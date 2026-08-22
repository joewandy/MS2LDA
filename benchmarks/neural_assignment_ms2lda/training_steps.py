"""Numerical update and anti-collapse phases for neural MS2LDA training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .core import HardContextQueue, routing_temperature, validation_metrics
from .data import ViewPair, iter_row_batches, sparse_batch
from .inventory import topic_inventory_summary
from .objectives import recycle_dead_prototypes, router_block_loss, topic_block_loss
from .regularizers import (
    cooccurrence_topic_constraint,
    nearest_neighbor_topic_constraint,
)
from .utils import atomic_torch_save

if TYPE_CHECKING:
    import scipy.sparse as sp


LOSS_NAMES = (
    "router_total",
    "router_base",
    "router_separation",
    "completion",
    "sinkhorn",
    "consistency",
    "topic_base",
    "local_decoder",
    "cooccurrence",
    "topic_separation",
)


@dataclass
class TrainingState:
    """All mutable state required for exact epoch-boundary continuation."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    epoch_start: int
    global_step: int
    elapsed_before: float
    history: list[dict[str, Any]]
    underuse_streak: np.ndarray
    recycle_counts: np.ndarray
    recycle_events: list[dict[str, Any]]
    context_queue: HardContextQueue


def _entry_losses(output: Any, beta: torch.Tensor) -> torch.Tensor:
    """Score routed token contexts for deterministic dead-topic recycling."""
    topics = output.theta[output.row_ids]
    words = beta[:, output.token_indices].T
    probability = torch.sum(topics * words, dim=1).clamp_min(1e-12)
    return -torch.log(probability)


def _weighted_topic_separation(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[Any, torch.Tensor]:
    """Return the supported nearest-topic margin and its weighted loss."""
    result = nearest_neighbor_topic_constraint(
        model,
        neighbors=int(config["neighbors"]),
        margin=float(config["margin"]),
    )
    return result, float(config["weight"]) * result.loss


def _apply_gradient_step(
    state: TrainingState,
    total: torch.Tensor,
    *,
    clip_norm: float,
    loss_failure: str,
    gradient_failure: str,
) -> str | None:
    """Apply one finite, clipped optimizer step or return its failure reason."""
    if not torch.isfinite(total):
        return loss_failure
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), clip_norm)
    if not torch.isfinite(gradient_norm):
        return gradient_failure
    state.optimizer.step()
    state.global_step += 1
    return None


def router_phase(  # noqa: PLR0913
    state: TrainingState,
    *,
    train: sp.csr_matrix,
    pair: ViewPair,
    epoch: int,
    temperature: float,
    balance_weight: float,
    top_k: int,
    protocol: dict[str, Any],
    totals: dict[str, float],
) -> tuple[int, str | None]:
    """Update routing parameters while treating the current decoder as fixed."""
    optimization = protocol["optimization"]
    model_config = protocol["model"]
    separation_config = protocol["topic_separation"]
    with torch.no_grad():
        cached_beta = state.model.topic_word_distribution().detach()
    batches = 0
    for rows in iter_row_batches(
        train.shape[0],
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        seed=int(protocol["seed"]) + epoch,
    ):
        left_batch = sparse_batch(pair.left, rows)
        right_batch = sparse_batch(pair.right, rows)
        state.optimizer.zero_grad(set_to_none=True)
        terms = router_block_loss(
            state.model,
            left_batch,
            right_batch,
            cached_beta=cached_beta,
            temperature=temperature,
            top_k=top_k,
            sinkhorn_weight=balance_weight,
            consistency_weight=float(optimization["theta_consistency_weight"]),
            sinkhorn_epsilon=float(model_config["sinkhorn_epsilon"]),
            sinkhorn_iterations=int(model_config["sinkhorn_iterations"]),
        )
        separation, weighted_separation = _weighted_topic_separation(
            state.model, separation_config
        )
        total = terms.total + weighted_separation
        failure = _apply_gradient_step(
            state,
            total,
            clip_norm=float(model_config["gradient_clip_norm"]),
            loss_failure="non_finite_router_loss",
            gradient_failure="non_finite_router_gradient",
        )
        if failure is not None:
            return batches, failure

        # Hard contexts are ranked under the same fixed decoder used for this
        # router block, making later prototype replacement deterministic.
        with torch.no_grad():
            for routed in (terms.left, terms.right):
                state.context_queue.add(
                    _entry_losses(routed, cached_beta),
                    routed.route_embeddings,
                    limit=32,
                )
        totals["router_total"] += float(total.detach())
        totals["router_base"] += float(terms.total.detach())
        totals["router_separation"] += float(separation.loss.detach())
        totals["completion"] += float(terms.completion.detach())
        totals["sinkhorn"] += float(terms.sinkhorn.detach())
        totals["consistency"] += float(terms.consistency.detach())
        batches += 1
    return batches, None


def topic_phase(  # noqa: PLR0913
    state: TrainingState,
    *,
    train: sp.csr_matrix,
    pair: ViewPair,
    epoch: int,
    temperature: float,
    top_k: int,
    graph_tensor: torch.Tensor,
    protocol: dict[str, Any],
    totals: dict[str, float],
) -> tuple[int, dict[str, float], str | None]:
    """Update prototypes and decoder while treating token routes as fixed."""
    optimization = protocol["optimization"]
    model_config = protocol["model"]
    graph_config = protocol["cooccurrence_regularization"]
    topic_rows = list(
        iter_row_batches(
            train.shape[0],
            batch_size=int(optimization["topic_update_batch_size"]),
            shuffle=True,
            seed=int(protocol["seed"]) + 100_003 + epoch,
        )
    )
    diagnostics: dict[str, float] = {}
    updates = 0
    for update in range(int(optimization["topic_updates_per_epoch"])):
        rows = topic_rows[update % len(topic_rows)]
        left_batch = sparse_batch(pair.left, rows)
        right_batch = sparse_batch(pair.right, rows)
        state.optimizer.zero_grad(set_to_none=True)
        terms = topic_block_loss(
            state.model,
            left_batch,
            right_batch,
            temperature=temperature,
            top_k=top_k,
            local_decoder_weight=float(optimization["local_decoder_weight"]),
        )
        separation, weighted_separation = _weighted_topic_separation(
            state.model, protocol["topic_separation"]
        )
        cooccurrence = cooccurrence_topic_constraint(
            state.model, graph_tensor, beta=terms.beta
        )
        weighted_cooccurrence = float(graph_config["weight"]) * cooccurrence.loss
        total = terms.total + weighted_cooccurrence + weighted_separation
        failure = _apply_gradient_step(
            state,
            total,
            clip_norm=float(model_config["gradient_clip_norm"]),
            loss_failure="non_finite_topic_loss",
            gradient_failure="non_finite_topic_gradient",
        )
        if failure is not None:
            return updates, diagnostics, failure
        totals["topic_base"] += float(terms.total.detach())
        totals["completion"] += float(terms.completion.detach())
        totals["local_decoder"] += float(terms.local_decoder.detach())
        totals["cooccurrence"] += float(cooccurrence.loss.detach())
        totals["topic_separation"] += float(separation.loss.detach())
        diagnostics = {**separation.diagnostics, **cooccurrence.diagnostics}
        updates += 1
    return updates, diagnostics, None


def validate_and_recycle(  # noqa: PLR0913
    state: TrainingState,
    *,
    output: Path,
    train: sp.csr_matrix,
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_full: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    protocol: dict[str, Any],
    epoch: int,
    top_k: int,
) -> tuple[dict[str, Any], list[int]]:
    """Evaluate one scheduled epoch, save it, and recycle persistently dead topics."""
    validation = validation_metrics(
        state.model,
        train=train,
        validation_observed=validation_observed,
        validation_completion=validation_completion,
        validation_full=validation_full,
        validation_records=validation_records,
        protocol=protocol,
        epoch=epoch,
        include_npmi=True,
    )
    usage = np.asarray(validation.pop("_usage"), dtype=np.float64)
    with torch.inference_mode():
        beta = state.model.topic_word_distribution().cpu().numpy()
    validation["topic_inventory"] = topic_inventory_summary(
        beta, usage, top_n=int(protocol["evaluation"]["topic_top_n"])
    )
    checkpoint_path = output / "validation_checkpoints" / f"epoch_{epoch:04d}.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "neural-ms2lda/selected-model-v1",
            "model": state.model.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "routing_temperature": routing_temperature(epoch, protocol),
            "top_k": top_k,
        },
    )

    anti_collapse = protocol["anti_collapse"]
    num_topics = int(protocol["model"]["num_topics"])
    underused = usage < (
        float(anti_collapse["recycle_usage_fraction_of_uniform"]) / num_topics
    )
    state.underuse_streak[underused] += 1
    state.underuse_streak[~underused] = 0
    eligible = np.flatnonzero(
        (state.underuse_streak >= int(anti_collapse["recycle_patience_validations"]))
        & (state.recycle_counts < int(anti_collapse["maximum_recycles_per_topic"]))
    )
    if (
        not len(eligible)
        or epoch > int(anti_collapse["recycle_through_epoch"])
        or not len(state.context_queue.heap)
    ):
        return validation, []

    # A persistent under-use streak, never one noisy batch, triggers reuse of
    # the highest-loss stored token context. Ties are deterministic by index.
    ordered = eligible[np.lexsort((eligible, usage[eligible]))]
    replacements = state.context_queue.pop_highest(len(ordered))
    ordered = ordered[: len(replacements)]
    if not len(ordered):
        return validation, []
    indices = torch.from_numpy(ordered.astype(np.int64, copy=False))
    recycle_dead_prototypes(
        state.model,
        state.optimizer,
        topic_indices=indices,
        replacements=replacements,
    )
    recycled = ordered.tolist()
    state.recycle_counts[ordered] += 1
    state.underuse_streak[ordered] = 0
    state.recycle_events.append({"epoch": epoch, "topic_indices": recycled})
    return validation, recycled
