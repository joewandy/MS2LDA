"""Alternating numerical updates for neural MS2LDA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .data import iter_row_batches, sparse_batch
from .objectives import (
    cooccurrence_topic_loss,
    router_block_loss,
    topic_block_loss,
    topic_separation_loss,
)

if TYPE_CHECKING:
    import scipy.sparse as sp


def routing_temperature(epoch: int, protocol: dict[str, Any]) -> float:
    """Linearly anneal the fixed top-2 routing temperature."""
    config = protocol["anti_collapse"]
    start = float(config["routing_temperature_start"])
    end = float(config["routing_temperature_end"])
    progress = min(
        max(epoch, 0) / max(float(config["routing_temperature_anneal_epochs"]), 1.0),
        1.0,
    )
    return start + progress * (end - start)


def sinkhorn_weight(epoch: int, protocol: dict[str, Any]) -> float:
    """Return the fixed anti-collapse balance schedule."""
    config = protocol["anti_collapse"]
    start = float(config["sinkhorn_weight_start"])
    hold = int(config["sinkhorn_weight_hold_epochs"])
    end = float(config["sinkhorn_weight_end"])
    end_epoch = int(protocol["optimization"]["maximum_epochs"])
    if epoch < hold:
        return start
    progress = min(max((epoch - hold) / max(end_epoch - hold, 1), 0.0), 1.0)
    return start + progress * (end - start)


@dataclass
class TrainingState:
    """Model and optimizer shared by the alternating phases."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


def _weighted_topic_separation(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.Tensor:
    """Return the weighted nearest-topic margin loss."""
    loss = topic_separation_loss(
        model,
        neighbors=int(config["neighbors"]),
        margin=float(config["margin"]),
    )
    return float(config["weight"]) * loss


def _apply_gradient_step(
    state: TrainingState,
    total: torch.Tensor,
    *,
    clip_norm: float,
    phase: str,
) -> None:
    """Apply one finite, clipped optimizer step or fail immediately."""
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite {phase} loss")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"non-finite {phase} gradient")
    state.optimizer.step()


def router_phase(  # noqa: PLR0913
    state: TrainingState,
    *,
    train: sp.csr_matrix,
    epoch: int,
    temperature: float,
    balance_weight: float,
    protocol: dict[str, Any],
) -> None:
    """Update routing parameters while treating the decoder as fixed."""
    optimization = protocol["optimization"]
    with torch.no_grad():
        cached_beta = state.model.topic_word_distribution().detach()
    for rows in iter_row_batches(
        train.shape[0],
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        seed=int(protocol["seed"]) + epoch,
    ):
        batch = sparse_batch(train, rows)
        state.optimizer.zero_grad(set_to_none=True)
        terms = router_block_loss(
            state.model,
            batch,
            cached_beta=cached_beta,
            temperature=temperature,
            sinkhorn_weight=balance_weight,
            sinkhorn_epsilon=float(protocol["anti_collapse"]["sinkhorn_epsilon"]),
            sinkhorn_iterations=int(protocol["anti_collapse"]["sinkhorn_iterations"]),
        )
        total = terms.total + _weighted_topic_separation(
            state.model,
            protocol["topic_separation"],
        )
        _apply_gradient_step(
            state,
            total,
            clip_norm=float(optimization["gradient_clip_norm"]),
            phase="router",
        )


def topic_phase(  # noqa: PLR0913
    state: TrainingState,
    *,
    train: sp.csr_matrix,
    epoch: int,
    temperature: float,
    graph_tensor: torch.Tensor,
    protocol: dict[str, Any],
) -> None:
    """Update prototypes and decoder while treating token routes as fixed."""
    optimization = protocol["optimization"]
    graph_config = protocol["cooccurrence_regularization"]
    topic_rows = list(
        iter_row_batches(
            train.shape[0],
            batch_size=int(optimization["topic_update_batch_size"]),
            shuffle=True,
            seed=int(protocol["seed"]) + 100_003 + epoch,
        )
    )
    for update in range(int(optimization["topic_updates_per_epoch"])):
        batch = sparse_batch(train, topic_rows[update % len(topic_rows)])
        state.optimizer.zero_grad(set_to_none=True)
        terms = topic_block_loss(
            state.model,
            batch,
            temperature=temperature,
        )
        cooccurrence = cooccurrence_topic_loss(
            state.model,
            graph_tensor,
            beta=terms.beta,
        )
        total = (
            terms.total
            + float(graph_config["weight"]) * cooccurrence
            + _weighted_topic_separation(
                state.model,
                protocol["topic_separation"],
            )
        )
        _apply_gradient_step(
            state,
            total,
            clip_norm=float(optimization["gradient_clip_norm"]),
            phase="topic",
        )
