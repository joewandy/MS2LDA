"""Numerical update and anti-collapse phases for neural MS2LDA training."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import ViewPair, iter_row_batches, sparse_batch
from .model import infer_theta
from .objectives import (
    cooccurrence_topic_loss,
    recycle_dead_prototypes,
    router_block_loss,
    topic_block_loss,
    topic_separation_loss,
)

if TYPE_CHECKING:
    import scipy.sparse as sp


@dataclass
class HardContextQueue:
    """Bounded deterministic queue of high-loss routing contexts."""

    capacity: int
    heap: list[tuple[float, int, torch.Tensor]]
    serial: int = 0

    @classmethod
    def empty(cls, capacity: int) -> HardContextQueue:
        """Create an empty queue with a fixed memory bound."""
        return cls(capacity=int(capacity), heap=[])

    def add(self, losses: torch.Tensor, contexts: torch.Tensor, *, limit: int) -> None:
        """Retain the highest-loss normalized route contexts seen so far."""
        if not len(losses):
            return
        selected = torch.topk(
            losses.detach(),
            k=min(int(limit), len(losses)),
            largest=True,
        ).indices
        for index in selected.tolist():
            item = (
                float(losses[index]),
                self.serial,
                contexts[index].detach().cpu().clone(),
            )
            self.serial += 1
            if len(self.heap) < self.capacity:
                heapq.heappush(self.heap, item)
            elif item[:2] > self.heap[0][:2]:
                heapq.heapreplace(self.heap, item)

    def pop_highest(self, count: int) -> torch.Tensor:
        """Remove and return up to ``count`` contexts in descending loss order."""
        selected = heapq.nlargest(min(int(count), len(self.heap)), self.heap)
        selected_ids = {serial for _, serial, _ in selected}
        self.heap = [item for item in self.heap if item[1] not in selected_ids]
        heapq.heapify(self.heap)
        if not selected:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack([item[2] for item in selected])


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


@torch.inference_mode()
def validation_topic_usage(
    model: torch.nn.Module,
    *,
    validation_full: sp.csr_matrix,
    protocol: dict[str, Any],
    epoch: int,
) -> np.ndarray:
    """Return mean full-spectrum topic usage for dead-topic detection."""
    batch_size = int(protocol["optimization"]["batch_size"])
    temperature = routing_temperature(epoch, protocol)
    full_theta = infer_theta(
        model,
        validation_full,
        batch_size=batch_size,
        temperature=temperature,
    )
    return full_theta.mean(axis=0).astype(np.float32)


@dataclass
class TrainingState:
    """Mutable quantities shared by the alternating optimization phases."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    underuse_streak: np.ndarray
    recycle_counts: np.ndarray
    context_queue: HardContextQueue


def _entry_losses(output: Any, batch: Any, beta: torch.Tensor) -> torch.Tensor:
    """Score routed token contexts for deterministic dead-topic recycling."""
    topics = output.theta[batch.row_ids]
    words = beta[:, batch.indices].T
    probability = torch.sum(topics * words, dim=1).clamp_min(1e-12)
    return -torch.log(probability)


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
    pair: ViewPair,
    epoch: int,
    temperature: float,
    balance_weight: float,
    protocol: dict[str, Any],
) -> None:
    """Update routing parameters while treating the current decoder as fixed."""
    optimization = protocol["optimization"]
    separation_config = protocol["topic_separation"]
    with torch.no_grad():
        cached_beta = state.model.topic_word_distribution().detach()
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
            sinkhorn_weight=balance_weight,
            consistency_weight=float(optimization["theta_consistency_weight"]),
            sinkhorn_epsilon=float(protocol["anti_collapse"]["sinkhorn_epsilon"]),
            sinkhorn_iterations=int(protocol["anti_collapse"]["sinkhorn_iterations"]),
        )
        weighted_separation = _weighted_topic_separation(state.model, separation_config)
        total = terms.total + weighted_separation
        _apply_gradient_step(
            state,
            total,
            clip_norm=float(optimization["gradient_clip_norm"]),
            phase="router",
        )

        # Hard contexts are ranked under the same fixed decoder used for this
        # router block, making later prototype replacement deterministic.
        with torch.no_grad():
            for routed, batch in (
                (terms.left, left_batch),
                (terms.right, right_batch),
            ):
                state.context_queue.add(
                    _entry_losses(routed, batch, cached_beta),
                    routed.route_embeddings,
                    limit=32,
                )


def topic_phase(  # noqa: PLR0913
    state: TrainingState,
    *,
    train: sp.csr_matrix,
    pair: ViewPair,
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
        rows = topic_rows[update % len(topic_rows)]
        left_batch = sparse_batch(pair.left, rows)
        right_batch = sparse_batch(pair.right, rows)
        state.optimizer.zero_grad(set_to_none=True)
        terms = topic_block_loss(
            state.model,
            left_batch,
            right_batch,
            temperature=temperature,
            local_decoder_weight=float(optimization["local_decoder_weight"]),
        )
        weighted_separation = _weighted_topic_separation(
            state.model, protocol["topic_separation"]
        )
        cooccurrence = cooccurrence_topic_loss(
            state.model, graph_tensor, beta=terms.beta
        )
        weighted_cooccurrence = float(graph_config["weight"]) * cooccurrence
        total = terms.total + weighted_cooccurrence + weighted_separation
        _apply_gradient_step(
            state,
            total,
            clip_norm=float(optimization["gradient_clip_norm"]),
            phase="topic",
        )


def validate_and_recycle(  # noqa: PLR0913
    state: TrainingState,
    *,
    validation_full: sp.csr_matrix,
    protocol: dict[str, Any],
    epoch: int,
) -> None:
    """Measure usage at one scheduled epoch and recycle persistently dead topics."""
    usage = validation_topic_usage(
        state.model,
        validation_full=validation_full,
        protocol=protocol,
        epoch=epoch,
    )
    usage = np.asarray(usage, dtype=np.float64)
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
    if not len(eligible) or not len(state.context_queue.heap):
        return

    # A persistent under-use streak, never one noisy batch, triggers reuse of
    # the highest-loss stored token context. Ties are deterministic by index.
    ordered = eligible[np.lexsort((eligible, usage[eligible]))]
    replacements = state.context_queue.pop_highest(len(ordered))
    ordered = ordered[: len(replacements)]
    if not len(ordered):
        return
    indices = torch.from_numpy(ordered.astype(np.int64, copy=False))
    recycle_dead_prototypes(
        state.model,
        state.optimizer,
        topic_indices=indices,
        replacements=replacements,
    )
    state.recycle_counts[ordered] += 1
    state.underuse_streak[ordered] = 0
