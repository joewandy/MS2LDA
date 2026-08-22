"""Exact-resume orchestration for the supported neural topic model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .cooccurrence import prepare_cooccurrence_graph, torch_sparse_graph
from .core import HardContextQueue, fresh_model, routing_temperature, sinkhorn_weight
from .data import ViewPair
from .training_steps import (
    LOSS_NAMES,
    TrainingState,
    router_phase,
    topic_phase,
    validate_and_recycle,
)
from .utils import atomic_torch_save, file_sha256, peak_rss_bytes, read_json, write_json

if TYPE_CHECKING:
    import scipy.sparse as sp


def _selection(
    history: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Select the predeclared final epoch without metric-based checkpoint search."""
    epoch = int(protocol["optimization"]["maximum_epochs"])
    selected = next((row for row in history if int(row["epoch"]) == epoch), None)
    if selected is None or selected.get("validation") is None:
        raise RuntimeError("the final epoch has no validation checkpoint")
    return {
        "selection_rule": "fixed_final_epoch",
        "epoch": epoch,
        "checkpoint": f"validation_checkpoints/epoch_{epoch:04d}.pt",
        "validation": selected["validation"],
    }


def _checkpoint_payload(
    *,
    state: TrainingState,
    epoch: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Capture every mutable state needed to continue at an epoch boundary."""
    return {
        "schema_version": "neural-ms2lda/training-checkpoint-v1",
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(state.global_step),
        "elapsed_seconds": float(elapsed_seconds),
        "history": state.history,
        "underuse_streak": state.underuse_streak,
        "recycle_counts": state.recycle_counts,
        "recycle_events": state.recycle_events,
        "context_queue": state.context_queue.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }


def _completed_result(
    output: Path, graph_manifest: dict[str, Any]
) -> dict[str, Any] | None:
    """Return an already completed result only after checking its dependencies."""
    complete_path = output / "complete.json"
    if not complete_path.is_file():
        return None
    result = read_json(complete_path)
    if result["cooccurrence_graph"] != graph_manifest:
        raise ValueError("model co-occurrence graph provenance changed")
    selected = output / result["selected"]["checkpoint"]
    if file_sha256(selected) != result["selected"]["checkpoint_sha256"]:
        raise ValueError("selected model checkpoint changed")
    return result


def _initialize_training_state(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
) -> TrainingState:
    """Build the optimizer state or restore its exact epoch-boundary snapshot."""
    seed = int(protocol["seed"])
    num_topics = int(protocol["model"]["num_topics"])
    torch.manual_seed(seed)
    model = fresh_model(root, protocol)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        epoch_start=0,
        global_step=0,
        elapsed_before=0.0,
        history=[],
        underuse_streak=np.zeros(num_topics, dtype=np.int64),
        recycle_counts=np.zeros(num_topics, dtype=np.int64),
        recycle_events=[],
        context_queue=HardContextQueue.empty(max(4 * num_topics, 512)),
    )
    latest_path = output / "checkpoint_latest.pt"
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        state.model.load_state_dict(checkpoint["model"])
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.epoch_start = int(checkpoint["epoch"])
        state.global_step = int(checkpoint["global_step"])
        state.elapsed_before = float(checkpoint["elapsed_seconds"])
        state.history = list(checkpoint["history"])
        state.underuse_streak = checkpoint["underuse_streak"]
        state.recycle_counts = checkpoint["recycle_counts"]
        state.recycle_events = list(checkpoint["recycle_events"])
        state.context_queue = HardContextQueue.from_state_dict(
            checkpoint["context_queue"]
        )
        torch.set_rng_state(checkpoint["torch_rng_state"])
    maximum_epochs = int(optimization["maximum_epochs"])
    if state.epoch_start > maximum_epochs:
        raise ValueError("checkpoint is beyond the resolved maximum epoch")
    return state


def _mean_losses(
    totals: dict[str, float], *, router_batches: int, topic_updates: int
) -> dict[str, float]:
    """Normalize each accumulated loss by the phase that generated it."""
    return {
        "router_total": totals["router_total"] / max(router_batches, 1),
        "router_base": totals["router_base"] / max(router_batches, 1),
        "router_separation": totals["router_separation"] / max(router_batches, 1),
        "completion": totals["completion"] / max(router_batches + topic_updates, 1),
        "sinkhorn": totals["sinkhorn"] / max(router_batches, 1),
        "consistency": totals["consistency"] / max(router_batches, 1),
        "topic_base": totals["topic_base"] / max(topic_updates, 1),
        "local_decoder": totals["local_decoder"] / max(topic_updates, 1),
        "cooccurrence": totals["cooccurrence"] / max(topic_updates, 1),
        "topic_separation": totals["topic_separation"] / max(topic_updates, 1),
    }


def _record_epoch(  # noqa: PLR0913
    state: TrainingState,
    *,
    output: Path,
    epoch: int,
    pair_index: int,
    elapsed: float,
    temperature: float,
    balance_weight: float,
    totals: dict[str, float],
    router_batches: int,
    topic_updates: int,
    diagnostics: dict[str, float],
    validation: dict[str, Any] | None,
    recycled: list[int],
    stable: bool,
) -> None:
    """Persist the auditable epoch row and exact mutable continuation state."""
    state.history.append(
        {
            "epoch": epoch,
            "global_step": state.global_step,
            "view_pair": pair_index,
            "elapsed_seconds": elapsed,
            "routing_temperature": temperature,
            "sinkhorn_weight": balance_weight,
            "losses": _mean_losses(
                totals,
                router_batches=router_batches,
                topic_updates=topic_updates,
            ),
            "regularizer_diagnostics": diagnostics,
            "validation": validation,
            "recycled_topics": recycled,
            "stable": stable,
        }
    )
    atomic_torch_save(
        output / "checkpoint_latest.pt",
        _checkpoint_payload(state=state, epoch=epoch, elapsed_seconds=elapsed),
    )
    write_json(output / "history.json", state.history)


def train_model(  # noqa: PLR0913
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    views: list[ViewPair],
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_full: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    protocol: dict[str, Any],
    heartbeat: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Train or exactly resume the resolved 40-epoch model."""
    root = Path(run_dir)
    output = root / "model"
    output.mkdir(parents=True, exist_ok=True)
    graph, graph_manifest = prepare_cooccurrence_graph(
        root, train=train, protocol=protocol
    )
    completed = _completed_result(output, graph_manifest)
    if completed is not None:
        return completed

    graph_tensor = torch_sparse_graph(graph)
    state = _initialize_training_state(root, output, protocol)
    num_topics = int(protocol["model"]["num_topics"])
    maximum_epochs = int(protocol["optimization"]["maximum_epochs"])
    validation_interval = int(protocol["optimization"]["validation_interval"])
    started = time.perf_counter()
    failure: str | None = None
    top_k = int(protocol["model"]["top_k"])
    for epoch in range(state.epoch_start, maximum_epochs):
        state.model.train()
        pair = views[epoch % len(views)]
        temperature = routing_temperature(epoch, protocol)
        balance_weight = sinkhorn_weight(epoch, protocol)
        totals = dict.fromkeys(LOSS_NAMES, 0.0)
        router_batches, failure = router_phase(
            state,
            train=train,
            pair=pair,
            epoch=epoch,
            temperature=temperature,
            balance_weight=balance_weight,
            top_k=top_k,
            protocol=protocol,
            totals=totals,
        )
        topic_updates = 0
        diagnostics: dict[str, float] = {}
        if failure is None:
            topic_updates, diagnostics, failure = topic_phase(
                state,
                train=train,
                pair=pair,
                epoch=epoch,
                temperature=temperature,
                top_k=top_k,
                graph_tensor=graph_tensor,
                protocol=protocol,
                totals=totals,
            )
        epochs_completed = epoch + 1
        validation = None
        recycled: list[int] = []
        if failure is None and epochs_completed % validation_interval == 0:
            validation, recycled = validate_and_recycle(
                state,
                output=output,
                train=train,
                validation_observed=validation_observed,
                validation_completion=validation_completion,
                validation_full=validation_full,
                validation_records=validation_records,
                protocol=protocol,
                epoch=epochs_completed,
                top_k=top_k,
            )
        elapsed = state.elapsed_before + time.perf_counter() - started
        _record_epoch(
            state,
            output=output,
            epoch=epochs_completed,
            pair_index=pair.pair_index,
            elapsed=elapsed,
            temperature=temperature,
            balance_weight=balance_weight,
            totals=totals,
            router_batches=router_batches,
            topic_updates=topic_updates,
            diagnostics=diagnostics,
            validation=validation,
            recycled=recycled,
            stable=failure is None,
        )
        if heartbeat is not None:
            heartbeat(
                stage="train_neural",
                epoch=epochs_completed,
                maximum_epochs=maximum_epochs,
                elapsed_seconds=elapsed,
                stable=failure is None,
            )
        if failure is not None:
            break

    if failure is not None:
        raise RuntimeError(f"neural training failed: {failure}")
    selected = _selection(state.history, protocol)
    selected_path = output / selected["checkpoint"]
    selected["checkpoint_sha256"] = file_sha256(selected_path)
    write_json(output / "selected.json", selected)
    result = {
        "schema_version": "neural-ms2lda/training-complete-v1",
        "num_topics": num_topics,
        "epochs_completed": int(state.history[-1]["epoch"]),
        "stable": True,
        "stop_reason": "maximum_epochs",
        "elapsed_seconds": float(state.history[-1]["elapsed_seconds"]),
        "selected": selected,
        "recycle_count_total": int(state.recycle_counts.sum()),
        "cooccurrence_graph": graph_manifest,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(output / "complete.json", result)
    return result


def load_selected_model(
    run_dir: str | Path, protocol: dict[str, Any]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the fixed final-epoch checkpoint and verify its digest."""
    directory = Path(run_dir)
    selected = read_json(directory / "model/selected.json")
    checkpoint_path = directory / "model" / selected["checkpoint"]
    if file_sha256(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    model = fresh_model(directory, protocol)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
