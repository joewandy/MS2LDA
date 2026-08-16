"""Exact-resume training for the single supported K=500 ERNTM model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .core import (
    HardContextQueue,
    fresh_model,
    routing_temperature,
    sinkhorn_weight,
    validation_metrics,
)
from .data import ViewPair, iter_row_batches, sparse_batch
from .inventory import topic_inventory_summary
from .model import recycle_dead_prototypes, router_block_loss, topic_block_loss
from .regularizers import erntm_topic_constraint
from .utils import atomic_torch_save, file_sha256, peak_rss_bytes, read_json, write_json

if TYPE_CHECKING:
    import scipy.sparse as sp


def _entry_losses(output: Any, beta: torch.Tensor) -> torch.Tensor:
    topics = output.theta[output.row_ids]
    words = beta[:, output.token_indices].T
    probability = torch.sum(topics * words, dim=1).clamp_min(1e-12)
    return -torch.log(probability)


def _selection(history: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in history if row.get("validation") is not None]
    if not candidates:
        raise RuntimeError("training produced no validation checkpoint")

    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        validation = row["validation"]
        mass99 = validation["topic_inventory"]["mass_coverages"]["mass_99"]
        nll = validation["document_completion"]["nll_per_token"]
        return (
            -float(mass99["distinct_topic_equivalents"]),
            float(nll),
            int(row["epoch"]),
        )

    selected = min(candidates, key=key)
    epoch = int(selected["epoch"])
    return {
        "epoch": epoch,
        "checkpoint": f"validation_checkpoints/epoch_{epoch:04d}.pt",
        "validation": selected["validation"],
        "selection_rule": "maximum_validation_mass99_distinct_topic_equivalents_then_lower_nll",
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    history: list[dict[str, Any]],
    underuse_streak: np.ndarray,
    recycle_counts: np.ndarray,
    recycle_events: list[dict[str, Any]],
    context_queue: HardContextQueue,
) -> dict[str, Any]:
    return {
        "schema_version": "neural-ms2lda/training-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "elapsed_seconds": float(elapsed_seconds),
        "history": history,
        "underuse_streak": underuse_streak,
        "recycle_counts": recycle_counts,
        "recycle_events": recycle_events,
        "context_queue": context_queue.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }


def train_model(
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
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        selected = output / result["selected"]["checkpoint"]
        if file_sha256(selected) != result["selected"]["checkpoint_sha256"]:
            raise ValueError("selected model checkpoint changed")
        return result

    seed = int(protocol["seed"])
    num_topics = int(protocol["model"]["num_topics"])
    maximum_epochs = int(protocol["optimization"]["maximum_epochs"])
    validation_interval = int(protocol["optimization"]["validation_interval"])
    torch.manual_seed(seed)
    model = fresh_model(root, protocol)
    optimization = protocol["optimization"]
    model_config = protocol["model"]
    anti_collapse = protocol["anti_collapse"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    latest_path = output / "checkpoint_latest.pt"
    epoch_start = 0
    global_step = 0
    elapsed_before = 0.0
    history: list[dict[str, Any]] = []
    underuse_streak = np.zeros(num_topics, dtype=np.int64)
    recycle_counts = np.zeros(num_topics, dtype=np.int64)
    recycle_events: list[dict[str, Any]] = []
    context_queue = HardContextQueue.empty(max(4 * num_topics, 512))
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        history = list(checkpoint["history"])
        underuse_streak = checkpoint["underuse_streak"]
        recycle_counts = checkpoint["recycle_counts"]
        recycle_events = list(checkpoint["recycle_events"])
        context_queue = HardContextQueue.from_state_dict(checkpoint["context_queue"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if epoch_start > maximum_epochs:
        raise ValueError("checkpoint is beyond the resolved maximum epoch")

    started = time.perf_counter()
    stable = True
    stop_reason = "maximum_epochs"
    top_k = int(model_config["top_k"])
    for epoch in range(epoch_start, maximum_epochs):
        model.train()
        pair = views[epoch % len(views)]
        temperature = routing_temperature(epoch, protocol)
        balance_weight = sinkhorn_weight(epoch, protocol)
        totals = {
            "router_total": 0.0,
            "completion": 0.0,
            "sinkhorn": 0.0,
            "consistency": 0.0,
            "topic_base": 0.0,
            "local_decoder": 0.0,
            "erntm": 0.0,
        }
        router_batches = 0
        topic_updates = 0
        regularizer_diagnostics: dict[str, float] = {}
        with torch.no_grad():
            cached_beta = model.topic_word_distribution().detach()
        for rows in iter_row_batches(
            train.shape[0],
            batch_size=int(optimization["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            left_batch = sparse_batch(pair.left, rows)
            right_batch = sparse_batch(pair.right, rows)
            optimizer.zero_grad(set_to_none=True)
            terms = router_block_loss(
                model,
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
            if not torch.isfinite(terms.total):
                stable = False
                stop_reason = "non_finite_router_loss"
                break
            terms.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(model_config["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                stable = False
                stop_reason = "non_finite_router_gradient"
                break
            optimizer.step()
            with torch.no_grad():
                for routed in (terms.left, terms.right):
                    context_queue.add(
                        _entry_losses(routed, cached_beta),
                        routed.route_embeddings,
                        limit=32,
                    )
            totals["router_total"] += float(terms.total.detach())
            totals["completion"] += float(terms.completion.detach())
            totals["sinkhorn"] += float(terms.sinkhorn.detach())
            totals["consistency"] += float(terms.consistency.detach())
            router_batches += 1
            global_step += 1

        if stable:
            topic_rows = list(
                iter_row_batches(
                    train.shape[0],
                    batch_size=int(optimization["topic_update_batch_size"]),
                    shuffle=True,
                    seed=seed + 100_003 + epoch,
                )
            )
            for update in range(int(optimization["topic_updates_per_epoch"])):
                rows = topic_rows[update % len(topic_rows)]
                left_batch = sparse_batch(pair.left, rows)
                right_batch = sparse_batch(pair.right, rows)
                optimizer.zero_grad(set_to_none=True)
                terms = topic_block_loss(
                    model,
                    left_batch,
                    right_batch,
                    temperature=temperature,
                    top_k=top_k,
                    local_decoder_weight=float(optimization["local_decoder_weight"]),
                )
                regularizer = erntm_topic_constraint(model)
                weighted_erntm = float(optimization["erntm_weight"]) * regularizer.loss
                total = terms.total + weighted_erntm
                if not torch.isfinite(total):
                    stable = False
                    stop_reason = "non_finite_topic_loss"
                    break
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(model_config["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    stable = False
                    stop_reason = "non_finite_topic_gradient"
                    break
                optimizer.step()
                totals["topic_base"] += float(terms.total.detach())
                totals["completion"] += float(terms.completion.detach())
                totals["local_decoder"] += float(terms.local_decoder.detach())
                totals["erntm"] += float(regularizer.loss.detach())
                regularizer_diagnostics = regularizer.diagnostics
                topic_updates += 1
                global_step += 1

        epochs_completed = epoch + 1
        validation = None
        recycled: list[int] = []
        if stable and epochs_completed % validation_interval == 0:
            validation = validation_metrics(
                model,
                train=train,
                validation_observed=validation_observed,
                validation_completion=validation_completion,
                validation_full=validation_full,
                validation_records=validation_records,
                protocol=protocol,
                epoch=epochs_completed,
            )
            usage = np.asarray(validation.pop("_usage"), dtype=np.float64)
            with torch.inference_mode():
                beta = model.topic_word_distribution().cpu().numpy()
            validation["topic_inventory"] = topic_inventory_summary(
                beta, usage, top_n=int(protocol["evaluation"]["topic_top_n"])
            )
            checkpoint_path = (
                output / "validation_checkpoints" / f"epoch_{epochs_completed:04d}.pt"
            )
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": "neural-ms2lda/selected-model-v1",
                    "model": model.state_dict(),
                    "epoch": epochs_completed,
                    "validation": validation,
                    "routing_temperature": routing_temperature(
                        epochs_completed, protocol
                    ),
                    "top_k": top_k,
                },
            )
            underused = usage < (
                float(anti_collapse["recycle_usage_fraction_of_uniform"]) / num_topics
            )
            underuse_streak[underused] += 1
            underuse_streak[~underused] = 0
            eligible = np.flatnonzero(
                (underuse_streak >= int(anti_collapse["recycle_patience_validations"]))
                & (recycle_counts < int(anti_collapse["maximum_recycles_per_topic"]))
            )
            if (
                len(eligible)
                and epochs_completed <= int(anti_collapse["recycle_through_epoch"])
                and len(context_queue.heap)
            ):
                ordered = eligible[np.lexsort((eligible, usage[eligible]))]
                replacements = context_queue.pop_highest(len(ordered))
                ordered = ordered[: len(replacements)]
                if len(ordered):
                    indices = torch.from_numpy(ordered.astype(np.int64, copy=False))
                    recycle_dead_prototypes(
                        model,
                        optimizer,
                        topic_indices=indices,
                        replacements=replacements,
                    )
                    recycled = ordered.tolist()
                    recycle_counts[ordered] += 1
                    underuse_streak[ordered] = 0
                    recycle_events.append(
                        {"epoch": epochs_completed, "topic_indices": recycled}
                    )

        elapsed = elapsed_before + time.perf_counter() - started
        row = {
            "epoch": epochs_completed,
            "global_step": global_step,
            "view_pair": pair.pair_index,
            "elapsed_seconds": elapsed,
            "routing_temperature": temperature,
            "sinkhorn_weight": balance_weight,
            "losses": {
                "router_total": totals["router_total"] / max(router_batches, 1),
                "completion": totals["completion"]
                / max(router_batches + topic_updates, 1),
                "sinkhorn": totals["sinkhorn"] / max(router_batches, 1),
                "consistency": totals["consistency"] / max(router_batches, 1),
                "topic_base": totals["topic_base"] / max(topic_updates, 1),
                "local_decoder": totals["local_decoder"] / max(topic_updates, 1),
                "erntm": totals["erntm"] / max(topic_updates, 1),
            },
            "regularizer_diagnostics": regularizer_diagnostics,
            "validation": validation,
            "recycled_topics": recycled,
            "stable": stable,
        }
        history.append(row)
        atomic_torch_save(
            latest_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epochs_completed,
                global_step=global_step,
                elapsed_seconds=elapsed,
                history=history,
                underuse_streak=underuse_streak,
                recycle_counts=recycle_counts,
                recycle_events=recycle_events,
                context_queue=context_queue,
            ),
        )
        write_json(output / "history.json", history)
        if heartbeat is not None:
            heartbeat(
                stage="train_neural",
                epoch=epochs_completed,
                maximum_epochs=maximum_epochs,
                elapsed_seconds=elapsed,
                stable=stable,
            )
        if not stable:
            break

    if not stable:
        raise RuntimeError(f"neural training failed: {stop_reason}")
    selected = _selection(history)
    selected_path = output / selected["checkpoint"]
    selected["checkpoint_sha256"] = file_sha256(selected_path)
    write_json(output / "selected.json", selected)
    result = {
        "schema_version": "neural-ms2lda/training-complete-v1",
        "num_topics": num_topics,
        "epochs_completed": int(history[-1]["epoch"]),
        "stable": True,
        "stop_reason": stop_reason,
        "elapsed_seconds": float(history[-1]["elapsed_seconds"]),
        "selected": selected,
        "recycle_count_total": int(recycle_counts.sum()),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(complete_path, result)
    return result


def load_selected_model(
    run_dir: str | Path, protocol: dict[str, Any]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the validation-selected checkpoint and verify its digest."""
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
