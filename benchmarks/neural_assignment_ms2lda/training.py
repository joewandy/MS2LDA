"""Exact-resume training for the supported K=500 neural topic model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .cooccurrence import prepare_cooccurrence_graph, torch_sparse_graph
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
from .regularizers import (
    cooccurrence_topic_constraint,
    nearest_neighbor_topic_constraint,
)
from .utils import atomic_torch_save, file_sha256, peak_rss_bytes, read_json, write_json

if TYPE_CHECKING:
    import scipy.sparse as sp


def _entry_losses(output: Any, beta: torch.Tensor) -> torch.Tensor:
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


def validation_gate_summary(
    validation: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the predeclared seed-42 development gates."""
    targets = protocol["development_gates"]
    coherence = validation["word_cooccurrence_npmi"]
    values = {
        "mean_npmi": float(coherence["mean_npmi"]),
        "undefined_pair_fraction": float(coherence["undefined_pair_fraction"]),
        "top_word_diversity": float(validation["top_word_diversity"]),
        "effective_topics_median": float(
            validation["mixture_diagnostics"]["effective_topic_count_median"]
        ),
        "validation_nll": float(validation["document_completion"]["nll_per_token"]),
    }
    gates = {
        "mean_npmi": values["mean_npmi"] >= float(targets["minimum_mean_npmi"]),
        "undefined_pair_fraction": values["undefined_pair_fraction"]
        <= float(targets["maximum_undefined_pair_fraction"]),
        "top_word_diversity": values["top_word_diversity"]
        >= float(targets["minimum_top_word_diversity"]),
        "effective_topics_median": values["effective_topics_median"]
        <= float(targets["maximum_effective_topics_median"]),
        "validation_nll": values["validation_nll"]
        <= float(targets["maximum_validation_nll"]),
    }
    return {
        "values": values,
        "gates": gates,
        "gates_met": int(sum(gates.values())),
        "all_gates_met": bool(all(gates.values())),
    }


def _selection(
    history: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    candidates = [row for row in history if row.get("validation") is not None]
    if not candidates:
        raise RuntimeError("training produced no validation checkpoint")

    def key(row: dict[str, Any]) -> tuple[float, ...]:
        validation = row["validation"]
        summary = validation_gate_summary(validation, protocol)
        gates = summary["gates"]
        values = summary["values"]
        admissibility_failures = int(not gates["top_word_diversity"]) + int(
            not gates["validation_nll"]
        )
        return (
            float(admissibility_failures),
            -float(summary["gates_met"]),
            -values["mean_npmi"],
            values["undefined_pair_fraction"],
            values["effective_topics_median"],
            values["validation_nll"],
            float(row["epoch"]),
        )

    selected = min(candidates, key=key)
    epoch = int(selected["epoch"])
    return {
        "epoch": epoch,
        "checkpoint": f"validation_checkpoints/epoch_{epoch:04d}.pt",
        "validation": selected["validation"],
        "validation_gate_summary": validation_gate_summary(
            selected["validation"], protocol
        ),
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
    graph_config = protocol["cooccurrence_regularization"]
    separation_config = protocol["topic_separation"]
    graph, graph_manifest = prepare_cooccurrence_graph(
        root, train=train, protocol=protocol
    )
    graph_tensor = torch_sparse_graph(graph)
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        if result["cooccurrence_graph"] != graph_manifest:
            raise ValueError("model co-occurrence graph provenance changed")
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
            "router_base": 0.0,
            "router_separation": 0.0,
            "completion": 0.0,
            "sinkhorn": 0.0,
            "consistency": 0.0,
            "topic_base": 0.0,
            "local_decoder": 0.0,
            "cooccurrence": 0.0,
            "topic_separation": 0.0,
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
            router_separation, weighted_router_separation = _weighted_topic_separation(
                model,
                separation_config,
            )
            router_total = terms.total + weighted_router_separation
            if not torch.isfinite(router_total):
                stable = False
                stop_reason = "non_finite_router_loss"
                break
            router_total.backward()
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
            totals["router_total"] += float(router_total.detach())
            totals["router_base"] += float(terms.total.detach())
            totals["router_separation"] += float(router_separation.loss.detach())
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
                separation, weighted_separation = _weighted_topic_separation(
                    model,
                    separation_config,
                )
                cooccurrence = cooccurrence_topic_constraint(
                    model, graph_tensor, beta=terms.beta
                )
                weighted_cooccurrence = (
                    float(graph_config["weight"]) * cooccurrence.loss
                )
                total = terms.total + weighted_cooccurrence + weighted_separation
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
                totals["cooccurrence"] += float(cooccurrence.loss.detach())
                totals["topic_separation"] += float(separation.loss.detach())
                regularizer_diagnostics = {
                    **separation.diagnostics,
                    **cooccurrence.diagnostics,
                }
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
                include_npmi=True,
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
                "router_base": totals["router_base"] / max(router_batches, 1),
                "router_separation": totals["router_separation"]
                / max(router_batches, 1),
                "completion": totals["completion"]
                / max(router_batches + topic_updates, 1),
                "sinkhorn": totals["sinkhorn"] / max(router_batches, 1),
                "consistency": totals["consistency"] / max(router_batches, 1),
                "topic_base": totals["topic_base"] / max(topic_updates, 1),
                "local_decoder": totals["local_decoder"] / max(topic_updates, 1),
                "cooccurrence": totals["cooccurrence"] / max(topic_updates, 1),
                "topic_separation": totals["topic_separation"] / max(topic_updates, 1),
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
    selected = _selection(history, protocol)
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
        "cooccurrence_graph": graph_manifest,
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
