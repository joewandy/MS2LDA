# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, PLR2004
"""Exact-resume alternating training for K=200 and K=1000 stages."""

from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import (
    ViewPair,
    build_token_features,
    iter_row_batches,
    iter_sparse_batches,
    load_vocabulary,
    prototype_seeding_weights,
    sparse_batch,
)
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
    sparse_npmi,
    top_word_diversity,
)
from .model import (
    NeuralAssignmentMS2LDA,
    initialize_model,
    recycle_dead_prototypes,
    router_block_loss,
    topic_block_loss,
)
from .utils import (
    atomic_save_numpy,
    atomic_torch_save,
    file_sha256,
    peak_rss_bytes,
    read_json,
    write_json,
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
        """Create an empty queue."""
        return cls(capacity=int(capacity), heap=[])

    def add(
        self,
        losses: torch.Tensor,
        contexts: torch.Tensor,
        *,
        limit: int,
    ) -> None:
        """Add only the highest-loss contexts from one batch."""
        if not len(losses):
            return
        count = min(int(limit), len(losses))
        selected = torch.topk(losses.detach(), k=count, largest=True).indices
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
        """Remove and return the highest-loss contexts."""
        selected = heapq.nlargest(min(int(count), len(self.heap)), self.heap)
        selected_ids = {serial for _, serial, _ in selected}
        self.heap = [item for item in self.heap if item[1] not in selected_ids]
        heapq.heapify(self.heap)
        if not selected:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack([item[2] for item in selected])

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe exact queue state."""
        return {
            "capacity": self.capacity,
            "serial": self.serial,
            "items": [
                {"loss": loss, "serial": serial, "context": context}
                for loss, serial, context in self.heap
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HardContextQueue:
        """Restore an exact queue state."""
        queue = cls.empty(int(state["capacity"]))
        queue.serial = int(state["serial"])
        queue.heap = [
            (float(row["loss"]), int(row["serial"]), row["context"])
            for row in state["items"]
        ]
        heapq.heapify(queue.heap)
        return queue


def routing_temperature(
    epoch: int,
    *,
    attempt: str,
    rescue_mode: str | None,
    protocol: dict[str, Any],
) -> float:
    """Return the frozen annealed routing temperature."""
    primary = protocol["primary"]
    start = float(primary["routing_temperature_start"])
    end = float(primary["routing_temperature_end"])
    if attempt == "rescue" and rescue_mode in {"diffuse", "both"}:
        end = float(protocol["rescue"]["diffuse_rescue_final_temperature"])
    progress = min(
        max(epoch, 0) / max(float(primary["routing_temperature_anneal_epochs"]), 1.0),
        1.0,
    )
    if progress >= 1.0:
        return end
    return start + progress * (end - start)


def sinkhorn_weight(
    epoch: int,
    *,
    attempt: str,
    protocol: dict[str, Any],
) -> float:
    """Return the primary or predefined rescue balance schedule."""
    config = protocol["rescue"] if attempt == "rescue" else protocol["primary"]
    start = float(config["sinkhorn_weight_start"])
    hold = int(config["sinkhorn_weight_hold_epochs"])
    end = float(config["sinkhorn_weight_end"])
    end_epoch = int(config["sinkhorn_weight_end_epoch"])
    if epoch < hold:
        return start
    progress = min(max((epoch - hold) / max(end_epoch - hold, 1), 0.0), 1.0)
    if progress >= 1.0:
        return end
    return start + progress * (end - start)


def _top_k(
    *,
    attempt: str,
    rescue_mode: str | None,
    protocol: dict[str, Any],
) -> int:
    if attempt == "rescue" and rescue_mode in {"diffuse", "both"}:
        return int(protocol["rescue"]["diffuse_rescue_top_k"])
    return int(protocol["model"]["top_k"])


def prepare_token_features(
    run_dir: str | Path,
    *,
    counts_dir: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Combine the frozen SGNS output with m/z and token-type features."""
    directory = Path(run_dir)
    output = directory / "token_features"
    complete_path = output / "complete.json"
    features_path = output / "features.npy"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(features_path) != result["features_sha256"]:
            msg = "neural-assignment token features changed"
            raise ValueError(msg)
        return result
    embeddings = np.load(directory / "sgns/embeddings.npy")
    vocabulary = load_vocabulary(counts_dir)
    features = build_token_features(
        embeddings,
        vocabulary,
        protocol["token_features"],
    )
    atomic_save_numpy(features_path, features)
    result = {
        "schema_version": "neural-assignment-ms2lda/token-features-v1",
        "shape": list(features.shape),
        "training_split_only": True,
        "chemical_labels_used": False,
        "dreams_used": False,
        "features_sha256": file_sha256(features_path),
    }
    write_json(complete_path, result)
    return result


def prepare_initialization(
    run_dir: str | Path,
    *,
    label: str,
    num_topics: int,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Create one immutable data-only k-means++ initialization."""
    directory = Path(run_dir)
    output = directory / "initializations" / label
    complete_path = output / "complete.json"
    checkpoint_path = output / "model_initialization.pt"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(checkpoint_path) != result["checkpoint_sha256"]:
            msg = f"{label} initialization changed"
            raise ValueError(msg)
        return result
    features = torch.from_numpy(
        np.load(directory / "token_features/features.npy"),
    )
    model, indices = initialize_model(
        features,
        num_topics=int(num_topics),
        protocol=protocol,
        seeding_weights=prototype_seeding_weights(train),
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "neural-assignment-ms2lda/initialization-v1",
            "label": label,
            "num_topics": int(num_topics),
            "model": model.state_dict(),
            "topic_initial_indices": indices,
            "seed": int(protocol["seed"]),
            "method": (
                "deterministic_sqrt_cf_idf_squared_weighted_"
                "kmeans_plus_plus_seeding_only"
            ),
            "lloyd_iterations": 0,
            "classical_topic_teacher_used": False,
        },
    )
    result = {
        "schema_version": "neural-assignment-ms2lda/initialization-complete-v1",
        "label": label,
        "num_topics": int(num_topics),
        "shared_by_primary_and_rescue": label == "k1000",
        "data_only": True,
        "lloyd_iterations": 0,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    write_json(complete_path, result)
    return result


def _fresh_model(
    run_dir: Path,
    *,
    initialization_label: str,
    num_topics: int,
    protocol: dict[str, Any],
) -> NeuralAssignmentMS2LDA:
    features = torch.from_numpy(np.load(run_dir / "token_features/features.npy"))
    checkpoint = torch.load(
        run_dir / "initializations" / initialization_label / "model_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_config = protocol["model"]
    model = NeuralAssignmentMS2LDA(
        features,
        num_topics=int(num_topics),
        projection_dimensions=int(model_config["projection_dimensions"]),
        router_hidden_dimensions=int(model_config["router_hidden_dimensions"]),
        beta_temperature=float(model_config["beta_temperature"]),
        topic_initial_indices=checkpoint["topic_initial_indices"],
        seed=int(protocol["seed"]) + int(num_topics),
    )
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def infer_theta(
    model: NeuralAssignmentMS2LDA,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    temperature: float,
    top_k: int,
    with_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Infer every mixture with exactly one deterministic routing pass."""
    model.eval()
    projected = model.projected_tokens()
    rows = []
    assignment_entropies = []
    selected_masses = []
    for batch in iter_sparse_batches(
        matrix,
        batch_size=int(batch_size),
        shuffle=False,
        seed=0,
    ):
        output = model.route(
            batch,
            temperature=float(temperature),
            top_k=int(top_k),
            straight_through=False,
            projected_tokens=projected,
        )
        rows.append(output.theta.cpu().numpy().astype(np.float32))
        if with_diagnostics:
            probabilities = output.assignments.clamp_min(1e-12)
            assignment_entropies.append(
                (
                    -torch.sum(
                        output.assignments * torch.log(probabilities),
                        dim=1,
                    )
                )
                .cpu()
                .numpy(),
            )
            selected_masses.append(
                torch.max(output.assignments, dim=1).values.cpu().numpy(),
            )
    theta = np.concatenate(rows, axis=0)
    if not with_diagnostics:
        return theta
    entropy = np.concatenate(assignment_entropies)
    mass = np.concatenate(selected_masses)
    return theta, {
        "routing_passes_per_spectrum": 1,
        "local_vb_steps": 0,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "assignment_entropy_mean": float(np.mean(entropy)),
        "maximum_assignment_mass_mean": float(np.mean(mass)),
        "nonzero_assignments_per_observed_token": int(top_k),
    }


@torch.inference_mode()
def validation_metrics(
    model: NeuralAssignmentMS2LDA,
    *,
    train: sp.csr_matrix,
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_full: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    batch_size: int,
    temperature: float,
    top_k: int,
    include_npmi: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Calculate selection metrics without reading the test matrices."""
    model.eval()
    beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
    observed_theta, routing = infer_theta(
        model,
        validation_observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
        with_diagnostics=True,
    )
    full_theta = infer_theta(
        model,
        validation_full,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    completion, _ = completion_metrics(
        observed_theta,
        beta,
        validation_completion,
        validation_records,
    )
    num_topics = model.num_topics
    result = {
        "document_completion": completion,
        "active_topics": active_topic_metrics(
            observed_theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"],
            ),
            corpus_threshold=1.0 / num_topics,
        ),
        "mixture_diagnostics": effective_topic_summary(full_theta),
        "top_word_diversity": top_word_diversity(
            beta,
            top_n=int(protocol["evaluation"]["topic_top_n"]),
        ),
        "routing": routing,
        "topic_usage": {
            "minimum": float(np.min(full_theta.mean(axis=0))),
            "median": float(np.median(full_theta.mean(axis=0))),
            "maximum": float(np.max(full_theta.mean(axis=0))),
        },
        "_usage": full_theta.mean(axis=0).astype(np.float32),
    }
    if include_npmi:
        result["word_cooccurrence_npmi"] = sparse_npmi(
            beta,
            train,
            top_n=int(protocol["evaluation"]["topic_top_n"]),
        )
    return result


def gate_checks(
    metrics: dict[str, Any],
    *,
    stage: str,
    stable: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen K=200 or K=1000 validation gate."""
    if stage not in {"k200", "k1000"}:
        msg = "real-data stage must be k200 or k1000"
        raise ValueError(msg)
    config = protocol[f"{stage}_gates"]
    active = int(metrics["active_topics"]["corpus_active_topics"])
    diversity = float(metrics["top_word_diversity"])
    effective = float(
        metrics["mixture_diagnostics"]["effective_topic_count_median"],
    )
    nll = float(metrics["document_completion"]["nll_per_token"])
    checks: dict[str, dict[str, Any]] = {
        "stable": {"pass": bool(stable), "actual": bool(stable), "required": True},
        "active_topics": {
            "pass": active >= int(config["minimum_active_topics"]),
            "actual": active,
            "minimum": int(config["minimum_active_topics"]),
        },
        "top_word_diversity": {
            "pass": diversity >= float(config["minimum_top_word_diversity"]),
            "actual": diversity,
            "minimum": float(config["minimum_top_word_diversity"]),
        },
        "median_effective_topics": {
            "pass": float(config["minimum_median_effective_topics"])
            <= effective
            <= float(config["maximum_median_effective_topics"]),
            "actual": effective,
            "minimum": float(config["minimum_median_effective_topics"]),
            "maximum": float(config["maximum_median_effective_topics"]),
        },
        "validation_nll": {
            "pass": nll <= float(config["maximum_validation_nll"]),
            "actual": nll,
            "maximum": float(config["maximum_validation_nll"]),
        },
    }
    if stage == "k200":
        npmi = float(metrics["word_cooccurrence_npmi"]["mean_npmi"])
        checks["mean_npmi"] = {
            "pass": npmi >= float(config["minimum_mean_npmi"]),
            "actual": npmi,
            "minimum": float(config["minimum_mean_npmi"]),
        }
    failed = [name for name, row in checks.items() if not row["pass"]]
    amendment = protocol.get("exploratory_amendment", {})
    waived = (
        list(amendment.get("waived_k200_blocking_failures", []))
        if stage == "k200"
        else []
    )
    unexpected_waivers = sorted(set(waived) - set(checks))
    if unexpected_waivers:
        msg = f"waiver names an unknown K=200 gate: {unexpected_waivers[0]}"
        raise ValueError(msg)
    waived_failures = [name for name in failed if name in waived]
    blocking_failures = [name for name in failed if name not in waived]
    return {
        "stage": stage,
        "checks": checks,
        "raw_pass": not failed,
        "pass": not blocking_failures,
        "failed": failed,
        "waived_failures": waived_failures,
        "blocking_failures": blocking_failures,
        "exploratory_amendment_id": (amendment.get("id") if waived_failures else None),
    }


def diagnose_collapse(
    validation_gate: dict[str, Any],
) -> tuple[bool, str | None, list[str]]:
    """Map validation failures onto the one permitted rescue type."""
    failed = set(validation_gate["failed"])
    collapse_failures = failed & {
        "active_topics",
        "top_word_diversity",
        "median_effective_topics",
    }
    if not collapse_failures:
        return False, None, []
    effective = validation_gate["checks"]["median_effective_topics"]
    diffuse = not effective["pass"] and float(effective["actual"]) > float(
        effective["maximum"],
    )
    underuse = bool(
        collapse_failures & {"active_topics", "top_word_diversity"}
        or (
            not effective["pass"]
            and float(effective["actual"]) < float(effective["minimum"])
        ),
    )
    mode = "both" if diffuse and underuse else "diffuse" if diffuse else "underuse"
    return True, mode, sorted(collapse_failures)


def _entry_losses(
    output: Any,
    beta: torch.Tensor,
) -> torch.Tensor:
    selected_topics = output.theta[output.row_ids]
    selected_words = beta[:, output.token_indices].T
    probability = torch.sum(selected_topics * selected_words, dim=1).clamp_min(1e-12)
    return -torch.log(probability)


def _recycling_config(
    attempt: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return protocol["rescue"] if attempt == "rescue" else protocol["primary"]


def _checkpoint_payload(
    *,
    model: NeuralAssignmentMS2LDA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    history: list[dict[str, Any]],
    best_key: tuple[int, float] | None,
    best_epoch: int | None,
    bad_validations: int,
    underuse_streak: np.ndarray,
    recycle_counts: np.ndarray,
    recycle_events: list[dict[str, Any]],
    context_queue: HardContextQueue,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "best_key": list(best_key) if best_key is not None else None,
        "best_epoch": best_epoch,
        "bad_validations": bad_validations,
        "underuse_streak": underuse_streak,
        "recycle_counts": recycle_counts,
        "recycle_events": recycle_events,
        "context_queue": context_queue.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }


def train_attempt(
    run_dir: str | Path,
    *,
    stage: str,
    attempt: str,
    initialization_label: str,
    train: sp.csr_matrix,
    views: list[ViewPair],
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_full: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    protocol: dict[str, Any],
    batch_size: int,
    rescue_mode: str | None = None,
) -> dict[str, Any]:
    """Train one bounded real-data attempt with exact epoch-level resume."""
    if stage not in {"k200", "k1000"}:
        msg = "training stage must be k200 or k1000"
        raise ValueError(msg)
    if attempt not in {"primary", "rescue"}:
        msg = "attempt must be primary or rescue"
        raise ValueError(msg)
    if attempt == "rescue" and stage != "k1000":
        msg = "the rescue is available only at K=1000"
        raise ValueError(msg)
    directory = Path(run_dir)
    output = directory / "stages" / stage / "attempts" / attempt
    complete_path = output / "complete.json"
    model_path = output / "model.pt"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(model_path) != result["model_sha256"]:
            msg = f"completed {stage}/{attempt} model changed"
            raise ValueError(msg)
        return result
    output.mkdir(parents=True, exist_ok=True)

    stage_config = protocol["stages"][stage]
    num_topics = int(stage_config["num_topics"])
    seed = int(protocol["seed"])
    torch.manual_seed(seed)
    model = _fresh_model(
        directory,
        initialization_label=initialization_label,
        num_topics=num_topics,
        protocol=protocol,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["optimization"]["learning_rate"]),
        weight_decay=float(protocol["optimization"]["weight_decay"]),
    )
    latest_path = output / "checkpoint_latest.pt"
    best_path = output / "checkpoint_best.pt"
    epoch_start = 0
    global_step = 0
    elapsed_before = 0.0
    history: list[dict[str, Any]] = []
    best_key: tuple[int, float] | None = None
    best_epoch: int | None = None
    bad_validations = 0
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
        best_key = (
            tuple(checkpoint["best_key"])
            if checkpoint["best_key"] is not None
            else None
        )
        best_epoch = checkpoint["best_epoch"]
        bad_validations = int(checkpoint["bad_validations"])
        underuse_streak = checkpoint["underuse_streak"]
        recycle_counts = checkpoint["recycle_counts"]
        recycle_events = list(checkpoint["recycle_events"])
        context_queue = HardContextQueue.from_state_dict(
            checkpoint["context_queue"],
        )
        torch.set_rng_state(checkpoint["torch_rng_state"])

    started = time.perf_counter()
    maximum_seconds = float(stage_config["maximum_hours"]) * 3600.0
    stable = True
    stop_reason = "maximum_epochs"
    epochs_completed = epoch_start
    selected_validation: dict[str, Any] | None = None
    selected_gate: dict[str, Any] | None = None
    model_config = protocol["model"]
    optimization = protocol["optimization"]
    recycle_config = _recycling_config(attempt, protocol)
    chosen_top_k = _top_k(
        attempt=attempt,
        rescue_mode=rescue_mode,
        protocol=protocol,
    )

    for epoch in range(epoch_start, int(stage_config["maximum_epochs"])):
        model.train()
        pair = views[epoch % len(views)]
        temperature = routing_temperature(
            epoch,
            attempt=attempt,
            rescue_mode=rescue_mode,
            protocol=protocol,
        )
        balance_weight = sinkhorn_weight(
            epoch,
            attempt=attempt,
            protocol=protocol,
        )
        totals = {
            "router_total": 0.0,
            "completion": 0.0,
            "sinkhorn": 0.0,
            "consistency": 0.0,
            "topic_total": 0.0,
            "local_decoder": 0.0,
        }
        router_batches = 0
        topic_updates = 0
        with torch.no_grad():
            cached_beta = model.topic_word_distribution().detach()
        for rows in iter_row_batches(
            train.shape[0],
            batch_size=int(batch_size),
            shuffle=True,
            seed=seed + epoch,
        ):
            elapsed = elapsed_before + time.perf_counter() - started
            if elapsed >= maximum_seconds:
                stop_reason = "wall_clock_cap"
                break
            left_batch = sparse_batch(pair.left, rows)
            right_batch = sparse_batch(pair.right, rows)
            optimizer.zero_grad(set_to_none=True)
            terms = router_block_loss(
                model,
                left_batch,
                right_batch,
                cached_beta=cached_beta,
                temperature=temperature,
                top_k=chosen_top_k,
                sinkhorn_weight=balance_weight,
                consistency_weight=float(
                    optimization["theta_consistency_weight"],
                ),
                sinkhorn_epsilon=float(model_config["sinkhorn_epsilon"]),
                sinkhorn_iterations=int(model_config["sinkhorn_iterations"]),
            )
            if not torch.isfinite(terms.total):
                stable = False
                stop_reason = "non_finite_router_loss"
                break
            terms.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(model_config["gradient_clip_norm"]),
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

        elapsed = elapsed_before + time.perf_counter() - started
        if elapsed >= maximum_seconds:
            stop_reason = "wall_clock_cap"
        if stable and stop_reason != "wall_clock_cap":
            topic_rows = list(
                iter_row_batches(
                    train.shape[0],
                    batch_size=int(optimization["topic_update_batch_size"]),
                    shuffle=True,
                    seed=seed + 100_003 + epoch,
                ),
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
                    top_k=chosen_top_k,
                    local_decoder_weight=float(
                        optimization["local_decoder_weight"],
                    ),
                )
                if not torch.isfinite(terms.total):
                    stable = False
                    stop_reason = "non_finite_topic_loss"
                    break
                terms.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(model_config["gradient_clip_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    stable = False
                    stop_reason = "non_finite_topic_gradient"
                    break
                optimizer.step()
                totals["topic_total"] += float(terms.total.detach())
                totals["completion"] += float(terms.completion.detach())
                totals["local_decoder"] += float(terms.local_decoder.detach())
                topic_updates += 1
                global_step += 1

        epochs_completed = epoch + 1
        validation = None
        gate = None
        recycled: list[int] = []
        if stable and epochs_completed % int(stage_config["validation_interval"]) == 0:
            final_temperature = routing_temperature(
                epochs_completed,
                attempt=attempt,
                rescue_mode=rescue_mode,
                protocol=protocol,
            )
            validation = validation_metrics(
                model,
                train=train,
                validation_observed=validation_observed,
                validation_completion=validation_completion,
                validation_full=validation_full,
                validation_records=validation_records,
                batch_size=int(batch_size),
                temperature=final_temperature,
                top_k=chosen_top_k,
                include_npmi=stage == "k200",
                protocol=protocol,
            )
            usage = np.asarray(validation.pop("_usage"), dtype=np.float64)
            gate = gate_checks(
                validation,
                stage=stage,
                stable=stable,
                protocol=protocol,
            )
            failures = len(gate["blocking_failures"])
            nll = float(validation["document_completion"]["nll_per_token"])
            candidate_key = (failures, nll)
            improved = (
                best_key is None
                or candidate_key[0] < best_key[0]
                or (
                    candidate_key[0] == best_key[0]
                    and candidate_key[1]
                    < best_key[1]
                    - float(
                        optimization["minimum_nll_improvement"],
                    )
                )
            )
            if improved:
                best_key = candidate_key
                best_epoch = epochs_completed
                bad_validations = 0
                selected_validation = validation
                selected_gate = gate
                atomic_torch_save(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "epoch": epochs_completed,
                        "validation": validation,
                        "gate": gate,
                    },
                )
            else:
                bad_validations += 1

            # Recycling changes the state used by the next epoch only. The
            # selected checkpoint above therefore matches its recorded
            # pre-recycling validation metrics exactly.
            underused = usage < (
                float(recycle_config["recycle_usage_fraction_of_uniform"]) / num_topics
            )
            underuse_streak[underused] += 1
            underuse_streak[~underused] = 0
            eligible = np.flatnonzero(
                (underuse_streak >= int(recycle_config["recycle_patience_validations"]))
                & (recycle_counts < int(recycle_config["maximum_recycles_per_topic"])),
            )
            if (
                len(eligible)
                and epochs_completed <= int(recycle_config["recycle_through_epoch"])
                and len(context_queue.heap)
            ):
                ordered = eligible[np.lexsort((eligible, usage[eligible]))]
                replacements = context_queue.pop_highest(len(ordered))
                ordered = ordered[: len(replacements)]
                if len(ordered):
                    topic_indices = torch.from_numpy(
                        ordered.astype(np.int64, copy=False),
                    )
                    recycle_dead_prototypes(
                        model,
                        optimizer,
                        topic_indices=topic_indices,
                        replacements=replacements,
                    )
                    recycled = ordered.tolist()
                    recycle_counts[ordered] += 1
                    underuse_streak[ordered] = 0
                    recycle_events.append(
                        {
                            "epoch": epochs_completed,
                            "topic_indices": recycled,
                            "usage_before": usage[ordered].tolist(),
                            "queue_remaining": len(context_queue.heap),
                        },
                    )

        elapsed = elapsed_before + time.perf_counter() - started
        row = {
            "epoch": epochs_completed,
            "global_step": global_step,
            "view_pair": pair.pair_index,
            "router_batches": router_batches,
            "topic_updates": topic_updates,
            "elapsed_seconds": elapsed,
            "routing_temperature": temperature,
            "sinkhorn_weight": balance_weight,
            "top_k": chosen_top_k,
            "losses": {
                "router_total": totals["router_total"] / max(router_batches, 1),
                "completion": totals["completion"]
                / max(router_batches + topic_updates, 1),
                "sinkhorn": totals["sinkhorn"] / max(router_batches, 1),
                "consistency": totals["consistency"] / max(router_batches, 1),
                "topic_total": totals["topic_total"] / max(topic_updates, 1),
                "local_decoder": totals["local_decoder"] / max(topic_updates, 1),
            },
            "validation": validation,
            "gate": gate,
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
                best_key=best_key,
                best_epoch=best_epoch,
                bad_validations=bad_validations,
                underuse_streak=underuse_streak,
                recycle_counts=recycle_counts,
                recycle_events=recycle_events,
                context_queue=context_queue,
            ),
        )
        write_json(output / "history.json", history)
        write_json(
            directory / "heartbeat.json",
            {
                "stage": f"train_{stage}_{attempt}",
                "epoch": epochs_completed,
                "maximum_epochs": int(stage_config["maximum_epochs"]),
                "elapsed_seconds": elapsed,
                "maximum_seconds": maximum_seconds,
                "stable": stable,
                "pid": os.getpid(),
                "torch_cpu_threads": torch.get_num_threads(),
            },
        )
        if not stable or stop_reason == "wall_clock_cap":
            break
        if epochs_completed >= int(
            stage_config["minimum_epochs"],
        ) and bad_validations >= int(stage_config["early_stopping_patience"]):
            stop_reason = "early_stopping"
            break

    if best_path.is_file():
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model"])
        selected_validation = selected["validation"]
        selected_gate = selected["gate"]
        best_epoch = int(selected["epoch"])
    else:
        final_temperature = routing_temperature(
            epochs_completed,
            attempt=attempt,
            rescue_mode=rescue_mode,
            protocol=protocol,
        )
        selected_validation = validation_metrics(
            model,
            train=train,
            validation_observed=validation_observed,
            validation_completion=validation_completion,
            validation_full=validation_full,
            validation_records=validation_records,
            batch_size=int(batch_size),
            temperature=final_temperature,
            top_k=chosen_top_k,
            include_npmi=stage == "k200",
            protocol=protocol,
        )
        selected_validation.pop("_usage")
        selected_gate = gate_checks(
            selected_validation,
            stage=stage,
            stable=stable,
            protocol=protocol,
        )
        best_epoch = epochs_completed

    atomic_torch_save(
        model_path,
        {
            "schema_version": "neural-assignment-ms2lda/model-v1",
            "stage": stage,
            "attempt": attempt,
            "rescue_mode": rescue_mode,
            "model": model.state_dict(),
            "selected_epoch": best_epoch,
            "selected_validation": selected_validation,
            "validation_gate": selected_gate,
            "routing_temperature": routing_temperature(
                int(best_epoch),
                attempt=attempt,
                rescue_mode=rescue_mode,
                protocol=protocol,
            ),
            "top_k": chosen_top_k,
            "encoder_passes_per_representation": 1,
            "local_vb_steps": 0,
        },
    )
    result = {
        "schema_version": "neural-assignment-ms2lda/training-complete-v1",
        "stage": stage,
        "attempt": attempt,
        "rescue_mode": rescue_mode,
        "stable": stable,
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "selected_epoch": best_epoch,
        "global_steps": global_step,
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "maximum_hours": float(stage_config["maximum_hours"]),
        "training_cpu_threads": int(protocol["training_cpu_threads"]),
        "batch_size": int(batch_size),
        "topic_updates_per_epoch": int(
            protocol["optimization"]["topic_updates_per_epoch"],
        ),
        "selected_validation": selected_validation,
        "validation_gate": selected_gate,
        "recycle_events": recycle_events,
        "recycle_count_total": int(recycle_counts.sum()),
        "maximum_recycles_for_any_topic": int(recycle_counts.max(initial=0)),
        "routing": {
            "top_k": chosen_top_k,
            "one_pass": True,
            "local_vb_steps": 0,
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "model_sha256": file_sha256(model_path),
    }
    write_json(complete_path, result)
    return result


def load_attempt_model(
    run_dir: str | Path,
    *,
    stage: str,
    attempt: str,
    initialization_label: str,
    protocol: dict[str, Any],
) -> tuple[NeuralAssignmentMS2LDA, dict[str, Any]]:
    """Load and verify one selected attempt checkpoint."""
    directory = Path(run_dir)
    complete = read_json(
        directory / "stages" / stage / "attempts" / attempt / "complete.json",
    )
    path = directory / "stages" / stage / "attempts" / attempt / "model.pt"
    if file_sha256(path) != complete["model_sha256"]:
        msg = f"{stage}/{attempt} selected model changed"
        raise ValueError(msg)
    num_topics = int(protocol["stages"][stage]["num_topics"])
    model = _fresh_model(
        directory,
        initialization_label=initialization_label,
        num_topics=num_topics,
        protocol=protocol,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def benchmark_batch_sizes(
    run_dir: str | Path,
    *,
    views: list[ViewPair],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Choose a parity-checked batch size by full router-block throughput."""
    directory = Path(run_dir)
    output = directory / "batch_benchmark.json"
    if output.is_file():
        return read_json(output)
    model = _fresh_model(
        directory,
        initialization_label="k1000",
        num_topics=int(protocol["stages"]["k1000"]["num_topics"]),
        protocol=protocol,
    )
    model.train()
    candidates = list(map(int, protocol["optimization"]["batch_size_candidates"]))
    maximum_documents = min(max(candidates), views[0].left.shape[0])
    subset = views[0].left[:maximum_documents].tocsr()
    temperature = float(protocol["primary"]["routing_temperature_start"])
    top_k = int(protocol["model"]["top_k"])
    reference_theta = None
    parity = []
    for batch_size in candidates:
        with torch.inference_mode():
            theta = infer_theta(
                model,
                subset,
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
            )
        if reference_theta is None:
            reference_theta = theta
        np.testing.assert_allclose(theta, reference_theta, rtol=0, atol=1e-6)
        parity.append(
            {
                "batch_size": batch_size,
                "maximum_absolute_error": float(
                    np.max(np.abs(theta - reference_theta)),
                ),
            },
        )

    model.train()
    rows = []
    pair = views[0]
    for batch_size in candidates:
        selected_rows = np.arange(batch_size, dtype=np.int64)
        left_batch = sparse_batch(pair.left, selected_rows)
        right_batch = sparse_batch(pair.right, selected_rows)
        elapsed = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                cached_beta = model.topic_word_distribution().detach()
            started = time.perf_counter()
            terms = router_block_loss(
                model,
                left_batch,
                right_batch,
                cached_beta=cached_beta,
                temperature=temperature,
                top_k=top_k,
                sinkhorn_weight=float(
                    protocol["primary"]["sinkhorn_weight_start"],
                ),
                consistency_weight=float(
                    protocol["optimization"]["theta_consistency_weight"],
                ),
                sinkhorn_epsilon=float(
                    protocol["model"]["sinkhorn_epsilon"],
                ),
                sinkhorn_iterations=int(
                    protocol["model"]["sinkhorn_iterations"],
                ),
            )
            terms.total.backward()
            elapsed.append(time.perf_counter() - started)
        median_elapsed = float(np.median(elapsed))
        rss = peak_rss_bytes()
        rows.append(
            {
                "batch_size": batch_size,
                "documents": batch_size,
                "timed_router_blocks": len(elapsed),
                "median_elapsed_seconds": median_elapsed,
                "documents_per_second": batch_size / max(median_elapsed, 1e-12),
                "peak_rss_bytes": rss,
                "within_rss_limit": rss
                <= int(protocol["optimization"]["maximum_rss_bytes"]),
            },
        )
    eligible = [row for row in rows if row["within_rss_limit"]]
    if not eligible:
        msg = "no candidate batch size remained below the 8 GB RSS gate"
        raise MemoryError(msg)
    selected = max(eligible, key=lambda row: row["documents_per_second"])
    result = {
        "schema_version": "neural-assignment-ms2lda/batch-benchmark-v1",
        "candidates": rows,
        "inference_parity": parity,
        "selected_batch_size": int(selected["batch_size"]),
        "selection": "fastest_full_router_block_below_8gb_rss",
        "model_outputs_equal_across_candidates": True,
    }
    write_json(output, result)
    return result
