# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, PLR2004
"""Resumable primary and collapse-only rescue training."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import (
    build_token_features,
    iter_sparse_batches,
    load_vocabulary,
)
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
    top_word_diversity,
)
from .model import NeuralMS2LDA, initialize_model
from .utils import (
    append_jsonl,
    atomic_save_numpy,
    atomic_torch_save,
    file_sha256,
    peak_rss_bytes,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    import scipy.sparse as sp


def prepare_initialization(
    run_dir: str | Path,
    counts_dir: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build fixed 64-D features and the one shared random initialization."""
    directory = Path(run_dir)
    output = directory / "initialization"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"neural initialization changed: {name}"
                raise ValueError(msg)
        return result
    embeddings = np.load(directory / "sgns/embeddings.npy")
    vocabulary = load_vocabulary(counts_dir)
    features = build_token_features(
        embeddings,
        vocabulary,
        protocol["token_features"],
    )
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "token_features.npy"
    atomic_save_numpy(features_path, features)
    model, topic_indices = initialize_model(torch.from_numpy(features), protocol)
    checkpoint_path = output / "model_initialization.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "fully-neural-ms2lda/initialization-v1",
            "model": model.state_dict(),
            "topic_initial_indices": topic_indices,
            "seed": int(protocol["seed"]),
            "data_only_topic_initialization": True,
            "tomotopy_or_nmf_warm_start": False,
        },
    )
    result = {
        "schema_version": "fully-neural-ms2lda/initialization-complete-v1",
        "features": list(features.shape),
        "topics": int(protocol["num_topics"]),
        "shared_by_primary_and_rescue": True,
        "data_only_random_token_centers": True,
        "output_sha256": {
            features_path.name: file_sha256(features_path),
            checkpoint_path.name: file_sha256(checkpoint_path),
        },
    }
    write_json(complete_path, result)
    return result


def _fresh_model(run_dir: Path, protocol: dict[str, Any]) -> NeuralMS2LDA:
    features = torch.from_numpy(np.load(run_dir / "initialization/token_features.npy"))
    model, _ = initialize_model(features, protocol)
    checkpoint = torch.load(
        run_dir / "initialization/model_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model


def _ecr_order(vocabulary_size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 811).permutation(vocabulary_size)


def _ecr_block(order: np.ndarray, step: int, size: int) -> torch.Tensor:
    positions = (np.arange(size, dtype=np.int64) + step * size) % len(order)
    return torch.from_numpy(order[positions])


@torch.inference_mode()
def infer_theta(
    model: NeuralMS2LDA,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    projected_tokens: torch.Tensor | None = None,
) -> np.ndarray:
    """Infer every mixture with exactly one deterministic encoder pass."""
    model.eval()
    tokens = model.projected_tokens() if projected_tokens is None else projected_tokens
    rows = []
    for batch in iter_sparse_batches(
        matrix,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        theta, _, _ = model.encode(
            batch,
            sample=False,
            projected_tokens=tokens,
        )
        rows.append(theta.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


@torch.inference_mode()
def _validation_metrics(
    model: NeuralMS2LDA,
    observed: sp.csr_matrix,
    completion: sp.csr_matrix,
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    theta = infer_theta(
        model,
        observed,
        batch_size=int(protocol["training"]["batch_size"]),
    )
    beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
    completion_result, _ = completion_metrics(theta, beta, completion, records)
    evaluation = protocol["evaluation"]
    return {
        "document_completion": completion_result,
        "active_topics": active_topic_metrics(
            theta,
            document_threshold=float(evaluation["document_active_threshold"]),
            corpus_threshold=float(evaluation["corpus_active_threshold"]),
        ),
        "mixture_diagnostics": effective_topic_summary(theta),
        "top_word_diversity": top_word_diversity(
            beta,
            top_n=int(evaluation["topic_top_n"]),
        ),
    }


def _checkpoint_payload(
    *,
    model: NeuralMS2LDA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    history: list[dict[str, Any]],
    best_key: tuple[int, float] | None,
    best_epoch: int | None,
    bad_validations: int,
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
        "torch_rng_state": torch.get_rng_state(),
    }


def train_attempt(
    run_dir: str | Path,
    *,
    attempt: str,
    train: sp.csr_matrix,
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    protocol: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Train one bounded attempt, resuming only from its own epoch checkpoints."""
    if attempt not in {"primary", "rescue"}:
        msg = "attempt must be primary or rescue"
        raise ValueError(msg)
    directory = Path(run_dir)
    output = directory / "attempts" / attempt
    complete_path = output / "complete.json"
    model_path = output / "model.pt"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(model_path) != result["model_sha256"]:
            msg = f"completed {attempt} model changed"
            raise ValueError(msg)
        return result
    output.mkdir(parents=True, exist_ok=True)
    seed = int(protocol["seed"])
    torch.manual_seed(seed)
    model = _fresh_model(directory, protocol)
    training = protocol["training"]
    model_config = protocol["model"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
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
        torch.set_rng_state(checkpoint["torch_rng_state"])
    ecr_order = _ecr_order(train.shape[1], seed)
    usage_weight = (
        float(protocol["rescue"]["usage_guard_weight"]) if attempt == "rescue" else 0.0
    )
    sparsity_weight = (
        float(protocol["rescue"]["sparsity_guard_weight"])
        if attempt == "rescue"
        else 0.0
    )
    started = time.perf_counter()
    maximum_seconds = float(training["maximum_hours_per_attempt"]) * 3600.0
    stable = True
    stop_reason = "maximum_epochs"
    epochs_completed = epoch_start
    for epoch in range(epoch_start, int(training["maximum_epochs"])):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "ecr": 0.0,
            "usage_guard": 0.0,
            "sparsity_guard": 0.0,
            "theta_entropy": 0.0,
        }
        encoder_batches = 0
        decoder_updates = 0
        with torch.no_grad():
            cached_tokens = model.projected_tokens().detach()
            cached_beta = model.topic_word_distribution(cached_tokens).detach()
        for batch in iter_sparse_batches(
            train,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            elapsed = elapsed_before + time.perf_counter() - started
            if elapsed >= maximum_seconds:
                stop_reason = "wall_clock_cap"
                break
            optimizer.zero_grad(set_to_none=True)
            kl_weight = min(
                1.0,
                (epoch + 1) / max(float(model_config["kl_warmup_epochs"]), 1.0),
            )
            terms = model.encoder_loss(
                batch,
                beta=cached_beta,
                projected_tokens=cached_tokens,
                kl_weight=kl_weight,
                usage_guard_weight=usage_weight,
                sparsity_guard_weight=sparsity_weight,
                target_effective_topics=float(
                    protocol["rescue"]["target_effective_topics"],
                ),
            )
            if not torch.isfinite(terms.total):
                stable = False
                stop_reason = "non_finite_loss"
                break
            terms.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(model_config["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                stable = False
                stop_reason = "non_finite_gradient"
                break
            optimizer.step()
            totals["loss"] += float(terms.total.detach())
            totals["reconstruction"] += float(terms.reconstruction.detach())
            totals["kl"] += float(terms.kl.detach())
            totals["usage_guard"] += float(terms.usage_guard.detach())
            totals["sparsity_guard"] += float(terms.sparsity_guard.detach())
            totals["theta_entropy"] += float(terms.theta_entropy.detach())
            encoder_batches += 1
            global_step += 1
        elapsed = elapsed_before + time.perf_counter() - started
        if elapsed >= maximum_seconds:
            stop_reason = "wall_clock_cap"
        if stable and stop_reason != "wall_clock_cap":
            topic_batches = iter_sparse_batches(
                train,
                batch_size=int(training["topic_update_batch_size"]),
                shuffle=True,
                seed=seed + 100_003 + epoch,
            )
            for topic_update in range(int(training["topic_updates_per_epoch"])):
                try:
                    topic_batch = next(topic_batches)
                except StopIteration:
                    break
                with torch.no_grad():
                    theta, _, _ = model.encode(
                        topic_batch,
                        sample=False,
                        projected_tokens=cached_tokens,
                    )
                optimizer.zero_grad(set_to_none=True)
                block = _ecr_block(
                    ecr_order,
                    epoch * int(training["topic_updates_per_epoch"]) + topic_update,
                    int(model_config["ecr_vocabulary_block"]),
                )
                decoder_total, decoder_reconstruction, decoder_ecr = model.decoder_loss(
                    topic_batch,
                    theta=theta.detach(),
                    ecr_token_indices=block,
                    ecr_weight=float(model_config["ecr_weight"]),
                    ecr_epsilon=float(model_config["ecr_epsilon"]),
                    ecr_iterations=int(model_config["ecr_iterations"]),
                )
                if not torch.isfinite(decoder_total):
                    stable = False
                    stop_reason = "non_finite_decoder_loss"
                    break
                decoder_total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(model_config["gradient_clip_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    stable = False
                    stop_reason = "non_finite_decoder_gradient"
                    break
                optimizer.step()
                totals["loss"] += float(decoder_total.detach())
                totals["reconstruction"] += float(decoder_reconstruction.detach())
                totals["ecr"] += float(decoder_ecr.detach())
                decoder_updates += 1
                global_step += 1
        epochs_completed = epoch + 1
        elapsed = elapsed_before + time.perf_counter() - started
        validation = None
        if stable and epochs_completed % int(training["validation_interval"]) == 0:
            validation = _validation_metrics(
                model,
                validation_observed,
                validation_completion,
                validation_records,
                protocol,
            )
            active = int(validation["active_topics"]["corpus_active_topics"])
            diversity = float(validation["top_word_diversity"])
            effective = float(
                validation["mixture_diagnostics"]["effective_topic_count_median"],
            )
            reference_effective = float(
                reference["full_spectrum_mixture"]["effective_topic_count_median"],
            )
            gates = protocol["hard_viability_gates"]
            tier = int(
                active < int(training["selection_minimum_active_topics"])
                or diversity < float(training["selection_minimum_diversity"])
                or effective
                < reference_effective
                * float(gates["effective_topics_reference_minimum_fraction"])
                or effective
                > reference_effective
                * float(gates["effective_topics_reference_maximum_multiple"]),
            )
            nll = float(validation["document_completion"]["nll_per_token"])
            candidate_key = (tier, nll)
            improved = (
                best_key is None
                or candidate_key[0] < best_key[0]
                or (
                    candidate_key[0] == best_key[0]
                    and candidate_key[1]
                    < best_key[1] - float(training["minimum_nll_improvement"])
                )
            )
            if improved:
                best_key = candidate_key
                best_epoch = epochs_completed
                bad_validations = 0
                atomic_torch_save(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "epoch": epochs_completed,
                        "validation": validation,
                    },
                )
            else:
                bad_validations += 1
        row = {
            "epoch": epochs_completed,
            "global_step": global_step,
            "encoder_batches": encoder_batches,
            "decoder_updates": decoder_updates,
            "elapsed_seconds": elapsed,
            "training": {
                "loss_per_optimizer_step": totals["loss"]
                / max(encoder_batches + decoder_updates, 1),
                "reconstruction_per_optimizer_step": totals["reconstruction"]
                / max(encoder_batches + decoder_updates, 1),
                "kl_per_encoder_batch": totals["kl"] / max(encoder_batches, 1),
                "ecr_per_decoder_update": totals["ecr"] / max(decoder_updates, 1),
                "usage_guard_per_encoder_batch": totals["usage_guard"]
                / max(encoder_batches, 1),
                "sparsity_guard_per_encoder_batch": totals["sparsity_guard"]
                / max(encoder_batches, 1),
                "theta_entropy_per_encoder_batch": totals["theta_entropy"]
                / max(encoder_batches, 1),
            },
            "validation": validation,
            "stable": stable,
        }
        history.append(row)
        append_jsonl(output / "history.jsonl", row)
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
            ),
        )
        write_json(
            directory / "heartbeat.json",
            {
                "stage": f"train_{attempt}",
                "attempt": attempt,
                "epoch": epochs_completed,
                "maximum_epochs": int(training["maximum_epochs"]),
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
            training["minimum_epochs"],
        ) and bad_validations >= int(training["early_stopping_patience"]):
            stop_reason = "early_stopping"
            break
    if best_path.is_file():
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model"])
        selected_validation = selected["validation"]
    else:
        selected_validation = _validation_metrics(
            model,
            validation_observed,
            validation_completion,
            validation_records,
            protocol,
        )
        best_epoch = epochs_completed
    atomic_torch_save(
        model_path,
        {
            "schema_version": "fully-neural-ms2lda/model-v1",
            "attempt": attempt,
            "model": model.state_dict(),
            "selected_epoch": best_epoch,
            "selected_validation": selected_validation,
            "single_encoder_pass": True,
            "local_vb_steps": 0,
        },
    )
    result = {
        "schema_version": "fully-neural-ms2lda/training-complete-v1",
        "attempt": attempt,
        "stable": stable,
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "selected_epoch": best_epoch,
        "global_steps": global_step,
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "maximum_hours": float(training["maximum_hours_per_attempt"]),
        "four_training_threads": int(protocol["training_cpu_threads"]) == 4,
        "rescue_guards": {
            "usage_weight": usage_weight,
            "sparsity_weight": sparsity_weight,
        },
        "selected_validation": selected_validation,
        "peak_rss_bytes": peak_rss_bytes(),
        "model_sha256": file_sha256(model_path),
    }
    write_json(complete_path, result)
    return result


def load_attempt_model(
    run_dir: str | Path,
    attempt: str,
    protocol: dict[str, Any],
) -> NeuralMS2LDA:
    """Load a completed selected checkpoint and verify its identity."""
    directory = Path(run_dir)
    complete = read_json(directory / "attempts" / attempt / "complete.json")
    path = directory / "attempts" / attempt / "model.pt"
    if file_sha256(path) != complete["model_sha256"]:
        msg = f"{attempt} selected model changed"
        raise ValueError(msg)
    model = _fresh_model(directory, protocol)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def validation_is_collapsed(
    result: dict[str, Any],
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply only pre-test collapse criteria to determine rescue eligibility."""
    failures = []
    if not result["stable"]:
        failures.append("unstable_training")
    validation = result["selected_validation"]
    active = validation["active_topics"]["corpus_active_topics"]
    reference_active = reference["active_topics"]["corpus_active_topics"]
    minimum_active = math.ceil(
        protocol["hard_viability_gates"]["active_topics_reference_fraction"]
        * reference_active,
    )
    if active < minimum_active:
        failures.append("corpus_active_topics")
    diversity = validation["top_word_diversity"]
    minimum_diversity = (
        reference["top_word_diversity"]
        - protocol["hard_viability_gates"]["top_word_diversity_absolute_drop"]
    )
    if diversity < minimum_diversity:
        failures.append("top_word_diversity")
    effective = validation["mixture_diagnostics"]["effective_topic_count_median"]
    reference_effective = reference["full_spectrum_mixture"][
        "effective_topic_count_median"
    ]
    minimum_effective = (
        reference_effective
        * protocol["hard_viability_gates"][
            "effective_topics_reference_minimum_fraction"
        ]
    )
    maximum_effective = (
        reference_effective
        * protocol["hard_viability_gates"][
            "effective_topics_reference_maximum_multiple"
        ]
    )
    if not minimum_effective <= effective <= maximum_effective:
        failures.append("median_effective_topics")
    return bool(failures), failures
