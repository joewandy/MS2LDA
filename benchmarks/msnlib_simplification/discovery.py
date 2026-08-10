# ruff: noqa: PLR0913, PLR0915, TRY301
"""Verified current discovery import and a genuinely DreaMS-free LDA arm."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.msnlib_validation.config import (
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)
from ms2lda_hybrid._variational import (
    EPSILON,
    estimate_dirichlet_alpha,
    expected_log_dirichlet,
    expected_topic_word_counts,
    local_vb,
    make_sparse_batch,
)

from .data import load_count_matrix, load_vocabulary_copy, prepare_count_inputs
from .runtime import configure_cpu_threads
from .spec import load_spec, verify_study

SOURCE_CHECKPOINT_VERSION = 3


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_complete(output: Path, discovery: str) -> dict[str, Any]:
    result = read_json(output / "complete.json")
    if result.get("discovery") != discovery:
        msg = "discovery completion identity changed"
        raise ValueError(msg)
    for name, digest in result["output_sha256"].items():
        if file_sha256(output / name) != digest:
            msg = f"discovery artifact changed: {name}"
            raise ValueError(msg)
    return result


def import_current_discovery(run_dir: str | Path) -> dict[str, Any]:
    """Import only alpha/lambda from the corrected finalized Hybrid model."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    output = directory / "discoveries" / "dreams_prior"
    if (output / "complete.json").is_file():
        return _verify_complete(output, "dreams_prior")
    source = Path(lock["source_run"])
    model_path = source / "core/seed_42/hybrid/model.pt"
    complete = read_json(source / "core/seed_42/hybrid/complete.json")
    if file_sha256(model_path) != complete["model_sha256"]:
        msg = "corrected Hybrid source model changed"
        raise ValueError(msg)
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "ms2lda-hybrid-reference"
        or payload.get("version") != SOURCE_CHECKPOINT_VERSION
    ):
        msg = "unsupported corrected Hybrid checkpoint"
        raise ValueError(msg)
    vocabulary = load_vocabulary_copy(directory)
    model_vocabulary = tuple(map(str, payload["vocabulary"]))
    if model_vocabulary != vocabulary:
        msg = "corrected Hybrid vocabulary changed"
        raise ValueError(msg)
    state = payload["core_state_dict"]
    alpha = state["alpha"].detach().cpu().numpy().astype(np.float32, copy=True)
    lambda_posterior = (
        state["lambda_posterior"].detach().cpu().numpy().astype(np.float32, copy=True)
    )
    posterior_mean = lambda_posterior / np.maximum(
        lambda_posterior.sum(axis=1, keepdims=True),
        EPSILON,
    )
    source_beta = np.load(source / "core/seed_42/hybrid/beta.npy")
    protocol_vocabulary = tuple(
        map(str, read_json(source / "vocabulary.json")["vocabulary"]),
    )
    columns = {word: index for index, word in enumerate(model_vocabulary)}
    posterior_mean_in_protocol_order = posterior_mean[
        :,
        [columns[word] for word in protocol_vocabulary],
    ]
    alignment_max_abs = float(
        np.max(np.abs(posterior_mean_in_protocol_order - source_beta)),
    )
    if not np.allclose(
        posterior_mean_in_protocol_order,
        source_beta,
        rtol=1e-6,
        atol=1e-8,
    ):
        msg = "imported current discovery does not reproduce source beta"
        raise ValueError(msg)
    protocol_columns = {word: index for index, word in enumerate(protocol_vocabulary)}
    beta = source_beta[
        :,
        [protocol_columns[word] for word in model_vocabulary],
    ]
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "snapshot.npz"
    _atomic_savez(snapshot, alpha=alpha, lambda_posterior=lambda_posterior, beta=beta)
    history = output / "history.json"
    write_json(
        history,
        read_json(source / "core/seed_42/hybrid/discovery_history.json"),
    )
    result = {
        "schema_version": "msnlib-simplification/discovery-v1",
        "discovery": "dreams_prior",
        "seed": 42,
        "topic_count": int(beta.shape[0]),
        "vocabulary_size": int(beta.shape[1]),
        "converged": bool(complete["discovery_converged"]),
        "epochs": int(complete["discovery_epochs"]),
        "source_model_sha256": complete["model_sha256"],
        "source_beta_sha256": complete["beta_sha256"],
        "alpha_exactly_recovered": True,
        "beta_exactly_recovered_by_frozen_source_reordering": True,
        "posterior_mean_alignment_max_abs": alignment_max_abs,
        "working_vocabulary_order": "corrected_hybrid_model_insertion_order",
        "output_sha256": {path.name: file_sha256(path) for path in (snapshot, history)},
    }
    write_json(output / "complete.json", result)
    return result


def _checkpoint_context(directory: Path) -> str:
    counts = prepare_count_inputs(directory)
    spec = load_spec(directory)
    return object_sha256(
        {
            "spec": spec.as_dict(),
            "train_matrix_sha256": counts["output_sha256"]["train.npz"],
            "vocabulary_sha256": counts["output_sha256"]["vocabulary.json"],
            "discovery": "symmetric_prior",
        },
    )


def _checkpoint_sidecars(output: Path) -> list[Path]:
    return sorted((output / "checkpoints").glob("checkpoint-*.json"), reverse=True)


def _load_checkpoint(output: Path, context: str) -> dict[str, Any] | None:
    rejected: list[dict[str, str]] = []
    for sidecar in _checkpoint_sidecars(output):
        try:
            metadata = read_json(sidecar)
            if metadata.get("context_sha256") != context:
                msg = "context mismatch"
                raise ValueError(msg)
            payload_path = sidecar.parent / str(metadata["file"])
            if payload_path.stat().st_size != int(metadata["bytes"]):
                msg = "byte-size mismatch"
                raise ValueError(msg)
            if file_sha256(payload_path) != metadata["sha256"]:
                msg = "hash mismatch"
                raise ValueError(msg)
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
            if payload.get("context_sha256") != context:
                msg = "payload context mismatch"
                raise ValueError(msg)
        except Exception as exc:  # noqa: BLE001 - fall back to older checkpoint
            rejected.append(
                {"sidecar": sidecar.name, "reason": f"{type(exc).__name__}: {exc}"},
            )
            continue
        write_json(
            output / "checkpoint_resume_audit.json",
            {"selected": metadata, "rejected_newer": rejected, "resumed": True},
        )
        return payload
    if _checkpoint_sidecars(output):
        msg = "no valid symmetric-discovery checkpoint remains"
        raise RuntimeError(msg)
    write_json(
        output / "checkpoint_resume_audit.json",
        {"selected": None, "rejected_newer": [], "resumed": False},
    )
    return None


def _save_checkpoint(
    output: Path,
    *,
    context: str,
    epoch: int,
    alpha: torch.Tensor,
    lambda_posterior: torch.Tensor,
    gamma: np.ndarray,
    history: list[dict[str, float]],
    stable_epochs: int,
    elapsed_seconds: float,
    keep: int,
) -> None:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"checkpoint-{epoch:04d}"
    payload_path = checkpoint_dir / f"{stem}.pt"
    temporary = payload_path.with_name(f".{payload_path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": "symmetric-discovery-checkpoint/v1",
        "context_sha256": context,
        "epoch": epoch,
        "alpha": alpha.detach().cpu(),
        "lambda_posterior": lambda_posterior.detach().cpu(),
        "gamma": torch.from_numpy(gamma),
        "history": copy.deepcopy(history),
        "stable_epochs": stable_epochs,
        "elapsed_seconds": elapsed_seconds,
    }
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(payload_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "context_sha256": context,
        "epoch": epoch,
        "file": payload_path.name,
        "bytes": payload_path.stat().st_size,
        "sha256": file_sha256(payload_path),
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(checkpoint_dir / f"{stem}.json", metadata)
    write_json(checkpoint_dir / "latest.json", metadata)
    for sidecar in _checkpoint_sidecars(output)[keep:]:
        old = read_json(sidecar)
        (sidecar.parent / str(old.get("file", ""))).unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def run_symmetric_discovery(run_dir: str | Path) -> dict[str, Any]:
    """Run classical variational LDA with eta plus expected counts only."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    spec = load_spec(directory)
    output = directory / "discoveries" / "symmetric_prior"
    if (output / "complete.json").is_file():
        return _verify_complete(output, "symmetric_prior")
    output.mkdir(parents=True, exist_ok=True)
    matrix = load_count_matrix(directory, "train")
    vocabulary = load_vocabulary_copy(directory)
    lock = read_json(directory / "simplification.lock.json")
    source_config = read_json(Path(lock["source_run"]) / "config.resolved.json")
    source_model = torch.load(
        Path(lock["source_run"]) / "core/seed_42/hybrid/model.pt",
        map_location="cpu",
        weights_only=True,
    )["config"]
    cpu_threads = configure_cpu_threads(directory, "training")
    context = _checkpoint_context(directory)
    checkpoint = _load_checkpoint(output, context)
    if checkpoint is None:
        alpha = torch.full(
            (spec.num_topics,),
            float(source_config["alpha"]),
            dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        raw = torch.empty(spec.num_topics, matrix.shape[1], dtype=torch.float32)
        raw.exponential_(1.0, generator=generator)
        means = raw / raw.sum(dim=1, keepdim=True).clamp_min(EPSILON)
        mass = float(matrix.sum()) / spec.num_topics
        lambda_posterior = float(source_config["eta"]) + mass * means
        totals = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        gamma = alpha.numpy()[None, :] + totals / spec.num_topics
        history: list[dict[str, float]] = []
        epoch = 0
        stable_epochs = 0
        elapsed_base = 0.0
    else:
        alpha = checkpoint["alpha"].float()
        lambda_posterior = checkpoint["lambda_posterior"].float()
        gamma = checkpoint["gamma"].numpy().astype(np.float32, copy=True)
        history = copy.deepcopy(checkpoint["history"])
        epoch = int(checkpoint["epoch"])
        stable_epochs = int(checkpoint["stable_epochs"])
        elapsed_base = float(checkpoint["elapsed_seconds"])
    started = time.perf_counter()
    converged = stable_epochs >= spec.global_patience
    for epoch_index in range(epoch + 1, spec.symmetric_max_epochs + 1):
        previous_lambda = lambda_posterior.clone()
        previous_alpha = alpha.clone()
        statistics = torch.zeros_like(lambda_posterior)
        updated_gamma = np.empty_like(gamma)
        expected_log_theta_sum = np.zeros(spec.num_topics, dtype=np.float64)
        expected_log_beta = expected_log_dirichlet(lambda_posterior)
        for start in range(0, matrix.shape[0], spec.batch_size):
            indices = np.arange(start, min(start + spec.batch_size, matrix.shape[0]))
            batch = make_sparse_batch(matrix, indices, device=torch.device("cpu"))
            initial = torch.from_numpy(gamma[indices])
            refined, phi = local_vb(
                batch,
                initial,
                alpha,
                expected_log_beta,
                steps=int(source_model["training_local_steps"]),
                tolerance=float(source_model["local_tolerance"]),
            )
            updated_gamma[indices] = refined.numpy()
            expected_log_theta_sum += (
                expected_log_dirichlet(refined.double()).sum(dim=0).numpy()
            )
            statistics += expected_topic_word_counts(
                batch,
                phi,
                num_topics=spec.num_topics,
                vocab_size=matrix.shape[1],
            )
        gamma = updated_gamma
        optimized_alpha = estimate_dirichlet_alpha(
            previous_alpha.numpy(),
            expected_log_theta_sum,
            matrix.shape[0],
        )
        alpha = torch.as_tensor(optimized_alpha, dtype=torch.float32)
        lambda_posterior = float(source_config["eta"]) + statistics
        lambda_change = float(
            (lambda_posterior - previous_lambda).abs().sum()
            / previous_lambda.abs().sum().clamp_min(EPSILON),
        )
        alpha_change = float(
            (alpha - previous_alpha).abs().sum()
            / previous_alpha.abs().sum().clamp_min(EPSILON),
        )
        history.append(
            {
                "epoch": float(epoch_index),
                "lambda_relative_change": lambda_change,
                "alpha_relative_change": alpha_change,
                "alpha_sum": float(alpha.sum()),
                "alpha_min": float(alpha.min()),
                "alpha_median": float(np.median(alpha.numpy())),
                "alpha_max": float(alpha.max()),
                "structured_prior_mass_fraction": 0.0,
            },
        )
        if (
            epoch_index >= spec.symmetric_min_epochs
            and lambda_change < float(source_model["global_tolerance"])
            and alpha_change < float(source_model["global_tolerance"])
        ):
            stable_epochs += 1
        else:
            stable_epochs = 0
        elapsed = elapsed_base + time.perf_counter() - started
        _save_checkpoint(
            output,
            context=context,
            epoch=epoch_index,
            alpha=alpha,
            lambda_posterior=lambda_posterior,
            gamma=gamma,
            history=history,
            stable_epochs=stable_epochs,
            elapsed_seconds=elapsed,
            keep=spec.checkpoint_keep,
        )
        if stable_epochs >= spec.global_patience:
            converged = True
            epoch = epoch_index
            break
        epoch = epoch_index
    elapsed_total = elapsed_base + time.perf_counter() - started
    if not converged:
        failure = {
            "schema_version": "msnlib-simplification/scientific-failure-v1",
            "discovery": "symmetric_prior",
            "reason": "maximum epochs reached without frozen convergence",
            "epochs": epoch,
            "elapsed_seconds": elapsed_total,
            "retuning_performed": False,
        }
        write_json(output / "scientific_failure.json", failure)
        raise RuntimeError(failure["reason"])
    beta = lambda_posterior.numpy()
    beta /= np.maximum(beta.sum(axis=1, keepdims=True), EPSILON)
    snapshot = output / "snapshot.npz"
    _atomic_savez(
        snapshot,
        alpha=alpha.numpy(),
        lambda_posterior=lambda_posterior.numpy(),
        beta=beta,
    )
    history_path = output / "history.json"
    write_json(history_path, history)
    result = {
        "schema_version": "msnlib-simplification/discovery-v1",
        "discovery": "symmetric_prior",
        "seed": spec.seed,
        "topic_count": spec.num_topics,
        "vocabulary_size": len(vocabulary),
        "converged": True,
        "epochs": epoch,
        "elapsed_seconds": elapsed_total,
        "cpu_threads": cpu_threads,
        "dreams_cache_loaded": False,
        "dreams_modules_initialized": False,
        "structured_prior_mass_fraction": 0.0,
        "retuning_performed": False,
        "checkpoint_context_sha256": context,
        "output_sha256": {
            path.name: file_sha256(path) for path in (snapshot, history_path)
        },
    }
    write_json(output / "complete.json", result)
    return result


def load_discovery(
    run_dir: str | Path,
    discovery: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load alpha, lambda, beta, and verified discovery metadata."""
    directory = Path(run_dir).expanduser().resolve()
    if discovery == "dreams_prior":
        import_current_discovery(directory)
    elif discovery == "symmetric_prior":
        run_symmetric_discovery(directory)
    else:
        msg = f"unknown discovery: {discovery}"
        raise ValueError(msg)
    output = directory / "discoveries" / discovery
    result = _verify_complete(output, discovery)
    values = np.load(output / "snapshot.npz")
    return (
        np.asarray(values["alpha"], dtype=np.float32),
        np.asarray(values["lambda_posterior"], dtype=np.float32),
        np.asarray(values["beta"], dtype=np.float32),
        result,
    )
