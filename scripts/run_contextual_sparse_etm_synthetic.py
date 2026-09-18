"""Run the four predeclared synthetic ETM ablation formulations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as nnf

from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    ContextualSparseETM,
    entmax15_document_mixture,
    reparameterized_gaussian,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.etm_baselines import ChannelBalancedETM
from benchmarks.neural_ms2lda.model_evaluation import (
    completion_metrics,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
    normalize_probability_rows,
    read_json_object,
    resolve_torch_device,
    sha256_file,
)
from benchmarks.neural_ms2lda.study_protocol import (
    SYNTHETIC_ARTIFACT_LABELS,
    SYNTHETIC_FORMULATIONS,
)
from benchmarks.neural_ms2lda.synthetic_msms import (
    ACTIVE_TOPIC_USAGE_THRESHOLD,
    EVALUATION_PROTOCOL,
    load_prepared_synthetic_seed,
    matched_truth_metrics,
)
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    raw_count_reconstruction_loss,
)
from benchmarks.neural_ms2lda.utils import (
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

Formulation = str
EPSILON = 1e-12
HIDDEN_WIDTH = 800


def build_synthetic_model(
    embeddings: np.ndarray,
    fitted_topics: int,
    fragment_mask: np.ndarray,
    *,
    formulation: Formulation,
    hidden: int = HIDDEN_WIDTH,
) -> ChannelBalancedETM | ContextualSparseETM:
    """Construct the decoder-only or decoder-plus-context ablation model."""
    if formulation not in SYNTHETIC_FORMULATIONS:
        raise ValueError(f"formulation must be one of {SYNTHETIC_FORMULATIONS}")
    if formulation.startswith("contextual_"):
        return ContextualSparseETM(
            embeddings,
            fitted_topics,
            fragment_mask,
            hidden=hidden,
        )
    return ChannelBalancedETM(
        embeddings,
        fitted_topics,
        fragment_mask,
        hidden=hidden,
    )


def document_topic_mixture(
    model: ChannelBalancedETM | ContextualSparseETM,
    normalized_bows: torch.Tensor,
    *,
    formulation: Formulation,
    sample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the selected softmax or entmax transform to one posterior."""
    mean, log_variance, kl = model.posterior(normalized_bows)
    latent = reparameterized_gaussian(mean, log_variance, sample=sample)
    theta = (
        entmax15_document_mixture(latent)
        if formulation.endswith("_entmax")
        else nnf.softmax(latent, dim=1)
    )
    return theta, kl


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def infer_document_topics(
    model: ChannelBalancedETM | ContextualSparseETM,
    matrix: sp.csr_matrix,
    *,
    formulation: Formulation,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run deterministic inference and report spectra per second."""
    model.eval()
    _synchronize(device)
    started = time.perf_counter()
    values: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        theta, _ = document_topic_mixture(
            model,
            dense_normalized(matrix, rows, device),
            formulation=formulation,
            sample=False,
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPSILON)


@torch.inference_mode()
def infer_context_evidence(
    model: ChannelBalancedETM | ContextualSparseETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray | None:
    """Return the explicit context-evidence distribution when it exists."""
    if not isinstance(model, ContextualSparseETM):
        return None
    model.eval()
    values: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        evidence = model.contextual_evidence(dense_normalized(matrix, rows, device))
        values.append(evidence.cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def _train_synthetic_model(
    model: ChannelBalancedETM | ContextualSparseETM,
    training_matrix: sp.csr_matrix,
    *,
    formulation: Formulation,
    label: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    """Optimize one formulation and return its complete epoch trace."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1.2e-6)
    random_generator = np.random.default_rng(seed + 7019)
    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    training_started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        model.train()
        order = random_generator.permutation(training_matrix.shape[0])
        reconstruction_values: list[float] = []
        kl_values: list[float] = []
        gradient_values: list[float] = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            normalized = dense_normalized(training_matrix, rows, device)
            theta, kl = document_topic_mixture(
                model,
                normalized,
                formulation=formulation,
                sample=True,
            )
            beta = model.topic_word_distribution()
            reconstruction, _ = raw_count_reconstruction_loss(
                theta,
                beta,
                training_matrix[rows],
                device,
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "synthetic ETM produced a non-finite objective"
                )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if not all(
                parameter.grad is None
                or bool(torch.all(torch.isfinite(parameter.grad)).item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("synthetic ETM produced non-finite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
        _synchronize(device)
        row = {
            "epoch": epoch_index + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "reconstruction_to_kl_ratio": float(
                np.mean(reconstruction_values) / max(np.mean(kl_values), EPSILON),
            ),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "context_scale": (
                float(model.context_scale.detach().cpu())
                if isinstance(model, ContextualSparseETM)
                else None
            ),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            "SYNTHETIC_ETM_EPOCH",
            json.dumps({"run": label, **row}, sort_keys=True),
            flush=True,
        )
    _synchronize(device)
    return history, time.perf_counter() - training_started


def _evaluate_synthetic_model(
    model: ChannelBalancedETM | ContextualSparseETM,
    dataset: Any,
    *,
    formulation: Formulation,
    label: str,
    batch_size: int,
    device: torch.device,
    training_seconds: float,
) -> dict[str, Any]:
    """Evaluate one fitted formulation against held-out data and known truth."""
    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.topic_word_distribution().cpu().numpy(),
            name=f"{label} synthetic beta",
        )
    theta_observed, observed_throughput = infer_document_topics(
        model,
        dataset.validation_observed,
        formulation=formulation,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = infer_document_topics(
        model,
        dataset.validation_full,
        formulation=formulation,
        batch_size=batch_size,
        device=device,
    )
    context_evidence = infer_context_evidence(
        model,
        dataset.validation_full,
        batch_size=batch_size,
        device=device,
    )
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        dataset.vocabulary,
        EVALUATION_PROTOCOL,
    )
    usage = theta_full.astype(np.float64).mean(axis=0)
    return {
        "dataset": dataset.summary,
        "heldout_completion": completion_metrics(
            theta_observed,
            beta,
            dataset.validation_completion,
            list(dataset.validation_records),
        ),
        "truth_recovery": matched_truth_metrics(
            beta,
            theta_full,
            dataset.true_beta,
            dataset.validation_true_theta,
        ),
        "theta_support": theta_support_diagnostics(theta_full),
        "context_evidence_support": (
            theta_support_diagnostics(context_evidence)
            if context_evidence is not None
            else None
        ),
        **diagnostics,
        "active_topics_mean_usage_gt_0_005": int(
            np.sum(usage > ACTIVE_TOPIC_USAGE_THRESHOLD),
        ),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full))
        ),
        "learned_context_scale": (
            float(model.context_scale.detach().cpu())
            if isinstance(model, ContextualSparseETM)
            else None
        ),
        "runtime": {
            "training_wall_seconds": training_seconds,
            "validation_observed_spectra_per_second": observed_throughput,
            "validation_full_spectra_per_second": full_throughput,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_synthetic(
    output_root: Path,
    *,
    seed: int,
    fitted_topics: int,
    formulation: Formulation,
    epochs: int,
    batch_size: int,
    device: torch.device,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> dict[str, Any]:
    """Train and evaluate one isolated truth-known ablation formulation."""
    if formulation not in SYNTHETIC_FORMULATIONS:
        raise ValueError(f"formulation must be one of {SYNTHETIC_FORMULATIONS}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    label = SYNTHETIC_ARTIFACT_LABELS[formulation]
    output = output_root / "synthetic_runs" / f"seed_{seed}_K_{fitted_topics}_{label}"
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json_object(result_path)
    output.mkdir(parents=True, exist_ok=True)
    dataset, embeddings, seed_directory = load_prepared_synthetic_seed(
        output_root,
        seed=seed,
    )
    if int(dataset.summary["training_documents"]) != int(training_documents):
        raise ValueError("prepared synthetic training-document count changed")
    if int(dataset.summary["validation_documents"]) != int(validation_documents):
        raise ValueError("prepared synthetic validation-document count changed")
    configure_deterministic_execution(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in dataset.vocabulary],
        dtype=bool,
    )
    model = build_synthetic_model(
        embeddings,
        fitted_topics,
        fragment_mask,
        formulation=formulation,
    ).to(device)
    config = {
        "evidence": "truth-known synthetic training and validation",
        "published_base": "Embedded Topic Model",
        "seed": int(seed),
        "true_topics": 18,
        "fitted_topics": int(fitted_topics),
        "formulation": formulation,
        "theta_transform": (
            "entmax15" if formulation.endswith("_entmax") else "softmax"
        ),
        "contextual_evidence": formulation.startswith("contextual_"),
        "reconstruction_scaling": "raw_counts",
        "hidden_dimensions": HIDDEN_WIDTH,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "device": str(device),
        "threads": int(threads),
        "training_documents": int(training_documents),
        "validation_documents": int(validation_documents),
        "implementation_class": type(model).__name__,
        "stopping_rule": "fixed epochs or immediate non-finite loss/gradient",
        "test_artifacts_accessed": False,
    }
    history, training_seconds = _train_synthetic_model(
        model,
        dataset.train,
        formulation=formulation,
        label=label,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    metrics = _evaluate_synthetic_model(
        model,
        dataset,
        formulation=formulation,
        label=label,
        batch_size=batch_size,
        device=device,
        training_seconds=training_seconds,
    )
    feature_path = seed_directory / "token_features/features.npy"
    provenance = {
        "synthetic_artifact_manifest": str(
            (seed_directory / "artifact_manifest.json").relative_to(output_root),
        ),
        "token_features": {
            "path": str(feature_path.relative_to(output_root)),
            "bytes": feature_path.stat().st_size,
            "sha256": sha256_file(feature_path),
        },
        "test_artifacts_accessed": False,
    }
    result = {
        "method": label,
        "config": config,
        "metrics": metrics,
        "training_history": history,
        "provenance": provenance,
    }
    write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synthetic formulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fitted-topics", required=True, type=int)
    parser.add_argument(
        "--formulation",
        required=True,
        choices=SYNTHETIC_FORMULATIONS,
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--training-documents", type=int, default=800)
    parser.add_argument("--validation-documents", type=int, default=160)
    arguments = parser.parse_args(argv)
    result = run_synthetic(
        arguments.output_root.expanduser().resolve(),
        seed=arguments.seed,
        fitted_topics=arguments.fitted_topics,
        formulation=arguments.formulation,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        device=resolve_torch_device(arguments.device),
        threads=arguments.threads,
        training_documents=arguments.training_documents,
        validation_documents=arguments.validation_documents,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
