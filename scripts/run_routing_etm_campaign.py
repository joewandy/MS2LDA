"""Run the bounded M1-component transplant screen on a published ETM base."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.routing_etm import (
    ROUTING_VARIANTS,
    RoutingInformedETM,
    RoutingVariant,
)
from benchmarks.neural_ms2lda.sparse_etm import (
    RECONSTRUCTION_SCALINGS,
    THETA_TRANSFORMS,
    ReconstructionScaling,
    ThetaTransform,
    dense_normalized,
    sparse_reconstruction_loss,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    write_json,
)
from scripts.run_sparse_etm_campaign import (
    EPS,
    SYNTHETIC_ACTIVE_USAGE_THRESHOLD,
    SYNTHETIC_EVALUATION_PROTOCOL,
    _matched_truth_metrics,
    configure,
    file_sha256,
    prepare_synthetic_seed,
    read_json,
    resolve_device,
    write_csv,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

PROMOTION_SEEDS = (11, 23, 37)
PRIMARY_TOPICS = 36
HIGH_K_TOPICS = 128


def method_label(
    variant: RoutingVariant,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
) -> str:
    """Return an artifact label, reusing the exact existing ETM control."""
    if variant == "etm":
        return f"balanced_etm_{theta_transform}_{reconstruction_scaling}"
    transform = "" if theta_transform == "softmax" else f"_{theta_transform}"
    return f"balanced_etm_routing_{variant}{transform}_{reconstruction_scaling}"


@torch.inference_mode()
def infer_theta(
    model: RoutingInformedETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Infer deterministic ETM mixtures and spectra-per-second throughput."""
    model.eval()
    values = []
    started = time.perf_counter()
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        theta, _ = model.theta(
            dense_normalized(matrix, rows, device),
            sample=False,
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPS)


@torch.inference_mode()
def infer_routing_evidence(
    model: RoutingInformedETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray | None:
    """Infer the auditable token-evidence distribution used by the posterior."""
    if model.routing_variant == "etm":
        return None
    model.eval()
    values = []
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        evidence = model.routing_evidence(dense_normalized(matrix, rows, device))
        values.append(evidence.cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def run_synthetic(  # noqa: PLR0913, PLR0915
    output_root: Path,
    *,
    seed: int,
    fitted_topics: int,
    routing_variant: RoutingVariant,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
    epochs: int,
    batch_size: int,
    device: torch.device,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> dict[str, Any]:
    """Train and evaluate one isolated routing-informed ETM formulation."""
    label = method_label(routing_variant, theta_transform, reconstruction_scaling)
    output = output_root / "synthetic_runs" / f"seed_{seed}_K_{fitted_topics}_{label}"
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "evidence": "truth-known synthetic training and validation only",
        "published_base": "Embedded Topic Model",
        "seed": int(seed),
        "true_topics": 18,
        "fitted_topics": int(fitted_topics),
        "routing_variant": routing_variant,
        "routing_temperature": 1.0,
        "theta_transform": theta_transform,
        "reconstruction_scaling": reconstruction_scaling,
        "posterior_change": (
            "centered log token evidence added to the Gaussian posterior mean"
            if routing_variant != "etm"
            else "none"
        ),
        "unchanged_components": [
            "fixed train-only 48D SGNS word embeddings",
            "50/50 fragment-loss ETM decoder",
            "Gaussian variational latent family",
            "standard-normal analytic KL",
            f"{reconstruction_scaling} multinomial reconstruction weighting",
        ],
        "excluded_m1_components": [
            "nonlinear context router",
            "document gate",
            "Sinkhorn balancing",
            "positive-NPMI regularization",
            "prototype separation",
            "alternating optimization",
        ],
        "hidden_dimensions": 800,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "device": str(device),
        "threads": int(threads),
        "training_documents": int(training_documents),
        "validation_documents": int(validation_documents),
        "stopping_rule": "fixed epochs or immediate non-finite loss/gradient",
        "candidate_test_artifacts_accessed": False,
    }
    write_json(output / "config.json", config)
    dataset, embeddings, seed_directory = prepare_synthetic_seed(
        output_root,
        seed=seed,
        threads=threads,
        training_documents=training_documents,
        validation_documents=validation_documents,
    )
    # Match the existing sparse-ETM runner exactly so every routing variant
    # starts from the paired control's parameters and sees the same batch order.
    configure(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in dataset.vocabulary],
        dtype=bool,
    )
    model = RoutingInformedETM(
        embeddings,
        fitted_topics,
        fragment_mask,
        routing_variant=routing_variant,
        theta_transform=theta_transform,
        routing_temperature=1.0,
        hidden=800,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=1.2e-6,
    )
    rng = np.random.default_rng(seed + 7019)
    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(dataset.train.shape[0])
        reconstruction_values = []
        kl_values = []
        gradient_values = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            theta, kl = model.theta(
                dense_normalized(dataset.train, rows, device),
                sample=True,
            )
            reconstruction, _ = sparse_reconstruction_loss(
                theta,
                model.beta(),
                dataset.train,
                rows,
                device,
                scaling=reconstruction_scaling,
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                message = "routing ETM produced a non-finite objective"
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            finite_gradients = all(
                parameter.grad is None
                or torch.all(torch.isfinite(parameter.grad)).item()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                message = "routing ETM produced non-finite gradients"
                raise FloatingPointError(message)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
        row = {
            "epoch": epoch + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "reconstruction_to_kl_ratio": float(
                np.mean(reconstruction_values) / max(np.mean(kl_values), EPS),
            ),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "context_scale": (
                float(model.context_scale.detach().cpu())
                if model.context_scale is not None
                else None
            ),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print(  # noqa: T201
            "ROUTING_ETM_EPOCH",
            json.dumps({"run": label, **row}, sort_keys=True),
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    model.eval()
    with torch.inference_mode():
        beta = model.beta().cpu().numpy().astype(np.float32)
    theta_observed, observed_throughput = infer_theta(
        model,
        dataset.validation_observed,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = infer_theta(
        model,
        dataset.validation_full,
        batch_size=batch_size,
        device=device,
    )
    routing_evidence = infer_routing_evidence(
        model,
        dataset.validation_full,
        batch_size=batch_size,
        device=device,
    )
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        dataset.vocabulary,
        SYNTHETIC_EVALUATION_PROTOCOL,
    )
    usage = theta_full.astype(np.float64).mean(axis=0)
    metrics = {
        "dataset": dataset.summary,
        "heldout_completion": completion_metrics(
            theta_observed,
            beta,
            dataset.validation_completion,
            list(dataset.validation_records),
        ),
        "truth_recovery": _matched_truth_metrics(
            beta,
            theta_full,
            dataset.true_beta,
            dataset.validation_true_theta,
        ),
        "theta_support": theta_support_diagnostics(theta_full),
        "routing_evidence_support": (
            theta_support_diagnostics(routing_evidence)
            if routing_evidence is not None
            else None
        ),
        **diagnostics,
        "active_topics_mean_usage_gt_0_005": int(
            np.sum(usage > SYNTHETIC_ACTIVE_USAGE_THRESHOLD),
        ),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full)),
        ),
        "learned_context_scale": (
            float(model.context_scale.detach().cpu())
            if model.context_scale is not None
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
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "validation_observed_theta.npy", theta_observed)
    atomic_save_numpy(output / "validation_full_theta.npy", theta_full)
    if routing_evidence is not None:
        atomic_save_numpy(output / "validation_routing_evidence.npy", routing_evidence)
    provenance = {
        "synthetic_artifact_manifest": str(seed_directory / "artifact_manifest.json"),
        "token_features": {
            "path": str(seed_directory / "token_features/features.npy"),
            "bytes": (seed_directory / "token_features/features.npy").stat().st_size,
            "sha256": file_sha256(seed_directory / "token_features/features.npy"),
        },
        "weights": {
            "path": str(output / "weights.pt"),
            "bytes": (output / "weights.pt").stat().st_size,
            "sha256": file_sha256(output / "weights.pt"),
        },
        "candidate_test_artifacts_accessed": False,
    }
    write_json(output / "provenance.json", provenance)
    result = {
        "method": label,
        "config": config,
        "metrics": metrics,
        "provenance": provenance,
    }
    write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one routing-informed ETM experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fitted-topics", required=True, type=int)
    parser.add_argument(
        "--routing-variant",
        required=True,
        choices=ROUTING_VARIANTS,
    )
    parser.add_argument(
        "--theta-transform",
        choices=THETA_TRANSFORMS,
        default="softmax",
    )
    parser.add_argument(
        "--reconstruction-scaling",
        choices=RECONSTRUCTION_SCALINGS,
        default="raw_counts",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--training-documents", type=int, default=800)
    parser.add_argument("--validation-documents", type=int, default=160)
    args = parser.parse_args(argv)
    result = run_synthetic(
        args.output_root.expanduser().resolve(),
        seed=args.seed,
        fitted_topics=args.fitted_topics,
        routing_variant=args.routing_variant,
        theta_transform=args.theta_transform,
        reconstruction_scaling=args.reconstruction_scaling,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=resolve_device(args.device),
        threads=args.threads,
        training_documents=args.training_documents,
        validation_documents=args.validation_documents,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
