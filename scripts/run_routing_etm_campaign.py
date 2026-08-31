"""Run the bounded M1-component transplant screen on a published ETM base."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from benchmarks.neural_ms2lda.contextual_sparse_etm import ContextualSparseETM
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.objectives import (
    beta_cooccurrence_topic_loss,
    completion_metrics,
    prepare_cooccurrence_graph,
    torch_sparse_graph,
)
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
    read_json_object,
    resolve_torch_device,
    sha256_file,
    write_csv_rows,
)
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
from benchmarks.neural_ms2lda.top2_token_etm import (
    TOP2_ROUTING_VARIANT,
    Top2TokenETM,
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
    prepare_synthetic_seed,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

PROMOTION_SEEDS = (11, 23, 37)
PRIMARY_TOPICS = 36
HIGH_K_TOPICS = 128
CampaignRoutingVariant = RoutingVariant | Literal["top2_token"]
CAMPAIGN_ROUTING_VARIANTS: tuple[CampaignRoutingVariant, ...] = (
    *ROUTING_VARIANTS,
    TOP2_ROUTING_VARIANT,
)
POSITIVE_NPMI_WEIGHT = 1.0
POSITIVE_NPMI_GRAPH_CONFIG = {
    "minimum_document_frequency": 10,
    "minimum_pair_frequency": 3,
    "maximum_neighbors": 16,
    "minimum_npmi": 0.0,
    "weight": POSITIVE_NPMI_WEIGHT,
}


def build_synthetic_model(  # noqa: PLR0913
    embeddings: np.ndarray,
    fitted_topics: int,
    fragment_mask: np.ndarray,
    *,
    routing_variant: CampaignRoutingVariant,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
    hidden: int = 800,
) -> torch.nn.Module:
    """Construct one synthetic formulation without duplicating model equations.

    The paper-facing treatment is instantiated from :class:`ContextualSparseETM`,
    the maintained implementation used for real-data training and inference.
    Historical configurable classes remain only for the controlled softmax and
    entmax ablations that are not themselves the proposed model.
    """
    is_contextual_sparse_etm = (
        routing_variant == "top2_context"
        and theta_transform == "entmax15"
        and reconstruction_scaling == "raw_counts"
    )
    if is_contextual_sparse_etm:
        return ContextualSparseETM(
            embeddings,
            fitted_topics,
            fragment_mask,
            hidden=hidden,
        )
    if routing_variant == TOP2_ROUTING_VARIANT:
        return Top2TokenETM(
            embeddings,
            fitted_topics,
            fragment_mask,
            theta_transform=theta_transform,
            routing_temperature=1.0,
            hidden=hidden,
        )
    return RoutingInformedETM(
        embeddings,
        fitted_topics,
        fragment_mask,
        routing_variant=routing_variant,
        theta_transform=theta_transform,
        routing_temperature=1.0,
        hidden=hidden,
    )


def _model_theta(
    model: torch.nn.Module,
    normalized_bows: torch.Tensor,
    *,
    sample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the document-mixture equation for any controlled formulation."""
    if isinstance(model, ContextualSparseETM):
        return model.document_topic_mixture(normalized_bows, sample=sample)
    return model.theta(normalized_bows, sample=sample)


def _model_beta(model: torch.nn.Module) -> torch.Tensor:
    """Evaluate the topic-word equation for any controlled formulation."""
    if isinstance(model, ContextualSparseETM):
        return model.topic_word_distribution()
    return model.beta()


def _model_evidence(
    model: torch.nn.Module,
    normalized_bows: torch.Tensor,
) -> torch.Tensor | None:
    """Return contextual evidence when the formulation defines it."""
    if isinstance(model, ContextualSparseETM):
        return model.contextual_evidence(normalized_bows)
    if model.routing_variant == "etm":
        return None
    return model.routing_evidence(normalized_bows)


def method_label(
    variant: CampaignRoutingVariant,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
    *,
    positive_npmi: bool = False,
) -> str:
    """Return an artifact label, reusing the exact existing ETM control."""
    if variant == "etm":
        label = f"balanced_etm_{theta_transform}_{reconstruction_scaling}"
    else:
        transform = "" if theta_transform == "softmax" else f"_{theta_transform}"
        label = f"balanced_etm_routing_{variant}{transform}_{reconstruction_scaling}"
    return f"{label}_positive_npmi" if positive_npmi else label


@torch.inference_mode()
def infer_theta(
    model: torch.nn.Module,
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
        theta, _ = _model_theta(
            model,
            dense_normalized(matrix, rows, device),
            sample=False,
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPS)


@torch.inference_mode()
def infer_routing_evidence(
    model: torch.nn.Module,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray | None:
    """Infer the auditable token-evidence distribution used by the posterior."""
    model.eval()
    values = []
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        evidence = _model_evidence(
            model,
            dense_normalized(matrix, rows, device),
        )
        if evidence is None:
            return None
        values.append(evidence.cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def run_synthetic(  # noqa: PLR0913, PLR0915
    output_root: Path,
    *,
    seed: int,
    fitted_topics: int,
    routing_variant: CampaignRoutingVariant,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
    epochs: int,
    batch_size: int,
    device: torch.device,
    threads: int,
    training_documents: int,
    validation_documents: int,
    positive_npmi: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one isolated routing-informed ETM formulation."""
    label = method_label(
        routing_variant,
        theta_transform,
        reconstruction_scaling,
        positive_npmi=positive_npmi,
    )
    output = output_root / "synthetic_runs" / f"seed_{seed}_K_{fitted_topics}_{label}"
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json_object(result_path)
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
        "positive_npmi": (
            {
                **POSITIVE_NPMI_GRAPH_CONFIG,
                "objective_application": "every ordinary ETM minibatch",
            }
            if positive_npmi
            else None
        ),
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
            *([] if positive_npmi else ["positive-NPMI regularization"]),
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
    graph = None
    graph_tensor = None
    if positive_npmi:
        graph = prepare_cooccurrence_graph(
            output,
            train=dataset.train,
            protocol={"cooccurrence_regularization": POSITIVE_NPMI_GRAPH_CONFIG},
        )
        graph_tensor = torch_sparse_graph(graph).to(device)
    # Match the existing sparse-ETM runner exactly so every routing variant
    # starts from the paired control's parameters and sees the same batch order.
    configure_deterministic_execution(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in dataset.vocabulary],
        dtype=bool,
    )
    model = build_synthetic_model(
        embeddings,
        fitted_topics,
        fragment_mask,
        routing_variant=routing_variant,
        theta_transform=theta_transform,
        reconstruction_scaling=reconstruction_scaling,
        hidden=800,
    ).to(device)
    config["implementation_class"] = type(model).__name__
    write_json(output / "config.json", config)
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
        npmi_values = []
        gradient_values = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            theta, kl = _model_theta(
                model,
                dense_normalized(dataset.train, rows, device),
                sample=True,
            )
            beta_batch = _model_beta(model)
            reconstruction, _ = sparse_reconstruction_loss(
                theta,
                beta_batch,
                dataset.train[rows],
                device,
                scaling=reconstruction_scaling,
            )
            npmi = (
                beta_cooccurrence_topic_loss(graph_tensor, beta=beta_batch)
                if graph_tensor is not None
                else reconstruction.new_zeros(())
            )
            objective = reconstruction + kl.mean() + POSITIVE_NPMI_WEIGHT * npmi
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
            npmi_values.append(float(npmi.detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
        row = {
            "epoch": epoch + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "positive_npmi_loss": (
                float(np.mean(npmi_values)) if positive_npmi else None
            ),
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
        write_csv_rows(output / "training_history.csv", history)
        print(  # noqa: T201
            "ROUTING_ETM_EPOCH",
            json.dumps({"run": label, **row}, sort_keys=True),
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    model.eval()
    with torch.inference_mode():
        beta_tensor = _model_beta(model)
        final_npmi_loss = (
            float(
                beta_cooccurrence_topic_loss(
                    graph_tensor,
                    beta=beta_tensor,
                ).cpu(),
            )
            if graph_tensor is not None
            else None
        )
        beta = beta_tensor.cpu().numpy().astype(np.float32)
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
        "positive_npmi": {
            "enabled": bool(positive_npmi),
            "weight": POSITIVE_NPMI_WEIGHT if positive_npmi else None,
            "final_loss": final_npmi_loss,
            "graph_nonzero_entries": int(graph.nnz) if graph is not None else None,
            "graph_mean_weight": (
                float(np.mean(graph.data)) if graph is not None else None
            ),
            "graph_max_weight": (
                float(np.max(graph.data)) if graph is not None else None
            ),
        },
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
            "sha256": sha256_file(seed_directory / "token_features/features.npy"),
        },
        "weights": {
            "path": str(output / "weights.pt"),
            "bytes": (output / "weights.pt").stat().st_size,
            "sha256": sha256_file(output / "weights.pt"),
        },
        "candidate_test_artifacts_accessed": False,
        "positive_npmi_graph": (
            {
                "path": str(output / "cooccurrence_graph/positive_npmi_graph.npz"),
                "bytes": (output / "cooccurrence_graph/positive_npmi_graph.npz")
                .stat()
                .st_size,
                "sha256": sha256_file(
                    output / "cooccurrence_graph/positive_npmi_graph.npz",
                ),
            }
            if positive_npmi
            else None
        ),
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
        choices=CAMPAIGN_ROUTING_VARIANTS,
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
    parser.add_argument(
        "--positive-npmi",
        action="store_true",
        help="add the frozen weight-1 train-derived positive-NPMI loss",
    )
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
        device=resolve_torch_device(args.device),
        threads=args.threads,
        training_documents=args.training_documents,
        validation_documents=args.validation_documents,
        positive_npmi=args.positive_npmi,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
