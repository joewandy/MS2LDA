"""Train and score the one synthetic-promoted routing-informed ETM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.followup import theta_distribution
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.routing_etm import RoutingInformedETM
from benchmarks.neural_ms2lda.sparse_etm import (
    dense_normalized,
    sparse_reconstruction_loss,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    write_json,
)
from scripts.run_msnlib_model_comparison import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
    chemical,
    entropy_diagnostics,
    save_validation,
    topic_word_diagnostics,
)
from scripts.run_routing_etm_campaign import infer_routing_evidence, infer_theta
from scripts.run_sparse_etm_campaign import (
    EPS,
    RuntimeMemoryTracker,
    _flat_theta_support,
    _validate_theta_contract,
    configure,
    file_sha256,
    prepare_real_validation_view,
    read_json,
    resolve_device,
    write_csv,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REAL_METHOD = "etm_balanced_routing_top2_entmax15_raw_counts"


def train_real_validation(  # noqa: C901, PLR0912, PLR0913, PLR0915
    real_run: Path,
    prepared_run: Path,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    """Train the single promoted model on frozen training/validation inputs."""
    real_run = real_run.expanduser().resolve()
    prepared_run = prepared_run.expanduser().resolve(strict=True)
    output = real_run / "models" / REAL_METHOD
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)

    input_manifest = prepare_real_validation_view(real_run, prepared_run)
    protocol = read_json(real_run / "protocol.json")
    seed = int(protocol["seed"])
    topics = int(protocol["model"]["num_topics"])
    data = real_run / "data"
    train = load_csr(data / "train.npz")
    observed = load_csr(data / "validation_observed.npz")
    completion = load_csr(data / "validation_completion.npz")
    full = load_csr(data / "validation_full.npz")
    records = load_heldout_records(data, "validation")
    vocabulary = load_vocabulary(data)
    features_path = real_run / "token_features" / "features.npy"
    features = np.load(features_path).astype(np.float32, copy=False)
    embeddings = np.array(features[:, :-2], dtype=np.float32, copy=True)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    if train.shape[1] != len(vocabulary) or embeddings.shape[0] != len(vocabulary):
        message = "frozen train, vocabulary, and SGNS shapes do not align"
        raise ValueError(message)
    if observed.shape != full.shape or completion.shape != full.shape:
        message = "frozen validation matrix shapes do not align"
        raise ValueError(message)
    if full.shape[0] != len(records):
        message = "frozen validation records and matrices differ"
        raise ValueError(message)

    configure(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary],
        dtype=bool,
    )
    model = RoutingInformedETM(
        embeddings,
        topics,
        fragment_mask,
        routing_variant="top2_context",
        theta_transform="entmax15",
        routing_temperature=1.0,
        hidden=800,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=1.2e-6,
    )
    rng = np.random.default_rng(seed + 7019)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    input_hashes = {
        Path(row["path"]).name: row["sha256"] for row in input_manifest["linked_inputs"]
    }
    training_contract = {
        "method": REAL_METHOD,
        "topics": topics,
        "routing_variant": "top2_context",
        "routing_temperature": 1.0,
        "theta_transform": "entmax15",
        "reconstruction_scaling": "raw_counts",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": seed + 7001,
        "optimizer": "Adam",
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "input_sha256": input_hashes,
    }
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["training_contract"] != training_contract:
            message = "routing-ETM checkpoint contract does not match"
            raise ValueError(message)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"])

    config = {
        "evidence": "frozen MSnLib training plus validation only",
        "prepared_run": str(prepared_run),
        "published_base": "Embedded Topic Model",
        "method": REAL_METHOD,
        "topics": topics,
        "fixed_train_only_sgns_dimensions": int(embeddings.shape[1]),
        "fragment_loss_beta_mass": [0.5, 0.5],
        "hidden_dimensions": 800,
        "routing_variant": "top2_context",
        "routing_temperature": 1.0,
        "context_parameters": 1,
        "posterior_evidence": (
            "centered bounded log of shared-geometry top-2 contextual token routes"
        ),
        "theta_transform": "alpha-entmax 1.5",
        "variational_posterior": "Gaussian with standard-normal analytic KL",
        "reconstruction_scaling": "raw_counts",
        "excluded_m1_components": [
            "nonlinear context router",
            "document gate",
            "Sinkhorn balancing",
            "positive-NPMI regularization",
            "prototype separation",
            "alternating optimization",
        ],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "seed": seed + 7001,
        "device": str(device),
        "threads": int(threads),
        "checkpoint_interval_epochs": 5,
        "resumed_from_epoch": start_epoch,
        "stopping_rule": "fixed epochs or immediate non-finite loss/gradient",
        "candidate_test_artifacts_accessed": False,
    }
    write_json(output / "config.json", config)
    write_json(output / "validation_access_audit.json", input_manifest)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory = RuntimeMemoryTracker(device)
    for epoch in range(start_epoch, int(epochs)):
        model.train()
        order = rng.permutation(train.shape[0])
        reconstruction_values = []
        kl_values = []
        gradient_values = []
        mass_values = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            theta, kl = model.theta(
                dense_normalized(train, rows, device),
                sample=True,
            )
            reconstruction, effective_mass = sparse_reconstruction_loss(
                theta,
                model.beta(),
                train,
                rows,
                device,
                scaling="raw_counts",
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                message = "real routing ETM produced non-finite loss"
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            finite_gradients = all(
                parameter.grad is None
                or torch.all(torch.isfinite(parameter.grad)).item()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                message = "real routing ETM produced non-finite gradients"
                raise FloatingPointError(message)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
            mass_values.append(effective_mass)
        memory.sample()
        row = {
            "epoch": epoch + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "reconstruction_to_kl_ratio": float(
                np.mean(reconstruction_values) / max(np.mean(kl_values), EPS),
            ),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "mean_effective_document_mass": float(np.mean(mass_values)),
            "context_scale": float(model.context_scale.detach().cpu()),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        if (epoch + 1) % 5 == 0 or epoch + 1 == int(epochs):
            atomic_torch_save(
                checkpoint_path,
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "history": history,
                    "training_contract": training_contract,
                },
            )
        print(  # noqa: T201
            "REAL_ROUTING_ETM_EPOCH",
            json.dumps(row, sort_keys=True),
            flush=True,
        )

    model.eval()
    with torch.inference_mode():
        beta = model.beta().cpu().numpy().astype(np.float32)
    theta_observed, observed_throughput = infer_theta(
        model,
        observed,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = infer_theta(
        model,
        full,
        batch_size=batch_size,
        device=device,
    )
    routing_evidence = infer_routing_evidence(
        model,
        full,
        batch_size=batch_size,
        device=device,
    )
    if routing_evidence is None:
        message = "promoted routing ETM produced no routing evidence"
        raise RuntimeError(message)
    _validate_theta_contract(theta_observed, name="validation observed")
    _validate_theta_contract(theta_full, name="validation full")
    _validate_theta_contract(routing_evidence, name="validation routing evidence")
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    support = theta_support_diagnostics(theta_full)
    evidence_support = theta_support_diagnostics(routing_evidence)
    _, top_words = topic_word_diagnostics(beta, vocabulary)
    metrics = {
        "document_completion": completion_metrics(
            theta_observed,
            beta,
            completion,
            records,
        ),
        **diagnostics,
        "theta_support": support,
        "routing_evidence_support": evidence_support,
        "theta_distribution": theta_distribution(theta_full),
        "theta_information": entropy_diagnostics(theta_full),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full))
            and np.all(np.isfinite(routing_evidence)),
        ),
        "learned_context_scale": float(model.context_scale.detach().cpu()),
        "runtime": {
            "training_wall_seconds": float(
                sum(float(row["seconds"]) for row in history),
            ),
            "validation_observed_spectra_per_second": observed_throughput,
            "validation_full_spectra_per_second": full_throughput,
            "memory": memory.result(),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    weights_path = output / "weights.pt"
    beta_path = output / "beta.npy"
    observed_path = output / "validation_observed_theta.npy"
    full_path = output / "validation_full_theta.npy"
    evidence_path = output / "validation_routing_evidence.npy"
    atomic_torch_save(
        weights_path,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    atomic_save_numpy(beta_path, beta)
    atomic_save_numpy(observed_path, theta_observed)
    atomic_save_numpy(full_path, theta_full)
    atomic_save_numpy(evidence_path, routing_evidence)
    write_csv(output / "top_words.csv", top_words)
    write_csv(output / "theta_support_summary.csv", [_flat_theta_support(support)])
    write_csv(
        output / "routing_evidence_support_summary.csv",
        [_flat_theta_support(evidence_support)],
    )
    write_json(
        output / "duplicate_component_summary.json",
        {
            "duplicate_components": diagnostics["topic_inventory"][
                "duplicate_components"
            ],
            "catastrophic_duplicate_component": diagnostics["topic_inventory"][
                "catastrophic_duplicate_component"
            ],
            "largest_strict_duplicate_component": diagnostics["topic_inventory"][
                "largest_strict_duplicate_component"
            ],
        },
    )
    write_json(
        output / "fragment_mass_summary.json",
        diagnostics["fragment_probability_mass"],
    )
    write_json(output / "metrics.json", metrics)
    save_validation(real_run, REAL_METHOD, beta, theta_full, metrics)
    local_artifacts = [
        weights_path,
        beta_path,
        observed_path,
        full_path,
        evidence_path,
        checkpoint_path,
    ]
    provenance = {
        "validation_inputs": input_manifest,
        "candidate_test_artifacts_accessed": False,
        "candidate_test_metrics_inspected": False,
        "local_artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in local_artifacts
        ],
    }
    write_json(output / "provenance.json", provenance)
    result = {
        "method": REAL_METHOD,
        "parameters": metrics["parameters"],
        "config": config,
        "metrics": metrics,
        "provenance": provenance,
    }
    write_json(result_path, result)
    return result


def score_real_validation(real_run: Path, data_root: Path) -> dict[str, Any]:
    """Run the unchanged leakage-controlled MAG/SOS validation evaluation."""
    return chemical(
        real_run.expanduser().resolve(),
        data_root.expanduser().resolve(),
        REAL_METHOD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run training or chemical scoring for the promoted candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--real-run", required=True, type=Path)
    train.add_argument("--prepared-run", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--threads", type=int, default=6)
    score = commands.add_parser("chemical")
    score.add_argument("--real-run", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_real_validation(
            args.real_run,
            args.prepared_run,
            device=resolve_device(args.device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    elif args.command == "chemical":
        result = score_real_validation(args.real_run, args.data_root)
    else:
        message = "unknown routing-ETM real command"
        raise RuntimeError(message)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
