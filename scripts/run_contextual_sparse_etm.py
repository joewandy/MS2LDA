"""Train and evaluate the canonical Contextual Sparse ETM.

The numerical path follows ``docs/research/neural_ms2lda_report.tex`` directly:
normalized spectral-word counts are encoded, contextual top-2 evidence shifts
the Gaussian posterior mean, 1.5-entmax produces ``theta``, and the
channel-balanced ETM decoder produces ``beta``.  Training minimizes the raw
pseudo-count reconstruction loss plus the analytic Gaussian KL divergence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from benchmarks.neural_ms2lda.contextual_sparse_etm import ContextualSparseETM
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.followup import theta_distribution
from benchmarks.neural_ms2lda.model_evaluation import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
    TRAINING_ACCESS_AUDIT_FILENAME,
    entropy_diagnostics,
    save_validation,
    score_chemical_validation,
    theta_support_diagnostics,
    topic_word_diagnostics,
)
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.reproducibility import (
    MemoryState,
    configure_deterministic_execution,
    flatten_support_summary,
    read_json_object,
    resolve_torch_device,
    runtime_memory_metrics,
    sample_runtime_memory,
    sha256_file,
    validate_probability_matrix,
    write_csv_rows,
)
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    raw_count_reconstruction_loss,
)
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

MODEL_NAME = "Contextual Sparse ETM"
REAL_METHOD = "contextual_sparse_etm"
REAL_TOPICS = 1000
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1.2e-6
CHECKPOINT_INTERVAL_EPOCHS = 5
EPSILON = 1e-12


class TrainingSettings(NamedTuple):
    """Transparent command-line settings with no behavior or hidden defaults."""

    epochs: int
    batch_size: int
    threads: int
    requested_seed: int | None = None


class ValidationData(NamedTuple):
    """Arrays and metadata loaded from the frozen validation-only view."""

    run_directory: Path
    train: sp.csr_matrix
    observed: sp.csr_matrix
    completion: sp.csr_matrix
    full: sp.csr_matrix
    records: list[dict[str, Any]]
    vocabulary: list[str]
    embeddings: np.ndarray
    fragment_mask: np.ndarray
    topics: int
    data_seed: int
    input_manifest: dict[str, Any]


class TrainingOutcome(NamedTuple):
    """The two small pieces of state needed after fitting."""

    history: list[dict[str, Any]]
    memory_state: MemoryState


class ExecutionState(NamedTuple):
    """Resolved execution values that are recorded in the run configuration."""

    training_seed: int
    device: torch.device
    resumed_from_epoch: int


def resolve_training_seed(data_seed: int, requested_seed: int | None) -> int:
    """Return the frozen default training seed or an explicit stability seed."""
    training_seed = (
        int(data_seed) + 7001 if requested_seed is None else int(requested_seed)
    )
    if training_seed < 0:
        message = "training seed must be non-negative"
        raise ValueError(message)
    return training_seed


def _validate_settings(settings: TrainingSettings) -> None:
    """Reject nonsensical execution settings before opening or writing a run."""
    if settings.epochs <= 0:
        message = "epochs must be positive"
        raise ValueError(message)
    if settings.batch_size <= 0:
        message = "batch size must be positive"
        raise ValueError(message)
    if settings.threads <= 0:
        message = "threads must be positive"
        raise ValueError(message)


def _load_validation_data(
    run_directory: Path,
) -> ValidationData:
    """Load and cross-check the frozen training and validation arrays."""
    input_manifest = read_json_object(
        run_directory / "validation_input_manifest.json",
    )
    if input_manifest.get("candidate_test_artifacts_accessed") is not False:
        message = "validation view does not preserve the test boundary"
        raise RuntimeError(message)
    protocol = read_json_object(run_directory / "protocol.json")
    topics = int(protocol["model"]["num_topics"])
    data_seed = int(protocol["seed"])
    data_directory = run_directory / "data"
    forbidden = sorted(
        path.name for path in data_directory.glob("test*") if path.is_file()
    )
    if forbidden:
        message = f"sealed training view exposes test files: {forbidden}"
        raise RuntimeError(message)
    train = load_csr(data_directory / "train.npz")
    observed = load_csr(data_directory / "validation_observed.npz")
    completion = load_csr(data_directory / "validation_completion.npz")
    full = load_csr(data_directory / "validation_full.npz")
    records = load_heldout_records(data_directory, "validation")
    vocabulary = load_vocabulary(data_directory)

    # The final two feature columns are non-embedding metadata.  The report
    # specifies unit-normalized, train-only SGNS coordinates, so normalization
    # is completed here and checked again by ContextualSparseETM.
    features_path = run_directory / "token_features" / "features.npy"
    features = np.load(features_path).astype(np.float32, copy=False)
    embeddings = np.array(features[:, :-2], dtype=np.float32, copy=True)
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(embedding_norms, 1e-8)

    vocabulary_size = len(vocabulary)
    if train.shape[1] != vocabulary_size or embeddings.shape[0] != vocabulary_size:
        message = "frozen train, vocabulary, and SGNS shapes do not align"
        raise ValueError(message)
    if observed.shape != full.shape or completion.shape != full.shape:
        message = "frozen validation matrix shapes do not align"
        raise ValueError(message)
    if full.shape[0] != len(records):
        message = "frozen validation records and matrices differ"
        raise ValueError(message)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary],
        dtype=bool,
    )
    return ValidationData(
        run_directory=run_directory,
        train=train,
        observed=observed,
        completion=completion,
        full=full,
        records=records,
        vocabulary=vocabulary,
        embeddings=embeddings,
        fragment_mask=fragment_mask,
        topics=topics,
        data_seed=data_seed,
        input_manifest=input_manifest,
    )


def _training_contract(
    data: ValidationData,
    settings: TrainingSettings,
    training_seed: int,
) -> dict[str, Any]:
    """Return the exact contract used to accept or reject a saved checkpoint."""
    input_hashes = {
        Path(row["path"]).name: row["sha256"]
        for row in data.input_manifest["linked_inputs"]
    }
    return {
        "method": REAL_METHOD,
        "topics": data.topics,
        "context_top_k": 2,
        "context_temperature": 1.0,
        "entmax_alpha": 1.5,
        "reconstruction": "raw_counts",
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "seed": training_seed,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_sha256": input_hashes,
    }


def _run_configuration(
    data: ValidationData,
    settings: TrainingSettings,
    execution: ExecutionState,
) -> dict[str, Any]:
    """Return a plain-language, machine-readable specification of this run."""
    return {
        "model_name": MODEL_NAME,
        "artifact_method_id": REAL_METHOD,
        "evidence": "frozen MSnLib training plus validation only",
        "prepared_run": data.input_manifest["prepared_run"],
        "published_base": "Embedded Topic Model",
        "topics": data.topics,
        "fixed_train_only_sgns_dimensions": int(data.embeddings.shape[1]),
        "fragment_loss_beta_mass": [0.5, 0.5],
        "hidden_dimensions": 800,
        "context_parameters": 1,
        "context_top_k": 2,
        "context_temperature": 1.0,
        "contextual_evidence": (
            "count-weighted top-2 assignments from leave-one-out token context"
        ),
        "posterior_offset": "centered log of evidence plus a fixed 1/K pseudocount",
        "evidence_pseudocount": "1/K",
        "numerical_probability_floor": EPSILON,
        "document_topic_transform": "1.5-entmax",
        "entmax_alpha": 1.5,
        "variational_posterior": "diagonal Gaussian with analytic standard-normal KL",
        "reconstruction": "raw pseudo-count multinomial negative log-likelihood",
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "data_split_seed": data.data_seed,
        "training_seed": execution.training_seed,
        "device": str(execution.device),
        "threads": settings.threads,
        "checkpoint_interval_epochs": CHECKPOINT_INTERVAL_EPOCHS,
        "resumed_from_epoch": execution.resumed_from_epoch,
        "stopping_rule": "fixed epochs or immediate non-finite loss/gradient",
        "candidate_test_artifacts_accessed": False,
    }


def _restore_checkpoint(
    checkpoint_path: Path,
    model: ContextualSparseETM,
    optimizer: torch.optim.Optimizer,
    random_generator: np.random.Generator,
    training_contract: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Restore model, optimizer, random states, and history from one checkpoint."""
    if not checkpoint_path.is_file():
        return 0, []
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["training_contract"] != training_contract:
        message = "Contextual Sparse ETM checkpoint contract does not match"
        raise ValueError(message)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    random_generator.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if model.rho.device.type == "cuda" and checkpoint["cuda_rng_states"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    return int(checkpoint["epoch"]), list(checkpoint["history"])


def _train_one_epoch(
    model: ContextualSparseETM,
    optimizer: torch.optim.Optimizer,
    train: sp.csr_matrix,
    random_generator: np.random.Generator,
    batch_size: int,
) -> dict[str, float]:
    """Apply one epoch of the report's reconstruction-plus-KL objective."""
    model.train()
    device = model.rho.device
    reconstruction_values: list[float] = []
    kl_values: list[float] = []
    gradient_values: list[float] = []
    mass_values: list[float] = []
    epoch_started = time.perf_counter()
    order = random_generator.permutation(train.shape[0])
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        x = dense_normalized(train, rows, device)
        theta, kl = model.document_topic_mixture(x, sample=True)
        beta = model.topic_word_distribution()
        reconstruction, effective_mass = raw_count_reconstruction_loss(
            theta,
            beta,
            train[rows],
            device,
        )
        objective = reconstruction + kl.mean()
        if not torch.isfinite(objective):
            message = "Contextual Sparse ETM produced a non-finite loss"
            raise FloatingPointError(message)

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if not all(
            parameter.grad is None or torch.all(torch.isfinite(parameter.grad)).item()
            for parameter in model.parameters()
        ):
            message = "Contextual Sparse ETM produced non-finite gradients"
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

    mean_reconstruction = float(np.mean(reconstruction_values))
    mean_kl = float(np.mean(kl_values))
    return {
        "reconstruction": mean_reconstruction,
        "kl": mean_kl,
        "reconstruction_to_kl_ratio": mean_reconstruction / max(mean_kl, EPSILON),
        "mean_gradient_norm": float(np.mean(gradient_values)),
        "mean_effective_document_mass": float(np.mean(mass_values)),
        "context_scale": float(model.context_scale.detach().cpu()),
        "seconds": time.perf_counter() - epoch_started,
    }


@torch.inference_mode()
def infer_document_topic_mixtures(
    model: ContextualSparseETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Infer deterministic ``theta`` with ``z = mu_tilde``.

    This is the inference case of equation ``eq:theta`` in the report.
    """
    model.eval()
    device = model.rho.device
    batches: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start,
            min(start + batch_size, matrix.shape[0]),
            dtype=np.int64,
        )
        x = dense_normalized(matrix, rows, device)
        theta, _ = model.document_topic_mixture(x, sample=False)
        batches.append(theta.cpu().numpy().astype(np.float32))
    elapsed = time.perf_counter() - started
    return np.concatenate(batches), matrix.shape[0] / max(elapsed, EPSILON)


@torch.inference_mode()
def infer_contextual_evidence(
    model: ContextualSparseETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
) -> np.ndarray:
    """Infer the auditable evidence matrix ``r`` from ``eq:document-evidence``."""
    model.eval()
    device = model.rho.device
    batches: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start,
            min(start + batch_size, matrix.shape[0]),
            dtype=np.int64,
        )
        x = dense_normalized(matrix, rows, device)
        r = model.contextual_evidence(x)
        batches.append(r.cpu().numpy().astype(np.float32))
    return np.concatenate(batches)


def _artifact_record(path: Path) -> dict[str, object]:
    """Return path, byte count, and digest for one locally retained artifact."""
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _evaluate_and_persist(
    model: ContextualSparseETM,
    data: ValidationData,
    output: Path,
    settings: TrainingSettings,
    training: TrainingOutcome,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run deterministic validation, calculate metrics, and save model outputs."""
    with torch.inference_mode():
        beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
    theta_observed, observed_throughput = infer_document_topic_mixtures(
        model,
        data.observed,
        batch_size=settings.batch_size,
    )
    theta_full, full_throughput = infer_document_topic_mixtures(
        model,
        data.full,
        batch_size=settings.batch_size,
    )
    evidence = infer_contextual_evidence(
        model,
        data.full,
        batch_size=settings.batch_size,
    )
    validate_probability_matrix(theta_observed, name="validation observed theta")
    validate_probability_matrix(theta_full, name="validation full theta")
    validate_probability_matrix(evidence, name="validation contextual evidence")

    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        data.vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    support = theta_support_diagnostics(theta_full)
    evidence_support = theta_support_diagnostics(evidence)
    _, top_words = topic_word_diagnostics(beta, data.vocabulary)
    metrics = {
        "document_completion": completion_metrics(
            theta_observed,
            beta,
            data.completion,
            data.records,
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
            and np.all(np.isfinite(evidence)),
        ),
        "learned_context_scale": float(model.context_scale.detach().cpu()),
        "runtime": {
            "training_wall_seconds": float(
                sum(float(row["seconds"]) for row in training.history),
            ),
            "validation_observed_spectra_per_second": observed_throughput,
            "validation_full_spectra_per_second": full_throughput,
            "memory": runtime_memory_metrics(
                training.memory_state,
                model.rho.device,
            ),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }

    weights_path = output / "weights.pt"
    beta_path = output / "beta.npy"
    observed_path = output / "validation_observed_theta.npy"
    full_path = output / "validation_full_theta.npy"
    evidence_path = output / "validation_routing_evidence.npy"
    checkpoint_path = output / "checkpoint.pt"
    atomic_torch_save(
        weights_path,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    atomic_save_numpy(beta_path, beta)
    atomic_save_numpy(observed_path, theta_observed)
    atomic_save_numpy(full_path, theta_full)
    atomic_save_numpy(evidence_path, evidence)
    write_csv_rows(output / "top_words.csv", top_words)
    write_csv_rows(
        output / "theta_support_summary.csv",
        [flatten_support_summary(support)],
    )
    write_csv_rows(
        output / "routing_evidence_support_summary.csv",
        [flatten_support_summary(evidence_support)],
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
    save_validation(data.run_directory, REAL_METHOD, beta, theta_full, metrics)

    provenance = {
        "validation_inputs": data.input_manifest,
        "candidate_test_artifacts_accessed": False,
        "candidate_test_metrics_inspected": False,
        "local_artifacts": [
            _artifact_record(path)
            for path in (
                weights_path,
                beta_path,
                observed_path,
                full_path,
                evidence_path,
                checkpoint_path,
            )
        ],
    }
    write_json(output / "provenance.json", provenance)
    return metrics, provenance


def train_real_validation(
    real_run: Path,
    *,
    device: torch.device,
    settings: TrainingSettings,
) -> dict[str, Any]:
    """Train the model on frozen training data and evaluate validation only."""
    _validate_settings(settings)
    run_directory = real_run.expanduser().resolve(strict=True)
    output = run_directory / "models" / REAL_METHOD
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json_object(result_path)

    data = _load_validation_data(run_directory)
    training_seed = resolve_training_seed(data.data_seed, settings.requested_seed)
    configure_deterministic_execution(training_seed, settings.threads)
    model = ContextualSparseETM(
        data.embeddings,
        data.topics,
        data.fragment_mask,
        hidden=800,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    random_generator = np.random.default_rng(training_seed + 18)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    contract = _training_contract(data, settings, training_seed)
    start_epoch, history = _restore_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        random_generator,
        contract,
    )
    write_json(
        output / "config.json",
        _run_configuration(
            data,
            settings,
            ExecutionState(
                training_seed=training_seed,
                device=device,
                resumed_from_epoch=start_epoch,
            ),
        ),
    )
    write_json(output / TRAINING_ACCESS_AUDIT_FILENAME, data.input_manifest)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory_state = sample_runtime_memory()
    for epoch_index in range(start_epoch, settings.epochs):
        epoch_metrics = _train_one_epoch(
            model,
            optimizer,
            data.train,
            random_generator,
            settings.batch_size,
        )
        row = {"epoch": epoch_index + 1, **epoch_metrics}
        history.append(row)
        memory_state = sample_runtime_memory(memory_state)
        write_csv_rows(output / "training_history.csv", history)
        if (
            epoch_index + 1
        ) % CHECKPOINT_INTERVAL_EPOCHS == 0 or epoch_index + 1 == settings.epochs:
            # Store every mutable state needed to continue the exact random and
            # optimizer trajectory after interruption.
            atomic_torch_save(
                checkpoint_path,
                {
                    "epoch": epoch_index + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "numpy_rng_state": random_generator.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "history": history,
                    "training_contract": contract,
                },
            )
        print(  # noqa: T201
            "CONTEXTUAL_SPARSE_ETM_EPOCH",
            json.dumps(row, sort_keys=True),
            flush=True,
        )

    model.eval()
    metrics, provenance = _evaluate_and_persist(
        model,
        data,
        output,
        settings,
        TrainingOutcome(history=history, memory_state=memory_state),
    )
    result = {
        "model": MODEL_NAME,
        "method": REAL_METHOD,
        "parameters": metrics["parameters"],
        "config": read_json_object(output / "config.json"),
        "metrics": metrics,
        "provenance": provenance,
    }
    write_json(result_path, result)
    return result


def score_real_validation(real_run: Path, data_root: Path) -> dict[str, Any]:
    """Run the unchanged leakage-controlled MAG/SOS validation evaluation."""
    return score_chemical_validation(
        real_run.expanduser().resolve(),
        data_root.expanduser().resolve(),
        REAL_METHOD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run training/inference or chemical validation scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--real-run", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--threads", type=int, default=6)
    train.add_argument(
        "--training-seed",
        type=int,
        help="override only model initialization and training order",
    )
    score = commands.add_parser("chemical")
    score.add_argument("--real-run", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_real_validation(
            args.real_run,
            device=resolve_torch_device(args.device),
            settings=TrainingSettings(
                epochs=args.epochs,
                batch_size=args.batch_size,
                threads=args.threads,
                requested_seed=args.training_seed,
            ),
        )
    elif args.command == "chemical":
        result = score_real_validation(args.real_run, args.data_root)
    else:
        message = "unknown Contextual Sparse ETM command"
        raise RuntimeError(message)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
