"""Run bounded truth-known sparse-ETM mechanism experiments.

The synthetic path never opens real MSnLib artifacts. A separate validation-
only real-data path is added only after the synthetic promotion decision.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment

from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
    train_token_features,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.followup import theta_distribution
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.sparse_etm import (
    RECONSTRUCTION_SCALINGS,
    THETA_TRANSFORMS,
    BalancedSparseETM,
    ReconstructionScaling,
    ThetaTransform,
    dense_normalized,
    sparse_reconstruction_loss,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.synthetic_msms import (
    SyntheticMsmsDataset,
    generate_synthetic_msms,
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

if TYPE_CHECKING:
    from collections.abc import Sequence

EPS = 1e-12
LOOSE_RECOVERY_COSINE = 0.20
PRIMARY_RECOVERY_COSINE = 0.50
SYNTHETIC_ACTIVE_USAGE_THRESHOLD = 0.005
REAL_METHOD = "etm_balanced_entmax15_distinct_words"
REAL_TOPICS = 1000
REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_PROTOCOL_PATH = REPO_ROOT / "benchmarks/neural_ms2lda/protocol.json"
SYNTHETIC_EVALUATION_PROTOCOL = {
    "active_topic_usage_threshold": 0.0005,
    "duplicate_cosine_thresholds": (0.95, 0.99, 0.999),
    "catastrophic_duplicate_component_fraction": 0.5,
    "top_word_count": 20,
    "channel_extreme_lower": 0.1,
    "channel_extreme_upper": 0.9,
}
REAL_VALIDATION_DATA_FILES = (
    "train.npz",
    "validation_observed.npz",
    "validation_completion.npz",
    "validation_full.npz",
    "validation_records.jsonl",
    "vocabulary.json",
)
REAL_MAG_INDEX_FILES = (
    "complete.json",
    "excluded_connectivity_keys.json",
    "kept_original_ids.npy",
    "spec2vec_filtered.faiss",
)


def configure(seed: int, threads: int) -> None:
    """Apply deterministic CPU/CUDA random and linear-algebra settings."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)  # noqa: FBT003
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def resolve_device(name: str) -> torch.device:
    """Resolve a requested CPU or CUDA device."""
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but is unavailable"
        raise RuntimeError(message)
    if selected not in {"cpu", "cuda"}:
        message = "device must be auto, cpu, or cuda"
        raise ValueError(message)
    return torch.device(selected)


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    with path.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic, small experiment tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_label(
    transform: ThetaTransform,
    scaling: ReconstructionScaling,
) -> str:
    """Return the stable artifact label for one isolated formulation."""
    return f"balanced_etm_{transform}_{scaling}"


def _save_dataset_artifacts(
    directory: Path,
    dataset: SyntheticMsmsDataset,
) -> dict[str, dict[str, Any]]:
    """Persist the small truth-known dataset so its exact identity is auditable."""
    directory.mkdir(parents=True, exist_ok=True)
    matrices = {
        "train": dataset.train,
        "validation_observed": dataset.validation_observed,
        "validation_completion": dataset.validation_completion,
        "validation_full": dataset.validation_full,
    }
    for name, matrix in matrices.items():
        path = directory / f"{name}.npz"
        if not path.is_file():
            sp.save_npz(path, matrix, compressed=False)
    arrays = {
        "true_beta": dataset.true_beta,
        "train_true_theta": dataset.train_true_theta,
        "validation_true_theta": dataset.validation_true_theta,
    }
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        if not path.is_file():
            atomic_save_numpy(path, array.astype(np.float32, copy=False))
    vocabulary_path = directory / "vocabulary.json"
    if not vocabulary_path.is_file():
        write_json(vocabulary_path, {"vocabulary": list(dataset.vocabulary)})
    records_path = directory / "validation_records.jsonl"
    if not records_path.is_file():
        with records_path.open("w", encoding="utf-8") as handle:
            for record in dataset.validation_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(directory / "summary.json", dataset.summary)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        path.name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    }
    write_json(directory / "artifact_manifest.json", manifest)
    return manifest


def prepare_synthetic_seed(
    output_root: Path,
    *,
    seed: int,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> tuple[SyntheticMsmsDataset, np.ndarray, Path]:
    """Generate one seed and train/reuse its single train-only SGNS table."""
    dataset = generate_synthetic_msms(
        seed=seed,
        training_documents=training_documents,
        validation_documents=validation_documents,
    )
    seed_directory = output_root / "synthetic_artifacts" / f"seed_{seed}"
    _save_dataset_artifacts(seed_directory, dataset)
    protocol = read_json(SYNTHETIC_PROTOCOL_PATH)
    configure(seed, threads)
    train_token_features(
        seed_directory / "token_features",
        dataset.train,
        list(dataset.vocabulary),
        protocol,
        seed=seed,
    )
    features_path = seed_directory / "token_features/features.npy"
    features = np.load(features_path).astype(np.float32, copy=False)
    embeddings = np.asarray(features[:, :-2], dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    return dataset, embeddings, seed_directory


@torch.inference_mode()
def infer_theta(
    model: BalancedSparseETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Infer deterministic mixtures and report spectra per second."""
    model.eval()
    values = []
    started = time.perf_counter()
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start,
            min(start + batch_size, matrix.shape[0]),
            dtype=np.int64,
        )
        theta, _ = model.theta(
            dense_normalized(matrix, rows, device),
            sample=False,
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPS)


def _matched_truth_metrics(
    learned_beta: np.ndarray,
    learned_theta: np.ndarray,
    true_beta: np.ndarray,
    true_theta: np.ndarray,
) -> dict[str, Any]:
    """Align planted motifs one-to-one and report beta/theta recovery."""
    learned_norm = learned_beta / np.maximum(
        np.linalg.norm(learned_beta, axis=1, keepdims=True),
        EPS,
    )
    true_norm = true_beta / np.maximum(
        np.linalg.norm(true_beta, axis=1, keepdims=True),
        EPS,
    )
    similarity = true_norm @ learned_norm.T
    true_rows, learned_rows = linear_sum_assignment(-similarity)
    matched = similarity[true_rows, learned_rows]
    aligned = np.zeros_like(true_theta, dtype=np.float64)
    aligned[:, true_rows] = learned_theta[:, learned_rows]
    numerator = np.sum(true_theta * aligned, axis=1)
    denominator = np.linalg.norm(true_theta, axis=1) * np.linalg.norm(aligned, axis=1)
    theta_cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )
    top_n = min(20, learned_beta.shape[1])
    jaccards = []
    for truth, learned in zip(true_rows, learned_rows, strict=True):
        truth_top = set(np.argsort(-true_beta[truth], kind="stable")[:top_n])
        learned_top = set(np.argsort(-learned_beta[learned], kind="stable")[:top_n])
        jaccards.append(len(truth_top & learned_top) / len(truth_top | learned_top))
    return {
        "true_beta_matched_cosine_mean": float(matched.mean()),
        "true_beta_matched_cosine_median": float(np.median(matched)),
        "true_beta_matched_cosine_minimum": float(matched.min()),
        "true_beta_top20_jaccard_mean": float(np.mean(jaccards)),
        "true_theta_cosine_mean": float(theta_cosine.mean()),
        "true_theta_cosine_median": float(np.median(theta_cosine)),
        "top_planted_motif_accuracy": float(
            np.mean(np.argmax(true_theta, axis=1) == np.argmax(aligned, axis=1)),
        ),
        "planted_motifs_recovered_cosine_gt_0_20": int(
            np.sum(matched > LOOSE_RECOVERY_COSINE),
        ),
        "planted_motifs_recovered_cosine_ge_0_50": int(
            np.sum(matched >= PRIMARY_RECOVERY_COSINE),
        ),
        "matching": [
            {
                "true_topic": int(truth),
                "learned_topic": int(learned),
                "beta_cosine": float(similarity[truth, learned]),
            }
            for truth, learned in zip(true_rows, learned_rows, strict=True)
        ],
    }


def run_synthetic(  # noqa: PLR0913, PLR0915
    output_root: Path,
    *,
    seed: int,
    fitted_topics: int,
    theta_transform: ThetaTransform,
    reconstruction_scaling: ReconstructionScaling,
    epochs: int,
    batch_size: int,
    device: torch.device,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> dict[str, Any]:
    """Train and evaluate one isolated synthetic sparse-ETM formulation."""
    label = method_label(theta_transform, reconstruction_scaling)
    output = output_root / "synthetic_runs" / f"seed_{seed}_K_{fitted_topics}_{label}"
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "evidence": "truth-known synthetic train and validation only",
        "seed": int(seed),
        "true_topics": 18,
        "fitted_topics": int(fitted_topics),
        "theta_transform": theta_transform,
        "reconstruction_scaling": reconstruction_scaling,
        "fixed_train_only_sgns_dimensions": 48,
        "fragment_loss_beta_mass": [0.5, 0.5],
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
    }
    write_json(output / "config.json", config)
    dataset, embeddings, seed_directory = prepare_synthetic_seed(
        output_root,
        seed=seed,
        threads=threads,
        training_documents=training_documents,
        validation_documents=validation_documents,
    )
    configure(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in dataset.vocabulary],
        dtype=bool,
    )
    model = BalancedSparseETM(
        embeddings,
        fitted_topics,
        fragment_mask,
        theta_transform=theta_transform,
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
        effective_mass_values = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            theta, kl = model.theta(
                dense_normalized(dataset.train, rows, device),
                sample=True,
            )
            reconstruction, effective_mass = sparse_reconstruction_loss(
                theta,
                model.beta(),
                dataset.train,
                rows,
                device,
                scaling=reconstruction_scaling,
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                message = "sparse ETM produced non-finite objective"
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            finite_gradients = all(
                parameter.grad is None
                or torch.all(torch.isfinite(parameter.grad)).item()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                message = "sparse ETM produced non-finite gradients"
                raise FloatingPointError(message)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
            effective_mass_values.append(effective_mass)
        row = {
            "epoch": epoch + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "reconstruction_to_kl_ratio": float(
                np.mean(reconstruction_values) / max(np.mean(kl_values), EPS),
            ),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "mean_effective_document_mass": float(np.mean(effective_mass_values)),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print(  # noqa: T201
            "SPARSE_ETM_EPOCH",
            json.dumps({"run": label, **row}),
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
        **diagnostics,
        "active_topics_mean_usage_gt_0_005": int(
            np.sum(usage > SYNTHETIC_ACTIVE_USAGE_THRESHOLD),
        ),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full)),
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
    provenance = {
        "synthetic_artifact_manifest": str(
            seed_directory / "artifact_manifest.json",
        ),
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


def _ensure_file_link(destination: Path, source: Path) -> None:
    """Create one immutable-input symlink and reject conflicting destinations."""
    resolved_source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != resolved_source:
            message = f"existing link has a different source: {destination}"
            raise ValueError(message)
        return
    if destination.exists():
        message = f"validation-view destination already exists: {destination}"
        raise FileExistsError(message)
    destination.symlink_to(resolved_source)


def prepare_real_validation_view(
    real_run: Path,
    prepared_run: Path,
) -> dict[str, Any]:
    """Expose only frozen training/validation inputs in a new writable run view."""
    source = prepared_run.expanduser().resolve(strict=True)
    destination = real_run.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = source / "protocol.json"
    protocol = read_json(protocol_path)
    if int(protocol["model"]["num_topics"]) != REAL_TOPICS:
        message = "real sparse-ETM validation requires the frozen K=1000 protocol"
        raise ValueError(message)
    write_json(destination / "protocol.json", protocol)

    linked_sources: list[Path] = []
    for name in REAL_VALIDATION_DATA_FILES:
        source_path = source / "data" / name
        _ensure_file_link(destination / "data" / name, source_path)
        linked_sources.append(source_path)
    features_path = source / "token_features" / "features.npy"
    _ensure_file_link(destination / "token_features" / "features.npy", features_path)
    linked_sources.append(features_path)
    for name in REAL_MAG_INDEX_FILES:
        source_path = source / "mag" / "index" / name
        _ensure_file_link(destination / "mag" / "index" / name, source_path)
        linked_sources.append(source_path)

    manifest = {
        "evidence_boundary": "training plus validation only",
        "prepared_run": str(source),
        "candidate_test_artifacts_accessed": False,
        "candidate_test_metrics_inspected": False,
        "linked_inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in [protocol_path, *linked_sources]
        ],
    }
    write_json(destination / "validation_input_manifest.json", manifest)
    return manifest


class RuntimeMemoryTracker:
    """Track process, system-available, and CUDA high-water memory."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_process_bytes = 0
        self.minimum_system_available_bytes: int | None = None
        self.sample()

    @staticmethod
    def _system_available_bytes() -> int | None:
        path = Path("/proc/meminfo")
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return None

    def sample(self) -> None:
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        self.peak_process_bytes = max(self.peak_process_bytes, usage)
        available = self._system_available_bytes()
        if available is not None:
            current = self.minimum_system_available_bytes
            self.minimum_system_available_bytes = (
                available if current is None else min(current, available)
            )

    def result(self) -> dict[str, Any]:
        self.sample()
        return {
            "measurement": (
                "sampled per epoch; process is Linux ru_maxrss and system is "
                "minimum /proc/meminfo MemAvailable"
            ),
            "peak_process_bytes": self.peak_process_bytes,
            "minimum_system_available_bytes": self.minimum_system_available_bytes,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(self.device))
                if self.device.type == "cuda"
                else None
            ),
        }


def _validate_theta_contract(theta: np.ndarray, *, name: str) -> None:
    """Fail closed unless inferred theta is a finite non-negative simplex."""
    if not np.all(np.isfinite(theta)):
        message = f"{name} theta contains non-finite values"
        raise FloatingPointError(message)
    if np.any(theta < 0):
        message = f"{name} theta contains negative values"
        raise FloatingPointError(message)
    row_sums = theta.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, atol=2e-6):
        maximum_deviation = float(np.max(np.abs(row_sums - 1.0)))
        message = (
            f"{name} theta rows do not sum to one: "
            f"minimum={row_sums.min():.9g}, maximum={row_sums.max():.9g}, "
            f"maximum_deviation={maximum_deviation:.9g}"
        )
        raise FloatingPointError(message)


def _flat_theta_support(summary: dict[str, object]) -> dict[str, object]:
    """Flatten support percentiles for a one-row reviewable CSV."""
    row = {
        key: value
        for key, value in summary.items()
        if key != "support_size_percentiles"
    }
    percentiles = dict(summary["support_size_percentiles"])  # type: ignore[arg-type]
    row.update({f"support_p{key}": value for key, value in percentiles.items()})
    return row


def train_real_validation(  # noqa: C901, PLR0912, PLR0913, PLR0915
    real_run: Path,
    prepared_run: Path,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    """Train the one promoted sparse ETM on frozen MSnLib validation inputs."""
    real_run = real_run.expanduser().resolve()
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
        message = "frozen MSnLib train, vocabulary, and SGNS shapes do not align"
        raise ValueError(message)
    if observed.shape != full.shape or completion.shape != full.shape:
        message = "frozen MSnLib validation matrix shapes do not align"
        raise ValueError(message)
    if full.shape[0] != len(records):
        message = "frozen validation records and matrices differ"
        raise ValueError(message)

    configure(seed + 7001, threads)
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary],
        dtype=bool,
    )
    model = BalancedSparseETM(
        embeddings,
        topics,
        fragment_mask,
        theta_transform="entmax15",
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
        "theta_transform": "entmax15",
        "reconstruction_scaling": "distinct_words",
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
            message = "real sparse-ETM checkpoint contract does not match"
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
        "prepared_run": str(prepared_run.expanduser().resolve()),
        "method": REAL_METHOD,
        "topics": topics,
        "fixed_train_only_sgns_dimensions": int(embeddings.shape[1]),
        "fragment_loss_beta_mass": [0.5, 0.5],
        "hidden_dimensions": 800,
        "theta_transform": "entmax15",
        "variational_posterior": "Gaussian with standard-normal latent KL",
        "reconstruction_scaling": "distinct_words",
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
        effective_mass_values = []
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
                scaling="distinct_words",
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                message = "real sparse ETM produced a non-finite objective"
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            finite_gradients = all(
                parameter.grad is None
                or torch.all(torch.isfinite(parameter.grad)).item()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                message = "real sparse ETM produced non-finite gradients"
                raise FloatingPointError(message)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
            effective_mass_values.append(effective_mass)
        memory.sample()
        row = {
            "epoch": epoch + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "reconstruction_to_kl_ratio": float(
                np.mean(reconstruction_values) / max(np.mean(kl_values), EPS),
            ),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "mean_effective_document_mass": float(np.mean(effective_mass_values)),
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
            "REAL_SPARSE_ETM_EPOCH",
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
    _validate_theta_contract(theta_observed, name="validation observed")
    _validate_theta_contract(theta_full, name="validation full")
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    support = theta_support_diagnostics(theta_full)
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
        "theta_distribution": theta_distribution(theta_full),
        "theta_information": entropy_diagnostics(theta_full),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full)),
        ),
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
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "validation_observed_theta.npy", theta_observed)
    atomic_save_numpy(output / "validation_full_theta.npy", theta_full)
    write_csv(output / "top_words.csv", top_words)
    write_csv(output / "theta_support_summary.csv", [_flat_theta_support(support)])
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
        output / "weights.pt",
        output / "beta.npy",
        output / "validation_observed_theta.npy",
        output / "validation_full_theta.npy",
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
    """Run one campaign command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic-run")
    synthetic.add_argument("--output-root", required=True, type=Path)
    synthetic.add_argument("--seed", required=True, type=int)
    synthetic.add_argument("--fitted-topics", required=True, type=int)
    synthetic.add_argument(
        "--theta-transform",
        required=True,
        choices=THETA_TRANSFORMS,
    )
    synthetic.add_argument(
        "--reconstruction-scaling",
        required=True,
        choices=RECONSTRUCTION_SCALINGS,
    )
    synthetic.add_argument("--epochs", type=int, default=120)
    synthetic.add_argument("--batch-size", type=int, default=200)
    synthetic.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    synthetic.add_argument("--threads", type=int, default=6)
    synthetic.add_argument("--training-documents", type=int, default=800)
    synthetic.add_argument("--validation-documents", type=int, default=160)
    real = commands.add_parser("real-run")
    real.add_argument("--real-run", required=True, type=Path)
    real.add_argument("--prepared-run", required=True, type=Path)
    real.add_argument("--epochs", type=int, default=120)
    real.add_argument("--batch-size", type=int, default=256)
    real.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    real.add_argument("--threads", type=int, default=6)
    score = commands.add_parser("real-chemical")
    score.add_argument("--real-run", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "synthetic-run":
        result = run_synthetic(
            args.output_root.expanduser().resolve(),
            seed=args.seed,
            fitted_topics=args.fitted_topics,
            theta_transform=args.theta_transform,
            reconstruction_scaling=args.reconstruction_scaling,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=resolve_device(args.device),
            threads=args.threads,
            training_documents=args.training_documents,
            validation_documents=args.validation_documents,
        )
    elif args.command == "real-run":
        result = train_real_validation(
            args.real_run,
            args.prepared_run,
            device=resolve_device(args.device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    elif args.command == "real-chemical":
        result = score_real_validation(args.real_run, args.data_root)
    else:
        message = "unknown campaign command"
        raise RuntimeError(message)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
