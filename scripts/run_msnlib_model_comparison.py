"""Validation-only MSnLib ETM and pooled-model comparison campaign.

The runner reuses the locked repository preparation, completion, and chemical
scoring machinery. Its candidate training/evaluation commands do not load or
score test matrices or result artifacts. The pre-existing shared MAG leakage
index may encode held-out compound identifiers from both splits, but candidate
selection remains based only on validation theta and validation chemistry.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import signal
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as nnf

from benchmarks.neural_ms2lda.artifacts import initialize_run
from benchmarks.neural_ms2lda.data import (
    iter_row_batches,
    load_csr,
    load_heldout_records,
    load_vocabulary,
    prepare_data,
    sparse_batch,
    train_token_features,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.followup import retemperature_theta, theta_distribution
from benchmarks.neural_ms2lda.model_evaluation import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
    TRAINING_ACCESS_AUDIT_FILENAME,
    entropy_diagnostics,
    save_validation,
    score_chemical_validation,
    topic_word_diagnostics,
)
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.pooled import (
    assignment_information_loss,
    batch_to_device,
    infer_pooled_theta,
    initialize_pooled_candidate,
)
from benchmarks.neural_ms2lda.reproducibility import normalize_probability_rows
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    read_json,
    write_json,
)
from scripts.run_published_topic_models_msnlib import (
    ECR,
    FixedETM,
    TopMostECRTM,
    ecrtm_completion,
    infer_ecrtm,
    sgns_only,
)

EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]
POOLED_ROOT = REPO_ROOT / "research/etm_ecrtm_msnlib/pooled_projected"
ETM_METHODS = ("etm", "etm_balanced", "etm_balanced_gated")
METHODS = ETM_METHODS + ("pooled_likelihood", "pooled_mi005")
GATED_ETM_PREFIX = "etm_balanced_gated_"
# Preserve the historical script-level callable for existing command imports.
chemical = score_chemical_validation
POOLED_PROTOCOLS = {
    "pooled_likelihood": POOLED_ROOT / "protocol_minimum.json",
    "pooled_mi005": POOLED_ROOT / "protocol_mi005.json",
}


def file_sha256(path: Path) -> str:
    """Hash an execution input without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_access_manifest(run: Path) -> dict[str, Any]:
    """Describe and hash every candidate data artifact opened by ETM training."""
    relative_paths = (
        "protocol.json",
        "data/train.npz",
        "data/validation_observed.npz",
        "data/validation_completion.npz",
        "data/validation_full.npz",
        "data/validation_records.jsonl",
        "data/vocabulary.json",
        "token_features/features.npy",
    )
    artifacts = []
    for relative in relative_paths:
        path = run / relative
        artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "evidence_boundary": "training plus validation only",
        "candidate_test_artifacts_loaded": False,
        "candidate_test_metrics_inspected": False,
        "data_loader": "load_etm_campaign_data",
        "loaded_artifacts": artifacts,
    }


class MemoryTracker:
    """Track process and accelerator allocation high-water marks."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_process_bytes = 0
        self.peak_mps_allocated_bytes = 0
        self.peak_mps_driver_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()

    def sample(self) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process_bytes = int(usage if platform.system() == "Darwin" else usage * 1024)
        self.peak_process_bytes = max(self.peak_process_bytes, process_bytes)
        if self.device.type == "mps":
            self.peak_mps_allocated_bytes = max(
                self.peak_mps_allocated_bytes,
                int(torch.mps.current_allocated_memory()),
            )
            self.peak_mps_driver_bytes = max(
                self.peak_mps_driver_bytes,
                int(torch.mps.driver_allocated_memory()),
            )

    def result(self) -> dict[str, Any]:
        self.sample()
        return {
            "measurement": "sampled high-water marks; process ru_maxrss is OS reported",
            "peak_process_bytes": self.peak_process_bytes,
            "peak_mps_allocated_bytes": (
                self.peak_mps_allocated_bytes if self.device.type == "mps" else None
            ),
            "peak_mps_driver_bytes": (
                self.peak_mps_driver_bytes if self.device.type == "mps" else None
            ),
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


class FragmentLossBalancedETM(FixedETM):
    """Canonical fixed-SGNS ETM with only channel-wise beta normalization."""

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        hidden: int = 800,
    ) -> None:
        super().__init__(embeddings, topics, hidden=hidden)
        mask = torch.as_tensor(fragment_mask, dtype=torch.bool)
        if mask.ndim != 1 or len(mask) != len(embeddings):
            raise ValueError("fragment mask must match the ETM vocabulary")
        if not torch.any(mask) or torch.all(mask):
            raise ValueError("fragment mask must contain fragments and losses")
        self.register_buffer("fragment_mask", mask, persistent=False)

    def beta(self) -> torch.Tensor:
        """Normalize fragment and loss logits independently to mass 0.5."""
        logits = self.alphas(self.rho).T
        probabilities = torch.empty_like(logits)
        probabilities[:, self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, self.fragment_mask], dim=1
        )
        probabilities[:, ~self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, ~self.fragment_mask], dim=1
        )
        return probabilities


class GatedFragmentLossBalancedETM(FragmentLossBalancedETM):
    """Balanced ETM with detached shared-geometry document evidence.

    The ordinary ETM variational encoder still produces ``theta``. A pooled
    count-weighted SGNS document vector is compared with ETM's existing topic
    vectors, and the resulting gate reweights ``theta`` during reconstruction.
    Detaching the gate prevents it from becoming a second learned encoder while
    preserving reconstruction gradients through ETM theta and beta.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        gate_temperature: float = 1.0,
        gate_gamma: float = 1.0,
        hidden: int = 800,
    ) -> None:
        super().__init__(embeddings, topics, fragment_mask, hidden=hidden)
        if not math.isfinite(gate_temperature) or gate_temperature <= 0:
            raise ValueError("gate temperature must be finite and positive")
        if not math.isfinite(gate_gamma) or gate_gamma < 0:
            raise ValueError("gate gamma must be finite and non-negative")
        self.gate_temperature = float(gate_temperature)
        self.gate_gamma = float(gate_gamma)

    def document_gate(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """Return shared-geometry gate probabilities before detachment."""
        document_geometry = nnf.normalize(normalized_bows @ self.rho, dim=1)
        topic_geometry = nnf.normalize(self.alphas.weight, dim=1)
        logits = 2.0 * (document_geometry @ topic_geometry.T) / self.gate_temperature
        return nnf.softmax(logits, dim=1)

    def apply_document_gate(
        self,
        theta: torch.Tensor,
        normalized_bows: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight ETM theta by detached evidence and renormalize rows."""
        if self.gate_gamma == 0.0:
            return theta
        gate = self.document_gate(normalized_bows).detach()
        gated = theta * gate.pow(self.gate_gamma)
        return gated / gated.sum(dim=1, keepdim=True).clamp_min(EPS)

    def theta(
        self,
        normalized_bows: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the detached gate in both stochastic training and inference."""
        theta, kl = super().theta(normalized_bows, sample=sample)
        return self.apply_document_gate(theta, normalized_bows), kl


def _float_label(value: float) -> str:
    """Return a filesystem-safe compact label for one frozen scalar."""
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def gated_method_name(gate_temperature: float, gate_gamma: float) -> str:
    """Name separately trained gate-strength variants without ambiguity."""
    return (
        f"{GATED_ETM_PREFIX}t{_float_label(gate_temperature)}"
        f"_g{_float_label(gate_gamma)}"
    )


def is_gated_etm_artifact(method: str) -> bool:
    """Return whether a result label belongs to this gated-ETM campaign."""
    return method.startswith(GATED_ETM_PREFIX)


def configure(seed: int, threads: int) -> None:
    """Apply the locked seed/thread contract."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_device(name: str) -> torch.device:
    """Resolve an operational device without changing model equations."""
    selected = name
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        else:
            selected = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if selected not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(selected)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a small deterministic CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def dense_normalized(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Create the canonical normalized-BOW ETM encoder input."""
    values = torch.from_numpy(matrix[rows].toarray().astype(np.float32, copy=False)).to(
        device
    )
    return values / values.sum(1, keepdim=True).clamp_min(1.0)


def load_etm_campaign_data(run: Path) -> dict[str, Any]:
    """Load the training and validation-only matrices used by ETM candidates."""
    data = run / "data"
    return {
        "train": load_csr(data / "train.npz"),
        "observed": load_csr(data / "validation_observed.npz"),
        "completion": load_csr(data / "validation_completion.npz"),
        "full": load_csr(data / "validation_full.npz"),
        "records": load_heldout_records(data, "validation"),
    }


def sparse_reconstruction(
    theta: torch.Tensor,
    beta: torch.Tensor,
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Exact raw-count ETM reconstruction over observed entries."""
    batch = matrix[rows].tocsr()
    if batch.nnz == 0:
        return theta.new_zeros(())
    row_ids = torch.from_numpy(
        np.repeat(np.arange(len(rows), dtype=np.int64), np.diff(batch.indptr))
    ).to(device)
    word_ids = torch.from_numpy(batch.indices.astype(np.int64, copy=False)).to(device)
    weights = torch.from_numpy(batch.data.astype(np.float32, copy=False)).to(device)
    probability = torch.sum(theta[row_ids] * beta[:, word_ids].T, dim=1).clamp_min(EPS)
    return -torch.sum(weights * torch.log(probability)) / len(rows)


@torch.inference_mode()
def infer_etm(
    model: FixedETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Infer deterministic canonical ETM mixtures and throughput."""
    model.eval()
    synchronize_device(device)
    started = time.perf_counter()
    values = []
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start, min(start + batch_size, matrix.shape[0]), dtype=np.int64
        )
        theta, _ = model.theta(dense_normalized(matrix, rows, device), sample=False)
        values.append(theta.cpu().numpy().astype(np.float32))
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / elapsed


def mixture_diagnostics(theta: np.ndarray, beta: np.ndarray) -> dict[str, Any]:
    """Measure inventory usage, per-spectrum sparsity, and topic redundancy."""
    values = theta.astype(np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), EPS)
    usage = values.mean(axis=0)
    entropy = -np.sum(values * np.log(np.clip(values, EPS, None)), axis=1)
    normalized = beta.astype(np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), EPS)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)
    return {
        "median_effective_topics_per_spectrum": float(np.median(np.exp(entropy))),
        "corpus_effective_topic_count": float(
            np.exp(-np.sum(usage * np.log(np.clip(usage, EPS, None))))
        ),
        "active_topics_mean_usage_gt_0_0005": int(np.sum(usage > 0.0005)),
        "active_topics_mean_usage_ge_1_over_k": int(np.sum(usage >= 1.0 / len(usage))),
        "maximum_mean_topic_usage": float(usage.max()),
        "mean_nearest_topic_beta_cosine": float(np.max(similarity, axis=1).mean()),
        "maximum_pairwise_beta_cosine": float(similarity.max()),
    }


def prepare(run: Path, data_root: Path) -> dict[str, Any]:
    """Run the exact repository preparation and train-only SGNS stages."""
    protocol = initialize_run(run, data_root=data_root)
    configure(int(protocol["seed"]), int(protocol["cpu_threads"]))
    data_result = prepare_data(run, data_root=data_root, protocol=protocol)
    train = load_csr(run / "data/train.npz")
    vocabulary = load_vocabulary(run / "data")
    feature_result = train_token_features(
        run / "token_features",
        train,
        vocabulary,
        protocol,
        seed=int(protocol["seed"]),
    )
    result = {
        "data": data_result,
        "token_features": feature_result,
        "train_shape": list(train.shape),
        "vocabulary_size": len(vocabulary),
        "evidence_boundary": "validation candidates only; test is not evaluated",
    }
    write_json(run / "comparison_preparation.json", result)
    return result


def _base_evaluation(
    *,
    theta_observed: np.ndarray,
    theta_full: np.ndarray,
    beta: np.ndarray,
    completion: sp.csr_matrix,
    records: list[dict[str, Any]],
    vocabulary: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, top_words = topic_word_diagnostics(beta, vocabulary)
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    metrics = {
        "document_completion": completion_metrics(
            theta_observed, beta, completion, records
        ),
        **diagnostics,
        "theta_distribution": theta_distribution(theta_full),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full))
        ),
    }
    return metrics, top_words


def train_etm(
    run: Path,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    method: str = "etm",
    gate_temperature: float = 1.0,
    gate_gamma: float = 1.0,
) -> dict[str, Any]:
    """Train one validation-only ETM architecture under the locked contract."""
    if method not in ETM_METHODS:
        raise ValueError(f"ETM method must be one of {ETM_METHODS}")
    if method != "etm_balanced_gated" and (
        gate_temperature != 1.0 or gate_gamma != 1.0
    ):
        raise ValueError("gate controls are only valid for etm_balanced_gated")
    artifact_method = (
        gated_method_name(gate_temperature, gate_gamma)
        if method == "etm_balanced_gated"
        else method
    )
    output = run / "models" / artifact_method
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 7001, int(protocol["cpu_threads"]))
    campaign_data = load_etm_campaign_data(run)
    train = campaign_data["train"]
    observed = campaign_data["observed"]
    completion = campaign_data["completion"]
    full = campaign_data["full"]
    records = campaign_data["records"]
    vocabulary = load_vocabulary(run / "data")
    embeddings = sgns_only(run / "token_features/features.npy")
    write_json(
        output / TRAINING_ACCESS_AUDIT_FILENAME,
        validation_access_manifest(run),
    )
    if method in {"etm_balanced", "etm_balanced_gated"}:
        fragment_mask = np.asarray(
            [word.startswith("frag@") for word in vocabulary], dtype=bool
        )
        if method == "etm_balanced_gated":
            model = GatedFragmentLossBalancedETM(
                embeddings,
                int(protocol["model"]["num_topics"]),
                fragment_mask,
                gate_temperature=gate_temperature,
                gate_gamma=gate_gamma,
                hidden=800,
            ).to(device)
        else:
            model = FragmentLossBalancedETM(
                embeddings,
                int(protocol["model"]["num_topics"]),
                fragment_mask,
                hidden=800,
            ).to(device)
    else:
        model = FixedETM(
            embeddings,
            int(protocol["model"]["num_topics"]),
            hidden=800,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1.2e-6)
    rng = np.random.default_rng(seed + 7019)
    memory = MemoryTracker(device)
    history: list[dict[str, Any]] = []
    synchronize_device(device)
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train.shape[0])
        reconstruction_total = 0.0
        kl_total = 0.0
        batches = 0
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            theta, kl = model.theta(dense_normalized(train, rows, device), sample=True)
            beta = model.beta()
            reconstruction = sparse_reconstruction(theta, beta, train, rows, device)
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("ETM produced a non-finite objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            reconstruction_total += float(reconstruction.detach().cpu())
            kl_total += float(kl.mean().detach().cpu())
            batches += 1
        memory.sample()
        synchronize_device(device)
        row = {
            "epoch": epoch + 1,
            "reconstruction": reconstruction_total / batches,
            "kl": kl_total / batches,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print(
            f"{artifact_method.upper()}_EPOCH",
            json.dumps(row, sort_keys=True),
            flush=True,
        )
    synchronize_device(device)
    fitting_seconds = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.beta().cpu().numpy(),
            name=f"{artifact_method} validation beta",
        )
    theta_observed, observed_throughput = infer_etm(
        model,
        observed,
        batch_size=int(batch_size),
        device=device,
    )
    theta_full, full_throughput = infer_etm(
        model,
        full,
        batch_size=int(batch_size),
        device=device,
    )
    metrics, top_words = _base_evaluation(
        theta_observed=theta_observed,
        theta_full=theta_full,
        beta=beta,
        completion=completion,
        records=records,
        vocabulary=vocabulary,
    )
    metrics["runtime"] = {
        "training_wall_seconds": fitting_seconds,
        "validation_observed_spectra_per_second": observed_throughput,
        "validation_full_spectra_per_second": full_throughput,
        "memory": memory.result(),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    config = {
        "architecture": (
            "fixed-pretrained-SGNS ETM with fragment/loss-balanced decoder "
            "and detached shared-geometry document gate"
            if method == "etm_balanced_gated"
            else (
                "fixed-pretrained-SGNS ETM with fragment/loss-balanced decoder"
                if method == "etm_balanced"
                else "canonical fixed-pretrained-SGNS ETM"
            )
        ),
        "embedding_dimensions": 48,
        "hidden_dimensions": 800,
        "topics": int(protocol["model"]["num_topics"]),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "device": str(device),
        "seed": seed + 7001,
        "decoder_normalization": (
            "independent fragment/loss softmaxes at 0.5 each"
            if method in {"etm_balanced", "etm_balanced_gated"}
            else "global topic-word softmax"
        ),
        "paired_reference_method": (
            "etm_balanced"
            if method == "etm_balanced_gated"
            else ("etm" if method == "etm_balanced" else None)
        ),
        "only_scientific_change": (
            "detached shared-geometry document gate participates in every "
            "training reconstruction"
            if method == "etm_balanced_gated"
            else (
                "beta normalization: global softmax -> fixed 0.5 fragment + 0.5 loss"
                if method == "etm_balanced"
                else None
            )
        ),
        "gate_temperature": (
            float(gate_temperature) if method == "etm_balanced_gated" else None
        ),
        "gate_gamma": float(gate_gamma) if method == "etm_balanced_gated" else None,
        "gate_logit_scale": 2.0 if method == "etm_balanced_gated" else None,
        "gate_detached_before_theta_multiplication": (
            True if method == "etm_balanced_gated" else None
        ),
        "gate_used_during_training": (True if method == "etm_balanced_gated" else None),
        "trained_separately": True,
    }
    write_json(output / "config.json", config)
    write_csv(output / "top_words.csv", top_words)
    write_json(
        output / "fragment_mass_summary.json",
        metrics["fragment_probability_mass"],
    )
    write_json(output / "metrics.json", metrics)
    write_csv(output / "theta_distribution.csv", [metrics["theta_distribution"]])
    write_json(
        output / "duplicate_component_summary.json",
        {
            "duplicate_components": metrics["topic_inventory"]["duplicate_components"],
            "largest_strict_duplicate_component": metrics["topic_inventory"][
                "largest_strict_duplicate_component"
            ],
            "catastrophic_duplicate_component": metrics["topic_inventory"][
                "catastrophic_duplicate_component"
            ],
        },
    )
    result = {
        "method": artifact_method,
        "architecture_method": method,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
        "metrics": metrics,
    }
    write_json(result_path, result)
    save_validation(run, artifact_method, beta, theta_full, metrics)
    return result


def pooled_protocol(method: str, base: dict[str, Any]) -> dict[str, Any]:
    """Overlay one frozen pooled configuration on the locked protocol."""
    result = json.loads(json.dumps(base))
    result.update(read_json(POOLED_PROTOCOLS[method]))
    return result


def train_pooled(
    run: Path,
    *,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    """Train one frozen pooled projected candidate."""
    output = run / "models" / method
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    base_protocol = read_json(run / "protocol.json")
    protocol = pooled_protocol(method, base_protocol)
    seed = int(protocol["seed"])
    configure(seed, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    observed = load_csr(run / "data/validation_observed.npz")
    completion = load_csr(run / "data/validation_completion.npz")
    full = load_csr(run / "data/validation_full.npz")
    records = load_heldout_records(run / "data", "validation")
    vocabulary = load_vocabulary(run / "data")
    features = torch.from_numpy(
        np.load(run / "token_features/features.npy").astype(np.float32)
    )
    model, initial_indices = initialize_pooled_candidate(
        features,
        num_topics=int(protocol["model"]["num_topics"]),
        protocol=protocol,
    )
    model = model.to(device)
    config = protocol["simple_candidate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    memory = MemoryTracker(device)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(config["maximum_epochs"])):
        model.train()
        nll_values = []
        information_values = []
        total_values = []
        epoch_started = time.perf_counter()
        for rows in iter_row_batches(
            train.shape[0],
            batch_size=int(config["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            batch = batch_to_device(sparse_batch(train, rows), device)
            optimizer.zero_grad(set_to_none=True)
            candidate = model.infer_batch(batch)
            nll = model.sparse_completion_nll(candidate.theta, candidate.beta, batch)
            information = assignment_information_loss(candidate.theta)
            total = nll + float(config["mi_weight"]) * information
            if not torch.isfinite(total):
                raise FloatingPointError(f"{method} produced a non-finite objective")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(f"{method} produced a non-finite gradient")
            optimizer.step()
            nll_values.append(float(nll.detach().cpu()))
            information_values.append(float(information.detach().cpu()))
            total_values.append(float(total.detach().cpu()))
        memory.sample()
        row = {
            "epoch": epoch + 1,
            "nll": float(np.mean(nll_values)),
            "information_regularizer": float(np.mean(information_values)),
            "total": float(np.mean(total_values)),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print(method.upper(), json.dumps(row, sort_keys=True), flush=True)
    fitting_seconds = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.topic_word_distribution().cpu().numpy(),
            name=f"{method} validation beta",
        )
    inference_started = time.perf_counter()
    theta_observed = infer_pooled_theta(
        model,
        observed,
        batch_size=int(config["batch_size"]),
        device=device,
    )
    observed_seconds = time.perf_counter() - inference_started
    inference_started = time.perf_counter()
    theta_full = infer_pooled_theta(
        model,
        full,
        batch_size=int(config["batch_size"]),
        device=device,
    )
    full_seconds = time.perf_counter() - inference_started
    metrics, top_words = _base_evaluation(
        theta_observed=theta_observed,
        theta_full=theta_full,
        beta=beta,
        completion=completion,
        records=records,
        vocabulary=vocabulary,
    )
    metrics["theta_entropy"] = entropy_diagnostics(theta_full)
    metrics["runtime"] = {
        "training_wall_seconds": fitting_seconds,
        "validation_observed_spectra_per_second": observed.shape[0] / observed_seconds,
        "validation_full_spectra_per_second": full.shape[0] / full_seconds,
        "memory": memory.result(),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    model_config = {
        "architecture": "PooledProjectedMS2LDA",
        **config,
        "topics": int(protocol["model"]["num_topics"]),
        "input_dimensions": int(model.input_dimensions),
        "topic_initial_indices": initial_indices.tolist(),
        "device": str(device),
        "seed": seed,
        "decoder_normalization": "independent fragment/loss softmaxes at 0.5 each",
        "reference_protocol": str(POOLED_PROTOCOLS[method].relative_to(REPO_ROOT)),
    }
    write_json(output / "config.json", model_config)
    write_csv(output / "top_words.csv", top_words)
    result = {
        "method": method,
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "config": model_config,
        "metrics": metrics,
    }
    write_json(result_path, result)
    save_validation(run, method, beta, theta_full, metrics)
    return result


def real_batch_smoke(run: Path, *, device: torch.device) -> dict[str, Any]:
    """Exercise one real full-K/V optimizer step for each primary architecture."""
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    threads = int(protocol["cpu_threads"])
    topics = int(protocol["model"]["num_topics"])
    train = load_csr(run / "data/train.npz")
    results: dict[str, Any] = {
        "device": str(device),
        "topics": topics,
        "vocabulary_size": train.shape[1],
        "finite": True,
    }

    configure(seed + 7001, threads)
    etm_rows = np.arange(min(256, train.shape[0]), dtype=np.int64)
    etm = FixedETM(
        sgns_only(run / "token_features/features.npy"), topics, hidden=800
    ).to(device)
    etm_optimizer = torch.optim.Adam(etm.parameters(), lr=0.005, weight_decay=1.2e-6)
    started = time.perf_counter()
    theta, kl = etm.theta(dense_normalized(train, etm_rows, device), sample=True)
    beta = etm.beta()
    reconstruction = sparse_reconstruction(theta, beta, train, etm_rows, device)
    etm_objective = reconstruction + kl.mean()
    etm_optimizer.zero_grad(set_to_none=True)
    etm_objective.backward()
    etm_optimizer.step()
    results["etm"] = {
        "batch_size": len(etm_rows),
        "step_seconds": time.perf_counter() - started,
        "objective": float(etm_objective.detach().cpu()),
        "finite": bool(torch.isfinite(etm_objective).detach().cpu()),
    }
    del beta, etm, etm_optimizer, theta
    if device.type == "mps":
        torch.mps.empty_cache()

    pooled = pooled_protocol("pooled_likelihood", protocol)
    configure(seed, threads)
    features = torch.from_numpy(
        np.load(run / "token_features/features.npy").astype(np.float32)
    )
    model, _ = initialize_pooled_candidate(features, num_topics=topics, protocol=pooled)
    model = model.to(device)
    config = pooled["simple_candidate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    pooled_rows = np.arange(
        min(int(config["batch_size"]), train.shape[0]), dtype=np.int64
    )
    batch = batch_to_device(sparse_batch(train, pooled_rows), device)
    started = time.perf_counter()
    candidate = model.infer_batch(batch)
    nll = model.sparse_completion_nll(candidate.theta, candidate.beta, batch)
    information = assignment_information_loss(candidate.theta)
    pooled_objective = nll + float(config["mi_weight"]) * information
    optimizer.zero_grad(set_to_none=True)
    pooled_objective.backward()
    optimizer.step()
    results["pooled_likelihood"] = {
        "batch_size": len(pooled_rows),
        "step_seconds": time.perf_counter() - started,
        "objective": float(pooled_objective.detach().cpu()),
        "finite": bool(torch.isfinite(pooled_objective).detach().cpu()),
    }
    results["finite"] = bool(
        results["etm"]["finite"] and results["pooled_likelihood"]["finite"]
    )
    write_json(run / "real_batch_smoke.json", results)
    return results


class ProbeECR(ECR):
    """ECR objective instrumented with iteration/residual evidence."""

    def __init__(self, *, max_iter: int) -> None:
        super().__init__(weight=100.0, alpha=20.0, max_iter=max_iter)
        self.iterations_run = 0
        self.final_residual: float | None = None

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        a = (
            torch.ones((cost.shape[0], 1), dtype=cost.dtype, device=cost.device)
            / cost.shape[0]
        )
        b = (
            torch.ones((cost.shape[1], 1), dtype=cost.dtype, device=cost.device)
            / cost.shape[1]
        )
        u = torch.ones_like(a) / a.shape[0]
        kernel = torch.exp(-cost * self.alpha)
        for iteration in range(self.max_iter):
            v = b / (kernel.T @ u + 1e-16)
            u = a / (kernel @ v + 1e-16)
            self.iterations_run = iteration + 1
            if iteration % 50 == 0:
                residual = torch.max(
                    torch.sum(torch.abs(v * (kernel.T @ u) - b), dim=0)
                )
                self.final_residual = float(residual.detach().cpu())
                if not math.isfinite(self.final_residual):
                    raise FloatingPointError(
                        "ECRTM Sinkhorn solver produced a non-finite residual"
                    )
                if self.final_residual <= 0.005:
                    break
        transport = u * (kernel * v.T)
        return self.weight * torch.sum(transport * cost)


@contextlib.contextmanager
def wall_alarm(seconds: float):
    """Interrupt an overlong feasibility probe while preserving a failure log."""
    if seconds <= 0:
        yield
        return

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"probe exceeded {seconds:.1f} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def ecrtm_probe(
    run: Path,
    *,
    device: torch.device,
    max_iter: int,
    batch_size: int,
    wall_cap_seconds: float,
) -> dict[str, Any]:
    """Probe full K/V ECRTM forward/backward on one real training batch."""
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"]) + 8001
    configure(seed, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    topics = int(protocol["model"]["num_topics"])
    model = TopMostECRTM(sgns_only(run / "token_features/features.npy"), topics)
    model.ecr = ProbeECR(max_iter=int(max_iter))
    model = model.to(device)
    rows = np.arange(min(int(batch_size), train.shape[0]), dtype=np.int64)
    bows = torch.from_numpy(train[rows].toarray().astype(np.float32, copy=False)).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    memory = MemoryTracker(device)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "started",
        "solver": (
            "canonical stopping rule" if max_iter >= 1000 else "bounded approximation"
        ),
        "max_iterations": int(max_iter),
        "topics": topics,
        "vocabulary_size": train.shape[1],
        "batch_size": len(rows),
        "device": str(device),
        "wall_cap_seconds": float(wall_cap_seconds),
    }
    try:
        with wall_alarm(float(wall_cap_seconds)):
            model.train()
            theta, kl = model.theta(bows, sample=True)
            beta = model.beta()
            reconstruction = nnf.softmax(model.decoder_bn(theta @ beta), dim=-1)
            reconstruction_loss = (
                -(bows * torch.log(reconstruction.clamp_min(EPS))).sum(dim=1).mean()
            )
            ecr_loss = model.ecr_loss()
            objective = reconstruction_loss + kl.mean() + ecr_loss
            residual = model.ecr.final_residual
            if residual is None or not math.isfinite(residual):
                raise FloatingPointError(
                    "ECRTM Sinkhorn solver produced no finite residual"
                )
            if int(max_iter) >= 1000 and residual > 0.005:
                raise FloatingPointError(
                    "canonical ECRTM Sinkhorn solver did not reach residual 0.005"
                )
            if not torch.isfinite(objective):
                raise FloatingPointError("ECRTM probe produced a non-finite loss")
            forward_seconds = time.perf_counter() - started
            optimizer.zero_grad(set_to_none=True)
            backward_started = time.perf_counter()
            objective.backward()
            optimizer.step()
            backward_seconds = time.perf_counter() - backward_started
            memory.sample()
        result.update(
            {
                "status": "complete",
                "forward_seconds": forward_seconds,
                "backward_and_step_seconds": backward_seconds,
                "total_seconds": time.perf_counter() - started,
                "iterations_run": model.ecr.iterations_run,
                "final_checked_residual": model.ecr.final_residual,
                "topic_model_loss": float(
                    (reconstruction_loss + kl.mean()).detach().cpu()
                ),
                "ecr_loss": float(ecr_loss.detach().cpu()),
                "finite": bool(torch.isfinite(objective).detach().cpu()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "failed_or_timed_out",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
                "iterations_run": model.ecr.iterations_run,
                "final_checked_residual": model.ecr.final_residual,
            }
        )
    result["memory"] = memory.result()
    output = run / "ecrtm_feasibility"
    output.mkdir(parents=True, exist_ok=True)
    label = "canonical" if max_iter >= 1000 else f"bounded_{max_iter}"
    write_json(output / f"{label}.json", result)
    return result


def train_ecrtm_canonical(  # noqa: PLR0915
    run: Path,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    max_iter: int,
) -> dict[str, Any]:
    """Train maintained ECRTM with the convergent canonical Sinkhorn rule."""
    if device.type != "cpu":
        raise ValueError("the validated canonical ECRTM path currently requires CPU")
    method = "ecrtm_canonical"
    calibrated_method = "ecrtm_canonical_tau030"
    output = run / "models" / method
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    if int(max_iter) < 1000:
        raise ValueError("canonical ECRTM requires max_iter >= 1000")
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 8001, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    observed = load_csr(run / "data/validation_observed.npz")
    completion = load_csr(run / "data/validation_completion.npz")
    full = load_csr(run / "data/validation_full.npz")
    records = load_heldout_records(run / "data", "validation")
    vocabulary = load_vocabulary(run / "data")
    topics = int(protocol["model"]["num_topics"])
    model = TopMostECRTM(sgns_only(run / "token_features/features.npy"), topics)
    model.ecr = ProbeECR(max_iter=int(max_iter))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rng = np.random.default_rng(seed + 8017)
    memory = MemoryTracker(device)
    history: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    training_contract = {
        "topics": topics,
        "batch_size": int(batch_size),
        "maximum_sinkhorn_iterations": int(max_iter),
        "seed": seed + 8001,
        "optimizer": "Adam",
        "learning_rate": 0.002,
        "ecr_weight": 100.0,
        "sinkhorn_alpha": 20.0,
        "sinkhorn_residual_tolerance": 0.005,
        "sinkhorn_check_interval": 50,
        "train_matrix_sha256": file_sha256(run / "data/train.npz"),
        "token_features_sha256": file_sha256(run / "token_features/features.npy"),
        "protocol_sha256": file_sha256(run / "protocol.json"),
    }
    start_epoch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["training_contract"] != training_contract:
            raise ValueError("canonical ECRTM checkpoint contract does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"])
    for epoch in range(start_epoch, int(epochs)):
        model.train()
        order = rng.permutation(train.shape[0])
        topic_total = 0.0
        ecr_total = 0.0
        iteration_values = []
        residual_values = []
        batches = 0
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            bows = torch.from_numpy(
                train[rows].toarray().astype(np.float32, copy=False)
            ).to(device)
            theta, kl = model.theta(bows, sample=True)
            beta_internal = model.beta()
            reconstruction = nnf.softmax(
                model.decoder_bn(theta @ beta_internal), dim=-1
            )
            reconstruction_loss = (
                -(bows * torch.log(reconstruction.clamp_min(EPS))).sum(dim=1).mean()
            )
            topic_loss = reconstruction_loss + kl.mean()
            ecr_loss = model.ecr_loss()
            residual = model.ecr.final_residual
            if residual is None or not math.isfinite(residual) or residual > 0.005:
                raise FloatingPointError(
                    "canonical ECRTM Sinkhorn solver did not reach residual 0.005"
                )
            objective = topic_loss + ecr_loss
            if not torch.isfinite(objective):
                raise FloatingPointError("canonical ECRTM produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            topic_total += float(topic_loss.detach().cpu())
            ecr_total += float(ecr_loss.detach().cpu())
            iteration_values.append(int(model.ecr.iterations_run))
            if model.ecr.final_residual is not None:
                residual_values.append(float(model.ecr.final_residual))
            batches += 1
        memory.sample()
        row = {
            "epoch": epoch + 1,
            "topic_model_loss": topic_total / batches,
            "ecr_loss": ecr_total / batches,
            "mean_sinkhorn_iterations": float(np.mean(iteration_values)),
            "maximum_sinkhorn_iterations": int(max(iteration_values)),
            "mean_final_checked_residual": float(np.mean(residual_values)),
            "maximum_final_checked_residual": float(max(residual_values)),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "history": history,
                "training_contract": training_contract,
            },
        )
        print("ECRTM_CANONICAL_EPOCH", json.dumps(row, sort_keys=True), flush=True)
    fitting_seconds = float(sum(float(row["seconds"]) for row in history))
    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.beta().cpu().numpy(),
            name=f"{method} validation beta",
        )
    inference_started = time.perf_counter()
    theta_observed = infer_ecrtm(model, observed, int(batch_size))
    observed_seconds = time.perf_counter() - inference_started
    inference_started = time.perf_counter()
    theta_full = infer_ecrtm(model, full, int(batch_size))
    full_seconds = time.perf_counter() - inference_started
    word_metrics, top_words = topic_word_diagnostics(beta, vocabulary)
    metrics = {
        "document_completion": ecrtm_completion(
            model, theta_observed, completion, records, int(batch_size)
        ),
        "topic_inventory": mixture_diagnostics(theta_full, beta),
        **word_metrics,
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full))
        ),
        "runtime": {
            "training_wall_seconds": fitting_seconds,
            "validation_observed_spectra_per_second": observed.shape[0]
            / observed_seconds,
            "validation_full_spectra_per_second": full.shape[0] / full_seconds,
            "memory": memory.result(),
        },
        "sinkhorn": {
            "solver": "canonical convergence check",
            "maximum_iterations": int(max_iter),
            "check_interval": 50,
            "residual_tolerance": 0.005,
            "final_epoch_mean_iterations": history[-1]["mean_sinkhorn_iterations"],
            "final_epoch_maximum_residual": history[-1][
                "maximum_final_checked_residual"
            ],
        },
    }
    config = {
        "architecture": "maintained TopMost-style ECRTM",
        "topics": topics,
        "embedding_dimensions": 48,
        "encoder_hidden_dimensions": 200,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.002,
        "device": str(device),
        "seed": seed + 8001,
        "ecr_weight": 100.0,
        "sinkhorn_alpha": 20.0,
        "sinkhorn_maximum_iterations": int(max_iter),
        "sinkhorn_residual_tolerance": 0.005,
        "sinkhorn_check_interval": 50,
        "numerical_approximation": False,
        "resumable_epoch_checkpoint": str(checkpoint_path),
        "resumed_from_epoch": start_epoch,
    }
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    write_json(output / "config.json", config)
    write_csv(output / "top_words.csv", top_words)
    write_json(
        output / "fragment_mass_summary.json",
        metrics["fragment_probability_mass"],
    )
    result = {
        "method": method,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
        "metrics": metrics,
    }
    write_json(result_path, result)
    save_validation(run, method, beta, theta_full, metrics)
    atomic_save_numpy(output / "validation_observed_theta.npy", theta_observed)

    calibrated_observed = retemperature_theta(
        theta_observed, source_temperature=1.0, target_temperature=0.30
    )
    calibrated_full = retemperature_theta(
        theta_full, source_temperature=1.0, target_temperature=0.30
    )
    calibrated_metrics = {
        "document_completion": ecrtm_completion(
            model, calibrated_observed, completion, records, int(batch_size)
        ),
        "topic_inventory": mixture_diagnostics(calibrated_full, beta),
        **word_metrics,
        "finite_stable": bool(
            np.all(np.isfinite(calibrated_observed))
            and np.all(np.isfinite(calibrated_full))
        ),
        "post_hoc_inference_temperature": 0.30,
        "beta_unchanged": True,
        "source_method": method,
    }
    calibrated_output = run / "models" / calibrated_method
    calibrated_output.mkdir(parents=True, exist_ok=True)
    calibrated_result = {
        "method": calibrated_method,
        "parameters": result["parameters"],
        "config": {
            "source_method": method,
            "post_hoc_inference_temperature": 0.30,
            "trained_separately": False,
        },
        "metrics": calibrated_metrics,
    }
    write_json(calibrated_output / "config.json", calibrated_result["config"])
    write_json(calibrated_output / "result.json", calibrated_result)
    write_csv(calibrated_output / "top_words.csv", top_words)
    save_validation(run, calibrated_method, beta, calibrated_full, calibrated_metrics)
    atomic_save_numpy(
        calibrated_output / "validation_observed_theta.npy", calibrated_observed
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--run", required=True, type=Path)
    preparation.add_argument("--data-root", required=True, type=Path)
    train = commands.add_parser("train")
    train.add_argument("--run", required=True, type=Path)
    train.add_argument("--method", required=True, choices=METHODS)
    train.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    train.add_argument("--etm-epochs", type=int, default=120)
    train.add_argument("--etm-batch-size", type=int, default=256)
    train.add_argument("--gate-temperature", type=float, default=1.0)
    train.add_argument("--gate-gamma", type=float, default=1.0)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--run", required=True, type=Path)
    smoke.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    score = commands.add_parser("chemical")
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    score.add_argument("--method", required=True)
    probe = commands.add_parser("ecrtm-probe")
    probe.add_argument("--run", required=True, type=Path)
    probe.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    probe.add_argument("--max-iter", type=int, required=True)
    probe.add_argument("--batch-size", type=int, default=8)
    probe.add_argument("--wall-cap-seconds", type=float, default=900.0)
    ecrtm_train = commands.add_parser("train-ecrtm-canonical")
    ecrtm_train.add_argument("--run", required=True, type=Path)
    ecrtm_train.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    ecrtm_train.add_argument("--epochs", type=int, default=40)
    ecrtm_train.add_argument("--batch-size", type=int, default=200)
    ecrtm_train.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    if args.command == "prepare":
        result = prepare(run, args.data_root.expanduser().resolve())
    elif args.command == "train":
        device = resolve_device(args.device)
        if args.method in ETM_METHODS:
            result = train_etm(
                run,
                device=device,
                epochs=args.etm_epochs,
                batch_size=args.etm_batch_size,
                method=args.method,
                gate_temperature=args.gate_temperature,
                gate_gamma=args.gate_gamma,
            )
        else:
            result = train_pooled(run, method=args.method, device=device)
    elif args.command == "smoke":
        result = real_batch_smoke(run, device=resolve_device(args.device))
    elif args.command == "chemical":
        result = chemical(run, args.data_root.expanduser().resolve(), args.method)
    elif args.command == "train-ecrtm-canonical":
        result = train_ecrtm_canonical(
            run,
            device=resolve_device(args.device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
        )
    else:
        result = ecrtm_probe(
            run,
            device=resolve_device(args.device),
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            wall_cap_seconds=args.wall_cap_seconds,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
