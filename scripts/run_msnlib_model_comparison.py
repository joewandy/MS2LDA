"""Validation-only MSnLib ETM and pooled-model comparison campaign.

The runner reuses the locked repository preparation, completion, and chemical
scoring machinery.  It never loads candidate test matrices or test records.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
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
from benchmarks.neural_ms2lda.chemical import run_chemical_scoring
from benchmarks.neural_ms2lda.data import (
    iter_row_batches,
    load_csr,
    load_heldout_records,
    load_vocabulary,
    prepare_data,
    sparse_batch,
    train_token_features,
)
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.pooled import (
    assignment_information_loss,
    batch_to_device,
    infer_pooled_theta,
    initialize_pooled_candidate,
)
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
    sgns_only,
)

EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]
POOLED_ROOT = REPO_ROOT / "research/etm_ecrtm_msnlib/pooled_projected"
METHODS = ("etm", "pooled_likelihood", "pooled_mi005")
POOLED_PROTOCOLS = {
    "pooled_likelihood": POOLED_ROOT / "protocol_minimum.json",
    "pooled_mi005": POOLED_ROOT / "protocol_mi005.json",
}


class MemoryTracker:
    """Track practical process and MPS allocation high-water marks."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_process_bytes = 0
        self.peak_mps_allocated_bytes = 0
        self.peak_mps_driver_bytes = 0
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
        }


def configure(seed: int, threads: int) -> None:
    """Apply the locked seed/thread contract."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def resolve_device(name: str) -> torch.device:
    """Resolve an operational device without changing model equations."""
    selected = name
    if selected == "auto":
        selected = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if selected not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
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
    started = time.perf_counter()
    values = []
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start, min(start + batch_size, matrix.shape[0]), dtype=np.int64
        )
        theta, _ = model.theta(dense_normalized(matrix, rows, device), sample=False)
        values.append(theta.cpu().numpy().astype(np.float32))
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


def entropy_diagnostics(theta: np.ndarray) -> dict[str, float]:
    """Return conditional entropy, marginal entropy, and their MI gap."""
    values = theta.astype(np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), EPS)
    conditional = float(
        np.mean(-np.sum(values * np.log(np.clip(values, EPS, None)), axis=1))
    )
    marginal = values.mean(axis=0)
    marginal_entropy = float(-np.sum(marginal * np.log(np.clip(marginal, EPS, None))))
    return {
        "mean_conditional_theta_entropy": conditional,
        "marginal_theta_entropy": marginal_entropy,
        "mutual_information": marginal_entropy - conditional,
    }


def topic_word_diagnostics(
    beta: np.ndarray,
    vocabulary: list[str],
    *,
    top_n: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize top-word uniqueness and fragment probability mass."""
    count = min(int(top_n), beta.shape[1])
    candidates = np.argpartition(-beta, count - 1, axis=1)[:, :count]
    scores = np.take_along_axis(beta, candidates, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    indices = np.take_along_axis(candidates, order, axis=1)
    rows = []
    for topic_id, topic_indices in enumerate(indices):
        rows.append(
            {
                "topic_id": topic_id,
                "top_words": " ".join(vocabulary[index] for index in topic_indices),
                "top_probabilities": " ".join(
                    f"{float(beta[topic_id, index]):.10g}" for index in topic_indices
                ),
            }
        )
    unique = len(set(indices.ravel().tolist())) / indices.size
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary], dtype=bool
    )
    fragment_mass = beta[:, fragment_mask].sum(axis=1).astype(np.float64)
    percentiles = {
        str(percentile): float(np.percentile(fragment_mass, percentile))
        for percentile in (1, 5, 25, 50, 75, 95, 99)
    }
    return (
        {
            "top_word_count": count,
            "top_word_uniqueness": float(unique),
            "fragment_probability_mass": {
                "minimum": float(fragment_mass.min()),
                "percentiles": percentiles,
                "median": float(np.median(fragment_mass)),
                "maximum": float(fragment_mass.max()),
                "extreme_definition": "fragment mass <0.1 or >0.9",
                "fraction_extreme_skew": float(
                    np.mean((fragment_mass < 0.1) | (fragment_mass > 0.9))
                ),
            },
        },
        rows,
    )


def save_validation(
    run: Path,
    method: str,
    beta: np.ndarray,
    theta: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Save only validation candidate arrays under the locked artifact layout."""
    output = run / "validation_evaluation" / method
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / "beta.npy", beta.astype(np.float32, copy=False))
    atomic_save_numpy(
        output / "validation_full_theta.npy",
        theta.astype(np.float32, copy=False),
    )
    write_json(
        output / "complete.json",
        {"method": method, "split": "validation", "metrics": metrics},
    )


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
    word_metrics, top_words = topic_word_diagnostics(beta, vocabulary)
    metrics = {
        "document_completion": completion_metrics(
            theta_observed, beta, completion, records
        ),
        "topic_inventory": mixture_diagnostics(theta_full, beta),
        **word_metrics,
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
) -> dict[str, Any]:
    """Train the canonical fixed-SGNS ETM on the locked training matrix."""
    method = "etm"
    output = run / "models" / method
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 7001, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    observed = load_csr(run / "data/validation_observed.npz")
    completion = load_csr(run / "data/validation_completion.npz")
    full = load_csr(run / "data/validation_full.npz")
    records = load_heldout_records(run / "data", "validation")
    vocabulary = load_vocabulary(run / "data")
    model = FixedETM(
        sgns_only(run / "token_features/features.npy"),
        int(protocol["model"]["num_topics"]),
        hidden=800,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1.2e-6)
    rng = np.random.default_rng(seed + 7019)
    memory = MemoryTracker(device)
    history: list[dict[str, Any]] = []
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
        row = {
            "epoch": epoch + 1,
            "reconstruction": reconstruction_total / batches,
            "kl": kl_total / batches,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print("ETM_EPOCH", json.dumps(row, sort_keys=True), flush=True)
    fitting_seconds = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        beta = model.beta().cpu().numpy().astype(np.float32)
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
        "architecture": "canonical fixed-pretrained-SGNS ETM",
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
        "decoder_normalization": "global topic-word softmax",
    }
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
        beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
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


def chemical(run: Path, data_root: Path, method: str) -> dict[str, Any]:
    """Run exact shared MAG/SOS scoring on validation only."""
    protocol = read_json(run / "protocol.json")
    result = run_chemical_scoring(
        run,
        method=method,
        data_root=data_root,
        protocol=protocol,
        split="validation",
    )
    rows = result["high_confidence_chemistry"].get("topic_scores", [])
    write_csv(run / "models" / method / "chemical_scores.csv", rows)
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
    train.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    train.add_argument("--etm-epochs", type=int, default=120)
    train.add_argument("--etm-batch-size", type=int, default=256)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--run", required=True, type=Path)
    smoke.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    score = commands.add_parser("chemical")
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    score.add_argument("--method", required=True, choices=METHODS + ("ecrtm",))
    probe = commands.add_parser("ecrtm-probe")
    probe.add_argument("--run", required=True, type=Path)
    probe.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    probe.add_argument("--max-iter", type=int, required=True)
    probe.add_argument("--batch-size", type=int, default=8)
    probe.add_argument("--wall-cap-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    if args.command == "prepare":
        result = prepare(run, args.data_root.expanduser().resolve())
    elif args.command == "train":
        device = resolve_device(args.device)
        if args.method == "etm":
            result = train_etm(
                run,
                device=device,
                epochs=args.etm_epochs,
                batch_size=args.etm_batch_size,
            )
        else:
            result = train_pooled(run, method=args.method, device=device)
    elif args.command == "smoke":
        result = real_batch_smoke(run, device=resolve_device(args.device))
    elif args.command == "chemical":
        result = chemical(run, args.data_root.expanduser().resolve(), args.method)
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
