"""Held-out likelihood, mixture diagnostics, and one-pass evaluation."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_trained_model
from .data import load_csr, load_heldout_records
from .model import infer_theta
from .objectives import completion_metrics
from .utils import atomic_save_numpy, read_json, write_json

PROBABILITY_FLOOR = 1e-12


def _normalized_mixtures(theta: np.ndarray) -> np.ndarray:
    """Return row-normalized mixtures with a uniform empty-row fallback."""
    values = np.asarray(theta, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > PROBABILITY_FLOOR,
    )
    values[totals[:, 0] <= PROBABILITY_FLOOR] = 1.0 / values.shape[1]
    return values


def _mixture_safety(
    observed_theta: np.ndarray,
    full_theta: np.ndarray,
) -> tuple[int, float]:
    """Measure corpus activity from observed halves and entropy from full spectra."""
    observed = _normalized_mixtures(observed_theta)
    full = _normalized_mixtures(full_theta)
    active = int((observed.mean(axis=0) >= 1.0 / observed.shape[1]).sum())
    entropy = -np.sum(full * np.log(np.clip(full, PROBABILITY_FLOOR, None)), axis=1)
    return active, float(np.median(np.exp(entropy)))


@torch.inference_mode()
def _warm_inference(
    model: Any,
    matrix: Any,
    *,
    subset_size: int,
    repeats: int,
    temperature: float,
) -> dict[str, Any]:
    """Measure resident-model throughput after one unmeasured warm-up pass."""
    documents = min(int(subset_size), matrix.shape[0])
    subset = matrix[:documents].tocsr()
    infer_theta(model, subset, batch_size=documents, temperature=temperature)
    elapsed = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        infer_theta(model, subset, batch_size=documents, temperature=temperature)
        elapsed.append(time.perf_counter() - started)
    seconds_per_spectrum = [value / documents for value in elapsed]
    median = float(statistics.median(seconds_per_spectrum))
    return {
        "documents": documents,
        "median_spectra_per_second": 1.0 / median,
    }


def evaluate_neural(
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Evaluate the trained model on validation or, after validation, test."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    run = Path(run_dir).expanduser().resolve()
    if (
        split == "test"
        and not (run / "validation_chemical/neural/complete.json").is_file()
    ):
        raise RuntimeError("validation must finish before test data are opened")
    group = "validation_evaluation" if split == "validation" else "evaluation"
    output = run / group / "neural"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        return read_json(complete_path)
    if torch.get_num_threads() != int(protocol["cpu_threads"]):
        raise ValueError("PyTorch thread count differs from the study protocol")

    data = run / "data"
    observed = load_csr(data / f"{split}_observed.npz")
    completion = load_csr(data / f"{split}_completion.npz")
    full = load_csr(data / f"{split}_full.npz")
    records = load_heldout_records(data, split)
    model, _, temperature = load_trained_model(run / "trained_model")
    batch_size = int(protocol["optimization"]["batch_size"])
    beta = model.topic_word_distribution().detach().cpu().numpy().astype(np.float32)
    theta = infer_theta(model, observed, batch_size=batch_size, temperature=temperature)
    full_theta = infer_theta(
        model, full, batch_size=batch_size, temperature=temperature
    )
    completion_summary = completion_metrics(theta, beta, completion, records)
    if not all(np.isfinite(values).all() for values in (beta, theta, full_theta)):
        raise FloatingPointError("neural evaluation produced non-finite values")
    if not np.isfinite(completion_summary["nll_per_token"]):
        raise FloatingPointError("neural completion NLL is non-finite")

    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / f"{split}_full_theta.npy", full_theta)
    if split == "validation":
        atomic_save_numpy(output / "beta.npy", beta)
        metrics = {"validation_document_completion": completion_summary}
    else:
        active, effective_median = _mixture_safety(theta, full_theta)
        metrics = {
            "test_document_completion": completion_summary,
            "active_topics": {"corpus_active_topics": active},
            "full_spectrum_mixture": {"effective_topic_count_median": effective_median},
            "warm_in_memory_batch_inference": _warm_inference(
                model,
                observed,
                subset_size=int(protocol["evaluation"]["latency_subset_size"]),
                repeats=int(protocol["evaluation"]["latency_repeats"]),
                temperature=temperature,
            ),
        }
    result = {
        "method": "neural",
        "split": split,
        "metrics": metrics,
    }
    write_json(complete_path, result)
    return result
