"""Held-out likelihood, collapse diagnostics, and one-pass evaluation."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_trained_model
from .data import load_csr, load_heldout_records, load_vocabulary
from .diagnostics import model_selection_diagnostics
from .model import infer_theta
from .objectives import completion_metrics
from .utils import atomic_save_numpy, read_json, write_json


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
    vocabulary = load_vocabulary(data)
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

    diagnostics = model_selection_diagnostics(
        full_theta,
        beta,
        vocabulary,
        protocol["evaluation"],
    )
    inventory = diagnostics["topic_inventory"]

    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / f"{split}_full_theta.npy", full_theta)
    write_json(output / "diagnostics.json", diagnostics)
    if split == "validation":
        atomic_save_numpy(output / "beta.npy", beta)
        metrics = {
            "validation_document_completion": completion_summary,
            **diagnostics,
        }
    else:
        metrics = {
            "test_document_completion": completion_summary,
            **diagnostics,
            # Compatibility aliases retained for the existing report generator.
            "active_topics": {
                "corpus_active_topics": inventory[
                    "active_topics_mean_usage_ge_1_over_k"
                ]
            },
            "full_spectrum_mixture": {
                "effective_topic_count_median": inventory[
                    "median_effective_topics_per_spectrum"
                ]
            },
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
