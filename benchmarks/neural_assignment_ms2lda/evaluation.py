"""Single-touch evaluation for the validation-selected neural model."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import infer_theta
from .data import load_csr, load_heldout_records
from .inventory import topic_inventory_summary
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
    sparse_npmi,
    top_word_diversity,
)
from .training import load_selected_model, validation_gate_summary
from .utils import (
    atomic_save_numpy,
    file_sha256,
    peak_rss_bytes,
    read_json,
    verify_output_hashes,
    write_json,
)


@torch.inference_mode()
def _latency(
    model: Any,
    matrix: Any,
    *,
    subset_size: int,
    repeats: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    documents = min(int(subset_size), matrix.shape[0])
    subset = matrix[:documents].tocsr()
    infer_theta(
        model,
        subset,
        batch_size=documents,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        infer_theta(
            model,
            subset,
            batch_size=documents,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed.append(time.perf_counter() - started)
    seconds = [value / documents for value in elapsed]
    median = float(statistics.median(seconds))
    return {
        "documents": documents,
        "repeats": int(repeats),
        "median_seconds_per_spectrum": median,
        "median_spectra_per_second": 1.0 / median,
        "p95_seconds_per_spectrum": float(np.percentile(seconds, 95)),
        "routing_passes": 1,
        "iterative_inference_steps": 0,
    }


def evaluate_neural(run_dir: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the selected checkpoint once on the untouched test split."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation/neural"
    complete_path = output / "complete.json"
    selected_path = directory / "model/selected.json"
    selected = read_json(selected_path)
    training_manifest = read_json(directory / "model/complete.json")
    if selected != training_manifest["selected"]:
        raise ValueError("selected model manifest changed")
    gate_summary = validation_gate_summary(selected["validation"], protocol)
    if gate_summary != selected.get("validation_gate_summary"):
        raise ValueError("selected model validation gates changed")
    if gate_summary["all_gates_met"] is not True:
        raise RuntimeError("test evaluation requires every validation gate to pass")
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        return result
    data = directory / "data"
    test_access_path = output / "test_access.json"
    write_json(
        test_access_path,
        {
            "schema_version": "neural-ms2lda/test-access-v1",
            "selected_model_sha256": file_sha256(selected_path),
            "selection_final_before_test": True,
            "validation_gates_met": True,
        },
    )
    train = load_csr(data / "train.npz")
    validation_observed = load_csr(data / "validation_observed.npz")
    validation_completion = load_csr(data / "validation_completion.npz")
    test_observed = load_csr(data / "test_observed.npz")
    test_completion = load_csr(data / "test_completion.npz")
    test_full = load_csr(data / "test_full.npz")
    validation_records = load_heldout_records(data, "validation")
    test_records = load_heldout_records(data, "test")
    model, checkpoint = load_selected_model(directory, protocol)
    temperature = float(checkpoint["routing_temperature"])
    top_k = int(checkpoint["top_k"])
    batch_size = int(protocol["optimization"]["batch_size"])
    started = time.perf_counter()
    beta = model.topic_word_distribution().detach().cpu().numpy().astype(np.float32)
    validation_theta = infer_theta(
        model,
        validation_observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    test_theta, routing = infer_theta(
        model,
        test_observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
        with_diagnostics=True,
    )
    test_full_theta = infer_theta(
        model,
        test_full,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    validation_completion_metrics, validation_losses = completion_metrics(
        validation_theta,
        beta,
        validation_completion,
        validation_records,
    )
    test_completion_metrics, test_losses = completion_metrics(
        test_theta, beta, test_completion, test_records
    )
    usage = test_full_theta.mean(axis=0)
    metrics = {
        "validation_document_completion": validation_completion_metrics,
        "test_document_completion": test_completion_metrics,
        "active_topics": active_topic_metrics(
            test_theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"]
            ),
            corpus_threshold=1.0 / beta.shape[0],
        ),
        "full_spectrum_mixture": effective_topic_summary(test_full_theta),
        "top_word_diversity": top_word_diversity(
            beta, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "word_cooccurrence_npmi": sparse_npmi(
            beta, train, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "topic_inventory": topic_inventory_summary(
            beta, usage, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "routing": routing,
        "cached_latency": _latency(
            model,
            test_observed,
            subset_size=int(protocol["evaluation"]["latency_subset_size"]),
            repeats=int(protocol["evaluation"]["latency_repeats"]),
            temperature=temperature,
            top_k=top_k,
        ),
    }
    stable = all(
        np.all(np.isfinite(values))
        for values in (beta, validation_theta, test_theta, test_full_theta)
    ) and np.isfinite(test_completion_metrics["nll_per_token"])
    if not stable:
        raise FloatingPointError("selected neural model produced non-finite evaluation")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "beta.npy": beta,
        "validation_observed_theta.npy": validation_theta,
        "validation_completion_nll.npy": validation_losses,
        "test_observed_theta.npy": test_theta,
        "test_completion_nll.npy": test_losses,
        "test_full_theta.npy": test_full_theta,
    }
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "schema_version": "neural-ms2lda/neural-evaluation-v1",
        "method": "neural_cooccurrence_margin_hierarchical_k500",
        "topic_count": int(beta.shape[0]),
        "selected_epoch": int(checkpoint["epoch"]),
        "stable": True,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {
            name: file_sha256(output / name)
            for name in (*arrays, test_access_path.name)
        },
    }
    write_json(complete_path, result)
    return result
