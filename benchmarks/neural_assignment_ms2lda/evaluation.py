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
from .training import load_selected_model
from .utils import (
    atomic_save_numpy,
    file_sha256,
    peak_rss_bytes,
    read_json,
    verify_output_hashes,
    write_json,
)


def _validation_result(  # noqa: PLR0913
    *,
    model: Any,
    data: Path,
    output: Path,
    temperature: float,
    top_k: int,
    batch_size: int,
    source_sha256: str,
) -> dict[str, Any]:
    """Evaluate a model using validation artifacts only."""
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        if result.get("source_sha256") != source_sha256:
            raise ValueError("validation model source changed")
        return result
    observed = load_csr(data / "validation_observed.npz")
    completion = load_csr(data / "validation_completion.npz")
    full = load_csr(data / "validation_full.npz")
    records = load_heldout_records(data, "validation")
    started = time.perf_counter()
    beta = model.topic_word_distribution().detach().cpu().numpy().astype(np.float32)
    theta = infer_theta(
        model,
        observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    full_theta = infer_theta(
        model,
        full,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    document_completion, losses = completion_metrics(theta, beta, completion, records)
    maximum = full_theta.max(axis=1)
    confident = maximum >= 0.5
    confident_topics = np.unique(full_theta[confident].argmax(axis=1))
    arrays = {
        "beta.npy": beta,
        "validation_observed_theta.npy": theta,
        "validation_full_theta.npy": full_theta,
        "validation_completion_nll.npy": losses,
    }
    stable = all(np.isfinite(values).all() for values in (beta, theta, full_theta))
    stable = stable and np.isfinite(document_completion["nll_per_token"])
    if not stable:
        raise FloatingPointError("model produced non-finite validation evidence")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "schema_version": "neural-ms2lda/validation-evaluation-v1",
        "method": "neural",
        "split": "validation",
        "topic_count": int(beta.shape[0]),
        "source_sha256": source_sha256,
        "stable": True,
        "metrics": {
            "validation_document_completion": document_completion,
            "high_confidence_spectra": int(confident.sum()),
            "distinct_high_confidence_topics": int(confident_topics.size),
            "full_spectrum_mixture": effective_topic_summary(full_theta),
        },
        "evaluation_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result


def evaluate_neural_validation(
    run_dir: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a trained neural checkpoint without opening the test split."""
    directory = Path(run_dir).expanduser().resolve()
    selected_path = directory / "model/selected.json"
    selected = read_json(selected_path)
    training = read_json(directory / "model/complete.json")
    if selected != training["selected"]:
        raise ValueError("selected model manifest changed")
    if selected.get("selection_rule") != "fixed_final_epoch":
        raise ValueError("validation evaluation requires fixed final epoch selection")
    if int(selected["epoch"]) != int(protocol["optimization"]["maximum_epochs"]):
        raise ValueError("validation evaluation requires the fixed final epoch")
    model, checkpoint = load_selected_model(directory, protocol)
    return _validation_result(
        model=model,
        data=directory / "data",
        output=directory / "validation_evaluation/neural",
        temperature=float(checkpoint["routing_temperature"]),
        top_k=int(checkpoint["top_k"]),
        batch_size=int(protocol["optimization"]["batch_size"]),
        source_sha256=str(selected["checkpoint_sha256"]),
    )


@torch.inference_mode()
def _latency(  # noqa: PLR0913
    model: Any,
    matrix: Any,
    *,
    subset_size: int,
    repeats: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    """Measure resident-model throughput after one unmeasured warm-up pass."""
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
        "cpu_threads": torch.get_num_threads(),
    }


def evaluate_neural(  # noqa: PLR0915
    run_dir: str | Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the selected checkpoint once on the untouched test split."""
    expected_threads = int(protocol["cpu_threads"])
    actual_threads = torch.get_num_threads()
    if actual_threads != expected_threads:
        raise ValueError(
            "neural evaluation thread count differs from the protocol: "
            f"expected {expected_threads}, observed {actual_threads}"
        )
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation/neural"
    complete_path = output / "complete.json"
    selected_path = directory / "model/selected.json"
    selected = read_json(selected_path)
    training_manifest = read_json(directory / "model/complete.json")
    if selected != training_manifest["selected"]:
        raise ValueError("selected model manifest changed")
    final_epoch = int(protocol["optimization"]["maximum_epochs"])
    if selected.get("selection_rule") != "fixed_final_epoch":
        raise ValueError("neural selection rule changed")
    if int(selected["epoch"]) != final_epoch:
        raise ValueError("test evaluation requires the fixed final epoch")
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
            "selection_rule": "fixed_final_epoch",
            "selected_epoch": final_epoch,
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
        "warm_in_memory_batch_inference": _latency(
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
        "method": "neural",
        "topic_count": int(beta.shape[0]),
        "selected_epoch": int(checkpoint["epoch"]),
        "cpu_threads": int(protocol["cpu_threads"]),
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
