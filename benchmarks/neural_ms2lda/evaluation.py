"""Single-touch evaluation for the validation-selected neural model."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_csr, load_heldout_records
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
)
from .model import infer_theta
from .training import load_selected_model
from .utils import (
    atomic_save_numpy,
    file_sha256,
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
    )
    full_theta = infer_theta(
        model,
        full,
        batch_size=batch_size,
        temperature=temperature,
    )
    document_completion, _ = completion_metrics(theta, beta, completion, records)
    maximum = full_theta.max(axis=1)
    confident = maximum >= 0.5
    confident_topics = np.unique(full_theta[confident].argmax(axis=1))
    arrays = {
        "beta.npy": beta,
        "validation_full_theta.npy": full_theta,
    }
    stable = all(np.isfinite(values).all() for values in (beta, theta, full_theta))
    stable = stable and np.isfinite(document_completion["nll_per_token"])
    if not stable:
        raise FloatingPointError("model produced non-finite validation evidence")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
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
) -> dict[str, Any]:
    """Measure resident-model throughput after one unmeasured warm-up pass."""
    documents = min(int(subset_size), matrix.shape[0])
    subset = matrix[:documents].tocsr()
    infer_theta(
        model,
        subset,
        batch_size=documents,
        temperature=temperature,
    )
    elapsed = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        infer_theta(
            model,
            subset,
            batch_size=documents,
            temperature=temperature,
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
            "selected_model_sha256": file_sha256(selected_path),
            "selection_final_before_test": True,
            "selection_rule": "fixed_final_epoch",
            "selected_epoch": final_epoch,
        },
    )
    test_observed = load_csr(data / "test_observed.npz")
    test_completion = load_csr(data / "test_completion.npz")
    test_full = load_csr(data / "test_full.npz")
    test_records = load_heldout_records(data, "test")
    model, checkpoint = load_selected_model(directory, protocol)
    temperature = float(checkpoint["routing_temperature"])
    batch_size = int(protocol["optimization"]["batch_size"])
    started = time.perf_counter()
    beta = model.topic_word_distribution().detach().cpu().numpy().astype(np.float32)
    test_theta = infer_theta(
        model,
        test_observed,
        batch_size=batch_size,
        temperature=temperature,
    )
    test_full_theta = infer_theta(
        model,
        test_full,
        batch_size=batch_size,
        temperature=temperature,
    )
    test_completion_metrics, _ = completion_metrics(
        test_theta, beta, test_completion, test_records
    )
    metrics = {
        "test_document_completion": test_completion_metrics,
        "active_topics": active_topic_metrics(
            test_theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"]
            ),
            corpus_threshold=1.0 / beta.shape[0],
        ),
        "full_spectrum_mixture": effective_topic_summary(test_full_theta),
        "warm_in_memory_batch_inference": _latency(
            model,
            test_observed,
            subset_size=int(protocol["evaluation"]["latency_subset_size"]),
            repeats=int(protocol["evaluation"]["latency_repeats"]),
            temperature=temperature,
        ),
    }
    stable = all(
        np.all(np.isfinite(values)) for values in (beta, test_theta, test_full_theta)
    ) and np.isfinite(test_completion_metrics["nll_per_token"])
    if not stable:
        raise FloatingPointError("selected neural model produced non-finite evaluation")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "test_full_theta.npy": test_full_theta,
    }
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "method": "neural",
        "topic_count": int(beta.shape[0]),
        "selected_epoch": int(checkpoint["epoch"]),
        "cpu_threads": int(protocol["cpu_threads"]),
        "stable": True,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "output_sha256": {
            name: file_sha256(output / name)
            for name in (*arrays, test_access_path.name)
        },
    }
    write_json(complete_path, result)
    return result
