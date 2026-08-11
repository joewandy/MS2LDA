# ruff: noqa: PLR0913, PLR0915
"""Single-touch held-out evaluation after the validation-only model decision."""

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
    sparse_npmi,
    top_word_diversity,
)
from .training import infer_theta, load_attempt_model
from .utils import (
    atomic_save_numpy,
    file_sha256,
    peak_rss_bytes,
    read_json,
    write_json,
)


def load_reference_summary(reference_run: str | Path) -> dict[str, Any]:
    """Extract the corrected Tomotopy comparator without loading its model."""
    root = Path(reference_run).expanduser().resolve()
    core = read_json(root / "core/seed_42/tomotopy/complete.json")
    chemical = read_json(
        root / "chemical_inference/seed_42/tomotopy/complete.json",
    )
    mag = read_json(root / "mag/seed_42/tomotopy/complete.json")
    association = {row["association_mode"]: row for row in mag["association_results"]}
    dominant = association["dominant_topic"]
    threshold = association["probability_ge_frozen_threshold"]
    metrics = core["metrics"]
    mixture = chemical["mixture_diagnostics"]["standard"]
    return {
        "schema_version": "neural-assignment-ms2lda/reference-summary-v1",
        "method": "tomotopy",
        "topic_count": int(core["topic_count"]),
        "active_topics": metrics["active_topics"],
        "top_word_diversity": float(metrics["top_word_diversity"]),
        "word_cooccurrence_npmi": metrics["word_cooccurrence_npmi"],
        "document_completion": metrics["document_completion"],
        "full_spectrum_mixture": mixture,
        "dominant_topic_chemistry": {
            "mean_sos": float(dominant["mean_sos"]),
            "sos_evaluable_coverage": float(dominant["sos_evaluable_coverage"]),
        },
        "high_confidence_chemistry": {
            "association_mode": "probability_ge_frozen_threshold",
            "membership_threshold": float(threshold["membership_threshold"]),
            "mean_sos": float(threshold["mean_sos"]),
            "sos_evaluable_coverage": float(threshold["sos_evaluable_coverage"]),
        },
        "cached_latency": core["cached_latency"],
        "source_files": {
            "core": str(root / "core/seed_42/tomotopy/complete.json"),
            "chemical_inference": str(
                root / "chemical_inference/seed_42/tomotopy/complete.json",
            ),
            "mag": str(root / "mag/seed_42/tomotopy/complete.json"),
        },
    }


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
        "routing_passes_per_representation": 1,
        "local_vb_steps": 0,
        "iterative_inference_steps": 0,
        "cached_projected_token_table": True,
    }


def nonchemical_test_gates(
    evaluation: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the absolute K=1000 non-chemical viability gates once."""
    metrics = evaluation["metrics"]
    config = protocol["k1000_gates"]
    active = int(metrics["active_topics"]["corpus_active_topics"])
    diversity = float(metrics["top_word_diversity"])
    effective = float(
        metrics["full_spectrum_mixture"]["effective_topic_count_median"],
    )
    nll = float(metrics["test_document_completion"]["nll_per_token"])
    checks = {
        "fully_neural_audit": {
            "pass": bool(evaluation["fully_neural_audit"]["fully_neural"]),
            "actual": bool(evaluation["fully_neural_audit"]["fully_neural"]),
            "required": True,
        },
        "stable": {
            "pass": bool(evaluation["stable"]),
            "actual": bool(evaluation["stable"]),
            "required": True,
        },
        "active_topics": {
            "pass": active >= int(config["minimum_active_topics"]),
            "actual": active,
            "minimum": int(config["minimum_active_topics"]),
        },
        "top_word_diversity": {
            "pass": diversity >= float(config["minimum_top_word_diversity"]),
            "actual": diversity,
            "minimum": float(config["minimum_top_word_diversity"]),
        },
        "median_effective_topics": {
            "pass": float(config["minimum_median_effective_topics"])
            <= effective
            <= float(config["maximum_median_effective_topics"]),
            "actual": effective,
            "minimum": float(config["minimum_median_effective_topics"]),
            "maximum": float(config["maximum_median_effective_topics"]),
        },
        "heldout_nll": {
            "pass": nll <= float(config["maximum_validation_nll"]),
            "actual": nll,
            "maximum": float(config["maximum_validation_nll"]),
        },
    }
    return {
        "checks": checks,
        "pass": all(row["pass"] for row in checks.values()),
        "failed": [name for name, row in checks.items() if not row["pass"]],
    }


def evaluate_selected(
    run_dir: str | Path,
    *,
    attempt: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Touch test data only for the validation-selected K=1000 attempt."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation" / attempt
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"evaluation artifact changed: {name}"
                raise ValueError(msg)
        return result
    selection_path = directory / "validation_selection.json"
    selection = read_json(selection_path)
    if not selection.get("decision_final_before_test"):
        msg = "validation selection was not frozen before test access"
        raise RuntimeError(msg)
    if selection["selected_attempt"] != attempt:
        msg = "requested test attempt differs from validation selection"
        raise ValueError(msg)

    lock = read_json(directory / "neural.lock.json")
    counts = Path(lock["source_run"]) / "shared/counts"
    # This marker is written before the first test matrix is opened.
    write_json(
        directory / "test_access.json",
        {
            "schema_version": "neural-assignment-ms2lda/test-access-v1",
            "selected_attempt": attempt,
            "validation_selection_sha256": file_sha256(selection_path),
            "selection_final_before_test": True,
            "test_access_count": 1,
        },
    )
    train = load_csr(counts / "train.npz")
    validation_observed = load_csr(counts / "validation_observed.npz")
    validation_completion = load_csr(counts / "validation_completion.npz")
    test_observed = load_csr(counts / "test_observed.npz")
    test_completion = load_csr(counts / "test_completion.npz")
    test_full = load_csr(counts / "test_full.npz")
    validation_records = load_heldout_records(counts, "validation")
    test_records = load_heldout_records(counts, "test")
    model, checkpoint = load_attempt_model(
        directory,
        stage="k1000",
        attempt=attempt,
        initialization_label="k1000",
        protocol=protocol,
    )
    batch_size = int(
        read_json(directory / "batch_benchmark.json")["selected_batch_size"],
    )
    temperature = float(checkpoint["routing_temperature"])
    top_k = int(checkpoint["top_k"])
    torch.set_num_threads(int(protocol["evaluation_cpu_threads"]))
    started = time.perf_counter()
    beta = model.topic_word_distribution().cpu().detach().numpy().astype(np.float32)
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
        test_theta,
        beta,
        test_completion,
        test_records,
    )
    stable = bool(
        np.all(np.isfinite(beta))
        and np.all(np.isfinite(validation_theta))
        and np.all(np.isfinite(test_theta))
        and np.all(np.isfinite(test_full_theta))
        and np.isfinite(test_completion_metrics["nll_per_token"]),
    )
    metrics = {
        "validation_document_completion": validation_completion_metrics,
        "test_document_completion": test_completion_metrics,
        "active_topics": active_topic_metrics(
            test_theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"],
            ),
            corpus_threshold=1.0 / model.num_topics,
        ),
        "full_spectrum_mixture": effective_topic_summary(test_full_theta),
        "top_word_diversity": top_word_diversity(
            beta,
            top_n=int(protocol["evaluation"]["topic_top_n"]),
        ),
        "word_cooccurrence_npmi": sparse_npmi(
            beta,
            train,
            top_n=int(protocol["evaluation"]["topic_top_n"]),
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
    training = read_json(
        directory / "stages/k1000/attempts" / attempt / "complete.json",
    )
    audit = read_json(directory / "candidate_audit.json")
    result = {
        "schema_version": "neural-assignment-ms2lda/evaluation-complete-v1",
        "attempt": attempt,
        "stable": bool(stable and training["stable"]),
        "fully_neural_audit": audit,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "evaluation_cpu_threads": int(protocol["evaluation_cpu_threads"]),
        "routing_passes_per_representation": 1,
        "local_vb_steps": 0,
        "iterative_test_inference_steps": 0,
        "test_touched_after_validation_selection": True,
        "validation_selection_sha256": file_sha256(selection_path),
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    gates = nonchemical_test_gates(result, protocol)
    write_json(output / "nonchemical_gates.json", gates)
    return result
