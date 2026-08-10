"""Held-out neural evaluation and reference-aligned gate calculations."""

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
from .utils import atomic_save_numpy, file_sha256, peak_rss_bytes, read_json, write_json


def load_reference_summary(reference_run: str | Path) -> dict[str, Any]:
    """Extract the corrected Tomotopy comparator from its immutable artifacts."""
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
        "schema_version": "fully-neural-ms2lda/reference-summary-v1",
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
    model,
    matrix,
    *,
    subset_size: int,
    repeats: int,
) -> dict[str, Any]:
    documents = min(int(subset_size), matrix.shape[0])
    subset = matrix[:documents].tocsr()
    projected_tokens = model.projected_tokens()
    infer_theta(
        model,
        subset,
        batch_size=documents,
        projected_tokens=projected_tokens,
    )
    elapsed = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        infer_theta(
            model,
            subset,
            batch_size=documents,
            projected_tokens=projected_tokens,
        )
        elapsed.append(time.perf_counter() - started)
    seconds_per_spectrum = [value / documents for value in elapsed]
    median = float(statistics.median(seconds_per_spectrum))
    return {
        "documents": documents,
        "repeats": int(repeats),
        "median_seconds_per_spectrum": median,
        "median_spectra_per_second": 1.0 / median,
        "p95_seconds_per_spectrum": float(np.percentile(seconds_per_spectrum, 95)),
        "encoder_passes_per_representation": 1,
        "local_vb_steps": 0,
        "cached_projected_token_table": True,
    }


def evaluate_attempt(
    run_dir: str | Path,
    *,
    attempt: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one selected attempt once on the fixed validation/test split."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation" / attempt
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"{attempt} evaluation artifact changed: {name}"
                raise ValueError(msg)
        return result

    lock = read_json(directory / "neural.lock.json")
    counts = Path(lock["source_run"]) / "shared/counts"
    train = load_csr(counts / "train.npz")
    validation_observed = load_csr(counts / "validation_observed.npz")
    validation_completion = load_csr(counts / "validation_completion.npz")
    test_observed = load_csr(counts / "test_observed.npz")
    test_completion = load_csr(counts / "test_completion.npz")
    test_full = load_csr(counts / "test_full.npz")
    validation_records = load_heldout_records(counts, "validation")
    test_records = load_heldout_records(counts, "test")
    model = load_attempt_model(directory, attempt, protocol)
    batch_size = int(protocol["training"]["batch_size"])
    evaluation = protocol["evaluation"]

    torch.set_num_threads(int(protocol["evaluation_cpu_threads"]))
    started = time.perf_counter()
    projected_tokens = model.projected_tokens()
    beta = (
        model.topic_word_distribution(projected_tokens)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    validation_theta = infer_theta(
        model,
        validation_observed,
        batch_size=batch_size,
        projected_tokens=projected_tokens,
    )
    test_theta = infer_theta(
        model,
        test_observed,
        batch_size=batch_size,
        projected_tokens=projected_tokens,
    )
    test_full_theta = infer_theta(
        model,
        test_full,
        batch_size=batch_size,
        projected_tokens=projected_tokens,
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
            document_threshold=float(evaluation["document_active_threshold"]),
            corpus_threshold=float(evaluation["corpus_active_threshold"]),
        ),
        "full_spectrum_mixture": effective_topic_summary(test_full_theta),
        "top_word_diversity": top_word_diversity(
            beta,
            top_n=int(evaluation["topic_top_n"]),
        ),
        "word_cooccurrence_npmi": sparse_npmi(
            beta,
            train,
            top_n=int(evaluation["topic_top_n"]),
        ),
        "cached_latency": _latency(
            model,
            test_observed,
            subset_size=int(evaluation["latency_subset_size"]),
            repeats=int(evaluation["latency_repeats"]),
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
    training = read_json(directory / "attempts" / attempt / "complete.json")
    audit = read_json(directory / "candidate_audit.json")
    result = {
        "schema_version": "fully-neural-ms2lda/evaluation-complete-v1",
        "attempt": attempt,
        "stable": bool(stable and training["stable"]),
        "fully_neural_audit": audit,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "evaluation_cpu_threads": int(protocol["evaluation_cpu_threads"]),
        "encoder_passes_per_representation": 1,
        "local_vb_steps": 0,
        "iterative_test_inference_steps": 0,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result


def nonchemical_hard_gates(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply every hard gate that does not require MAG."""
    thresholds = protocol["hard_viability_gates"]
    metrics = candidate["metrics"]
    active = int(metrics["active_topics"]["corpus_active_topics"])
    reference_active = int(reference["active_topics"]["corpus_active_topics"])
    diversity = float(metrics["top_word_diversity"])
    reference_diversity = float(reference["top_word_diversity"])
    effective = float(
        metrics["full_spectrum_mixture"]["effective_topic_count_median"],
    )
    reference_effective = float(
        reference["full_spectrum_mixture"]["effective_topic_count_median"],
    )
    nll = float(metrics["test_document_completion"]["nll_per_token"])
    reference_nll = float(reference["document_completion"]["nll_per_token"])
    checks = {
        "fully_neural_audit": {
            "pass": bool(candidate["fully_neural_audit"]["fully_neural"]),
            "actual": bool(candidate["fully_neural_audit"]["fully_neural"]),
            "required": True,
        },
        "stable": {
            "pass": bool(candidate["stable"]),
            "actual": bool(candidate["stable"]),
            "required": True,
        },
        "corpus_active_topics": {
            "pass": active
            >= thresholds["active_topics_reference_fraction"] * reference_active,
            "actual": active,
            "minimum": thresholds["active_topics_reference_fraction"]
            * reference_active,
            "reference": reference_active,
        },
        "top_word_diversity": {
            "pass": diversity
            >= reference_diversity - thresholds["top_word_diversity_absolute_drop"],
            "actual": diversity,
            "minimum": reference_diversity
            - thresholds["top_word_diversity_absolute_drop"],
            "reference": reference_diversity,
        },
        "median_effective_topics": {
            "pass": reference_effective
            * thresholds["effective_topics_reference_minimum_fraction"]
            <= effective
            <= reference_effective
            * thresholds["effective_topics_reference_maximum_multiple"],
            "actual": effective,
            "minimum": reference_effective
            * thresholds["effective_topics_reference_minimum_fraction"],
            "maximum": reference_effective
            * thresholds["effective_topics_reference_maximum_multiple"],
            "reference": reference_effective,
        },
        "heldout_nll": {
            "pass": nll <= reference_nll * thresholds["nll_reference_maximum_fraction"],
            "actual": nll,
            "maximum": reference_nll * thresholds["nll_reference_maximum_fraction"],
            "reference": reference_nll,
        },
    }
    return {
        "checks": checks,
        "pass": all(row["pass"] for row in checks.values()),
    }
