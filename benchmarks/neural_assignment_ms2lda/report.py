"""Machine-readable evidence report assembled only from frozen manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import file_sha256, read_json, write_json


def _method_row(
    result: dict[str, Any],
    chemistry: dict[str, Any],
    *,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = result["metrics"]
    dominant = chemistry["dominant_topic_chemistry"]
    confident = chemistry["high_confidence_chemistry"]
    row = {
        "method": result["method"],
        "topics": int(result["topic_count"]),
        "test_nll": float(metrics["test_document_completion"]["nll_per_token"]),
        "top_word_diversity": float(metrics["top_word_diversity"]),
        "mean_npmi": float(metrics["word_cooccurrence_npmi"]["mean_npmi"]),
        "undefined_pair_fraction": float(
            metrics["word_cooccurrence_npmi"]["undefined_pair_fraction"]
        ),
        "active_topics": int(metrics["active_topics"]["corpus_active_topics"]),
        "effective_topics_median": float(
            metrics["full_spectrum_mixture"]["effective_topic_count_median"]
        ),
        "spectra_per_second": float(
            metrics["cached_latency"]["median_spectra_per_second"]
        ),
        "annotation_coverage": float(chemistry["annotation_coverage"]),
        "annotated_topics": int(
            round(float(chemistry["annotation_coverage"]) * chemistry["topics"])
        ),
        "dominant_eligible_topics": int(dominant["eligible_topics"]),
        "dominant_mean_sos": dominant["mean_sos"],
        "high_confidence_eligible_topics": int(confident["eligible_topics"]),
        "high_confidence_mean_sos": confident["mean_sos"],
        "high_confidence_associated_spectra": int(confident["associated_spectra"]),
        "inference_workers": int(
            result.get("inference_workers", result.get("cpu_threads", 1))
        ),
        "peak_rss_bytes": int(
            result.get("source_peak_rss_bytes", result["peak_rss_bytes"])
        ),
    }
    inventory = metrics["topic_inventory"]
    row["mass99_distinct_topic_equivalents"] = float(
        inventory["mass_coverages"]["mass_99"]["distinct_topic_equivalents"]
    )
    if training is None:
        row["training_seconds"] = float(result["training_seconds_total"])
        row["training_workers"] = int(result["training_workers"])
        row["training_reused"] = bool(result["training_reused"])
    else:
        row["training_seconds"] = float(training["elapsed_seconds"])
        row["training_workers"] = int(result["cpu_threads"])
        row["training_reused"] = False
        row["peak_rss_bytes"] = int(training["peak_rss_bytes"])
    return row


def build_machine_report(run_dir: str | Path) -> dict[str, Any]:
    """Build the final comparison from completed evaluation artifacts."""
    directory = Path(run_dir).expanduser().resolve()
    neural = read_json(directory / "evaluation/neural/complete.json")
    comparator = read_json(directory / "evaluation/tomotopy/complete.json")
    neural_training = read_json(directory / "model/complete.json")
    protocol = read_json(directory / "protocol.resolved.json")
    neural_chemistry = read_json(directory / "chemical/neural/complete.json")
    comparator_chemistry = read_json(directory / "chemical/tomotopy/complete.json")
    rows = [
        _method_row(neural, neural_chemistry, training=neural_training),
        _method_row(comparator, comparator_chemistry),
    ]
    result = {
        "schema_version": "neural-ms2lda/research-report-v1",
        "title": "Neural MS2LDA on MSnLib",
        "evidence_scope": "single-seed applied-method reproducibility checkpoint",
        "headline": "A controlled K=1000, six-thread comparison on seed 42.",
        "comparison_contract": {
            "seed": int(protocol["seed"]),
            "topics": int(protocol["model"]["num_topics"]),
            "cpu_threads": int(protocol["cpu_threads"]),
            "neural_selected_epoch": int(neural["selected_epoch"]),
            "tomotopy_inference_iterations": int(comparator["inference_iterations"]),
            "tomotopy_training_reused": True,
        },
        "methods": rows,
        "training_contract": {
            "unsupervised": True,
            "tomotopy_teacher": False,
            "dreams_input": False,
            "variational_bayes": False,
            "chemistry_labels": False,
            "test_information": False,
        },
        "limitations": [
            "single fixed seed",
            "research checkpoint rather than production replacement",
            "cached CPU throughput compares native inference algorithms rather than equal operation counts",
            "Tomotopy training is reused while its held-out inference is recomputed",
        ],
        "source_sha256": {
            "protocol": file_sha256(directory / "protocol.resolved.json"),
            "neural_evaluation": file_sha256(
                directory / "evaluation/neural/complete.json"
            ),
            "tomotopy_evaluation": file_sha256(
                directory / "evaluation/tomotopy/complete.json"
            ),
            "neural_chemistry": file_sha256(
                directory / "chemical/neural/complete.json"
            ),
            "tomotopy_chemistry": file_sha256(
                directory / "chemical/tomotopy/complete.json"
            ),
        },
    }
    write_json(directory / "report/report.json", result)
    return result
