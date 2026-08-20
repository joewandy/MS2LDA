"""Machine-readable evidence report assembled only from frozen manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import file_sha256, read_json, write_json


def _architecture_copy(protocol: dict[str, Any]) -> tuple[str, str]:
    """Describe the architecture recorded by this run's frozen protocol."""
    if "hierarchical_routing" not in protocol:
        return (
            "Collapse-resistant fully neural MS2LDA on MSnLib",
            "The K=500 ERNTM model is a working, collapse-resistant, fast fully "
            "neural discovery model; Tomotopy remains stronger on coherence and "
            "chemical interpretation and remains the production comparator.",
        )
    routing = protocol["hierarchical_routing"]
    if routing.get("method") != "local_document_product_of_experts":
        raise ValueError("cannot report an unknown hierarchical routing architecture")
    return (
        "Hierarchical co-occurrence neural MS2LDA on MSnLib",
        "The K=500 hierarchical co-occurrence and topic-margin model is a "
        "working, collapse-resistant, fast fully neural discovery model; "
        "Tomotopy remains the production comparator.",
    )


def _method_row(result: dict[str, Any], chemistry: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    dominant = chemistry["dominant_topic_chemistry"]
    confident = chemistry["high_confidence_chemistry"]
    row = {
        "method": result["method"],
        "topics": int(result["topic_count"]),
        "test_nll": float(metrics["test_document_completion"]["nll_per_token"]),
        "top_word_diversity": float(metrics["top_word_diversity"]),
        "mean_npmi": float(metrics["word_cooccurrence_npmi"]["mean_npmi"]),
        "active_topics": int(metrics["active_topics"]["corpus_active_topics"]),
        "spectra_per_second": float(
            metrics["cached_latency"]["median_spectra_per_second"]
        ),
        "annotation_coverage": float(chemistry["annotation_coverage"]),
        "dominant_eligible_topics": int(dominant["eligible_topics"]),
        "dominant_mean_sos": dominant["mean_sos"],
        "high_confidence_eligible_topics": int(confident["eligible_topics"]),
        "high_confidence_mean_sos": confident["mean_sos"],
    }
    inventory = metrics.get("topic_inventory")
    if inventory is not None:
        row["mass99_distinct_topic_equivalents"] = float(
            inventory["mass_coverages"]["mass_99"]["distinct_topic_equivalents"]
        )
    return row


def build_machine_report(run_dir: str | Path) -> dict[str, Any]:
    """Build the final comparison from completed evaluation artifacts."""
    directory = Path(run_dir).expanduser().resolve()
    protocol = read_json(directory / "protocol.resolved.json")
    title, headline = _architecture_copy(protocol)
    neural = read_json(directory / "evaluation/neural/complete.json")
    comparator = read_json(directory / "evaluation/tomotopy/complete.json")
    neural_chemistry = read_json(directory / "chemical/neural/complete.json")
    comparator_chemistry = read_json(directory / "chemical/tomotopy/complete.json")
    rows = [
        _method_row(neural, neural_chemistry),
        _method_row(comparator, comparator_chemistry),
    ]
    result = {
        "schema_version": "neural-ms2lda/research-report-v1",
        "title": title,
        "evidence_scope": "single-seed applied-method reproducibility checkpoint",
        "headline": headline,
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
            "no claim of Tomotopy parity",
            "research checkpoint rather than production replacement",
            "chemical annotation coverage remains lower for the neural model",
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
