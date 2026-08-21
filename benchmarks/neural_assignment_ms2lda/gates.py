"""Predeclared validation gate for the seed-42 document-mixture experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import file_sha256, read_json, write_json

MINIMUM_GAP_CLOSED = 0.5
MINIMUM_ANNOTATION_COVERAGE = 0.496
MAXIMUM_MEAN_SOS_DROP = 0.02
MAXIMUM_VALIDATION_NLL = 8.5266
PRIOR_RUNTIME_REFERENCE_SECONDS = 5571.0


def _useful_high_confidence_topics(chemistry: dict[str, Any]) -> int:
    bands = chemistry["high_confidence_chemistry"]["sos_bands"]
    return int(bands["high_gt_0_8"]) + int(bands["intermediate_0_6_to_0_8"])


def evaluate_validation_gate(run_dir: str | Path) -> dict[str, Any]:
    """Decide whether the candidate may be evaluated once on test."""
    directory = Path(run_dir).expanduser().resolve()
    paths = {
        "current_neural_chemistry": directory
        / "validation_chemical/current_neural/complete.json",
        "candidate_neural_chemistry": directory
        / "validation_chemical/candidate_neural/complete.json",
        "tomotopy_chemistry": directory / "validation_chemical/tomotopy/complete.json",
        "candidate_validation": directory
        / "validation_evaluation/candidate_neural/complete.json",
        "candidate_training": directory / "model/complete.json",
        "candidate_selected": directory / "model/selected.json",
    }
    evidence = {name: read_json(path) for name, path in paths.items()}
    current = evidence["current_neural_chemistry"]
    candidate = evidence["candidate_neural_chemistry"]
    comparator = evidence["tomotopy_chemistry"]
    evaluation = evidence["candidate_validation"]
    training = evidence["candidate_training"]
    selected = evidence["candidate_selected"]

    current_useful = _useful_high_confidence_topics(current)
    candidate_useful = _useful_high_confidence_topics(candidate)
    comparator_useful = _useful_high_confidence_topics(comparator)
    gap = comparator_useful - current_useful
    closed_fraction = (
        (candidate_useful - current_useful) / gap
        if gap > 0
        else float(candidate_useful >= comparator_useful)
    )
    current_mean_sos = float(current["high_confidence_chemistry"]["mean_sos"])
    candidate_mean_sos = float(candidate["high_confidence_chemistry"]["mean_sos"])
    validation_nll = float(
        evaluation["metrics"]["validation_document_completion"]["nll_per_token"]
    )
    training_seconds = float(training["elapsed_seconds"])
    checks = {
        "gap_closed": closed_fraction >= MINIMUM_GAP_CLOSED,
        "annotation_coverage": float(candidate["annotation_coverage"])
        >= MINIMUM_ANNOTATION_COVERAGE,
        "mean_high_confidence_sos": candidate_mean_sos
        >= current_mean_sos - MAXIMUM_MEAN_SOS_DROP,
        "validation_nll": validation_nll <= MAXIMUM_VALIDATION_NLL,
        "stable": bool(training["stable"]) and bool(evaluation["stable"]),
    }
    accepted = all(checks.values())
    result = {
        "schema_version": "neural-ms2lda/validation-gate-v1",
        "decision": "accepted" if accepted else "rejected",
        "test_evaluation_authorized": accepted,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "paper_outcome": {
            "current_neural_useful_high_confidence_motifs": current_useful,
            "candidate_neural_useful_high_confidence_motifs": candidate_useful,
            "tomotopy_useful_high_confidence_motifs": comparator_useful,
            "validation_gap": gap,
            "closed_fraction": closed_fraction,
        },
        "measurements": {
            "candidate_annotation_coverage": float(candidate["annotation_coverage"]),
            "current_mean_high_confidence_sos": current_mean_sos,
            "candidate_mean_high_confidence_sos": candidate_mean_sos,
            "candidate_validation_nll": validation_nll,
            "candidate_training_seconds": training_seconds,
        },
        "reported_context": {
            "training_within_prior_10_percent": training_seconds
            <= PRIOR_RUNTIME_REFERENCE_SECONDS,
            "prior_10_percent_runtime_seconds": PRIOR_RUNTIME_REFERENCE_SECONDS,
        },
        "thresholds": {
            "minimum_gap_closed": MINIMUM_GAP_CLOSED,
            "minimum_annotation_coverage": MINIMUM_ANNOTATION_COVERAGE,
            "maximum_mean_sos_drop": MAXIMUM_MEAN_SOS_DROP,
            "maximum_validation_nll": MAXIMUM_VALIDATION_NLL,
        },
        "checks": checks,
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    write_json(directory / "validation_gate.json", result)
    return result
