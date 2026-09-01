"""Validate and load the sealed evidence consumed by the manuscript renderer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .study_protocol import (
    METHOD,
    SYNTHETIC_ARTIFACT_LABELS,
    SYNTHETIC_DISPLAY_LABELS,
    TRAINING_SEEDS,
)

EXPECTED_METHOD = METHOD
EXPECTED_TRAINING_SEEDS = list(TRAINING_SEEDS)
EXPECTED_SYNTHETIC_FORMULATIONS = 4
EXPECTED_HIGH_K_ROWS = 3
BALANCED_SOFTMAX = SYNTHETIC_DISPLAY_LABELS[
    SYNTHETIC_ARTIFACT_LABELS["balanced_softmax"]
]
BALANCED_ENTMAX = SYNTHETIC_DISPLAY_LABELS[SYNTHETIC_ARTIFACT_LABELS["balanced_entmax"]]
CONTEXT_SOFTMAX = SYNTHETIC_DISPLAY_LABELS[
    SYNTHETIC_ARTIFACT_LABELS["contextual_softmax"]
]
CONTEXT_ENTMAX = SYNTHETIC_DISPLAY_LABELS[
    SYNTHETIC_ARTIFACT_LABELS["contextual_entmax"]
]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    """Hash one evidence artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package_integrity(root: Path, checkpoint: dict[str, Any]) -> None:
    """Verify the package seal and every compact evidence artifact it owns."""
    manifest_path = root / "fresh_evidence_manifest.json"
    if _sha256_file(manifest_path) != checkpoint["fresh_evidence_manifest_sha256"]:
        msg = "fresh evidence manifest changed after checkpointing"
        raise ValueError(msg)
    manifest = _json(manifest_path)
    for row in manifest["packaged_files"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            msg = f"packaged evidence changed after sealing: {path}"
            raise ValueError(msg)


def require_reportable_claims(acceptance: dict[str, Any]) -> None:
    """Reject affirmative manuscript prose when a declared direction failed."""
    if acceptance.get("all_passed") is not True:
        failed = sorted(
            name for name, passed in acceptance.get("checks", {}).items() if not passed
        )
        msg = (
            "directional claims failed; revise the report before rendering: "
            + ", ".join(
                failed,
            )
        )
        raise ValueError(msg)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_field(row: dict[str, str], key: str) -> float:
    """Read one required numeric CSV field."""
    value = row.get(key)
    if value is None or value == "":
        msg = f"missing {key!r} in {row}"
        raise ValueError(msg)
    return float(value)


def integer_field(row: dict[str, str], key: str) -> int:
    """Read one required integer-valued CSV field."""
    return round(float_field(row, key))


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        msg = f"{name} changed: {actual} != {expected}"
        raise ValueError(msg)


def _require_manuscript_claims(evidence: dict[str, Any]) -> None:
    """Verify every data-dependent comparison stated by the static manuscript."""
    comparison = evidence["comparison"]
    canonical = comparison["canonical ETM"]
    balanced = comparison["balanced ETM"]
    contextual = comparison["Contextual Sparse ETM"]
    tomotopy = comparison["Tomotopy LDA"]
    synthetic = {row["formulation"]: row for row in evidence["synthetic"]}
    high_k = {row["formulation"]: row for row in evidence["high_k"]}
    base = synthetic[BALANCED_SOFTMAX]
    sparse = synthetic[BALANCED_ENTMAX]
    context_dense = synthetic[CONTEXT_SOFTMAX]
    complete = synthetic[CONTEXT_ENTMAX]
    high_complete = high_k[CONTEXT_ENTMAX]

    checks = {
        "k36_context_improves_beta": float_field(
            context_dense,
            "mean_true_beta_cosine",
        )
        > float_field(base, "mean_true_beta_cosine"),
        "k36_context_improves_theta": float_field(
            context_dense,
            "mean_true_theta_cosine",
        )
        > float_field(base, "mean_true_theta_cosine"),
        "k36_context_improves_nll": float_field(context_dense, "mean_nll")
        < float_field(base, "mean_nll"),
        "k36_context_uses_more_winners": float_field(
            context_dense,
            "mean_unique_top1_topics",
        )
        > float_field(base, "mean_unique_top1_topics"),
        "k36_entmax_alone_reduces_beta": float_field(
            sparse,
            "mean_true_beta_cosine",
        )
        < float_field(base, "mean_true_beta_cosine"),
        "k36_entmax_alone_reduces_theta": float_field(
            sparse,
            "mean_true_theta_cosine",
        )
        < float_field(base, "mean_true_theta_cosine"),
        "k36_entmax_alone_uses_fewer_winners": float_field(
            sparse,
            "mean_unique_top1_topics",
        )
        < float_field(base, "mean_unique_top1_topics"),
        "k36_complete_has_strongest_beta": float_field(
            complete,
            "mean_true_beta_cosine",
        )
        == max(float_field(row, "mean_true_beta_cosine") for row in synthetic.values()),
        "k36_complete_has_strongest_theta": float_field(
            complete,
            "mean_true_theta_cosine",
        )
        == max(
            float_field(row, "mean_true_theta_cosine") for row in synthetic.values()
        ),
        "k36_complete_nll_is_worse_than_context_softmax": float_field(
            complete,
            "mean_nll",
        )
        > float_field(context_dense, "mean_nll"),
        "high_k_complete_has_lowest_nll": float_field(high_complete, "nll")
        == min(float_field(row, "nll") for row in high_k.values()),
        "balancing_raises_optimized_motifs": integer_field(
            balanced,
            "optimized_motifs",
        )
        > integer_field(canonical, "optimized_motifs"),
        "balancing_raises_evaluable_motifs": integer_field(
            balanced,
            "evaluable_motifs",
        )
        > integer_field(canonical, "evaluable_motifs"),
        "balancing_raises_useful_motifs": integer_field(balanced, "useful_motifs")
        > integer_field(canonical, "useful_motifs"),
        "contextual_has_fewer_optimized_motifs_than_balanced": integer_field(
            contextual,
            "optimized_motifs",
        )
        < integer_field(balanced, "optimized_motifs"),
        "contextual_nll_is_lower_than_tomotopy": float_field(
            contextual,
            "completion_nll",
        )
        < float_field(tomotopy, "completion_nll"),
        "etm_controls_have_more_effective_topics": min(
            float_field(canonical, "median_effective_topics"),
            float_field(balanced, "median_effective_topics"),
        )
        > float_field(contextual, "median_effective_topics"),
        "etm_controls_have_fewer_unique_winners": max(
            integer_field(canonical, "unique_top1_topics"),
            integer_field(balanced, "unique_top1_topics"),
        )
        < integer_field(contextual, "unique_top1_topics"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        msg = (
            "manuscript result prose is not supported by fresh evidence: "
            + ", ".join(
                failed,
            )
        )
        raise ValueError(msg)


def load_report_evidence(evidence_root: Path) -> dict[str, Any]:  # noqa: C901
    """Load one sealed bundle and enforce every manuscript-facing contract."""
    root = evidence_root.expanduser().resolve(strict=True)
    preparation = _json(root / "preparation_summary.json")
    protocol = _json(root / "protocol.json")
    checkpoint = _json(root / "checkpoint_manifest.json")
    validate_package_integrity(root, checkpoint)
    config = _json(root / "config.json")
    metrics = _json(root / "metrics.json")
    comparison = {row["model"]: row for row in _csv(root / "comparison.csv")}
    synthetic = _csv(root / "synthetic_summary.csv")
    high_k = _csv(root / "high_k_stress.csv")
    stability = _json(root / "stability_summary.json")
    tomotopy = _json(root / "tomotopy.json")
    acceptance = _json(root / "acceptance.json")
    data_quality = _json(root / "data_quality.json")

    config_method = config.get("artifact_method_id", config.get("method"))
    if checkpoint["method"] != EXPECTED_METHOD or config_method != EXPECTED_METHOD:
        raise ValueError("Contextual Sparse ETM method identity changed")
    if checkpoint["test_released_after_model_and_validation_freeze"] is not True:
        raise ValueError(
            "test split was not released after model and validation freeze"
        )
    if checkpoint["acceptance_all_passed"] is not acceptance["all_passed"]:
        raise ValueError("checkpoint and detailed claim checks disagree")
    require_reportable_claims(acceptance)
    if data_quality["status"] != "pass":
        raise ValueError("fresh evidence did not pass data-quality checks")
    if config["candidate_test_artifacts_accessed"] is not False:
        raise ValueError("training configuration indicates test access")
    if (
        stability["direction_checks"]["test_released_only_after_model_freeze"]
        is not True
    ):
        raise ValueError("stability evidence does not preserve split ordering")
    if (
        stability["runs"] != len(EXPECTED_TRAINING_SEEDS)
        or stability["training_seeds"] != EXPECTED_TRAINING_SEEDS
    ):
        raise ValueError("expected exactly the three frozen training seeds")

    data = preparation["data"]
    if data["leakage_audit"]["leaked_compounds"] != 0:
        raise ValueError("compound leakage detected")
    if data["leakage_audit"]["leaked_groups"] != 0:
        raise ValueError("split-group leakage detected")
    if data["split"]["seed"] != protocol["seed"]:
        raise ValueError("preparation and protocol split seeds differ")
    if protocol["chemistry"].get("spectrum_topic_assignment") != "dominant_topic":
        raise ValueError("chemical evaluation must use dominant-topic assignment")
    if data["vocabulary"]["vocabulary_size"] != preparation["vocabulary_size"]:
        raise ValueError("vocabulary size mismatch")

    required_models = {"canonical ETM", "balanced ETM", "Contextual Sparse ETM"}
    if not required_models.issubset(comparison):
        raise ValueError("comparison is missing a required ETM baseline")
    test_spectra = int(data["split"]["spectrum_counts"]["test"])
    if any(
        integer_field(row, "spectrum_topic_associations") != test_spectra
        for row in comparison.values()
    ):
        raise ValueError("every model must associate each test spectrum exactly once")
    proposed_row = comparison["Contextual Sparse ETM"]
    if proposed_row["finite_stable"] != "True":
        raise ValueError("Contextual Sparse ETM is not marked finite and stable")
    chemistry = metrics["test_chemistry"]
    completion = metrics["document_completion"]
    for key, source_key in (
        ("optimized_motifs", "optimized_motifs"),
        ("evaluable_motifs", "eligible_topics"),
        ("useful_motifs", "useful_motifs"),
    ):
        if integer_field(proposed_row, key) != int(chemistry[source_key]):
            msg = f"Contextual Sparse ETM {key} disagrees with metrics.json"
            raise ValueError(msg)
    _close(
        float_field(proposed_row, "mean_sos"), chemistry["mean_sos"], name="mean SOS"
    )
    _close(
        float_field(proposed_row, "completion_nll"),
        completion["nll_per_token"],
        name="completion NLL",
    )
    if int(metrics["parameters"]) != integer_field(proposed_row, "parameters"):
        raise ValueError("unexpected Contextual Sparse ETM parameter count")
    if int(metrics["parameters"]) - integer_field(
        comparison["canonical ETM"],
        "parameters",
    ) != int(config["context_parameters"]):
        raise ValueError("Contextual Sparse ETM parameter increment changed")

    if tomotopy.get("method") != "tomotopy":
        raise ValueError("fresh Tomotopy comparator is missing")
    tomotopy_nll = float(tomotopy["test"]["completion_nll"])
    _close(
        float_field(comparison["Tomotopy LDA"], "completion_nll"),
        tomotopy_nll,
        name="Tomotopy completion NLL",
    )

    if len(synthetic) != EXPECTED_SYNTHETIC_FORMULATIONS or {
        int(row["seeds"]) for row in synthetic
    } != {len(EXPECTED_TRAINING_SEEDS)}:
        raise ValueError("expected four three-seed K=36 synthetic formulations")
    if {int(row["k"]) for row in synthetic} != {36}:
        raise ValueError("synthetic summary K changed")
    if len(high_k) != EXPECTED_HIGH_K_ROWS or {
        int(row["fitted_topics"]) for row in high_k
    } != {128}:
        raise ValueError("expected three K=128 rows")
    if {int(row["true_topics"]) for row in high_k} != {18}:
        raise ValueError("high-K planted topic count changed")

    evidence = {
        "preparation": preparation,
        "protocol": protocol,
        "config": config,
        "metrics": metrics,
        "comparison": comparison,
        "synthetic": synthetic,
        "high_k": high_k,
        "stability": stability,
        "tomotopy": tomotopy,
        "tomotopy_nll": tomotopy_nll,
        "evidence_root": root,
    }
    _require_manuscript_claims(evidence)
    return evidence
