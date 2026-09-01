"""Package one sealed Contextual Sparse ETM clean-room reproduction.

Only artifacts owned by the supplied reproduction UUID are accepted.  The
packager verifies stage hashes, split-release ordering, probability matrices,
MAG failure counts, and scientific integrity checks before writing the
compact evidence bundle consumed by the LaTeX report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.neural_ms2lda.chemical import score_precomputed_annotations
from benchmarks.neural_ms2lda.data import load_heldout_records
from benchmarks.neural_ms2lda.evidence_bundle import (
    assert_no_machine_paths as _assert_no_machine_paths,
)
from benchmarks.neural_ms2lda.evidence_bundle import (
    copy_raw_evidence as _copy_raw_evidence,
)
from benchmarks.neural_ms2lda.evidence_bundle import (
    path_replacements as _path_replacements,
)
from benchmarks.neural_ms2lda.evidence_bundle import (
    rewrite_json_as_portable as _rewrite_json_as_portable,
)
from benchmarks.neural_ms2lda.evidence_bundle import (
    write_package_seals as _write_package_seals,
)
from benchmarks.neural_ms2lda.evidence_bundle import (
    write_summary_artifacts as _write_summary_artifacts,
)
from benchmarks.neural_ms2lda.reproduction_audit import (
    probability_audit,
    read_json,
    sha256_file,
    validate_model_views,
    verify_stage_records,
)
from benchmarks.neural_ms2lda.reproduction_plan import (
    reproduction_paths,
)
from benchmarks.neural_ms2lda.study_protocol import (
    FINAL_SYNTHETIC_LABEL,
    METHOD,
    NEURAL_DEVICE,
    SYNTHETIC_DISPLAY_LABELS,
    SYNTHETIC_SEEDS,
    TRAINING_SEEDS,
    load_protocol,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FORMULATION_LABELS = SYNTHETIC_DISPLAY_LABELS
PRIMARY_SYNTHETIC_TOPICS = 36
HIGH_K_SYNTHETIC_TOPICS = 128
PLANTED_SYNTHETIC_TOPICS = 18
MAXIMUM_EFFECTIVE_TOPICS = 5.0
MINIMUM_UNIQUE_WINNERS = 800
MAXIMUM_HIGH_K_SUPPORT = 3.0
SUMMARY_FIELDS = (
    "optimized_motifs",
    "evaluable_motifs",
    "useful_motifs",
    "mean_sos",
    "median_sos",
    "completion_nll",
    "median_effective_topics",
    "median_exact_support",
    "unique_top1_topics",
    "corpus_effective_topics",
    "learned_context_scale",
    "training_wall_seconds",
)


def _chemistry_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize shared MAG/SOS output into the paper's summary schema."""
    if "mag_failures" not in result:
        msg = "fresh chemical evidence lacks explicit MAG exception accounting"
        raise RuntimeError(msg)
    if result.get("heldout_compounds_excluded_from_mag") is not True:
        msg = "fresh chemical evidence does not exclude held-out compounds from MAG"
        raise RuntimeError(msg)
    summary = dict(result["chemical_evaluation"])
    if summary.get("association_rule") != "dominant_topic":
        msg = "chemical evidence must use dominant-topic assignment"
        raise RuntimeError(msg)
    summary.pop("topic_scores", None)
    topics = int(result["topics"])
    bands = summary["sos_bands"]
    eligible = int(summary["eligible_topics"])
    band_total = sum(
        int(bands[name])
        for name in (
            "high_gt_0_8",
            "intermediate_0_6_to_0_8",
            "low_lt_0_6",
        )
    )
    if band_total != eligible:
        msg = f"SOS bands account for {band_total} motifs but {eligible} are evaluable"
        raise RuntimeError(msg)
    failures = result["mag_failures"]
    total_failures = 0
    for kind in ("clustering", "optimization"):
        count = int(failures[f"{kind}_count"])
        topic_ids = failures[f"{kind}_topic_ids"]
        if count < 0 or count != len(topic_ids):
            msg = f"MAG {kind} exception count and topic IDs disagree"
            raise RuntimeError(msg)
        total_failures += count
    if total_failures:
        msg = f"fresh chemical evidence contains {total_failures} MAG exceptions"
        raise RuntimeError(msg)
    optimized = round(float(result["annotation_coverage"]) * topics)
    summary.update(
        {
            "optimized_motifs": optimized,
            "useful_motifs": int(bands["high_gt_0_8"])
            + int(bands["intermediate_0_6_to_0_8"]),
            "annotation_coverage": float(result["annotation_coverage"]),
            "heldout_compounds_excluded_from_mag": bool(
                result["heldout_compounds_excluded_from_mag"],
            ),
            "mag_failures": result["mag_failures"],
            "sos_band_accounting_valid": True,
            "split": str(result["split"]),
        },
    )
    return summary


def _chemical_evaluation_result(
    run: Path,
    *,
    method: str,
    split: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute chemical scores from frozen mixtures and MAG annotations.

    Model fitting, inference and beta-dependent MAG annotation remain frozen.
    Recomputing this inexpensive final layer ensures that every packaged result
    uses the model-neutral dominant-topic association rule.
    """
    if split not in {"validation", "test"}:
        msg = "chemical split must be validation or test"
        raise ValueError(msg)
    evaluation_group = (
        "validation_evaluation" if split == "validation" else "evaluation"
    )
    theta_path = run / evaluation_group / method / f"{split}_full_theta.npy"
    records_path = run / "data" / f"{split}_records.jsonl"
    annotations_path = run / "mag" / "annotations" / method / "annotations.jsonl"
    annotation_summary_path = run / "mag" / "annotations" / method / "complete.json"

    theta = np.load(theta_path, mmap_mode="r")
    records = load_heldout_records(run / "data", split)
    if theta.shape[0] != len(records):
        msg = "full mixtures and held-out records differ"
        raise ValueError(msg)
    annotations = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    annotation = read_json(annotation_summary_path)
    summary = score_precomputed_annotations(
        theta=theta,
        records=records,
        annotations=annotations,
        fingerprint_threshold=float(protocol["chemistry"]["mag_fingerprint_threshold"]),
    )
    if summary["association_rule"] != "dominant_topic":
        msg = "chemical evaluation did not use dominant-topic assignment"
        raise RuntimeError(msg)
    if int(summary["associated_spectra"]) != len(records):
        msg = "dominant-topic evaluation must assign every spectrum once"
        raise RuntimeError(msg)
    return {
        "method": method,
        "annotation_method": method,
        "split": split,
        "topics": len(annotations),
        "annotation_coverage": annotation["annotation_coverage"],
        "chemical_evaluation": summary,
        "heldout_compounds_excluded_from_mag": annotation[
            "heldout_compounds_excluded_from_mag"
        ],
        "mag_failures": annotation["mag_failures"],
        "evidence_inputs": {
            "theta_sha256": sha256_file(theta_path),
            "records_sha256": sha256_file(records_path),
            "annotations_sha256": sha256_file(annotations_path),
            "annotation_summary_sha256": sha256_file(annotation_summary_path),
        },
    }


def _synthetic_row(result: Mapping[str, Any], *, stage: str) -> dict[str, object]:
    """Extract one truth-known result row without rounding away evidence."""
    config = result["config"]
    _require_neural_device(config.get("device"), label=f"synthetic stage {stage}")
    metrics = result["metrics"]
    recovery = metrics["truth_recovery"]
    support = metrics["theta_support"]
    inventory = metrics["topic_inventory"]
    return {
        "stage": stage,
        "seed": int(config["seed"]),
        "k": int(config["fitted_topics"]),
        "true_topics": int(config["true_topics"]),
        "formulation": FORMULATION_LABELS[result["method"]],
        "implementation_class": config.get("implementation_class", "unknown"),
        "nll": float(metrics["heldout_completion"]["nll_per_token"]),
        "true_beta_cosine": float(recovery["true_beta_matched_cosine_mean"]),
        "true_theta_cosine": float(recovery["true_theta_cosine_mean"]),
        "top_motif_accuracy": float(recovery["top_planted_motif_accuracy"]),
        "planted_motifs_recovered_cosine_ge_0_50": int(
            recovery["planted_motifs_recovered_cosine_ge_0_50"],
        ),
        "median_effective_topics": float(
            support["median_effective_topics_per_spectrum"],
        ),
        "median_exact_support": float(support["median_exact_support"]),
        "active_topics_gt_0_005": int(metrics["active_topics_mean_usage_gt_0_005"]),
        "unique_top1_topics": int(inventory["unique_top1_topics"]),
        "mean_nearest_beta_cosine": float(inventory["mean_nearest_topic_beta_cosine"]),
        "maximum_beta_cosine": float(inventory["maximum_pairwise_beta_cosine"]),
        "catastrophic_duplicate": bool(inventory["catastrophic_duplicate_component"]),
    }


def _require_neural_device(value: object, *, label: str) -> None:
    """Reject evidence not executed on the reproduction's required device."""
    if str(value).split(":", maxsplit=1)[0] != NEURAL_DEVICE:
        msg = f"{label} was not executed on {NEURAL_DEVICE}: {value}"
        raise RuntimeError(msg)


def _synthetic_tables(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read exactly the study's 12 K=36 and three K=128 fits."""
    synthetic_root = reproduction_paths(root).synthetic / "synthetic_runs"
    rows = []
    for result_path in sorted(synthetic_root.glob("*/result.json")):
        result = read_json(result_path)
        topics = int(result["config"]["fitted_topics"])
        rows.append(
            _synthetic_row(
                result,
                stage=(
                    "multi_seed" if topics == PRIMARY_SYNTHETIC_TOPICS else "high_k"
                ),
            ),
        )
    primary = [row for row in rows if row["k"] == PRIMARY_SYNTHETIC_TOPICS]
    high_k = [row for row in rows if row["k"] == HIGH_K_SYNTHETIC_TOPICS]
    if (
        len(primary) != len(SYNTHETIC_SEEDS) * len(FORMULATION_LABELS)
        or len(
            high_k,
        )
        != len(FORMULATION_LABELS) - 1
    ):
        msg = "expected exactly 12 K=36 and three K=128 synthetic fits"
        raise RuntimeError(msg)
    if {int(row["seed"]) for row in primary} != set(SYNTHETIC_SEEDS):
        msg = "synthetic seed set differs from the frozen plan"
        raise RuntimeError(msg)
    summary = []
    for formulation in FORMULATION_LABELS.values():
        selected = [row for row in primary if row["formulation"] == formulation]
        if len(selected) != len(SYNTHETIC_SEEDS):
            msg = f"synthetic formulation is incomplete: {formulation}"
            raise RuntimeError(msg)
        summary.append(
            {
                "formulation": formulation,
                "seeds": len(selected),
                "k": PRIMARY_SYNTHETIC_TOPICS,
                "mean_nll": statistics.mean(float(row["nll"]) for row in selected),
                "mean_true_beta_cosine": statistics.mean(
                    float(row["true_beta_cosine"]) for row in selected
                ),
                "mean_true_theta_cosine": statistics.mean(
                    float(row["true_theta_cosine"]) for row in selected
                ),
                "mean_median_effective_topics": statistics.mean(
                    float(row["median_effective_topics"]) for row in selected
                ),
                "mean_active_topics_gt_0_005": statistics.mean(
                    int(row["active_topics_gt_0_005"]) for row in selected
                ),
                "mean_unique_top1_topics": statistics.mean(
                    int(row["unique_top1_topics"]) for row in selected
                ),
                "decision": (
                    "promoted formulation"
                    if formulation == FINAL_SYNTHETIC_LABEL
                    else "ablation/control"
                ),
            },
        )
    for row in high_k:
        row["fitted_topics"] = row.pop("k")
        row["decision"] = (
            "promote to real validation"
            if row["formulation"] == FINAL_SYNTHETIC_LABEL
            else "ablation/control"
        )
    return primary, summary, high_k


def _model_row(
    label: str,
    evaluation: Mapping[str, Any],
    chemistry_result: Mapping[str, Any],
    training_result: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
) -> dict[str, object]:
    """Extract one frozen-model test comparison row."""
    metrics = evaluation["metrics"]
    chemistry = _chemistry_summary(chemistry_result)
    inventory = metrics["topic_inventory"]
    support = metrics.get("theta_support", {})
    return {
        "model": label,
        "optimized_motifs": int(chemistry["optimized_motifs"]),
        "evaluable_motifs": int(chemistry["eligible_topics"]),
        "useful_motifs": int(chemistry["useful_motifs"]),
        "useful_fraction_evaluable": (
            int(chemistry["useful_motifs"]) / int(chemistry["eligible_topics"])
        ),
        "spectrum_topic_associations": int(chemistry["associated_spectra"]),
        "mean_sos": float(chemistry["mean_sos"]),
        "median_sos": float(chemistry["median_sos"]),
        "completion_nll": float(metrics["document_completion"]["nll_per_token"]),
        "median_effective_topics": float(
            inventory["median_effective_topics_per_spectrum"],
        ),
        "mean_effective_topics": float(inventory["mean_effective_topics_per_spectrum"]),
        "median_exact_support": support.get("median_exact_support", ""),
        "p95_exact_support": support.get("support_size_percentiles", {}).get("95", ""),
        "unique_top1_topics": int(inventory["unique_top1_topics"]),
        "active_topics_gt_0_0005": int(
            inventory["active_topics_above_usage_threshold"],
        ),
        "corpus_effective_topics": float(inventory["corpus_effective_topic_count"]),
        "maximum_mean_topic_usage": float(inventory["maximum_mean_topic_usage"]),
        "mean_nearest_beta_cosine": float(inventory["mean_nearest_topic_beta_cosine"]),
        "maximum_beta_cosine": float(inventory["maximum_pairwise_beta_cosine"]),
        "catastrophic_duplicate_component": bool(
            inventory["catastrophic_duplicate_component"],
        ),
        "training_seconds": float(training_metrics["runtime"]["training_wall_seconds"]),
        "parameters": int(training_result["parameters"]),
        "finite_stable": bool(metrics["finite_stable"]),
        "mag_clustering_failures": int(chemistry["mag_failures"]["clustering_count"]),
        "mag_optimization_failures": int(
            chemistry["mag_failures"]["optimization_count"],
        ),
        "heldout_compounds_excluded_from_mag": bool(
            chemistry["heldout_compounds_excluded_from_mag"],
        ),
        "sos_band_accounting_valid": bool(chemistry["sos_band_accounting_valid"]),
    }


def _validation_model_row(
    label: str,
    training_result: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    chemistry_result: Mapping[str, Any],
) -> dict[str, object]:
    """Extract the development-split row retained beside final test evidence."""
    chemistry = _chemistry_summary(chemistry_result)
    inventory = training_metrics["topic_inventory"]
    support = training_metrics.get("theta_support", {})
    return {
        "model": label,
        "optimized_motifs": int(chemistry["optimized_motifs"]),
        "evaluable_motifs": int(chemistry["eligible_topics"]),
        "useful_motifs": int(chemistry["useful_motifs"]),
        "useful_fraction_evaluable": (
            int(chemistry["useful_motifs"]) / int(chemistry["eligible_topics"])
        ),
        "spectrum_topic_associations": int(chemistry["associated_spectra"]),
        "mean_sos": float(chemistry["mean_sos"]),
        "median_sos": float(chemistry["median_sos"]),
        "completion_nll": float(
            training_metrics["document_completion"]["nll_per_token"],
        ),
        "median_effective_topics": float(
            inventory["median_effective_topics_per_spectrum"],
        ),
        "median_exact_support": support.get("median_exact_support", ""),
        "unique_top1_topics": int(inventory["unique_top1_topics"]),
        "finite_stable": bool(training_metrics["finite_stable"]),
        "parameters": int(training_result["parameters"]),
    }


def _tomotopy_evidence(
    run: Path,
    protocol: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, object],
    dict[str, object],
    dict[str, Any],
    dict[str, Any],
]:
    """Build LDA summaries from one frozen fit and its full-spectrum mixtures."""
    validation_raw = read_json(run / "tomotopy/validation_only_result.json")
    test_raw = read_json(run / "tomotopy/test_result.json")
    validation_result = _chemical_evaluation_result(
        run,
        method="tomotopy",
        split="validation",
        protocol=protocol,
    )
    test_result = _chemical_evaluation_result(
        run,
        method="tomotopy",
        split="test",
        protocol=protocol,
    )
    validation_chemistry = _chemistry_summary(validation_result)
    test_chemistry = _chemistry_summary(test_result)
    validation_completion = validation_raw["validation"]["metrics"][
        "validation_document_completion"
    ]
    test_completion = test_raw["evaluation"]["metrics"]["test_document_completion"]
    summary = {
        "method": "tomotopy",
        "training": validation_raw["training"],
        "validation": {
            **validation_chemistry,
            "completion_nll": float(validation_completion["nll_per_token"]),
            "document_completion": validation_completion,
        },
        "test": {
            **test_chemistry,
            "completion_nll": float(test_completion["nll_per_token"]),
            "document_completion": test_completion,
        },
        "validation_access_audit": validation_raw["validation_access_audit"],
        "test_access_audit": {
            "model_sha256": test_raw["model_sha256"],
            "model_unchanged_after_evaluation": test_raw[
                "model_unchanged_after_evaluation"
            ],
            "training_or_optimization_performed": False,
        },
    }
    test_row = {
        "model": "Tomotopy LDA",
        "optimized_motifs": test_chemistry["optimized_motifs"],
        "evaluable_motifs": test_chemistry["eligible_topics"],
        "useful_motifs": test_chemistry["useful_motifs"],
        "useful_fraction_evaluable": (
            test_chemistry["useful_motifs"] / test_chemistry["eligible_topics"]
        ),
        "spectrum_topic_associations": test_chemistry["associated_spectra"],
        "mean_sos": test_chemistry["mean_sos"],
        "median_sos": test_chemistry["median_sos"],
        "completion_nll": summary["test"]["completion_nll"],
        "median_effective_topics": "",
        "mean_effective_topics": "",
        "median_exact_support": "",
        "p95_exact_support": "",
        "unique_top1_topics": "",
        "active_topics_gt_0_0005": "",
        "corpus_effective_topics": "",
        "maximum_mean_topic_usage": "",
        "mean_nearest_beta_cosine": "",
        "maximum_beta_cosine": "",
        "catastrophic_duplicate_component": "",
        "training_seconds": summary["training"]["training_seconds_total"],
        "parameters": "",
        "finite_stable": True,
        "mag_clustering_failures": test_chemistry["mag_failures"]["clustering_count"],
        "mag_optimization_failures": test_chemistry["mag_failures"][
            "optimization_count"
        ],
        "heldout_compounds_excluded_from_mag": bool(
            test_chemistry["heldout_compounds_excluded_from_mag"],
        ),
        "sos_band_accounting_valid": bool(
            test_chemistry["sos_band_accounting_valid"],
        ),
    }
    validation_row = {
        "model": "Tomotopy LDA",
        "optimized_motifs": validation_chemistry["optimized_motifs"],
        "evaluable_motifs": validation_chemistry["eligible_topics"],
        "useful_motifs": validation_chemistry["useful_motifs"],
        "useful_fraction_evaluable": (
            validation_chemistry["useful_motifs"]
            / validation_chemistry["eligible_topics"]
        ),
        "spectrum_topic_associations": validation_chemistry["associated_spectra"],
        "mean_sos": validation_chemistry["mean_sos"],
        "median_sos": validation_chemistry["median_sos"],
        "completion_nll": summary["validation"]["completion_nll"],
        "median_effective_topics": "",
        "median_exact_support": "",
        "unique_top1_topics": "",
        "finite_stable": True,
        "parameters": "",
    }
    chemistry = {"validation": validation_result, "test": test_result}
    test_raw = {**test_raw, "chemistry": test_result}
    return summary, test_row, validation_row, chemistry, test_raw


def _real_evidence(
    root: Path,
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, Any],
    list[dict[str, object]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Extract final test, development validation, and multiseed evidence."""
    paths = reproduction_paths(root)
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    chemical_results: dict[str, Any] = {
        "controls": {},
        "contextual": {},
        "tomotopy": {},
    }
    for label, method in (("canonical ETM", "etm"), ("balanced ETM", "etm_balanced")):
        model = paths.controls / "models" / method
        training_result = read_json(model / "result.json")
        _require_neural_device(
            training_result["config"].get("device"),
            label=f"{label} training",
        )
        training_metrics = training_result["metrics"]
        test_evaluation = read_json(
            paths.controls / "evaluation" / method / "complete.json",
        )
        _require_neural_device(
            test_evaluation.get("device"),
            label=f"{label} test inference",
        )
        validation_chemistry = _chemical_evaluation_result(
            paths.controls,
            method=method,
            split="validation",
            protocol=protocol,
        )
        test_chemistry = _chemical_evaluation_result(
            paths.controls,
            method=method,
            split="test",
            protocol=protocol,
        )
        chemical_results["controls"][method] = {
            "validation": validation_chemistry,
            "test": test_chemistry,
        }
        rows.append(
            _model_row(
                label,
                test_evaluation,
                test_chemistry,
                training_result,
                training_metrics,
            ),
        )
        validation_rows.append(
            _validation_model_row(
                label,
                training_result,
                training_metrics,
                validation_chemistry,
            ),
        )

    seed_rows = []
    proposed: dict[str, Any] | None = None
    for seed in TRAINING_SEEDS:
        model = paths.contextual[seed] / "models" / METHOD
        training_result = read_json(model / "result.json")
        training_metrics = training_result["metrics"]
        config = training_result["config"]
        _require_neural_device(
            config.get("device"),
            label=f"Contextual Sparse ETM seed {seed} training",
        )
        validation_chemistry = _chemical_evaluation_result(
            paths.contextual[seed],
            method=METHOD,
            split="validation",
            protocol=protocol,
        )
        if int(config["training_seed"]) != seed:
            msg = f"seed {seed} does not match its declared training seed"
            raise RuntimeError(msg)
        test_evaluation = read_json(
            paths.contextual[seed] / "evaluation" / METHOD / "complete.json",
        )
        _require_neural_device(
            test_evaluation.get("device"),
            label=f"Contextual Sparse ETM seed {seed} test inference",
        )
        test_chemistry = _chemical_evaluation_result(
            paths.contextual[seed],
            method=METHOD,
            split="test",
            protocol=protocol,
        )
        chemical_results["contextual"][seed] = {
            "validation": validation_chemistry,
            "test": test_chemistry,
        }
        row = _model_row(
            "Contextual Sparse ETM",
            test_evaluation,
            test_chemistry,
            training_result,
            training_metrics,
        )
        row.update(
            {
                "training_seed": seed,
                "learned_context_scale": float(
                    training_metrics["learned_context_scale"],
                ),
            },
        )
        seed_rows.append(row)
        if seed == TRAINING_SEEDS[0]:
            test_metrics = dict(test_evaluation["metrics"])
            test_metrics.update(
                {
                    "test_chemistry": _chemistry_summary(test_chemistry),
                    "parameters": int(training_result["parameters"]),
                    "learned_context_scale": float(
                        training_metrics["learned_context_scale"],
                    ),
                    "training_runtime": training_metrics["runtime"],
                },
            )
            proposed = {
                "config": config,
                "metrics": test_metrics,
                "validation_metrics": {
                    **training_metrics,
                    "validation_chemistry": _chemistry_summary(validation_chemistry),
                },
            }
            rows.append(row)
            validation_rows.append(
                _validation_model_row(
                    "Contextual Sparse ETM",
                    training_result,
                    training_metrics,
                    validation_chemistry,
                ),
            )
    if proposed is None:
        msg = "primary Contextual Sparse ETM seed is missing"
        raise RuntimeError(msg)

    (
        tomotopy,
        tomotopy_test_row,
        tomotopy_validation_row,
        chemical_results["tomotopy"],
        tomotopy_test_raw,
    ) = _tomotopy_evidence(paths.tomotopy, protocol)
    rows.append(tomotopy_test_row)
    validation_rows.append(tomotopy_validation_row)
    return (
        rows,
        validation_rows,
        tomotopy,
        seed_rows,
        proposed,
        chemical_results,
        tomotopy_test_raw,
    )


def _stability(
    seed_rows: Sequence[Mapping[str, object]],
    comparators: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Compute descriptive multiseed summaries and directional checks."""
    by_seed = [
        {
            "training_seed": int(row["training_seed"]),
            "optimized_motifs": int(row["optimized_motifs"]),
            "evaluable_motifs": int(row["evaluable_motifs"]),
            "useful_motifs": int(row["useful_motifs"]),
            "mean_sos": float(row["mean_sos"]),
            "median_sos": float(row["median_sos"]),
            "completion_nll": float(row["completion_nll"]),
            "median_effective_topics": float(row["median_effective_topics"]),
            "median_exact_support": float(row["median_exact_support"]),
            "unique_top1_topics": int(row["unique_top1_topics"]),
            "corpus_effective_topics": float(row["corpus_effective_topics"]),
            "learned_context_scale": float(row["learned_context_scale"]),
            "training_wall_seconds": float(row["training_seconds"]),
            "finite_stable": bool(row["finite_stable"]),
            "catastrophic_duplicate_component": bool(
                row["catastrophic_duplicate_component"],
            ),
            "mag_clustering_failures": int(row["mag_clustering_failures"]),
            "mag_optimization_failures": int(row["mag_optimization_failures"]),
            "heldout_compounds_excluded_from_mag": bool(
                row["heldout_compounds_excluded_from_mag"],
            ),
            "sos_band_accounting_valid": bool(row["sos_band_accounting_valid"]),
        }
        for row in seed_rows
    ]
    aggregate = {}
    for field in SUMMARY_FIELDS:
        values = [float(row[field]) for row in by_seed]
        aggregate[field] = {
            "mean": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
            "sample_standard_deviation": statistics.stdev(values),
        }
    other = [row for row in comparators if row["model"] != "Contextual Sparse ETM"]
    return {
        "schema_version": 2,
        "method": METHOD,
        "reported_split": "test",
        "training_seeds": list(TRAINING_SEEDS),
        "runs": len(by_seed),
        "by_seed": by_seed,
        "aggregate": aggregate,
        "direction_checks": {
            "all_finite_stable": all(row["finite_stable"] for row in by_seed),
            "no_catastrophic_duplicate_component_on_any_seed": all(
                not row["catastrophic_duplicate_component"] for row in by_seed
            ),
            "zero_mag_exceptions_on_all_seeds": all(
                row["mag_clustering_failures"] == 0
                and row["mag_optimization_failures"] == 0
                for row in by_seed
            ),
            "heldout_compounds_excluded_from_mag_on_all_seeds": all(
                row["heldout_compounds_excluded_from_mag"] for row in by_seed
            ),
            "sos_bands_account_for_evaluable_motifs_on_all_seeds": all(
                row["sos_band_accounting_valid"] for row in by_seed
            ),
            "test_released_only_after_model_freeze": True,
            "primary_seed_exceeds_every_comparator_evaluable": int(
                by_seed[0]["evaluable_motifs"],
            )
            > max(int(row["evaluable_motifs"]) for row in other),
            "primary_seed_exceeds_every_comparator_useful": int(
                by_seed[0]["useful_motifs"],
            )
            > max(int(row["useful_motifs"]) for row in other),
        },
    }


def _exact_data_checks(
    preparation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply immutable data and configuration gates."""
    data = preparation["data"]
    actual = {
        "source_spectra": int(data["parsing"]["parsed_blocks"]),
        "retained_spectra": int(data["parsing"]["retained_spectra"]),
        "connectivity_groups": int(data["leakage_audit"]["connectivity_groups"]),
        "split_groups": int(data["leakage_audit"]["split_groups"]),
        "train_spectra": int(data["split"]["spectrum_counts"]["train"]),
        "validation_spectra": int(data["split"]["spectrum_counts"]["validation"]),
        "test_spectra": int(data["split"]["spectrum_counts"]["test"]),
        "vocabulary_size": int(data["vocabulary"]["vocabulary_size"]),
        "leaked_compounds": int(data["leakage_audit"]["leaked_compounds"]),
        "leaked_split_groups": int(data["leakage_audit"]["leaked_groups"]),
        "topics": int(protocol["model"]["num_topics"]),
    }
    expected = {
        "source_spectra": 41_568,
        "retained_spectra": 38_888,
        "connectivity_groups": 38_465,
        "split_groups": 28_572,
        "train_spectra": 27_222,
        "validation_spectra": 3_889,
        "test_spectra": 7_777,
        "vocabulary_size": 21_233,
        "leaked_compounds": 0,
        "leaked_split_groups": 0,
        "topics": 1000,
    }
    checks = {
        key: {
            "expected": expected[key],
            "actual": actual[key],
            "passed": actual[key] == expected[key],
        }
        for key in expected
    }
    return {
        "all_passed": all(row["passed"] for row in checks.values()),
        "checks": checks,
    }


def _claim_checks(
    comparison: Sequence[Mapping[str, object]],
    stability: Mapping[str, Any],
    high_k: Sequence[Mapping[str, object]],
    expected_test_spectra: int,
) -> dict[str, Any]:
    """Evaluate the report's directional claims."""
    models = {str(row["model"]): row for row in comparison}
    proposed = models["Contextual Sparse ETM"]
    controls = [models["canonical ETM"], models["balanced ETM"]]
    high_k_proposed = next(
        row for row in high_k if row["formulation"] == FINAL_SYNTHETIC_LABEL
    )
    checks = {
        "proposed_has_most_evaluable_motifs": int(proposed["evaluable_motifs"])
        > max(
            int(row["evaluable_motifs"])
            for name, row in models.items()
            if name != "Contextual Sparse ETM"
        ),
        "proposed_has_most_useful_motifs": int(proposed["useful_motifs"])
        > max(
            int(row["useful_motifs"])
            for name, row in models.items()
            if name != "Contextual Sparse ETM"
        ),
        "all_models_assign_every_test_spectrum_once": all(
            int(row["spectrum_topic_associations"]) == expected_test_spectra
            for row in models.values()
        ),
        "dense_etm_controls_have_lower_completion_nll": all(
            float(row["completion_nll"]) < float(proposed["completion_nll"])
            for row in controls
        ),
        "proposed_median_effective_topics_at_most_five": float(
            proposed["median_effective_topics"],
        )
        <= MAXIMUM_EFFECTIVE_TOPICS,
        "proposed_unique_winners_at_least_800": int(proposed["unique_top1_topics"])
        >= MINIMUM_UNIQUE_WINNERS,
        "all_three_seeds_avoid_catastrophic_duplicates": stability["direction_checks"][
            "no_catastrophic_duplicate_component_on_any_seed"
        ],
        "all_three_seeds_have_zero_mag_exceptions": stability["direction_checks"][
            "zero_mag_exceptions_on_all_seeds"
        ],
        "high_k_recovers_all_18_planted_motifs": int(
            high_k_proposed["planted_motifs_recovered_cosine_ge_0_50"],
        )
        == int(high_k_proposed["true_topics"])
        == PLANTED_SYNTHETIC_TOPICS,
        "high_k_median_support_at_most_three": float(
            high_k_proposed["median_exact_support"],
        )
        <= MAXIMUM_HIGH_K_SUPPORT,
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def _chemical_integrity_checks(
    comparison: Sequence[Mapping[str, object]],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply leakage, SOS-accounting, and MAG-exception gates to every fit."""
    checks = {
        "comparison_models_have_zero_mag_exceptions": all(
            int(row["mag_clustering_failures"]) == 0
            and int(row["mag_optimization_failures"]) == 0
            for row in comparison
        ),
        "all_contextual_seeds_have_zero_mag_exceptions": bool(
            stability["direction_checks"]["zero_mag_exceptions_on_all_seeds"],
        ),
        "comparison_models_exclude_heldout_compounds_from_mag": all(
            bool(row["heldout_compounds_excluded_from_mag"]) for row in comparison
        ),
        "all_contextual_seeds_exclude_heldout_compounds_from_mag": bool(
            stability["direction_checks"][
                "heldout_compounds_excluded_from_mag_on_all_seeds"
            ],
        ),
        "comparison_model_sos_bands_account_for_evaluable_motifs": all(
            bool(row["sos_band_accounting_valid"]) for row in comparison
        ),
        "all_contextual_seed_sos_bands_account_for_evaluable_motifs": bool(
            stability["direction_checks"][
                "sos_bands_account_for_evaluable_motifs_on_all_seeds"
            ],
        ),
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def _build_package(
    raw_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build a verified package in a new staging directory."""
    manifest, stage_records = verify_stage_records(raw_root)
    _require_neural_device(
        manifest.get("neural_execution_device"),
        label="reproduction manifest",
    )
    validation_views = validate_model_views(raw_root)
    probability = probability_audit(raw_root)
    paths = reproduction_paths(raw_root)
    preparation = read_json(paths.prepared / "preparation_summary.json")
    protocol = load_protocol()
    data_checks = _exact_data_checks(preparation, protocol)
    if not data_checks["all_passed"]:
        msg = "immutable data/configuration checks failed"
        raise RuntimeError(msg)

    primary, synthetic_summary, high_k = _synthetic_tables(raw_root)
    (
        comparison,
        validation_comparison,
        tomotopy,
        seed_rows,
        proposed,
        chemical_results,
        tomotopy_test_raw,
    ) = _real_evidence(raw_root, protocol)
    stability = _stability(seed_rows, comparison)
    claims = _claim_checks(
        comparison,
        stability,
        high_k,
        int(preparation["data"]["split"]["spectrum_counts"]["test"]),
    )
    chemical_integrity = _chemical_integrity_checks(comparison, stability)
    chemical_integrity_passed = chemical_integrity["all_passed"]
    data_quality = {
        "status": "pass" if chemical_integrity_passed else "fail",
        "exact_data_checks": data_checks,
        "validation_views": validation_views,
        "probability_matrices": probability,
        "chemical_integrity": chemical_integrity,
    }
    if not chemical_integrity_passed:
        msg = "chemical integrity checks make the clean reproduction incomplete"
        raise RuntimeError(msg)

    evidence = {
        "preparation": preparation,
        "protocol": protocol,
        "proposed": proposed,
        "tomotopy": tomotopy,
        "stability": stability,
        "claims": claims,
        "data_quality": data_quality,
        "comparison": comparison,
        "validation_comparison": validation_comparison,
        "primary": primary,
        "synthetic_summary": synthetic_summary,
        "high_k": high_k,
        "chemical_results": chemical_results,
        "tomotopy_test_raw": tomotopy_test_raw,
    }
    destination.mkdir(parents=True)
    _write_summary_artifacts(destination, evidence)
    _copy_raw_evidence(
        paths,
        destination,
        manifest=manifest,
        claims=claims,
        chemical_results=chemical_results,
        tomotopy_test_raw=tomotopy_test_raw,
    )
    replacements = _path_replacements(paths, manifest)
    _rewrite_json_as_portable(destination, replacements)
    _write_package_seals(
        destination,
        manifest=manifest,
        stage_records=stage_records,
        claims=claims,
        data_quality=data_quality,
        replacements=replacements,
    )
    _rewrite_json_as_portable(destination, replacements)
    _assert_no_machine_paths(destination)
    return {
        "status": "packaged",
        "output": str(destination),
        "reproduction_id": manifest["reproduction_id"],
        "claim_checks_passed": claims["all_passed"],
        "data_quality": data_quality["status"],
        "reported_split": "test",
    }


def package_reproduction(root: Path, output: Path) -> dict[str, Any]:
    """Verify and atomically package one complete clean-room reproduction."""
    raw_root = root.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve()
    if destination.exists():
        msg = f"package output already exists: {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        msg = f"package staging directory already exists: {staging}"
        raise FileExistsError(msg)
    try:
        result = _build_package(raw_root, staging)
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    result["output"] = str(destination)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Package a completed reproduction into a compact report input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = package_reproduction(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
