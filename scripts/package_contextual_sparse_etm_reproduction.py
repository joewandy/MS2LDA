"""Package one sealed Contextual Sparse ETM clean-room reproduction.

Only artifacts owned by the supplied reproduction UUID are accepted.  The
packager verifies stage hashes, split-release ordering, probability matrices,
MAG failure counts, and predeclared scientific claims before writing the
compact evidence bundle consumed by the LaTeX report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.reproduction_audit import (
    file_record,
    probability_audit,
    read_json,
    sha256_file,
    validate_model_views,
    verify_stage_records,
    write_csv,
    write_json,
)
from benchmarks.neural_ms2lda.reproduction_plan import (
    METHOD,
    SYNTHETIC_SEEDS,
    TRAINING_SEEDS,
    ReproductionPaths,
    reproduction_paths,
    stage_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FORMULATION_LABELS = {
    "balanced_etm_softmax_raw_counts": "balanced ETM softmax raw",
    "balanced_etm_entmax15_raw_counts": "balanced ETM plus entmax15",
    "balanced_etm_routing_top2_context_raw_counts": (
        "balanced ETM plus top-2-context routing and softmax"
    ),
    "balanced_etm_routing_top2_context_entmax15_raw_counts": (
        "balanced ETM plus top-2-context routing and entmax15"
    ),
}
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
COMPACT_MODEL_FILES = (
    "result.json",
    "training_history.csv",
    "theta_support_summary.csv",
    "routing_evidence_support_summary.csv",
    "duplicate_component_summary.json",
    "fragment_mass_summary.json",
    "top_words.csv",
    "validation_access_audit.json",
    "provenance.json",
)
COMPACT_CONTROL_FILES = (
    "result.json",
    "training_history.csv",
    "duplicate_component_summary.json",
    "fragment_mass_summary.json",
    "top_words.csv",
    "validation_access_audit.json",
)
COMPACT_TEST_EVALUATION_FILES = ("complete.json", "test_access_audit.json")


def _chemistry_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize shared MAG/SOS output into the paper's summary schema."""
    if "mag_failures" not in result:
        msg = "fresh chemical evidence lacks explicit MAG exception accounting"
        raise RuntimeError(msg)
    if result.get("heldout_compounds_excluded_from_mag") is not True:
        msg = "fresh chemical evidence does not exclude held-out compounds from MAG"
        raise RuntimeError(msg)
    summary = dict(result["high_confidence_chemistry"])
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
    for kind in ("clustering", "optimization"):
        count = int(failures[f"{kind}_count"])
        topic_ids = failures[f"{kind}_topic_ids"]
        if count < 0 or count != len(topic_ids):
            msg = f"MAG {kind} exception count and topic IDs disagree"
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


def _synthetic_row(result: Mapping[str, Any], *, stage: str) -> dict[str, object]:
    """Extract one truth-known result row without rounding away evidence."""
    config = result["config"]
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


def _synthetic_tables(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read exactly the predeclared 12 K=36 and three K=128 fits."""
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
                    if "top-2-context routing and entmax15" in formulation
                    else "ablation/control"
                ),
            },
        )
    for row in high_k:
        row["fitted_topics"] = row.pop("k")
        row["decision"] = (
            "promote to real validation"
            if "top-2-context routing and entmax15" in str(row["formulation"])
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


def _real_evidence(
    root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, Any],
    list[dict[str, object]],
    dict[str, Any],
]:
    """Extract final test, development validation, and multiseed evidence."""
    paths = reproduction_paths(root)
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for label, method in (("canonical ETM", "etm"), ("balanced ETM", "etm_balanced")):
        model = paths.controls / "models" / method
        training_result = read_json(model / "result.json")
        training_metrics = training_result["metrics"]
        validation_chemistry = read_json(
            paths.controls / "validation_chemical" / method / "complete.json",
        )
        rows.append(
            _model_row(
                label,
                read_json(paths.controls / "evaluation" / method / "complete.json"),
                read_json(paths.controls / "chemical" / method / "complete.json"),
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
        validation_chemistry = read_json(
            paths.contextual[seed] / "validation_chemical" / METHOD / "complete.json",
        )
        if (
            int(config["training_seed"]) != seed
            or int(config["resumed_from_epoch"]) != 0
        ):
            msg = f"seed {seed} was not a fresh epoch-zero training run"
            raise RuntimeError(msg)
        test_evaluation = read_json(
            paths.contextual[seed] / "evaluation" / METHOD / "complete.json",
        )
        row = _model_row(
            "Contextual Sparse ETM",
            test_evaluation,
            read_json(paths.contextual[seed] / "chemical" / METHOD / "complete.json"),
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
                    "test_chemistry": _chemistry_summary(
                        read_json(
                            paths.contextual[seed]
                            / "chemical"
                            / METHOD
                            / "complete.json",
                        ),
                    ),
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

    tomotopy_validation_raw = read_json(
        paths.tomotopy / "tomotopy/validation_only_result.json",
    )
    tomotopy_test_raw = read_json(paths.tomotopy / "tomotopy/test_result.json")
    tomotopy_validation_chemistry = _chemistry_summary(
        read_json(paths.tomotopy / "validation_chemical/tomotopy/complete.json"),
    )
    tomotopy_validation_completion = tomotopy_validation_raw["validation"]["metrics"][
        "validation_document_completion"
    ]
    tomotopy_test_chemistry = _chemistry_summary(tomotopy_test_raw["chemistry"])
    tomotopy_test_completion = tomotopy_test_raw["evaluation"]["metrics"][
        "test_document_completion"
    ]
    tomotopy = {
        "method": "tomotopy",
        "training": tomotopy_validation_raw["training"],
        "validation": {
            **tomotopy_validation_chemistry,
            "high_confidence_evaluable_motifs": tomotopy_validation_chemistry[
                "eligible_topics"
            ],
            "useful_high_confidence_motifs": tomotopy_validation_chemistry[
                "useful_motifs"
            ],
            "completion_nll": float(tomotopy_validation_completion["nll_per_token"]),
            "document_completion": tomotopy_validation_completion,
        },
        "test": {
            **tomotopy_test_chemistry,
            "high_confidence_evaluable_motifs": tomotopy_test_chemistry[
                "eligible_topics"
            ],
            "useful_high_confidence_motifs": tomotopy_test_chemistry["useful_motifs"],
            "completion_nll": float(tomotopy_test_completion["nll_per_token"]),
            "document_completion": tomotopy_test_completion,
        },
        "validation_access_audit": tomotopy_validation_raw["validation_access_audit"],
        "test_access_audit": {
            "model_sha256": tomotopy_test_raw["model_sha256"],
            "model_unchanged_after_evaluation": tomotopy_test_raw[
                "model_unchanged_after_evaluation"
            ],
            "training_or_optimization_performed": False,
        },
    }
    rows.append(
        {
            "model": "Tomotopy LDA",
            "optimized_motifs": tomotopy_test_chemistry["optimized_motifs"],
            "evaluable_motifs": tomotopy_test_chemistry["eligible_topics"],
            "useful_motifs": tomotopy_test_chemistry["useful_motifs"],
            "mean_sos": tomotopy_test_chemistry["mean_sos"],
            "median_sos": tomotopy_test_chemistry["median_sos"],
            "completion_nll": tomotopy["test"]["completion_nll"],
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
            "training_seconds": tomotopy["training"]["training_seconds_total"],
            "parameters": "",
            "finite_stable": True,
            "mag_clustering_failures": tomotopy_test_chemistry["mag_failures"][
                "clustering_count"
            ],
            "mag_optimization_failures": tomotopy_test_chemistry["mag_failures"][
                "optimization_count"
            ],
            "heldout_compounds_excluded_from_mag": bool(
                tomotopy_test_chemistry["heldout_compounds_excluded_from_mag"],
            ),
            "sos_band_accounting_valid": bool(
                tomotopy_test_chemistry["sos_band_accounting_valid"],
            ),
        },
    )
    validation_rows.append(
        {
            "model": "Tomotopy LDA",
            "optimized_motifs": tomotopy_validation_chemistry["optimized_motifs"],
            "evaluable_motifs": tomotopy_validation_chemistry["eligible_topics"],
            "useful_motifs": tomotopy_validation_chemistry["useful_motifs"],
            "mean_sos": tomotopy_validation_chemistry["mean_sos"],
            "median_sos": tomotopy_validation_chemistry["median_sos"],
            "completion_nll": tomotopy["validation"]["completion_nll"],
            "median_effective_topics": "",
            "median_exact_support": "",
            "unique_top1_topics": "",
            "finite_stable": True,
            "parameters": "",
        },
    )
    return rows, validation_rows, tomotopy, seed_rows, proposed


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
    """Apply predeclared immutable data and configuration gates."""
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
) -> dict[str, Any]:
    """Evaluate the directional claims frozen before results were opened."""
    models = {str(row["model"]): row for row in comparison}
    proposed = models["Contextual Sparse ETM"]
    tomotopy = models["Tomotopy LDA"]
    controls = [models["canonical ETM"], models["balanced ETM"]]
    high_k_proposed = next(
        row
        for row in high_k
        if "top-2-context routing and entmax15" in str(row["formulation"])
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
        "tomotopy_has_higher_conditional_mean_sos": float(tomotopy["mean_sos"])
        > float(proposed["mean_sos"]),
        "tomotopy_has_higher_conditional_median_sos": float(tomotopy["median_sos"])
        > float(proposed["median_sos"]),
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
        "high_k_recovers_all_18_planted_winners": int(
            high_k_proposed["unique_top1_topics"],
        )
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


def _copy_compact(source: Path, destination: Path, names: Sequence[str]) -> None:
    """Copy required compact files and fail on omissions."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if not path.is_file():
            msg = f"missing compact evidence file: {path}"
            raise FileNotFoundError(msg)
        target = destination / name
        shutil.copy2(path, target)


def _readme(
    manifest: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> str:
    """Return the human-readable clean-room handoff bundled with the evidence."""
    return f"""# Contextual Sparse ETM clean-room reproduction

This bundle was generated from reproduction `{manifest['reproduction_id']}` at
source commit `{manifest['source']['commit']}`. Models were fitted on training
spectra, selected and ablated on validation spectra, frozen, and then evaluated
on the fixed test split. `validation_comparison.csv` records development-split
evidence; `comparison.csv` and `stability_by_seed.csv` contain final test results.

## Acceptance status

Predeclared directional claims passed: **{claims['all_passed']}**. Inspect
`acceptance.json`, `data_quality.json`, `fresh_evidence_manifest.json`, and the
CSV/JSON result tables for the complete evidence trail.
"""


def _copy_model_evidence(
    run: Path,
    destination: Path,
    *,
    method: str,
    model_files: Sequence[str],
) -> None:
    """Copy compact validation, frozen-test, and split-boundary evidence."""
    _copy_compact(run / "models" / method, destination, model_files)
    _copy_compact(
        run / "evaluation" / method,
        destination / "test_evaluation",
        COMPACT_TEST_EVALUATION_FILES,
    )
    shutil.copy2(
        run / "chemical" / method / "complete.json",
        destination / "test_chemical.json",
    )
    shutil.copy2(
        run / "validation_chemical" / method / "complete.json",
        destination / "validation_chemical.json",
    )
    for name in ("validation_input_manifest.json", "test_input_manifest.json"):
        shutil.copy2(run / name, destination / name)


def _write_summary_artifacts(
    destination: Path,
    evidence: Mapping[str, Any],
) -> None:
    """Write compact machine-readable summaries and tables."""
    json_outputs = {
        "preparation_summary.json": evidence["preparation"],
        "protocol.json": evidence["protocol"],
        "config.json": evidence["proposed"]["config"],
        "metrics.json": evidence["proposed"]["metrics"],
        "validation_metrics.json": evidence["proposed"]["validation_metrics"],
        "tomotopy.json": evidence["tomotopy"],
        "stability_summary.json": evidence["stability"],
        "acceptance.json": evidence["claims"],
        "data_quality.json": evidence["data_quality"],
    }
    csv_outputs = {
        "comparison.csv": evidence["comparison"],
        "validation_comparison.csv": evidence["validation_comparison"],
        "synthetic_by_seed.csv": evidence["primary"],
        "synthetic_summary.csv": evidence["synthetic_summary"],
        "high_k_stress.csv": evidence["high_k"],
        "stability_by_seed.csv": evidence["stability"]["by_seed"],
    }
    for name, value in json_outputs.items():
        write_json(destination / name, value)
    for name, rows in csv_outputs.items():
        write_csv(destination / name, rows)


def _copy_raw_evidence(
    paths: ReproductionPaths,
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> None:
    """Copy compact model, split-boundary, and stage-provenance artifacts."""
    for seed in TRAINING_SEEDS:
        _copy_model_evidence(
            paths.contextual[seed],
            destination / "contextual" / f"seed_{seed}",
            method=METHOD,
            model_files=COMPACT_MODEL_FILES,
        )
    for method in ("etm", "etm_balanced"):
        _copy_model_evidence(
            paths.controls,
            destination / "controls" / method,
            method=method,
            model_files=COMPACT_CONTROL_FILES,
        )
    for result_path in sorted(
        (paths.synthetic / "synthetic_runs").glob("*/result.json"),
    ):
        target = (
            destination / "synthetic_results" / result_path.parent.name / "result.json"
        )
        target.parent.mkdir(parents=True)
        shutil.copy2(result_path, target)
    for source, name in (
        (paths.tomotopy / "tomotopy/validation_only_result.json", "tomotopy_raw.json"),
        (paths.tomotopy / "tomotopy/test_result.json", "tomotopy_test_raw.json"),
        (paths.assets / "acquisition_manifest.json", "acquisition_manifest.json"),
        (paths.root / "reproduction_manifest.json", "reproduction_manifest.json"),
    ):
        shutil.copy2(source, destination / name)
    stage_directory = destination / "stage_records"
    stage_directory.mkdir()
    for stage in stage_plan(paths):
        shutil.copy2(
            paths.stages / f"{stage.name}.json",
            stage_directory / f"{stage.name}.json",
        )
    tomotopy_boundaries = destination / "tomotopy"
    tomotopy_boundaries.mkdir()
    for name in ("validation_input_manifest.json", "test_input_manifest.json"):
        shutil.copy2(paths.tomotopy / name, tomotopy_boundaries / name)
    (destination / "README.md").write_text(
        _readme(manifest, claims),
        encoding="utf-8",
    )


def _write_package_seals(
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    claims: Mapping[str, Any],
    data_quality: Mapping[str, Any],
) -> None:
    """Seal all compact files and write the report-facing checkpoint."""
    raw_outputs = [
        output_row for stage in stage_records for output_row in stage.get("outputs", [])
    ]
    seal = {
        "schema_version": 1,
        "reproduction_id": manifest["reproduction_id"],
        "source": manifest["source"],
        "split_protocol": (
            "fit on train; select and ablate on validation; evaluate frozen models "
            "on test"
        ),
        "method": METHOD,
        "training_seeds": list(TRAINING_SEEDS),
        "synthetic_seeds": list(SYNTHETIC_SEEDS),
        "stage_count": len(stage_records),
        "raw_stage_outputs": raw_outputs,
        "packaged_files": [
            file_record(path, relative_to=destination)
            for path in sorted(
                path for path in destination.rglob("*") if path.is_file()
            )
        ],
    }
    write_json(destination / "fresh_evidence_manifest.json", seal)
    checkpoint = {
        "schema_version": 2,
        "method": METHOD,
        "split_protocol": seal["split_protocol"],
        "reproduction_id": manifest["reproduction_id"],
        "source_commit": manifest["source"]["commit"],
        "test_released_after_model_and_validation_freeze": True,
        "fresh_evidence_manifest_sha256": sha256_file(
            destination / "fresh_evidence_manifest.json",
        ),
        "acceptance_all_passed": claims["all_passed"],
        "data_quality_status": data_quality["status"],
    }
    write_json(destination / "checkpoint_manifest.json", checkpoint)


def _build_package(
    raw_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build a verified package in a new staging directory."""
    manifest, stage_records = verify_stage_records(raw_root)
    validation_views = validate_model_views(raw_root)
    probability = probability_audit(raw_root)
    paths = reproduction_paths(raw_root)
    preparation = read_json(paths.prepared / "comparison_preparation.json")
    protocol = read_json(paths.prepared / "protocol.json")
    data_checks = _exact_data_checks(preparation, protocol)
    if not data_checks["all_passed"]:
        msg = "immutable data/configuration checks failed"
        raise RuntimeError(msg)

    primary, synthetic_summary, high_k = _synthetic_tables(raw_root)
    comparison, validation_comparison, tomotopy, seed_rows, proposed = _real_evidence(
        raw_root,
    )
    stability = _stability(seed_rows, comparison)
    claims = _claim_checks(comparison, stability, high_k)
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
    }
    destination.mkdir(parents=True)
    _write_summary_artifacts(destination, evidence)
    _copy_raw_evidence(
        paths,
        destination,
        manifest=manifest,
        claims=claims,
    )
    _write_package_seals(
        destination,
        manifest=manifest,
        stage_records=stage_records,
        claims=claims,
        data_quality=data_quality,
    )
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
