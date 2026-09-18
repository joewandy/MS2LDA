"""Historical clean-room aggregation and acceptance criteria.

These reproduce the archived study, not selection gates for the current model.
Scientific summaries live here; packaging and filesystem publication do not.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .reproduction_audit import read_json
from .reproduction_plan import reproduction_paths
from .study_protocol import (
    FINAL_SYNTHETIC_LABEL,
    METHOD,
    NEURAL_DEVICE,
    SYNTHETIC_ARTIFACT_LABELS,
    SYNTHETIC_DISPLAY_LABELS,
    SYNTHETIC_SEEDS,
    TRAINING_SEEDS,
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


def synthetic_row(result: Mapping[str, Any], *, stage: str) -> dict[str, object]:
    """Extract one truth-known result row without rounding away evidence."""
    config = result["config"]
    require_neural_device(config.get("device"), label=f"synthetic stage {stage}")
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


def require_neural_device(value: object, *, label: str) -> None:
    """Reject evidence not executed on the reproduction's required device."""
    if str(value).split(":", maxsplit=1)[0] != NEURAL_DEVICE:
        msg = f"{label} was not executed on {NEURAL_DEVICE}: {value}"
        raise RuntimeError(msg)


def synthetic_tables(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read exactly the study's 12 K=36 and three K=128 fits."""
    synthetic_root = reproduction_paths(root).synthetic / "synthetic_runs"
    rows = []
    for result_path in sorted(synthetic_root.glob("*/result.json")):
        result = read_json(result_path)
        topics = int(result["config"]["fitted_topics"])
        rows.append(
            synthetic_row(
                result,
                stage=(
                    "multi_seed" if topics == PRIMARY_SYNTHETIC_TOPICS else "high_k"
                ),
            ),
        )
    primary = [row for row in rows if row["k"] == PRIMARY_SYNTHETIC_TOPICS]
    high_k = [row for row in rows if row["k"] == HIGH_K_SYNTHETIC_TOPICS]
    expected = Counter(
        (PRIMARY_SYNTHETIC_TOPICS, seed, formulation)
        for seed in SYNTHETIC_SEEDS
        for formulation in FORMULATION_LABELS.values()
    )
    expected.update(
        (
            HIGH_K_SYNTHETIC_TOPICS,
            SYNTHETIC_SEEDS[0],
            FORMULATION_LABELS[SYNTHETIC_ARTIFACT_LABELS[name]],
        )
        for name in ("balanced_softmax", "balanced_entmax", "contextual_entmax")
    )
    actual = Counter((row["k"], row["seed"], row["formulation"]) for row in rows)
    if actual != expected:
        raise RuntimeError(
            "synthetic fits differ from the frozen seed/formulation plan"
        )
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


def stability(
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


def exact_data_checks(
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


def claim_checks(
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


def chemical_integrity_checks(
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
