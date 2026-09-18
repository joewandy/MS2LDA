"""Model-neutral chemical and predictive comparisons for sealed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .chemical import score_precomputed_annotations
from .data import load_heldout_records
from .reproduction_audit import read_json, sha256_file
from .reproduction_plan import reproduction_paths
from .reproduction_summaries import require_neural_device
from .study_protocol import METHOD, TRAINING_SEEDS

if TYPE_CHECKING:
    from collections.abc import Mapping


def chemistry_summary(result: Mapping[str, Any]) -> dict[str, Any]:
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


def chemical_evaluation_result(
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


def chemistry_fields(label: str, chemistry: Mapping[str, Any]) -> dict[str, object]:
    """Keep the shared LDA/neural comparison schema and conversions in one place."""
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
    }


def model_row(
    label: str,
    evaluation: Mapping[str, Any],
    chemistry_result: Mapping[str, Any],
    training_result: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
) -> dict[str, object]:
    """Extract one frozen-model test comparison row."""
    metrics = evaluation["metrics"]
    chemistry = chemistry_summary(chemistry_result)
    inventory = metrics["topic_inventory"]
    support = metrics.get("theta_support", {})
    return {
        **chemistry_fields(label, chemistry),
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


def validation_model_row(
    label: str,
    training_result: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    chemistry_result: Mapping[str, Any],
) -> dict[str, object]:
    """Extract the development-split row retained beside final test evidence."""
    chemistry = chemistry_summary(chemistry_result)
    inventory = training_metrics["topic_inventory"]
    support = training_metrics.get("theta_support", {})
    return {
        **chemistry_fields(label, chemistry),
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


def tomotopy_evidence(
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
    validation_result = chemical_evaluation_result(
        run,
        method="tomotopy",
        split="validation",
        protocol=protocol,
    )
    test_result = chemical_evaluation_result(
        run,
        method="tomotopy",
        split="test",
        protocol=protocol,
    )
    validation_chemistry = chemistry_summary(validation_result)
    test_chemistry = chemistry_summary(test_result)
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
        **chemistry_fields("Tomotopy LDA", test_chemistry),
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
        **chemistry_fields("Tomotopy LDA", validation_chemistry),
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


def real_evidence(
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
        require_neural_device(
            training_result["config"].get("device"),
            label=f"{label} training",
        )
        training_metrics = training_result["metrics"]
        test_evaluation = read_json(
            paths.controls / "evaluation" / method / "complete.json",
        )
        require_neural_device(
            test_evaluation.get("device"),
            label=f"{label} test inference",
        )
        validation_chemistry = chemical_evaluation_result(
            paths.controls,
            method=method,
            split="validation",
            protocol=protocol,
        )
        test_chemistry = chemical_evaluation_result(
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
            model_row(
                label,
                test_evaluation,
                test_chemistry,
                training_result,
                training_metrics,
            ),
        )
        validation_rows.append(
            validation_model_row(
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
        require_neural_device(
            config.get("device"),
            label=f"Contextual Sparse ETM seed {seed} training",
        )
        validation_chemistry = chemical_evaluation_result(
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
        require_neural_device(
            test_evaluation.get("device"),
            label=f"Contextual Sparse ETM seed {seed} test inference",
        )
        test_chemistry = chemical_evaluation_result(
            paths.contextual[seed],
            method=METHOD,
            split="test",
            protocol=protocol,
        )
        chemical_results["contextual"][seed] = {
            "validation": validation_chemistry,
            "test": test_chemistry,
        }
        row = model_row(
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
                    "test_chemistry": chemistry_summary(test_chemistry),
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
                    "validation_chemistry": chemistry_summary(validation_chemistry),
                },
            }
            rows.append(row)
            validation_rows.append(
                validation_model_row(
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
    ) = tomotopy_evidence(paths.tomotopy, protocol)
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
