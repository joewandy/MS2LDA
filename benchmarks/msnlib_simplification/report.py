"""Neutral result bundle, grouped uncertainty, and completeness verification."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.msnlib_validation.config import (
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)

from .chemical import score_all_chemical_results
from .data import heldout_metadata
from .evaluation import (
    finalize_test,
    metric_rows,
    topic_matching,
)
from .spec import (
    ARM_IDS,
    BUDGETS,
    load_spec,
    verify_archived_study,
    verify_study,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        msg = "cannot write an empty result table"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_bootstrap(
    values: np.ndarray,
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(groups),):
        msg = "bootstrap values and groups are not aligned"
        raise ValueError(msg)
    unique = sorted(set(groups))
    group_index = {value: index for index, value in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for value, group in zip(array, groups, strict=True):
        if np.isfinite(value):
            index = group_index[group]
            sums[index] += value
            counts[index] += 1
    finite = np.isfinite(array)
    estimate = float(np.mean(array[finite])) if np.any(finite) else None
    if not np.any(counts):
        return {
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
            "groups": len(unique),
            "observations": int(np.sum(finite)),
            "replicates": replicates,
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        denominator = counts[selected].sum()
        draws[replicate] = sums[selected].sum() / denominator if denominator else np.nan
    return {
        "estimate": estimate,
        "ci_low": float(np.nanpercentile(draws, 2.5)),
        "ci_high": float(np.nanpercentile(draws, 97.5)),
        "groups": len(unique),
        "observations": int(np.sum(finite)),
        "replicates": replicates,
    }


def build_bootstrap(run_dir: str | Path) -> dict[str, Any]:
    """Compute fixed-seed scaffold-group bootstrap summaries."""
    directory = Path(run_dir).expanduser().resolve()
    spec = load_spec(directory)
    output = directory / "report" / "bootstrap.jsonl"
    complete_path = directory / "report" / "bootstrap_complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(output) != result["rows_sha256"]:
            msg = "bootstrap rows changed"
            raise ValueError(msg)
        return result
    rows: list[dict[str, Any]] = []
    baseline = "dreams_prior__dreams_semi"
    for split_index, split in enumerate(("validation", "test")):
        metadata = heldout_metadata(directory, split)
        groups = [str(row["scaffold_key"]) for row in metadata]
        baseline_arrays = {
            budget: np.load(
                directory
                / "evaluation"
                / split
                / "observed"
                / "arms"
                / baseline
                / f"per_document_{budget}.npz",
            )
            for budget in BUDGETS
        }
        for _arm_index, arm_id in enumerate(ARM_IDS):
            for budget in BUDGETS:
                values = np.load(
                    directory
                    / "evaluation"
                    / split
                    / "observed"
                    / "arms"
                    / arm_id
                    / f"per_document_{budget}.npz",
                )
                for metric in (
                    "nll_per_token",
                    "cosine_to_reference",
                    "js_to_reference",
                ):
                    seed = spec.seed + split_index * 100_000 + budget * 1_000
                    summary = _group_bootstrap(
                        values[metric],
                        groups,
                        replicates=spec.bootstrap_replicates,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "split": split,
                            "arm_id": arm_id,
                            "budget": budget,
                            "metric": metric,
                            "contrast": "absolute",
                            **summary,
                        },
                    )
                    delta = values[metric] - baseline_arrays[budget][metric]
                    delta_summary = _group_bootstrap(
                        delta,
                        groups,
                        replicates=spec.bootstrap_replicates,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "split": split,
                            "arm_id": arm_id,
                            "budget": budget,
                            "metric": metric,
                            "contrast": f"paired_delta_vs_{baseline}",
                            **delta_summary,
                        },
                    )
    _write_jsonl(output, rows)
    result = {
        "schema_version": "msnlib-simplification/bootstrap-v1",
        "rows": len(rows),
        "replicates": spec.bootstrap_replicates,
        "resampling_unit": "structural_scaffold_group",
        "same_draws_used_for_paired_contrasts": True,
        "chemical_compound_scores_available_for_later_grouped_analysis": True,
        "rows_sha256": file_sha256(output),
    }
    write_json(complete_path, result)
    return result


def build_report(run_dir: str | Path) -> dict[str, Any]:
    """Build factual tables and stop before interpretation or model selection."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    finalize_test(directory)
    score_all_chemical_results(directory)
    matching = topic_matching(directory)
    bootstrap = build_bootstrap(directory)
    output = directory / "report"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"report artifact changed: {name}"
                raise ValueError(msg)
        return result
    core_rows = metric_rows(directory)
    chemical_rows = _read_jsonl(directory / "chemical/scores/summaries.jsonl")
    metrics_path = output / "metrics.csv"
    chemical_path = output / "chemical_metrics.csv"
    summary_path = output / "collection_summary.json"
    readme_path = output / "README.md"
    _write_csv(metrics_path, core_rows)
    _write_csv(chemical_path, chemical_rows)
    source = Path(lock["source_run"])
    source_hybrid = read_json(source / "core/seed_42/hybrid/complete.json")
    source_tomotopy = read_json(source / "core/seed_42/tomotopy/complete.json")
    summary = {
        "schema_version": "msnlib-simplification/collection-summary-v1",
        "scientific_status": "collection_complete_interpretation_deferred",
        "selection_or_adoption_performed": False,
        "seed": 42,
        "topic_count": 1000,
        "arms": list(ARM_IDS),
        "budgets": list(BUDGETS),
        "validation_primary_then_single_test_pass": True,
        "metrics_rows": len(core_rows),
        "chemical_rows": len(chemical_rows),
        "bootstrap": bootstrap,
        "topic_matching": matching,
        "prespecified_preservation_margins": {
            "sos_absolute": load_spec(directory).sos_margin,
            "coverage_fraction": load_spec(directory).coverage_fraction,
            "nll_relative": load_spec(directory).nll_relative_margin,
            "npmi_absolute": load_spec(directory).npmi_margin,
            "encoder_p05_cosine_absolute": load_spec(directory).cosine_p05_margin,
        },
        "current_hybrid_reference": source_hybrid,
        "tomotopy_reference": source_tomotopy,
        "claim_boundary": {
            "single_seed_posthoc": True,
            "chemical_evidence_is_indirect": True,
            "peak_to_fragment_ground_truth_available": False,
            "model_replacement_authorized": False,
            "winner_declared": False,
        },
    }
    write_json(summary_path, summary)
    readme_path.write_text(
        "# HybridLDA simplification result bundle\n\n"
        "This directory contains mechanically collected seed-42 results for the "
        "frozen factorial study. It deliberately makes no model-selection or "
        "adoption recommendation. Use `collection_summary.json`, `metrics.csv`, "
        "`chemical_metrics.csv`, and `bootstrap.jsonl` for the later analysis.\n",
        encoding="utf-8",
    )
    outputs = (metrics_path, chemical_path, summary_path, readme_path)
    result = {
        "schema_version": "msnlib-simplification/report-complete-v1",
        "spec_sha256": lock["spec_sha256"],
        "interpretation_deferred": True,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
        "bootstrap_complete_sha256": file_sha256(output / "bootstrap_complete.json"),
    }
    write_json(complete_path, result)
    return result


def verify_results(run_dir: str | Path) -> dict[str, Any]:
    """Verify every required arm, budget, split, and neutral report artifact."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory, verify_source_files=True)
    required: list[Path] = [
        directory / "environment_manifest.json",
        directory / "models_frozen.json",
        directory / "evaluation/validation/frozen.json",
        directory / "evaluation/test/complete.json",
        directory / "chemical/scores/complete.json",
        directory / "report/complete.json",
    ]
    for split in ("validation", "test"):
        for representation in ("observed", "full"):
            for arm_id in ARM_IDS:
                root = (
                    directory / "evaluation" / split / representation / "arms" / arm_id
                )
                required.append(root / "inference_complete.json")
                required.extend(root / f"theta_{budget}.npy" for budget in BUDGETS)
                if representation == "observed":
                    required.append(root / "metrics_complete.json")
    missing = [path for path in required if not path.is_file()]
    if missing:
        msg = f"result bundle is incomplete: {missing[0]}"
        raise RuntimeError(msg)
    inventory = {
        str(path.relative_to(directory)): file_sha256(path) for path in sorted(required)
    }
    result = {
        "schema_version": "msnlib-simplification/result-verification-v1",
        "spec_sha256": lock["spec_sha256"],
        "required_artifacts": len(required),
        "missing_artifacts": 0,
        "arms": len(ARM_IDS),
        "budgets": len(BUDGETS),
        "splits": 2,
        "representations": 2,
        "inventory_sha256": object_sha256(inventory),
        "complete": True,
    }
    write_json(directory / "verification.json", result)
    return result


def verify_archived_results(
    run_dir: str | Path,
    *,
    frozen_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a completed bundle without requiring the live checkout to match."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_archived_study(
        directory,
        frozen_source_root=frozen_source_root,
    )
    expected = read_json(directory / "verification.json")
    required: list[Path] = [
        directory / "environment_manifest.json",
        directory / "models_frozen.json",
        directory / "evaluation/validation/frozen.json",
        directory / "evaluation/test/complete.json",
        directory / "chemical/scores/complete.json",
        directory / "report/complete.json",
    ]
    for split in ("validation", "test"):
        for representation in ("observed", "full"):
            for arm_id in ARM_IDS:
                root = (
                    directory / "evaluation" / split / representation / "arms" / arm_id
                )
                required.append(root / "inference_complete.json")
                required.extend(root / f"theta_{budget}.npy" for budget in BUDGETS)
                if representation == "observed":
                    required.append(root / "metrics_complete.json")
    missing = [path for path in required if not path.is_file()]
    if missing:
        msg = f"archived result bundle is incomplete: {missing[0]}"
        raise RuntimeError(msg)
    inventory = {
        str(path.relative_to(directory)): file_sha256(path) for path in sorted(required)
    }
    observed = {
        "schema_version": "msnlib-simplification/result-verification-v1",
        "spec_sha256": lock["spec_sha256"],
        "required_artifacts": len(required),
        "missing_artifacts": 0,
        "arms": len(ARM_IDS),
        "budgets": len(BUDGETS),
        "splits": 2,
        "representations": 2,
        "inventory_sha256": object_sha256(inventory),
        "complete": True,
    }
    if observed != expected:
        msg = "archived verification record does not match the result inventory"
        raise ValueError(msg)

    recovery_path = directory / "recovery_provenance.json"
    if recovery_path.is_file():
        recovery = read_json(recovery_path)
        recovery_files = {
            "recovery_driver": directory / recovery["recovery_driver"]["path"],
            "recovery_runner": directory / recovery["recovery_runner"]["path"],
        }
        for name, path in recovery_files.items():
            if file_sha256(path) != recovery[name]["sha256"]:
                msg = f"archived recovery file changed: {path.name}"
                raise ValueError(msg)
        for name, digest in recovery["chemical_output_sha256"].items():
            if file_sha256(directory / "chemical/scores" / name) != digest:
                msg = f"archived chemical output changed: {name}"
                raise ValueError(msg)

    return {
        **observed,
        "archived_source_files": lock["verified_frozen_source_files"],
        "archived_source_root": lock["verified_frozen_source_root"],
        "recovery_provenance_verified": recovery_path.is_file(),
    }
