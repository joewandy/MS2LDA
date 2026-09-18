"""Verify and atomically package one sealed clean-room reproduction.

Historical scientific comparisons and gates live in the research package.
This command owns orchestration and publication, not model/metric definitions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.evidence_bundle import (
    assert_no_machine_paths,
    copy_raw_evidence,
    path_replacements,
    rewrite_json_as_portable,
    write_package_seals,
    write_summary_artifacts,
)
from benchmarks.neural_ms2lda.reproduction_audit import (
    probability_audit,
    read_json,
    validate_model_views,
    verify_stage_records,
)
from benchmarks.neural_ms2lda.reproduction_comparison import real_evidence
from benchmarks.neural_ms2lda.reproduction_plan import reproduction_paths
from benchmarks.neural_ms2lda.reproduction_summaries import (
    chemical_integrity_checks,
    claim_checks,
    exact_data_checks,
    require_neural_device,
    stability,
    synthetic_tables,
)
from benchmarks.neural_ms2lda.study_protocol import load_protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_package(
    raw_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build a verified package in a new staging directory."""
    manifest, stage_records = verify_stage_records(raw_root)
    require_neural_device(
        manifest.get("neural_execution_device"),
        label="reproduction manifest",
    )
    validation_views = validate_model_views(raw_root)
    probability = probability_audit(raw_root)
    paths = reproduction_paths(raw_root)
    preparation = read_json(paths.prepared / "preparation_summary.json")
    protocol = load_protocol()
    data_checks = exact_data_checks(preparation, protocol)
    if not data_checks["all_passed"]:
        msg = "immutable data/configuration checks failed"
        raise RuntimeError(msg)

    primary, synthetic_summary, high_k = synthetic_tables(raw_root)
    (
        comparison,
        validation_comparison,
        tomotopy,
        seed_rows,
        proposed,
        chemical_results,
        tomotopy_test_raw,
    ) = real_evidence(raw_root, protocol)
    stability_summary = stability(seed_rows, comparison)
    claims = claim_checks(
        comparison,
        stability_summary,
        high_k,
        int(preparation["data"]["split"]["spectrum_counts"]["test"]),
    )
    chemical_integrity = chemical_integrity_checks(comparison, stability_summary)
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
        "stability": stability_summary,
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
    write_summary_artifacts(destination, evidence)
    copy_raw_evidence(
        paths,
        destination,
        manifest=manifest,
        claims=claims,
        chemical_results=chemical_results,
        tomotopy_test_raw=tomotopy_test_raw,
    )
    replacements = path_replacements(paths, manifest)
    rewrite_json_as_portable(destination, replacements)
    write_package_seals(
        destination,
        manifest=manifest,
        stage_records=stage_records,
        claims=claims,
        data_quality=data_quality,
        replacements=replacements,
    )
    rewrite_json_as_portable(destination, replacements)
    assert_no_machine_paths(destination)
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
