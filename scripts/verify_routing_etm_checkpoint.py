"""Verify the frozen Routing ETM validation checkpoint without opening test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

DEFAULT_MANIFEST = Path(
    "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/"
    "checkpoint_manifest.json",
)
FLOAT_TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        message = f"expected a JSON object: {path}"
        raise TypeError(message)
    return value


def _repo_path(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root):
        message = f"checkpoint path escapes repository: {relative}"
        raise ValueError(message)
    return path


def _compare(
    errors: list[str],
    *,
    label: str,
    actual: object,
    expected: object,
) -> None:
    if isinstance(expected, float):
        try:
            matches = math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _verify_hashed_files(
    repo_root: Path,
    rows: Iterable[Mapping[str, object]],
    errors: list[str],
    *,
    source_commit: str | None = None,
) -> int:
    checks = 0
    for row in rows:
        relative = str(row["path"])
        if source_commit is not None:
            completed = subprocess.run(  # noqa: S603
                ["git", "show", f"{source_commit}:{relative}"],  # noqa: S607
                cwd=repo_root,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                errors.append(
                    f"missing checkpoint source at {source_commit}: {relative}",
                )
                continue
            checks += 1
            _compare(
                errors,
                label=f"{relative} bytes at {source_commit}",
                actual=len(completed.stdout),
                expected=int(row["bytes"]),
            )
            _compare(
                errors,
                label=f"{relative} sha256 at {source_commit}",
                actual=_sha256_bytes(completed.stdout),
                expected=str(row["sha256"]),
            )
            continue
        path = _repo_path(repo_root, relative)
        if not path.is_file():
            errors.append(f"missing checkpoint file: {relative}")
            continue
        checks += 1
        _compare(
            errors,
            label=f"{relative} bytes",
            actual=path.stat().st_size,
            expected=int(row["bytes"]),
        )
        _compare(
            errors,
            label=f"{relative} sha256",
            actual=_sha256(path),
            expected=str(row["sha256"]),
        )
    return checks


def _routing_metrics(metrics: Mapping[str, Any]) -> dict[str, object]:
    chemistry = metrics["validation_chemistry"]
    inventory = metrics["topic_inventory"]
    support = metrics["theta_support"]
    return {
        "completion_nll": metrics["document_completion"]["nll_per_token"],
        "evaluable_motifs": chemistry["eligible_topics"],
        "mean_sos": chemistry["mean_sos"],
        "median_effective_topics": inventory["median_effective_topics_per_spectrum"],
        "median_exact_support": support["median_exact_support"],
        "median_sos": chemistry["median_sos"],
        "optimized_motifs": chemistry["optimized_motifs"],
        "unique_top1_topics": inventory["unique_top1_topics"],
        "useful_motifs": chemistry["useful_motifs"],
    }


def _locked_metrics(results: Mapping[str, Any], method: str) -> dict[str, object]:
    methods = {row["method"]: row for row in results["methods"]}
    chemistry = methods[method]["validation"]
    completion = results["secondary"]["completion_nll_per_token"][method]
    return {
        "completion_nll": completion["validation"],
        "evaluable_motifs": chemistry["high_confidence_evaluable_motifs"],
        "mean_sos": chemistry["mean_sos"],
        "median_sos": chemistry["median_sos"],
        "optimized_motifs": chemistry["optimized_motifs"],
        "useful_motifs": chemistry["useful_high_confidence_motifs"],
    }


def _verify_expected_metrics(
    errors: list[str],
    *,
    label: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> int:
    for name, expected_value in expected.items():
        _compare(
            errors,
            label=f"{label}.{name}",
            actual=actual.get(name),
            expected=expected_value,
        )
    return len(expected)


def _verify_comparison_row(
    comparison_path: Path,
    expected: Mapping[str, object],
    errors: list[str],
) -> int:
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["model"] == "routing-informed sparse ETM"]
    if len(matches) != 1:
        errors.append(
            "comparison must contain exactly one routing-informed sparse ETM row",
        )
        return 0
    row = matches[0]
    columns = {
        "optimized_motifs": int,
        "evaluable_motifs": int,
        "useful_motifs": int,
        "mean_sos": float,
        "median_sos": float,
        "completion_nll": float,
        "median_effective_topics": float,
        "median_exact_support": int,
        "unique_top1_topics": int,
    }
    for name, converter in columns.items():
        _compare(
            errors,
            label=f"comparison.routing_etm.{name}",
            actual=converter(row[name]),
            expected=expected[name],
        )
    _compare(
        errors,
        label="comparison.routing_etm.passed_all_frozen_gates",
        actual=row["passed_all_frozen_gates"],
        expected="False",
    )
    return len(columns) + 1


def _verify_chemistry_invariants(metrics: Mapping[str, Any], errors: list[str]) -> int:
    chemistry = metrics["validation_chemistry"]
    bands = chemistry["sos_bands"]
    evaluable = (
        bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"] + bands["low_lt_0_6"]
    )
    useful = bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
    _compare(
        errors,
        label="routing chemistry SOS bands sum to evaluable motifs",
        actual=evaluable,
        expected=chemistry["eligible_topics"],
    )
    _compare(
        errors,
        label="routing chemistry useful motifs equal high plus intermediate bands",
        actual=useful,
        expected=chemistry["useful_motifs"],
    )
    _compare(
        errors,
        label="routing optimized motifs derive from annotation coverage",
        actual=chemistry["optimized_motifs"],
        expected=round(chemistry["annotation_coverage"] * chemistry["total_topics"]),
    )
    return 3


def _verify_access_audit(audit: Mapping[str, Any], errors: list[str]) -> int:
    false_flags = (
        "candidate_test_artifacts_accessed",
        "candidate_test_chemistry_loaded",
        "candidate_test_mag_or_sos_computed",
        "candidate_test_metrics_inspected",
    )
    for name in false_flags:
        _compare(
            errors,
            label=f"access audit {name}",
            actual=audit[name],
            expected=False,
        )
    _compare(
        errors,
        label="access audit chemical split",
        actual=audit["chemical_split"],
        expected="validation",
    )
    return len(false_flags) + 1


def _verify_external_rows(
    rows: Iterable[Mapping[str, Any]],
    errors: list[str],
    *,
    require_all: bool,
    label: str,
) -> int:
    checks = 0
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file():
            if require_all:
                errors.append(f"missing {label}: {path}")
            continue
        checks += 1
        _compare(
            errors,
            label=f"{label} {path} bytes",
            actual=path.stat().st_size,
            expected=int(row["bytes"]),
        )
        _compare(
            errors,
            label=f"{label} {path} sha256",
            actual=_sha256(path),
            expected=str(row["sha256"]),
        )
    return checks


def verify_checkpoint(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    verify_local_artifacts: bool = False,
    verify_inputs: bool = False,
    require_external: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable integrity and consistency result."""
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    manifest_file = (
        manifest_path
        if manifest_path.is_absolute()
        else (root / manifest_path).resolve()
    )
    manifest = _read_json(manifest_file)
    errors: list[str] = []
    checks = _verify_hashed_files(
        root,
        manifest["implementation_files"],
        errors,
        source_commit=str(manifest["source_implementation_commit"]),
    )
    checks += _verify_hashed_files(root, manifest["evidence_files"], errors)

    metrics = _read_json(_repo_path(root, str(manifest["metrics_source"])))
    expected = manifest["expected_validation_metrics"]
    checks += _verify_expected_metrics(
        errors,
        label="routing_etm",
        actual=_routing_metrics(metrics),
        expected=expected["routing_etm"],
    )
    checks += _verify_chemistry_invariants(metrics, errors)

    locked = _read_json(_repo_path(root, str(manifest["locked_comparator_source"])))
    checks += _verify_expected_metrics(
        errors,
        label="m1",
        actual=_locked_metrics(locked, "neural"),
        expected=expected["m1"],
    )
    checks += _verify_expected_metrics(
        errors,
        label="tomotopy",
        actual=_locked_metrics(locked, "tomotopy"),
        expected=expected["tomotopy"],
    )
    checks += _verify_comparison_row(
        _repo_path(root, str(manifest["comparison_source"])),
        expected["routing_etm"],
        errors,
    )
    audit_path = _repo_path(
        root,
        str(manifest["validation_access_audit_source"]),
    )
    audit = _read_json(audit_path)
    checks += _verify_access_audit(audit, errors)

    provenance = _read_json(_repo_path(root, str(manifest["provenance_source"])))
    if verify_local_artifacts:
        checks += _verify_external_rows(
            provenance["local_artifacts"],
            errors,
            require_all=require_external,
            label="local model artifact",
        )
    if verify_inputs:
        checks += _verify_external_rows(
            provenance["validation_inputs"]["linked_inputs"],
            errors,
            require_all=require_external,
            label="validation input",
        )

    return {
        "checkpoint_id": manifest["checkpoint_id"],
        "checks_completed": checks,
        "errors": errors,
        "external_files_required": require_external,
        "local_artifacts_checked": verify_local_artifacts,
        "status": "passed" if not errors else "failed",
        "validation_inputs_checked": verify_inputs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify committed evidence and optionally large local artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-local-artifacts", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="fail if a referenced local artifact or input is unavailable",
    )
    args = parser.parse_args(argv)
    result = verify_checkpoint(
        args.manifest,
        verify_local_artifacts=args.verify_local_artifacts,
        verify_inputs=args.verify_inputs,
        require_external=args.require_external,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
