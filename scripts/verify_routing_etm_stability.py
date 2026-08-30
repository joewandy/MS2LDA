"""Verify the packaged Routing ETM real training-seed stability study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.package_routing_etm_stability import (
    BASELINE_RESULTS,
    DEFAULT_OUTPUT,
    aggregate_rows,
    seed_row,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_MANIFEST = DEFAULT_OUTPUT / "checkpoint_manifest.json"
FLOAT_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        message = f"expected a JSON object: {path}"
        raise TypeError(message)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _repo_path(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root):
        message = f"checkpoint path escapes repository: {relative}"
        raise ValueError(message)
    return path


def _verify_files(
    repo_root: Path,
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> int:
    checks = 0
    for row in rows:
        relative = str(row["path"])
        path = _repo_path(repo_root, relative)
        if not path.is_file():
            errors.append(f"missing packaged file: {relative}")
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


def _verify_external(
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
    *,
    required: bool,
    label: str,
) -> int:
    checks = 0
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file():
            if required:
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


def _recompute_rows(repo_root: Path) -> list[dict[str, object]]:
    baseline_metrics = _read_json(repo_root / BASELINE_RESULTS / "metrics.json")
    baseline_config = _read_json(repo_root / BASELINE_RESULTS / "config.json")
    rows = [
        seed_row(
            baseline_metrics,
            training_seed=int(baseline_config["seed"]),
        ),
    ]
    package_root = repo_root / DEFAULT_OUTPUT
    for seed in (23, 37):
        metrics = _read_json(package_root / f"seed_{seed}" / "metrics.json")
        rows.append(seed_row(metrics, training_seed=seed))
    return rows


def _verify_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _compare(
        errors,
        label="CSV row count",
        actual=len(rows),
        expected=len(expected_rows),
    )
    checks = 1
    for index, (actual, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        for name, expected_value in expected.items():
            value: object = actual[name]
            if isinstance(expected_value, bool):
                value = value == "True"
            elif isinstance(expected_value, int):
                value = int(value)
            elif isinstance(expected_value, float):
                value = float(value)
            _compare(
                errors,
                label=f"CSV row {index} {name}",
                actual=value,
                expected=expected_value,
            )
            checks += 1
    return checks


def verify_stability_package(
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
    checks = _verify_files(root, manifest["implementation_files"], errors)
    checks += _verify_files(root, manifest["packaged_files"], errors)

    summary_path = root / DEFAULT_OUTPUT / "stability_summary.json"
    summary = _read_json(summary_path)
    expected_rows = _recompute_rows(root)
    _compare(
        errors,
        label="summary by-seed rows",
        actual=summary["by_seed"],
        expected=expected_rows,
    )
    checks += 1
    expected_aggregate = aggregate_rows(expected_rows, summary["comparators"])
    for name in ("training_seeds", "runs", "aggregate", "direction_checks"):
        _compare(
            errors,
            label=f"summary {name}",
            actual=summary[name],
            expected=expected_aggregate[name],
        )
        checks += 1
    checks += _verify_csv(
        root / DEFAULT_OUTPUT / "stability_by_seed.csv",
        expected_rows,
        errors,
    )

    for seed in (23, 37):
        audit = _read_json(
            root / DEFAULT_OUTPUT / f"seed_{seed}" / "validation_access_audit.json",
        )
        for name in (
            "candidate_test_artifacts_accessed",
            "candidate_test_chemistry_loaded",
            "candidate_test_mag_or_sos_computed",
            "candidate_test_metrics_inspected",
        ):
            _compare(
                errors,
                label=f"seed {seed} audit {name}",
                actual=audit[name],
                expected=False,
            )
            checks += 1

    if verify_local_artifacts:
        for seed, rows in manifest["external_local_artifacts"].items():
            checks += _verify_external(
                rows,
                errors,
                required=require_external,
                label=f"seed {seed} local artifact",
            )
    if verify_inputs:
        checks += _verify_external(
            manifest["validation_inputs"],
            errors,
            required=require_external,
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
    """Verify committed stability evidence and optional large local files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-local-artifacts", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--require-external", action="store_true")
    args = parser.parse_args(argv)
    result = verify_stability_package(
        args.manifest,
        verify_local_artifacts=args.verify_local_artifacts,
        verify_inputs=args.verify_inputs,
        require_external=args.require_external,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
