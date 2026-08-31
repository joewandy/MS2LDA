"""Integrity checks and small I/O helpers for clean scientific reproductions."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.neural_ms2lda.reproducibility import validate_probability_matrix
from benchmarks.neural_ms2lda.reproduction_plan import (
    METHOD,
    TRAINING_SEEDS,
    probability_artifact_paths,
    reproduction_paths,
    stage_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other top-level types."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"expected a JSON object: {path}"
        raise TypeError(msg)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one stable JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write one deterministic compact table, including heterogeneous rows.

    Result tables deliberately mix shared comparison fields with method-specific
    diagnostics.  Preserve the first-seen column order while taking the union of
    every row's keys so a later method cannot be rejected or silently truncated.
    Missing values are emitted as empty CSV cells by :class:`csv.DictWriter`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    """Return a path, size, and digest record."""
    resolved = path.resolve(strict=True)
    displayed = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None and resolved.is_relative_to(relative_to.resolve())
        else str(resolved)
    )
    return {
        "path": displayed,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_stage_records(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify every planned stage, command, timestamp, and output digest."""
    paths = reproduction_paths(root)
    manifest = read_json(paths.root / "reproduction_manifest.json")
    if manifest.get("status") != "raw_results_complete":
        msg = "reproduction is not marked raw_results_complete"
        raise RuntimeError(msg)
    records = []
    previous_finished: datetime | None = None
    for stage in stage_plan(paths):
        record_path = paths.stages / f"{stage.name}.json"
        record = read_json(record_path)
        if record.get("name") != stage.name or record.get("status") != "complete":
            msg = f"stage is not complete: {stage.name}"
            raise RuntimeError(msg)
        if record.get("command") != list(stage.command):
            msg = f"stage command differs from the frozen plan: {stage.name}"
            raise RuntimeError(msg)
        started = datetime.fromisoformat(str(record["started_utc"]))
        finished = datetime.fromisoformat(str(record["finished_utc"]))
        if finished < started or (
            previous_finished is not None and started < previous_finished
        ):
            msg = f"stage chronology is invalid: {stage.name}"
            raise RuntimeError(msg)
        previous_finished = finished
        recorded_outputs = {row["path"]: row for row in record.get("outputs", [])}
        if set(recorded_outputs) != {str(output) for output in stage.outputs}:
            msg = f"stage output ownership differs from plan: {stage.name}"
            raise RuntimeError(msg)
        for output in stage.outputs:
            expected = recorded_outputs[str(output)]
            actual = file_record(output)
            if (
                actual["bytes"] != expected["bytes"]
                or actual["sha256"] != expected["sha256"]
            ):
                msg = f"stage output changed after sealing: {output}"
                raise RuntimeError(msg)
        records.append(record)
    return manifest, records


def _manifest_identity(manifest: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    """Return path-independent validation input identities."""
    return [
        (Path(row["path"]).name, int(row["bytes"]), str(row["sha256"]))
        for row in manifest["linked_inputs"]
    ]


def verify_linked_inputs(
    manifest: Mapping[str, Any],
    *,
    field: str,
) -> None:
    """Re-hash source and model-visible paths recorded by one input manifest."""
    for row in manifest[field]:
        if "linked_path" not in row:
            msg = f"{field} row lacks its model-visible linked path"
            raise RuntimeError(msg)
        expected_size = int(row["bytes"])
        expected_digest = str(row["sha256"])
        for key in ("path", "linked_path"):
            path = Path(row[key])
            if (
                not path.is_file()
                or path.stat().st_size != expected_size
                or sha256_file(path) != expected_digest
            ):
                msg = f"manifest-owned input changed: {path}"
                raise RuntimeError(msg)


def validate_model_views(root: Path) -> dict[str, Any]:
    """Prove identical inputs and verify that test followed model freezing."""
    paths = reproduction_paths(root)
    runs = {
        "controls": paths.controls,
        "tomotopy": paths.tomotopy,
        **{
            f"contextual_seed_{seed}": paths.contextual[seed] for seed in TRAINING_SEEDS
        },
    }
    reference: list[tuple[str, int, str]] | None = None
    test_reference: list[tuple[str, int, str]] | None = None
    for name, run in runs.items():
        validation_manifest = read_json(run / "validation_input_manifest.json")
        if validation_manifest.get("candidate_test_artifacts_accessed") is not False:
            msg = f"{name} does not preserve the validation boundary"
            raise RuntimeError(msg)
        verify_linked_inputs(validation_manifest, field="linked_inputs")
        identity = _manifest_identity(validation_manifest)
        if reference is None:
            reference = identity
        elif identity != reference:
            msg = f"{name} validation inputs differ from controls"
            raise RuntimeError(msg)

        test_manifest = read_json(run / "test_input_manifest.json")
        if test_manifest.get("exposed_after_validation") is not True:
            msg = f"{name} test release was not validation-gated"
            raise RuntimeError(msg)
        verify_linked_inputs(test_manifest, field="linked_test_inputs")
        test_identity = [
            (Path(row["path"]).name, int(row["bytes"]), str(row["sha256"]))
            for row in test_manifest["linked_test_inputs"]
        ]
        if test_reference is None:
            test_reference = test_identity
        elif test_identity != test_reference:
            msg = f"{name} test inputs differ from controls"
            raise RuntimeError(msg)
        for row in (
            *test_manifest["frozen_models"],
            *test_manifest["completed_validation_outputs"],
        ):
            path = Path(row["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(row["bytes"])
                or sha256_file(path) != row["sha256"]
            ):
                msg = f"frozen artifact changed after test release: {path}"
                raise RuntimeError(msg)
    return {
        "identical_validation_inputs": True,
        "identical_test_inputs": True,
        "test_released_after_validation_and_model_freeze": True,
        "model_views": list(runs),
        "validation_input_identity": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest in reference or []
        ],
        "test_input_identity": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest in test_reference or []
        ],
    }


def probability_audit(root: Path) -> dict[str, Any]:
    """Validate every paper-facing beta and validation/test theta matrix."""
    paths = reproduction_paths(root)
    methods = {
        "canonical ETM": (paths.controls, "etm"),
        "balanced ETM": (paths.controls, "etm_balanced"),
        "Tomotopy LDA": (paths.tomotopy, "tomotopy"),
        **{
            f"Contextual Sparse ETM seed {seed}": (paths.contextual[seed], METHOD)
            for seed in TRAINING_SEEDS
        },
    }
    rows = []
    for label, (run, method) in methods.items():
        beta_path, theta_path, test_theta_path = probability_artifact_paths(run, method)
        beta = np.load(beta_path, mmap_mode="r")
        theta = np.load(theta_path, mmap_mode="r")
        test_theta = np.load(test_theta_path, mmap_mode="r")
        validate_probability_matrix(beta, name=f"{label} beta")
        validate_probability_matrix(theta, name=f"{label} validation theta")
        validate_probability_matrix(test_theta, name=f"{label} test theta")
        rows.append(
            {
                "model": label,
                "beta_shape": list(beta.shape),
                "theta_shape": list(theta.shape),
                "test_theta_shape": list(test_theta.shape),
                "beta_sha256": sha256_file(beta_path),
                "validation_theta_sha256": sha256_file(theta_path),
                "test_theta_sha256": sha256_file(test_theta_path),
                "passed": True,
            },
        )
    return {"all_passed": True, "matrices": rows}
