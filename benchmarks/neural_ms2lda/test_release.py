"""Frozen model ownership and atomic release of the held-out test inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .reproduction_audit import file_record as _file_record
from .reproduction_audit import verify_linked_inputs
from .utils import read_json, sha256_file, write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

TEST_DATA_FILES = (
    "test_observed.npz",
    "test_completion.npz",
    "test_full.npz",
    "test_records.jsonl",
)


def _model_files(run: Path, method: str) -> tuple[Path, ...]:
    """Resolve every model-side artifact that must remain frozen for test use."""
    if method == "tomotopy":
        return (run / "tomotopy/model.bin",)
    output = run / "models" / method
    return (output / "weights.pt", output / "config.json", output / "result.json")


def verify_released_model(
    run_directory: Path,
    *,
    method: str,
    model_path: Path,
) -> dict[str, object]:
    """Verify that one unchanged fitted model was named at test release."""
    run = run_directory.expanduser().resolve(strict=True)
    manifest = read_json(run / "test_input_manifest.json")
    if method not in manifest.get("methods", []):
        msg = f"method was not frozen when test inputs were released: {method}"
        raise RuntimeError(msg)
    expected_inputs = {run / "data" / name for name in TEST_DATA_FILES}
    input_rows = manifest.get("linked_test_inputs", [])
    if (
        manifest.get("exposed_after_validation") is not True
        or len(input_rows) != len(expected_inputs)
        or {Path(row["linked_path"]).absolute() for row in input_rows}
        != expected_inputs
    ):
        raise RuntimeError("test-release manifest has incomplete test inputs")
    verify_linked_inputs(manifest, field="linked_test_inputs")
    expected_files = {
        path.expanduser().resolve(strict=True) for path in _model_files(run, method)
    }
    method_records = [
        row for row in manifest.get("frozen_models", []) if row.get("method") == method
    ]
    recorded_files = {
        Path(row["path"]).expanduser().resolve(strict=True) for row in method_records
    }
    if recorded_files != expected_files:
        msg = f"test-release manifest has incomplete frozen inputs for {method}"
        raise RuntimeError(msg)
    for row in method_records:
        path = Path(row["path"]).expanduser().resolve(strict=True)
        if int(row["bytes"]) != path.stat().st_size or str(
            row["sha256"],
        ) != sha256_file(
            path,
        ):
            msg = f"frozen model changed after test release: {path}"
            raise RuntimeError(msg)

    resolved_model = model_path.expanduser().resolve(strict=True)
    matches = [
        row
        for row in method_records
        if Path(row["path"]).resolve(strict=True) == resolved_model
    ]
    if len(matches) != 1:
        msg = f"test-release manifest does not uniquely own model: {resolved_model}"
        raise RuntimeError(msg)
    return matches[0]  # Every frozen artifact was already re-hashed above.


def expose_test_view(
    run_directory: Path,
    prepared_run: Path,
    *,
    methods: Sequence[str],
) -> dict[str, Any]:
    """Link test inputs after proving every named method finished validation."""
    run = run_directory.expanduser().resolve(strict=True)
    prepared = prepared_run.expanduser().resolve(strict=True)
    if not methods or len(set(methods)) != len(methods):
        msg = "one or more unique methods are required"
        raise ValueError(msg)
    manifest_path = run / "test_input_manifest.json"
    if manifest_path.exists():
        msg = f"test view already exists: {manifest_path}"
        raise FileExistsError(msg)

    frozen_models = []
    validation_outputs = []
    for method in methods:
        evaluation = run / "validation_evaluation" / method / "complete.json"
        chemistry = run / "validation_chemical" / method / "complete.json"
        model_files = _model_files(run, method)
        for path in (*model_files, evaluation, chemistry):
            if not path.is_file():
                msg = f"test exposure requires completed validation artifact: {path}"
                raise FileNotFoundError(
                    msg,
                )
        for model_file in model_files:
            record = _file_record(model_file)
            record["method"] = method
            frozen_models.append(record)
        validation_outputs.extend((_file_record(evaluation), _file_record(chemistry)))

    linked_inputs = []
    planned_links = []
    for name in TEST_DATA_FILES:
        source = prepared / "data" / name
        destination = run / "data" / name
        if destination.exists() or destination.is_symlink():
            msg = f"test input already exposed: {destination}"
            raise FileExistsError(msg)
        record = _file_record(source)
        record["linked_path"] = str(destination)
        linked_inputs.append(record)
        planned_links.append((source, destination))

    result = {
        "split": "test",
        "exposed_after_validation": True,
        "methods": list(methods),
        "frozen_models": frozen_models,
        "completed_validation_outputs": validation_outputs,
        "linked_test_inputs": linked_inputs,
    }
    # Validate all sources and destinations before exposing any test data.
    # If link creation or manifest publication fails, undo only this call's links.
    created_links = []
    try:
        for source, destination in planned_links:
            destination.symlink_to(source)
            created_links.append(destination)
        write_json(manifest_path, result)
    except Exception:
        for destination in reversed(created_links):
            destination.unlink()
        raise
    return result
