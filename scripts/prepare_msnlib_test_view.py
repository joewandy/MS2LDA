"""Expose the fixed MSnLib test split only after validation is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.reproducibility import sha256_file
from benchmarks.neural_ms2lda.utils import read_json, write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

TEST_DATA_FILES = (
    "test_observed.npz",
    "test_completion.npz",
    "test_full.npz",
    "test_records.jsonl",
)


def _file_record(path: Path) -> dict[str, object]:
    """Return immutable identity evidence for one file."""
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _model_file(run: Path, method: str) -> Path:
    """Resolve the frozen fitted-model artifact for one method."""
    if method == "tomotopy":
        return run / "tomotopy/model.bin"
    return run / "models" / method / "weights.pt"


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
    resolved_model = model_path.expanduser().resolve(strict=True)
    matches = [
        row
        for row in manifest.get("frozen_models", [])
        if Path(row["path"]).resolve(strict=True) == resolved_model
    ]
    if len(matches) != 1:
        msg = f"test-release manifest does not uniquely own model: {resolved_model}"
        raise RuntimeError(msg)
    record = matches[0]
    if int(record["bytes"]) != resolved_model.stat().st_size or str(
        record["sha256"],
    ) != sha256_file(resolved_model):
        msg = f"frozen model changed after test release: {resolved_model}"
        raise RuntimeError(msg)
    return record


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
        model = _model_file(run, method)
        evaluation = run / "validation_evaluation" / method / "complete.json"
        chemistry = run / "validation_chemical" / method / "complete.json"
        for path in (model, evaluation, chemistry):
            if not path.is_file():
                msg = f"test exposure requires completed validation artifact: {path}"
                raise FileNotFoundError(
                    msg,
                )
        frozen_models.append(_file_record(model))
        validation_outputs.extend((_file_record(evaluation), _file_record(chemistry)))

    linked_inputs = []
    for name in TEST_DATA_FILES:
        source = prepared / "data" / name
        destination = run / "data" / name
        if destination.exists() or destination.is_symlink():
            msg = f"test input already exposed: {destination}"
            raise FileExistsError(msg)
        source.resolve(strict=True)
        destination.symlink_to(source)
        linked_inputs.append(_file_record(source))

    result = {
        "split": "test",
        "exposed_after_validation": True,
        "methods": list(methods),
        "frozen_models": frozen_models,
        "completed_validation_outputs": validation_outputs,
        "linked_test_inputs": linked_inputs,
    }
    write_json(manifest_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the test partition for frozen models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared-run", required=True, type=Path)
    parser.add_argument("--method", required=True, action="append")
    args = parser.parse_args(argv)
    result = expose_test_view(
        args.run,
        args.prepared_run,
        methods=args.method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
