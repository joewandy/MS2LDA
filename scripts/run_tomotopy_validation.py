"""Train and evaluate Tomotopy without opening or scoring test spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.tomotopy import evaluate_tomotopy, train_tomotopy
from benchmarks.neural_ms2lda.utils import read_json, write_json
from scripts.prepare_msnlib_validation_view import forbidden_test_files

if TYPE_CHECKING:
    from collections.abc import Sequence


def _assert_validation_only(run_directory: Path) -> None:
    """Reject any model-facing test input or test-result artifact."""
    forbidden = forbidden_test_files(run_directory)
    if forbidden:
        message = f"Tomotopy validation run exposes test inputs: {forbidden}"
        raise RuntimeError(message)
    forbidden_outputs = [
        path
        for path in (
            run_directory / "evaluation/tomotopy/complete.json",
            run_directory / "chemical/tomotopy/complete.json",
        )
        if path.exists()
    ]
    if forbidden_outputs:
        message = f"Tomotopy validation run contains test outputs: {forbidden_outputs}"
        raise RuntimeError(message)


def run_validation_only_tomotopy(
    run_directory: Path,
) -> dict[str, Any]:
    """Fit Tomotopy and compute document completion on validation only."""
    run = run_directory.expanduser().resolve(strict=True)
    _assert_validation_only(run)
    protocol = read_json(run / "protocol.json")

    training = train_tomotopy(run, protocol)
    validation = evaluate_tomotopy(run, protocol, split="validation")
    _assert_validation_only(run)

    audit = {
        "evidence_boundary": "training and validation spectra only",
        "test_spectra_exposed_to_model_run": False,
        "test_metrics_computed": False,
        "test_chemistry_computed": False,
        "validation_completion_computed": True,
        "validation_chemistry_computed_in_training_process": False,
    }
    write_json(run / "tomotopy/validation_access_audit.json", audit)
    result = {
        "method": "tomotopy",
        "training": training,
        "validation": validation,
        "validation_access_audit": audit,
    }
    write_json(run / "tomotopy/validation_only_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation-only comparator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_validation_only_tomotopy(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
