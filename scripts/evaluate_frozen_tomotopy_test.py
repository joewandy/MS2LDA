"""Evaluate the frozen Tomotopy model on the fixed MSnLib test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.chemical import run_chemical_scoring
from benchmarks.neural_ms2lda.reproducibility import sha256_file
from benchmarks.neural_ms2lda.tomotopy import evaluate_tomotopy
from benchmarks.neural_ms2lda.utils import read_json, write_json
from scripts.prepare_msnlib_test_view import verify_released_model

if TYPE_CHECKING:
    from collections.abc import Sequence


def evaluate_frozen_tomotopy_test(
    run_directory: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Compute test completion and MAG/SOS without changing the fitted model."""
    run = run_directory.expanduser().resolve(strict=True)
    assets = data_root.expanduser().resolve(strict=True)
    if not (run / "test_input_manifest.json").is_file():
        msg = "test inputs have not been released for Tomotopy"
        raise FileNotFoundError(msg)
    model_path = run / "tomotopy/model.bin"
    release_record = verify_released_model(
        run,
        method="tomotopy",
        model_path=model_path,
    )
    output = run / "tomotopy/test_result.json"
    if output.is_file():
        result = read_json(output)
        if (
            result.get("model_sha256") != release_record["sha256"]
            or result.get("model_unchanged_after_evaluation") is not True
        ):
            msg = "cached Tomotopy test result is not bound to the frozen model"
            raise RuntimeError(msg)
        return result
    before = sha256_file(model_path)
    protocol = read_json(run / "protocol.json")
    evaluation = evaluate_tomotopy(run, protocol, split="test")
    chemistry = run_chemical_scoring(
        run,
        method="tomotopy",
        data_root=assets,
        protocol=protocol,
        split="test",
    )
    unchanged = sha256_file(model_path) == before
    if not unchanged:
        msg = "frozen Tomotopy model changed during test evaluation"
        raise RuntimeError(msg)
    result = {
        "method": "tomotopy",
        "split": "test",
        "evaluation": evaluation,
        "chemistry": chemistry,
        "model_sha256": before,
        "model_unchanged_after_evaluation": unchanged,
        "training_or_optimization_performed": False,
    }
    write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the frozen Tomotopy comparator on test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = evaluate_frozen_tomotopy_test(args.run, args.data_root)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
