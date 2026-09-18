"""Create a sealed train-and-validation view of one prepared MSnLib run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.neural_ms2lda.reproducibility import prepare_validation_view
from benchmarks.neural_ms2lda.utils import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence


def forbidden_test_files(run_directory: Path) -> list[str]:
    """Return any test-split files accidentally exposed in the sealed view."""
    data_directory = run_directory / "data"
    return sorted(path.name for path in data_directory.glob("test*") if path.is_file())


def create_validation_view(
    run_directory: Path,
    prepared_run: Path,
    *,
    expected_topics: int = 1000,
) -> dict[str, object]:
    """Link immutable train/validation inputs and fail closed on test exposure."""
    manifest = prepare_validation_view(
        run_directory.expanduser().resolve(),
        prepared_run.expanduser().resolve(strict=True),
        expected_topics=expected_topics,
    )
    forbidden = forbidden_test_files(run_directory.expanduser().resolve())
    if forbidden:
        message = f"sealed validation view exposes test files: {forbidden}"
        raise RuntimeError(message)
    result = {
        **manifest,
        "test_spectra_exposed_to_model_run": False,
        "forbidden_test_files": forbidden,
    }
    write_json(run_directory / "validation_input_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Create one validation-only input view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared-run", required=True, type=Path)
    parser.add_argument("--expected-topics", type=int, default=1000)
    args = parser.parse_args(argv)
    result = create_validation_view(
        args.run,
        args.prepared_run,
        expected_topics=args.expected_topics,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
