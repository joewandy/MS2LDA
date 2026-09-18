"""Expose the fixed MSnLib test split only after validation is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.neural_ms2lda.test_release import expose_test_view

if TYPE_CHECKING:
    from collections.abc import Sequence


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
