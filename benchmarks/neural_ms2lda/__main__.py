"""Command-line entry point for the neural MS2LDA reproducibility workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_process_threads() -> None:
    """Set the frozen CPU allowance before importing numerical libraries.

    PyTorch can change its own pool later, but BLAS libraries commonly read
    their limits at import time. Reading the JSON with the standard library
    here ensures the documented ``python -m`` entry point sets them in time.
    """
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    count = str(int(protocol["cpu_threads"]))
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = count


def _main() -> int:
    """Configure numerical threads before importing and dispatching the workflow."""
    _configure_process_threads()
    return main()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the workflow")
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--run", required=True, type=Path)
    status = commands.add_parser("status", help="show progress")
    status.add_argument("--run", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one CLI command after the process thread limit is set."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        from .pipeline import run_pipeline

        result = run_pipeline(args.run, data_root=args.data_root)
    elif args.command == "status":
        from .pipeline import status

        result = status(args.run)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
