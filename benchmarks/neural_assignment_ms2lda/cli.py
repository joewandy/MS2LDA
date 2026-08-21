"""Command-line interface for the clean neural MS2LDA checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or exactly resume the full workflow")
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--tomotopy-reference-run", required=True, type=Path)
    status = commands.add_parser("status", help="show progress")
    status.add_argument("--run", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify provenance and artifacts")
    verify.add_argument("--run", required=True, type=Path)
    verify.add_argument("--data-root", type=Path)
    verify.add_argument("--large-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        from .orchestrator import run_pipeline

        result = run_pipeline(
            args.run,
            data_root=args.data_root,
            tomotopy_reference_run=args.tomotopy_reference_run,
        )
    elif args.command == "status":
        from .orchestrator import status

        result = status(args.run)
    elif args.command == "verify":
        from .config import verify_run

        result = verify_run(
            args.run,
            data_root=args.data_root,
            verify_large_inputs=args.large_inputs,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
