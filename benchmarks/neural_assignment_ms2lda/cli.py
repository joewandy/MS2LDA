"""Command-line interface for the staged neural-assignment MS2LDA study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or exactly resume the staged study")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--source", required=True, type=Path)
    run.add_argument("--reference", required=True, type=Path)
    chemical = commands.add_parser("chemical", help="run MAG in its pinned env")
    chemical.add_argument("--run", required=True, type=Path)
    chemical.add_argument("--attempt", required=True, choices=("primary", "rescue"))
    progress = commands.add_parser("status", help="show a compact progress snapshot")
    progress.add_argument("--run", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify frozen provenance")
    verify.add_argument("--run", required=True, type=Path)
    verify.add_argument("--frozen-source", action="store_true")
    verify.add_argument("--skip-large-inputs", action="store_true")
    smoke = commands.add_parser("smoke", help="run tiny alternating mechanics")
    smoke.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command without eager PyTorch import."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        from .orchestrator import run_study

        result = run_study(
            args.run,
            source_run=args.source,
            reference_run=args.reference,
        )
    elif args.command == "chemical":
        from .chemical import run_chemical_scoring
        from .config import verify_run
        from .utils import read_json

        verify_run(args.run)
        protocol = read_json(args.run / "protocol.resolved.json")
        result = run_chemical_scoring(
            args.run,
            attempt=args.attempt,
            protocol=protocol,
        )
    elif args.command == "status":
        from .orchestrator import status

        result = status(args.run)
    elif args.command == "verify":
        from .config import verify_run

        result = verify_run(
            args.run,
            require_live_code=not args.frozen_source,
            verify_large_inputs=not args.skip_large_inputs,
        )
    elif args.command == "smoke":
        from .smoke import run_smoke

        result = run_smoke(args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
