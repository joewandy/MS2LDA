# ruff: noqa: C901, PLR0911, PLR0912
"""Command-line interface for the frozen simplification study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .spec import ARM_IDS, DISCOVERY_IDS

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.msnlib_simplification",
        description="HybridLDA simplification collection without model selection",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--source-run", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, default=Path.cwd())
    for name in (
        "preflight",
        "prepare-counts",
        "import-current-discovery",
        "run-symmetric-discovery",
        "freeze-models",
        "prepare-full-validation-dreams",
        "finalize-validation",
        "finalize-test",
        "score-chemical",
        "report",
        "verify",
        "run",
        "status",
    ):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
    archived = commands.add_parser("verify-archive")
    archived.add_argument("--run", type=Path, required=True)
    archived.add_argument("--frozen-source-root", type=Path)
    targets = commands.add_parser("build-targets")
    targets.add_argument("--run", type=Path, required=True)
    targets.add_argument("--discovery", choices=DISCOVERY_IDS, required=True)
    train = commands.add_parser("train-arm")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--arm", choices=ARM_IDS, required=True)
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--run", type=Path, required=True)
    annotate.add_argument("--discovery", choices=DISCOVERY_IDS, required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "freeze":
        from .spec import freeze_study

        return freeze_study(
            run_dir=args.run,
            source_run=args.source_run,
            repo_root=args.repo_root,
        )
    if args.command == "preflight":
        from .spec import preflight

        return preflight(args.run)
    if args.command == "prepare-counts":
        from .data import prepare_count_inputs

        return prepare_count_inputs(args.run)
    if args.command == "import-current-discovery":
        from .discovery import import_current_discovery

        return import_current_discovery(args.run)
    if args.command == "run-symmetric-discovery":
        from .discovery import run_symmetric_discovery

        return run_symmetric_discovery(args.run)
    if args.command == "build-targets":
        from .encoders import build_direct_targets

        return build_direct_targets(args.run, args.discovery)
    if args.command == "train-arm":
        from .encoders import train_encoder

        return train_encoder(args.run, args.arm)
    if args.command == "freeze-models":
        from .evaluation import freeze_models

        return freeze_models(args.run)
    if args.command == "prepare-full-validation-dreams":
        from .data import prepare_full_validation_dreams_embeddings

        return prepare_full_validation_dreams_embeddings(args.run)
    if args.command == "finalize-validation":
        from .evaluation import finalize_validation

        return finalize_validation(args.run)
    if args.command == "finalize-test":
        from .evaluation import finalize_test

        return finalize_test(args.run)
    if args.command == "annotate":
        from .chemical import annotate_discovery

        return annotate_discovery(args.run, args.discovery)
    if args.command == "score-chemical":
        from .chemical import score_all_chemical_results

        return score_all_chemical_results(args.run)
    if args.command == "report":
        from .report import build_report

        return build_report(args.run)
    if args.command == "verify":
        from .report import verify_results

        return verify_results(args.run)
    if args.command == "verify-archive":
        from .report import verify_archived_results

        return verify_archived_results(
            args.run,
            frozen_source_root=args.frozen_source_root,
        )
    if args.command == "run":
        from .orchestrator import run_overnight

        return run_overnight(args.run)
    if args.command == "status":
        from .orchestrator import status

        return status(args.run)
    msg = f"unsupported command: {args.command}"
    raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI turns worker failures into status
        sys.stderr.write(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
        )
        return 1
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
