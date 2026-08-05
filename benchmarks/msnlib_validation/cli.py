"""Single command-line driver for the leakage-safe MSnLib benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import load_config

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "full-msnlib-k1000.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.msnlib_validation",
        description="Frozen, leakage-safe full MSnLib HybridLDA validation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-inputs")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--no-allocation-probe", action="store_true")

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--data-root", type=Path, required=True)
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, default=Path.cwd())
    freeze.add_argument("--test-results-inspected", action="store_true")

    freeze_derived = commands.add_parser("freeze-derived")
    freeze_derived.add_argument("--config", type=Path, required=True)
    freeze_derived.add_argument("--data-root", type=Path, required=True)
    freeze_derived.add_argument("--run", type=Path, required=True)
    freeze_derived.add_argument("--repo-root", type=Path, default=Path.cwd())
    freeze_derived.add_argument("--source-run", type=Path, required=True)
    freeze_derived.add_argument("--reason")

    reuse = commands.add_parser("reuse-core-artifacts")
    reuse.add_argument("--run", type=Path, required=True)
    reuse.add_argument("--source-run", type=Path, required=True)

    for name in ("run-core", "run-mag", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name == "run-mag":
            command.add_argument("--data-root", type=Path, required=True)

    worker = commands.add_parser("_run-model")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--method", choices=("tomotopy", "hybrid"), required=True)
    worker.add_argument("--seed", type=int, required=True)

    mag_index = commands.add_parser("_build-mag-index")
    mag_index.add_argument("--run", type=Path, required=True)
    mag_index.add_argument("--data-root", type=Path, required=True)

    mag_worker = commands.add_parser("_run-mag-model")
    mag_worker.add_argument("--run", type=Path, required=True)
    mag_worker.add_argument("--data-root", type=Path, required=True)
    mag_worker.add_argument("--method", choices=("tomotopy", "hybrid"), required=True)
    mag_worker.add_argument("--seed", type=int, required=True)

    raw_dreams = commands.add_parser("_run-raw-dreams")
    raw_dreams.add_argument("--run", type=Path, required=True)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path)
    return parser


def _append_execution(run_dir: Path, *, status: str) -> None:
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "status": status,
    }
    with (run_dir / "execution_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "validate-inputs":
        from .protocol import validate_inputs

        return validate_inputs(
            load_config(args.config), args.data_root, output_path=args.output
        )
    if args.command == "preflight":
        from .preflight import run_preflight

        result = run_preflight(
            load_config(args.config),
            data_root=args.data_root,
            output_path=args.output,
            allocation_probe=not args.no_allocation_probe,
        )
        if not result["passed"]:
            raise RuntimeError("full-scale resource preflight failed")
        return result
    if args.command == "freeze":
        from .protocol import freeze_protocol

        return freeze_protocol(
            load_config(args.config),
            config_path=args.config,
            data_root=args.data_root,
            run_dir=args.run,
            repo_root=args.repo_root,
            test_results_inspected=args.test_results_inspected,
        )
    if args.command == "freeze-derived":
        from .protocol import (
            freeze_protocol,
            validate_execution_only_derivation,
        )

        config = load_config(args.config)
        derivation = validate_execution_only_derivation(
            args.source_run,
            config,
            args.reason,
        )
        return freeze_protocol(
            config,
            config_path=args.config,
            data_root=args.data_root,
            run_dir=args.run,
            repo_root=args.repo_root,
            test_results_inspected=True,
            derivation=derivation,
        )
    if args.command == "reuse-core-artifacts":
        from .reuse import reuse_core_artifacts

        return reuse_core_artifacts(args.source_run, args.run)
    if args.command == "run-core":
        from .models import run_all_core_models

        return run_all_core_models(args.run)
    if args.command == "_run-model":
        from .models import run_hybrid_seed, run_tomotopy_seed

        if args.method == "tomotopy":
            return run_tomotopy_seed(args.run, args.seed)
        return run_hybrid_seed(args.run, args.seed)
    if args.command == "run-mag":
        from .mag import run_all_mag

        return run_all_mag(args.run, data_root=args.data_root)
    if args.command == "_build-mag-index":
        from .mag import build_filtered_mag_index

        return build_filtered_mag_index(args.run, data_root=args.data_root)
    if args.command == "_run-mag-model":
        from .mag import run_mag_for_model

        return run_mag_for_model(
            args.run,
            data_root=args.data_root,
            seed=args.seed,
            method=args.method,
        )
    if args.command == "_run-raw-dreams":
        from .mag import run_raw_dreams_baseline

        return run_raw_dreams_baseline(args.run)
    if args.command == "report":
        from .report import build_report

        return build_report(args.run)
    if args.command == "smoke":
        from .smoke import run_smoke

        return run_smoke(args.output)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one benchmark stage and emit its JSON result."""
    args = _parser().parse_args(argv)
    run = getattr(args, "run", None)
    if run is not None and args.command not in {"freeze", "freeze-derived"}:
        _append_execution(run, status="started")
    try:
        result = _run(args)
    except BaseException:
        if run is not None and args.command not in {"freeze", "freeze-derived"}:
            _append_execution(run, status="failed")
        raise
    if run is not None:
        _append_execution(run, status="completed")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
