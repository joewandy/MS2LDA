"""Orchestrate a cache-free train/validation/test model reproduction.

The controller creates one UUID-bound root, records the source checkout and
environment, and runs every paper-facing fit in a dependency-ordered sequence.
Each stage has an append-only log and a completion record containing its exact
command and output hashes.  A stage refuses pre-existing outputs unless its own
completion record proves they were created by this reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.neural_ms2lda.reproduction_plan import (
    NEURAL_DEVICE,
    ReproductionPaths,
    Stage,
    acceptance_policy,
    reproduction_paths,
    stage_plan,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
MINIMUM_IDLE_MEMORY_BYTES = 10_000_000_000
MAXIMUM_IDLE_LOAD_AVERAGE = 4.0


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"expected a JSON object: {path}"
        raise TypeError(msg)
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    """Write one stable JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_output(command: Sequence[str], *, cwd: Path = REPO) -> str:
    """Return a read-only command result for provenance."""
    completed = subprocess.run(  # noqa: S603
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    return output if completed.returncode == 0 else f"unavailable: {output}"


def source_state() -> dict[str, object]:
    """Return the exact Git state that owns the scientific execution."""
    commit = command_output(("git", "rev-parse", "HEAD"))
    status = command_output(("git", "status", "--porcelain"))
    if status and not status.startswith("unavailable:"):
        msg = "clean-room execution requires a clean source checkout"
        raise RuntimeError(msg)
    return {
        "commit": commit,
        "worktree": str(REPO),
        "branch": command_output(("git", "branch", "--show-current")),
        "clean": True,
    }


def initialize(root: Path) -> dict[str, object]:
    """Create a new empty reproduction root and freeze its execution contract."""
    paths = reproduction_paths(root)
    if paths.root.exists():
        msg = f"clean-room root already exists: {paths.root}"
        raise FileExistsError(msg)
    paths.root.mkdir(parents=True)
    paths.logs.mkdir()
    paths.stages.mkdir()
    state = source_state()
    conda = shutil.which("conda")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "reproduction_id": str(uuid.uuid4()),
        "created_utc": utc_now(),
        "source": state,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "kernel": command_output(("uname", "-a")),
            "cpu": command_output(("lscpu",)),
            "gpu": command_output(("nvidia-smi", "-q")),
            "conda_packages": (
                command_output((conda, "list", "--export"))
                if conda is not None
                else "unavailable: conda executable not found"
            ),
            "pip_packages": command_output((sys.executable, "-m", "pip", "freeze")),
        },
        "paths": {
            "root": str(paths.root),
            "assets": str(paths.assets),
            "prepared": str(paths.prepared),
            "synthetic": str(paths.synthetic),
            "controls": str(paths.controls),
            "tomotopy": str(paths.tomotopy),
            "contextual": {
                str(key): str(value) for key, value in paths.contextual.items()
            },
        },
        "acceptance_policy": acceptance_policy(),
        "neural_execution_device": NEURAL_DEVICE,
        "split_protocol": (
            "fit on train; select and ablate on validation; evaluate frozen models "
            "on test"
        ),
        "status": "initialized",
    }
    write_json(paths.root / "reproduction_manifest.json", manifest)
    return manifest


def _available_memory_bytes() -> int:
    """Read Linux MemAvailable."""
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    msg = "cannot read MemAvailable"
    raise RuntimeError(msg)


def assert_idle_system() -> dict[str, object]:
    """Fail before timed fits if another compute workload would taint timing."""
    available = _available_memory_bytes()
    load_average = os.getloadavg()[0]
    gpu_processes = command_output(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
    )
    active_gpu_processes = [line for line in gpu_processes.splitlines() if line.strip()]
    snapshot = {
        "utc": utc_now(),
        "available_memory_bytes": available,
        "one_minute_load_average": load_average,
        "gpu_compute_processes": active_gpu_processes,
    }
    if available < MINIMUM_IDLE_MEMORY_BYTES:
        msg = f"system-load gate failed: only {available} bytes available"
        raise RuntimeError(msg)
    if load_average > MAXIMUM_IDLE_LOAD_AVERAGE:
        msg = f"system-load gate failed: load average is {load_average}"
        raise RuntimeError(msg)
    if active_gpu_processes:
        msg = (
            "system-load gate failed: active GPU compute processes "
            f"{active_gpu_processes}"
        )
        raise RuntimeError(
            msg,
        )
    return snapshot


def _output_records(outputs: Sequence[Path]) -> list[dict[str, object]]:
    """Validate and hash every declared stage output."""
    records = []
    for output in outputs:
        if not output.is_file():
            msg = f"stage did not create declared output: {output}"
            raise FileNotFoundError(msg)
        records.append(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "modified_ns": output.stat().st_mtime_ns,
            },
        )
    return records


def _verify_recorded_outputs(
    outputs: Sequence[Path],
    recorded: object,
) -> list[dict[str, object]]:
    """Re-hash sealed outputs and reject missing, extra, or changed records."""
    if not isinstance(recorded, list):
        msg = "sealed stage output records must be a list"
        raise TypeError(msg)
    actual = _output_records(outputs)
    expected_by_path = {
        str(row["path"]): row for row in recorded if isinstance(row, dict)
    }
    if set(expected_by_path) != {str(path) for path in outputs}:
        msg = "sealed stage output ownership differs from the current plan"
        raise RuntimeError(msg)
    for row in actual:
        expected = expected_by_path[str(row["path"])]
        if row["bytes"] != expected.get("bytes") or row["sha256"] != expected.get(
            "sha256",
        ):
            msg = f"sealed stage output changed: {row['path']}"
            raise RuntimeError(msg)
    return actual


def run_stage(paths: ReproductionPaths, stage: Stage) -> dict[str, object]:
    """Run one uncached stage, stream its log, and seal its outputs."""
    record_path = paths.stages / f"{stage.name}.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("status") != "complete":
            msg = f"stage has a non-complete record: {stage.name}"
            raise RuntimeError(msg)
        _verify_recorded_outputs(stage.outputs, record.get("outputs"))
        return record
    pre_existing = [str(path) for path in stage.outputs if path.exists()]
    if pre_existing:
        msg = f"unowned cached outputs exist for {stage.name}: {pre_existing}"
        raise FileExistsError(
            msg,
        )

    load_snapshot = assert_idle_system() if stage.requires_idle_system else None
    started = utc_now()
    started_ns = time.time_ns()
    log_path = paths.logs / f"{stage.name}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "OPENBLAS_NUM_THREADS": "6",
            "NUMEXPR_NUM_THREADS": "6",
            "NUMBA_CACHE_DIR": str(paths.root / "runtime_cache/numba"),
            "MPLCONFIGDIR": str(paths.root / "runtime_cache/matplotlib"),
        },
    )
    Path(environment["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603
            list(stage.command),
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            msg = "stage subprocess has no output stream"
            raise RuntimeError(msg)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)  # noqa: T201
        return_code = process.wait()
    if return_code != 0:
        failed = {
            "name": stage.name,
            "status": "failed",
            "command": list(stage.command),
            "started_utc": started,
            "finished_utc": utc_now(),
            "return_code": return_code,
            "log": str(log_path),
            "pre_stage_load": load_snapshot,
        }
        write_json(record_path, failed)
        raise subprocess.CalledProcessError(return_code, stage.command)

    outputs = _output_records(stage.outputs)
    if any(int(row["modified_ns"]) < started_ns for row in outputs):
        msg = f"stage output predates execution: {stage.name}"
        raise RuntimeError(msg)
    record = {
        "name": stage.name,
        "status": "complete",
        "command": list(stage.command),
        "started_utc": started,
        "finished_utc": utc_now(),
        "return_code": 0,
        "log": str(log_path),
        "pre_stage_load": load_snapshot,
        "outputs": outputs,
    }
    write_json(record_path, record)
    return record


def run(root: Path, *, stop_after: str | None = None) -> dict[str, object]:
    """Execute or resume the clean-room plan through the requested stage."""
    paths = reproduction_paths(root)
    manifest_path = paths.root / "reproduction_manifest.json"
    if not manifest_path.is_file():
        msg = "initialize the clean-room root before running it"
        raise FileNotFoundError(msg)
    manifest = read_json(manifest_path)
    current_source = source_state()
    if current_source["commit"] != manifest["source"]["commit"]:
        msg = "source commit differs from the initialized reproduction"
        raise RuntimeError(msg)

    completed = []
    plan = stage_plan(paths)
    if stop_after is not None and stop_after not in {stage.name for stage in plan}:
        msg = f"unknown stop-after stage: {stop_after}"
        raise ValueError(msg)
    for stage in plan:
        run_stage(paths, stage)
        completed.append(stage.name)
        if stage.name == stop_after:
            break
    manifest["status"] = (
        "raw_results_complete" if len(completed) == len(plan) else "partial"
    )
    manifest["last_completed_stage"] = completed[-1] if completed else None
    manifest["updated_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "completed_stages": completed,
        "total_stages": len(plan),
        "manifest": str(manifest_path),
    }


def status(root: Path) -> dict[str, object]:
    """Return stage status without opening scientific result payloads."""
    paths = reproduction_paths(root)
    stages = stage_plan(paths)
    return {
        "root": str(paths.root),
        "initialized": (paths.root / "reproduction_manifest.json").is_file(),
        "stages": {
            stage.name: (
                read_json(paths.stages / f"{stage.name}.json").get("status")
                if (paths.stages / f"{stage.name}.json").is_file()
                else "pending"
            )
            for stage in stages
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize, execute, or inspect one reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize_command = commands.add_parser("initialize")
    initialize_command.add_argument("--root", required=True, type=Path)
    run_command = commands.add_parser("run")
    run_command.add_argument("--root", required=True, type=Path)
    run_command.add_argument("--stop-after")
    status_command = commands.add_parser("status")
    status_command.add_argument("--root", required=True, type=Path)
    plan_command = commands.add_parser("plan")
    plan_command.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        result = initialize(args.root)
    elif args.command == "run":
        result = run(args.root, stop_after=args.stop_after)
    elif args.command == "status":
        result = status(args.root)
    else:
        result = {
            "stages": [
                {
                    "name": stage.name,
                    "command": list(stage.command),
                    "outputs": [str(path) for path in stage.outputs],
                    "requires_idle_system": stage.requires_idle_system,
                }
                for stage in stage_plan(reproduction_paths(args.root))
            ],
        }
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
