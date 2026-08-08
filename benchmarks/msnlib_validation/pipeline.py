"""Unattended, resumable orchestration for one frozen MSnLib run."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import file_sha256, read_json, write_json
from .protocol import verify_protocol


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _completed_artifact(run_dir: Path, relative_path: str) -> dict[str, Any]:
    path = run_dir / relative_path
    if not path.is_file():
        raise RuntimeError(f"pipeline stage did not create {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _run_stage(
    *,
    run_dir: Path,
    name: str,
    command: Sequence[str],
    completion_path: str,
    state: dict[str, Any],
) -> None:
    stages = state.setdefault("stages", {})
    stages[name] = {
        "command": list(command),
        "started_utc": _utc_now(),
        "status": "running",
    }
    write_json(run_dir / "pipeline_state.json", state)
    try:
        subprocess.run(command, check=True)
    except BaseException as exc:
        stages[name].update(
            {
                "completed_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "status": "failed",
            }
        )
        write_json(run_dir / "pipeline_state.json", state)
        raise
    stages[name].update(
        {
            "artifact": _completed_artifact(run_dir, completion_path),
            "completed_utc": _utc_now(),
            "status": "completed",
        }
    )
    write_json(run_dir / "pipeline_state.json", state)


def run_pipeline(
    run_dir: str | Path,
    *,
    data_root: str | Path,
    mag_environment: str = "ms2lda-msnlib-mag",
) -> dict[str, Any]:
    """Run every post-freeze stage, safely resuming completed work.

    Hybrid/DreaMS stages run in fresh processes using the current Python
    interpreter. MAG runs in its separately pinned Conda environment. Each
    underlying stage remains responsible for validating and resuming its own
    artifacts; this function records their ordering and durable completion.
    """
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    frozen_data_root = Path(lock["data_root"]).expanduser().resolve()
    requested_data_root = Path(data_root).expanduser().resolve()
    if frozen_data_root != requested_data_root:
        raise ValueError("pipeline data root differs from the frozen protocol")
    if not mag_environment.strip():
        raise ValueError("MAG Conda environment cannot be empty")

    state_path = directory / "pipeline_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("pipeline state belongs to another frozen protocol")
    else:
        state = {
            "created_utc": _utc_now(),
            "data_root": str(frozen_data_root),
            "mag_environment": mag_environment,
            "protocol_sha256": lock["protocol_sha256"],
            "stages": {},
        }
    state.update(
        {
            "last_process_id": os.getpid(),
            "last_started_utc": _utc_now(),
            "status": "running",
        }
    )
    write_json(state_path, state)

    module = "benchmarks.msnlib_validation"
    stages = (
        (
            "core",
            [sys.executable, "-m", module, "run-core", "--run", str(directory)],
            "core/complete.json",
        ),
        (
            "chemical_inference",
            [
                sys.executable,
                "-m",
                module,
                "run-chemical-inference",
                "--run",
                str(directory),
            ],
            "chemical_inference/complete.json",
        ),
        (
            "raw_dreams",
            [
                sys.executable,
                "-m",
                module,
                "_run-raw-dreams",
                "--run",
                str(directory),
            ],
            "mag/raw_dreams/complete.json",
        ),
        (
            "mag",
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                mag_environment,
                "python",
                "-m",
                module,
                "run-mag",
                "--run",
                str(directory),
                "--data-root",
                str(frozen_data_root),
            ],
            "mag/complete.json",
        ),
        (
            "report",
            [sys.executable, "-m", module, "report", "--run", str(directory)],
            "report/complete.json",
        ),
    )
    try:
        for name, command, completion_path in stages:
            _run_stage(
                run_dir=directory,
                name=name,
                command=command,
                completion_path=completion_path,
                state=state,
            )
    except BaseException:
        state.update({"completed_utc": _utc_now(), "status": "failed"})
        write_json(state_path, state)
        raise
    state.update({"completed_utc": _utc_now(), "status": "completed"})
    write_json(state_path, state)
    return state
