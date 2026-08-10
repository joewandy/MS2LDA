# ruff: noqa: PLR0915
"""Sequential, resumable, failure-isolating overnight orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.msnlib_validation.config import file_sha256, read_json, write_json

from .spec import ARM_IDS, load_spec, verify_study


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Task:
    """One isolated worker command and its dependency names."""

    name: str
    arguments: tuple[str, ...]
    requires: tuple[str, ...] = ()
    environment: str = "ms2lda-hybrid"


def _tasks() -> tuple[Task, ...]:
    tasks: list[Task] = [
        Task("prepare_counts", ("prepare-counts",)),
        Task("discovery_current", ("import-current-discovery",), ("prepare_counts",)),
        Task(
            "targets_current",
            ("build-targets", "--discovery", "dreams_prior"),
            ("discovery_current",),
        ),
    ]
    arm_tasks: list[str] = []
    for discovery in ("dreams_prior", "symmetric_prior"):
        if discovery == "symmetric_prior":
            tasks.extend(
                (
                    Task(
                        "discovery_symmetric",
                        ("run-symmetric-discovery",),
                        ("prepare_counts",),
                    ),
                    Task(
                        "targets_symmetric",
                        ("build-targets", "--discovery", "symmetric_prior"),
                        ("discovery_symmetric",),
                    ),
                ),
            )
        discovery_task = (
            "discovery_current"
            if discovery == "dreams_prior"
            else "discovery_symmetric"
        )
        target_task = (
            "targets_current" if discovery == "dreams_prior" else "targets_symmetric"
        )
        for arm_id in (value for value in ARM_IDS if value.startswith(discovery)):
            _, inference = arm_id.split("__", 1)
            dependencies = [discovery_task]
            if inference.endswith("_direct"):
                dependencies.append(target_task)
            name = f"arm_{arm_id}"
            arm_tasks.append(name)
            tasks.append(
                Task(name, ("train-arm", "--arm", arm_id), tuple(dependencies)),
            )
    tasks.extend(
        (
            Task("freeze_models", ("freeze-models",), tuple(arm_tasks)),
            Task(
                "full_validation_dreams",
                ("prepare-full-validation-dreams",),
                ("freeze_models",),
            ),
            Task(
                "validation",
                ("finalize-validation",),
                ("freeze_models", "full_validation_dreams"),
            ),
            Task("test", ("finalize-test",), ("validation",)),
            Task(
                "annotations_current",
                ("annotate", "--discovery", "dreams_prior"),
                ("test",),
            ),
            Task(
                "annotations_symmetric",
                ("annotate", "--discovery", "symmetric_prior"),
                ("test",),
                environment="ms2lda-msnlib-mag",
            ),
            Task(
                "chemical_scores",
                ("score-chemical",),
                ("annotations_current", "annotations_symmetric"),
            ),
            Task("report", ("report",), ("chemical_scores",)),
            Task("verify", ("verify",), ("report",)),
        ),
    )
    return tuple(tasks)


def _append_event(directory: Path, value: dict[str, Any]) -> None:
    path = directory / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _process_stats(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"child_pid": None, "cpu_percent": None, "rss_bytes": None}
    result = subprocess.run(  # noqa: S603 - fixed executable and numeric PID
        ["/bin/ps", "-o", "%cpu=,rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != len(("cpu", "rss")):
        return {"child_pid": pid, "cpu_percent": None, "rss_bytes": None}
    return {
        "child_pid": pid,
        "cpu_percent": float(fields[0]),
        "rss_bytes": int(fields[1]) * 1024,
    }


def _heartbeat_loop(
    directory: Path,
    shared: dict[str, Any],
    stop: threading.Event,
    interval: int,
) -> None:
    while not stop.is_set():
        stat = os.statvfs(directory)
        heartbeat = {
            "created_utc": _utc_now(),
            "runner_pid": os.getpid(),
            "stage": shared.get("stage"),
            "stage_started_monotonic": shared.get("stage_started"),
            "runner_elapsed_seconds": time.monotonic() - shared["runner_started"],
            "free_disk_bytes": stat.f_bavail * stat.f_frsize,
            **_process_stats(shared.get("child_pid")),
        }
        write_json(directory / "heartbeat.json", heartbeat)
        if stop.wait(interval):
            break


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _acquire_lock(directory: Path) -> Path:
    path = directory / "runner.lock"
    if path.exists():
        try:
            existing = int(read_json(path)["pid"])
        except (KeyError, OSError, TypeError, ValueError):
            existing = -1
        if existing > 0 and _pid_alive(existing):
            msg = f"overnight runner is already active as PID {existing}"
            raise RuntimeError(msg)
        stale = directory / f"runner.lock.stale.{int(time.time())}.json"
        path.replace(stale)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_utc": _utc_now()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _command(task: Task, directory: Path) -> list[str]:
    base = [
        "python",
        "-m",
        "benchmarks.msnlib_simplification",
        *task.arguments,
        "--run",
        str(directory),
    ]
    if task.environment == os.environ.get("CONDA_DEFAULT_ENV"):
        return [
            sys.executable,
            "-m",
            "benchmarks.msnlib_simplification",
            *task.arguments,
            "--run",
            str(directory),
        ]
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        task.environment,
        *base,
    ]


def run_overnight(run_dir: str | Path) -> dict[str, Any]:
    """Run all stages sequentially and preserve independent failure evidence."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    spec = load_spec(directory)
    lock_path = _acquire_lock(directory)
    state_path = directory / "run_state.json"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("spec_sha256") != lock["spec_sha256"]:
            msg = "runner state belongs to another specification"
            raise ValueError(msg)
    else:
        state = {
            "schema_version": "msnlib-simplification/runner-state-v1",
            "created_utc": _utc_now(),
            "spec_sha256": lock["spec_sha256"],
            "tasks": {},
        }
    state.update(
        {
            "status": "running",
            "last_runner_pid": os.getpid(),
            "last_started_utc": _utc_now(),
        },
    )
    write_json(state_path, state)
    shared: dict[str, Any] = {
        "runner_started": time.monotonic(),
        "stage": "starting",
        "stage_started": time.monotonic(),
        "child_pid": None,
    }
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(directory, shared, stop, spec.heartbeat_seconds),
        daemon=True,
    )
    heartbeat.start()
    completed: set[str] = {
        name
        for name, value in state.get("tasks", {}).items()
        if value.get("status") == "completed"
    }
    try:
        for task in _tasks():
            previous = state["tasks"].get(task.name, {})
            if previous.get("status") == "completed":
                completed.add(task.name)
                continue
            unmet = [name for name in task.requires if name not in completed]
            if unmet:
                state["tasks"][task.name] = {
                    "status": "skipped",
                    "completed_utc": _utc_now(),
                    "unmet_dependencies": unmet,
                }
                write_json(state_path, state)
                _append_event(
                    directory,
                    {
                        "created_utc": _utc_now(),
                        "event": "skipped",
                        "task": task.name,
                        "unmet": unmet,
                    },
                )
                continue
            command = _command(task, directory)
            log_path = directory / "logs" / f"{task.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            task_state = {
                "status": "running",
                "started_utc": _utc_now(),
                "command": command,
                "log": str(log_path.relative_to(directory)),
                "attempt": int(previous.get("attempt", 0)) + 1,
            }
            state["tasks"][task.name] = task_state
            write_json(state_path, state)
            _append_event(
                directory,
                {"created_utc": _utc_now(), "event": "started", "task": task.name},
            )
            shared.update(
                {
                    "stage": task.name,
                    "stage_started": time.monotonic(),
                    "child_pid": None,
                },
            )
            started = time.perf_counter()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n[{_utc_now()}] starting attempt {task_state['attempt']}\n",
                )
                handle.flush()
                process = subprocess.Popen(  # noqa: S603 - frozen internal task argv
                    command,
                    cwd=lock["repo_root"],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                shared["child_pid"] = process.pid
                returncode = process.wait()
            shared["child_pid"] = None
            task_state.update(
                {
                    "completed_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "returncode": returncode,
                    "status": "completed" if returncode == 0 else "failed",
                },
            )
            write_json(state_path, state)
            _append_event(
                directory,
                {
                    "created_utc": _utc_now(),
                    "event": task_state["status"],
                    "task": task.name,
                    "returncode": returncode,
                },
            )
            if returncode == 0:
                completed.add(task.name)
        required = {task.name for task in _tasks()}
        statuses = {
            name: state["tasks"].get(name, {}).get("status", "missing")
            for name in sorted(required)
        }
        succeeded = [name for name, status in statuses.items() if status == "completed"]
        failed = [name for name, status in statuses.items() if status == "failed"]
        skipped = [name for name, status in statuses.items() if status == "skipped"]
        missing = [name for name, status in statuses.items() if status == "missing"]
        complete = len(succeeded) == len(required)
        sentinel = {
            "schema_version": "msnlib-simplification/overnight-complete-v1",
            "created_utc": _utc_now(),
            "complete": complete,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "missing": missing,
            "report_sha256": (
                file_sha256(directory / "report/complete.json")
                if (directory / "report/complete.json").is_file()
                else None
            ),
            "verification_sha256": (
                file_sha256(directory / "verification.json")
                if (directory / "verification.json").is_file()
                else None
            ),
            "interpretation_performed": False,
        }
        write_json(directory / "overnight_complete.json", sentinel)
        state.update(
            {
                "completed_utc": _utc_now(),
                "status": "completed" if complete else "incomplete",
                "overnight_complete_sha256": file_sha256(
                    directory / "overnight_complete.json",
                ),
            },
        )
        write_json(state_path, state)
        if not complete:
            msg = f"overnight collection incomplete: failed={failed}, skipped={skipped}"
            raise RuntimeError(
                msg,
            )
        return sentinel
    finally:
        shared["stage"] = "finished"
        stop.set()
        heartbeat.join(timeout=5)
        lock_path.unlink(missing_ok=True)


def status(run_dir: str | Path) -> dict[str, Any]:
    """Return a concise health/status snapshot for humans and automation."""
    directory = Path(run_dir).expanduser().resolve()
    result: dict[str, Any] = {
        "run": str(directory),
        "lock_present": (directory / "runner.lock").is_file(),
    }
    if (directory / "runner.lock").is_file():
        lock = read_json(directory / "runner.lock")
        result["runner_pid"] = lock.get("pid")
        result["runner_alive"] = bool(
            isinstance(lock.get("pid"), int) and _pid_alive(lock["pid"]),
        )
    if (directory / "heartbeat.json").is_file():
        result["heartbeat"] = read_json(directory / "heartbeat.json")
    if (directory / "run_state.json").is_file():
        state = read_json(directory / "run_state.json")
        result["status"] = state.get("status")
        result["tasks"] = {
            name: value.get("status") for name, value in state.get("tasks", {}).items()
        }
        result["completed_tasks"] = sum(
            value == "completed" for value in result["tasks"].values()
        )
        result["total_tasks"] = len(_tasks())
    if (directory / "overnight_complete.json").is_file():
        result["completion"] = read_json(directory / "overnight_complete.json")
    return result
