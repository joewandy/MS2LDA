"""Run the five approved validation-only baseline fits with bounded concurrency.

Three independent, one-worker Tomotopy fits run alongside one GPU/annotation
queue. Each fit has a fresh sealed view, immutable execution-source snapshot
and durable stage log. The shared lock serializes neural fitting and chemistry.
This is a fixed experiment runner, not a model-selection or tuning framework.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from benchmarks.neural_ms2lda.baseline_repeats import runtime_versions
from benchmarks.neural_ms2lda.utils import sha256_file, write_json

REPO = Path(__file__).resolve().parents[1]


def utc_now():
    """Record wall-clock provenance separately from measured training durations."""
    return datetime.now(timezone.utc).isoformat()


def snapshot_source(root: Path):
    """Freeze executed Python bytes so concurrent editorial work cannot affect fits."""
    destination = root / "source_snapshot"
    destination.mkdir()
    sources = []
    for package in ("benchmarks", "scripts", "MS2LDA"):
        for source in sorted((REPO / package).rglob("*.py")):
            relative = source.relative_to(REPO)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources.append({"path": str(relative), "sha256": sha256_file(target)})
    for name in ("environment.yml", "pyproject.toml"):
        shutil.copy2(REPO / name, destination / name)
        sources.append({"path": name, "sha256": sha256_file(destination / name)})
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record execution-source provenance")
    revision = subprocess.check_output(  # noqa: S603 - fixed read-only Git command
        [git, "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    status = subprocess.check_output(  # noqa: S603 - fixed read-only Git command
        [git, "status", "--porcelain=v1"], cwd=REPO, text=True
    )
    write_json(
        root / "source_manifest.json",
        {
            "created_utc": utc_now(),
            "revision": revision,
            "dirty": bool(status.strip()),
            "git_status": status,
            "python": sys.executable,
            "runtime": runtime_versions(),
            "sources": sources,
        },
    )
    return destination


def run_stage(root, snapshot, run, name, module, arguments):
    """Persist an attempt record before launch and leave failed outputs intact."""
    log_path = root / "logs" / f"{run.name}.{name}.log"
    record_path = root / "stages" / f"{run.name}.{name}.json"
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    record = {
        "run": str(run),
        "stage": name,
        "attempt": 1,
        "command": command,
        "cwd": str(snapshot),
        "started_utc": utc_now(),
        "log": str(log_path),
        "status": "starting",
    }
    write_json(record_path, record)
    environment = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONUNBUFFERED="1")
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = (
                subprocess.Popen(  # noqa: S603 - fixed modules, no shell evaluation
                    command,
                    cwd=snapshot,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
            record.update(pid=process.pid, status="running")
            write_json(record_path, record)
            print(
                f"START {run.name} {name} pid={process.pid} log={log_path}", flush=True
            )
            code = process.wait()
    except Exception as error:
        # A bookkeeping failure must not leave an unobserved child continuing to
        # write artifacts. Terminate/reap only the process created by this call.
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait()
        record.update(
            finished_utc=utc_now(),
            status="failed",
            return_code=None if process is None else process.returncode,
            error=repr(error),
        )
        write_json(record_path, record)
        raise RuntimeError(
            f"{run.name}/{name} could not complete its launch record"
        ) from error
    record.update(
        finished_utc=utc_now(),
        return_code=code,
        status="complete" if code == 0 else "failed",
    )
    write_json(record_path, record)
    print(f"END {run.name} {name} exit={code}", flush=True)
    if code:
        raise RuntimeError(f"{run.name}/{name} failed; retain its log and artifacts")


def run_queue(kind, seeds, root, snapshot, prepared, assets, annotation_lock):
    """Run each prescribed fit once; never replace a failed seed with another."""
    failures = []
    for seed in seeds:
        run = root / "real" / f"{kind}_seed{seed}_attempt1"
        try:
            run_stage(
                root,
                snapshot,
                run,
                "seal",
                "scripts.prepare_msnlib_validation_view",
                ["--run", run, "--prepared-run", prepared],
            )
            with annotation_lock if kind == "etm" else nullcontext():
                if kind == "etm":
                    module = "scripts.run_etm_controls"
                    arguments = [
                        "train",
                        "--run",
                        run,
                        "--method",
                        "etm",
                        "--device",
                        "cuda",
                        "--epochs",
                        120,
                        "--batch-size",
                        256,
                    ]
                else:
                    module = "scripts.run_tomotopy_validation"
                    arguments = ["--run", run]
                run_stage(
                    root,
                    snapshot,
                    run,
                    "fit_validation",
                    module,
                    [*arguments, "--training-seed", seed],
                )
            with annotation_lock:
                run_stage(
                    root,
                    snapshot,
                    run,
                    "chemistry",
                    "benchmarks.neural_ms2lda.chemical",
                    [
                        "--run",
                        run,
                        "--data-root",
                        assets,
                        "--method",
                        kind,
                        "--split",
                        "validation",
                    ],
                )
        except RuntimeError as error:
            failures.append({"kind": kind, "training_seed": seed, "error": str(error)})
    return failures


def main():
    """Freeze sources and run the approved three-LDA/two-ETM fit inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "stages").mkdir()
    snapshot = snapshot_source(root)
    driver = {"pid": os.getpid(), "started_utc": utc_now(), "status": "running"}
    write_json(root / "driver.json", driver)
    annotation_lock = Lock()
    # LDA fits are scientifically independent: concurrency changes wall-clock
    # scheduling, not the one-worker sampling recipe inside any fitted model.
    # ETM remains a single queue so GPU fitting and chemistry cannot overlap.
    plan = (
        ("tomotopy", (11,)),
        ("tomotopy", (23,)),
        ("tomotopy", (42,)),
        ("etm", (7012, 7024)),
    )
    with ThreadPoolExecutor(max_workers=4) as workers:
        queues = [
            workers.submit(
                run_queue,
                kind,
                seeds,
                root,
                snapshot,
                args.prepared_run.resolve(),
                args.data_root.resolve(),
                annotation_lock,
            )
            for kind, seeds in plan
        ]
        failures = [failure for queue in queues for failure in queue.result()]
    driver.update(
        finished_utc=utc_now(),
        status="failed" if failures else "complete",
        failures=failures,
    )
    write_json(root / "driver.json", driver)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
