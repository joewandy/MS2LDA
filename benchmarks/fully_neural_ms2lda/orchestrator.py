"""Resumable orchestration for the bounded fully neural study."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .config import REPO_ROOT, initialize_run, verify_run
from .data import load_csr, load_heldout_records
from .embeddings import train_sgns
from .evaluation import (
    evaluate_attempt,
    load_reference_summary,
    nonchemical_hard_gates,
)
from .report import build_report
from .training import prepare_initialization, train_attempt, validation_is_collapsed
from .utils import read_json, write_json


def _heartbeat(directory: Path, stage: str, **details: Any) -> None:
    write_json(
        directory / "heartbeat.json",
        {
            "stage": stage,
            "pid": os.getpid(),
            "torch_cpu_threads": torch.get_num_threads(),
            **details,
        },
    )


def _configure_torch(threads: int) -> None:
    torch.set_num_threads(int(threads))
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _run_chemical_environment(directory: Path, attempt: str) -> None:
    conda = shutil.which("conda")
    if conda is None:
        msg = "conda is required for the pinned MAG environment"
        raise FileNotFoundError(msg)
    log = directory / "logs" / f"chemical_{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "ms2lda-msnlib-mag",
        "python",
        "-m",
        "benchmarks.fully_neural_ms2lda",
        "chemical",
        "--run",
        str(directory),
        "--attempt",
        attempt,
    ]
    environment = os.environ.copy()
    numba_cache = directory / "runtime_cache" / "numba"
    matplotlib_cache = directory / "runtime_cache" / "matplotlib"
    numba_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment["NUMBA_CACHE_DIR"] = str(numba_cache)
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(  # noqa: S603
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_study(
    run_dir: str | Path,
    *,
    source_run: str | Path,
    reference_run: str | Path,
) -> dict[str, Any]:
    """Run, resume, and finish the frozen primary/rescue experiment."""
    directory = Path(run_dir).expanduser().resolve()
    lock = initialize_run(
        directory,
        source_run=source_run,
        reference_run=reference_run,
    )
    verify_run(directory)
    protocol = read_json(directory / "protocol.resolved.json")
    _configure_torch(int(protocol["training_cpu_threads"]))
    counts = Path(lock["source_run"]) / "shared/counts"
    train = load_csr(counts / "train.npz")
    validation_observed = load_csr(counts / "validation_observed.npz")
    validation_completion = load_csr(counts / "validation_completion.npz")
    validation_records = load_heldout_records(counts, "validation")

    _heartbeat(directory, "sgns")
    train_sgns(directory / "sgns", train, protocol["sgns"], seed=protocol["seed"])
    _heartbeat(directory, "initialization")
    prepare_initialization(directory, counts, protocol)
    reference = load_reference_summary(lock["reference_run"])
    write_json(directory / "reference_summary.json", reference)

    _heartbeat(directory, "train_primary")
    primary = train_attempt(
        directory,
        attempt="primary",
        train=train,
        validation_observed=validation_observed,
        validation_completion=validation_completion,
        validation_records=validation_records,
        protocol=protocol,
        reference=reference,
    )
    collapsed, collapse_reasons = validation_is_collapsed(primary, reference, protocol)
    rescue_decision = {
        "schema_version": "fully-neural-ms2lda/rescue-decision-v1",
        "made_before_test_evaluation": True,
        "primary_collapsed": collapsed,
        "collapse_reasons": collapse_reasons,
        "rescue_eligible": collapsed,
        "same_initialization_required": True,
    }
    write_json(directory / "rescue_decision.json", rescue_decision)
    attempts = ["primary"]
    if collapsed:
        _configure_torch(int(protocol["training_cpu_threads"]))
        _heartbeat(directory, "train_rescue", reasons=collapse_reasons)
        train_attempt(
            directory,
            attempt="rescue",
            train=train,
            validation_observed=validation_observed,
            validation_completion=validation_completion,
            validation_records=validation_records,
            protocol=protocol,
            reference=reference,
        )
        attempts.append("rescue")

    _configure_torch(int(protocol["evaluation_cpu_threads"]))
    for attempt in attempts:
        _heartbeat(directory, f"evaluate_{attempt}")
        evaluation = evaluate_attempt(directory, attempt=attempt, protocol=protocol)
        nonchemical = nonchemical_hard_gates(evaluation, reference, protocol)
        write_json(
            directory / "evaluation" / attempt / "nonchemical_hard_gates.json",
            nonchemical,
        )
        if nonchemical["pass"]:
            _heartbeat(directory, f"chemical_{attempt}")
            _run_chemical_environment(directory, attempt)

    _heartbeat(directory, "report")
    report = build_report(
        directory,
        attempts=attempts,
        reference=reference,
        protocol=protocol,
    )
    write_json(
        directory / "complete.json",
        {
            "schema_version": "fully-neural-ms2lda/study-complete-v1",
            "attempts": attempts,
            "decision": report["decision"],
            "selected_attempt": report["selected_attempt"],
            "report": "report/report.json",
        },
    )
    _heartbeat(
        directory,
        "complete",
        decision=report["decision"],
        selected_attempt=report["selected_attempt"],
    )
    return report


def status(run_dir: str | Path) -> dict[str, Any]:
    """Return a compact, read-only progress snapshot."""
    directory = Path(run_dir).expanduser().resolve()
    if not directory.exists():
        return {"run": str(directory), "stage": "not_started"}
    heartbeat = (
        read_json(directory / "heartbeat.json")
        if (directory / "heartbeat.json").is_file()
        else {"stage": "initialized"}
    )
    attempts = {}
    for attempt in ("primary", "rescue"):
        complete = directory / "attempts" / attempt / "complete.json"
        latest = directory / "attempts" / attempt / "checkpoint_latest.pt"
        attempts[attempt] = {
            "complete": complete.is_file(),
            "checkpoint_available": latest.is_file(),
            "evaluation_complete": (
                directory / "evaluation" / attempt / "complete.json"
            ).is_file(),
            "chemical_complete": (
                directory / "chemical" / attempt / "complete.json"
            ).is_file(),
        }
        if complete.is_file():
            result = read_json(complete)
            attempts[attempt].update(
                {
                    "epochs_completed": result["epochs_completed"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "stop_reason": result["stop_reason"],
                },
            )
    return {
        "run": str(directory),
        **heartbeat,
        "attempts": attempts,
        "study_complete": (directory / "complete.json").is_file(),
        "process_python": sys.executable,
    }
