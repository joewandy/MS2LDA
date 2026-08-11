# ruff: noqa: C901, PLR0913, PLR0915, S603
"""Dependency-gated unattended orchestration for the bounded study."""

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
from .data import (
    load_csr,
    load_heldout_records,
    load_view_pairs,
    prepare_training_views,
)
from .embeddings import train_sgns
from .evaluation import (
    evaluate_selected,
    load_reference_summary,
    nonchemical_test_gates,
)
from .report import build_report, chemical_gate_checks
from .synthetic import run_synthetic_gate
from .training import (
    benchmark_batch_sizes,
    diagnose_collapse,
    prepare_initialization,
    prepare_token_features,
    train_attempt,
)
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


def _require_merged_clean_main(lock: dict[str, Any]) -> None:
    git = lock["git"]
    if git["branch"] != "main" or git["status_porcelain"]:
        msg = "the scientific runner must start from a clean merged fork main"
        raise RuntimeError(
            msg,
        )


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
        "benchmarks.neural_assignment_ms2lda",
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
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _finish(
    directory: Path,
    *,
    decision: str,
    furthest_stage: str,
    selected_attempt: str | None,
    protocol: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    report = build_report(
        directory,
        decision=decision,
        furthest_stage=furthest_stage,
        selected_attempt=selected_attempt,
        protocol=protocol,
        reference=reference,
    )
    write_json(
        directory / "complete.json",
        {
            "schema_version": "neural-assignment-ms2lda/study-complete-v1",
            "decision": decision,
            "furthest_stage": furthest_stage,
            "selected_attempt": selected_attempt,
            "report": "report/report.json",
        },
    )
    _heartbeat(
        directory,
        "complete",
        decision=decision,
        furthest_stage=furthest_stage,
        selected_attempt=selected_attempt,
    )
    return report


def _validation_selection(
    directory: Path,
    attempts: list[str],
) -> dict[str, Any]:
    candidates = []
    for attempt in attempts:
        result = read_json(
            directory / "stages/k1000/attempts" / attempt / "complete.json",
        )
        gate = result["validation_gate"]
        nll = float(
            result["selected_validation"]["document_completion"]["nll_per_token"],
        )
        candidates.append(
            {
                "attempt": attempt,
                "gate": gate,
                "validation_nll": nll,
                "selected_epoch": int(result["selected_epoch"]),
            },
        )
    candidates.sort(
        key=lambda row: (
            not row["gate"]["pass"],
            len(row["gate"]["failed"]),
            row["validation_nll"],
            row["attempt"] != "primary",
        ),
    )
    selected = candidates[0]
    return {
        "schema_version": "neural-assignment-ms2lda/validation-selection-v1",
        "decision_final_before_test": True,
        "test_metrics_inspected": False,
        "candidates": candidates,
        "selected_attempt": selected["attempt"],
        "selected_validation_gate": selected["gate"],
        "selection_rule": "pass_first_then_fewest_gate_failures_then_lowest_nll",
    }


def run_study(
    run_dir: str | Path,
    *,
    source_run: str | Path,
    reference_run: str | Path,
) -> dict[str, Any]:
    """Run or exactly resume every eligible stage and stop automatically."""
    directory = Path(run_dir).expanduser().resolve()
    lock = initialize_run(
        directory,
        source_run=source_run,
        reference_run=reference_run,
    )
    verify_run(directory)
    _require_merged_clean_main(lock)
    protocol = read_json(directory / "protocol.resolved.json")
    _configure_torch(int(protocol["training_cpu_threads"]))
    counts = Path(lock["source_run"]) / "shared/counts"
    train = load_csr(counts / "train.npz")
    validation_observed = load_csr(counts / "validation_observed.npz")
    validation_completion = load_csr(counts / "validation_completion.npz")
    validation_full = load_csr(counts / "validation_full.npz")
    validation_records = load_heldout_records(counts, "validation")
    reference = load_reference_summary(lock["reference_run"])
    write_json(directory / "reference_summary.json", reference)

    _heartbeat(directory, "synthetic")
    synthetic = run_synthetic_gate(directory, protocol=protocol)
    if not synthetic["pass"]:
        return _finish(
            directory,
            decision="stop_after_genuine_synthetic_failure",
            furthest_stage="synthetic",
            selected_attempt=None,
            protocol=protocol,
            reference=reference,
        )

    _heartbeat(directory, "sgns")
    train_sgns(
        directory / "sgns",
        train,
        protocol["sgns"],
        seed=int(protocol["seed"]),
    )
    _heartbeat(directory, "token_features")
    prepare_token_features(directory, counts_dir=counts, protocol=protocol)
    _heartbeat(directory, "physical_peak_views")
    prepare_training_views(
        directory,
        counts_dir=counts,
        reference_run=lock["reference_run"],
        protocol=protocol,
    )
    views = load_view_pairs(directory, protocol)

    _heartbeat(directory, "initialize_k200")
    prepare_initialization(
        directory,
        label="k200",
        num_topics=int(protocol["stages"]["k200"]["num_topics"]),
        train=train,
        protocol=protocol,
    )
    _heartbeat(directory, "initialize_k1000")
    prepare_initialization(
        directory,
        label="k1000",
        num_topics=int(protocol["stages"]["k1000"]["num_topics"]),
        train=train,
        protocol=protocol,
    )
    _heartbeat(directory, "batch_benchmark")
    batch_benchmark = benchmark_batch_sizes(
        directory,
        views=views,
        protocol=protocol,
    )
    batch_size = int(batch_benchmark["selected_batch_size"])

    _heartbeat(directory, "train_k200")
    k200 = train_attempt(
        directory,
        stage="k200",
        attempt="primary",
        initialization_label="k200",
        train=train,
        views=views,
        validation_observed=validation_observed,
        validation_completion=validation_completion,
        validation_full=validation_full,
        validation_records=validation_records,
        protocol=protocol,
        batch_size=batch_size,
    )
    if not k200["validation_gate"]["pass"]:
        return _finish(
            directory,
            decision="stop_after_k200_viability_failure",
            furthest_stage="k200",
            selected_attempt=None,
            protocol=protocol,
            reference=reference,
        )

    _heartbeat(directory, "train_k1000_primary")
    primary = train_attempt(
        directory,
        stage="k1000",
        attempt="primary",
        initialization_label="k1000",
        train=train,
        views=views,
        validation_observed=validation_observed,
        validation_completion=validation_completion,
        validation_full=validation_full,
        validation_records=validation_records,
        protocol=protocol,
        batch_size=batch_size,
    )
    collapse, rescue_mode, collapse_reasons = diagnose_collapse(
        primary["validation_gate"],
    )
    rescue_decision = {
        "schema_version": "neural-assignment-ms2lda/rescue-decision-v1",
        "made_before_test_evaluation": True,
        "primary_gate_pass": primary["validation_gate"]["pass"],
        "primary_collapsed": collapse,
        "collapse_reasons": collapse_reasons,
        "rescue_mode": rescue_mode,
        "rescue_eligible": bool(collapse and not primary["validation_gate"]["pass"]),
        "same_initialization_required": True,
        "maximum_total_attempts": 2,
    }
    write_json(directory / "rescue_decision.json", rescue_decision)
    attempts = ["primary"]
    if rescue_decision["rescue_eligible"]:
        _configure_torch(int(protocol["training_cpu_threads"]))
        _heartbeat(
            directory,
            "train_k1000_rescue",
            rescue_mode=rescue_mode,
            reasons=collapse_reasons,
        )
        train_attempt(
            directory,
            stage="k1000",
            attempt="rescue",
            initialization_label="k1000",
            train=train,
            views=views,
            validation_observed=validation_observed,
            validation_completion=validation_completion,
            validation_full=validation_full,
            validation_records=validation_records,
            protocol=protocol,
            batch_size=batch_size,
            rescue_mode=rescue_mode,
        )
        attempts.append("rescue")

    selection = _validation_selection(directory, attempts)
    write_json(directory / "validation_selection.json", selection)
    selected_attempt = str(selection["selected_attempt"])
    _configure_torch(int(protocol["evaluation_cpu_threads"]))
    _heartbeat(directory, "evaluate_selected", attempt=selected_attempt)
    evaluation = evaluate_selected(
        directory,
        attempt=selected_attempt,
        protocol=protocol,
    )
    nonchemical = nonchemical_test_gates(evaluation, protocol)
    if nonchemical["pass"]:
        _heartbeat(directory, "chemical", attempt=selected_attempt)
        _run_chemical_environment(directory, selected_attempt)
        chemical = read_json(
            directory / "chemical" / selected_attempt / "complete.json",
        )
        chemical_gate = chemical_gate_checks(chemical, protocol)
    else:
        chemical_gate = None

    if nonchemical["pass"] and chemical_gate is not None and chemical_gate["pass"]:
        decision = "retain_viable_neural_assignment_model"
        furthest = "chemical"
    elif nonchemical["pass"]:
        decision = "nonchemical_viable_but_chemical_gate_failed"
        furthest = "chemical"
    else:
        decision = "bounded_k1000_attempt_failed_return_to_model_design"
        furthest = "k1000_test"
    return _finish(
        directory,
        decision=decision,
        furthest_stage=furthest,
        selected_attempt=selected_attempt,
        protocol=protocol,
        reference=reference,
    )


def status(run_dir: str | Path) -> dict[str, Any]:
    """Return a compact read-only progress snapshot."""
    directory = Path(run_dir).expanduser().resolve()
    if not directory.exists():
        return {"run": str(directory), "stage": "not_started"}
    heartbeat = (
        read_json(directory / "heartbeat.json")
        if (directory / "heartbeat.json").is_file()
        else {"stage": "initialized"}
    )
    stages: dict[str, Any] = {
        "synthetic_complete": (directory / "stages/synthetic/complete.json").is_file(),
        "batch_benchmark_complete": (directory / "batch_benchmark.json").is_file(),
        "k200_complete": (
            directory / "stages/k200/attempts/primary/complete.json"
        ).is_file(),
        "validation_selection_complete": (
            directory / "validation_selection.json"
        ).is_file(),
        "test_accessed": (directory / "test_access.json").is_file(),
    }
    for attempt in ("primary", "rescue"):
        root = directory / "stages/k1000/attempts" / attempt
        stages[f"k1000_{attempt}"] = {
            "checkpoint_available": (root / "checkpoint_latest.pt").is_file(),
            "complete": (root / "complete.json").is_file(),
            "evaluation_complete": (
                directory / "evaluation" / attempt / "complete.json"
            ).is_file(),
            "chemical_complete": (
                directory / "chemical" / attempt / "complete.json"
            ).is_file(),
        }
        if (root / "complete.json").is_file():
            result = read_json(root / "complete.json")
            stages[f"k1000_{attempt}"].update(
                {
                    "epochs_completed": result["epochs_completed"],
                    "selected_epoch": result["selected_epoch"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "gate_pass": result["validation_gate"]["pass"],
                },
            )
    return {
        "run": str(directory),
        **heartbeat,
        "stages": stages,
        "study_complete": (directory / "complete.json").is_file(),
        "process_python": sys.executable,
    }
