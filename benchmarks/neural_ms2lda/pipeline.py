"""Unattended orchestration for the clean MSnLib workflow."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    REPO_ROOT,
    build_results,
    initialize_run,
)
from .data import (
    load_csr,
    load_view_pairs,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from .evaluation import evaluate_neural
from .tomotopy import evaluate_tomotopy, train_tomotopy
from .training import train_model
from .utils import read_json, write_json


def _heartbeat(directory: Path, **details: Any) -> None:
    """Atomically expose the latest long-running stage without loading arrays."""
    write_json(
        directory / "heartbeat.json",
        {"pid": os.getpid(), "torch_cpu_threads": torch.get_num_threads(), **details},
    )


def _configure_threads(count: int) -> None:
    """Apply one CPU-thread allowance to PyTorch and numerical backends."""
    torch.set_num_threads(int(count))
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(count)


def _chemical_subprocess(
    directory: Path,
    *,
    method: str,
    data_root: str | Path,
    cpu_threads: int,
    split: str,
) -> None:
    """Run MAG in its pinned environment while preserving the thread contract."""
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("conda is required for the pinned MAG environment")
    log = directory / "logs" / f"chemical_{split}_{method}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(directory / "runtime_cache/numba")
    environment["MPLCONFIGDIR"] = str(directory / "runtime_cache/matplotlib")
    Path(environment["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(cpu_threads)
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "ms2lda-msnlib-mag",
        "python",
        "-m",
        "benchmarks.neural_ms2lda.chemical",
        "--run",
        str(directory),
        "--data-root",
        str(Path(data_root).expanduser().resolve()),
        "--method",
        method,
        "--split",
        split,
    ]
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_pipeline(
    run_dir: str | Path,
    *,
    data_root: str | Path,
) -> dict[str, Any]:
    """Run acquisition-independent processing through the canonical results."""
    directory = Path(run_dir).expanduser().resolve()
    protocol = initialize_run(
        directory,
        data_root=data_root,
    )
    cpu_threads = int(protocol["cpu_threads"])
    _configure_threads(cpu_threads)
    _heartbeat(directory, stage="prepare_data")
    prepare_data(directory, data_root=data_root, protocol=protocol)
    data = directory / "data"
    train = load_csr(data / "train.npz")
    _heartbeat(directory, stage="train_token_features")
    train_token_features(
        directory / "token_features",
        train,
        load_vocabulary(data),
        protocol,
        seed=int(protocol["seed"]),
    )
    views = load_view_pairs(directory, protocol)
    validation_full = load_csr(data / "validation_full.npz")
    train_model(
        directory,
        train=train,
        views=views,
        validation_full=validation_full,
        protocol=protocol,
        heartbeat=lambda **details: _heartbeat(directory, **details),
    )
    _heartbeat(directory, stage="train_tomotopy")
    train_tomotopy(
        directory,
        protocol,
        heartbeat=lambda **details: _heartbeat(directory, **details),
    )
    _heartbeat(directory, stage="evaluate_validation_neural")
    evaluate_neural(directory, protocol, split="validation")
    _heartbeat(directory, stage="evaluate_validation_tomotopy")
    evaluate_tomotopy(directory, protocol, split="validation")
    for method in ("neural", "tomotopy"):
        if not (directory / "validation_chemical" / method / "complete.json").is_file():
            _heartbeat(directory, stage="chemical", split="validation", method=method)
            _chemical_subprocess(
                directory,
                method=method,
                data_root=data_root,
                cpu_threads=cpu_threads,
                split="validation",
            )
    _heartbeat(directory, stage="evaluate_test_neural")
    evaluate_neural(directory, protocol, split="test")
    _heartbeat(directory, stage="evaluate_test_tomotopy")
    evaluate_tomotopy(directory, protocol, split="test")
    for method in ("neural", "tomotopy"):
        if not (directory / "chemical" / method / "complete.json").is_file():
            _heartbeat(directory, stage="chemical", split="test", method=method)
            _chemical_subprocess(
                directory,
                method=method,
                data_root=data_root,
                cpu_threads=cpu_threads,
                split="test",
            )
    results = build_results(directory)
    write_json(
        directory / "complete.json",
        {"results": "results.json", "trained_model": "trained_model"},
    )
    _heartbeat(directory, stage="complete")
    return results


def status(run_dir: str | Path) -> dict[str, Any]:
    """Return a compact progress snapshot without loading scientific arrays."""
    directory = Path(run_dir).expanduser().resolve()
    heartbeat = (
        read_json(directory / "heartbeat.json")
        if (directory / "heartbeat.json").is_file()
        else None
    )
    stages = {
        "data": directory / "data/complete.json",
        "token_features": directory / "token_features/complete.json",
        "neural_model": directory / "trained_model/complete.json",
        "tomotopy_model": directory / "tomotopy/complete.json",
        "neural_validation": directory / "validation_evaluation/neural/complete.json",
        "tomotopy_validation": directory
        / "validation_evaluation/tomotopy/complete.json",
        "neural_validation_chemistry": directory
        / "validation_chemical/neural/complete.json",
        "tomotopy_validation_chemistry": directory
        / "validation_chemical/tomotopy/complete.json",
        "neural_evaluation": directory / "evaluation/neural/complete.json",
        "tomotopy_evaluation": directory / "evaluation/tomotopy/complete.json",
        "neural_chemistry": directory / "chemical/neural/complete.json",
        "tomotopy_chemistry": directory / "chemical/tomotopy/complete.json",
        "results": directory / "results.json",
    }
    return {
        "run": str(directory),
        "heartbeat": heartbeat,
        "stages": {name: path.is_file() for name, path in stages.items()},
        "complete": (directory / "complete.json").is_file(),
    }
