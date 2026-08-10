"""Frozen CPU-thread policy for preparation, training, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .spec import load_spec, verify_study

ThreadPhase = Literal["training", "evaluation"]


def configure_cpu_threads(run_dir: str | Path, phase: ThreadPhase) -> int:
    """Apply and return the frozen CPU-thread count for one benchmark phase."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    spec = load_spec(directory)
    if phase == "training":
        threads = spec.training_cpu_threads
    elif phase == "evaluation":
        threads = spec.evaluation_cpu_threads
    else:
        msg = f"unknown CPU-thread phase: {phase}"
        raise ValueError(msg)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return threads
