"""Backfill the expanded diagnostic contract for an existing validation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.neural_ms2lda.data import load_vocabulary
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.utils import read_json, write_json


def backfill(run_dir: str | Path, *, update_complete: bool = False) -> dict[str, Any]:
    """Compute validation-only diagnostics from locked beta/theta artifacts."""
    run = Path(run_dir).expanduser().resolve()
    evaluation = run / "validation_evaluation/neural"
    beta_path = evaluation / "beta.npy"
    theta_path = evaluation / "validation_full_theta.npy"
    if not beta_path.is_file() or not theta_path.is_file():
        raise FileNotFoundError("validation beta/theta artifacts are required")
    protocol = read_json(run / "protocol.json")
    beta = np.load(beta_path, mmap_mode="r")
    theta = np.load(theta_path, mmap_mode="r")
    vocabulary = load_vocabulary(run / "data")
    diagnostics = model_selection_diagnostics(
        theta,
        beta,
        vocabulary,
        protocol["evaluation"],
    )
    write_json(evaluation / "diagnostics.json", diagnostics)
    if update_complete:
        complete_path = evaluation / "complete.json"
        complete = read_json(complete_path)
        complete.setdefault("metrics", {}).update(diagnostics)
        write_json(complete_path, complete)
    return {
        "run": str(run),
        "split": "validation",
        "updated_complete": bool(update_complete),
        "diagnostics": diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--update-complete", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(backfill(args.run, update_complete=args.update_complete), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
