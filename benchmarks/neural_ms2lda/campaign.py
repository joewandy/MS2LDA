"""Validation-only runner for the bounded minimal-neural campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import PACKAGE_ROOT, initialize_run, load_trained_model
from .data import load_csr, load_vocabulary, prepare_data, train_token_features
from .evaluation import evaluate_neural
from .mag import build_filtered_mag_index
from .objectives import prepare_cooccurrence_graph
from .pipeline import _chemical_subprocess, _configure_threads
from .training import train_model
from .utils import read_json, write_json

DATA_FILES = (
    "complete.json",
    "train.npz",
    "validation_completion.npz",
    "validation_full.npz",
    "validation_observed.npz",
    "validation_records.jsonl",
    "vocabulary.json",
)
ABSOLUTE_FLOORS = {
    "optimized_motifs": 650,
    "evaluable_motifs": 296,
    "useful_motifs": 176,
    "mean_sos": 0.6223,
}
STANDARD_RETENTION = {
    "optimized_motifs": 0.95,
    "evaluable_motifs": 0.95,
    "useful_motifs": 0.95,
    "mean_sos": 0.99,
}
TIE_RETENTION = {
    "optimized_motifs": 0.94,
    "evaluable_motifs": 0.94,
    "useful_motifs": 0.94,
    "mean_sos": 0.98,
}
NLL_REPORTING_REFERENCE = 1.01
NLL_SELECTION_CEILING = 1.05
U1_BASELINE = {
    "optimized_motifs": 843,
    "evaluable_motifs": 429,
    "useful_motifs": 268,
    "mean_sos": 0.6506700669726432,
    "median_sos": 0.6440677966101696,
    "validation_nll": 8.832002635285642,
}


def _copy_path(source: Path, destination: Path) -> None:
    """Copy an immutable shared artifact into a fresh candidate run."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _test_data_files(run: Path) -> list[str]:
    """Return forbidden test-split files materialized in a candidate run."""
    data = run / "data"
    return sorted(path.name for path in data.glob("test*") if path.is_file())


def prepare_shared(run_dir: str | Path, *, data_root: str | Path) -> dict[str, Any]:
    """Create the one full-data staging bundle used to seal candidate runs."""
    run = Path(run_dir).expanduser().resolve()
    protocol = initialize_run(run, data_root=data_root)
    _configure_threads(int(protocol["cpu_threads"]))
    prepare_data(run, data_root=data_root, protocol=protocol)
    data = run / "data"
    train = load_csr(data / "train.npz")
    train_token_features(
        run / "token_features",
        train,
        load_vocabulary(data),
        protocol,
        seed=int(protocol["seed"]),
    )
    prepare_cooccurrence_graph(run, train=train, protocol=protocol)

    accepted = PACKAGE_ROOT / "results/seed42/trained_model"
    state = torch.load(accepted / "weights.pt", map_location="cpu", weights_only=True)
    generated = np.load(run / "token_features/features.npy")
    if not np.array_equal(generated, state["token_features"].cpu().numpy()):
        raise ValueError("generated token features differ from accepted U1")
    if load_vocabulary(data) != list(map(str, read_json(accepted / "vocabulary.json"))):
        raise ValueError("generated vocabulary differs from accepted U1")
    result = {
        "test_materialization": "staging only",
        "token_features_match_u1": True,
        "vocabulary_matches_u1": True,
    }
    write_json(run / "campaign_shared.json", result)
    return result


def prepare_shared_index(
    run_dir: str | Path,
    *,
    data_root: str | Path,
) -> dict[str, Any]:
    """Build the leakage-filtered MAG index before test records are sealed away."""
    run = Path(run_dir).expanduser().resolve()
    protocol = read_json(run / "protocol.json")
    return build_filtered_mag_index(run, data_root=data_root, protocol=protocol)


def initialize_candidate(
    run_dir: str | Path,
    *,
    shared_dir: str | Path,
    data_root: str | Path,
    candidate: str,
) -> dict[str, Any]:
    """Create a run containing train/validation evidence but no test records."""
    run = Path(run_dir).expanduser().resolve()
    shared = Path(shared_dir).expanduser().resolve()
    protocol = initialize_run(run, data_root=data_root)
    if not (shared / "mag/index/complete.json").is_file():
        raise FileNotFoundError("shared leakage-filtered MAG index is incomplete")
    for name in DATA_FILES:
        _copy_path(shared / "data" / name, run / "data" / name)
    for relative in ("token_features", "cooccurrence_graph", "mag/index"):
        _copy_path(shared / relative, run / relative)
    forbidden = _test_data_files(run)
    if forbidden:
        raise RuntimeError(f"candidate run exposes test files: {forbidden}")
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to bind candidate provenance")
    revision = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(  # noqa: S603
        [git, "status", "--porcelain"],
        cwd=PACKAGE_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("candidate source must be committed before training")
    manifest = {
        "candidate": candidate,
        "source_commit": revision,
        "test_files": forbidden,
        "validation_only": True,
    }
    write_json(run / "campaign_manifest.json", manifest)
    return {"protocol": protocol, "manifest": manifest}


def candidate_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Extract the fixed validation metrics used by the campaign gates."""
    run = Path(run_dir).expanduser().resolve()
    evaluation = read_json(run / "validation_evaluation/neural/complete.json")
    chemistry = read_json(run / "validation_chemical/neural/complete.json")
    scored = chemistry["high_confidence_chemistry"]
    bands = scored["sos_bands"]
    model, _, _ = load_trained_model(run / "trained_model")
    return {
        "optimized_motifs": int(
            round(float(chemistry["annotation_coverage"]) * chemistry["topics"])
        ),
        "evaluable_motifs": int(scored["eligible_topics"]),
        "useful_motifs": int(bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]),
        "mean_sos": float(scored["mean_sos"]),
        "median_sos": float(scored["median_sos"]),
        "validation_nll": float(
            evaluation["metrics"]["validation_document_completion"]["nll_per_token"]
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def assess_candidate(
    metrics: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Apply the predeclared chemistry-first decision rule."""
    finite = all(np.isfinite(float(value)) for value in metrics.values())
    absolute = {
        name: float(metrics[name]) >= floor for name, floor in ABSOLUTE_FLOORS.items()
    }
    relative = {
        name: float(metrics[name]) >= factor * float(baseline[name])
        for name, factor in STANDARD_RETENTION.items()
    }
    tie = {
        name: float(metrics[name]) >= factor * float(baseline[name])
        for name, factor in TIE_RETENTION.items()
    }
    misses = [name for name, passed in relative.items() if not passed]
    nll_101 = float(metrics["validation_nll"]) <= (
        NLL_REPORTING_REFERENCE * float(baseline["validation_nll"])
    )
    nll_105 = float(metrics["validation_nll"]) <= (
        NLL_SELECTION_CEILING * float(baseline["validation_nll"])
    )
    standard = finite and all(absolute.values()) and not misses and nll_105
    borderline = (
        finite
        and all(absolute.values())
        and nll_105
        and len(misses) == 1
        and tie[misses[0]]
    )
    retained = standard or borderline
    return {
        "absolute_gates": absolute,
        "borderline": borderline,
        "chemistry_relative_gates": relative,
        "failure_note": (
            ""
            if retained
            else ", ".join(
                [name for name, passed in absolute.items() if not passed]
                + misses
                + ([] if nll_105 else ["validation_nll_105"])
                + ([] if finite else ["numerically_stable"])
            )
        ),
        "nll_101_reporting_reference": nll_101,
        "nll_105_selection_ceiling": nll_105,
        "numerically_stable": finite,
        "retained": retained,
        "tie_gates": tie,
    }


def run_candidate(  # noqa: PLR0913
    run_dir: str | Path,
    *,
    shared_dir: str | Path,
    data_root: str | Path,
    candidate: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Train and score one candidate without opening the test split."""
    run = Path(run_dir).expanduser().resolve()
    initialized = initialize_candidate(
        run,
        shared_dir=shared_dir,
        data_root=data_root,
        candidate=candidate,
    )
    protocol = initialized["protocol"]
    _configure_threads(int(protocol["cpu_threads"]))
    train_model(run, train=load_csr(run / "data/train.npz"), protocol=protocol)
    evaluate_neural(run, protocol, split="validation")
    _chemical_subprocess(
        run,
        method="neural",
        data_root=data_root,
        cpu_threads=int(protocol["cpu_threads"]),
        split="validation",
    )
    metrics = candidate_metrics(run)
    result = {
        **initialized["manifest"],
        "baseline": baseline,
        "metrics": metrics,
        "decision": assess_candidate(metrics, baseline),
    }
    write_json(run / "candidate_results.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-shared")
    prepare.add_argument("--run", required=True, type=Path)
    prepare.add_argument("--data-root", required=True, type=Path)
    index = commands.add_parser("prepare-shared-index")
    index.add_argument("--run", required=True, type=Path)
    index.add_argument("--data-root", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--shared", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--candidate", required=True)
    run.add_argument("--baseline", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the isolated campaign utility."""
    args = _parser().parse_args(argv)
    if args.command == "prepare-shared":
        result = prepare_shared(args.run, data_root=args.data_root)
    elif args.command == "prepare-shared-index":
        result = prepare_shared_index(args.run, data_root=args.data_root)
    else:
        baseline = read_json(args.baseline)["metrics"] if args.baseline else U1_BASELINE
        result = run_candidate(
            args.run,
            shared_dir=args.shared,
            data_root=args.data_root,
            candidate=args.candidate,
            baseline=baseline,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
