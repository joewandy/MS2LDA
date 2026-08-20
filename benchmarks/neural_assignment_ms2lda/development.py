"""Validation-only seed-42 architecture experiment runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import code_manifest, load_protocol
from .data import load_csr, load_heldout_records, load_view_pairs
from .training import train_model
from .utils import file_sha256, object_sha256, read_json, write_json

READ_ONLY_STAGES = (
    "data",
    "training_views",
    "embeddings",
    "token_features",
    "initialization",
)


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def _source_evidence(source: Path) -> dict[str, str]:
    names = (
        "protocol.resolved.json",
        "data/complete.json",
        "data/train.npz",
        "data/validation_observed.npz",
        "data/validation_completion.npz",
        "data/validation_full.npz",
        "data/heldout_records.jsonl",
        "training_views/complete.json",
        "token_features/features.npy",
        "token_features/complete.json",
        "initialization/model_initialization.pt",
        "initialization/complete.json",
    )
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source run is incomplete: {missing}")
    return {name: file_sha256(source / name) for name in names}


def _validate_frozen_protocol(source: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "seed",
        "input_files",
        "preprocessing",
        "sgns",
        "token_features",
        "model",
        "views",
        "anti_collapse",
        "evaluation",
    )
    changed = [key for key in keys if source[key] != current[key]]
    source_optimization = source["optimization"]
    current_optimization = current["optimization"]
    architecture_optimization_keys = {"erntm_weight"}
    changed.extend(
        f"optimization.{key}"
        for key, value in source_optimization.items()
        if key not in architecture_optimization_keys
        and current_optimization.get(key) != value
    )
    if changed:
        raise ValueError(f"frozen benchmark fields changed: {changed}")


def _link_read_only_stages(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in READ_ONLY_STAGES:
        target = source / name
        link = output / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"experiment stage link changed: {name}")
        elif link.exists():
            raise ValueError(f"experiment stage must be a read-only link: {name}")
        else:
            link.symlink_to(target, target_is_directory=True)


def run_development_experiment(
    source_run: str | Path,
    output_run: str | Path,
    *,
    hypothesis: str,
) -> dict[str, Any]:
    """Run one architecture hypothesis without loading any test matrix."""
    source = Path(source_run).expanduser().resolve()
    output = Path(output_run).expanduser().resolve()
    protocol = load_protocol()
    source_protocol = read_json(source / "protocol.resolved.json")
    _validate_frozen_protocol(source_protocol, protocol)
    evidence = _source_evidence(source)
    lock = {
        "schema_version": "neural-ms2lda/development-lock-v1",
        "hypothesis": str(hypothesis),
        "seed": int(protocol["seed"]),
        "source_run": str(source),
        "source_artifact_sha256": evidence,
        "protocol_sha256": object_sha256(protocol),
        "code_sha256": object_sha256(code_manifest()),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "test_matrices_loaded": False,
        "test_evaluation_performed": False,
    }
    lock_path = output / "development.lock.json"
    if lock_path.is_file() and read_json(lock_path) != lock:
        raise ValueError("development experiment provenance changed")
    _link_read_only_stages(source, output)
    if not lock_path.is_file():
        write_json(output / "protocol.resolved.json", protocol)
        write_json(lock_path, lock)

    torch.set_num_threads(int(protocol["training_cpu_threads"]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(protocol["seed"]))
    data = output / "data"
    train = load_csr(data / "train.npz")
    result = train_model(
        output,
        train=train,
        views=load_view_pairs(output, protocol),
        validation_observed=load_csr(data / "validation_observed.npz"),
        validation_completion=load_csr(data / "validation_completion.npz"),
        validation_full=load_csr(data / "validation_full.npz"),
        validation_records=load_heldout_records(data, "validation"),
        protocol=protocol,
        heartbeat=lambda **details: write_json(output / "heartbeat.json", details),
    )
    summary = {
        "schema_version": "neural-ms2lda/development-result-v1",
        "hypothesis": str(hypothesis),
        "selected_epoch": int(result["selected"]["epoch"]),
        "selected_checkpoint_sha256": result["selected"]["checkpoint_sha256"],
        "validation_gate_summary": result["selected"]["validation_gate_summary"],
        "development_gates": protocol["development_gates"],
        "test_matrices_loaded": False,
        "test_evaluation_performed": False,
    }
    write_json(output / "validation_result.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args(argv)
    result = run_development_experiment(
        args.source_run, args.run, hypothesis=args.hypothesis
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
