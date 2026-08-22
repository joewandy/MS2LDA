"""Validation-only seed-42 architecture experiment runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import REPO_ROOT, code_manifest, load_protocol
from .core import prepare_initialization
from .data import load_csr, load_heldout_records, load_view_pairs
from .training import train_model
from .utils import (
    file_sha256,
    object_sha256,
    read_json,
    verify_output_hashes,
    write_json,
    write_jsonl,
)

READ_ONLY_STAGES = (
    "training_views",
    "token_features",
)
DEVELOPMENT_DATA_FILES = (
    "train.npz",
    "validation_observed.npz",
    "validation_completion.npz",
    "validation_full.npz",
    "validation_records.jsonl",
    "vocabulary.json",
)
LEGACY_HELDOUT_RECORDS = "heldout_records.jsonl"
MAG_FILES = (
    "excluded_connectivity_keys.json",
    "kept_original_ids.npy",
    "spec2vec_filtered.faiss",
)


def _verify_declared_file(
    stage: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    legacy_key: str | None = None,
) -> None:
    expected = manifest.get("output_sha256", {}).get(name)
    if expected is None and legacy_key is not None:
        expected = manifest.get(legacy_key)
    path = stage / name
    if expected is None or file_sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _verify_source_artifacts(source: Path, validation_records: Path) -> None:
    data = source / "data"
    data_manifest = read_json(data / "complete.json")
    declared = tuple(
        name for name in DEVELOPMENT_DATA_FILES if name != "validation_records.jsonl"
    )
    for name in (*declared, validation_records.name):
        _verify_declared_file(data, data_manifest, name)
    _verify_declared_file(data, data_manifest, "split_manifest.jsonl")
    training_views = source / "training_views"
    verify_output_hashes(training_views, read_json(training_views / "complete.json"))
    token_features = source / "token_features"
    _verify_declared_file(
        token_features,
        read_json(token_features / "complete.json"),
        "features.npy",
        legacy_key="features_sha256",
    )
    mag = source / "mag/index"
    verify_output_hashes(mag, read_json(mag / "manifest.json"))
    excluded = set(
        read_json(mag / "excluded_connectivity_keys.json")["connectivity_keys"]
    )
    heldout = set()
    with (data / "split_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] in {"validation", "test"}:
                    heldout.add(str(row["connectivity_key"]))
    if excluded != heldout:
        raise ValueError("MAG exclusion does not exactly cover held-out compounds")


def _source_evidence(source: Path) -> dict[str, str]:
    declared = tuple(
        name for name in DEVELOPMENT_DATA_FILES if name != "validation_records.jsonl"
    )
    names = (
        "protocol.resolved.json",
        "data/complete.json",
        "data/split_manifest.jsonl",
        *(f"data/{name}" for name in declared),
        "training_views/complete.json",
        "token_features/features.npy",
        "token_features/complete.json",
        "mag/index/manifest.json",
        *(f"mag/index/{name}" for name in MAG_FILES),
    )
    validation_records = source / "data/validation_records.jsonl"
    if not validation_records.is_file():
        validation_records = source / "data" / LEGACY_HELDOUT_RECORDS
    missing = [name for name in names if not (source / name).is_file()]
    if not validation_records.is_file():
        missing.append("data/validation_records.jsonl")
    if missing:
        raise FileNotFoundError(f"source run is incomplete: {missing}")
    _verify_source_artifacts(source, validation_records)
    evidence = {name: file_sha256(source / name) for name in names}
    evidence[f"data/{validation_records.name}"] = file_sha256(validation_records)
    return evidence


def _validate_frozen_protocol(source: dict[str, Any], current: dict[str, Any]) -> None:
    sections = (
        "input_files",
        "preprocessing",
        "sgns",
        "token_features",
        "model",
        "views",
        "anti_collapse",
        "evaluation",
    )
    changed = [] if source["seed"] == current["seed"] else ["seed"]
    for section in sections:
        ignored = (
            {
                "document_mixture_weight",
                "document_topic_prior_weight",
                "num_topics",
                "beta_temperature",
                "token_type_balance",
                "normalize_token_type_evidence",
            }
            if section == "model"
            else set()
        )
        changed.extend(
            f"{section}.{key}"
            for key, value in current[section].items()
            if key not in ignored and source[section].get(key) != value
        )
    source_optimization = source["optimization"]
    current_optimization = current["optimization"]
    frozen_optimization_keys = (
        "batch_size",
        "topic_update_batch_size",
        "topic_updates_per_epoch",
        "learning_rate",
        "weight_decay",
        "local_decoder_weight",
        "theta_consistency_weight",
        "maximum_epochs",
        "validation_interval",
    )
    changed.extend(
        f"optimization.{key}"
        for key in frozen_optimization_keys
        if source_optimization.get(key) != current_optimization.get(key)
    )
    if changed:
        raise ValueError(f"frozen benchmark fields changed: {changed}")


def _development_protocol(source: dict[str, Any]) -> dict[str, Any]:
    """Use the single supported protocol with frozen source data and views."""
    del source
    return load_protocol()


def _link(target: Path, link: Path, *, directory: bool) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError(f"experiment input link changed: {link.name}")
    elif link.exists():
        raise ValueError(f"experiment input must be a read-only link: {link.name}")
    else:
        link.symlink_to(target, target_is_directory=directory)


def _link_read_only_inputs(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data"
    if data.is_symlink():
        raise ValueError("development data must not expose the complete source split")
    data.mkdir(exist_ok=True)
    for name in DEVELOPMENT_DATA_FILES:
        if name == "validation_records.jsonl":
            continue
        _link(source / "data" / name, data / name, directory=False)
    validation_records = source / "data/validation_records.jsonl"
    if validation_records.is_file():
        _link(validation_records, data / validation_records.name, directory=False)
    else:
        rows = []
        with (source / "data" / LEGACY_HELDOUT_RECORDS).open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("split") == "validation":
                        rows.append(row)
        if not rows:
            raise ValueError("legacy held-out records contain no validation rows")
        write_jsonl(data / "validation_records.jsonl", rows)
    for name in READ_ONLY_STAGES:
        _link(source / name, output / name, directory=True)
    _link(source / "mag", output / "mag", directory=True)


def run_development_experiment(
    source_run: str | Path,
    output_run: str | Path,
    *,
    hypothesis: str,
) -> dict[str, Any]:
    """Run one architecture hypothesis without loading any test matrix."""
    source = Path(source_run).expanduser().resolve()
    output = Path(output_run).expanduser().resolve()
    source_protocol = read_json(source / "protocol.resolved.json")
    protocol = _development_protocol(source_protocol)
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
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    lock_path = output / "development.lock.json"
    if lock_path.is_file() and read_json(lock_path) != lock:
        raise ValueError("development experiment provenance changed")
    _link_read_only_inputs(source, output)
    if not lock_path.is_file():
        write_json(output / "protocol.resolved.json", protocol)
        write_json(lock_path, lock)

    torch.set_num_threads(int(protocol["cpu_threads"]))
    data = output / "data"
    train = load_csr(data / "train.npz")
    prepare_initialization(output, train=train, protocol=protocol)
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
        "selection_rule": result["selected"]["selection_rule"],
        "selected_checkpoint_sha256": result["selected"]["checkpoint_sha256"],
        "validation": result["selected"]["validation"],
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
