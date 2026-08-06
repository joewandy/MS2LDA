"""Immutable protocol construction and verification."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import (
    BenchmarkConfig,
    environment_manifest,
    file_sha256,
    git_state,
    load_config,
    object_sha256,
    read_json,
    resolve_input_paths,
    write_json,
)
from .data import (
    SpectrumRecord,
    assign_scaffold_splits,
    audit_split_disjointness,
    build_training_vocabulary,
    completion_document,
    load_records,
)

LOCK_FILENAME = "protocol.lock.json"
EXECUTION_ONLY_DERIVATION_FIELDS = frozenset(
    {"protocol_name", "hybrid_training_cpu_threads"}
)
CONVERGENCE_CONTINUATION_FIELDS = frozenset({"hybrid_max_epochs"})
CHEMICAL_CORRECTION_FIELDS = frozenset({"protocol_name"})


def _source_files(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "benchmarks" / "msnlib_validation",
        repo_root / "ms2lda_hybrid",
    ]
    files = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json", ".md"}
        )
    return sorted(files)


def code_manifest(repo_root: str | Path) -> dict[str, str]:
    """Hash every benchmark and Hybrid source file affecting a run."""
    root = Path(repo_root).resolve()
    return {
        str(path.relative_to(root)): file_sha256(path) for path in _source_files(root)
    }


def validate_inputs(
    config: BenchmarkConfig,
    data_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify every configured external file against size and SHA-256."""
    resolved = resolve_input_paths(config, data_root)
    rows = {}
    for name, path in resolved.items():
        specification = config.input_files[name]
        if not path.is_file():
            raise FileNotFoundError(f"missing input {name}: {path}")
        size = path.stat().st_size
        expected_size = specification.get("bytes")
        if expected_size is not None and size != int(expected_size):
            raise ValueError(
                f"input size mismatch for {name}: expected {expected_size}, found {size}"
            )
        digest = file_sha256(path)
        if digest != specification["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}: {path}")
        rows[name] = {
            "path": str(path),
            "bytes": size,
            "sha256": digest,
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "files": rows,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    if output_path is not None:
        write_json(output_path, manifest)
    return manifest


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _split_rows(
    records: Sequence[SpectrumRecord], assignments: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "spectrum_id": record.spectrum_id,
            "split": assignments[record.spectrum_id],
            "feature_id": record.feature_id,
            "connectivity_key": record.connectivity_key,
            "scaffold": record.scaffold_key,
            "split_group": record.split_group,
            "smiles": record.smiles,
        }
        for record in records
    ]


def _completion_rows(
    records: Sequence[SpectrumRecord],
    assignments: dict[str, str],
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        split = assignments[record.spectrum_id]
        if split not in {"validation", "test"}:
            continue
        try:
            completion = completion_document(
                record,
                observed_fraction=config.completion_observed_fraction,
                seed=config.completion_seed,
            )
        except ValueError as exc:
            rows.append(
                {
                    "spectrum_id": record.spectrum_id,
                    "split": split,
                    "eligible": False,
                    "reason": str(exc),
                    "observed_peak_indices": [],
                    "completion_peak_indices": [],
                }
            )
            continue
        rows.append(
            {
                "spectrum_id": record.spectrum_id,
                "split": split,
                "eligible": True,
                "reason": "",
                "observed_peak_indices": [
                    group.original_index for group in completion.observed_groups
                ],
                "completion_peak_indices": [
                    group.original_index for group in completion.completion_groups
                ],
            }
        )
    return rows


def freeze_protocol(
    config: BenchmarkConfig,
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
    repo_root: str | Path,
    test_results_inspected: bool = False,
    derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create exact manifests and an immutable lock before test evaluation."""
    destination = Path(run_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / LOCK_FILENAME
    if lock_path.exists():
        raise FileExistsError(f"protocol is already frozen: {lock_path}")
    existing = [
        path for path in destination.iterdir() if path.name != "input_manifest.json"
    ]
    if existing:
        raise FileExistsError(
            f"run directory contains unfrozen files: {', '.join(path.name for path in existing)}"
        )
    input_path = destination / "input_manifest.json"
    if input_path.exists():
        input_manifest = read_json(input_path)
        expected = validate_inputs(config, data_root)
        if input_manifest.get("files") != expected.get("files"):
            raise ValueError("existing input manifest does not match configured inputs")
    else:
        input_manifest = validate_inputs(config, data_root, output_path=input_path)
    paths = resolve_input_paths(config, data_root)
    records, data_summary = load_records(paths["mgf"], config)
    assignments, split_summary = assign_scaffold_splits(
        records,
        fractions=config.split_fractions,
        seed=config.split_seed,
    )
    leakage_audit = audit_split_disjointness(records, assignments)
    vocabulary, vocabulary_summary = build_training_vocabulary(
        records,
        assignments,
        min_df=config.min_df,
        min_cf=config.min_cf,
        rm_top=config.rm_top,
    )
    split_rows = _split_rows(records, assignments)
    completion_rows = _completion_rows(records, assignments, config)
    split_path = destination / "split_manifest.jsonl"
    model_assignments_path = destination / "model_assignments.json"
    completion_path = destination / "completion_manifest.jsonl"
    vocabulary_path = destination / "vocabulary.json"
    resolved_config_path = destination / "config.resolved.json"
    _write_jsonl(split_path, split_rows)
    write_json(model_assignments_path, assignments)
    _write_jsonl(completion_path, completion_rows)
    write_json(vocabulary_path, {"vocabulary": list(vocabulary)})
    write_json(resolved_config_path, config.as_dict())
    summary = {
        "data": data_summary,
        "split": split_summary,
        "leakage_audit": leakage_audit,
        "vocabulary": vocabulary_summary,
        "completion": {
            "rows": len(completion_rows),
            "eligible": sum(bool(row["eligible"]) for row in completion_rows),
            "ineligible": sum(not bool(row["eligible"]) for row in completion_rows),
            "peak_group_atomicity": True,
            "observed_intensity_renormalized_after_split": True,
        },
    }
    summary_path = destination / "freeze_summary.json"
    environment_path = destination / "environment.json"
    write_json(summary_path, summary)
    write_json(environment_path, environment_manifest())
    root = Path(repo_root).resolve()
    sources = code_manifest(root)
    source_manifest_path = destination / "code_manifest.json"
    write_json(source_manifest_path, sources)
    artifacts = {
        path.name: file_sha256(path)
        for path in (
            input_path,
            resolved_config_path,
            split_path,
            model_assignments_path,
            completion_path,
            vocabulary_path,
            summary_path,
            environment_path,
            source_manifest_path,
        )
    }
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_name": config.protocol_name,
        "config_source": str(Path(config_path).resolve()),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "repo_root": str(root),
        "command": " ".join(map(shlex.quote, sys.argv)),
        "git": git_state(root),
        "artifacts": artifacts,
        "source_manifest_sha256": object_sha256(sources),
        "test_results_inspected": bool(test_results_inspected),
    }
    if derivation is not None:
        lock["derivation"] = derivation
    lock["protocol_sha256"] = object_sha256(lock)
    write_json(lock_path, lock)
    return lock


def validate_execution_only_derivation(
    source_run: str | Path,
    target_config: BenchmarkConfig,
    reason: str | None = None,
) -> dict[str, Any]:
    """Prove a post-inspection continuation changes execution settings only."""
    source_directory = Path(source_run).expanduser().resolve()
    source_lock = verify_protocol(source_directory, verify_code=False)
    source_config = load_config(source_directory / "config.resolved.json")
    source_values = source_config.as_dict()
    target_values = target_config.as_dict()
    differences = {
        key: {"source": source_values.get(key), "target": target_values.get(key)}
        for key in sorted(set(source_values) | set(target_values))
        if source_values.get(key) != target_values.get(key)
    }
    difference_fields = set(differences)
    normalized_reason = str(reason).strip() if reason is not None else ""
    if difference_fields == {"protocol_name"}:
        if not normalized_reason:
            raise ValueError(
                "protocol-name-only derivation requires an explicit execution reason"
            )
    elif difference_fields == EXECUTION_ONLY_DERIVATION_FIELDS:
        if target_config.hybrid_training_cpu_threads <= 1:
            raise ValueError("derived Hybrid training must request limited parallelism")
    else:
        raise ValueError(
            "derived protocol must differ only in protocol_name, optionally with "
            "hybrid_training_cpu_threads"
        )
    result = {
        "created_after_source_test_results_inspected": True,
        "differences": differences,
        "execution_only": True,
        "scientific_settings_frozen_by_source": True,
        "source_config_sha256": object_sha256(source_values),
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_run": str(source_directory),
        "target_config_sha256": object_sha256(target_values),
    }
    if normalized_reason:
        result["reason"] = normalized_reason
    return result


def validate_convergence_continuation_derivation(
    source_run: str | Path,
    target_config: BenchmarkConfig,
    reason: str | None = None,
) -> dict[str, Any]:
    """Freeze a training-only continuation without changing its stop rule."""
    source_directory = Path(source_run).expanduser().resolve()
    source_lock = verify_protocol(source_directory, verify_code=False)
    source_config = load_config(source_directory / "config.resolved.json")
    source_values = source_config.as_dict()
    target_values = target_config.as_dict()
    differences = {
        key: {"source": source_values.get(key), "target": target_values.get(key)}
        for key in sorted(set(source_values) | set(target_values))
        if source_values.get(key) != target_values.get(key)
    }
    if target_config.hybrid_max_epochs <= source_config.hybrid_max_epochs:
        raise ValueError("convergence continuation must increase hybrid_max_epochs")
    if set(differences) != CONVERGENCE_CONTINUATION_FIELDS:
        raise ValueError("convergence continuation may change only hybrid_max_epochs")
    normalized_reason = str(reason).strip() if reason is not None else ""
    if not normalized_reason:
        raise ValueError("convergence continuation requires an explicit reason")
    return {
        "all_other_settings_frozen_by_source": True,
        "created_after_source_test_results_inspected": True,
        "differences": differences,
        "execution_only": False,
        "kind": "training_convergence_continuation",
        "reason": normalized_reason,
        "source_config_sha256": object_sha256(source_values),
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_run": str(source_directory),
        "stopping_rule_unchanged": True,
        "target_config_sha256": object_sha256(target_values),
        "trigger_uses_training_state_only": True,
    }


def validate_chemical_evaluation_correction_derivation(
    source_run: str | Path,
    target_config: BenchmarkConfig,
    reason: str | None = None,
) -> dict[str, Any]:
    """Disclose a post-inspection correction to the chemical endpoint only."""
    source_directory = Path(source_run).expanduser().resolve()
    source_lock = verify_protocol(source_directory, verify_code=False)
    source_config = load_config(source_directory / "config.resolved.json")
    source_values = source_config.as_dict()
    target_values = target_config.as_dict()
    differences = {
        key: {"source": source_values.get(key), "target": target_values.get(key)}
        for key in sorted(set(source_values) | set(target_values))
        if source_values.get(key) != target_values.get(key)
    }
    if set(differences) != CHEMICAL_CORRECTION_FIELDS:
        raise ValueError("chemical evaluation correction may change only protocol_name")
    normalized_reason = str(reason).strip() if reason is not None else ""
    if not normalized_reason:
        raise ValueError("chemical evaluation correction requires an explicit reason")
    return {
        "confirmatory": False,
        "core_model_artifacts_unchanged": True,
        "created_after_source_test_results_inspected": True,
        "differences": differences,
        "execution_only": False,
        "kind": "chemical_evaluation_correction",
        "reason": normalized_reason,
        "source_config_sha256": object_sha256(source_values),
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_run": str(source_directory),
        "target_config_sha256": object_sha256(target_values),
    }


def verify_protocol(run_dir: str | Path, *, verify_code: bool = True) -> dict[str, Any]:
    """Refuse execution after any frozen artifact or benchmark source changes."""
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / LOCK_FILENAME)
    expected_protocol_hash = lock.pop("protocol_sha256")
    if object_sha256(lock) != expected_protocol_hash:
        raise ValueError("protocol lock self-hash mismatch")
    lock["protocol_sha256"] = expected_protocol_hash
    for name, expected_hash in lock["artifacts"].items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"frozen artifact changed: {path}")
    if verify_code:
        current = code_manifest(lock["repo_root"])
        if object_sha256(current) != lock["source_manifest_sha256"]:
            raise ValueError("benchmark or Hybrid source changed after protocol freeze")
    return lock


def verify_frozen_input_files(
    run_dir: str | Path,
    *,
    names: set[str] | None = None,
    lock: dict[str, Any] | None = None,
) -> dict[str, dict[str, int | str]]:
    """Rehash selected external inputs immediately before a consuming stage."""
    directory = Path(run_dir).expanduser().resolve()
    frozen_lock = verify_protocol(directory) if lock is None else lock
    manifest = read_json(directory / "input_manifest.json")
    files = manifest.get("files", {})
    selected = set(files) if names is None else set(names)
    missing = selected - set(files)
    if missing:
        raise ValueError(f"frozen input names are missing: {sorted(missing)}")
    verified = {}
    for name in sorted(selected):
        row = files[name]
        path = Path(row["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {path}")
        size = path.stat().st_size
        if size != int(row["bytes"]):
            raise ValueError(f"frozen input byte size changed: {name}")
        digest = file_sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"frozen input SHA-256 changed: {name}")
        verified[name] = {"bytes": size, "path": str(path), "sha256": digest}
    if (
        Path(frozen_lock["data_root"]).expanduser().resolve()
        != Path(manifest["data_root"]).expanduser().resolve()
    ):
        raise ValueError("input manifest data root differs from protocol lock")
    return verified


def load_assignments(run_dir: str | Path) -> dict[str, str]:
    """Load the chemistry-free identifier-to-split map."""
    return {
        str(key): str(value)
        for key, value in read_json(Path(run_dir) / "model_assignments.json").items()
    }


def load_vocabulary(run_dir: str | Path) -> tuple[str, ...]:
    """Load the frozen training-only vocabulary."""
    payload = read_json(Path(run_dir) / "vocabulary.json")
    return tuple(map(str, payload["vocabulary"]))


def load_completion_rows(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load the frozen document-completion manifest keyed by spectrum ID."""
    rows = {}
    with (Path(run_dir) / "completion_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["spectrum_id"])] = row
    return rows
