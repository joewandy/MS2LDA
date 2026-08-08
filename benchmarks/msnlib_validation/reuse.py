"""Hash-audited reuse of completed core artifacts by derived protocols."""

from __future__ import annotations

import copy
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import file_sha256, load_config, read_json, write_json
from .protocol import (
    validate_chemical_evaluation_correction_derivation,
    validate_convergence_continuation_derivation,
    validate_execution_only_derivation,
    validate_implementation_correction_derivation,
    verify_protocol,
)

IDENTICAL_FROZEN_ARTIFACTS = (
    "split_manifest.jsonl",
    "model_assignments.json",
    "completion_manifest.jsonl",
    "vocabulary.json",
)


def _validate_target_derivation(
    source_directory: Path,
    target_config: Any,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the recorded derivation with its matching strict validator."""
    reason = recorded.get("reason")
    if recorded.get("kind") == "training_convergence_continuation":
        return validate_convergence_continuation_derivation(
            source_directory,
            target_config,
            reason,
        )
    if recorded.get("kind") == "chemical_evaluation_correction":
        return validate_chemical_evaluation_correction_derivation(
            source_directory,
            target_config,
            reason,
        )
    return validate_execution_only_derivation(
        source_directory,
        target_config,
        reason,
    )


def _inventory(directory: Path) -> dict[str, dict[str, int | str]]:
    """Hash every regular file below a generated artifact directory."""
    return {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _copytree_atomic(source: Path, destination: Path) -> Path:
    """Copy one directory completely before exposing its final path."""
    if destination.exists():
        raise FileExistsError(f"reuse destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.import-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"reuse staging directory already exists: {temporary}")
    try:
        shutil.copytree(source, temporary)
        return temporary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verify_feature_source(
    source_directory: Path,
    source_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, int | str]]]:
    """Verify the completed source cache against its own manifest."""
    feature_dir = source_directory / "features"
    manifest = read_json(feature_dir / "manifest.json")
    if manifest.get("protocol_sha256") != source_protocol_sha256:
        raise ValueError("source feature cache belongs to another protocol")
    for name, digest in manifest.get("output_sha256", {}).items():
        path = feature_dir / str(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"source feature cache changed: {name}")
    if int(manifest.get("rows", -1)) < 1 or int(manifest.get("train_rows", -1)) < 1:
        raise ValueError("source feature cache is incomplete")
    return manifest, _inventory(feature_dir)


def _verify_tomotopy_source(
    source_directory: Path,
    *,
    seed: int,
    topic_count: int,
    workers: int,
    parallel: int,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, int | str]]]:
    """Verify a completed Tomotopy result and all hashes it declares."""
    result_dir = source_directory / "core" / f"seed_{seed}" / "tomotopy"
    result = read_json(result_dir / "complete.json")
    expected = {
        "method": "tomotopy",
        "seed": seed,
        "topic_count": topic_count,
        "training_workers_requested": workers,
        "training_parallel_scheme_value": parallel,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"source Tomotopy {name} mismatch for seed {seed}")
    declared = {
        "beta.npy": result["beta_sha256"],
        "test_theta.npy": result["theta_sha256"],
        "model.bin": result["model_sha256"],
    }
    for name, digest in declared.items():
        if file_sha256(result_dir / name) != digest:
            raise ValueError(f"source Tomotopy artifact changed: {name}")
    return result_dir, result, _inventory(result_dir)


def _verify_hybrid_source(
    source_directory: Path,
    *,
    seed: int,
    topic_count: int,
    training_threads: int,
    inference_threads: int,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, int | str]]] | None:
    """Verify an optional completed Hybrid arm before derived-run reuse."""
    result_dir = source_directory / "core" / f"seed_{seed}" / "hybrid"
    complete_path = result_dir / "complete.json"
    if not complete_path.is_file():
        return None
    result = read_json(complete_path)
    expected = {
        "method": "hybrid",
        "seed": seed,
        "topic_count": topic_count,
        "training_cpu_threads": training_threads,
        "inference_cpu_threads": inference_threads,
        "reference_converged": True,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"source Hybrid {name} mismatch for seed {seed}")
    reference_steps = str(result["reference_steps"])
    theta_hashes = result.get("theta_sha256")
    expected_steps = {"0", "2", reference_steps}
    if not isinstance(theta_hashes, dict) or set(theta_hashes) != expected_steps:
        raise ValueError(f"source Hybrid theta hashes mismatch for seed {seed}")
    declared = {
        "beta.npy": result["beta_sha256"],
        "model.pt": result["model_sha256"],
        **{f"test_theta_{steps}.npy": theta_hashes[steps] for steps in expected_steps},
    }
    for name, digest in declared.items():
        if file_sha256(result_dir / name) != digest:
            raise ValueError(f"source Hybrid artifact changed: {name}")
    return result_dir, result, _inventory(result_dir)


def _verify_hybrid_continuation_source(
    source_directory: Path,
    *,
    seed: int,
    source_max_epochs: int,
    checkpoint_keep: int,
    training_threads: int,
) -> tuple[Path, list[tuple[Path, dict[str, Any]]]]:
    """Verify the retained nonconverged checkpoints selected for continuation."""
    result_dir = source_directory / "core" / f"seed_{seed}" / "hybrid"
    if (result_dir / "complete.json").exists():
        raise ValueError("a completed Hybrid result does not need continuation")
    checkpoint_dir = result_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("checkpoint-*.json"), reverse=True)
    required = min(2, checkpoint_keep)
    if len(candidates) < required:
        raise ValueError("source continuation lacks two checkpoint generations")
    selected: list[tuple[Path, dict[str, Any]]] = []
    for sidecar in candidates[:checkpoint_keep]:
        metadata = read_json(sidecar)
        filename = str(metadata.get("file", ""))
        binary = (checkpoint_dir / filename).resolve()
        if not filename or binary.parent != checkpoint_dir.resolve():
            raise ValueError("source checkpoint filename escapes its directory")
        if not binary.is_file():
            raise FileNotFoundError(f"source checkpoint is missing: {binary}")
        if binary.stat().st_size != int(metadata.get("bytes", -1)):
            raise ValueError("source checkpoint byte size mismatch")
        if file_sha256(binary) != metadata.get("sha256"):
            raise ValueError("source checkpoint SHA-256 mismatch")
        if metadata.get("phase") != "discovery":
            raise ValueError("source continuation checkpoint is not discovery state")
        if int(metadata.get("training_cpu_threads", -1)) != training_threads:
            raise ValueError("source checkpoint training thread count mismatch")
        selected.append((sidecar, metadata))
    latest = read_json(checkpoint_dir / "latest.json")
    if latest != selected[0][1]:
        raise ValueError("source latest checkpoint does not match newest sidecar")
    if int(latest.get("phase_epoch", -1)) != source_max_epochs:
        raise ValueError("source checkpoint did not reach its frozen maximum")
    contexts = {str(metadata.get("context_sha256")) for _, metadata in selected}
    if len(contexts) != 1:
        raise ValueError("source checkpoint generations have different contexts")
    return result_dir, list(reversed(selected))


def _import_hybrid_continuation(
    *,
    source_directory: Path,
    target_directory: Path,
    source_config: Any,
    target_config: Any,
    target_lock: dict[str, Any],
    seed: int,
    checkpoints: list[tuple[Path, dict[str, Any]]],
    common_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Rebind verified source checkpoints to the derived protocol context."""
    from ms2lda_hybrid import HybridLDAModel

    from .models import _hybrid_checkpoint_context, prepare_documents
    from .runtime import load_feature_cache

    target_result_dir = target_directory / "core" / f"seed_{seed}" / "hybrid"
    if target_result_dir.exists():
        manifest = read_json(target_result_dir / "continuation_import.json")
        if manifest.get("source_protocol_sha256") != common_provenance.get(
            "source_protocol_sha256"
        ):
            raise FileExistsError("target Hybrid continuation has different provenance")
        return manifest

    data = prepare_documents(target_directory)
    _, _, _, feature_manifest = load_feature_cache(target_directory)
    target_context = _hybrid_checkpoint_context(
        directory=target_directory,
        lock=target_lock,
        config=target_config,
        seed=seed,
        data=data,
        feature_manifest=feature_manifest,
    )
    temporary = (
        target_result_dir.parent / f".{target_result_dir.name}.import-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"reuse staging directory already exists: {temporary}")
    checkpoint_dir = temporary / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    imported = []
    try:
        for source_sidecar, source_metadata in checkpoints:
            source_binary = source_sidecar.parent / str(source_metadata["file"])
            target_binary = checkpoint_dir / source_binary.name
            migration = HybridLDAModel.extend_training_checkpoint(
                source_binary,
                target_binary,
                source_context_sha256=str(source_metadata["context_sha256"]),
                target_context_sha256=target_context,
                target_max_epochs=target_config.hybrid_max_epochs,
            )
            metadata = copy.deepcopy(source_metadata)
            metadata.update(
                {
                    "bytes": target_binary.stat().st_size,
                    "context_sha256": target_context,
                    "continuation_import": True,
                    "sha256": file_sha256(target_binary),
                    "source_checkpoint_sha256": source_metadata["sha256"],
                    "source_sidecar_sha256": file_sha256(source_sidecar),
                }
            )
            target_sidecar = checkpoint_dir / source_sidecar.name
            write_json(target_sidecar, metadata)
            imported.append({"metadata": metadata, "migration": migration})
        newest = imported[-1]["metadata"]
        write_json(checkpoint_dir / "latest.json", newest)
        with (checkpoint_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            import json

            for row in imported:
                handle.write(
                    json.dumps(row["metadata"], sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        manifest = {
            **common_provenance,
            "all_other_settings_unchanged": True,
            "imported_checkpoints": imported,
            "seed": seed,
            "source_max_epochs": source_config.hybrid_max_epochs,
            "stopping_rule_unchanged": True,
            "target_context_sha256": target_context,
            "target_max_epochs": target_config.hybrid_max_epochs,
            "trigger_uses_training_state_only": True,
        }
        write_json(temporary / "continuation_import.json", manifest)
        os.replace(temporary, target_result_dir)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def reuse_core_artifacts(
    source_run: str | Path,
    target_run: str | Path,
) -> dict[str, Any]:
    """Reuse verified feature and completed core outputs in a derivative."""
    source_directory = Path(source_run).expanduser().resolve()
    target_directory = Path(target_run).expanduser().resolve()
    source_lock = verify_protocol(source_directory, verify_code=False)
    target_lock = verify_protocol(target_directory)
    source_config = load_config(source_directory / "config.resolved.json")
    target_config = load_config(target_directory / "config.resolved.json")
    recorded_derivation = target_lock.get("derivation", {})
    if recorded_derivation.get("kind") == "implementation_correction":
        raise ValueError(
            "implementation corrections may reuse Tomotopy only; use reuse-tomotopy"
        )
    derivation = _validate_target_derivation(
        source_directory,
        target_config,
        recorded_derivation,
    )
    if recorded_derivation != derivation:
        raise ValueError("target lock does not contain the validated derivation")
    if target_lock.get("test_results_inspected") is not True:
        raise ValueError("derived lock must disclose prior test-result inspection")

    source_inputs = read_json(source_directory / "input_manifest.json")["files"]
    target_inputs = read_json(target_directory / "input_manifest.json")["files"]
    if source_inputs != target_inputs:
        raise ValueError("source and target input manifests differ")
    for name in IDENTICAL_FROZEN_ARTIFACTS:
        if source_lock["artifacts"][name] != target_lock["artifacts"][name]:
            raise ValueError(f"source and target frozen artifacts differ: {name}")

    source_feature_manifest, source_feature_inventory = _verify_feature_source(
        source_directory,
        source_lock["protocol_sha256"],
    )
    tomotopy_sources = {}
    hybrid_sources = {}
    continuation_sources = {}
    is_continuation = derivation.get("kind") == "training_convergence_continuation"
    for seed in target_config.seeds:
        tomotopy_sources[seed] = _verify_tomotopy_source(
            source_directory,
            seed=seed,
            topic_count=target_config.num_topics,
            workers=target_config.tomotopy_training_workers,
            parallel=target_config.tomotopy_training_parallel,
        )
        hybrid_sources[seed] = _verify_hybrid_source(
            source_directory,
            seed=seed,
            topic_count=target_config.num_topics,
            training_threads=target_config.hybrid_training_cpu_threads,
            inference_threads=target_config.hybrid_inference_cpu_threads,
        )
        if is_continuation:
            _, continuation_sources[seed] = _verify_hybrid_continuation_source(
                source_directory,
                seed=seed,
                source_max_epochs=source_config.hybrid_max_epochs,
                checkpoint_keep=source_config.hybrid_checkpoint_keep,
                training_threads=source_config.hybrid_training_cpu_threads,
            )

    created_utc = datetime.now(timezone.utc).isoformat()
    common_provenance = {
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_run": str(source_directory),
        "target_protocol_sha256": target_lock["protocol_sha256"],
        "target_run": str(target_directory),
    }
    target_feature_dir = target_directory / "features"
    if target_feature_dir.exists():
        existing = read_json(target_feature_dir / "manifest.json")
        if existing.get("reuse_provenance") != common_provenance:
            raise FileExistsError("target feature cache has different provenance")
        for name, digest in existing["output_sha256"].items():
            if file_sha256(target_feature_dir / name) != digest:
                raise ValueError(f"target reused feature cache changed: {name}")
    else:
        temporary = _copytree_atomic(
            source_directory / "features",
            target_feature_dir,
        )
        imported_manifest = copy.deepcopy(source_feature_manifest)
        imported_manifest["extraction_seconds_this_process"] = 0.0
        imported_manifest["protocol_sha256"] = target_lock["protocol_sha256"]
        imported_manifest["reuse_provenance"] = common_provenance
        write_json(temporary / "manifest.json", imported_manifest)
        os.replace(temporary, target_feature_dir)

    reused_tomotopy = {}
    for seed, (source_result_dir, result, inventory) in tomotopy_sources.items():
        target_result_dir = target_directory / "core" / f"seed_{seed}" / "tomotopy"
        provenance = {
            **common_provenance,
            "source_inventory": inventory,
            "source_result_sha256": file_sha256(source_result_dir / "complete.json"),
        }
        if target_result_dir.exists():
            if read_json(target_result_dir / "reuse_provenance.json") != provenance:
                raise FileExistsError(
                    f"target Tomotopy seed {seed} has different provenance"
                )
            target_inventory = _inventory(target_result_dir)
            target_inventory.pop("reuse_provenance.json", None)
            if target_inventory != inventory:
                raise ValueError(
                    f"target reused Tomotopy seed {seed} changed after import"
                )
        else:
            temporary = _copytree_atomic(source_result_dir, target_result_dir)
            write_json(temporary / "reuse_provenance.json", provenance)
            os.replace(temporary, target_result_dir)
        reused_tomotopy[str(seed)] = {
            "inventory": inventory,
            "result_sha256": file_sha256(target_result_dir / "complete.json"),
            "training_iterations": result["training_iterations"],
        }

    reused_hybrid = {}
    for seed, source in hybrid_sources.items():
        if source is None:
            continue
        source_result_dir, result, inventory = source
        target_result_dir = target_directory / "core" / f"seed_{seed}" / "hybrid"
        provenance = {
            **common_provenance,
            "source_inventory": inventory,
            "source_result_sha256": file_sha256(source_result_dir / "complete.json"),
        }
        if target_result_dir.exists():
            if read_json(target_result_dir / "reuse_provenance.json") != provenance:
                raise FileExistsError(
                    f"target Hybrid seed {seed} has different provenance"
                )
            target_inventory = _inventory(target_result_dir)
            target_inventory.pop("reuse_provenance.json", None)
            if target_inventory != inventory:
                raise ValueError(
                    f"target reused Hybrid seed {seed} changed after import"
                )
        else:
            temporary = _copytree_atomic(source_result_dir, target_result_dir)
            write_json(temporary / "reuse_provenance.json", provenance)
            os.replace(temporary, target_result_dir)
        reused_hybrid[str(seed)] = {
            "discovery_epochs": result["discovery_epochs"],
            "inventory": inventory,
            "reference_steps": result["reference_steps"],
            "result_sha256": file_sha256(target_result_dir / "complete.json"),
        }

    continued_hybrid = {}
    for seed, checkpoints in continuation_sources.items():
        continued_hybrid[str(seed)] = _import_hybrid_continuation(
            source_directory=source_directory,
            target_directory=target_directory,
            source_config=source_config,
            target_config=target_config,
            target_lock=target_lock,
            seed=seed,
            checkpoints=checkpoints,
            common_provenance=common_provenance,
        )

    summary = {
        "chemical_evidence": False,
        "created_utc": created_utc,
        "derivation": derivation,
        "feature_source_inventory": source_feature_inventory,
        "reused": {
            "continued_hybrid": continued_hybrid,
            "features": True,
            "hybrid": reused_hybrid,
            "tomotopy": reused_tomotopy,
        },
        "software_provenance_only": True,
        "source_test_results_already_inspected": True,
    }
    write_json(target_directory / "core_artifact_reuse.json", summary)
    return summary


def reuse_tomotopy_artifacts(
    source_run: str | Path,
    target_run: str | Path,
) -> dict[str, Any]:
    """Import only hash-verified Tomotopy models into an implementation correction."""
    source_directory = Path(source_run).expanduser().resolve()
    target_directory = Path(target_run).expanduser().resolve()
    source_lock = verify_protocol(source_directory, verify_code=False)
    target_lock = verify_protocol(target_directory)
    source_config = load_config(source_directory / "config.resolved.json")
    target_config = load_config(target_directory / "config.resolved.json")
    recorded = target_lock.get("derivation", {})
    derivation = validate_implementation_correction_derivation(
        source_directory,
        target_config,
        recorded.get("reason"),
    )
    if recorded != derivation:
        raise ValueError("target lock does not contain the validated derivation")
    if target_lock.get("test_results_inspected") is not True:
        raise ValueError("implementation correction must disclose prior results")

    source_inputs = read_json(source_directory / "input_manifest.json")["files"]
    target_inputs = read_json(target_directory / "input_manifest.json")["files"]
    if source_inputs != target_inputs:
        raise ValueError("source and target input manifests differ")
    for name in IDENTICAL_FROZEN_ARTIFACTS:
        if source_lock["artifacts"].get(name) != target_lock["artifacts"].get(name):
            raise ValueError(f"source and target frozen artifacts differ: {name}")

    allowed_config_differences = {
        "protocol_name",
        "prior_test_results_inspected",
        "evaluation_timing",
    }
    source_values = source_config.as_dict()
    target_values = target_config.as_dict()
    differing = {
        key
        for key in set(source_values) | set(target_values)
        if source_values.get(key) != target_values.get(key)
    }
    if differing != allowed_config_differences:
        raise ValueError(
            "Tomotopy reuse requires identical scientific and execution configuration"
        )

    created_utc = datetime.now(timezone.utc).isoformat()
    common = {
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_run": str(source_directory),
        "target_protocol_sha256": target_lock["protocol_sha256"],
        "target_run": str(target_directory),
    }
    reused: dict[str, Any] = {}
    for seed in target_config.seeds:
        source_result_dir, result, inventory = _verify_tomotopy_source(
            source_directory,
            seed=seed,
            topic_count=target_config.num_topics,
            workers=target_config.tomotopy_training_workers,
            parallel=target_config.tomotopy_training_parallel,
        )
        target_result_dir = target_directory / "core" / f"seed_{seed}" / "tomotopy"
        provenance = {
            **common,
            "reuse_scope": "tomotopy_core_only",
            "source_inventory": inventory,
            "source_result_sha256": file_sha256(source_result_dir / "complete.json"),
        }
        if target_result_dir.exists():
            if read_json(target_result_dir / "reuse_provenance.json") != provenance:
                raise FileExistsError(
                    f"target Tomotopy seed {seed} has different provenance"
                )
            target_inventory = _inventory(target_result_dir)
            target_inventory.pop("reuse_provenance.json", None)
            if target_inventory != inventory:
                raise ValueError(
                    f"target reused Tomotopy seed {seed} changed after import"
                )
        else:
            temporary = _copytree_atomic(source_result_dir, target_result_dir)
            write_json(temporary / "reuse_provenance.json", provenance)
            os.replace(temporary, target_result_dir)
        reused[str(seed)] = {
            "inventory": inventory,
            "result_sha256": file_sha256(target_result_dir / "complete.json"),
            "training_iterations": result["training_iterations"],
        }

    summary = {
        "chemical_evidence": False,
        "created_utc": created_utc,
        "derivation": derivation,
        "forbidden_reuse": ["features", "hybrid"],
        "reuse_scope": "tomotopy_core_only",
        "reused": {"tomotopy": reused},
        "software_provenance_only": True,
        "source_test_results_already_inspected": True,
    }
    write_json(target_directory / "tomotopy_artifact_reuse.json", summary)
    return summary
