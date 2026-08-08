"""Resumable, training-only DreaMS feature construction."""

from __future__ import annotations

import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ms2lda_hybrid.dreams_features import (
    DREAMS_EMBEDDING_DIM,
    DreaMSFeatureBatch,
    DreaMSFeatureExtractor,
    parse_spectral_word,
)

from .checkpoint_hashes import extend_row_hash_ledger, validate_row_hash_ledger
from .config import file_sha256, load_config, read_json, resolve_input_paths, write_json
from .data import (
    SpectrumRecord,
    load_records,
    renormalize_peak_groups,
    split_records,
    to_matchms_spectrum,
)
from .protocol import (
    load_assignments,
    load_completion_rows,
    load_vocabulary,
    verify_frozen_input_files,
    verify_protocol,
)
from .runtime import peak_rss_bytes

_WORD_POOL_STRATEGY = "physical_peak_identity_v3"
_FEATURE_CHECKPOINT_SCHEMA = "msnlib-feature-checkpoint/v3"
_FEATURE_CHECKPOINT_FORMAT = "atomic-generations-v3"
_CHECKPOINT_ARRAY_NAMES = ("sums.npy", "weights.npy", "completed.npy")


def _empty_word_pool_counters() -> dict[str, int]:
    """Return counters that can be safely accumulated across checkpoints."""
    return {
        "training_peak_groups": 0,
        "matched_peak_groups": 0,
        "unmatched_peak_groups": 0,
        "matched_token_occurrences": 0,
        "unmatched_token_occurrences": 0,
        "retained_dreams_peak_states": 0,
        "fragment_collision_documents": 0,
        "fragment_collision_words": 0,
        "fragment_collision_extra_peak_groups": 0,
        "neutral_loss_collision_documents": 0,
        "neutral_loss_collision_words": 0,
        "neutral_loss_collision_extra_peak_groups": 0,
    }


def _peak_identity_key(value: float) -> int:
    """Return the exact float32 m/z identity used by pinned DreaMS."""
    converted = np.asarray(value, dtype=np.float32)
    if converted.ndim != 0 or not np.isfinite(converted) or converted <= 0:
        raise ValueError("peak identity requires a finite positive m/z")
    return int(converted.view(np.uint32))


def _validated_pool_counters(value: Any) -> dict[str, int]:
    expected = _empty_word_pool_counters()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("feature checkpoint has invalid word-pool counters")
    result = {}
    for name, count in value.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("feature checkpoint has invalid word-pool counters")
        result[name] = count
    return result


def _observed_only_record(
    record: SpectrumRecord,
    completion_row: dict[str, Any],
    *,
    significant_digits: int,
) -> SpectrumRecord:
    if not completion_row.get("eligible"):
        raise ValueError(f"ineligible completion record: {record.spectrum_id}")
    expected = set(map(int, completion_row["observed_peak_indices"]))
    observed = tuple(
        group for group in record.peak_groups if group.original_index in expected
    )
    if {group.original_index for group in observed} != expected:
        raise ValueError(f"completion peak indices changed for {record.spectrum_id}")
    from dataclasses import replace

    return replace(
        record,
        peak_groups=renormalize_peak_groups(
            observed,
            precursor_mz=record.precursor_mz,
            significant_digits=significant_digits,
        ),
    )


def _update_word_pool(
    *,
    records: Sequence[SpectrumRecord],
    feature_batch: DreaMSFeatureBatch,
    vocabulary_index: dict[str, int],
    sums: np.ndarray,
    weights: np.ndarray,
    counters: dict[str, int] | None = None,
) -> dict[str, int]:
    """Pool each physical peak's words against that peak's contextual state.

    Token rounding is intentionally not used to locate a contextual state. Two
    physical peaks may round to the same vocabulary word, but each still
    contributes its own DreaMS state and its own intensity-derived multiplicity.
    """
    if len(records) != len(feature_batch.identifiers):
        raise ValueError("record and feature counts differ")
    if counters is None:
        counters = _empty_word_pool_counters()
    elif set(counters) != set(_empty_word_pool_counters()):
        raise ValueError("word-pool counters have an unexpected schema")
    for row, record in enumerate(records):
        if record.spectrum_id != feature_batch.identifiers[row]:
            raise ValueError("feature identifiers are not row aligned")
        mask = feature_batch.peak_mask[row]
        peak_mz = feature_batch.peak_mz[row, mask]
        states = feature_batch.peak_embeddings[row, mask].astype(np.float32)
        group_keys = [_peak_identity_key(group.mz) for group in record.peak_groups]
        if len(set(group_keys)) != len(group_keys):
            raise ValueError(
                f"physical peak identity is ambiguous after float32 conversion: "
                f"{record.spectrum_id}"
            )
        state_keys = [_peak_identity_key(value) for value in peak_mz]
        if len(set(state_keys)) != len(state_keys):
            raise ValueError(
                f"DreaMS retained duplicate peak identities: {record.spectrum_id}"
            )
        unknown_states = set(state_keys) - set(group_keys)
        if unknown_states:
            raise ValueError(
                f"DreaMS peak identities do not match their source record: "
                f"{record.spectrum_id}"
            )
        state_by_identity = dict(zip(state_keys, states, strict=True))
        counters["retained_dreams_peak_states"] += len(state_by_identity)
        word_peak_groups: Counter[str] = Counter()
        for group in record.peak_groups:
            for word in set(group.tokens):
                if word in vocabulary_index and parse_spectral_word(word) is not None:
                    word_peak_groups[word] += 1
        collision_kinds: set[str] = set()
        for word, group_count in word_peak_groups.items():
            if group_count <= 1:
                continue
            parsed = parse_spectral_word(word)
            assert parsed is not None
            kind = parsed[0]
            prefix = "fragment" if kind == "frag" else "neutral_loss"
            counters[f"{prefix}_collision_words"] += 1
            counters[f"{prefix}_collision_extra_peak_groups"] += group_count - 1
            collision_kinds.add(prefix)
        for prefix in collision_kinds:
            counters[f"{prefix}_collision_documents"] += 1

        for group, identity in zip(record.peak_groups, group_keys, strict=True):
            eligible = [
                (word, count)
                for word, count in Counter(group.tokens).items()
                if word in vocabulary_index and parse_spectral_word(word) is not None
            ]
            if not eligible:
                continue
            counters["training_peak_groups"] += 1
            token_occurrences = sum(count for _, count in eligible)
            state = state_by_identity.get(identity)
            if state is None:
                counters["unmatched_peak_groups"] += 1
                counters["unmatched_token_occurrences"] += token_occurrences
                continue
            counters["matched_peak_groups"] += 1
            counters["matched_token_occurrences"] += token_occurrences
            for word, count in eligible:
                column = vocabulary_index[word]
                sums[column] += float(count) * state
                weights[column] += float(count)
    return counters


def _checkpoint_generation_directories(checkpoint_dir: Path) -> list[Path]:
    """Return published checkpoint generations from newest to oldest."""
    if not checkpoint_dir.exists():
        return []
    candidates = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("generation-"):
            continue
        try:
            generation = int(path.name.removeprefix("generation-"))
        except ValueError:
            continue
        candidates.append((generation, path))
    return [path for _, path in sorted(candidates, reverse=True)]


def _fsync_path(path: Path) -> None:
    """Best-effort durability barrier for a newly written file or directory."""
    flags = os.O_RDONLY
    if path.is_dir() and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_feature_checkpoint_generation(
    path: Path,
    *,
    protocol_sha256: str,
    total_rows: int,
    vocabulary_size: int,
    embedding_dim: int,
    global_embeddings: np.ndarray,
    identifiers_sha256: str,
) -> dict[str, Any]:
    """Read and fully verify one atomic feature-checkpoint generation."""
    state = read_json(path / "state.json")
    if state.get("schema") != _FEATURE_CHECKPOINT_SCHEMA:
        raise ValueError("feature checkpoint schema mismatch")
    if state.get("protocol_sha256") != protocol_sha256:
        raise ValueError("feature checkpoint protocol hash mismatch")
    if state.get("word_pool_strategy") != _WORD_POOL_STRATEGY:
        raise ValueError("feature checkpoint word-pool strategy mismatch")
    if int(state.get("total_rows", -1)) != total_rows:
        raise ValueError("feature checkpoint row count mismatch")
    if int(state.get("vocabulary_size", -1)) != vocabulary_size:
        raise ValueError("feature checkpoint vocabulary size mismatch")
    if int(state.get("embedding_dim", -1)) != embedding_dim:
        raise ValueError("feature checkpoint embedding dimension mismatch")
    if state.get("identifiers_sha256") != identifiers_sha256:
        raise ValueError("feature checkpoint identifiers changed")
    try:
        expected_generation = int(path.name.removeprefix("generation-"))
    except ValueError as exc:
        raise ValueError("invalid feature checkpoint directory name") from exc
    if int(state.get("generation", -1)) != expected_generation:
        raise ValueError("feature checkpoint generation mismatch")
    hashes = state.get("output_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_CHECKPOINT_ARRAY_NAMES):
        raise ValueError("feature checkpoint array manifest is invalid")
    for name in _CHECKPOINT_ARRAY_NAMES:
        array_path = path / name
        if not array_path.is_file() or file_sha256(array_path) != hashes[name]:
            raise ValueError(f"feature checkpoint array changed: {name}")

    sums = np.load(path / "sums.npy", allow_pickle=False)
    weights = np.load(path / "weights.npy", allow_pickle=False)
    completed = np.load(path / "completed.npy", allow_pickle=False)
    if sums.shape != (vocabulary_size, embedding_dim) or sums.dtype != np.float32:
        raise ValueError("feature checkpoint sums have invalid shape or dtype")
    if weights.shape != (vocabulary_size,) or weights.dtype != np.float64:
        raise ValueError("feature checkpoint weights have invalid shape or dtype")
    if completed.shape != (total_rows,) or completed.dtype != np.bool_:
        raise ValueError("feature checkpoint completion mask is invalid")
    if not np.all(np.isfinite(sums)) or not np.all(np.isfinite(weights)):
        raise ValueError("feature checkpoint arrays contain non-finite values")
    if np.any(weights < 0):
        raise ValueError("feature checkpoint weights cannot be negative")
    if (
        global_embeddings.shape != (total_rows, embedding_dim)
        or global_embeddings.dtype != np.float32
    ):
        raise ValueError("global embedding checkpoint has invalid shape or dtype")
    completed_rows = int(state.get("completed_rows", -1))
    if completed_rows < 0 or completed_rows > total_rows:
        raise ValueError("feature checkpoint completed row count is invalid")
    if not np.all(completed[:completed_rows]) or np.any(completed[completed_rows:]):
        raise ValueError("feature checkpoint completion mask is non-contiguous")
    global_embedding_chunks = validate_row_hash_ledger(
        global_embeddings,
        state.get("global_embedding_chunks"),
        completed_rows=completed_rows,
    )
    cumulative_seconds = float(state.get("cumulative_extraction_seconds", -1.0))
    if not np.isfinite(cumulative_seconds) or cumulative_seconds < 0:
        raise ValueError("feature checkpoint extraction time is invalid")
    return {
        "state": state,
        "sums": sums,
        "weights": weights,
        "completed": completed,
        "counters": _validated_pool_counters(state.get("word_pool_counters")),
        "global_embedding_chunks": global_embedding_chunks,
    }


def _load_latest_feature_checkpoint(
    checkpoint_dir: Path,
    *,
    protocol_sha256: str,
    total_rows: int,
    vocabulary_size: int,
    embedding_dim: int,
    global_embeddings: np.ndarray,
    identifiers_sha256: str,
) -> dict[str, Any] | None:
    """Restore the newest valid generation, falling back from corrupt ones."""
    candidates = _checkpoint_generation_directories(checkpoint_dir)
    rejected = []
    for path in candidates:
        try:
            restored = _read_feature_checkpoint_generation(
                path,
                protocol_sha256=protocol_sha256,
                total_rows=total_rows,
                vocabulary_size=vocabulary_size,
                embedding_dim=embedding_dim,
                global_embeddings=global_embeddings,
                identifiers_sha256=identifiers_sha256,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            rejected.append({"generation": path.name, "reason": str(exc)})
            continue
        restored["rejected_newer_generations"] = rejected
        return restored
    if candidates:
        raise RuntimeError(
            "no valid atomic feature checkpoint generation remains; "
            "delete the incomplete feature cache and restart extraction"
        )
    return None


def _write_feature_checkpoint_generation(
    checkpoint_dir: Path,
    *,
    generation: int,
    protocol_sha256: str,
    completed: np.ndarray,
    sums: np.ndarray,
    weights: np.ndarray,
    cumulative_extraction_seconds: float,
    word_pool_counters: dict[str, int],
    global_embeddings: np.ndarray,
    global_embedding_chunks: Sequence[dict[str, Any]],
    checkpoint_start: int,
    checkpoint_end: int,
    identifiers_sha256: str,
    keep: int = 2,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish one all-or-nothing feature checkpoint and retain two fallbacks."""
    if generation < 1:
        raise ValueError("feature checkpoint generation must be positive")
    if keep < 2:
        raise ValueError("at least two feature checkpoint generations are required")
    if sums.ndim != 2 or sums.dtype != np.float32:
        raise ValueError("feature checkpoint sums must be a float32 matrix")
    if weights.shape != (sums.shape[0],) or weights.dtype != np.float64:
        raise ValueError("feature checkpoint weights must align with sums")
    if completed.ndim != 1 or completed.dtype != np.bool_:
        raise ValueError("feature checkpoint completed mask must be boolean")
    if (
        global_embeddings.shape != (len(completed), sums.shape[1])
        or global_embeddings.dtype != np.float32
    ):
        raise ValueError("global embeddings must align with feature checkpoint")
    if (
        not np.isfinite(cumulative_extraction_seconds)
        or cumulative_extraction_seconds < 0
    ):
        raise ValueError("feature checkpoint extraction time is invalid")
    counters = _validated_pool_counters(word_pool_counters)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"generation-{generation:08d}"
    destination = checkpoint_dir / stem
    if destination.exists():
        raise FileExistsError(f"feature checkpoint already exists: {destination}")
    temporary = checkpoint_dir / f".{stem}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir()

    def fault(stage: str) -> None:
        if fault_hook is not None:
            fault_hook(stage)

    try:
        np.save(temporary / "sums.npy", np.asarray(sums, dtype=np.float32))
        _fsync_path(temporary / "sums.npy")
        fault("after_sums")
        np.save(temporary / "weights.npy", np.asarray(weights, dtype=np.float64))
        _fsync_path(temporary / "weights.npy")
        fault("after_weights")
        np.save(temporary / "completed.npy", np.asarray(completed, dtype=np.bool_))
        _fsync_path(temporary / "completed.npy")
        fault("after_completed")
        completed_array = np.asarray(completed, dtype=np.bool_)
        false_rows = np.flatnonzero(~completed_array)
        completed_rows = int(false_rows[0]) if len(false_rows) else len(completed_array)
        if np.any(completed_array[completed_rows:]):
            raise ValueError("feature checkpoint completion mask is non-contiguous")
        if checkpoint_end != completed_rows or checkpoint_start >= checkpoint_end:
            raise ValueError("feature checkpoint row interval is invalid")
        updated_embedding_chunks = extend_row_hash_ledger(
            global_embedding_chunks,
            global_embeddings,
            start=checkpoint_start,
            end=checkpoint_end,
        )
        state = {
            "schema": _FEATURE_CHECKPOINT_SCHEMA,
            "generation": generation,
            "protocol_sha256": protocol_sha256,
            "word_pool_strategy": _WORD_POOL_STRATEGY,
            "total_rows": len(completed_array),
            "completed_rows": completed_rows,
            "vocabulary_size": int(np.asarray(weights).shape[0]),
            "embedding_dim": int(np.asarray(sums).shape[1]),
            "identifiers_sha256": identifiers_sha256,
            "global_embedding_chunks": updated_embedding_chunks,
            "cumulative_extraction_seconds": float(cumulative_extraction_seconds),
            "word_pool_counters": counters,
            "output_sha256": {
                name: file_sha256(temporary / name) for name in _CHECKPOINT_ARRAY_NAMES
            },
        }
        write_json(temporary / "state.json", state)
        _fsync_path(temporary / "state.json")
        _fsync_path(temporary)
        fault("after_state")
        os.replace(temporary, destination)
        _fsync_path(checkpoint_dir)
        fault("after_publish")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    retained = 0
    for path in _checkpoint_generation_directories(checkpoint_dir):
        try:
            _read_feature_checkpoint_generation(
                path,
                protocol_sha256=protocol_sha256,
                total_rows=len(completed),
                vocabulary_size=len(weights),
                embedding_dim=sums.shape[1],
                global_embeddings=global_embeddings,
                identifiers_sha256=identifiers_sha256,
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        retained += 1
        if retained > keep:
            shutil.rmtree(path)
    return state


def _ensure_feature_checkpoint_format(
    feature_dir: Path,
    *,
    protocol_sha256: str,
    rows: int,
    vocabulary_size: int,
    identifiers_sha256: str,
) -> None:
    """Initialize v3 state or reject ambiguous legacy additive checkpoints."""
    format_path = feature_dir / "checkpoint_format.json"
    legacy_paths = tuple(
        feature_dir / name
        for name in (
            "completed.npy",
            "word_embedding_sums.npy",
            "word_embedding_weights.npy",
            "progress.json",
        )
    )
    if not format_path.exists():
        if any(path.exists() for path in legacy_paths):
            raise RuntimeError(
                "legacy or partial feature checkpoint state detected; its pooled "
                "sums cannot be resumed safely, so use a new feature directory"
            )
        write_json(
            format_path,
            {
                "format": _FEATURE_CHECKPOINT_FORMAT,
                "protocol_sha256": protocol_sha256,
                "rows": rows,
                "vocabulary_size": vocabulary_size,
                "embedding_dim": DREAMS_EMBEDDING_DIM,
                "word_pool_strategy": _WORD_POOL_STRATEGY,
                "identifiers_sha256": identifiers_sha256,
            },
        )
        return
    state = read_json(format_path)
    expected = {
        "format": _FEATURE_CHECKPOINT_FORMAT,
        "protocol_sha256": protocol_sha256,
        "rows": rows,
        "vocabulary_size": vocabulary_size,
        "embedding_dim": DREAMS_EMBEDDING_DIM,
        "word_pool_strategy": _WORD_POOL_STRATEGY,
        "identifiers_sha256": identifiers_sha256,
    }
    if state != expected:
        raise ValueError("feature checkpoint format does not match this run")


def prepare_features(
    run_dir: str | Path,
    *,
    extraction_batch_size: int = 16,
    checkpoint_every_chunks: int = 25,
) -> dict[str, Any]:
    """Extract reusable global features and pool train-only word features."""
    lock = verify_protocol(run_dir)
    directory = Path(run_dir).expanduser().resolve()
    verify_frozen_input_files(directory, names={"mgf"}, lock=lock)
    feature_dir = directory / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = feature_dir / "manifest.json"
    if final_manifest_path.exists():
        manifest = read_json(final_manifest_path)
        if manifest.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("feature cache belongs to another frozen protocol")
        if manifest.get("word_pool_strategy") != _WORD_POOL_STRATEGY:
            raise RuntimeError(
                "feature cache uses an incompatible or unsafe word-pooling strategy; "
                "use a new feature directory and rebuild Hybrid features"
            )
        for name, digest in manifest["output_sha256"].items():
            if file_sha256(feature_dir / name) != digest:
                raise ValueError(f"feature cache changed: {name}")
        return manifest

    config = load_config(directory / "config.resolved.json")
    inputs = resolve_input_paths(config, lock["data_root"])
    records, _ = load_records(inputs["mgf"], config)
    assignments = load_assignments(directory)
    completion_rows = load_completion_rows(directory)
    vocabulary = load_vocabulary(directory)
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}
    if len(vocabulary_index) != len(vocabulary):
        raise ValueError("frozen vocabulary is not unique")

    identifiers = [record.spectrum_id for record in records]
    identifiers_path = feature_dir / "identifiers.json"
    identifiers_payload = {"identifiers": identifiers}
    if identifiers_path.exists():
        if read_json(identifiers_path) != identifiers_payload:
            raise ValueError("feature identifiers changed")
    else:
        write_json(identifiers_path, identifiers_payload)
    identifiers_sha256 = file_sha256(identifiers_path)
    global_path = feature_dir / "global_embeddings.npy"
    completed_path = feature_dir / "completed.npy"
    checkpoint_dir = feature_dir / "checkpoint_generations"
    shape = (len(records), DREAMS_EMBEDDING_DIM)
    _ensure_feature_checkpoint_format(
        feature_dir,
        protocol_sha256=lock["protocol_sha256"],
        rows=len(records),
        vocabulary_size=len(vocabulary),
        identifiers_sha256=identifiers_sha256,
    )
    checkpoint_candidates = _checkpoint_generation_directories(checkpoint_dir)
    if checkpoint_candidates and not global_path.exists():
        raise RuntimeError(
            "feature checkpoint exists but global embeddings are missing"
        )
    had_global_embeddings = global_path.exists()
    global_embeddings = np.lib.format.open_memmap(
        global_path,
        mode="r+" if had_global_embeddings else "w+",
        dtype=np.float32,
        shape=shape,
    )
    restored = _load_latest_feature_checkpoint(
        checkpoint_dir,
        protocol_sha256=lock["protocol_sha256"],
        total_rows=len(records),
        vocabulary_size=len(vocabulary),
        embedding_dim=DREAMS_EMBEDDING_DIM,
        global_embeddings=global_embeddings,
        identifiers_sha256=identifiers_sha256,
    )
    if restored is not None:
        completed = restored["completed"]
        sums = restored["sums"]
        weights = restored["weights"]
        pool_counters = restored["counters"]
        global_embedding_chunks = restored["global_embedding_chunks"]
        previous_extraction_seconds = float(
            restored["state"]["cumulative_extraction_seconds"]
        )
        checkpoint_generation = max(
            int(path.name.removeprefix("generation-"))
            for path in _checkpoint_generation_directories(checkpoint_dir)
        )
        last_valid_checkpoint_generation = int(restored["state"]["generation"])
        rejected_checkpoint_generations = restored["rejected_newer_generations"]
    else:
        # If this is a pre-first-checkpoint interruption, ``w+`` safely
        # overwrites any global rows because no additive state was published.
        if had_global_embeddings:
            del global_embeddings
            global_embeddings = np.lib.format.open_memmap(
                global_path, mode="w+", dtype=np.float32, shape=shape
            )
        completed = np.zeros(len(records), dtype=np.bool_)
        sums = np.zeros((len(vocabulary), DREAMS_EMBEDDING_DIM), dtype=np.float32)
        weights = np.zeros(len(vocabulary), dtype=np.float64)
        pool_counters = _empty_word_pool_counters()
        global_embedding_chunks = []
        previous_extraction_seconds = 0.0
        checkpoint_generation = 0
        last_valid_checkpoint_generation = 0
        rejected_checkpoint_generations = []
    first_incomplete = np.flatnonzero(~completed)
    start_row = int(first_incomplete[0]) if len(first_incomplete) else len(records)
    if np.any(completed[start_row:]):
        raise ValueError("feature progress is non-contiguous")

    extractor_started = time.perf_counter()
    extractor = DreaMSFeatureExtractor(device="cpu")
    initialization_seconds = time.perf_counter() - extractor_started
    extraction_seconds = 0.0
    chunks = 0
    checkpoint_start = start_row
    for start in range(start_row, len(records), extraction_batch_size):
        end = min(start + extraction_batch_size, len(records))
        source_records = records[start:end]
        extraction_records = []
        for record in source_records:
            split = assignments[record.spectrum_id]
            if split == "train":
                extraction_records.append(record)
            else:
                extraction_records.append(
                    _observed_only_record(
                        record,
                        completion_rows[record.spectrum_id],
                        significant_digits=config.significant_digits,
                    )
                )
        spectra = [to_matchms_spectrum(record) for record in extraction_records]
        started = time.perf_counter()
        batch = extractor.extract(
            spectra,
            identifiers=[record.spectrum_id for record in source_records],
            batch_size=extraction_batch_size,
        )
        extraction_seconds += time.perf_counter() - started
        global_embeddings[start:end] = batch.spectrum_embeddings
        train_positions = [
            index
            for index, record in enumerate(source_records)
            if assignments[record.spectrum_id] == "train"
        ]
        if train_positions:
            train_records = [source_records[index] for index in train_positions]
            train_batch = DreaMSFeatureBatch(
                identifiers=tuple(record.spectrum_id for record in train_records),
                spectrum_embeddings=batch.spectrum_embeddings[train_positions],
                peak_embeddings=batch.peak_embeddings[train_positions],
                peak_mz=batch.peak_mz[train_positions],
                peak_mask=batch.peak_mask[train_positions],
                precursor_mz=batch.precursor_mz[train_positions],
                provenance=batch.provenance,
            )
            _update_word_pool(
                records=train_records,
                feature_batch=train_batch,
                vocabulary_index=vocabulary_index,
                sums=sums,
                weights=weights,
                counters=pool_counters,
            )
        global_embeddings.flush()
        chunks += 1
        if chunks % checkpoint_every_chunks == 0 or end == len(records):
            completed[checkpoint_start:end] = True
            checkpoint_generation += 1
            checkpoint_state = _write_feature_checkpoint_generation(
                checkpoint_dir,
                generation=checkpoint_generation,
                protocol_sha256=lock["protocol_sha256"],
                completed=completed,
                sums=sums,
                weights=weights,
                cumulative_extraction_seconds=(
                    previous_extraction_seconds + extraction_seconds
                ),
                word_pool_counters=pool_counters,
                global_embeddings=global_embeddings,
                global_embedding_chunks=global_embedding_chunks,
                checkpoint_start=checkpoint_start,
                checkpoint_end=end,
                identifiers_sha256=identifiers_sha256,
            )
            global_embedding_chunks = checkpoint_state["global_embedding_chunks"]
            last_valid_checkpoint_generation = checkpoint_generation
            checkpoint_start = end

    if not bool(np.all(completed)):
        raise RuntimeError("feature extraction ended with incomplete rows")
    if (
        pool_counters["matched_peak_groups"] + pool_counters["unmatched_peak_groups"]
        != pool_counters["training_peak_groups"]
    ):
        raise RuntimeError("peak identity assignment counts do not reconcile")

    word_embeddings = np.divide(
        sums,
        weights[:, None],
        out=np.zeros_like(sums),
        where=weights[:, None] > 0,
    )
    word_embeddings_path = feature_dir / "word_embeddings.npy"
    np.save(word_embeddings_path, word_embeddings)
    np.save(completed_path, completed)
    output_paths = (
        identifiers_path,
        global_path,
        completed_path,
        word_embeddings_path,
    )
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "rows": len(records),
        "train_rows": len(split_records(records, assignments, "train")),
        "embedding_dim": DREAMS_EMBEDDING_DIM,
        "training_only_word_pool": True,
        "word_pool_strategy": _WORD_POOL_STRATEGY,
        "peak_identity_mapping": "exact_float32_source_mz",
        "discarded_peak_state_policy": "unmatched",
        "word_pool_counters": pool_counters,
        "nontraining_features_use_observed_peak_groups_only": True,
        "nontraining_observed_intensity_renormalized_after_split": True,
        "matched_word_embeddings": int((weights > 0).sum()),
        "unmatched_word_embeddings": int((weights == 0).sum()),
        "last_checkpoint_generation": last_valid_checkpoint_generation,
        "global_embedding_chunks": len(global_embedding_chunks),
        "rejected_newer_checkpoint_generations": rejected_checkpoint_generations,
        "extractor_initialization_seconds": initialization_seconds,
        "extraction_seconds_this_process": extraction_seconds,
        "cumulative_extraction_seconds": (
            previous_extraction_seconds + extraction_seconds
        ),
        "peak_rss_bytes": peak_rss_bytes(),
        "extractor_provenance": extractor.provenance,
        "output_sha256": {path.name: file_sha256(path) for path in output_paths},
    }
    write_json(final_manifest_path, manifest)
    return manifest
