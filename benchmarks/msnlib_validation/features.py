"""Resumable, training-only DreaMS feature construction."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ms2lda_hybrid.dreams_features import (
    DREAMS_EMBEDDING_DIM,
    DreaMSFeatureBatch,
    DreaMSFeatureExtractor,
    parse_spectral_word,
)

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
    mz_tolerance: float = 0.02,
) -> None:
    """Accumulate count-weighted contextual peak states from training rows only."""
    if len(records) != len(feature_batch.identifiers):
        raise ValueError("record and feature counts differ")
    for row, record in enumerate(records):
        if record.spectrum_id != feature_batch.identifiers[row]:
            raise ValueError("feature identifiers are not row aligned")
        mask = feature_batch.peak_mask[row]
        peak_mz = feature_batch.peak_mz[row, mask]
        states = feature_batch.peak_embeddings[row, mask].astype(np.float32)
        if not len(peak_mz):
            continue
        for word, count in Counter(record.words).items():
            column = vocabulary_index.get(word)
            parsed = parse_spectral_word(word)
            if column is None or parsed is None:
                continue
            kind, value = parsed
            target = value if kind == "frag" else record.precursor_mz - value
            if target <= 0:
                continue
            peak = int(np.argmin(np.abs(peak_mz - target)))
            if abs(float(peak_mz[peak]) - target) <= mz_tolerance:
                sums[column] += float(count) * states[peak]
                weights[column] += float(count)


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
    if not identifiers_path.exists():
        write_json(identifiers_path, {"identifiers": identifiers})
    global_path = feature_dir / "global_embeddings.npy"
    completed_path = feature_dir / "completed.npy"
    sums_path = feature_dir / "word_embedding_sums.npy"
    weights_path = feature_dir / "word_embedding_weights.npy"
    progress_path = feature_dir / "progress.json"
    shape = (len(records), DREAMS_EMBEDDING_DIM)
    if global_path.exists():
        global_embeddings = np.lib.format.open_memmap(
            global_path, mode="r+", dtype=np.float32, shape=shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="r+", dtype=np.bool_, shape=(len(records),)
        )
        sums = np.load(sums_path)
        weights = np.load(weights_path)
        progress = read_json(progress_path)
        previous_extraction_seconds = float(
            progress.get("cumulative_extraction_seconds", 0.0)
        )
    else:
        global_embeddings = np.lib.format.open_memmap(
            global_path, mode="w+", dtype=np.float32, shape=shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=(len(records),)
        )
        completed[:] = False
        sums = np.zeros((len(vocabulary), DREAMS_EMBEDDING_DIM), dtype=np.float32)
        weights = np.zeros(len(vocabulary), dtype=np.float64)
        np.save(sums_path, sums)
        np.save(weights_path, weights)
        previous_extraction_seconds = 0.0
        write_json(
            progress_path,
            {"completed_rows": 0, "cumulative_extraction_seconds": 0.0},
        )
    first_incomplete = np.flatnonzero(~np.asarray(completed))
    start_row = int(first_incomplete[0]) if len(first_incomplete) else len(records)
    if np.any(np.asarray(completed)[start_row:]):
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
            )
        global_embeddings.flush()
        chunks += 1
        if chunks % checkpoint_every_chunks == 0 or end == len(records):
            # Commit pooled sums before marking their source rows complete. If
            # interrupted between checkpoints, those rows are recomputed rather
            # than silently omitted from the training-only word pool.
            np.save(sums_path, sums)
            np.save(weights_path, weights)
            completed[checkpoint_start:end] = True
            completed.flush()
            checkpoint_start = end
            write_json(
                progress_path,
                {
                    "completed_rows": end,
                    "total_rows": len(records),
                    "extraction_seconds_this_process": extraction_seconds,
                    "cumulative_extraction_seconds": (
                        previous_extraction_seconds + extraction_seconds
                    ),
                    "peak_rss_bytes": peak_rss_bytes(),
                },
            )

    if not bool(np.all(np.asarray(completed))):
        raise RuntimeError("feature extraction ended with incomplete rows")

    word_embeddings = np.divide(
        sums,
        weights[:, None],
        out=np.zeros_like(sums),
        where=weights[:, None] > 0,
    )
    word_embeddings_path = feature_dir / "word_embeddings.npy"
    np.save(word_embeddings_path, word_embeddings)
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
        "nontraining_features_use_observed_peak_groups_only": True,
        "nontraining_observed_intensity_renormalized_after_split": True,
        "matched_word_embeddings": int((weights > 0).sum()),
        "unmatched_word_embeddings": int((weights == 0).sum()),
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
