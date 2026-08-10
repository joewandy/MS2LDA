# ruff: noqa: C901, PLR0915
"""Count-only matrices and explicitly isolated DreaMS feature access."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from benchmarks.msnlib_validation.config import (
    file_sha256,
    load_config,
    read_json,
    write_json,
)
from benchmarks.msnlib_validation.data import (
    SpectrumRecord,
    load_records,
    renormalize_peak_groups,
    split_records,
    to_matchms_spectrum,
)
from benchmarks.msnlib_validation.protocol import (
    load_assignments,
    load_completion_rows,
    load_vocabulary,
)

from .runtime import configure_cpu_threads
from .spec import verify_study


def _completion_groups(
    record: SpectrumRecord,
    row: dict[str, Any],
    *,
    significant_digits: int,
) -> tuple[list[str], list[str], SpectrumRecord]:
    """Reproduce frozen peak-group completion without importing feature workers."""
    if not row.get("eligible"):
        msg = f"held-out spectrum is ineligible: {record.spectrum_id}"
        raise ValueError(msg)
    observed_indices = set(map(int, row["observed_peak_indices"]))
    completion_indices = set(map(int, row["completion_peak_indices"]))
    all_indices = {group.original_index for group in record.peak_groups}
    if (
        observed_indices & completion_indices
        or observed_indices | completion_indices != all_indices
    ):
        msg = f"completion groups changed: {record.spectrum_id}"
        raise ValueError(msg)
    observed_groups = tuple(
        group
        for group in record.peak_groups
        if group.original_index in observed_indices
    )
    completion_groups = tuple(
        group
        for group in record.peak_groups
        if group.original_index in completion_indices
    )
    observed_groups = renormalize_peak_groups(
        observed_groups,
        precursor_mz=record.precursor_mz,
        significant_digits=significant_digits,
    )
    observed_record = replace(record, peak_groups=observed_groups)
    observed_words = [token for group in observed_groups for token in group.tokens]
    completion_words = [token for group in completion_groups for token in group.tokens]
    return observed_words, completion_words, observed_record


if TYPE_CHECKING:
    from collections.abc import Sequence


def _atomic_save_npz(path: Path, matrix: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        sp.save_npz(temporary, matrix, compressed=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _matrix(
    documents: Sequence[Sequence[str]],
    vocabulary: Sequence[str],
) -> sp.csr_matrix:
    index = {word: column for column, word in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, words in enumerate(documents):
        counts: dict[int, float] = {}
        for word in words:
            column = index.get(word)
            if column is not None:
                counts[column] = counts.get(column, 0.0) + 1.0
        rows.extend([row] * len(counts))
        columns.extend(counts)
        values.extend(counts.values())
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float32,
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_records(run_dir: Path) -> tuple[Any, list[SpectrumRecord]]:
    lock = verify_study(run_dir)
    source = Path(lock["source_run"])
    source_lock = read_json(source / "protocol.lock.json")
    config = load_config(source / "config.resolved.json")
    mgf = Path(source_lock["data_root"]) / config.input_files["mgf"]["relative_path"]
    records, _ = load_records(mgf, config)
    return config, records


def _source_model_vocabulary(source: Path) -> tuple[str, ...]:
    """Recover the Hybrid model's insertion order without loading DreaMS."""
    import torch

    payload = torch.load(
        source / "core/seed_42/hybrid/model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model_vocabulary = tuple(map(str, payload["vocabulary"]))
    protocol_vocabulary = load_vocabulary(source)
    same_size = len(model_vocabulary) == len(protocol_vocabulary)
    same_words = set(model_vocabulary) == set(protocol_vocabulary)
    if not same_size or not same_words:
        msg = "corrected Hybrid and protocol vocabularies contain different words"
        raise ValueError(msg)
    return model_vocabulary


def prepare_count_inputs(run_dir: str | Path) -> dict[str, Any]:
    """Materialize chemistry-free matrices without loading any DreaMS cache."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    output = directory / "shared" / "counts"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"count input changed: {name}"
                raise ValueError(msg)
        return result

    source = Path(lock["source_run"])
    config, records = _source_records(directory)
    assignments = load_assignments(source)
    completion_rows = load_completion_rows(source)
    vocabulary = _source_model_vocabulary(source)
    vocabulary_set = set(vocabulary)
    output.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, sp.csr_matrix] = {}
    metadata_rows: list[dict[str, Any]] = []
    identifiers: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        selected = split_records(records, assignments, split)
        identifiers[split] = [record.spectrum_id for record in selected]
        if split == "train":
            matrices["train"] = _matrix(
                [record.words for record in selected],
                vocabulary,
            )
        else:
            observed_documents: list[list[str]] = []
            completion_documents: list[list[str]] = []
            full_documents: list[list[str]] = []
            for record in selected:
                observed, completion, _ = _completion_groups(
                    record,
                    completion_rows[record.spectrum_id],
                    significant_digits=config.significant_digits,
                )
                observed_documents.append(observed)
                completion_documents.append(completion)
                full_documents.append(record.words)
                metadata_rows.append(
                    {
                        "split": split,
                        "spectrum_id": record.spectrum_id,
                        "connectivity_key": record.connectivity_key,
                        "scaffold_key": record.scaffold_key,
                        "smiles": record.smiles,
                        "observed_tokens": len(observed),
                        "observed_oov_tokens": sum(
                            word not in vocabulary_set for word in observed
                        ),
                        "completion_tokens": len(completion),
                        "completion_oov_tokens": sum(
                            word not in vocabulary_set for word in completion
                        ),
                    },
                )
            matrices[f"{split}_observed"] = _matrix(observed_documents, vocabulary)
            matrices[f"{split}_completion"] = _matrix(completion_documents, vocabulary)
            matrices[f"{split}_full"] = _matrix(full_documents, vocabulary)

    outputs: list[Path] = []
    for name, matrix in matrices.items():
        path = output / f"{name}.npz"
        _atomic_save_npz(path, matrix)
        outputs.append(path)
    identifiers_path = output / "identifiers.json"
    write_json(identifiers_path, identifiers)
    outputs.append(identifiers_path)
    metadata_path = output / "heldout_records.jsonl"
    _write_jsonl(metadata_path, metadata_rows)
    outputs.append(metadata_path)
    vocabulary_path = output / "vocabulary.json"
    write_json(vocabulary_path, {"vocabulary": list(vocabulary)})
    outputs.append(vocabulary_path)
    result = {
        "schema_version": "msnlib-simplification/count-inputs-v1",
        "spec_sha256": lock["spec_sha256"],
        "chemical_labels_used_for_model_inputs": False,
        "dreams_cache_loaded": False,
        "vocabulary_size": len(vocabulary),
        "vocabulary_order": "corrected_hybrid_model_insertion_order",
        "matrix_shapes": {
            name: list(matrix.shape) for name, matrix in matrices.items()
        },
        "matrix_nnz": {name: int(matrix.nnz) for name, matrix in matrices.items()},
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(complete_path, result)
    return result


def load_count_matrix(run_dir: str | Path, name: str) -> sp.csr_matrix:
    """Load one verified count matrix."""
    directory = Path(run_dir).expanduser().resolve()
    complete = prepare_count_inputs(directory)
    filename = f"{name}.npz"
    path = directory / "shared" / "counts" / filename
    if complete["output_sha256"].get(filename) != file_sha256(path):
        msg = f"count matrix changed: {name}"
        raise ValueError(msg)
    return sp.load_npz(path).tocsr().astype(np.float32)


def load_vocabulary_copy(run_dir: str | Path) -> tuple[str, ...]:
    """Load the study-local frozen vocabulary."""
    path = (
        Path(run_dir).expanduser().resolve() / "shared" / "counts" / "vocabulary.json"
    )
    prepare_count_inputs(run_dir)
    return tuple(map(str, read_json(path)["vocabulary"]))


def load_identifiers(run_dir: str | Path, split: str) -> tuple[str, ...]:
    """Load row identifiers for a frozen split."""
    prepare_count_inputs(run_dir)
    value = read_json(
        Path(run_dir).expanduser().resolve() / "shared" / "counts" / "identifiers.json",
    )
    return tuple(map(str, value[split]))


def load_observed_dreams_embeddings(run_dir: str | Path, split: str) -> np.ndarray:
    """Load observed-peak DreaMS embeddings only for an explicit DreaMS arm."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    source = Path(lock["source_run"])
    feature_manifest = read_json(source / "features" / "manifest.json")
    for name, digest in feature_manifest["output_sha256"].items():
        if file_sha256(source / "features" / name) != digest:
            msg = f"source DreaMS feature changed: {name}"
            raise ValueError(msg)
    source_ids = tuple(
        map(str, read_json(source / "features" / "identifiers.json")["identifiers"]),
    )
    row_by_id = {identifier: row for row, identifier in enumerate(source_ids)}
    requested = load_identifiers(directory, split)
    rows = [row_by_id[identifier] for identifier in requested]
    values = np.load(source / "features" / "global_embeddings.npy", mmap_mode="r")
    return np.asarray(values[rows], dtype=np.float32)


def _full_validation_paths(directory: Path) -> tuple[Path, Path, Path]:
    root = directory / "shared" / "dreams_full_validation"
    return root / "embeddings.npy", root / "completed.npy", root / "complete.json"


def prepare_full_validation_dreams_embeddings(
    run_dir: str | Path,
    *,
    batch_size: int = 16,
    checkpoint_rows: int = 400,
) -> dict[str, Any]:
    """Extract resumable full-spectrum validation embeddings for chemical SOS."""
    from ms2lda_hybrid.dreams_features import (
        DREAMS_EMBEDDING_DIM,
        DreaMSFeatureExtractor,
    )

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    embeddings_path, completed_path, complete_path = _full_validation_paths(directory)
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(complete_path.parent / name) != digest:
                msg = f"full validation DreaMS cache changed: {name}"
                raise ValueError(msg)
        return result
    _, records = _source_records(directory)
    source = Path(lock["source_run"])
    assignments = load_assignments(source)
    selected = split_records(records, assignments, "validation")
    expected_ids = load_identifiers(directory, "validation")
    if tuple(record.spectrum_id for record in selected) != expected_ids:
        msg = "validation record order changed"
        raise ValueError(msg)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(selected), DREAMS_EMBEDDING_DIM)
    existed = embeddings_path.exists()
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="r+" if existed else "w+",
        dtype=np.float32,
        shape=shape,
    )
    if completed_path.exists():
        completed = np.load(completed_path)
        if completed.shape != (len(selected),):
            msg = "full validation checkpoint shape changed"
            raise ValueError(msg)
    else:
        completed = np.zeros(len(selected), dtype=np.bool_)
    pending = np.flatnonzero(~completed)
    start = int(pending[0]) if len(pending) else len(selected)
    if np.any(completed[start:]):
        msg = "full validation checkpoint is not contiguous"
        raise ValueError(msg)
    cpu_threads = configure_cpu_threads(directory, "training")
    extractor = DreaMSFeatureExtractor(device="cpu")
    started = time.perf_counter()
    checkpoint_start = start
    for row in range(start, len(selected), batch_size):
        stop = min(row + batch_size, len(selected))
        batch_records = selected[row:stop]
        batch = extractor.extract(
            [to_matchms_spectrum(record) for record in batch_records],
            identifiers=[record.spectrum_id for record in batch_records],
            batch_size=batch_size,
        )
        embeddings[row:stop] = batch.spectrum_embeddings
        embeddings.flush()
        if stop - checkpoint_start >= checkpoint_rows or stop == len(selected):
            completed[checkpoint_start:stop] = True
            temporary = completed_path.with_name(
                f".{completed_path.name}.{os.getpid()}.tmp.npy",
            )
            np.save(temporary, completed)
            temporary.replace(completed_path)
            checkpoint_start = stop
    if not bool(np.all(completed)):
        msg = "full validation DreaMS extraction is incomplete"
        raise RuntimeError(msg)
    identifiers_path = complete_path.parent / "identifiers.json"
    write_json(identifiers_path, {"identifiers": list(expected_ids)})
    result = {
        "schema_version": "msnlib-simplification/full-validation-dreams-v1",
        "rows": len(selected),
        "embedding_dim": DREAMS_EMBEDDING_DIM,
        "full_spectrum_peak_groups": True,
        "chemical_labels_used_for_extraction": False,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "cpu_threads": cpu_threads,
        "extractor_provenance": extractor.provenance,
        "output_sha256": {
            path.name: file_sha256(path)
            for path in (embeddings_path, completed_path, identifiers_path)
        },
    }
    write_json(complete_path, result)
    return result


def load_full_dreams_embeddings(run_dir: str | Path, split: str) -> np.ndarray:
    """Load full-spectrum embeddings for chemical-only inference."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    if split == "validation":
        prepare_full_validation_dreams_embeddings(directory)
        path, _, _ = _full_validation_paths(directory)
        return np.load(path, mmap_mode="r")
    if split != "test":
        msg = "full-spectrum DreaMS embeddings exist only for held-out splits"
        raise ValueError(
            msg,
        )
    source = Path(lock["source_run"])
    manifest = read_json(source / "chemical_inference/features/manifest.json")
    path = source / "chemical_inference/features/full_test_embeddings.npy"
    if file_sha256(path) != manifest["output_sha256"][path.name]:
        msg = "source full-test DreaMS embeddings changed"
        raise ValueError(msg)
    identifiers = tuple(
        map(
            str,
            read_json(source / "chemical_inference/features/identifiers.json")[
                "identifiers"
            ],
        ),
    )
    if identifiers != load_identifiers(directory, "test"):
        msg = "source full-test DreaMS row order changed"
        raise ValueError(msg)
    return np.load(path, mmap_mode="r")


def heldout_metadata(run_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load chemical/group labels only for evaluation and bootstrapping."""
    path = (
        Path(run_dir).expanduser().resolve()
        / "shared"
        / "counts"
        / "heldout_records.jsonl"
    )
    prepare_count_inputs(run_dir)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    if tuple(row["spectrum_id"] for row in rows) != load_identifiers(run_dir, split):
        msg = "held-out metadata row order changed"
        raise ValueError(msg)
    return rows
