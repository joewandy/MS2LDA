"""Chemistry-free sparse inputs and physical peak-group training views."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

from .inputs import (
    config_from_protocol,
    resolve_input_paths,
)
from .spectra import (
    PeakGroup,
    assign_scaffold_splits,
    audit_split_disjointness,
    build_training_vocabulary,
    completion_document,
    filtered_words,
    load_records,
    renormalize_peak_groups,
    split_records,
)
from .utils import (
    file_sha256,
    read_json,
    verify_output_hashes,
    write_json,
    write_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True)
class SparseBatch:
    """One count-exact sparse mini-batch with local row identifiers."""

    indices: torch.Tensor
    weights: torch.Tensor
    row_ids: torch.Tensor
    document_totals: torch.Tensor
    documents: int


@dataclass(frozen=True)
class ViewPair:
    """Two deterministic 80-percent physical-peak views."""

    left: sp.csr_matrix
    right: sp.csr_matrix
    pair_index: int


def load_csr(path: str | Path) -> sp.csr_matrix:
    """Load one SciPy CSR archive produced by this workflow."""
    matrix = sp.load_npz(Path(path)).tocsr().astype(np.float32)
    matrix.sort_indices()
    return matrix


def load_vocabulary(counts_dir: str | Path) -> list[str]:
    """Load the frozen corrected vocabulary in model column order."""
    payload = read_json(Path(counts_dir) / "vocabulary.json")
    vocabulary = list(map(str, payload["vocabulary"]))
    if len(vocabulary) != len(set(vocabulary)):
        msg = "neural-assignment vocabulary contains duplicates"
        raise ValueError(msg)
    return vocabulary


def load_heldout_records(counts_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load held-out evaluation metadata in fixed matrix row order."""
    if split not in {"validation", "test"}:
        msg = "held-out split must be validation or test"
        raise ValueError(msg)
    rows = []
    path = Path(counts_dir) / f"{split}_records.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") != split:
                    raise ValueError(f"unexpected split in {path.name}")
                rows.append(row)
    return rows


def sparse_batch(matrix: sp.csr_matrix, rows: NDArray[np.int64]) -> SparseBatch:
    """Materialize one CSR slice as tensors without densifying vocabulary counts."""
    batch = matrix[rows].tocsr()
    lengths = np.diff(batch.indptr).astype(np.int64, copy=False)
    row_ids = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    totals = np.asarray(batch.sum(axis=1)).ravel().astype(np.float32)
    return SparseBatch(
        indices=torch.from_numpy(batch.indices.astype(np.int64, copy=False)),
        weights=torch.from_numpy(batch.data.astype(np.float32, copy=False)),
        row_ids=torch.from_numpy(row_ids),
        document_totals=torch.from_numpy(totals),
        documents=len(rows),
    )


def iter_row_batches(
    documents: int,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[NDArray[np.int64]]:
    """Yield a deterministic shared row order for aligned matrices."""
    order = np.arange(documents, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, documents, batch_size):
        yield order[start : start + batch_size]


def iter_sparse_batches(
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
) -> Iterator[SparseBatch]:
    """Yield count-exact inference batches in source order."""
    for rows in iter_row_batches(
        matrix.shape[0],
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        yield sparse_batch(matrix, rows)


def token_masses_and_types(vocabulary: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse fragment/loss identities without using chemical labels."""
    masses = np.empty(len(vocabulary), dtype=np.float32)
    types = np.zeros((len(vocabulary), 2), dtype=np.float32)
    for index, token in enumerate(vocabulary):
        prefix, separator, value = token.partition("@")
        if not separator or prefix not in {"frag", "loss"}:
            msg = f"unsupported MS2LDA token: {token}"
            raise ValueError(msg)
        masses[index] = float(value)
        types[index, 0 if prefix == "frag" else 1] = 1.0
    return masses, types


def build_token_features(
    sgns_embeddings: np.ndarray,
    vocabulary: Sequence[str],
    feature_config: dict[str, Any],
) -> np.ndarray:
    """Combine train-only SGNS, Fourier mass, and fragment/loss type features."""
    embeddings = np.asarray(sgns_embeddings, dtype=np.float32).copy()
    if embeddings.shape[0] != len(vocabulary):
        msg = "SGNS embeddings and vocabulary do not align"
        raise ValueError(msg)
    masses, token_types = token_masses_and_types(vocabulary)
    scaled = masses / float(feature_config["mass_scale"])
    columns = []
    for frequency in feature_config["fourier_frequencies"]:
        phase = 2.0 * math.pi * float(frequency) * scaled
        columns.extend((np.sin(phase), np.cos(phase)))
    fourier = np.stack(columns, axis=1).astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    fourier /= math.sqrt(fourier.shape[1] / 2.0)
    combined = np.concatenate((embeddings, fourier, token_types), axis=1)
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8)
    if not np.all(np.isfinite(combined)):
        msg = "constructed token features are not finite"
        raise ValueError(msg)
    return combined.astype(np.float32, copy=False)


def corpus_frequencies(matrix: sp.csr_matrix) -> np.ndarray:
    """Return training-only token frequencies."""
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)


def prototype_seeding_weights(matrix: sp.csr_matrix) -> np.ndarray:
    """Return sqrt-corpus-frequency times squared-IDF seeding weights."""
    frequencies = corpus_frequencies(matrix)
    document_frequency = np.diff(matrix.tocsc().indptr).astype(np.float64)
    inverse_document_frequency = np.log(
        (1.0 + matrix.shape[0]) / (1.0 + document_frequency),
    )
    weights = np.sqrt(frequencies) * np.square(inverse_document_frequency)
    if not np.any(weights > 0):
        return np.ones(matrix.shape[1], dtype=np.float64)
    return weights


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
    result = sp.csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float32,
    )
    result.sort_indices()
    return result


def _atomic_save_npz(path: Path, matrix: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        sp.save_npz(temporary, matrix, compressed=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data(  # noqa: PLR0915
    run_dir: str | Path, *, data_root: str | Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Build the split, first-seen vocabulary, and sparse matrices from raw MGF."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "data"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        return result
    config = config_from_protocol(protocol)
    mgf = resolve_input_paths(protocol, data_root)["mgf"]
    records, parsing = load_records(mgf, config)
    prep = protocol["preprocessing"]
    assignments, split_summary = assign_scaffold_splits(
        records,
        fractions=tuple(map(float, prep["split_fractions"])),
        seed=int(prep["split_seed"]),
    )
    leakage = audit_split_disjointness(records, assignments)
    vocabulary, vocabulary_summary = build_training_vocabulary(
        records,
        assignments,
        min_df=int(prep["min_df"]),
        min_cf=int(prep["min_cf"]),
        rm_top=int(prep["rm_top"]),
    )
    vocabulary_set = set(vocabulary)
    output.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, sp.csr_matrix] = {}
    identifiers: dict[str, list[str]] = {}
    heldout_rows: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    split_rows: list[dict[str, Any]] = []
    for record in records:
        split = assignments[record.spectrum_id]
        split_rows.append(
            {
                "spectrum_id": record.spectrum_id,
                "split": split,
                "connectivity_key": record.connectivity_key,
                "scaffold_key": record.scaffold_key,
                "split_group": record.split_group,
            }
        )
    for split in ("train", "validation", "test"):
        selected = split_records(records, assignments, split)
        identifiers[split] = [record.spectrum_id for record in selected]
        if split == "train":
            matrices["train"] = _matrix(
                [filtered_words(record, vocabulary_set) for record in selected],
                vocabulary,
            )
            continue
        full_documents: list[list[str]] = []
        observed_documents: list[list[str]] = []
        completion_documents: list[list[str]] = []
        for record in selected:
            completion = completion_document(
                record,
                observed_fraction=float(prep["completion_observed_fraction"]),
                seed=int(prep["completion_seed"]),
            )
            observed_groups = renormalize_peak_groups(
                completion.observed_groups,
                precursor_mz=record.precursor_mz,
                significant_digits=config.significant_digits,
            )
            observed_words = [
                token for group in observed_groups for token in group.tokens
            ]
            full_words = filtered_words(record, vocabulary_set)
            completion_words = [
                token
                for token in completion.completion_words
                if token in vocabulary_set
            ]
            observed_filtered = [
                token for token in observed_words if token in vocabulary_set
            ]
            full_documents.append(full_words)
            observed_documents.append(observed_filtered)
            completion_documents.append(completion_words)
            heldout_rows[split].append(
                {
                    "spectrum_id": record.spectrum_id,
                    "split": split,
                    "smiles": record.smiles,
                    "connectivity_key": record.connectivity_key,
                    "scaffold_key": record.scaffold_key,
                    "observed_oov_tokens": len(observed_words) - len(observed_filtered),
                    "completion_oov_tokens": (
                        len(completion.completion_words) - len(completion_words)
                    ),
                }
            )
        matrices[f"{split}_full"] = _matrix(full_documents, vocabulary)
        matrices[f"{split}_observed"] = _matrix(observed_documents, vocabulary)
        matrices[f"{split}_completion"] = _matrix(completion_documents, vocabulary)
    outputs: list[Path] = []
    for name, matrix in matrices.items():
        path = output / f"{name}.npz"
        _atomic_save_npz(path, matrix)
        outputs.append(path)
    vocabulary_path = output / "vocabulary.json"
    write_json(
        vocabulary_path,
        {
            "vocabulary": list(vocabulary),
            **vocabulary_summary,
        },
    )
    identifiers_path = output / "identifiers.json"
    write_json(identifiers_path, identifiers)
    split_path = output / "split_manifest.jsonl"
    heldout_paths = []
    for split, rows in heldout_rows.items():
        path = output / f"{split}_records.jsonl"
        write_jsonl(path, rows)
        heldout_paths.append(path)
    write_jsonl(split_path, split_rows)
    outputs.extend((vocabulary_path, identifiers_path, *heldout_paths, split_path))
    result = {
        "raw_mgf": {"path": str(mgf), "sha256": file_sha256(mgf)},
        "parsing": parsing,
        "split": split_summary,
        "leakage_audit": leakage,
        "vocabulary": vocabulary_summary,
        "matrix_shapes": {
            name: list(matrix.shape) for name, matrix in matrices.items()
        },
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(complete_path, result)
    return result


def _stable_rank(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_view_peak_groups(  # noqa: PLR0913
    peak_groups: Sequence[PeakGroup],
    *,
    spectrum_id: str,
    seed: int,
    pair_index: int,
    side: str,
    retained_fraction: float,
) -> tuple[PeakGroup, ...]:
    """Select whole physical peak groups deterministically for one view."""
    if side not in {"left", "right"}:
        msg = "view side must be left or right"
        raise ValueError(msg)
    if not peak_groups:
        msg = "cannot mask a spectrum without peak groups"
        raise ValueError(msg)
    ranked = sorted(
        peak_groups,
        key=lambda group: _stable_rank(
            seed,
            pair_index,
            side,
            spectrum_id,
            group.original_index,
        ),
    )
    retained = max(1, round(len(ranked) * retained_fraction))
    if len(ranked) > 1:
        retained = min(retained, len(ranked) - 1)
    selected = {group.original_index for group in ranked[:retained]}
    return tuple(group for group in peak_groups if group.original_index in selected)


def _view_words(  # noqa: PLR0913
    record: Any,
    *,
    seed: int,
    pair_index: int,
    side: str,
    retained_fraction: float,
    significant_digits: int,
) -> list[str]:
    groups = select_view_peak_groups(
        record.peak_groups,
        spectrum_id=record.spectrum_id,
        seed=seed,
        pair_index=pair_index,
        side=side,
        retained_fraction=retained_fraction,
    )
    normalized = renormalize_peak_groups(
        groups,
        precursor_mz=record.precursor_mz,
        significant_digits=significant_digits,
    )
    return [token for group in normalized for token in group.tokens]


def prepare_training_views(
    run_dir: str | Path,
    *,
    counts_dir: str | Path,
    data_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild four chemistry-free paired views from frozen physical peaks."""
    directory = Path(run_dir).expanduser().resolve()
    counts = Path(counts_dir).expanduser().resolve()
    output = directory / "training_views"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        return result

    config = config_from_protocol(protocol)
    mgf = resolve_input_paths(protocol, data_root)["mgf"]
    if file_sha256(mgf) != str(protocol["input_files"]["mgf"]["sha256"]):
        raise ValueError("frozen raw MGF changed before view construction")
    records, parsing = load_records(mgf, config)
    by_id = {record.spectrum_id: record for record in records}
    identifiers = read_json(counts / "identifiers.json")["train"]
    selected = [by_id[str(identifier)] for identifier in identifiers]
    vocabulary = load_vocabulary(counts)
    frozen_train = load_csr(counts / "train.npz")
    rebuilt_train = _matrix([record.words for record in selected], vocabulary)
    if rebuilt_train.shape != frozen_train.shape or (rebuilt_train != frozen_train).nnz:
        msg = "raw peak groups do not reproduce the frozen train counts"
        raise ValueError(msg)

    output.mkdir(parents=True, exist_ok=True)
    view_config = protocol["views"]
    outputs: list[Path] = []
    pair_summaries = []
    for pair_index in range(int(view_config["pairs"])):
        summary: dict[str, Any] = {"pair_index": pair_index}
        for side in ("left", "right"):
            documents = [
                _view_words(
                    record,
                    seed=int(protocol["seed"]),
                    pair_index=pair_index,
                    side=side,
                    retained_fraction=float(
                        view_config["retained_peak_group_fraction"],
                    ),
                    significant_digits=config.significant_digits,
                )
                for record in selected
            ]
            matrix = _matrix(documents, vocabulary)
            path = output / f"pair_{pair_index}_{side}.npz"
            _atomic_save_npz(path, matrix)
            outputs.append(path)
            summary[side] = {
                "shape": list(matrix.shape),
                "nnz": int(matrix.nnz),
                "token_mass": float(matrix.sum()),
                "empty_documents": int(np.sum(np.diff(matrix.indptr) == 0)),
            }
        pair_summaries.append(summary)
    result = {
        "raw_mgf": {
            "path": str(mgf),
            "sha256": file_sha256(mgf),
        },
        "train_identifiers_sha256": file_sha256(counts / "identifiers.json"),
        "frozen_train_counts_reproduced_exactly": True,
        "retained_peak_group_fraction": float(
            view_config["retained_peak_group_fraction"],
        ),
        "pairs": pair_summaries,
        "raw_parser_summary": {
            "parsed_blocks": int(parsing["parsed_blocks"]),
            "retained_spectra": int(parsing["retained_spectra"]),
        },
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(complete_path, result)
    return result


def load_view_pairs(run_dir: str | Path, protocol: dict[str, Any]) -> list[ViewPair]:
    """Load all verified paired training views."""
    output = Path(run_dir).expanduser().resolve() / "training_views"
    complete = read_json(output / "complete.json")
    verify_output_hashes(output, complete)
    pairs = []
    for pair_index in range(int(protocol["views"]["pairs"])):
        paths = {
            side: output / f"pair_{pair_index}_{side}.npz" for side in ("left", "right")
        }
        pairs.append(
            ViewPair(
                left=load_csr(paths["left"]),
                right=load_csr(paths["right"]),
                pair_index=pair_index,
            ),
        )
    return pairs
