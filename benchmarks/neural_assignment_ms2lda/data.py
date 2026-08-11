# ruff: noqa: PLR0913
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

from benchmarks.msnlib_validation.config import load_config
from benchmarks.msnlib_validation.data import (
    PeakGroup,
    load_records,
    renormalize_peak_groups,
)

from .utils import file_sha256, read_json, write_json

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True)
class SparseBatch:
    """One count-exact sparse mini-batch with local row identifiers."""

    indices: torch.Tensor
    offsets: torch.Tensor
    weights: torch.Tensor
    row_ids: torch.Tensor
    document_totals: torch.Tensor
    rows: NDArray[np.int64]

    @property
    def documents(self) -> int:
        """Return the number of documents, including empty ones."""
        return len(self.rows)


@dataclass(frozen=True)
class ViewPair:
    """Two deterministic 80-percent physical-peak views."""

    left: sp.csr_matrix
    right: sp.csr_matrix
    pair_index: int


def load_csr(path: str | Path) -> sp.csr_matrix:
    """Load either the benchmark CSR archive or a SciPy sparse archive."""
    source = Path(path)
    try:
        matrix = sp.load_npz(source).tocsr().astype(np.float32)
    except ValueError:
        with np.load(source, allow_pickle=False) as archive:
            shape = tuple(map(int, archive["shape"]))
            matrix = sp.csr_matrix(
                (archive["data"], archive["indices"], archive["indptr"]),
                shape=shape,
                dtype=np.float32,
            )
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
    with (Path(counts_dir) / "heldout_records.jsonl").open(
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
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
        offsets=torch.from_numpy(batch.indptr.astype(np.int64, copy=False)),
        weights=torch.from_numpy(batch.data.astype(np.float32, copy=False)),
        row_ids=torch.from_numpy(row_ids),
        document_totals=torch.from_numpy(totals),
        rows=rows,
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
    shuffle: bool,
    seed: int,
) -> Iterator[SparseBatch]:
    """Yield count-exact sparse batches."""
    for rows in iter_row_batches(
        matrix.shape[0],
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
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
    expected = int(feature_config["output_dimensions"])
    if combined.shape != (len(vocabulary), expected):
        msg = "constructed token features differ from the protocol"
        raise ValueError(msg)
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


def _stable_rank(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_view_peak_groups(
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


def _view_words(
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
    reference_run: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild four chemistry-free paired views from frozen physical peaks."""
    directory = Path(run_dir).expanduser().resolve()
    counts = Path(counts_dir).expanduser().resolve()
    reference = Path(reference_run).expanduser().resolve()
    output = directory / "training_views"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"training view changed: {name}"
                raise ValueError(msg)
        return result

    config = load_config(reference / "config.resolved.json")
    source_lock = read_json(reference / "protocol.lock.json")
    mgf = (
        Path(source_lock["data_root"]) / str(config.input_files["mgf"]["relative_path"])
    ).resolve()
    if file_sha256(mgf) != str(config.input_files["mgf"]["sha256"]):
        msg = "frozen raw MGF changed before view construction"
        raise ValueError(msg)
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
        "schema_version": "neural-assignment-ms2lda/training-views-v1",
        "raw_mgf": {
            "path": str(mgf),
            "sha256": file_sha256(mgf),
        },
        "train_identifiers_sha256": file_sha256(counts / "identifiers.json"),
        "frozen_train_counts_reproduced_exactly": True,
        "physical_peak_groups_atomic": True,
        "retained_peak_group_fraction": float(
            view_config["retained_peak_group_fraction"],
        ),
        "pairs": pair_summaries,
        "chemistry_fields_in_model_inputs": [],
        "model_inputs": [
            "token identities",
            "token counts",
            "fragment/loss type",
            "m/z",
        ],
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
    pairs = []
    for pair_index in range(int(protocol["views"]["pairs"])):
        paths = {
            side: output / f"pair_{pair_index}_{side}.npz" for side in ("left", "right")
        }
        for path in paths.values():
            if complete["output_sha256"][path.name] != file_sha256(path):
                msg = f"training view changed: {path.name}"
                raise ValueError(msg)
        pairs.append(
            ViewPair(
                left=load_csr(paths["left"]),
                right=load_csr(paths["right"]),
                pair_index=pair_index,
            ),
        )
    return pairs
