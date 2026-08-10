"""Sparse count loading and training-only token feature construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@dataclass(frozen=True)
class SparseBatch:
    """One count-weighted sparse mini-batch."""

    indices: torch.Tensor
    offsets: torch.Tensor
    weights: torch.Tensor
    row_ids: torch.Tensor
    document_totals: torch.Tensor
    rows: NDArray[np.int64]


def load_csr(path: str | Path) -> sp.csr_matrix:
    """Load the explicit CSR representation written by the fixed benchmark."""
    with np.load(Path(path), allow_pickle=False) as archive:
        shape = tuple(map(int, archive["shape"]))
        matrix = sp.csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=shape,
        )
    matrix.sort_indices()
    return matrix


def load_vocabulary(counts_dir: str | Path) -> list[str]:
    """Load the insertion-ordered corrected vocabulary."""
    with (Path(counts_dir) / "vocabulary.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    vocabulary = list(map(str, payload["vocabulary"]))
    if len(vocabulary) != len(set(vocabulary)):
        msg = "neural vocabulary contains duplicate tokens"
        raise ValueError(msg)
    return vocabulary


def load_heldout_records(counts_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load held-out chemical records in matrix row order."""
    if split not in {"validation", "test"}:
        msg = "held-out split must be validation or test"
        raise ValueError(msg)
    records = []
    with (Path(counts_dir) / "heldout_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    records.append(row)
    return records


def iter_sparse_batches(
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[SparseBatch]:
    """Yield count-exact CSR batches without materializing dense documents."""
    order = np.arange(matrix.shape[0], dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        batch = matrix[rows].tocsr()
        lengths = np.diff(batch.indptr).astype(np.int64, copy=False)
        row_ids = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
        weights = batch.data.astype(np.float32, copy=False)
        totals = np.asarray(batch.sum(axis=1)).ravel().astype(np.float32)
        yield SparseBatch(
            indices=torch.from_numpy(batch.indices.astype(np.int64, copy=False)),
            offsets=torch.from_numpy(batch.indptr.astype(np.int64, copy=False)),
            weights=torch.from_numpy(weights),
            row_ids=torch.from_numpy(row_ids),
            document_totals=torch.from_numpy(totals),
            rows=rows,
        )


def token_masses_and_types(vocabulary: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse fragment/loss identities without any chemical labels."""
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
    vocabulary: list[str],
    feature_config: dict[str, Any],
) -> np.ndarray:
    """Combine SGNS, Fourier m/z, and token-type blocks into 64 dimensions."""
    embeddings = np.array(sgns_embeddings, dtype=np.float32, copy=True)
    if embeddings.shape[0] != len(vocabulary):
        msg = "SGNS embeddings and vocabulary do not align"
        raise ValueError(msg)
    masses, token_types = token_masses_and_types(vocabulary)
    scaled = masses / float(feature_config["mass_scale"])
    fourier_columns = []
    for frequency in feature_config["fourier_frequencies"]:
        phase = 2.0 * math.pi * float(frequency) * scaled
        fourier_columns.extend((np.sin(phase), np.cos(phase)))
    fourier = np.stack(fourier_columns, axis=1).astype(np.float32)

    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    fourier /= math.sqrt(fourier.shape[1] / 2.0)
    combined = np.concatenate((embeddings, fourier, token_types), axis=1)
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8)
    expected = int(feature_config["output_dimensions"])
    if combined.shape != (len(vocabulary), expected):
        msg = "constructed token feature dimensions differ from protocol"
        raise ValueError(msg)
    if not np.all(np.isfinite(combined)):
        msg = "constructed token features are not finite"
        raise ValueError(msg)
    return combined.astype(np.float32, copy=False)


def corpus_frequencies(matrix: sp.csr_matrix) -> np.ndarray:
    """Return training-only token counts."""
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
