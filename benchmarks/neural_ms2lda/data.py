"""Chemistry-free sparse inputs and fixed train-only token features."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as nnf

from .spectra import (
    assign_scaffold_splits,
    audit_split_disjointness,
    build_training_vocabulary,
    completion_document,
    filtered_words,
    input_paths,
    load_records,
    preprocessing_config,
    renormalize_peak_groups,
    split_records,
)
from .utils import (
    atomic_save_numpy,
    read_json,
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


def token_types(vocabulary: Sequence[str]) -> np.ndarray:
    """Return fragment/loss indicators without using chemical labels."""
    types = np.zeros((len(vocabulary), 2), dtype=np.float32)
    for index, token in enumerate(vocabulary):
        prefix, separator, value = token.partition("@")
        if not separator or prefix not in {"frag", "loss"}:
            msg = f"unsupported MS2LDA token: {token}"
            raise ValueError(msg)
        float(value)
        types[index, 0 if prefix == "frag" else 1] = 1.0
    return types


def build_token_features(
    sgns_embeddings: np.ndarray,
    vocabulary: Sequence[str],
) -> np.ndarray:
    """Combine train-only SGNS with fragment/loss type indicators."""
    embeddings = np.asarray(sgns_embeddings, dtype=np.float32).copy()
    if embeddings.shape[0] != len(vocabulary):
        msg = "SGNS embeddings and vocabulary do not align"
        raise ValueError(msg)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    combined = np.concatenate((embeddings, token_types(vocabulary)), axis=1)
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8)
    if not np.all(np.isfinite(combined)):
        msg = "constructed token features are not finite"
        raise ValueError(msg)
    return combined.astype(np.float32, copy=False)


def corpus_frequencies(matrix: sp.csr_matrix) -> np.ndarray:
    """Return training-only token frequencies."""
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)


MAX_PAIR_RESAMPLES = 8


class _Sgns(nn.Module):
    """Two embedding tables trained by skip-gram negative sampling."""

    def __init__(self, vocabulary_size: int, dimensions: int) -> None:
        super().__init__()
        self.source = nn.Embedding(vocabulary_size, dimensions, sparse=True)
        self.context = nn.Embedding(vocabulary_size, dimensions, sparse=True)
        nn.init.uniform_(self.source.weight, -0.5 / dimensions, 0.5 / dimensions)
        nn.init.zeros_(self.context.weight)

    def forward(
        self,
        source: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        source_values = self.source(source)
        positive = torch.sum(source_values * self.context(context), dim=1)
        negative = torch.einsum("bd,bnd->bn", source_values, self.context(negatives))
        return -(nnf.logsigmoid(positive) + nnf.logsigmoid(-negative).sum(dim=1)).mean()


def _positive_pairs(
    matrix: sp.csr_matrix,
    *,
    pairs_per_document: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic, count-weighted within-spectrum token pairs."""
    rng = np.random.default_rng(seed)
    total = matrix.shape[0] * pairs_per_document
    sources = np.empty(total, dtype=np.int64)
    contexts = np.empty(total, dtype=np.int64)
    cursor = 0
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        words = matrix.indices[start:stop]
        counts = matrix.data[start:stop].astype(np.float64, copy=False)
        if not len(words):
            continue
        probability = counts / counts.sum()
        left = rng.choice(words, size=pairs_per_document, p=probability)
        right = rng.choice(words, size=pairs_per_document, p=probability)
        if len(words) > 1:
            same = left == right
            attempts = 0
            while np.any(same) and attempts < MAX_PAIR_RESAMPLES:
                right[same] = rng.choice(words, size=int(same.sum()), p=probability)
                same = left == right
                attempts += 1
            if np.any(same):
                positions = {int(word): index for index, word in enumerate(words)}
                right[same] = [
                    words[(positions[int(word)] + 1) % len(words)]
                    for word in left[same]
                ]
        count = len(left)
        sources[cursor : cursor + count] = left
        contexts[cursor : cursor + count] = right
        cursor += count
    return sources[:cursor], contexts[:cursor]


def train_token_features(
    output_dir: str | Path,
    matrix: sp.csr_matrix,
    vocabulary: list[str],
    protocol: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Train SGNS, append mass/type features, and write the fixed token table."""
    output = Path(output_dir)
    complete_path = output / "complete.json"
    features_path = output / "features.npy"
    if complete_path.is_file() and features_path.is_file():
        return read_json(complete_path)
    output.mkdir(parents=True, exist_ok=True)
    config = protocol["sgns"]
    torch.manual_seed(seed)
    model = _Sgns(matrix.shape[1], int(config["dimensions"]))
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    frequencies = corpus_frequencies(matrix)
    negative_probability = np.power(frequencies, float(config["negative_power"]))
    negative_probability /= negative_probability.sum()
    for epoch in range(int(config["epochs"])):
        sources, contexts = _positive_pairs(
            matrix,
            pairs_per_document=int(config["positive_pairs_per_document"]),
            seed=seed + 1009 * epoch,
        )
        rng = np.random.default_rng(seed + 2027 * epoch)
        order = rng.permutation(len(sources))
        for begin in range(0, len(order), int(config["batch_size"])):
            selected = order[begin : begin + int(config["batch_size"])]
            negatives = rng.choice(
                matrix.shape[1],
                size=(len(selected), int(config["negative_samples"])),
                p=negative_probability,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                torch.from_numpy(sources[selected]),
                torch.from_numpy(contexts[selected]),
                torch.from_numpy(negatives),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("SGNS produced a non-finite loss")
            loss.backward()
            optimizer.step()
    embeddings = 0.5 * (model.source.weight.detach() + model.context.weight.detach())
    embeddings = nnf.normalize(embeddings, dim=1).cpu().numpy().astype(np.float32)
    features = build_token_features(embeddings, vocabulary)
    atomic_save_numpy(features_path, features)
    result = {
        "feature_dimensions": int(features.shape[1]),
    }
    write_json(complete_path, result)
    return result


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
        return read_json(complete_path)
    config = preprocessing_config(protocol)
    mgf = input_paths(protocol, data_root)["mgf"]
    records, parsing = load_records(mgf, config)
    prep = protocol["preprocessing"]
    assignments, split_summary = assign_scaffold_splits(
        records,
        fractions=tuple(map(float, prep["split_fractions"])),
        seed=int(protocol["seed"]),
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
    heldout_rows: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    for split in ("train", "validation", "test"):
        selected = split_records(records, assignments, split)
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
            observed_groups, completion_groups = completion_document(
                record,
                observed_fraction=float(prep["completion_observed_fraction"]),
                seed=int(protocol["seed"]),
            )
            observed_groups = renormalize_peak_groups(
                observed_groups,
                precursor_mz=record.precursor_mz,
                significant_digits=config.significant_digits,
            )
            observed_words = [
                token for group in observed_groups for token in group.tokens
            ]
            full_words = filtered_words(record, vocabulary_set)
            raw_completion_words = [
                token for group in completion_groups for token in group.tokens
            ]
            completion_words = [
                token for token in raw_completion_words if token in vocabulary_set
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
                        len(raw_completion_words) - len(completion_words)
                    ),
                }
            )
        matrices[f"{split}_full"] = _matrix(full_documents, vocabulary)
        matrices[f"{split}_observed"] = _matrix(observed_documents, vocabulary)
        matrices[f"{split}_completion"] = _matrix(completion_documents, vocabulary)
    for name, matrix in matrices.items():
        path = output / f"{name}.npz"
        _atomic_save_npz(path, matrix)
    vocabulary_path = output / "vocabulary.json"
    write_json(
        vocabulary_path,
        {
            "vocabulary": list(vocabulary),
            **vocabulary_summary,
        },
    )
    for split, rows in heldout_rows.items():
        path = output / f"{split}_records.jsonl"
        write_jsonl(path, rows)
    result = {
        "parsing": parsing,
        "split": split_summary,
        "leakage_audit": leakage,
        "vocabulary": vocabulary_summary,
    }
    write_json(complete_path, result)
    return result
