# ruff: noqa: PLR0913
"""Train-only word co-occurrence graph construction for neural topic learning."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

from .utils import file_sha256, read_json, write_json

if TYPE_CHECKING:
    from numpy.typing import NDArray


def positive_npmi_graph(
    matrix: sp.csr_matrix,
    *,
    minimum_document_frequency: int,
    minimum_pair_frequency: int,
    maximum_neighbors: int,
    minimum_npmi: float,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Build a deterministic symmetric graph of strong train-document pairs."""
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("co-occurrence graph needs at least two documents and words")
    if minimum_document_frequency < 1 or minimum_pair_frequency < 1:
        raise ValueError("co-occurrence frequency thresholds must be positive")
    if maximum_neighbors < 1:
        raise ValueError("maximum_neighbors must be positive")

    binary = matrix.tocsr().astype(np.float32, copy=True)
    binary.data.fill(1.0)
    document_frequency = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
    pair_counts = (binary.T @ binary).tocsr()
    pair_counts.setdiag(0)
    pair_counts.eliminate_zeros()
    documents = float(matrix.shape[0])

    graph_rows: list[NDArray[np.int64]] = []
    graph_columns: list[NDArray[np.int64]] = []
    graph_values: list[NDArray[np.float32]] = []
    directed_edges = 0
    for row in range(pair_counts.shape[0]):
        if document_frequency[row] < minimum_document_frequency:
            continue
        start, stop = pair_counts.indptr[row], pair_counts.indptr[row + 1]
        columns = pair_counts.indices[start:stop].astype(np.int64, copy=False)
        counts = pair_counts.data[start:stop].astype(np.float64, copy=False)
        eligible = (counts >= minimum_pair_frequency) & (
            document_frequency[columns] >= minimum_document_frequency
        )
        if not np.any(eligible):
            continue
        columns = columns[eligible]
        counts = counts[eligible]
        joint = counts / documents
        independent = (
            document_frequency[row] * document_frequency[columns] / documents**2
        )
        scores = np.empty_like(joint)
        certain = joint >= 1.0
        scores[certain] = 1.0
        uncertain = ~certain
        scores[uncertain] = np.log(joint[uncertain] / independent[uncertain]) / -np.log(
            joint[uncertain]
        )
        finite = np.isfinite(scores) & (scores > float(minimum_npmi))
        if not np.any(finite):
            continue
        columns = columns[finite]
        scores = scores[finite]
        order = np.lexsort((columns, -scores))[:maximum_neighbors]
        columns = columns[order]
        scores = scores[order]
        graph_rows.append(np.full(len(columns), row, dtype=np.int64))
        graph_columns.append(columns)
        graph_values.append(scores.astype(np.float32, copy=False))
        directed_edges += len(columns)

    if not graph_rows:
        raise RuntimeError("co-occurrence thresholds produced an empty graph")
    rows = np.concatenate(graph_rows)
    columns = np.concatenate(graph_columns)
    values = np.concatenate(graph_values)
    directed = sp.csr_matrix(
        (values, (rows, columns)),
        shape=(matrix.shape[1], matrix.shape[1]),
        dtype=np.float32,
    )
    # Mutual nearest-neighbour edges suppress high-degree corpus hubs: both
    # words must rank the pair among their strongest train-only associations.
    graph = directed.minimum(directed.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.diff(graph.indptr)
    diagnostics = {
        "documents": int(matrix.shape[0]),
        "vocabulary_size": int(matrix.shape[1]),
        "candidate_pair_entries": int(pair_counts.nnz),
        "directed_pruned_edges": int(directed_edges),
        "symmetric_directed_edges": int(graph.nnz),
        "undirected_edges": int(graph.nnz // 2),
        "words_with_neighbors": int(np.count_nonzero(degree)),
        "mean_degree_nonempty": float(degree[degree > 0].mean()),
        "maximum_degree": int(degree.max()),
        "mean_edge_npmi": float(graph.data.mean()),
        "minimum_edge_npmi": float(graph.data.min()),
        "maximum_edge_npmi": float(graph.data.max()),
    }
    if not math.isfinite(diagnostics["mean_edge_npmi"]):
        raise FloatingPointError("co-occurrence graph contains non-finite weights")
    return graph, diagnostics


def prepare_cooccurrence_graph(
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Create or verify the frozen train-only graph for one model run."""
    directory = Path(run_dir) / "cooccurrence_graph"
    graph_path = directory / "positive_npmi_graph.npz"
    complete_path = directory / "complete.json"
    config = protocol["cooccurrence_regularization"]
    if complete_path.is_file():
        result = read_json(complete_path)
        if result["config"] != config:
            raise ValueError("co-occurrence graph configuration changed")
        if file_sha256(graph_path) != result["graph_sha256"]:
            raise ValueError("co-occurrence graph changed")
        return sp.load_npz(graph_path).tocsr(), result

    graph, diagnostics = positive_npmi_graph(
        train,
        minimum_document_frequency=int(config["minimum_document_frequency"]),
        minimum_pair_frequency=int(config["minimum_pair_frequency"]),
        maximum_neighbors=int(config["maximum_neighbors"]),
        minimum_npmi=float(config["minimum_npmi"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".positive_npmi_graph.tmp.npz"
    sp.save_npz(temporary, graph, compressed=True)
    os.replace(temporary, graph_path)
    result = {
        "schema_version": "neural-ms2lda/cooccurrence-graph-v1",
        "training_split_only": True,
        "config": config,
        "diagnostics": diagnostics,
        "graph_sha256": file_sha256(graph_path),
    }
    write_json(complete_path, result)
    return graph, result


def torch_sparse_graph(graph: sp.csr_matrix) -> torch.Tensor:
    """Convert a SciPy graph to a coalesced CPU sparse tensor."""
    values = graph.tocoo()
    indices = torch.from_numpy(
        np.vstack((values.row, values.col)).astype(np.int64, copy=False)
    )
    weights = torch.from_numpy(values.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, weights, values.shape).coalesce()
