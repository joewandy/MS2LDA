"""Training objectives and anti-collapse operations for neural MS2LDA."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as nnf

from .model import AssignmentOutput, NeuralMS2LDA
from .utils import read_json, write_json

if TYPE_CHECKING:
    from typing import Any

    import scipy.sparse as sp

    from .data import SparseBatch

MATRIX_DIMENSIONS = 2
PROBABILITY_FLOOR = 1e-12
MIN_GRAPH_DIMENSION = 2


def completion_metrics(
    theta: np.ndarray,
    beta: np.ndarray,
    completion: sp.csr_matrix,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score held-out token counts under the exact mixture ``theta @ beta``."""
    total_loss = 0.0
    in_vocabulary = 0
    out_of_vocabulary = 0
    eligible = 0
    for row in range(completion.shape[0]):
        start, stop = completion.indptr[row], completion.indptr[row + 1]
        words = completion.indices[start:stop]
        counts = completion.data[start:stop]
        token_count = int(counts.sum())
        out_of_vocabulary += int(records[row]["completion_oov_tokens"])
        if not token_count:
            continue
        probability = theta[row] @ beta[:, words]
        total_loss -= float(
            np.sum(counts * np.log(np.clip(probability, PROBABILITY_FLOOR, None)))
        )
        in_vocabulary += token_count
        eligible += 1
    total = in_vocabulary + out_of_vocabulary
    return {
        "nll_per_token": total_loss / in_vocabulary,
        "in_vocabulary_tokens": in_vocabulary,
        "out_of_vocabulary_tokens": out_of_vocabulary,
        "oov_fraction": out_of_vocabulary / total,
        "eligible_documents": eligible,
        "total_documents": completion.shape[0],
    }


def positive_npmi_graph(  # noqa: C901, PLR0915
    matrix: sp.csr_matrix,
    *,
    minimum_document_frequency: int,
    minimum_pair_frequency: int,
    maximum_neighbors: int,
    minimum_npmi: float,
) -> sp.csr_matrix:
    """Build the mutual-neighbour positive-NPMI graph from training documents."""
    if min(matrix.shape) < MIN_GRAPH_DIMENSION:
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
    graph_rows = []
    graph_columns = []
    graph_values = []
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
    if not graph_rows:
        raise RuntimeError("co-occurrence thresholds produced an empty graph")
    directed = sp.csr_matrix(
        (
            np.concatenate(graph_values),
            (np.concatenate(graph_rows), np.concatenate(graph_columns)),
        ),
        shape=(matrix.shape[1], matrix.shape[1]),
        dtype=np.float32,
    )
    graph = directed.minimum(directed.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz == 0:
        raise RuntimeError("mutual-neighbour pruning produced an empty graph")
    if not np.isfinite(graph.data).all():
        raise FloatingPointError("co-occurrence graph contains non-finite weights")
    return graph


def prepare_cooccurrence_graph(
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
) -> sp.csr_matrix:
    """Create or reuse the train-only NPMI graph for one run."""
    directory = Path(run_dir) / "cooccurrence_graph"
    graph_path = directory / "positive_npmi_graph.npz"
    complete_path = directory / "complete.json"
    config = protocol["cooccurrence_regularization"]
    if complete_path.is_file():
        if read_json(complete_path)["config"] != config:
            raise ValueError("co-occurrence graph configuration changed")
        return sp.load_npz(graph_path).tocsr()
    graph = positive_npmi_graph(
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
    write_json(complete_path, {"config": config})
    return graph


def torch_sparse_graph(graph: sp.csr_matrix) -> torch.Tensor:
    """Convert a SciPy graph to a coalesced CPU sparse tensor."""
    values = graph.tocoo()
    indices = torch.from_numpy(
        np.vstack((values.row, values.col)).astype(np.int64, copy=False)
    )
    weights = torch.from_numpy(values.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, weights, values.shape).coalesce()


@dataclass(frozen=True)
class RouterLossTerms:
    """Differentiable router objective and its routed full-spectrum batch."""

    total: torch.Tensor
    output: AssignmentOutput


@dataclass(frozen=True)
class TopicLossTerms:
    """Decoder/prototype objective evaluated at fixed assignments."""

    total: torch.Tensor
    beta: torch.Tensor


def cooccurrence_topic_loss(
    model: NeuralMS2LDA,
    graph: torch.Tensor,
    *,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Penalize topics that assign little mass to positive-NPMI neighbours."""
    if graph.layout != torch.sparse_coo or graph.shape != (
        model.vocabulary_size,
        model.vocabulary_size,
    ):
        raise ValueError("co-occurrence graph does not match the model vocabulary")
    propagated = torch.sparse.mm(graph, beta.T)
    affinity = torch.sum(beta.T * propagated, dim=0)
    return -torch.mean(torch.log(affinity.clamp_min(1e-12)))


def topic_separation_loss(
    model: NeuralMS2LDA,
    *,
    neighbors: int,
    margin: float,
) -> torch.Tensor:
    """Penalize nearest prototype cosine similarities above ``margin``."""
    if not 0 < neighbors < model.num_topics:
        raise ValueError("nearest-neighbour count must be between zero and K")
    if not -1.0 < margin < 1.0:
        raise ValueError("topic-separation margin must be inside (-1, 1)")
    topics = nnf.normalize(model.topic_prototypes, dim=1)
    similarities = topics @ topics.T
    diagonal = torch.eye(model.num_topics, dtype=torch.bool, device=similarities.device)
    nearest = torch.topk(
        similarities.masked_fill(diagonal, float("-inf")),
        k=int(neighbors),
        dim=1,
    ).values
    return torch.mean(torch.square(nnf.relu(nearest - float(margin))))


def balanced_sinkhorn_targets(
    logits: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Project routing scores onto equal-mass topics in log space.

    The returned matrix has unit row sums and aggregate column mass ``N / K``.
    It is used only as a detached training target; inference remains a single
    top-k softmax pass and therefore has no Sinkhorn iterations.
    """
    if logits.ndim != MATRIX_DIMENSIONS or not logits.numel():
        raise ValueError("Sinkhorn logits must be a non-empty matrix")
    observations, topics = logits.shape
    log_kernel = logits / float(epsilon)
    log_row_mass = logits.new_full((observations,), -math.log(observations))
    log_topic_mass = logits.new_full((topics,), -math.log(topics))
    log_u = torch.zeros_like(log_row_mass)
    log_v = torch.zeros_like(log_topic_mass)
    for _ in range(int(iterations)):
        log_u = log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_topic_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return plan * observations


def router_block_loss(
    model: NeuralMS2LDA,
    batch: SparseBatch,
    *,
    cached_beta: torch.Tensor,
    temperature: float,
    sinkhorn_weight: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> RouterLossTerms:
    """Optimize full-spectrum routing with likelihood and balanced usage.

    ``beta`` is detached for this block. Sinkhorn targets resist topic collapse.
    """
    output = model.route(batch, temperature=temperature, straight_through=True)
    completion = model.sparse_completion_nll(
        output.theta,
        cached_beta,
        batch,
    )
    with torch.no_grad():
        targets = balanced_sinkhorn_targets(
            output.logits.detach(),
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
        )
    log_probabilities = nnf.log_softmax(output.logits / float(temperature), dim=1)
    balance = -torch.mean(torch.sum(targets * log_probabilities, dim=1))
    total = completion + sinkhorn_weight * balance
    return RouterLossTerms(total=total, output=output)


def topic_block_loss(
    model: NeuralMS2LDA,
    batch: SparseBatch,
    *,
    temperature: float,
) -> TopicLossTerms:
    """Optimize topic geometry against fixed one-pass assignments.

    Detaching the routes makes this the decoder/prototype half of an alternating
    optimization. Full-spectrum likelihood supplies the probabilistic objective.
    """
    with torch.no_grad():
        projected = model.projected_tokens()
        output = model.route(
            batch,
            temperature=temperature,
            straight_through=False,
            projected_tokens=projected,
        )
    beta = model.topic_word_distribution()
    completion = model.sparse_completion_nll(output.theta.detach(), beta, batch)
    return TopicLossTerms(total=completion, beta=beta)
