"""Topic geometry and train-only co-occurrence constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import NeuralAssignmentMS2LDA


@dataclass(frozen=True)
class DiversityRegularizerResult:
    """A scalar regularizer and diagnostics that are safe to serialize."""

    loss: torch.Tensor
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class CooccurrenceRegularizerResult:
    """A graph-coherence scalar and serializable diagnostics."""

    loss: torch.Tensor
    diagnostics: dict[str, float]


def erntm_topic_constraint(
    model: NeuralAssignmentMS2LDA,
) -> DiversityRegularizerResult:
    """Apply the ERNTM identity-target cross-entropy topic constraint.

    Shao et al. (2022) define the constraint as cross-entropy between the
    topic-topic similarity matrix and the identity matrix.  The prototypes are
    normalized here, matching the geometry already used by this model.
    """
    topics = F.normalize(model.topic_prototypes, dim=1)
    similarities = topics @ topics.T
    targets = torch.arange(model.num_topics, device=similarities.device)
    loss = F.cross_entropy(similarities, targets)
    with torch.no_grad():
        off_diagonal = similarities[
            ~torch.eye(
                model.num_topics,
                dtype=torch.bool,
                device=similarities.device,
            )
        ]
        diagnostics = {
            "mean_off_diagonal_cosine": float(off_diagonal.mean()),
            "maximum_off_diagonal_cosine": float(off_diagonal.max()),
        }
    return DiversityRegularizerResult(loss=loss, diagnostics=diagnostics)


def cooccurrence_topic_constraint(
    model: NeuralAssignmentMS2LDA,
    graph: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
) -> CooccurrenceRegularizerResult:
    """Maximize within-topic probability assigned to positive-NPMI graph edges."""
    if graph.layout != torch.sparse_coo or graph.shape != (
        model.vocabulary_size,
        model.vocabulary_size,
    ):
        raise ValueError("co-occurrence graph does not match the model vocabulary")
    probabilities = model.topic_word_distribution() if beta is None else beta
    propagated = torch.sparse.mm(graph, probabilities.T)
    affinity = torch.sum(probabilities.T * propagated, dim=0)
    loss = -torch.mean(torch.log(affinity.clamp_min(1e-12)))
    with torch.no_grad():
        diagnostics = {
            "cooccurrence_affinity_mean": float(affinity.mean()),
            "cooccurrence_affinity_median": float(affinity.median()),
            "cooccurrence_affinity_minimum": float(affinity.min()),
            "cooccurrence_affinity_maximum": float(affinity.max()),
        }
    return CooccurrenceRegularizerResult(loss=loss, diagnostics=diagnostics)


def nearest_neighbor_topic_constraint(
    model: NeuralAssignmentMS2LDA,
    *,
    neighbors: int,
    margin: float,
) -> DiversityRegularizerResult:
    """Penalize each topic's nearest prototype neighbours above a cosine margin."""
    if not 0 < neighbors < model.num_topics:
        raise ValueError("nearest-neighbour count must be between zero and K")
    if not -1.0 < margin < 1.0:
        raise ValueError("topic-separation margin must be inside (-1, 1)")
    topics = F.normalize(model.topic_prototypes, dim=1)
    similarities = topics @ topics.T
    diagonal = torch.eye(model.num_topics, dtype=torch.bool, device=similarities.device)
    off_diagonal = similarities.masked_fill(diagonal, float("-inf"))
    nearest = torch.topk(off_diagonal, k=int(neighbors), dim=1).values
    violations = F.relu(nearest - float(margin))
    loss = torch.mean(torch.square(violations))
    with torch.no_grad():
        closest = nearest[:, 0]
        diagnostics = {
            "nearest_topic_margin": float(margin),
            "nearest_topic_cosine_mean": float(closest.mean()),
            "nearest_topic_cosine_median": float(closest.median()),
            "nearest_topic_cosine_p95": float(torch.quantile(closest, 0.95)),
            "nearest_topic_cosine_maximum": float(closest.max()),
            "hard_neighbor_fraction_above_margin": float(
                torch.mean((nearest > float(margin)).to(dtype=torch.float32))
            ),
        }
    return DiversityRegularizerResult(loss=loss, diagnostics=diagnostics)
