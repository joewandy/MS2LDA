"""ERNTM topic diversity constraint used by the supported model."""

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
