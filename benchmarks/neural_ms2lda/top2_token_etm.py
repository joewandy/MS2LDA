"""Zero-parameter top-2 token evidence for the Routing ETM ablation."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as nnf

from .routing_etm import EPS, RoutingInformedETM
from .sparse_etm import ThetaTransform

TOP2_ROUTING_VARIANT = "top2_token"


class Top2TokenETM(RoutingInformedETM):
    """ETM posterior informed by direct, parameter-free top-2 token votes."""

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        theta_transform: ThetaTransform = "softmax",
        routing_temperature: float = 1.0,
        hidden: int = 800,
    ) -> None:
        if int(topics) < 2:
            raise ValueError("top-2 routing requires at least two topics")
        super().__init__(
            embeddings,
            topics,
            fragment_mask,
            routing_variant="soft_token",
            theta_transform=theta_transform,
            routing_temperature=routing_temperature,
            hidden=hidden,
        )
        self.routing_variant = TOP2_ROUTING_VARIANT  # type: ignore[assignment]

    def routing_evidence(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """Return direct token evidence restricted to two topics per word."""
        if normalized_bows.ndim != 2 or normalized_bows.shape[1] != self.rho.shape[0]:
            raise ValueError("normalized BOW rows must match the ETM vocabulary")
        if not torch.all(torch.isfinite(normalized_bows)) or torch.any(
            normalized_bows < 0,
        ):
            raise ValueError("normalized BOW values must be finite and non-negative")

        row_ids, word_ids = torch.nonzero(normalized_bows > 0, as_tuple=True)
        documents = normalized_bows.shape[0]
        topics = self.alphas.weight.shape[0]
        if not len(row_ids):
            return normalized_bows.new_full((documents, topics), 1.0 / topics)

        weights = normalized_bows[row_ids, word_ids]
        tokens = nnf.normalize(self.rho[word_ids], dim=1)
        topic_vectors = nnf.normalize(self.alphas.weight, dim=1)
        logits = tokens @ topic_vectors.T
        selected_logits, selected_topics = torch.topk(logits, k=2, dim=1)
        selected_mass = nnf.softmax(
            selected_logits / self.routing_temperature,
            dim=1,
        )
        flat_evidence = normalized_bows.new_zeros(documents * topics)
        flat_indices = (row_ids.unsqueeze(1) * topics + selected_topics).reshape(-1)
        flat_evidence.index_add_(
            0,
            flat_indices,
            (weights.unsqueeze(1) * selected_mass).reshape(-1),
        )
        evidence = flat_evidence.reshape(documents, topics)
        totals = evidence.sum(dim=1, keepdim=True)
        normalized = evidence / totals.clamp_min(EPS)
        empty = totals.squeeze(1) <= 0
        if torch.any(empty):
            normalized = normalized.clone()
            normalized[empty] = 1.0 / topics
        return normalized
