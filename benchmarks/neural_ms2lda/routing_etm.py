"""Routing-informed variational inference for the published ETM generator."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from .sparse_etm import BalancedSparseETM, ThetaTransform

EPS = 1e-12
RoutingVariant = Literal[
    "etm",
    "soft_token",
    "soft_context",
    "top2_context",
]
ROUTING_VARIANTS: tuple[RoutingVariant, ...] = (
    "etm",
    "soft_token",
    "soft_context",
    "top2_context",
)


class RoutingInformedETM(BalancedSparseETM):
    """Balanced fixed-SGNS ETM with a bounded inference-only adaptation.

    The ETM topic-word decoder, multinomial likelihood, Gaussian variational
    latent, and standard-normal KL are unchanged. For the three routing variants,
    observed tokens are scored against the same topic embeddings that define
    ``beta``. Their aggregated evidence is added to the Gaussian posterior mean
    before the selected published simplex mapping.

    ``soft_token`` uses each fixed SGNS token vector directly. ``soft_context``
    adds one learned scalar times the leave-one-out spectrum mean.
    ``top2_context`` makes the same contextual assignments exactly top-2 before
    document aggregation. No M1 gate, Sinkhorn target, NPMI loss, prototype
    separation, or alternating optimizer is present.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        routing_variant: RoutingVariant = "etm",
        theta_transform: ThetaTransform = "softmax",
        routing_temperature: float = 1.0,
        hidden: int = 800,
    ) -> None:
        if routing_variant not in ROUTING_VARIANTS:
            raise ValueError(f"unknown routing variant: {routing_variant}")
        if routing_variant == "top2_context" and int(topics) < 2:
            raise ValueError("top-2 routing requires at least two topics")
        if float(routing_temperature) <= 0:
            raise ValueError("routing temperature must be positive")
        super().__init__(
            embeddings,
            topics,
            fragment_mask,
            theta_transform=theta_transform,
            hidden=hidden,
        )
        self.routing_variant = routing_variant
        self.routing_temperature = float(routing_temperature)
        if routing_variant in {"soft_context", "top2_context"}:
            self.context_scale = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("context_scale", None)

    def routing_evidence(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """Return one shared-geometry token-evidence distribution per row."""
        if self.routing_variant == "etm":
            raise RuntimeError("the ETM control has no routing evidence")
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
        routes = tokens
        if self.context_scale is not None:
            document_sums = normalized_bows @ self.rho
            denominator = (1.0 - weights).clamp_min(EPS)
            context = (
                document_sums[row_ids] - weights.unsqueeze(1) * tokens
            ) / denominator.unsqueeze(1)
            routes = nnf.normalize(
                tokens + self.context_scale * context,
                dim=1,
            )

        topic_vectors = nnf.normalize(self.alphas.weight, dim=1)
        logits = routes @ topic_vectors.T
        if self.routing_variant == "top2_context":
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
        else:
            assignments = nnf.softmax(
                logits / self.routing_temperature,
                dim=1,
            )
            evidence = normalized_bows.new_zeros((documents, topics))
            evidence.index_add_(0, row_ids, weights.unsqueeze(1) * assignments)

        totals = evidence.sum(dim=1, keepdim=True)
        normalized = evidence / totals.clamp_min(EPS)
        empty = totals.squeeze(1) <= 0
        if torch.any(empty):
            normalized = normalized.clone()
            normalized[empty] = 1.0 / topics
        return normalized

    def encode(
        self,
        normalized_bows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the ordinary ETM posterior, optionally informed by routing."""
        hidden = self.encoder(normalized_bows)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        if self.routing_variant != "etm":
            evidence = self.routing_evidence(normalized_bows)
            # Adding a uniform pseudocount of 1/K bounds the offset for topics
            # outside the sparse route. Row centering removes an irrelevant
            # softmax constant and makes uniform evidence an exact no-op.
            offset = torch.log(evidence + 1.0 / evidence.shape[1])
            mu = mu + offset - offset.mean(dim=1, keepdim=True)
        kl = -0.5 * torch.sum(
            1.0 + logvar - mu.square() - logvar.exp(),
            dim=1,
        )
        return mu, logvar, kl
