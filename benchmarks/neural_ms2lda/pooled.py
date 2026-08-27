"""Pooled projected Neural MS2LDA research candidate.

This module adapts the frozen reference implementation in
``research/etm_ecrtm_msnlib/pooled_projected`` to the benchmark's sparse-batch
contract.  It is intentionally separate from the locked M1 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from .data import SparseBatch, iter_sparse_batches

if TYPE_CHECKING:
    import scipy.sparse as sp


@dataclass(frozen=True)
class PooledOutput:
    """One-pass document mixture and shared topic-word decoder."""

    theta: torch.Tensor
    beta: torch.Tensor
    document_logits: torch.Tensor


def assignment_information_loss(theta: torch.Tensor) -> torch.Tensor:
    """Return ``E[H(theta_d)] - H(E[theta_d])``."""
    probabilities = theta.clamp_min(1e-12)
    conditional = -torch.sum(probabilities * torch.log(probabilities), dim=1).mean()
    marginal = probabilities.mean(dim=0)
    marginal_entropy = -torch.sum(marginal * torch.log(marginal.clamp_min(1e-12)))
    return conditional - marginal_entropy


def batch_to_device(batch: SparseBatch, device: torch.device) -> SparseBatch:
    """Move one immutable sparse batch to the training device."""
    return SparseBatch(
        indices=batch.indices.to(device),
        weights=batch.weights.to(device),
        row_ids=batch.row_ids.to(device),
        document_totals=batch.document_totals.to(device),
        documents=batch.documents,
    )


class PooledProjectedMS2LDA(nn.Module):
    """Minimal pooled encoder with a fragment/loss-balanced decoder."""

    def __init__(
        self,
        token_features: torch.Tensor,
        *,
        num_topics: int,
        projection_dimensions: int,
        theta_temperature: float,
        beta_temperature: float,
        topic_initial_indices: torch.Tensor,
        seed: int,
    ) -> None:
        super().__init__()
        if token_features.ndim != 2:
            raise ValueError("token features must be a matrix")
        if len(topic_initial_indices) != int(num_topics):
            raise ValueError("topic initialization does not match topic count")
        self.num_topics = int(num_topics)
        self.vocabulary_size = int(token_features.shape[0])
        self.input_dimensions = int(token_features.shape[1])
        self.projection_dimensions = int(projection_dimensions)
        self.theta_temperature = float(theta_temperature)
        self.beta_temperature = float(beta_temperature)
        if self.theta_temperature <= 0 or self.beta_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.register_buffer("token_features", token_features.detach().clone())
        type_features = token_features[:, -2:]
        fragment_mask = type_features[:, 0] > type_features[:, 1]
        if not torch.any(fragment_mask) or torch.all(fragment_mask):
            raise ValueError("token features must contain fragments and losses")
        self.register_buffer("fragment_mask", fragment_mask, persistent=False)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.token_projection = nn.Linear(
                self.input_dimensions,
                self.projection_dimensions,
                bias=False,
            )
            nn.init.orthogonal_(self.token_projection.weight)
        with torch.no_grad():
            initial = self.projected_tokens()[topic_initial_indices].clone()
        self.topic_prototypes = nn.Parameter(initial)

    def projected_tokens(self) -> torch.Tensor:
        """Return normalized projected token positions."""
        return nnf.normalize(self.token_projection(self.token_features), dim=1)

    def topic_word_distribution(
        self,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the frozen candidate's 50/50 fragment/loss decoder."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        topics = nnf.normalize(self.topic_prototypes, dim=1)
        logits = 2.0 * topics @ tokens.T / self.beta_temperature
        probabilities = torch.empty_like(logits)
        probabilities[:, self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, self.fragment_mask], dim=1
        )
        probabilities[:, ~self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, ~self.fragment_mask], dim=1
        )
        return probabilities

    def infer_batch(
        self,
        batch: SparseBatch,
        *,
        projected_tokens: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> PooledOutput:
        """Infer ``theta`` from a count-weighted pooled spectrum."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        weighted = tokens[batch.indices] * batch.weights.unsqueeze(1)
        document_sums = weighted.new_zeros(
            (batch.documents, self.projection_dimensions)
        )
        document_sums.index_add_(0, batch.row_ids, weighted)
        document_vectors = nnf.normalize(document_sums, dim=1)
        topics = nnf.normalize(self.topic_prototypes, dim=1)
        document_logits = 2.0 * document_vectors @ topics.T / self.theta_temperature
        theta = nnf.softmax(document_logits, dim=1)
        decoder = self.topic_word_distribution(tokens) if beta is None else beta
        return PooledOutput(
            theta=theta,
            beta=decoder,
            document_logits=document_logits,
        )

    @staticmethod
    def sparse_completion_nll(
        theta: torch.Tensor,
        beta: torch.Tensor,
        target: SparseBatch,
    ) -> torch.Tensor:
        """Score nonzero counts under the exact topic mixture."""
        selected_topics = theta[target.row_ids]
        selected_words = beta[:, target.indices].T
        probability = torch.sum(selected_topics * selected_words, dim=1).clamp_min(
            1e-12
        )
        total = target.weights.sum().clamp_min(1.0)
        return -torch.sum(target.weights * torch.log(probability)) / total


@torch.inference_mode()
def infer_pooled_theta(
    model: PooledProjectedMS2LDA,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Infer all document mixtures without local optimization."""
    model.eval()
    projected = model.projected_tokens()
    beta = model.topic_word_distribution(projected)
    rows = []
    for batch in iter_sparse_batches(matrix, batch_size=int(batch_size)):
        moved = batch_to_device(batch, device)
        rows.append(
            model.infer_batch(
                moved,
                projected_tokens=projected,
                beta=beta,
            )
            .theta.cpu()
            .numpy()
            .astype(np.float32)
        )
    return np.concatenate(rows, axis=0)


def initialize_pooled_candidate(
    token_features: torch.Tensor,
    *,
    num_topics: int,
    protocol: dict[str, Any],
) -> tuple[PooledProjectedMS2LDA, torch.Tensor]:
    """Construct the reference model's deterministic token initialization."""
    seed = int(protocol["seed"])
    config = protocol["simple_candidate"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(token_features), generator=generator)[
        : int(num_topics)
    ]
    model = PooledProjectedMS2LDA(
        token_features,
        num_topics=int(num_topics),
        projection_dimensions=int(config["projection_dimensions"]),
        theta_temperature=float(config["theta_temperature"]),
        beta_temperature=float(config["beta_temperature"]),
        topic_initial_indices=indices,
        seed=seed + int(num_topics),
    )
    return model, indices
