# ruff: noqa: N812, PLR0913, PLR2004
"""Collapse-resistant fully neural MS2LDA model and exact objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .data import SparseBatch


@dataclass(frozen=True)
class LossTerms:
    """Differentiable training terms and detached diagnostics."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    ecr: torch.Tensor
    usage_guard: torch.Tensor
    sparsity_guard: torch.Tensor
    theta_entropy: torch.Tensor


@dataclass(frozen=True)
class EncoderLossTerms:
    """Encoder-only exact reconstruction, KL, and collapse guards."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    usage_guard: torch.Tensor
    sparsity_guard: torch.Tensor
    theta_entropy: torch.Tensor


def balanced_sinkhorn_plan(
    cost: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Compute a differentiable balanced OT plan in the log domain."""
    if cost.ndim != 2 or not cost.numel():
        msg = "Sinkhorn cost must be a non-empty matrix"
        raise ValueError(msg)
    words, topics = cost.shape
    log_kernel = -cost / epsilon
    log_word_mass = cost.new_full((words,), -math.log(words))
    log_topic_mass = cost.new_full((topics,), -math.log(topics))
    log_u = torch.zeros_like(log_word_mass)
    log_v = torch.zeros_like(log_topic_mass)
    for _ in range(iterations):
        log_u = log_word_mass - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0),
            dim=1,
        )
        log_v = log_topic_mass - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1),
            dim=0,
        )
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


class NeuralMS2LDA(nn.Module):
    """Direct amortized topic model with ECR-style topic separation."""

    def __init__(
        self,
        token_features: torch.Tensor,
        *,
        num_topics: int,
        hidden_dimensions: int,
        topic_word_temperature: float,
        dropout: float,
        topic_initial_indices: torch.Tensor,
    ) -> None:
        super().__init__()
        if token_features.ndim != 2:
            msg = "token features must be a matrix"
            raise ValueError(msg)
        vocabulary_size, dimensions = token_features.shape
        if len(topic_initial_indices) != num_topics:
            msg = "topic initialization does not match topic count"
            raise ValueError(msg)
        self.num_topics = int(num_topics)
        self.vocabulary_size = int(vocabulary_size)
        self.dimensions = int(dimensions)
        self.topic_word_temperature = float(topic_word_temperature)
        self.register_buffer("token_features", token_features.clone())
        self.token_projection = nn.Linear(dimensions, dimensions, bias=False)
        nn.init.eye_(self.token_projection.weight)
        initial_topics = token_features[topic_initial_indices].clone()
        self.topic_embeddings = nn.Parameter(initial_topics)
        self.encoder_hidden = nn.Linear(dimensions, hidden_dimensions)
        self.encoder_norm = nn.LayerNorm(hidden_dimensions)
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_mean = nn.Linear(hidden_dimensions, num_topics)
        self.encoder_log_variance = nn.Linear(hidden_dimensions, num_topics)
        nn.init.zeros_(self.encoder_mean.bias)
        nn.init.constant_(self.encoder_log_variance.bias, -1.0)

    def projected_tokens(self) -> torch.Tensor:
        """Return normalized trainable projections of fixed token features."""
        return F.normalize(self.token_projection(self.token_features), dim=1)

    def topic_word_distribution(
        self,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return row-normalized neural topic/token distributions."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        topics = F.normalize(self.topic_embeddings, dim=1)
        logits = 2.0 * topics @ tokens.T / self.topic_word_temperature
        return F.softmax(logits, dim=1)

    def encode(
        self,
        batch: SparseBatch,
        *,
        sample: bool,
        projected_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer a logistic-normal topic mixture in one encoder pass."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        bag = F.embedding_bag(
            batch.indices,
            tokens,
            batch.offsets,
            mode="sum",
            per_sample_weights=batch.weights,
            include_last_offset=True,
        )
        bag = bag / batch.document_totals.clamp_min(1.0).unsqueeze(1)
        hidden = F.softplus(self.encoder_hidden(bag))
        hidden = self.encoder_dropout(self.encoder_norm(hidden))
        mean = self.encoder_mean(hidden)
        log_variance = self.encoder_log_variance(hidden).clamp(-8.0, 6.0)
        latent = (
            mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            if sample
            else mean
        )
        return F.softmax(latent, dim=1), mean, log_variance

    def ecr_loss(
        self,
        token_indices: torch.Tensor,
        *,
        epsilon: float,
        iterations: int,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply balanced sampled embedding-clustering optimal transport."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        words = tokens[token_indices]
        topics = F.normalize(self.topic_embeddings, dim=1)
        cost = (2.0 - 2.0 * words @ topics.T).clamp_min(0.0)
        # The optimal-transport envelope gradient with respect to cost is the
        # transport plan itself. Detaching the finite Sinkhorn solve preserves
        # that gradient while avoiding a 20-iteration autograd graph.
        with torch.no_grad():
            plan = balanced_sinkhorn_plan(
                cost.detach(),
                epsilon=epsilon,
                iterations=iterations,
            )
        return torch.sum(plan * cost)

    @staticmethod
    def sparse_reconstruction_loss(
        theta: torch.Tensor,
        beta: torch.Tensor,
        batch: SparseBatch,
    ) -> torch.Tensor:
        """Score exact theta-times-beta probabilities only where counts exist."""
        selected_topics = theta[batch.row_ids]
        selected_words = beta[:, batch.indices].T
        observed = torch.sum(selected_topics * selected_words, dim=1).clamp_min(1e-12)
        return -torch.sum(batch.weights * torch.log(observed)) / torch.sum(
            batch.weights,
        ).clamp_min(1.0)

    def encoder_loss(
        self,
        batch: SparseBatch,
        *,
        beta: torch.Tensor,
        projected_tokens: torch.Tensor,
        kl_weight: float,
        usage_guard_weight: float = 0.0,
        sparsity_guard_weight: float = 0.0,
        target_effective_topics: float = 9.0,
    ) -> EncoderLossTerms:
        """Optimize the encoder against a fixed exact topic distribution."""
        theta, mean, log_variance = self.encode(
            batch,
            sample=True,
            projected_tokens=projected_tokens,
        )
        reconstruction = self.sparse_reconstruction_loss(theta, beta, batch)
        kl_per_document = 0.5 * torch.sum(
            torch.exp(log_variance) + mean.square() - 1.0 - log_variance,
            dim=1,
        )
        kl = torch.mean(kl_per_document / batch.document_totals.clamp_min(1.0))
        usage = theta.mean(dim=0).clamp_min(1e-12)
        usage_guard = -torch.mean(torch.log(usage * self.num_topics))
        entropy = -torch.sum(theta * torch.log(theta.clamp_min(1e-12)), dim=1)
        target_entropy = math.log(target_effective_topics)
        sparsity_guard = torch.mean(
            ((entropy - target_entropy) / max(target_entropy, 1e-6)).square(),
        )
        total = (
            reconstruction
            + kl_weight * kl
            + usage_guard_weight * usage_guard
            + sparsity_guard_weight * sparsity_guard
        )
        return EncoderLossTerms(
            total=total,
            reconstruction=reconstruction,
            kl=kl,
            usage_guard=usage_guard,
            sparsity_guard=sparsity_guard,
            theta_entropy=entropy.mean(),
        )

    def decoder_loss(
        self,
        batch: SparseBatch,
        *,
        theta: torch.Tensor,
        ecr_token_indices: torch.Tensor,
        ecr_weight: float,
        ecr_epsilon: float,
        ecr_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize topic/token geometry with one exact full-vocabulary normalizer."""
        projected = self.projected_tokens()
        beta = self.topic_word_distribution(projected)
        reconstruction = self.sparse_reconstruction_loss(theta, beta, batch)
        ecr = self.ecr_loss(
            ecr_token_indices,
            epsilon=ecr_epsilon,
            iterations=ecr_iterations,
            projected_tokens=projected,
        )
        return reconstruction + ecr_weight * ecr, reconstruction, ecr

    def loss(
        self,
        batch: SparseBatch,
        ecr_token_indices: torch.Tensor,
        *,
        kl_weight: float,
        ecr_weight: float,
        ecr_epsilon: float,
        ecr_iterations: int,
        usage_guard_weight: float = 0.0,
        sparsity_guard_weight: float = 0.0,
        target_effective_topics: float = 9.0,
    ) -> LossTerms:
        """Compute exact theta-times-beta multinomial reconstruction and guards."""
        projected = self.projected_tokens()
        beta = self.topic_word_distribution(projected)
        theta, mean, log_variance = self.encode(
            batch,
            sample=True,
            projected_tokens=projected,
        )
        reconstruction = self.sparse_reconstruction_loss(theta, beta, batch)
        kl_per_document = 0.5 * torch.sum(
            torch.exp(log_variance) + mean.square() - 1.0 - log_variance,
            dim=1,
        )
        kl = torch.mean(kl_per_document / batch.document_totals.clamp_min(1.0))
        ecr = self.ecr_loss(
            ecr_token_indices,
            epsilon=ecr_epsilon,
            iterations=ecr_iterations,
            projected_tokens=projected,
        )
        usage = theta.mean(dim=0).clamp_min(1e-12)
        usage_guard = -torch.mean(torch.log(usage * self.num_topics))
        entropy = -torch.sum(theta * torch.log(theta.clamp_min(1e-12)), dim=1)
        target_entropy = math.log(target_effective_topics)
        sparsity_guard = torch.mean(
            ((entropy - target_entropy) / max(target_entropy, 1e-6)).square(),
        )
        total = (
            reconstruction
            + kl_weight * kl
            + ecr_weight * ecr
            + usage_guard_weight * usage_guard
            + sparsity_guard_weight * sparsity_guard
        )
        return LossTerms(
            total=total,
            reconstruction=reconstruction,
            kl=kl,
            ecr=ecr,
            usage_guard=usage_guard,
            sparsity_guard=sparsity_guard,
            theta_entropy=entropy.mean(),
        )


def initialize_model(
    token_features: torch.Tensor,
    protocol: dict[str, Any],
) -> tuple[NeuralMS2LDA, torch.Tensor]:
    """Create the single data-only initialization shared by both attempts."""
    seed = int(protocol["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed + 4049)
    permutation = torch.randperm(len(token_features), generator=generator)
    topic_indices = permutation[: int(protocol["num_topics"])]
    torch.manual_seed(seed)
    model_config = protocol["model"]
    model = NeuralMS2LDA(
        token_features,
        num_topics=int(protocol["num_topics"]),
        hidden_dimensions=int(model_config["hidden_dimensions"]),
        topic_word_temperature=float(model_config["topic_word_temperature"]),
        dropout=float(model_config["dropout"]),
        topic_initial_indices=topic_indices,
    )
    with torch.no_grad():
        model.topic_embeddings.add_(
            0.001
            * torch.randn(
                model.topic_embeddings.shape,
                generator=generator,
                dtype=model.topic_embeddings.dtype,
            ),
        )
    return model, topic_indices
