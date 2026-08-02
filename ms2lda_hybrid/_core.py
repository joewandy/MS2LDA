"""PyTorch components for the hybrid document encoder and word prior."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_functional

from ._variational import EPSILON, SparseBatch, expected_log_dirichlet
from .config import HybridLDAConfig


class HybridLDACore(nn.Module):
    """Parameters for the document encoder and structured word prior."""

    def __init__(self, vocab_size: int, config: HybridLDAConfig) -> None:
        """Create tensors with the dimensions declared by ``config``."""
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.document_projector = nn.Sequential(
                nn.LayerNorm(config.embedding_dim),
                nn.Linear(config.embedding_dim, config.feature_projection_dim),
                nn.GELU(),
                nn.LayerNorm(config.feature_projection_dim),
            )
            self.word_projector = nn.Sequential(
                nn.LayerNorm(config.embedding_dim),
                nn.Linear(
                    config.embedding_dim,
                    config.feature_projection_dim,
                    bias=False,
                ),
            )
            self.word_type_embedding = nn.Embedding(3, config.feature_projection_dim)
            self.word_mz_projector = nn.Sequential(
                nn.Linear(1, config.feature_projection_dim),
                nn.GELU(),
                nn.Linear(
                    config.feature_projection_dim,
                    config.feature_projection_dim,
                ),
            )
            self.topic_embeddings = nn.Parameter(
                torch.empty(config.num_topics, config.feature_projection_dim)
            )
            nn.init.normal_(
                self.topic_embeddings,
                std=1.0 / math.sqrt(config.feature_projection_dim),
            )
            input_size = config.num_topics + config.feature_projection_dim
            self.encoder = nn.Sequential(
                nn.Linear(input_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_topics),
            )
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)

        self.register_buffer("alpha", torch.from_numpy(config.alpha_vector()))
        self.register_buffer(
            "lambda_posterior",
            torch.full(
                (config.num_topics, self.vocab_size),
                config.eta + 1.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "word_context_embeddings",
            torch.zeros(self.vocab_size, config.embedding_dim),
        )
        self.register_buffer(
            "word_context_observed",
            torch.zeros(self.vocab_size, dtype=torch.bool),
        )
        self.register_buffer("word_mz", torch.zeros(self.vocab_size, 1))
        self.register_buffer(
            "word_type",
            torch.full((self.vocab_size,), 2, dtype=torch.long),
        )

    def initialize_topics(self, total_tokens: float) -> None:
        """Initialize free topic-word factors from a seeded random simplex."""
        if not np.isfinite(total_tokens) or total_tokens <= 0:
            raise ValueError("at least one finite token is required")
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        raw = torch.empty(
            self.config.num_topics,
            self.vocab_size,
            dtype=torch.float32,
        ).exponential_(1.0, generator=generator)
        means = raw / raw.sum(dim=1, keepdim=True).clamp_min(EPSILON)
        mass = total_tokens / self.config.num_topics
        self.lambda_posterior.copy_((self.config.eta + mass * means).to(self.device))

    @property
    def device(self) -> torch.device:
        """Device holding the model buffers and parameters."""
        return self.lambda_posterior.device

    def set_word_features(
        self,
        contextual_embeddings: np.ndarray,
        observed: np.ndarray,
        mz_values: np.ndarray,
        word_types: np.ndarray,
    ) -> None:
        """Copy aligned contextual, mass, and type features into model buffers."""
        expected = (self.vocab_size, self.config.embedding_dim)
        if contextual_embeddings.shape != expected:
            raise ValueError(f"word embeddings must have shape {expected}")
        if observed.shape != (self.vocab_size,):
            raise ValueError("observed must contain one flag per word")
        if mz_values.shape != (self.vocab_size,):
            raise ValueError("mz_values must contain one value per word")
        if word_types.shape != (self.vocab_size,):
            raise ValueError("word_types must contain one value per word")
        self.word_context_embeddings.copy_(
            torch.as_tensor(contextual_embeddings, device=self.device)
        )
        self.word_context_observed.copy_(torch.as_tensor(observed, device=self.device))
        self.word_mz.copy_(
            torch.as_tensor(mz_values, device=self.device).reshape(-1, 1)
        )
        self.word_type.copy_(torch.as_tensor(word_types, device=self.device))

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Parameters trained to amortize the local VB posterior."""
        return [*self.encoder.parameters(), *self.document_projector.parameters()]

    def prior_parameters(self) -> list[nn.Parameter]:
        """Parameters trained to construct the bounded topic-word prior."""
        return [
            *self.word_projector.parameters(),
            *self.word_type_embedding.parameters(),
            *self.word_mz_projector.parameters(),
            self.topic_embeddings,
        ]

    def beta_mean(self) -> torch.Tensor:
        """Return posterior-mean topic-word probabilities, shape ``K x V``."""
        return self.lambda_posterior / self.lambda_posterior.sum(
            dim=1, keepdim=True
        ).clamp_min(EPSILON)

    def _word_topic_evidence(
        self,
        batch: SparseBatch,
        word_topic: torch.Tensor,
    ) -> torch.Tensor:
        """Average current topic evidence over each document's observed words."""
        counts = batch.word_counts * batch.word_mask
        evidence = (counts.unsqueeze(-1) * word_topic[batch.word_ids]).sum(dim=1)
        evidence = evidence / batch.totals.clamp_min(1.0)
        empty = batch.totals <= 0
        if torch.any(empty):
            evidence = torch.where(
                empty,
                torch.full_like(evidence, 1.0 / self.config.num_topics),
                evidence,
            )
        return evidence / evidence.sum(dim=1, keepdim=True).clamp_min(EPSILON)

    def encode(
        self,
        batch: SparseBatch,
        document_embeddings: torch.Tensor,
        word_topic: torch.Tensor,
    ) -> torch.Tensor:
        """Predict initial ``gamma`` from topic evidence and a DreaMS embedding."""
        expected = (batch.word_ids.shape[0], self.config.embedding_dim)
        if tuple(document_embeddings.shape) != expected:
            raise ValueError(f"document embeddings must have shape {expected}")
        evidence = self._word_topic_evidence(batch, word_topic)
        projected = self.document_projector(document_embeddings)
        residual = self.encoder(torch.cat([evidence, projected], dim=1))
        topic_mean = torch.softmax(
            evidence.clamp_min(EPSILON).log() + residual,
            dim=1,
        )
        return self.alpha.unsqueeze(0) + batch.totals * topic_mean

    def _projected_words(self) -> torch.Tensor:
        """Combine contextual peak, normalized mass, and token-type features."""
        context = self.word_projector(self.word_context_embeddings)
        context = context * self.word_context_observed.unsqueeze(-1)
        mz = self.word_mz_projector(self.word_mz)
        token_type = self.word_type_embedding(self.word_type)
        return nn_functional.normalize(context + mz + token_type, dim=1)

    def structured_prior(self, total_tokens: float, epoch: int) -> torch.Tensor:
        """Return ``eta + r[e] rho N/K p[k,v]`` for every topic and word."""
        baseline = torch.full_like(self.lambda_posterior, self.config.eta)
        topics = nn_functional.normalize(self.topic_embeddings, dim=1)
        logits = topics @ self._projected_words().transpose(0, 1)
        distribution = torch.softmax(logits / self.config.prior_temperature, dim=1)
        warmup = min(max(float(epoch), 0.0) / self.config.prior_warmup_epochs, 1.0)
        topic_mass = total_tokens / self.config.num_topics
        structured_mass = warmup * self.config.prior_mass_fraction * topic_mass
        return baseline + structured_mass * distribution

    def prior_loss(self, total_tokens: float, epoch: int) -> torch.Tensor:
        """Empirical-Bayes loss for the structured prior parameters."""
        prior = self.structured_prior(total_tokens, epoch)
        posterior = self.lambda_posterior.detach()
        expected_log_beta = expected_log_dirichlet(posterior)
        expected_log_prior = (
            torch.lgamma(prior.sum(dim=1))
            - torch.lgamma(prior).sum(dim=1)
            + ((prior - 1.0) * expected_log_beta).sum(dim=1)
        ).mean() / self.vocab_size
        topics = nn_functional.normalize(self.topic_embeddings, dim=1)
        gram = topics @ topics.transpose(0, 1)
        identity = torch.eye(
            self.config.num_topics,
            device=self.device,
            dtype=gram.dtype,
        )
        orthogonality = ((gram - identity) ** 2).mean()
        return -expected_log_prior + self.config.topic_diversity_weight * orthogonality

    @torch.no_grad()
    def update_topics(self, statistics: torch.Tensor, prior: torch.Tensor) -> None:
        """Apply the global VB update ``lambda = prior + expected counts``."""
        if (
            statistics.shape != self.lambda_posterior.shape
            or not bool(torch.all(torch.isfinite(statistics)))
            or bool(torch.any(statistics < 0))
        ):
            raise ValueError("invalid expected topic-word counts")
        if (
            prior.shape != self.lambda_posterior.shape
            or not bool(torch.all(torch.isfinite(prior)))
            or bool(torch.any(prior <= 0))
        ):
            raise ValueError("invalid Dirichlet prior")
        self.lambda_posterior.copy_(prior + statistics)
