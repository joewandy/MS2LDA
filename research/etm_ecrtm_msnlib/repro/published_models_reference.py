"""Compact reference implementations used to reason about ETM/ECRTM.

This file is intentionally independent of the production M1 code.  It captures
only the recognizable published-model pieces used in the synthetic study and in
`scripts/run_published_topic_models_msnlib.py`.

References
----------
ETM: Dieng, Ruiz & Blei, TACL 2020; original authors' implementation
`adjidieng/ETM`, commit cbb67bf484282e66df00cd2166bf8dc740a95a1d.

ECRTM: Wu et al., ICML 2023; maintained implementation in `bobxwu/TopMost`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-12


class FixedEmbeddingETM(nn.Module):
    """Original ETM form with fixed pretrained word embeddings.

    `normalized_bows` are normalized only for the amortized encoder.  The
    reconstruction term should still be evaluated against raw count BOWs.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        num_topics: int,
        hidden_size: int = 800,
    ) -> None:
        super().__init__()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        vocab_size, embedding_size = embeddings.shape
        self.register_buffer("rho", torch.from_numpy(embeddings))
        self.alphas = nn.Linear(embedding_size, num_topics, bias=False)
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_q_theta = nn.Linear(hidden_size, num_topics)
        self.logsigma_q_theta = nn.Linear(hidden_size, num_topics)

    def get_beta(self) -> torch.Tensor:
        # V x K logits, then softmax over V for every topic.
        return F.softmax(self.alphas(self.rho), dim=0).T

    def encode(self, normalized_bows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.q_theta(normalized_bows)
        mu = self.mu_q_theta(hidden)
        logvar = self.logsigma_q_theta(hidden)
        kl = -0.5 * torch.sum(
            1.0 + logvar - mu.square() - logvar.exp(),
            dim=1,
        )
        return mu, logvar, kl

    def get_theta(
        self,
        normalized_bows: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar, kl = self.encode(normalized_bows)
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return F.softmax(z, dim=1), kl

    def loss(self, bows: torch.Tensor, normalized_bows: torch.Tensor) -> torch.Tensor:
        theta, kl = self.get_theta(normalized_bows, sample=self.training)
        beta = self.get_beta()
        probability = (theta @ beta).clamp_min(EPS)
        reconstruction = -(bows * probability.log()).sum(dim=1)
        return (reconstruction + kl).mean()


class EmbeddingClusteringRegularization(nn.Module):
    """ECR optimal-transport regularizer from ECRTM.

    The published standalone implementation allows up to 1000 Sinkhorn
    iterations with a 0.005 stopping tolerance.  The synthetic multi-seed screen
    used `max_iter=50` as an explicitly labelled numerical approximation after a
    spot check showed nearly identical scientific metrics.
    """

    def __init__(
        self,
        weight: float = 100.0,
        sinkhorn_alpha: float = 20.0,
        max_iter: int = 1000,
        stop_threshold: float = 0.005,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.sinkhorn_alpha = float(sinkhorn_alpha)
        self.max_iter = int(max_iter)
        self.stop_threshold = float(stop_threshold)

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        topics, words = cost.shape
        a = torch.ones((topics, 1), device=cost.device, dtype=cost.dtype) / topics
        b = torch.ones((words, 1), device=cost.device, dtype=cost.dtype) / words
        u = torch.ones_like(a) / topics
        kernel = torch.exp(-cost * self.sinkhorn_alpha)
        v = torch.ones_like(b)
        for iteration in range(self.max_iter):
            v = b / (kernel.T @ u + 1e-16)
            u = a / (kernel @ v + 1e-16)
            if iteration % 50 == 0:
                recovered_b = v * (kernel.T @ u)
                error = torch.max(torch.sum(torch.abs(recovered_b - b), dim=0))
                if float(error.detach()) <= self.stop_threshold:
                    break
        transport = u * (kernel * v.T)
        return self.weight * torch.sum(transport * cost)


class TopMostStyleECRTM(nn.Module):
    """ECRTM equations following the maintained TopMost implementation."""

    def __init__(
        self,
        pretrained_embeddings: np.ndarray,
        num_topics: int,
        encoder_units: int = 200,
        beta_temperature: float = 0.2,
        ecr_weight: float = 100.0,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_max_iter: int = 1000,
        dirichlet_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        embeddings = F.normalize(
            torch.as_tensor(pretrained_embeddings, dtype=torch.float32),
            dim=1,
        )
        vocab_size, embedding_size = embeddings.shape
        self.num_topics = int(num_topics)
        self.beta_temperature = float(beta_temperature)

        self.fc11 = nn.Linear(vocab_size, encoder_units)
        self.fc12 = nn.Linear(encoder_units, encoder_units)
        self.fc21 = nn.Linear(encoder_units, num_topics)
        self.fc22 = nn.Linear(encoder_units, num_topics)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        # Pretrained SGNS is an initialization, not a frozen ECRTM embedding.
        self.word_embeddings = nn.Parameter(embeddings.clone())
        topic_embeddings = torch.empty((num_topics, embedding_size))
        nn.init.trunc_normal_(topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(topic_embeddings, dim=1))

        concentration = np.full((1, num_topics), float(dirichlet_alpha), dtype=np.float32)
        mu2 = (np.log(concentration).T - np.mean(np.log(concentration), axis=1)).T
        var2 = (
            ((1.0 / concentration) * (1.0 - 2.0 / num_topics)).T
            + (1.0 / num_topics**2) * np.sum(1.0 / concentration, axis=1)
        ).T
        self.register_buffer("mu2", torch.from_numpy(mu2))
        self.register_buffer("var2", torch.from_numpy(var2))

        self.ecr = EmbeddingClusteringRegularization(
            weight=ecr_weight,
            sinkhorn_alpha=sinkhorn_alpha,
            max_iter=sinkhorn_max_iter,
        )

    @staticmethod
    def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(x.square(), dim=1, keepdim=True)
            + torch.sum(y.square(), dim=1)
            - 2.0 * (x @ y.T)
        )

    def get_beta(self) -> torch.Tensor:
        cost = self.pairwise_squared_distance(
            self.topic_embeddings,
            self.word_embeddings,
        )
        # This is the published ECRTM topic/word weight matrix.  Its columns are
        # normalized across topics; topic-word ranking is read from each row.
        return F.softmax(-cost / self.beta_temperature, dim=0)

    def encode(self, bows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.softplus(self.fc11(bows))
        hidden = F.softplus(self.fc12(hidden))
        mu = self.mean_bn(self.fc21(hidden))
        logvar = self.logvar_bn(self.fc22(hidden))
        return mu, logvar

    def get_theta(
        self,
        bows: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(bows)
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        theta = F.softmax(z, dim=1)

        var = logvar.exp()
        kl = 0.5 * (
            (
                var / self.var2
                + (mu - self.mu2).square() / self.var2
                + self.var2.log()
                - logvar
            ).sum(dim=1)
            - self.num_topics
        )
        return theta, kl

    def decoder_probability(self, theta: torch.Tensor) -> torch.Tensor:
        beta = self.get_beta()
        return F.softmax(self.decoder_bn(theta @ beta), dim=-1)

    def ecr_loss(self) -> torch.Tensor:
        cost = self.pairwise_squared_distance(
            self.topic_embeddings,
            self.word_embeddings,
        )
        return self.ecr(cost)

    def loss(self, bows: torch.Tensor) -> torch.Tensor:
        theta, kl = self.get_theta(bows, sample=self.training)
        probability = self.decoder_probability(theta).clamp_min(EPS)
        reconstruction = -(bows * probability.log()).sum(dim=1)
        return (reconstruction + kl).mean() + self.ecr_loss()


def sharpen_theta(theta: np.ndarray, temperature: float = 0.30) -> np.ndarray:
    """Frozen synthetic-derived ECRTM theta calibration candidate."""
    values = np.asarray(theta, dtype=np.float64)
    logits = np.log(np.clip(values, EPS, None)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    return values.astype(np.float32)
