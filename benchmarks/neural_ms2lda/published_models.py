"""Small published-family prototypes, isolated from production MS2LDA.

ProdLDA follows https://pyro.ai/examples/prodlda.html in plain PyTorch. The
Dirichlet encoder follows Burkhardt and Kramer (JMLR 2019, 20:131), using the
authors' implicit-gradient/analytic-KL variant instead of their RSVI estimator.
The additive LDA and embedded decoders are explicitly labelled adaptations.
Inference freezes ordinary training EMA statistics; it never fits on validation.
"""

from __future__ import annotations

import numpy as np
import torch
from entmax import entmax15
from torch import nn
from torch.distributions import Dirichlet, kl_divergence
from torch.nn import functional as F

from .contextual_sparse_etm import diagonal_gaussian_kl

VARIANTS = ("prodlda", "dvae_poe", "dirichlet_lda", "dirichlet_etm", "prodlda_entmax")


def fixed_scale_batchnorm(width: int, *, learn_bias: bool) -> nn.BatchNorm1d:
    """Match affine-free Pyro or unit-scale/learned-bias TensorFlow BatchNorm."""
    layer = nn.BatchNorm1d(width, affine=learn_bias)
    if learn_bias:
        layer.weight.requires_grad_(False)
    return layer


class LogisticNormalEncoder(nn.Module):
    """Pyro tutorial's two-softplus Gaussian encoder and standard-normal KL."""

    def __init__(
        self, vocabulary: int, topics: int, hidden: int = 100, *, sparse: bool = False
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.hidden = nn.Sequential(
            nn.Linear(vocabulary, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Dropout(0.2),
        )
        self.mean = nn.Linear(hidden, topics)
        self.logvar = nn.Linear(hidden, topics)
        self.mean_bn = fixed_scale_batchnorm(topics, learn_bias=False)
        self.logvar_bn = fixed_scale_batchnorm(topics, learn_bias=False)

    def forward(self, counts: torch.Tensor, *, sample: bool):
        encoded = self.hidden(counts)
        mean = self.mean_bn(self.mean(encoded))
        logvar = self.logvar_bn(self.logvar(encoded))
        latent = (
            mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        )
        theta = entmax15(latent, dim=-1) if self.sparse else latent.softmax(dim=-1)
        return theta, diagonal_gaussian_kl(mean, logvar)


class DirichletEncoder(nn.Module):
    """One neural concentration head with exact Dirichlet sampling and KL."""

    def __init__(
        self,
        vocabulary: int,
        topics: int,
        hidden: int = 100,
        concentration: float = 0.02,
    ) -> None:
        super().__init__()
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError(
                "per-topic prior concentration must be positive and finite"
            )
        self.hidden = nn.Sequential(
            nn.Linear(vocabulary, hidden), nn.ReLU(), nn.Dropout(0.25)
        )
        self.head = nn.Linear(hidden, topics)
        self.head_bn = fixed_scale_batchnorm(topics, learn_bias=True)
        self.register_buffer("prior", torch.full((topics,), float(concentration)))

    def posterior(self, counts: torch.Tensor) -> Dirichlet:
        concentration = F.softplus(self.head_bn(self.head(self.hidden(counts))))
        return Dirichlet(concentration.clamp_min(1e-5))

    def forward(self, counts: torch.Tensor, *, sample: bool):
        posterior = self.posterior(counts)
        theta = posterior.rsample() if sample else posterior.mean
        return theta, kl_divergence(posterior, Dirichlet(self.prior))


class ProductDecoder(nn.Module):
    """Normalized product-of-experts, NOT an additive mixture of emissions."""

    def __init__(
        self, vocabulary: int, topics: int, *, dropout: float, learn_bias: bool
    ) -> None:
        super().__init__()
        self.weights = nn.Linear(topics, vocabulary, bias=False)
        self.normalization = fixed_scale_batchnorm(vocabulary, learn_bias=learn_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(
            self.normalization(self.weights(self.dropout(theta))), dim=1
        )


class MixtureDecoder(nn.Module):
    """Classical additive LDA, optionally factorizing logits as in ETM."""

    def __init__(
        self, vocabulary: int, topics: int, embeddings: np.ndarray | None = None
    ) -> None:
        super().__init__()
        if embeddings is None:
            self.logits = nn.Parameter(torch.empty(topics, vocabulary))
            nn.init.xavier_uniform_(self.logits)
            self.register_buffer("rho", None)
        else:
            self.register_buffer("rho", torch.as_tensor(embeddings).float().clone())
            self.logits = nn.Parameter(torch.empty(topics, embeddings.shape[1]))
            # Same initialization as the canonical ETM topic embedding layer.
            nn.init.kaiming_uniform_(self.logits, a=5**0.5)

    def topic_probabilities(self) -> torch.Tensor:
        logits = self.logits if self.rho is None else self.logits @ self.rho.T
        return logits.softmax(dim=1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta @ self.topic_probabilities()).clamp_min(1e-30).log()


class PublishedTopicModel(nn.Module):
    """Compose an explicit published encoder and decoder without hidden losses."""

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, topics: int, *, product: bool
    ) -> None:
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.topics, self.product = topics, product

    def forward(self, counts: torch.Tensor, *, sample: bool):
        theta, kl = self.encoder(counts, sample=sample)
        return self.decoder(theta), theta, kl

    @torch.inference_mode()
    def topic_prototypes(self) -> torch.Tensor:
        """Conditional p(word | theta=e_k); for additive models these are beta."""
        if self.training:
            raise ValueError("topic export requires frozen evaluation mode")
        if not self.product:
            return self.decoder.topic_probabilities()
        device = next(self.parameters()).device
        return self.decoder(torch.eye(self.topics, device=device)).exp()


def build_published_model(
    variant: str, embeddings: np.ndarray, topics: int, *, hidden: int = 100
) -> PublishedTopicModel:
    """Build one fixed recipe; no parameter selection based on held-out data."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown published variant: {variant}")
    if topics < 2 or hidden < 1 or embeddings.ndim != 2:
        raise ValueError("invalid topics, hidden width, or embedding matrix")
    vocabulary = len(embeddings)
    if variant in ("prodlda", "prodlda_entmax"):
        return PublishedTopicModel(
            LogisticNormalEncoder(
                vocabulary, topics, hidden, sparse=variant == "prodlda_entmax"
            ),
            ProductDecoder(vocabulary, topics, dropout=0.2, learn_bias=False),
            topics,
            product=True,
        )
    encoder = DirichletEncoder(vocabulary, topics, hidden)
    if variant == "dvae_poe":
        decoder = ProductDecoder(vocabulary, topics, dropout=0.0, learn_bias=True)
    else:
        decoder = MixtureDecoder(
            vocabulary, topics, embeddings if variant == "dirichlet_etm" else None
        )
    return PublishedTopicModel(encoder, decoder, topics, product=variant == "dvae_poe")
