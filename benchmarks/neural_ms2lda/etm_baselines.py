"""Published ETM controls used by the Contextual Sparse ETM study.

Only two architectures live here: the canonical Embedded Topic Model (ETM)
with fixed train-only SGNS word embeddings, and the same model with independent
fragment and neutral-loss decoder normalization.  The implementations expose
the same small mathematical interface as :class:`ContextualSparseETM` so the
training and held-out evaluation code can remain ordinary functions.
"""

from __future__ import annotations

from os import PathLike

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from .contextual_sparse_etm import (
    channel_balanced_topic_word_distribution,
    diagonal_gaussian_kl,
    reparameterized_gaussian,
)

DEFAULT_HIDDEN_WIDTH = 800


def load_sgns_embeddings(path: str | bytes | PathLike[str]) -> np.ndarray:
    """Load and row-normalize the SGNS coordinates from the feature table."""
    features = np.load(path).astype(np.float32, copy=False)
    if features.ndim != 2 or features.shape[1] <= 2:
        raise ValueError("token features must contain SGNS coordinates and two flags")
    embeddings = np.array(features[:, :-2], dtype=np.float32, copy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("SGNS embeddings must have finite non-zero row norms")
    embeddings /= norms
    return embeddings


class CanonicalETM(nn.Module):
    """Dieng, Ruiz and Blei's ETM with fixed pretrained word embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        *,
        hidden: int = DEFAULT_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        rho = np.asarray(embeddings, dtype=np.float32)
        if rho.ndim != 2 or not rho.shape[0] or not rho.shape[1]:
            raise ValueError("embeddings must be a non-empty matrix")
        if not np.all(np.isfinite(rho)):
            raise ValueError("embeddings must be finite")
        if int(topics) <= 0 or int(hidden) <= 0:
            raise ValueError("topics and hidden width must be positive")
        self.register_buffer("rho", torch.from_numpy(rho.copy()))
        self.alphas = nn.Linear(rho.shape[1], int(topics), bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(rho.shape[0], int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
        )
        self.mu = nn.Linear(int(hidden), int(topics))
        self.logvar = nn.Linear(int(hidden), int(topics))

    def topic_word_distribution(self) -> torch.Tensor:
        """Return canonical ETM ``beta`` using one vocabulary softmax."""
        return nnf.softmax(self.alphas(self.rho), dim=0).T

    def posterior(
        self,
        normalized_bows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return posterior mean, log variance and analytic Gaussian KL."""
        encoded = self.encoder(normalized_bows)
        mean = self.mu(encoded)
        log_variance = self.logvar(encoded)
        return mean, log_variance, diagonal_gaussian_kl(mean, log_variance)

    def document_topic_mixture(
        self,
        normalized_bows: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return canonical softmax ``theta`` and KL for training or inference."""
        mean, log_variance, kl = self.posterior(normalized_bows)
        latent = reparameterized_gaussian(mean, log_variance, sample=sample)
        return nnf.softmax(latent, dim=1), kl


class ChannelBalancedETM(CanonicalETM):
    """Canonical ETM with only fragment/loss channel normalization changed."""

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        hidden: int = DEFAULT_HIDDEN_WIDTH,
    ) -> None:
        super().__init__(embeddings, topics, hidden=hidden)
        mask = np.asarray(fragment_mask, dtype=bool)
        if mask.shape != (len(embeddings),):
            raise ValueError("fragment mask must match the ETM vocabulary")
        if not mask.any() or mask.all():
            raise ValueError("fragment mask must contain fragments and losses")
        self.register_buffer(
            "fragment_mask",
            torch.from_numpy(mask.copy()),
            persistent=False,
        )

    def topic_word_distribution(self) -> torch.Tensor:
        """Return ``beta`` with fixed half-mass fragment and loss channels."""
        return channel_balanced_topic_word_distribution(
            self.rho,
            self.alphas.weight,
            self.fragment_mask,
        )
