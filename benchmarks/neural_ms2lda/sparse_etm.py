"""Principled sparse-theta variants of the fixed-SGNS Embedded Topic Model."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from entmax import entmax15, sparsemax
from torch import nn
from torch.nn import functional as nnf

from .model_evaluation import theta_support_diagnostics
from .topic_model_training import (
    RECONSTRUCTION_SCALINGS,
    ReconstructionScaling,
    dense_normalized,
    sparse_reconstruction_loss,
)

ThetaTransform = Literal["softmax", "entmax15", "sparsemax"]
THETA_TRANSFORMS: tuple[ThetaTransform, ...] = (
    "softmax",
    "entmax15",
    "sparsemax",
)
__all__ = [
    "RECONSTRUCTION_SCALINGS",
    "THETA_TRANSFORMS",
    "BalancedSparseETM",
    "ReconstructionScaling",
    "ThetaTransform",
    "dense_normalized",
    "sparse_reconstruction_loss",
    "theta_support_diagnostics",
    "transform_theta",
]


def transform_theta(logits: torch.Tensor, transform: ThetaTransform) -> torch.Tensor:
    """Map unconstrained ETM logits to one probability-simplex row per document."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("theta logits must be a non-empty document-topic matrix")
    if transform == "softmax":
        theta = nnf.softmax(logits, dim=1)
    elif transform == "entmax15":
        theta = entmax15(logits, dim=1)
    elif transform == "sparsemax":
        theta = sparsemax(logits, dim=1)
    else:
        raise ValueError(f"unknown theta transform: {transform}")
    if not torch.all(torch.isfinite(theta)):
        raise FloatingPointError("theta transform produced non-finite values")
    totals = theta.sum(dim=1, keepdim=True)
    if torch.any(totals <= 0):
        raise FloatingPointError("theta transform produced a zero-mass row")
    # At large K the tested entmax kernels can accumulate small float32 simplex
    # error. Numerical row normalization preserves exact zeros and ranks while
    # enforcing the probability contract used by the ETM decoder.
    return theta / totals


class BalancedSparseETM(nn.Module):
    """Fixed-SGNS ETM with balanced beta and a selectable theta transform.

    With ``theta_transform='softmax'`` this is the existing fragment/loss-
    balanced ETM reference. The sparse variants replace only the final mapping
    from sampled Gaussian logits to the topic simplex.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        theta_transform: ThetaTransform = "softmax",
        hidden: int = 800,
    ) -> None:
        super().__init__()
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError("embeddings must be a non-empty word-feature matrix")
        if int(topics) <= 0 or int(hidden) <= 0:
            raise ValueError("topics and hidden dimensions must be positive")
        if theta_transform not in THETA_TRANSFORMS:
            raise ValueError(f"unknown theta transform: {theta_transform}")
        mask = np.asarray(fragment_mask, dtype=bool)
        if mask.shape != (values.shape[0],):
            raise ValueError("fragment mask must match the ETM vocabulary")
        if not mask.any() or mask.all():
            raise ValueError("fragment mask must contain fragments and losses")

        self.theta_transform = theta_transform
        self.register_buffer("rho", torch.from_numpy(values.copy()))
        self.register_buffer(
            "fragment_mask",
            torch.from_numpy(mask.copy()),
            persistent=False,
        )
        self.alphas = nn.Linear(values.shape[1], int(topics), bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(values.shape[0], int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
        )
        self.mu = nn.Linear(int(hidden), int(topics))
        self.logvar = nn.Linear(int(hidden), int(topics))

    def beta(self) -> torch.Tensor:
        """Return topic-word probabilities with fixed 50/50 channel mass."""
        logits = self.alphas(self.rho).T
        probabilities = torch.empty_like(logits)
        probabilities[:, self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, self.fragment_mask],
            dim=1,
        )
        probabilities[:, ~self.fragment_mask] = 0.5 * nnf.softmax(
            logits[:, ~self.fragment_mask],
            dim=1,
        )
        return probabilities

    def encode(
        self,
        normalized_bows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Gaussian posterior parameters and latent-space KL per row."""
        hidden = self.encoder(normalized_bows)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        kl = -0.5 * torch.sum(
            1.0 + logvar - mu.square() - logvar.exp(),
            dim=1,
        )
        return mu, logvar, kl

    def theta(
        self,
        normalized_bows: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer one stochastic or deterministic document-topic mixture."""
        mu, logvar, kl = self.encode(normalized_bows)
        logits = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return transform_theta(logits, self.theta_transform), kl
