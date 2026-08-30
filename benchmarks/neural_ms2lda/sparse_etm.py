"""Principled sparse-theta variants of the fixed-SGNS Embedded Topic Model."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from entmax import entmax15, sparsemax
from torch import nn
from torch.nn import functional as nnf

from .diagnostics import normalize_mixtures

EPS = 1e-12
ThetaTransform = Literal["softmax", "entmax15", "sparsemax"]
ReconstructionScaling = Literal["raw_counts", "distinct_words", "unit_mass"]
THETA_TRANSFORMS: tuple[ThetaTransform, ...] = (
    "softmax",
    "entmax15",
    "sparsemax",
)
RECONSTRUCTION_SCALINGS: tuple[ReconstructionScaling, ...] = (
    "raw_counts",
    "distinct_words",
    "unit_mass",
)


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


def dense_normalized(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return the canonical row-normalized dense ETM encoder input."""
    values = torch.from_numpy(
        matrix[rows].toarray().astype(np.float32, copy=False),
    ).to(device)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1.0)


def sparse_reconstruction_loss(
    theta: torch.Tensor,
    beta: torch.Tensor,
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
    *,
    scaling: ReconstructionScaling,
) -> tuple[torch.Tensor, float]:
    """Evaluate observed-word reconstruction with a predeclared mass scaling.

    ``raw_counts`` is the canonical multinomial pseudo-count objective.
    ``distinct_words`` preserves each row's relative intensity weights while
    setting its total effective mass to its number of observed nonzero words.
    ``unit_mass`` treats the row-normalized intensity vector as one observation.
    """
    if scaling not in RECONSTRUCTION_SCALINGS:
        raise ValueError(f"unknown reconstruction scaling: {scaling}")
    batch = matrix[rows].tocsr()
    if batch.nnz == 0:
        return theta.new_zeros(()), 0.0
    lengths = np.diff(batch.indptr).astype(np.int64, copy=False)
    row_ids_array = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    row_ids = torch.from_numpy(row_ids_array).to(device)
    word_ids = torch.from_numpy(
        batch.indices.astype(np.int64, copy=False),
    ).to(device)
    weights_array = batch.data.astype(np.float32, copy=True)
    raw_totals = np.asarray(batch.sum(axis=1)).ravel().astype(np.float32)
    if scaling == "distinct_words":
        target_totals = lengths.astype(np.float32)
    elif scaling == "unit_mass":
        target_totals = (raw_totals > 0).astype(np.float32)
    else:
        target_totals = raw_totals
    if scaling != "raw_counts":
        scale = np.divide(
            target_totals,
            raw_totals,
            out=np.zeros_like(target_totals),
            where=raw_totals > 0,
        )
        weights_array *= scale[row_ids_array]
    weights = torch.from_numpy(weights_array).to(device)
    probability = torch.sum(
        theta[row_ids] * beta[:, word_ids].T,
        dim=1,
    ).clamp_min(EPS)
    per_document = theta.new_zeros((len(rows),))
    per_document.index_add_(0, row_ids, -weights * torch.log(probability))
    return per_document.mean(), float(target_totals.mean())


def theta_support_diagnostics(theta: np.ndarray) -> dict[str, object]:
    """Summarize exact support, entropy-effective topics and confidence."""
    mixtures = normalize_mixtures(theta)
    support = np.count_nonzero(mixtures > 0.0, axis=1)
    entropy = -np.sum(
        np.where(
            mixtures > 0.0,
            mixtures * np.log(np.clip(mixtures, EPS, None)),
            0.0,
        ),
        axis=1,
    )
    maximum = mixtures.max(axis=1)
    percentiles = {
        str(percentile): float(np.percentile(support, percentile))
        for percentile in (1, 5, 25, 50, 75, 95, 99)
    }
    return {
        "minimum_exact_support": int(support.min()),
        "support_size_percentiles": percentiles,
        "median_exact_support": float(np.median(support)),
        "mean_exact_support": float(support.mean()),
        "maximum_exact_support": int(support.max()),
        "fraction_support_le_3": float(np.mean(support <= 3)),
        "median_effective_topics_per_spectrum": float(np.median(np.exp(entropy))),
        "mean_effective_topics_per_spectrum": float(np.mean(np.exp(entropy))),
        "median_maximum_theta": float(np.median(maximum)),
        "fraction_max_theta_ge_0_5": float(np.mean(maximum >= 0.5)),
        "fraction_max_theta_ge_0_3": float(np.mean(maximum >= 0.3)),
        "fraction_max_theta_ge_0_2": float(np.mean(maximum >= 0.2)),
    }
