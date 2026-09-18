"""ETM with two independently ablatable changes from AVITM (ICLR 2017).

The decoder, softmax mixture and encoder architecture are ordinary ETM.
The optional changes are (1) AVITM's diagonal Laplace approximation to a
symmetric Dirichlet prior and (2) batch normalization of the posterior heads.
This is an ETM adaptation, not a reproduction of the full AVITM architecture.
There is no contextual routing, channel constraint or exact sparsity transform.

Sources: https://aclanthology.org/2020.tacl-1.29/ and
https://arxiv.org/abs/1703.01488 (equations 6--7 and section 3.4).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .etm_baselines import CanonicalETM


def symmetric_dirichlet_logit_variance(topics: int, concentration: float) -> float:
    """Return AVITM's Laplace variance for symmetric Dirichlet(a,...,a).

    Equation 6 reduces to (1 - 1/K) / a, with zero prior mean. Concentration
    is PER TOPIC; the total Dirichlet concentration is K*a. This is a
    logistic-normal approximation, not an exact Dirichlet distribution.
    """
    if topics < 2:
        raise ValueError("the Dirichlet approximation needs at least two topics")
    if not math.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    return (1.0 - 1.0 / topics) / concentration


def isotropic_gaussian_kl(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    prior_variance: torch.Tensor,
) -> torch.Tensor:
    """Compute KL[N(mean,diag(exp(log_variance))) || N(0,v I)] per row."""
    return 0.5 * (
        (log_variance.exp() + mean.square()) / prior_variance
        - 1.0
        + prior_variance.log()
        - log_variance
    ).sum(dim=1)


class PriorETM(CanonicalETM):
    """An unchanged ETM decoder with a published prior/normalization recipe.

    ``concentration=None`` retains ETM's standard-normal prior. Set
    ``batch_normalize=False`` to remove the second modification independently.
    BatchNorm uses training statistics only; evaluation requires ``eval()``.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        *,
        hidden: int = 800,
        concentration: float | None = 0.02,
        batch_normalize: bool = True,
        learn_normalization_scale: bool = True,
    ) -> None:
        super().__init__(embeddings, topics, hidden=hidden)
        variance = (
            1.0
            if concentration is None
            else symmetric_dirichlet_logit_variance(topics, concentration)
        )
        self.register_buffer("prior_variance", torch.tensor(variance))
        self.mean_normalization = (
            nn.BatchNorm1d(topics) if batch_normalize else nn.Identity()
        )
        self.variance_normalization = (
            nn.BatchNorm1d(topics) if batch_normalize else nn.Identity()
        )
        if batch_normalize and not learn_normalization_scale:
            # AVITM's tf.contrib.layers.batch_norm defaults to scale=False.
            # The published ECRTM implementation likewise freezes both scales.
            self.mean_normalization.weight.requires_grad_(False)
            self.variance_normalization.weight.requires_grad_(False)

    def posterior(
        self, normalized_bows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Gaussian parameters and KL against the configured prior."""
        encoded = self.encoder(normalized_bows)
        mean = self.mean_normalization(self.mu(encoded))
        log_variance = self.variance_normalization(self.logvar(encoded))
        return (
            mean,
            log_variance,
            isotropic_gaussian_kl(mean, log_variance, self.prior_variance),
        )
