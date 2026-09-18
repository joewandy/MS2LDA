"""Two bounded simplifications evaluated against Contextual Sparse ETM.

Both keep an explicit ETM topic-word distribution and a neural Gaussian
encoder. Entmax maps the Gaussian latent variable to the simplex in BOTH the
generative model and variational reconstruction; its KL remains in latent
Gaussian space. It does not preserve ETM's logistic-normal distribution on theta.
"""

from __future__ import annotations

import numpy as np
import torch

from .contextual_sparse_etm import (
    ContextualSparseETM,
    entmax15_document_mixture,
    reparameterized_gaussian,
)
from .prior_etm import PriorETM


class BatchNormSparseETM(PriorETM):
    """ETM with posterior-head BatchNorm and an entmax latent-to-mixture link.

    There is no context, channel constraint, sparse prior or extra loss. The
    learned/fixed-scale switch is the existing normalization-recipe comparison.
    Final BatchNorm moments must be fitted to training rows before inference.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        *,
        hidden: int = 800,
        learn_normalization_scale: bool = False,
    ) -> None:
        super().__init__(
            embeddings,
            topics,
            hidden=hidden,
            concentration=None,
            batch_normalize=True,
            learn_normalization_scale=learn_normalization_scale,
        )

    def document_topic_mixture(
        self, normalized_bows: torch.Tensor, *, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance, kl = self.posterior(normalized_bows)
        latent = reparameterized_gaussian(mean, log_variance, sample=sample)
        return entmax15_document_mixture(latent), kl


class UnbalancedContextualETM(ContextualSparseETM):
    """Remove only the 50/50 channel constraint from the current model.

    The posterior, embeddings, parameter initialization and stochastic objective
    are inherited without modification. Each topic normalizes once over the
    entire fragment/loss vocabulary, exactly as in canonical ETM.
    """

    def topic_word_distribution(self) -> torch.Tensor:
        return torch.softmax(self.alphas(self.rho), dim=0).T
