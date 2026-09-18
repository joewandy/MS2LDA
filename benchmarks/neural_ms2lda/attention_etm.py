"""MLP-free attention ETM with an explicit, replaceable posterior encoder.

Research only. This shell reuses the frozen model's tensor equations without
changing the ablation implementation or any benchmark already in progress.
The default encoder is exactly ``reduced_evidence_only``. A replacement encoder
may consume a peak batch or spectrum embeddings instead of normalized bows;
the Gaussian prior, entmax link and token-emission decoder stay independent.
No raw-spectrum foundation model or peak-set likelihood is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    centered_log_evidence_offset,
    channel_balanced_topic_word_distribution,
    contextual_top2_evidence,
    diagonal_gaussian_kl,
    entmax15_document_mixture,
    reparameterized_gaussian,
)


class TokenAttentionEncoder(nn.Module):
    """Learned topic attention plus one global log variance per topic; no MLP.

    Encoder contract: ``forward(observations, *, topic_embeddings,
    word_embeddings) -> (mean, log_variance)``, two ``(batch, K)`` tensors.
    The topic model does not inspect ``observations``. Only this default encoder
    requires normalized bows. Replacement encoders can ignore the supplied word
    embeddings, and can use or ignore the shared topic vectors. Neither tensor
    is registered again inside the encoder, so ownership and gradients remain
    unambiguous. Replacing the encoder does not replace the token likelihood.
    """

    def __init__(self, topics: int) -> None:
        super().__init__()
        if topics < 2:
            raise ValueError("topic attention requires at least two topics")
        self.context_scale = nn.Parameter(torch.ones(()))
        self.global_logvar = nn.Parameter(torch.zeros(topics))

    def forward(self, observations, *, topic_embeddings, word_embeddings):
        evidence = contextual_top2_evidence(
            observations,
            word_embeddings,
            topic_embeddings,
            self.context_scale,
        )
        mean = centered_log_evidence_offset(evidence)
        return mean, self.global_logvar.expand_as(mean)


class AttentionETM(nn.Module):
    """Fixed ETM geometry/decoder and Gaussian-entmax core, separate from input.

    A custom ``encoder`` must be an ``nn.Module`` following the contract above.
    Raw-spectrum encoders would still need aligned token targets when training
    this decoder. Fully tokenizer-free modelling requires a different emission
    model and an independently validated motif interpretation, both deferred.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        rho = np.asarray(embeddings, dtype=np.float32)
        if rho.ndim != 2 or not all(rho.shape) or not np.isfinite(rho).all():
            raise ValueError("embeddings must be a finite non-empty matrix")
        if not np.allclose(np.linalg.norm(rho, axis=1), 1, rtol=1e-5, atol=1e-6):
            raise ValueError("word embeddings must be row-wise unit normalized")
        if int(topics) != topics or topics < 2:
            raise ValueError("topics must be an integer of at least two")
        mask = np.asarray(fragment_mask, dtype=bool)
        if mask.shape != (rho.shape[0],) or not mask.any() or mask.all():
            raise ValueError("fragment_mask must identify both vocabulary channels")
        if encoder is not None and not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module")
        self.register_buffer("rho", torch.from_numpy(rho.copy()))
        # Unlike the historical checkpoint, this shell saves its channel mask.
        self.register_buffer("fragment_mask", torch.from_numpy(mask.copy()))
        self.alphas = nn.Linear(rho.shape[1], int(topics), bias=False)
        self.encoder = (
            TokenAttentionEncoder(int(topics)) if encoder is None else encoder
        )

    def topic_word_distribution(self) -> torch.Tensor:
        return channel_balanced_topic_word_distribution(
            self.rho, self.alphas.weight, self.fragment_mask
        )

    def posterior(self, observations):
        mean, logvar = self.encoder(
            observations,
            topic_embeddings=self.alphas.weight,
            word_embeddings=self.rho,
        )
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(logvar, torch.Tensor)
            or mean.ndim != 2
            or mean.shape[1] != self.alphas.out_features
            or logvar.shape != mean.shape
        ):
            raise ValueError("encoder must return matching batch-by-topic tensors")
        for value in (mean, logvar):
            if value.device != self.rho.device or value.dtype != self.rho.dtype:
                raise ValueError("encoder outputs must match model device and dtype")
            if not torch.isfinite(value).all():
                raise FloatingPointError("encoder returned non-finite parameters")
        kl = diagonal_gaussian_kl(mean, logvar)
        if not torch.isfinite(kl).all():
            raise FloatingPointError("encoder parameters overflowed Gaussian KL")
        return mean, logvar, kl

    def document_topic_mixture(self, observations, *, sample: bool):
        mean, logvar, kl = self.posterior(observations)
        z = reparameterized_gaussian(mean, logvar, sample=sample)
        return entmax15_document_mixture(z), kl

    @classmethod
    def from_evidence_only_state(
        cls, state: Mapping[str, torch.Tensor], fragment_mask: np.ndarray
    ) -> AttentionETM:
        """Import exactly the MLP-free ablation state without changing its math.

        Historical checkpoints omit the channel mask, so the caller must supply
        it from the same audited vocabulary. This returns a CPU float32 model;
        call ``.to(...)`` and ``.eval()`` explicitly for evaluation. Initialization
        consumes a different RNG sequence from the ablation's create-then-delete
        MLP; this is a state compatibility path, not a seed-replay guarantee.
        """
        keys = {"rho", "alphas.weight", "context_scale", "global_logvar"}
        if set(state) != keys:
            raise ValueError("expected exactly an evidence-only ablation state")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("checkpoint contains non-finite tensors")
        model = cls(
            state["rho"].detach().cpu().numpy(),
            state["alphas.weight"].shape[0],
            fragment_mask,
        )
        model.load_state_dict(
            {
                "rho": state["rho"],
                "fragment_mask": model.fragment_mask,
                "alphas.weight": state["alphas.weight"],
                "encoder.context_scale": state["context_scale"],
                "encoder.global_logvar": state["global_logvar"],
            },
            strict=True,
        )
        return model
