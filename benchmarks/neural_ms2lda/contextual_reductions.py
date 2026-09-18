"""Surgical reductions of the frozen Contextual Sparse ETM, research only.

Each switch removes or simplifies a named mechanism. Shared parameters retain
the frozen model's seeded initialization. No switch changes the training loss.

The manuscript's chosen model is ``variant="reduced_document_context"``.
Follow just that variant to read the current method: channel-balanced ETM
decoder (inherited), two-layer Gaussian MLP (inherited), whole-spectrum top-2
evidence, a centered log mean offset, and Gaussian-to-entmax mixtures.
Other named switches are retained to reproduce the archived ablations, not
combined automatically or selected using validation data inside this module.

Equation labels below refer to the detailed formulation in
``docs/research/contextual_sparse_etm_supplement.tex``. The companion
``contextual_sparse_etm_report.tex`` explains the base model and enhancements.
For a batch of B spectra, V words, K topics and embedding dimension E:
``x`` is B x V, ``rho`` is V x E, ``alpha`` is K x E, and ``r``/``theta`` are B x K.
Spectrum, word and topic indices correspond to the paper's d, w and k.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    EPSILON,
    ContextualSparseETM,
    centered_log_evidence_offset,
    diagonal_gaussian_kl,
    entmax15_document_mixture,
    leave_one_out_context,
    reparameterized_gaussian,
    unit_normalize_rows,
)

REDUCTION_VARIANTS = {
    "reduced_no_loo": ("no_loo",),
    "reduced_document_context": ("document_context",),
    "reduced_fixed_context": ("fixed_context",),
    "reduced_full_routing": ("full_routing",),
    "reduced_linear_evidence": ("linear_evidence",),
    "reduced_softmax": ("softmax",),
    "reduced_shallow": ("shallow",),
    "reduced_evidence_only": ("evidence_only",),
    "reduced_scaled_full_routing": ("full_routing", "scaled_routing"),
    "reduced_fixed_context_top1": ("fixed_context", "top1"),
    # Same one-layer form; the declared follow-on recipe supplies hidden=100.
    "reduced_shallow_narrow": ("shallow",),
    # Predeclared combination/top-1 follow-ups, run only after the first screen.
    "reduced_no_loo_shallow": ("no_loo", "shallow"),
    "reduced_no_loo_top1": ("no_loo", "top1"),
    "reduced_no_loo_shallow_top1": ("no_loo", "shallow", "top1"),
    "reduced_document_context_shallow": ("document_context", "shallow"),
    "reduced_document_context_fixed": ("document_context", "fixed_context"),
    "reduced_document_context_fixed_shallow": (
        "document_context",
        "fixed_context",
        "shallow",
    ),
    "reduced_attention_document_fixed": (
        "document_context",
        "fixed_context",
        "evidence_only",
    ),
}


def pooled_evidence(
    x: torch.Tensor,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    context_scale: torch.Tensor,
    *,
    context: str = "leave_one_out",
    routing: int | None = 2,
    score_scale: float = 1.0,
) -> torch.Tensor:
    """Compute r_dk from contextual word directions and count-weighted attention.

    Current-model settings are ``context="document", routing=2, score_scale=1``:

    1. ``s = x @ normalize(rho)`` forms the full-spectrum mean embedding.
    2. ``h_dw = normalize(rho_hat_w + context_scale * s_d)`` implements
       ``eq:document-context``. The scored word is INCLUDED in s_d.
    3. Cosine scores ``h_dw @ alpha_hat.T`` give a local softmax on the two
       highest scoring topics, with all other entries exactly zero.
    4. ``r_dk = sum_w x_dw * pi_dwk`` is ``eq:document-evidence``.

    ``context_scale`` is the paper's learned scalar lambda_ctx, not a network.
    Attention pi is deterministic evidence for the Gaussian encoder, not an
    additional latent variable, likelihood factor or exact word-topic posterior.
    Top-2 is a PER-WORD choice: different words may support many spectrum topics.

    The remaining settings describe archived ablations: ``routing=None`` uses
    all topics, and ``routing=1`` is hard routing.

    Without context, the attention is vocabulary-level and r = x A. This
    removes both the leave-one-out computation and the learned context scalar.
    Top-1 has zero routing gradient almost everywhere; decoder/MLP still train.
    """
    if context not in {"none", "document", "leave_one_out"}:
        raise ValueError("unknown context rule")
    if routing not in {None, 1, 2}:
        raise ValueError("routing must be all topics, top-1 or top-2")
    if x.ndim != 2 or x.shape[1] != rho.shape[0]:
        raise ValueError("x must match the embedding vocabulary")
    if not torch.isfinite(x).all() or torch.any(x < 0):
        raise ValueError("x must be finite and nonnegative")
    masses = x.sum(1)
    if not torch.all((masses == 0) | torch.isclose(masses, torch.ones_like(masses))):
        raise ValueError("nonempty x rows must sum to one")
    rho_hat, alpha_hat = unit_normalize_rows(rho), unit_normalize_rows(alpha)

    def attention(h: torch.Tensor) -> torch.Tensor:
        # N observed (document, word) pairs -> N x K scores. With unit vectors
        # and score_scale=1 these are the paper's untempered cosine scores a_dwk.
        scores = (h @ alpha_hat.T) * score_scale
        if routing is None:
            return torch.softmax(scores, dim=1)
        values, indices = torch.topk(scores, k=routing, dim=1)
        # Normalize only selected scores, then put pi_dwk back into K columns.
        return torch.zeros_like(scores).scatter(
            1, indices, torch.softmax(values, dim=1)
        )

    if context == "none":
        r = x @ attention(unit_normalize_rows(rho_hat))
    else:
        # Work only on observed words. No B x V x E context tensor is required.
        documents, words = torch.nonzero(x, as_tuple=True)
        surrounding = (
            (x @ rho_hat)[documents]
            if context == "document"
            else leave_one_out_context(x, rho_hat, documents, words)
        )
        h = unit_normalize_rows(rho_hat[words] + context_scale * surrounding)
        # Each observed word contributes its normalized pseudo-count x_dw.
        # index_add sums contributions from the same document into one r_d row.
        weights = x[documents, words, None] * attention(h)
        r = torch.zeros(
            x.shape[0], alpha.shape[0], device=x.device, dtype=x.dtype
        ).index_add(0, documents, weights)
    mass = r.sum(1, keepdim=True)
    # Nonempty rows sum to one in exact arithmetic. Correct accumulation error;
    # an empty-input row uses uniform evidence, hence zero centered mean offset.
    return torch.where(mass > 0, r / mass.clamp_min(EPSILON), 1.0 / alpha.shape[0])


def linear_evidence_offset(evidence: torch.Tensor) -> torch.Tensor:
    """First-order Taylor approximation of the centred log at uniform evidence.

    d log(r + 1/K) / dr at r=1/K is K/2, fixing the scale without tuning.
    """
    return 0.5 * evidence.shape[1] * (evidence - evidence.mean(1, keepdim=True))


class ReducedContextualETM(ContextualSparseETM):
    """ETM parameter state plus explicit, historically tested reductions.

    This subclass preserves checkpoint names and seeded parameter initialization.
    In particular, ablated parameters are constructed before removal so shared
    parameters start identically in paired experiments. Reordering initialization
    would change those experiments even if the equations remained the same.
    No per-spectrum parameters, extra losses or trainable routing network exist.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        variant: str,
        hidden: int = 800,
    ) -> None:
        if variant not in REDUCTION_VARIANTS:
            raise ValueError(f"unknown reduction: {variant}")
        super().__init__(embeddings, topics, fragment_mask, hidden=hidden)
        self.reductions = REDUCTION_VARIANTS[variant]
        if "no_loo" in self.reductions or "fixed_context" in self.reductions:
            del self.context_scale
            self.register_buffer(
                "context_scale",
                torch.tensor(0.0 if "no_loo" in self.reductions else 1.0),
            )
        if "shallow" in self.reductions:
            self.encoder = self.encoder[:2]
        if "evidence_only" in self.reductions:
            # A learned topic-attention encoder remains; only the residual MLP
            # and its document-dependent variance are removed. q is Gaussian.
            del self.encoder, self.mu, self.logvar
            self.global_logvar = nn.Parameter(torch.zeros(topics))

    def contextual_evidence(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """Return B x K evidence; the selected variant uses full-document context.

        The switches below only dispatch archived experiments. For
        ``reduced_document_context`` they resolve to (document, top-2, scale 1).
        """
        if any(
            flag in self.reductions
            for flag in ("no_loo", "document_context", "full_routing", "top1")
        ):
            context = (
                "none"
                if "no_loo" in self.reductions
                else (
                    "document"
                    if "document_context" in self.reductions
                    else "leave_one_out"
                )
            )
            routing = (
                None
                if "full_routing" in self.reductions
                else 1 if "top1" in self.reductions else 2
            )
            return pooled_evidence(
                normalized_bows,
                self.rho,
                self.alphas.weight,
                self.context_scale,
                context=context,
                routing=routing,
                score_scale=(
                    self.rho.shape[1] ** 0.5
                    if "scaled_routing" in self.reductions
                    else 1.0
                ),
            )
        return super().contextual_evidence(normalized_bows)

    def posterior(self, normalized_bows: torch.Tensor):
        """Return (m, ell, KL): B x K means/log variances and B KL values.

        Current model: m = mu_phi(x) + center(log(r + 1/K)), while
        ell = ell_phi(x) remains the ETM MLP log-variance head. This is
        ``eq:base-posterior`` plus ``eq:posterior-offset``. Adding the offset
        affects BOTH reconstruction and KL; the latter must use shifted m.
        No attention-specific penalty is added to the Gaussian ELBO.
        """
        r = self.contextual_evidence(normalized_bows)
        offset = (
            linear_evidence_offset(r)
            if "linear_evidence" in self.reductions
            else centered_log_evidence_offset(r)
        )
        if "evidence_only" in self.reductions:
            mean = offset
            logvar = self.global_logvar.expand_as(mean)
        else:
            encoded = self.encoder(normalized_bows)
            mean = self.mu(encoded) + offset
            logvar = self.logvar(encoded)
        return mean, logvar, diagonal_gaussian_kl(mean, logvar)

    def document_topic_mixture(self, normalized_bows: torch.Tensor, *, sample: bool):
        """Map one Gaussian draw (training) or its mean (inference) to theta.

        z = m + exp(ell/2) * epsilon, epsilon ~ N(0,I), then theta = entmax1.5(z)
        implements ``eq:entmax`` and the sample used in ``eq:elbo``.
        ``sample=False`` returns entmax(m), not E_q[entmax(z)]. The KL is in
        Gaussian latent space, so no simplex-density/Jacobian term is needed.
        Softmax below is an archived ablation, not the selected model's map.
        """
        mean, logvar, kl = self.posterior(normalized_bows)
        z = reparameterized_gaussian(mean, logvar, sample=sample)
        theta = (
            torch.softmax(z, dim=1)
            if "softmax" in self.reductions
            else entmax15_document_mixture(z)
        )
        return theta, kl
