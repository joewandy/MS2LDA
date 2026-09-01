"""Executable equations for the Contextual Sparse Embedded Topic Model.

This module is the canonical implementation of the model described in
``docs/research/neural_ms2lda_report.tex``.  The notation in the docstrings
matches the notation in that report:

``x``
    row-normalized spectral-word pseudo-counts;
``rho`` and ``alpha``
    fixed word vectors and learned topic vectors;
``beta``
    the channel-balanced topic-word probability matrix;
``r`` and ``o``
    contextual topic evidence and its centred log offset;
``mu_tilde`` and ``log_sigma_squared``
    the shifted diagonal-Gaussian posterior parameters; and
``z`` and ``theta``
    a reparameterized latent vector and its 1.5-entmax topic mixture.

The model-specific mathematics is deliberately implemented as small tensor
functions.  :class:`ContextualSparseETM` exists only because PyTorch needs an
``nn.Module`` to register trainable parameters and checkpoint them.  It does
not inherit from an experimental model base class and has no selectable modes,
routing strategies, or configurable helper objects.
"""

from __future__ import annotations

import numpy as np
import torch
from entmax import entmax15
from torch import nn
from torch.nn import functional as nnf

EPSILON = 1e-12
FRAGMENT_CHANNEL_MASS = 0.5
TOPICS_PER_TOKEN = 2
ROUTING_TEMPERATURE = 1.0
DEFAULT_HIDDEN_WIDTH = 800


def unit_normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    """Return row-wise unit vectors using the report's ``normalize`` operator.

    For an input matrix ``v`` this computes

    ``v / max(||v||_2, epsilon)``

    independently for each row.  Keeping this operation explicit makes the
    geometry used by contextual evidence easy to compare with equations
    ``eq:loo-context``--``eq:routing-score`` in the report.
    """
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("vectors must be a non-empty row matrix")
    row_norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors / row_norms.clamp_min(EPSILON)


def channel_balanced_topic_word_distribution(
    word_embeddings: torch.Tensor,
    topic_embeddings: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the channel-balanced ETM decoder ``beta`` (equation ``eq:beta``).

    Parameters
    ----------
    word_embeddings
        ``rho`` with shape ``(vocabulary, embedding_dimensions)``.
    topic_embeddings
        ``alpha`` with shape ``(topics, embedding_dimensions)``.
    fragment_mask
        Boolean vector of length ``vocabulary``.  False entries are
        neutral-loss words.

    Returns
    -------
    torch.Tensor
        ``beta`` with shape ``(topics, vocabulary)``.  Every topic allocates
        exactly half of its probability mass to fragments and half to losses.
    """
    if word_embeddings.ndim != 2 or topic_embeddings.ndim != 2:
        raise ValueError("word and topic embeddings must be matrices")
    if word_embeddings.shape[1] != topic_embeddings.shape[1]:
        raise ValueError("word and topic embedding dimensions must match")
    if fragment_mask.dtype != torch.bool or fragment_mask.shape != (
        word_embeddings.shape[0],
    ):
        raise ValueError("fragment_mask must be one Boolean value per word")
    if not torch.any(fragment_mask) or torch.all(fragment_mask):
        raise ValueError("fragment_mask must contain fragments and losses")

    # The orientation mirrors alpha_k^T rho_w in equation (eq:beta): rows are
    # topics and columns are spectral words.
    topic_word_logits = (word_embeddings @ topic_embeddings.T).T
    beta = torch.empty_like(topic_word_logits)
    beta[:, fragment_mask] = FRAGMENT_CHANNEL_MASS * nnf.softmax(
        topic_word_logits[:, fragment_mask],
        dim=1,
    )
    beta[:, ~fragment_mask] = (1.0 - FRAGMENT_CHANNEL_MASS) * nnf.softmax(
        topic_word_logits[:, ~fragment_mask],
        dim=1,
    )
    return beta


def leave_one_out_context(
    normalized_bows: torch.Tensor,
    unit_word_embeddings: torch.Tensor,
    document_indices: torch.Tensor,
    word_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute ``rho_bar_(d\\w)`` for observed words (equation ``eq:loo-context``).

    ``document_indices`` and ``word_indices`` identify the non-zero entries of
    ``normalized_bows``.  The returned matrix has one row per such entry, so it
    can be processed without materializing a document-by-word-by-embedding
    tensor.
    """
    weights = normalized_bows[document_indices, word_indices]
    document_embedding_sums = normalized_bows @ unit_word_embeddings
    numerators = (
        document_embedding_sums[document_indices]
        - weights.unsqueeze(1) * unit_word_embeddings[word_indices]
    )
    denominators = (1.0 - weights).clamp_min(EPSILON)
    return numerators / denominators.unsqueeze(1)


def contextual_top2_evidence(
    normalized_bows: torch.Tensor,
    word_embeddings: torch.Tensor,
    topic_embeddings: torch.Tensor,
    context_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute document evidence ``r`` from the contextual-routing equations.

    This implements equations ``eq:loo-context`` through
    ``eq:document-evidence`` in the report.

    Each observed word is combined with its weighted leave-one-out context,
    compared with every topic in the shared embedding geometry, and assigned
    only to its two highest-scoring topics.  The local softmax assignments are
    then aggregated with the normalized spectral-word weights ``x_dw``.

    Zero-word rows are outside the fitted-data regime but receive a documented
    uniform fallback so inference remains total and numerically explicit.
    """
    if (
        normalized_bows.ndim != 2
        or normalized_bows.shape[1] != word_embeddings.shape[0]
    ):
        raise ValueError("normalized_bows must have one column per word embedding")
    if topic_embeddings.ndim != 2 or topic_embeddings.shape[0] < TOPICS_PER_TOKEN:
        raise ValueError("contextual top-2 evidence requires at least two topics")
    if word_embeddings.shape[1] != topic_embeddings.shape[1]:
        raise ValueError("word and topic embedding dimensions must match")
    if context_scale.numel() != 1:
        raise ValueError("context_scale must be one scalar")
    if not torch.all(torch.isfinite(normalized_bows)) or torch.any(normalized_bows < 0):
        raise ValueError("normalized_bows must be finite and non-negative")
    bow_masses = normalized_bows.sum(dim=1)
    non_empty_documents = bow_masses > EPSILON
    if torch.any(non_empty_documents) and not torch.allclose(
        bow_masses[non_empty_documents],
        torch.ones_like(bow_masses[non_empty_documents]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("non-empty normalized_bows rows must sum to one")

    document_count = normalized_bows.shape[0]
    topic_count = topic_embeddings.shape[0]
    document_indices, word_indices = torch.nonzero(
        normalized_bows > 0,
        as_tuple=True,
    )
    if document_indices.numel() == 0:
        return normalized_bows.new_full(
            (document_count, topic_count),
            1.0 / topic_count,
        )

    x_dw = normalized_bows[document_indices, word_indices]
    rho_hat = unit_normalize_rows(word_embeddings)
    alpha_hat = unit_normalize_rows(topic_embeddings)
    rho_bar = leave_one_out_context(
        normalized_bows,
        rho_hat,
        document_indices,
        word_indices,
    )

    # h_dw from equation (eq:contextual-word).
    contextual_words = unit_normalize_rows(
        rho_hat[word_indices] + context_scale * rho_bar,
    )

    # a_dwk from equation (eq:routing-score).  The temperature is fixed at one
    # in the reported model but remains visible here to preserve the equation.
    routing_scores = (contextual_words @ alpha_hat.T) / ROUTING_TEMPERATURE
    top_scores, top_topics = torch.topk(
        routing_scores,
        k=TOPICS_PER_TOKEN,
        dim=1,
    )
    # q_dwk from equation (eq:top2-route).
    q_dwk = nnf.softmax(top_scores, dim=1)

    # r_dk = sum_w x_dw q_dwk (equation eq:document-evidence).  Flattening the
    # document/topic coordinates lets index_add_ accumulate repeated topics
    # without a dense document-by-word-by-topic tensor.
    flattened_r = normalized_bows.new_zeros(document_count * topic_count)
    flattened_indices = (
        document_indices.unsqueeze(1) * topic_count + top_topics
    ).reshape(-1)
    flattened_r.index_add_(
        0,
        flattened_indices,
        (x_dw.unsqueeze(1) * q_dwk).reshape(-1),
    )
    r = flattened_r.reshape(document_count, topic_count)

    # In exact arithmetic each non-empty row already sums to one.  Normalize
    # once more to remove float32 accumulation error, as stated in the report.
    row_masses = r.sum(dim=1, keepdim=True)
    normalized_r = r / row_masses.clamp_min(EPSILON)
    empty_documents = row_masses.squeeze(1) <= 0
    if torch.any(empty_documents):
        normalized_r = normalized_r.clone()
        normalized_r[empty_documents] = 1.0 / topic_count
    return normalized_r


def centered_log_evidence_offset(evidence: torch.Tensor) -> torch.Tensor:
    """Compute posterior offset ``o`` from equation ``eq:posterior-offset``.

    The fixed ``1 / K`` pseudocount keeps every logarithm finite.  Subtracting
    the row mean removes the additive constant that is irrelevant to the final
    simplex mapping and makes uniform evidence an exact zero offset.
    """
    if evidence.ndim != 2 or evidence.shape[1] == 0:
        raise ValueError("evidence must be a non-empty document-topic matrix")
    if not torch.all(torch.isfinite(evidence)) or torch.any(evidence < 0):
        raise ValueError("evidence must be finite and non-negative")
    if not torch.allclose(
        evidence.sum(dim=1),
        torch.ones(evidence.shape[0], device=evidence.device, dtype=evidence.dtype),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("evidence rows must sum to one")
    topic_count = evidence.shape[1]
    log_evidence = torch.log(evidence + 1.0 / topic_count)
    offset = log_evidence - log_evidence.mean(dim=1, keepdim=True)
    # A float32 reduction can leave an equal-valued row at roughly 1e-8 rather
    # than zero.  Enforce the stated exact no-op for uniform evidence.
    uniform_rows = torch.all(evidence == evidence[:, :1], dim=1, keepdim=True)
    return torch.where(uniform_rows, torch.zeros_like(offset), offset)


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL[q(z|x) || N(0,I)]`` per document (equation ``eq:kl``)."""
    if posterior_mean.shape != posterior_log_variance.shape:
        raise ValueError("posterior mean and log variance must have equal shapes")
    return -0.5 * torch.sum(
        1.0
        + posterior_log_variance
        - posterior_mean.square()
        - posterior_log_variance.exp(),
        dim=1,
    )


def reparameterized_gaussian(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    *,
    sample: bool,
) -> torch.Tensor:
    """Return ``z`` from equation ``eq:theta`` for training or inference.

    Training uses one reparameterized standard-normal draw.  Deterministic
    inference sets ``z`` equal to the shifted posterior mean, exactly as stated
    beneath equation ``eq:entmax-closed-form`` in the report.
    """
    if not sample:
        return posterior_mean
    standard_normal = torch.randn_like(posterior_mean)
    posterior_scale = torch.exp(0.5 * posterior_log_variance)
    return posterior_mean + posterior_scale * standard_normal


def entmax15_document_mixture(latent_logits: torch.Tensor) -> torch.Tensor:
    """Map ``z`` to sparse ``theta`` with 1.5-entmax.

    This is equations ``eq:theta`` and ``eq:entmax-closed-form``.
    """
    if latent_logits.ndim != 2 or latent_logits.shape[1] == 0:
        raise ValueError("latent logits must be a non-empty document-topic matrix")
    theta = entmax15(latent_logits, dim=1)
    if not torch.all(torch.isfinite(theta)):
        raise FloatingPointError("1.5-entmax produced non-finite probabilities")
    row_masses = theta.sum(dim=1, keepdim=True)
    if torch.any(row_masses <= 0):
        raise FloatingPointError("1.5-entmax produced a zero-mass row")
    # This correction preserves entmax's exact zeros and ranking while removing
    # only finite-precision simplex error.
    return theta / row_masses


class ContextualSparseETM(nn.Module):
    """Minimal stateful shell around the Contextual Sparse ETM equations.

    PyTorch modules are used only for the learned ETM topic vectors, two-layer
    encoder, Gaussian posterior heads, and scalar context weight.  All decoder
    and posterior mathematics is delegated to the pure tensor functions above.

    The parameter and buffer names intentionally match the frozen experimental
    checkpoint, allowing those weights to be loaded without conversion.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        topics: int,
        fragment_mask: np.ndarray,
        *,
        hidden: int = DEFAULT_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        rho = np.asarray(embeddings, dtype=np.float32)
        if rho.ndim != 2 or not rho.shape[0] or not rho.shape[1]:
            raise ValueError("embeddings must be a non-empty word-feature matrix")
        if not np.all(np.isfinite(rho)):
            raise ValueError("embeddings must be finite")
        if int(topics) < TOPICS_PER_TOKEN or int(hidden) <= 0:
            raise ValueError("topics must be at least two and hidden must be positive")
        rho_norms = np.linalg.norm(rho, axis=1)
        if not np.allclose(rho_norms, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("word embeddings must be row-wise unit normalized")

        mask = np.asarray(fragment_mask, dtype=bool)
        if mask.shape != (rho.shape[0],):
            raise ValueError("fragment_mask must match the ETM vocabulary")
        if not mask.any() or mask.all():
            raise ValueError("fragment_mask must contain fragments and losses")

        self.register_buffer("rho", torch.from_numpy(rho.copy()))
        self.register_buffer(
            "fragment_mask",
            torch.from_numpy(mask.copy()),
            persistent=False,
        )
        self.alphas = nn.Linear(rho.shape[1], int(topics), bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(rho.shape[0], int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
        )
        self.mu = nn.Linear(int(hidden), int(topics))
        self.logvar = nn.Linear(int(hidden), int(topics))
        self.context_scale = nn.Parameter(torch.ones(()))

    def topic_word_distribution(self) -> torch.Tensor:
        """Return ``beta`` from the channel-balanced ETM decoder."""
        return channel_balanced_topic_word_distribution(
            self.rho,
            self.alphas.weight,
            self.fragment_mask,
        )

    def contextual_evidence(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """Return the document-topic evidence matrix ``r``."""
        return contextual_top2_evidence(
            normalized_bows,
            self.rho,
            self.alphas.weight,
            self.context_scale,
        )

    def posterior(
        self,
        normalized_bows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``mu_tilde``, ``log_sigma_squared``, and analytic KL per row.

        This method is the direct executable sequence of equations
        ``eq:encoder`` and ``eq:posterior-offset`` followed by equation
        ``eq:kl``.
        """
        encoded_documents = self.encoder(normalized_bows)
        mu = self.mu(encoded_documents)
        log_sigma_squared = self.logvar(encoded_documents)
        r = self.contextual_evidence(normalized_bows)
        o = centered_log_evidence_offset(r)
        mu_tilde = mu + o
        kl = diagonal_gaussian_kl(mu_tilde, log_sigma_squared)
        return mu_tilde, log_sigma_squared, kl

    def document_topic_mixture(
        self,
        normalized_bows: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse ``theta`` and KL for training or deterministic inference."""
        mu_tilde, log_sigma_squared, kl = self.posterior(normalized_bows)
        z = reparameterized_gaussian(
            mu_tilde,
            log_sigma_squared,
            sample=sample,
        )
        theta = entmax15_document_mixture(z)
        return theta, kl
