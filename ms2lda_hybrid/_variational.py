"""Sparse variational-LDA equations shared by training and benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import digamma, gammaln, polygamma

EPSILON = 1e-12
ALPHA_FLOOR = 1e-9


@dataclass(frozen=True)
class SparseBatch:
    """Padded nonzero words for one batch.

    Word tensors have shape ``batch x positions``; ``totals`` has shape
    ``batch x 1``. The mask distinguishes real entries from padding.
    """

    word_ids: torch.Tensor
    word_counts: torch.Tensor
    word_mask: torch.Tensor
    totals: torch.Tensor


def dirichlet_prior_objective(
    alpha: np.ndarray,
    expected_log_theta_sum: np.ndarray,
    document_count: int,
) -> float:
    """Return the alpha-dependent variational objective for one corpus."""
    values = np.asarray(alpha, dtype=np.float64)
    expected_logs = np.asarray(expected_log_theta_sum, dtype=np.float64)
    if (
        values.ndim != 1
        or expected_logs.shape != values.shape
        or document_count < 1
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(expected_logs))
        or np.any(values <= 0)
    ):
        raise ValueError("invalid Dirichlet alpha objective inputs")
    normalizer = document_count * (gammaln(values.sum()) - float(gammaln(values).sum()))
    return float(normalizer + np.dot(values - 1.0, expected_logs))


def estimate_dirichlet_alpha(
    initial_alpha: np.ndarray,
    expected_log_theta_sum: np.ndarray,
    document_count: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Estimate an asymmetric document-topic prior with guarded Newton steps.

    ``expected_log_theta_sum`` is the training-corpus sum of
    ``E_q[log(theta_d)]``. Candidate Newton steps are backtracked until they
    remain positive and do not reduce the alpha-dependent variational
    objective.
    """
    alpha = np.asarray(initial_alpha, dtype=np.float64).copy()
    expected_logs = np.asarray(expected_log_theta_sum, dtype=np.float64)
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid Dirichlet alpha optimizer settings")
    objective = dirichlet_prior_objective(alpha, expected_logs, document_count)
    for _ in range(max_iterations):
        gradient = (
            document_count * (digamma(alpha.sum()) - digamma(alpha)) + expected_logs
        )
        diagonal = -document_count * polygamma(1, alpha)
        shared = document_count * float(polygamma(1, alpha.sum()))
        correction = float(
            np.sum(gradient / diagonal) / (1.0 / shared + np.sum(1.0 / diagonal))
        )
        step = -(gradient - correction) / diagonal
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("non-finite Dirichlet alpha Newton step")

        scale = 1.0
        accepted = False
        while scale >= 2.0**-30:
            candidate = alpha + scale * step
            if np.all(candidate >= ALPHA_FLOOR):
                candidate_objective = dirichlet_prior_objective(
                    candidate,
                    expected_logs,
                    document_count,
                )
                if candidate_objective >= objective - 1e-10:
                    accepted = True
                    break
            scale *= 0.5
        if not accepted:
            raise FloatingPointError("Dirichlet alpha line search failed")

        relative_change = float(
            np.max(np.abs(candidate - alpha) / np.maximum(alpha, ALPHA_FLOOR))
        )
        alpha = candidate
        objective = candidate_objective
        if relative_change < tolerance:
            break
    if not np.all(np.isfinite(alpha)) or np.any(alpha < ALPHA_FLOOR):
        raise FloatingPointError("invalid optimized Dirichlet alpha")
    return alpha


def make_sparse_batch(
    matrix: sp.csr_matrix,
    indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
) -> SparseBatch:
    """Pad selected CSR rows without constructing a dense vocabulary matrix."""
    subset = matrix[np.asarray(indices, dtype=np.int64)].tocsr()
    lengths = np.diff(subset.indptr)
    width = max(int(lengths.max()) if lengths.size else 0, 1)
    word_ids = np.zeros((subset.shape[0], width), dtype=np.int64)
    word_counts = np.zeros((subset.shape[0], width), dtype=np.float32)
    word_mask = np.zeros((subset.shape[0], width), dtype=bool)
    for row, length in enumerate(lengths):
        if not length:
            continue
        start = subset.indptr[row]
        end = subset.indptr[row + 1]
        word_ids[row, :length] = subset.indices[start:end]
        word_counts[row, :length] = subset.data[start:end]
        word_mask[row, :length] = True
    counts = torch.from_numpy(word_counts).to(device)
    mask = torch.from_numpy(word_mask).to(device)
    return SparseBatch(
        word_ids=torch.from_numpy(word_ids).to(device),
        word_counts=counts,
        word_mask=mask,
        totals=(counts * mask).sum(dim=1, keepdim=True),
    )


def observed_token_nll(
    matrix: sp.csr_matrix,
    theta: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Return mean negative log likelihood per observed token."""
    matrix = matrix.tocsr()
    loss = 0.0
    tokens = 0.0
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        words = matrix.indices[start:end]
        counts = matrix.data[start:end]
        probabilities = theta[row] @ beta[:, words]
        loss -= float(np.sum(counts * np.log(np.clip(probabilities, EPSILON, None))))
        tokens += float(counts.sum())
    return loss / max(tokens, EPSILON)


def expected_log_dirichlet(parameters: torch.Tensor) -> torch.Tensor:
    """Compute ``E[log p]`` for rows of Dirichlet parameters."""
    return torch.digamma(parameters) - torch.digamma(
        parameters.sum(dim=1, keepdim=True)
    )


def responsibilities(
    batch: SparseBatch,
    gamma: torch.Tensor,
    expected_log_beta: torch.Tensor,
) -> torch.Tensor:
    """Return topic responsibilities for the nonzero words in a sparse batch."""
    expected_log_theta = expected_log_dirichlet(gamma)
    word_values = expected_log_beta[:, batch.word_ids].permute(1, 2, 0)
    return torch.softmax(expected_log_theta.unsqueeze(1) + word_values, dim=2)


def local_vb(
    batch: SparseBatch,
    initial_gamma: torch.Tensor,
    alpha: torch.Tensor,
    expected_log_beta: torch.Tensor,
    *,
    steps: int,
    tolerance: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate the two local LDA updates for a batch of documents.

    Each iteration evaluates the topic responsibilities and then applies
    ``gamma[d,k] = alpha[k] + sum_v x[d,v] phi[d,v,k]``. The returned
    responsibilities are recomputed from the final ``gamma``.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    gamma = initial_gamma
    counts = batch.word_counts * batch.word_mask
    for _ in range(steps):
        phi = responsibilities(batch, gamma, expected_log_beta)
        updated = alpha.unsqueeze(0) + (counts.unsqueeze(-1) * phi).sum(dim=1)
        change = ((updated - gamma).abs() / gamma.abs().clamp_min(EPSILON)).amax()
        gamma = updated
        if tolerance is not None and float(change) < tolerance:
            break
    return gamma, responsibilities(batch, gamma, expected_log_beta)


def local_document_elbo(
    batch: SparseBatch,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    expected_log_beta: torch.Tensor,
) -> torch.Tensor:
    """Return the encoder-dependent local LDA ELBO for each document.

    The global topic posterior is fixed. The categorical factor is optimized
    analytically and collapsed into a stable ``logsumexp``.
    """
    expected_shape = (batch.word_ids.shape[0], alpha.numel())
    if tuple(gamma.shape) != expected_shape:
        raise ValueError(f"gamma must have shape {expected_shape}")
    if expected_log_beta.ndim != 2 or expected_log_beta.shape[0] != alpha.numel():
        raise ValueError("expected_log_beta has incompatible topic dimensions")

    expected_log_theta = expected_log_dirichlet(gamma)
    word_values = expected_log_beta[:, batch.word_ids].permute(1, 2, 0)
    logits = expected_log_theta.unsqueeze(1) + word_values
    counts = batch.word_counts * batch.word_mask
    token_bound = (counts * torch.logsumexp(logits, dim=2)).sum(dim=1)
    negative_dirichlet_kl = (
        torch.lgamma(alpha.sum())
        - torch.lgamma(alpha).sum()
        - torch.lgamma(gamma.sum(dim=1))
        + torch.lgamma(gamma).sum(dim=1)
        + ((alpha.unsqueeze(0) - gamma) * expected_log_theta).sum(dim=1)
    )
    return negative_dirichlet_kl + token_bound


def corpus_elbo_minibatch_scale(
    *,
    corpus_documents: int,
    batch_documents: int,
    corpus_tokens: float,
) -> float:
    """Scale a uniform document minibatch to the corpus-per-token objective."""
    if (
        corpus_documents < 1
        or batch_documents < 1
        or not np.isfinite(corpus_tokens)
        or corpus_tokens <= 0
    ):
        raise ValueError("corpus and minibatch sizes must be positive")
    return corpus_documents / (batch_documents * corpus_tokens)


def expected_topic_word_counts(
    batch: SparseBatch,
    phi: torch.Tensor,
    *,
    num_topics: int,
    vocab_size: int,
) -> torch.Tensor:
    """Compute ``sum_d x[d,v] phi[d,v,k]`` without a dense count matrix."""
    statistics = torch.zeros(
        (num_topics, vocab_size),
        device=phi.device,
        dtype=phi.dtype,
    )
    for row in range(batch.word_ids.shape[0]):
        observed = batch.word_mask[row]
        words = batch.word_ids[row, observed]
        weighted_phi = (
            batch.word_counts[row, observed].unsqueeze(1) * phi[row, observed]
        )
        statistics.index_add_(1, words, weighted_phi.transpose(0, 1))
    return statistics
