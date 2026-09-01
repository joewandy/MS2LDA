"""Focused tests for the zero-parameter top-2 token ablation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.sparse_etm import (
    BalancedSparseETM,
    dense_normalized,
    sparse_reconstruction_loss,
)
from benchmarks.neural_ms2lda.top2_token_etm import Top2TokenETM


def _inputs() -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.2, 0.8],
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    fragment_mask = np.asarray([True, True, True, False, False, False])
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [4.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    return embeddings, fragment_mask, matrix


def test_top2_token_adds_no_parameters_or_state() -> None:
    embeddings, fragment_mask, _ = _inputs()
    torch.manual_seed(29)
    reference = BalancedSparseETM(
        embeddings,
        5,
        fragment_mask,
        theta_transform="entmax15",
        hidden=7,
    )
    torch.manual_seed(29)
    candidate = Top2TokenETM(
        embeddings,
        5,
        fragment_mask,
        theta_transform="entmax15",
        hidden=7,
    )
    assert candidate.context_scale is None
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    assert all(
        torch.equal(reference.state_dict()[key], candidate.state_dict()[key])
        for key in reference.state_dict()
    )
    assert sum(parameter.numel() for parameter in reference.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )


def test_top2_token_evidence_is_sparse_finite_simplex() -> None:
    embeddings, fragment_mask, matrix = _inputs()
    model = Top2TokenETM(embeddings, 5, fragment_mask, hidden=7)
    normalized = dense_normalized(matrix, np.arange(2), torch.device("cpu"))
    evidence = model.routing_evidence(normalized)
    nonzero_words = torch.count_nonzero(normalized, dim=1)
    assert torch.all(torch.isfinite(evidence))
    assert torch.all(evidence >= 0)
    assert torch.allclose(evidence.sum(dim=1), torch.ones(2), atol=1e-7)
    assert torch.all(torch.count_nonzero(evidence, dim=1) <= 2 * nonzero_words)
    assert torch.any(evidence == 0)


def test_top2_token_entmax_has_finite_gradients_and_exact_zeros() -> None:
    embeddings, fragment_mask, matrix = _inputs()
    model = Top2TokenETM(
        embeddings,
        5,
        fragment_mask,
        theta_transform="entmax15",
        hidden=7,
    )
    rows = np.arange(2)
    normalized = dense_normalized(matrix, rows, torch.device("cpu"))
    theta, kl = model.theta(normalized, sample=True)
    reconstruction, _ = sparse_reconstruction_loss(
        theta,
        model.beta(),
        matrix[rows],
        torch.device("cpu"),
        scaling="raw_counts",
    )
    objective = reconstruction + kl.mean()
    objective.backward()
    assert torch.isfinite(objective)
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    assert torch.any(theta == 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(2), atol=1e-7)


def test_top2_token_requires_two_topics() -> None:
    embeddings, fragment_mask, _ = _inputs()
    with pytest.raises(ValueError, match="at least two topics"):
        Top2TokenETM(embeddings, 1, fragment_mask, hidden=7)
