"""Focused tests for the isolated pooled validation candidate."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.data import sparse_batch
from benchmarks.neural_ms2lda.followup import (
    retemperature_theta,
    theta_distribution,
    top_rank_stability,
)
from benchmarks.neural_ms2lda.pooled import (
    assignment_information_loss,
    initialize_pooled_candidate,
)


def _protocol(mi_weight: float = 0.0) -> dict[str, object]:
    return {
        "seed": 42,
        "simple_candidate": {
            "projection_dimensions": 4,
            "theta_temperature": 0.24,
            "beta_temperature": 0.18,
            "mi_weight": mi_weight,
        },
    }


def test_pooled_candidate_is_finite_and_channel_balanced() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.2, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.2, 0.8, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model, _ = initialize_pooled_candidate(features, num_topics=2, protocol=_protocol())
    matrix = sp.csr_matrix(np.asarray([[2, 1, 0, 3], [0, 2, 4, 1]], dtype=np.float32))
    output = model.infer_batch(sparse_batch(matrix, np.arange(2, dtype=np.int64)))
    assert torch.all(torch.isfinite(output.theta))
    assert torch.allclose(output.theta.sum(dim=1), torch.ones(2))
    assert torch.allclose(output.beta[:, :2].sum(dim=1), torch.full((2,), 0.5))
    assert torch.allclose(output.beta[:, 2:].sum(dim=1), torch.full((2,), 0.5))


def test_assignment_information_loss_is_negative_mutual_information() -> None:
    theta = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    value = assignment_information_loss(theta)
    conditional = -torch.sum(theta * torch.log(theta), dim=1).mean()
    marginal = theta.mean(dim=0)
    marginal_entropy = -torch.sum(marginal * torch.log(marginal))
    assert torch.allclose(value, conditional - marginal_entropy)
    assert value < 0


def test_retemperature_preserves_rank_and_sharpens() -> None:
    theta = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]], dtype=np.float32)
    sharpened = retemperature_theta(
        theta,
        source_temperature=0.24,
        target_temperature=0.12,
    )
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert np.all(sharpened.max(axis=1) > theta.max(axis=1))
    assert top_rank_stability(theta, sharpened) == {
        "top1_assignment_stability": 1.0,
        "top3_set_stability": 1.0,
    }
    assert theta_distribution(sharpened)["median_effective_topics_per_spectrum"] < (
        theta_distribution(theta)["median_effective_topics_per_spectrum"]
    )


def test_retemperature_identity_is_numerically_stable() -> None:
    theta = np.asarray([[0.0, 0.0, 1.0], [1e-30, 0.4, 0.6]], dtype=np.float32)
    calibrated = retemperature_theta(
        theta,
        source_temperature=0.24,
        target_temperature=0.24,
    )
    assert np.all(np.isfinite(calibrated))
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(calibrated, axis=1), np.argmax(theta, axis=1))
