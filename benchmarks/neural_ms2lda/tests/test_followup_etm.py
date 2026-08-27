"""Tests for the paired fragment/loss-balanced ETM decoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.run_msnlib_model_comparison import FragmentLossBalancedETM, ProbeECR
from scripts.run_msnlib_neural_followup import (
    _completion_nll,
    _largest_component_members,
)


def test_balanced_etm_changes_only_decoder_normalization() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    model = FragmentLossBalancedETM(
        embeddings,
        topics=3,
        fragment_mask=np.asarray([True, True, False, False]),
        hidden=4,
    )
    beta = model.beta()
    assert beta.shape == (3, 4)
    assert torch.allclose(beta.sum(dim=1), torch.ones(3))
    assert torch.allclose(beta[:, :2].sum(dim=1), torch.full((3,), 0.5))
    assert torch.allclose(beta[:, 2:].sum(dim=1), torch.full((3,), 0.5))


def test_balanced_etm_rejects_one_channel_vocabulary() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    try:
        FragmentLossBalancedETM(
            embeddings,
            topics=2,
            fragment_mask=np.ones(3, dtype=bool),
            hidden=4,
        )
    except ValueError as exc:
        assert "fragments and losses" in str(exc)
    else:
        raise AssertionError("one-channel vocabulary should be rejected")


def test_followup_reads_locked_completion_metric_field() -> None:
    assert _completion_nll({"nll_per_token": 8.5}) == 8.5


def test_largest_redundancy_component_is_identified() -> None:
    similarity = np.asarray(
        [
            [-1.0, 0.9995, 0.1, 0.2],
            [0.9995, -1.0, 0.9996, 0.1],
            [0.1, 0.9996, -1.0, 0.2],
            [0.2, 0.1, 0.2, -1.0],
        ]
    )
    assert np.array_equal(
        _largest_component_members(similarity, 0.999), np.asarray([0, 1, 2])
    )


def test_canonical_ecr_solver_is_finite_and_differentiable() -> None:
    cost = torch.rand(
        4, 7, generator=torch.Generator().manual_seed(42), requires_grad=True
    )
    solver = ProbeECR(max_iter=1000)
    loss = solver(cost)
    loss.backward()
    assert torch.isfinite(loss)
    assert cost.grad is not None
    assert torch.all(torch.isfinite(cost.grad))
    assert 1 <= solver.iterations_run <= 1000
    assert solver.final_residual is not None


def test_canonical_ecr_solver_rejects_nonfinite_residual() -> None:
    cost = torch.full((4, 7), float("nan"), requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite residual"):
        ProbeECR(max_iter=1000)(cost)
