"""Focused tests for the Contextual Sparse ETM posterior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scripts.run_contextual_sparse_etm import REAL_METHOD, resolve_training_seed

from benchmarks.neural_ms2lda.chemical import run_chemical_scoring
from benchmarks.neural_ms2lda.routing_etm import (
    ROUTING_VARIANTS,
    RoutingInformedETM,
)
from benchmarks.neural_ms2lda.sparse_etm import (
    BalancedSparseETM,
    dense_normalized,
    sparse_reconstruction_loss,
)


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


def test_real_training_seed_override_preserves_legacy_default() -> None:
    assert resolve_training_seed(42, None) == 7043
    assert resolve_training_seed(42, 23) == 23
    with pytest.raises(ValueError, match="must be non-negative"):
        resolve_training_seed(42, -1)


def test_etm_control_is_exact_balanced_etm() -> None:
    embeddings, fragment_mask, matrix = _inputs()
    torch.manual_seed(19)
    reference = BalancedSparseETM(
        embeddings,
        5,
        fragment_mask,
        theta_transform="softmax",
        hidden=7,
    )
    torch.manual_seed(19)
    candidate = RoutingInformedETM(
        embeddings,
        5,
        fragment_mask,
        routing_variant="etm",
        hidden=7,
    )
    normalized = dense_normalized(
        matrix,
        np.arange(matrix.shape[0]),
        torch.device("cpu"),
    )
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    assert all(
        torch.equal(reference.state_dict()[key], candidate.state_dict()[key])
        for key in reference.state_dict()
    )
    assert torch.equal(reference.beta(), candidate.beta())
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            reference.encode(normalized),
            candidate.encode(normalized),
            strict=True,
        )
    )


@pytest.mark.parametrize("variant", ROUTING_VARIANTS[1:])
def test_routing_evidence_is_a_finite_simplex(variant: str) -> None:
    embeddings, fragment_mask, matrix = _inputs()
    model = RoutingInformedETM(
        embeddings,
        5,
        fragment_mask,
        routing_variant=variant,  # type: ignore[arg-type]
        hidden=7,
    )
    normalized = dense_normalized(
        matrix,
        np.arange(matrix.shape[0]),
        torch.device("cpu"),
    )
    evidence = model.routing_evidence(normalized)
    assert evidence.shape == (2, 5)
    assert torch.all(torch.isfinite(evidence))
    assert torch.all(evidence >= 0)
    assert torch.allclose(evidence.sum(dim=1), torch.ones(2), atol=1e-7)
    if variant == "top2_context":
        nonzero_words = torch.count_nonzero(normalized, dim=1)
        support = torch.count_nonzero(evidence, dim=1)
        assert torch.all(support <= 2 * nonzero_words)
        assert torch.any(evidence == 0)
    else:
        assert torch.all(evidence > 0)


def test_top2_context_routes_one_word_to_exactly_two_topics() -> None:
    embeddings, fragment_mask, _ = _inputs()
    model = RoutingInformedETM(
        embeddings,
        5,
        fragment_mask,
        routing_variant="top2_context",
        hidden=7,
    )
    normalized = torch.zeros((1, len(embeddings)))
    normalized[0, 2] = 1.0
    evidence = model.routing_evidence(normalized)
    assert torch.count_nonzero(evidence).item() == 2
    assert torch.allclose(evidence.sum(dim=1), torch.ones(1), atol=1e-7)


def test_routing_informed_entmax_theta_has_exact_zeros() -> None:
    embeddings, fragment_mask, matrix = _inputs()
    model = RoutingInformedETM(
        embeddings,
        5,
        fragment_mask,
        routing_variant="top2_context",
        theta_transform="entmax15",
        hidden=7,
    )
    with torch.no_grad():
        model.mu.weight.zero_()
        model.mu.bias.copy_(torch.linspace(-8.0, 8.0, 5))
    normalized = dense_normalized(
        matrix,
        np.arange(matrix.shape[0]),
        torch.device("cpu"),
    )
    theta, _ = model.theta(normalized, sample=False)
    assert torch.any(theta == 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(2), atol=1e-7)


@pytest.mark.parametrize("variant", ROUTING_VARIANTS)
def test_routing_etm_has_finite_gradients_and_deterministic_inference(
    variant: str,
) -> None:
    embeddings, fragment_mask, matrix = _inputs()
    rows = np.arange(matrix.shape[0])
    model = RoutingInformedETM(
        embeddings,
        5,
        fragment_mask,
        routing_variant=variant,  # type: ignore[arg-type]
        hidden=7,
    )
    normalized = dense_normalized(matrix, rows, torch.device("cpu"))
    first, _ = model.theta(normalized, sample=False)
    second, _ = model.theta(normalized, sample=False)
    assert torch.equal(first, second)
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
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(value is not None for value in gradients)
    assert all(
        torch.all(torch.isfinite(value)) for value in gradients if value is not None
    )
    assert torch.all(theta >= 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(2), atol=1e-7)


def test_top2_requires_at_least_two_topics() -> None:
    embeddings, fragment_mask, _ = _inputs()
    with pytest.raises(ValueError, match="at least two topics"):
        RoutingInformedETM(
            embeddings,
            1,
            fragment_mask,
            routing_variant="top2_context",
            hidden=7,
        )


def test_routing_etm_method_is_registered_for_chemistry(tmp_path: Path) -> None:
    output = tmp_path / "validation_chemical" / REAL_METHOD
    output.mkdir(parents=True)
    payload = {"method": REAL_METHOD, "split": "validation"}
    (output / "complete.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = run_chemical_scoring(
        tmp_path,
        method=REAL_METHOD,
        data_root=tmp_path,
        protocol={},
        split="validation",
    )

    assert result == payload
