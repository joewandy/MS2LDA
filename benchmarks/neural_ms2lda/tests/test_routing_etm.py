"""Focused tests for the Contextual Sparse ETM posterior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scripts.run_routing_etm_real import REAL_METHOD, resolve_training_seed
from torch.nn import functional as nnf

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


def _closed_form_entmax15(logits: torch.Tensor) -> torch.Tensor:
    """Evaluate the report's threshold definition independently by bisection."""
    scaled = logits.to(torch.float64) / 2.0
    lower = scaled.min(dim=1, keepdim=True).values - 1.0
    upper = scaled.max(dim=1, keepdim=True).values
    for _ in range(80):
        threshold = (lower + upper) / 2.0
        mass = (
            torch.clamp(scaled - threshold, min=0)
            .square()
            .sum(
                dim=1,
                keepdim=True,
            )
        )
        lower = torch.where(mass > 1.0, threshold, lower)
        upper = torch.where(mass > 1.0, upper, threshold)
    values = torch.clamp(scaled - upper, min=0).square()
    return (values / values.sum(dim=1, keepdim=True)).to(logits.dtype)


def test_reported_contextual_sparse_etm_equations_match_code() -> None:
    embeddings, fragment_mask, matrix = _inputs()
    topics = 5
    hidden_width = 7
    torch.manual_seed(29)
    model = RoutingInformedETM(
        embeddings,
        topics,
        fragment_mask,
        routing_variant="top2_context",
        theta_transform="entmax15",
        hidden=hidden_width,
    )
    with torch.no_grad():
        model.context_scale.fill_(0.35)
        model.mu.bias.copy_(torch.linspace(-4.0, 4.0, topics))
    rows = np.arange(matrix.shape[0])
    normalized = dense_normalized(matrix, rows, torch.device("cpu"))

    logits = (model.rho @ model.alphas.weight.T).T
    expected_beta = torch.empty_like(logits)
    mask = model.fragment_mask
    expected_beta[:, mask] = 0.5 * torch.softmax(logits[:, mask], dim=1)
    expected_beta[:, ~mask] = 0.5 * torch.softmax(logits[:, ~mask], dim=1)
    assert torch.allclose(model.beta(), expected_beta, atol=1e-7)
    assert torch.allclose(expected_beta.sum(dim=1), torch.ones(topics), atol=1e-7)

    rho = nnf.normalize(model.rho, dim=1)
    alpha = nnf.normalize(model.alphas.weight, dim=1)
    expected_evidence = torch.zeros((len(rows), topics), dtype=normalized.dtype)
    for document, bow in enumerate(normalized):
        document_sum = bow @ rho
        for word in torch.nonzero(bow > 0, as_tuple=False).flatten():
            weight = bow[word]
            context = (document_sum - weight * rho[word]) / torch.clamp(
                1.0 - weight,
                min=1e-12,
            )
            route = nnf.normalize(
                (rho[word] + model.context_scale * context).unsqueeze(0),
                dim=1,
            ).squeeze(0)
            scores = route @ alpha.T
            selected = torch.topk(scores, k=2).indices
            local = torch.softmax(scores[selected], dim=0)
            expected_evidence[document, selected] += weight * local
    expected_evidence /= expected_evidence.sum(dim=1, keepdim=True)
    actual_evidence = model.routing_evidence(normalized)
    assert torch.allclose(actual_evidence, expected_evidence, atol=1e-7)

    encoder_hidden = model.encoder(normalized)
    base_mu = model.mu(encoder_hidden)
    expected_logvar = model.logvar(encoder_hidden)
    offset = torch.log(expected_evidence + 1.0 / topics)
    offset -= offset.mean(dim=1, keepdim=True)
    expected_mu = base_mu + offset
    expected_kl = -0.5 * torch.sum(
        1.0 + expected_logvar - expected_mu.square() - expected_logvar.exp(),
        dim=1,
    )
    actual_mu, actual_logvar, actual_kl = model.encode(normalized)
    assert torch.allclose(actual_mu, expected_mu, atol=1e-7)
    assert torch.allclose(actual_logvar, expected_logvar, atol=1e-7)
    assert torch.allclose(actual_kl, expected_kl, atol=1e-6)

    expected_theta = _closed_form_entmax15(expected_mu)
    actual_theta, _ = model.theta(normalized, sample=False)
    assert torch.allclose(actual_theta, expected_theta, atol=2e-6)
    assert torch.any(actual_theta == 0)

    counts = torch.from_numpy(matrix.toarray().astype(np.float32))
    probabilities = actual_theta @ expected_beta
    expected_reconstruction = -(counts * probabilities.log()).sum(dim=1).mean()
    actual_reconstruction, effective_mass = sparse_reconstruction_loss(
        actual_theta,
        model.beta(),
        matrix,
        rows,
        torch.device("cpu"),
        scaling="raw_counts",
    )
    assert torch.allclose(actual_reconstruction, expected_reconstruction, atol=1e-6)
    assert effective_mass == pytest.approx(float(counts.sum(dim=1).mean()))

    vocabulary, dimensions = embeddings.shape
    expected_parameters = (
        vocabulary * hidden_width
        + hidden_width
        + hidden_width * hidden_width
        + hidden_width
        + 2 * (topics * hidden_width + topics)
        + topics * dimensions
        + 1
    )
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected_parameters
    )


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
        matrix,
        rows,
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
