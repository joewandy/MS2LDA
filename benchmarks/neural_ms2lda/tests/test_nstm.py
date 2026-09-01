"""Focused tests for the reference Neural Sinkhorn Topic Model port."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.neural_ms2lda.nstm import (
    NeuralSinkhornTopicModel,
    prepare_documents,
    reference_sinkhorn_cost,
)


def test_paper_document_distribution_is_scale_invariant() -> None:
    counts = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    original = prepare_documents(counts, "paper_l1")
    scaled = prepare_documents(100 * counts, "paper_l1")
    assert torch.equal(original.encoder_input, original.transport_distribution)
    assert torch.equal(original.encoder_input, original.reconstruction_weights)
    assert torch.allclose(original.encoder_input, scaled.encoder_input)
    assert torch.allclose(original.encoder_input.sum(dim=1), torch.ones(2))


def test_released_code_mode_preserves_reference_count_behaviour() -> None:
    counts = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    prepared = prepare_documents(counts, "released_code")
    assert torch.equal(prepared.encoder_input, counts)
    assert torch.equal(prepared.reconstruction_weights, counts)
    assert torch.allclose(
        prepared.transport_distribution,
        torch.softmax(counts, dim=1),
    )


def test_empty_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_documents(torch.zeros((1, 3)), "paper_l1")


def test_reference_sinkhorn_is_finite_and_differentiable() -> None:
    ground_cost = torch.tensor(
        [[0.1, 0.4, 0.8], [0.7, 0.2, 0.3]],
        requires_grad=True,
    )
    topic_logits = torch.tensor([[0.4, -0.2], [0.1, 0.7]], requires_grad=True)
    topic_mass = torch.softmax(topic_logits, dim=1)
    word_mass = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    result = reference_sinkhorn_cost(
        ground_cost,
        topic_mass,
        word_mass,
        alpha=5.0,
        maximum_iterations=200,
    )
    result.cost.mean().backward()
    assert torch.all(torch.isfinite(result.cost))
    assert torch.all(result.cost >= 0)
    assert 0 < result.iterations <= 200
    assert np.isfinite(result.marginal_error)
    assert ground_cost.grad is not None
    assert topic_logits.grad is not None
    assert torch.all(torch.isfinite(ground_cost.grad))
    assert torch.all(torch.isfinite(topic_logits.grad))


@pytest.mark.parametrize("mode", ["paper_l1", "released_code"])
def test_nstm_has_finite_gradients_and_deterministic_inference(mode: str) -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    counts = torch.tensor([[4.0, 2.0, 1.0, 0.0], [0.0, 1.0, 3.0, 2.0]])
    model = NeuralSinkhornTopicModel(
        embeddings,
        3,
        hidden=5,
        input_mode=mode,  # type: ignore[arg-type]
        sinkhorn_maximum_iterations=200,
    )
    model.eval()
    first = model.theta(counts)
    second = model.theta(counts)
    beta = model.beta()
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(beta.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.all(beta >= 0)

    model.train()
    output = model(counts)
    output.loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert torch.isfinite(output.loss)
    assert all(value is not None for value in gradients)
    assert all(
        torch.all(torch.isfinite(value)) for value in gradients if value is not None
    )
