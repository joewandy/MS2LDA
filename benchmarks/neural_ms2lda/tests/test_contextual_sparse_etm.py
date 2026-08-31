"""Equation-level tests for the canonical Contextual Sparse ETM."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scripts.run_contextual_sparse_etm import infer_document_topic_mixtures
from torch.nn import functional as nnf

from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    EPSILON,
    FRAGMENT_CHANNEL_MASS,
    ROUTING_TEMPERATURE,
    TOPICS_PER_TOKEN,
    ContextualSparseETM,
    centered_log_evidence_offset,
    channel_balanced_topic_word_distribution,
    contextual_top2_evidence,
    diagonal_gaussian_kl,
    entmax15_document_mixture,
)
from benchmarks.neural_ms2lda.routing_etm import RoutingInformedETM
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    raw_count_reconstruction_loss,
)


def _scientific_inputs() -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Return a tiny paired-channel spectrum matrix with unit word vectors."""
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
    counts = sp.csr_matrix(
        np.asarray(
            [
                [4.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    return embeddings, fragment_mask, counts


def _closed_form_entmax15(logits: torch.Tensor) -> torch.Tensor:
    """Evaluate the report's threshold definition independently by bisection."""
    scaled_logits = logits.to(torch.float64) / 2.0
    lower = scaled_logits.min(dim=1, keepdim=True).values - 1.0
    upper = scaled_logits.max(dim=1, keepdim=True).values
    for _ in range(80):
        threshold = (lower + upper) / 2.0
        mass = (
            torch.clamp(scaled_logits - threshold, min=0)
            .square()
            .sum(
                dim=1,
                keepdim=True,
            )
        )
        lower = torch.where(mass > 1.0, threshold, lower)
        upper = torch.where(mass > 1.0, upper, threshold)
    theta = torch.clamp(scaled_logits - upper, min=0).square()
    return (theta / theta.sum(dim=1, keepdim=True)).to(logits.dtype)


def test_channel_balanced_decoder_matches_equation_beta() -> None:
    embeddings, fragment_mask, _ = _scientific_inputs()
    rho = torch.from_numpy(embeddings)
    alpha = torch.tensor(
        [
            [0.4, -0.2, 0.3],
            [-0.1, 0.7, 0.2],
            [0.5, 0.1, -0.6],
        ],
    )
    mask = torch.from_numpy(fragment_mask)

    logits = (rho @ alpha.T).T
    expected_beta = torch.empty_like(logits)
    expected_beta[:, mask] = FRAGMENT_CHANNEL_MASS * torch.softmax(
        logits[:, mask],
        dim=1,
    )
    expected_beta[:, ~mask] = (1.0 - FRAGMENT_CHANNEL_MASS) * torch.softmax(
        logits[:, ~mask],
        dim=1,
    )

    beta = channel_balanced_topic_word_distribution(rho, alpha, mask)

    assert torch.allclose(beta, expected_beta, atol=1e-7)
    assert torch.allclose(beta.sum(dim=1), torch.ones(alpha.shape[0]), atol=1e-7)
    assert torch.allclose(
        beta[:, mask].sum(dim=1),
        torch.full((alpha.shape[0],), FRAGMENT_CHANNEL_MASS),
        atol=1e-7,
    )


def test_contextual_top2_evidence_matches_equations() -> None:
    embeddings, _, counts = _scientific_inputs()
    x = dense_normalized(counts, np.arange(counts.shape[0]), torch.device("cpu"))
    rho = torch.from_numpy(embeddings)
    alpha = torch.tensor(
        [
            [0.7, 0.1, -0.2],
            [0.0, 0.8, 0.2],
            [-0.2, 0.1, 0.9],
            [0.4, -0.6, 0.1],
            [-0.5, 0.5, 0.3],
        ],
    )
    context_scale = torch.tensor(0.35)

    rho_hat = nnf.normalize(rho, dim=1)
    alpha_hat = nnf.normalize(alpha, dim=1)
    expected_r = torch.zeros((x.shape[0], alpha.shape[0]))
    for document_index, document in enumerate(x):
        document_embedding_sum = document @ rho_hat
        for word_index in torch.nonzero(document > 0, as_tuple=False).flatten():
            x_dw = document[word_index]
            rho_bar = (
                document_embedding_sum - x_dw * rho_hat[word_index]
            ) / torch.clamp(1.0 - x_dw, min=EPSILON)
            h_dw = nnf.normalize(
                (rho_hat[word_index] + context_scale * rho_bar).unsqueeze(0),
                dim=1,
            ).squeeze(0)
            scores = (h_dw @ alpha_hat.T) / ROUTING_TEMPERATURE
            top_topics = torch.topk(scores, k=TOPICS_PER_TOKEN).indices
            q_dwk = torch.softmax(scores[top_topics], dim=0)
            expected_r[document_index, top_topics] += x_dw * q_dwk
    expected_r /= expected_r.sum(dim=1, keepdim=True)

    r = contextual_top2_evidence(x, rho, alpha, context_scale)

    assert torch.allclose(r, expected_r, atol=1e-7)
    assert torch.allclose(r.sum(dim=1), torch.ones(x.shape[0]), atol=1e-7)
    assert torch.all(torch.count_nonzero(r, dim=1) <= TOPICS_PER_TOKEN * 4)


def test_posterior_offset_kl_and_entmax_match_report_equations() -> None:
    evidence = torch.tensor(
        [
            [0.70, 0.20, 0.10, 0.00],
            [0.25, 0.25, 0.25, 0.25],
        ],
    )
    base_mu = torch.tensor(
        [
            [-3.0, -1.0, 1.0, 4.0],
            [-2.0, -0.5, 0.5, 2.0],
        ],
    )
    log_sigma_squared = torch.tensor(
        [
            [-0.4, 0.2, -0.1, 0.5],
            [0.1, -0.3, 0.4, -0.2],
        ],
    )

    expected_log_evidence = torch.log(evidence + 1.0 / evidence.shape[1])
    expected_o = expected_log_evidence - expected_log_evidence.mean(
        dim=1,
        keepdim=True,
    )
    o = centered_log_evidence_offset(evidence)
    assert torch.allclose(o, expected_o, atol=1e-7)
    assert torch.equal(o[1], torch.zeros_like(o[1]))

    mu_tilde = base_mu + o
    expected_kl = -0.5 * torch.sum(
        1.0 + log_sigma_squared - mu_tilde.square() - log_sigma_squared.exp(),
        dim=1,
    )
    kl = diagonal_gaussian_kl(mu_tilde, log_sigma_squared)
    assert torch.allclose(kl, expected_kl, atol=1e-7)

    expected_theta = _closed_form_entmax15(mu_tilde)
    theta = entmax15_document_mixture(mu_tilde)
    assert torch.allclose(theta, expected_theta, atol=2e-6)
    assert torch.any(theta == 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(theta.shape[0]), atol=1e-7)


def test_model_objective_matches_report_end_to_end() -> None:
    embeddings, fragment_mask, counts = _scientific_inputs()
    topics = 5
    hidden_width = 7
    torch.manual_seed(29)
    model = ContextualSparseETM(
        embeddings,
        topics,
        fragment_mask,
        hidden=hidden_width,
    )
    with torch.no_grad():
        model.context_scale.fill_(0.35)
        model.mu.bias.copy_(torch.linspace(-4.0, 4.0, topics))

    rows = np.arange(counts.shape[0])
    x = dense_normalized(counts, rows, torch.device("cpu"))
    theta, kl = model.document_topic_mixture(x, sample=False)
    beta = model.topic_word_distribution()
    probabilities = theta @ beta
    dense_counts = torch.from_numpy(counts.toarray().astype(np.float32))
    expected_reconstruction = -(dense_counts * probabilities.log()).sum(dim=1).mean()
    reconstruction, effective_mass = raw_count_reconstruction_loss(
        theta,
        beta,
        counts[rows],
        torch.device("cpu"),
    )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-6)
    assert effective_mass == pytest.approx(float(dense_counts.sum(dim=1).mean()))
    objective = reconstruction + kl.mean()
    objective.backward()
    assert torch.isfinite(objective)
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )

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


def test_frozen_routing_checkpoint_loads_without_conversion() -> None:
    """The cleaned model must preserve the frozen checkpoint tensor contract."""
    embeddings, fragment_mask, counts = _scientific_inputs()
    topics = 5
    hidden_width = 7
    torch.manual_seed(41)
    frozen_model = RoutingInformedETM(
        embeddings,
        topics,
        fragment_mask,
        routing_variant="top2_context",
        theta_transform="entmax15",
        routing_temperature=ROUTING_TEMPERATURE,
        hidden=hidden_width,
    )
    cleaned_model = ContextualSparseETM(
        embeddings,
        topics,
        fragment_mask,
        hidden=hidden_width,
    )
    cleaned_model.load_state_dict(frozen_model.state_dict(), strict=True)
    x = dense_normalized(counts, np.arange(counts.shape[0]), torch.device("cpu"))

    frozen_theta, frozen_kl = frozen_model.theta(x, sample=False)
    cleaned_theta, cleaned_kl = cleaned_model.document_topic_mixture(x, sample=False)

    assert frozen_model.state_dict().keys() == cleaned_model.state_dict().keys()
    assert torch.allclose(
        frozen_model.beta(),
        cleaned_model.topic_word_distribution(),
        atol=1e-7,
    )
    assert torch.allclose(
        frozen_model.routing_evidence(x),
        cleaned_model.contextual_evidence(x),
        atol=2e-7,
    )
    assert torch.allclose(frozen_theta, cleaned_theta, atol=2e-7)
    assert torch.allclose(frozen_kl, cleaned_kl, atol=2e-6)


def test_model_rejects_embeddings_that_violate_the_report_contract() -> None:
    embeddings, fragment_mask, _ = _scientific_inputs()
    embeddings[0] *= 2.0
    with pytest.raises(ValueError, match="unit normalized"):
        ContextualSparseETM(embeddings, 5, fragment_mask, hidden=7)


def test_contextual_evidence_rejects_unnormalized_encoder_rows() -> None:
    embeddings, fragment_mask, _ = _scientific_inputs()
    model = ContextualSparseETM(embeddings, 5, fragment_mask, hidden=7)
    unnormalized_counts = torch.tensor(
        [[2.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError, match="rows must sum to one"):
        model.contextual_evidence(unnormalized_counts)


def test_empty_spectrum_has_explicit_uniform_evidence_fallback() -> None:
    embeddings, fragment_mask, _ = _scientific_inputs()
    model = ContextualSparseETM(embeddings, 5, fragment_mask, hidden=7)
    x = torch.zeros((1, len(embeddings)))

    evidence = model.contextual_evidence(x)

    assert torch.equal(evidence, torch.full((1, 5), 0.2))
    assert torch.equal(
        centered_log_evidence_offset(evidence),
        torch.zeros_like(evidence),
    )


def test_batched_inference_is_the_deterministic_theta_equation() -> None:
    embeddings, fragment_mask, counts = _scientific_inputs()
    model = ContextualSparseETM(embeddings, 5, fragment_mask, hidden=7)
    x = dense_normalized(
        counts,
        np.arange(counts.shape[0]),
        torch.device("cpu"),
    )
    expected_theta, _ = model.document_topic_mixture(x, sample=False)

    inferred_theta, throughput = infer_document_topic_mixtures(
        model,
        counts,
        batch_size=1,
    )

    assert np.array_equal(inferred_theta, expected_theta.detach().numpy())
    assert throughput > 0
