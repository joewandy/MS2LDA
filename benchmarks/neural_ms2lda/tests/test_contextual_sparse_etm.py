"""Equation-level tests for the canonical Contextual Sparse ETM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scripts.run_contextual_sparse_etm import (
    _load_validation_data,
    infer_document_topic_mixtures,
)
from scripts.run_msnlib_model_comparison import FragmentLossBalancedETM
from scripts.run_routing_etm_campaign import build_synthetic_model
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
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    raw_count_reconstruction_loss,
)
from benchmarks.neural_ms2lda.utils import write_json


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


def test_fragment_loss_balanced_etm_changes_only_decoder_normalization() -> None:
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


def test_fragment_loss_balanced_etm_rejects_one_channel_vocabulary() -> None:
    with pytest.raises(ValueError, match="fragments and losses"):
        FragmentLossBalancedETM(
            np.eye(3, dtype=np.float32),
            topics=2,
            fragment_mask=np.ones(3, dtype=bool),
            hidden=4,
        )


@pytest.mark.parametrize(
    ("routing_variant", "theta_transform", "is_complete_model"),
    [
        ("etm", "softmax", False),
        ("etm", "entmax15", False),
        ("top2_context", "softmax", False),
        ("top2_context", "entmax15", True),
    ],
)
def test_published_synthetic_formulations_are_finite_and_deterministic(
    routing_variant: str,
    theta_transform: str,
    is_complete_model: bool,
) -> None:
    """Exercise every component comparison reported in the manuscript."""
    embeddings, fragment_mask, counts = _scientific_inputs()
    model = build_synthetic_model(
        embeddings,
        5,
        fragment_mask,
        routing_variant=routing_variant,
        theta_transform=theta_transform,
        reconstruction_scaling="raw_counts",
        hidden=7,
    ).eval()
    x = dense_normalized(counts, np.arange(counts.shape[0]), torch.device("cpu"))

    if isinstance(model, ContextualSparseETM):
        beta = model.topic_word_distribution()
        first_theta, first_kl = model.document_topic_mixture(x, sample=False)
        second_theta, second_kl = model.document_topic_mixture(x, sample=False)
    else:
        beta = model.beta()
        first_theta, first_kl = model.theta(x, sample=False)
        second_theta, second_kl = model.theta(x, sample=False)

    assert isinstance(model, ContextualSparseETM) is is_complete_model
    assert torch.all(torch.isfinite(beta))
    assert torch.allclose(beta.sum(dim=1), torch.ones(5), atol=1e-7)
    assert torch.allclose(first_theta.sum(dim=1), torch.ones(2), atol=1e-7)
    assert torch.equal(first_theta, second_theta)
    assert torch.equal(first_kl, second_kl)


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


def test_training_loader_rejects_any_test_file_in_the_sealed_view(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "validation_input_manifest.json",
        {
            "prepared_run": "/not-opened-by-training",
            "candidate_test_artifacts_accessed": False,
        },
    )
    write_json(tmp_path / "protocol.json", {"model": {"num_topics": 1000}, "seed": 42})
    test_file = tmp_path / "data/test_full.npz"
    test_file.parent.mkdir(parents=True)
    test_file.touch()

    with pytest.raises(RuntimeError, match="sealed training view exposes test files"):
        _load_validation_data(tmp_path)


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
