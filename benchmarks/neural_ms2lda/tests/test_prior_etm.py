"""Independent distribution checks and inference contracts for minimal ETM."""

import io
from copy import deepcopy

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.distributions import Independent, Normal, kl_divergence

from benchmarks.neural_ms2lda.batchnorm_calibration import calibrate_posterior_batchnorm
from benchmarks.neural_ms2lda.etm_baselines import CanonicalETM
from benchmarks.neural_ms2lda.model_variants import build_model
from benchmarks.neural_ms2lda.prior_etm import (
    PriorETM,
    isotropic_gaussian_kl,
    symmetric_dirichlet_logit_variance,
)
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    training_batches,
)


def embeddings():
    values = np.random.default_rng(13).normal(size=(12, 6)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_prior_matches_general_laplace_formula():
    alpha = np.full(36, 0.02)
    general = (1 - 2 / len(alpha)) / alpha + np.sum(1 / alpha) / len(alpha) ** 2
    np.testing.assert_allclose(symmetric_dirichlet_logit_variance(36, 0.02), general)


@pytest.mark.parametrize("concentration", [0, -1, float("nan"), float("inf")])
def test_invalid_prior_rejected(concentration):
    with pytest.raises(ValueError):
        symmetric_dirichlet_logit_variance(36, concentration)


@pytest.mark.parametrize("prior_variance", [1.0, 48.611111])
def test_kl_matches_torch_distributions(prior_variance):
    torch.manual_seed(10)
    mean = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    log_variance = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    variance = torch.tensor(prior_variance, dtype=torch.float64)
    posterior = Independent(Normal(mean, (0.5 * log_variance).exp()), 1)
    prior = Independent(
        Normal(torch.zeros_like(mean), variance.sqrt().expand_as(mean)), 1
    )
    actual = isotropic_gaussian_kl(mean, log_variance, variance)
    torch.testing.assert_close(actual, kl_divergence(posterior, prior))
    actual.sum().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(log_variance.grad).all()
    zero = isotropic_gaussian_kl(
        torch.zeros_like(mean), variance.log().expand_as(mean), variance
    )
    torch.testing.assert_close(zero, torch.zeros(5, dtype=torch.float64))


def test_disabling_additions_recovers_original_etm():
    torch.manual_seed(23)
    base = CanonicalETM(embeddings(), 4, hidden=8).eval()
    model = PriorETM(
        embeddings(), 4, hidden=8, concentration=None, batch_normalize=False
    ).eval()
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    bows = torch.softmax(torch.randn(3, 12), dim=1)
    for actual, expected in zip(
        model.document_topic_mixture(bows, sample=False),
        base.document_topic_mixture(bows, sample=False),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        model.topic_word_distribution(), base.topic_word_distribution()
    )


def test_batchnorm_checkpoint_and_batch_independent_validation():
    torch.manual_seed(37)
    model = PriorETM(embeddings(), 4, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(4):
        bows = torch.softmax(torch.randn(8, 12), dim=1)
        theta, kl = model.document_topic_mixture(bows, sample=True)
        prediction = theta @ model.topic_word_distribution()
        loss = -(bows * prediction.log()).sum(dim=1).mean() + kl.mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        optimizer.step()
    model.eval()
    before = model.mean_normalization.running_mean.clone()
    theta, _ = model.document_topic_mixture(bows, sample=False)
    separate = torch.cat(
        [model.document_topic_mixture(row[None], sample=False)[0] for row in bows]
    )
    torch.testing.assert_close(theta, separate)
    torch.testing.assert_close(theta.sum(dim=1), torch.ones(8))
    torch.testing.assert_close(model.mean_normalization.running_mean, before)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = PriorETM(embeddings(), 4, hidden=8).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(
        theta, restored.document_topic_mixture(bows, sample=False)[0]
    )


@pytest.mark.parametrize("variant", ["batchnorm_fixed", "prior_batchnorm_fixed"])
def test_published_fixed_scale_normalization_cannot_shrink_away(variant):
    model = build_model(
        variant,
        embeddings(),
        [f"frag@{i}" for i in range(12)],
        4,
        hidden=8,
        concentration=0.02,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bows = torch.softmax(torch.randn(8, 12), dim=1)
    theta, kl = model.document_topic_mixture(bows, sample=True)
    loss = -(bows * (theta @ model.topic_word_distribution()).log()).sum() + kl.sum()
    loss.backward()
    optimizer.step()
    for layer in (model.mean_normalization, model.variance_normalization):
        assert not layer.weight.requires_grad and layer.weight.grad is None
        torch.testing.assert_close(layer.weight, torch.ones(4))
        assert layer.bias.grad is not None
        assert torch.isfinite(layer.bias.grad).all()
    assert torch.isfinite(model.mu.weight.grad).all()


def test_calibration_matches_full_training_moments_without_changing_weights():
    model = PriorETM(embeddings(), 4, hidden=8, learn_normalization_scale=False)
    with torch.no_grad():
        model.mu.bias.add_(100.0)
    train = sp.csr_matrix(np.random.default_rng(7).uniform(0, 4, size=(11, 12)))
    original_parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    alternate = deepcopy(model)
    device = torch.device("cpu")
    with torch.no_grad():
        encoded = model.encoder(dense_normalized(train, np.arange(11), device))
        raw_heads = (model.mu(encoded), model.logvar(encoded))
    report = calibrate_posterior_batchnorm(model, train, batch_size=3, device=device)
    calibrate_posterior_batchnorm(alternate, train, batch_size=7, device=device)
    assert report["training_rows"] == 11 and report["validation_rows_used"] == 0
    assert report["test_rows_used"] == 0 and not model.training
    for layer, other, raw in zip(
        (model.mean_normalization, model.variance_normalization),
        (alternate.mean_normalization, alternate.variance_normalization),
        raw_heads,
        strict=True,
    ):
        torch.testing.assert_close(layer.running_mean, raw.double().mean(0).float())
        torch.testing.assert_close(
            layer.running_var, raw.double().var(0).float(), rtol=1e-3, atol=1e-8
        )
        torch.testing.assert_close(layer.running_mean, other.running_mean)
        torch.testing.assert_close(
            layer.running_var, other.running_var, rtol=1e-3, atol=1e-8
        )
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original_parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "size,batch_size", [(2, 200), (201, 200), (401, 200), (8, 3), (9, 3)]
)
def test_batches_keep_every_row_without_singletons(size, batch_size):
    order = np.random.default_rng(3).permutation(size)
    batches = training_batches(order, batch_size)
    np.testing.assert_array_equal(np.concatenate(batches), order)
    assert all(len(batch) >= 2 for batch in batches)
