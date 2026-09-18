"""Scientific checks for the broader, validation-only model-family review."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from entmax import entmax15
from torch.distributions import Dirichlet, Normal, kl_divergence

from benchmarks.neural_ms2lda.model_evaluation import completion_metrics
from benchmarks.neural_ms2lda.published_evaluation import (
    infer_published,
    published_completion,
)
from benchmarks.neural_ms2lda.published_models import (
    VARIANTS,
    DirichletEncoder,
    LogisticNormalEncoder,
    build_published_model,
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_valid_probabilities_finite_gradients_and_frozen_inference(variant):
    torch.manual_seed(41)
    embeddings = np.random.default_rng(51).normal(size=(13, 5)).astype("float32")
    model = build_published_model(variant, embeddings, 4, hidden=7)
    counts = torch.randint(0, 7, (8, 13)).float()
    logp, theta, kl = model(counts, sample=True)
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(8))
    torch.testing.assert_close(theta.sum(1), torch.ones(8))
    loss = -(counts * logp).sum(1).mean() + kl.mean()
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    model.eval()
    before = {name: value.clone() for name, value in model.named_buffers()}
    matrix = sp.csr_matrix(counts.numpy())
    first = infer_published(model, matrix, batch_size=8, device="cpu")
    split = infer_published(model, matrix, batch_size=3, device="cpu")
    np.testing.assert_allclose(first, split, rtol=1e-5, atol=1e-7)
    prototypes = model.topic_prototypes()
    torch.testing.assert_close(prototypes.sum(1), torch.ones(4))
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_gaussian_encoder_matches_tutorial_math():
    encoder = LogisticNormalEncoder(11, 5, hidden=7).eval()
    counts = torch.rand(6, 11)
    theta, kl = encoder(counts, sample=False)
    hidden = encoder.hidden(counts)
    mean = encoder.mean_bn(encoder.mean(hidden))
    logvar = encoder.logvar_bn(encoder.logvar(hidden))
    expected = kl_divergence(Normal(mean, (0.5 * logvar).exp()), Normal(0, 1)).sum(1)
    torch.testing.assert_close(kl, expected)
    torch.testing.assert_close(theta, mean.softmax(1))
    assert encoder.mean_bn.affine is False


def test_sparse_extension_changes_only_latent_link():
    embeddings = np.ones((13, 5), dtype="float32")
    reference = build_published_model("prodlda", embeddings, 8, hidden=7).eval()
    sparse = build_published_model("prodlda_entmax", embeddings, 8, hidden=7).eval()
    sparse.load_state_dict(reference.state_dict(), strict=True)
    counts = torch.rand(6, 13) * 20
    theta, kl = sparse.encoder(counts, sample=False)
    _, expected_kl = reference.encoder(counts, sample=False)
    hidden = reference.encoder.hidden(counts)
    mean = reference.encoder.mean_bn(reference.encoder.mean(hidden))
    torch.testing.assert_close(theta, entmax15(mean, dim=1))
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(sparse.topic_prototypes(), reference.topic_prototypes())


def test_dirichlet_mean_and_independent_analytic_kl():
    encoder = DirichletEncoder(9, 4, hidden=6).double().eval()
    counts = torch.rand(5, 9).double()
    posterior = encoder.posterior(counts)
    a, b = posterior.concentration, encoder.prior
    theta, kl = encoder(counts, sample=False)
    expected = (
        torch.lgamma(a.sum(-1))
        - torch.lgamma(a).sum(-1)
        - torch.lgamma(b.sum())
        + torch.lgamma(b).sum()
        + ((a - b) * (torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True)))).sum(
            -1
        )
    )
    torch.testing.assert_close(kl, expected)
    torch.testing.assert_close(theta, a / a.sum(-1, keepdim=True))
    assert encoder.head_bn.weight.requires_grad is False
    assert encoder.head_bn.bias.requires_grad is True
    # The direct sampler, including small concentrations, has pathwise gradients.
    small = torch.full((3, 4), 0.02, dtype=torch.float64, requires_grad=True)
    Dirichlet(small).rsample().square().sum().backward()
    assert torch.isfinite(small.grad).all()


@pytest.mark.parametrize("variant", ("dirichlet_lda", "dirichlet_etm"))
def test_additive_completion_matches_existing_evaluator(variant):
    model = build_published_model(
        variant, np.ones((6, 3), "float32"), 3, hidden=4
    ).eval()
    theta = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype="float32")
    counts = sp.csr_matrix([[1, 0, 3, 0, 1, 0], [2, 3, 0, 0, 0, 1]])
    records = [{"completion_oov_tokens": 2}, {"completion_oov_tokens": 0}]
    actual = published_completion(
        model, theta, counts, records, batch_size=2, device="cpu"
    )
    expected = completion_metrics(
        theta, model.topic_prototypes().numpy(), counts, records
    )
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=2e-7)


def test_product_decoder_export_is_conditional_but_not_an_additive_emission():
    model = build_published_model(
        "prodlda", np.ones((7, 3), "float32"), 3, hidden=4
    ).eval()
    with torch.no_grad():
        model.decoder.weights.weight.mul_(10)
        model.decoder.normalization.running_mean.copy_(torch.linspace(-1, 1, 7))
        model.decoder.normalization.running_var.copy_(torch.linspace(0.2, 2, 7))
    beta = model.topic_prototypes()
    torch.testing.assert_close(beta, model.decoder(torch.eye(3)).exp())
    theta = torch.tensor([[0.2, 0.3, 0.5]])
    probability = model.decoder(theta).exp()
    geometric = (theta @ beta.log()).softmax(1)
    torch.testing.assert_close(probability, geometric)
    assert not torch.allclose(probability, theta @ beta, atol=1e-3)
    counts = sp.csr_matrix([[1, 2, 1, 3, 0, 2, 4]])
    records = [{"completion_oov_tokens": 0}]
    actual = published_completion(
        model, theta.numpy(), counts, records, batch_size=1, device="cpu"
    )
    expected = (
        -(torch.as_tensor(counts.toarray()) * probability.log()).sum() / counts.sum()
    )
    assert actual["nll_per_token"] == pytest.approx(float(expected.detach()), rel=1e-6)


def test_invalid_variant_and_training_mode_export_rejected():
    with pytest.raises(ValueError, match="unknown"):
        build_published_model("unknown", np.ones((4, 2)), 3)
    model = build_published_model("prodlda", np.ones((4, 2)), 3)
    with pytest.raises(ValueError, match="frozen"):
        model.topic_prototypes()
