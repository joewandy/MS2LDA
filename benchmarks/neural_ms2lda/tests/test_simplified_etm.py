"""Independent probability and ablation contracts for the simplified models."""

import numpy as np
import pytest
import torch
from entmax import entmax15
from torch.distributions import Independent, Normal, kl_divergence

from benchmarks.neural_ms2lda.contextual_sparse_etm import ContextualSparseETM
from benchmarks.neural_ms2lda.simplified_etm import (
    BatchNormSparseETM,
    UnbalancedContextualETM,
)


def inputs():
    rng = np.random.default_rng(19)
    rho = rng.normal(size=(16, 6)).astype(np.float32)
    rho /= np.linalg.norm(rho, axis=1, keepdims=True)
    bows = torch.tensor(rng.dirichlet(np.ones(16), 9), dtype=torch.float32)
    return rho, bows, np.arange(16) < 5


@pytest.mark.parametrize("learn_scale", [False, True])
def test_sparse_batchnorm_uses_latent_gaussian_kl_and_frozen_entmax(learn_scale):
    rho, bows, _ = inputs()
    model = BatchNormSparseETM(rho, 7, hidden=12, learn_normalization_scale=learn_scale)
    model.train()
    # Nontrivial stored moments are essential for exercising frozen inference.
    theta, kl = model.document_topic_mixture(bows, sample=True)
    (theta.square().sum() + kl.mean()).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        model.mean_normalization.bias.copy_(torch.linspace(-4, 4, 7))
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    mean, logvar, actual_kl = model.posterior(bows)
    q = Independent(Normal(mean, (0.5 * logvar).exp()), 1)
    p = Independent(Normal(torch.zeros_like(mean), torch.ones_like(mean)), 1)
    torch.testing.assert_close(actual_kl, kl_divergence(q, p))
    actual, _ = model.document_topic_mixture(bows, sample=False)
    expected = entmax15(mean, dim=1)
    expected /= expected.sum(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    assert (actual == 0).any()
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(len(bows)))
    parts = [model.document_topic_mixture(x, sample=False)[0] for x in bows.split(2)]
    torch.testing.assert_close(actual, torch.cat(parts), atol=2e-6, rtol=2e-6)
    for name, value in state_before.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_removing_channel_balance_preserves_posterior_but_restores_etm_decoder():
    rho, bows, mask = inputs()
    torch.manual_seed(10)
    current = ContextualSparseETM(rho, 7, mask, hidden=12)
    candidate = UnbalancedContextualETM(rho, 7, mask, hidden=12)
    candidate.load_state_dict(current.state_dict(), strict=True)
    for actual, expected in zip(candidate.posterior(bows), current.posterior(bows)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_beta = torch.softmax(candidate.alphas.weight @ candidate.rho.T, dim=1)
    torch.testing.assert_close(candidate.topic_word_distribution(), expected_beta)
    torch.testing.assert_close(expected_beta.sum(dim=1), torch.ones(7))
    assert not torch.allclose(expected_beta[:, mask].sum(dim=1), torch.full((7,), 0.5))
