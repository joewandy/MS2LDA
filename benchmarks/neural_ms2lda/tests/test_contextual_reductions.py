"""Independent identities, paired initialization and gradient checks."""

import numpy as np
import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from benchmarks.neural_ms2lda.contextual_reductions import (
    REDUCTION_VARIANTS,
    ReducedContextualETM,
    linear_evidence_offset,
    pooled_evidence,
)
from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    ContextualSparseETM,
    centered_log_evidence_offset,
    contextual_top2_evidence,
    leave_one_out_context,
    unit_normalize_rows,
)


def inputs():
    rng = np.random.default_rng(333)
    rho = rng.normal(size=(17, 6)).astype(np.float32)
    rho /= np.linalg.norm(rho, axis=1, keepdims=True)
    x = torch.from_numpy(rng.dirichlet(np.ones(17), 8).astype(np.float32))
    x[0] = 0
    x[1] = 0
    x[1, 0] = 1
    return rho, x, np.arange(17) < 8


@pytest.mark.parametrize("variant", REDUCTION_VARIANTS)
def test_reductions_are_finite_neural_gaussians_with_paired_initialization(variant):
    rho, x, mask = inputs()
    torch.manual_seed(51)
    current = ContextualSparseETM(rho, 9, mask, hidden=11)
    torch.manual_seed(51)
    model = ReducedContextualETM(rho, 9, mask, hidden=11, variant=variant)
    for name, parameter in model.named_parameters():
        if name != "global_logvar":
            torch.testing.assert_close(
                parameter, current.state_dict()[name], rtol=0, atol=0
            )
    mean, logvar, kl = model.posterior(x)
    expected = kl_divergence(
        Independent(Normal(mean, (0.5 * logvar).exp()), 1),
        Independent(Normal(torch.zeros_like(mean), torch.ones_like(mean)), 1),
    )
    torch.testing.assert_close(kl, expected)
    theta, kl = model.document_topic_mixture(x, sample=True)
    beta = model.topic_word_distribution()
    (-(x * (theta @ beta).log()).sum() + kl.mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    torch.testing.assert_close(theta.sum(1), torch.ones(len(x)))
    model.eval()
    before = {n: p.clone() for n, p in model.state_dict().items()}
    whole = model.document_topic_mixture(x, sample=False)[0]
    parts = torch.cat(
        [model.document_topic_mixture(t, sample=False)[0] for t in x.split(3)]
    )
    torch.testing.assert_close(whole, parts, rtol=2e-5, atol=2e-6)
    for n, p in before.items():
        torch.testing.assert_close(p, model.state_dict()[n], rtol=0, atol=0)


@pytest.mark.parametrize("context,scale", [("none", 0.0), ("leave_one_out", 0.6)])
def test_pooled_evidence_matches_frozen_definition_and_topic_gradient(context, scale):
    rho, x, _ = inputs()
    rho, x = torch.from_numpy(rho).double(), x.double()
    torch.manual_seed(32)
    alpha = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(scale, dtype=torch.float64)
    actual = pooled_evidence(x, rho, alpha, c, context=context)
    expected = contextual_top2_evidence(x, rho, alpha, c)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(actual)
    grad_actual = torch.autograd.grad(
        (actual * weights).sum(), alpha, retain_graph=True
    )[0]
    grad_expected = torch.autograd.grad((expected * weights).sum(), alpha)[0]
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-10, atol=1e-10)


def test_log_rewrite_and_softmax_multiplicative_identity():
    r = torch.softmax(torch.randn(7, 13, dtype=torch.float64), dim=1)
    offset = centered_log_evidence_offset(r)
    simple = torch.log1p(13 * r)
    simple -= simple.mean(1, keepdim=True)
    torch.testing.assert_close(offset, simple, rtol=1e-12, atol=1e-12)
    mean = torch.randn_like(r)
    multiplicative = mean.exp() * (1 + 13 * r)
    multiplicative /= multiplicative.sum(1, keepdim=True)
    torch.testing.assert_close(torch.softmax(mean + offset, 1), multiplicative)
    # Translation invariance of theta does NOT imply invariance of Gaussian KL.
    raw = torch.log1p(13 * r)
    assert not torch.allclose(raw.square().sum(1), offset.square().sum(1))


@pytest.mark.parametrize("scale", [-0.8, 0.0, 0.6, 2.0])
def test_whole_spectrum_report_equation_and_both_parameter_gradients(scale):
    rho, x, _ = inputs()
    rho, x = torch.from_numpy(rho).double(), x.double()
    torch.manual_seed(32)
    alpha = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(scale, dtype=torch.float64, requires_grad=True)
    actual = pooled_evidence(x, rho, alpha, c, context="document")

    def unit(value):
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    rho_hat, alpha_hat = unit(rho), unit(alpha)
    expected_rows = []
    for weights in x:
        if weights.sum() == 0:
            expected_rows.append(torch.full((9,), 1 / 9, dtype=torch.float64))
            continue
        context = (weights[:, None] * rho_hat).sum(0)
        pooled = torch.zeros(9, dtype=torch.float64)
        for w in torch.nonzero(weights).flatten():
            h = unit(rho_hat[w] + c * context)
            scores = alpha_hat @ h
            indices = scores.argsort(descending=True)[:2]
            local = torch.zeros_like(scores).scatter(
                0, indices, scores[indices].softmax(0)
            )
            pooled = pooled + weights[w] * local
        expected_rows.append(pooled / pooled.sum())
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        (actual * probe).sum(), (alpha, c), retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * probe).sum(), (alpha, c))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", [-0.8, 0.0, 0.6, 2.0])
def test_compact_loo_direction_including_singleton(scale):
    rho, x, _ = inputs()
    rho = unit_normalize_rows(torch.from_numpy(rho).double())
    x = x.double()
    documents, words = torch.nonzero(x, as_tuple=True)
    weights = x[documents, words]
    original = unit_normalize_rows(
        rho[words] + scale * leave_one_out_context(x, rho, documents, words)
    )
    compact = unit_normalize_rows(
        (1 - (1 + scale) * weights[:, None]) * rho[words] + scale * (x @ rho)[documents]
    )
    compact = torch.where((weights == 1)[:, None], rho[words], compact)
    torch.testing.assert_close(original, compact, atol=1e-12, rtol=1e-12)


def test_linear_offset_is_the_first_order_log_expansion():
    deviation = torch.tensor([[1.0, -1.0, 2.0, -2.0]], dtype=torch.float64) * 1e-6
    r = 0.25 + deviation
    torch.testing.assert_close(
        linear_evidence_offset(r),
        centered_log_evidence_offset(r),
        atol=1e-10,
        rtol=1e-5,
    )


def test_fixed_context_has_identical_outputs_but_no_trainable_scalar():
    rho, x, mask = inputs()
    torch.manual_seed(41)
    current = ContextualSparseETM(rho, 9, mask, hidden=11)
    torch.manual_seed(41)
    model = ReducedContextualETM(
        rho, 9, mask, hidden=11, variant="reduced_fixed_context"
    )
    for a, b in zip(current.posterior(x), model.posterior(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert "context_scale" not in dict(model.named_parameters())


def test_narrow_encoder_keeps_the_seeded_decoder_initialization():
    rho, _, mask = inputs()
    torch.manual_seed(333)
    wide = ReducedContextualETM(rho, 9, mask, hidden=80, variant="reduced_shallow")
    torch.manual_seed(333)
    narrow = ReducedContextualETM(
        rho, 9, mask, hidden=10, variant="reduced_shallow_narrow"
    )
    torch.testing.assert_close(wide.alphas.weight, narrow.alphas.weight, atol=0, rtol=0)
    assert narrow.encoder[0].out_features == 10
    assert len(narrow.encoder) == 2
    assert sum(p.numel() for p in narrow.parameters()) < sum(
        p.numel() for p in wide.parameters()
    )


def test_mean_head_gauge_reduction_preserves_sampled_predictions_and_tightens_kl():
    rho, x, mask = inputs()
    model = ContextualSparseETM(rho, 9, mask, hidden=11).double()
    x = x.double()
    mean, logvar, before_kl = model.posterior(x)
    torch.manual_seed(199)
    before_theta = model.document_topic_mixture(x, sample=True)[0]
    with torch.no_grad():
        model.mu.weight -= model.mu.weight.mean(0, keepdim=True)
        model.mu.bias -= model.mu.bias.mean()
    centred_mean, after_logvar, after_kl = model.posterior(x)
    torch.manual_seed(199)
    after_theta = model.document_topic_mixture(x, sample=True)[0]
    torch.testing.assert_close(before_theta, after_theta, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(logvar, after_logvar, atol=0, rtol=0)
    torch.testing.assert_close(centred_mean, mean - mean.mean(1, keepdim=True))
    torch.testing.assert_close(before_kl - after_kl, 4.5 * mean.mean(1).square())
