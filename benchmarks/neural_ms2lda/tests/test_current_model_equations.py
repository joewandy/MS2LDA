"""Executable reading examples for the current paper, without fitting any model.

All tests use small CPU tensors or synthetic records. The reference calculation
spells out the selected whole-spectrum equations rather than calling the model's
equation helpers. Tests also distinguish mathematical targets from numerical and
data-preparation conventions that must remain visible in the manuscript.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda import chemical
from benchmarks.neural_ms2lda.contextual_reductions import (
    ReducedContextualETM,
    pooled_evidence,
)
from benchmarks.neural_ms2lda.contextual_sparse_etm import entmax15_document_mixture
from benchmarks.neural_ms2lda.model_evaluation import completion_metrics
from benchmarks.neural_ms2lda.model_inference import infer_document_topics
from benchmarks.neural_ms2lda.spectra import PeakGroup, renormalize_peak_groups
from benchmarks.neural_ms2lda.synthetic_msms import (
    SyntheticPeak,
    _completion_views,
    matched_truth_metrics,
)
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    sparse_reconstruction_loss,
)


def _unit(values):
    """Paper normalization: divide by the Euclidean norm floored at 1e-12."""
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _reference_entmax(scores):
    """Solve sum_k [z_k/2 - tau]_+^2 = 1 by scalar-threshold bisection.

    This deliberately does not call the entmax package. It is a forward-value
    reference, not an alternate differentiable implementation of its threshold.
    """
    scaled = scores / 2
    lower = scaled.min(dim=1, keepdim=True).values - 1
    upper = scaled.max(dim=1, keepdim=True).values
    for _ in range(80):
        threshold = (lower + upper) / 2
        mass = (scaled - threshold).clamp_min(0).square().sum(1, keepdim=True)
        lower = torch.where(mass > 1, threshold, lower)
        upper = torch.where(mass > 1, upper, threshold)
    return (scaled - upper).clamp_min(0).square()


def _selected_fixture():
    """Provide unequal document lengths, both channels, and an empty input row."""
    rng = np.random.default_rng(73)
    rho = rng.normal(size=(6, 3)).astype(np.float32)
    rho /= np.linalg.norm(rho, axis=1, keepdims=True)
    torch.manual_seed(61)
    model = ReducedContextualETM(
        rho,
        topics=5,
        fragment_mask=np.asarray([True, True, True, False, False, False]),
        variant="reduced_document_context",
        hidden=7,
    ).double()
    with torch.no_grad():
        model.mu.bias.copy_(torch.linspace(-2, 2, 5, dtype=torch.float64))
    counts = sp.csr_matrix(
        [[4, 2, 0, 1, 0, 0], [0, 0, 3, 1, 2, 5], [0, 0, 0, 0, 0, 0]],
        dtype=np.float64,
    )
    return model, counts


def _reference_posterior_and_decoder(model, x):
    """Evaluate the paper one spectrum/word at a time using only learned state."""
    rho, alpha = model.rho, model.alphas.weight
    rho_hat, alpha_hat = _unit(rho), _unit(alpha)
    topics = alpha.shape[0]
    evidence_rows = []
    for weights in x:
        if weights.sum() == 0:
            evidence_rows.append(weights.new_full((topics,), 1 / topics))
            continue
        # eq:document-context: the mean includes the word being scored.
        spectrum_mean = (weights[:, None] * rho_hat).sum(0)
        evidence = weights.new_zeros(topics)
        for word in torch.nonzero(weights).flatten():
            direction = _unit(rho_hat[word] + model.context_scale * spectrum_mean)
            scores = alpha_hat @ direction
            selected = scores.argsort(descending=True)[:2]
            # eq:document-evidence: local softmax only on this word's top two.
            local = scores[selected].exp()
            local = local / local.sum()
            evidence = evidence.scatter_add(0, selected, weights[word] * local)
        evidence_rows.append(evidence / evidence.sum())
    r = torch.stack(evidence_rows)
    log_evidence = torch.log1p(topics * r)
    offset = log_evidence - log_evidence.mean(1, keepdim=True)

    # Write the two inherited affine/ReLU layers and Gaussian heads explicitly.
    first, second = model.encoder[0], model.encoder[2]
    hidden = (x @ first.weight.T + first.bias).clamp_min(0)
    hidden = (hidden @ second.weight.T + second.bias).clamp_min(0)
    mean = hidden @ model.mu.weight.T + model.mu.bias + offset
    logvar = hidden @ model.logvar.weight.T + model.logvar.bias

    # eq:beta: raw inner products, word normalization within each channel.
    logits = alpha @ rho.T
    rows = []
    for scores in logits:
        beta = torch.zeros_like(scores)
        for mask in (model.fragment_mask, ~model.fragment_mask):
            numerator = scores[mask].exp()
            beta[mask] = 0.5 * numerator / numerator.sum()
        rows.append(beta)
    return mean, logvar, r, torch.stack(rows)


def test_selected_model_end_to_end_matches_independent_equations(monkeypatch):
    """Check current posterior, both maps, balanced decoder and raw-count loss."""
    model, counts = _selected_fixture()
    raw = torch.from_numpy(counts.toarray())
    x = raw / raw.sum(1, keepdim=True).clamp_min(1)
    expected_mean, expected_logvar, expected_r, expected_beta = (
        _reference_posterior_and_decoder(model, x)
    )
    mean, logvar, kl = model.posterior(x)
    expected_kl = 0.5 * (
        expected_mean.square() + expected_logvar.exp() - 1 - expected_logvar
    ).sum(1)
    torch.testing.assert_close(mean, expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logvar, expected_logvar, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kl, expected_kl, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.contextual_evidence(x), expected_r)

    # Fix epsilon rather than fitting a model or approximating an expectation.
    noise = torch.linspace(-0.7, 0.9, mean.numel(), dtype=mean.dtype).reshape_as(mean)
    monkeypatch.setattr(torch, "randn_like", lambda values: noise)
    sampled_theta, sampled_kl = model.document_topic_mixture(x, sample=True)
    expected_z = expected_mean + (expected_logvar / 2).exp() * noise
    torch.testing.assert_close(
        sampled_theta, _reference_entmax(expected_z), rtol=1e-12, atol=1e-12
    )
    inferred_theta, _ = model.document_topic_mixture(x, sample=False)
    torch.testing.assert_close(
        inferred_theta, _reference_entmax(expected_mean), rtol=1e-12, atol=1e-12
    )
    assert torch.any(inferred_theta == 0)
    assert not torch.allclose(sampled_theta, inferred_theta)
    beta = model.topic_word_distribution()
    torch.testing.assert_close(beta, expected_beta, rtol=1e-12, atol=1e-12)

    # eq:elbo: raw counts inside each sum, then mean over ALL spectrum rows.
    # The empty row has no reconstruction contribution but still has a KL.
    reconstruction, target_mass = sparse_reconstruction_loss(
        sampled_theta, beta, counts, torch.device("cpu"), scaling="raw_counts"
    )
    expected_reconstruction = (
        -(raw * (sampled_theta @ expected_beta).clamp_min(1e-12).log()).sum(1).mean()
    )
    torch.testing.assert_close(reconstruction, expected_reconstruction)
    torch.testing.assert_close(sampled_kl, expected_kl)
    assert target_mass == pytest.approx(float(raw.sum(1).mean()))
    objective = reconstruction + sampled_kl.mean()
    objective.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.context_scale.grad.abs() > 0


def test_entmax_gradient_matches_the_analytic_active_support_jacobian():
    """At a smooth point, J = diag(s) - s*s.T/sum(s), with s = sqrt(theta).

    Differentiate theta_k = [z_k/2 - tau]_+^2 and sum(theta) = 1 to eliminate
    the threshold derivative. Inactive rows/columns and the common-shift
    direction must have zero derivative. This is independent of bisection.
    """
    scores = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    probabilities = entmax15_document_mixture(scores[None])[0]
    support_scale = probabilities.sqrt()
    expected = (
        torch.diag(support_scale)
        - torch.outer(support_scale, support_scale) / support_scale.sum()
    )
    actual = torch.autograd.functional.jacobian(
        lambda values: entmax15_document_mixture(values[None])[0], scores
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual.sum(1), torch.zeros_like(scores), atol=1e-12, rtol=0
    )
    assert probabilities[2] == 0
    assert torch.count_nonzero(actual[2]) == 0


@pytest.mark.parametrize("probability", [0.25, 1e-16, 0.0])
def test_reconstruction_probability_floor_has_the_documented_gradient(probability):
    """The floor caps surprise and stops the gradient only below its boundary."""
    beta = torch.tensor(
        [[probability, 1 - probability]], dtype=torch.float64, requires_grad=True
    )
    theta = torch.ones((1, 1), dtype=torch.float64)
    counts = sp.csr_matrix([[3, 2]], dtype=np.float64)
    actual, _ = sparse_reconstruction_loss(
        theta, beta, counts, torch.device("cpu"), scaling="raw_counts"
    )
    expected = -3 * np.log(max(probability, 1e-12)) - 2 * np.log(1 - probability)
    assert actual.item() == pytest.approx(expected)
    (gradient,) = torch.autograd.grad(actual, beta)
    expected_gradient = -3 / probability if probability > 1e-12 else 0
    assert gradient[0, 0].item() == pytest.approx(expected_gradient)
    assert gradient[0, 1].item() == pytest.approx(-2 / (1 - probability))


def test_zero_observed_mass_remains_eligible_when_withheld_mass_is_positive():
    """A zero encoder row is not a reason to remove its withheld token counts."""
    model, _ = _selected_fixture()
    # The shared batch loader uses float32, as it does for the saved fits.
    model = model.float()
    observed = sp.csr_matrix([[0] * 6, [2, 1, 0, 0, 0, 0]], dtype=np.float32)
    x = dense_normalized(observed, np.arange(2), torch.device("cpu"))
    assert torch.equal(x[0], torch.zeros(6))
    r = model.contextual_evidence(x)
    torch.testing.assert_close(r[0], torch.full((5,), 1 / 5))
    with torch.no_grad():
        hidden = model.encoder(x)
        mean, _, _ = model.posterior(x)
        # Uniform r removes only the contextual offset, not the learned MLP.
        torch.testing.assert_close(mean[0], model.mu(hidden)[0], rtol=0, atol=0)
    theta, _ = infer_document_topics(model, observed, batch_size=2)
    torch.testing.assert_close(
        torch.from_numpy(theta[0]), _reference_entmax(mean[:1].double())[0].float()
    )
    completion = sp.csr_matrix([[8, 0, 0, 0, 0, 0], [0] * 6], dtype=np.float32)
    beta = model.topic_word_distribution().detach().numpy()
    result = completion_metrics(
        theta,
        beta,
        completion,
        [{"completion_oov_tokens": 3}, {"completion_oov_tokens": 2}],
    )
    assert result["eligible_documents"] == 1
    assert result["total_documents"] == 2
    assert result["in_vocabulary_tokens"] == 8
    assert result["out_of_vocabulary_tokens"] == 5
    assert result["nll_per_token"] == pytest.approx(
        -np.log(max(float(theta[0] @ beta[:, 0]), 1e-12))
    )


def test_completion_uses_token_weighting_and_the_same_probability_floor():
    theta = np.ones((2, 1))
    beta = np.asarray([[1e-16, 1 - 1e-16]])
    counts = sp.csr_matrix([[3, 0], [0, 1]])
    result = completion_metrics(theta, beta, counts, [{"completion_oov_tokens": 0}] * 2)
    expected = (-3 * np.log(1e-12) - np.log(1 - 1e-16)) / 4
    assert result["nll_per_token"] == pytest.approx(expected)


def test_real_and_synthetic_observed_count_preparation_are_distinct():
    """Synthetic counts stay fixed; real peaks are re-max-normalized and rounded."""
    document = tuple(
        SyntheticPeak(f"frag@{mz:.2f}", f"loss@{100 - mz:.2f}", count, 0)
        for mz, count in ((10, 20), (20, 40), (30, 100), (40, 60))
    )
    vocabulary = tuple(word for peak in document for word in (peak.fragment, peak.loss))
    observed, held, _ = _completion_views([document], vocabulary, seed=1)
    # This fixed split observes the first two peaks, neither the full maximum.
    np.testing.assert_array_equal(observed.toarray(), [[20, 20, 40, 40, 0, 0, 0, 0]])
    np.testing.assert_array_equal(
        (observed + held).toarray(), [[20, 20, 40, 40, 100, 100, 60, 60]]
    )
    real_groups = tuple(
        PeakGroup(index, float(mz), intensity, ())
        for index, (mz, intensity) in enumerate(((10, 0.2), (20, 0.4)))
    )
    rebuilt = renormalize_peak_groups(
        real_groups, precursor_mz=100, significant_digits=2
    )
    assert [group.intensity for group in rebuilt] == [0.5, 1.0]
    assert [group.tokens.count(f"frag@{group.mz}") for group in rebuilt] == [50, 100]


def _matching_inputs():
    """Two planted words/topics and one genuinely unmatched fitted topic."""
    return {
        "learned_beta": np.eye(3),
        "learned_theta": np.asarray([[0, 0, 1.0], [0.2, 0.8, 0]]),
        "true_beta": np.eye(3)[:2],
        "true_theta": np.asarray([[1.0, 0], [0.2, 0.8]]),
    }


def test_matching_scores_all_planted_topics_and_zero_aligned_mass():
    result = matched_truth_metrics(**_matching_inputs())
    assert result["true_beta_matched_cosine_mean"] == pytest.approx(1)
    assert result["true_theta_cosine_mean"] == pytest.approx(0.5)
    assert result["planted_motifs_recovered_cosine_ge_0_50"] == 2
    assert len(result["matching"]) == 2


@pytest.mark.parametrize("name", tuple(_matching_inputs()))
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
def test_matching_rejects_nonfinite_or_negative_values(name, bad_value):
    values = _matching_inputs()
    values[name][0, 0] = bad_value
    with pytest.raises(ValueError, match="finite nonnegative"):
        matched_truth_metrics(**values)


@pytest.mark.parametrize("name", tuple(_matching_inputs()))
@pytest.mark.parametrize("failure", ["not_matrix", "empty", "zero_mass"])
def test_matching_rejects_undefined_cosine_inputs(name, failure):
    values = _matching_inputs()
    if failure == "not_matrix":
        values[name] = values[name][0]
    elif failure == "empty":
        values[name] = values[name][:0]
    else:
        values[name][0] = 0
    with pytest.raises(ValueError, match="non-empty matrix|positive mass"):
        matched_truth_metrics(**values)


@pytest.mark.parametrize(
    "name,columns,rows,error",
    [
        ("learned_beta", 2, None, "same vocabulary"),
        ("learned_theta", 2, None, "fitted topics"),
        ("true_theta", 1, None, "planted topics"),
        ("true_theta", None, 1, "same spectrum rows"),
    ],
)
def test_matching_rejects_misaligned_shapes(name, columns, rows, error):
    values = _matching_inputs()
    # Use positive matrices so only the deliberately wrong shape is rejected.
    shape = values[name].shape
    values[name] = np.ones((rows or shape[0], columns or shape[1]))
    with pytest.raises(ValueError, match=error):
        matched_truth_metrics(**values)


def test_matching_rejects_fewer_fitted_than_planted_topics():
    values = _matching_inputs()
    values["learned_beta"] = values["learned_beta"][:1]
    values["learned_theta"] = np.ones((2, 1))
    with pytest.raises(ValueError, match="K must be at least planted count R"):
        matched_truth_metrics(**values)


def test_available_empty_consensus_is_not_missing_annotation(monkeypatch):
    monkeypatch.setattr(chemical, "maccs_fingerprint", lambda _: np.asarray([1, 0]))
    monkeypatch.setattr(
        chemical,
        "consensus_fingerprint",
        lambda smiles, _: None if smiles[0] == "missing" else np.zeros(2, dtype=bool),
    )
    result = chemical.score_precomputed_annotations(
        theta=np.eye(2),
        records=[
            {"connectivity_key": "a", "smiles": "a"},
            {"connectivity_key": "b", "smiles": "b"},
        ],
        annotations=[
            {
                "topic_id": 0,
                "optimized_feature_count": 1,
                "clustered_smiles": ["missing"],
            },
            {
                "topic_id": 1,
                "optimized_feature_count": 1,
                "clustered_smiles": ["empty"],
            },
        ],
        fingerprint_threshold=0.8,
    )
    assert result["eligible_topics"] == 1
    assert result["topic_scores"][0]["sos"] is None
    assert result["topic_scores"][1]["sos"] == 0
    assert result["mean_sos"] == 0


def test_top_two_support_bound_excludes_uniform_evidence_for_short_documents():
    """If 2 * observed types < K, nonempty top-2 evidence cannot be uniform."""
    torch.manual_seed(19)
    x = torch.tensor([[0.2, 0, 0.5, 0, 0.3], [0, 0.6, 0, 0.4, 0]])
    r = pooled_evidence(
        x, torch.randn(5, 3), torch.randn(7, 3), torch.tensor(1.0), context="document"
    )
    assert torch.all(torch.count_nonzero(r, dim=1) <= 2 * torch.count_nonzero(x, dim=1))
    assert torch.all(torch.any(r == 0, dim=1))
    assert not torch.any(torch.all(r == r[:, :1], dim=1))
