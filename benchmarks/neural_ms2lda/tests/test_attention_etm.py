"""Numerical compatibility and input/decoder separation, not a DreaMS test."""

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.neural_ms2lda.attention_etm import AttentionETM, TokenAttentionEncoder
from benchmarks.neural_ms2lda.contextual_reductions import ReducedContextualETM


def inputs():
    rng = np.random.default_rng(349)
    rho = rng.normal(size=(17, 6)).astype(np.float32)
    rho /= np.linalg.norm(rho, axis=1, keepdims=True)
    x = torch.from_numpy(rng.dirichlet(np.ones(17), 8).astype(np.float32))
    x[0] = 0
    x[1] = 0
    x[1, 0] = 1
    return rho, x, np.arange(17) < 8


def test_import_has_identical_predictions_gradients_and_parameter_count():
    rho, x, mask = inputs()
    old = ReducedContextualETM(rho, 9, mask, hidden=11, variant="reduced_evidence_only")
    with torch.no_grad():
        old.context_scale.fill_(0.63)
        old.global_logvar.copy_(torch.linspace(-1, 0, 9))
    new = AttentionETM.from_evidence_only_state(old.state_dict(), mask)
    old.eval()
    new.eval()
    assert sum(p.numel() for p in new.parameters()) == 9 * 6 + 9 + 1
    assert "fragment_mask" in new.state_dict()
    assert not new.rho.requires_grad
    for a, b in zip(old.posterior(x), new.posterior(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for model in (old, new):
        torch.manual_seed(761)
        theta, kl = model.document_topic_mixture(x, sample=True)
        beta = model.topic_word_distribution()
        (-(x * (theta @ beta).log()).sum() + kl.mean()).backward()
    for name, parameter in old.named_parameters():
        mapped = (
            f"encoder.{name}" if name in {"context_scale", "global_logvar"} else name
        )
        torch.testing.assert_close(
            parameter.grad, dict(new.named_parameters())[mapped].grad, rtol=0, atol=0
        )
    for sample in (False, True):
        torch.manual_seed(6)
        before = old.document_topic_mixture(x, sample=sample)
        torch.manual_seed(6)
        after = new.document_topic_mixture(x, sample=sample)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        old.topic_word_distribution(), new.topic_word_distribution()
    )


class ToyPeakEncoder(nn.Module):
    """A contract test double only: no pretrained model or scientific baseline."""

    def __init__(self, topics):
        super().__init__()
        self.projection = nn.Linear(2, topics)
        self.logvar = nn.Parameter(torch.zeros(topics))

    def forward(self, observations, *, topic_embeddings, word_embeddings):
        # Deliberately neither a BOW input nor a word-embedding operation.
        peaks, mask = observations["peaks"], observations["valid"]
        pooled = (peaks * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        mean = self.projection(pooled)
        return mean, self.logvar.expand_as(mean)


def test_peak_batch_can_replace_encoder_without_changing_decoder_or_core():
    rho, x, mask = inputs()
    replacement = ToyPeakEncoder(9)
    model = AttentionETM(rho, 9, mask, encoder=replacement)
    observations = {"peaks": torch.rand(8, 4, 2), "valid": torch.ones(8, 4)}
    before = model.topic_word_distribution().detach().clone()
    theta, kl = model.document_topic_mixture(observations, sample=True)
    loss = -(x * (theta @ model.topic_word_distribution()).log()).sum() + kl.mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.encoder is replacement
    assert set(dict(model.named_parameters())) == {
        "alphas.weight",
        "encoder.logvar",
        "encoder.projection.weight",
        "encoder.projection.bias",
    }
    torch.testing.assert_close(theta.sum(1), torch.ones(8))
    torch.testing.assert_close(before, model.topic_word_distribution(), rtol=0, atol=0)
    torch.testing.assert_close(
        before[:, model.fragment_mask].sum(1), torch.full((9,), 0.5)
    )


def test_round_trip_preserves_channel_mask_and_frozen_inference():
    rho, x, mask = inputs()
    model = AttentionETM(rho, 9, mask).eval()
    clone = AttentionETM(rho, 9, ~mask).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    saved = {key: value.clone() for key, value in clone.state_dict().items()}
    for a, b in zip(
        model.document_topic_mixture(x, sample=False),
        clone.document_topic_mixture(x, sample=False),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in saved.items():
        torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(
        model.topic_word_distribution(), clone.topic_word_distribution()
    )


def test_import_rejects_mlp_state_instead_of_silently_dropping_parameters():
    rho, _, mask = inputs()
    old = ReducedContextualETM(rho, 9, mask, hidden=11, variant="reduced_shallow")
    with pytest.raises(ValueError, match="exactly an evidence-only"):
        AttentionETM.from_evidence_only_state(old.state_dict(), mask)


def test_encoder_shape_and_finiteness_are_enforced():
    rho, x, mask = inputs()
    with pytest.raises(TypeError, match="nn.Module"):
        AttentionETM(rho, 9, mask, encoder=lambda observations: observations)
    wrong_topics = AttentionETM(rho, 9, mask, encoder=TokenAttentionEncoder(8))
    with pytest.raises(RuntimeError):
        wrong_topics.posterior(x)
    model = AttentionETM(rho, 9, mask)
    with torch.no_grad():
        model.encoder.global_logvar.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.posterior(x)
