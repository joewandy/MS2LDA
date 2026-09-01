"""Tests for the paired fragment/loss-balanced ETM decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scripts import run_msnlib_model_comparison as comparison
from scripts.package_gated_etm_temperature import (
    select_nll_preserving_temperature,
    should_add_intermediate,
)
from scripts.run_msnlib_model_comparison import (
    FragmentLossBalancedETM,
    GatedFragmentLossBalancedETM,
    ProbeECR,
    gated_method_name,
    resolve_device,
)
from scripts.run_msnlib_neural_followup import (
    _completion_nll,
    _largest_component_members,
)
from scripts.run_published_topic_models_msnlib import FixedETM


def test_balanced_etm_changes_only_decoder_normalization() -> None:
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


def test_balanced_etm_rejects_one_channel_vocabulary() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    try:
        FragmentLossBalancedETM(
            embeddings,
            topics=2,
            fragment_mask=np.ones(3, dtype=bool),
            hidden=4,
        )
    except ValueError as exc:
        assert "fragments and losses" in str(exc)
    else:
        raise AssertionError("one-channel vocabulary should be rejected")


def _gated_etm(*, gamma: float = 1.0) -> GatedFragmentLossBalancedETM:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    return GatedFragmentLossBalancedETM(
        embeddings,
        topics=3,
        fragment_mask=np.asarray([True, True, False, False]),
        gate_temperature=1.0,
        gate_gamma=gamma,
        hidden=4,
    )


def test_gated_theta_is_finite_nonnegative_and_normalized() -> None:
    model = _gated_etm()
    bows = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]])
    theta, _ = model.theta(bows, sample=False)
    assert torch.all(torch.isfinite(theta))
    assert torch.all(theta >= 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(2))


def test_gate_gamma_zero_recovers_ordinary_etm_theta() -> None:
    model = _gated_etm(gamma=0.0)
    bows = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]])
    expected, expected_kl = FixedETM.theta(model, bows, sample=False)
    actual, actual_kl = model.theta(bows, sample=False)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_kl, expected_kl)


def test_document_gate_is_detached_before_theta_reweighting() -> None:
    model = _gated_etm()
    bows = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    theta = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    assert model.document_gate(bows).requires_grad
    gated = model.apply_document_gate(theta, bows)
    (gated * torch.tensor([[1.0, 2.0, 4.0]])).sum().backward()
    assert theta.grad is not None
    assert torch.any(theta.grad != 0)
    assert model.alphas.weight.grad is None


def test_gated_etm_keeps_exact_channel_balance_and_parameter_count() -> None:
    model = _gated_etm()
    reference = FragmentLossBalancedETM(
        model.rho.detach().numpy(),
        topics=3,
        fragment_mask=np.asarray([True, True, False, False]),
        hidden=4,
    )
    beta = model.beta()
    assert torch.allclose(beta[:, :2].sum(dim=1), torch.full((3,), 0.5))
    assert torch.allclose(beta[:, 2:].sum(dim=1), torch.full((3,), 0.5))
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_gated_etm_inference_is_deterministic_for_identical_inputs() -> None:
    model = _gated_etm().eval()
    bows = torch.tensor([[0.4, 0.1, 0.2, 0.3]])
    first, _ = model.theta(bows.repeat(2, 1), sample=False)
    second, _ = model.theta(bows.repeat(2, 1), sample=False)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])


def test_gated_artifact_name_records_temperature_and_gamma() -> None:
    assert gated_method_name(1.0, 0.5) == "etm_balanced_gated_t1_g0p5"


def test_cuda_device_requires_available_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_temperature_selection_retains_nll_gate_and_maximizes_breadth() -> None:
    rows = [
        {
            "theta_temperature": 0.9,
            "gate_completion_nll": True,
            "evaluable_motifs": 274,
            "useful_motifs": 172,
            "mean_sos": 0.64,
            "gate_useful": False,
            "gate_mean_sos": False,
        },
        {
            "theta_temperature": 0.8,
            "gate_completion_nll": True,
            "evaluable_motifs": 316,
            "useful_motifs": 192,
            "mean_sos": 0.63,
            "gate_useful": False,
            "gate_mean_sos": False,
        },
        {
            "theta_temperature": 0.7,
            "gate_completion_nll": False,
            "evaluable_motifs": 353,
            "useful_motifs": 214,
            "mean_sos": 0.64,
            "gate_useful": False,
            "gate_mean_sos": False,
        },
    ]
    assert select_nll_preserving_temperature(rows) == 0.8
    assert not should_add_intermediate(rows)


def test_etm_campaign_matrix_routing_is_validation_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []

    def fake_load(path: object) -> object:
        opened.append(Path(str(path)).name)
        return object()

    def fake_records(path: object, split: str) -> list[dict[str, str]]:
        opened.append(f"records:{split}")
        return [{"split": split}]

    monkeypatch.setattr(comparison, "load_csr", fake_load)
    monkeypatch.setattr(comparison, "load_heldout_records", fake_records)
    loaded = comparison.load_etm_campaign_data(tmp_path)
    assert set(loaded) == {"train", "observed", "completion", "full", "records"}
    assert opened
    assert all("test" not in path for path in opened)
    assert any(path.endswith(":validation") for path in opened)


def test_followup_reads_locked_completion_metric_field() -> None:
    assert _completion_nll({"nll_per_token": 8.5}) == 8.5


def test_largest_redundancy_component_is_identified() -> None:
    similarity = np.asarray(
        [
            [-1.0, 0.9995, 0.1, 0.2],
            [0.9995, -1.0, 0.9996, 0.1],
            [0.1, 0.9996, -1.0, 0.2],
            [0.2, 0.1, 0.2, -1.0],
        ]
    )
    assert np.array_equal(
        _largest_component_members(similarity, 0.999), np.asarray([0, 1, 2])
    )


def test_canonical_ecr_solver_is_finite_and_differentiable() -> None:
    cost = torch.rand(
        4, 7, generator=torch.Generator().manual_seed(42), requires_grad=True
    )
    solver = ProbeECR(max_iter=1000)
    loss = solver(cost)
    loss.backward()
    assert torch.isfinite(loss)
    assert cost.grad is not None
    assert torch.all(torch.isfinite(cost.grad))
    assert 1 <= solver.iterations_run <= 1000
    assert solver.final_residual is not None


def test_canonical_ecr_solver_rejects_nonfinite_residual() -> None:
    cost = torch.full((4, 7), float("nan"), requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite residual"):
        ProbeECR(max_iter=1000)(cost)
