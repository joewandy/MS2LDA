"""Focused mathematical tests for principled sparse-ETM mechanisms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scripts.run_sparse_etm_campaign import REAL_METHOD, REAL_TOPICS

from benchmarks.neural_ms2lda.chemical import run_chemical_scoring
from benchmarks.neural_ms2lda.reproducibility import (
    VALIDATION_DATA_FILES,
    VALIDATION_MAG_INDEX_FILES,
    prepare_validation_view,
)
from benchmarks.neural_ms2lda.sparse_etm import (
    BalancedSparseETM,
    dense_normalized,
    sparse_reconstruction_loss,
    theta_support_diagnostics,
    transform_theta,
)


@pytest.mark.parametrize("name", ["softmax", "entmax15", "sparsemax"])
def test_theta_transforms_are_finite_simplex_mappings(name: str) -> None:
    logits = torch.tensor([[-2.0, 0.0, 0.5], [1.0, 2.0, 3.5]])
    theta = transform_theta(logits, name)  # type: ignore[arg-type]
    assert torch.all(torch.isfinite(theta))
    assert torch.all(theta >= 0)
    assert torch.allclose(theta.sum(dim=1), torch.ones(2), atol=1e-7)
    if name == "softmax":
        assert torch.all(theta > 0)
    else:
        assert torch.any(theta == 0)
    if name == "sparsemax":
        assert torch.allclose(theta[0], torch.tensor([0.0, 0.25, 0.75]))


@pytest.mark.parametrize("name", ["entmax15", "sparsemax"])
def test_sparse_transforms_enforce_large_k_simplex(name: str) -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((7, REAL_TOPICS), generator=generator)
    theta = transform_theta(logits, name)  # type: ignore[arg-type]
    row_sums = theta.sum(dim=1, dtype=torch.float64)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-6)
    assert torch.any(theta == 0)


@pytest.mark.parametrize("name", ["softmax", "entmax15", "sparsemax"])
def test_sparse_etm_has_finite_gradients_and_deterministic_inference(
    name: str,
) -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    matrix = sp.csr_matrix(
        np.asarray([[4, 2, 1, 0], [0, 1, 3, 2]], dtype=np.float32),
    )
    rows = np.asarray([0, 1], dtype=np.int64)
    model = BalancedSparseETM(
        embeddings,
        3,
        np.asarray([True, True, False, False]),
        theta_transform=name,  # type: ignore[arg-type]
        hidden=5,
    )
    normalized = dense_normalized(matrix, rows, torch.device("cpu"))
    first, _ = model.theta(normalized, sample=False)
    second, _ = model.theta(normalized, sample=False)
    assert torch.equal(first, second)
    theta, kl = model.theta(normalized, sample=True)
    reconstruction, _ = sparse_reconstruction_loss(
        theta,
        model.beta(),
        matrix[rows],
        torch.device("cpu"),
        scaling="raw_counts",
    )
    objective = reconstruction + kl.mean()
    objective.backward()
    assert torch.isfinite(objective)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None for value in gradients)
    assert all(
        torch.all(torch.isfinite(value)) for value in gradients if value is not None
    )


def test_distinct_word_scaling_is_invariant_to_pseudocount_multiplier() -> None:
    base = sp.csr_matrix(np.asarray([[2, 1, 0], [0, 3, 1]], dtype=np.float32))
    scaled = base * 100
    theta = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    beta = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    base_distinct, base_mass = sparse_reconstruction_loss(
        theta,
        beta,
        base,
        torch.device("cpu"),
        scaling="distinct_words",
    )
    scaled_distinct, scaled_mass = sparse_reconstruction_loss(
        theta,
        beta,
        scaled,
        torch.device("cpu"),
        scaling="distinct_words",
    )
    base_raw, _ = sparse_reconstruction_loss(
        theta,
        beta,
        base,
        torch.device("cpu"),
        scaling="raw_counts",
    )
    scaled_raw, _ = sparse_reconstruction_loss(
        theta,
        beta,
        scaled,
        torch.device("cpu"),
        scaling="raw_counts",
    )
    assert torch.allclose(base_distinct, scaled_distinct, atol=1e-6)
    assert base_mass == scaled_mass == 2.0
    assert torch.allclose(scaled_raw, 100 * base_raw, atol=1e-4)


def test_theta_support_diagnostics_report_exact_zeros() -> None:
    summary = theta_support_diagnostics(
        np.asarray([[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert summary["minimum_exact_support"] == 1
    assert summary["maximum_exact_support"] == 2
    assert summary["median_exact_support"] == 1.5
    assert summary["fraction_support_le_3"] == 1.0


def test_real_validation_view_exposes_only_declared_inputs(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    real_run = tmp_path / "real"
    (prepared / "data").mkdir(parents=True)
    (prepared / "token_features").mkdir()
    (prepared / "mag/index").mkdir(parents=True)
    (prepared / "protocol.json").write_text(
        json.dumps({"model": {"num_topics": REAL_TOPICS}}),
        encoding="utf-8",
    )
    for name in VALIDATION_DATA_FILES:
        (prepared / "data" / name).write_bytes(name.encode())
    (prepared / "token_features/features.npy").write_bytes(b"features")
    for name in VALIDATION_MAG_INDEX_FILES:
        (prepared / "mag/index" / name).write_bytes(name.encode())
    sentinel = prepared / "data/candidate_test_sentinel.bin"
    sentinel.write_bytes(b"must not be exposed")

    manifest = prepare_validation_view(
        real_run,
        prepared,
        expected_topics=REAL_TOPICS,
    )

    assert manifest["candidate_test_artifacts_accessed"] is False
    assert manifest["candidate_test_metrics_inspected"] is False
    assert not (real_run / "data" / sentinel.name).exists()
    assert all(
        (real_run / "data" / name).is_symlink() for name in VALIDATION_DATA_FILES
    )
    assert (real_run / "token_features/features.npy").is_symlink()
    assert all(
        (real_run / "mag/index" / name).is_symlink()
        for name in VALIDATION_MAG_INDEX_FILES
    )


def test_promoted_sparse_method_is_registered_for_chemistry(tmp_path: Path) -> None:
    output = tmp_path / "validation_chemical" / REAL_METHOD
    output.mkdir(parents=True)
    payload = {"method": REAL_METHOD, "split": "validation"}
    (output / "complete.json").write_text(json.dumps(payload), encoding="utf-8")

    result = run_chemical_scoring(
        tmp_path,
        method=REAL_METHOD,
        data_root=tmp_path,
        protocol={},
        split="validation",
    )

    assert result == payload
