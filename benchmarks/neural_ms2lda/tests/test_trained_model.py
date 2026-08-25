"""Numerical regression tests for the published seed-42 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.artifacts import load_trained_model
from benchmarks.neural_ms2lda.data import sparse_batch
from benchmarks.neural_ms2lda.utils import read_json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PACKAGE_ROOT / "results/seed42"
MODEL_ROOT = RESULTS_ROOT / "trained_model"


def _reference_batch(vocabulary_size: int) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray([1.0, 2.0, 1.0, 3.0, 1.0], dtype=np.float32),
            np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
            np.asarray([0, 2, 5, 5], dtype=np.int32),
        ),
        shape=(3, vocabulary_size),
        dtype=np.float32,
    )


def test_published_model_reproduces_reference_probabilities() -> None:
    """Lock support and probabilities across compatible numerical backends."""
    model, vocabulary, temperature = load_trained_model(MODEL_ROOT)
    batch = sparse_batch(
        _reference_batch(len(vocabulary)), np.arange(3, dtype=np.int64)
    )
    with torch.inference_mode():
        beta = model.topic_word_distribution().cpu().numpy()
        theta = (
            model.route(
                batch,
                temperature=temperature,
                straight_through=False,
            )
            .theta.cpu()
            .numpy()
        )

    expected_indices = (
        np.asarray([55, 473, 620, 661]),
        np.asarray([50, 83, 293, 564, 825, 828]),
    )
    expected_values = (
        np.asarray(
            [0.10213348, 0.022363424, 0.69833755, 0.17716552],
            dtype=np.float32,
        ),
        np.asarray(
            [
                0.1227046,
                0.052162364,
                0.20571087,
                0.22919762,
                0.046178792,
                0.34404582,
            ],
            dtype=np.float32,
        ),
    )
    for row, (indices, values) in enumerate(
        zip(expected_indices, expected_values, strict=True)
    ):
        assert np.array_equal(np.flatnonzero(theta[row]), indices)
        np.testing.assert_allclose(theta[row, indices], values, rtol=5e-6, atol=1e-7)
    assert np.array_equal(theta[2], np.full(1000, 0.001, dtype=np.float32))
    assert beta.shape == (1000, 21233)
    np.testing.assert_allclose(
        beta[[0, 42, 999], [0, 42, 21232]],
        np.asarray(
            [
                2.6324029022362083e-05,
                0.0007532279123552144,
                7.333456778724212e-06,
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-10,
    )


def test_paper_results_are_exact() -> None:
    """Keep every paper-facing comparison value unchanged by refactoring."""
    results = read_json(RESULTS_ROOT / "results.json")
    methods = {row["method"]: row for row in results["methods"]}
    assert methods["neural"]["validation"] == {
        "annotation_coverage": 0.843,
        "high_confidence_evaluable_motifs": 429,
        "mean_sos": 0.6506700669726432,
        "median_sos": 0.6440677966101696,
        "optimized_motifs": 843,
        "sos_bands": {
            "high_gt_0_8": 71,
            "intermediate_0_6_to_0_8": 197,
            "low_lt_0_6": 161,
        },
        "useful_high_confidence_motifs": 268,
    }
    assert results["secondary"]["completion_nll_per_token"] == {
        "neural": {"validation": 8.832002635285642, "test": 8.841165651073034},
        "tomotopy": {"validation": 9.662228074924426, "test": 9.756948055261505},
    }
    assert methods["tomotopy"]["validation"]["annotation_coverage"] == 0.607
    assert methods["neural"]["test"]["annotation_coverage"] == 0.843
    assert methods["tomotopy"]["test"]["annotation_coverage"] == 0.607


def test_ablation_ledger_selects_u1_and_excludes_shallow_models() -> None:
    ledger = read_json(RESULTS_ROOT / "ablation_results.json")
    rows = {row["experiment"]: row for row in ledger}
    candidate = rows["selected_deep_U1"]
    assert candidate["retained"] is True
    assert candidate["test_evaluated"] is True
    assert candidate["architecture_eligibility"] == "eligible_deep"
    assert candidate["parameter_count"] == 233_600
    assert candidate["metrics"] == {
        "annotation_coverage": 0.843,
        "evaluable_motifs": 429,
        "mean_sos": 0.6506700669726432,
        "median_sos": 0.6440677966101696,
        "optimized_motifs": 843,
        "useful_motifs": 268,
        "validation_nll": 8.832002635285642,
    }
    assert all(candidate["gates"].values())
    assert "selected_deep_control" not in rows
    for experiment in ("U7", "locked_U7_retrain", "S1", "S8", "locked_S8_retrain"):
        assert rows[experiment]["architecture_eligibility"].startswith("excluded")
    assert all(not any("test" in key for key in row["metrics"]) for row in ledger)
