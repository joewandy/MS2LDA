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
    """Lock interpretable numerical values without coupling tests to file bytes."""
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
        np.asarray([129, 464, 628, 883]),
        np.asarray([113, 261, 435, 758, 983]),
    )
    expected_values = (
        np.asarray(
            [0.22317973, 0.10929480, 0.016402945, 0.6511225],
            dtype=np.float32,
        ),
        np.asarray(
            [0.69798553, 0.056570835, 0.18551093, 0.004249601, 0.0556831],
            dtype=np.float32,
        ),
    )
    for row, (indices, values) in enumerate(
        zip(expected_indices, expected_values, strict=True)
    ):
        assert np.array_equal(np.flatnonzero(theta[row]), indices)
        assert np.array_equal(theta[row, indices], values)
    assert np.array_equal(theta[2], np.full(1000, 0.001, dtype=np.float32))
    assert beta.shape == (1000, 21233)
    assert beta[0, 0] == np.float32(0.06091761961579323)
    assert beta[42, 42] == np.float32(5.1844017434632406e-05)
    assert beta[999, 21232] == np.float32(2.4032394776440924e-06)


def test_paper_results_are_exact() -> None:
    """Keep every paper-facing comparison value unchanged by refactoring."""
    results = read_json(RESULTS_ROOT / "results.json")
    methods = {row["method"]: row for row in results["methods"]}
    assert methods["neural"]["validation"] == {
        "annotation_coverage": 0.663,
        "high_confidence_evaluable_motifs": 312,
        "mean_sos": 0.6323301481310782,
        "median_sos": 0.6363636363636365,
        "optimized_motifs": 663,
        "sos_bands": {
            "high_gt_0_8": 49,
            "intermediate_0_6_to_0_8": 136,
            "low_lt_0_6": 127,
        },
        "useful_high_confidence_motifs": 185,
    }
    assert results["secondary"]["completion_nll_per_token"] == {
        "neural": {"validation": 8.501446912771746, "test": 8.522600207027194},
        "tomotopy": {"validation": 9.662228074924426, "test": 9.756948055261505},
    }
    assert methods["tomotopy"]["validation"]["annotation_coverage"] == 0.607
    assert methods["neural"]["test"]["annotation_coverage"] == 0.663
    assert methods["tomotopy"]["test"]["annotation_coverage"] == 0.607
