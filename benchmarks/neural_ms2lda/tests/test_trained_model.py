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
        np.asarray([383, 456, 691, 800]),
        np.asarray([61, 553, 609, 728, 731, 785]),
    )
    expected_values = (
        np.asarray(
            [0.3621624, 0.00024500815, 0.6224439, 0.015148607],
            dtype=np.float32,
        ),
        np.asarray(
            [
                0.0020063557,
                0.25020465,
                0.00077795435,
                0.19118232,
                0.23770817,
                0.3181206,
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
                7.370563980657607e-05,
                5.212986798142083e-05,
                1.4015468877914827e-05,
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
        "annotation_coverage": 0.853,
        "high_confidence_evaluable_motifs": 396,
        "mean_sos": 0.6366813215122081,
        "median_sos": 0.6323809523809524,
        "optimized_motifs": 853,
        "sos_bands": {
            "high_gt_0_8": 64,
            "intermediate_0_6_to_0_8": 170,
            "low_lt_0_6": 162,
        },
        "useful_high_confidence_motifs": 234,
    }
    assert results["secondary"]["completion_nll_per_token"] == {
        "neural": {"validation": 9.122051611660526, "test": 9.135820892827928},
        "tomotopy": {"validation": 9.662228074924426, "test": 9.756948055261505},
    }
    assert methods["tomotopy"]["validation"]["annotation_coverage"] == 0.607
    assert methods["neural"]["test"]["annotation_coverage"] == 0.853
    assert methods["tomotopy"]["test"]["annotation_coverage"] == 0.607
