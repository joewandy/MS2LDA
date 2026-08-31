"""Tests for probability-matrix persistence helpers."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.neural_ms2lda.reproducibility import (
    normalize_probability_rows,
    validate_probability_matrix,
)


def test_normalize_probability_rows_repairs_float32_sum_drift() -> None:
    """Wide-decoder rounding is removed without changing probability ratios."""
    original = np.asarray(
        [
            [0.10002, 0.20004, 0.30006, 0.40008],
            [0.50010, 0.25005, 0.125025, 0.125025],
        ],
        dtype=np.float32,
    )

    normalized = normalize_probability_rows(original, name="test beta")

    assert normalized.dtype == np.float32
    validate_probability_matrix(normalized, name="normalized test beta")
    np.testing.assert_allclose(
        normalized[:, 0] / normalized[:, 1],
        original[:, 0] / original[:, 1],
        rtol=2e-7,
    )


@pytest.mark.parametrize(
    "invalid",
    (
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.5, -0.5]], dtype=np.float32),
        np.asarray([[0.5, np.nan]], dtype=np.float32),
    ),
)
def test_normalize_probability_rows_rejects_invalid_rows(
    invalid: np.ndarray,
) -> None:
    """Invalid decoder outputs fail before they can enter the evidence set."""
    with pytest.raises(FloatingPointError):
        normalize_probability_rows(invalid, name="invalid beta")
