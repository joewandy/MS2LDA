"""Tests for the model-selection diagnostic contract."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.neural_ms2lda.diagnostics import (
    fragment_loss_mass_diagnostics,
    model_selection_diagnostics,
    normalize_mixtures,
    topic_inventory_diagnostics,
)


def test_normalize_mixtures_uses_uniform_empty_row_fallback() -> None:
    theta = np.asarray([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    normalized = normalize_mixtures(theta)
    np.testing.assert_allclose(normalized[0], [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(normalized[1], [0.5, 0.5])


def test_inventory_detects_duplicate_component_and_unused_topic() -> None:
    theta = np.asarray(
        [[0.9, 0.1, 0.0], [0.1, 0.1, 0.8], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    beta = np.asarray(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.7, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )
    diagnostics = topic_inventory_diagnostics(
        theta,
        beta,
        active_usage_threshold=0.05,
        duplicate_cosine_thresholds=(0.95, 0.999),
        top_word_count=2,
        catastrophic_duplicate_component_fraction=0.5,
    )
    strict = diagnostics["duplicate_components"][-1]
    assert strict["pair_count"] == 1
    assert strict["duplicate_component_count"] == 1
    assert strict["largest_component_size"] == 2
    assert diagnostics["unique_top1_topics"] == 2
    assert diagnostics["topics_never_top1"] == 1
    assert diagnostics["catastrophic_duplicate_component"] is True
    assert diagnostics["top_word_uniqueness"] == pytest.approx(4.0 / 6.0)


def test_fragment_loss_mass_diagnostics_reports_extreme_topics() -> None:
    beta = np.asarray(
        [
            [0.45, 0.45, 0.05, 0.05],
            [0.05, 0.05, 0.45, 0.45],
        ],
        dtype=np.float32,
    )
    vocabulary = ["frag@10.00", "frag@20.00", "loss@5.00", "loss@15.00"]
    diagnostics = fragment_loss_mass_diagnostics(beta, vocabulary)
    assert diagnostics["minimum"] == pytest.approx(0.1)
    assert diagnostics["maximum"] == pytest.approx(0.9)
    assert diagnostics["median"] == pytest.approx(0.5)
    assert diagnostics["fraction_extreme_skew"] == pytest.approx(0.0)


def test_model_selection_diagnostics_uses_protocol_thresholds() -> None:
    theta = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    beta = np.asarray(
        [[0.4, 0.1, 0.4, 0.1], [0.1, 0.4, 0.1, 0.4]], dtype=np.float32
    )
    vocabulary = ["frag@1.00", "loss@1.00", "frag@2.00", "loss@2.00"]
    protocol = {
        "active_topic_usage_threshold": 0.0005,
        "duplicate_cosine_thresholds": [0.95, 0.99, 0.999],
        "catastrophic_duplicate_component_fraction": 0.5,
        "top_word_count": 2,
        "channel_extreme_lower": 0.1,
        "channel_extreme_upper": 0.9,
    }
    diagnostics = model_selection_diagnostics(theta, beta, vocabulary, protocol)
    assert set(diagnostics) == {"topic_inventory", "fragment_probability_mass"}
    assert diagnostics["topic_inventory"]["topics"] == 2
    assert diagnostics["fragment_probability_mass"]["median"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "theta",
    [
        np.asarray([[1.0, -0.1]], dtype=np.float32),
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
        np.asarray([], dtype=np.float32),
    ],
)
def test_invalid_mixtures_fail_closed(theta: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_mixtures(theta)
