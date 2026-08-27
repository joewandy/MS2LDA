"""Validation-only diagnostics for the MSnLib neural-model follow-up."""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


def retemperature_theta(
    theta: np.ndarray,
    *,
    source_temperature: float,
    target_temperature: float,
) -> np.ndarray:
    """Change softmax temperature from probabilities without changing rank."""
    if source_temperature <= 0 or target_temperature <= 0:
        raise ValueError("theta temperatures must be positive")
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("theta must be a non-empty matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta must contain finite non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        totals,
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=totals > 0,
    )
    logits = np.log(np.clip(values, EPS, None)) * (
        float(source_temperature) / float(target_temperature)
    )
    logits -= logits.max(axis=1, keepdims=True)
    sharpened = np.exp(logits)
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    return sharpened.astype(np.float32)


def theta_distribution(theta: np.ndarray) -> dict[str, Any]:
    """Summarize mixture sparsity and locked membership-threshold reach."""
    values = np.asarray(theta, dtype=np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), EPS)
    entropy = -np.sum(values * np.log(np.clip(values, EPS, None)), axis=1)
    effective = np.exp(entropy)
    maximum = values.max(axis=1)
    percentiles = {
        f"p{percentile}": float(np.percentile(maximum, percentile))
        for percentile in (10, 25, 50, 75, 90, 95)
    }
    top1 = np.argmax(values, axis=1)
    return {
        "median_effective_topics_per_spectrum": float(np.median(effective)),
        "mean_effective_topics_per_spectrum": float(np.mean(effective)),
        "median_max_theta": float(np.median(maximum)),
        **{f"max_theta_{name}": value for name, value in percentiles.items()},
        "fraction_max_theta_ge_0_5": float(np.mean(maximum >= 0.5)),
        "fraction_max_theta_ge_0_3": float(np.mean(maximum >= 0.3)),
        "fraction_max_theta_ge_0_2": float(np.mean(maximum >= 0.2)),
        "unique_top1_topics": int(len(np.unique(top1))),
    }


def top_rank_stability(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Check that post-hoc temperature changes preserve top-1/top-3 ranks."""
    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("theta matrices must have identical two-dimensional shapes")
    top1 = np.argmax(left, axis=1) == np.argmax(right, axis=1)
    left_top3 = np.sort(np.argpartition(-left, 2, axis=1)[:, :3], axis=1)
    right_top3 = np.sort(np.argpartition(-right, 2, axis=1)[:, :3], axis=1)
    return {
        "top1_assignment_stability": float(np.mean(top1)),
        "top3_set_stability": float(np.mean(np.all(left_top3 == right_top3, axis=1))),
    }
