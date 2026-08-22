"""Capacity-aware topic inventory diagnostics for comparing different K values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MATRIX_DIMENSIONS = 2


def _top_word_summary(
    beta: np.ndarray,
    indices: np.ndarray,
    top_n: int,
) -> dict[str, Any]:
    """Summarize top-word reuse for an explicitly selected topic subset."""
    if not len(indices):
        return {
            "topics": 0,
            "unique_top_words": 0,
            "top_word_diversity": 0.0,
            "distinct_topic_equivalents": 0.0,
        }
    selected = beta[indices]
    count = min(int(top_n), selected.shape[1])
    top_words = np.argpartition(selected, -count, axis=1)[:, -count:]
    unique = len(np.unique(top_words))
    diversity = unique / (len(indices) * count)
    return {
        "topics": len(indices),
        "unique_top_words": unique,
        "top_word_diversity": float(diversity),
        # This is the number of non-overlapping top-N topic slots represented
        # by the observed unique vocabulary, not a claim about true chemistry.
        "distinct_topic_equivalents": float(unique / count),
    }


def topic_inventory_summary(
    beta: np.ndarray,
    usage: np.ndarray,
    *,
    top_n: int,
) -> dict[str, Any]:
    """Summarize used and non-redundant capacity without assuming K=1000.

    Mass-covering inventories avoid comparing different K values solely with a
    threshold that itself changes as ``1 / K``.  The 90%, 95%, and 99% views
    are all retained so that no conclusion rests on one cutoff.
    """
    topics = np.asarray(beta, dtype=np.float64)
    weights = np.asarray(usage, dtype=np.float64)
    if (
        topics.ndim != MATRIX_DIMENSIONS
        or weights.ndim != 1
        or len(weights) != len(topics)
    ):
        msg = "beta and corpus topic usage do not align"
        raise ValueError(msg)
    if not len(topics) or topics.shape[1] == 0:
        msg = "topic inventory requires a non-empty beta matrix"
        raise ValueError(msg)
    if not np.all(np.isfinite(topics)) or not np.all(np.isfinite(weights)):
        msg = "topic inventory inputs must be finite"
        raise ValueError(msg)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        msg = "corpus topic usage must have positive mass"
        raise ValueError(msg)
    weights /= total
    order = np.argsort(-weights, kind="stable")
    cumulative = np.cumsum(weights[order])
    mass_coverages: dict[str, Any] = {}
    for fraction in (0.90, 0.95, 0.99):
        count = int(np.searchsorted(cumulative, fraction, side="left") + 1)
        indices = order[:count]
        mass_coverages[f"mass_{int(fraction * 100)}"] = {
            "target_mass": fraction,
            "actual_mass": float(weights[indices].sum()),
            **_top_word_summary(topics, indices, int(top_n)),
        }

    uniform = 1.0 / len(weights)
    positive = weights[weights > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    strict = np.flatnonzero(weights >= uniform)
    permissive = np.flatnonzero(weights >= 0.1 * uniform)
    return {
        "topic_count": len(weights),
        "corpus_effective_topic_count": float(math.exp(entropy)),
        "usage_at_least_uniform": {
            "threshold": uniform,
            **_top_word_summary(topics, strict, int(top_n)),
        },
        "usage_at_least_tenth_uniform": {
            "threshold": 0.1 * uniform,
            **_top_word_summary(topics, permissive, int(top_n)),
        },
        "mass_coverages": mass_coverages,
        "all_topics": _top_word_summary(
            topics,
            np.arange(len(topics), dtype=np.int64),
            int(top_n),
        ),
    }
