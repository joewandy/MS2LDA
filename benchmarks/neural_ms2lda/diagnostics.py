"""Model-selection diagnostics for neural topic-model inventories.

These metrics are deliberately independent of MAG/SOS.  They detect failure
modes that can be hidden by good likelihood or by a technically optimizable
motif spectrum: diffuse document mixtures, unused topics, duplicate topic-word
distributions, almost-uniform topics, and fragment/loss channel imbalance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PROBABILITY_FLOOR = 1e-12


def normalize_mixtures(theta: np.ndarray) -> np.ndarray:
    """Return finite row-normalized mixtures with a uniform empty-row fallback."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("theta must be a non-empty document-topic matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta must contain finite non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > PROBABILITY_FLOOR,
    )
    normalized[totals[:, 0] <= PROBABILITY_FLOOR] = 1.0 / values.shape[1]
    return normalized


def normalize_topics(beta: np.ndarray) -> np.ndarray:
    """Return row-normalized topic-word probabilities."""
    values = np.asarray(beta, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("beta must be a non-empty topic-word matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("beta must contain finite non-negative values")
    totals = values.sum(axis=1, keepdims=True, dtype=np.float64)
    if np.any(totals <= PROBABILITY_FLOOR):
        raise ValueError("every beta row must have positive probability mass")
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def _duplicate_component_summary(
    similarity: np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    """Summarize connected components induced by a cosine threshold."""
    topics = int(similarity.shape[0])
    rows, columns = np.where(np.triu(similarity >= float(threshold), k=1))
    parent = np.arange(topics, dtype=np.int64)
    size = np.ones(topics, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for left, right in zip(rows.tolist(), columns.tolist(), strict=True):
        union(int(left), int(right))

    roots = np.asarray([find(topic) for topic in range(topics)], dtype=np.int64)
    component_sizes = np.unique(roots, return_counts=True)[1]
    duplicate_sizes = component_sizes[component_sizes > 1]
    return {
        "threshold": float(threshold),
        "pair_count": int(len(rows)),
        "duplicate_component_count": int(len(duplicate_sizes)),
        "topics_in_duplicate_components": int(duplicate_sizes.sum()),
        "largest_component_size": int(component_sizes.max(initial=1)),
    }


def topic_inventory_diagnostics(
    theta: np.ndarray,
    beta: np.ndarray,
    *,
    active_usage_threshold: float = 0.0005,
    duplicate_cosine_thresholds: Sequence[float] = (0.95, 0.99, 0.999),
    top_word_count: int = 20,
    catastrophic_duplicate_component_fraction: float = 0.5,
) -> dict[str, Any]:
    """Measure mixture sparsity, topic use, beta concentration, and duplication."""
    mixtures = normalize_mixtures(theta)
    topics = normalize_topics(beta)
    if mixtures.shape[1] != topics.shape[0]:
        raise ValueError("theta topic count must match beta row count")
    if not 0 <= float(active_usage_threshold) <= 1:
        raise ValueError("active usage threshold must lie in [0, 1]")
    if not 0 < int(top_word_count) <= topics.shape[1]:
        raise ValueError("top word count must lie within the vocabulary")
    if not 0 < float(catastrophic_duplicate_component_fraction) <= 1:
        raise ValueError("catastrophic component fraction must lie in (0, 1]")

    usage = mixtures.mean(axis=0)
    document_entropy = -np.sum(
        mixtures * np.log(np.clip(mixtures, PROBABILITY_FLOOR, None)), axis=1
    )
    corpus_entropy = -np.sum(
        usage * np.log(np.clip(usage, PROBABILITY_FLOOR, None))
    )

    norms = np.linalg.norm(topics, axis=1, keepdims=True)
    normalized_topics = np.divide(
        topics,
        np.maximum(norms, PROBABILITY_FLOOR),
        out=np.zeros_like(topics),
    )
    similarity = normalized_topics @ normalized_topics.T
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.max(axis=1)

    top1 = np.argmax(mixtures, axis=1)
    unique_top1 = int(np.unique(top1).size)
    top_n = int(top_word_count)
    top_indices = np.argpartition(-topics, top_n - 1, axis=1)[:, :top_n]
    top_mass = np.take_along_axis(topics, top_indices, axis=1).sum(axis=1)
    top_word_uniqueness = float(np.unique(top_indices).size / top_indices.size)
    beta_entropy = -np.sum(
        topics * np.log(np.clip(topics, PROBABILITY_FLOOR, None)), axis=1
    )

    thresholds = tuple(sorted({float(value) for value in duplicate_cosine_thresholds}))
    if not thresholds or thresholds[0] < -1 or thresholds[-1] > 1:
        raise ValueError("duplicate cosine thresholds must lie in [-1, 1]")
    components = [
        _duplicate_component_summary(similarity, threshold) for threshold in thresholds
    ]
    strictest = components[-1]
    catastrophic_size = int(
        np.ceil(float(catastrophic_duplicate_component_fraction) * topics.shape[0])
    )

    return {
        "documents": int(mixtures.shape[0]),
        "topics": int(topics.shape[0]),
        "active_topic_usage_threshold": float(active_usage_threshold),
        "active_topics_above_usage_threshold": int(
            np.sum(usage > float(active_usage_threshold))
        ),
        "active_topics_mean_usage_ge_1_over_k": int(
            np.sum(usage >= 1.0 / topics.shape[0])
        ),
        "maximum_mean_topic_usage": float(usage.max()),
        "median_effective_topics_per_spectrum": float(np.median(np.exp(document_entropy))),
        "mean_effective_topics_per_spectrum": float(np.mean(np.exp(document_entropy))),
        "corpus_effective_topic_count": float(np.exp(corpus_entropy)),
        "unique_top1_topics": unique_top1,
        "topics_never_top1": int(topics.shape[0] - unique_top1),
        "mean_nearest_topic_beta_cosine": float(nearest.mean()),
        "median_nearest_topic_beta_cosine": float(np.median(nearest)),
        "maximum_pairwise_beta_cosine": float(similarity.max()),
        "duplicate_components": components,
        "strictest_duplicate_cosine_threshold": float(thresholds[-1]),
        "largest_strict_duplicate_component": int(
            strictest["largest_component_size"]
        ),
        "catastrophic_duplicate_component_fraction": float(
            catastrophic_duplicate_component_fraction
        ),
        "catastrophic_duplicate_component": bool(
            int(strictest["largest_component_size"]) >= catastrophic_size
        ),
        "top_word_count": top_n,
        "top_word_uniqueness": top_word_uniqueness,
        "median_beta_effective_words": float(np.median(np.exp(beta_entropy))),
        "median_beta_max_probability": float(np.median(topics.max(axis=1))),
        "median_beta_top_n_mass": float(np.median(top_mass)),
    }


def fragment_loss_mass_diagnostics(
    beta: np.ndarray,
    vocabulary: Sequence[str],
    *,
    lower_extreme: float = 0.1,
    upper_extreme: float = 0.9,
) -> dict[str, Any]:
    """Summarize per-topic probability assigned to fragment words."""
    topics = normalize_topics(beta)
    words = list(map(str, vocabulary))
    if len(words) != topics.shape[1]:
        raise ValueError("vocabulary length must match beta columns")
    fragment_mask = np.asarray([word.startswith("frag@") for word in words])
    loss_mask = np.asarray([word.startswith("loss@") for word in words])
    if not fragment_mask.any() or not loss_mask.any():
        raise ValueError("vocabulary must contain fragment and neutral-loss words")
    if np.any(~(fragment_mask | loss_mask)):
        raise ValueError("vocabulary contains an unknown token channel")
    if not 0 <= lower_extreme < upper_extreme <= 1:
        raise ValueError("channel extreme bounds are invalid")

    fragment_mass = topics[:, fragment_mask].sum(axis=1, dtype=np.float64)
    percentiles = {
        str(percentile): float(np.percentile(fragment_mass, percentile))
        for percentile in (1, 5, 25, 50, 75, 95, 99)
    }
    return {
        "minimum": float(fragment_mass.min()),
        "percentiles": percentiles,
        "median": float(np.median(fragment_mass)),
        "maximum": float(fragment_mass.max()),
        "extreme_definition": (
            f"fragment mass <{lower_extreme:g} or >{upper_extreme:g}"
        ),
        "fraction_extreme_skew": float(
            np.mean(
                (fragment_mass < float(lower_extreme))
                | (fragment_mass > float(upper_extreme))
            )
        ),
    }


def model_selection_diagnostics(
    theta: np.ndarray,
    beta: np.ndarray,
    vocabulary: Sequence[str],
    evaluation_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the repository's frozen diagnostic reporting contract."""
    return {
        "topic_inventory": topic_inventory_diagnostics(
            theta,
            beta,
            active_usage_threshold=float(
                evaluation_protocol["active_topic_usage_threshold"]
            ),
            duplicate_cosine_thresholds=tuple(
                map(float, evaluation_protocol["duplicate_cosine_thresholds"])
            ),
            top_word_count=int(evaluation_protocol["top_word_count"]),
            catastrophic_duplicate_component_fraction=float(
                evaluation_protocol["catastrophic_duplicate_component_fraction"]
            ),
        ),
        "fragment_probability_mass": fragment_loss_mass_diagnostics(
            beta,
            vocabulary,
            lower_extreme=float(evaluation_protocol["channel_extreme_lower"]),
            upper_extreme=float(evaluation_protocol["channel_extreme_upper"]),
        ),
    }
