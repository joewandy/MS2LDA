"""Chemistry-association helpers shared by neural and Tomotopy evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

ASSOCIATION_MODES = ("dominant_topic", "probability_ge_frozen_threshold")


def associated_record_indices(
    theta: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> dict[int, list[int]]:
    """Map topics to held-out rows under the two frozen association rules."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("theta must be a non-empty document-topic matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta must contain finite non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        totals,
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=totals > 0,
    )
    associated: defaultdict[int, list[int]] = defaultdict(list)
    if mode == "dominant_topic":
        for row, topic in enumerate(np.argmax(values, axis=1)):
            associated[int(topic)].append(row)
    elif mode == "probability_ge_frozen_threshold":
        if not 0 <= threshold <= 1:
            raise ValueError("association threshold must lie in [0, 1]")
        rows, topics = np.nonzero(values >= float(threshold))
        for row, topic in zip(rows, topics, strict=True):
            associated[int(topic)].append(int(row))
    else:
        raise ValueError(f"unsupported association mode: {mode}")
    return dict(associated)
