"""Evaluation metrics aligned to the corrected Tomotopy benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import scipy.sparse as sp

PROBABILITY_FLOOR = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return finite probability rows."""
    matrix = np.asarray(values, dtype=np.float64)
    denominator = matrix.sum(axis=1, keepdims=True)
    result = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > PROBABILITY_FLOOR,
    )
    result[denominator[:, 0] <= PROBABILITY_FLOOR] = 1.0 / matrix.shape[1]
    return result


def active_topic_metrics(
    theta: np.ndarray,
    *,
    document_threshold: float,
    corpus_threshold: float,
) -> dict[str, float | int]:
    """Use the same active-topic definitions as the corrected reference."""
    values = normalize_rows(theta)
    per_document = (values >= document_threshold).sum(axis=1)
    return {
        "document_active_mean": float(per_document.mean()),
        "document_active_median": float(np.median(per_document)),
        "document_active_p95": float(np.percentile(per_document, 95)),
        "corpus_active_topics": int((values.mean(axis=0) >= corpus_threshold).sum()),
    }


def effective_topic_summary(theta: np.ndarray) -> dict[str, float]:
    """Summarize entropy-derived effective topics per spectrum."""
    values = normalize_rows(theta)
    entropy = -np.sum(values * np.log(np.clip(values, PROBABILITY_FLOOR, None)), axis=1)
    effective = np.exp(entropy)
    maximum = values.max(axis=1)
    return {
        "effective_topic_count_mean": float(effective.mean()),
        "effective_topic_count_median": float(np.median(effective)),
        "effective_topic_count_p95": float(np.percentile(effective, 95)),
        "maximum_topic_probability_median": float(np.median(maximum)),
        "maximum_topic_probability_p95": float(np.percentile(maximum, 95)),
        "maximum_topic_probability_maximum": float(maximum.max()),
    }


def completion_metrics(
    theta: np.ndarray,
    beta: np.ndarray,
    completion: sp.csr_matrix,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    """Score held-out completion counts under exact theta times beta."""
    losses = np.full(completion.shape[0], np.nan, dtype=np.float64)
    total_loss = 0.0
    in_vocabulary = 0
    out_of_vocabulary = 0
    eligible = 0
    for row in range(completion.shape[0]):
        start, stop = completion.indptr[row], completion.indptr[row + 1]
        words = completion.indices[start:stop]
        counts = completion.data[start:stop]
        token_count = int(counts.sum())
        out_of_vocabulary += int(records[row]["completion_oov_tokens"])
        if not token_count:
            continue
        probabilities = theta[row] @ beta[:, words]
        loss = -float(
            np.sum(counts * np.log(np.clip(probabilities, PROBABILITY_FLOOR, None)))
        )
        losses[row] = loss / token_count
        total_loss += loss
        in_vocabulary += token_count
        eligible += 1
    total = in_vocabulary + out_of_vocabulary
    return (
        {
            "nll_per_token": total_loss / in_vocabulary,
            "in_vocabulary_tokens": in_vocabulary,
            "out_of_vocabulary_tokens": out_of_vocabulary,
            "oov_fraction": out_of_vocabulary / total,
            "eligible_documents": eligible,
            "total_documents": completion.shape[0],
        },
        losses,
    )
