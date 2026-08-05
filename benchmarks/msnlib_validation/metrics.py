"""Pure metric definitions for machine-readable MSnLib reporting."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

EPSILON = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return finite probability rows, using uniform values for empty rows."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("expected a non-empty matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("probability matrices must be finite and nonnegative")
    denominator = matrix.sum(axis=1, keepdims=True)
    result = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > EPSILON,
    )
    empty = denominator[:, 0] <= EPSILON
    result[empty] = 1.0 / matrix.shape[1]
    return result


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate aligned row-wise cosine similarity."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("cosine inputs must be aligned matrices")
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPSILON,
    )


def jensen_shannon_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate aligned Jensen-Shannon divergence in natural-log units."""
    a = normalize_rows(left)
    b = normalize_rows(right)
    midpoint = 0.5 * (a + b)

    def kl(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        terms = np.where(
            values > 0,
            values * (np.log(np.clip(values, EPSILON, None)) - np.log(reference)),
            0.0,
        )
        return terms.sum(axis=1)

    return 0.5 * (kl(a, midpoint) + kl(b, midpoint))


def document_completion_nll(
    theta: np.ndarray,
    beta: np.ndarray,
    heldout_documents: Sequence[Sequence[str]],
    vocabulary: Sequence[str],
) -> dict[str, float | int]:
    """Score completion words under fixed topic mixtures and topic-word rows."""
    theta_values = normalize_rows(theta)
    beta_values = normalize_rows(beta)
    if theta_values.shape[0] != len(heldout_documents):
        raise ValueError("theta rows and held-out documents differ")
    if beta_values.shape != (theta_values.shape[1], len(vocabulary)):
        raise ValueError("beta does not align with theta and vocabulary")
    columns = {str(word): index for index, word in enumerate(vocabulary)}
    negative_log_likelihood = 0.0
    in_vocabulary_tokens = 0
    out_of_vocabulary_tokens = 0
    eligible_documents = 0
    for row, document in enumerate(heldout_documents):
        counts = Counter(map(str, document))
        eligible = False
        for word, count in counts.items():
            column = columns.get(word)
            if column is None:
                out_of_vocabulary_tokens += int(count)
                continue
            probability = float(theta_values[row] @ beta_values[:, column])
            negative_log_likelihood -= int(count) * math.log(max(probability, EPSILON))
            in_vocabulary_tokens += int(count)
            eligible = True
        eligible_documents += int(eligible)
    value = (
        negative_log_likelihood / in_vocabulary_tokens
        if in_vocabulary_tokens
        else math.nan
    )
    total = in_vocabulary_tokens + out_of_vocabulary_tokens
    return {
        "nll_per_token": float(value),
        "in_vocabulary_tokens": in_vocabulary_tokens,
        "out_of_vocabulary_tokens": out_of_vocabulary_tokens,
        "oov_fraction": out_of_vocabulary_tokens / total if total else math.nan,
        "eligible_documents": eligible_documents,
        "total_documents": len(heldout_documents),
    }


def active_topic_metrics(
    theta: np.ndarray,
    *,
    document_threshold: float,
    corpus_threshold: float,
) -> dict[str, float | int]:
    """Report document-level and corpus-level active-topic counts."""
    values = normalize_rows(theta)
    per_document = (values >= document_threshold).sum(axis=1)
    return {
        "document_active_mean": float(per_document.mean()),
        "document_active_median": float(np.median(per_document)),
        "document_active_p95": float(np.percentile(per_document, 95)),
        "corpus_active_topics": int((values.mean(axis=0) >= corpus_threshold).sum()),
    }


def top_word_diversity(beta: np.ndarray, *, top_n: int) -> float:
    """Fraction of unique vocabulary positions among all topic top words."""
    values = np.asarray(beta)
    if values.ndim != 2 or top_n < 1:
        raise ValueError("beta must be a matrix and top_n positive")
    count = min(top_n, values.shape[1])
    indices = np.argsort(-values, axis=1, kind="stable")[:, :count]
    return float(len(set(indices.ravel().tolist())) / indices.size)


def word_cooccurrence_npmi(
    beta: np.ndarray,
    training_documents: Sequence[Sequence[str]],
    vocabulary: Sequence[str],
    *,
    top_n: int,
) -> dict[str, float | int]:
    """Compute binary-document top-word NPMI using training documents only."""
    if not training_documents:
        raise ValueError("training_documents cannot be empty")
    count = min(top_n, len(vocabulary))
    top_indices = np.argsort(-np.asarray(beta), axis=1, kind="stable")[:, :count]
    requested_words = set(map(int, top_indices.ravel()))
    requested_pairs = {
        pair
        for topic in top_indices
        for pair in combinations(sorted(map(int, topic)), 2)
    }
    requested_neighbors: dict[int, set[int]] = {}
    for first, second in requested_pairs:
        requested_neighbors.setdefault(first, set()).add(second)
    columns = {str(word): index for index, word in enumerate(vocabulary)}
    document_frequency: Counter[int] = Counter()
    pair_frequency: Counter[tuple[int, int]] = Counter()
    for document in training_documents:
        present_set = {
            columns[word]
            for word in document
            if word in columns and columns[word] in requested_words
        }
        present = sorted(present_set)
        document_frequency.update(present)
        for first in present:
            pair_frequency.update(
                (first, second)
                for second in requested_neighbors.get(first, ())
                if second in present_set
            )
    total_documents = len(training_documents)
    topic_scores = []
    undefined_pairs = 0
    for topic in top_indices:
        pair_scores = []
        for first, second in combinations(sorted(map(int, topic)), 2):
            joint_count = pair_frequency[(first, second)]
            if joint_count == 0:
                pair_scores.append(-1.0)
                undefined_pairs += 1
                continue
            joint = joint_count / total_documents
            first_probability = document_frequency[first] / total_documents
            second_probability = document_frequency[second] / total_documents
            if joint == 1.0:
                pair_scores.append(1.0)
                continue
            pmi = math.log(joint / (first_probability * second_probability))
            pair_scores.append(pmi / -math.log(joint))
        topic_scores.append(float(np.mean(pair_scores)) if pair_scores else math.nan)
    return {
        "mean_npmi": float(np.nanmean(topic_scores)),
        "median_topic_npmi": float(np.nanmedian(topic_scores)),
        "undefined_pairs_scored_as_minus_one": undefined_pairs,
    }


def optimal_topic_matching(
    left_beta: np.ndarray,
    right_beta: np.ndarray,
    *,
    top_n: int,
) -> dict[str, object]:
    """Match topics by maximum cosine and report cosine/top-word Jaccard."""
    left = normalize_rows(left_beta)
    right = normalize_rows(right_beta)
    if left.shape != right.shape:
        raise ValueError("topic matrices must have identical aligned shapes")
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPSILON)
    right_norm = right / np.maximum(
        np.linalg.norm(right, axis=1, keepdims=True), EPSILON
    )
    similarity = left_norm @ right_norm.T
    left_rows, right_rows = linear_sum_assignment(-similarity)
    matched_cosines = similarity[left_rows, right_rows]
    count = min(top_n, left.shape[1])
    left_top = np.argsort(-left, axis=1, kind="stable")[:, :count]
    right_top = np.argsort(-right, axis=1, kind="stable")[:, :count]
    jaccards = []
    for left_topic, right_topic in zip(left_rows, right_rows, strict=True):
        a = set(map(int, left_top[left_topic]))
        b = set(map(int, right_top[right_topic]))
        jaccards.append(len(a & b) / len(a | b) if a or b else 1.0)
    return {
        "left_topic_ids": left_rows.astype(int).tolist(),
        "right_topic_ids": right_rows.astype(int).tolist(),
        "matched_cosine_mean": float(np.mean(matched_cosines)),
        "matched_cosine_median": float(np.median(matched_cosines)),
        "top_word_jaccard_mean": float(np.mean(jaccards)),
        "top_word_jaccard_median": float(np.median(jaccards)),
    }


def convergence_metrics(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Summarize one inference budget against a long-refinement reference."""
    cosine = cosine_rows(candidate, reference)
    divergence = jensen_shannon_rows(candidate, reference)
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "js_mean": float(np.mean(divergence)),
        "js_median": float(np.median(divergence)),
        "js_p95": float(np.percentile(divergence, 95)),
    }


def calculate_sos(annotation_fp: np.ndarray, molecule_fp: np.ndarray) -> float:
    """Return historical substructure-overlap arithmetic for one molecule."""
    annotation = np.asarray(annotation_fp, dtype=bool)
    molecule = np.asarray(molecule_fp, dtype=bool)
    if annotation.shape != molecule.shape:
        raise ValueError("fingerprints must have identical shapes")
    denominator = int(annotation.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(annotation, molecule).sum() / denominator)
