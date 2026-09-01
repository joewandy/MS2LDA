"""Shared validation metrics and artifact writers for fitted topic models.

This module contains ordinary functions only.  Keeping these operations out of
command-line scripts lets every model runner use the same validation protocol
without importing a large experimental campaign as a pseudo-library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .diagnostics import normalize_mixtures
from .study_protocol import (
    TRAINING_ACCESS_AUDIT_FILENAME,
    VALIDATION_ACCESS_AUDIT_FILENAME,
)
from .utils import atomic_save_numpy, read_json, write_json

EPSILON = 1e-12
MODEL_SELECTION_EVALUATION_PROTOCOL = {
    "active_topic_usage_threshold": 0.0005,
    "duplicate_cosine_thresholds": (0.95, 0.99, 0.999),
    "catastrophic_duplicate_component_fraction": 0.5,
    "top_word_count": 20,
    "channel_extreme_lower": 0.1,
    "channel_extreme_upper": 0.9,
}


def completion_metrics(
    theta: np.ndarray,
    beta: np.ndarray,
    completion: sp.csr_matrix,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score held-out token counts under the exact mixture ``theta @ beta``."""
    if theta.ndim != 2 or beta.ndim != 2:
        raise ValueError("theta and beta must be matrices")
    if theta.shape[1] != beta.shape[0] or completion.shape != (
        theta.shape[0],
        beta.shape[1],
    ):
        raise ValueError("theta, beta and completion matrix shapes do not align")
    if len(records) != completion.shape[0]:
        raise ValueError("completion records must match completion rows")
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
        probability = theta[row] @ beta[:, words]
        total_loss -= float(
            np.sum(counts * np.log(np.clip(probability, EPSILON, None))),
        )
        in_vocabulary += token_count
        eligible += 1
    if in_vocabulary <= 0:
        raise ValueError("completion split contains no in-vocabulary tokens")
    total = in_vocabulary + out_of_vocabulary
    return {
        "nll_per_token": total_loss / in_vocabulary,
        "in_vocabulary_tokens": in_vocabulary,
        "out_of_vocabulary_tokens": out_of_vocabulary,
        "oov_fraction": out_of_vocabulary / total,
        "eligible_documents": eligible,
        "total_documents": completion.shape[0],
    }


def finalize_validation_access_audit(run: Path, method: str) -> dict[str, Any]:
    """Extend the sealed training-access record after validation chemistry.

    Model fitting writes a training-only record under a distinct filename. The
    chemistry stage owns the final audit path so the clean-room runner can
    reject cached chemistry without mistaking the training record for a prior
    chemistry result.
    """
    model_output = run / "models" / method
    training_path = model_output / TRAINING_ACCESS_AUDIT_FILENAME
    if not training_path.is_file():
        message = f"missing sealed training-access record: {training_path}"
        raise FileNotFoundError(message)
    audit = read_json(training_path)
    audit.update(
        {
            "chemical_split": "validation",
            "candidate_test_chemistry_loaded": False,
            "candidate_test_mag_or_sos_computed": False,
            "reused_leakage_filtered_mag_index": str(
                run / "mag/index/spec2vec_filtered.faiss"
            ),
        }
    )
    write_json(model_output / VALIDATION_ACCESS_AUDIT_FILENAME, audit)
    return audit


def entropy_diagnostics(theta: np.ndarray) -> dict[str, float]:
    """Return conditional entropy, marginal entropy, and their MI gap."""
    values = theta.astype(np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), EPSILON)
    conditional = float(
        np.mean(-np.sum(values * np.log(np.clip(values, EPSILON, None)), axis=1))
    )
    marginal = values.mean(axis=0)
    marginal_entropy = float(
        -np.sum(marginal * np.log(np.clip(marginal, EPSILON, None)))
    )
    return {
        "mean_conditional_theta_entropy": conditional,
        "marginal_theta_entropy": marginal_entropy,
        "mutual_information": marginal_entropy - conditional,
    }


def theta_support_diagnostics(theta: np.ndarray) -> dict[str, object]:
    """Summarize exact mixture support, effective topics, and confidence."""
    mixtures = normalize_mixtures(theta)
    support = np.count_nonzero(mixtures > 0.0, axis=1)
    entropy = -np.sum(
        np.where(
            mixtures > 0.0,
            mixtures * np.log(np.clip(mixtures, EPSILON, None)),
            0.0,
        ),
        axis=1,
    )
    maximum = mixtures.max(axis=1)
    percentiles = {
        str(percentile): float(np.percentile(support, percentile))
        for percentile in (1, 5, 25, 50, 75, 95, 99)
    }
    return {
        "minimum_exact_support": int(support.min()),
        "support_size_percentiles": percentiles,
        "median_exact_support": float(np.median(support)),
        "mean_exact_support": float(support.mean()),
        "maximum_exact_support": int(support.max()),
        "fraction_support_le_3": float(np.mean(support <= 3)),
        "median_effective_topics_per_spectrum": float(np.median(np.exp(entropy))),
        "mean_effective_topics_per_spectrum": float(np.mean(np.exp(entropy))),
        "median_maximum_theta": float(np.median(maximum)),
        "fraction_max_theta_ge_0_5": float(np.mean(maximum >= 0.5)),
        "fraction_max_theta_ge_0_3": float(np.mean(maximum >= 0.3)),
        "fraction_max_theta_ge_0_2": float(np.mean(maximum >= 0.2)),
    }


def mixture_distribution_summary(theta: np.ndarray) -> dict[str, float | int]:
    """Summarize mixture concentration and the number of document winners.

    This is the compact compatibility view retained in model result files.  The
    richer exact-support and inventory diagnostics remain the canonical inputs
    to the paper.
    """
    mixtures = normalize_mixtures(theta)
    entropy = -np.sum(
        mixtures * np.log(np.clip(mixtures, EPSILON, None)),
        axis=1,
    )
    maximum = mixtures.max(axis=1)
    percentiles = {
        f"max_theta_p{percentile}": float(np.percentile(maximum, percentile))
        for percentile in (10, 25, 50, 75, 90, 95)
    }
    return {
        "median_effective_topics_per_spectrum": float(np.median(np.exp(entropy))),
        "mean_effective_topics_per_spectrum": float(np.mean(np.exp(entropy))),
        "median_max_theta": float(np.median(maximum)),
        **percentiles,
        "fraction_max_theta_ge_0_5": float(np.mean(maximum >= 0.5)),
        "fraction_max_theta_ge_0_3": float(np.mean(maximum >= 0.3)),
        "fraction_max_theta_ge_0_2": float(np.mean(maximum >= 0.2)),
        "unique_top1_topics": int(len(np.unique(np.argmax(mixtures, axis=1)))),
    }


def topic_word_diagnostics(
    beta: np.ndarray,
    vocabulary: list[str],
    *,
    top_n: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize top-word uniqueness and fragment probability mass."""
    count = min(int(top_n), beta.shape[1])
    candidates = np.argpartition(-beta, count - 1, axis=1)[:, :count]
    scores = np.take_along_axis(beta, candidates, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    indices = np.take_along_axis(candidates, order, axis=1)
    rows = [
        {
            "topic_id": topic_id,
            "top_words": " ".join(vocabulary[index] for index in topic_indices),
            "top_probabilities": " ".join(
                f"{float(beta[topic_id, index]):.10g}" for index in topic_indices
            ),
        }
        for topic_id, topic_indices in enumerate(indices)
    ]
    unique = len(set(indices.ravel().tolist())) / indices.size
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary],
        dtype=bool,
    )
    fragment_mass = beta[:, fragment_mask].sum(axis=1).astype(np.float64)
    percentiles = {
        str(percentile): float(np.percentile(fragment_mass, percentile))
        for percentile in (1, 5, 25, 50, 75, 95, 99)
    }
    return (
        {
            "top_word_count": count,
            "top_word_uniqueness": float(unique),
            "fragment_probability_mass": {
                "minimum": float(fragment_mass.min()),
                "percentiles": percentiles,
                "median": float(np.median(fragment_mass)),
                "maximum": float(fragment_mass.max()),
                "extreme_definition": "fragment mass <0.1 or >0.9",
                "fraction_extreme_skew": float(
                    np.mean((fragment_mass < 0.1) | (fragment_mass > 0.9))
                ),
            },
        },
        rows,
    )


def save_validation(
    run: Path,
    method: str,
    beta: np.ndarray,
    theta: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Save validation arrays and metrics under the locked artifact layout."""
    output = run / "validation_evaluation" / method
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / "beta.npy", beta.astype(np.float32, copy=False))
    atomic_save_numpy(
        output / "validation_full_theta.npy",
        theta.astype(np.float32, copy=False),
    )
    write_json(
        output / "complete.json",
        {"method": method, "split": "validation", "metrics": metrics},
    )


def score_chemical_validation(
    run: Path,
    data_root: Path,
    method: str,
) -> dict[str, Any]:
    """Run the shared leakage-controlled MAG/SOS protocol on validation only."""
    # Chemical scoring has optional heavy dependencies.  Import it only for the
    # explicit scoring command so model training and inference stay lightweight.
    from .chemical import run_chemical_scoring

    protocol = read_json(run / "protocol.json")
    result = run_chemical_scoring(
        run,
        method=method,
        data_root=data_root,
        protocol=protocol,
        split="validation",
    )
    finalize_validation_access_audit(run, method)
    return result
