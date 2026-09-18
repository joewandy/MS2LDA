"""Full-spectrum motif stability and redundancy for the six locked fits.

Hungarian assignments maximize total full-beta cosine, not chemical agreement.
All matches are reported, including weak forced matches. Fifteen pairwise
comparisons share six fits and must never be treated as 15 independent runs.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .chemical_assessment_io import fit_paths, read_jsonl
from .utils import write_json, write_jsonl


def normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize finite nonzero vectors; explicitly flag unscorable rows."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a matrix")
    valid = np.all(np.isfinite(matrix), axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(matrix), matrix, 0), axis=1)
    valid &= norms > 0
    normalized = np.divide(
        matrix, norms[:, None], out=np.zeros_like(matrix), where=valid[:, None]
    )
    return normalized, valid


def load_beta(fit: dict, vocabulary_size: int) -> np.ndarray:
    """Check common vocabulary width and finite, positive probability rows."""
    beta = np.load(fit_paths(fit)["beta"], allow_pickle=False)
    if (
        beta.ndim != 2
        or beta.shape[1] != vocabulary_size
        or not np.all(np.isfinite(beta))
        or np.any(beta < 0)
    ):
        raise ValueError("invalid saved topic-word matrix")
    # Some neural checkpoints are Fortran-ordered float32 arrays. A float32
    # reduction over 21,233 words can accumulate 1e-4 error even though the
    # stored probabilities sum to one at float64 precision. Do not loosen the
    # probability tolerance or alter their values: validate with float64 sums.
    if not np.allclose(beta.sum(axis=1, dtype=np.float64), 1, atol=1e-5, rtol=0):
        raise ValueError("beta rows are not probability distributions")
    return beta


def match_topics(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return optimal one-to-one IDs, cosine scores, reciprocal-NN flags.

    Similarity uses ALL vocabulary columns. Exact nearest-neighbour ties use
    the lowest topic index, so reciprocal flags are conservative for duplicate
    topics and depend on this documented tie rule. Hungarian ties need not be
    chemically unique; interpreting a forced match requires its actual score.
    """
    a, valid_a = normalize_rows(left)
    b, valid_b = normalize_rows(right)
    if a.shape[1] != b.shape[1] or not valid_a.all() or not valid_b.all():
        raise ValueError("matching requires finite nonzero vectors in one vocabulary")
    similarity = np.clip(a @ b.T, -1, 1)
    rows, cols = linear_sum_assignment(similarity, maximize=True)
    reciprocal = (np.argmax(similarity, axis=1)[rows] == cols) & (
        np.argmax(similarity, axis=0)[cols] == rows
    )
    return rows, cols, similarity[rows, cols], reciprocal


def jaccard(left, right) -> float | None:
    """Set intersection/union; two empty supports have no estimable agreement."""
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else None


def profile_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    """Cosine of signed feature-enrichment effects, undefined for a zero vector."""
    normalized, valid = normalize_rows(np.array([left, right]))
    return float(np.clip(normalized[0] @ normalized[1], -1, 1)) if valid.all() else None


def run_stability(protocol: dict, output: Path) -> None:
    """Compare every pair and every within-fit neighbour with verified support."""
    from .chemical_assessment import describe, finish_stage, verify_stage
    from .data import load_vocabulary

    verify_stage(output, "specificity")
    vocabulary = load_vocabulary(Path(protocol["prepared"]) / "data")
    feature_data = np.load(output / "primary_50_enrichment.npz", allow_pickle=False)
    effects = (feature_data["count"] - feature_data["expected_count"]) / np.maximum(
        1, feature_data["support"]
    )[:, None]
    fits = protocol["fits"]
    matrices = [load_beta(f, len(vocabulary)) for f in fits]
    inventories = [read_jsonl(output / f"inventory/{f['id']}.jsonl") for f in fits]
    top_words = [np.argsort(-b, axis=1, kind="stable")[:, :20] for b in matrices]
    offsets = np.cumsum([0] + [len(b) for b in matrices])
    redundant, redundancy_summaries = [], []
    for fit, beta, inventory in zip(fits, matrices, inventories, strict=True):
        normalized, _ = normalize_rows(beta)
        similarities = np.clip(normalized @ normalized.T, 0, 1)
        np.fill_diagonal(similarities, -np.inf)
        nearest = np.argmax(similarities, axis=1)
        profiles = [
            None if r["annotation_maccs"] is None else tuple(r["annotation_maccs"])
            for r in inventory
        ]
        counts = Counter(p for p in profiles if p is not None)
        for k, neighbour in enumerate(nearest):
            redundant.append(
                {
                    "fit": fit["id"],
                    "model": fit["model"],
                    "topic_id": k,
                    "nearest_topic_id": int(neighbour),
                    "nearest_beta_cosine": float(similarities[k, neighbour]),
                    "identical_MAG_profile_other_topics": (
                        None if profiles[k] is None else counts[profiles[k]] - 1
                    ),
                }
            )
        redundancy_summaries.append(
            {
                "fit": fit["id"],
                "model": fit["model"],
                "nearest_beta_cosine": describe(
                    similarities[np.arange(len(beta)), nearest]
                ),
                "available_MAG_profiles": sum(p is not None for p in profiles),
                "distinct_MAG_profiles": len(counts),
                "topics_sharing_MAG_profile": sum(
                    count for count in counts.values() if count > 1
                ),
            }
        )
    matched, pair_summaries = [], []
    for i, j in combinations(range(len(fits)), 2):
        rows, cols, scores, reciprocal = match_topics(matrices[i], matrices[j])
        kind = (
            f"within_{fits[i]['model']}"
            if fits[i]["model"] == fits[j]["model"]
            else "cross_model"
        )
        pair_rows = []
        for k, ell, score, mutual in zip(rows, cols, scores, reciprocal, strict=True):
            pair_rows.append(
                {
                    "left_fit": fits[i]["id"],
                    "right_fit": fits[j]["id"],
                    "kind": kind,
                    "left_topic": int(k),
                    "right_topic": int(ell),
                    "beta_cosine": float(score),
                    "reciprocal_nearest": bool(mutual),
                    "top20_jaccard": jaccard(top_words[i][k], top_words[j][ell]),
                    "compound_jaccard": jaccard(
                        inventories[i][k]["compound_ids"],
                        inventories[j][ell]["compound_ids"],
                    ),
                    "feature_effect_cosine": profile_cosine(
                        effects[offsets[i] + k], effects[offsets[j] + ell]
                    ),
                }
            )
        matched += pair_rows
        pair_summaries.append(
            {
                "left_fit": fits[i]["id"],
                "right_fit": fits[j]["id"],
                "kind": kind,
                "reciprocal_fraction": float(reciprocal.mean()),
                **{
                    field: describe(
                        [r[field] for r in pair_rows if r[field] is not None]
                    )
                    for field in (
                        "beta_cosine",
                        "top20_jaccard",
                        "compound_jaccard",
                        "feature_effect_cosine",
                    )
                },
            }
        )
        print(
            f"Matched all {len(rows)} topics: {fits[i]['id']} / {fits[j]['id']}",
            flush=True,
        )
    write_jsonl(output / "topic_matches.jsonl", matched)
    write_jsonl(output / "redundancy.jsonl", redundant)
    summary = {
        "pairs": pair_summaries,
        "redundancy": redundancy_summaries,
        "interpretation": (
            "Dependent fit-pair comparisons, not independent chemical replication."
        ),
    }
    write_json(output / "stability_summary.json", summary)
    finish_stage(
        output,
        "stability",
        [
            output / name
            for name in (
                "topic_matches.jsonl",
                "redundancy.jsonl",
                "stability_summary.json",
            )
        ],
        summary,
    )
