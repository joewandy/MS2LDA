"""Directed motif recovery without forcing a one-to-one correspondence.

For source topic k and target topic l, S[k,l] is a spectral cosine. The
best score m[k] = max_l S[k,l] determines recovery C(t) = mean(m >= t).
The second-largest score determines whether there are at least two candidate
counterparts. These two order statistics replace a threshold-specific graph
construction: no arbitrary edge cutoff or clustering algorithm is required.

All functions operate on fixed fitted motifs. Nothing here trains a model or
uses chemical agreement to select the closest spectral counterpart.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

import numpy as np

from .motif_correspondence import jaccard, normalize_rows

MATCH_DTYPE = np.dtype(
    [
        (name, "i2")
        for name in (
            "source_fit",
            "target_fit",
            "cohort",
            "similarity",
            "source_topic",
            "target_topic",
            "second_topic",
            "source_compounds",
            "target_compounds",
        )
    ]
    + [
        (name, "f8")
        for name in (
            "best_score",
            "second_score",
            "full_beta_cosine",
            "fragment_cosine",
            "loss_cosine",
            "top20_jaccard",
            "compound_jaccard",
            "feature_effect_cosine",
        )
    ]
    + [("reciprocal", "?"), ("source_mag", "?"), ("target_mag", "?")]
)
METRICS = (
    "best_score",
    "full_beta_cosine",
    "fragment_cosine",
    "loss_cosine",
    "top20_jaccard",
    "compound_jaccard",
    "feature_effect_cosine",
)


def cohort_topics(inventory: list[dict], name: str) -> np.ndarray:
    """Select each inventory independently BEFORE searching for counterparts."""
    support = np.array([len(r["compound_ids"]) for r in inventory])
    available = np.array([r["annotation_maccs"] is not None for r in inventory])
    masks = {
        "all": np.ones(len(inventory), dtype=bool),
        "recurring": support >= 2,
        "recurring_evaluable": (support >= 2) & available,
    }
    if name not in masks:
        raise ValueError(f"unknown overlap cohort: {name}")
    return np.flatnonzero(masks[name])


def spectral_vectors(beta: np.ndarray, fragment: np.ndarray) -> dict:
    """Normalize complete and channel-restricted probability vectors separately.

    A fragment and a neutral loss never share a coordinate, even at equal mass.
    Both saved model families give positive total probability to both channels.
    Reject a missing/zero channel rather than pretending its cosine is zero.
    """
    if fragment.dtype != bool or fragment.shape != (beta.shape[1],):
        raise ValueError("fragment mask must match the ordered vocabulary")
    result = {}
    for name, matrix in (
        ("full", beta),
        ("fragment", beta[:, fragment]),
        ("loss", beta[:, ~fragment]),
    ):
        values, valid = normalize_rows(matrix)
        if not valid.all():
            raise ValueError(f"nonfinite or zero {name} spectral vector")
        result[name] = values
    return result


def similarity_matrices(left: dict, right: dict) -> dict:
    """Compute full-beta cosine and the equal-channel sensitivity, both in [0,1]."""
    matrices = {
        name: np.clip(left[name] @ right[name].T, 0, 1)
        for name in ("full", "fragment", "loss")
    }
    matrices["balanced"] = (matrices["fragment"] + matrices["loss"]) / 2
    return matrices


def nearest_two(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return best/second IDs and reciprocal flags; ties use lowest column ID.

    Target topics may be reused. A single target has no second counterpart
    (ID -1), rather than a fictional zero-similarity topic. Empty inventories
    cannot support a recovery denominator and are rejected at this boundary.
    """
    s = np.asarray(similarity, dtype=float)
    if s.ndim != 2 or min(s.shape) == 0 or not np.isfinite(s).all():
        raise ValueError("nearest search needs a nonempty finite matrix")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("spectral cosine must lie in [0,1]")
    best = s.argmax(axis=1)
    remaining = s.copy()
    remaining[np.arange(len(s)), best] = -np.inf
    second = remaining.argmax(axis=1) if s.shape[1] > 1 else np.full(len(s), -1)
    reciprocal = s.argmax(axis=0)[best] == np.arange(len(s))
    return best, second, reciprocal


def directed_records(
    source: dict,
    target: dict,
    matrices: dict,
    *,
    source_id: int,
    target_id: int,
    cohort_id: int,
    similarity_id: int,
    cohorts: list[str],
) -> np.ndarray:
    """Select by spectra, then measure chemical/support agreement on those pairs.

    Integer fit/cohort/similarity codes refer to protocol list positions.
    NaN is reserved for undefined set/profile agreement; it is not zero
    chemical evidence. Arrays are saved without Python objects or pickles.
    """
    a = cohort_topics(source["inventory"], cohorts[cohort_id])
    b = cohort_topics(target["inventory"], cohorts[cohort_id])
    key = ("full", "balanced")[similarity_id]
    scores = matrices[key][np.ix_(a, b)]
    best, second, reciprocal = nearest_two(scores)
    targets = b[best]
    rows = np.zeros(len(a), dtype=MATCH_DTYPE)
    for field, value in (
        ("source_fit", source_id),
        ("target_fit", target_id),
        ("cohort", cohort_id),
        ("similarity", similarity_id),
        ("source_topic", a),
        ("target_topic", targets),
        ("second_topic", np.where(second >= 0, b[np.maximum(second, 0)], -1)),
        ("best_score", scores[np.arange(len(a)), best]),
        (
            "second_score",
            np.where(
                second >= 0, scores[np.arange(len(a)), np.maximum(second, 0)], np.nan
            ),
        ),
        ("reciprocal", reciprocal),
    ):
        rows[field] = value
    for field, matrix_name in (
        ("full_beta_cosine", "full"),
        ("fragment_cosine", "fragment"),
        ("loss_cosine", "loss"),
    ):
        rows[field] = matrices[matrix_name][a, targets]
    for index, (k, ell) in enumerate(zip(a, targets, strict=True)):
        left, right = source["inventory"][k], target["inventory"][ell]
        rows["source_compounds"][index] = len(left["compound_ids"])
        rows["target_compounds"][index] = len(right["compound_ids"])
        rows["source_mag"][index] = left["annotation_maccs"] is not None
        rows["target_mag"][index] = right["annotation_maccs"] is not None
        for field, x, y in (
            ("compound_jaccard", left["compound_ids"], right["compound_ids"]),
            ("top20_jaccard", source["top_words"][k], target["top_words"][ell]),
        ):
            value = jaccard(x, y)
            rows[field][index] = np.nan if value is None else value
    valid = source["effect_valid"][a] & target["effect_valid"][targets]
    effects = np.einsum("ij,ij->i", source["effects"][a], target["effects"][targets])
    rows["feature_effect_cosine"] = np.where(valid, np.clip(effects, -1, 1), np.nan)
    return rows


def recovery_curves(rows: np.ndarray, thresholds: np.ndarray) -> dict:
    """Complementary 0/1/2+ neighbour fractions use a fixed source denominator.

    C(t) = mean(best >= t), M(t) = mean(second >= t), so fractions are
    unmatched = 1-C, single = C-M, multiple = M. A missing second target
    never counts. This is graph degree, not a fragmentation split/merge test.
    """
    if not len(rows) or not np.isfinite(rows["best_score"]).all():
        raise ValueError("coverage needs a nonempty finite source inventory")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a finite vector")
    best = rows["best_score"][:, None] >= thresholds
    second = rows["second_score"][:, None] >= thresholds
    if np.any(second & ~best):
        raise ValueError("second-best score cannot exceed best score")
    recovered, multiple = best.mean(axis=0), second.mean(axis=0)
    return {
        "coverage": recovered.tolist(),
        "unmatched": (1 - recovered).tolist(),
        "single": (recovered - multiple).tolist(),
        "multiple": multiple.tolist(),
    }


def describe_finite(values: np.ndarray) -> dict:
    """Retain the undefined denominator; no NaN becomes a zero measurement."""
    good = np.asarray(values)[np.isfinite(values)]
    return {
        "n": len(good),
        "undefined": len(values) - len(good),
        "mean": float(good.mean()) if len(good) else None,
        "median": float(np.median(good)) if len(good) else None,
    }


def aggregate_defined(values: list) -> dict:
    """Summarize available fit statistics while exposing missing fit counts."""
    valid = [value for value in values if value is not None]
    return {
        "n": len(valid),
        "undefined": len(values) - len(valid),
        "mean": mean(valid) if valid else None,
        "sample_sd": stdev(valid) if len(valid) > 1 else None,
    }


def summarize_matches(rows: np.ndarray, protocol: dict, fits: list[dict]) -> dict:
    """Average targets within each source fit; never pool runs before matching.

    Three source-fit summaries are the descriptive aggregation unit. Their
    shared target runs still create dependence, so the SD is not a confidence
    interval and the 9 cross-model pairs are not independent replicates.
    """
    grid = protocol["threshold_grid"]
    thresholds = np.linspace(grid["minimum"], grid["maximum"], grid["points"])
    pair_summaries, source_groups = [], defaultdict(list)
    for i, source in enumerate(fits):
        for j, target in enumerate(fits):
            if i == j:
                continue
            direction = f"{source['model']}_to_{target['model']}"
            for c, cohort in enumerate(protocol["cohorts"]):
                for s, similarity in enumerate(protocol["similarities"]):
                    subset = rows[
                        (rows["source_fit"] == i)
                        & (rows["target_fit"] == j)
                        & (rows["cohort"] == c)
                        & (rows["similarity"] == s)
                    ]
                    item = {
                        "source_fit": source["id"],
                        "target_fit": target["id"],
                        "direction": direction,
                        "cohort": cohort,
                        "similarity": similarity,
                        "source_topics": len(subset),
                        "target_topics": int(
                            protocol["cohort_sizes"][target["id"]][cohort]
                        ),
                        "reciprocal_fraction": float(subset["reciprocal"].mean()),
                        "metrics": {m: describe_finite(subset[m]) for m in METRICS},
                        "curves": recovery_curves(subset, thresholds),
                    }
                    pair_summaries.append(item)
                    source_groups[(direction, cohort, similarity, source["id"])].append(
                        item
                    )
    source_summaries, final_groups = [], defaultdict(list)
    for (direction, cohort, similarity, source), pairs in source_groups.items():
        item = {
            "direction": direction,
            "cohort": cohort,
            "similarity": similarity,
            "source_fit": source,
            "targets": len(pairs),
            "source_topics": pairs[0]["source_topics"],
            "metrics": {
                m: aggregate_defined([p["metrics"][m]["median"] for p in pairs])["mean"]
                for m in METRICS
            },
            "metric_target_counts": {
                m: sum(p["metrics"][m]["median"] is not None for p in pairs)
                for m in METRICS
            },
            "curves": {
                k: np.mean([p["curves"][k] for p in pairs], axis=0).tolist()
                for k in pairs[0]["curves"]
            },
        }
        source_summaries.append(item)
        final_groups[(direction, cohort, similarity)].append(item)
    groups = [
        {
            "direction": direction,
            "cohort": cohort,
            "similarity": similarity,
            "source_fits": len(sources),
            "metrics": {
                m: aggregate_defined([r["metrics"][m] for r in sources])
                for m in METRICS
            },
        }
        for (direction, cohort, similarity), sources in final_groups.items()
    ]
    return {
        "thresholds": thresholds.tolist(),
        "pairs": pair_summaries,
        "sources": source_summaries,
        "groups": groups,
    }
