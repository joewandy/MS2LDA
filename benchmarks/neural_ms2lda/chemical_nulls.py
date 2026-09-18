"""Compound-conditional null models, with equations expressed as small functions.

Arrays use T topics, N distinct compounds, F molecular fingerprint positions.
``members[k]`` is C_k: local compound indices, without duplicates. ``blocks``
partition 0..N-1 into acquisition/mass strata. A compound may belong to several
topics, but the SAME shuffled fingerprint is used for it in every topic/fit.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit
from scipy.stats import beta as beta_distribution
from scipy.stats import hypergeom


def validate_partition(blocks: list[np.ndarray], n: int) -> None:
    """Reject overlapping, missing, empty or out-of-range null strata."""
    if not blocks or any(len(b) == 0 for b in blocks):
        raise ValueError("strata must be nonempty")
    if not np.array_equal(np.sort(np.concatenate(blocks)), np.arange(n)):
        raise ValueError("strata must partition all compounds exactly once")


def validate_members(members: list[np.ndarray], n: int) -> None:
    """A topic contains a set of compound indices, not replicated spectra."""
    for ids in members:
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n):
            raise ValueError("topic members must be unique valid compound indices")


def by_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini--Yekutieli q_i = min_{j>=rank(i)} m H_m p_(j)/j.

    All declared family slots enter m, including unavailable/constant tests
    supplied as p=1. This correction allows dependent tests only when their
    marginal p-values are valid; it cannot repair post-selection bias.
    """
    p = np.asarray(pvalues, dtype=float)
    if not p.size or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a nonempty finite array in [0,1]")
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.arange(1, len(flat) + 1, dtype=float)
    scaled = flat[order] * len(flat) * np.sum(1 / ranks) / ranks
    q = np.empty_like(flat)
    q[order] = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
    return q.reshape(p.shape)


def permutation_indices(
    rng: np.random.Generator, blocks: list[np.ndarray], batch: int, n: int
) -> np.ndarray:
    """Independent uniform within-block permutations; no bit-wise shuffling."""
    indices = np.empty((batch, n), dtype=np.int64)
    for block in blocks:
        indices[:, block] = rng.permuted(
            np.broadcast_to(block, (batch, len(block))), axis=1
        )
    return indices


@njit(cache=True)
def _permutation_accumulators(permutations, overlaps, pointers, members, observed):
    """Count inclusive integer-score tails without floating-point tie ambiguity.

    overlaps[k,c] = |A_k intersect F_c|. The SOS denominator |C_k| |A_k|
    is fixed within a topic and cancels in every tail comparison. Returning
    aggregate sums avoids storing the full B-by-T Monte Carlo score matrix.
    """
    topics = len(observed)
    hits = np.zeros(topics, dtype=np.int64)
    sums = np.zeros(topics, dtype=np.float64)
    for b in range(len(permutations)):
        for k in range(topics):
            total = 0
            for j in range(pointers[k], pointers[k + 1]):
                total += overlaps[k, permutations[b, members[j]]]
            hits[k] += total >= observed[k]
            sums[k] += total
    return hits, sums


def sos_permutations(
    fingerprints: np.ndarray,
    annotations: np.ndarray,
    available: np.ndarray,
    members: list[np.ndarray],
    blocks: list[np.ndarray],
    *,
    permutations: int,
    seed: int,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Evaluate fixed MAG consensus SOS against a compound-conditional null.

    fingerprints: N-by-F boolean matrix; annotations: T-by-F boolean matrix.
    Missing annotations are zero-filled with available=False; an available
    all-zero annotation has observed SOS=0 and an uninformative constant null.
    Expected SOS and null variance are analytic finite-population quantities.
    Wilson-like precision is not used at boundary counts: exact binomial
    Clopper--Pearson bounds describe Monte Carlo *tail probability* precision,
    not confidence in chemistry or uncertainty from fitting the topic model.
    """
    if permutations < 1 or batch_size < 1:
        raise ValueError("positive permutation and batch counts required")
    fp, ann = np.asarray(fingerprints, dtype=bool), np.asarray(annotations, dtype=bool)
    if fp.ndim != 2 or ann.ndim != 2 or fp.shape[1] != ann.shape[1]:
        raise ValueError("fingerprints and annotations must share feature columns")
    if len(members) != len(ann) or np.shape(available) != (len(ann),):
        raise ValueError(
            "annotation availability and membership must cover every topic"
        )
    validate_partition(blocks, len(fp))
    validate_members(members, len(fp))
    # Integer intersections make exact tie handling inexpensive. MACCS has
    # 167 stored bits, safely below uint16's upper bound.
    if fp.shape[1] > np.iinfo(np.uint16).max:
        raise ValueError("fingerprint too wide for integer intersection kernel")
    overlaps = ann.astype(np.uint16) @ fp.astype(np.uint16).T
    sizes = np.array([len(c) for c in members])
    denominators = sizes * ann.sum(axis=1)
    observed = np.array(
        [overlaps[k, c].sum() for k, c in enumerate(members)], dtype=np.int64
    )
    expected = np.zeros(len(ann))
    variance = np.zeros(len(ann))
    movable = np.zeros(len(ann), dtype=int)
    block_id = np.empty(len(fp), dtype=int)
    for s, block in enumerate(blocks):
        block_id[block] = s
    for k, ids in enumerate(members):
        for s, count in zip(*np.unique(block_id[ids], return_counts=True), strict=True):
            block = blocks[s]
            population = overlaps[k, block].astype(float)
            expected[k] += count * population.mean()
            if len(block) > 1:
                movable[k] += count
                variance[k] += (
                    count * (len(block) - count) / (len(block) - 1) * population.var()
                )
    pointers = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    flat_members = np.concatenate(members).astype(np.int64)
    hits, sums = np.zeros(len(ann), dtype=np.int64), np.zeros(len(ann))
    rng = np.random.default_rng(seed)
    for start in range(0, permutations, batch_size):
        indices = permutation_indices(
            rng, blocks, min(batch_size, permutations - start), len(fp)
        )
        b_hits, b_sums = _permutation_accumulators(
            indices, overlaps, pointers, flat_members, observed
        )
        hits += b_hits
        sums += b_sums
    safe_denominator = np.maximum(1, denominators)
    eligible = np.asarray(available, dtype=bool) & (sizes > 0)
    informative = eligible & (variance > 0)
    p = (1 + hits) / (1 + permutations)
    p[~informative] = 1
    # Exact binomial limits remain nonzero in width at 0/B and B/B.
    lower = np.zeros(len(ann))
    upper = np.ones(len(ann))
    mask = hits > 0
    lower[mask] = beta_distribution.ppf(
        0.025, hits[mask], permutations - hits[mask] + 1
    )
    mask = hits < permutations
    upper[mask] = beta_distribution.ppf(
        0.975, hits[mask] + 1, permutations - hits[mask]
    )
    return {
        "eligible": eligible,
        "informative": informative,
        "support": sizes,
        "movable": movable,
        "observed": observed / safe_denominator,
        "expected": expected / safe_denominator,
        "excess": (observed - expected) / safe_denominator,
        "null_sd": np.sqrt(variance) / safe_denominator,
        "mc_mean": sums / permutations / safe_denominator,
        "tail_hits": hits,
        "p": p,
        "q": by_adjust(p),
        "tail_mc_lower": lower,
        "tail_mc_upper": upper,
    }


@lru_cache(maxsize=100_000)
def hypergeometric_pmf(total: int, positives: int, drawn: int) -> np.ndarray:
    """P(X=x), x=0..drawn, for sampling without replacement in one stratum."""
    if not 0 <= positives <= total or not 0 <= drawn <= total or total < 1:
        raise ValueError("invalid hypergeometric population")
    pmf = hypergeom.pmf(np.arange(drawn + 1), total, positives, drawn)
    return pmf / pmf.sum()


def stratified_tail(strata: tuple[tuple[int, int, int], ...], observed: int) -> float:
    """P(sum_s X_s >= observed) via exact finite-support convolution.

    Each tuple is (N_s, M_s, n_ks): population size, feature-positive count,
    motif support. Floating-point arithmetic approximates exact combinatorial
    probabilities; this is not a normal/chi-square/asymptotic test.
    """
    if not strata or observed < 0 or observed > sum(s[2] for s in strata):
        raise ValueError("observed count outside stratified support")
    pmf = np.array([1.0])
    for total, positives, drawn in strata:
        pmf = np.convolve(pmf, hypergeometric_pmf(total, positives, drawn))
    return float(np.clip(pmf[observed:].sum() / pmf.sum(), 0, 1))


def feature_enrichment(
    fingerprints: np.ndarray, members: list[np.ndarray], blocks: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """Feature prevalence and one-sided conditional count enrichment, T-by-F.

    expected[k,f] = sum_s n_ks M_sf/N_s; the matched-background prevalence is
    expected/|C_k|. Effect = (observed-expected)/|C_k|. Background includes the
    motif's own compounds, as required by the conditional randomization null.
    Ratios with zero background are undefined, not infinite evidence.
    """
    fp = np.asarray(fingerprints, dtype=bool)
    validate_partition(blocks, len(fp))
    validate_members(members, len(fp))
    block_id = np.empty(len(fp), dtype=int)
    population_counts = []
    for s, block in enumerate(blocks):
        block_id[block] = s
        population_counts.append(fp[block].sum(axis=0))
    shape = (len(members), fp.shape[1])
    counts = np.zeros(shape, dtype=np.uint16)
    expected, variance = np.zeros(shape), np.zeros(shape)
    p = np.ones(shape)
    sizes = np.array([len(ids) for ids in members])
    for k, ids in enumerate(members):
        if not len(ids):
            continue
        counts[k] = fp[ids].sum(axis=0)
        strata = list(zip(*np.unique(block_id[ids], return_counts=True), strict=True))
        for s, n in strata:
            total = len(blocks[s])
            prevalence = population_counts[s] / total
            expected[k] += n * prevalence
            if total > 1:
                variance[k] += (
                    n * (total - n) / (total - 1) * prevalence * (1 - prevalence)
                )
        for f in np.flatnonzero((counts[k] > 0) & (variance[k] > 0)):
            parameters = tuple(
                (len(blocks[s]), int(population_counts[s][f]), int(n))
                for s, n in strata
            )
            p[k, f] = stratified_tail(parameters, int(counts[k, f]))
    denominator = np.maximum(1, sizes)[:, None]
    return {
        "count": counts,
        "support": sizes,
        "expected_count": expected,
        "prevalence": counts / denominator,
        "background": expected / denominator,
        "effect": (counts - expected) / denominator,
        "ratio": np.divide(
            counts, expected, out=np.full(shape, np.nan), where=expected > 0
        ),
        "informative": variance > 0,
        "p": p,
        "q": by_adjust(p),
    }
