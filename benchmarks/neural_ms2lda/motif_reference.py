"""Typed MotifDB parsing and corroborative, non-expert reference matching.

Manual labels are retained as reference descriptions, not verified identities
of learned motifs. Spec2Vec correspondence and MAG share representation/library
history and are correlated evidence. A missing hit does not demonstrate novelty.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .chemical_assessment_io import read_jsonl
from .motif_correspondence import load_beta, normalize_rows
from .utils import read_json, write_json, write_jsonl


def parse_channel(masses: list, intensities: list) -> list[tuple[float, float]]:
    """Validate paired null/NaN padding; merge duplicate masses; drop zero weight.

    The local MotifDB export uses nonstandard JSON NaN, not JSON null, to pad
    the other channel's rows. Only paired missing mass/intensity is padding;
    one missing value or infinity still makes the reference malformed.
    """
    if len(masses) != len(intensities):
        raise ValueError("mass/intensity channel lengths differ")
    peaks = defaultdict(list)
    for mass, intensity in zip(masses, intensities, strict=True):
        missing_mass = mass is None or math.isnan(float(mass))
        missing_intensity = intensity is None or math.isnan(float(intensity))
        if missing_mass and missing_intensity:
            continue
        if missing_mass or missing_intensity:
            raise ValueError("unpaired null mass/intensity")
        m, weight = float(mass), float(intensity)
        if not math.isfinite(m) or not math.isfinite(weight) or m <= 0 or weight < 0:
            raise ValueError("invalid mass/intensity")
        if weight > 0:
            peaks[m].append(weight)
    return [(m, math.fsum(w)) for m, w in sorted(peaks.items())]


def typed_signature(fragments: list, losses: list) -> tuple:
    """Exact typed mass/intensity signature after joint maximum normalization."""
    maximum = max((p[1] for p in fragments + losses), default=0)
    if maximum <= 0:
        raise ValueError("empty reference spectrum")
    return tuple(
        (channel, float(m).hex(), float(w / maximum).hex())
        for channel, peaks in (("fragment", fragments), ("loss", losses))
        for m, w in peaks
    )


def parse_references(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Deduplicate spectral signatures, preserving source IDs and label conflicts."""
    references, excluded = {}, []
    for path in paths:
        data = read_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("ms2"), list):
            raise ValueError(f"unexpected MotifDB schema: {path}")
        for index, row in enumerate(data["ms2"]):
            provenance = {
                "source_file": str(path),
                "source_row": index,
                **{
                    key: row.get(key)
                    for key in (
                        "motif_id",
                        "motifset",
                        "short_annotation",
                        "annotation",
                        "paper_url",
                        "analysis_polarity",
                    )
                },
            }
            try:
                if (
                    "positive" not in str(row.get("analysis_polarity", "")).lower()
                    or row.get("charge") != 1
                ):
                    raise ValueError(
                        "not an explicitly positive singly charged reference"
                    )
                fragments = parse_channel(row["frag_mz"], row["frag_intens"])
                losses = parse_channel(row["loss_mz"], row["loss_intens"])
                signature = typed_signature(fragments, losses)
            except (KeyError, TypeError, ValueError) as error:
                excluded.append({**provenance, "reason": str(error)})
                continue
            if signature not in references:
                identity = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
                maximum = max(w for _, w in fragments + losses)
                references[signature] = {
                    "reference_id": identity,
                    "fragments": [(m, w / maximum) for m, w in fragments],
                    "losses": [(m, w / maximum) for m, w in losses],
                    "sources": [],
                }
            references[signature]["sources"].append(provenance)
    return sorted(references.values(), key=lambda r: r["reference_id"]), excluded


def reference_spectrum(reference: dict):
    """Use the repository's spectrum/document types; fragments and losses separate."""
    from MS2LDA.Mass2Motif import Mass2Motif

    fragments = np.array(reference["fragments"], dtype=float).reshape(-1, 2)
    losses = np.array(reference["losses"], dtype=float).reshape(-1, 2)
    return Mass2Motif(
        fragments[:, 0],
        fragments[:, 1],
        losses[:, 0],
        losses[:, 1],
        metadata={"id": reference["reference_id"], "charge": 1},
    )


def checked_embeddings(model, spectra: list) -> tuple[np.ndarray, list[dict]]:
    """Use existing Spec2Vec calculation but reject zero/nonfinite embeddings.

    Weighted vocabulary coverage is sum sqrt(intensity) for known words divided
    by that quantity for all words, consistent with the fixed embedding power.
    It measures representation coverage, not reference-match correctness.
    """
    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings
    from MS2LDA.Mass2MotifDocument import Mass2MotifDocument

    embeddings = calc_embeddings(model, spectra)
    normalized, valid = normalize_rows(embeddings)
    rows = []
    for spectrum, good in zip(spectra, valid, strict=True):
        document = Mass2MotifDocument(spectrum)
        weights = np.sqrt(np.asarray(document.weights))
        known = np.array([word in model.model.wv for word in document.words])
        rows.append(
            {
                "scorable": bool(good),
                "words": len(known),
                "known_words": int(known.sum()),
                "weighted_vocabulary_coverage": (
                    float(weights[known].sum() / weights.sum())
                    if weights.sum() > 0
                    else None
                ),
                "reason": None if good else "zero_or_nonfinite_embedding",
            }
        )
    return normalized, rows


def mass_overlap(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    """Maximum cardinality one-to-one matching of sorted masses within tolerance.

    Equal-width one-dimensional compatibility intervals permit the two-pointer
    greedy algorithm: matching the earliest compatible pair leaves the largest
    feasible suffix. No fragment can match a neutral loss (caller separates
    channels). A 1e-12 arithmetic tolerance protects exact decimal boundaries.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("nonnegative finite mass tolerance required")
    a, b = np.sort(left), np.sort(right)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("masses must be finite")
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance + 1e-12:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return matched


def direct_overlaps(query, reference, tolerances: list[float]) -> list[dict]:
    """Return count and both denominators, never interpreting absence as zero fit."""
    rows = []
    for tolerance in tolerances:
        for channel in ("peaks", "losses"):
            a, b = getattr(query, channel).mz, getattr(reference, channel).mz
            matched = mass_overlap(a, b, tolerance)
            rows.append(
                {
                    "tolerance_da": tolerance,
                    "channel": "fragment" if channel == "peaks" else "loss",
                    "matched": matched,
                    "query_features": len(a),
                    "reference_features": len(b),
                    "query_coverage": matched / len(a) if len(a) else None,
                    "reference_coverage": matched / len(b) if len(b) else None,
                }
            )
    return rows


def run_reference_assessment(protocol: dict, output: Path) -> None:
    """Rank every scorable reference; retain top-five detailed typed overlaps."""
    from MS2LDA.Add_On.Spec2Vec.annotation import load_s2v_model

    from .chemical_assessment import describe, finish_stage, verify_stage
    from .data import load_vocabulary
    from .mag import topic_spectra

    verify_stage(output, "specificity")
    references, excluded = parse_references([Path(p) for p in protocol["references"]])
    spectra = [reference_spectrum(r) for r in references]
    if not spectra:
        raise ValueError("no parseable positive reference spectra")
    model = load_s2v_model(protocol["spec2vec_model"])
    ref_embeddings, coverage = checked_embeddings(model, spectra)
    for r, c in zip(references, coverage, strict=True):
        r["embedding"] = c
    valid = np.array([r["scorable"] for r in coverage])
    if not valid.any():
        raise ValueError("no scorable reference embeddings")
    vocabulary = load_vocabulary(Path(protocol["prepared"]) / "data")
    hits, queries, full_scores, fit_summary = [], [], [], []
    for fit in protocol["fits"]:
        beta = load_beta(fit, len(vocabulary))
        query_spectra = topic_spectra(beta, vocabulary, top_n=20, significant_digits=2)
        embeddings, query_coverage = checked_embeddings(model, query_spectra)
        similarity = np.clip(embeddings @ ref_embeddings.T, -1, 1)
        similarity[:, ~valid] = np.nan
        similarity[~np.array([r["scorable"] for r in query_coverage])] = np.nan
        full_scores.append(similarity)
        support = read_jsonl(output / f"inventory/{fit['id']}.jsonl")
        best = []
        for k, c in enumerate(query_coverage):
            queries.append(
                {"fit": fit["id"], "model": fit["model"], "topic_id": k, **c}
            )
            if not c["scorable"]:
                continue
            rank = np.argsort(
                -np.where(np.isfinite(similarity[k]), similarity[k], -np.inf),
                kind="stable",
            )
            for position, index in enumerate(
                rank[: min(protocol["reference_ranked_hits"], int(valid.sum()))],
                start=1,
            ):
                overlaps = direct_overlaps(
                    query_spectra[k],
                    spectra[index],
                    protocol["reference_tolerances_da"],
                )
                row = {
                    "fit": fit["id"],
                    "model": fit["model"],
                    "topic_id": k,
                    "rank": position,
                    "reference_id": references[index]["reference_id"],
                    "reference_index": int(index),
                    "spec2vec_cosine": float(similarity[k, index]),
                    "direct_overlaps": overlaps,
                    "support_compounds": len(support[k]["compound_ids"]),
                }
                hits.append(row)
                if position == 1:
                    best.append(row)
        fit_summary.append(
            {
                "fit": fit["id"],
                "model": fit["model"],
                "scorable_topics": len(best),
                "top_reference_cosine": describe([r["spec2vec_cosine"] for r in best]),
                "top_reference_with_any_direct_overlap": sum(
                    any(
                        o["matched"] > 0
                        for o in r["direct_overlaps"]
                        if o["tolerance_da"] == 0.01
                    )
                    for r in best
                ),
                "distinct_top_references": len({r["reference_id"] for r in best}),
                "top_reference_query_coverage": describe(
                    [
                        sum(
                            o["matched"]
                            for o in r["direct_overlaps"]
                            if o["tolerance_da"] == 0.01
                        )
                        / 20
                        for r in best
                    ]
                ),
            }
        )
        print(f"Ranked references for {fit['id']}", flush=True)
    write_json(
        output / "references.json", {"references": references, "excluded": excluded}
    )
    write_jsonl(output / "reference_hits.jsonl", hits)
    write_jsonl(output / "query_embeddings_coverage.jsonl", queries)
    np.savez_compressed(
        output / "reference_similarities.npz", cosine=np.concatenate(full_scores)
    )
    summary = {
        "raw_reference_records": sum(
            len(read_json(Path(p))["ms2"]) for p in protocol["references"]
        ),
        "unique_reference_signatures": len(references),
        "excluded_records": len(excluded),
        "scorable_references": int(valid.sum()),
        "conflicting_label_signatures": sum(
            len({(s["short_annotation"], s["annotation"]) for s in r["sources"]}) > 1
            for r in references
        ),
        "fits": fit_summary,
    }
    finish_stage(
        output,
        "references",
        [
            output / name
            for name in (
                "references.json",
                "reference_hits.jsonl",
                "query_embeddings_coverage.jsonl",
                "reference_similarities.npz",
            )
        ],
        summary,
    )
