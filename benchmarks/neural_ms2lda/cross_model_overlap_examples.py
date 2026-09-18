"""Deterministic, explicitly illustrative spectral counterparts for inspection."""

from __future__ import annotations

import numpy as np

from .motif_reference import mass_overlap


def quantile_example(rows: np.ndarray, field: str, quantile: float):
    """Pick the score nearest a prespecified quantile, then stable fit/topic IDs."""
    if not len(rows) or not np.isfinite(rows[field]).all():
        raise ValueError("example quantiles need finite nonempty scores")
    value = float(np.quantile(rows[field], quantile))
    order = np.lexsort(
        (
            rows["source_topic"],
            rows["target_fit"],
            rows["source_fit"],
            abs(rows[field] - value),
        )
    )
    return rows[order[0]], value


def spectral_example(fit: dict, topic: int, vocabulary: list[str]) -> dict:
    """Retain typed top-20 beta weights, not simulated experimental intensities."""
    inventory = fit["inventory"][topic]
    return {
        "fit": fit["specification"]["id"],
        "model": fit["specification"]["model"],
        "topic_id": int(topic),
        "support_compounds": len(inventory["compound_ids"]),
        "compound_ids": inventory["compound_ids"],
        "MAG_available": inventory["annotation_maccs"] is not None,
        "words": [
            {"token": vocabulary[w], "probability": float(fit["beta"][topic, w])}
            for w in fit["top_words"][topic]
        ],
    }


def example_overlap(source: dict, target: dict, tolerances: list[float]) -> list:
    """Report direct one-to-one mass overlap separately for fragments and losses."""
    rows = []
    for channel in ("frag", "loss"):
        a, b = [
            np.array(
                [
                    float(w["token"].split("@")[1])
                    for w in spectrum["words"]
                    if w["token"].startswith(channel + "@")
                ]
            )
            for spectrum in (source, target)
        ]
        for tolerance in tolerances:
            rows.append(
                {
                    "channel": channel,
                    "tolerance_da": tolerance,
                    "matched": mass_overlap(a, b, tolerance),
                    "source_words": len(a),
                    "target_words": len(b),
                }
            )
    return rows


def select_examples(
    rows: np.ndarray, fits: list[dict], vocabulary: list[str], protocol: dict
) -> dict:
    """Display median/upper-score pairs plus one-to-two spectral correspondence.

    Selection uses only spectral score and recurrence/MAG availability. It does
    not select on chemical agreement, reference annotations, or visual appeal.
    Repeated examples are permitted and disclosed rather than silently replaced
    to improve the story. Exact source/target IDs remain in the evidence.
    """
    config = protocol["example_selection"]
    cohort = protocol["cohorts"].index(config["cohort"])
    metric = protocol["similarities"].index(config["similarity"])
    source_models = np.array([f["specification"]["model"] for f in fits])
    subset = rows[(rows["cohort"] == cohort) & (rows["similarity"] == metric)]
    examples = []
    for model in ("selected", "tomotopy"):
        cross = subset[
            (source_models[subset["source_fit"]] == model)
            & (source_models[subset["target_fit"]] != model)
        ]
        selections = [("best_score", q) for q in config["display_quantiles"]]
        if model == "selected":
            selections.append(("second_score", 0.9))
        for field, quantile in selections:
            row, target_quantile = quantile_example(cross, field, quantile)
            a, b = int(row["source_fit"]), int(row["target_fit"])
            source = spectral_example(fits[a], int(row["source_topic"]), vocabulary)
            targets = [int(row["target_topic"])]
            if field == "second_score":
                targets.append(int(row["second_topic"]))
            target_spectra = [spectral_example(fits[b], t, vocabulary) for t in targets]
            examples.append(
                {
                    "direction": f"{model}_to_{source_models[b]}",
                    "selection_field": field,
                    "selection_quantile": quantile,
                    "pooled_quantile_score": target_quantile,
                    "best_score": float(row["best_score"]),
                    "second_score": float(row["second_score"]),
                    "source": source,
                    "targets": target_spectra,
                    "direct_overlaps": [
                        example_overlap(source, t, config["mass_tolerances_da"])
                        for t in target_spectra
                    ],
                    "matched_metrics": {
                        field: float(row[field]) if np.isfinite(row[field]) else None
                        for field in (
                            "fragment_cosine",
                            "loss_cosine",
                            "compound_jaccard",
                            "feature_effect_cosine",
                        )
                    },
                }
            )
    return {"selection": config, "examples": examples}
