"""Fast deterministic software-validation smoke mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import write_json
from .data import (
    PeakGroup,
    SpectrumRecord,
    assign_scaffold_splits,
    audit_split_disjointness,
    build_training_vocabulary,
    completion_document,
)
from .mag import audit_mag_exclusion
from .metrics import (
    active_topic_metrics,
    calculate_sos,
    convergence_metrics,
    document_completion_nll,
    optimal_topic_matching,
    top_word_diversity,
    word_cooccurrence_npmi,
)


def _records() -> list[SpectrumRecord]:
    rows = []
    for compound in range(18):
        connectivity = f"CONNECTIVITY{compound:03d}"
        scaffold = f"scaffold:ring-{compound // 2:02d}"
        for replicate in range(2 if compound % 4 == 0 else 1):
            groups = tuple(
                PeakGroup(
                    original_index=peak,
                    mz=50.0 + peak + compound,
                    intensity=1.0 - peak * 0.1,
                    tokens=(
                        f"frag@{50 + peak + compound}.0",
                        f"loss@{150 - peak - compound}.0",
                    ),
                )
                for peak in range(4)
            )
            identifier = f"synthetic-{compound}-{replicate}"
            rows.append(
                SpectrumRecord(
                    spectrum_id=identifier,
                    feature_id=identifier,
                    smiles=f"synthetic-smiles-{compound}",
                    supplied_inchikey=connectivity,
                    connectivity_key=connectivity,
                    scaffold_key=scaffold,
                    split_group=scaffold,
                    precursor_mz=200.0,
                    peak_groups=groups,
                    declared_num_peaks=4,
                    parsed_num_peaks=4,
                    compound_name="synthetic",
                    metadata={},
                )
            )
    return rows


def run_smoke(output_path: str | Path | None = None) -> dict[str, Any]:
    """Exercise split, completion, metrics, matching, and MAG exclusion."""
    records = _records()
    assignments, split_summary = assign_scaffold_splits(
        records, fractions=(0.6, 0.2, 0.2), seed=42
    )
    leakage = audit_split_disjointness(records, assignments)
    vocabulary, vocabulary_summary = build_training_vocabulary(
        records, assignments, min_df=1, min_cf=0, rm_top=0
    )
    test = [row for row in records if assignments[row.spectrum_id] == "test"]
    completions = [
        completion_document(row, observed_fraction=0.5, seed=42) for row in test
    ]
    rng = np.random.default_rng(42)
    topics = 4
    beta = rng.dirichlet(np.ones(len(vocabulary)), size=topics)
    theta = rng.dirichlet(np.ones(topics), size=len(test))
    refined = 0.95 * theta + 0.05 / topics
    metrics = {
        "document_completion": document_completion_nll(
            theta,
            beta,
            [row.completion_words for row in completions],
            vocabulary,
        ),
        "active_topics": active_topic_metrics(
            theta, document_threshold=0.1, corpus_threshold=0.1
        ),
        "top_word_diversity": top_word_diversity(beta, top_n=3),
        "npmi": word_cooccurrence_npmi(
            beta,
            [row.words for row in records if assignments[row.spectrum_id] == "train"],
            vocabulary,
            top_n=3,
        ),
        "convergence": convergence_metrics(refined, theta),
        "matching": optimal_topic_matching(beta, beta[:, ::-1][:, ::-1], top_n=3),
        "sos": calculate_sos(np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0])),
    }
    heldout_connectivity = {
        row.connectivity_key
        for row in records
        if assignments[row.spectrum_id] != "train"
    }
    retained_connectivity = [
        row.connectivity_key
        for row in records
        if assignments[row.spectrum_id] == "train"
    ]
    result = {
        "mode": "deterministic_synthetic_smoke",
        "software_validation_only": True,
        "chemical_evidence": False,
        "seed": 42,
        "spectra": len(records),
        "split": split_summary,
        "leakage_audit": leakage,
        "vocabulary": vocabulary_summary,
        "completion_peak_group_atomicity": all(
            not (
                {group.original_index for group in row.observed_groups}
                & {group.original_index for group in row.completion_groups}
            )
            for row in completions
        ),
        "mag_exclusion": audit_mag_exclusion(
            heldout_connectivity, retained_connectivity
        ),
        "metrics": metrics,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
