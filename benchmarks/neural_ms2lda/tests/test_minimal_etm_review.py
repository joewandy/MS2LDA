"""Decision tolerances must not conceal missing runs or metric trade-offs."""

from copy import deepcopy

import pytest

from benchmarks.neural_ms2lda.review_tables import (
    real_gate,
    synthetic_gate,
)
from scripts.generate_minimal_etm_review import (
    SYNTHETIC_VARIANTS,
    synthetic_groups,
)


def synthetic_rows():
    return [
        {
            "data": "synthetic",
            "topics": k,
            "variant": variant,
            "seed": seed,
            "epochs": 120,
            "batch_size": 200,
            "momentum": 0.9,
            "count_scaling": "raw_counts",
            "reused_training_weights": False,
            "beta_recovery": 0.6,
            "theta_recovery": 0.9,
            "median_effective_topics": 2.0,
            "catastrophic_duplicate": False,
            "recovered_motifs": 17,
        }
        for k in (36, 128)
        for variant in SYNTHETIC_VARIANTS
        for seed in (11, 23, 37)
    ]


def test_synthetic_review_rejects_missing_duplicate_or_mismatched_runs():
    rows = synthetic_rows()
    groups = synthetic_groups(rows)
    assert len(groups) == 8
    with pytest.raises(ValueError, match="seeds"):
        synthetic_groups(rows[:-1])
    with pytest.raises(ValueError, match="seeds"):
        synthetic_groups([*rows, rows[-1]])
    rows[-1]["momentum"] = 0.99
    with pytest.raises(ValueError, match="recipe"):
        synthetic_groups(rows)


def test_high_k_retention_and_any_catastrophic_run_are_binding():
    current = synthetic_rows()[:3]
    candidate = deepcopy(current)
    assert synthetic_gate(candidate, current, 128)["passed"]
    candidate[0]["recovered_motifs"] = 11
    assert not synthetic_gate(candidate, current, 128)["passed"]
    assert synthetic_gate(candidate, current, 36)["passed"]
    candidate[0]["catastrophic_duplicate"] = True
    assert not synthetic_gate(candidate, current, 36)["passed"]


def real_row():
    return {
        "evaluable_motifs": 672,
        "useful_motifs": 400,
        "mean_sos": 0.636,
        "completion_nll": 9.515,
        "median_effective_topics": 3.66,
        "unique_top1_topics": 838,
        "catastrophic_duplicate": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("useful_motifs", 379),
        ("evaluable_motifs", 638),
        ("median_effective_topics", 5.01),
        ("unique_top1_topics", 799),
        ("mean_sos", 0.615),
        ("completion_nll", 9.706),
        ("catastrophic_duplicate", True),
    ],
)
def test_each_real_requirement_can_independently_block_promotion(field, value):
    reference = real_row()
    candidate = {**reference, field: value}
    assert not real_gate(candidate, [reference, reference])["passed"]


def test_both_current_references_must_be_retained():
    reference = real_row()
    assert real_gate(reference, [reference, reference])["passed"]
    stronger = {**reference, "useful_motifs": 430}
    assert not real_gate(reference, [reference, stronger])["passed"]
    with pytest.raises(ValueError, match="non-finite"):
        real_gate({**reference, "mean_sos": float("nan")}, [reference])
