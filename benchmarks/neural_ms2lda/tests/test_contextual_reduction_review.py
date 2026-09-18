"""The staged deletion screen must enforce its recipe and complete inventory."""

from copy import deepcopy

import pytest

from benchmarks.neural_ms2lda.report_comparison import BASELINE_SEEDS
from benchmarks.neural_ms2lda.review_tables import synthetic_gate
from scripts.generate_contextual_reduction_review import (
    ATOMIC,
    DESCRIPTIVE_SYNTHETIC,
    SOS_FIELDS,
    check_real_inventory,
    current_model_report,
    failed_inventory,
    paired_tradeoffs,
    repeat_summary,
    sos_line,
    synthetic_inventory,
    validate_recipe,
)


def failure():
    return {
        "variant": "reduced_linear_evidence",
        "seed": 42,
        "training_seed": 7043,
        "topics": 1000,
        "selection_split": "validation",
        "test_matrices_loaded": False,
        "planned_epochs": 120,
        "last_completed_checkpoint_epoch": 80,
        "status": "non_finite_training_loss",
        "metrics": None,
        "source_sha256": {"worker.log": "a" * 64},
    }


def test_failure_is_visible_but_never_a_completed_or_zero_score():
    record = failure()
    failures = failed_inventory({"runs": [record]})
    key = (record["variant"], 42)
    assert failures[key]["metrics"] is None
    check_real_inventory({}, failures, {key})
    with pytest.raises(ValueError, match="complete and failed"):
        check_real_inventory({key: {}}, failures, {key})
    with pytest.raises(ValueError, match="missing or unexpected"):
        check_real_inventory({}, {}, {key})
    with pytest.raises(ValueError, match="duplicate"):
        failed_inventory({"runs": [record, record]})


@pytest.mark.parametrize(
    "field,value",
    [
        ("variant", "unknown"),
        ("training_seed", 42),
        ("test_matrices_loaded", True),
        ("selection_split", "test"),
        ("status", "running"),
        ("metrics", {"useful_motifs": 0}),
        ("last_completed_checkpoint_epoch", 120),
        ("source_sha256", {}),
    ],
)
def test_failure_boundary_and_provenance_are_enforced(field, value):
    record = failure()
    record[field] = value
    with pytest.raises(ValueError):
        failed_inventory({"runs": [record]})


def row(variant, k, seed, recovery=0.2):
    return {
        "variant": variant,
        "topics": k,
        "seed": seed,
        "training_seed": seed + 7001,
        "data": "synthetic",
        "epochs": 120,
        "batch_size": 200,
        "hidden": 100 if variant == "reduced_shallow_narrow" else 800,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 1.2e-6,
        "normalization_statistics": "not_applicable",
        "count_scaling": "raw_counts",
        "decoder": "additive_mixture",
        "beta_meaning": "emissions",
        "framework": "pytorch",
        "reused_training_weights": False,
        "beta_recovery": recovery,
        "theta_recovery": recovery,
        "median_effective_topics": 2,
        "recovered_motifs": 17,
        "catastrophic_duplicate": False,
    }


def inventory():
    return (
        [row(v, 128, s) for v in ATOMIC for s in (11, 23, 37)]
        + [
            row("reduced_scaled_full_routing", k, s)
            for k in (36, 128)
            for s in (11, 23, 37)
        ]
        + [row("reduced_shallow_narrow", k, s) for k in (36, 128) for s in (11, 23, 37)]
        + [
            row("reduced_attention_document_fixed", k, s)
            for k in (36, 128)
            for s in (11, 23, 37)
        ]
        + [row(v, 36, s) for v in DESCRIPTIVE_SYNTHETIC for s in (11, 23, 37)],
        [row("contextual", k, s, 0.8) for k in (36, 128) for s in (11, 23, 37)],
    )


def test_all_failed_groups_are_retained_and_do_not_advance():
    rows, controls = inventory()
    groups, gates, passed = synthetic_inventory(rows, controls)
    assert len(groups) == len(gates) == 16
    assert not passed
    assert all((36, v) in groups for v in DESCRIPTIVE_SYNTHETIC)
    with pytest.raises(ValueError, match="exactly"):
        synthetic_inventory(rows[:-1], controls)
    with pytest.raises(ValueError, match="exactly"):
        synthetic_inventory([*rows, rows[0]], controls)


def test_passed_groups_require_lower_k_and_unplanned_groups_are_rejected():
    rows, controls = inventory()
    variant = "reduced_linear_evidence"
    for r in rows:
        if r["variant"] == variant:
            r.update(beta_recovery=0.8, theta_recovery=0.8)
    with pytest.raises(ValueError, match="missing required"):
        synthetic_inventory(rows, controls)
    rows.extend(row(variant, 36, s, 0.8) for s in (11, 23, 37))
    assert synthetic_inventory(rows, controls)[2] == {variant}
    rows.extend(row("reduced_full_routing", 36, s) for s in (11, 23, 37))
    with pytest.raises(ValueError, match="unexpected"):
        synthetic_inventory(rows, controls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden", 100),
        ("momentum", 0.99),
        ("count_scaling", "unit_mass"),
        ("training_seed", 11),
        ("reused_training_weights", True),
        ("decoder", "product_of_experts"),
        ("beta_meaning", "prototypes"),
    ],
)
def test_no_recipe_or_semantic_drift(field, value):
    value_row = row(ATOMIC[0], 128, 11)
    validate_recipe(value_row)
    value_row[field] = value
    with pytest.raises(ValueError):
        validate_recipe(value_row)


def test_exact_ninety_percent_boundary_passes_without_relaxing_threshold():
    current = [row("contextual", 128, s, 0.8) for s in (11, 23, 37)]
    for r, recovered in zip(current, (18, 16, 16)):
        r["recovered_motifs"] = recovered
    candidate = deepcopy(current)
    for r in candidate:
        r["recovered_motifs"] = 15
    assert synthetic_gate(candidate, current, 128)["passed"]
    candidate[0]["recovered_motifs"] = 14
    assert not synthetic_gate(candidate, current, 128)["passed"]


def test_tradeoffs_report_effect_sizes_without_a_new_acceptance_rule():
    reference = {
        "completion_nll": 9.5,
        "mean_sos": 0.62,
        "evaluable_motifs": 600,
        "useful_motifs": 400,
        "unique_top1_topics": 800,
        "median_effective_topics": 4,
        "parameters": 1000,
        **{key: 300 for key in SOS_FIELDS},
    }
    candidate = dict(reference, completion_nll=9.75, useful_motifs=404, parameters=900)
    effect = paired_tradeoffs(candidate, reference)
    assert effect["nll_difference_nats_per_token"] == 0.25
    assert effect["perplexity_change_percent"] == pytest.approx(28.4025416688)
    assert effect["relative_changes_percent"]["useful_motifs"] == pytest.approx(1)
    assert effect["relative_changes_percent"]["parameters"] == pytest.approx(-10)
    assert "passed" not in effect
    assert sos_line("Reference", reference).endswith(chr(92) * 2)


def test_repeat_summary_is_descriptive_and_requires_all_three_seeds():
    rows = [
        {
            "seed": seed,
            "parameters": 49001,
            **{
                field: value
                for field in (
                    "evaluable_motifs",
                    "useful_motifs",
                    "mean_sos",
                    "completion_nll",
                    "median_effective_topics",
                    "unique_top1_topics",
                )
            },
        }
        for seed, value in zip((11, 23, 42), (1, 2, 3))
    ]
    result = repeat_summary(rows)
    assert result["completion_nll"] == {"mean": 2, "sample_sd": 1}
    with pytest.raises(ValueError, match="exactly"):
        repeat_summary(rows[:-1])
    rows[0]["parameters"] = 1234
    with pytest.raises(ValueError, match="parameter"):
        repeat_summary(rows)


def current_report_inputs():
    variant = "reduced_document_context"
    selected = [row(variant, k, s) for k in (36, 128) for s in (11, 23, 37)]
    for value in selected:
        value["completion_nll"] = 6.2
    for seed, n in zip((11, 23, 42), (1, 2, 3)):
        value = row(variant, 1000, seed)
        value.update(
            data="MSnLib validation",
            parameters=19278001,
            evaluable_motifs=650 + 10 * n,
            useful_motifs=370 + 10 * n,
            mean_sos=0.6 + 0.01 * n,
            completion_nll=9.5 + 0.01 * n,
            median_effective_topics=3 + 0.1 * n,
            unique_top1_topics=800 + n,
        )
        selected.append(value)
    baselines = [
        {
            "model": model,
            "training_seed": seed,
            "run": f"{model}_{seed}",
            "spectrum_topic_associations": 3889,
            "completion_documents": 3888,
            "completion_tokens": 1277983,
            "optimized_motifs": 900,
            "evaluable_motifs": (573 if model == "Tomotopy LDA" else 199) + n,
            "useful_motifs": (359 if model == "Tomotopy LDA" else 118) + n,
            "mean_sos": 0.642 + n * 0.001,
            "completion_nll": (9.689 if model == "Tomotopy LDA" else 8.677) + n * 0.001,
            "median_effective_topics": (4 if model == "Tomotopy LDA" else 40.46)
            + n * 0.01,
        }
        for model, seeds in BASELINE_SEEDS.items()
        for n, seed in zip((-1, 0, 1), seeds)
    ]
    return selected, baselines


def test_current_report_is_tomotopy_focused_and_never_inherits_old_model_scores():
    rows, baselines = current_report_inputs()
    # Irrelevant predecessor and alternative records must not enter the report.
    rows.append(dict(rows[-1], variant="contextual", useful_motifs=999))
    parts, summary = current_model_report(rows, baselines)
    assert summary["real"]["useful_motifs"] == {"mean": 390, "sample_sd": 10}
    assert summary["reported_split"] == "validation"
    assert summary["historical_test_results_used"] is False
    assert summary["primary_comparator"] == "Tomotopy LDA"
    assert summary["secondary_base_check"] == "canonical ETM"
    table = "\n".join(parts["validation_table"])
    assert r"Tomotopy LDA & 3 & $573.0\pm1.0$ & $359.0\pm1.0$" in table
    assert r"Plain ETM & 3 & $199.0\pm1.0$ & $118.0\pm1.0$" in table
    assert "Contextual Sparse ETM & 3 &" in table
    assert table.index("Tomotopy LDA") < table.index("Plain ETM")
    assert table.index("Plain ETM") < table.index("Contextual Sparse ETM")
    assert r"\boldsymbol{390.0\pm10.0}" in table and "--" not in table
    assert "balanced" not in table
    assert "999" not in table and "998" not in table
    macros = "\n".join(parts["macros"])
    assert "ReferenceCanonicalEffective}{40.46}" in macros
    assert "ReferenceLDAEffective}{4.00}" in macros
    assert "ReferenceLDAUsefulSD}{1.0}" in macros
    assert "ReferenceCanonicalUsefulSD}{1.0}" in macros
    assert "CurrentUsefulDifference}{+31.0}" in macros
    assert len(parts["synthetic_table"]) == 2
    assert [v.split(" & ")[0] for v in parts["stability_table"]] == [
        "7012",
        "7024",
        "7043",
    ]


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed", "recipe"])
def test_current_report_requires_exact_selected_fits_and_recipe(change):
    rows, baselines = current_report_inputs()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "wrong_seed":
        rows[0]["seed"] = 42
    else:
        rows[0]["batch_size"] = 256
    with pytest.raises(ValueError):
        current_model_report(rows, baselines)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "test", "wrong_seed", "alternative"]
)
def test_current_report_rejects_missing_baselines_and_historical_test_substitution(
    change,
):
    rows, baselines = current_report_inputs()
    if change == "missing":
        baselines.pop()
    elif change == "duplicate":
        baselines[0] = baselines[1]
    elif change == "wrong_seed":
        baselines[0]["training_seed"] = 999
    elif change == "alternative":
        baselines.append(dict(baselines[0], model="balanced ETM"))
    else:
        baselines[1]["spectrum_topic_associations"] = 7777
    with pytest.raises(ValueError):
        current_model_report(rows, baselines)
