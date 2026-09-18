"""Independent small-matrix checks of directed recovery and its denominators."""

from itertools import product

import numpy as np
import pytest

from benchmarks.neural_ms2lda.cross_model_overlap import (
    MATCH_DTYPE,
    METRICS,
    aggregate_defined,
    cohort_topics,
    describe_finite,
    directed_records,
    nearest_two,
    recovery_curves,
    similarity_matrices,
    spectral_vectors,
    summarize_matches,
)
from benchmarks.neural_ms2lda.cross_model_overlap_evidence import (
    read_overlap,
    seal_overlap,
    validate_protocol,
)
from benchmarks.neural_ms2lda.cross_model_overlap_examples import (
    example_overlap,
    quantile_example,
)
from benchmarks.neural_ms2lda.utils import write_json


def test_cohorts_keep_missing_annotations_separate_from_zero_fingerprints():
    inventory = [
        {"compound_ids": [], "annotation_maccs": [0, 0]},
        {"compound_ids": [1], "annotation_maccs": [1, 0]},
        {"compound_ids": [1, 2], "annotation_maccs": None},
        {"compound_ids": [3, 4], "annotation_maccs": [0, 0]},
    ]
    np.testing.assert_array_equal(cohort_topics(inventory, "all"), [0, 1, 2, 3])
    np.testing.assert_array_equal(cohort_topics(inventory, "recurring"), [2, 3])
    np.testing.assert_array_equal(cohort_topics(inventory, "recurring_evaluable"), [3])
    with pytest.raises(ValueError, match="unknown"):
        cohort_topics(inventory, "favourable")


def test_nearest_two_allows_target_reuse_and_resolves_exact_ties():
    scores = np.array([[0.9, 0.1, 0.9], [0.9, 0.2, 0.7], [0.0, 0.8, 0.8]])
    best, second, reciprocal = nearest_two(scores)
    np.testing.assert_array_equal(best, [0, 0, 1])
    np.testing.assert_array_equal(second, [2, 2, 2])
    np.testing.assert_array_equal(reciprocal, [True, False, True])


def test_degree_curves_agree_with_exhaustive_threshold_graph():
    # Quantized ties test inclusive >= boundaries as well as 0/1 endpoints.
    rng = np.random.default_rng(19)
    scores = rng.integers(0, 5, size=(13, 7)) / 4
    best, second, _ = nearest_two(scores)
    rows = np.zeros(len(scores), dtype=MATCH_DTYPE)
    rows["best_score"] = scores[np.arange(len(scores)), best]
    rows["second_score"] = scores[np.arange(len(scores)), second]
    thresholds = np.linspace(0, 1, 101)
    curves = recovery_curves(rows, thresholds)
    for index, t in enumerate(thresholds):
        degree = (scores >= t).sum(axis=1)
        for key, expected in (
            ("coverage", (degree >= 1).mean()),
            ("unmatched", (degree == 0).mean()),
            ("single", (degree == 1).mean()),
            ("multiple", (degree >= 2).mean()),
        ):
            assert curves[key][index] == pytest.approx(expected)
    assert np.all(np.diff(curves["coverage"]) <= 0)
    np.testing.assert_allclose(
        np.array(curves["unmatched"]) + curves["single"] + curves["multiple"], 1
    )


def test_single_target_has_no_fictional_second_neighbour():
    best, second, reciprocal = nearest_two(np.array([[0.5], [0.2]]))
    np.testing.assert_array_equal(best, [0, 0])
    np.testing.assert_array_equal(second, [-1, -1])
    np.testing.assert_array_equal(reciprocal, [True, False])
    rows = np.zeros(2, dtype=MATCH_DTYPE)
    rows["best_score"] = [0.5, 0.2]
    rows["second_score"] = np.nan
    curves = recovery_curves(rows, np.array([0.0, 0.5, 1.0]))
    assert curves["coverage"] == [1, 0.5, 0]
    assert curves["multiple"] == [0, 0, 0]


@pytest.mark.parametrize("matrix", [np.empty((0, 2)), [[np.nan]], [[-0.1]], [[1.1]]])
def test_invalid_cosines_fail_closed(matrix):
    with pytest.raises(ValueError):
        nearest_two(np.asarray(matrix))


def test_channel_balance_does_not_confuse_equal_mass_fragment_and_loss():
    mask = np.array([True, True, False, False])
    a = spectral_vectors(np.array([[0.89, 0.01, 0.09, 0.01]]), mask)
    b = spectral_vectors(np.array([[0.089, 0.001, 0.001, 0.909]]), mask)
    matrices = similarity_matrices(a, b)
    assert matrices["fragment"][0, 0] == pytest.approx(1)
    assert matrices["loss"][0, 0] < 0.12
    assert matrices["balanced"][0, 0] == pytest.approx(
        (matrices["fragment"][0, 0] + matrices["loss"][0, 0]) / 2
    )
    assert matrices["full"][0, 0] != pytest.approx(matrices["balanced"][0, 0])
    with pytest.raises(ValueError, match="loss"):
        spectral_vectors(np.array([[0.5, 0.5, 0, 0]]), mask)


def _fit(inventory):
    return {
        "inventory": inventory,
        "top_words": np.array([[0, 1], [2, 3], [0, 2]]),
        "effects": np.eye(3),
        "effect_valid": np.ones(3, dtype=bool),
    }


def test_filtering_precedes_search_and_chemical_agreement_cannot_choose_match():
    inventory = [
        {"compound_ids": [1, 2], "annotation_maccs": [1]},
        {"compound_ids": [3], "annotation_maccs": [1]},
        {"compound_ids": [4, 5], "annotation_maccs": [0]},
    ]
    source, target = _fit(inventory), _fit(inventory)
    # The best full-inventory target for topic 0 is an ineligible singleton;
    # the eligible replacement 2 shares no compounds and orthogonal effects.
    scores = np.array([[0.1, 0.99, 0.8], [0.5, 0.8, 0.7], [0.9, 0.1, 0.3]])
    matrices = {key: scores for key in ("full", "balanced", "fragment", "loss")}
    rows = directed_records(
        source,
        target,
        matrices,
        source_id=0,
        target_id=3,
        cohort_id=0,
        similarity_id=0,
        cohorts=["recurring_evaluable"],
    )
    np.testing.assert_array_equal(rows["source_topic"], [0, 2])
    np.testing.assert_array_equal(rows["target_topic"], [2, 0])
    np.testing.assert_allclose(rows["best_score"], [0.8, 0.9])
    np.testing.assert_array_equal(rows["compound_jaccard"], [0, 0])
    np.testing.assert_array_equal(rows["feature_effect_cosine"], [0, 0])
    assert rows.dtype.hasobject is False


def test_missing_chemical_profiles_do_not_become_zero_agreement():
    summary = describe_finite(np.array([np.nan, 0, 1]))
    assert summary == {"n": 2, "undefined": 1, "mean": 0.5, "median": 0.5}
    assert aggregate_defined([None, None])["mean"] is None
    assert aggregate_defined([None, 1])["sample_sd"] is None


def test_aggregation_averages_target_fits_instead_of_best_target_run():
    fits = [
        {"id": str(i), "model": "selected" if i < 3 else "tomotopy"} for i in range(6)
    ]
    blocks = []
    for i, j in product(range(6), repeat=2):
        if i == j:
            continue
        rows = np.zeros(2, dtype=MATCH_DTYPE)
        rows["source_fit"], rows["target_fit"] = i, j
        rows["source_topic"] = [0, 1]
        for field in METRICS:
            rows[field] = [j / 10, j / 10 + 0.02]
        rows["second_score"] = 0
        blocks.append(rows)
    protocol = {
        "cohorts": ["all"],
        "similarities": ["full_beta"],
        "threshold_grid": {"minimum": 0, "maximum": 1, "points": 11},
        "cohort_sizes": {str(i): {"all": 2} for i in range(6)},
    }
    summary = summarize_matches(np.concatenate(blocks), protocol, fits)
    group = next(
        g for g in summary["groups"] if g["direction"] == "selected_to_tomotopy"
    )
    assert group["source_fits"] == 3
    assert group["metrics"]["best_score"]["mean"] == pytest.approx(0.41)
    assert group["metrics"]["best_score"]["sample_sd"] == 0
    assert len(summary["pairs"]) == 30
    assert len(summary["sources"]) == 12


def test_example_selection_is_quantile_based_with_stable_identity_ties():
    rows = np.zeros(3, dtype=MATCH_DTYPE)
    rows["source_fit"] = [2, 1, 0]
    rows["best_score"] = [0.9, 0.5, 0.5]
    row, quantile = quantile_example(rows, "best_score", 0.5)
    assert quantile == 0.5
    assert row["source_fit"] == 0
    source = {"words": [{"token": "frag@50.0"}, {"token": "loss@50.0"}]}
    target = {"words": [{"token": "loss@50.0"}]}
    overlaps = example_overlap(source, target, [0.01])
    assert overlaps[0]["matched"] == 0
    assert overlaps[1]["matched"] == 1


def test_array_code_order_and_fixed_grid_are_not_silently_relabelled():
    protocol = {
        "cohorts": ["all", "recurring_evaluable", "recurring"],
        "similarities": ["full_beta", "balanced_channels"],
        "threshold_grid": {"minimum": 0, "maximum": 1, "points": 201},
    }
    validate_protocol(protocol)
    for key, value in (
        ("similarities", ["balanced_channels", "full_beta"]),
        ("cohorts", ["all"]),
        ("threshold_grid", {"minimum": 0, "maximum": 1, "points": 2}),
    ):
        with pytest.raises(ValueError):
            validate_protocol({**protocol, key: value})


def test_reader_rejects_missing_payload_manifest_and_seal_overwrite(tmp_path):
    write_json(tmp_path / "complete.json", {"payloads": {}})
    with pytest.raises(ValueError, match="manifest is incomplete"):
        read_overlap(tmp_path)
    with pytest.raises(FileExistsError, match="new empty directory"):
        seal_overlap(tmp_path / "missing_protocol.json", tmp_path)
