"""Small exhaustive oracles for the post-hoc compound-level assessments."""

from itertools import combinations, permutations, product
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.neural_ms2lda.chemical_assessment import (
    restrict_members,
    write_maccs_definitions,
)
from benchmarks.neural_ms2lda.chemical_assessment_io import cohort_strata
from benchmarks.neural_ms2lda.chemical_nulls import (
    _permutation_accumulators,
    by_adjust,
    feature_enrichment,
    permutation_indices,
    sos_permutations,
    stratified_tail,
    validate_partition,
)
from benchmarks.neural_ms2lda.motif_correspondence import (
    jaccard,
    load_beta,
    match_topics,
    normalize_rows,
    profile_cosine,
)
from benchmarks.neural_ms2lda.motif_reference import (
    direct_overlaps,
    mass_overlap,
    parse_channel,
    parse_references,
    typed_signature,
)
from benchmarks.neural_ms2lda.utils import read_json, write_json


def test_maccs_definition_export_covers_native_positions(tmp_path):
    write_maccs_definitions(tmp_path)
    keys = read_json(tmp_path / "maccs_definitions.json")["keys"]
    assert [r["position"] for r in keys] == list(range(1, 167))
    assert keys[124]["special_definition"]
    assert keys[165]["special_definition"]


def test_exact_stratified_tail_against_all_combinations():
    blocks = [[1, 1, 0, 0], [1, 0, 0]]
    totals = [
        sum(a) + sum(b)
        for a, b in product(combinations(blocks[0], 2), combinations(blocks[1], 1))
    ]
    for observed in range(4):
        expected = np.mean(np.array(totals) >= observed)
        assert stratified_tail(((4, 2, 2), (3, 1, 1)), observed) == pytest.approx(
            expected
        )


def test_exhaustive_integer_null_kernel_and_shared_compounds():
    fp = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint16)
    ann = np.array([[1, 1], [1, 0]], dtype=np.uint16)
    members = np.array([0, 2, 0, 1])  # compound zero supports BOTH topics
    pointers = np.array([0, 2, 4])
    overlap = ann @ fp.T
    observed = np.array([3, 1])
    shuffled = np.array(
        [a + b for a, b in product(permutations([0, 1]), permutations([2, 3]))]
    )
    counts, sums = _permutation_accumulators(
        shuffled, overlap, pointers, members, observed
    )
    brute = np.array(
        [
            [
                sum(overlap[k, perm[c]] for c in members[pointers[k] : pointers[k + 1]])
                for k in range(2)
            ]
            for perm in shuffled
        ]
    )
    np.testing.assert_array_equal(counts, (brute >= observed).sum(axis=0))
    np.testing.assert_allclose(sums, brute.sum(axis=0))


def test_sos_permutation_boundaries_analytic_expectation_and_determinism():
    fp = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    members = [np.array([0, 2]), np.array([0]), np.array([3]), np.array([], dtype=int)]
    ann = np.array([[1, 1], [0, 0], [1, 1], [1, 0]], dtype=bool)
    blocks = [np.array([0, 1, 2]), np.array([3])]
    kwargs = dict(permutations=999, seed=9, batch_size=100)
    result = sos_permutations(
        fp, ann, np.ones(4, dtype=bool), members, blocks, **kwargs
    )
    repeat = sos_permutations(
        fp, ann, np.ones(4, dtype=bool), members, blocks, **kwargs
    )
    assert result["observed"][0] == 0.75
    assert result["expected"][0] == pytest.approx(2 / 3)
    assert result["mc_mean"][0] == pytest.approx(2 / 3, abs=0.03)
    assert result["p"][1:].tolist() == [1, 1, 1]
    assert result["informative"].tolist() == [True, False, False, False]
    assert result["eligible"].tolist() == [True, True, True, False]
    assert result["tail_mc_lower"][1] < 1  # B/B is not infinitely precise
    for key in result:
        np.testing.assert_array_equal(result[key], repeat[key])


def test_feature_null_all_positions_constants_and_empty_topics():
    fp = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0]])
    result = feature_enrichment(
        fp, [np.array([0]), np.array([], dtype=int)], [np.arange(4)]
    )
    assert result["p"][0, 0] == pytest.approx(0.5)
    assert result["effect"][0, 0] == 0.5
    assert result["ratio"][0, 0] == 2
    assert result["p"][0, 1:].tolist() == [1, 1]
    assert np.isnan(result["ratio"][0, 2])
    assert np.all(result["p"][1] == 1)


def test_by_matches_independent_scipy_and_keeps_unavailable_slots():
    from scipy.stats import false_discovery_control

    p = np.array([0.001, 0.001, 0.03, 1, 1, 0.2])
    np.testing.assert_allclose(by_adjust(p), false_discovery_control(p, method="by"))
    assert by_adjust(np.ones(3)).tolist() == [1, 1, 1]
    for invalid in ([], [np.nan], [-0.1], [1.01]):
        with pytest.raises(ValueError):
            by_adjust(np.array(invalid))


def test_permutations_never_cross_blocks_and_partition_rejects_duplicates():
    blocks = [np.array([0, 2, 4]), np.array([1]), np.array([3, 5])]
    validate_partition(blocks, 6)
    shuffled = permutation_indices(np.random.default_rng(19), blocks, 40, 6)
    for block in blocks:
        assert np.all(np.sort(shuffled[:, block], axis=1) == block)
    with pytest.raises(ValueError):
        validate_partition([np.array([0, 1]), np.array([1, 2])], 3)


def test_scaffold_cohort_deterministic_and_restricted_compounds_shared():
    compounds = [
        dict(
            connectivity_key=str(i),
            scaffold_key=s,
            acquisition_profile=[("H", "60")],
            median_precursor_mz=101 + i,
        )
        for i, s in enumerate(["a", "a", "acyclic:2", "acyclic:3"])
    ]
    chosen, blocks = cohort_strata(compounds, 50, True)
    assert len(chosen) == 3 and {2, 3} <= set(chosen)
    np.testing.assert_array_equal(cohort_strata(compounds, 50, True)[0], chosen)
    validate_partition(blocks, 3)
    fits = [{"members": [np.array([0, 2]), np.array([0, 1])]}] * 2
    members = restrict_members(fits, chosen, 4)
    np.testing.assert_array_equal(members[0], members[2])


def test_topic_matching_identical_reordered_duplicate_and_orthogonal():
    left = np.eye(3)
    rows, cols, scores, reciprocal = match_topics(left, left[[2, 0, 1]])
    np.testing.assert_array_equal(cols, [1, 2, 0])
    np.testing.assert_allclose(scores, 1)
    assert reciprocal.all()
    _, _, duplicate_scores, duplicate_reciprocal = match_topics(
        np.ones((3, 4)), np.ones((3, 4))
    )
    np.testing.assert_allclose(duplicate_scores, 1)
    assert duplicate_reciprocal.sum() <= 1
    _, _, unrelated, _ = match_topics([[1, 0, 0]], [[0, 1, 0]])
    assert unrelated[0] == 0
    assert jaccard([], []) is None
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert profile_cosine(np.array([0, 0]), np.array([1, 0])) is None


def test_missing_embeddings_not_silently_matched():
    normalized, valid = normalize_rows(np.array([[0, 0], [np.nan, 1], [3, 4]]))
    assert valid.tolist() == [False, False, True]
    np.testing.assert_allclose(normalized[:2], 0)
    with pytest.raises(ValueError):
        match_topics([[0, 0]], [[1, 0]])


def test_fortran_float32_beta_is_validated_in_double_precision(tmp_path):
    path = tmp_path / "validation_evaluation/test/beta.npy"
    path.parent.mkdir(parents=True)
    beta = np.asfortranarray(np.full((3, 21233), 1 / 21233, dtype=np.float32))
    assert np.max(abs(beta.sum(1) - 1)) > 1e-5
    np.save(path, beta)
    np.testing.assert_array_equal(
        load_beta({"path": str(tmp_path), "method": "test"}, 21233), beta
    )
    np.save(path, beta * 0.99)
    with pytest.raises(ValueError):
        load_beta({"path": str(tmp_path), "method": "test"}, 21233)


def test_assessment_report_requires_intact_payload_and_null_family(tmp_path):
    from benchmarks.neural_ms2lda.chemical_assessment_report import (
        read_assessment_evidence,
    )
    from benchmarks.neural_ms2lda.utils import sha256_file

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    fits = [
        {"id": f"{model}_{i}", "model": model}
        for model in ("selected", "tomotopy")
        for i in range(3)
    ]
    protocol = {"fits": fits, "configurations": [{"id": "primary_50"}]}
    write_json(tmp_path / "protocol.json", protocol)
    write_json(evidence / "protocol.json", protocol)
    write_json(
        evidence / "input_seal.json",
        {"protocol_sha256": sha256_file(tmp_path / "protocol.json")},
    )
    write_json(evidence / "payload.json", {"example": 1})
    payloads = {
        "payload.json": {
            "bytes": (evidence / "payload.json").stat().st_size,
            "sha256": sha256_file(evidence / "payload.json"),
        }
    }
    summary = {
        "primary_50": {
            "fits": [{"fit": f["id"]} for f in fits],
            "sos_family_size": 6000,
            "feature_family_size": 996000,
            "permutations": 99999,
        }
    }
    for stage, value in (
        ("specificity", summary),
        ("stability", {"pairs": [{}] * 15}),
        ("references", {"fits": [{"fit": f["id"]} for f in fits]}),
    ):
        write_json(
            evidence / f"{stage}_complete.json",
            {"payloads": payloads, "summary": value},
        )
    assert read_assessment_evidence(evidence)["protocol"] == protocol
    summary["primary_50"]["permutations"] = 999
    write_json(
        evidence / "specificity_complete.json",
        {"payloads": payloads, "summary": summary},
    )
    with pytest.raises(ValueError, match="budget"):
        read_assessment_evidence(evidence)
    summary["primary_50"]["permutations"] = 99999
    write_json(
        evidence / "specificity_complete.json",
        {"payloads": payloads, "summary": summary},
    )
    write_json(evidence / "payload.json", {"example": 2})
    with pytest.raises(ValueError, match="payload changed"):
        read_assessment_evidence(evidence)


def test_reference_parser_keeps_manual_conflicts_and_rejects_malformed(tmp_path):
    row = dict(
        charge=1,
        analysis_polarity="positive ionisation mode",
        motif_id="motif_1",
        motifset="test",
        frag_mz=[100, None],
        frag_intens=[1, None],
        loss_mz=[None, 50],
        loss_intens=[None, 0.5],
        annotation="first",
        short_annotation="one",
        auto_annotation=["NOT GROUND TRUTH"],
    )
    path = tmp_path / "reference.json"
    write_json(
        path,
        {
            "ms2": [
                row,
                {**row, "annotation": "conflicting", "auto_annotation": ["ignored"]},
                {**row, "frag_intens": [None, None]},
            ]
        },
    )
    refs, excluded = parse_references([path])
    assert len(refs) == 1 and len(refs[0]["sources"]) == 2
    assert len(excluded) == 1
    assert "auto_annotation" not in str(refs)
    assert typed_signature([(100, 2)], [(50, 1)]) == typed_signature(
        [(100, 1)], [(50, 0.5)]
    )
    assert typed_signature([(100, 1)], []) != typed_signature([], [(100, 1)])
    assert parse_channel([100, 100, None], [0.3, 0.7, None]) == [(100, 1)]
    assert parse_channel([100, np.nan], [1, np.nan]) == [(100, 1)]
    with pytest.raises(ValueError):
        parse_channel([100, np.nan], [1, 0.5])
    with pytest.raises(ValueError):
        parse_channel([np.inf], [np.inf])


def test_mass_overlap_matches_exhaustive_bipartite_oracle_and_channel_separation():
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(5)
    for _ in range(20):
        a, b = rng.integers(0, 15, 7), rng.integers(0, 15, 5)
        compatible = abs(a[:, None] - b) <= 1
        i, j = linear_sum_assignment(compatible.astype(int), maximize=True)
        assert mass_overlap(a, b, 1) == compatible[i, j].sum()
    query = SimpleNamespace(
        peaks=SimpleNamespace(mz=np.array([100.0])),
        losses=SimpleNamespace(mz=np.array([])),
    )
    reference = SimpleNamespace(
        peaks=SimpleNamespace(mz=np.array([])),
        losses=SimpleNamespace(mz=np.array([100.0])),
    )
    assert sum(r["matched"] for r in direct_overlaps(query, reference, [0.01])) == 0
    assert mass_overlap(np.array([100.0]), np.array([100.01]), 0.01) == 1
