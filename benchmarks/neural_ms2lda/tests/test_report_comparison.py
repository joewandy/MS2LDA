"""The focused paper must use complete fits and genuine between-fit variation."""

import csv
import json
from copy import deepcopy

import pytest

from benchmarks.neural_ms2lda.report_comparison import (
    BASELINE_SEEDS,
    baseline_summaries,
    comparison_table,
    read_baseline_evidence,
)
from benchmarks.neural_ms2lda.utils import sha256_file


def baseline_rows():
    """Six toy fits with exact, independently known sample means and SDs."""
    return [
        {
            "model": model,
            "training_seed": seed,
            "run": f"{model}_{seed}",
            "spectrum_topic_associations": 3889,
            "completion_documents": 3888,
            "completion_tokens": 1277983,
            "optimized_motifs": 800,
            "evaluable_motifs": 600 + offset,
            "useful_motifs": 350 + offset,
            "mean_sos": 0.6 + offset * 0.01,
            "completion_nll": 9 + offset * 0.1,
            "median_effective_topics": 4 + offset * 0.1,
        }
        for model, seeds in BASELINE_SEEDS.items()
        for seed, offset in zip(seeds, (-1, 0, 1))
    ]


def test_summaries_are_fit_means_and_sample_sds_not_errors_over_spectra():
    summaries = baseline_summaries(baseline_rows())
    for model in BASELINE_SEEDS:
        assert summaries[model]["fits"] == 3
        assert summaries[model]["useful_motifs"] == {"mean": 350, "sample_sd": 1}
        assert summaries[model]["completion_nll"] == {
            "mean": 9,
            "sample_sd": pytest.approx(0.1),
        }
        assert summaries[model]["median_effective_topics"]["mean"] == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_seed", 11.1),
        ("completion_documents", 3889),
        ("completion_tokens", 1277982),
        ("spectrum_topic_associations", 7777),
        ("optimized_motifs", 1001),
        ("optimized_motifs", 500),
        ("evaluable_motifs", 350.5),
        ("useful_motifs", -1),
        ("useful_motifs", 700),
        ("mean_sos", 1.01),
        ("mean_sos", "nan"),
        ("completion_nll", "inf"),
        ("completion_nll", -1),
        ("median_effective_topics", 0),
        ("median_effective_topics", 1001),
        ("run", ""),
    ],
)
def test_invalid_population_values_or_recipe_identity_are_rejected(field, value):
    rows = baseline_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        baseline_summaries(rows)


def test_duplicate_run_identity_cannot_count_as_independent_fits():
    rows = baseline_rows()
    rows[0]["run"] = rows[1]["run"]
    with pytest.raises(ValueError, match="distinct"):
        baseline_summaries(rows)


def test_empty_evaluable_population_has_no_numeric_conditional_mean():
    rows = baseline_rows()
    rows[0].update(evaluable_motifs=0, useful_motifs=0, mean_sos=0)
    with pytest.raises(ValueError, match="undefined"):
        baseline_summaries(rows)


def test_best_means_are_directional_ties_are_bold_and_sparsity_is_not_ranked():
    baselines = baseline_summaries(baseline_rows())
    chosen = deepcopy(baselines["Tomotopy LDA"])
    chosen["evaluable_motifs"]["mean"] = 700
    chosen["useful_motifs"]["mean"] = 400
    chosen["mean_sos"]["mean"] = 0.5
    chosen["completion_nll"]["mean"] = 10
    chosen["median_effective_topics"]["mean"] = 2
    lines = comparison_table(baselines, chosen)
    for baseline in lines[:2]:
        assert r"\boldsymbol{0.600\pm0.010}" in baseline
        assert r"\boldsymbol{9.000\pm0.100}" in baseline
        assert baseline.count(r"\boldsymbol") == 2
    assert r"\boldsymbol{700.0\pm1.0}" in lines[-1]
    assert r"\boldsymbol{400.0\pm1.0}" in lines[-1]
    assert r"$2.00\pm0.10$" in lines[-1]
    assert lines[-1].count(r"\boldsymbol") == 2


def test_highlighting_uses_full_precision_not_rounded_display_ties():
    baselines = baseline_summaries(baseline_rows())
    baselines["canonical ETM"]["completion_nll"]["mean"] = 8.9999
    chosen = deepcopy(baselines["Tomotopy LDA"])
    lines = comparison_table(baselines, chosen)
    assert r"\boldsymbol{9.000\pm0.100}" not in lines[0]
    assert r"\boldsymbol{9.000\pm0.100}" in lines[1]


def write_evidence(directory):
    """Make a minimal public bundle; never access the live scientific evidence."""
    rows = baseline_rows()
    path = directory / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payloads = [
        "runs.csv",
        "summary.json",
        "protocol.json",
        "execution_source_manifest.json",
        "execution_source.tar.gz",
        *(
            f"{row['run']}/{name}"
            for row in rows
            for name in (
                "result.json",
                "chemical.json",
                "validation_input_manifest.json",
            )
        ),
    ]
    for name in payloads[1:]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"toy integrity fixture; never real scientific evidence")
    manifest = {
        "schema_version": 1,
        "reported_split": "validation",
        "historical_tomotopy_included": False,
        "selected_model_refitted": False,
        "records": [
            {key: row[key] for key in ("model", "training_seed", "run")} for row in rows
        ],
        "sources": [
            {
                "path": name,
                "bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in payloads
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_public_manifest_round_trip_requires_no_training_dependencies(tmp_path):
    write_evidence(tmp_path)
    rows = read_baseline_evidence(tmp_path)
    assert baseline_summaries(rows) == baseline_summaries(baseline_rows())


@pytest.mark.parametrize(
    "change",
    [
        "scope",
        "old_lda",
        "refit",
        "size",
        "hash",
        "path",
        "missing",
        "incomplete",
        "duplicate",
        "run",
    ],
)
def test_corrupt_or_misidentified_public_evidence_cannot_generate_a_report(
    tmp_path, change
):
    manifest = write_evidence(tmp_path)
    if change == "scope":
        manifest["reported_split"] = "test"
    elif change == "old_lda":
        manifest["historical_tomotopy_included"] = True
    elif change == "refit":
        manifest["selected_model_refitted"] = True
    elif change == "size":
        manifest["sources"][0]["bytes"] += 1
    elif change == "hash":
        manifest["sources"][0]["sha256"] = "0" * 64
    elif change == "path":
        manifest["sources"].append({"path": "../unrelated.csv"})
    elif change == "missing":
        manifest["sources"] = []
    elif change == "incomplete":
        manifest["sources"] = manifest["sources"][:1]
    elif change == "duplicate":
        manifest["sources"].append(manifest["sources"][0])
    else:
        manifest["records"][0]["run"] = "substituted_run"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        read_baseline_evidence(tmp_path)
