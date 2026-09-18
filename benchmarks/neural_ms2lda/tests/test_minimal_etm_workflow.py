"""Exercise fitting, sealed validation export and experiment evidence together."""

import argparse
import json
from copy import copy

import pytest
import torch

from benchmarks.neural_ms2lda.chemical import run_chemical_scoring
from scripts.prepare_msnlib_validation_view import create_validation_view
from scripts.run_minimal_etm import run
from scripts.summarize_minimal_etm import (
    repository_code_path,
    sos_sensitivity,
    summarize,
)

from .test_workflow import _prepare_mini_prepared_source


def test_minimal_etm_train_export_and_no_test_access(tmp_path):
    prepared, _ = _prepare_mini_prepared_source(tmp_path)
    view = tmp_path / "candidate"
    create_validation_view(view, prepared, expected_topics=4)
    output = view / "models/minimal_etm"
    args = argparse.Namespace(
        synthetic_root=None,
        validation_run=view,
        output=output,
        variant="batchnorm",
        topics=4,
        seed=42,
        epochs=2,
        batch_size=4,
        hidden=8,
        concentration=0.02,
        learning_rate=0.005,
        momentum=0.99,
        count_scaling="raw_counts",
        device="cpu",
        threads=1,
    )
    result = run(args)
    assert result["config"]["test_matrices_loaded"] is False
    assert (output / "checkpoint.pt").is_file()
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert type(checkpoint["config"]["torch"]) is str
    assert all(
        isinstance(value, torch.Tensor) for value in checkpoint["state_dict"].values()
    )
    assert (view / "validation_evaluation/minimal_etm/beta.npy").is_file()
    assert not list((view / "data").glob("test*"))
    with pytest.raises(FileExistsError, match="fresh output"):
        run(args)
    calibrated_view = tmp_path / "recalibrated"
    create_validation_view(calibrated_view, prepared, expected_topics=4)
    recalibration_args = copy(args)
    recalibration_args.validation_run = calibrated_view
    recalibration_args.output = calibrated_view / "models/minimal_etm"
    recalibration_args.recalibrate_from = output
    recalibrated = run(recalibration_args)
    assert recalibrated["config"]["parent_checkpoint_sha256"]
    assert recalibrated["config"]["normalization_statistics"] == "training"
    new_checkpoint = torch.load(
        recalibration_args.output / "checkpoint.pt", weights_only=True
    )
    for name, value in checkpoint["state_dict"].items():
        if not any(
            buffer in name
            for buffer in ("running_mean", "running_var", "num_batches_tracked")
        ):
            torch.testing.assert_close(
                value, new_checkpoint["state_dict"][name], rtol=0, atol=0
            )
    chemistry = {
        "split": "validation",
        "topics": 4,
        "annotation_coverage": 0.75,
        "heldout_compounds_excluded_from_mag": True,
        "mag_failures": {"clustering_count": 0, "optimization_count": 0},
        "chemical_evaluation": {
            "eligible_topics": 3,
            "mean_sos": 0.7,
            "sos_bands": {
                "high_gt_0_8": 1,
                "intermediate_0_6_to_0_8": 1,
                "low_lt_0_6": 1,
            },
            "topic_scores": [
                {"eligible": True, "sos": 0.9},
                {"eligible": True, "sos": 0.6},
                {"eligible": True, "sos": 0.5},
                {"eligible": False, "sos": None},
            ],
        },
    }
    chemistry_path = view / "validation_chemical/minimal_etm/complete.json"
    chemistry_path.parent.mkdir(parents=True)
    chemistry_path.write_text(json.dumps(chemistry))
    rows = summarize(tmp_path, tmp_path / "evidence")
    assert len(rows) == 2
    chemistry_row = next(row for row in rows if row["optimized_motifs"] != "")
    assert chemistry_row["optimized_motifs"] == 3
    assert chemistry_row["evaluable_motifs"] == 3
    assert chemistry_row["useful_motifs"] == 2
    assert chemistry_row["sos_count_ge_0.5"] == 3
    assert chemistry_row["sos_count_ge_0.6"] == 2
    assert chemistry_row["sos_count_ge_0.9"] == 1
    chemistry["mag_failures"]["clustering_count"] = 1
    chemistry_path.write_text(json.dumps(chemistry))
    with pytest.raises(ValueError, match="MAG failures"):
        summarize(tmp_path, tmp_path / "invalid-evidence")
    chemistry["mag_failures"]["clustering_count"] = 0
    chemistry["heldout_compounds_excluded_from_mag"] = False
    chemistry_path.write_text(json.dumps(chemistry))
    with pytest.raises(ValueError, match="held-out compounds"):
        summarize(tmp_path, tmp_path / "invalid-evidence")
    # Even a manifest claiming isolation must not permit an exposed test file.
    args.output = view / "other-output"
    (view / "data/test_full.npz").touch()
    with pytest.raises(RuntimeError, match="exposes test"):
        run(args)


def test_development_chemical_method_cannot_evaluate_test(tmp_path):
    with pytest.raises(ValueError, match="validation-only"):
        run_chemical_scoring(
            tmp_path,
            method="minimal_etm",
            data_root=tmp_path,
            protocol={},
            split="test",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_sos_sensitivity_rejects_invalid_scores(value):
    with pytest.raises(ValueError, match="invalid eligible"):
        sos_sensitivity(
            {
                "topic_scores": [{"eligible": True, "sos": value}],
                "eligible_topics": 1,
            }
        )


@pytest.mark.parametrize(
    "prefix", ["/home/person/git/MS2LDA/", "/Users/person/MS2LDA/", ""]
)
def test_archived_code_hashes_are_portable(prefix):
    for name in ("scripts/run_minimal_etm.py", "benchmarks/neural_ms2lda/prior_etm.py"):
        assert repository_code_path(prefix + name) == name
