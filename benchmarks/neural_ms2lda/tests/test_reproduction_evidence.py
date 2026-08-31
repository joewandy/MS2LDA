"""Tests for clean-room evidence ownership and package sealing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from scripts import generate_routing_etm_report as report_generator
from scripts import package_contextual_sparse_etm_reproduction as packager
from scripts.generate_routing_etm_report import (
    _require_reportable_claims,
    _validate_package_integrity,
)

from benchmarks.neural_ms2lda.reproduction_audit import (
    file_record,
    sha256_file,
    verify_linked_inputs,
    write_csv,
    write_json,
)


def test_report_rejects_changed_packaged_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "comparison.csv"
    artifact.write_text("model,value\nETM,1\n", encoding="utf-8")
    manifest = {
        "packaged_files": [file_record(artifact, relative_to=tmp_path)],
    }
    manifest_path = tmp_path / "fresh_evidence_manifest.json"
    write_json(manifest_path, manifest)
    checkpoint = {"fresh_evidence_manifest_sha256": sha256_file(manifest_path)}
    _validate_package_integrity(tmp_path, checkpoint)

    artifact.write_text("model,value\nETM,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="packaged evidence changed"):
        _validate_package_integrity(tmp_path, checkpoint)


def test_fresh_chemistry_requires_explicit_mag_exception_counts() -> None:
    result: dict[str, Any] = {
        "high_confidence_chemistry": {},
        "topics": 1,
        "annotation_coverage": 0.0,
        "heldout_compounds_excluded_from_mag": True,
        "split": "test",
    }
    with pytest.raises(RuntimeError, match="lacks explicit MAG exception"):
        packager._chemistry_summary(result)


def test_packager_rejects_non_cuda_neural_evidence() -> None:
    packager._require_neural_device("cuda", label="test fixture")
    with pytest.raises(RuntimeError, match="was not executed on cuda"):
        packager._require_neural_device("mps", label="test fixture")


def _valid_chemistry_result() -> dict[str, Any]:
    """Return one minimal package-facing chemical result."""
    return {
        "topics": 2,
        "annotation_coverage": 0.5,
        "heldout_compounds_excluded_from_mag": True,
        "split": "test",
        "mag_failures": {
            "clustering_count": 0,
            "clustering_topic_ids": [],
            "optimization_count": 0,
            "optimization_topic_ids": [],
        },
        "high_confidence_chemistry": {
            "eligible_topics": 1,
            "mean_sos": 0.7,
            "median_sos": 0.7,
            "sos_bands": {
                "high_gt_0_8": 0,
                "intermediate_0_6_to_0_8": 1,
                "low_lt_0_6": 0,
            },
        },
    }


def test_fresh_chemistry_rejects_heldout_mag_leakage() -> None:
    result = _valid_chemistry_result()
    result["heldout_compounds_excluded_from_mag"] = False
    with pytest.raises(RuntimeError, match="does not exclude held-out compounds"):
        packager._chemistry_summary(result)


def test_fresh_chemistry_rejects_incomplete_sos_band_accounting() -> None:
    result = _valid_chemistry_result()
    result["high_confidence_chemistry"]["eligible_topics"] = 2
    with pytest.raises(RuntimeError, match="SOS bands account for 1 motifs but 2"):
        packager._chemistry_summary(result)


def test_fresh_chemistry_rejects_recorded_mag_exceptions() -> None:
    result = _valid_chemistry_result()
    result["mag_failures"]["clustering_count"] = 1
    result["mag_failures"]["clustering_topic_ids"] = [7]
    with pytest.raises(RuntimeError, match="contains 1 MAG exceptions"):
        packager._chemistry_summary(result)


def test_csv_writer_preserves_method_specific_columns(tmp_path: Path) -> None:
    artifact = tmp_path / "comparison.csv"
    write_csv(
        artifact,
        [
            {"model": "canonical ETM", "score": 1.0},
            {
                "model": "Contextual Sparse ETM",
                "score": 2.0,
                "training_seed": 7043,
                "learned_context_scale": 0.25,
            },
        ],
    )

    with artifact.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "model",
        "score",
        "training_seed",
        "learned_context_scale",
    ]
    assert rows[0]["training_seed"] == ""
    assert rows[1]["training_seed"] == "7043"


def test_chemical_integrity_gate_includes_every_contextual_seed() -> None:
    comparison = [
        {
            "mag_clustering_failures": 0,
            "mag_optimization_failures": 0,
            "heldout_compounds_excluded_from_mag": True,
            "sos_band_accounting_valid": True,
        },
    ]
    stability = {
        "direction_checks": {
            "zero_mag_exceptions_on_all_seeds": False,
            "heldout_compounds_excluded_from_mag_on_all_seeds": True,
            "sos_bands_account_for_evaluable_motifs_on_all_seeds": True,
        },
    }
    result = packager._chemical_integrity_checks(comparison, stability)
    assert result["checks"]["comparison_models_have_zero_mag_exceptions"] is True
    assert result["checks"]["all_contextual_seeds_have_zero_mag_exceptions"] is False
    assert result["all_passed"] is False


def test_report_rejects_failed_directional_claims() -> None:
    acceptance = {
        "all_passed": False,
        "checks": {"expected_direction": False},
    }
    with pytest.raises(ValueError, match="expected_direction"):
        _require_reportable_claims(acceptance)


def test_report_rejects_an_unsupported_sentence_level_comparison() -> None:
    comparison = {
        "canonical ETM": {
            "optimized_motifs": "10",
            "evaluable_motifs": "5",
            "useful_motifs": "3",
            "mean_sos": "0.5",
            "completion_nll": "1.0",
            "median_effective_topics": "10",
            "unique_top1_topics": "100",
        },
        "balanced ETM": {
            "optimized_motifs": "9",
            "evaluable_motifs": "6",
            "useful_motifs": "4",
            "mean_sos": "0.6",
            "completion_nll": "0.9",
            "median_effective_topics": "9",
            "unique_top1_topics": "200",
        },
        "Contextual Sparse ETM": {
            "optimized_motifs": "8",
            "evaluable_motifs": "20",
            "useful_motifs": "10",
            "mean_sos": "0.7",
            "completion_nll": "1.1",
            "median_effective_topics": "2",
            "unique_top1_topics": "900",
        },
        "Tomotopy LDA": {
            "optimized_motifs": "8",
            "evaluable_motifs": "8",
            "useful_motifs": "5",
            "mean_sos": "0.8",
            "median_sos": "0.8",
            "completion_nll": "1.2",
        },
    }
    synthetic = [
        {
            "formulation": formulation,
            "mean_true_beta_cosine": str(beta),
            "mean_true_theta_cosine": str(theta),
            "mean_nll": str(nll),
            "mean_unique_top1_topics": str(unique),
        }
        for formulation, beta, theta, nll, unique in (
            (report_generator.BALANCED_SOFTMAX, 0.5, 0.5, 2.0, 10),
            (report_generator.BALANCED_ENTMAX, 0.4, 0.4, 2.2, 5),
            (report_generator.CONTEXT_SOFTMAX, 0.6, 0.6, 1.8, 12),
            (report_generator.CONTEXT_ENTMAX, 0.7, 0.7, 1.9, 13),
        )
    ]
    high_k = [
        {"formulation": formulation, "nll": str(nll)}
        for formulation, nll in (
            (report_generator.BALANCED_SOFTMAX, 2.0),
            (report_generator.BALANCED_ENTMAX, 1.9),
            (report_generator.CONTEXT_ENTMAX, 1.8),
        )
    ]
    with pytest.raises(ValueError, match="balancing_raises_optimized_motifs"):
        report_generator._require_manuscript_claims(
            {"comparison": comparison, "synthetic": synthetic, "high_k": high_k},
        )


def test_report_render_failure_does_not_mix_generated_fragments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated"
    output.mkdir()
    existing = output / "routing_etm_macros.tex"
    existing.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(report_generator, "_validate_and_load", lambda _root: {})
    generators = (
        "_generate_macros",
        "_generate_synthetic_table",
        "_generate_high_k_table",
        "_generate_test_table",
        "_generate_stability_table",
        "_generate_diagnostics_table",
        "_generate_hyperparameters",
    )
    artifact_names = sorted(
        report_generator.EXPECTED_OUTPUTS - {"routing_etm_code_table.tex"}
    )
    for function_name, artifact_name in zip(generators, artifact_names, strict=True):
        monkeypatch.setattr(
            report_generator,
            function_name,
            lambda _evidence, name=artifact_name: (name, "new\n"),
        )

    def fail_last() -> tuple[str, str]:
        raise ValueError("deliberate late render failure")

    monkeypatch.setattr(report_generator, "_generate_code_map", fail_last)
    with pytest.raises(ValueError, match="deliberate late render failure"):
        report_generator.generate(tmp_path / "evidence", output)
    assert existing.read_text(encoding="utf-8") == "old\n"
    assert list(output.iterdir()) == [existing]


def test_linked_input_manifest_is_rehashed_at_packaging(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    linked = tmp_path / "visible.bin"
    source.write_bytes(b"sealed")
    linked.symlink_to(source)
    manifest = {
        "linked_inputs": [
            {
                "path": str(source),
                "linked_path": str(linked),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            },
        ],
    }
    verify_linked_inputs(manifest, field="linked_inputs")
    linked.unlink()
    linked.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="manifest-owned input changed"):
        verify_linked_inputs(manifest, field="linked_inputs")


def test_failed_package_build_leaves_no_partial_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "package"

    def fail_after_writing(_raw: Path, staging: Path) -> dict[str, Any]:
        staging.mkdir()
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("deliberate packaging failure")

    monkeypatch.setattr(packager, "_build_package", fail_after_writing)
    with pytest.raises(RuntimeError, match="deliberate packaging failure"):
        packager.package_reproduction(raw, output)
    assert not output.exists()
    assert not (tmp_path / ".package.staging").exists()
