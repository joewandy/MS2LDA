"""Tests for the sealed validation-only simplification campaign."""

from __future__ import annotations

from pathlib import Path

from benchmarks.neural_ms2lda.campaign import (
    U1_BASELINE,
    _test_data_files,
    assess_candidate,
)


def test_candidate_test_file_audit_is_explicit(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "validation_full.npz").touch()
    assert _test_data_files(tmp_path) == []
    (data / "test_records.jsonl").touch()
    assert _test_data_files(tmp_path) == ["test_records.jsonl"]


def test_chemistry_first_standard_pass() -> None:
    metrics = {**U1_BASELINE, "parameter_count": 167_168}
    decision = assess_candidate(metrics, U1_BASELINE)
    assert decision["retained"] is True
    assert decision["borderline"] is False


def test_one_narrow_chemistry_miss_is_borderline() -> None:
    metrics = {
        **U1_BASELINE,
        "useful_motifs": 253,
        "parameter_count": 167_168,
    }
    decision = assess_candidate(metrics, U1_BASELINE)
    assert decision["chemistry_relative_gates"]["useful_motifs"] is False
    assert decision["tie_gates"]["useful_motifs"] is True
    assert decision["retained"] is True
    assert decision["borderline"] is True


def test_two_misses_or_large_nll_loss_fail() -> None:
    chemistry_misses = {
        **U1_BASELINE,
        "useful_motifs": 253,
        "mean_sos": 0.642,
        "parameter_count": 167_168,
    }
    assert assess_candidate(chemistry_misses, U1_BASELINE)["retained"] is False
    poor_nll = {
        **U1_BASELINE,
        "validation_nll": U1_BASELINE["validation_nll"] * 1.051,
        "parameter_count": 167_168,
    }
    assert assess_candidate(poor_nll, U1_BASELINE)["retained"] is False
