"""Tests for clean-room evidence ownership and package sealing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts import package_contextual_sparse_etm_reproduction as packager
from scripts.generate_routing_etm_report import _validate_package_integrity

from benchmarks.neural_ms2lda.reproduction_audit import (
    file_record,
    sha256_file,
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
