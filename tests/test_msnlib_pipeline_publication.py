"""Focused tests for unattended execution and committed result evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.msnlib_validation.pipeline as validation_pipeline
import benchmarks.msnlib_validation.publication as validation_publication
from benchmarks.msnlib_validation.config import file_sha256, read_json, write_json


def test_pipeline_runs_all_stages_and_records_failure_safe_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    protocol = "a" * 64
    monkeypatch.setattr(
        validation_pipeline,
        "verify_protocol",
        lambda _: {"data_root": str(data_root), "protocol_sha256": protocol},
    )
    outputs = iter(
        (
            "core/complete.json",
            "chemical_inference/complete.json",
            "mag/raw_dreams/complete.json",
            "mag/complete.json",
            "report/complete.json",
        )
    )
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(list(command))
        path = tmp_path / next(outputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"completed": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_pipeline.subprocess, "run", fake_run)
    state = validation_pipeline.run_pipeline(
        tmp_path, data_root=data_root, mag_environment="legacy-mag"
    )

    assert state["status"] == "completed"
    assert list(state["stages"]) == [
        "core",
        "chemical_inference",
        "raw_dreams",
        "mag",
        "report",
    ]
    assert all(row["status"] == "completed" for row in state["stages"].values())
    assert commands[3][:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "legacy-mag",
        "python",
    ]
    assert read_json(tmp_path / "pipeline_state.json")["status"] == "completed"


def test_pipeline_rejects_changed_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    monkeypatch.setattr(
        validation_pipeline,
        "verify_protocol",
        lambda _: {"data_root": str(frozen), "protocol_sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="data root"):
        validation_pipeline.run_pipeline(tmp_path, data_root=tmp_path / "other")


def test_publication_export_is_exact_and_self_verifying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    report = run / "report"
    report.mkdir(parents=True)
    protocol = "b" * 64
    summary_path = report / "summary.json"
    fragment_path = report / "manuscript_results.tex"
    write_json(summary_path, {"result": {"sos": 0.625}})
    fragment_path.write_text("\\newcommand{\\ExampleSOS}{0.6250}\n", encoding="utf-8")
    output_hashes = {
        "summary.json": file_sha256(summary_path),
        "manuscript_results.tex": file_sha256(fragment_path),
    }
    write_json(
        report / "complete.json",
        {"output_sha256": output_hashes, "protocol_sha256": protocol},
    )
    monkeypatch.setattr(
        validation_publication,
        "verify_protocol",
        lambda _: {
            "derivation": {"kind": "implementation_correction"},
            "git": {"commit": "c" * 40, "dirty": False},
            "protocol_name": "corrected",
            "protocol_sha256": protocol,
            "source_manifest_sha256": "d" * 64,
            "test_results_inspected": True,
        },
    )
    docs = tmp_path / "docs"
    checkpoint = docs / "checkpoint.json"
    latex = docs / "results.tex"
    manifest = docs / "publication.json"

    exported = validation_publication.export_publication_artifacts(
        run,
        checkpoint_path=checkpoint,
        latex_path=latex,
        manifest_path=manifest,
    )

    assert exported == validation_publication.verify_publication_artifacts(
        checkpoint_path=checkpoint,
        latex_path=latex,
        manifest_path=manifest,
    )
    assert read_json(checkpoint)["report_summary"]["result"]["sos"] == 0.625
    assert latex.read_text(encoding="utf-8") == fragment_path.read_text(
        encoding="utf-8"
    )

    latex.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validation_publication.verify_publication_artifacts(
            checkpoint_path=checkpoint,
            latex_path=latex,
            manifest_path=manifest,
        )
