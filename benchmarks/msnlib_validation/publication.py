"""Export and verify small, Git-suitable evidence from a completed real run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import file_sha256, read_json, write_json
from .protocol import verify_protocol

CHECKPOINT_SCHEMA = "hybrid-lda-seed42-checkpoint/v2"
PUBLICATION_SCHEMA = "msnlib-validation/publication-manifest-v1"


def _atomic_copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_report(run_dir: Path, protocol_sha256: str) -> tuple[dict[str, Any], Path]:
    report_dir = run_dir / "report"
    complete = read_json(report_dir / "complete.json")
    if complete.get("protocol_sha256") != protocol_sha256:
        raise ValueError("completed report belongs to another frozen protocol")
    for name, digest in complete.get("output_sha256", {}).items():
        path = report_dir / str(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"completed report artifact changed: {name}")
    summary_path = report_dir / "summary.json"
    fragment_path = report_dir / "manuscript_results.tex"
    if not summary_path.is_file() or not fragment_path.is_file():
        raise FileNotFoundError("completed report lacks publication artifacts")
    return complete, fragment_path


def export_publication_artifacts(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    latex_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Export an exact report snapshot and its generated LaTeX fragment."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    report_complete, source_fragment = _verify_report(
        directory, lock["protocol_sha256"]
    )
    report_summary_path = directory / "report" / "summary.json"
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol": {
            "derivation": lock.get("derivation"),
            "git": lock.get("git"),
            "name": lock["protocol_name"],
            "prior_test_results_inspected": bool(lock.get("test_results_inspected")),
            "sha256": lock["protocol_sha256"],
            "source_manifest_sha256": lock["source_manifest_sha256"],
        },
        "report_complete_sha256": file_sha256(directory / "report" / "complete.json"),
        "report_summary": read_json(report_summary_path),
        "report_summary_sha256": file_sha256(report_summary_path),
    }
    checkpoint_destination = Path(checkpoint_path)
    latex_destination = Path(latex_path)
    manifest_destination = Path(manifest_path)
    write_json(checkpoint_destination, checkpoint)
    _atomic_copy_text(source_fragment, latex_destination)
    manifest = {
        "schema_version": PUBLICATION_SCHEMA,
        "protocol_sha256": lock["protocol_sha256"],
        "source_report_outputs": report_complete["output_sha256"],
        "published": {
            checkpoint_destination.name: file_sha256(checkpoint_destination),
            latex_destination.name: file_sha256(latex_destination),
        },
    }
    write_json(manifest_destination, manifest)
    return manifest


def verify_publication_artifacts(
    *,
    checkpoint_path: str | Path,
    latex_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify committed evidence without requiring the multi-gigabyte run."""
    checkpoint = Path(checkpoint_path)
    latex = Path(latex_path)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != PUBLICATION_SCHEMA:
        raise ValueError("unsupported publication manifest schema")
    expected = manifest.get("published", {})
    actual = {
        checkpoint.name: file_sha256(checkpoint),
        latex.name: file_sha256(latex),
    }
    if expected != actual:
        raise ValueError("committed publication artifacts changed")
    payload = read_json(checkpoint)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("protocol", {}).get("sha256") != manifest.get("protocol_sha256"):
        raise ValueError("checkpoint and publication protocol hashes differ")
    return manifest
