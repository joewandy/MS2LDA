"""Portable copying and sealing for a verified reproduction evidence bundle.

Scientific calculations and claim checks deliberately live in the packager
script.  This module contains only the mechanical boundary: copy the selected
artifacts, normalize host paths, and cryptographically seal the resulting
directory.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .reproduction_audit import file_record, sha256_file, write_csv, write_json
from .reproduction_plan import ReproductionPaths, stage_plan
from .study_protocol import METHOD, SYNTHETIC_SEEDS, TRAINING_SEEDS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

COMPACT_MODEL_FILES = (
    "result.json",
    "training_history.csv",
    "theta_support_summary.csv",
    "context_evidence_support_summary.csv",
    "duplicate_component_summary.json",
    "fragment_mass_summary.json",
    "top_words.csv",
    "validation_access_audit.json",
    "provenance.json",
)
COMPACT_CONTROL_FILES = (
    "result.json",
    "training_history.csv",
    "duplicate_component_summary.json",
    "fragment_mass_summary.json",
    "top_words.csv",
    "validation_access_audit.json",
)
COMPACT_TEST_EVALUATION_FILES = ("complete.json", "test_access_audit.json")
MACHINE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\\\|\\\\wsl(?:\.localhost)?\\|/(?:home|Users|tmp|mnt)/)",
)
LOCAL_FILE_URL_PATTERN = re.compile(
    r"file:///(?:home|Users|tmp|mnt)/[^\s]+",
)


def portable_value(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    """Replace checkout-specific paths inside a JSON-compatible value."""
    if isinstance(value, dict):
        return {key: portable_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item, replacements) for item in value]
    if isinstance(value, str):
        portable = value
        for source, label in replacements:
            portable = portable.replace(source, label)
        # ``pip freeze`` can retain ephemeral conda/feedstock build locations
        # that are unrelated to the current user's home or reproduction root.
        # Keep the dependency declaration while removing the non-portable local
        # build URL (for example ``file:///home/conda/...``).
        return LOCAL_FILE_URL_PATTERN.sub("file://<local-build-path>", portable)
    return value


def path_replacements(
    paths: ReproductionPaths,
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return longest-first path substitutions for portable committed evidence."""
    environment = manifest.get("environment", {})
    manifest_paths = manifest.get("paths", {})
    current_python_prefix = Path(sys.executable).resolve().parent.parent
    candidates = (
        (str(paths.root), "<reproduction-root>"),
        (str(manifest_paths.get("root", "")), "<reproduction-root>"),
        (str(manifest.get("source", {}).get("worktree", "")), "<source-checkout>"),
        (str(environment.get("prefix", "")), "<python-prefix>"),
        (str(current_python_prefix), "<python-prefix>"),
        (str(Path.home()), "<home>"),
    )
    return tuple(
        sorted(
            ((source, label) for source, label in candidates if source),
            key=lambda row: len(row[0]),
            reverse=True,
        ),
    )


def rewrite_json_as_portable(
    destination: Path,
    replacements: Sequence[tuple[str, str]],
) -> None:
    """Normalize every JSON path after raw evidence is verified and copied."""
    for path in sorted(destination.rglob("*.json")):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(portable_value(value, replacements), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def assert_no_machine_paths(destination: Path) -> None:
    """Reject committed text evidence containing host-specific absolute paths."""
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.suffix not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        match = MACHINE_PATH_PATTERN.search(text)
        if match is not None:
            msg = f"machine-specific absolute path remains in {path}: {match.group()}"
            raise RuntimeError(msg)


def _copy_compact(source: Path, destination: Path, names: Sequence[str]) -> None:
    """Copy required compact files and fail on omissions."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if not path.is_file():
            msg = f"missing compact evidence file: {path}"
            raise FileNotFoundError(msg)
        shutil.copy2(path, destination / name)


def _readme(manifest: Mapping[str, Any], claims: Mapping[str, Any]) -> str:
    """Return the human-readable clean-room handoff bundled with the evidence."""
    return f"""# Contextual Sparse ETM clean-room reproduction

This bundle was generated from reproduction `{manifest['reproduction_id']}` at
source commit `{manifest['source']['commit']}`. Models were fitted on training
spectra, selected and ablated on validation spectra, frozen, and then evaluated
on the fixed test split. `validation_comparison.csv` records development-split
evidence; `comparison.csv` and `stability_by_seed.csv` contain final test results.
Chemical scores are computed from the sealed full-spectrum topic mixtures and
MAG annotations using one dominant-topic association per spectrum.

## Evidence checks

Scientific integrity and directional checks passed: **{claims['all_passed']}**. Inspect
`acceptance.json`, `data_quality.json`, `fresh_evidence_manifest.json`, and the
CSV/JSON result tables for the complete evidence trail.

`reproduction_manifest.json` records the frozen fit and inference provenance.
`acceptance.json` is the authoritative audited interpretation of its scientific
gates. In particular, high-K planted-motif recovery is counted by one-to-one
truth matching at topic-word cosine at least 0.50; it is not the same quantity
as the number of fitted topics that become a document-level top-1 winner.
"""


def _copy_model_evidence(
    run: Path,
    destination: Path,
    *,
    method: str,
    model_files: Sequence[str],
    validation_chemistry: Mapping[str, Any],
    test_chemistry: Mapping[str, Any],
) -> None:
    """Copy compact validation, frozen-test, and split-boundary evidence."""
    _copy_compact(run / "models" / method, destination, model_files)
    _copy_compact(
        run / "evaluation" / method,
        destination / "test_evaluation",
        COMPACT_TEST_EVALUATION_FILES,
    )
    write_json(destination / "test_chemical.json", test_chemistry)
    write_json(destination / "validation_chemical.json", validation_chemistry)
    for name in ("validation_input_manifest.json", "test_input_manifest.json"):
        shutil.copy2(run / name, destination / name)


def write_summary_artifacts(
    destination: Path,
    evidence: Mapping[str, Any],
) -> None:
    """Write compact machine-readable summaries and tables."""
    json_outputs = {
        "preparation_summary.json": evidence["preparation"],
        "protocol.json": evidence["protocol"],
        "config.json": evidence["proposed"]["config"],
        "metrics.json": evidence["proposed"]["metrics"],
        "validation_metrics.json": evidence["proposed"]["validation_metrics"],
        "tomotopy.json": evidence["tomotopy"],
        "stability_summary.json": evidence["stability"],
        "acceptance.json": evidence["claims"],
        "data_quality.json": evidence["data_quality"],
    }
    csv_outputs = {
        "comparison.csv": evidence["comparison"],
        "validation_comparison.csv": evidence["validation_comparison"],
        "synthetic_by_seed.csv": evidence["primary"],
        "synthetic_summary.csv": evidence["synthetic_summary"],
        "high_k_stress.csv": evidence["high_k"],
        "stability_by_seed.csv": evidence["stability"]["by_seed"],
    }
    for name, value in json_outputs.items():
        write_json(destination / name, value)
    for name, rows in csv_outputs.items():
        write_csv(destination / name, rows)


def copy_raw_evidence(
    paths: ReproductionPaths,
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    claims: Mapping[str, Any],
    chemical_results: Mapping[str, Any],
    tomotopy_test_raw: Mapping[str, Any],
) -> None:
    """Copy compact model, split-boundary, and stage-provenance artifacts."""
    for seed in TRAINING_SEEDS:
        _copy_model_evidence(
            paths.contextual[seed],
            destination / "contextual" / f"seed_{seed}",
            method=METHOD,
            model_files=COMPACT_MODEL_FILES,
            validation_chemistry=chemical_results["contextual"][seed]["validation"],
            test_chemistry=chemical_results["contextual"][seed]["test"],
        )
    for method in ("etm", "etm_balanced"):
        _copy_model_evidence(
            paths.controls,
            destination / "controls" / method,
            method=method,
            model_files=COMPACT_CONTROL_FILES,
            validation_chemistry=chemical_results["controls"][method]["validation"],
            test_chemistry=chemical_results["controls"][method]["test"],
        )
    for result_path in sorted(
        (paths.synthetic / "synthetic_runs").glob("*/result.json"),
    ):
        target = (
            destination / "synthetic_results" / result_path.parent.name / "result.json"
        )
        target.parent.mkdir(parents=True)
        shutil.copy2(result_path, target)
    for source, name in (
        (paths.tomotopy / "tomotopy/validation_only_result.json", "tomotopy_raw.json"),
        (paths.assets / "acquisition_manifest.json", "acquisition_manifest.json"),
    ):
        shutil.copy2(source, destination / name)
    reproduction_manifest = {
        key: value for key, value in manifest.items() if key != "acceptance_policy"
    }
    reproduction_manifest["chemical_evaluation"] = {
        "association_rule": "dominant_topic",
        "derived_from_frozen_full_spectrum_mixtures": True,
        "derived_from_frozen_mag_annotations": True,
    }
    write_json(destination / "reproduction_manifest.json", reproduction_manifest)
    write_json(destination / "tomotopy_test_raw.json", tomotopy_test_raw)
    stage_directory = destination / "stage_records"
    stage_directory.mkdir()
    for stage in stage_plan(paths):
        shutil.copy2(
            paths.stages / f"{stage.name}.json",
            stage_directory / f"{stage.name}.json",
        )
    tomotopy_boundaries = destination / "tomotopy"
    tomotopy_boundaries.mkdir()
    for name in ("validation_input_manifest.json", "test_input_manifest.json"):
        shutil.copy2(paths.tomotopy / name, tomotopy_boundaries / name)
    (destination / "README.md").write_text(
        _readme(manifest, claims),
        encoding="utf-8",
    )


def write_package_seals(
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    claims: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    replacements: Sequence[tuple[str, str]],
) -> None:
    """Seal all compact files and write the report-facing checkpoint."""
    raw_outputs = [
        output_row for stage in stage_records for output_row in stage.get("outputs", [])
    ]
    seal = {
        "schema_version": 1,
        "reproduction_id": manifest["reproduction_id"],
        "source": portable_value(manifest["source"], replacements),
        "split_protocol": (
            "fit on train; select and ablate on validation; evaluate frozen models "
            "on test"
        ),
        "method": METHOD,
        "training_seeds": list(TRAINING_SEEDS),
        "synthetic_seeds": list(SYNTHETIC_SEEDS),
        "stage_count": len(stage_records),
        "raw_stage_outputs": portable_value(raw_outputs, replacements),
        "packaged_files": [
            file_record(path, relative_to=destination)
            for path in sorted(
                path for path in destination.rglob("*") if path.is_file()
            )
        ],
    }
    write_json(destination / "fresh_evidence_manifest.json", seal)
    checkpoint = {
        "schema_version": 2,
        "method": METHOD,
        "split_protocol": seal["split_protocol"],
        "reproduction_id": manifest["reproduction_id"],
        "source_commit": manifest["source"]["commit"],
        "test_released_after_model_and_validation_freeze": True,
        "fresh_evidence_manifest_sha256": sha256_file(
            destination / "fresh_evidence_manifest.json",
        ),
        "acceptance_all_passed": claims["all_passed"],
        "data_quality_status": data_quality["status"],
    }
    write_json(destination / "checkpoint_manifest.json", checkpoint)
