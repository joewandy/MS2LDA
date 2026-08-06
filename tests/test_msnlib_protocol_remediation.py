"""Regression tests for the independent MSnLib protocol review findings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.msnlib_validation import cli
from benchmarks.msnlib_validation import protocol as validation_protocol
from benchmarks.msnlib_validation import reuse as validation_reuse
from benchmarks.msnlib_validation.config import (
    file_sha256,
    load_config,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "benchmarks" / "msnlib_validation" / "configs"


def test_historical_config_retains_prespecified_timing_defaults() -> None:
    config = load_config(CONFIG_DIR / "indicative-msnlib-k1000-seed42.json")

    assert config.evaluation_timing == "prespecified"
    assert config.prior_test_results_inspected is False


def test_peak_pooling_config_discloses_posthoc_implementation_correction() -> None:
    config = load_config(
        CONFIG_DIR / "indicative-msnlib-k1000-seed42-peak-pooling-correction.json"
    )

    assert config.evaluation_timing == "posthoc_implementation_correction"
    assert config.prior_test_results_inspected is True
    with pytest.raises(ValueError, match="are inconsistent"):
        replace(config, prior_test_results_inspected=False)


@pytest.mark.parametrize("command", ["validate-inputs", "preflight", "freeze"])
def test_entry_commands_require_explicit_config(command: str) -> None:
    arguments = [command, "--data-root", "/data"]
    if command == "freeze":
        arguments.extend(["--run", "/run"])

    with pytest.raises(SystemExit):
        cli._parser().parse_args(arguments)


def test_direct_freeze_cannot_override_immutable_inspection_status() -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "freeze",
                "--config",
                "/config.json",
                "--data-root",
                "/data",
                "--run",
                "/run",
                "--test-results-inspected",
            ]
        )


def test_mag_production_dependency_closure_is_hashed() -> None:
    files = {
        str(path.relative_to(REPO_ROOT))
        for path in validation_protocol._mag_source_files(REPO_ROOT)
    }

    assert {
        "MS2LDA/utils.py",
        "MS2LDA/Add_On/Spec2Vec/annotation.py",
        "MS2LDA/Add_On/Spec2Vec/annotation_refined.py",
        "MS2LDA/Mass2Motif.py",
        "MS2LDA/Mass2MotifDocument.py",
        "MS2LDA/Visualisation/visualisation.py",
    } <= files
    manifest = validation_protocol.code_manifest(REPO_ROOT)
    assert all(name in manifest for name in files)


def test_implementation_correction_derivation_freezes_scientific_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_config = load_config(
        CONFIG_DIR / "indicative-msnlib-k1000-seed42-chemical-correction.json"
    )
    target_config = load_config(
        CONFIG_DIR / "indicative-msnlib-k1000-seed42-peak-pooling-correction.json"
    )
    write_json(source / "config.resolved.json", source_config.as_dict())
    monkeypatch.setattr(
        validation_protocol,
        "verify_protocol",
        lambda *_args, **_kwargs: {"protocol_sha256": "a" * 64},
    )

    derivation = validation_protocol.validate_implementation_correction_derivation(
        source,
        target_config,
        "Correct peak-level DreaMS pooling after independent review.",
    )

    assert derivation["kind"] == "implementation_correction"
    assert derivation["tomotopy_core_reusable"] is True
    assert derivation["feature_cache_reusable"] is False
    assert derivation["hybrid_artifacts_reusable"] is False
    assert set(derivation["differences"]) == {
        "evaluation_timing",
        "prior_test_results_inspected",
        "protocol_name",
    }
    with pytest.raises(ValueError, match="may change only"):
        validation_protocol.validate_implementation_correction_derivation(
            source,
            replace(target_config, alpha=0.7),
            "Invalid scientific change.",
        )


def _make_tomotopy_reuse_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    source_config = load_config(
        CONFIG_DIR / "indicative-msnlib-k1000-seed42-chemical-correction.json"
    )
    target_config = load_config(
        CONFIG_DIR / "indicative-msnlib-k1000-seed42-peak-pooling-correction.json"
    )
    write_json(source / "config.resolved.json", source_config.as_dict())
    write_json(target / "config.resolved.json", target_config.as_dict())
    input_files = {"mgf": {"bytes": 1, "path": "/data/input", "sha256": "f" * 64}}
    write_json(source / "input_manifest.json", {"files": input_files})
    write_json(target / "input_manifest.json", {"files": input_files})
    frozen_hashes = {}
    for name in validation_reuse.IDENTICAL_FROZEN_ARTIFACTS:
        (source / name).write_text(name, encoding="utf-8")
        (target / name).write_text(name, encoding="utf-8")
        frozen_hashes[name] = file_sha256(source / name)

    tomotopy = source / "core" / "seed_42" / "tomotopy"
    tomotopy.mkdir(parents=True)
    for name, contents in {
        "beta.npy": b"beta",
        "test_theta.npy": b"theta",
        "model.bin": b"model",
    }.items():
        (tomotopy / name).write_bytes(contents)
    write_json(
        tomotopy / "complete.json",
        {
            "beta_sha256": file_sha256(tomotopy / "beta.npy"),
            "method": "tomotopy",
            "model_sha256": file_sha256(tomotopy / "model.bin"),
            "seed": 42,
            "theta_sha256": file_sha256(tomotopy / "test_theta.npy"),
            "topic_count": 1000,
            "training_iterations": 1200,
            "training_parallel_scheme_value": 3,
            "training_workers_requested": 6,
        },
    )
    derivation = {
        "kind": "implementation_correction",
        "reason": "Correct peak-level DreaMS pooling after independent review.",
    }
    source_lock = {
        "artifacts": frozen_hashes,
        "protocol_sha256": "a" * 64,
    }
    target_lock = {
        "artifacts": frozen_hashes,
        "derivation": derivation,
        "protocol_sha256": "b" * 64,
        "test_results_inspected": True,
    }
    return source, target, source_lock, target_lock


def test_tomotopy_reuse_never_imports_features_or_hybrid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, source_lock, target_lock = _make_tomotopy_reuse_fixture(tmp_path)
    recorded = target_lock["derivation"]
    validated = {**recorded, "validated": True}
    target_lock["derivation"] = validated
    monkeypatch.setattr(
        validation_reuse,
        "verify_protocol",
        lambda path, **_kwargs: (
            source_lock if Path(path).resolve() == source.resolve() else target_lock
        ),
    )
    monkeypatch.setattr(
        validation_reuse,
        "validate_implementation_correction_derivation",
        lambda *_args, **_kwargs: validated,
    )
    (source / "features").mkdir()
    (source / "features" / "must-not-copy").write_text("features", encoding="utf-8")
    hybrid = source / "core" / "seed_42" / "hybrid"
    hybrid.mkdir()
    (hybrid / "must-not-copy").write_text("hybrid", encoding="utf-8")

    first = validation_reuse.reuse_tomotopy_artifacts(source, target)
    second = validation_reuse.reuse_tomotopy_artifacts(source, target)

    assert first["reuse_scope"] == "tomotopy_core_only"
    assert second["reused"]["tomotopy"]["42"]["training_iterations"] == 1200
    assert (target / "core" / "seed_42" / "tomotopy" / "model.bin").is_file()
    assert not (target / "features").exists()
    assert not (target / "core" / "seed_42" / "hybrid").exists()
    assert first["forbidden_reuse"] == ["features", "hybrid"]


def test_general_reuse_refuses_implementation_correction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, source_lock, target_lock = _make_tomotopy_reuse_fixture(tmp_path)
    monkeypatch.setattr(
        validation_reuse,
        "verify_protocol",
        lambda path, **_kwargs: (
            source_lock if Path(path).resolve() == source.resolve() else target_lock
        ),
    )

    with pytest.raises(ValueError, match="reuse Tomotopy only"):
        validation_reuse.reuse_core_artifacts(source, target)


def test_tomotopy_reuse_rejects_scientific_config_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, target, source_lock, target_lock = _make_tomotopy_reuse_fixture(tmp_path)
    target_config = load_config(target / "config.resolved.json")
    write_json(
        target / "config.resolved.json", replace(target_config, alpha=0.7).as_dict()
    )
    validated = {**target_lock["derivation"], "validated": True}
    target_lock["derivation"] = validated
    monkeypatch.setattr(
        validation_reuse,
        "verify_protocol",
        lambda path, **_kwargs: (
            source_lock if Path(path).resolve() == source.resolve() else target_lock
        ),
    )
    monkeypatch.setattr(
        validation_reuse,
        "validate_implementation_correction_derivation",
        lambda *_args, **_kwargs: validated,
    )

    with pytest.raises(ValueError, match="identical scientific"):
        validation_reuse.reuse_tomotopy_artifacts(source, target)
