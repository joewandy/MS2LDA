"""Focused regression tests for MAG reuse and Tomotopy recovery."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import benchmarks.msnlib_validation.mag as validation_mag
import benchmarks.msnlib_validation.models as validation_models
from benchmarks.msnlib_validation.config import file_sha256, write_json


class _SavedModel:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def save(self, path: str) -> None:
        if self.fail:
            raise OSError("injected save failure")
        Path(path).write_bytes(self.payload)


class _Loader:
    @staticmethod
    def load(path: str) -> bytes:
        return Path(path).read_bytes()


def _history(iteration: int) -> list[dict[str, float | int]]:
    return [
        {
            "iteration": iteration,
            "ll_per_word": -1.0,
            "perplexity": 2.0,
            "cumulative_training_seconds": float(iteration),
        }
    ]


def test_tomotopy_checkpoint_falls_back_to_previous_valid_generation(
    tmp_path: Path,
) -> None:
    context = "a" * 64
    validation_models._save_tomotopy_checkpoint(
        _SavedModel(b"first"),
        tmp_path,
        context_sha256=context,
        history=_history(10),
    )
    newest = validation_models._save_tomotopy_checkpoint(
        _SavedModel(b"second"),
        tmp_path,
        context_sha256=context,
        history=_history(20),
    )
    (tmp_path / "checkpoints" / newest["file"]).write_bytes(b"corrupt")

    model, history, audit = validation_models._restore_tomotopy_checkpoint(
        SimpleNamespace(LDAModel=_Loader),
        tmp_path,
        context_sha256=context,
    )

    assert model == b"first"
    assert history[-1]["iteration"] == 10
    assert audit["selected_checkpoint"]["iteration"] == 10
    assert len(audit["rejected_newer_checkpoints"]) == 1
    assert "mismatch" in audit["rejected_newer_checkpoints"][0]["reason"]


def test_tomotopy_checkpoint_keeps_two_generations_and_survives_save_failure(
    tmp_path: Path,
) -> None:
    context = "b" * 64
    for iteration in (10, 20, 30):
        validation_models._save_tomotopy_checkpoint(
            _SavedModel(str(iteration).encode()),
            tmp_path,
            context_sha256=context,
            history=_history(iteration),
        )
    sidecars = validation_models._tomotopy_checkpoint_sidecars(tmp_path)
    assert len(sidecars) == 2
    assert [
        validation_models._verified_tomotopy_checkpoint(path, context_sha256=context)[
            "iteration"
        ]
        for path in sidecars
    ] == [30, 20]

    with pytest.raises(OSError, match="injected save failure"):
        validation_models._save_tomotopy_checkpoint(
            _SavedModel(b"never-published", fail=True),
            tmp_path,
            context_sha256=context,
            history=_history(40),
        )
    model, history, _ = validation_models._restore_tomotopy_checkpoint(
        SimpleNamespace(LDAModel=_Loader),
        tmp_path,
        context_sha256=context,
    )
    assert model == b"30"
    assert history[-1]["iteration"] == 30


def test_tomotopy_checkpoint_survives_fault_before_sidecar_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = "e" * 64
    validation_models._save_tomotopy_checkpoint(
        _SavedModel(b"first"),
        tmp_path,
        context_sha256=context,
        history=_history(10),
    )
    original_write_json = validation_models.write_json

    def fail_sidecar(path, value):
        destination = Path(path)
        if destination.name.startswith("checkpoint-"):
            raise OSError("injected sidecar publication failure")
        return original_write_json(destination, value)

    with monkeypatch.context() as context_patch:
        context_patch.setattr(validation_models, "write_json", fail_sidecar)
        with pytest.raises(OSError, match="sidecar publication failure"):
            validation_models._save_tomotopy_checkpoint(
                _SavedModel(b"unpublished"),
                tmp_path,
                context_sha256=context,
                history=_history(20),
            )

    model, history, _ = validation_models._restore_tomotopy_checkpoint(
        SimpleNamespace(LDAModel=_Loader),
        tmp_path,
        context_sha256=context,
    )
    assert model == b"first"
    assert history[-1]["iteration"] == 10


def test_tomotopy_checkpoint_rejects_wrong_context_and_unsafe_legacy_pair(
    tmp_path: Path,
) -> None:
    validation_models._save_tomotopy_checkpoint(
        _SavedModel(b"model"),
        tmp_path,
        context_sha256="c" * 64,
        history=_history(10),
    )
    with pytest.raises(RuntimeError, match="no valid Tomotopy checkpoint"):
        validation_models._restore_tomotopy_checkpoint(
            SimpleNamespace(LDAModel=_Loader),
            tmp_path,
            context_sha256="d" * 64,
        )

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "model.bin.partial").write_bytes(b"old")
    with pytest.raises(RuntimeError, match="legacy Tomotopy checkpoint"):
        validation_models._restore_tomotopy_checkpoint(
            SimpleNamespace(LDAModel=_Loader),
            legacy,
            context_sha256="d" * 64,
        )


def test_topic_spectra_use_frozen_significant_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def create_spectrum(words, topic_id, **kwargs):
        calls.append((words, topic_id, kwargs))
        return topic_id

    package = ModuleType("MS2LDA")
    utilities = ModuleType("MS2LDA.utils")
    utilities.create_spectrum = create_spectrum
    package.utils = utilities
    monkeypatch.setitem(sys.modules, "MS2LDA", package)
    monkeypatch.setitem(sys.modules, "MS2LDA.utils", utilities)
    spectra = validation_mag._topic_spectra(
        np.asarray([[0.7, 0.3]]),
        ["frag@100.123", "loss@20.123"],
        2,
        significant_digits=3,
    )
    assert spectra == [0]
    assert calls[0][2]["significant_digits"] == 3


def _write_bound_mag_fixture(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    config_path = directory / "config.resolved.json"
    write_json(config_path, {"fixture": True})
    monkeypatch.setattr(
        validation_mag,
        "load_config",
        lambda _: SimpleNamespace(num_topics=2, seeds=(42,)),
    )
    core = directory / "core" / "seed_42" / "tomotopy"
    core.mkdir(parents=True)
    (core / "beta.npy").write_bytes(b"beta")
    (core / "test_theta.npy").write_bytes(b"core theta")
    write_json(
        core / "complete.json",
        {
            "method": "tomotopy",
            "seed": 42,
            "topic_count": 2,
            "beta_sha256": file_sha256(core / "beta.npy"),
            "theta_sha256": file_sha256(core / "test_theta.npy"),
        },
    )
    chemical = directory / "chemical_inference" / "seed_42" / "tomotopy"
    chemical.mkdir(parents=True)
    theta_name = "test_full_theta_standard.npy"
    (chemical / theta_name).write_bytes(b"chemical theta")
    write_json(
        chemical / "complete.json",
        {
            "protocol_sha256": "p" * 64,
            "theta_sha256": {theta_name: file_sha256(chemical / theta_name)},
        },
    )
    output = directory / "mag" / "seed_42" / "tomotopy"
    output.mkdir(parents=True)
    (output / "topics.jsonl").write_text("{}\n", encoding="utf-8")
    bindings = validation_mag._mag_input_bindings(
        directory,
        seed=42,
        method="tomotopy",
        protocol_sha256="p" * 64,
    )
    result = {
        "protocol_sha256": "p" * 64,
        "method": "tomotopy",
        "seed": 42,
        "motif_optimization_loss_err": 1,
        "topics_sha256": file_sha256(output / "topics.jsonl"),
        "input_bindings": bindings,
    }
    write_json(output / "complete.json", result)
    return result


def test_completed_mag_result_is_bound_to_core_and_chemical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write_bound_mag_fixture(tmp_path, monkeypatch)
    assert (
        validation_mag._verify_completed_mag_result(
            tmp_path,
            seed=42,
            method="tomotopy",
            protocol_sha256="p" * 64,
        )
        == expected
    )

    beta = tmp_path / "core" / "seed_42" / "tomotopy" / "beta.npy"
    beta.write_bytes(b"changed")
    with pytest.raises(ValueError, match="topic matrix changed"):
        validation_mag._verify_completed_mag_result(
            tmp_path,
            seed=42,
            method="tomotopy",
            protocol_sha256="p" * 64,
        )


def test_completed_mag_result_rejects_topics_and_optimization_setting_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _write_bound_mag_fixture(tmp_path, monkeypatch)
    output = tmp_path / "mag" / "seed_42" / "tomotopy"
    (output / "topics.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="topic rows changed"):
        validation_mag._verify_completed_mag_result(
            tmp_path,
            seed=42,
            method="tomotopy",
            protocol_sha256="p" * 64,
        )

    (output / "topics.jsonl").write_text("{}\n", encoding="utf-8")
    result["motif_optimization_loss_err"] = 0.5
    write_json(output / "complete.json", result)
    with pytest.raises(ValueError, match="optimization setting changed"):
        validation_mag._verify_completed_mag_result(
            tmp_path,
            seed=42,
            method="tomotopy",
            protocol_sha256="p" * 64,
        )


def test_run_all_mag_validates_every_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_sha256 = "p" * 64
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "complete.json").write_text("{}", encoding="utf-8")
    chemical = tmp_path / "chemical_inference"
    chemical.mkdir()
    write_json(
        chemical / "complete.json",
        {
            "protocol_sha256": protocol_sha256,
            "full_spectrum_peak_groups": True,
        },
    )
    for method in ("tomotopy", "hybrid"):
        result = tmp_path / "mag" / "seed_42" / method
        result.mkdir(parents=True)
        (result / "complete.json").write_text("{}", encoding="utf-8")
    dreams = tmp_path / "mag" / "raw_dreams"
    dreams.mkdir(parents=True)
    (dreams / "complete.json").write_text("{}", encoding="utf-8")
    index = tmp_path / "mag" / "index"
    index.mkdir(parents=True)
    (index / "manifest.json").write_text("{}", encoding="utf-8")

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        validation_mag,
        "verify_protocol",
        lambda _: {
            "protocol_sha256": protocol_sha256,
            "data_root": str(tmp_path),
            "repo_root": str(tmp_path),
        },
    )
    monkeypatch.setattr(validation_mag, "verify_frozen_input_files", lambda *a, **k: {})
    monkeypatch.setattr(
        validation_mag,
        "load_config",
        lambda _: SimpleNamespace(seeds=(42,)),
    )
    monkeypatch.setattr(
        validation_mag,
        "build_filtered_mag_index",
        lambda *a, **k: calls.append(("index", None)) or {"validated": True},
    )

    def verify_result(*_args, method, **_kwargs):
        calls.append(("result", method))
        return {"method": method, "seed": 42}

    monkeypatch.setattr(validation_mag, "_verify_completed_mag_result", verify_result)
    monkeypatch.setattr(
        validation_mag,
        "_verify_raw_dreams_result",
        lambda *a, **k: calls.append(("raw", None)) or {"method": "raw_dreams"},
    )
    monkeypatch.setattr(validation_mag, "environment_manifest", lambda: {})

    result = validation_mag.run_all_mag(tmp_path, data_root=tmp_path)

    assert calls == [
        ("index", None),
        ("result", "tomotopy"),
        ("result", "hybrid"),
        ("raw", None),
    ]
    assert result["index"] == {"validated": True}
