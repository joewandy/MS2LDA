"""Focused physical-peak pooling and atomic feature-checkpoint tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.msnlib_validation.features as feature_module
from benchmarks.msnlib_validation.data import PeakGroup, SpectrumRecord
from benchmarks.msnlib_validation.features import (
    _empty_word_pool_counters,
    _ensure_feature_checkpoint_format,
    _load_latest_feature_checkpoint,
    _read_feature_checkpoint_generation,
    _update_word_pool,
    _write_feature_checkpoint_generation,
)
from ms2lda_hybrid.dreams_features import DreaMSFeatureBatch

PROTOCOL_SHA256 = "a" * 64


def _record(identifier: str, groups: tuple[PeakGroup, ...]) -> SpectrumRecord:
    return SpectrumRecord(
        spectrum_id=identifier,
        feature_id=identifier,
        smiles="CC",
        supplied_inchikey="OTMSDBZUPAUEDD-UHFFFAOYSA-N",
        connectivity_key="OTMSDBZUPAUEDD",
        scaffold_key="",
        split_group=f"acyclic:{identifier}",
        precursor_mz=150.0,
        peak_groups=groups,
        declared_num_peaks=len(groups),
        parsed_num_peaks=len(groups),
        compound_name=identifier,
        metadata={},
    )


def _load(checkpoint_dir: Path) -> dict[str, object]:
    restored = _load_latest_feature_checkpoint(
        checkpoint_dir,
        protocol_sha256=PROTOCOL_SHA256,
        total_rows=2,
        vocabulary_size=2,
        embedding_dim=3,
    )
    assert restored is not None
    return restored


def _save(
    checkpoint_dir: Path,
    generation: int,
    *,
    completed: np.ndarray,
    sums: np.ndarray,
    weights: np.ndarray,
    counters: dict[str, int],
    fault_hook=None,
) -> None:
    _write_feature_checkpoint_generation(
        checkpoint_dir,
        generation=generation,
        protocol_sha256=PROTOCOL_SHA256,
        completed=completed,
        sums=sums,
        weights=weights,
        cumulative_extraction_seconds=float(generation),
        word_pool_counters=counters,
        fault_hook=fault_hook,
    )


def test_rounded_word_collisions_preserve_each_physical_peak_context() -> None:
    groups = (
        PeakGroup(
            original_index=0,
            mz=100.011,
            intensity=0.02,
            tokens=("frag@100.0", "frag@100.0", "loss@50.0", "loss@50.0"),
        ),
        PeakGroup(
            original_index=1,
            mz=100.039,
            intensity=0.03,
            tokens=(
                "frag@100.0",
                "frag@100.0",
                "frag@100.0",
                "loss@50.0",
                "loss@50.0",
                "loss@50.0",
            ),
        ),
    )
    record = _record("spectrum-1", groups)
    batch = DreaMSFeatureBatch(
        identifiers=(record.spectrum_id,),
        spectrum_embeddings=np.zeros((1, 3), dtype=np.float32),
        peak_embeddings=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32
        ),
        peak_mz=np.asarray([[100.011, 100.039]], dtype=np.float32),
        peak_mask=np.asarray([[True, True]], dtype=np.bool_),
        precursor_mz=np.asarray([150.0], dtype=np.float32),
        provenance={},
    )
    sums = np.zeros((2, 3), dtype=np.float32)
    weights = np.zeros(2, dtype=np.float64)

    counters = _update_word_pool(
        records=[record],
        feature_batch=batch,
        vocabulary_index={"frag@100.0": 0, "loss@50.0": 1},
        sums=sums,
        weights=weights,
    )

    expected = np.asarray([2.0, 6.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(sums[0], expected)
    np.testing.assert_array_equal(sums[1], expected)
    np.testing.assert_array_equal(weights, np.asarray([5.0, 5.0]))
    assert counters["training_peak_groups"] == 2
    assert counters["matched_peak_groups"] == 2
    assert counters["matched_token_occurrences"] == 10
    assert counters["fragment_collision_documents"] == 1
    assert counters["fragment_collision_words"] == 1
    assert counters["fragment_collision_extra_peak_groups"] == 1
    assert counters["neutral_loss_collision_documents"] == 1
    assert counters["neutral_loss_collision_words"] == 1
    assert counters["neutral_loss_collision_extra_peak_groups"] == 1


def test_prepare_features_records_peak_pool_counters_in_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    groups = (
        PeakGroup(0, 100.011, 0.02, ("frag@100.0", "loss@50.0")),
        PeakGroup(1, 100.039, 0.03, ("frag@100.0", "loss@50.0")),
    )
    record = _record("spectrum-1", groups)

    class FakeExtractor:
        provenance = {"test": True}

        def __init__(self, *, device: str) -> None:
            assert device == "cpu"

        def extract(self, spectra, *, identifiers, batch_size):
            assert len(spectra) == batch_size == 1
            return DreaMSFeatureBatch(
                identifiers=tuple(identifiers),
                spectrum_embeddings=np.zeros((1, 1024), dtype=np.float32),
                peak_embeddings=np.stack(
                    [
                        np.stack(
                            [
                                np.full(1024, 1.0, dtype=np.float16),
                                np.full(1024, 2.0, dtype=np.float16),
                            ]
                        )
                    ]
                ),
                peak_mz=np.asarray([[100.011, 100.039]], dtype=np.float32),
                peak_mask=np.asarray([[True, True]], dtype=np.bool_),
                precursor_mz=np.asarray([150.0], dtype=np.float32),
                provenance=self.provenance,
            )

    monkeypatch.setattr(
        feature_module,
        "verify_protocol",
        lambda _: {"protocol_sha256": PROTOCOL_SHA256, "data_root": str(tmp_path)},
    )
    monkeypatch.setattr(
        feature_module, "verify_frozen_input_files", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        feature_module,
        "load_config",
        lambda _: SimpleNamespace(significant_digits=1),
    )
    monkeypatch.setattr(
        feature_module, "resolve_input_paths", lambda *_: {"mgf": tmp_path / "x.mgf"}
    )
    monkeypatch.setattr(feature_module, "load_records", lambda *_: ([record], {}))
    monkeypatch.setattr(
        feature_module, "load_assignments", lambda _: {record.spectrum_id: "train"}
    )
    monkeypatch.setattr(feature_module, "load_completion_rows", lambda _: {})
    monkeypatch.setattr(
        feature_module,
        "load_vocabulary",
        lambda _: ["frag@100.0", "loss@50.0"],
    )
    monkeypatch.setattr(feature_module, "to_matchms_spectrum", lambda _: object())
    monkeypatch.setattr(feature_module, "DreaMSFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(feature_module, "peak_rss_bytes", lambda: 123)

    manifest = feature_module.prepare_features(
        tmp_path, extraction_batch_size=1, checkpoint_every_chunks=1
    )

    assert manifest["word_pool_strategy"] == "physical_peak_group_v2"
    assert manifest["word_pool_counters"]["matched_peak_groups"] == 2
    assert manifest["word_pool_counters"]["fragment_collision_documents"] == 1
    assert manifest["word_pool_counters"]["neutral_loss_collision_documents"] == 1
    assert manifest["last_checkpoint_generation"] == 1


@pytest.mark.parametrize(
    "fault_stage",
    ["after_sums", "after_weights", "after_completed", "after_state", "after_publish"],
)
def test_checkpoint_fault_recovery_equals_uninterrupted_accumulation(
    tmp_path: Path, fault_stage: str
) -> None:
    checkpoint_dir = tmp_path / fault_stage
    base_sums = np.asarray([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    base_weights = np.asarray([1.0, 1.0], dtype=np.float64)
    base_completed = np.asarray([True, False], dtype=np.bool_)
    base_counters = _empty_word_pool_counters()
    base_counters["matched_peak_groups"] = 1
    _save(
        checkpoint_dir,
        1,
        completed=base_completed,
        sums=base_sums,
        weights=base_weights,
        counters=base_counters,
    )

    delta_sums = np.asarray([[4.0, 0.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    delta_weights = np.asarray([2.0, 3.0], dtype=np.float64)
    expected_sums = base_sums + delta_sums
    expected_weights = base_weights + delta_weights
    expected_completed = np.asarray([True, True], dtype=np.bool_)
    expected_counters = dict(base_counters)
    expected_counters["matched_peak_groups"] += 2

    def fail_at(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"injected fault at {stage}")

    with pytest.raises(RuntimeError, match="injected fault"):
        _save(
            checkpoint_dir,
            2,
            completed=expected_completed,
            sums=expected_sums,
            weights=expected_weights,
            counters=expected_counters,
            fault_hook=fail_at,
        )

    restored = _load(checkpoint_dir)
    if fault_stage == "after_publish":
        recovered_sums = restored["sums"]
        recovered_weights = restored["weights"]
        recovered_completed = restored["completed"]
        recovered_counters = restored["counters"]
    else:
        recovered_sums = restored["sums"] + delta_sums
        recovered_weights = restored["weights"] + delta_weights
        recovered_completed = expected_completed
        recovered_counters = dict(restored["counters"])
        recovered_counters["matched_peak_groups"] += 2
    np.testing.assert_array_equal(recovered_sums, expected_sums)
    np.testing.assert_array_equal(recovered_weights, expected_weights)
    np.testing.assert_array_equal(recovered_completed, expected_completed)
    assert recovered_counters == expected_counters
    assert not list(checkpoint_dir.glob(".*.tmp"))


def test_corrupt_newest_generation_falls_back_and_two_verified_are_retained(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    arrays = []
    for generation in range(1, 4):
        sums = np.full((2, 3), generation, dtype=np.float32)
        weights = np.full(2, generation, dtype=np.float64)
        completed = np.asarray([True, generation >= 2], dtype=np.bool_)
        arrays.append((sums, weights, completed))
        _save(
            checkpoint_dir,
            generation,
            completed=completed,
            sums=sums,
            weights=weights,
            counters=counters,
        )

    assert not (checkpoint_dir / "generation-00000001").exists()
    assert (checkpoint_dir / "generation-00000002").exists()
    newest = checkpoint_dir / "generation-00000003"
    np.save(newest / "weights.npy", np.asarray([999.0, 999.0], dtype=np.float64))

    restored = _load(checkpoint_dir)
    assert restored["state"]["generation"] == 2
    np.testing.assert_array_equal(restored["sums"], arrays[1][0])
    assert restored["rejected_newer_generations"][0]["generation"] == newest.name


def test_legacy_additive_checkpoint_is_rejected(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "word_embedding_sums.npy", np.zeros((2, 3)))

    with pytest.raises(RuntimeError, match="cannot be resumed safely"):
        _ensure_feature_checkpoint_format(
            feature_dir,
            protocol_sha256=PROTOCOL_SHA256,
            rows=2,
            vocabulary_size=2,
        )


def test_checkpoint_generation_rejects_noncontiguous_completion(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    with pytest.raises(ValueError, match="non-contiguous"):
        _save(
            checkpoint_dir,
            1,
            completed=np.asarray([False, True], dtype=np.bool_),
            sums=np.zeros((2, 3), dtype=np.float32),
            weights=np.zeros(2, dtype=np.float64),
            counters=counters,
        )

    assert not list(checkpoint_dir.glob("generation-*"))


def test_checkpoint_reader_detects_array_tampering(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    _save(
        checkpoint_dir,
        1,
        completed=np.asarray([True, False], dtype=np.bool_),
        sums=np.zeros((2, 3), dtype=np.float32),
        weights=np.zeros(2, dtype=np.float64),
        counters=counters,
    )
    generation = checkpoint_dir / "generation-00000001"
    np.save(generation / "sums.npy", np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="array changed"):
        _read_feature_checkpoint_generation(
            generation,
            protocol_sha256=PROTOCOL_SHA256,
            total_rows=2,
            vocabulary_size=2,
            embedding_dim=3,
        )
