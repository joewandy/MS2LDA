"""Focused physical-peak pooling and atomic feature-checkpoint tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.msnlib_validation.chemical as chemical_module
import benchmarks.msnlib_validation.features as feature_module
import ms2lda_hybrid.dreams_features as dreams_features_module
from benchmarks.msnlib_validation.chemical import (
    _load_chemical_feature_checkpoint,
    _write_chemical_feature_checkpoint,
)
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
IDENTIFIERS_SHA256 = "b" * 64


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


def _load(checkpoint_dir: Path, global_embeddings: np.ndarray) -> dict[str, object]:
    restored = _load_latest_feature_checkpoint(
        checkpoint_dir,
        protocol_sha256=PROTOCOL_SHA256,
        total_rows=len(global_embeddings),
        vocabulary_size=2,
        embedding_dim=3,
        global_embeddings=global_embeddings,
        identifiers_sha256=IDENTIFIERS_SHA256,
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
    global_embeddings: np.ndarray,
    global_embedding_chunks: list[dict[str, object]],
    fault_hook=None,
) -> dict[str, object]:
    completed_rows = (
        int(np.flatnonzero(~completed)[0]) if not np.all(completed) else len(completed)
    )
    checkpoint_start = (
        int(global_embedding_chunks[-1]["end"]) if global_embedding_chunks else 0
    )
    return _write_feature_checkpoint_generation(
        checkpoint_dir,
        generation=generation,
        protocol_sha256=PROTOCOL_SHA256,
        completed=completed,
        sums=sums,
        weights=weights,
        cumulative_extraction_seconds=float(generation),
        word_pool_counters=counters,
        global_embeddings=global_embeddings,
        global_embedding_chunks=global_embedding_chunks,
        checkpoint_start=checkpoint_start,
        checkpoint_end=completed_rows,
        identifiers_sha256=IDENTIFIERS_SHA256,
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
    assert counters["retained_dreams_peak_states"] == 2
    assert counters["matched_peak_groups"] + counters["unmatched_peak_groups"] == 2
    assert counters["fragment_collision_documents"] == 1
    assert counters["fragment_collision_words"] == 1
    assert counters["fragment_collision_extra_peak_groups"] == 1
    assert counters["neutral_loss_collision_documents"] == 1
    assert counters["neutral_loss_collision_words"] == 1
    assert counters["neutral_loss_collision_extra_peak_groups"] == 1


def test_top_100_truncation_cannot_lend_a_nearby_peak_state() -> None:
    ordinary = tuple(
        PeakGroup(index, 10.0 + index, 1.0, (f"unused@{index}",)) for index in range(99)
    )
    retained = PeakGroup(99, 117.066757, 1.0, ("frag@117.07",))
    discarded = PeakGroup(100, 117.069855, 0.01, ("frag@117.07",))
    record = _record("top-100", (*ordinary, retained, discarded))
    retained_groups = record.peak_groups[:100]
    states = np.zeros((1, 100, 3), dtype=np.float32)
    states[0, 99] = np.asarray([7.0, 0.0, 0.0], dtype=np.float32)
    batch = DreaMSFeatureBatch(
        identifiers=(record.spectrum_id,),
        spectrum_embeddings=np.zeros((1, 3), dtype=np.float32),
        peak_embeddings=states,
        peak_mz=np.asarray([[group.mz for group in retained_groups]], dtype=np.float32),
        peak_mask=np.ones((1, 100), dtype=np.bool_),
        precursor_mz=np.asarray([150.0], dtype=np.float32),
        provenance={"n_highest_peaks": 100},
    )
    sums = np.zeros((1, 3), dtype=np.float32)
    weights = np.zeros(1, dtype=np.float64)

    counters = _update_word_pool(
        records=[record],
        feature_batch=batch,
        vocabulary_index={"frag@117.07": 0},
        sums=sums,
        weights=weights,
    )

    np.testing.assert_array_equal(sums[0], np.asarray([7.0, 0.0, 0.0]))
    np.testing.assert_array_equal(weights, np.asarray([1.0]))
    assert counters["matched_peak_groups"] == 1
    assert counters["unmatched_peak_groups"] == 1
    assert counters["training_peak_groups"] == 2
    assert counters["fragment_collision_extra_peak_groups"] == 1


def test_float32_duplicate_peak_identity_fails_closed() -> None:
    first = 117.066757
    second = float(np.nextafter(np.float64(first), np.float64(np.inf)))
    assert np.float32(first) == np.float32(second)
    record = _record(
        "ambiguous",
        (
            PeakGroup(0, first, 1.0, ("frag@117.07",)),
            PeakGroup(1, second, 0.5, ("frag@117.07",)),
        ),
    )
    batch = DreaMSFeatureBatch(
        identifiers=(record.spectrum_id,),
        spectrum_embeddings=np.zeros((1, 3), dtype=np.float32),
        peak_embeddings=np.ones((1, 1, 3), dtype=np.float32),
        peak_mz=np.asarray([[first]], dtype=np.float32),
        peak_mask=np.ones((1, 1), dtype=np.bool_),
        precursor_mz=np.asarray([150.0], dtype=np.float32),
        provenance={},
    )

    with pytest.raises(ValueError, match="ambiguous after float32"):
        _update_word_pool(
            records=[record],
            feature_batch=batch,
            vocabulary_index={"frag@117.07": 0},
            sums=np.zeros((1, 3), dtype=np.float32),
            weights=np.zeros(1, dtype=np.float64),
        )


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

    assert manifest["word_pool_strategy"] == "physical_peak_identity_v3"
    assert manifest["peak_identity_mapping"] == "exact_float32_source_mz"
    assert manifest["discarded_peak_state_policy"] == "unmatched"
    assert manifest["word_pool_counters"]["matched_peak_groups"] == 2
    assert manifest["word_pool_counters"]["unmatched_peak_groups"] == 0
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
    global_embeddings = np.asarray(
        [[10.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    base_counters = _empty_word_pool_counters()
    base_counters["matched_peak_groups"] = 1
    base_state = _save(
        checkpoint_dir,
        1,
        completed=base_completed,
        sums=base_sums,
        weights=base_weights,
        counters=base_counters,
        global_embeddings=global_embeddings,
        global_embedding_chunks=[],
    )

    delta_sums = np.asarray([[4.0, 0.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    delta_weights = np.asarray([2.0, 3.0], dtype=np.float64)
    expected_sums = base_sums + delta_sums
    expected_weights = base_weights + delta_weights
    expected_completed = np.asarray([True, True], dtype=np.bool_)
    global_embeddings[1] = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
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
            global_embeddings=global_embeddings,
            global_embedding_chunks=base_state["global_embedding_chunks"],
            fault_hook=fail_at,
        )

    restored = _load(checkpoint_dir, global_embeddings)
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
    global_embeddings = np.zeros((3, 3), dtype=np.float32)
    global_embedding_chunks: list[dict[str, object]] = []
    arrays = []
    for generation in range(1, 4):
        sums = np.full((2, 3), generation, dtype=np.float32)
        weights = np.full(2, generation, dtype=np.float64)
        completed = np.arange(3) < generation
        global_embeddings[generation - 1] = generation * 10.0
        arrays.append((sums, weights, completed))
        state = _save(
            checkpoint_dir,
            generation,
            completed=completed,
            sums=sums,
            weights=weights,
            counters=counters,
            global_embeddings=global_embeddings,
            global_embedding_chunks=global_embedding_chunks,
        )
        global_embedding_chunks = state["global_embedding_chunks"]

    assert not (checkpoint_dir / "generation-00000001").exists()
    assert (checkpoint_dir / "generation-00000002").exists()
    newest = checkpoint_dir / "generation-00000003"
    np.save(newest / "weights.npy", np.asarray([999.0, 999.0], dtype=np.float64))

    restored = _load(checkpoint_dir, global_embeddings)
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
            identifiers_sha256=IDENTIFIERS_SHA256,
        )


def test_checkpoint_generation_rejects_noncontiguous_completion(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    global_embeddings = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="non-contiguous"):
        _save(
            checkpoint_dir,
            1,
            completed=np.asarray([False, True], dtype=np.bool_),
            sums=np.zeros((2, 3), dtype=np.float32),
            weights=np.zeros(2, dtype=np.float64),
            counters=counters,
            global_embeddings=global_embeddings,
            global_embedding_chunks=[],
        )

    assert not list(checkpoint_dir.glob("generation-*"))


def test_checkpoint_reader_detects_array_tampering(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    global_embeddings = np.zeros((2, 3), dtype=np.float32)
    _save(
        checkpoint_dir,
        1,
        completed=np.asarray([True, False], dtype=np.bool_),
        sums=np.zeros((2, 3), dtype=np.float32),
        weights=np.zeros(2, dtype=np.float64),
        counters=counters,
        global_embeddings=global_embeddings,
        global_embedding_chunks=[],
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
            global_embeddings=global_embeddings,
            identifiers_sha256=IDENTIFIERS_SHA256,
        )


def test_global_embedding_tampering_falls_back_to_verified_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    counters = _empty_word_pool_counters()
    global_embeddings = np.zeros((2, 3), dtype=np.float32)
    global_embeddings[0] = 1.0
    first = _save(
        checkpoint_dir,
        1,
        completed=np.asarray([True, False], dtype=np.bool_),
        sums=np.zeros((2, 3), dtype=np.float32),
        weights=np.zeros(2, dtype=np.float64),
        counters=counters,
        global_embeddings=global_embeddings,
        global_embedding_chunks=[],
    )
    global_embeddings[1] = 2.0
    _save(
        checkpoint_dir,
        2,
        completed=np.asarray([True, True], dtype=np.bool_),
        sums=np.ones((2, 3), dtype=np.float32),
        weights=np.ones(2, dtype=np.float64),
        counters=counters,
        global_embeddings=global_embeddings,
        global_embedding_chunks=first["global_embedding_chunks"],
    )
    global_embeddings[1, 0] = 999.0

    restored = _load(checkpoint_dir, global_embeddings)

    assert restored["state"]["generation"] == 1
    assert "rows 1:2 changed" in restored["rejected_newer_generations"][0]["reason"]


def test_checkpoint_rejects_changed_identifier_inventory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    global_embeddings = np.ones((2, 3), dtype=np.float32)
    _save(
        checkpoint_dir,
        1,
        completed=np.asarray([True, False], dtype=np.bool_),
        sums=np.zeros((2, 3), dtype=np.float32),
        weights=np.zeros(2, dtype=np.float64),
        counters=_empty_word_pool_counters(),
        global_embeddings=global_embeddings,
        global_embedding_chunks=[],
    )

    with pytest.raises(RuntimeError, match="no valid atomic feature checkpoint"):
        _load_latest_feature_checkpoint(
            checkpoint_dir,
            protocol_sha256=PROTOCOL_SHA256,
            total_rows=2,
            vocabulary_size=2,
            embedding_dim=3,
            global_embeddings=global_embeddings,
            identifiers_sha256="c" * 64,
        )


def test_chemical_embedding_checkpoint_falls_back_after_tampering(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "chemical"
    embeddings = np.zeros((2, 3), dtype=np.float32)
    embeddings[0] = 1.0
    first = _write_chemical_feature_checkpoint(
        checkpoint_dir,
        generation=1,
        protocol_sha256=PROTOCOL_SHA256,
        identifiers_sha256=IDENTIFIERS_SHA256,
        embeddings=embeddings,
        embedding_chunks=[],
        start=0,
        end=1,
        cumulative_extraction_seconds=1.0,
    )
    embeddings[1] = 2.0
    _write_chemical_feature_checkpoint(
        checkpoint_dir,
        generation=2,
        protocol_sha256=PROTOCOL_SHA256,
        identifiers_sha256=IDENTIFIERS_SHA256,
        embeddings=embeddings,
        embedding_chunks=first["embedding_chunks"],
        start=1,
        end=2,
        cumulative_extraction_seconds=2.0,
    )
    embeddings[1, 0] = 999.0

    restored = _load_chemical_feature_checkpoint(
        checkpoint_dir,
        protocol_sha256=PROTOCOL_SHA256,
        identifiers_sha256=IDENTIFIERS_SHA256,
        embeddings=embeddings,
    )

    assert restored is not None
    assert restored["generation"] == 1
    assert "rows 1:2 changed" in restored["rejected_newer_generations"][0]["reason"]


def test_chemical_checkpoint_fault_before_publish_keeps_previous_generation(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "chemical"
    embeddings = np.ones((2, 3), dtype=np.float32)
    first = _write_chemical_feature_checkpoint(
        checkpoint_dir,
        generation=1,
        protocol_sha256=PROTOCOL_SHA256,
        identifiers_sha256=IDENTIFIERS_SHA256,
        embeddings=embeddings,
        embedding_chunks=[],
        start=0,
        end=1,
        cumulative_extraction_seconds=1.0,
    )

    def fail(stage: str) -> None:
        if stage == "after_state":
            raise RuntimeError("injected chemical checkpoint fault")

    with pytest.raises(RuntimeError, match="injected chemical checkpoint fault"):
        _write_chemical_feature_checkpoint(
            checkpoint_dir,
            generation=2,
            protocol_sha256=PROTOCOL_SHA256,
            identifiers_sha256=IDENTIFIERS_SHA256,
            embeddings=embeddings,
            embedding_chunks=first["embedding_chunks"],
            start=1,
            end=2,
            cumulative_extraction_seconds=2.0,
            fault_hook=fail,
        )

    restored = _load_chemical_feature_checkpoint(
        checkpoint_dir,
        protocol_sha256=PROTOCOL_SHA256,
        identifiers_sha256=IDENTIFIERS_SHA256,
        embeddings=embeddings,
    )
    assert restored is not None
    assert restored["generation"] == 1
    assert not list(checkpoint_dir.glob(".*.tmp"))


def test_prepare_full_test_features_publishes_authenticated_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _record(
        "chemical-test",
        (PeakGroup(0, 100.0, 1.0, ("frag@100.0",)),),
    )

    class FakeExtractor:
        provenance = {"test": True}

        def __init__(self, *, device: str) -> None:
            assert device == "cpu"

        def extract(self, spectra, *, identifiers, batch_size):
            assert len(spectra) == batch_size == 1
            return SimpleNamespace(
                identifiers=tuple(identifiers),
                spectrum_embeddings=np.ones((1, 1024), dtype=np.float32),
            )

    monkeypatch.setattr(
        chemical_module,
        "verify_protocol",
        lambda _: {"protocol_sha256": PROTOCOL_SHA256, "data_root": str(tmp_path)},
    )
    monkeypatch.setattr(
        chemical_module, "verify_frozen_input_files", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        chemical_module,
        "load_config",
        lambda _: SimpleNamespace(hybrid_inference_cpu_threads=1),
    )
    monkeypatch.setattr(chemical_module, "_test_records", lambda *_: [record])
    monkeypatch.setattr(chemical_module, "to_matchms_spectrum", lambda _: object())
    monkeypatch.setattr(dreams_features_module, "DreaMSFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(chemical_module, "peak_rss_bytes", lambda: 123)

    manifest = chemical_module.prepare_full_test_features(
        tmp_path, extraction_batch_size=1, checkpoint_every_chunks=1
    )

    assert manifest["checkpoint_format"] == "atomic-generations-v2"
    assert manifest["last_checkpoint_generation"] == 1
    assert manifest["embedding_chunks"] == 1
    assert manifest["rejected_newer_checkpoint_generations"] == []
