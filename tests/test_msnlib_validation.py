"""Focused leakage and metric tests for the MSnLib benchmark driver."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.msnlib_validation.features as validation_features
import benchmarks.msnlib_validation.models as validation_models
import benchmarks.msnlib_validation.protocol as validation_protocol
import benchmarks.msnlib_validation.reuse as validation_reuse
from benchmarks.msnlib_validation.config import (
    BenchmarkConfig,
    file_sha256,
    load_config,
    read_json,
    write_json,
)
from benchmarks.msnlib_validation.data import (
    PeakGroup,
    SpectrumRecord,
    assign_scaffold_splits,
    audit_split_disjointness,
    build_training_vocabulary,
    completion_document,
    iter_mgf,
    load_records,
    renormalize_peak_groups,
)
from benchmarks.msnlib_validation.mag import (
    _require_frozen_data_root,
    audit_mag_exclusion,
)
from benchmarks.msnlib_validation.metrics import (
    active_topic_metrics,
    calculate_sos,
    convergence_metrics,
    document_completion_nll,
    normalize_rows,
    optimal_topic_matching,
    top_word_diversity,
    word_cooccurrence_npmi,
)
from benchmarks.msnlib_validation.smoke import run_smoke


def _config(*, expected_spectra: int = 1) -> BenchmarkConfig:
    inputs = {
        name: {"relative_path": name, "sha256": "0" * 64}
        for name in (
            "mgf",
            "spec2vec_model",
            "spec2vec_embeddings",
            "spec2vec_db",
        )
    }
    return BenchmarkConfig(
        protocol_name="test",
        evidence_scope="confirmatory",
        seeds=(42, 43, 44, 45, 46),
        split_seed=42,
        completion_seed=42,
        split_fractions=(0.6, 0.2, 0.2),
        num_topics=3,
        min_mz=0.0,
        max_mz=2000.0,
        max_fragments=1000,
        min_fragments=2,
        min_intensity=0.01,
        max_intensity=1.0,
        significant_digits=2,
        min_df=1,
        min_cf=0,
        rm_top=0,
        alpha=0.6,
        eta=0.1,
        tomotopy_max_iterations=10,
        tomotopy_step_size=5,
        tomotopy_convergence_window=2,
        tomotopy_convergence_threshold=0.01,
        tomotopy_inference_iterations=5,
        tomotopy_training_workers=1,
        tomotopy_training_parallel=1,
        hybrid_max_epochs=21,
        hybrid_global_patience=2,
        hybrid_inference_epochs=2,
        hybrid_batch_size=2,
        hybrid_training_cpu_threads=1,
        hybrid_inference_cpu_threads=1,
        hybrid_checkpoint_keep=2,
        hybrid_reference_steps=4,
        hybrid_reference_extension_steps=8,
        reference_median_cosine=0.99,
        reference_fifth_percentile_cosine=0.95,
        completion_observed_fraction=0.5,
        topic_top_n=2,
        motif_spectrum_top_n=2,
        document_active_threshold=0.1,
        corpus_active_threshold=0.1,
        membership_threshold=0.5,
        mag_search_k=5,
        mag_unique_molecules=2,
        mag_cluster_cosine=0.9,
        mag_fingerprint_threshold=0.8,
        latency_subset_size=2,
        latency_repeats=2,
        expected_spectra=expected_spectra,
        input_files=inputs,
    )


def _record(
    identifier: str,
    *,
    connectivity: str,
    group: str,
    unique_word: str | None = None,
) -> SpectrumRecord:
    groups = (
        PeakGroup(0, 100.0, 1.0, ("frag@100.0", "loss@50.0")),
        PeakGroup(1, 110.0, 0.5, ("frag@110.0", "loss@40.0")),
        PeakGroup(2, 120.0, 0.25, ((unique_word or "frag@120.0"), "loss@30.0")),
    )
    return SpectrumRecord(
        spectrum_id=identifier,
        feature_id=identifier,
        smiles="C",
        supplied_inchikey=connectivity,
        connectivity_key=connectivity,
        scaffold_key=group,
        split_group=group,
        precursor_mz=150.0,
        peak_groups=groups,
        declared_num_peaks=3,
        parsed_num_peaks=3,
        compound_name="fixture",
        metadata={},
    )


def _split_records() -> list[SpectrumRecord]:
    records = []
    for group in range(12):
        records.append(
            _record(
                f"group-{group}-a",
                connectivity=f"compound-{group}",
                group=f"scaffold-{group // 2}",
            )
        )
        if group % 3 == 0:
            records.append(
                _record(
                    f"group-{group}-replicate",
                    connectivity=f"compound-{group}",
                    group=f"scaffold-{group // 2}",
                )
            )
    return records


def test_scaffold_split_is_disjoint_and_repeated_compounds_stay_together() -> None:
    records = _split_records()
    assignments, _ = assign_scaffold_splits(records, fractions=(0.6, 0.2, 0.2), seed=42)
    audit = audit_split_disjointness(records, assignments)
    assert audit["leaked_compounds"] == 0
    assert audit["leaked_groups"] == 0
    for connectivity in {row.connectivity_key for row in records}:
        assert (
            len(
                {
                    assignments[row.spectrum_id]
                    for row in records
                    if row.connectivity_key == connectivity
                }
            )
            == 1
        )


def test_split_is_deterministic_and_input_order_independent() -> None:
    records = _split_records()
    first, first_summary = assign_scaffold_splits(
        records, fractions=(0.6, 0.2, 0.2), seed=42
    )
    second, second_summary = assign_scaffold_splits(
        list(reversed(records)), fractions=(0.6, 0.2, 0.2), seed=42
    )
    assert first == second
    assert first_summary == second_summary


def test_leakage_audit_rejects_repeated_compound_crossing_splits() -> None:
    records = [
        _record("a", connectivity="same", group="acyclic:same"),
        _record("b", connectivity="same", group="acyclic:same"),
    ]
    with pytest.raises(ValueError, match="split leakage"):
        audit_split_disjointness(records, {"a": "train", "b": "test"})


def test_vocabulary_is_constructed_from_training_spectra_only() -> None:
    train = _record("train", connectivity="a", group="a")
    test = _record("test", connectivity="b", group="b", unique_word="frag@test-only")
    vocabulary, summary = build_training_vocabulary(
        [train, test], {"train": "train", "test": "test"}, min_df=1, min_cf=0, rm_top=0
    )
    assert "frag@test-only" not in vocabulary
    assert summary["source_split"] == "train"


def test_document_completion_keeps_peak_derived_words_atomic() -> None:
    record = _record("a", connectivity="a", group="a")
    first = completion_document(record, observed_fraction=0.5, seed=42)
    second = completion_document(record, observed_fraction=0.5, seed=42)
    assert first == second
    observed = {group.original_index for group in first.observed_groups}
    completion = {group.original_index for group in first.completion_groups}
    assert not observed & completion
    assert observed | completion == {0, 1, 2}
    for group in first.observed_groups + first.completion_groups:
        assert len(group.tokens) == 2


def test_observed_peak_groups_are_renormalized_without_heldout_maximum() -> None:
    record = _record("a", connectivity="a", group="a")
    observed = renormalize_peak_groups(
        record.peak_groups[1:], precursor_mz=record.precursor_mz, significant_digits=2
    )
    assert [group.intensity for group in observed] == [1.0, 0.5]
    assert observed[0].tokens.count("frag@110.0") == 100
    assert observed[0].tokens.count("loss@40.0") == 100
    assert observed[1].tokens.count("frag@120.0") == 50


def test_mag_exclusion_audit_accepts_clean_rows_and_rejects_leaks() -> None:
    assert audit_mag_exclusion({"heldout"}, ["train-a", "train-b"]) == {
        "retained_rows": 2,
        "retained_leak_rows": 0,
    }
    with pytest.raises(RuntimeError, match="exclusion audit failed"):
        audit_mag_exclusion({"heldout"}, ["train", "heldout"])


def test_mag_data_root_must_match_frozen_protocol(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    other = tmp_path / "other"
    frozen.mkdir()
    other.mkdir()

    assert _require_frozen_data_root({"data_root": str(frozen)}, frozen) == frozen
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        _require_frozen_data_root({"data_root": str(frozen)}, other)


def test_metric_sanity_for_identity_permutation_and_known_sos() -> None:
    beta = np.asarray([[0.8, 0.2, 0.0], [0.0, 0.1, 0.9]])
    theta = np.asarray([[0.75, 0.25], [0.2, 0.8]])
    nll = document_completion_nll(theta, beta, [["a", "a"], ["c"]], ["a", "b", "c"])
    assert nll["nll_per_token"] == pytest.approx(-(2 * np.log(0.6) + np.log(0.72)) / 3)
    convergence = convergence_metrics(theta, theta)
    assert convergence["cosine_median"] == pytest.approx(1.0)
    assert convergence["js_mean"] == pytest.approx(0.0)
    matching = optimal_topic_matching(beta, beta[::-1], top_n=2)
    assert matching["matched_cosine_mean"] == pytest.approx(1.0)
    assert matching["top_word_jaccard_mean"] == pytest.approx(1.0)
    assert calculate_sos(np.asarray([1, 1, 0]), np.asarray([1, 0, 1])) == 0.5
    assert 0 <= top_word_diversity(beta, top_n=2) <= 1
    assert (
        active_topic_metrics(theta, document_threshold=0.1, corpus_threshold=0.1)[
            "corpus_active_topics"
        ]
        == 2
    )


def test_npmi_handles_always_cooccurring_words_and_negative_values_fail() -> None:
    result = word_cooccurrence_npmi(
        np.asarray([[0.6, 0.4]]), [["a", "b"], ["a", "b"]], ["a", "b"], top_n=2
    )
    assert result["mean_npmi"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_rows(np.asarray([[1.0, -1.0]]))


def test_mgf_parser_rejects_malformed_peak_and_unterminated_block(tmp_path) -> None:
    malformed = tmp_path / "malformed.mgf"
    malformed.write_text("BEGIN IONS\nUSI=x\nnot-a-peak\nEND IONS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed peak"):
        list(iter_mgf(malformed))
    unterminated = tmp_path / "unterminated.mgf"
    unterminated.write_text("BEGIN IONS\nUSI=x\n100 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unterminated"):
        list(iter_mgf(unterminated))


def test_load_records_rejects_missing_required_metadata(tmp_path) -> None:
    path = tmp_path / "missing.mgf"
    path.write_text(
        "BEGIN IONS\nUSI=x\nFEATURE_ID=x\nINCHIKEY=X\nPRECURSOR_MZ=150\n"
        "100 1\n110 .5\nEND IONS\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing metadata.*smiles"):
        load_records(path, _config(), require_expected_count=False)


def test_declared_peak_mismatch_is_audited_not_trusted(tmp_path, monkeypatch) -> None:
    path = tmp_path / "mismatch.mgf"
    path.write_text(
        "BEGIN IONS\nUSI=x\nFEATURE_ID=x\nSMILES=C\nINCHIKEY=X\n"
        "PRECURSOR_MZ=150\nNUM_PEAKS=99\n100 1\n110 .5\nEND IONS\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "benchmarks.msnlib_validation.data._structure_keys",
        lambda smiles, inchikey: ("X", "", "acyclic:X"),
    )
    records, summary = load_records(path, _config())
    assert len(records) == 1
    assert summary["num_peaks_mismatches"][0]["parsed"] == 2


def test_synthetic_smoke_is_deterministic_and_not_chemical_evidence() -> None:
    first = run_smoke()
    second = run_smoke()
    assert first == second
    assert first["software_validation_only"] is True
    assert first["chemical_evidence"] is False
    assert first["completion_peak_group_atomicity"] is True


def test_committed_full_configuration_is_paper_scale() -> None:
    config = load_config("benchmarks/msnlib_validation/configs/full-msnlib-k1000.json")
    assert config.expected_spectra == 41_568
    assert config.num_topics == 1_000
    assert config.seeds == (42, 43, 44, 45, 46)
    assert config.tomotopy_max_iterations == 5_000
    assert config.evidence_scope == "confirmatory"
    assert config.tomotopy_training_workers == 1
    assert config.tomotopy_training_parallel == 1
    assert config.hybrid_training_cpu_threads == 1
    assert config.hybrid_inference_cpu_threads == 1
    assert config.hybrid_checkpoint_keep == 2
    assert config.motif_spectrum_top_n == 20


def test_indicative_configuration_changes_only_scope_seed_and_parallelism() -> None:
    full = load_config("benchmarks/msnlib_validation/configs/full-msnlib-k1000.json")
    indicative = load_config(
        "benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42.json"
    )
    full_values = full.as_dict()
    indicative_values = indicative.as_dict()
    allowed = {
        "evidence_scope",
        "hybrid_max_epochs",
        "hybrid_training_cpu_threads",
        "protocol_name",
        "seeds",
        "tomotopy_training_parallel",
        "tomotopy_training_workers",
    }
    assert {
        key for key in full_values if full_values[key] != indicative_values[key]
    } == allowed
    assert indicative.evidence_scope == "indicative_single_seed"
    assert indicative.seeds == (42,)
    assert indicative.num_topics == 1_000
    assert indicative.expected_spectra == 41_568
    assert indicative.hybrid_max_epochs == 50
    assert indicative.hybrid_training_cpu_threads == 4
    assert indicative.hybrid_inference_cpu_threads == 1
    assert indicative.tomotopy_training_parallel == 3
    assert indicative.tomotopy_training_workers == 6


def test_confirmatory_scope_requires_all_prespecified_seeds() -> None:
    with pytest.raises(ValueError, match="seeds 42 through 46"):
        replace(_config(), seeds=(42,))


def test_precheckpoint_frozen_config_loads_with_recorded_thread_defaults(
    tmp_path: Path,
) -> None:
    payload = read_json(
        "benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42.json"
    )
    for name in (
        "hybrid_training_cpu_threads",
        "hybrid_inference_cpu_threads",
        "hybrid_checkpoint_keep",
    ):
        payload.pop(name)
    path = tmp_path / "legacy.json"
    write_json(path, payload)

    restored = load_config(path)
    assert restored.hybrid_training_cpu_threads == 1
    assert restored.hybrid_inference_cpu_threads == 1
    assert restored.hybrid_checkpoint_keep == 2


def test_hybrid_inference_threads_and_checkpoint_generations_are_guarded() -> None:
    with pytest.raises(ValueError, match="exactly one CPU thread"):
        replace(_config(), hybrid_inference_cpu_threads=2)
    with pytest.raises(ValueError, match="at least two"):
        replace(_config(), hybrid_checkpoint_keep=1)


def test_feature_cache_must_match_the_frozen_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_json(
        feature_dir / "manifest.json",
        {"output_sha256": {}, "protocol_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        validation_features,
        "verify_protocol",
        lambda _: {"protocol_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        validation_features,
        "verify_frozen_input_files",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="another frozen protocol"):
        validation_features.prepare_features(tmp_path)


def test_frozen_external_inputs_are_rehashed_before_use(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    mgf = data_root / "input.mgf"
    mgf.write_bytes(b"original")
    write_json(
        tmp_path / "input_manifest.json",
        {
            "data_root": str(data_root),
            "files": {
                "mgf": {
                    "bytes": mgf.stat().st_size,
                    "path": str(mgf),
                    "sha256": file_sha256(mgf),
                }
            },
        },
    )
    lock = {"data_root": str(data_root)}

    assert validation_protocol.verify_frozen_input_files(
        tmp_path,
        names={"mgf"},
        lock=lock,
    )["mgf"]["sha256"] == file_sha256(mgf)
    mgf.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        validation_protocol.verify_frozen_input_files(
            tmp_path,
            names={"mgf"},
            lock=lock,
        )


def test_completed_core_artifacts_are_hash_verified(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "tomotopy"
    output.mkdir()
    for name, value in {
        "beta.npy": b"beta",
        "model.bin": b"model",
        "test_theta.npy": b"theta",
    }.items():
        (output / name).write_bytes(value)
    write_json(
        output / "complete.json",
        {
            "beta_sha256": file_sha256(output / "beta.npy"),
            "evidence_scope": config.evidence_scope,
            "method": "tomotopy",
            "model_sha256": file_sha256(output / "model.bin"),
            "seed": 42,
            "theta_sha256": file_sha256(output / "test_theta.npy"),
            "topic_count": config.num_topics,
            "training_parallel_scheme_value": config.tomotopy_training_parallel,
            "training_workers_requested": config.tomotopy_training_workers,
        },
    )

    assert (
        validation_models._verify_completed_core_result(
            output,
            method="tomotopy",
            seed=42,
            config=config,
        )["method"]
        == "tomotopy"
    )
    (output / "beta.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validation_models._verify_completed_core_result(
            output,
            method="tomotopy",
            seed=42,
            config=config,
        )


def test_core_worker_parallelism_is_confined_to_hybrid_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        seeds=(42,),
        hybrid_training_cpu_threads=4,
        hybrid_inference_cpu_threads=1,
        evidence_scope="indicative_single_seed",
        tomotopy_training_workers=0,
        tomotopy_training_parallel=3,
    )
    calls = []
    monkeypatch.setattr(
        validation_models,
        "verify_protocol",
        lambda _: {"repo_root": str(tmp_path), "protocol_sha256": "a" * 64},
    )
    monkeypatch.setattr(validation_models, "load_config", lambda _: config)
    monkeypatch.setattr(validation_models, "prepare_features", lambda _: {})
    monkeypatch.setattr(
        validation_models,
        "_verify_completed_core_result",
        lambda _output, *, method, seed, config: {
            "method": method,
            "seed": seed,
        },
    )
    monkeypatch.setattr(validation_models, "write_json", lambda *_: None)

    def capture_run(command, *, cwd, env, check):
        calls.append({"command": command, "cwd": cwd, "env": env, "check": check})

    monkeypatch.setattr(validation_models.subprocess, "run", capture_run)
    validation_models.run_all_core_models(tmp_path)

    assert len(calls) == 2
    by_method = {
        call["command"][call["command"].index("--method") + 1]: call for call in calls
    }
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert by_method["tomotopy"]["env"][name] == "1"
        assert by_method["hybrid"]["env"][name] == "4"


def test_derived_protocol_rejects_scientific_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_config(
        "benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42.json"
    )
    target = replace(
        source,
        protocol_name="parallel-continuation",
        hybrid_training_cpu_threads=3,
    )
    monkeypatch.setattr(
        validation_protocol,
        "verify_protocol",
        lambda *_args, **_kwargs: {"protocol_sha256": "a" * 64},
    )
    monkeypatch.setattr(validation_protocol, "load_config", lambda _: source)

    derivation = validation_protocol.validate_execution_only_derivation(
        "/source",
        target,
    )
    assert derivation["execution_only"] is True
    assert set(derivation["differences"]) == {
        "protocol_name",
        "hybrid_training_cpu_threads",
    }
    name_only = replace(source, protocol_name="runtime-fix")
    with pytest.raises(ValueError, match="explicit execution reason"):
        validation_protocol.validate_execution_only_derivation("/source", name_only)
    named_derivation = validation_protocol.validate_execution_only_derivation(
        "/source",
        name_only,
        "isolate raw-DreaMS from conflicting OpenMP runtimes",
    )
    assert set(named_derivation["differences"]) == {"protocol_name"}
    assert named_derivation["reason"].startswith("isolate raw-DreaMS")
    with pytest.raises(ValueError, match="must differ only"):
        validation_protocol.validate_execution_only_derivation(
            "/source",
            replace(target, alpha=target.alpha + 0.1),
        )


def test_completed_core_artifact_reuse_is_hashed_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    input_files = {"mgf": {"sha256": "a" * 64}}
    write_json(source / "input_manifest.json", {"files": input_files})
    write_json(target / "input_manifest.json", {"files": input_files})
    frozen_hashes = {}
    for name in validation_reuse.IDENTICAL_FROZEN_ARTIFACTS:
        (source / name).write_text(name, encoding="utf-8")
        (target / name).write_text(name, encoding="utf-8")
        frozen_hashes[name] = file_sha256(source / name)

    source_protocol = "b" * 64
    target_protocol = "c" * 64
    derivation = {"execution_only": True, "source_protocol_sha256": source_protocol}
    source_lock = {
        "artifacts": frozen_hashes,
        "protocol_sha256": source_protocol,
    }
    target_lock = {
        "artifacts": frozen_hashes,
        "derivation": derivation,
        "protocol_sha256": target_protocol,
        "test_results_inspected": True,
    }
    monkeypatch.setattr(
        validation_reuse,
        "verify_protocol",
        lambda path, **_kwargs: (
            source_lock if Path(path).resolve() == source.resolve() else target_lock
        ),
    )
    config = SimpleNamespace(
        seeds=(42,),
        num_topics=3,
        hybrid_inference_cpu_threads=1,
        hybrid_training_cpu_threads=4,
        tomotopy_training_workers=0,
        tomotopy_training_parallel=3,
    )
    monkeypatch.setattr(validation_reuse, "load_config", lambda _: config)
    monkeypatch.setattr(
        validation_reuse,
        "validate_execution_only_derivation",
        lambda *_: derivation,
    )

    feature_dir = source / "features"
    feature_dir.mkdir()
    (feature_dir / "global_embeddings.npy").write_bytes(b"features")
    write_json(
        feature_dir / "manifest.json",
        {
            "output_sha256": {
                "global_embeddings.npy": file_sha256(
                    feature_dir / "global_embeddings.npy"
                )
            },
            "protocol_sha256": source_protocol,
            "rows": 2,
            "train_rows": 1,
        },
    )
    tomotopy = source / "core" / "seed_42" / "tomotopy"
    tomotopy.mkdir(parents=True)
    for name, value in {
        "beta.npy": b"beta",
        "test_theta.npy": b"theta",
        "model.bin": b"model",
    }.items():
        (tomotopy / name).write_bytes(value)
    write_json(
        tomotopy / "complete.json",
        {
            "beta_sha256": file_sha256(tomotopy / "beta.npy"),
            "method": "tomotopy",
            "model_sha256": file_sha256(tomotopy / "model.bin"),
            "seed": 42,
            "theta_sha256": file_sha256(tomotopy / "test_theta.npy"),
            "topic_count": 3,
            "training_iterations": 10,
            "training_parallel_scheme_value": 3,
            "training_workers_requested": 0,
        },
    )
    hybrid = source / "core" / "seed_42" / "hybrid"
    hybrid.mkdir(parents=True)
    for name, value in {
        "beta.npy": b"hybrid-beta",
        "model.pt": b"hybrid-model",
        "test_theta_0.npy": b"hybrid-theta-0",
        "test_theta_2.npy": b"hybrid-theta-2",
        "test_theta_4.npy": b"hybrid-theta-4",
    }.items():
        (hybrid / name).write_bytes(value)
    write_json(
        hybrid / "complete.json",
        {
            "beta_sha256": file_sha256(hybrid / "beta.npy"),
            "discovery_epochs": 21,
            "inference_cpu_threads": 1,
            "method": "hybrid",
            "model_sha256": file_sha256(hybrid / "model.pt"),
            "reference_converged": True,
            "reference_steps": 4,
            "seed": 42,
            "theta_sha256": {
                "0": file_sha256(hybrid / "test_theta_0.npy"),
                "2": file_sha256(hybrid / "test_theta_2.npy"),
                "4": file_sha256(hybrid / "test_theta_4.npy"),
            },
            "topic_count": 3,
            "training_cpu_threads": 4,
        },
    )

    first = validation_reuse.reuse_core_artifacts(source, target)
    second = validation_reuse.reuse_core_artifacts(source, target)
    assert first["reused"]["features"] is True
    assert second["reused"]["tomotopy"]["42"]["training_iterations"] == 10
    assert second["reused"]["hybrid"]["42"]["reference_steps"] == 4
    imported = validation_reuse.read_json(target / "features" / "manifest.json")
    assert imported["protocol_sha256"] == target_protocol
    assert imported["reuse_provenance"]["source_protocol_sha256"] == source_protocol
    assert validation_reuse.read_json(
        target / "core" / "seed_42" / "tomotopy" / "complete.json"
    )["model_sha256"] == file_sha256(
        target / "core" / "seed_42" / "tomotopy" / "model.bin"
    )
    assert validation_reuse.read_json(
        target / "core" / "seed_42" / "hybrid" / "complete.json"
    )["model_sha256"] == file_sha256(
        target / "core" / "seed_42" / "hybrid" / "model.pt"
    )


def test_raw_dreams_import_does_not_eagerly_load_scipy_metrics() -> None:
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'benchmarks.msnlib_validation.metrics' or (
        name == 'metrics' and level == 1
    ):
        raise RuntimeError('eager metrics import')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import benchmarks.msnlib_validation.mag
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
