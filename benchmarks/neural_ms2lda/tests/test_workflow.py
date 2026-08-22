"""Provenance, orchestration, resume, and portable-bundle tests."""

from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.neural_ms2lda import __main__ as module_entry
from benchmarks.neural_ms2lda import artifacts as neural_artifacts
from benchmarks.neural_ms2lda import chemical, pipeline
from benchmarks.neural_ms2lda.artifacts import (
    build_results,
    load_bundle,
    load_protocol,
    package_bundle,
)
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_view_pairs,
    prepare_data,
    prepare_training_views,
)
from benchmarks.neural_ms2lda.embeddings import train_sgns
from benchmarks.neural_ms2lda.evaluation import (
    evaluate_neural,
    evaluate_neural_validation,
)
from benchmarks.neural_ms2lda.initialization import (
    prepare_initialization,
    prepare_token_features,
)
from benchmarks.neural_ms2lda.mag import _connectivity_key, build_filtered_mag_index
from benchmarks.neural_ms2lda.tomotopy import (
    _alpha_evidence,
    _converged,
    _infer_theta,
)
from benchmarks.neural_ms2lda.training import train_model
from benchmarks.neural_ms2lda.utils import file_sha256, write_json, write_jsonl
from scripts.download_msnlib_validation_assets import (
    RECORD_API,
    RECORD_ID,
    safe_zip_members,
    validate_acquisition_manifest,
)

from ._support import chemistry_result, mini_protocol, write_mini_mgf


def test_module_entry_pins_numerical_threads_before_dispatch(monkeypatch: Any) -> None:
    """The documented direct entry point must honor the six-thread protocol."""
    for name in module_entry.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "99")
    module_entry._configure_process_threads()
    assert {os.environ[name] for name in module_entry.THREAD_ENVIRONMENT_VARIABLES} == {
        "6"
    }


def _prepare_mini_training_scaffold(
    run: Path, *, data_root: Path, protocol: dict[str, Any]
) -> None:
    """Create every deterministic input that precedes neural optimization."""
    prepare_data(run, data_root=data_root, protocol=protocol)
    data = run / "data"
    prepare_training_views(run, counts_dir=data, data_root=data_root, protocol=protocol)
    train = load_csr(data / "train.npz")
    train_sgns(run / "embeddings", train, protocol["sgns"], seed=42)
    prepare_token_features(run, counts_dir=data, protocol=protocol)
    prepare_initialization(run, train=train, protocol=protocol)


def _mini_training_arguments(run: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Load the aligned sparse inputs passed to ``train_model``."""
    data = run / "data"
    return {
        "train": load_csr(data / "train.npz"),
        "views": load_view_pairs(run, protocol),
        "validation_observed": load_csr(data / "validation_observed.npz"),
        "validation_completion": load_csr(data / "validation_completion.npz"),
        "validation_full": load_csr(data / "validation_full.npz"),
        "validation_records": load_heldout_records(data, "validation"),
    }


def test_training_interface_exposes_no_test_inputs() -> None:
    """Model development receives training and validation evidence only."""
    assert not any("test" in name for name in inspect.signature(train_model).parameters)


def test_validation_records_do_not_open_the_test_file(tmp_path: Path) -> None:
    validation = {"split": "validation", "spectrum_id": "validation-1"}
    (tmp_path / "validation_records.jsonl").write_text(
        json.dumps(validation) + "\n", encoding="utf-8"
    )
    (tmp_path / "test_records.jsonl").write_text("not-json\n", encoding="utf-8")
    assert load_heldout_records(tmp_path, "validation") == [validation]


def test_pipeline_finishes_validation_before_test(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Prove the final runner's only test access occurs after validation outputs."""
    protocol = load_protocol()
    events: list[str] = []
    monkeypatch.setattr(pipeline, "initialize_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "read_json", lambda path: protocol)
    monkeypatch.setattr(pipeline, "prepare_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_csr", lambda path: object())
    monkeypatch.setattr(
        pipeline, "prepare_training_views", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(pipeline, "train_sgns", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline, "prepare_token_features", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline, "prepare_initialization", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(pipeline, "load_view_pairs", lambda *args: [])
    monkeypatch.setattr(pipeline, "load_heldout_records", lambda *args: [])
    monkeypatch.setattr(pipeline, "train_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "train_tomotopy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "evaluate_neural_validation",
        lambda *args: events.append("validation-neural"),
    )

    def tomotopy_evaluation(*args: Any, split: str, **kwargs: Any) -> None:
        events.append(f"{split}-tomotopy")

    monkeypatch.setattr(pipeline, "evaluate_tomotopy", tomotopy_evaluation)
    monkeypatch.setattr(
        pipeline,
        "evaluate_neural",
        lambda *args: events.append("test-neural"),
    )

    def chemistry(*args: Any, split: str, method: str, **kwargs: Any) -> None:
        events.append(f"{split}-chemistry-{method}")

    monkeypatch.setattr(pipeline, "_chemical_subprocess", chemistry)
    monkeypatch.setattr(pipeline, "package_bundle", lambda *args: {})
    monkeypatch.setattr(pipeline, "build_results", lambda path: {"ok": True})
    monkeypatch.setattr(
        pipeline, "verify_run", lambda *args, **kwargs: {"verified": True}
    )
    pipeline.run_pipeline(tmp_path / "run", data_root=tmp_path)
    first_test = min(
        index for index, event in enumerate(events) if event.startswith("test")
    )
    validation_events = [
        index for index, event in enumerate(events) if event.startswith("validation")
    ]
    assert validation_events
    assert max(validation_events) < first_test


def test_mag_index_excludes_validation_and_test_compounds(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Both held-out splits must be absent before MAG nearest-neighbour search."""
    embeddings = tmp_path / "embeddings.npy"
    np.save(embeddings, np.eye(3, 300, dtype=np.float32))
    database = tmp_path / "spectra.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE spectra (id INTEGER, smiles TEXT)")
        connection.executemany(
            "INSERT INTO spectra VALUES (?, ?)",
            [(0, "CCO"), (1, "CCN"), (2, "CCC")],
        )
        connection.commit()
    finally:
        connection.close()
    run = tmp_path / "run"
    write_jsonl(
        run / "data/split_manifest.jsonl",
        [
            {"split": "validation", "connectivity_key": _connectivity_key("CCO")},
            {"split": "test", "connectivity_key": _connectivity_key("CCN")},
            {"split": "train", "connectivity_key": _connectivity_key("CCC")},
        ],
    )

    class FakeIndex:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.rows = 0

        def add(self, values: np.ndarray) -> None:
            self.rows += len(values)

    def write_index(index: FakeIndex, path: str) -> None:
        Path(path).write_bytes(f"{index.dimensions}:{index.rows}".encode())

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(IndexFlatIP=FakeIndex, write_index=write_index),
    )
    protocol = {
        "input_files": {
            "spec2vec_embeddings": {"relative_path": embeddings.name},
            "spec2vec_db": {"relative_path": database.name},
        }
    }
    manifest = build_filtered_mag_index(run, data_root=tmp_path, protocol=protocol)
    assert manifest["excluded_connectivity_keys"] == 2
    assert manifest["excluded_reference_rows"] == 2
    assert manifest["retained_reference_rows"] == 1
    assert manifest["retained_leak_rows"] == 0


def test_mag_annotations_are_generated_once_per_model(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Validation and test scoring share one model-level annotation artifact."""
    run = tmp_path / "run"
    evaluation = run / "validation_evaluation/neural"
    evaluation.mkdir(parents=True)
    np.save(evaluation / "beta.npy", np.asarray([[0.6, 0.4]], dtype=np.float32))
    write_json(run / "data/vocabulary.json", {"vocabulary": ["frag@1", "loss@1"]})
    monkeypatch.setattr(chemical, "_verified_chemical_inputs", lambda *args: {})
    monkeypatch.setattr(
        chemical,
        "build_filtered_mag_index",
        lambda *args, **kwargs: {"retained_leak_rows": 0},
    )
    monkeypatch.setattr(chemical, "topic_spectra", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        chemical, "_mag_matches", lambda *args, **kwargs: (object(), [object()])
    )
    calls = 0

    def annotate(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return [
            {
                "topic_id": 0,
                "clustered_smiles": [],
                "optimized_feature_count": 0,
                "cluster_error": "",
                "optimization_error": "",
                "retrieved_smiles": [],
                "retrieved_scores": [],
            }
        ]

    monkeypatch.setattr(chemical, "_annotate_topics", annotate)
    protocol = {
        "chemistry": {"motif_spectrum_top_n": 2},
        "preprocessing": {"significant_digits": 2},
    }
    first, _ = chemical._shared_annotations(
        run, method="neural", data_root=tmp_path, protocol=protocol
    )
    second, _ = chemical._shared_annotations(
        run, method="neural", data_root=tmp_path, protocol=protocol
    )
    assert first == second
    assert calls == 1


def test_verify_run_checks_the_train_only_cooccurrence_graph(tmp_path: Path) -> None:
    mgf = tmp_path / "input.mgf"
    mgf.write_text("BEGIN IONS\nEND IONS\n", encoding="utf-8")
    protocol = copy.deepcopy(load_protocol())
    protocol["input_files"] = {
        "mgf": {
            "relative_path": mgf.name,
            "bytes": mgf.stat().st_size,
            "sha256": file_sha256(mgf),
        }
    }
    run = tmp_path / "run"
    write_json(run / "protocol.resolved.json", protocol)
    write_json(
        run / "run.lock.json",
        {
            "data_root": str(tmp_path),
            "protocol_sha256": neural_artifacts.object_sha256(protocol),
            "inputs": neural_artifacts.verify_inputs(protocol, tmp_path, names={"mgf"}),
            "code": neural_artifacts.code_manifest(),
        },
    )
    data_artifact = run / "data/train.npz"
    data_artifact.parent.mkdir(parents=True)
    data_artifact.write_bytes(b"frozen training data")
    data_manifest = {
        "leakage_audit": {"leaked_compounds": 0, "leaked_groups": 0},
        "vocabulary": {
            "source_split": "train",
            "order": "raw_training_spectra_first_seen",
        },
        "output_sha256": {data_artifact.name: file_sha256(data_artifact)},
    }
    write_json(run / "data/complete.json", data_manifest)
    graph = run / "cooccurrence_graph/positive_npmi_graph.npz"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"frozen train-only graph")
    graph_manifest = {"output_sha256": {graph.name: file_sha256(graph)}}
    write_json(graph.parent / "complete.json", graph_manifest)
    selected_checkpoint = run / "model/selected.pt"
    selected_checkpoint.parent.mkdir(parents=True)
    selected_checkpoint.write_bytes(b"selected model")
    selected = {
        "checkpoint": selected_checkpoint.name,
        "checkpoint_sha256": file_sha256(selected_checkpoint),
        "selection_rule": "fixed_final_epoch",
        "epoch": int(protocol["optimization"]["maximum_epochs"]),
    }
    write_json(
        run / "model/complete.json",
        {"selected": selected, "cooccurrence_graph": graph_manifest},
    )
    write_json(run / "model/selected.json", selected)
    result = neural_artifacts.verify_run(run, data_root=tmp_path)
    assert "cooccurrence_graph/complete.json" in result["manifests_present"]
    graph.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        neural_artifacts.verify_run(run, data_root=tmp_path)


def test_neural_test_evaluation_requires_six_threads_and_final_epoch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    protocol = load_protocol()
    selected = {
        "selection_rule": "fixed_final_epoch",
        "epoch": int(protocol["optimization"]["maximum_epochs"]) - 2,
    }
    write_json(run / "model/selected.json", selected)
    write_json(run / "model/complete.json", {"selected": selected})
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with pytest.raises(ValueError, match="thread count differs"):
            evaluate_neural(run, protocol)
        torch.set_num_threads(6)
        with pytest.raises(ValueError, match="fixed final epoch"):
            evaluate_neural(run, protocol)
    finally:
        torch.set_num_threads(original_threads)
    assert not (run / "evaluation/neural/test_access.json").exists()


def test_tomotopy_empty_document_uses_the_learned_prior() -> None:
    calls = []

    class FakeModel:
        k = 2
        alpha = np.asarray([0.3, 0.7], dtype=np.float32)

        @staticmethod
        def make_doc(words: list[str]) -> list[str]:
            assert words
            return words

        @staticmethod
        def infer(
            documents: list[list[str]], **kwargs: object
        ) -> tuple[list[list[float]], None]:
            calls.append(kwargs)
            return [[0.8, 0.2] for _ in documents], None

    theta = _infer_theta(FakeModel(), [[], ["frag@100.0"], []], iterations=5, workers=6)
    assert np.allclose(theta[0], [0.3, 0.7])
    assert np.allclose(theta[1], [0.8, 0.2])
    assert calls == [{"iter": 5, "workers": 6, "parallel": 1, "together": False}]


def test_tomotopy_alpha_and_convergence_evidence() -> None:
    class FakeModel:
        k = 2
        alpha = np.asarray([0.1, 0.2])
        optim_interval = 10

    evidence = _alpha_evidence(FakeModel(), {"alpha": 0.6})
    assert evidence["initial_value"] == 0.6
    assert evidence["learned_minimum"] == 0.1
    history = [
        {"perplexity": 100.0},
        {"perplexity": 100.1},
        {"perplexity": 100.2},
    ]
    assert _converged(history, window=2, threshold=0.005)
    FakeModel.alpha = np.asarray([0.1, 0.0])
    with pytest.raises(ValueError, match="not positive"):
        _alpha_evidence(FakeModel(), {"alpha": 0.6})


def test_epoch_boundary_resume_is_bitwise_equivalent(tmp_path: Path) -> None:
    """An interrupted run must recover the exact uninterrupted model state."""
    mgf = tmp_path / "mini.mgf"
    write_mini_mgf(mgf)
    protocol = mini_protocol(mgf)
    scaffold = tmp_path / "scaffold"
    _prepare_mini_training_scaffold(scaffold, data_root=tmp_path, protocol=protocol)

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    shutil.copytree(scaffold, uninterrupted)
    shutil.copytree(scaffold, resumed)
    uninterrupted_result = train_model(
        uninterrupted,
        protocol=protocol,
        **_mini_training_arguments(uninterrupted, protocol),
    )

    first_epoch = copy.deepcopy(protocol)
    first_epoch["optimization"]["maximum_epochs"] = 1
    train_model(
        resumed,
        protocol=first_epoch,
        **_mini_training_arguments(resumed, first_epoch),
    )
    # Simulate a process that finished its first invocation but is asked to
    # continue to the predeclared epoch two from the saved mutable state.
    (resumed / "model/complete.json").unlink()
    resumed_result = train_model(
        resumed,
        protocol=protocol,
        **_mini_training_arguments(resumed, protocol),
    )

    def selected_checkpoint(run: Path, result: dict[str, Any]) -> dict[str, Any]:
        return torch.load(
            run / "model" / result["selected"]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

    expected = selected_checkpoint(uninterrupted, uninterrupted_result)
    actual = selected_checkpoint(resumed, resumed_result)
    assert expected.keys() == actual.keys()
    assert expected["epoch"] == actual["epoch"] == 2
    assert expected["validation"] == actual["validation"]
    assert expected["routing_temperature"] == actual["routing_temperature"]
    assert expected["model"].keys() == actual["model"].keys()
    for name in expected["model"]:
        assert torch.equal(expected["model"][name], actual["model"][name]), name


def test_miniature_mgf_through_report_and_bundle(tmp_path: Path) -> None:
    mgf = tmp_path / "mini.mgf"
    write_mini_mgf(mgf)
    protocol = mini_protocol(mgf)
    run = tmp_path / "run"
    _prepare_mini_training_scaffold(run, data_root=tmp_path, protocol=protocol)
    data = run / "data"
    train = load_csr(data / "train.npz")
    result = train_model(
        run,
        protocol=protocol,
        **_mini_training_arguments(run, protocol),
    )
    selected_hash = result["selected"]["checkpoint_sha256"]
    resumed = train_model(
        run,
        protocol=protocol,
        **_mini_training_arguments(run, protocol),
    )
    assert resumed["selected"]["checkpoint_sha256"] == selected_hash
    write_json(run / "protocol.resolved.json", protocol)
    validation = evaluate_neural_validation(run, protocol)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(6)
        neural_test = evaluate_neural(run, protocol)
    finally:
        torch.set_num_threads(original_threads)
    assert neural_test["method"] == "neural"
    assert "warm_in_memory_batch_inference" in neural_test["metrics"]

    tomotopy_validation = copy.deepcopy(validation)
    tomotopy_validation.update({"method": "tomotopy", "model_sha256": "test"})
    tomotopy_test = copy.deepcopy(neural_test)
    tomotopy_test.update({"method": "tomotopy", "model_sha256": "test"})
    write_json(
        run / "validation_evaluation/tomotopy/complete.json", tomotopy_validation
    )
    write_json(run / "evaluation/tomotopy/complete.json", tomotopy_test)
    model_binary = run / "tomotopy/model.bin"
    model_binary.parent.mkdir(parents=True, exist_ok=True)
    model_binary.write_bytes(b"miniature Tomotopy model")
    write_json(
        run / "tomotopy/complete.json",
        {
            "training_seconds_total": 10.0,
            "training_workers": 6,
            "model_sha256": file_sha256(model_binary),
            "output_sha256": {"model.bin": file_sha256(model_binary)},
        },
    )
    chemistry = chemistry_result()
    for path in (
        "validation_chemical/neural/complete.json",
        "validation_chemical/tomotopy/complete.json",
        "chemical/neural/complete.json",
        "chemical/tomotopy/complete.json",
    ):
        write_json(run / path, chemistry)
    write_json(
        run / "run.lock.json",
        {
            "protocol_sha256": neural_artifacts.object_sha256(protocol),
            "inputs": {"mgf": {"sha256": file_sha256(mgf)}},
            "code": {},
            "environment": {},
            "discovery_audit": {"forbidden_dependencies_found": []},
        },
    )
    bundle = run / "model_bundle"
    packaged = package_bundle(run, bundle)
    report = build_results(run)
    assert [row["method"] for row in report["methods"]] == ["neural", "tomotopy"]
    assert report["comparison_contract"]["association_probability_threshold"] == 0.5
    loaded, vocabulary, manifest = load_bundle(bundle)
    assert loaded.num_topics == 4
    assert len(vocabulary) == train.shape[1]
    assert manifest["selected_epoch"] == 2
    assert set(packaged["files"]) == {
        "model.pt",
        "protocol.json",
        "provenance.json",
        "vocabulary.json",
    }


def test_zip_safety_and_acquisition_manifest_are_immutable(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_zip_members(archive)

    archives = {"Data.zip": {"bytes": 1, "md5": "a", "sha256": "b"}}
    extracted = {"input.mgf": {"bytes": 2, "sha256": "c"}}
    manifest = {
        "zenodo_record": RECORD_ID,
        "zenodo_api": RECORD_API,
        "archives": archives,
        "extracted_inputs": extracted,
    }
    path = tmp_path / "acquisition_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    assert validate_acquisition_manifest(tmp_path, archives, extracted) == path
    assert path.read_bytes() == before
