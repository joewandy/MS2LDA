"""Workflow ordering, leakage, and model-artifact tests."""

from __future__ import annotations

import inspect
import json
import os
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
from benchmarks.neural_ms2lda import chemical, pipeline
from benchmarks.neural_ms2lda.artifacts import (
    build_results,
    load_protocol,
    load_trained_model,
)
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_view_pairs,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from benchmarks.neural_ms2lda.evaluation import evaluate_neural
from benchmarks.neural_ms2lda.mag import _connectivity_key, build_filtered_mag_index
from benchmarks.neural_ms2lda.tomotopy import (
    _converged,
    _infer_theta,
    _validate_alpha,
)
from benchmarks.neural_ms2lda.training import train_model
from benchmarks.neural_ms2lda.utils import write_json, write_jsonl
from scripts.download_msnlib_validation_assets import safe_zip_members

from ._support import chemistry_result, mini_protocol, write_mini_mgf


def test_module_entry_pins_numerical_threads_before_dispatch(monkeypatch: Any) -> None:
    for name in module_entry.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "99")
    module_entry._configure_process_threads()
    assert {os.environ[name] for name in module_entry.THREAD_ENVIRONMENT_VARIABLES} == {
        "6"
    }


def _prepare_mini_training_scaffold(
    run: Path, *, data_root: Path, protocol: dict[str, Any]
) -> None:
    prepare_data(run, data_root=data_root, protocol=protocol)
    data = run / "data"
    train = load_csr(data / "train.npz")
    train_token_features(
        run / "token_features",
        train,
        load_vocabulary(data),
        protocol,
        seed=int(protocol["seed"]),
    )


def _mini_training_arguments(run: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    data = run / "data"
    return {
        "train": load_csr(data / "train.npz"),
        "views": load_view_pairs(run, protocol),
        "validation_full": load_csr(data / "validation_full.npz"),
    }


def test_training_interface_exposes_no_test_inputs() -> None:
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
    protocol = load_protocol()
    events: list[str] = []
    monkeypatch.setattr(pipeline, "initialize_run", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(pipeline, "prepare_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_csr", lambda path: object())
    monkeypatch.setattr(pipeline, "load_vocabulary", lambda path: [])
    monkeypatch.setattr(pipeline, "train_token_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_view_pairs", lambda *args: [])
    monkeypatch.setattr(pipeline, "train_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "train_tomotopy", lambda *args, **kwargs: None)

    def neural(*args: Any, split: str, **kwargs: Any) -> None:
        events.append(f"{split}-neural")

    def tomotopy(*args: Any, split: str, **kwargs: Any) -> None:
        events.append(f"{split}-tomotopy")

    def chemistry(*args: Any, split: str, method: str, **kwargs: Any) -> None:
        events.append(f"{split}-chemistry-{method}")

    monkeypatch.setattr(pipeline, "evaluate_neural", neural)
    monkeypatch.setattr(pipeline, "evaluate_tomotopy", tomotopy)
    monkeypatch.setattr(pipeline, "_chemical_subprocess", chemistry)
    monkeypatch.setattr(pipeline, "build_results", lambda path: {"ok": True})
    pipeline.run_pipeline(tmp_path / "run", data_root=tmp_path)
    first_test = min(i for i, event in enumerate(events) if event.startswith("test"))
    assert (
        max(i for i, event in enumerate(events) if event.startswith("validation"))
        < first_test
    )


def test_mag_index_excludes_validation_and_test_compounds(
    monkeypatch: Any, tmp_path: Path
) -> None:
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
    for split, smiles in (("validation", "CCO"), ("test", "CCN")):
        write_jsonl(
            run / f"data/{split}_records.jsonl",
            [
                {
                    "split": split,
                    "connectivity_key": _connectivity_key(smiles),
                }
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
            "spec2vec_embeddings": embeddings.name,
            "spec2vec_db": database.name,
        }
    }
    result = build_filtered_mag_index(run, data_root=tmp_path, protocol=protocol)
    assert result["excluded_connectivity_keys"] == 2
    assert result["excluded_reference_rows"] == 2
    assert result["retained_reference_rows"] == 1
    assert result["retained_leak_rows"] == 0


def test_mag_annotations_are_generated_once_per_model(
    monkeypatch: Any, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    evaluation = run / "validation_evaluation/neural"
    evaluation.mkdir(parents=True)
    np.save(evaluation / "beta.npy", np.asarray([[0.6, 0.4]], dtype=np.float32))
    write_json(run / "data/vocabulary.json", {"vocabulary": ["frag@1", "loss@1"]})
    monkeypatch.setattr(chemical, "_chemical_inputs", lambda *args: {})
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


def test_test_evaluation_requires_completed_validation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="validation must finish"):
        evaluate_neural(tmp_path, load_protocol(), split="test")
    assert not (tmp_path / "evaluation/neural").exists()


def test_tomotopy_empty_document_uses_the_learned_prior() -> None:
    calls = []

    class FakeModel:
        k = 2
        alpha = np.asarray([0.3, 0.7], dtype=np.float32)

        @staticmethod
        def make_doc(words: list[str]) -> list[str]:
            return words

        @staticmethod
        def infer(documents: list[list[str]], **kwargs: object):
            calls.append(kwargs)
            return [[0.8, 0.2] for _ in documents], None

    theta = _infer_theta(FakeModel(), [[], ["frag@100.0"], []], iterations=5, workers=6)
    assert np.allclose(theta[0], [0.3, 0.7])
    assert np.allclose(theta[1], [0.8, 0.2])
    assert calls == [{"iter": 5, "workers": 6, "parallel": 1, "together": False}]


def test_tomotopy_alpha_and_convergence() -> None:
    class FakeModel:
        k = 2
        alpha = np.asarray([0.1, 0.2])
        optim_interval = 10

    _validate_alpha(FakeModel())
    history = [
        {"perplexity": 100.0},
        {"perplexity": 100.1},
        {"perplexity": 100.2},
    ]
    assert _converged(history, window=2, threshold=0.005)
    FakeModel.alpha = np.asarray([0.1, 0.0])
    with pytest.raises(ValueError, match="not positive"):
        _validate_alpha(FakeModel())


def test_miniature_mgf_through_results_and_model(tmp_path: Path) -> None:
    mgf = tmp_path / "mini.mgf"
    write_mini_mgf(mgf)
    protocol = mini_protocol(mgf)
    run = tmp_path / "run"
    write_json(run / "protocol.json", protocol)
    _prepare_mini_training_scaffold(run, data_root=tmp_path, protocol=protocol)
    train_model(run, protocol=protocol, **_mini_training_arguments(run, protocol))
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(6)
        validation = evaluate_neural(run, protocol, split="validation")
        write_json(run / "validation_chemical/neural/complete.json", chemistry_result())
        test = evaluate_neural(run, protocol, split="test")
    finally:
        torch.set_num_threads(original_threads)
    assert "warm_in_memory_batch_inference" in test["metrics"]

    write_json(run / "validation_evaluation/tomotopy/complete.json", validation)
    write_json(run / "evaluation/tomotopy/complete.json", test)
    write_json(
        run / "tomotopy/complete.json",
        {
            "training_iterations": 10,
            "training_seconds_total": 10.0,
        },
    )
    chemistry = chemistry_result()
    for path in (
        "validation_chemical/tomotopy/complete.json",
        "chemical/neural/complete.json",
        "chemical/tomotopy/complete.json",
    ):
        write_json(run / path, chemistry)
    report = build_results(run)
    model, vocabulary, temperature = load_trained_model(run / "trained_model")
    assert [row["method"] for row in report["methods"]] == ["neural", "tomotopy"]
    assert report["study"]["association_probability_threshold"] == 0.5
    assert model.num_topics == 4
    assert len(vocabulary) == load_csr(run / "data/train.npz").shape[1]
    assert temperature == pytest.approx(0.1)


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_zip_members(archive)
