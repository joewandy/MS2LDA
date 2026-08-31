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
from scripts.download_msnlib_validation_assets import safe_zip_members
from scripts.evaluate_frozen_etm_test import evaluate_test
from scripts.prepare_msnlib_test_view import expose_test_view
from scripts.prepare_msnlib_validation_view import create_validation_view
from scripts.run_contextual_sparse_etm import TrainingSettings, train_real_validation
from scripts.run_contextual_sparse_etm_reproduction import (
    run_stage,
)
from scripts.run_contextual_sparse_etm_reproduction import (
    sha256_file as reproduction_sha256_file,
)
from scripts.run_msnlib_model_comparison import train_etm

from benchmarks.neural_ms2lda import __main__ as module_entry
from benchmarks.neural_ms2lda import chemical, pipeline
from benchmarks.neural_ms2lda.artifacts import (
    build_results,
    initialize_run,
    load_protocol,
    load_trained_model,
)
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from benchmarks.neural_ms2lda.evaluation import evaluate_neural
from benchmarks.neural_ms2lda.mag import _connectivity_key, build_filtered_mag_index
from benchmarks.neural_ms2lda.model_evaluation import (
    TRAINING_ACCESS_AUDIT_FILENAME,
    VALIDATION_ACCESS_AUDIT_FILENAME,
)
from benchmarks.neural_ms2lda.reproduction_plan import (
    Stage,
    probability_artifact_paths,
    reproduction_paths,
    stage_plan,
)
from benchmarks.neural_ms2lda.tomotopy import (
    _converged,
    _infer_theta,
    _validate_alpha,
)
from benchmarks.neural_ms2lda.training import train_model
from benchmarks.neural_ms2lda.utils import read_json, write_json, write_jsonl

from ._support import chemistry_result, mini_protocol, write_mini_mgf


def test_module_entry_pins_numerical_threads_before_dispatch(monkeypatch: Any) -> None:
    for name in module_entry.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "99")
    module_entry._configure_process_threads()
    assert {os.environ[name] for name in module_entry.THREAD_ENVIRONMENT_VARIABLES} == {
        "6"
    }


def test_run_is_bound_to_its_original_data_root(tmp_path: Path) -> None:
    protocol = load_protocol()

    def data_root(name: str) -> Path:
        root = tmp_path / name
        mgf = root / protocol["input_files"]["mgf"]
        mgf.parent.mkdir(parents=True)
        mgf.touch()
        return root

    first = data_root("first")
    second = data_root("second")
    run = tmp_path / "run"
    initialize_run(run, data_root=first)
    assert (run / "data_root.txt").read_text(encoding="utf-8") == f"{first}\n"
    initialize_run(run, data_root=first)
    with pytest.raises(ValueError, match="differs from the original"):
        initialize_run(run, data_root=second)


def test_old_stage_outputs_are_not_silently_adopted(tmp_path: Path) -> None:
    protocol = load_protocol()
    data_root = tmp_path / "inputs"
    mgf = data_root / protocol["input_files"]["mgf"]
    mgf.parent.mkdir(parents=True)
    mgf.touch()
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks a data-root binding"):
        initialize_run(run, data_root=data_root)


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


def _mini_training_arguments(run: Path) -> dict[str, Any]:
    data = run / "data"
    return {"train": load_csr(data / "train.npz")}


def _prepare_mini_frozen_source(root: Path) -> tuple[Path, dict[str, Any]]:
    """Create one tiny prepared run reusable by validation-only model views."""
    mgf = root / "mini.mgf"
    write_mini_mgf(mgf)
    protocol = mini_protocol(mgf)
    prepared = root / "prepared"
    write_json(prepared / "protocol.json", protocol)
    _prepare_mini_training_scaffold(
        prepared,
        data_root=root,
        protocol=protocol,
    )
    write_json(prepared / "mag/index/complete.json", {"retained_leak_rows": 0})
    write_json(prepared / "mag/index/excluded_connectivity_keys.json", [])
    (prepared / "mag/index").mkdir(parents=True, exist_ok=True)
    np.save(prepared / "mag/index/kept_original_ids.npy", np.asarray([], dtype=int))
    (prepared / "mag/index/spec2vec_filtered.faiss").write_bytes(b"mini-index")
    return prepared, protocol


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


def test_chemical_subprocess_uses_active_unified_interpreter(
    monkeypatch: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    interpreter = "/opt/ms2lda-neural/bin/python"
    monkeypatch.setattr(pipeline.sys, "executable", interpreter)
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline._chemical_subprocess(
        tmp_path / "run",
        method="neural",
        data_root=tmp_path / "inputs",
        cpu_threads=6,
        split="validation",
    )
    assert captured["command"][:3] == [
        interpreter,
        "-m",
        "benchmarks.neural_ms2lda.chemical",
    ]
    assert "ms2lda-msnlib-mag" not in captured["command"]
    assert captured["kwargs"]["check"] is True


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
    second, summary = chemical._shared_annotations(
        run, method="neural", data_root=tmp_path, protocol=protocol
    )
    assert first == second
    assert calls == 1
    assert summary["mag_failures"] == {
        "clustering_count": 0,
        "clustering_topic_ids": [],
        "optimization_count": 0,
        "optimization_topic_ids": [],
    }


def test_mag_annotation_failures_are_recorded(monkeypatch: Any) -> None:
    class BrokenClustering:
        @staticmethod
        def __call__(**kwargs: Any) -> None:
            raise RuntimeError("cluster failure")

    annotation_module = SimpleNamespace(
        hit_clustering=BrokenClustering(),
        motif_optimization=lambda *args, **kwargs: [],
    )
    monkeypatch.setitem(
        sys.modules,
        "MS2LDA.Add_On.Spec2Vec.annotation_refined",
        annotation_module,
    )
    rows = chemical._annotate_topics(
        [object()],
        [object()],
        object(),
        {"chemistry": {"mag_cluster_cosine": 0.5}},
    )
    assert rows[0]["clustering_failure"] == {
        "exception_type": "RuntimeError",
        "message": "cluster failure",
    }
    assert rows[0]["optimization_failure"] is None


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
    train_model(run, protocol=protocol, **_mini_training_arguments(run))
    assert torch.are_deterministic_algorithms_enabled()
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


def test_frozen_etm_is_evaluated_only_after_validation(tmp_path: Path) -> None:
    prepared, _ = _prepare_mini_frozen_source(tmp_path)

    run = tmp_path / "frozen-etm"
    manifest = create_validation_view(run, prepared, expected_topics=4)
    assert manifest["test_spectra_exposed_to_model_run"] is False
    train_etm(
        run,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        method="etm",
    )
    write_json(run / "validation_chemical/etm/complete.json", chemistry_result())
    release = expose_test_view(run, prepared, methods=["etm"])
    result = evaluate_test(
        run,
        method="etm",
        device=torch.device("cpu"),
        batch_size=4,
        threads=1,
    )
    assert release["split"] == "test"
    assert result["split"] == "test"
    assert result["weights_unchanged_after_evaluation"] is True
    assert result["metrics"]["document_completion"]["eligible_documents"] > 0
    weights = run / "models/etm/weights.pt"
    weights.write_bytes(weights.read_bytes() + b"changed-after-release")
    with pytest.raises(RuntimeError, match="frozen model changed after test release"):
        evaluate_test(
            run,
            method="etm",
            device=torch.device("cpu"),
            batch_size=4,
            threads=1,
        )


def test_training_audits_do_not_preempt_chemistry_stage_outputs(
    tmp_path: Path,
) -> None:
    """Exercise the runner boundary that separates training and chemistry."""
    paths = reproduction_paths(tmp_path / "reproduction")
    paths.logs.mkdir(parents=True)
    paths.stages.mkdir(parents=True)
    prepared, _ = _prepare_mini_frozen_source(tmp_path)

    create_validation_view(paths.controls, prepared, expected_topics=4)
    train_etm(
        paths.controls,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        method="etm",
    )
    contextual_run = paths.contextual[7043]
    create_validation_view(contextual_run, prepared, expected_topics=4)
    train_real_validation(
        contextual_run,
        device=torch.device("cpu"),
        settings=TrainingSettings(
            epochs=1,
            batch_size=4,
            threads=1,
            requested_seed=7043,
        ),
    )

    stages = {stage.name: stage for stage in stage_plan(paths)}
    cases = (
        (
            paths.controls,
            "etm",
            stages["canonical_etm_train"],
            stages["canonical_etm_chemistry"],
        ),
        (
            contextual_run,
            "contextual_sparse_etm",
            stages["contextual_seed_7043_train"],
            stages["contextual_seed_7043_chemistry"],
        ),
    )
    finalizer = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            (
                "from benchmarks.neural_ms2lda.model_evaluation import "
                "finalize_validation_access_audit"
            ),
            "from benchmarks.neural_ms2lda.utils import write_json",
            "run = Path(sys.argv[1])",
            "method = sys.argv[2]",
            "complete = Path(sys.argv[3])",
            "write_json(complete, {'split': 'validation'})",
            "finalize_validation_access_audit(run, method)",
        )
    )
    for run, method, training_stage, chemistry_stage in cases:
        training_audit = run / "models" / method / TRAINING_ACCESS_AUDIT_FILENAME
        final_audit = run / "models" / method / VALIDATION_ACCESS_AUDIT_FILENAME
        assert training_audit in training_stage.outputs
        assert final_audit in chemistry_stage.outputs
        assert training_audit.is_file()
        assert not final_audit.exists()
        assert all(not output.exists() for output in chemistry_stage.outputs)

        command = (
            sys.executable,
            "-c",
            finalizer,
            str(run),
            method,
            str(chemistry_stage.outputs[0]),
        )
        runnable_stage = Stage(
            chemistry_stage.name,
            command,
            chemistry_stage.outputs,
            chemistry_stage.requires_idle_system,
        )
        record = run_stage(paths, runnable_stage)
        assert record["status"] == "complete"
        assert read_json(final_audit)["chemical_split"] == "validation"


def test_clean_reproduction_plan_has_unique_stage_ownership(tmp_path: Path) -> None:
    paths = reproduction_paths(tmp_path / "reproduction")
    stages = stage_plan(paths)
    names = [stage.name for stage in stages]
    outputs = [str(path) for stage in stages for path in stage.outputs]
    assert len(stages) == 54
    assert len(names) == len(set(names))
    assert len(outputs) == len(set(outputs))
    assert {
        name
        for name in names
        if name.startswith("seal_validation_view_contextual_seed")
    } == {
        "seal_validation_view_contextual_seed_7043",
        "seal_validation_view_contextual_seed_23",
        "seal_validation_view_contextual_seed_37",
    }
    owned = {path for stage in stages for path in stage.outputs}
    method_runs = (
        (paths.controls, "etm"),
        (paths.controls, "etm_balanced"),
        (paths.tomotopy, "tomotopy"),
        *((paths.contextual[seed], "contextual_sparse_etm") for seed in (7043, 23, 37)),
    )
    for run, method in method_runs:
        assert set(probability_artifact_paths(run, method)) <= owned


def test_completed_stage_rejects_changed_sealed_output(tmp_path: Path) -> None:
    paths = reproduction_paths(tmp_path / "reproduction")
    paths.stages.mkdir(parents=True)
    output = paths.root / "owned.txt"
    output.write_text("sealed\n", encoding="utf-8")
    stage = Stage("sealed_stage", (sys.executable, "-c", "pass"), (output,))
    record = {
        "name": stage.name,
        "status": "complete",
        "outputs": [
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": reproduction_sha256_file(output),
            },
        ],
    }
    write_json(paths.stages / f"{stage.name}.json", record)
    assert run_stage(paths, stage) == record
    output.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed stage output changed"):
        run_stage(paths, stage)


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_zip_members(archive)
