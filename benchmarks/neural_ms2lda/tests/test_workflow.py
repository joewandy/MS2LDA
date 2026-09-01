"""Workflow ordering, leakage, and model-artifact tests."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda import chemical
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from benchmarks.neural_ms2lda.mag import _connectivity_key, build_filtered_mag_index
from benchmarks.neural_ms2lda.reproducibility import resolve_torch_device
from benchmarks.neural_ms2lda.reproduction_plan import (
    Stage,
    probability_artifact_paths,
    reproduction_paths,
    stage_plan,
)
from benchmarks.neural_ms2lda.study_protocol import (
    TRAINING_ACCESS_AUDIT_FILENAME,
    VALIDATION_ACCESS_AUDIT_FILENAME,
    initialize_run,
    load_protocol,
)
from benchmarks.neural_ms2lda.tomotopy import (
    _converged,
    _infer_theta,
    _validate_alpha,
)
from benchmarks.neural_ms2lda.utils import read_json, write_json, write_jsonl
from scripts import run_etm_controls as controls
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
from scripts.run_etm_controls import train_control

from ._support import chemistry_result, mini_protocol, write_mini_mgf


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


def test_preexisting_stage_outputs_are_not_silently_adopted(tmp_path: Path) -> None:
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


def _prepare_mini_prepared_source(root: Path) -> tuple[Path, dict[str, Any]]:
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


def test_validation_records_do_not_open_the_test_file(tmp_path: Path) -> None:
    validation = {"split": "validation", "spectrum_id": "validation-1"}
    (tmp_path / "validation_records.jsonl").write_text(
        json.dumps(validation) + "\n", encoding="utf-8"
    )
    (tmp_path / "test_records.jsonl").write_text("not-json\n", encoding="utf-8")
    assert load_heldout_records(tmp_path, "validation") == [validation]


def test_cuda_device_request_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_etm_control_loader_opens_validation_data_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []

    def fake_load(path: object) -> sp.csr_matrix:
        opened.append(Path(str(path)).name)
        return sp.csr_matrix(np.ones((2, 4), dtype=np.float32))

    def fake_records(path: object, split: str) -> list[dict[str, str]]:
        opened.append(f"records:{split}")
        return [{"split": split}, {"split": split}]

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(controls, "load_csr", fake_load)
    monkeypatch.setattr(controls, "load_heldout_records", fake_records)
    monkeypatch.setattr(controls, "load_vocabulary", lambda _: ["a", "b", "c", "d"])
    monkeypatch.setattr(
        controls,
        "load_sgns_embeddings",
        lambda _: np.ones((4, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        controls,
        "read_json_object",
        lambda path: (
            {"model": {"num_topics": 4}}
            if Path(path).name == "protocol.json"
            else {"candidate_test_artifacts_accessed": False}
        ),
    )

    loaded = controls.load_control_data(tmp_path)

    assert loaded.train.shape == (2, 4)
    assert opened
    assert all("test" not in path for path in opened)
    assert "records:validation" in opened


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


def test_selected_etm_is_evaluated_only_after_validation(tmp_path: Path) -> None:
    prepared, _ = _prepare_mini_prepared_source(tmp_path)

    run = tmp_path / "selected-etm"
    manifest = create_validation_view(run, prepared, expected_topics=4)
    assert manifest["test_spectra_exposed_to_model_run"] is False
    train_control(
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
    prepared, _ = _prepare_mini_prepared_source(tmp_path)

    create_validation_view(paths.controls, prepared, expected_topics=4)
    train_control(
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
    assert len(stages) == 57
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

    device_commands = [stage.command for stage in stages if "--device" in stage.command]
    assert device_commands
    assert all(
        command[command.index("--device") + 1] == "cuda" for command in device_commands
    )


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


def test_failed_stage_is_auditable_and_can_retry_cleanly(tmp_path: Path) -> None:
    paths = reproduction_paths(tmp_path / "reproduction")
    paths.logs.mkdir(parents=True)
    paths.stages.mkdir(parents=True)
    marker = paths.root / "first-attempt.marker"
    output = paths.root / "owned.txt"
    program = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "marker, output = map(Path, sys.argv[1:])",
            "if not marker.exists():",
            "    marker.write_text('failed\\n', encoding='utf-8')",
            "    output.write_text('partial\\n', encoding='utf-8')",
            "    raise SystemExit(3)",
            "output.write_text('complete\\n', encoding='utf-8')",
        )
    )
    stage = Stage(
        "retryable_stage",
        (sys.executable, "-c", program, str(marker), str(output)),
        (output,),
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_stage(paths, stage)
    failed = paths.stages / "retryable_stage.attempt-1.failed.json"
    assert failed.is_file()
    assert read_json(failed)["discarded_partial_outputs"] == [str(output)]
    assert not output.exists()
    assert not (paths.stages / "retryable_stage.json").exists()

    record = run_stage(paths, stage)
    assert record["status"] == "complete"
    assert record["attempt"] == 2
    assert output.read_text(encoding="utf-8") == "complete\n"
    assert (paths.logs / "retryable_stage.attempt-1.log").is_file()
    assert (paths.logs / "retryable_stage.attempt-2.log").is_file()


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_zip_members(archive)
