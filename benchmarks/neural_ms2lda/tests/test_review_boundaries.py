"""Regression coverage for the thermo-nuclear review's integrity boundaries."""

import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda import (
    reproducibility,
    reproduction_summaries,
    test_release,
    validation_data,
)
from benchmarks.neural_ms2lda.chemical import score_precomputed_annotations
from benchmarks.neural_ms2lda.contextual_sparse_etm import centered_log_evidence_offset
from benchmarks.neural_ms2lda.data import load_csr
from benchmarks.neural_ms2lda.utils import input_identity, sha256_file, write_json
from scripts import run_contextual_sparse_etm_reproduction as reproduction_runner
from scripts.prepare_msnlib_validation_view import create_validation_view

from .test_workflow import _prepare_mini_prepared_source


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_count_loader_rejects_invalid_values(tmp_path, value):
    path = tmp_path / "counts.npz"
    sp.save_npz(path, sp.csr_matrix([[value, 1.0]]))
    with pytest.raises(ValueError, match="finite nonnegative"):
        load_csr(path)


@pytest.mark.parametrize("topics", [2, 4, 7])
def test_uniform_evidence_keeps_the_centered_log_derivative(topics):
    evidence = torch.full(
        (1, topics), 1 / topics, dtype=torch.float64, requires_grad=True
    )
    offset = centered_log_evidence_offset(evidence)
    assert torch.equal(offset, torch.zeros_like(offset))
    # Probe a simplex-tangent direction. J = (K/2)(I - 11^T/K), so this
    # zero-sum contrast has gradient (K/2) * contrast, not zero.
    contrast = torch.zeros_like(evidence)
    contrast[0, 0], contrast[0, 1] = 1, -1
    (gradient,) = torch.autograd.grad((offset * contrast).sum(), evidence)
    torch.testing.assert_close(gradient, topics / 2 * contrast, rtol=1e-12, atol=1e-12)


def test_resumed_input_identity_preserves_roles_across_relocation():
    source = {"/old/train.npz": "a", "/old/validation_full.npz": "b"}
    relocated = {"/new/train.npz": "a", "/new/validation_full.npz": "b"}
    swapped = {"/new/train.npz": "b", "/new/validation_full.npz": "a"}
    assert input_identity(source) == input_identity(relocated)
    assert input_identity(source) != input_identity(swapped)
    with pytest.raises(ValueError, match="unique roles"):
        input_identity({"/a/train.npz": "a", "/b/train.npz": "b"})


def test_training_rechecks_sealed_bytes_before_opening_arrays(tmp_path, monkeypatch):
    prepared, _ = _prepare_mini_prepared_source(tmp_path)
    view = tmp_path / "candidate"
    create_validation_view(view, prepared, expected_topics=4)
    # Modify the source behind the linked input after its manifest was written.
    with (prepared / "data/train.npz").open("ab") as handle:
        handle.write(b"modified")

    def unexpected_read(_):
        pytest.fail("array data was opened before the seal was checked")

    monkeypatch.setattr(validation_data, "load_csr", unexpected_read)
    with pytest.raises(RuntimeError, match="manifest-owned input changed"):
        validation_data.load_validation_inputs(view)


def test_training_requires_manifest_coverage(tmp_path):
    write_json(tmp_path / "protocol.json", {})
    write_json(
        tmp_path / "validation_input_manifest.json",
        {"candidate_test_artifacts_accessed": False, "linked_inputs": []},
    )
    with pytest.raises(RuntimeError, match="seal every required input"):
        validation_data.load_validation_inputs(tmp_path)


def test_validation_columns_must_match_training_vocabulary(tmp_path, monkeypatch):
    prepared, _ = _prepare_mini_prepared_source(tmp_path)
    view = tmp_path / "candidate"
    create_validation_view(view, prepared, expected_topics=4)
    original_load = validation_data.load_csr

    def mismatched_validation(path):
        matrix = original_load(path)
        return matrix[:, :-1] if Path(path).name.startswith("validation_") else matrix

    monkeypatch.setattr(validation_data, "load_csr", mismatched_validation)
    with pytest.raises(ValueError, match="validation matrices and vocabulary"):
        validation_data.load_validation_inputs(view)


@pytest.mark.parametrize("ids", [[0, 0], [0], [0, 2]])
def test_chemical_scoring_requires_exact_topic_coverage(ids):
    with pytest.raises(ValueError, match="every topic exactly once"):
        score_precomputed_annotations(
            theta=np.array([[0.8, 0.2]]),
            records=[{}],
            annotations=[{"topic_id": topic} for topic in ids],
            fingerprint_threshold=0.8,
        )


def test_chemical_scoring_requires_matching_records():
    with pytest.raises(ValueError, match="every theta row"):
        score_precomputed_annotations(
            theta=np.array([[0.8, 0.2]]),
            records=[],
            annotations=[],
            fingerprint_threshold=0.8,
        )


def test_atomic_json_rejects_nonfinite_values_without_overwriting(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"score": 0.6})
    before = sha256_file(path)
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert sha256_file(path) == before
    assert list(tmp_path.iterdir()) == [path]


def test_research_modules_do_not_import_command_line_runners():
    package = Path(__file__).parents[1]
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("scripts"), path.name
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith("scripts") for alias in node.names
                ), path.name


def test_research_python_modules_stay_below_one_thousand_lines():
    root = Path(__file__).parents[3]
    paths = list((root / "benchmarks/neural_ms2lda").rglob("*.py"))
    paths += list((root / "scripts").glob("*.py"))
    oversized = {
        str(path.relative_to(root)): len(path.read_text().splitlines())
        for path in paths
        if len(path.read_text().splitlines()) > 1000
    }
    assert not oversized, oversized


@pytest.mark.parametrize("platform, expected", [("linux", 123 * 1024), ("darwin", 123)])
def test_process_memory_has_platform_correct_units(monkeypatch, platform, expected):
    monkeypatch.setattr(reproducibility.sys, "platform", platform)
    monkeypatch.setattr(
        reproducibility.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=123)
    )
    assert reproducibility.sample_runtime_memory()["peak_process_bytes"] == expected


def test_clean_room_does_not_claim_clean_source_if_git_fails(monkeypatch):
    monkeypatch.setattr(
        reproduction_runner, "command_output", lambda _: "unavailable: git failed"
    )
    with pytest.raises(RuntimeError, match="cannot establish source"):
        reproduction_runner.source_state()


def _release_fixture(tmp_path):
    prepared = tmp_path / "prepared"
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    for name in test_release.TEST_DATA_FILES:
        path = prepared / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed fixture")
    for name in (
        "models/etm/weights.pt",
        "models/etm/config.json",
        "models/etm/result.json",
        "validation_evaluation/etm/complete.json",
        "validation_chemical/etm/complete.json",
    ):
        write_json(run / name, {})
    return prepared, run


def test_test_release_rolls_back_its_links_if_publication_fails(tmp_path, monkeypatch):
    prepared, run = _release_fixture(tmp_path)

    def fail_publication(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(test_release, "write_json", fail_publication)
    with pytest.raises(OSError, match="full disk"):
        test_release.expose_test_view(run, prepared, methods=["etm"])
    assert not list((run / "data").iterdir())
    assert not (run / "test_input_manifest.json").exists()
    assert len(list((prepared / "data").iterdir())) == 4


def test_test_release_checks_all_sources_before_creating_links(tmp_path):
    prepared, run = _release_fixture(tmp_path)
    (prepared / "data/test_records.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        test_release.expose_test_view(run, prepared, methods=["etm"])
    assert not list((run / "data").iterdir())


def test_frozen_evaluation_rechecks_released_test_bytes(tmp_path):
    prepared, run = _release_fixture(tmp_path)
    test_release.expose_test_view(run, prepared, methods=["etm"])
    (prepared / "data/test_full.npz").write_bytes(b"changed fixture")
    with pytest.raises(RuntimeError, match="manifest-owned input changed"):
        test_release.verify_released_model(
            run, method="etm", model_path=run / "models/etm/weights.pt"
        )


@pytest.mark.parametrize("mutation", [None, "duplicate_seed", "extra_k"])
def test_synthetic_packaging_requires_exact_fit_inventory(tmp_path, mutation):
    root = Path(__file__).parents[3]
    source = (
        root
        / "research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room"
        / "synthetic_results"
    )
    destination = tmp_path / "synthetic/synthetic_runs"
    shutil.copytree(source, destination)
    path = next(
        path
        for path in destination.glob("*/result.json")
        if "seed_11_K_36" in str(path)
    )
    if mutation is not None:
        result = json.loads(path.read_text())
        if mutation == "duplicate_seed":
            result["config"]["seed"] = 23
            write_json(path, result)
        else:
            result["config"]["fitted_topics"] = 999
            write_json(destination / "unexpected/result.json", result)
        with pytest.raises(RuntimeError, match="frozen seed/formulation plan"):
            reproduction_summaries.synthetic_tables(tmp_path)
    else:
        primary, summary, high_k = reproduction_summaries.synthetic_tables(tmp_path)
        assert (len(primary), len(summary), len(high_k)) == (12, 4, 3)
