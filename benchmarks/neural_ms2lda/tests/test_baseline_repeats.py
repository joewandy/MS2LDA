"""Repeated baseline fits vary optimization randomness, not the data protocol."""

from threading import Barrier

import numpy as np
import pytest
import torch

from benchmarks.neural_ms2lda.baseline_repeat_evidence import (
    FROZEN,
    check_chemical_associations,
    check_etm_recipe,
    comparison_row,
    package,
    publish_evidence,
    scheduling_handoff,
    scientific_inputs,
    stage_provenance,
)
from benchmarks.neural_ms2lda.baseline_repeats import (
    baseline_fit_identity,
    require_cached_fit,
    resolve_training_seed,
)
from benchmarks.neural_ms2lda.test_release import expose_test_view
from benchmarks.neural_ms2lda.tomotopy import evaluate_tomotopy, train_tomotopy
from benchmarks.neural_ms2lda.utils import read_json_object, sha256_file, write_json
from scripts.prepare_msnlib_validation_view import create_validation_view
from scripts.run_etm_controls import train_control

from .test_workflow import _prepare_mini_prepared_source


def test_training_seed_defaults_and_overrides():
    protocol = {"seed": 42}
    assert resolve_training_seed(protocol, None) == 42
    assert resolve_training_seed(protocol, None, offset=7001) == 7043
    assert resolve_training_seed(protocol, 7012, offset=7001) == 7012
    assert resolve_training_seed(protocol, 7024, offset=7001) + 18 == 7042
    assert protocol == {"seed": 42}


@pytest.mark.parametrize("seed", [-1, True, 2**31, 1.5])
def test_training_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="training seed"):
        resolve_training_seed({"seed": 42}, seed)


def test_baseline_cache_rejects_wrong_seed_recipe_and_inputs(tmp_path):
    prepared, _ = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "repeat"
    create_validation_view(run, prepared, expected_topics=4)
    expected = baseline_fit_identity(run, 11, {"epochs": 1})
    require_cached_fit({"fit_identity": expected}, expected)
    for field, changed in (
        ("training_seed", 23),
        ("recipe", {"epochs": 2}),
        ("input_sha256", {}),
    ):
        with pytest.raises(RuntimeError, match="cached baseline"):
            require_cached_fit({"fit_identity": {**expected, field: changed}}, expected)
    with (prepared / "data/train.npz").open("ab") as handle:
        handle.write(b"modified after sealing")
    with pytest.raises(RuntimeError, match="manifest-owned input changed"):
        baseline_fit_identity(run, 11, {"epochs": 1})


def test_etm_repeat_seed_and_shuffle_leave_protocol_unchanged(tmp_path):
    prepared, _ = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "etm"
    create_validation_view(run, prepared, expected_topics=4)
    before = sha256_file(prepared / "protocol.json")
    kwargs = dict(method="etm", device=torch.device("cpu"), epochs=1, batch_size=4)
    result = train_control(run, **kwargs, training_seed=7012)
    assert result["config"]["seed"] == 7012
    assert result["config"]["shuffle_seed"] == 7030
    assert sha256_file(prepared / "protocol.json") == before
    assert train_control(run, **kwargs, training_seed=7012) == result
    with pytest.raises(RuntimeError, match="cached baseline"):
        train_control(run, **kwargs, training_seed=7024)
    assert not list((run / "data").glob("test*"))
    (run / "validation_evaluation/etm/beta.npy").unlink()
    with pytest.raises(RuntimeError, match="missing required validation artifacts"):
        train_control(run, **kwargs, training_seed=7012)


def test_tomotopy_repeat_seed_leaves_protocol_unchanged(tmp_path):
    prepared, protocol = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "lda"
    create_validation_view(run, prepared, expected_topics=4)
    before = sha256_file(prepared / "protocol.json")
    result = train_tomotopy(run, protocol, training_seed=11)
    assert result["training_seed"] == 11
    assert sha256_file(prepared / "protocol.json") == before
    assert train_tomotopy(run, protocol, training_seed=11) == result
    with pytest.raises(RuntimeError, match="cached baseline"):
        train_tomotopy(run, protocol, training_seed=23)
    assert not list((run / "data").glob("test*"))
    (run / "tomotopy/model.bin").unlink()
    with pytest.raises(RuntimeError, match="checkpoint is missing or changed"):
        train_tomotopy(run, protocol, training_seed=11)


@pytest.mark.parametrize("changed", ["model", "training", "beta", "theta", "legacy"])
def test_tomotopy_evaluation_cache_rejects_changed_fit_or_outputs(tmp_path, changed):
    prepared, protocol = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "lda"
    create_validation_view(run, prepared, expected_topics=4)
    train_tomotopy(run, protocol, training_seed=11)
    result = evaluate_tomotopy(run, protocol, split="validation")
    assert evaluate_tomotopy(run, protocol, split="validation") == result
    evaluation = run / "validation_evaluation/tomotopy"
    if changed == "legacy":
        result.pop("artifact_sha256")
        write_json(evaluation / "complete.json", result)
    elif changed == "training":
        path = run / "tomotopy/complete.json"
        training = read_json_object(path)
        training["training_seconds_total"] += 1
        write_json(path, training)
    else:
        path = {
            "model": run / "tomotopy/model.bin",
            "beta": evaluation / "beta.npy",
            "theta": evaluation / "validation_full_theta.npy",
        }[changed]
        with path.open("ab") as handle:
            handle.write(b"changed after evaluation")
    with pytest.raises(RuntimeError, match="checkpoint changed|identity or artifacts"):
        evaluate_tomotopy(run, protocol, split="validation")


@pytest.mark.parametrize("missing", ["model", "training", "beta", "theta"])
def test_tomotopy_evaluation_cache_rejects_missing_artifacts(tmp_path, missing):
    prepared, protocol = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "lda"
    create_validation_view(run, prepared, expected_topics=4)
    train_tomotopy(run, protocol, training_seed=11)
    evaluate_tomotopy(run, protocol, split="validation")
    evaluation = run / "validation_evaluation/tomotopy"
    path = {
        "model": run / "tomotopy/model.bin",
        "training": run / "tomotopy/complete.json",
        "beta": evaluation / "beta.npy",
        "theta": evaluation / "validation_full_theta.npy",
    }[missing]
    path.unlink()
    with pytest.raises(RuntimeError, match="complete training|missing required"):
        evaluate_tomotopy(run, protocol, split="validation")


def test_released_tomotopy_test_evaluation_preserves_the_training_boundary(tmp_path):
    """Use only miniature fixture test data, never the real sealed study test set."""
    prepared, protocol = _prepare_mini_prepared_source(tmp_path)
    run = tmp_path / "lda"
    create_validation_view(run, prepared, expected_topics=4)
    train_tomotopy(run, protocol, training_seed=11)
    evaluate_tomotopy(run, protocol, split="validation")
    write_json(run / "validation_chemical/tomotopy/complete.json", {"fixture": True})
    before = sha256_file(run / "tomotopy/model.bin")
    expose_test_view(run, prepared, methods=["tomotopy"])
    result = evaluate_tomotopy(run, protocol, split="test")
    assert result["split"] == "test"
    assert evaluate_tomotopy(run, protocol, split="test") == result
    assert sha256_file(run / "tomotopy/model.bin") == before
    with pytest.raises(RuntimeError, match="exposes test inputs"):
        train_tomotopy(run, protocol, training_seed=11)
    with (prepared / "data/test_completion.npz").open("ab") as handle:
        handle.write(b"changed after authorized test release")
    with pytest.raises(RuntimeError, match="manifest-owned input changed"):
        evaluate_tomotopy(run, protocol, split="test")


def test_historical_etm_row_uses_its_actual_recipe_and_metrics():
    source = FROZEN / "controls/etm"
    result = read_json_object(source / "result.json")
    chemistry = read_json_object(source / "validation_chemical.json")
    check_etm_recipe(result, 7043)
    row = comparison_row(
        "canonical ETM",
        7043,
        "frozen_etm_seed7043",
        result["metrics"]["document_completion"],
        chemistry,
        result["metrics"]["theta_distribution"],
        result["parameters"],
    )
    assert (row["evaluable_motifs"], row["useful_motifs"]) == (199, 118)
    assert row["completion_tokens"] == 1277983
    assert row["completion_documents"] == 3888
    assert row["unique_top1_topics"] == 272
    assert row["median_effective_topics"] == pytest.approx(40.46365024045499)
    result["config"]["batch_size"] = 200
    with pytest.raises(ValueError, match="recipe differs"):
        check_etm_recipe(result, 7043)


def test_new_etm_recipe_rejects_unlike_optimizer_or_shuffle():
    result = read_json_object(FROZEN / "controls/etm/result.json")
    recipe = {
        "method": "etm",
        "topics": 1000,
        "epochs": 120,
        "batch_size": 256,
        "hidden": 800,
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "adam_betas": [0.9, 0.999],
        "device": "cuda",
        "shuffle_seed": 7061,
    }
    result["fit_identity"] = {"recipe": recipe}
    check_etm_recipe(result, 7043)
    for key, value in (("adam_betas", [0.5, 0.999]), ("shuffle_seed", 7043)):
        result["fit_identity"]["recipe"] = {**recipe, key: value}
        with pytest.raises(ValueError, match="optimizer or shuffle"):
            check_etm_recipe(result, 7043)


def test_chemistry_checks_per_topic_associations_not_only_the_total():
    theta = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    scores = [
        {"topic_id": 0, "associated_spectra": 2},
        {"topic_id": 1, "associated_spectra": 1},
    ]
    chemistry = {"chemical_evaluation": {"topic_scores": scores}}
    check_chemical_associations(chemistry, theta)
    scores[0]["associated_spectra"], scores[1]["associated_spectra"] = 1, 2
    with pytest.raises(ValueError, match="this fit's mixtures"):
        check_chemical_associations(chemistry, theta)
    scores[0]["associated_spectra"], scores[1]["associated_spectra"] = 2, 1
    scores.append(dict(scores[0]))
    with pytest.raises(ValueError, match="this fit's mixtures"):
        check_chemical_associations(chemistry, theta)


def test_scientific_inputs_preserve_roles_but_ignore_protocol_export_format():
    manifest = {
        "linked_inputs": [
            {"linked_path": "/first/protocol.json", "sha256": "serialized-original"},
            {"linked_path": "/first/data/train.npz", "sha256": "training"},
            {"linked_path": "/first/data/validation_full.npz", "sha256": "validation"},
        ]
    }
    before = scientific_inputs(manifest)
    manifest["linked_inputs"][0]["sha256"] = "canonical-json-export"
    assert scientific_inputs(manifest) == before
    manifest["linked_inputs"][1]["sha256"] = "validation"
    manifest["linked_inputs"][2]["sha256"] = "training"
    assert scientific_inputs(manifest) != before
    manifest["linked_inputs"].append(dict(manifest["linked_inputs"][1]))
    with pytest.raises(ValueError, match="duplicate"):
        scientific_inputs(manifest)


def test_packaging_refuses_incomplete_runs_without_publishing(tmp_path):
    root, output = tmp_path / "run", tmp_path / "evidence"
    write_json(root / "driver.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="must finish"):
        package(root, output)
    assert not output.exists()


def test_handoff_requires_completed_fits_and_exact_runner_bytes(tmp_path):
    audit = tmp_path / "parallel_handoff"
    audit.mkdir()
    runner = audit / "handoff_runner.py"
    runner.write_text("# exact executed handoff\n", encoding="utf-8")
    record = {
        "status": "supervisor_resumed",
        "model_recipe_changed": False,
        "source_snapshot_changed": False,
        "runner_sha256": sha256_file(runner),
        "completed": [{"training_seed": 23}, {"training_seed": 42}],
    }
    write_json(audit / "handoff.json", record)
    assert scheduling_handoff(tmp_path) == record
    runner.write_text("# altered after execution\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="handoff"):
        scheduling_handoff(tmp_path)


def test_actual_parallel_launches_remain_distinct_from_cache_rechecks(tmp_path):
    identity = "tomotopy_seed23_attempt1"
    for origin in (tmp_path, tmp_path / "parallel_handoff"):
        (origin / "logs").mkdir(parents=True)
        for name in ("seal", "fit_validation", "chemistry"):
            write_json(
                origin / "stages" / f"{identity}.{name}.json",
                {
                    "run": str(tmp_path / "real" / identity),
                    "stage": name,
                    "status": "complete",
                    "return_code": 0,
                },
            )
            (origin / "logs" / f"{identity}.{name}.log").write_text(
                str(origin), encoding="utf-8"
            )
    payloads, logs, records = stage_provenance(tmp_path, identity, {"completed": []})
    assert len(payloads) == len(logs) == 6
    assert set(records) == {"execution", "cache_recheck"}
    assert "/parallel_handoff/" in records["execution"][1]["raw_record"]["path"]
    assert "/parallel_handoff/" not in records["cache_recheck"][1]["raw_record"]["path"]
    stage_path = tmp_path / "parallel_handoff/stages" / f"{identity}.chemistry.json"
    failed = read_json_object(stage_path)
    failed.update(status="failed", return_code=1)
    write_json(stage_path, failed)
    with pytest.raises(RuntimeError, match="incomplete, failed or mismatched"):
        stage_provenance(tmp_path, identity, {"completed": []})


def test_future_driver_submits_independent_lda_fits_and_one_gpu_queue(
    tmp_path, monkeypatch
):
    from scripts import run_baseline_repeats as driver

    # All four jobs must enter before any exits: a regression to serial LDA
    # scheduling fails the barrier rather than quietly extending experiment time.
    barrier, submissions = Barrier(4, timeout=5), []

    def record_queue(kind, seeds, root, snapshot, prepared, assets, lock):
        with lock:
            submissions.append((kind, seeds, id(lock)))
        barrier.wait()
        return []

    root = tmp_path / "repeats"
    monkeypatch.setattr(driver, "snapshot_source", lambda destination: destination)
    monkeypatch.setattr(driver, "run_queue", record_queue)
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [
            "run_baseline_repeats",
            "--root",
            str(root),
            "--prepared-run",
            str(tmp_path / "prepared"),
            "--data-root",
            str(tmp_path / "assets"),
        ],
    )
    assert driver.main() == 0
    assert {(kind, seeds) for kind, seeds, _ in submissions} == {
        ("tomotopy", (11,)),
        ("tomotopy", (23,)),
        ("tomotopy", (42,)),
        ("etm", (7012, 7024)),
    }
    assert len({lock for _, _, lock in submissions}) == 1
    record = read_json_object(root / "driver.json")
    assert record["status"] == "complete"
    assert record["finished_utc"] >= record["started_utc"]


def test_failed_process_launch_keeps_a_failed_stage_record(tmp_path, monkeypatch):
    from scripts import run_baseline_repeats as driver

    (tmp_path / "logs").mkdir()
    run = tmp_path / "real/tomotopy_seed11_attempt1"
    record_path = tmp_path / "stages" / f"{run.name}.fit_validation.json"

    def fail_launch(*args, **kwargs):
        assert read_json_object(record_path)["status"] == "starting"
        raise OSError("simulated launch failure")

    monkeypatch.setattr(driver.subprocess, "Popen", fail_launch)
    with pytest.raises(RuntimeError, match="launch record"):
        driver.run_stage(tmp_path, tmp_path, run, "fit_validation", "example", [])
    record = read_json_object(record_path)
    assert record["status"] == "failed"
    assert (
        record["return_code"] is None
    )  # No child existed; do not invent an exit code.
    assert "simulated launch failure" in record["error"]
    assert record["finished_utc"] >= record["started_utc"]


def _publication_inputs(tmp_path):
    """Small mechanical artifact fixture; no scientific metrics or fitting."""
    root = tmp_path / "run"
    source = root / "source_snapshot/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("# immutable executed source\n", encoding="utf-8")
    log = root / "actual.log"
    log.write_text(f"source={root}\n", encoding="utf-8")
    return {
        "root": root,
        "payloads": {
            "execution_source_manifest.json": {
                "sources": [{"path": "example.py", "sha256": sha256_file(source)}]
            },
            "postprocessing_source_manifest.json": [],
            "summary.json": {"test_fixture": True},
            "protocol.json": {"test_fixture": True},
        },
        "text_artifacts": {"run/execution.log": log},
        "rows": [{"model": "test fixture", "run": "example"}],
        "manifest": {"schema_version": 1, "reported_split": "validation"},
        "handoff": None,
    }


def test_publication_seals_every_file_and_keeps_originals(tmp_path):
    inputs = _publication_inputs(tmp_path)
    output = tmp_path / "public"
    publish_evidence(output, **inputs)
    manifest = read_json_object(output / "manifest.json")
    expected_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert {row["path"] for row in manifest["sources"]} == expected_paths
    for row in manifest["sources"]:
        path = output / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256_file(path) == row["sha256"]
    assert (inputs["root"] / "actual.log").is_file()
    assert "<repeat-root>" in (output / "run/execution.log").read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".baseline-evidence-*"))


def test_publication_failure_cleans_only_temporary_copies(tmp_path, monkeypatch):
    from benchmarks.neural_ms2lda import baseline_repeat_evidence as evidence

    inputs = _publication_inputs(tmp_path)
    output = tmp_path / "public"
    original = inputs["root"] / "source_snapshot/example.py"
    before = sha256_file(original)

    def fail_archive(*args, **kwargs):
        raise OSError("simulated archive failure")

    monkeypatch.setattr(evidence.tarfile, "open", fail_archive)
    with pytest.raises(OSError, match="simulated archive failure"):
        publish_evidence(output, **inputs)
    assert not output.exists()
    assert not list(tmp_path.glob(".baseline-evidence-*"))
    assert sha256_file(original) == before
    assert (inputs["root"] / "actual.log").is_file()


def test_publication_does_not_overwrite_an_existing_destination(tmp_path):
    inputs = _publication_inputs(tmp_path)
    output = tmp_path / "public"
    output.mkdir()
    keeper = output / "previous.txt"
    keeper.write_text("preserve existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh evidence"):
        publish_evidence(output, **inputs)
    assert keeper.read_text(encoding="utf-8") == "preserve existing evidence"
    assert not list(tmp_path.glob(".baseline-evidence-*"))
