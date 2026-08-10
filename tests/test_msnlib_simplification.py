# ruff: noqa: PLR2004, S101
"""Focused tests for the frozen HybridLDA simplification study."""

from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.msnlib_simplification import chemical, runtime
from benchmarks.msnlib_simplification.data import _matrix
from benchmarks.msnlib_simplification.encoders import (
    EvidenceEncoder,
    _evidence,
    _restore_encoder_checkpoint,
    _save_encoder_checkpoint,
    analytic_gamma,
    uniform_gamma,
)
from benchmarks.msnlib_simplification.orchestrator import _tasks
from benchmarks.msnlib_simplification.report import _group_bootstrap
from benchmarks.msnlib_simplification.spec import (
    ARM_IDS,
    BUDGETS,
    DISCOVERY_IDS,
    INFERENCE_IDS,
    SimplificationSpec,
    verify_archived_study,
)
from benchmarks.msnlib_validation.config import file_sha256, object_sha256, write_json
from ms2lda_hybrid._variational import make_sparse_batch


def test_frozen_factorial_matrix_contains_exactly_ten_arms() -> None:
    spec = SimplificationSpec()

    assert spec.discoveries == DISCOVERY_IDS
    assert spec.inference_modes == INFERENCE_IDS
    assert spec.budgets == BUDGETS == (0, 1, 2, 50)
    assert spec.training_cpu_threads == 4
    assert spec.evaluation_cpu_threads == 1
    assert len(ARM_IDS) == 10
    assert len(set(ARM_IDS)) == len(ARM_IDS)
    assert {arm for arm in ARM_IDS if arm.startswith("symmetric_prior__topic_")} == {
        "symmetric_prior__topic_semi",
        "symmetric_prior__topic_direct",
    }


def test_spec_rejects_posthoc_matrix_and_seed_changes() -> None:
    spec = SimplificationSpec()

    with pytest.raises(ValueError, match="seed 42"):
        replace(spec, seed=43)
    with pytest.raises(ValueError, match="inference matrix"):
        replace(spec, inference_modes=("analytic",))
    with pytest.raises(ValueError, match="budgets"):
        replace(spec, budgets=(0, 2, 50))
    with pytest.raises(ValueError, match="four training threads"):
        replace(spec, training_cpu_threads=1)
    with pytest.raises(ValueError, match="four training threads"):
        replace(spec, evaluation_cpu_threads=4)


def test_archived_study_verifies_frozen_source_without_live_checkout(tmp_path) -> None:
    run_dir = tmp_path / "run"
    source_root = run_dir / "frozen_source"
    source_root.mkdir(parents=True)
    source_file = source_root / "example.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    spec = SimplificationSpec()
    manifest = {"example.py": file_sha256(source_file)}
    lock = {
        "spec_sha256": object_sha256(spec.as_dict()),
        "code_manifest_sha256": object_sha256(manifest),
    }
    lock["lock_sha256"] = object_sha256(lock)
    write_json(run_dir / "spec.resolved.json", spec.as_dict())
    write_json(run_dir / "code_manifest.json", manifest)
    write_json(run_dir / "simplification.lock.json", lock)

    verified = verify_archived_study(run_dir)

    assert verified["verified_frozen_source_files"] == 1
    assert verified["verified_frozen_source_root"] == str(source_root)


def test_archived_study_rejects_frozen_source_drift(tmp_path) -> None:
    run_dir = tmp_path / "run"
    source_root = run_dir / "frozen_source"
    source_root.mkdir(parents=True)
    source_file = source_root / "example.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    spec = SimplificationSpec()
    manifest = {"example.py": file_sha256(source_file)}
    lock = {
        "spec_sha256": object_sha256(spec.as_dict()),
        "code_manifest_sha256": object_sha256(manifest),
    }
    lock["lock_sha256"] = object_sha256(lock)
    write_json(run_dir / "spec.resolved.json", spec.as_dict())
    write_json(run_dir / "code_manifest.json", manifest)
    write_json(run_dir / "simplification.lock.json", lock)
    source_file.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="archived source file changed"):
        verify_archived_study(run_dir)


def test_cpu_thread_policy_separates_training_from_evaluation(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runtime, "verify_study", lambda _: {})
    monkeypatch.setattr(runtime, "load_spec", lambda _: SimplificationSpec())
    monkeypatch.setattr(runtime.torch, "set_num_threads", calls.append)
    monkeypatch.setattr(runtime.torch, "set_num_interop_threads", lambda _: None)

    assert runtime.configure_cpu_threads("unused", "training") == 4
    assert runtime.configure_cpu_threads("unused", "evaluation") == 1
    assert calls == [4, 1]


def test_completed_symmetric_annotation_skips_build_only_imports(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "chemical/annotations/symmetric_prior"
    output.mkdir(parents=True)
    (output / "complete.json").write_text("{}\n", encoding="utf-8")
    expected = {"discovery": "symmetric_prior"}
    monkeypatch.setattr(chemical, "verify_study", lambda _: {})
    monkeypatch.setattr(chemical, "_verify_annotations", lambda *_: expected)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "faiss" or name.startswith("MS2LDA"):
            message = f"unexpected build-only import: {name}"
            raise AssertionError(message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert chemical.annotate_symmetric_discovery(tmp_path) == expected


def test_count_matrix_preserves_multiplicity_and_omits_oov_words() -> None:
    matrix = _matrix(
        [["frag@1", "frag@1", "loss@2", "outside"], ["loss@2"]],
        ("frag@1", "loss@2"),
    )

    np.testing.assert_array_equal(matrix.toarray(), [[2.0, 1.0], [0.0, 1.0]])


def _batch() -> tuple[object, torch.Tensor, torch.Tensor]:
    matrix = sp.csr_matrix(
        np.asarray([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]], dtype=np.float32),
    )
    batch = make_sparse_batch(matrix, [0, 1], device=torch.device("cpu"))
    alpha = torch.asarray([0.2, 0.3], dtype=torch.float32)
    word_topic = torch.asarray(
        [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]],
        dtype=torch.float32,
    )
    return batch, alpha, word_topic


def test_analytic_initializer_uses_only_count_topic_evidence() -> None:
    batch, alpha, word_topic = _batch()

    gamma = analytic_gamma(batch, alpha, word_topic)
    evidence = _evidence(batch, word_topic)

    np.testing.assert_allclose(
        gamma.numpy(),
        (alpha.unsqueeze(0) + batch.totals * evidence).numpy(),
    )
    np.testing.assert_allclose(
        gamma.sum(dim=1).numpy(),
        (alpha.sum() + batch.totals[:, 0]).numpy(),
    )


def test_uniform_initializer_is_independent_of_word_identity() -> None:
    batch, alpha, _ = _batch()

    gamma = uniform_gamma(batch, alpha)
    residual = gamma - alpha.unsqueeze(0)

    np.testing.assert_allclose(residual[:, 0], residual[:, 1])


def test_topic_only_encoder_rejects_dreams_embeddings() -> None:
    batch, alpha, word_topic = _batch()
    evidence = _evidence(batch, word_topic)
    encoder = EvidenceEncoder(
        num_topics=2,
        embedding_dim=4,
        hidden_size=5,
        feature_projection_dim=3,
        use_dreams=False,
        seed=42,
    )

    gamma = encoder(evidence, batch.totals, alpha, None)

    assert encoder.document_projector is None
    assert gamma.shape == (2, 2)
    with pytest.raises(ValueError, match="must not receive"):
        encoder(evidence, batch.totals, alpha, torch.zeros(2, 4))


def test_dreams_encoder_requires_aligned_embeddings() -> None:
    batch, alpha, word_topic = _batch()
    evidence = _evidence(batch, word_topic)
    encoder = EvidenceEncoder(
        num_topics=2,
        embedding_dim=4,
        hidden_size=5,
        feature_projection_dim=3,
        use_dreams=True,
        seed=42,
    )

    assert encoder(evidence, batch.totals, alpha, torch.zeros(2, 4)).shape == (2, 2)
    with pytest.raises(ValueError, match="requires"):
        encoder(evidence, batch.totals, alpha, None)


def test_encoder_initialization_is_seed_deterministic() -> None:
    settings = {
        "num_topics": 3,
        "embedding_dim": 4,
        "hidden_size": 5,
        "feature_projection_dim": 2,
        "use_dreams": False,
        "seed": 42,
    }

    first = EvidenceEncoder(**settings)
    second = EvidenceEncoder(**settings)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_group_bootstrap_is_deterministic_and_reports_denominators() -> None:
    values = np.asarray([1.0, 2.0, 10.0, np.nan])
    groups = ["a", "a", "b", "c"]

    first = _group_bootstrap(values, groups, replicates=100, seed=42)
    second = _group_bootstrap(values, groups, replicates=100, seed=42)

    assert first == second
    assert first["estimate"] == pytest.approx(13 / 3)
    assert first["groups"] == 3
    assert first["observations"] == 3
    assert first["replicates"] == 100


def test_overnight_dependency_graph_covers_every_arm_before_freeze() -> None:
    tasks = _tasks()
    by_name = {task.name: task for task in tasks}
    arm_tasks = {f"arm_{arm}" for arm in ARM_IDS}

    assert arm_tasks <= set(by_name)
    assert set(by_name["freeze_models"].requires) == arm_tasks
    current_arm_indices = [
        index
        for index, task in enumerate(tasks)
        if task.name.startswith("arm_dreams_prior__")
    ]
    symmetric_discovery_index = next(
        index for index, task in enumerate(tasks) if task.name == "discovery_symmetric"
    )
    assert max(current_arm_indices) < symmetric_discovery_index
    assert by_name["test"].requires == ("validation",)
    assert by_name["report"].requires == ("chemical_scores",)
    assert by_name["verify"].requires == ("report",)


def test_encoder_resume_falls_back_from_corrupt_newest_generation(tmp_path) -> None:
    settings = {
        "num_topics": 2,
        "embedding_dim": 4,
        "hidden_size": 5,
        "feature_projection_dim": 3,
        "use_dreams": False,
        "seed": 42,
    }
    encoder = EvidenceEncoder(**settings)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)
    context = "a" * 64
    for epoch in (1, 2):
        _save_encoder_checkpoint(
            tmp_path,
            context=context,
            epoch=epoch,
            encoder=encoder,
            optimizer=optimizer,
            rng=rng,
            history=[{"inference_epoch": float(epoch)}],
            elapsed_seconds=float(epoch),
            keep=2,
        )
    (tmp_path / "checkpoints/checkpoint-0002.pt").write_bytes(b"corrupt")
    restored = EvidenceEncoder(**settings)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)

    epoch, _, history, elapsed = _restore_encoder_checkpoint(
        tmp_path,
        context=context,
        encoder=restored,
        optimizer=restored_optimizer,
        seed=42,
    )

    assert epoch == 1
    assert history == [{"inference_epoch": 1.0}]
    assert elapsed == 1.0
