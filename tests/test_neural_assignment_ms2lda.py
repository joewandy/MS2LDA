# ruff: noqa: PLR2004, S101, S603, S607, SLF001, TC003
"""Focused architecture, safety, and runner tests for neural assignment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.msnlib_validation.data import PeakGroup
from benchmarks.neural_assignment_ms2lda.config import (
    CONTINUATION_PROTOCOL_PATH,
    REPO_ROOT,
    code_manifest,
    load_protocol,
    resolve_protocol_path,
    static_candidate_audit,
)
from benchmarks.neural_assignment_ms2lda.data import (
    build_token_features,
    prototype_seeding_weights,
    select_view_peak_groups,
    sparse_batch,
)
from benchmarks.neural_assignment_ms2lda.model import (
    NeuralAssignmentMS2LDA,
    balanced_sinkhorn_targets,
    deterministic_kmeans_plus_plus,
    initialize_model,
    recycle_dead_prototypes,
    router_block_loss,
    topic_block_loss,
)
from benchmarks.neural_assignment_ms2lda.report import _markdown
from benchmarks.neural_assignment_ms2lda.smoke import run_smoke
from benchmarks.neural_assignment_ms2lda.synthetic import generate_synthetic
from benchmarks.neural_assignment_ms2lda.training import (
    HardContextQueue,
    diagnose_collapse,
    gate_checks,
    infer_theta,
    routing_temperature,
    sinkhorn_weight,
)


def _matrix() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.asarray(
            [
                [3, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 3, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1],
            ],
            dtype=np.float32,
        ),
    )


def _model() -> NeuralAssignmentMS2LDA:
    torch.manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(12, 64), dim=1)
    model, _ = initialize_model(
        features,
        num_topics=4,
        protocol=load_protocol(),
        seeding_weights=prototype_seeding_weights(_matrix()),
    )
    return model


def test_protocol_freezes_stages_architecture_and_budgets() -> None:
    protocol = load_protocol()
    assert protocol["stages"]["synthetic"]["num_topics"] == 32
    assert protocol["stages"]["k200"]["num_topics"] == 200
    assert protocol["stages"]["k1000"]["num_topics"] == 1000
    assert protocol["training_cpu_threads"] == 4
    assert protocol["evaluation_cpu_threads"] == 1
    assert protocol["model"]["top_k"] == 2
    assert protocol["optimization"]["topic_updates_per_epoch"] == 4
    assert protocol["stop_rule"]["maximum_k1000_attempts"] == 2
    assert protocol["stop_rule"]["no_automatic_annotation_redirect"] is True


def test_exploratory_continuation_changes_only_the_k200_blocking_policy() -> None:
    base = load_protocol()
    continuation = load_protocol(CONTINUATION_PROTOCOL_PATH)
    amendment = continuation["exploratory_amendment"]
    assert amendment["declared_after_validation_observation"] is True
    assert amendment["test_data_touched_in_trigger_run"] is False
    assert amendment["waived_k200_blocking_failures"] == ["active_topics"]
    assert continuation["k200_gates"] == base["k200_gates"]
    assert continuation["k1000_gates"] == base["k1000_gates"]
    assert continuation["chemical_gates"] == base["chemical_gates"]
    assert continuation["rescue"] == base["rescue"]

    metrics = {
        "active_topics": {"corpus_active_topics": 67},
        "top_word_diversity": 0.871,
        "mixture_diagnostics": {"effective_topic_count_median": 11.03},
        "document_completion": {"nll_per_token": 8.292},
        "word_cooccurrence_npmi": {"mean_npmi": -0.515},
    }
    original_gate = gate_checks(
        metrics,
        stage="k200",
        stable=True,
        protocol=base,
    )
    continuation_gate = gate_checks(
        metrics,
        stage="k200",
        stable=True,
        protocol=continuation,
    )
    assert original_gate["raw_pass"] is False
    assert original_gate["pass"] is False
    assert original_gate["blocking_failures"] == ["active_topics"]
    assert continuation_gate["raw_pass"] is False
    assert continuation_gate["pass"] is True
    assert continuation_gate["failed"] == ["active_topics"]
    assert continuation_gate["waived_failures"] == ["active_topics"]
    assert continuation_gate["blocking_failures"] == []


def test_protocol_selection_rejects_uncommitted_paths(tmp_path: Path) -> None:
    assert resolve_protocol_path() != resolve_protocol_path(CONTINUATION_PROTOCOL_PATH)
    with np.testing.assert_raises(ValueError):
        resolve_protocol_path(tmp_path / "ad_hoc.json")


def test_report_discloses_raw_failure_and_exploratory_waiver() -> None:
    rendered = _markdown(
        {
            "decision": "continue_to_k1000",
            "furthest_stage": "k200",
            "selected_attempt": None,
            "exploratory_amendment": {"id": "screen-waiver"},
            "gates": {
                "synthetic": None,
                "k200": {
                    "pass": True,
                    "raw_pass": False,
                    "waived_failures": ["active_topics"],
                },
                "k1000_validation": None,
                "k1000_test": None,
                "chemical": None,
            },
        },
    )
    assert "declared after the v1 K=200 validation result" in rendered
    assert "PASS WITH EXPLORATORY WAIVER (active_topics)" in rendered


def test_candidate_audit_proves_fully_neural_one_pass_contract() -> None:
    audit = static_candidate_audit(load_protocol())
    assert audit["violations"] == []
    assert audit["fully_neural"] is True
    assert audit["encoder_passes_per_representation"] == 1
    assert audit["local_vb_steps"] == 0
    assert audit["dreams_used"] is False
    assert audit["classical_topic_teacher_used"] is False
    assert audit["chemistry_fields_in_model_inputs"] == []


def test_package_entrypoint_does_not_eagerly_import_torch() -> None:
    check = (
        "import sys, benchmarks.neural_assignment_ms2lda; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT, check=True)


def test_code_manifest_covers_runner_protocol_and_shared_dependencies() -> None:
    manifest = code_manifest()
    assert "scripts/run_neural_assignment_ms2lda.sh" in manifest
    assert "scripts/run_neural_assignment_ms2lda_k1000_continuation.sh" in manifest
    assert "benchmarks/neural_assignment_ms2lda/protocol.json" in manifest
    assert (
        "benchmarks/neural_assignment_ms2lda/protocol_k1000_continuation.json"
        in manifest
    )
    assert "benchmarks/neural_assignment_ms2lda/model.py" in manifest
    assert "benchmarks/neural_assignment_ms2lda/training.py" in manifest
    assert "benchmarks/fully_neural_ms2lda/embeddings.py" in manifest
    assert "docs/research/neural_assignment_ms2lda_protocol.md" in manifest


def test_token_features_are_64d_and_do_not_require_labels() -> None:
    protocol = load_protocol()
    embeddings = np.arange(144, dtype=np.float32).reshape(3, 48) + 1
    result = build_token_features(
        embeddings,
        ["frag@100.0", "loss@18.01", "frag@250.25"],
        protocol["token_features"],
    )
    assert result.shape == (3, 64)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_prototype_seed_weights_downweight_ubiquitous_background() -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [5, 1, 0],
                [5, 0, 1],
                [5, 1, 0],
                [5, 0, 1],
            ],
            dtype=np.float32,
        ),
    )
    weights = prototype_seeding_weights(matrix)
    assert weights[0] == 0
    assert weights[1] > 0
    assert weights[2] > 0


def test_sinkhorn_targets_are_balanced_and_differentiable() -> None:
    torch.manual_seed(3)
    logits = torch.rand(41, 7, requires_grad=True)
    targets = balanced_sinkhorn_targets(logits, epsilon=0.2, iterations=100)
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(41), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        targets.sum(dim=0),
        torch.full((7,), 41 / 7),
        atol=1e-4,
        rtol=0,
    )
    torch.sum(targets * logits).backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_kmeans_plus_plus_is_seeded_unique_and_has_no_lloyd_step() -> None:
    torch.manual_seed(5)
    features = torch.randn(30, 8)
    first = deterministic_kmeans_plus_plus(features, clusters=9, seed=42)
    second = deterministic_kmeans_plus_plus(features, clusters=9, seed=42)
    torch.testing.assert_close(first, second)
    assert len(torch.unique(first)) == 9


def test_routing_is_normalized_sparse_and_count_aggregated() -> None:
    model = _model()
    matrix = _matrix()
    batch = sparse_batch(matrix, np.arange(matrix.shape[0], dtype=np.int64))
    output = model.route(batch, temperature=0.2, top_k=2, straight_through=False)
    torch.testing.assert_close(
        output.assignments.sum(dim=1),
        torch.ones(len(output.assignments)),
    )
    assert torch.all((output.assignments > 0).sum(dim=1) == 2)
    torch.testing.assert_close(output.theta.sum(dim=1), torch.ones(matrix.shape[0]))
    expected = model.aggregate_theta(
        output.assignments,
        row_ids=batch.row_ids,
        weights=batch.weights,
        documents=batch.documents,
    )
    torch.testing.assert_close(output.theta, expected)


def test_leave_one_token_out_context_changes_route_embedding() -> None:
    model = _model()
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    batch = sparse_batch(matrix, np.arange(2, dtype=np.int64))
    routes = model._route_embeddings(batch, model.projected_tokens())
    token_zero_rows = torch.nonzero(batch.indices == 0, as_tuple=False).flatten()
    assert len(token_zero_rows) == 2
    assert not torch.allclose(routes[token_zero_rows[0]], routes[token_zero_rows[1]])


def test_router_and_topic_blocks_have_finite_gradients() -> None:
    model = _model()
    matrix = _matrix()
    right = matrix[:, np.roll(np.arange(matrix.shape[1]), 1)].tocsr()
    rows = np.arange(matrix.shape[0], dtype=np.int64)
    left_batch = sparse_batch(matrix, rows)
    right_batch = sparse_batch(right, rows)
    router = router_block_loss(
        model,
        left_batch,
        right_batch,
        cached_beta=model.topic_word_distribution().detach(),
        temperature=0.5,
        top_k=2,
        sinkhorn_weight=0.25,
        consistency_weight=0.1,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=40,
    )
    router.total.backward()
    assert torch.isfinite(router.total)
    assert model.context_router[0].weight.grad is not None
    model.zero_grad(set_to_none=True)
    topic = topic_block_loss(
        model,
        left_batch,
        right_batch,
        temperature=0.5,
        top_k=2,
        local_decoder_weight=0.25,
    )
    topic.total.backward()
    assert torch.isfinite(topic.total)
    assert model.topic_prototypes.grad is not None


def test_one_pass_inference_is_batch_partition_invariant() -> None:
    model = _model()
    first = infer_theta(model, _matrix(), batch_size=1, temperature=0.1, top_k=2)
    second = infer_theta(model, _matrix(), batch_size=4, temperature=0.1, top_k=2)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)


def test_recycling_replaces_only_named_rows_and_resets_adam_state() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.topic_prototypes.square().sum().backward()
    optimizer.step()
    before = model.topic_prototypes.detach().clone()
    rows = torch.tensor([1, 3])
    replacements = torch.nn.functional.normalize(torch.randn(2, 128), dim=1)
    recycle_dead_prototypes(
        model,
        optimizer,
        topic_indices=rows,
        replacements=replacements,
    )
    torch.testing.assert_close(model.topic_prototypes[rows], replacements)
    torch.testing.assert_close(model.topic_prototypes[[0, 2]], before[[0, 2]])
    state = optimizer.state[model.topic_prototypes]
    assert torch.all(state["exp_avg"][rows] == 0)
    assert torch.all(state["exp_avg_sq"][rows] == 0)


def test_hard_context_queue_round_trips_exactly() -> None:
    queue = HardContextQueue.empty(3)
    queue.add(
        torch.tensor([1.0, 4.0, 2.0, 3.0]),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        limit=4,
    )
    restored = HardContextQueue.from_state_dict(queue.state_dict())
    torch.testing.assert_close(queue.pop_highest(3), restored.pop_highest(3))


def test_peak_group_masking_never_splits_fragment_and_loss_tokens() -> None:
    groups = (
        PeakGroup(0, 100.0, 1.0, ("frag@100.0", "loss@50.0") * 3),
        PeakGroup(1, 120.0, 0.8, ("frag@120.0", "loss@30.0") * 2),
        PeakGroup(2, 140.0, 0.5, ("frag@140.0", "loss@10.0")),
    )
    selected = select_view_peak_groups(
        groups,
        spectrum_id="spectrum",
        seed=42,
        pair_index=0,
        side="left",
        retained_fraction=0.8,
    )
    assert 0 < len(selected) < len(groups)
    for group in selected:
        assert group in groups
        assert group.tokens == groups[group.original_index].tokens


def test_real_stage_gates_and_rescue_diagnosis_are_validation_only() -> None:
    protocol = load_protocol()
    metrics = {
        "active_topics": {"corpus_active_topics": 80},
        "top_word_diversity": 0.7,
        "mixture_diagnostics": {"effective_topic_count_median": 60.0},
        "document_completion": {"nll_per_token": 9.0},
    }
    gate = gate_checks(metrics, stage="k1000", stable=True, protocol=protocol)
    collapsed, mode, reasons = diagnose_collapse(gate)
    assert collapsed is True
    assert mode == "both"
    assert set(reasons) == {"active_topics", "median_effective_topics"}


def test_primary_and_rescue_schedules_match_frozen_contract() -> None:
    protocol = load_protocol()
    assert (
        routing_temperature(0, attempt="primary", rescue_mode=None, protocol=protocol)
        == 0.5
    )
    assert (
        routing_temperature(40, attempt="primary", rescue_mode=None, protocol=protocol)
        == 0.1
    )
    assert (
        routing_temperature(
            40,
            attempt="rescue",
            rescue_mode="diffuse",
            protocol=protocol,
        )
        == 0.07
    )
    assert sinkhorn_weight(0, attempt="primary", protocol=protocol) == 0.25
    assert sinkhorn_weight(80, attempt="rescue", protocol=protocol) == 0.15


def test_synthetic_generator_exercises_both_prespecified_problems() -> None:
    separable = generate_synthetic("separable", seed=42, num_topics=8)
    long_tail = generate_synthetic("long_tail_shared_background", seed=42, num_topics=8)
    assert separable.true_beta.shape == long_tail.true_beta.shape
    assert separable.train.shape[0] == 768
    assert len(separable.views) == 4
    assert not np.allclose(separable.true_beta, long_tail.true_beta)


def test_smoke_exercises_both_alternating_blocks(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    result = run_smoke(output)
    assert result["pass"] is True
    assert result["single_routing_pass"] is True
    assert result["local_vb_steps"] == 0
    assert output.is_file()


def test_durable_runner_has_valid_bash_and_merge_gate() -> None:
    path = REPO_ROOT / "scripts/run_neural_assignment_ms2lda.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS=4" in source
    assert "screen -dmS" in source
    assert "caffeinate -dimsu" in source
    assert "NEURAL_ASSIGNMENT_LAUNCH_TOKEN" in source
    assert "NEURAL_ASSIGNMENT_PROTOCOL" in source
    assert "NEURAL_ASSIGNMENT_SESSION_NAME" in source
    assert "NEURAL_ASSIGNMENT_LAUNCH_WAIT_SECONDS:-60" in source
    assert "git branch --show-current" in source
    assert "cmp -s" in source
    continuation = (
        REPO_ROOT / "scripts/run_neural_assignment_ms2lda_k1000_continuation.sh"
    )
    subprocess.run(["bash", "-n", str(continuation)], check=True)
    continuation_source = continuation.read_text(encoding="utf-8")
    assert "protocol_k1000_continuation.json" in continuation_source
    assert "seed42-v2-k1000-continuation" in continuation_source
