# ruff: noqa: PLR2004, S101, TC003
"""Focused mechanics and protocol checks for the fully neural benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.fully_neural_ms2lda.config import (
    REPO_ROOT,
    code_manifest,
    load_protocol,
    static_candidate_audit,
)
from benchmarks.fully_neural_ms2lda.data import (
    build_token_features,
    iter_sparse_batches,
)
from benchmarks.fully_neural_ms2lda.evaluation import nonchemical_hard_gates
from benchmarks.fully_neural_ms2lda.model import (
    NeuralMS2LDA,
    balanced_sinkhorn_plan,
)
from benchmarks.fully_neural_ms2lda.report import attempt_scorecard
from benchmarks.fully_neural_ms2lda.smoke import run_smoke
from benchmarks.fully_neural_ms2lda.training import validation_is_collapsed


def _model_and_batch() -> tuple[NeuralMS2LDA, object]:
    torch.manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(11, 6), dim=1)
    model = NeuralMS2LDA(
        features,
        num_topics=4,
        hidden_dimensions=8,
        topic_word_temperature=0.5,
        dropout=0.0,
        topic_initial_indices=torch.arange(4),
    )
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0],
                [0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 4, 1, 0, 0, 0, 2, 0],
            ],
            dtype=np.float32,
        ),
    )
    batch = next(iter_sparse_batches(matrix, batch_size=3, shuffle=False, seed=0))
    return model, batch


def test_frozen_protocol_has_intended_architecture_and_budgets() -> None:
    protocol = load_protocol()
    assert protocol["num_topics"] == 1000
    assert protocol["training_cpu_threads"] == 4
    assert protocol["evaluation_cpu_threads"] == 1
    assert protocol["sgns"]["dimensions"] == 48
    assert protocol["token_features"]["output_dimensions"] == 64
    assert protocol["model"]["ecr_vocabulary_block"] == 4096
    assert protocol["model"]["ecr_iterations"] == 20
    assert protocol["training"]["maximum_hours_per_attempt"] == 12.0
    assert protocol["rescue"]["eligible_only_for_collapse"] is True
    assert protocol["rescue"]["same_initialization"] is True


def test_candidate_static_audit_excludes_all_forbidden_dependencies() -> None:
    audit = static_candidate_audit(load_protocol())
    assert audit["fully_neural"] is True
    assert audit["violations"] == []
    assert audit["local_vb_steps"] == 0
    assert audit["dreams_used"] is False
    assert audit["tomotopy_or_nmf_initialization_used"] is False


def test_code_manifest_covers_runner_protocol_and_candidate_modules() -> None:
    manifest = code_manifest()
    assert "scripts/run_fully_neural_ms2lda.sh" in manifest
    assert "benchmarks/fully_neural_ms2lda/protocol.json" in manifest
    assert "benchmarks/fully_neural_ms2lda/model.py" in manifest
    assert "benchmarks/fully_neural_ms2lda/training.py" in manifest


def test_package_entrypoint_does_not_eagerly_load_torch() -> None:
    check = (
        "import sys, benchmarks.fully_neural_ms2lda; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", check],
        cwd=REPO_ROOT,
        check=True,
    )


def test_token_features_combine_sgns_mass_and_type_without_labels() -> None:
    config = {
        "fourier_frequencies": [1, 2],
        "mass_scale": 1000.0,
        "type_dimensions": 2,
        "output_dimensions": 12,
    }
    embeddings = np.arange(18, dtype=np.float32).reshape(3, 6) + 1
    result = build_token_features(
        embeddings,
        ["frag@100.00", "loss@18.01", "frag@250.25"],
        config,
    )
    assert result.shape == (3, 12)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
    assert result[0, -2] > 0
    assert result[1, -1] > 0


def test_sinkhorn_plan_has_balanced_marginals_and_gradients() -> None:
    torch.manual_seed(3)
    cost = torch.rand(13, 5, requires_grad=True)
    plan = balanced_sinkhorn_plan(cost, epsilon=0.2, iterations=80)
    torch.testing.assert_close(
        plan.sum(dim=1),
        torch.full((13,), 1 / 13),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        plan.sum(dim=0),
        torch.full((5,), 1 / 5),
        atol=1e-4,
        rtol=0,
    )
    torch.sum(plan * cost).backward()
    assert cost.grad is not None
    assert torch.all(torch.isfinite(cost.grad))


def test_sparse_reconstruction_matches_dense_theta_times_beta() -> None:
    model, batch = _model_and_batch()
    theta, _, _ = model.encode(batch, sample=False)
    beta = model.topic_word_distribution()
    sparse_loss = model.sparse_reconstruction_loss(theta, beta, batch)
    dense = theta @ beta
    selected = dense[batch.row_ids, batch.indices].clamp_min(1e-12)
    expected = -torch.sum(batch.weights * torch.log(selected)) / batch.weights.sum()
    torch.testing.assert_close(sparse_loss, expected)


def test_encoder_is_one_pass_and_rescue_guards_are_finite() -> None:
    model, batch = _model_and_batch()
    projected = model.projected_tokens().detach()
    beta = model.topic_word_distribution(projected).detach()
    terms = model.encoder_loss(
        batch,
        beta=beta,
        projected_tokens=projected,
        kl_weight=0.2,
        usage_guard_weight=0.25,
        sparsity_guard_weight=0.05,
        target_effective_topics=2.0,
    )
    assert torch.isfinite(terms.total)
    assert torch.isfinite(terms.usage_guard)
    assert torch.isfinite(terms.sparsity_guard)
    terms.total.backward()
    assert model.encoder_mean.weight.grad is not None
    assert model.topic_embeddings.grad is None


def test_nonchemical_hard_gates_are_reference_relative() -> None:
    protocol = load_protocol()
    reference = {
        "active_topics": {"corpus_active_topics": 100},
        "top_word_diversity": 0.8,
        "full_spectrum_mixture": {"effective_topic_count_median": 8.0},
        "document_completion": {"nll_per_token": 10.0},
    }
    candidate = {
        "stable": True,
        "fully_neural_audit": {"fully_neural": True},
        "metrics": {
            "active_topics": {"corpus_active_topics": 70},
            "top_word_diversity": 0.65,
            "full_spectrum_mixture": {"effective_topic_count_median": 2.0},
            "test_document_completion": {"nll_per_token": 11.0},
        },
    }
    result = nonchemical_hard_gates(candidate, reference, protocol)
    assert result["pass"] is True


def test_rescue_is_eligible_for_document_mixture_collapse() -> None:
    protocol = load_protocol()
    reference = {
        "active_topics": {"corpus_active_topics": 100},
        "top_word_diversity": 0.8,
        "full_spectrum_mixture": {"effective_topic_count_median": 8.0},
    }
    training_result = {
        "stable": True,
        "selected_validation": {
            "active_topics": {"corpus_active_topics": 100},
            "top_word_diversity": 0.8,
            "mixture_diagnostics": {"effective_topic_count_median": 500.0},
        },
    }
    collapsed, reasons = validation_is_collapsed(
        training_result,
        reference,
        protocol,
    )
    assert collapsed is True
    assert reasons == ["median_effective_topics"]


def test_competitive_misses_do_not_veto_hard_viability() -> None:
    protocol = load_protocol()
    reference = {
        "active_topics": {"corpus_active_topics": 100},
        "top_word_diversity": 0.8,
        "word_cooccurrence_npmi": {"mean_npmi": -0.3},
        "full_spectrum_mixture": {"effective_topic_count_median": 8.0},
        "document_completion": {"nll_per_token": 10.0},
        "cached_latency": {"median_seconds_per_spectrum": 0.06},
        "dominant_topic_chemistry": {"mean_sos": 0.62},
        "high_confidence_chemistry": {"sos_evaluable_coverage": 0.33},
    }
    candidate = {
        "attempt": "primary",
        "stable": True,
        "fully_neural_audit": {"fully_neural": True},
        "metrics": {
            "active_topics": {"corpus_active_topics": 80},
            "top_word_diversity": 0.68,
            "word_cooccurrence_npmi": {"mean_npmi": -0.37},
            "full_spectrum_mixture": {"effective_topic_count_median": 8.0},
            "test_document_completion": {"nll_per_token": 10.8},
            "cached_latency": {"median_seconds_per_spectrum": 0.02},
        },
    }
    chemical = {
        "dominant_topic_chemistry": {"mean_sos": 0.54},
        "high_confidence_chemistry": {"sos_evaluable_coverage": 0.2},
    }
    result = attempt_scorecard(
        candidate,
        chemical=chemical,
        reference=reference,
        protocol=protocol,
    )
    assert result["hard_viability"]["pass"] is True
    assert result["competitive_scorecard"]["all_pass"] is False


def test_synthetic_smoke_exercises_both_alternating_blocks(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    result = run_smoke(output)
    assert result["pass"] is True
    assert result["single_encoder_pass"] is True
    assert result["local_vb_steps"] == 0
    assert output.is_file()


def test_durable_runner_has_valid_bash_syntax() -> None:
    path = REPO_ROOT / "scripts/run_fully_neural_ms2lda.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)  # noqa: S603, S607
    source = path.read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS=4" in source
    assert "screen -dmS" in source
    assert "caffeinate -dimsu" in source
