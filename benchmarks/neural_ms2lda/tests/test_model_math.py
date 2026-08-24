"""Equation-level tests for the supported neural model and its safeguards."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.artifacts import load_protocol, load_trained_model
from benchmarks.neural_ms2lda.chemical import _sos_bands
from benchmarks.neural_ms2lda.data import (
    build_token_features,
    sparse_batch,
)
from benchmarks.neural_ms2lda.model import (
    DOCUMENT_MIXTURE_EXPONENT,
    TOPICS_PER_TOKEN,
    initialize_model,
)
from benchmarks.neural_ms2lda.objectives import (
    balanced_sinkhorn_targets,
    cooccurrence_topic_loss,
    positive_npmi_graph,
    router_block_loss,
    topic_block_loss,
    topic_separation_loss,
    torch_sparse_graph,
)
from benchmarks.neural_ms2lda.optimization import (
    _weighted_topic_separation,
)
from benchmarks.neural_ms2lda.spectra import (
    SpectrumRecord,
    audit_split_disjointness,
    build_training_vocabulary,
)
from benchmarks.neural_ms2lda.utils import read_json

from ._support import spectrum_record, token_features


def test_first_seen_training_vocabulary_excludes_test_words() -> None:
    records = [
        spectrum_record("a", ["frag@2.0", "frag@1.0", "frag@2.0"]),
        spectrum_record("b", ["loss@3.0", "frag@1.0"]),
        spectrum_record("c", ["frag@9.0"]),
    ]
    vocabulary, summary = build_training_vocabulary(
        records,
        {"a": "train", "b": "train", "c": "test"},
        min_df=1,
        min_cf=0,
        rm_top=0,
    )
    assert vocabulary == ("frag@2.0", "frag@1.0", "loss@3.0")
    assert summary["order"] == "raw_training_spectra_first_seen"


def test_split_audit_rejects_compound_leakage() -> None:
    records = [
        spectrum_record("a", ["frag@1.0"]),
        spectrum_record("b", ["frag@2.0"]),
    ]
    records[1] = SpectrumRecord(
        **{**records[1].__dict__, "connectivity_key": records[0].connectivity_key}
    )
    with pytest.raises(ValueError, match="split leakage"):
        audit_split_disjointness(records, {"a": "train", "b": "test"})


def test_token_features_are_sgns_plus_fragment_loss_indicators() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    vocabulary = ["frag@1.0", "loss@2.0", "frag@3.0", "loss@4.0"]
    features = build_token_features(embeddings, vocabulary)
    assert features.shape == (4, 6)
    assert np.all(features[[0, 2], -2] > 0)
    assert np.all(features[[1, 3], -1] > 0)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)


def test_fixed_equal_channel_decoder_matches_the_report_equation() -> None:
    protocol = load_protocol()
    features = token_features(10, fragments=2)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    projected = model.projected_tokens()
    topics = torch.nn.functional.normalize(model.topic_prototypes, dim=1)
    logits = 2.0 * topics @ projected.T / model.beta_temperature
    fragment_logits = logits[:, :2]
    loss_logits = logits[:, 2:]
    expected = torch.empty_like(logits)
    expected[:, :2] = 0.5 * torch.softmax(fragment_logits, dim=1)
    expected[:, 2:] = 0.5 * torch.softmax(loss_logits, dim=1)
    actual = model.topic_word_distribution(projected)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.allclose(actual.sum(dim=1), torch.ones(4), atol=1e-6)


def test_equal_logits_are_neutral_to_unequal_vocabulary_sizes() -> None:
    protocol = load_protocol()
    model, _ = initialize_model(
        token_features(10, fragments=2), num_topics=4, protocol=protocol
    )
    beta = model.topic_word_distribution(torch.zeros(10, model.projection_dimensions))
    assert torch.allclose(beta[:, :2].sum(dim=1), torch.full((4,), 0.5))
    assert torch.allclose(beta[:, 2:].sum(dim=1), torch.full((4,), 0.5))
    assert torch.allclose(beta[:, :2], torch.full((4, 2), 0.25))
    assert torch.allclose(beta[:, 2:], torch.full((4, 8), 0.0625))


def test_decoder_supplies_finite_prototype_gradients() -> None:
    protocol = load_protocol()
    model, _ = initialize_model(
        token_features(10, fragments=2), num_topics=4, protocol=protocol
    )
    beta = model.topic_word_distribution()
    (-torch.log(beta[:, 0]).mean()).backward()
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()


def test_sinkhorn_top2_and_topic_gradients_are_finite() -> None:
    protocol = load_protocol()
    model, _ = initialize_model(token_features(20), num_topics=4, protocol=protocol)
    matrix = sp.csr_matrix(np.eye(4, 20, dtype=np.float32) + 1.0)
    batch = sparse_batch(matrix, np.arange(4, dtype=np.int64))
    router = router_block_loss(
        model,
        batch,
        cached_beta=model.topic_word_distribution().detach(),
        temperature=0.5,
        sinkhorn_weight=0.25,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=20,
    )
    assert torch.all((router.output.assignments > 0).sum(dim=1) == 2)
    router.total.backward()
    assert model.context_projection.weight.grad is not None
    model.zero_grad(set_to_none=True)
    topic = topic_block_loss(
        model,
        batch,
        temperature=0.5,
    )
    topic.total.backward()
    assert model.topic_prototypes.grad is not None
    targets = balanced_sinkhorn_targets(
        torch.randn(40, 4, generator=torch.Generator().manual_seed(42)),
        epsilon=0.2,
        iterations=100,
    )
    assert torch.allclose(targets.sum(dim=1), torch.ones(40), atol=1e-5)
    assert torch.allclose(targets.sum(dim=0), torch.full((4,), 10.0), atol=1e-5)


def test_document_score_is_shared_by_tokens_in_one_spectrum() -> None:
    protocol = load_protocol()
    model, _ = initialize_model(token_features(20), num_topics=4, protocol=protocol)
    matrix = sp.csr_matrix(np.eye(2, 20, dtype=np.float32) + 1.0)
    batch = sparse_batch(matrix, np.arange(2, dtype=np.int64))
    projected = model.projected_tokens()
    routes, document_routes = model._route_embeddings(batch, projected)
    topics = torch.nn.functional.normalize(model.topic_prototypes, dim=1)
    local_logits = routes @ topics.T
    document_logits = document_routes @ topics.T
    output = model.route(
        batch,
        temperature=0.5,
        straight_through=False,
        projected_tokens=projected,
    )
    added = output.logits - local_logits
    assert torch.allclose(added, document_logits[batch.row_ids])
    for row in range(batch.documents):
        rows = added[batch.row_ids == row]
        assert torch.allclose(rows, rows[:1].expand_as(rows))


def test_document_gate_preserves_support_and_empty_fallback() -> None:
    assignments = torch.tensor(
        [[0.6, 0.4, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
        requires_grad=True,
    )
    logits = torch.tensor(
        [[1.0, -1.0, 4.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        requires_grad=True,
    )
    model, _ = initialize_model(
        token_features(8), num_topics=4, protocol=load_protocol()
    )
    theta = model.aggregate_theta(
        assignments,
        row_ids=torch.tensor([0, 0]),
        weights=torch.tensor([1.0, 2.0]),
        documents=2,
        document_logits=logits,
        temperature=0.5,
    )
    assert torch.allclose(theta.sum(dim=1), torch.ones(2))
    assert torch.equal(theta[0, 2:], torch.zeros(2))
    assert torch.allclose(theta[1], torch.full((4,), 0.25))
    theta[0, 0].backward()
    assert assignments.grad is not None and torch.isfinite(assignments.grad).all()
    assert logits.grad is None


def test_train_only_regularizers_supply_finite_topic_gradients() -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    graph = positive_npmi_graph(
        matrix,
        minimum_document_frequency=1,
        minimum_pair_frequency=1,
        maximum_neighbors=2,
        minimum_npmi=0.0,
    )
    model, _ = initialize_model(
        token_features(6), num_topics=4, protocol=load_protocol()
    )
    beta = model.topic_word_distribution()
    cooccurrence = cooccurrence_topic_loss(model, torch_sparse_graph(graph), beta=beta)
    separation = topic_separation_loss(model, neighbors=2, margin=0.3)
    (cooccurrence + separation).backward()
    assert torch.isfinite(cooccurrence)
    assert torch.isfinite(separation)
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()


def test_protocol_and_model_artifact_expose_only_the_current_architecture() -> None:
    protocol = load_protocol()
    assert protocol["cpu_threads"] == 6
    assert protocol["model"] == {
        "num_topics": 1000,
        "projection_dimensions": 128,
        "beta_temperature": 0.18,
    }
    assert "token_features" not in protocol
    assert "views" not in protocol
    assert "local_decoder_weight" not in protocol["optimization"]
    assert "theta_consistency_weight" not in protocol["optimization"]
    assert "validation_interval" not in protocol["optimization"]
    assert "num_topics" not in protocol["tomotopy"]
    model, _ = initialize_model(token_features(20), num_topics=4, protocol=protocol)
    separation_config = copy.deepcopy(protocol["topic_separation"])
    separation_config["neighbors"] = 2
    weighted = _weighted_topic_separation(model, separation_config)
    expected = float(separation_config["weight"]) * topic_separation_loss(
        model,
        neighbors=int(separation_config["neighbors"]),
        margin=float(separation_config["margin"]),
    )
    assert torch.allclose(weighted, expected)

    artifact = Path(__file__).parents[1] / "results/seed42/trained_model"
    assert sorted(path.name for path in artifact.iterdir()) == [
        "model.json",
        "vocabulary.json",
        "weights.pt",
    ]
    assert set(read_json(artifact / "model.json")) == {
        "beta_temperature",
        "routing_temperature",
    }
    trained_model, vocabulary, temperature = load_trained_model(artifact)
    assert trained_model.num_topics == 1000
    assert trained_model.input_dimensions == 50
    assert isinstance(trained_model.context_projection, torch.nn.Linear)
    assert DOCUMENT_MIXTURE_EXPONENT == 0.75
    assert TOPICS_PER_TOKEN == 2
    assert temperature == pytest.approx(0.1)
    assert len(vocabulary) == trained_model.vocabulary_size


def test_report_constants_match_the_executable_model() -> None:
    """Prevent the paper's displayed model from drifting from the implementation."""
    report = (
        Path(__file__).parents[3] / "docs/research/neural_ms2lda_report.tex"
    ).read_text(encoding="utf-8")
    protocol = load_protocol()
    assert rf"\tau_\beta={protocol['model']['beta_temperature']}" in report
    assert r"\frac{1}{2}" in report
    assert rf"\gamma={DOCUMENT_MIXTURE_EXPONENT}" in report
    assert TOPICS_PER_TOKEN == 2
    assert "Only the two largest entries" in report


def test_sos_bands_include_boundaries_exactly_once() -> None:
    bands = _sos_bands([0.0, 0.5999, 0.6, 0.7, 0.8, 0.8001, 1.0])
    assert bands == {
        "high_gt_0_8": 2,
        "intermediate_0_6_to_0_8": 3,
        "low_lt_0_6": 2,
    }
    assert sum(bands.values()) == 7
