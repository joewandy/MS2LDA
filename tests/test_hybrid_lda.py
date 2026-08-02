"""Focused correctness tests for the hybrid MS2LDA reference implementation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from MS2LDA.dreams_features import (
    DreaMSFeatureBatch,
    parse_spectral_word,
    pool_word_embeddings,
)
from MS2LDA.hybrid_lda import (
    HybridLDAConfig,
    HybridLDAModel,
    _expected_log_dirichlet,
    _expected_topic_word_counts,
    _local_vb,
    _make_sparse_batch,
)


def small_config(*, seed: int = 7) -> HybridLDAConfig:
    return HybridLDAConfig(
        num_topics=3,
        embedding_dim=4,
        hidden_size=12,
        feature_projection_dim=6,
        training_local_steps=3,
        batch_size=3,
        encoder_updates_per_epoch=1,
        prior_warmup_epochs=1,
        prior_training_epochs=2,
        max_epochs=3,
        global_patience=2,
        seed=seed,
    )


def test_package_import_does_not_load_the_full_workflow() -> None:
    """Reference submodules must not require the optional application stack."""
    project_root = Path(__file__).resolve().parents[1]
    check = "import sys, MS2LDA.hybrid_lda; assert 'MS2LDA.run' not in sys.modules"

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", check],
        cwd=project_root,
        check=True,
    )


def documents() -> tuple[list[list[str]], np.ndarray, dict[str, np.ndarray]]:
    words = [
        ["frag@100.00", "frag@100.00", "loss@50.00"],
        ["frag@100.00", "loss@50.00", "frag@110.00"],
        ["frag@200.00", "frag@200.00", "loss@60.00"],
        ["frag@200.00", "loss@60.00", "frag@210.00"],
        ["frag@300.00", "frag@300.00", "loss@70.00"],
        ["frag@300.00", "loss@70.00", "frag@310.00"],
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.9, 0.1, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.1],
            [0.1, 0.9, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.1, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    vocabulary = dict.fromkeys(word for document in words for word in document)
    word_embeddings = {
        word: np.asarray(
            [index == 0, index == 1, index >= 2, 0.1],
            dtype=np.float32,
        )
        for index, word in enumerate(vocabulary)
    }
    return words, embeddings, word_embeddings


def prepared_model(*, seed: int = 7) -> HybridLDAModel:
    words, embeddings, word_embeddings = documents()
    model = HybridLDAModel(small_config(seed=seed))
    model.set_word_embeddings(word_embeddings)
    for document, embedding in zip(words, embeddings, strict=True):
        model.add_doc(document, embedding=embedding)
    model.train(0)
    return model


def topic_matrix(model: HybridLDAModel) -> np.ndarray:
    return np.vstack([model.get_topic_word_dist(topic) for topic in range(model.k)])


def document_topics(model: HybridLDAModel) -> np.ndarray:
    return np.vstack([document.get_topic_dist() for document in model.docs])


def test_config_rejects_invalid_invariants() -> None:
    with pytest.raises(ValueError, match="prior training"):
        HybridLDAConfig(
            num_topics=3,
            embedding_dim=4,
            prior_warmup_epochs=3,
            prior_training_epochs=2,
        )
    with pytest.raises(ValueError, match="alpha"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, alpha=(0.1, 0.2))
    with pytest.raises(ValueError, match="fixed-prior epoch"):
        HybridLDAConfig(
            num_topics=3,
            embedding_dim=4,
            prior_warmup_epochs=1,
            prior_training_epochs=5,
            max_epochs=5,
        )


def test_sparse_batch_preserves_counts_without_dense_vocabulary_tensor() -> None:
    matrix = sp.csr_matrix([[3.0, 0.0, 2.0], [0.0, 7.0, 0.0]])
    batch = _make_sparse_batch(matrix, [0, 1], device=torch.device("cpu"))

    assert batch.word_ids.shape == batch.word_counts.shape
    torch.testing.assert_close(batch.totals, torch.tensor([[5.0], [7.0]]))
    assert float((batch.word_counts * batch.word_mask).sum()) == 12.0


def test_one_local_vb_step_matches_the_two_lda_equations() -> None:
    matrix = sp.csr_matrix([[2.0, 1.0]])
    batch = _make_sparse_batch(matrix, [0], device=torch.device("cpu"))
    alpha = torch.tensor([0.5, 0.5])
    initial_gamma = torch.tensor([[1.0, 1.0]])
    expected_log_beta = torch.tensor([[0.0, -2.0], [-2.0, 0.0]])

    initial_phi = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    expected_gamma = alpha + (torch.tensor([2.0, 1.0])[:, None] * initial_phi).sum(0)
    gamma, final_phi = _local_vb(
        batch,
        initial_gamma,
        alpha,
        expected_log_beta,
        steps=1,
        tolerance=None,
    )

    torch.testing.assert_close(gamma[0], expected_gamma)
    torch.testing.assert_close(
        final_phi,
        torch.softmax(
            _expected_log_dirichlet(gamma).unsqueeze(1)
            + expected_log_beta.transpose(0, 1).unsqueeze(0),
            dim=2,
        ),
    )


def test_local_updates_preserve_dirichlet_mass_and_token_counts() -> None:
    model = prepared_model()
    assert model._matrix is not None and model._core is not None
    batch = _make_sparse_batch(model._matrix, [0, 1], device=model.device)
    core = model._core
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    gamma = core.encode(batch, model._embedding_batch(model.docs, [0, 1]), word_topic)
    refined, phi = _local_vb(
        batch,
        gamma,
        core.alpha,
        expected_log_beta,
        steps=2,
        tolerance=None,
    )
    expected_mass = batch.totals[:, 0] + core.alpha.sum()

    torch.testing.assert_close(refined.sum(dim=1), expected_mass)
    statistics = _expected_topic_word_counts(
        batch,
        phi,
        num_topics=model.k,
        vocab_size=model.num_vocabs,
    )
    assert float(statistics.sum()) == pytest.approx(float(batch.totals.sum()))


def test_structured_prior_has_declared_fixed_mass() -> None:
    model = prepared_model()
    assert model._core is not None and model._matrix is not None
    core = model._core
    tokens = float(model._matrix.sum())
    prior = core.structured_prior(tokens, epoch=1)
    expected = (
        core.vocab_size * model.config.eta
        + model.config.prior_mass_fraction * tokens / model.k
    )

    torch.testing.assert_close(
        prior.sum(dim=1),
        torch.full((model.k,), expected),
    )
    assert bool(torch.all(prior > 0))


def test_model_construction_does_not_change_global_torch_rng() -> None:
    words, embeddings, word_embeddings = documents()
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    model = HybridLDAModel(small_config())
    model.set_word_embeddings(word_embeddings)
    for document, embedding in zip(words, embeddings, strict=True):
        model.add_doc(document, embedding=embedding)
    model.train(0)

    torch.testing.assert_close(torch.rand(5), expected)


def test_convergence_starts_only_after_the_prior_is_fixed() -> None:
    words, embeddings, word_embeddings = documents()
    config = replace(
        small_config(),
        global_tolerance=1e9,
        global_patience=1,
        max_epochs=4,
    )
    model = HybridLDAModel(config)
    model.set_word_embeddings(word_embeddings)
    for document, embedding in zip(words, embeddings, strict=True):
        model.add_doc(document, embedding=embedding)

    model.train(2)
    assert not model.converged
    model.train(1)
    assert model.converged


def test_training_inference_and_safe_checkpoint_round_trip(tmp_path: Path) -> None:
    model = prepared_model()
    model.train(2)
    words, embeddings, _ = documents()
    query = model.make_doc(words[0], embedding=embeddings[0])
    original_theta, original_ll = model.infer(query, iter=5)
    checkpoint = tmp_path / "hybrid.bin"
    model.save(checkpoint)

    restored = HybridLDAModel.load(checkpoint)
    restored_query = restored.make_doc(words[0], embedding=embeddings[0])
    restored_theta, restored_ll = restored.infer(restored_query, iter=5)

    np.testing.assert_array_equal(topic_matrix(restored), topic_matrix(model))
    np.testing.assert_array_equal(restored_theta, original_theta)
    assert restored_ll == original_ll
    assert restored.docs == []
    assert restored._encoder_optimizer is None
    assert restored._prior_optimizer is None
    with pytest.raises(RuntimeError, match="cannot resume"):
        restored.train(1)


def test_same_seed_reproduces_trained_topics() -> None:
    first = prepared_model(seed=11)
    second = prepared_model(seed=11)
    first.train(2)
    second.train(2)

    np.testing.assert_array_equal(topic_matrix(first), topic_matrix(second))
    np.testing.assert_array_equal(document_topics(first), document_topics(second))


def test_document_embedding_affects_zero_step_amortized_inference() -> None:
    model = prepared_model()
    model.train(2)
    words, embeddings, _ = documents()
    first = model.make_doc(words[0], embedding=embeddings[0])
    second = model.make_doc(words[0], embedding=embeddings[4])

    first_theta, _ = model.infer(first, iter=0)
    second_theta, _ = model.infer(second, iter=0)

    assert not np.allclose(first_theta, second_theta)


def test_tomotopy_shaped_document_and_topic_accessors() -> None:
    model = prepared_model()
    model.train(1)

    assert model.num_vocabs == len(model.vocabs)
    assert model.docs[0].get_topic_dist().sum() == pytest.approx(1.0)
    assert len(model.get_topic_words(0, top_n=2)) == 2
    assert model.get_topic_word_dist(0).sum() == pytest.approx(1.0)
    assert np.isfinite(model.ll_per_word)
    assert np.isfinite(model.perplexity)


def feature_batch() -> DreaMSFeatureBatch:
    peak_embeddings = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return DreaMSFeatureBatch(
        identifiers=("a", "b"),
        spectrum_embeddings=np.ones((2, 4), dtype=np.float32),
        peak_embeddings=peak_embeddings,
        peak_mz=np.asarray([[100.0, 150.0], [100.0, 150.0]], dtype=np.float32),
        peak_mask=np.ones((2, 2), dtype=bool),
        precursor_mz=np.asarray([150.0, 150.0], dtype=np.float32),
        provenance={"model": "test"},
    )


def test_peak_states_pool_into_fragment_and_loss_words() -> None:
    features = feature_batch()
    documents = [
        ["frag@100.00", "loss@50.00"],
        ["frag@100.00", "frag@100.00", "loss@50.00"],
    ]
    pooled = pool_word_embeddings(documents, features)

    np.testing.assert_allclose(
        pooled["frag@100.00"],
        np.asarray([7.0 / 3.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        pooled["loss@50.00"],
        np.asarray([2.0, 0.0, 0.0, 0.0]),
    )


def test_spectral_word_parser_is_shared_and_strict() -> None:
    assert parse_spectral_word("Frag@100.5") == ("frag", 100.5)
    assert parse_spectral_word("loss@20") == ("loss", 20.0)
    assert parse_spectral_word("frag@nan") is None
    assert parse_spectral_word("loss@-1") is None
    assert parse_spectral_word("other@10") is None


def test_dreams_feature_cache_round_trip(tmp_path: Path) -> None:
    features = feature_batch()
    path = tmp_path / "features.h5"
    features.save(path)
    restored = DreaMSFeatureBatch.load(path)

    assert restored.identifiers == features.identifiers
    assert restored.provenance == features.provenance
    np.testing.assert_array_equal(
        restored.spectrum_embeddings,
        features.spectrum_embeddings,
    )
    np.testing.assert_array_equal(restored.peak_embeddings, features.peak_embeddings)
