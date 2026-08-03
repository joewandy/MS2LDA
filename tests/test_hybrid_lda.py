"""Focused correctness tests for the hybrid MS2LDA reference implementation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.inference_baselines import fit_posterior_regression_baseline
from benchmarks.semi_amortized_inference import METHODS, run_seed
from ms2lda_hybrid import HybridLDAConfig, HybridLDAModel
from ms2lda_hybrid._variational import (
    corpus_elbo_minibatch_scale as _corpus_elbo_minibatch_scale,
)
from ms2lda_hybrid._variational import (
    expected_log_dirichlet as _expected_log_dirichlet,
)
from ms2lda_hybrid._variational import (
    expected_topic_word_counts as _expected_topic_word_counts,
)
from ms2lda_hybrid._variational import (
    local_document_elbo as _local_document_elbo,
)
from ms2lda_hybrid._variational import (
    local_vb as _local_vb,
)
from ms2lda_hybrid._variational import (
    make_sparse_batch as _make_sparse_batch,
)
from ms2lda_hybrid.dreams_features import (
    DreaMSFeatureBatch,
    parse_spectral_word,
    pool_word_embeddings,
    spectrum_arrays,
)


def small_config(*, seed: int = 7) -> HybridLDAConfig:
    return HybridLDAConfig(
        num_topics=3,
        embedding_dim=4,
        hidden_size=12,
        feature_projection_dim=6,
        training_local_steps=3,
        batch_size=3,
        inference_epochs=2,
        prior_warmup_epochs=1,
        prior_training_epochs=2,
        max_epochs=3,
        global_patience=2,
        seed=seed,
    )


def test_package_import_does_not_load_the_full_workflow() -> None:
    """The reference package must not import the production workflow."""
    project_root = Path(__file__).resolve().parents[1]
    check = "import sys, ms2lda_hybrid; assert 'MS2LDA' not in sys.modules"

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


def prepared_model(
    config: HybridLDAConfig | None = None,
    *,
    seed: int = 7,
) -> HybridLDAModel:
    words, embeddings, word_embeddings = documents()
    model = HybridLDAModel(small_config(seed=seed) if config is None else config)
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
    with pytest.raises(ValueError, match="finite"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, encoder_learning_rate=np.nan)
    with pytest.raises(ValueError, match="eta"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, eta=np.nan)
    with pytest.raises(ValueError, match="alpha"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, alpha=np.nan)
    with pytest.raises(ValueError, match="positive integers"):
        HybridLDAConfig(num_topics=3.5, embedding_dim=4)
    with pytest.raises(ValueError, match="seed"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        HybridLDAConfig(num_topics=3, embedding_dim=4, seed=2**64)


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


def test_local_vb_tolerance_is_relative_below_unit_gamma() -> None:
    matrix = sp.csr_matrix([[0.1]])
    batch = _make_sparse_batch(matrix, [0], device=torch.device("cpu"))
    alpha = torch.tensor([0.1, 0.1])
    initial_gamma = torch.tensor([[0.1, 0.1]])
    expected_log_beta = torch.tensor([[0.0], [-2.0]])

    one_step, _ = _local_vb(
        batch,
        initial_gamma,
        alpha,
        expected_log_beta,
        steps=1,
        tolerance=None,
    )
    two_steps, _ = _local_vb(
        batch,
        initial_gamma,
        alpha,
        expected_log_beta,
        steps=2,
        tolerance=None,
    )
    adaptive, _ = _local_vb(
        batch,
        initial_gamma,
        alpha,
        expected_log_beta,
        steps=2,
        tolerance=0.1,
    )

    assert not torch.allclose(one_step, two_steps)
    torch.testing.assert_close(adaptive, two_steps)


def test_sparse_local_elbo_matches_expanded_responsibility_formula() -> None:
    matrix = sp.csr_matrix([[2.0, 1.0], [0.0, 3.0]])
    batch = _make_sparse_batch(matrix, [0, 1], device=torch.device("cpu"))
    alpha = torch.tensor([0.3, 0.7])
    gamma = torch.tensor([[2.1, 1.9], [0.8, 3.2]])
    expected_log_beta = torch.tensor([[-0.2, -2.0], [-1.7, -0.1]])

    expected_log_theta = _expected_log_dirichlet(gamma)
    word_values = expected_log_beta[:, batch.word_ids].permute(1, 2, 0)
    logits = expected_log_theta.unsqueeze(1) + word_values
    phi = torch.softmax(logits, dim=2)
    counts = batch.word_counts * batch.word_mask
    categorical = (
        counts.unsqueeze(-1) * phi * (logits - phi.clamp_min(1e-30).log())
    ).sum(dim=(1, 2))
    negative_kl = (
        torch.lgamma(alpha.sum())
        - torch.lgamma(alpha).sum()
        - torch.lgamma(gamma.sum(dim=1))
        + torch.lgamma(gamma).sum(dim=1)
        + ((alpha.unsqueeze(0) - gamma) * expected_log_theta).sum(dim=1)
    )

    collapsed = _local_document_elbo(
        batch,
        gamma,
        alpha,
        expected_log_beta,
    )
    torch.testing.assert_close(collapsed, negative_kl + categorical)


def test_document_minibatch_scale_is_unbiased_for_unequal_documents() -> None:
    document_elbos = np.asarray([-1.0, -5.0, -20.0])
    corpus_tokens = 26.0
    expected = float(document_elbos.sum() / corpus_tokens)

    for batch_size in (1, 2):
        estimates = []
        for indices in combinations(range(len(document_elbos)), batch_size):
            scale = _corpus_elbo_minibatch_scale(
                corpus_documents=len(document_elbos),
                batch_documents=batch_size,
                corpus_tokens=corpus_tokens,
            )
            estimates.append(float(document_elbos[list(indices)].sum() * scale))
        assert np.mean(estimates) == pytest.approx(expected)


def test_coordinate_refinement_does_not_reduce_local_elbo() -> None:
    matrix = sp.csr_matrix(
        [[8.0, 1.0, 0.0, 0.0], [4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 1.0, 8.0]]
    )
    batch = _make_sparse_batch(matrix, [0, 1, 2], device=torch.device("cpu"))
    alpha = torch.tensor([0.2, 0.2])
    lambda_posterior = torch.tensor([[30.0, 10.0, 1.0, 1.0], [1.0, 1.0, 10.0, 30.0]])
    expected_log_beta = _expected_log_dirichlet(lambda_posterior)
    initial = alpha.unsqueeze(0) + batch.totals / 2.0
    previous = _local_document_elbo(batch, initial, alpha, expected_log_beta)

    for steps in (1, 2, 3, 5, 10):
        gamma, _ = _local_vb(
            batch,
            initial,
            alpha,
            expected_log_beta,
            steps=steps,
            tolerance=None,
        )
        current = _local_document_elbo(batch, gamma, alpha, expected_log_beta)
        assert bool(torch.all(current >= previous - 1e-5))
        previous = current


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


def test_unrolled_refinement_backpropagates_only_to_encoder() -> None:
    model = prepared_model()
    assert model._matrix is not None and model._core is not None
    core = model._core
    batch = _make_sparse_batch(model._matrix, [0, 1, 2], device=model.device)
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior.detach().clone())
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1).detach()
    core.zero_grad(set_to_none=True)
    gamma_zero = core.encode(
        batch,
        model._embedding_batch(model.docs, [0, 1, 2]),
        word_topic,
    )
    gamma_refined, _ = _local_vb(
        batch,
        gamma_zero,
        core.alpha,
        expected_log_beta,
        steps=2,
        tolerance=None,
    )
    loss = -_local_document_elbo(
        batch,
        gamma_refined,
        core.alpha,
        expected_log_beta,
    ).mean()
    loss.backward()

    final_gradient = core.encoder[-1].weight.grad
    assert final_gradient is not None
    assert bool(torch.all(torch.isfinite(final_gradient)))
    assert float(final_gradient.norm()) > 1e-8
    assert all(parameter.grad is None for parameter in core.prior_parameters())


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


def test_global_update_rejects_nonfinite_expected_counts() -> None:
    model = prepared_model()
    assert model._core is not None and model._matrix is not None
    core = model._core
    prior = core.structured_prior(float(model._matrix.sum()), epoch=1)
    statistics = torch.zeros_like(core.lambda_posterior)
    statistics[0, 0] = torch.nan

    with pytest.raises(ValueError, match="expected topic-word counts"):
        core.update_topics(statistics, prior)


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
    config = replace(
        small_config(),
        num_topics=np.int64(3),
        embedding_dim=np.int64(4),
        alpha=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        eta=np.float32(0.01),
        seed=np.uint64(7),
    )
    model = prepared_model(config)
    model.train(2)
    model.finalize_inference()
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
    assert restored._prior_optimizer is None
    assert restored.inference_finalized
    assert type(restored.config.num_topics) is int
    assert type(restored.config.eta) is float
    assert type(restored.config.seed) is int
    assert isinstance(restored.config.alpha, tuple)
    assert all(type(value) is float for value in restored.config.alpha)
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
    model.finalize_inference()
    words, embeddings, _ = documents()
    first = model.make_doc(words[0], embedding=embeddings[0])
    second = model.make_doc(words[0], embedding=embeddings[4])

    first_theta, _ = model.infer(first, iter=0)
    second_theta, _ = model.infer(second, iter=0)

    assert not np.allclose(first_theta, second_theta)


def test_post_discovery_inference_training_cannot_change_topics() -> None:
    model = prepared_model()
    model.train(2)
    assert model._core is not None
    core = model._core
    topics_before = core.lambda_posterior.detach().clone()
    prior_before = [parameter.detach().clone() for parameter in core.prior_parameters()]
    encoder_before = [
        parameter.detach().clone() for parameter in core.encoder_parameters()
    ]

    history = model.finalize_inference()

    assert len(history) == 2
    assert all(np.isfinite(list(metrics.values())).all() for metrics in history)
    assert set(history[0]) == {
        "inference_epoch",
        "loss",
        "refined_elbo_per_token",
        "zero_step_elbo_per_token",
        "encoder_gradient_norm",
    }
    torch.testing.assert_close(core.lambda_posterior, topics_before, rtol=0, atol=0)
    for current, previous in zip(core.prior_parameters(), prior_before, strict=True):
        torch.testing.assert_close(current, previous, rtol=0, atol=0)
    assert any(
        not torch.equal(current, previous)
        for current, previous in zip(
            core.encoder_parameters(),
            encoder_before,
            strict=True,
        )
    )


def test_finalization_rejects_untrained_model_without_preparing_it() -> None:
    words, embeddings, _ = documents()
    model = HybridLDAModel(small_config())
    model.add_doc(words[0], embedding=embeddings[0])

    with pytest.raises(RuntimeError, match="topic-discovery epoch"):
        model.finalize_inference()

    assert model._core is None
    model.add_doc(words[1], embedding=embeddings[1])
    assert len(model.docs) == 2


def test_discovery_never_updates_document_encoder() -> None:
    model = prepared_model(seed=17)
    assert model._core is not None
    before = [
        parameter.detach().clone() for parameter in model._core.encoder_parameters()
    ]

    model.train(2)

    for current, expected in zip(
        model._core.encoder_parameters(),
        before,
        strict=True,
    ):
        torch.testing.assert_close(current, expected, rtol=0, atol=0)


def test_finalization_is_required_and_permanently_freezes_discovery(
    tmp_path: Path,
) -> None:
    model = prepared_model(seed=19)
    model.train(1)
    words, embeddings, _ = documents()

    with pytest.raises(RuntimeError, match="finalize"):
        model.make_doc(words[0], embedding=embeddings[0])
    with pytest.raises(RuntimeError, match="finalize"):
        model.save(tmp_path / "unfinished.bin")

    model.finalize_inference()
    assert model.inference_finalized
    with pytest.raises(RuntimeError, match="cannot resume"):
        model.train(1)
    with pytest.raises(RuntimeError, match="already"):
        model.finalize_inference()


def test_same_seed_reproduces_finalized_encoder() -> None:
    first = prepared_model(seed=23)
    second = prepared_model(seed=23)
    first.train(2)
    second.train(2)

    first_history = first.finalize_inference()
    second_history = second.finalize_inference()

    assert first_history == second_history
    assert first._core is not None and second._core is not None
    for left, right in zip(
        first._core.encoder_parameters(),
        second._core.encoder_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_finalization_failure_rolls_back_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = prepared_model(seed=27)
    control = prepared_model(seed=27)
    failed.train(1)
    control.train(1)
    assert failed._core is not None and control._core is not None
    encoder_before = [
        parameter.detach().clone() for parameter in failed._core.encoder_parameters()
    ]
    module_mode_before = failed._core.training
    original_step = torch.optim.Adam.step

    def fail_after_update(
        optimizer: torch.optim.Adam,
        closure=None,
    ) -> None:
        original_step(optimizer, closure=closure)
        raise FloatingPointError("intentional optimizer failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", fail_after_update)
        with pytest.raises(FloatingPointError, match="intentional"):
            failed.finalize_inference()

    assert not failed.inference_finalized
    assert failed._core.training is module_mode_before
    for current, expected in zip(
        failed._core.encoder_parameters(),
        encoder_before,
        strict=True,
    ):
        torch.testing.assert_close(current, expected, rtol=0, atol=0)

    failed_history = failed.finalize_inference()
    control_history = control.finalize_inference()
    assert failed_history == control_history
    for current, expected in zip(
        failed._core.encoder_parameters(),
        control._core.encoder_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(current, expected, rtol=0, atol=0)


def test_benchmark_regression_baseline_handles_multiple_minibatches() -> None:
    model = prepared_model(seed=29)
    model.train(1)
    assert len(model.docs) > model.config.batch_size
    assert model._core is not None
    topics_before = model._core.lambda_posterior.detach().clone()

    history = fit_posterior_regression_baseline(
        model,
        epochs=2,
        target_steps=1,
    )

    assert len(history) == 2
    assert np.isfinite(list(history[0].values())).all()
    assert not model.inference_finalized
    torch.testing.assert_close(
        model._core.lambda_posterior,
        topics_before,
        rtol=0,
        atol=0,
    )


def test_synthetic_benchmark_reference_includes_every_inference_basin() -> None:
    run = run_seed(
        SimpleNamespace(
            train_documents=24,
            test_documents=12,
            tokens_per_document=12,
            encoder_epochs=2,
        ),
        seed=13,
    )

    for method in METHODS:
        report = run["methods"][method]
        assert set(report["reference_source_counts"]) == {"symmetric", *METHODS}
        assert sum(report["reference_source_counts"].values()) == 12
        assert all(
            budget["elbo_gap_per_token"] >= -1e-6
            for budget in report["budgets"].values()
        )


def test_adaptive_inference_can_stop_after_one_exact_update() -> None:
    model = prepared_model()
    model.train(2)
    model.finalize_inference()
    words, embeddings, _ = documents()
    fixed = model.make_doc(words[0], embedding=embeddings[0])
    adaptive = model.make_doc(words[0], embedding=embeddings[0])

    fixed_theta, fixed_ll = model.infer(fixed, iter=1)
    adaptive_theta, adaptive_ll = model.infer(
        adaptive,
        iter=50,
        tolerance=1e9,
    )

    np.testing.assert_array_equal(adaptive_theta, fixed_theta)
    assert adaptive_ll == fixed_ll
    with pytest.raises(ValueError, match="tolerance"):
        model.infer(adaptive, iter=5, tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        model.infer(adaptive, iter=5, tolerance=np.nan)


def test_training_document_refresh_does_not_use_encoder_or_dreams_embedding() -> None:
    model = prepared_model()
    model.train(2)
    assert model._core is not None
    model._refresh_training_documents()
    expected = document_topics(model)
    with torch.no_grad():
        for parameter in model._core.encoder_parameters():
            parameter.add_(torch.randn_like(parameter) * 100.0)
    for document in model.docs:
        document.embedding[:] = np.random.default_rng(123).normal(
            size=document.embedding.shape
        )
    model._refresh_training_documents()

    np.testing.assert_array_equal(document_topics(model), expected)


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


def test_spectrum_arrays_rejects_all_zero_intensities() -> None:
    spectrum = SimpleNamespace(
        peaks=SimpleNamespace(
            mz=np.asarray([100.0, 150.0], dtype=np.float32),
            intensities=np.zeros(2, dtype=np.float32),
        ),
        metadata={"precursor_mz": 200.0},
    )

    with pytest.raises(ValueError, match="positive intensity"):
        spectrum_arrays(spectrum)


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


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -np.inf, 0.0])
def test_peak_pooling_rejects_nonfinite_or_nonpositive_tolerance(
    tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        pool_word_embeddings([[], []], feature_batch(), mz_tolerance=tolerance)


def test_feature_batch_rejects_invalid_peak_metadata() -> None:
    features = feature_batch()
    invalid_peak_mz = features.peak_mz.copy()
    invalid_peak_mz[0, 0] = np.nan

    with pytest.raises(ValueError, match="observed peak"):
        replace(features, peak_mz=invalid_peak_mz)
    with pytest.raises(ValueError, match="precursor"):
        replace(
            features,
            precursor_mz=np.asarray([150.0, np.inf], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="peak_mask"):
        replace(features, peak_mask=features.peak_mask.astype(np.int8))


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
