"""Tests for the paired fragment/loss truth-known simulator."""

from __future__ import annotations

import numpy as np

from benchmarks.neural_ms2lda.synthetic_msms import generate_synthetic_msms


def test_synthetic_msms_is_deterministic_and_train_only() -> None:
    first = generate_synthetic_msms(
        seed=11,
        true_topics=6,
        training_documents=32,
        validation_documents=8,
        minimum_document_frequency=2,
    )
    second = generate_synthetic_msms(
        seed=11,
        true_topics=6,
        training_documents=32,
        validation_documents=8,
        minimum_document_frequency=2,
    )
    assert first.vocabulary == second.vocabulary
    assert (first.train != second.train).nnz == 0
    assert (first.validation_observed != second.validation_observed).nnz == 0
    assert np.array_equal(first.true_beta, second.true_beta)
    assert np.array_equal(first.validation_true_theta, second.validation_true_theta)


def test_synthetic_msms_preserves_probability_and_completion_contracts() -> None:
    dataset = generate_synthetic_msms(
        seed=23,
        true_topics=6,
        training_documents=40,
        validation_documents=10,
        minimum_document_frequency=2,
    )
    assert dataset.train.shape == (40, len(dataset.vocabulary))
    assert dataset.validation_full.shape == (10, len(dataset.vocabulary))
    assert np.allclose(dataset.true_beta.sum(axis=1), 1.0)
    assert np.allclose(dataset.train_true_theta.sum(axis=1), 1.0)
    assert np.allclose(dataset.validation_true_theta.sum(axis=1), 1.0)
    assert all(word.startswith(("frag@", "loss@")) for word in dataset.vocabulary)
    assert all(record["split"] == "validation" for record in dataset.validation_records)
    assert np.all(
        np.asarray(dataset.validation_observed.sum(axis=1)).ravel() > 0,
    )
    assert np.all(
        np.asarray(dataset.validation_completion.sum(axis=1)).ravel() > 0,
    )
