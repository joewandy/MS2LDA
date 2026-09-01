"""Cross-cutting data, chemistry, and manuscript consistency tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.neural_ms2lda.chemical import _sos_bands
from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    CONTEXT_TEMPERATURE,
    FRAGMENT_CHANNEL_MASS,
    TOPICS_PER_TOKEN,
)
from benchmarks.neural_ms2lda.data import build_token_features
from benchmarks.neural_ms2lda.spectra import (
    SpectrumRecord,
    audit_split_disjointness,
    build_training_vocabulary,
)
from scripts.generate_contextual_sparse_etm_report import _generate_code_map

from ._support import spectrum_record


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


def test_report_constants_match_the_executable_model() -> None:
    research_directory = Path(__file__).parents[3] / "docs/research"
    report = (research_directory / "contextual_sparse_etm_report.tex").read_text(
        encoding="utf-8"
    )
    _, code_map = _generate_code_map()

    assert FRAGMENT_CHANNEL_MASS == 0.5
    assert CONTEXT_TEMPERATURE == 1.0
    assert TOPICS_PER_TOKEN == 2
    assert r"\tfrac12" in report
    assert r"\tau_c=1.0" in report
    assert r"\tfrac1K" in report
    assert r"\centerop" in report
    assert r"\entmax" in report
    assert r"\label{eq:reconstruction}" in report
    assert r"\label{eq:kl}" in report
    assert "two largest scores" in report
    assert r"contextual\_sparse\_etm.py" in code_map
    assert r"topic\_model\_training.py" in code_map
    assert r"Reconstruction loss (Eq.~\ref{eq:reconstruction})" in code_map
    assert r"Gaussian posterior and KL (Eqs.~\ref{eq:encoder}, \ref{eq:kl})" in code_map


def test_sos_bands_include_boundaries_exactly_once() -> None:
    bands = _sos_bands([0.0, 0.5999, 0.6, 0.7, 0.8, 0.8001, 1.0])

    assert bands == {
        "high_gt_0_8": 2,
        "intermediate_0_6_to_0_8": 3,
        "low_lt_0_6": 2,
    }
    assert sum(bands.values()) == 7
