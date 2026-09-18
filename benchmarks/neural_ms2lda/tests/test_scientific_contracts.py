"""Cross-cutting data, chemistry, and manuscript consistency tests."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.neural_ms2lda.chemical import _associated_record_indices, _sos_bands
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


def test_research_commands_are_excluded_from_the_production_package() -> None:
    """Only the production CLI belongs in the installed scripts namespace."""
    import tomllib

    root = Path(__file__).parents[3]
    configuration = tomllib.loads((root / "pyproject.toml").read_text())
    research_commands = {
        str(path.relative_to(root))
        for path in (root / "scripts").glob("*.py")
        if path.name not in {"__init__.py", "ms2lda_runfull.py"}
    }
    assert research_commands <= set(configuration["tool"]["poetry"]["exclude"])


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


def test_archived_report_constants_match_the_frozen_executable_model() -> None:
    review_directory = (
        Path(__file__).parents[3] / "research/minimal_neural_etm/review_20260907"
    )
    report = (review_directory / "model_form_review_report_20260908.tex").read_text(
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


def test_current_report_selects_whole_spectrum_context_and_tomotopy_comparison() -> (
    None
):
    directory = Path(__file__).parents[3] / "docs/research"
    report = (directory / "contextual_sparse_etm_report.tex").read_text(
        encoding="utf-8"
    )
    supplement = (directory / "contextual_sparse_etm_supplement.tex").read_text(
        encoding="utf-8"
    )
    # The main paper keeps the base ETM and points to full enhanced equations.
    for label in ("eq:base-beta", "eq:base-generator", "eq:base-posterior"):
        assert rf"\label{{{label}}}" in report
    assert r"\frac12" in supplement
    assert "unit-temperature local softmax" in supplement
    assert "two topics with largest cosine scores" in supplement
    assert r"s_d=\sum_w x_{dw}\widehat\rho_w" in supplement
    assert r"\newcommand{\ctxscale}{\lambda_{\mathrm{ctx}}}" in supplement
    assert r"h_{dw}=\normalize(\widehat\rho_w+\ctxscale\,s_d)" in supplement
    assert r"r_{dk}=\sum_w x_{dw}\pi_{dwk}" in supplement
    assert r"\log(1+Kr_{dk})-\frac1K" in supplement
    assert r"\entmax" in supplement and r"\label{eq:elbo}" in supplement
    assert r"\label{eq:kl}" in supplement
    assert r"\texttt{reduced\_document\_context}" in supplement
    assert "Primary comparison: Tomotopy" in supplement
    supplementary_references = set(re.findall(r"\\ref\{supp-([^}]+)\}", report))
    assert {
        "sec:balance",
        "sec:context",
        "sec:sparsity",
        "sec:objective",
        "sec:comparators",
        "sec:metrics",
        "sec:overlap-methods",
        "app:repro",
    } <= supplementary_references
    assert supplementary_references <= set(
        re.findall(r"\\label\{([^}]+)\}", supplement)
    )
    assert r"\externaldocument[supp-]{contextual_sparse_etm_supplement}" in report
    assert r"\externaldocument[main-]{contextual_sparse_etm_report}" in supplement
    inputs = re.findall(r"\\input\{([^}]+)\}", report)
    assert set(inputs) == {
        "generated/current_contextual_etm_macros.tex",
        "generated/current_contextual_etm_validation_table.tex",
        "generated/current_contextual_etm_synthetic_table.tex",
        "generated/chemical_assessment_macros.tex",
        "generated/chemical_assessment_table.tex",
        "generated/chemical_assessment_reference_examples.tex",
        "generated/cross_model_overlap_macros.tex",
        "generated/cross_model_overlap_table.tex",
        "figures/contextual_enhancement_examples.tex",
        "figures/spectral_word_construction.tex",
        "figures/selected_model_explanation.tex",
        "contextual_sparse_etm_references.tex",
    }
    assert len(inputs) == 12
    supplementary_inputs = re.findall(r"\\input\{([^}]+)\}", supplement)
    assert "contextual_sparse_etm_references.tex" in supplementary_inputs
    assert not any("test_table" in name for name in inputs + supplementary_inputs)
    main = report
    assert r"\appendix" not in main
    assert not any(seed in main for seed in ("7012", "7024", "7043"))
    assert r"\label{app:repro}" in supplement
    assert all(seed in supplement for seed in ("7012", "7024", "7043"))
    assert re.findall(r"\\section\{([^}]+)\}", main) == [
        "Introduction",
        "Related work",
        "Materials and methods",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    assert (
        main.index(r"\section{Discussion}")
        < main.index(r"\section{Conclusion}")
        < main.index(r"\section*{Data and code availability}")
    )
    discussion = main.split(r"\section{Discussion}", 1)[1].split(
        r"\section{Conclusion}", 1
    )[0]
    subsections = re.split(r"\\subsection\{([^}]+)\}", discussion)
    assert subsections[1::2] == [
        "Discovery breadth, chemical specificity and predictive fit",
        "Compact mixtures and the ETM foundation",
        "Limitations of the current evidence",
        "Implications for validation and future encoders",
    ]
    assert r"\paragraph" not in discussion
    assert r"\textbf" not in discussion
    # Each subsection develops its argument in multiple full prose paragraphs,
    # instead of relabelling the old one-paragraph, bold-led fragments.
    for body in subsections[2::2]:
        assert len(re.split(r"\n\s*\n", body.strip())) >= 2
    assert main.index(r"\label{sec:metrics}") < main.index(r"\section{Results}")
    assert "Synthetic spectra: controlled recovery" in main
    assert "Lib: comparison with Tomotopy and the ETM base" in main
    assert r"\subsection{Agreement and differences with Tomotopy}" in main
    assert r"\label{eq:directed-recovery}" in supplement
    assert "filtered before matching" in main


def test_current_methods_state_numerical_and_evaluation_conventions() -> None:
    directory = Path(__file__).parents[3] / "docs/research"
    report = (directory / "contextual_sparse_etm_report.tex").read_text(
        encoding="utf-8"
    )
    supplement = (directory / "contextual_sparse_etm_supplement.tex").read_text(
        encoding="utf-8"
    )
    main_methods = report.split(r"\section{Results}")[0]
    assert r"\ref{supp-sec:objective}" in main_methods
    assert r"\ref{supp-sec:metrics}" in main_methods
    methods = " ".join((main_methods + "\n" + supplement).split())
    assert "not a direct pairwise spectral-cosine cutoff" in methods
    assert "complete-linkage Euclidean" in methods
    assert "missing consensus is not evaluable" in methods
    assert "already discretized" in methods
    assert "no observed in-vocabulary words" in methods
    assert r"\varepsilon=10^{-12}" in methods
    for label in ("eq:completion", "eq:effective", "eq:sos"):
        assert rf"\label{{{label}}}" in methods


def test_sos_bands_include_boundaries_exactly_once() -> None:
    bands = _sos_bands([0.0, 0.5999, 0.6, 0.7, 0.8, 0.8001, 1.0])

    assert bands == {
        "high_gt_0_8": 2,
        "intermediate_0_6_to_0_8": 3,
        "low_lt_0_6": 2,
    }
    assert sum(bands.values()) == 7


def test_dominant_topic_assignment_is_one_per_spectrum_and_deterministic() -> None:
    theta = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [2.0, 5.0, 1.0],
            [0.4, 0.4, 0.2],
        ],
        dtype=np.float64,
    )

    associated = _associated_record_indices(theta)

    assert associated == {0: [0, 2], 1: [1]}
    assert sum(len(rows) for rows in associated.values()) == len(theta)


def test_dominant_topic_assignment_rejects_zero_mass_rows() -> None:
    with pytest.raises(ValueError, match="positive probability mass"):
        _associated_record_indices(np.asarray([[0.0, 0.0]]))
