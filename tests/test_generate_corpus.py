"""Contract tests for spectral-word corpus generation."""

import hashlib

import numpy as np
import pytest
from matchms import Spectrum

from MS2LDA.Mass2Motif import Mass2Motif
from MS2LDA.Preprocessing.generate_corpus import (
    combine_features,
    features_to_words,
    map_doc2spec,
)


def _spectrum_with_losses() -> Mass2Motif:
    return Mass2Motif(
        frag_mz=np.asarray([100.004, 200.006]),
        frag_intensities=np.asarray([0.25, 1.0]),
        loss_mz=np.asarray([50.004, 75.006]),
        loss_intensities=np.asarray([0.4, 0.1]),
        metadata={"precursor_mz": 300.0},
    )


def test_dda_uses_independent_fragment_and_loss_pseudocounts() -> None:
    document = features_to_words(
        [_spectrum_with_losses()],
        significant_figures=2,
        acquisition_type="DDA",
    )[0]

    assert document.count("frag@100.0") == 25
    assert document.count("frag@200.01") == 100
    assert document.count("loss@50.0") == 40
    assert document.count("loss@75.01") == 10


def test_dia_excludes_neutral_losses() -> None:
    document = features_to_words(
        [_spectrum_with_losses()],
        acquisition_type="DIA",
    )[0]

    assert document
    assert all(word.startswith("frag@") for word in document)


def test_word_masses_are_rounded_integer_pseudocounts() -> None:
    spectrum = Spectrum(
        mz=np.asarray([100.0, 200.0, 300.0]),
        intensities=np.asarray([0.004, 0.01, 0.505]),
        metadata={"precursor_mz": 400.0},
    )

    document = features_to_words([spectrum], acquisition_type="DIA")[0]

    assert "frag@100.0" not in document
    assert document.count("frag@200.0") == 1
    assert document.count("frag@300.0") == 50


def test_empty_spectrum_produces_an_empty_document() -> None:
    spectrum = Spectrum(
        mz=np.asarray([]),
        intensities=np.asarray([]),
        metadata={"precursor_mz": 200.0},
    )

    assert features_to_words([spectrum], acquisition_type="DDA") == [[]]


def test_empty_dataset_and_invalid_acquisition_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="no spectra"):
        features_to_words([])
    with pytest.raises(ValueError, match="acquisition_type"):
        features_to_words([_spectrum_with_losses()], acquisition_type="unknown")


def test_map_doc2spec_uses_the_document_token_order() -> None:
    documents = [
        ["frag@100.0", "loss@50.0", "loss@50.0"],
        ["frag@200.0"],
    ]
    spectra = [_spectrum_with_losses(), _spectrum_with_losses()]

    mapping = map_doc2spec(documents, spectra)

    first_key = hashlib.md5("".join(documents[0]).encode("utf-8")).hexdigest()
    second_key = hashlib.md5("".join(documents[1]).encode("utf-8")).hexdigest()
    assert mapping[first_key] is spectra[0]
    assert mapping[second_key] is spectra[1]


def test_map_doc2spec_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        map_doc2spec([["frag@100.0"]], [])


def test_combine_features_preserves_channel_and_spectrum_order() -> None:
    fragments = [["frag@100.0", "frag@200.0"], []]
    losses = [["loss@50.0"], ["loss@75.0"]]

    assert combine_features(fragments, losses) == [
        ["frag@100.0", "frag@200.0", "loss@50.0"],
        ["loss@75.0"],
    ]


def test_combine_features_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        combine_features([["frag@100.0"]], [[], []])
