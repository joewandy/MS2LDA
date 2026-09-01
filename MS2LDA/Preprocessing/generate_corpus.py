import hashlib
from itertools import chain

import numpy as np


def map_doc2spec(feature_words, spectra):
    """Map each generated document hash to its source spectrum.

    ``feature_words`` contains one ordered word-token list per spectrum. The
    same token concatenation is used by :func:`MS2LDA.utils.retrieve_spec4doc`.
    A length mismatch is an error because truncation would corrupt that
    correspondence.
    """
    documents = list(feature_words)
    source_spectra = list(spectra)
    if len(documents) != len(source_spectra):
        raise ValueError("feature documents and spectra must have equal lengths")

    return {
        hashlib.md5("".join(document).encode("utf-8")).hexdigest(): spectrum
        for document, spectrum in zip(documents, source_spectra)
    }


def features_to_words(spectra, significant_figures=2, acquisition_type="DDA"):
    """Convert normalized spectra to intensity-weighted word documents.

    Fragment and neutral-loss intensities are independently rounded to integer
    pseudo-counts on a 0--100 scale. DDA documents include both channels; DIA
    documents include fragments only.
    """
    if acquisition_type not in {"DDA", "DIA"}:
        raise ValueError("acquisition_type must be 'DDA' or 'DIA'")

    dataset_frag = []
    dataset_loss = []

    for spectrum in spectra:
        fragment_counts = np.rint(spectrum.peaks.intensities * 100).astype(int)

        frag_with_n_digits = [
            ["frag@" + str(round(mz, significant_figures))] for mz in spectrum.peaks.mz
        ]
        frag_multiplied_intensities = [
            fragment * count
            for fragment, count in zip(frag_with_n_digits, fragment_counts)
        ]
        dataset_frag.append(list(chain.from_iterable(frag_multiplied_intensities)))

        if acquisition_type == "DIA":
            continue

        loss_counts = np.rint(spectrum.losses.intensities * 100).astype(int)
        loss_with_n_digits = [
            ["loss@" + str(round(mz, significant_figures))] for mz in spectrum.losses.mz
        ]
        loss_multiplied_intensities = [
            loss * count for loss, count in zip(loss_with_n_digits, loss_counts)
        ]
        flattened_losses = chain.from_iterable(loss_multiplied_intensities)
        dataset_loss.append(
            [loss for loss in flattened_losses if float(loss[5:]) > 0.01]
        )

    if not dataset_frag:
        raise ValueError("no spectra were supplied; no vocabulary was generated")
    if acquisition_type == "DIA":
        return dataset_frag
    return combine_features(dataset_frag, dataset_loss)


def combine_features(dataset_frag, dataset_loss):
    """Combine corresponding fragment and neutral-loss word lists."""
    if len(dataset_frag) != len(dataset_loss):
        raise ValueError("fragment and loss datasets must have equal lengths")
    return [
        spectrum_frag + spectrum_loss
        for spectrum_frag, spectrum_loss in zip(dataset_frag, dataset_loss)
    ]
