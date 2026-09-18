"""One validated train/validation input boundary shared by every ETM runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .data import load_csr, load_heldout_records, load_vocabulary
from .etm_baselines import load_sgns_embeddings
from .reproducibility import VALIDATION_DATA_FILES, read_json_object
from .reproduction_audit import verify_linked_inputs

if TYPE_CHECKING:
    import scipy.sparse as sp


class ValidationInputs(NamedTuple):
    """The train and validation artifacts visible to an ETM control."""

    train: sp.csr_matrix
    observed: sp.csr_matrix
    completion: sp.csr_matrix
    full: sp.csr_matrix
    records: list[dict[str, Any]]
    vocabulary: list[str]
    embeddings: np.ndarray
    protocol: dict[str, Any]
    input_manifest: dict[str, Any]


def load_validation_inputs(run: Path) -> ValidationInputs:
    """Load only the sealed training and validation view."""
    data = run / "data"
    protocol = read_json_object(run / "protocol.json")
    input_manifest = read_json_object(run / "validation_input_manifest.json")
    if input_manifest.get("candidate_test_artifacts_accessed") is not False:
        raise RuntimeError("validation view does not preserve the test boundary")
    forbidden = sorted(path.name for path in data.glob("test*") if path.is_file())
    if forbidden:
        raise RuntimeError(f"sealed training view exposes test files: {forbidden}")
    # A manifest flag alone is not proof that the bytes still match the seal.
    # Check only inputs consumed by training/inference, not the large MAG index.
    required = {
        run.absolute() / "protocol.json",
        run.absolute() / "token_features/features.npy",
        *(run.absolute() / "data" / name for name in VALIDATION_DATA_FILES),
    }
    declared = {
        Path(row["linked_path"]).absolute(): row
        for row in input_manifest.get("linked_inputs", [])
    }
    if not required.issubset(declared):
        raise RuntimeError("validation manifest does not seal every required input")
    verify_linked_inputs(
        {"linked_inputs": [declared[path] for path in sorted(required)]},
        field="linked_inputs",
    )
    vocabulary = load_vocabulary(data)
    embeddings = load_sgns_embeddings(run / "token_features/features.npy")
    train = load_csr(data / "train.npz")
    observed = load_csr(data / "validation_observed.npz")
    completion = load_csr(data / "validation_completion.npz")
    full = load_csr(data / "validation_full.npz")
    records = load_heldout_records(data, "validation")
    if train.shape[1] != len(vocabulary) or embeddings.shape[0] != len(vocabulary):
        raise ValueError("train matrix, vocabulary and SGNS features do not align")
    if observed.shape != completion.shape or observed.shape != full.shape:
        raise ValueError("validation matrices do not have identical shapes")
    if full.shape[1] != len(vocabulary):
        raise ValueError("validation matrices and vocabulary do not align")
    if full.shape[0] != len(records):
        raise ValueError("validation records and matrices do not align")
    return ValidationInputs(
        train=train,
        observed=observed,
        completion=completion,
        full=full,
        records=records,
        vocabulary=vocabulary,
        embeddings=embeddings,
        protocol=protocol,
        input_manifest=input_manifest,
    )
