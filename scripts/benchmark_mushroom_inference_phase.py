#!/usr/bin/env python
"""Validate post-hoc semi-amortized inference on the frozen mushroom model.

This is deliberately a one-off migration benchmark for trusted historical
artifacts.  It reconstructs the exact random document-completion split used by
the earlier mushroom benchmark, migrates a legacy hybrid checkpoint into the
current reference implementation, and compares only document-inference
objectives while keeping every topic and structured-prior value frozen.

The three arms are:

* the historical encoder without further training;
* equal-epoch posterior distillation against the final frozen topics; and
* a fixed differentiable objective consisting of the two-step local ELBO plus
  0.1 times the encoder-only local ELBO.

The test DreaMS cache must have been extracted from observed physical peaks
only.  Held-out words are used solely for posterior-mean predictive NLL.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MS2LDA.dreams_features import spectrum_arrays  # noqa: E402
from MS2LDA.hybrid_lda import (  # noqa: E402
    EPSILON,
    HybridLDAConfig,
    HybridLDAModel,
    _expected_log_dirichlet,
    _local_document_elbo,
    _local_vb,
    _make_sparse_batch,
    observed_token_nll,
)

BUDGETS = (0, 1, 2, 5, 20)
LEGACY_FORMAT = "ms2lda-hybrid-lda"
LEGACY_VERSION = 2
ENCODER_PREFIXES = ("encoder.", "document_projector.")


@dataclass(frozen=True)
class ReconstructedSplit:
    """Exact train/test data reconstructed from historical artifacts."""

    vocabulary: list[str]
    train_matrix: sp.csr_matrix
    observed_matrix: sp.csr_matrix
    heldout_matrix: sp.csr_matrix
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_identifiers: tuple[str, ...]
    test_identifiers: tuple[str, ...]
    observed_words: list[list[str]]
    observed_embeddings: np.ndarray
    manifest: dict[str, Any]
    completion_metadata: dict[str, Any]
    cache_metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse the explicit artifact and experiment configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare post-hoc inference objectives on the frozen random-split "
            "mushroom benchmark. Historical checkpoint and spectrum-map inputs "
            "are unsafe serialization formats and require explicit trust."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--legacy-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--spectra-map", required=True, type=Path)
    parser.add_argument("--observed-features", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument(
        "--legacy-run-summary",
        type=Path,
        help="Optional historical run_summary.json; defaults beside checkpoint.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trust-legacy-artifacts",
        action="store_true",
        help=(
            "Acknowledge that the checkpoint and spectrum map are trusted local "
            "artifacts before torch.load(weights_only=False) and pickle.load."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--refinement-steps", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int)
    parser.add_argument("--reference-steps", type=int, default=100)
    parser.add_argument("--reference-tolerance", type=float, default=1e-7)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--observed-fraction", type=float, default=0.5)
    parser.add_argument("--mz-tolerance", type=float, default=0.02)
    parser.add_argument("--legacy-nll-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def progress(message: str) -> None:
    """Write concise progress without contaminating a possible JSON stream."""
    print(message, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    """Hash indices using the historical little-endian int64 encoding."""
    normalized = np.asarray(values, dtype="<i8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def hash_sparse(matrix: sp.csr_matrix) -> str:
    """Hash CSR arrays and shape using the historical benchmark convention."""
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensors with shape and dtype metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    """Raise a useful provenance error on an exact mismatch."""
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: got {actual!r}, expected {expected!r}")


def tomotopy_to_csr(model: Any) -> tuple[sp.csr_matrix, list[str]]:
    """Recover the exact retained count corpus from a Tomotopy model."""
    vocabulary = list(model.used_vocabs)
    word_index = {word: index for index, word in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, document in enumerate(model.docs):
        counts: dict[int, float] = {}
        for raw_word in document.words:
            word = model.vocabs[int(raw_word)]
            column = word_index.get(word)
            if column is not None:
                counts[column] = counts.get(column, 0.0) + 1.0
        for column, count in counts.items():
            rows.append(row)
            columns.append(column)
            values.append(count)
    return (
        sp.csr_matrix(
            (values, (rows, columns)),
            shape=(len(model.docs), len(vocabulary)),
            dtype=np.float32,
        ),
        vocabulary,
    )


def split_documents(
    num_documents: int,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the deterministic historical random split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    shuffled = np.random.default_rng(seed).permutation(num_documents)
    train_size = int(round(train_fraction * num_documents))
    return np.sort(shuffled[:train_size]), np.sort(shuffled[train_size:])


def restrict_to_training_vocabulary(
    train: sp.csr_matrix,
    test: sp.csr_matrix,
    vocabulary: list[str],
) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
    """Remove words absent from training without inspecting test frequency."""
    active = np.asarray(train.sum(axis=0)).ravel() > 0
    retained = [word for word, keep in zip(vocabulary, active, strict=True) if keep]
    return (
        train[:, active].tocsr().astype(np.float32),
        test[:, active].tocsr().astype(np.float32),
        retained,
    )


def load_aligned_spectra(
    path: Path,
    *,
    expected_documents: int,
) -> tuple[list[Any], list[str]]:
    """Load a trusted historical document-to-spectrum mapping in order."""
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - guarded by explicit CLI trust
    if isinstance(value, dict):
        spectra = list(value.values())
    elif isinstance(value, (list, tuple)):
        spectra = list(value)
    else:
        raise TypeError("spectra-map must contain a dictionary or spectrum sequence")
    if len(spectra) != expected_documents:
        raise ValueError(
            f"spectra-map contains {len(spectra)} rows; expected {expected_documents}"
        )
    identifiers: list[str] = []
    for index, spectrum in enumerate(spectra):
        getter = getattr(spectrum, "get", None)
        identifier = getter("id", None) if callable(getter) else None
        identifiers.append(str(identifier or f"spec_{index}"))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("spectrum identifiers are not unique")
    return spectra, identifiers


def _word_target_mz(word: str, precursor_mz: float) -> float | None:
    """Map a fragment or neutral-loss token back to a physical peak mass."""
    prefix, separator, raw_value = word.partition("@")
    if not separator or prefix not in {"frag", "loss"}:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if prefix == "frag" else precursor_mz - value


def _physical_word_groups(
    columns: np.ndarray,
    counts: np.ndarray,
    spectrum: Any,
    vocabulary: list[str],
    *,
    mz_tolerance: float,
) -> tuple[dict[tuple[str, int], list[tuple[int, float]]], int, int]:
    """Group fragment/loss words arising from one physical spectrum peak."""
    mz_values, _, precursor = spectrum_arrays(spectrum)
    groups: dict[tuple[str, int], list[tuple[int, float]]] = {}
    mapped_words = 0
    unmapped_words = 0
    for column, count in zip(columns, counts, strict=True):
        target_mz = _word_target_mz(vocabulary[int(column)], precursor)
        group: tuple[str, int]
        if target_mz is not None and target_mz > 0:
            peak_index = int(np.argmin(np.abs(mz_values - target_mz)))
            if abs(float(mz_values[peak_index]) - target_mz) <= mz_tolerance:
                group = ("peak", peak_index)
                mapped_words += 1
            else:
                group = ("word", int(column))
                unmapped_words += 1
        else:
            group = ("word", int(column))
            unmapped_words += 1
        groups.setdefault(group, []).append((int(column), float(count)))
    return groups, mapped_words, unmapped_words


def completion_eligible_rows(
    matrix: sp.csr_matrix,
    spectra: list[Any],
    vocabulary: list[str],
    *,
    mz_tolerance: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Find spectra supporting a leak-free two-sided physical-peak split."""
    if matrix.shape[0] != len(spectra):
        raise ValueError("test matrix and spectra are not row aligned")
    matrix = matrix.tocsr()
    eligible: list[int] = []
    reasons = {
        "fewer_than_two_feature_groups": 0,
        "no_mapped_physical_peak": 0,
    }
    for row, spectrum in enumerate(spectra):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        groups, _, _ = _physical_word_groups(
            matrix.indices[start:end],
            matrix.data[start:end],
            spectrum,
            vocabulary,
            mz_tolerance=mz_tolerance,
        )
        if len(groups) < 2:
            reasons["fewer_than_two_feature_groups"] += 1
        elif not any(group[0] == "peak" for group in groups):
            reasons["no_mapped_physical_peak"] += 1
        else:
            eligible.append(row)
    return np.asarray(eligible, dtype=np.int64), reasons


def paired_completion_split(  # noqa: PLR0913
    matrix: sp.csr_matrix,
    spectra: list[Any],
    vocabulary: list[str],
    *,
    observed_fraction: float,
    seed: int,
    mz_tolerance: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, Any]]:
    """Recreate the physical-peak completion assignment exactly."""
    if not 0 < observed_fraction < 1:
        raise ValueError("observed_fraction must lie strictly between zero and one")
    if matrix.shape[0] != len(spectra):
        raise ValueError("test matrix and spectra are not row aligned")
    matrix = matrix.tocsr()
    observed = sp.lil_matrix(matrix.shape, dtype=np.float32)
    heldout = sp.lil_matrix(matrix.shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    assignment_hash = hashlib.sha256()
    mapped_words = 0
    unmapped_words = 0

    for row, spectrum in enumerate(spectra):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        groups, mapped, unmapped = _physical_word_groups(
            matrix.indices[start:end],
            matrix.data[start:end],
            spectrum,
            vocabulary,
            mz_tolerance=mz_tolerance,
        )
        mapped_words += mapped
        unmapped_words += unmapped
        group_keys = list(groups)
        if len(group_keys) < 2:
            raise ValueError(f"test row {row} has fewer than two feature groups")
        shuffled = rng.permutation(len(group_keys))
        observed_count = int(round(observed_fraction * len(group_keys)))
        observed_count = min(max(observed_count, 1), len(group_keys) - 1)
        selected_groups = {
            group_keys[int(index)] for index in shuffled[:observed_count]
        }
        selected_peaks_list: list[int] = []
        for group, entries in groups.items():
            destination = observed if group in selected_groups else heldout
            for column, count in entries:
                destination[row, column] = count
            if group in selected_groups and group[0] == "peak":
                selected_peaks_list.append(group[1])
        selected_peaks = np.asarray(sorted(selected_peaks_list), dtype=np.int64)
        if not len(selected_peaks):
            physical_groups = [group for group in group_keys if group[0] == "peak"]
            if not physical_groups:
                raise ValueError(f"test row {row} has no mapped physical peak")
            moved = physical_groups[0]
            for column, count in groups[moved]:
                heldout[row, column] = 0
                observed[row, column] = count
            selected_peaks = np.asarray([moved[1]], dtype=np.int64)
        assignment_hash.update(np.asarray(selected_peaks, dtype="<i8").tobytes())

    observed_csr = observed.tocsr()
    heldout_csr = heldout.tocsr()
    observed_csr.eliminate_zeros()
    heldout_csr.eliminate_zeros()
    difference = (observed_csr + heldout_csr - matrix).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        raise RuntimeError("physical-peak completion did not preserve counts")
    return (
        observed_csr,
        heldout_csr,
        {
            "assignment_sha256": assignment_hash.hexdigest(),
            "mapped_word_columns": mapped_words,
            "unmapped_word_columns": unmapped_words,
            "mz_tolerance": mz_tolerance,
        },
    )


def csr_to_documents(
    matrix: sp.csr_matrix,
    vocabulary: list[str],
) -> list[list[str]]:
    """Expand the integer count matrix into the model's token-list format."""
    matrix = matrix.tocsr()
    documents: list[list[str]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        words: list[str] = []
        for column, raw_count in zip(
            matrix.indices[start:end], matrix.data[start:end], strict=True
        ):
            count = int(round(float(raw_count)))
            if not np.isclose(raw_count, count):
                raise ValueError("mushroom corpus contains a non-integer count")
            words.extend([vocabulary[int(column)]] * count)
        documents.append(words)
    return documents


def read_observed_cache(
    path: Path,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    """Read only IDs and global embeddings from the partial-spectrum cache."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("the real-data benchmark requires h5py") from exc
    with h5py.File(path, "r") as handle:
        identifiers = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["identifiers"][:]
        )
        embeddings = np.asarray(handle["spectrum_embeddings"][:], dtype=np.float32)
        raw_metadata = handle.attrs.get(
            "metadata_json", handle.attrs.get("provenance_json", "{}")
        )
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode()
    metadata = json.loads(str(raw_metadata))
    if embeddings.shape[0] != len(identifiers) or embeddings.ndim != 2:
        raise ValueError("observed cache IDs and spectrum embeddings are misaligned")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("observed cache contains non-finite spectrum embeddings")
    return identifiers, embeddings, metadata


def reconstruct_split(args: argparse.Namespace) -> ReconstructedSplit:
    """Rebuild and hash-check the exact random physical-peak split."""
    import tomotopy as tp

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    _require_equal("split mode", manifest.get("mode"), "random")
    split_seed = int(manifest["split_seed"])
    completion_seed = int(manifest["completion_seed"])
    _require_equal(
        "completion mode",
        manifest.get("completion", {}).get("mode"),
        "physical_peak_pairs",
    )
    _require_equal(
        "completion m/z tolerance",
        float(manifest["completion"]["mz_tolerance"]),
        float(args.mz_tolerance),
    )

    reference = tp.LDAModel.load(str(args.reference_model))
    full_matrix, full_vocabulary = tomotopy_to_csr(reference)
    spectra, spectrum_identifiers = load_aligned_spectra(
        args.spectra_map,
        expected_documents=full_matrix.shape[0],
    )
    train_indices, assigned_test_indices = split_documents(
        full_matrix.shape[0],
        train_fraction=args.train_fraction,
        seed=split_seed,
    )
    train, test, vocabulary = restrict_to_training_vocabulary(
        full_matrix[train_indices],
        full_matrix[assigned_test_indices],
        full_vocabulary,
    )
    selected_spectra = [spectra[index] for index in assigned_test_indices]
    eligible_rows, exclusion_reasons = completion_eligible_rows(
        test,
        selected_spectra,
        vocabulary,
        mz_tolerance=args.mz_tolerance,
    )
    if not len(eligible_rows):
        raise RuntimeError("no test spectra support physical-peak completion")
    test_indices = assigned_test_indices[eligible_rows]
    test = test[eligible_rows]
    selected_spectra = [selected_spectra[int(index)] for index in eligible_rows]
    observed, heldout, completion_metadata = paired_completion_split(
        test,
        selected_spectra,
        vocabulary,
        observed_fraction=args.observed_fraction,
        seed=completion_seed,
        mz_tolerance=args.mz_tolerance,
    )
    completion_metadata.update(
        {
            "mode": "physical_peak_pairs",
            "excluded_documents": int(len(assigned_test_indices) - len(test_indices)),
            "exclusion_reasons": exclusion_reasons,
        }
    )

    actual_hashes = {
        "train_indices_sha256": hash_array(train_indices),
        "assigned_test_indices_sha256": hash_array(assigned_test_indices),
        "test_indices_sha256": hash_array(test_indices),
        "train_matrix_sha256": hash_sparse(train),
        "test_observed_matrix_sha256": hash_sparse(observed),
        "test_heldout_matrix_sha256": hash_sparse(heldout),
    }
    for key, actual in actual_hashes.items():
        _require_equal(key, actual, manifest[key])
    _require_equal(
        "completion assignment",
        completion_metadata["assignment_sha256"],
        manifest["completion"]["assignment_sha256"],
    )
    for key in (
        "mapped_word_columns",
        "unmapped_word_columns",
        "excluded_documents",
        "exclusion_reasons",
    ):
        _require_equal(
            f"completion {key}",
            completion_metadata[key],
            manifest["completion"][key],
        )
    scalar_checks = {
        "train_documents": train.shape[0],
        "test_documents": observed.shape[0],
        "assigned_test_documents": len(assigned_test_indices),
        "vocab_size": train.shape[1],
        "train_tokens": int(train.sum()),
        "test_observed_tokens": int(observed.sum()),
        "test_heldout_tokens": int(heldout.sum()),
    }
    for key, actual in scalar_checks.items():
        _require_equal(key, actual, manifest[key])
    original_test = full_matrix[test_indices]
    coverage = float(test.sum()) / max(float(original_test.sum()), 1.0)
    if not np.isclose(
        coverage,
        float(manifest["test_vocabulary_coverage"]),
        rtol=0,
        atol=1e-15,
    ):
        raise RuntimeError("test vocabulary coverage does not match manifest")

    cache_ids, cache_embeddings, cache_metadata = read_observed_cache(
        args.observed_features
    )
    expected_test_ids = tuple(spectrum_identifiers[index] for index in test_indices)
    _require_equal("observed cache identifiers", cache_ids, expected_test_ids)
    expected_variant = (
        "observed_peak_pairs:random:" f"{completion_metadata['assignment_sha256']}"
    )
    _require_equal(
        "observed cache variant", cache_metadata.get("variant"), expected_variant
    )
    spectra_map_sha256 = sha256_file(args.spectra_map)
    expected_cache_source = hashlib.sha256(
        (
            spectra_map_sha256
            + hash_array(test_indices)
            + completion_metadata["assignment_sha256"]
        ).encode("utf-8")
    ).hexdigest()
    _require_equal(
        "observed cache source hash",
        cache_metadata.get("source_sha256"),
        expected_cache_source,
    )

    return ReconstructedSplit(
        vocabulary=vocabulary,
        train_matrix=train,
        observed_matrix=observed,
        heldout_matrix=heldout,
        train_indices=train_indices,
        test_indices=test_indices,
        train_identifiers=tuple(spectrum_identifiers[index] for index in train_indices),
        test_identifiers=expected_test_ids,
        observed_words=csr_to_documents(observed, vocabulary),
        observed_embeddings=cache_embeddings,
        manifest=manifest,
        completion_metadata=completion_metadata,
        cache_metadata=cache_metadata,
    )


def load_legacy_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted, self-contained legacy checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require_equal("legacy checkpoint format", payload.get("format"), LEGACY_FORMAT)
    _require_equal("legacy checkpoint version", payload.get("version"), LEGACY_VERSION)
    if not isinstance(payload.get("documents"), list):
        raise TypeError("legacy checkpoint does not contain training documents")
    if not isinstance(payload.get("word_embeddings"), dict):
        raise TypeError("legacy checkpoint does not contain word embeddings")
    return payload


def config_from_legacy(
    payload: dict[str, Any],
    *,
    device: str,
) -> HybridLDAConfig:
    """Translate the legacy wrapper configuration into the reference config."""
    values = payload["init_parameters"]
    _require_equal("legacy hidden-layer count", int(values["num_hidden_layers"]), 2)
    _require_equal("legacy dropout", float(values["dropout"]), 0.0)
    for name in ("rm_top", "min_cf", "min_df"):
        _require_equal(f"legacy {name}", int(values[name]), 0)
    if not bool(values.get("use_encoder", False)):
        raise ValueError("legacy checkpoint does not use an encoder")
    alpha_values = np.asarray(values["alpha"], dtype=np.float32)
    alpha: float | tuple[float, ...]
    if alpha_values.ndim == 0:
        alpha = float(alpha_values)
    else:
        alpha = tuple(float(value) for value in alpha_values)
    del device  # Device belongs to HybridLDAModel rather than its config.
    return HybridLDAConfig(
        num_topics=int(values["k"]),
        embedding_dim=int(values["embedding_dim"]),
        alpha=alpha,
        eta=float(values["eta"]),
        hidden_size=int(values["hidden_size"]),
        feature_projection_dim=int(values["feature_projection_dim"]),
        training_local_steps=int(values["local_steps"]),
        batch_size=int(values["batch_size"]),
        encoder_learning_rate=float(values["encoder_learning_rate"]),
        encoder_updates_per_epoch=int(values["encoder_updates_per_epoch"]),
        prior_mass_fraction=float(values["prior_mass_fraction"]),
        prior_warmup_epochs=int(values["prior_warmup_epochs"]),
        prior_training_epochs=int(values["prior_training_epochs"]),
        prior_temperature=float(values["prior_temperature"]),
        prior_learning_rate=float(values["prior_learning_rate"]),
        topic_diversity_weight=float(values["topic_diversity_weight"]),
        local_tolerance=float(values["local_tolerance"]),
        global_tolerance=float(values["global_tolerance"]),
        global_patience=int(values["global_patience"]),
        max_epochs=int(values["max_epochs"]),
        seed=int(values["seed"]),
    )


def translate_legacy_state(
    target: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map legacy dropout-separated encoder indices onto the fixed architecture."""
    source_keys = {
        "encoder.2.weight": "encoder.3.weight",
        "encoder.2.bias": "encoder.3.bias",
        "encoder.4.weight": "encoder.6.weight",
        "encoder.4.bias": "encoder.6.bias",
    }
    translated: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for target_name, target_value in target.items():
        source_name = source_keys.get(target_name, target_name)
        if source_name not in source:
            raise KeyError(f"legacy state lacks {source_name!r} for {target_name!r}")
        source_value = source[source_name]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"state shape mismatch for {target_name}: "
                f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
            )
        translated[target_name] = source_value
        consumed.add(source_name)
    unexpected = set(source) - consumed
    if unexpected:
        raise ValueError(f"unmapped legacy state keys: {sorted(unexpected)}")
    return translated


def frozen_state(model: HybridLDAModel) -> dict[str, torch.Tensor]:
    """Clone every core tensor that is not owned by document inference."""
    if model._core is None:
        raise RuntimeError("model core is not prepared")
    return {
        name: value.detach().cpu().clone()
        for name, value in model._core.state_dict().items()
        if not name.startswith(ENCODER_PREFIXES)
    }


def encoder_state(model: HybridLDAModel) -> dict[str, torch.Tensor]:
    """Clone document-projector and encoder tensors for provenance."""
    if model._core is None:
        raise RuntimeError("model core is not prepared")
    return {
        name: value.detach().cpu().clone()
        for name, value in model._core.state_dict().items()
        if name.startswith(ENCODER_PREFIXES)
    }


def assert_state_equal(
    actual_model: HybridLDAModel,
    expected: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    """Require bitwise equality for every frozen tensor."""
    actual = frozen_state(actual_model)
    _require_equal(f"{label} frozen state keys", set(actual), set(expected))
    changed = [name for name in actual if not torch.equal(actual[name], expected[name])]
    if changed:
        raise RuntimeError(f"{label} changed frozen tensors: {changed}")


def build_migrated_model(
    payload: dict[str, Any],
    data: ReconstructedSplit,
    *,
    device: str,
) -> tuple[HybridLDAModel, dict[str, Any]]:
    """Rebuild documents, validate vocabulary alignment, and migrate state."""
    config = config_from_legacy(payload, device=device)
    model = HybridLDAModel(config, device=device)
    model.set_word_embeddings(payload["word_embeddings"])
    documents = payload["documents"]
    embeddings = payload.get("document_embeddings", [])
    identifiers = tuple(str(value) for value in payload.get("document_uids", []))
    _require_equal("legacy training identifiers", identifiers, data.train_identifiers)
    if len(documents) != len(embeddings):
        raise ValueError("legacy document words and embeddings are misaligned")
    for words, embedding in zip(documents, embeddings, strict=True):
        model.add_doc(words, embedding=embedding)
    # Avoid train(0): that public method refreshes document posteriors against
    # newly initialized topics before the migrated topic state is installed.
    model._prepare()
    if model._core is None or model._matrix is None:
        raise RuntimeError("migrated model failed to prepare")
    core = model._core
    _require_equal(
        "legacy vocabulary size", model.num_vocabs, data.train_matrix.shape[1]
    )
    _require_equal(
        "legacy vocabulary set", set(model.used_vocabs), set(data.vocabulary)
    )

    # Reorder the internal insertion-order matrix to the historical external
    # vocabulary before checking the split-manifest hash.
    internal_columns = np.asarray(
        [model._vocab_index[word] for word in data.vocabulary], dtype=np.int64
    )
    external_order_matrix = model._matrix[:, internal_columns].tocsr()
    external_order_matrix.sort_indices()
    _require_equal(
        "migrated training matrix",
        hash_sparse(external_order_matrix),
        data.manifest["train_matrix_sha256"],
    )

    legacy_state = payload["core_state_dict"]
    prepared_state = core.state_dict()
    for buffer_name in (
        "word_context_embeddings",
        "word_context_observed",
        "word_mz",
        "word_type",
    ):
        if not torch.equal(
            prepared_state[buffer_name].cpu(), legacy_state[buffer_name].cpu()
        ):
            raise RuntimeError(f"vocabulary alignment check failed for {buffer_name}")
    translated = translate_legacy_state(prepared_state, legacy_state)
    core.load_state_dict(translated, strict=True)
    if not torch.equal(
        core.lambda_posterior.cpu(), legacy_state["lambda_posterior"].cpu()
    ):
        raise RuntimeError("topic posterior was not migrated bit-exactly")

    gamma = payload.get("gamma_state")
    if not isinstance(gamma, np.ndarray) or gamma.shape != (len(model.docs), model.k):
        raise ValueError("legacy gamma_state is missing or has the wrong shape")
    model._gamma = np.asarray(gamma, dtype=np.float32).copy()
    model.history = list(payload.get("history", []))
    model._epochs = int(payload.get("epochs", len(model.history)))
    model._converged = bool(payload.get("converged", False))
    model._stable_epochs = int(payload.get("stable_epochs", 0))
    if payload.get("numpy_rng_state") is not None:
        model._rng.bit_generator.state = copy.deepcopy(payload["numpy_rng_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    return model, {
        "legacy_format": payload["format"],
        "legacy_version": payload["version"],
        "legacy_epochs": model._epochs,
        "legacy_converged": model._converged,
        "vocab_size": model.num_vocabs,
        "encoder_key_mapping": {
            "encoder.0": "encoder.0",
            "encoder.3": "encoder.2",
            "encoder.6": "encoder.4",
        },
    }


def make_test_data(
    model: HybridLDAModel,
    data: ReconstructedSplit,
) -> tuple[list[Any], sp.csr_matrix, sp.csr_matrix]:
    """Index observed words and align heldout columns to the model vocabulary."""
    documents = [
        model.make_doc(words, embedding=embedding)
        for words, embedding in zip(
            data.observed_words, data.observed_embeddings, strict=True
        )
    ]
    observed = model._documents_to_matrix(documents)
    _require_equal(
        "internal observed token count",
        int(observed.sum()),
        int(data.observed_matrix.sum()),
    )
    external_index = {word: index for index, word in enumerate(data.vocabulary)}
    external_columns = np.asarray(
        [external_index[word] for word in model.used_vocabs], dtype=np.int64
    )
    heldout = data.heldout_matrix[:, external_columns].tocsr()
    _require_equal(
        "internal heldout token count",
        int(heldout.sum()),
        int(data.heldout_matrix.sum()),
    )
    return documents, observed, heldout


@torch.no_grad()
def collect_arm_inference(
    model: HybridLDAModel,
    data: ReconstructedSplit,
    *,
    reference_steps: int,
    reference_tolerance: float,
    include_symmetric_reference: bool,
) -> tuple[dict[str, Any], sp.csr_matrix, sp.csr_matrix]:
    """Collect finite-budget and long-solve posterior arrays for one arm."""
    if model._core is None:
        raise RuntimeError("model core is not prepared")
    documents, observed, heldout = make_test_data(model, data)
    core = model._core
    core.eval()
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    num_documents = observed.shape[0]
    budget_values = {
        str(budget): {
            "elbo": np.empty(num_documents, dtype=np.float32),
            "theta": np.empty((num_documents, model.k), dtype=np.float32),
        }
        for budget in BUDGETS
    }
    long_elbo = np.empty(num_documents, dtype=np.float32)
    long_theta = np.empty((num_documents, model.k), dtype=np.float32)
    long_tail_gain = np.empty(num_documents, dtype=np.float32)
    symmetric_elbo = (
        np.empty(num_documents, dtype=np.float32)
        if include_symmetric_reference
        else None
    )
    symmetric_theta = (
        np.empty((num_documents, model.k), dtype=np.float32)
        if include_symmetric_reference
        else None
    )
    symmetric_tail_gain = (
        np.empty(num_documents, dtype=np.float32)
        if include_symmetric_reference
        else None
    )

    for start in range(0, num_documents, model.config.batch_size):
        indices = np.arange(start, min(start + model.config.batch_size, num_documents))
        batch = _make_sparse_batch(observed, indices, device=model.device)
        gamma_zero = core.encode(
            batch,
            model._embedding_batch(documents, indices),
            word_topic,
        )
        for budget in BUDGETS:
            gamma = gamma_zero
            if budget:
                gamma, _ = _local_vb(
                    batch,
                    gamma_zero,
                    core.alpha,
                    expected_log_beta,
                    steps=budget,
                    tolerance=None,
                )
            elbo = _local_document_elbo(batch, gamma, core.alpha, expected_log_beta)
            theta = gamma / gamma.sum(dim=1, keepdim=True)
            budget_values[str(budget)]["elbo"][indices] = elbo.cpu().numpy()
            budget_values[str(budget)]["theta"][indices] = theta.cpu().numpy()

        converged, _ = _local_vb(
            batch,
            gamma_zero,
            core.alpha,
            expected_log_beta,
            steps=reference_steps,
            tolerance=reference_tolerance,
        )
        converged_elbo = _local_document_elbo(
            batch, converged, core.alpha, expected_log_beta
        )
        tail, _ = _local_vb(
            batch,
            converged,
            core.alpha,
            expected_log_beta,
            steps=1,
            tolerance=None,
        )
        tail_elbo = _local_document_elbo(
            batch,
            tail,
            core.alpha,
            expected_log_beta,
        )
        tail_improved = tail_elbo > converged_elbo
        long_tail_gain[indices] = (
            torch.clamp(
                tail_elbo - converged_elbo,
                min=0.0,
            )
            .cpu()
            .numpy()
        )
        converged = torch.where(
            tail_improved.unsqueeze(1),
            tail,
            converged,
        )
        converged_elbo = torch.maximum(converged_elbo, tail_elbo)
        long_elbo[indices] = converged_elbo.cpu().numpy()
        long_theta[indices] = (
            (converged / converged.sum(dim=1, keepdim=True)).cpu().numpy()
        )

        if include_symmetric_reference:
            symmetric = core.alpha.unsqueeze(0) + batch.totals / model.k
            symmetric, _ = _local_vb(
                batch,
                symmetric,
                core.alpha,
                expected_log_beta,
                steps=reference_steps,
                tolerance=reference_tolerance,
            )
            symmetric_bound = _local_document_elbo(
                batch, symmetric, core.alpha, expected_log_beta
            )
            symmetric_tail, _ = _local_vb(
                batch,
                symmetric,
                core.alpha,
                expected_log_beta,
                steps=1,
                tolerance=None,
            )
            symmetric_tail_bound = _local_document_elbo(
                batch,
                symmetric_tail,
                core.alpha,
                expected_log_beta,
            )
            symmetric_improved = symmetric_tail_bound > symmetric_bound
            assert (
                symmetric_elbo is not None
                and symmetric_theta is not None
                and symmetric_tail_gain is not None
            )
            symmetric_tail_gain[indices] = (
                torch.clamp(
                    symmetric_tail_bound - symmetric_bound,
                    min=0.0,
                )
                .cpu()
                .numpy()
            )
            symmetric = torch.where(
                symmetric_improved.unsqueeze(1),
                symmetric_tail,
                symmetric,
            )
            symmetric_bound = torch.maximum(
                symmetric_bound,
                symmetric_tail_bound,
            )
            symmetric_elbo[indices] = symmetric_bound.cpu().numpy()
            symmetric_theta[indices] = (
                (symmetric / symmetric.sum(dim=1, keepdim=True)).cpu().numpy()
            )

    result: dict[str, Any] = {
        "budgets": budget_values,
        "long_elbo": long_elbo,
        "long_theta": long_theta,
        "long_tail_gain": long_tail_gain,
        "beta": core.beta_mean().cpu().numpy(),
    }
    if include_symmetric_reference:
        result["symmetric_elbo"] = symmetric_elbo
        result["symmetric_theta"] = symmetric_theta
        result["symmetric_tail_gain"] = symmetric_tail_gain
    return result, observed, heldout


def _synchronize(device: torch.device) -> None:
    """Synchronize accelerator work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_fixed_budget(
    model: HybridLDAModel,
    words: list[list[str]],
    embeddings: np.ndarray,
    *,
    budget: int,
) -> np.ndarray:
    """Run end-to-end cached-feature inference, including sparse batching."""
    if model._core is None:
        raise RuntimeError("model core is not prepared")
    documents = [
        model.make_doc(document, embedding=embedding)
        for document, embedding in zip(words, embeddings, strict=True)
    ]
    matrix = model._documents_to_matrix(documents)
    core = model._core
    core.eval()
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    outputs: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], model.config.batch_size):
        indices = np.arange(
            start, min(start + model.config.batch_size, matrix.shape[0])
        )
        batch = _make_sparse_batch(matrix, indices, device=model.device)
        gamma = core.encode(
            batch,
            model._embedding_batch(documents, indices),
            word_topic,
        )
        if budget:
            gamma, _ = _local_vb(
                batch,
                gamma,
                core.alpha,
                expected_log_beta,
                steps=budget,
                tolerance=None,
            )
        outputs.append((gamma / gamma.sum(dim=1, keepdim=True)).cpu().numpy())
    return np.vstack(outputs).astype(np.float32, copy=False)


def timed_budgets(
    model: HybridLDAModel,
    data: ReconstructedSplit,
    *,
    repeats: int,
) -> dict[str, Any]:
    """Time document construction, sparse batching, encoding, and refinement."""
    timings: dict[str, Any] = {}
    for budget in BUDGETS:
        run_fixed_budget(
            model,
            data.observed_words,
            data.observed_embeddings,
            budget=budget,
        )
        durations: list[float] = []
        for _ in range(repeats):
            _synchronize(model.device)
            started = time.perf_counter()
            run_fixed_budget(
                model,
                data.observed_words,
                data.observed_embeddings,
                budget=budget,
            )
            _synchronize(model.device)
            durations.append(time.perf_counter() - started)
        per_document = 1000.0 * np.asarray(durations) / len(data.observed_words)
        timings[str(budget)] = {
            "seconds": durations,
            "median_ms_per_document": float(np.median(per_document)),
            "repeats": repeats,
            "includes": "document_indexing+sparse_batching+encoder+local_vb",
        }
    return timings


def training_callback(arm: str, total_epochs: int):
    """Construct a sparse progress callback for inference-network fitting."""

    def callback(metrics: dict[str, float]) -> None:
        epoch = int(metrics["inference_epoch"])
        if epoch == 1 or epoch == total_epochs or epoch % 4 == 0:
            values = [f"loss={metrics['loss']:.5g}"]
            for key, label in (
                ("refined_elbo_per_token", "refined_elbo/token"),
                ("zero_step_elbo_per_token", "zero_elbo/token"),
                ("teacher_mean_kl", "teacher_kl"),
            ):
                if key in metrics:
                    values.append(f"{label}={metrics[key]:.5g}")
            progress(f"  {arm} epoch={epoch}/{total_epochs} " + " ".join(values))

    return callback


def fit_arm(
    name: str,
    model: HybridLDAModel,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], float]:
    """Run one fixed post-hoc objective or preserve the historical encoder."""
    if name == "historical":
        return [], 0.0
    if name == "distillation":
        weights = {
            "refined_elbo_weight": 0.0,
            "zero_step_elbo_weight": 0.0,
            "distillation_weight": 1.0,
        }
    elif name == "unrolled_plus_zero":
        weights = {
            "refined_elbo_weight": 1.0,
            "zero_step_elbo_weight": 0.1,
            "distillation_weight": 0.0,
        }
    else:  # pragma: no cover - internal caller controls names
        raise ValueError(f"unknown arm {name!r}")
    started = time.perf_counter()
    history = model.fit_inference_network(
        epochs=args.epochs,
        refinement_steps=args.refinement_steps,
        teacher_steps=args.teacher_steps,
        reset_optimizer=True,
        progress_callback=training_callback(name, args.epochs),
        **weights,
    )
    return history, time.perf_counter() - started


def common_reference(
    raw_arms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Select the best long-solve ELBO per document across all initializers."""
    source_names = [f"{name}_long" for name in raw_arms]
    elbo_sources = [raw_arms[name]["long_elbo"] for name in raw_arms]
    theta_sources = [raw_arms[name]["long_theta"] for name in raw_arms]
    historical = raw_arms["historical"]
    source_names.append("symmetric_long")
    elbo_sources.append(historical["symmetric_elbo"])
    theta_sources.append(historical["symmetric_theta"])
    stacked_elbo = np.stack(elbo_sources)
    stacked_theta = np.stack(theta_sources)
    best_source = np.argmax(stacked_elbo, axis=0)
    rows = np.arange(stacked_elbo.shape[1])
    best_elbo = stacked_elbo[best_source, rows]
    best_theta = stacked_theta[best_source, rows]
    counts = {
        name: int(np.sum(best_source == index))
        for index, name in enumerate(source_names)
    }
    return best_elbo, best_theta, counts


def finalize_metrics(  # noqa: PLR0913
    raw: dict[str, Any],
    *,
    reference_elbo: np.ndarray,
    reference_theta: np.ndarray,
    observed_tokens: float,
    heldout: sp.csr_matrix,
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Calculate signed common-reference gaps, KL, and heldout NLL."""
    budgets: dict[str, Any] = {}
    beta = raw["beta"]
    for budget in BUDGETS:
        values = raw["budgets"][str(budget)]
        elbo = values["elbo"]
        theta = values["theta"]
        signed_gap = reference_elbo - elbo
        reference_kl = (
            reference_theta
            * (
                np.log(np.clip(reference_theta, EPSILON, None))
                - np.log(np.clip(theta, EPSILON, None))
            )
        ).sum(axis=1)
        budgets[str(budget)] = {
            "observed_elbo_per_token": float(elbo.sum()) / observed_tokens,
            "signed_reference_gap_per_token": (
                float(signed_gap.sum()) / observed_tokens
            ),
            "mean_signed_reference_gap": float(signed_gap.mean()),
            "min_signed_reference_gap": float(signed_gap.min()),
            "negative_gap_documents": int(np.sum(signed_gap < 0)),
            "reference_mean_kl": float(reference_kl.mean()),
            "heldout_token_nll": observed_token_nll(heldout, theta, beta),
            "timing": timing[str(budget)],
        }
    return {"budgets": budgets}


def legacy_reproduction(
    path: Path,
    historical_metrics: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare migrated historical inference to its original run summary."""
    if not path.is_file():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    for budget, expected_values in payload.get("test_inference", {}).items():
        if budget not in historical_metrics["budgets"]:
            continue
        expected = float(expected_values["heldout_token_nll"])
        actual = float(historical_metrics["budgets"][budget]["heldout_token_nll"])
        difference = actual - expected
        comparisons[budget] = {
            "expected_heldout_token_nll": expected,
            "actual_heldout_token_nll": actual,
            "difference": difference,
        }
        if abs(difference) > tolerance:
            raise RuntimeError(
                f"migrated budget-{budget} NLL differs from legacy result by "
                f"{difference:.6g}, exceeding {tolerance:.6g}"
            )
    return {
        "available": True,
        "path": str(path),
        "tolerance": tolerance,
        "comparisons": comparisons,
    }


def comparison_gate(arms: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified smoke-test gate against equal-epoch distillation."""
    baseline = arms["distillation"]["metrics"]["budgets"]
    candidate = arms["unrolled_plus_zero"]["metrics"]["budgets"]
    baseline_gap = float(baseline["2"]["signed_reference_gap_per_token"])
    candidate_gap = float(candidate["2"]["signed_reference_gap_per_token"])
    gap_reduction: float | None = None
    if abs(baseline_gap) > EPSILON:
        gap_reduction = (baseline_gap - candidate_gap) / abs(baseline_gap)
    zero_nll_change = float(
        candidate["0"]["heldout_token_nll"] - baseline["0"]["heldout_token_nll"]
    )
    two_nll_change = float(
        candidate["2"]["heldout_token_nll"] - baseline["2"]["heldout_token_nll"]
    )
    two_vs_five_nll_change = float(
        candidate["2"]["heldout_token_nll"] - baseline["5"]["heldout_token_nll"]
    )
    checks = {
        "two_step_gap_reduction_at_least_10_percent": (
            gap_reduction is not None and gap_reduction >= 0.10
        ),
        "zero_step_nll_noninferior_within_0_01": zero_nll_change <= 0.01,
        "two_step_nll_noninferior_within_0_01": two_nll_change <= 0.01,
    }
    return {
        "two_step_gap_reduction": gap_reduction,
        "zero_step_nll_change": zero_nll_change,
        "two_step_nll_change": two_nll_change,
        "candidate_two_vs_distillation_five_nll_change": two_vs_five_nll_change,
        "checks": checks,
        "passes_smoke_gate": all(checks.values()),
    }


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or ill-defined execution before deserialization."""
    if not args.trust_legacy_artifacts:
        raise PermissionError(
            "Refusing unsafe historical deserialization. Re-run with "
            "--trust-legacy-artifacts only after verifying these local inputs."
        )
    for name in (
        "legacy_checkpoint",
        "reference_model",
        "spectra_map",
        "observed_features",
        "split_manifest",
    ):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.legacy_run_summary is not None:
        args.legacy_run_summary = args.legacy_run_summary.expanduser().resolve()
    else:
        args.legacy_run_summary = (
            args.legacy_checkpoint.parent / "run_summary.json"
        ).resolve()
    if args.epochs < 1 or args.refinement_steps < 1:
        raise ValueError("epochs and refinement_steps must be positive")
    if args.teacher_steps is not None and args.teacher_steps < 1:
        raise ValueError("teacher_steps must be positive")
    if (
        args.reference_steps < 1
        or not np.isfinite(args.reference_tolerance)
        or args.reference_tolerance <= 0
    ):
        raise ValueError("reference solve settings must be positive")
    if args.timing_repeats < 1 or args.threads < 1:
        raise ValueError("timing_repeats and threads must be positive")
    if (
        not np.isfinite(args.mz_tolerance)
        or args.mz_tolerance <= 0
        or not np.isfinite(args.legacy_nll_tolerance)
        or args.legacy_nll_tolerance < 0
    ):
        raise ValueError("m/z and legacy NLL tolerances must be valid")
    inputs = {
        args.legacy_checkpoint,
        args.reference_model,
        args.spectra_map,
        args.observed_features,
        args.split_manifest,
        args.legacy_run_summary,
    }
    if args.output in inputs:
        raise ValueError("output must not overwrite a benchmark input")


def main() -> None:
    """Run reconstruction, migration, frozen-topic training, and evaluation."""
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    progress("Trusted legacy deserialization explicitly enabled.")
    artifact_paths = {
        "legacy_checkpoint": args.legacy_checkpoint,
        "reference_model": args.reference_model,
        "spectra_map": args.spectra_map,
        "observed_features": args.observed_features,
        "split_manifest": args.split_manifest,
    }
    if args.legacy_run_summary.is_file():
        artifact_paths["legacy_run_summary"] = args.legacy_run_summary
    artifact_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}

    progress("Reconstructing and verifying the random physical-peak split ...")
    data = reconstruct_split(args)
    progress(
        f"  verified train={data.train_matrix.shape[0]} test={data.observed_matrix.shape[0]} "
        f"vocab={data.train_matrix.shape[1]}"
    )
    progress("Loading trusted legacy checkpoint and validating migration ...")
    legacy_payload = load_legacy_checkpoint(args.legacy_checkpoint)
    checkpoint_seed = int(legacy_payload["init_parameters"]["seed"])

    raw_arms: dict[str, dict[str, Any]] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    common_initial_encoder_hash: str | None = None
    common_frozen_hash: str | None = None
    migration_report: dict[str, Any] | None = None
    heldout_for_metrics: sp.csr_matrix | None = None
    observed_tokens = float(data.observed_matrix.sum())

    for arm_index, arm_name in enumerate(
        ("historical", "distillation", "unrolled_plus_zero")
    ):
        progress(f"Building arm={arm_name} from the identical frozen checkpoint ...")
        model, current_migration = build_migrated_model(
            legacy_payload, data, device=args.device
        )
        if migration_report is None:
            migration_report = current_migration
        initial_encoder_hash = tensor_state_sha256(encoder_state(model))
        initial_frozen = frozen_state(model)
        frozen_hash = tensor_state_sha256(initial_frozen)
        if common_initial_encoder_hash is None:
            common_initial_encoder_hash = initial_encoder_hash
            common_frozen_hash = frozen_hash
        else:
            _require_equal(
                f"{arm_name} initial encoder hash",
                initial_encoder_hash,
                common_initial_encoder_hash,
            )
            _require_equal(
                f"{arm_name} initial frozen hash", frozen_hash, common_frozen_hash
            )

        history, training_seconds = fit_arm(arm_name, model, args)
        assert_state_equal(model, initial_frozen, label=arm_name)
        final_encoder_hash = tensor_state_sha256(encoder_state(model))
        if arm_name != "historical" and final_encoder_hash == initial_encoder_hash:
            raise RuntimeError(f"{arm_name} did not update the encoder")
        progress(f"Evaluating arm={arm_name} with a long local reference solve ...")
        raw, _, heldout = collect_arm_inference(
            model,
            data,
            reference_steps=args.reference_steps,
            reference_tolerance=args.reference_tolerance,
            include_symmetric_reference=arm_index == 0,
        )
        if heldout_for_metrics is None:
            heldout_for_metrics = heldout
        elif hash_sparse(heldout) != hash_sparse(heldout_for_metrics):
            raise RuntimeError("heldout matrix alignment differs between arms")
        progress(f"Timing arm={arm_name}; DreaMS extraction remains cached ...")
        timing = timed_budgets(model, data, repeats=args.timing_repeats)
        raw_arms[arm_name] = raw
        tail_gain = raw["long_tail_gain"]
        reference_diagnostics: dict[str, Any] = {
            "continuation_positive_documents": int(np.sum(tail_gain > 0)),
            "continuation_max_elbo_gain": float(tail_gain.max()),
            "continuation_gain_per_observed_token": float(tail_gain.sum())
            / observed_tokens,
        }
        if arm_name == "historical":
            symmetric_tail_gain = raw["symmetric_tail_gain"]
            reference_diagnostics["symmetric_continuation_positive_documents"] = int(
                np.sum(symmetric_tail_gain > 0)
            )
            reference_diagnostics["symmetric_continuation_max_elbo_gain"] = float(
                symmetric_tail_gain.max()
            )
            reference_diagnostics["symmetric_continuation_gain_per_observed_token"] = (
                float(symmetric_tail_gain.sum()) / observed_tokens
            )
        arm_reports[arm_name] = {
            "objective": (
                "none"
                if arm_name == "historical"
                else (
                    "final-topic posterior distillation"
                    if arm_name == "distillation"
                    else "refined local ELBO + 0.1 * zero-step local ELBO"
                )
            ),
            "training_seconds": training_seconds,
            "training_history": history,
            "initial_encoder_sha256": initial_encoder_hash,
            "final_encoder_sha256": final_encoder_hash,
            "frozen_state_sha256": frozen_hash,
            "frozen_state_bit_exact_after_training": True,
            "reference_diagnostics": reference_diagnostics,
            "timing": timing,
        }
        del model
        gc.collect()

    assert heldout_for_metrics is not None
    reference_elbo, reference_theta, source_counts = common_reference(raw_arms)
    for arm_name, raw in raw_arms.items():
        arm_reports[arm_name]["metrics"] = finalize_metrics(
            raw,
            reference_elbo=reference_elbo,
            reference_theta=reference_theta,
            observed_tokens=observed_tokens,
            heldout=heldout_for_metrics,
            timing=arm_reports[arm_name].pop("timing"),
        )

    reproduction = legacy_reproduction(
        args.legacy_run_summary,
        arm_reports["historical"]["metrics"],
        tolerance=args.legacy_nll_tolerance,
    )
    gate = comparison_gate(arm_reports)
    fixed_default_configuration = (
        args.epochs == 12
        and args.refinement_steps == 2
        and args.teacher_steps is None
        and args.reference_steps == 100
        and args.reference_tolerance == 1e-7
        and args.timing_repeats == 5
        and args.threads == 1
        and args.device == "cpu"
        and args.train_fraction == 0.8
        and args.observed_fraction == 0.5
        and args.mz_tolerance == 0.02
        and args.legacy_nll_tolerance == 1e-5
    )
    gate["fixed_default_configuration"] = fixed_default_configuration
    gate["passes_fixed_configuration_smoke_gate"] = (
        gate["passes_smoke_gate"] if fixed_default_configuration else None
    )
    split_report = {
        "mode": "random",
        "train_documents": data.train_matrix.shape[0],
        "test_documents": data.observed_matrix.shape[0],
        "vocab_size": data.train_matrix.shape[1],
        "train_tokens": int(data.train_matrix.sum()),
        "observed_tokens": int(data.observed_matrix.sum()),
        "heldout_tokens": int(data.heldout_matrix.sum()),
        "manifest_hashes_verified": True,
        "cache_identifiers_verified": True,
        "completion": data.completion_metadata,
    }
    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "post-hoc semi-amortized inference; topics remain frozen",
        "configuration": {
            "checkpoint_seed": checkpoint_seed,
            "epochs": args.epochs,
            "refinement_steps": args.refinement_steps,
            "teacher_steps": (
                int(legacy_payload["init_parameters"]["local_steps"])
                if args.teacher_steps is None
                else args.teacher_steps
            ),
            "reference_steps": args.reference_steps,
            "reference_tolerance": args.reference_tolerance,
            "budgets": list(BUDGETS),
            "timing_repeats": args.timing_repeats,
            "threads": args.threads,
            "device": args.device,
            "train_fraction": args.train_fraction,
            "observed_fraction": args.observed_fraction,
            "mz_tolerance": args.mz_tolerance,
        },
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_sha256": artifact_hashes,
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "split": split_report,
        "migration": migration_report,
        "common_reference": {
            "definition": (
                "per-document maximum ELBO among long solves initialized from "
                "all three arm encoders and a symmetric gamma"
            ),
            "selected_source_counts": source_counts,
            "negative finite-budget gaps_are_not_clamped": True,
        },
        "arms": arm_reports,
        "legacy_reproduction": reproduction,
        "comparison_to_equal_epoch_distillation": gate,
        "interpretation_boundary": (
            "This benchmark evaluates document inference only. Frozen topics mean "
            "it provides no new evidence about motif discovery or chemical quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress(
        f"Wrote {args.output}; smoke_gate={gate['passes_smoke_gate']} "
        f"gap_reduction={gate['two_step_gap_reduction']}"
    )


if __name__ == "__main__":
    main()
