"""Numerical tripwires for the accepted seed-42 research checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.artifacts import load_bundle
from benchmarks.neural_ms2lda.data import sparse_batch
from benchmarks.neural_ms2lda.utils import file_sha256, object_sha256, read_json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PACKAGE_ROOT / "results/seed42"
BUNDLE_ROOT = RESULTS_ROOT / "model_bundle"

CHECKPOINT_SHA256 = "639c1f37c613d908b59e3a85b7dc701e33a3f92fd7476a3257b47298143dfbc6"
BETA_SHA256 = "9a71e51fac05d2f3a23f5632b177f3a3cba596f297e9441410afcbdb0222ea6a"
THETA_SHA256 = "b0a7ffdae66dd9d221d3b49783ea11232f168f9500eea80c98668322354182ba"
PRIMARY_METHODS_SHA256 = (
    "63a20a72d69e332518a08c58b378e8d8d301a540d1e0d1703f794473326d19d2"
)


def _array_sha256(value: np.ndarray) -> str:
    """Hash numerical content together with the array's interpretation."""
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _reference_batch(vocabulary_size: int) -> sp.csr_matrix:
    """Return a fixed two-observed-plus-one-empty spectrum batch."""
    return sp.csr_matrix(
        (
            np.asarray([1.0, 2.0, 1.0, 3.0, 1.0], dtype=np.float32),
            np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
            np.asarray([0, 2, 5, 5], dtype=np.int32),
        ),
        shape=(3, vocabulary_size),
        dtype=np.float32,
    )


def test_accepted_checkpoint_outputs_are_exact() -> None:
    """Refactoring must not alter the selected weights, beta, or one-pass theta."""
    assert file_sha256(BUNDLE_ROOT / "model.pt") == CHECKPOINT_SHA256
    model, vocabulary, _ = load_bundle(BUNDLE_ROOT)
    batch = sparse_batch(
        _reference_batch(len(vocabulary)), np.arange(3, dtype=np.int64)
    )
    with torch.inference_mode():
        beta = model.topic_word_distribution().cpu().numpy()
        theta = (
            model.route(
                batch,
                temperature=0.1,
                straight_through=False,
            )
            .theta.cpu()
            .numpy()
        )
    assert _array_sha256(beta) == BETA_SHA256
    assert _array_sha256(theta) == THETA_SHA256


def test_accepted_paper_results_are_fixed() -> None:
    """Keep the peer-facing comparison unchanged while its plumbing is simplified."""
    results = read_json(RESULTS_ROOT / "results.json")
    assert object_sha256(results["methods"]) == PRIMARY_METHODS_SHA256
    methods = {row["method"]: row for row in results["methods"]}
    assert methods["neural"]["validation"] == {
        "annotation_coverage": 0.663,
        "high_confidence_evaluable_motifs": 312,
        "mean_sos": 0.6323301481310782,
        "median_sos": 0.6363636363636365,
        "optimized_motifs": 663,
        "sos_bands": {
            "high_gt_0_8": 49,
            "intermediate_0_6_to_0_8": 136,
            "low_lt_0_6": 127,
        },
        "useful_high_confidence_motifs": 185,
    }
    assert results["secondary_diagnostics"]["completion_nll_per_token"] == {
        "neural": {
            "validation": 8.501446912771746,
            "test": 8.522600207027194,
        },
        "tomotopy": {
            "validation": 9.662228074924426,
            "test": 9.756948055261505,
        },
    }
    assert methods["tomotopy"]["validation"]["annotation_coverage"] == 0.607
    assert methods["neural"]["test"]["annotation_coverage"] == 0.663
    assert methods["tomotopy"]["test"]["annotation_coverage"] == 0.607
