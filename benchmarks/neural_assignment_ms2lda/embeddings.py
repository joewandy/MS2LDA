"""Pinned train-only SGNS implementation shared with the preserved v1 study."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks.fully_neural_ms2lda.embeddings import train_sgns as _train_sgns

if TYPE_CHECKING:
    from pathlib import Path

    import scipy.sparse as sp


def train_sgns(
    output_dir: str | Path,
    matrix: sp.csr_matrix,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run the already validated SGNS feature builder unchanged."""
    return _train_sgns(output_dir, matrix, config, seed=seed)
