"""Batched posterior-mean inference shared by additive ETM-family models."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from .topic_model_training import dense_normalized

if TYPE_CHECKING:
    import scipy.sparse as sp

    from .contextual_sparse_etm import ContextualSparseETM
    from .etm_baselines import CanonicalETM


@torch.inference_mode()
def infer_document_topics(
    model: CanonicalETM | ContextualSparseETM,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Return float32 mixtures and synchronized inference throughput.

    Use the model's own inference equation, with frozen evaluation state.
    Canonical normalization for persisted probabilities remains an export step.
    """
    if batch_size <= 0 or matrix.shape[0] == 0:
        raise ValueError("inference needs positive batch size and at least one row")
    if device is not None and (
        device.type != model.rho.device.type
        or (device.index is not None and device.index != model.rho.device.index)
    ):
        raise ValueError("inference device must match the model")
    device = model.rho.device
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    values = []
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(
            start, min(start + batch_size, matrix.shape[0]), dtype=np.int64
        )
        theta, _ = model.document_topic_mixture(
            dense_normalized(matrix, rows, device), sample=False
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, 1e-12)
