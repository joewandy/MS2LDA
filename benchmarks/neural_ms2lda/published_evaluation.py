"""Frozen-decoder evaluation for additive and product-of-experts topic models."""

from __future__ import annotations

import numpy as np
import torch

from .reproducibility import normalize_probability_rows


def dense_counts(matrix, rows, device) -> torch.Tensor:
    """Published encoders consume raw counts, not ETM's normalized inputs."""
    return torch.as_tensor(matrix[rows].toarray(), dtype=torch.float32, device=device)


@torch.inference_mode()
def infer_published(model, matrix, *, batch_size: int, device) -> np.ndarray:
    """Infer deterministic posterior summaries with immutable BatchNorm buffers."""
    model.eval()
    result = []
    for start in range(0, matrix.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, matrix.shape[0]))
        theta, _ = model.encoder(dense_counts(matrix, rows, device), sample=False)
        result.append(theta.cpu().numpy())
    return normalize_probability_rows(np.concatenate(result), name="published theta")


@torch.inference_mode()
def published_completion(model, theta, completion, records, *, batch_size: int, device):
    """Score actual decoder probabilities; PoE must never use theta @ beta."""
    if len(theta) != completion.shape[0] or len(records) != len(theta):
        raise ValueError("theta, completion, and records must have matching rows")
    model.eval()
    loss, tokens, eligible = 0.0, 0, 0
    for start in range(0, len(theta), batch_size):
        stop = min(start + batch_size, len(theta))
        logp = model.decoder(
            torch.as_tensor(theta[start:stop], dtype=torch.float32, device=device)
        )
        block = completion[start:stop].tocoo()
        selected = logp[
            torch.as_tensor(block.row, device=device).long(),
            torch.as_tensor(block.col, device=device).long(),
        ]
        # The frozen benchmark clips all models' token probabilities at 1e-12.
        # Retain an unclipped model NLL separately in future likelihood studies.
        selected = selected.clamp_min(float(np.log(1e-12))).double()
        loss -= float(
            (selected * torch.as_tensor(block.data, device=device).double()).sum()
        )
        tokens += int(block.data.sum())
        eligible += int(np.count_nonzero(np.asarray(block.sum(axis=1)).ravel()))
    if tokens <= 0:
        raise ValueError("completion has no in-vocabulary tokens")
    oov = sum(int(row["completion_oov_tokens"]) for row in records)
    return {
        "nll_per_token": loss / tokens,
        "in_vocabulary_tokens": tokens,
        "out_of_vocabulary_tokens": oov,
        "oov_fraction": oov / (tokens + oov),
        "eligible_documents": eligible,
        "total_documents": len(theta),
        "decoder": "product_of_experts" if model.product else "additive_mixture",
        "posterior_summary": (
            (
                "entmax_gaussian_mean"
                if model.encoder.sparse
                else "softmax_gaussian_mean"
            )
            if model.product and hasattr(model.encoder, "mean")
            else "dirichlet_mean"
        ),
    }
