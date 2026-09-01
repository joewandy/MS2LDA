"""Functional count-matrix operations used to train neural topic models."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch

EPSILON = 1e-12
ReconstructionScaling = Literal["raw_counts", "distinct_words", "unit_mass"]
RECONSTRUCTION_SCALINGS: tuple[ReconstructionScaling, ...] = (
    "raw_counts",
    "distinct_words",
    "unit_mass",
)


def dense_normalized(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return row-normalized spectral-word counts for the ETM encoder.

    The returned tensor is ``x`` in equation ``eq:normalized-bow`` of the
    report.  Empty rows remain all-zero; prepared training and validation
    spectra are non-empty after preprocessing.
    """
    values = torch.from_numpy(
        matrix[rows].toarray().astype(np.float32, copy=False),
    ).to(device)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1.0)


def sparse_reconstruction_loss(
    theta: torch.Tensor,
    beta: torch.Tensor,
    count_batch: sp.csr_matrix,
    device: torch.device,
    *,
    scaling: ReconstructionScaling,
) -> tuple[torch.Tensor, float]:
    """Evaluate the observed-word multinomial reconstruction loss.

    For the reported ``raw_counts`` setting, this is equation
    ``eq:reconstruction``:

    ``-mean_d sum_w c_dw log(sum_k theta_dk beta_kw)``.

    Only non-zero ``c_dw`` entries are gathered, avoiding a dense
    document-by-vocabulary reconstruction matrix.  The other scaling choices
    are retained solely for controlled experimental comparisons.
    """
    if scaling not in RECONSTRUCTION_SCALINGS:
        raise ValueError(f"unknown reconstruction scaling: {scaling}")
    batch = count_batch.tocsr()
    if batch.nnz == 0:
        return theta.new_zeros(()), 0.0

    lengths = np.diff(batch.indptr).astype(np.int64, copy=False)
    row_ids_array = np.repeat(np.arange(batch.shape[0], dtype=np.int64), lengths)
    row_ids = torch.from_numpy(row_ids_array).to(device)
    word_ids = torch.from_numpy(
        batch.indices.astype(np.int64, copy=False),
    ).to(device)
    weights_array = batch.data.astype(np.float32, copy=True)
    raw_totals = np.asarray(batch.sum(axis=1)).ravel().astype(np.float32)

    if scaling == "distinct_words":
        target_totals = lengths.astype(np.float32)
    elif scaling == "unit_mass":
        target_totals = (raw_totals > 0).astype(np.float32)
    else:
        target_totals = raw_totals
    if scaling != "raw_counts":
        scale = np.divide(
            target_totals,
            raw_totals,
            out=np.zeros_like(target_totals),
            where=raw_totals > 0,
        )
        weights_array *= scale[row_ids_array]

    c_dw = torch.from_numpy(weights_array).to(device)
    # p(w|d) = sum_k theta_dk beta_kw, equation (eq:mixture), evaluated only
    # at observed document-word coordinates.
    p_w_given_d = torch.sum(
        theta[row_ids] * beta[:, word_ids].T,
        dim=1,
    ).clamp_min(EPSILON)
    per_document = theta.new_zeros((batch.shape[0],))
    per_document.index_add_(0, row_ids, -c_dw * torch.log(p_w_given_d))
    return per_document.mean(), float(target_totals.mean())


def raw_count_reconstruction_loss(
    theta: torch.Tensor,
    beta: torch.Tensor,
    count_batch: sp.csr_matrix,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Return exactly the report's raw pseudo-count reconstruction term."""
    return sparse_reconstruction_loss(
        theta,
        beta,
        count_batch,
        device,
        scaling="raw_counts",
    )
