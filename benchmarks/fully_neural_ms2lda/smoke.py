"""Fast synthetic mechanics check for the neural model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

from .data import iter_sparse_batches
from .model import NeuralMS2LDA
from .utils import write_json

if TYPE_CHECKING:
    from pathlib import Path


def run_smoke(output: str | Path | None = None) -> dict[str, Any]:
    """Exercise encoder and alternating decoder updates on tiny sparse counts."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    counts = rng.poisson(0.5, size=(24, 31)).astype(np.float32)
    counts[counts < 1] = 0
    for row in range(len(counts)):
        if not counts[row].any():
            counts[row, row % counts.shape[1]] = 1
    matrix = sp.csr_matrix(counts)
    features = torch.nn.functional.normalize(torch.randn(31, 8), dim=1)
    model = NeuralMS2LDA(
        features,
        num_topics=6,
        hidden_dimensions=12,
        topic_word_temperature=0.4,
        dropout=0.0,
        topic_initial_indices=torch.arange(6),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    batch = next(
        iter_sparse_batches(matrix, batch_size=12, shuffle=False, seed=42),
    )
    with torch.no_grad():
        projected = model.projected_tokens().detach()
        beta = model.topic_word_distribution(projected).detach()
    encoder = model.encoder_loss(
        batch,
        beta=beta,
        projected_tokens=projected,
        kl_weight=0.1,
    )
    optimizer.zero_grad(set_to_none=True)
    encoder.total.backward()
    optimizer.step()
    with torch.no_grad():
        theta, _, _ = model.encode(batch, sample=False)
    decoder, reconstruction, ecr = model.decoder_loss(
        batch,
        theta=theta.detach(),
        ecr_token_indices=torch.arange(16),
        ecr_weight=1.0,
        ecr_epsilon=0.1,
        ecr_iterations=5,
    )
    optimizer.zero_grad(set_to_none=True)
    decoder.backward()
    optimizer.step()
    final_beta = model.topic_word_distribution().detach()
    result = {
        "schema_version": "fully-neural-ms2lda/smoke-v1",
        "pass": bool(
            torch.isfinite(encoder.total)
            and torch.isfinite(decoder)
            and torch.allclose(final_beta.sum(dim=1), torch.ones(6), atol=1e-5),
        ),
        "encoder_loss": float(encoder.total.detach()),
        "decoder_loss": float(decoder.detach()),
        "decoder_reconstruction": float(reconstruction.detach()),
        "ecr": float(ecr.detach()),
        "single_encoder_pass": True,
        "local_vb_steps": 0,
    }
    if output is not None:
        write_json(output, result)
    return result
