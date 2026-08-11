"""Tiny deterministic exercise of both alternating optimization blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

from .config import load_protocol, static_candidate_audit
from .data import prototype_seeding_weights, sparse_batch
from .model import (
    balanced_sinkhorn_targets,
    initialize_model,
    router_block_loss,
    topic_block_loss,
)
from .training import infer_theta
from .utils import write_json

if TYPE_CHECKING:
    from pathlib import Path


def run_smoke(output: str | Path | None = None) -> dict[str, Any]:
    """Run finite forward/backward checks without touching scientific data."""
    protocol = load_protocol()
    torch.manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(18, 64), dim=1)
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [3, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    right = matrix[:, np.roll(np.arange(matrix.shape[1]), 1)].tocsr()
    rows = np.arange(matrix.shape[0], dtype=np.int64)
    left_batch = sparse_batch(matrix, rows)
    right_batch = sparse_batch(right, rows)
    model, _ = initialize_model(
        features,
        num_topics=4,
        protocol=protocol,
        seeding_weights=prototype_seeding_weights(matrix),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    beta = model.topic_word_distribution().detach()
    router = router_block_loss(
        model,
        left_batch,
        right_batch,
        cached_beta=beta,
        temperature=0.5,
        top_k=2,
        sinkhorn_weight=0.25,
        consistency_weight=0.1,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=20,
    )
    optimizer.zero_grad(set_to_none=True)
    router.total.backward()
    optimizer.step()
    topic = topic_block_loss(
        model,
        left_batch,
        right_batch,
        temperature=0.5,
        top_k=2,
        local_decoder_weight=0.25,
    )
    optimizer.zero_grad(set_to_none=True)
    topic.total.backward()
    optimizer.step()
    theta = infer_theta(
        model,
        matrix,
        batch_size=2,
        temperature=0.1,
        top_k=2,
    )
    sinkhorn = balanced_sinkhorn_targets(
        torch.randn(40, 4),
        epsilon=0.2,
        iterations=100,
    )
    result = {
        "schema_version": "neural-assignment-ms2lda/smoke-v1",
        "pass": bool(
            torch.isfinite(router.total)
            and torch.isfinite(topic.total)
            and np.all(np.isfinite(theta))
            and np.allclose(theta.sum(axis=1), 1.0),
        ),
        "router_loss": float(router.total.detach()),
        "topic_loss": float(topic.total.detach()),
        "theta_shape": list(theta.shape),
        "sinkhorn_row_error": float(
            torch.max(torch.abs(sinkhorn.sum(dim=1) - 1.0)),
        ),
        "sinkhorn_topic_error": float(
            torch.max(
                torch.abs(
                    sinkhorn.sum(dim=0) - sinkhorn.shape[0] / sinkhorn.shape[1],
                ),
            ),
        ),
        "single_routing_pass": True,
        "local_vb_steps": 0,
        "candidate_audit": static_candidate_audit(protocol),
    }
    if output is not None:
        write_json(output, result)
    return result
