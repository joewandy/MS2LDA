"""Single-phase training loop for the pooled projected reference candidate.

The implementing agent should wire this as a separate benchmark command rather
than replacing the locked M1 path until real MSnLib gates pass.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .data import iter_row_batches, sparse_batch
from .simple_candidate import (
    PooledProjectedMS2LDA,
    assignment_information_loss,
    initialize_simple_candidate,
)
from .utils import read_json, write_json

if TYPE_CHECKING:
    import scipy.sparse as sp


def candidate_batch_loss(
    model: PooledProjectedMS2LDA,
    batch: Any,
    *,
    mi_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return likelihood plus the sole optional anti-collapse term."""
    output = model.infer_batch(batch)
    nll = model.sparse_completion_nll(output.theta, output.beta, batch)
    information = assignment_information_loss(output.theta)
    total = nll + float(mi_weight) * information
    return total, {
        "nll": float(nll.detach()),
        "information_regularizer": float(information.detach()),
    }


def train_simple_candidate(
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
    heartbeat: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Fit the pooled candidate with one AdamW phase and no graph products."""
    torch.set_num_threads(int(protocol["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    run = Path(run_dir)
    output = run / "simple_candidate_model"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        return read_json(complete_path)
    output.mkdir(parents=True, exist_ok=True)
    features = torch.from_numpy(np.load(run / "token_features/features.npy"))
    topics = int(protocol["model"]["num_topics"])
    model, initial_indices = initialize_simple_candidate(
        features,
        num_topics=topics,
        protocol=protocol,
    )
    config = protocol["simple_candidate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["maximum_epochs"])
    started = time.perf_counter()
    final_nll = float("nan")
    for epoch in range(epochs):
        model.train()
        losses = []
        for rows in iter_row_batches(
            train.shape[0],
            batch_size=int(config["batch_size"]),
            shuffle=True,
            seed=int(protocol["seed"]) + epoch,
        ):
            batch = sparse_batch(train, rows)
            optimizer.zero_grad(set_to_none=True)
            total, diagnostics = candidate_batch_loss(
                model,
                batch,
                mi_weight=float(config["mi_weight"]),
            )
            if not torch.isfinite(total):
                raise FloatingPointError("simple candidate produced non-finite loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("simple candidate produced non-finite gradient")
            optimizer.step()
            losses.append(diagnostics["nll"])
        final_nll = float(np.mean(losses))
        if heartbeat is not None:
            heartbeat(
                stage="train_simple_candidate",
                epoch=epoch + 1,
                maximum_epochs=epochs,
                elapsed_seconds=time.perf_counter() - started,
                train_nll=final_nll,
            )

    torch.save(model.state_dict(), output / "weights.pt")
    write_json(
        output / "model.json",
        {
            "class": "PooledProjectedMS2LDA",
            "num_topics": model.num_topics,
            "vocabulary_size": model.vocabulary_size,
            "input_dimensions": model.input_dimensions,
            "projection_dimensions": model.projection_dimensions,
            "theta_temperature": model.theta_temperature,
            "beta_temperature": model.beta_temperature,
            "mi_weight": float(config["mi_weight"]),
            "topic_initial_indices": initial_indices.tolist(),
        },
    )
    shutil.copy2(run / "data/vocabulary.json", output / "vocabulary.json")
    result = {
        "fitting_seconds": float(time.perf_counter() - started),
        "final_train_nll": final_nll,
        "learned_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    write_json(complete_path, result)
    return result
