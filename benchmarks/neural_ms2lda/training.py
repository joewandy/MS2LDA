"""Deterministic optimization of the single supported neural MS2LDA model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .artifacts import save_trained_model
from .data import load_vocabulary
from .model import initialize_model
from .objectives import prepare_cooccurrence_graph, torch_sparse_graph
from .optimization import (
    TrainingState,
    router_phase,
    routing_temperature,
    sinkhorn_weight,
    topic_phase,
)
from .utils import read_json, write_json

if TYPE_CHECKING:
    import scipy.sparse as sp


def _new_training_state(
    run: Path, train: sp.csr_matrix, protocol: dict[str, Any]
) -> TrainingState:
    """Construct the sole deterministic initialization and optimizer."""
    seed = int(protocol["seed"])
    topics = int(protocol["model"]["num_topics"])
    torch.manual_seed(seed)
    features = torch.from_numpy(np.load(run / "token_features/features.npy"))
    model, _ = initialize_model(
        features,
        num_topics=topics,
        protocol=protocol,
    )
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    return TrainingState(model=model, optimizer=optimizer)


def train_model(
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
    heartbeat: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Fit the fixed 40-epoch model from its deterministic initialization."""
    torch.set_num_threads(int(protocol["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    run = Path(run_dir)
    output = run / "trained_model"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        return read_json(complete_path)
    output.mkdir(parents=True, exist_ok=True)
    graph = prepare_cooccurrence_graph(run, train=train, protocol=protocol)
    graph_tensor = torch_sparse_graph(graph)
    state = _new_training_state(run, train, protocol)
    optimization = protocol["optimization"]
    maximum_epochs = int(optimization["maximum_epochs"])
    started = time.perf_counter()
    for epoch in range(maximum_epochs):
        state.model.train()
        temperature = routing_temperature(epoch, protocol)
        router_phase(
            state,
            train=train,
            epoch=epoch,
            temperature=temperature,
            balance_weight=sinkhorn_weight(epoch, protocol),
            protocol=protocol,
        )
        topic_phase(
            state,
            train=train,
            epoch=epoch,
            temperature=temperature,
            graph_tensor=graph_tensor,
            protocol=protocol,
        )
        completed_epoch = epoch + 1
        elapsed = time.perf_counter() - started
        if heartbeat is not None:
            heartbeat(
                stage="train_neural",
                epoch=completed_epoch,
                maximum_epochs=maximum_epochs,
                elapsed_seconds=elapsed,
            )
    save_trained_model(
        output,
        state.model,
        load_vocabulary(run / "data"),
        routing_temperature=routing_temperature(maximum_epochs, protocol),
    )
    result = {"fitting_seconds": float(elapsed)}
    write_json(complete_path, result)
    return result
