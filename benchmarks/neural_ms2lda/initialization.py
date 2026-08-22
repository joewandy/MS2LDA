"""Deterministic token features and prototype initialization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import build_token_features, load_vocabulary, prototype_seeding_weights
from .model import NeuralMS2LDA, initialize_model
from .utils import (
    atomic_save_numpy,
    atomic_torch_save,
    file_sha256,
    read_json,
    verify_output_hashes,
    write_json,
)

if TYPE_CHECKING:
    import scipy.sparse as sp


def prepare_token_features(
    run_dir: str | Path,
    *,
    counts_dir: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Combine train-only SGNS, mass Fourier, and fragment/loss features."""
    directory = Path(run_dir)
    output = directory / "token_features"
    complete_path = output / "complete.json"
    features_path = output / "features.npy"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        return result
    embeddings = np.load(directory / "embeddings/embeddings.npy")
    vocabulary = load_vocabulary(counts_dir)
    features = build_token_features(embeddings, vocabulary, protocol["token_features"])
    atomic_save_numpy(features_path, features)
    result = {
        "shape": list(features.shape),
        "output_sha256": {features_path.name: file_sha256(features_path)},
    }
    write_json(complete_path, result)
    return result


def prepare_initialization(
    run_dir: str | Path,
    *,
    train: sp.csr_matrix,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Create the deterministic weighted k-means++ prototype initialization."""
    directory = Path(run_dir)
    output = directory / "initialization"
    checkpoint_path = output / "model_initialization.pt"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        return result
    features = torch.from_numpy(np.load(directory / "token_features/features.npy"))
    model, indices = initialize_model(
        features,
        num_topics=int(protocol["model"]["num_topics"]),
        protocol=protocol,
        seeding_weights=prototype_seeding_weights(train),
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "model": model.state_dict(),
            "topic_initial_indices": indices,
            "seed": int(protocol["seed"]),
            "method": "weighted_kmeans_plus_plus_without_lloyd_updates",
        },
    )
    result = {
        "num_topics": int(protocol["model"]["num_topics"]),
        "output_sha256": {checkpoint_path.name: file_sha256(checkpoint_path)},
    }
    write_json(complete_path, result)
    return result


def fresh_model(run_dir: str | Path, protocol: dict[str, Any]) -> NeuralMS2LDA:
    """Restore the immutable initialization into the final architecture."""
    directory = Path(run_dir)
    features = torch.from_numpy(np.load(directory / "token_features/features.npy"))
    checkpoint = torch.load(
        directory / "initialization/model_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = protocol["model"]
    model = NeuralMS2LDA(
        features,
        num_topics=int(config["num_topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        router_hidden_dimensions=int(config["router_hidden_dimensions"]),
        beta_temperature=float(config["beta_temperature"]),
        topic_initial_indices=checkpoint["topic_initial_indices"],
        seed=int(protocol["seed"]) + int(config["num_topics"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model
