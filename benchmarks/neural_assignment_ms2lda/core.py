"""Shared inference, schedules, and immutable initialization primitives."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import (
    build_token_features,
    iter_sparse_batches,
    load_vocabulary,
    prototype_seeding_weights,
)
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
    sparse_npmi,
    top_word_diversity,
)
from .model import NeuralAssignmentMS2LDA, initialize_model
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


@dataclass
class HardContextQueue:
    """Bounded deterministic queue of high-loss routing contexts."""

    capacity: int
    heap: list[tuple[float, int, torch.Tensor]]
    serial: int = 0

    @classmethod
    def empty(cls, capacity: int) -> HardContextQueue:
        return cls(capacity=int(capacity), heap=[])

    def add(self, losses: torch.Tensor, contexts: torch.Tensor, *, limit: int) -> None:
        if not len(losses):
            return
        selected = torch.topk(
            losses.detach(), k=min(int(limit), len(losses)), largest=True
        ).indices
        for index in selected.tolist():
            item = (
                float(losses[index]),
                self.serial,
                contexts[index].detach().cpu().clone(),
            )
            self.serial += 1
            if len(self.heap) < self.capacity:
                heapq.heappush(self.heap, item)
            elif item[:2] > self.heap[0][:2]:
                heapq.heapreplace(self.heap, item)

    def pop_highest(self, count: int) -> torch.Tensor:
        selected = heapq.nlargest(min(int(count), len(self.heap)), self.heap)
        selected_ids = {serial for _, serial, _ in selected}
        self.heap = [item for item in self.heap if item[1] not in selected_ids]
        heapq.heapify(self.heap)
        if not selected:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack([item[2] for item in selected])

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "serial": self.serial,
            "items": [
                {"loss": loss, "serial": serial, "context": context}
                for loss, serial, context in self.heap
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HardContextQueue:
        queue = cls.empty(int(state["capacity"]))
        queue.serial = int(state["serial"])
        queue.heap = [
            (float(row["loss"]), int(row["serial"]), row["context"])
            for row in state["items"]
        ]
        heapq.heapify(queue.heap)
        return queue


def routing_temperature(epoch: int, protocol: dict[str, Any]) -> float:
    """Linearly anneal the fixed top-2 routing temperature."""
    config = protocol["anti_collapse"]
    start = float(config["routing_temperature_start"])
    end = float(config["routing_temperature_end"])
    progress = min(
        max(epoch, 0) / max(float(config["routing_temperature_anneal_epochs"]), 1.0),
        1.0,
    )
    return start + progress * (end - start)


def sinkhorn_weight(epoch: int, protocol: dict[str, Any]) -> float:
    """Return the frozen anti-collapse balance schedule."""
    config = protocol["anti_collapse"]
    start = float(config["sinkhorn_weight_start"])
    hold = int(config["sinkhorn_weight_hold_epochs"])
    end = float(config["sinkhorn_weight_end"])
    end_epoch = int(config["sinkhorn_weight_end_epoch"])
    if epoch < hold:
        return start
    progress = min(max((epoch - hold) / max(end_epoch - hold, 1), 0.0), 1.0)
    return start + progress * (end - start)


def prepare_token_features(
    run_dir: str | Path, *, counts_dir: str | Path, protocol: dict[str, Any]
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
        "schema_version": "neural-ms2lda/token-features-v1",
        "shape": list(features.shape),
        "output_sha256": {features_path.name: file_sha256(features_path)},
    }
    write_json(complete_path, result)
    return result


def prepare_initialization(
    run_dir: str | Path, *, train: sp.csr_matrix, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Create the deterministic data-only prototype initialization."""
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
            "schema_version": "neural-ms2lda/initialization-v1",
            "model": model.state_dict(),
            "topic_initial_indices": indices,
            "seed": int(protocol["seed"]),
            "method": "weighted_kmeans_plus_plus_seeding_without_lloyd_updates",
        },
    )
    result = {
        "schema_version": "neural-ms2lda/initialization-complete-v1",
        "num_topics": int(protocol["model"]["num_topics"]),
        "output_sha256": {
            checkpoint_path.name: file_sha256(checkpoint_path),
        },
    }
    write_json(complete_path, result)
    return result


def fresh_model(
    run_dir: str | Path, protocol: dict[str, Any]
) -> NeuralAssignmentMS2LDA:
    """Instantiate a model from the immutable initialization."""
    directory = Path(run_dir)
    features = torch.from_numpy(np.load(directory / "token_features/features.npy"))
    checkpoint = torch.load(
        directory / "initialization/model_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = protocol["model"]
    model = NeuralAssignmentMS2LDA(
        features,
        num_topics=int(config["num_topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        router_hidden_dimensions=int(config["router_hidden_dimensions"]),
        beta_temperature=float(config["beta_temperature"]),
        document_mixture_weight=float(config["document_mixture_weight"]),
        document_topic_prior_weight=float(config["document_topic_prior_weight"]),
        topic_initial_indices=checkpoint["topic_initial_indices"],
        seed=int(protocol["seed"]) + int(config["num_topics"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def infer_theta(
    model: NeuralAssignmentMS2LDA,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    temperature: float,
    top_k: int,
    with_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Infer each document mixture in one deterministic neural pass."""
    model.eval()
    projected = model.projected_tokens()
    rows: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    selected_masses: list[np.ndarray] = []
    for batch in iter_sparse_batches(
        matrix, batch_size=int(batch_size), shuffle=False, seed=0
    ):
        output = model.route(
            batch,
            temperature=float(temperature),
            top_k=int(top_k),
            straight_through=False,
            projected_tokens=projected,
        )
        rows.append(output.theta.cpu().numpy().astype(np.float32))
        if with_diagnostics:
            probabilities = output.assignments.clamp_min(1e-12)
            entropies.append(
                (-torch.sum(output.assignments * torch.log(probabilities), dim=1))
                .cpu()
                .numpy()
            )
            selected_masses.append(
                torch.max(output.assignments, dim=1).values.cpu().numpy()
            )
    theta = np.concatenate(rows, axis=0)
    if not with_diagnostics:
        return theta
    return theta, {
        "routing_passes_per_spectrum": 1,
        "local_vb_steps": 0,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "assignment_entropy_mean": float(np.mean(np.concatenate(entropies))),
        "maximum_assignment_mass_mean": float(np.mean(np.concatenate(selected_masses))),
    }


@torch.inference_mode()
def validation_metrics(
    model: NeuralAssignmentMS2LDA,
    *,
    train: sp.csr_matrix,
    validation_observed: sp.csr_matrix,
    validation_completion: sp.csr_matrix,
    validation_full: sp.csr_matrix,
    validation_records: list[dict[str, Any]],
    protocol: dict[str, Any],
    epoch: int | None = None,
    include_npmi: bool = False,
) -> dict[str, Any]:
    """Calculate selection metrics without reading the test partition."""
    batch_size = int(protocol["optimization"]["batch_size"])
    temperature = routing_temperature(
        int(protocol["optimization"]["maximum_epochs"] if epoch is None else epoch),
        protocol,
    )
    top_k = int(protocol["model"]["top_k"])
    beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
    observed_theta, routing = infer_theta(
        model,
        validation_observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
        with_diagnostics=True,
    )
    full_theta = infer_theta(
        model,
        validation_full,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    completion, _ = completion_metrics(
        observed_theta, beta, validation_completion, validation_records
    )
    result: dict[str, Any] = {
        "document_completion": completion,
        "active_topics": active_topic_metrics(
            observed_theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"]
            ),
            corpus_threshold=1.0 / model.num_topics,
        ),
        "mixture_diagnostics": effective_topic_summary(full_theta),
        "top_word_diversity": top_word_diversity(
            beta, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "routing": routing,
        "_usage": full_theta.mean(axis=0).astype(np.float32),
    }
    if include_npmi:
        result["word_cooccurrence_npmi"] = sparse_npmi(
            beta, train, top_n=int(protocol["evaluation"]["topic_top_n"])
        )
    return result
