"""PyTorch port of the reference Neural Sinkhorn Topic Model (NSTM).

The model equations and defaults follow Zhao et al., ICLR 2021, and the
authors' MIT-licensed TensorFlow implementation at commit
``610d1604d5467289028714ed0ce684dfb5ef8a7b``:
https://github.com/ethanhezhao/NeuralSinkhornTopicModel

The paper defines L1-normalized document distributions.  The released script
instead feeds raw counts to the encoder/reconstruction term and applies a
softmax to counts for the transport marginal.  Both behaviours are explicit
here so that domain experiments cannot silently conflate the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

DOCUMENT_INPUT_MODES = ("paper_l1", "released_code")
DocumentInputMode = Literal["paper_l1", "released_code"]
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class PreparedDocuments:
    """Encoder input and the two word-space targets used by NSTM."""

    encoder_input: torch.Tensor
    transport_distribution: torch.Tensor
    reconstruction_weights: torch.Tensor


@dataclass(frozen=True)
class SinkhornResult:
    """Per-document entropic OT costs and convergence diagnostics."""

    cost: torch.Tensor
    iterations: int
    marginal_error: float


@dataclass(frozen=True)
class NSTMOutput:
    """Differentiable NSTM objective and its inspectable components."""

    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sinkhorn_loss: torch.Tensor
    theta: torch.Tensor
    sinkhorn_iterations: int
    sinkhorn_marginal_error: float


def prepare_documents(
    counts: torch.Tensor,
    mode: DocumentInputMode,
) -> PreparedDocuments:
    """Prepare document inputs according to the paper or released script."""
    if counts.ndim != MATRIX_DIMENSIONS or counts.shape[1] == 0:
        raise ValueError("counts must be a non-empty document-word matrix")
    if not torch.all(torch.isfinite(counts)) or torch.any(counts < 0):
        raise ValueError("counts must be finite and non-negative")
    totals = counts.sum(dim=1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("NSTM does not accept empty documents")
    if mode == "paper_l1":
        distribution = counts / totals
        return PreparedDocuments(
            encoder_input=distribution,
            transport_distribution=distribution,
            reconstruction_weights=distribution,
        )
    if mode == "released_code":
        return PreparedDocuments(
            encoder_input=counts,
            transport_distribution=nnf.softmax(counts, dim=1),
            reconstruction_weights=counts,
        )
    raise ValueError(f"unknown NSTM document input mode: {mode}")


def reference_sinkhorn_cost(
    ground_cost: torch.Tensor,
    topic_mass: torch.Tensor,
    word_mass: torch.Tensor,
    *,
    alpha: float = 20.0,
    maximum_iterations: int = 1000,
    stop_tolerance: float = 0.005,
    check_interval: int = 20,
) -> SinkhornResult:
    """Compute the differentiable Sinkhorn cost used by reference NSTM.

    ``ground_cost`` is K by V while ``topic_mass`` and ``word_mass`` are B by
    K and B by V.  The primal scaling iterations intentionally mirror the
    released algorithm.  Denominators are clamped only at the dtype's smallest
    normal value to turn silent division-by-zero into a finite numerical
    contract without changing ordinary reference computations.
    """
    if ground_cost.ndim != MATRIX_DIMENSIONS:
        raise ValueError("ground_cost must be a topic-word matrix")
    if topic_mass.ndim != MATRIX_DIMENSIONS or word_mass.ndim != MATRIX_DIMENSIONS:
        raise ValueError("topic_mass and word_mass must be matrices")
    if topic_mass.shape[0] != word_mass.shape[0]:
        raise ValueError("topic and word marginals must have the same batch size")
    if ground_cost.shape != (topic_mass.shape[1], word_mass.shape[1]):
        raise ValueError("ground cost dimensions do not match the marginals")
    if float(alpha) <= 0 or int(maximum_iterations) <= 0:
        raise ValueError("alpha and maximum_iterations must be positive")
    if float(stop_tolerance) < 0 or int(check_interval) <= 0:
        raise ValueError("stop tolerance and check interval are invalid")
    tensors = (ground_cost, topic_mass, word_mass)
    if any(not torch.all(torch.isfinite(value)) for value in tensors):
        raise ValueError("Sinkhorn inputs must be finite")
    has_negative = (
        torch.any(ground_cost < 0)
        or torch.any(topic_mass < 0)
        or torch.any(word_mass < 0)
    )
    if has_negative:
        raise ValueError("Sinkhorn costs and marginals must be non-negative")
    topic_totals = topic_mass.sum(dim=1)
    word_totals = word_mass.sum(dim=1)
    if not torch.allclose(topic_totals, word_totals, atol=1e-5, rtol=1e-5):
        raise ValueError("topic and word marginals must carry equal mass")
    if torch.any(topic_totals <= 0):
        raise ValueError("Sinkhorn marginals must carry positive mass")

    topics = topic_mass.T
    words = word_mass.T
    kernel = torch.exp(-ground_cost * float(alpha))
    floor = torch.finfo(kernel.dtype).tiny
    left = torch.ones_like(topics) / topics.shape[0]
    right = torch.ones_like(words) / words.shape[0]
    marginal_error = float("inf")
    completed_iterations = 0
    for iteration in range(int(maximum_iterations)):
        right = words / torch.matmul(kernel.T, left).clamp_min(floor)
        left = topics / torch.matmul(kernel, right).clamp_min(floor)
        completed_iterations = iteration + 1
        should_check = iteration % int(check_interval) == 0
        should_check = should_check or completed_iterations == int(maximum_iterations)
        if should_check:
            reconstructed_words = right * torch.matmul(kernel.T, left)
            error = torch.sum(torch.abs(reconstructed_words - words), dim=0).max()
            marginal_error = float(error.detach().cpu())
            if marginal_error <= float(stop_tolerance):
                break

    cost = torch.sum(
        left * torch.matmul(kernel * ground_cost, right),
        dim=0,
    )
    if not torch.all(torch.isfinite(cost)):
        raise FloatingPointError("NSTM Sinkhorn iterations produced a non-finite cost")
    return SinkhornResult(
        cost=cost,
        iterations=completed_iterations,
        marginal_error=marginal_error,
    )


class NeuralSinkhornTopicModel(nn.Module):
    """Reference NSTM with fixed pretrained word embeddings."""

    def __init__(
        self,
        pretrained_word_embeddings: np.ndarray,
        topics: int,
        *,
        hidden: int = 200,
        dropout: float = 0.25,
        reconstruction_weight: float = 0.07,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_maximum_iterations: int = 1000,
        sinkhorn_stop_tolerance: float = 0.005,
        input_mode: DocumentInputMode = "paper_l1",
    ) -> None:
        super().__init__()
        embeddings = np.asarray(pretrained_word_embeddings, dtype=np.float32)
        if embeddings.ndim != MATRIX_DIMENSIONS or min(embeddings.shape) <= 0:
            raise ValueError("word embeddings must be a non-empty matrix")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("word embeddings must be finite")
        if np.any(np.linalg.norm(embeddings, axis=1) <= 0):
            raise ValueError("every word embedding must have non-zero norm")
        if int(topics) <= 1 or int(hidden) <= 0:
            message = "NSTM needs at least two topics and a positive hidden size"
            raise ValueError(message)
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if float(reconstruction_weight) < 0:
            raise ValueError("reconstruction_weight must be non-negative")
        if input_mode not in DOCUMENT_INPUT_MODES:
            raise ValueError(f"unknown NSTM document input mode: {input_mode}")

        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.register_buffer(
            "word_embeddings",
            torch.from_numpy(normalized.astype(np.float32, copy=False)),
        )
        self.encoder_hidden = nn.Linear(embeddings.shape[0], int(hidden))
        self.encoder_dropout = nn.Dropout(float(dropout))
        self.encoder_topics = nn.Linear(int(hidden), int(topics))
        self.encoder_batch_norm = nn.BatchNorm1d(int(topics))
        self.encoder_batch_norm.weight.requires_grad_(False)
        self.topic_embeddings = nn.Parameter(
            torch.empty((int(topics), embeddings.shape[1]), dtype=torch.float32),
        )
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.input_mode = input_mode
        self.reconstruction_weight = float(reconstruction_weight)
        self.sinkhorn_alpha = float(sinkhorn_alpha)
        self.sinkhorn_maximum_iterations = int(sinkhorn_maximum_iterations)
        self.sinkhorn_stop_tolerance = float(sinkhorn_stop_tolerance)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of fixed word embeddings."""
        return int(self.word_embeddings.shape[0])

    @property
    def num_topics(self) -> int:
        """Return the fitted topic count."""
        return int(self.topic_embeddings.shape[0])

    def topic_word_similarity(self) -> torch.Tensor:
        """Return the reference cosine-similarity topic-word matrix."""
        topics = nnf.normalize(self.topic_embeddings, dim=1)
        return torch.matmul(topics, self.word_embeddings.T)

    def beta(self) -> torch.Tensor:
        """Return per-topic virtual-decoder word probabilities for reporting."""
        return nnf.softmax(self.topic_word_similarity(), dim=1)

    def theta_from_encoder_input(self, encoder_input: torch.Tensor) -> torch.Tensor:
        """Infer topic proportions from already prepared encoder inputs."""
        hidden = nnf.relu(self.encoder_hidden(encoder_input))
        hidden = self.encoder_dropout(hidden)
        logits = self.encoder_batch_norm(self.encoder_topics(hidden))
        return nnf.softmax(logits, dim=1)

    def theta(self, counts: torch.Tensor) -> torch.Tensor:
        """Infer topic proportions from non-negative document counts."""
        prepared = prepare_documents(counts, self.input_mode)
        return self.theta_from_encoder_input(prepared.encoder_input)

    def decode_log_probabilities(
        self,
        theta: torch.Tensor,
        *,
        similarity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the exact virtual-decoder log probabilities from the paper."""
        values = self.topic_word_similarity() if similarity is None else similarity
        return nnf.log_softmax(torch.matmul(theta, values), dim=1)

    def forward(self, counts: torch.Tensor) -> NSTMOutput:
        """Evaluate the joint reconstruction plus Sinkhorn objective."""
        prepared = prepare_documents(counts, self.input_mode)
        theta = self.theta_from_encoder_input(prepared.encoder_input)
        similarity = self.topic_word_similarity()
        sinkhorn = reference_sinkhorn_cost(
            1.0 - similarity,
            theta,
            prepared.transport_distribution,
            alpha=self.sinkhorn_alpha,
            maximum_iterations=self.sinkhorn_maximum_iterations,
            stop_tolerance=self.sinkhorn_stop_tolerance,
        )
        reconstruction = -torch.sum(
            prepared.reconstruction_weights
            * self.decode_log_probabilities(theta, similarity=similarity),
            dim=1,
        )
        reconstruction_loss = reconstruction.mean()
        sinkhorn_loss = sinkhorn.cost.mean()
        loss = self.reconstruction_weight * reconstruction_loss + sinkhorn_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("NSTM produced a non-finite objective")
        return NSTMOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sinkhorn_loss=sinkhorn_loss,
            theta=theta,
            sinkhorn_iterations=sinkhorn.iterations,
            sinkhorn_marginal_error=sinkhorn.marginal_error,
        )
