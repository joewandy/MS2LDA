"""Training objectives and anti-collapse operations for neural MS2LDA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as nnf

from .model import AssignmentOutput, NeuralMS2LDA

if TYPE_CHECKING:
    from .data import SparseBatch

MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class RouterLossTerms:
    """Differentiable router objective and its two routed views."""

    total: torch.Tensor
    completion: torch.Tensor
    sinkhorn: torch.Tensor
    consistency: torch.Tensor
    left: AssignmentOutput
    right: AssignmentOutput


@dataclass(frozen=True)
class TopicLossTerms:
    """Decoder/prototype objective evaluated at fixed assignments."""

    total: torch.Tensor
    completion: torch.Tensor
    local_decoder: torch.Tensor
    beta: torch.Tensor


def cooccurrence_topic_loss(
    model: NeuralMS2LDA,
    graph: torch.Tensor,
    *,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Penalize topics that assign little mass to positive-NPMI neighbours."""
    if graph.layout != torch.sparse_coo or graph.shape != (
        model.vocabulary_size,
        model.vocabulary_size,
    ):
        raise ValueError("co-occurrence graph does not match the model vocabulary")
    propagated = torch.sparse.mm(graph, beta.T)
    affinity = torch.sum(beta.T * propagated, dim=0)
    return -torch.mean(torch.log(affinity.clamp_min(1e-12)))


def topic_separation_loss(
    model: NeuralMS2LDA,
    *,
    neighbors: int,
    margin: float,
) -> torch.Tensor:
    """Penalize nearest prototype cosine similarities above ``margin``."""
    if not 0 < neighbors < model.num_topics:
        raise ValueError("nearest-neighbour count must be between zero and K")
    if not -1.0 < margin < 1.0:
        raise ValueError("topic-separation margin must be inside (-1, 1)")
    topics = nnf.normalize(model.topic_prototypes, dim=1)
    similarities = topics @ topics.T
    diagonal = torch.eye(model.num_topics, dtype=torch.bool, device=similarities.device)
    nearest = torch.topk(
        similarities.masked_fill(diagonal, float("-inf")),
        k=int(neighbors),
        dim=1,
    ).values
    return torch.mean(torch.square(nnf.relu(nearest - float(margin))))


def balanced_sinkhorn_targets(
    logits: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Project routing scores onto equal-mass topics in log space.

    The returned matrix has unit row sums and aggregate column mass ``N / K``.
    It is used only as a detached training target; inference remains a single
    top-k softmax pass and therefore has no Sinkhorn iterations.
    """
    if logits.ndim != MATRIX_DIMENSIONS or not logits.numel():
        raise ValueError("Sinkhorn logits must be a non-empty matrix")
    observations, topics = logits.shape
    log_kernel = logits / float(epsilon)
    log_row_mass = logits.new_full((observations,), -math.log(observations))
    log_topic_mass = logits.new_full((topics,), -math.log(topics))
    log_u = torch.zeros_like(log_row_mass)
    log_v = torch.zeros_like(log_topic_mass)
    for _ in range(int(iterations)):
        log_u = log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_topic_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return plan * observations


def _theta_consistency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the Jensen-Shannon divergence between paired-view mixtures."""
    midpoint = 0.5 * (left + right)
    left_kl = torch.sum(
        left
        * (torch.log(left.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))),
        dim=1,
    )
    right_kl = torch.sum(
        right
        * (torch.log(right.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))),
        dim=1,
    )
    return 0.5 * torch.mean(left_kl + right_kl)


def router_block_loss(  # noqa: PLR0913
    model: NeuralMS2LDA,
    left_batch: SparseBatch,
    right_batch: SparseBatch,
    *,
    cached_beta: torch.Tensor,
    temperature: float,
    sinkhorn_weight: float,
    consistency_weight: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> RouterLossTerms:
    """Optimize routing with completion, balanced usage, and view agreement.

    ``beta`` is detached for this block. Cross-view completion teaches each
    partial spectrum to predict the other; Sinkhorn targets resist collapse;
    Jensen-Shannon consistency makes the paired views represent one spectrum.
    """
    left = model.route(left_batch, temperature=temperature, straight_through=True)
    right = model.route(right_batch, temperature=temperature, straight_through=True)
    completion = 0.5 * (
        model.sparse_completion_nll(left.theta, cached_beta, right_batch)
        + model.sparse_completion_nll(right.theta, cached_beta, left_batch)
    )
    sinkhorn_terms = []
    for routed in (left, right):
        with torch.no_grad():
            targets = balanced_sinkhorn_targets(
                routed.logits.detach(),
                epsilon=sinkhorn_epsilon,
                iterations=sinkhorn_iterations,
            )
        log_probabilities = nnf.log_softmax(routed.logits / float(temperature), dim=1)
        sinkhorn_terms.append(
            -torch.mean(torch.sum(targets * log_probabilities, dim=1))
        )
    sinkhorn = 0.5 * (sinkhorn_terms[0] + sinkhorn_terms[1])
    consistency = _theta_consistency(left.theta, right.theta)
    total = completion + sinkhorn_weight * sinkhorn + consistency_weight * consistency
    return RouterLossTerms(
        total=total,
        completion=completion,
        sinkhorn=sinkhorn,
        consistency=consistency,
        left=left,
        right=right,
    )


def _local_decoder_loss(
    beta: torch.Tensor,
    output: AssignmentOutput,
    batch: SparseBatch,
) -> torch.Tensor:
    """Make a routed topic emit the token that selected it."""
    log_emission = torch.log(beta[:, batch.indices].T.clamp_min(1e-12))
    per_token = -torch.sum(output.assignments.detach() * log_emission, dim=1)
    return torch.sum(batch.weights * per_token) / batch.weights.sum().clamp_min(1.0)


def topic_block_loss(  # noqa: PLR0913
    model: NeuralMS2LDA,
    left_batch: SparseBatch,
    right_batch: SparseBatch,
    *,
    temperature: float,
    local_decoder_weight: float,
) -> TopicLossTerms:
    """Optimize topic geometry against fixed one-pass assignments.

    Detaching the routes makes this the decoder/prototype half of an alternating
    optimization. Cross-view completion supplies the probabilistic objective;
    the smaller local term keeps each selected prototype faithful to its token.
    """
    with torch.no_grad():
        projected = model.projected_tokens()
        left = model.route(
            left_batch,
            temperature=temperature,
            straight_through=False,
            projected_tokens=projected,
        )
        right = model.route(
            right_batch,
            temperature=temperature,
            straight_through=False,
            projected_tokens=projected,
        )
    beta = model.topic_word_distribution()
    completion = 0.5 * (
        model.sparse_completion_nll(left.theta.detach(), beta, right_batch)
        + model.sparse_completion_nll(right.theta.detach(), beta, left_batch)
    )
    local = 0.5 * (
        _local_decoder_loss(beta, left, left_batch)
        + _local_decoder_loss(beta, right, right_batch)
    )
    return TopicLossTerms(
        total=completion + float(local_decoder_weight) * local,
        completion=completion,
        local_decoder=local,
        beta=beta,
    )


def _reset_optimizer_rows(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    rows: torch.Tensor,
) -> None:
    """Clear Adam-like state for deterministically replaced parameter rows."""
    state = optimizer.state.get(parameter, {})
    with torch.no_grad():
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows] = 0


def recycle_dead_prototypes(
    model: NeuralMS2LDA,
    optimizer: torch.optim.Optimizer,
    *,
    topic_indices: torch.Tensor,
    replacements: torch.Tensor,
) -> None:
    """Replace named underused prototypes and clear their stale optimizer state."""
    expected = (len(topic_indices), model.projection_dimensions)
    if topic_indices.ndim != 1 or replacements.shape != expected:
        raise ValueError("recycling indices and replacements do not align")
    if len(torch.unique(topic_indices)) != len(topic_indices):
        raise ValueError("a prototype cannot be recycled twice in one operation")
    with torch.no_grad():
        model.topic_prototypes[topic_indices] = nnf.normalize(replacements, dim=1)
    _reset_optimizer_rows(optimizer, model.topic_prototypes, topic_indices)
