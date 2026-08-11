# ruff: noqa: N812, PLR0913, PLR2004
"""One-pass peak-to-topic routing model and alternating neural objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .data import SparseBatch


@dataclass(frozen=True)
class AssignmentOutput:
    """Sparse-forward token assignments and their document aggregation."""

    theta: torch.Tensor
    assignments: torch.Tensor
    soft_assignments: torch.Tensor
    logits: torch.Tensor
    route_embeddings: torch.Tensor
    row_ids: torch.Tensor
    token_indices: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class RouterLossTerms:
    """Router-block loss terms plus queue diagnostics."""

    total: torch.Tensor
    completion: torch.Tensor
    sinkhorn: torch.Tensor
    consistency: torch.Tensor
    left: AssignmentOutput
    right: AssignmentOutput


@dataclass(frozen=True)
class TopicLossTerms:
    """Topic-block exact completion and local decoder terms."""

    total: torch.Tensor
    completion: torch.Tensor
    local_decoder: torch.Tensor


def balanced_sinkhorn_targets(
    logits: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return row-normalized assignments with equal aggregate topic mass."""
    if logits.ndim != 2 or not logits.numel():
        msg = "Sinkhorn logits must be a non-empty matrix"
        raise ValueError(msg)
    observations, topics = logits.shape
    log_kernel = logits / float(epsilon)
    log_row_mass = logits.new_full((observations,), -math.log(observations))
    log_topic_mass = logits.new_full((topics,), -math.log(topics))
    log_u = torch.zeros_like(log_row_mass)
    log_v = torch.zeros_like(log_topic_mass)
    for _ in range(int(iterations)):
        log_u = log_row_mass - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0),
            dim=1,
        )
        log_v = log_topic_mass - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1),
            dim=0,
        )
    plan = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return plan * observations


def deterministic_kmeans_plus_plus(
    features: torch.Tensor,
    *,
    clusters: int,
    seed: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose D-squared prototype seeds without Lloyd refinement."""
    if features.ndim != 2 or not features.numel():
        msg = "k-means++ features must be a non-empty matrix"
        raise ValueError(msg)
    observations = features.shape[0]
    if not 0 < clusters <= observations:
        msg = "cluster count must not exceed feature count"
        raise ValueError(msg)
    values = F.normalize(features.detach().to(dtype=torch.float64), dim=1)
    probabilities = (
        torch.ones(observations, dtype=torch.float64)
        if weights is None
        else weights.detach().to(dtype=torch.float64).clamp_min(0)
    )
    if float(probabilities.sum()) <= 0:
        msg = "k-means++ weights must have positive mass"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ),
    )
    selected = [first]
    closest = (2.0 - 2.0 * (values @ values[first])).clamp_min(0.0)
    closest[first] = 0.0
    for _ in range(1, clusters):
        sampling = closest * probabilities
        sampling[torch.tensor(selected)] = 0.0
        if float(sampling.sum()) <= 1e-15:
            remaining = torch.ones(observations, dtype=torch.bool)
            remaining[torch.tensor(selected)] = False
            next_index = int(torch.flatnonzero(remaining)[0])
        else:
            next_index = int(
                torch.multinomial(
                    sampling,
                    num_samples=1,
                    generator=generator,
                ),
            )
        selected.append(next_index)
        distance = (2.0 - 2.0 * (values @ values[next_index])).clamp_min(0.0)
        closest = torch.minimum(closest, distance)
        closest[torch.tensor(selected)] = 0.0
    return torch.tensor(selected, dtype=torch.int64)


class NeuralAssignmentMS2LDA(nn.Module):
    """Route each observed token directly to learned neural topic prototypes."""

    def __init__(
        self,
        token_features: torch.Tensor,
        *,
        num_topics: int,
        projection_dimensions: int,
        router_hidden_dimensions: int,
        beta_temperature: float,
        topic_initial_indices: torch.Tensor,
        seed: int,
    ) -> None:
        super().__init__()
        if token_features.ndim != 2:
            msg = "token features must be a matrix"
            raise ValueError(msg)
        if len(topic_initial_indices) != num_topics:
            msg = "topic initialization does not match topic count"
            raise ValueError(msg)
        self.num_topics = int(num_topics)
        self.vocabulary_size = int(token_features.shape[0])
        self.input_dimensions = int(token_features.shape[1])
        self.projection_dimensions = int(projection_dimensions)
        self.beta_temperature = float(beta_temperature)
        self.register_buffer("token_features", token_features.detach().clone())

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.token_projection = nn.Linear(
                self.input_dimensions,
                self.projection_dimensions,
                bias=False,
            )
            nn.init.orthogonal_(self.token_projection.weight)
            self.context_router = nn.Sequential(
                nn.Linear(2 * self.projection_dimensions, router_hidden_dimensions),
                nn.GELU(),
                nn.LayerNorm(router_hidden_dimensions),
                nn.Linear(router_hidden_dimensions, self.projection_dimensions),
            )
            nn.init.normal_(self.context_router[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.context_router[-1].bias)

        with torch.no_grad():
            projected = self.projected_tokens()
            initial = projected[topic_initial_indices].clone()
        self.topic_prototypes = nn.Parameter(initial)

    def projected_tokens(self) -> torch.Tensor:
        """Project the fixed 64-D token table into normalized neural geometry."""
        return F.normalize(self.token_projection(self.token_features), dim=1)

    def topic_word_distribution(
        self,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Derive exact row-normalized beta from the shared neural geometry."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        topics = F.normalize(self.topic_prototypes, dim=1)
        logits = 2.0 * topics @ tokens.T / self.beta_temperature
        return F.softmax(logits, dim=1)

    def _route_embeddings(
        self,
        batch: SparseBatch,
        projected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_values = projected_tokens[batch.indices]
        weighted = token_values * batch.weights.unsqueeze(1)
        document_sums = token_values.new_zeros(
            (batch.documents, self.projection_dimensions),
        )
        document_sums.index_add_(0, batch.row_ids, weighted)
        context_numerator = document_sums[batch.row_ids] - weighted
        context_denominator = (
            batch.document_totals[batch.row_ids] - batch.weights
        ).clamp_min(1.0)
        context = context_numerator / context_denominator.unsqueeze(1)
        correction = self.context_router(torch.cat((token_values, context), dim=1))
        return F.normalize(token_values + correction, dim=1)

    @staticmethod
    def aggregate_theta(
        assignments: torch.Tensor,
        *,
        row_ids: torch.Tensor,
        weights: torch.Tensor,
        documents: int,
    ) -> torch.Tensor:
        """Aggregate count-weighted token assignments into document mixtures."""
        topic_mass = assignments.new_zeros((documents, assignments.shape[1]))
        topic_mass.index_add_(0, row_ids, assignments * weights.unsqueeze(1))
        totals = topic_mass.sum(dim=1, keepdim=True)
        theta = topic_mass / totals.clamp_min(1e-12)
        empty = totals.squeeze(1) <= 0
        if torch.any(empty):
            theta = theta.clone()
            theta[empty] = 1.0 / assignments.shape[1]
        return theta

    def route(
        self,
        batch: SparseBatch,
        *,
        temperature: float,
        top_k: int,
        straight_through: bool,
        projected_tokens: torch.Tensor | None = None,
    ) -> AssignmentOutput:
        """Perform one deterministic token-routing pass."""
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        routes = self._route_embeddings(batch, tokens)
        topics = F.normalize(self.topic_prototypes, dim=1)
        logits = routes @ topics.T
        soft = F.softmax(logits / float(temperature), dim=1)
        selected_k = min(int(top_k), self.num_topics)
        indices = torch.topk(soft, k=selected_k, dim=1).indices
        values = torch.gather(soft, 1, indices)
        values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        hard = torch.zeros_like(soft).scatter(1, indices, values)
        assignments = hard + soft - soft.detach() if straight_through else hard
        theta = self.aggregate_theta(
            assignments,
            row_ids=batch.row_ids,
            weights=batch.weights,
            documents=batch.documents,
        )
        return AssignmentOutput(
            theta=theta,
            assignments=assignments,
            soft_assignments=soft,
            logits=logits,
            route_embeddings=routes,
            row_ids=batch.row_ids,
            token_indices=batch.indices,
            weights=batch.weights,
        )

    @staticmethod
    def sparse_completion_nll(
        theta: torch.Tensor,
        beta: torch.Tensor,
        target: SparseBatch,
    ) -> torch.Tensor:
        """Score exact theta-times-beta probabilities at nonzero target tokens."""
        selected_topics = theta[target.row_ids]
        selected_words = beta[:, target.indices].T
        probability = torch.sum(selected_topics * selected_words, dim=1).clamp_min(
            1e-12,
        )
        return -torch.sum(
            target.weights * torch.log(probability),
        ) / target.weights.sum().clamp_min(1.0)


def _theta_consistency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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


def router_block_loss(
    model: NeuralAssignmentMS2LDA,
    left_batch: SparseBatch,
    right_batch: SparseBatch,
    *,
    cached_beta: torch.Tensor,
    temperature: float,
    top_k: int,
    sinkhorn_weight: float,
    consistency_weight: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> RouterLossTerms:
    """Compute symmetric masked completion, balanced routing, and consistency."""
    left = model.route(
        left_batch,
        temperature=temperature,
        top_k=top_k,
        straight_through=True,
    )
    right = model.route(
        right_batch,
        temperature=temperature,
        top_k=top_k,
        straight_through=True,
    )
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
        log_probabilities = F.log_softmax(
            routed.logits / float(temperature),
            dim=1,
        )
        sinkhorn_terms.append(
            -torch.mean(torch.sum(targets * log_probabilities, dim=1)),
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
) -> torch.Tensor:
    log_emission = torch.log(beta[:, output.token_indices].T.clamp_min(1e-12))
    per_token = -torch.sum(output.assignments.detach() * log_emission, dim=1)
    return torch.sum(output.weights * per_token) / output.weights.sum().clamp_min(1.0)


def topic_block_loss(
    model: NeuralAssignmentMS2LDA,
    left_batch: SparseBatch,
    right_batch: SparseBatch,
    *,
    temperature: float,
    top_k: int,
    local_decoder_weight: float,
) -> TopicLossTerms:
    """Update beta geometry against detached one-pass assignments."""
    with torch.no_grad():
        projected = model.projected_tokens()
        left = model.route(
            left_batch,
            temperature=temperature,
            top_k=top_k,
            straight_through=False,
            projected_tokens=projected,
        )
        right = model.route(
            right_batch,
            temperature=temperature,
            top_k=top_k,
            straight_through=False,
            projected_tokens=projected,
        )
    beta = model.topic_word_distribution()
    completion = 0.5 * (
        model.sparse_completion_nll(left.theta.detach(), beta, right_batch)
        + model.sparse_completion_nll(right.theta.detach(), beta, left_batch)
    )
    local = 0.5 * (_local_decoder_loss(beta, left) + _local_decoder_loss(beta, right))
    return TopicLossTerms(
        total=completion + float(local_decoder_weight) * local,
        completion=completion,
        local_decoder=local,
    )


def reset_optimizer_rows(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    rows: torch.Tensor,
) -> None:
    """Clear Adam-like optimizer state for deterministically replaced rows."""
    state = optimizer.state.get(parameter, {})
    with torch.no_grad():
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows] = 0


def recycle_dead_prototypes(
    model: NeuralAssignmentMS2LDA,
    optimizer: torch.optim.Optimizer,
    *,
    topic_indices: torch.Tensor,
    replacements: torch.Tensor,
) -> None:
    """Replace only named dead prototype rows and reset their optimizer state."""
    if topic_indices.ndim != 1 or replacements.shape != (
        len(topic_indices),
        model.projection_dimensions,
    ):
        msg = "recycling indices and replacements do not align"
        raise ValueError(msg)
    unique = torch.unique(topic_indices)
    if len(unique) != len(topic_indices):
        msg = "a prototype cannot be recycled twice in one operation"
        raise ValueError(msg)
    with torch.no_grad():
        model.topic_prototypes[topic_indices] = F.normalize(replacements, dim=1)
    reset_optimizer_rows(optimizer, model.topic_prototypes, topic_indices)


def initialize_model(
    token_features: torch.Tensor,
    *,
    num_topics: int,
    protocol: dict[str, Any],
    seeding_weights: np.ndarray | None = None,
) -> tuple[NeuralAssignmentMS2LDA, torch.Tensor]:
    """Create a deterministic data-only k-means++ seed state."""
    seed = int(protocol["seed"])
    model_config = protocol["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + int(num_topics))
        temporary_projection = nn.Linear(
            int(model_config["input_dimensions"]),
            int(model_config["projection_dimensions"]),
            bias=False,
        )
        nn.init.orthogonal_(temporary_projection.weight)
    projected = F.normalize(temporary_projection(token_features), dim=1)
    weights = (
        torch.from_numpy(np.asarray(seeding_weights, dtype=np.float64))
        if seeding_weights is not None
        else None
    )
    initial_indices = deterministic_kmeans_plus_plus(
        projected,
        clusters=int(num_topics),
        seed=seed + 4049 + int(num_topics),
        weights=weights,
    )
    model = NeuralAssignmentMS2LDA(
        token_features,
        num_topics=int(num_topics),
        projection_dimensions=int(model_config["projection_dimensions"]),
        router_hidden_dimensions=int(model_config["router_hidden_dimensions"]),
        beta_temperature=float(model_config["beta_temperature"]),
        topic_initial_indices=initial_indices,
        seed=seed + int(num_topics),
    )
    return model, initial_indices
