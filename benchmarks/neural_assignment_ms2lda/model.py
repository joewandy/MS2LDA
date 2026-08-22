"""One-pass peak-to-topic routing model and probabilistic decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

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
    values = nnf.normalize(features.detach().to(dtype=torch.float64), dim=1)
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
    """Route spectrum tokens to prototypes and decode topics over vocabulary.

    The model learns a shared normalized geometry for fixed token features and
    ``K`` topic prototypes. A token route combines its own embedding, a
    leave-one-out spectrum context, and one spectrum-level prototype score.
    The final top-2 route is sparse, so one forward pass produces the document
    mixture ``theta``. The same prototypes define the topic-word probabilities
    ``beta``; consequently the completion likelihood is the ordinary topic
    mixture ``p(w | d) = sum_k theta[d,k] * beta[k,w]``.

    The last two fixed token-feature columns identify fragments and neutral
    losses. They are buffers, not labels used by a prediction head.
    """

    def __init__(  # noqa: PLR0913
        self,
        token_features: torch.Tensor,
        *,
        num_topics: int,
        projection_dimensions: int,
        router_hidden_dimensions: int,
        beta_temperature: float,
        token_type_balance: float,
        document_mixture_weight: float,
        document_topic_prior_weight: float,
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
        if self.beta_temperature <= 0:
            raise ValueError("decoder temperature must be positive")
        self.token_type_balance = float(token_type_balance)
        if not 0.0 <= self.token_type_balance <= 1.0:
            msg = "token type balance must be between zero and one"
            raise ValueError(msg)
        self.document_mixture_weight = float(document_mixture_weight)
        if self.document_mixture_weight < 0:
            raise ValueError("document mixture exponent must be non-negative")
        self.document_topic_prior_weight = float(document_topic_prior_weight)
        self.register_buffer("token_features", token_features.detach().clone())
        type_features = token_features[:, -2:]
        fragment_mask = type_features[:, 0] > type_features[:, 1]
        if self.token_type_balance and (
            not torch.any(fragment_mask) or torch.all(fragment_mask)
        ):
            msg = "token type balancing requires fragments and losses"
            raise ValueError(msg)
        self.register_buffer("fragment_mask", fragment_mask, persistent=False)

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
        """Project the fixed token table into normalized neural geometry."""
        return nnf.normalize(self.token_projection(self.token_features), dim=1)

    def topic_word_distribution(
        self,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the topic-word matrix ``beta`` from neural topic geometry.

        For topic ``k`` and token ``w``, the unnormalized log evidence is a
        temperature-scaled cosine similarity. Fragment and neutral-loss
        channels are normalized separately. Their channel evidence is the
        log-mean-exp, rather than log-sum-exp, so a larger vocabulary cannot
        acquire probability mass merely by containing more words. The final
        channel mass is pulled by ``token_type_balance`` toward an equal split,
        while rankings within each channel remain unchanged.

        Returns:
            A ``[num_topics, vocabulary_size]`` row-stochastic tensor.
        """
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        topics = nnf.normalize(self.topic_prototypes, dim=1)
        logits = 2.0 * topics @ tokens.T / self.beta_temperature
        balance = self.token_type_balance
        if not balance:
            return nnf.softmax(logits, dim=1)
        fragment_logits = logits[:, self.fragment_mask]
        loss_logits = logits[:, ~self.fragment_mask]
        type_logits = torch.cat(
            (
                torch.logsumexp(fragment_logits, dim=1, keepdim=True),
                torch.logsumexp(loss_logits, dim=1, keepdim=True),
            ),
            dim=1,
        )
        vocabulary_sizes = type_logits.new_tensor(
            (fragment_logits.shape[1], loss_logits.shape[1]),
        )
        type_logits = type_logits - torch.log(vocabulary_sizes).unsqueeze(0)
        fragment_mass = nnf.softmax(type_logits, dim=1)[:, :1]
        balanced_fragment_mass = (1.0 - balance) * fragment_mass + 0.5 * balance
        probabilities = torch.empty_like(logits)
        probabilities[:, self.fragment_mask] = balanced_fragment_mass * nnf.softmax(
            fragment_logits, dim=1
        )
        probabilities[:, ~self.fragment_mask] = (
            1.0 - balanced_fragment_mass
        ) * nnf.softmax(loss_logits, dim=1)
        return probabilities

    def _route_embeddings(
        self,
        batch: SparseBatch,
        projected_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextual token routes and whole-spectrum evidence.

        For every observed token, the context is the count-weighted document
        sum with that token occurrence removed. This leave-one-out construction
        prevents the context MLP from trivially copying the token it routes.
        The document vector uses the unnormalized count-weighted sum followed by
        row normalization, matching the equations in the scientific report.
        """
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
        token_routes = nnf.normalize(token_values + correction, dim=1)
        document_routes = nnf.normalize(document_sums, dim=1)
        return token_routes, document_routes

    @staticmethod
    def aggregate_theta(  # noqa: PLR0913
        assignments: torch.Tensor,
        *,
        row_ids: torch.Tensor,
        weights: torch.Tensor,
        documents: int,
        document_logits: torch.Tensor | None = None,
        temperature: float = 1.0,
        document_mixture_weight: float = 0.0,
    ) -> torch.Tensor:
        """Form ``theta`` from sparse token mass and detached document evidence.

        If ``m[d,k]`` is count-weighted routed mass and ``g[d,k]`` is the
        document softmax, this computes ``normalize(m * g**gamma)``. Detaching
        ``g`` prevents the sharpening path from learning to game its own gate;
        the document score still learns through the routing logits. Positive
        multiplication preserves exact zero token support. A spectrum with no
        routed mass receives the explicit uniform fallback.
        """
        topic_mass = assignments.new_zeros((documents, assignments.shape[1]))
        topic_mass.index_add_(0, row_ids, assignments * weights.unsqueeze(1))
        mixture_weight = float(document_mixture_weight)
        if mixture_weight < 0:
            raise ValueError("document mixture exponent must be non-negative")
        if mixture_weight:
            if document_logits is None or document_logits.shape != topic_mass.shape:
                msg = "document logits must match document topic mass"
                raise ValueError(msg)
            if float(temperature) <= 0:
                raise ValueError("document gate temperature must be positive")
            gate = nnf.softmax(document_logits / float(temperature), dim=1).detach()
            topic_mass = topic_mass * gate.pow(mixture_weight)
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
        """Perform one token-routing pass and return a sparse document mixture.

        Training uses a straight-through estimator: forward values are the
        renormalized top-2 probabilities, while gradients follow the full
        softmax. Inference uses the same top-2 forward calculation without any
        local optimization loop.
        """
        if float(temperature) <= 0:
            raise ValueError("routing temperature must be positive")
        if int(top_k) < 1:
            raise ValueError("top-k routing requires at least one topic")
        tokens = (
            self.projected_tokens() if projected_tokens is None else projected_tokens
        )
        routes, document_routes = self._route_embeddings(batch, tokens)
        topics = nnf.normalize(self.topic_prototypes, dim=1)
        local_logits = routes @ topics.T
        document_logits = document_routes @ topics.T
        logits = local_logits + (
            self.document_topic_prior_weight * document_logits[batch.row_ids]
        )
        soft = nnf.softmax(logits / float(temperature), dim=1)
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
            document_logits=document_logits,
            temperature=temperature,
            document_mixture_weight=self.document_mixture_weight,
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
        """Return count-weighted negative log ``theta @ beta`` probability.

        Only nonzero sparse target entries are materialized, but their
        probabilities are exactly those of the dense topic mixture.
        """
        selected_topics = theta[target.row_ids]
        selected_words = beta[:, target.indices].T
        probability = torch.sum(selected_topics * selected_words, dim=1).clamp_min(
            1e-12,
        )
        return -torch.sum(
            target.weights * torch.log(probability),
        ) / target.weights.sum().clamp_min(1.0)


def initialize_model(
    token_features: torch.Tensor,
    *,
    num_topics: int,
    protocol: dict[str, Any],
    seeding_weights: np.ndarray | None = None,
) -> tuple[NeuralAssignmentMS2LDA, torch.Tensor]:
    """Create the deterministic frequency/IDF-weighted k-means++ seed state."""
    seed = int(protocol["seed"])
    model_config = protocol["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + int(num_topics))
        temporary_projection = nn.Linear(
            int(token_features.shape[1]),
            int(model_config["projection_dimensions"]),
            bias=False,
        )
        nn.init.orthogonal_(temporary_projection.weight)
    projected = nnf.normalize(temporary_projection(token_features), dim=1)
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
        token_type_balance=float(model_config["token_type_balance"]),
        document_mixture_weight=float(model_config["document_mixture_weight"]),
        document_topic_prior_weight=float(model_config["document_topic_prior_weight"]),
        topic_initial_indices=initial_indices,
        seed=seed + int(num_topics),
    )
    return model, initial_indices
