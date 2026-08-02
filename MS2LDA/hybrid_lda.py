"""Reference DreaMS-conditioned MS2LDA with neural initialization and VB.

The model retains Dirichlet LDA and exact expected-count topic updates.  Frozen
DreaMS spectrum embeddings initialize local inference, while pooled peak
embeddings parameterize a fixed-mass topic-word prior. ``HybridLDAModel`` adds
the small document API needed for Tomotopy-shaped MS2LDA workflows.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as nn_functional

from MS2LDA.dreams_features import parse_spectral_word

EPSILON = 1e-12
CHECKPOINT_FORMAT = "ms2lda-hybrid-reference"
CHECKPOINT_VERSION = 2
TopicWord = tuple[str, float]


@dataclass(frozen=True)
class HybridLDAConfig:
    """Scientific and optimization settings for the reference model.

    The neural architecture is intentionally fixed to the two-hidden-layer
    network described in the method paper. Only settings that are useful for
    fitting or controlled experiments remain configurable.
    """

    # LDA and input dimensions.
    num_topics: int
    embedding_dim: int
    alpha: float | tuple[float, ...] = 0.1
    eta: float = 0.01

    # Amortized document encoder.
    hidden_size: int = 256
    feature_projection_dim: int = 128
    training_local_steps: int = 50
    batch_size: int = 128
    encoder_learning_rate: float = 1e-3
    encoder_updates_per_epoch: int = 5

    # Empirical-Bayes topic-word prior.
    prior_mass_fraction: float = 0.05
    prior_warmup_epochs: int = 15
    prior_training_epochs: int = 20
    prior_temperature: float = 0.5
    prior_learning_rate: float = 1e-3
    topic_diversity_weight: float = 1e-3

    # Local and global stopping rules.
    local_tolerance: float = 1e-4
    global_tolerance: float = 1e-3
    global_patience: int = 3
    max_epochs: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject configurations that would make the algorithm ill-defined."""
        positive_integers = {
            "num_topics": self.num_topics,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "feature_projection_dim": self.feature_projection_dim,
            "training_local_steps": self.training_local_steps,
            "batch_size": self.batch_size,
            "encoder_updates_per_epoch": self.encoder_updates_per_epoch,
            "prior_warmup_epochs": self.prior_warmup_epochs,
            "prior_training_epochs": self.prior_training_epochs,
            "global_patience": self.global_patience,
            "max_epochs": self.max_epochs,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.encoder_learning_rate <= 0 or self.prior_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive")
        if self.local_tolerance <= 0 or self.global_tolerance <= 0:
            raise ValueError("convergence tolerances must be positive")
        if not 0 <= self.prior_mass_fraction <= 1:
            raise ValueError("prior_mass_fraction must lie between zero and one")
        if self.topic_diversity_weight < 0:
            raise ValueError("topic_diversity_weight cannot be negative")
        if self.prior_training_epochs < self.prior_warmup_epochs:
            raise ValueError("prior training must cover the prior warmup")
        if self.max_epochs <= self.prior_training_epochs:
            raise ValueError("max_epochs must include at least one fixed-prior epoch")
        self.alpha_vector()

    def alpha_vector(self) -> np.ndarray:
        """Return one positive alpha value per topic."""
        values = np.asarray(self.alpha, dtype=np.float32)
        if values.ndim == 0:
            values = np.repeat(values, self.num_topics)
        if values.shape != (self.num_topics,) or np.any(values <= 0):
            raise ValueError("alpha must be positive and scalar or one value per topic")
        return values


@dataclass(frozen=True)
class _SparseBatch:
    """Padded nonzero words for one batch.

    Word tensors have shape ``batch x positions``; ``totals`` has shape
    ``batch x 1``. The mask distinguishes real entries from padding.
    """

    word_ids: torch.Tensor
    word_counts: torch.Tensor
    word_mask: torch.Tensor
    totals: torch.Tensor


def _make_sparse_batch(
    matrix: sp.csr_matrix,
    indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
) -> _SparseBatch:
    """Pad selected CSR rows without constructing a dense vocabulary matrix."""
    subset = matrix[np.asarray(indices, dtype=np.int64)].tocsr()
    lengths = np.diff(subset.indptr)
    width = max(int(lengths.max()) if lengths.size else 0, 1)
    word_ids = np.zeros((subset.shape[0], width), dtype=np.int64)
    word_counts = np.zeros((subset.shape[0], width), dtype=np.float32)
    word_mask = np.zeros((subset.shape[0], width), dtype=bool)
    for row, length in enumerate(lengths):
        if not length:
            continue
        start = subset.indptr[row]
        end = subset.indptr[row + 1]
        word_ids[row, :length] = subset.indices[start:end]
        word_counts[row, :length] = subset.data[start:end]
        word_mask[row, :length] = True
    counts = torch.from_numpy(word_counts).to(device)
    mask = torch.from_numpy(word_mask).to(device)
    return _SparseBatch(
        word_ids=torch.from_numpy(word_ids).to(device),
        word_counts=counts,
        word_mask=mask,
        totals=(counts * mask).sum(dim=1, keepdim=True),
    )


def observed_token_nll(
    matrix: sp.csr_matrix,
    theta: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Return mean negative log likelihood per observed token."""
    matrix = matrix.tocsr()
    loss = 0.0
    tokens = 0.0
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        words = matrix.indices[start:end]
        counts = matrix.data[start:end]
        probabilities = theta[row] @ beta[:, words]
        loss -= float(np.sum(counts * np.log(np.clip(probabilities, EPSILON, None))))
        tokens += float(counts.sum())
    return loss / max(tokens, EPSILON)


def _expected_log_dirichlet(parameters: torch.Tensor) -> torch.Tensor:
    """Compute ``E[log p]`` for rows of Dirichlet parameters.

    This is used for both document-topic parameters ``gamma`` and topic-word
    parameters ``lambda`` in the equations from the method paper.
    """
    return torch.digamma(parameters) - torch.digamma(
        parameters.sum(dim=1, keepdim=True)
    )


def _responsibilities(
    batch: _SparseBatch,
    gamma: torch.Tensor,
    expected_log_beta: torch.Tensor,
) -> torch.Tensor:
    """Return ``phi[d, v, k]`` for the nonzero words in a sparse batch.

    ``phi[d,v,k]`` is proportional to
    ``exp(E[log theta[d,k]] + E[log beta[k,v]])``. Padded word positions are
    harmless because their counts are zero in every subsequent calculation.
    """
    expected_log_theta = _expected_log_dirichlet(gamma)
    word_values = expected_log_beta[:, batch.word_ids].permute(1, 2, 0)
    return torch.softmax(expected_log_theta.unsqueeze(1) + word_values, dim=2)


# The mathematical inputs stay explicit; bundling them into a state object
# would hide the two coordinate updates this reference is meant to show.
def _local_vb(  # noqa: PLR0913
    batch: _SparseBatch,
    initial_gamma: torch.Tensor,
    alpha: torch.Tensor,
    expected_log_beta: torch.Tensor,
    *,
    steps: int,
    tolerance: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate the two local LDA updates for a batch of documents.

    Each iteration evaluates ``phi`` with :func:`_responsibilities`, then
    applies ``gamma[d,k] = alpha[k] + sum_v x[d,v] phi[d,v,k]``. The returned
    ``phi`` is recomputed from the final ``gamma`` so the global expected
    counts correspond to the returned document posterior.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    gamma = initial_gamma
    counts = batch.word_counts * batch.word_mask
    for _ in range(steps):
        phi = _responsibilities(batch, gamma, expected_log_beta)
        updated = alpha.unsqueeze(0) + (counts.unsqueeze(-1) * phi).sum(dim=1)
        change = ((updated - gamma).abs() / gamma.abs().clamp_min(1.0)).amax()
        gamma = updated
        if tolerance is not None and float(change) < tolerance:
            break
    return gamma, _responsibilities(batch, gamma, expected_log_beta)


def _expected_topic_word_counts(
    batch: _SparseBatch,
    phi: torch.Tensor,
    *,
    num_topics: int,
    vocab_size: int,
) -> torch.Tensor:
    """Compute ``sum_d x[d,v] phi[d,v,k]`` without a dense count matrix."""
    statistics = torch.zeros(
        (num_topics, vocab_size),
        device=phi.device,
        dtype=phi.dtype,
    )
    # The short document loop mirrors the mathematical sum over d. Each
    # ``index_add_`` writes the observed words into the K x V result.
    for row in range(batch.word_ids.shape[0]):
        observed = batch.word_mask[row]
        words = batch.word_ids[row, observed]
        weighted_phi = (
            batch.word_counts[row, observed].unsqueeze(1) * phi[row, observed]
        )
        statistics.index_add_(1, words, weighted_phi.transpose(0, 1))
    return statistics


class _HybridLDACore(nn.Module):
    """PyTorch parameters for the document encoder and structured word prior.

    The orchestration wrapper is separate because ``nn.Module.train(mode)``
    conflicts with the Tomotopy-shaped ``HybridLDAModel.train(iter)`` API.
    Keeping tensors here also makes checkpoint contents explicit.
    """

    def __init__(self, vocab_size: int, config: HybridLDAConfig) -> None:
        """Create tensors with the dimensions declared by ``config``."""
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.document_projector = nn.Sequential(
                nn.LayerNorm(config.embedding_dim),
                nn.Linear(config.embedding_dim, config.feature_projection_dim),
                nn.GELU(),
                nn.LayerNorm(config.feature_projection_dim),
            )
            self.word_projector = nn.Sequential(
                nn.LayerNorm(config.embedding_dim),
                nn.Linear(
                    config.embedding_dim,
                    config.feature_projection_dim,
                    bias=False,
                ),
            )
            self.word_type_embedding = nn.Embedding(3, config.feature_projection_dim)
            self.word_mz_projector = nn.Sequential(
                nn.Linear(1, config.feature_projection_dim),
                nn.GELU(),
                nn.Linear(
                    config.feature_projection_dim,
                    config.feature_projection_dim,
                ),
            )
            self.topic_embeddings = nn.Parameter(
                torch.empty(config.num_topics, config.feature_projection_dim)
            )
            nn.init.normal_(
                self.topic_embeddings,
                std=1.0 / math.sqrt(config.feature_projection_dim),
            )
            input_size = config.num_topics + config.feature_projection_dim
            self.encoder = nn.Sequential(
                nn.Linear(input_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_topics),
            )
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)

        self.register_buffer("alpha", torch.from_numpy(config.alpha_vector()))
        self.register_buffer(
            "lambda_posterior",
            torch.full(
                (config.num_topics, self.vocab_size),
                config.eta + 1.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "word_context_embeddings",
            torch.zeros(self.vocab_size, config.embedding_dim),
        )
        self.register_buffer(
            "word_context_observed",
            torch.zeros(self.vocab_size, dtype=torch.bool),
        )
        self.register_buffer("word_mz", torch.zeros(self.vocab_size, 1))
        self.register_buffer(
            "word_type",
            torch.full((self.vocab_size,), 2, dtype=torch.long),
        )

    def initialize_topics(self, total_tokens: float) -> None:
        """Initialize free topic-word factors from a seeded random simplex."""
        if total_tokens <= 0:
            raise ValueError("at least one token is required")
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        raw = torch.empty(
            self.config.num_topics,
            self.vocab_size,
            dtype=torch.float32,
        ).exponential_(1.0, generator=generator)
        means = raw / raw.sum(dim=1, keepdim=True).clamp_min(EPSILON)
        mass = total_tokens / self.config.num_topics
        self.lambda_posterior.copy_((self.config.eta + mass * means).to(self.device))

    @property
    def device(self) -> torch.device:
        """Device holding the model buffers and parameters."""
        return self.lambda_posterior.device

    def set_word_features(
        self,
        contextual_embeddings: np.ndarray,
        observed: np.ndarray,
        mz_values: np.ndarray,
        word_types: np.ndarray,
    ) -> None:
        """Copy aligned contextual, mass, and type features into model buffers."""
        expected = (self.vocab_size, self.config.embedding_dim)
        if contextual_embeddings.shape != expected:
            raise ValueError(f"word embeddings must have shape {expected}")
        if observed.shape != (self.vocab_size,):
            raise ValueError("observed must contain one flag per word")
        if mz_values.shape != (self.vocab_size,):
            raise ValueError("mz_values must contain one value per word")
        if word_types.shape != (self.vocab_size,):
            raise ValueError("word_types must contain one value per word")
        self.word_context_embeddings.copy_(
            torch.as_tensor(contextual_embeddings, device=self.device)
        )
        self.word_context_observed.copy_(torch.as_tensor(observed, device=self.device))
        self.word_mz.copy_(
            torch.as_tensor(mz_values, device=self.device).reshape(-1, 1)
        )
        self.word_type.copy_(torch.as_tensor(word_types, device=self.device))

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Parameters trained to amortize the local VB posterior."""
        return [*self.encoder.parameters(), *self.document_projector.parameters()]

    def prior_parameters(self) -> list[nn.Parameter]:
        """Parameters trained to construct the bounded topic-word prior."""
        return [
            *self.word_projector.parameters(),
            *self.word_type_embedding.parameters(),
            *self.word_mz_projector.parameters(),
            self.topic_embeddings,
        ]

    def beta_mean(self) -> torch.Tensor:
        """Return posterior-mean topic-word probabilities, shape ``K x V``."""
        return self.lambda_posterior / self.lambda_posterior.sum(
            dim=1, keepdim=True
        ).clamp_min(EPSILON)

    def _word_topic_evidence(
        self,
        batch: _SparseBatch,
        word_topic: torch.Tensor,
    ) -> torch.Tensor:
        """Average current topic evidence over each document's observed words."""
        counts = batch.word_counts * batch.word_mask
        evidence = (counts.unsqueeze(-1) * word_topic[batch.word_ids]).sum(dim=1)
        evidence = evidence / batch.totals.clamp_min(1.0)
        empty = batch.totals <= 0
        if torch.any(empty):
            evidence = torch.where(
                empty,
                torch.full_like(evidence, 1.0 / self.config.num_topics),
                evidence,
            )
        return evidence / evidence.sum(dim=1, keepdim=True).clamp_min(EPSILON)

    def encode(
        self,
        batch: _SparseBatch,
        document_embeddings: torch.Tensor,
        word_topic: torch.Tensor,
    ) -> torch.Tensor:
        """Predict initial ``gamma`` from topic evidence and a DreaMS embedding.

        The network predicts a residual on top of the current LDA evidence,
        then preserves the Dirichlet mass invariant
        ``sum(gamma[d]) = sum(alpha) + N[d]``.
        """
        expected = (batch.word_ids.shape[0], self.config.embedding_dim)
        if tuple(document_embeddings.shape) != expected:
            raise ValueError(f"document embeddings must have shape {expected}")
        evidence = self._word_topic_evidence(batch, word_topic)
        projected = self.document_projector(document_embeddings)
        residual = self.encoder(torch.cat([evidence, projected], dim=1))
        topic_mean = torch.softmax(
            evidence.clamp_min(EPSILON).log() + residual,
            dim=1,
        )
        return self.alpha.unsqueeze(0) + batch.totals * topic_mean

    def _projected_words(self) -> torch.Tensor:
        """Combine contextual peak, normalized mass, and token-type features."""
        context = self.word_projector(self.word_context_embeddings)
        context = context * self.word_context_observed.unsqueeze(-1)
        mz = self.word_mz_projector(self.word_mz)
        token_type = self.word_type_embedding(self.word_type)
        return nn_functional.normalize(context + mz + token_type, dim=1)

    def structured_prior(self, total_tokens: float, epoch: int) -> torch.Tensor:
        """Return ``eta + r[e] rho N/K p[k,v]`` for every topic and word."""
        baseline = torch.full_like(self.lambda_posterior, self.config.eta)
        topics = nn_functional.normalize(self.topic_embeddings, dim=1)
        logits = topics @ self._projected_words().transpose(0, 1)
        distribution = torch.softmax(logits / self.config.prior_temperature, dim=1)
        warmup = min(max(float(epoch), 0.0) / self.config.prior_warmup_epochs, 1.0)
        topic_mass = total_tokens / self.config.num_topics
        structured_mass = warmup * self.config.prior_mass_fraction * topic_mass
        return baseline + structured_mass * distribution

    def prior_loss(
        self,
        total_tokens: float,
        epoch: int,
    ) -> torch.Tensor:
        """Empirical-Bayes loss for the structured prior parameters."""
        prior = self.structured_prior(total_tokens, epoch)
        posterior = self.lambda_posterior.detach()
        expected_log_beta = _expected_log_dirichlet(posterior)
        expected_log_prior = (
            torch.lgamma(prior.sum(dim=1))
            - torch.lgamma(prior).sum(dim=1)
            + ((prior - 1.0) * expected_log_beta).sum(dim=1)
        ).mean() / self.vocab_size
        topics = nn_functional.normalize(self.topic_embeddings, dim=1)
        gram = topics @ topics.transpose(0, 1)
        identity = torch.eye(
            self.config.num_topics,
            device=self.device,
            dtype=gram.dtype,
        )
        orthogonality = ((gram - identity) ** 2).mean()
        return -expected_log_prior + self.config.topic_diversity_weight * orthogonality

    @torch.no_grad()
    def update_topics(self, statistics: torch.Tensor, prior: torch.Tensor) -> None:
        """Apply the global VB update ``lambda = prior + expected counts``."""
        if statistics.shape != self.lambda_posterior.shape:
            raise ValueError("statistics shape does not match topic posterior")
        if prior.shape != self.lambda_posterior.shape or torch.any(prior <= 0):
            raise ValueError("invalid Dirichlet prior")
        self.lambda_posterior.copy_(prior + statistics)


@dataclass
class HybridDocument:
    """One tokenized spectrum and its aligned pretrained embedding.

    ``words`` contains vocabulary indices after the document is attached to a
    fitted model. The raw strings and embedding remain available so a document
    can be re-indexed safely by another compatible model.
    """

    raw_words: list[str] = field(repr=False)
    embedding: np.ndarray = field(repr=False)
    _topic_dist: np.ndarray = field(repr=False)
    words: list[int] = field(default_factory=list)

    def get_topic_dist(self) -> np.ndarray:
        """Return a defensive copy of the document-topic probabilities."""
        return self._topic_dist.copy()


class HybridLDAModel:
    """Fit free LDA topics and amortize inference with frozen DreaMS features.

    The public surface deliberately mirrors only the small part of Tomotopy
    used by model-facing MS2LDA code: documents, ``train``, ``infer``, topic
    accessors, and safe save/load. Unsupported Tomotopy options are not
    accepted silently.
    """

    def __init__(
        self,
        config: HybridLDAConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Create an unfitted reference model on ``device``."""
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.docs: list[HybridDocument] = []
        self.used_vocabs: list[str] = []
        self.history: list[dict[str, float]] = []
        self._word_embeddings: dict[str, np.ndarray] = {}
        self._vocab_index: dict[str, int] = {}
        self._matrix: sp.csr_matrix | None = None
        self._core: _HybridLDACore | None = None
        self._encoder_optimizer: torch.optim.Optimizer | None = None
        self._prior_optimizer: torch.optim.Optimizer | None = None
        self._gamma: np.ndarray | None = None
        self._epochs = 0
        self._converged = False
        self._stable_epochs = 0
        self._rng = np.random.default_rng(config.seed)
        self._inference_only = False

    @property
    def alpha(self) -> np.ndarray:
        """Document-topic Dirichlet prior, one value per topic."""
        if self._core is None:
            return self.config.alpha_vector()
        return self._core.alpha.detach().cpu().numpy().copy()

    @property
    def k(self) -> int:
        """Number of topics (Tomotopy-compatible name)."""
        return self.config.num_topics

    @property
    def vocabs(self) -> list[str]:
        """Retained vocabulary (Tomotopy-compatible name)."""
        return self.used_vocabs

    @property
    def num_vocabs(self) -> int:
        """Number of retained spectral words."""
        return len(self.used_vocabs)

    @property
    def converged(self) -> bool:
        """Whether the fixed-prior global stopping rule was satisfied."""
        return self._converged

    @property
    def ll_per_word(self) -> float:
        """Plug-in mean log likelihood using posterior-mean theta and beta."""
        if self._matrix is None or self._core is None:
            return float("-inf")
        theta = np.vstack([document.get_topic_dist() for document in self.docs])
        return -observed_token_nll(self._matrix, theta, self._beta_numpy())

    @property
    def perplexity(self) -> float:
        """Exponentiated negative plug-in log likelihood per word."""
        return float(np.exp(-self.ll_per_word))

    def _validate_embedding(
        self,
        embedding: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Return one finite float32 embedding with the configured dimension."""
        values = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if values.shape != (self.config.embedding_dim,):
            raise ValueError(
                f"embedding must have shape ({self.config.embedding_dim},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("embedding contains non-finite values")
        return values.copy()

    def set_word_embeddings(
        self,
        embeddings: Mapping[str, np.ndarray | Sequence[float]],
    ) -> None:
        """Provide pooled train-only peak embeddings before adding topics."""
        if self._core is not None:
            raise RuntimeError("word embeddings must be supplied before training")
        validated: dict[str, np.ndarray] = {}
        for word, embedding in embeddings.items():
            validated[str(word)] = self._validate_embedding(embedding)
        self._word_embeddings = validated

    def _new_document(
        self,
        words: Sequence[str],
        embedding: np.ndarray | Sequence[float],
    ) -> HybridDocument:
        """Construct a string-token document before vocabulary indexing."""
        topic_dist = self.config.alpha_vector()
        topic_dist /= topic_dist.sum()
        return HybridDocument(
            raw_words=[str(word) for word in words],
            embedding=self._validate_embedding(embedding),
            _topic_dist=topic_dist,
        )

    def add_doc(
        self,
        words: Sequence[str],
        *,
        embedding: np.ndarray | Sequence[float],
    ) -> int:
        """Add one training spectrum and return its document index."""
        if self._core is not None:
            raise RuntimeError("documents cannot be added after training starts")
        self.docs.append(self._new_document(words, embedding))
        return len(self.docs) - 1

    @staticmethod
    def _word_feature(word: str) -> tuple[int, float]:
        """Encode ``frag@mass``/``loss@mass`` as type and scaled mass."""
        parsed = parse_spectral_word(word)
        if parsed is None:
            return 2, 0.0
        word_kind, value = parsed
        word_type = 0 if word_kind == "frag" else 1
        normalized = np.log1p(value) / np.log1p(2000.0)
        return word_type, float(normalized)

    def _prepare_vocabulary(self) -> None:
        """Build the insertion-order vocabulary and training CSR matrix."""
        if not self.docs:
            raise ValueError("at least one document is required")
        insertion_order: dict[str, None] = {}
        for document in self.docs:
            for word in document.raw_words:
                insertion_order.setdefault(word, None)
        self._set_vocabulary(insertion_order)
        self._matrix = self._documents_to_matrix(self.docs)

    def _set_vocabulary(self, vocabulary: Sequence[str]) -> None:
        """Install one vocabulary and re-index attached documents."""
        self.used_vocabs = [str(word) for word in vocabulary]
        self._vocab_index = {word: index for index, word in enumerate(self.used_vocabs)}
        for document in self.docs:
            self._index_document(document)

    def _index_document(self, document: HybridDocument) -> None:
        """Map known raw words to this model's integer vocabulary IDs."""
        document.words = [
            self._vocab_index[word]
            for word in document.raw_words
            if word in self._vocab_index
        ]

    def _documents_to_matrix(
        self,
        documents: Sequence[HybridDocument],
    ) -> sp.csr_matrix:
        """Count indexed words into a ``documents x vocabulary`` CSR matrix."""
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row, document in enumerate(documents):
            self._index_document(document)
            counts = Counter(document.words)
            rows.extend([row] * len(counts))
            columns.extend(counts)
            values.extend(float(count) for count in counts.values())
        return sp.csr_matrix(
            (values, (rows, columns)),
            shape=(len(documents), self.num_vocabs),
            dtype=np.float32,
        )

    def _build_core(self, *, initialize_topics: bool) -> None:
        """Construct neural parameters and aligned vocabulary-level buffers."""
        self._core = _HybridLDACore(self.num_vocabs, self.config).to(self.device)
        contextual = np.zeros(
            (self.num_vocabs, self.config.embedding_dim),
            dtype=np.float32,
        )
        observed = np.zeros(self.num_vocabs, dtype=bool)
        word_types = np.empty(self.num_vocabs, dtype=np.int64)
        mz_values = np.empty(self.num_vocabs, dtype=np.float32)
        for index, word in enumerate(self.used_vocabs):
            word_types[index], mz_values[index] = self._word_feature(word)
            embedding = self._word_embeddings.get(word)
            if embedding is not None:
                contextual[index] = embedding
                observed[index] = True
        self._core.set_word_features(contextual, observed, mz_values, word_types)
        if initialize_topics:
            if self._matrix is None:
                raise RuntimeError("training matrix is not prepared")
            self._core.initialize_topics(float(self._matrix.sum()))

    def _build_optimizers(self) -> None:
        """Create training-only optimizers; inference checkpoints omit them."""
        if self._core is None:
            raise RuntimeError("model core is not prepared")
        self._encoder_optimizer = torch.optim.Adam(
            self._core.encoder_parameters(),
            lr=self.config.encoder_learning_rate,
        )
        self._prior_optimizer = torch.optim.Adam(
            self._core.prior_parameters(),
            lr=self.config.prior_learning_rate,
        )

    def _prepare(self) -> None:
        """Build the training matrix, core tensors, optimizers, and initial gamma."""
        if self._core is not None:
            return
        self._prepare_vocabulary()
        self._build_core(initialize_topics=True)
        self._build_optimizers()
        if self._matrix is None:
            raise RuntimeError("training matrix is not prepared")
        totals = np.asarray(self._matrix.sum(axis=1)).reshape(-1, 1)
        self._gamma = self.alpha.reshape(1, -1) + totals / self.k

    def _embedding_batch(
        self,
        documents: Sequence[HybridDocument],
        indices: Sequence[int] | np.ndarray,
    ) -> torch.Tensor:
        """Stack selected document embeddings on the model device."""
        values = np.vstack([documents[int(index)].embedding for index in indices])
        return torch.from_numpy(values.astype(np.float32, copy=False)).to(self.device)

    def _train_encoder(
        self,
        gamma_targets: np.ndarray,
        word_topic: torch.Tensor,
    ) -> float:
        """Fit the amortizer to stop-gradient local-VB topic proportions."""
        if self._core is None or self._matrix is None:
            raise RuntimeError("model is not prepared")
        if self._encoder_optimizer is None:
            raise RuntimeError("encoder optimizer is not prepared")
        core = self._core
        core.train()
        total_loss = 0.0
        batches = 0
        for _ in range(self.config.encoder_updates_per_epoch):
            shuffled = self._rng.permutation(len(self.docs))
            for start in range(0, len(self.docs), self.config.batch_size):
                indices = shuffled[start : start + self.config.batch_size]
                batch = _make_sparse_batch(self._matrix, indices, device=self.device)
                target_gamma = torch.from_numpy(gamma_targets[indices]).to(self.device)
                predicted_gamma = core.encode(
                    batch,
                    self._embedding_batch(self.docs, indices),
                    word_topic,
                )
                target_theta = target_gamma / target_gamma.sum(dim=1, keepdim=True)
                predicted_theta = predicted_gamma / predicted_gamma.sum(
                    dim=1, keepdim=True
                )
                loss = (
                    (
                        target_theta
                        * (
                            target_theta.clamp_min(EPSILON).log()
                            - predicted_theta.clamp_min(EPSILON).log()
                        )
                    )
                    .sum(dim=1)
                    .mean()
                )
                self._encoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.encoder_parameters(), 10.0)
                self._encoder_optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
        return total_loss / max(batches, 1)

    def _fit_epoch(self) -> dict[str, float]:
        """Run one local-VB, amortizer, and global expected-count cycle."""
        if self._core is None or self._matrix is None or self._gamma is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        core.eval()
        # E-step: refine gamma and phi against the current free topics. The
        # document encoder does not participate in discovery; it learns these
        # local solutions afterward so new-spectrum inference can be fast.
        statistics = torch.zeros_like(core.lambda_posterior)
        gamma_targets = np.empty((len(self.docs), self.k), dtype=np.float32)
        with torch.no_grad():
            expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
            word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
            for start in range(0, len(self.docs), self.config.batch_size):
                indices = np.arange(
                    start,
                    min(start + self.config.batch_size, len(self.docs)),
                )
                batch = _make_sparse_batch(self._matrix, indices, device=self.device)
                initial = torch.from_numpy(self._gamma[indices]).to(self.device)
                gamma, phi = _local_vb(
                    batch,
                    initial,
                    core.alpha,
                    expected_log_beta,
                    steps=self.config.training_local_steps,
                    tolerance=self.config.local_tolerance,
                )
                gamma_targets[indices] = gamma.cpu().numpy()
                statistics += _expected_topic_word_counts(
                    batch,
                    phi,
                    num_topics=self.k,
                    vocab_size=self.num_vocabs,
                )
        self._gamma = gamma_targets.copy()

        # Amortization step: teach the encoder to imitate the refined E-step.
        encoder_mean_kl = self._train_encoder(gamma_targets, word_topic)

        # M-step: lambda is exactly structured prior plus expected counts.
        previous = core.lambda_posterior.detach().clone()
        epoch = self._epochs + 1
        total_tokens = float(self._matrix.sum())
        prior = core.structured_prior(total_tokens, epoch)
        core.update_topics(statistics, prior)
        prior_loss_value = 0.0

        # Empirical-Bayes step: update the bounded chemical prior only during
        # the declared training window, then keep it stationary.
        if epoch <= self.config.prior_training_epochs:
            if self._prior_optimizer is None:
                raise RuntimeError("prior optimizer is not prepared")
            prior_loss = core.prior_loss(total_tokens, epoch)
            self._prior_optimizer.zero_grad(set_to_none=True)
            prior_loss.backward()
            torch.nn.utils.clip_grad_norm_(core.prior_parameters(), 10.0)
            self._prior_optimizer.step()
            prior_loss_value = float(prior_loss.detach().cpu())
            core.update_topics(statistics, core.structured_prior(total_tokens, epoch))
        denominator = previous.abs().sum().clamp_min(EPSILON)
        lambda_change = float(
            ((core.lambda_posterior - previous).abs().sum() / denominator)
            .detach()
            .cpu()
        )
        self._epochs = epoch
        metrics = {
            "epoch": float(epoch),
            "encoder_mean_kl": encoder_mean_kl,
            "lambda_relative_change": lambda_change,
            "prior_loss": prior_loss_value,
        }
        self.history.append(metrics)
        return metrics

    def train(
        self,
        iter: int | None = None,
        *,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Train for at most ``iter`` global variational epochs.

        Convergence is counted only after the structured prior has stopped
        learning. ``progress_callback`` receives a copy of each epoch's four
        scalar diagnostics.
        """
        if self._inference_only:
            raise RuntimeError("a loaded inference artifact cannot resume training")
        epochs = self.config.max_epochs if iter is None else int(iter)
        if epochs < 0:
            raise ValueError("iter cannot be negative")
        self._prepare()
        for _ in range(epochs):
            if self._converged or self._epochs >= self.config.max_epochs:
                break
            metrics = self._fit_epoch()
            if (
                self._epochs > self.config.prior_training_epochs
                and metrics["lambda_relative_change"] < self.config.global_tolerance
            ):
                self._stable_epochs += 1
                self._converged = self._stable_epochs >= self.config.global_patience
            else:
                self._stable_epochs = 0
            if progress_callback is not None:
                progress_callback(dict(metrics))
        self._refresh_training_documents()

    @torch.no_grad()
    def _infer_matrix(
        self,
        matrix: sp.csr_matrix,
        documents: Sequence[HybridDocument],
        *,
        refinement_steps: int,
    ) -> np.ndarray:
        """Run the neural initializer and optional local VB on a sparse matrix."""
        if self._core is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        core.eval()
        expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
        word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
        gamma_parts: list[np.ndarray] = []
        for start in range(0, matrix.shape[0], self.config.batch_size):
            indices = np.arange(
                start,
                min(start + self.config.batch_size, matrix.shape[0]),
            )
            batch = _make_sparse_batch(matrix, indices, device=self.device)
            gamma = core.encode(
                batch,
                self._embedding_batch(documents, indices),
                word_topic,
            )
            if refinement_steps:
                gamma, _ = _local_vb(
                    batch,
                    gamma,
                    core.alpha,
                    expected_log_beta,
                    steps=refinement_steps,
                    tolerance=None,
                )
            gamma_parts.append(gamma.cpu().numpy())
        gamma_matrix = np.vstack(gamma_parts).astype(np.float32, copy=False)
        return gamma_matrix / np.maximum(
            gamma_matrix.sum(axis=1, keepdims=True), EPSILON
        )

    def _refresh_training_documents(self) -> None:
        """Store final locally refined topic means on training documents."""
        if self._matrix is None:
            raise RuntimeError("training matrix is not prepared")
        theta = self._infer_matrix(
            self._matrix,
            self.docs,
            refinement_steps=self.config.training_local_steps,
        )
        for row, document in enumerate(self.docs):
            document._topic_dist = theta[row]

    def make_doc(
        self,
        words: Sequence[str],
        *,
        embedding: np.ndarray | Sequence[float],
    ) -> HybridDocument:
        """Create, but do not add, a vocabulary-indexed inference document."""
        if self._core is None:
            raise RuntimeError("the model must be trained or loaded first")
        document = self._new_document(words, embedding)
        self._index_document(document)
        return document

    def infer(
        self,
        doc: HybridDocument | Sequence[HybridDocument],
        *,
        iter: int = 5,
    ) -> tuple[np.ndarray, float] | tuple[list[np.ndarray], np.ndarray]:
        """Infer topic mixtures; ``iter`` is the number of local VB updates.

        The accompanying likelihood is a posterior-mean plug-in score for the
        supplied document, not a training ELBO or Tomotopy Gibbs likelihood.
        """
        steps = int(iter)
        if steps < 0:
            raise ValueError("iter cannot be negative")
        single = isinstance(doc, HybridDocument)
        documents = [doc] if single else list(doc)
        if not documents:
            return [], np.asarray([], dtype=np.float32)
        matrix = self._documents_to_matrix(documents)
        theta = self._infer_matrix(
            matrix,
            documents,
            refinement_steps=steps,
        )
        for row, document in enumerate(documents):
            document._topic_dist = theta[row]
        beta = self._beta_numpy()
        log_likelihoods = np.zeros(len(documents), dtype=np.float32)
        for row in range(len(documents)):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            words = matrix.indices[start:end]
            counts = matrix.data[start:end]
            probabilities = theta[row] @ beta[:, words]
            log_likelihoods[row] = float(
                np.sum(counts * np.log(np.clip(probabilities, EPSILON, None)))
            )
        if single:
            return theta[0], float(log_likelihoods[0])
        return [row.copy() for row in theta], log_likelihoods

    def _beta_numpy(self) -> np.ndarray:
        """Return all posterior-mean topic-word rows as float32 NumPy."""
        if self._core is None:
            raise RuntimeError("the model must be trained or loaded first")
        return self._core.beta_mean().detach().cpu().numpy().astype(np.float32)

    def get_topic_word_dist(self, topic_id: int) -> np.ndarray:
        """Return posterior-mean probabilities for one topic over the vocabulary."""
        topic = int(topic_id)
        if not 0 <= topic < self.k:
            raise IndexError("topic_id is out of range")
        return self._beta_numpy()[topic].copy()

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> list[TopicWord]:
        """Return the highest-probability words for one topic."""
        distribution = self.get_topic_word_dist(topic_id)
        count = min(max(int(top_n), 0), self.num_vocabs)
        indices = np.argsort(-distribution, kind="stable")[:count]
        return [
            (self.used_vocabs[index], float(distribution[index])) for index in indices
        ]

    def save(self, filename: str | Path) -> None:
        """Save a safe inference artifact without documents or optimizer state."""
        if self._core is None:
            raise RuntimeError("the model must be trained before it can be saved")
        payload: dict[str, object] = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "vocabulary": self.used_vocabs,
            "core_state_dict": self._core.state_dict(),
        }
        torch.save(payload, Path(filename))

    @classmethod
    def load(
        cls,
        filename: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> HybridLDAModel:
        """Load a checkpoint written by :meth:`save` without arbitrary pickle."""
        payload = torch.load(Path(filename), map_location=device, weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("not an MS2LDA hybrid reference checkpoint")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        model = cls(HybridLDAConfig(**payload["config"]), device=device)
        model._set_vocabulary(payload["vocabulary"])
        model._build_core(initialize_topics=False)
        if model._core is None:
            raise RuntimeError("checkpoint core was not constructed")
        model._core.load_state_dict(payload["core_state_dict"])
        model._inference_only = True
        return model
