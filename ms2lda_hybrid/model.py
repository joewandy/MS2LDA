"""DreaMS-conditioned LDA with classical discovery and neural local inference.

``HybridLDAModel`` has one explicit two-stage lifecycle. Classical
variational LDA first discovers topics from sparse expected counts and a
bounded DreaMS-conditioned word prior. The finalized topics are then frozen
while a DreaMS document encoder learns through two differentiable local
updates. The document encoder never contributes counts to topic discovery.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ._core import HybridLDACore as _HybridLDACore
from ._variational import (
    EPSILON,
    observed_token_nll,
)
from ._variational import (
    corpus_elbo_minibatch_scale as _corpus_elbo_minibatch_scale,
)
from ._variational import (
    expected_log_dirichlet as _expected_log_dirichlet,
)
from ._variational import (
    expected_topic_word_counts as _expected_topic_word_counts,
)
from ._variational import (
    local_document_elbo as _local_document_elbo,
)
from ._variational import (
    local_vb as _local_vb,
)
from ._variational import (
    make_sparse_batch as _make_sparse_batch,
)
from .config import HybridLDAConfig
from .dreams_features import parse_spectral_word

CHECKPOINT_FORMAT = "ms2lda-hybrid-reference"
CHECKPOINT_VERSION = 3
INFERENCE_REFINEMENT_STEPS = 2
ZERO_STEP_ELBO_WEIGHT = 0.1
TopicWord = tuple[str, float]


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
    """Discover LDA topics, then finalize a semi-amortized inference encoder.

    Add all training documents before calling :meth:`train`. Topic discovery
    can be advanced incrementally with repeated ``train(iter=...)`` calls.
    Once the topics are satisfactory, call :meth:`finalize_inference` exactly
    once; this freezes discovery permanently and fits the only supported neural
    document-inference objective. New-document :meth:`infer` and :meth:`save`
    are intentionally unavailable until that finalization succeeds.

    The deliberately small experiment-facing surface covers documents, topic
    accessors, frozen-topic inference, and safe checkpoint I/O. It is not a
    drop-in replacement for Tomotopy's sampled token assignments or the
    production visualization/export workflow.
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
        self._prior_optimizer: torch.optim.Optimizer | None = None
        self._gamma: np.ndarray | None = None
        self._epochs = 0
        self._converged = False
        self._stable_epochs = 0
        self._rng = np.random.default_rng(config.seed)
        self._inference_only = False
        self._inference_finalized = False

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
        """Whether topic discovery met its fixed-prior stopping rule.

        This flag does not imply that document inference has been finalized;
        inspect :attr:`inference_finalized` for that separate lifecycle state.
        """
        return self._converged

    @property
    def inference_finalized(self) -> bool:
        """Whether the frozen-topic semi-amortized encoder is ready for use."""
        return self._inference_finalized

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

    def _build_prior_optimizer(self) -> None:
        """Create the discovery-only structured-prior optimizer."""
        if self._core is None:
            raise RuntimeError("model core is not prepared")
        self._prior_optimizer = torch.optim.Adam(
            self._core.prior_parameters(),
            lr=self.config.prior_learning_rate,
        )

    def _prepare(self) -> None:
        """Build the training matrix, core tensors, prior optimizer, and gamma."""
        if self._core is not None:
            return
        self._prepare_vocabulary()
        self._build_core(initialize_topics=True)
        self._build_prior_optimizer()
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

    @torch.no_grad()
    def _refine_training_posteriors(self, *, steps: int) -> np.ndarray:
        """Solve local posteriors against the current frozen topic posterior.

        The initializer is retained classical-VB state, never the document
        encoder.  This keeps the targets and reported training likelihoods
        independent of the global DreaMS document embeddings.
        """
        if self._core is None or self._matrix is None or self._gamma is None:
            raise RuntimeError("model is not prepared")
        if steps < 1:
            raise ValueError("steps must be positive")
        core = self._core
        core.eval()
        expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
        targets = np.empty((len(self.docs), self.k), dtype=np.float32)
        for start in range(0, len(self.docs), self.config.batch_size):
            indices = np.arange(
                start,
                min(start + self.config.batch_size, len(self.docs)),
            )
            batch = _make_sparse_batch(self._matrix, indices, device=self.device)
            initial = torch.as_tensor(
                self._gamma[indices],
                device=self.device,
                dtype=core.lambda_posterior.dtype,
            )
            gamma, _ = _local_vb(
                batch,
                initial,
                core.alpha,
                expected_log_beta,
                steps=steps,
                tolerance=self.config.local_tolerance,
            )
            targets[indices] = gamma.cpu().numpy()
        return targets

    def _optimize_inference_encoder(
        self,
        *,
        expected_log_beta: torch.Tensor,
        word_topic: torch.Tensor,
    ) -> list[dict[str, float]]:
        """Optimize the fixed two-step ELBO against frozen topics."""
        if self._core is None or self._matrix is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        optimizer = torch.optim.Adam(
            core.encoder_parameters(),
            lr=self.config.encoder_learning_rate,
        )
        corpus_tokens = float(self._matrix.sum())
        corpus_documents = len(self.docs)
        phase_history: list[dict[str, float]] = []
        for phase_epoch in range(1, self.config.inference_epochs + 1):
            core.train()
            shuffled = self._rng.permutation(corpus_documents)
            loss_sum = 0.0
            refined_elbo_sum = 0.0
            zero_elbo_sum = 0.0
            token_sum = 0.0
            gradient_norm_sum = 0.0
            batches = 0
            for start in range(0, corpus_documents, self.config.batch_size):
                indices = shuffled[start : start + self.config.batch_size]
                batch = _make_sparse_batch(self._matrix, indices, device=self.device)
                gamma_zero = core.encode(
                    batch,
                    self._embedding_batch(self.docs, indices),
                    word_topic,
                )
                gamma_refined, _ = _local_vb(
                    batch,
                    gamma_zero,
                    core.alpha,
                    expected_log_beta,
                    steps=INFERENCE_REFINEMENT_STEPS,
                    tolerance=None,
                )
                refined_elbo = _local_document_elbo(
                    batch,
                    gamma_refined,
                    core.alpha,
                    expected_log_beta,
                )
                zero_elbo = _local_document_elbo(
                    batch,
                    gamma_zero,
                    core.alpha,
                    expected_log_beta,
                )

                # D/(B*T) turns a uniform document minibatch sum into the
                # corpus-token objective. Random reshuffling visits every
                # document exactly once per epoch.
                elbo_scale = _corpus_elbo_minibatch_scale(
                    corpus_documents=corpus_documents,
                    batch_documents=len(indices),
                    corpus_tokens=corpus_tokens,
                )
                loss = (
                    -(refined_elbo.sum() + ZERO_STEP_ELBO_WEIGHT * zero_elbo.sum())
                    * elbo_scale
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite semi-amortized ELBO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    core.encoder_parameters(),
                    10.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                loss_sum += float(loss.detach().cpu())
                refined_elbo_sum += float(refined_elbo.sum().detach().cpu())
                zero_elbo_sum += float(zero_elbo.sum().detach().cpu())
                token_sum += float(batch.totals.sum().detach().cpu())
                gradient_norm_sum += float(gradient_norm.detach().cpu())
                batches += 1
            metrics = {
                "inference_epoch": float(phase_epoch),
                "loss": loss_sum / max(batches, 1),
                "refined_elbo_per_token": refined_elbo_sum / max(token_sum, EPSILON),
                "zero_step_elbo_per_token": zero_elbo_sum / max(token_sum, EPSILON),
                "encoder_gradient_norm": gradient_norm_sum / max(batches, 1),
            }
            phase_history.append(metrics)
        return phase_history

    def finalize_inference(self) -> list[dict[str, float]]:
        """Freeze discovery and fit the supported document-inference encoder.

        This one-way operation is the second required training stage. A fresh
        Adam optimizer updates only the spectrum projector and document
        encoder for ``config.inference_epochs``. For every document, the
        encoder predicts ``gamma[0]``; two fixed local-LDA coordinate updates
        produce ``gamma[2]``; and gradients minimize the negative local ELBO at
        ``gamma[2]`` plus ``0.1`` times the negative ELBO at ``gamma[0]``.

        The complete topic posterior and every structured-prior parameter are
        snapshotted and checked bit-for-bit after training. Encoder-derived
        responsibilities are never used by the discovery update. Once this
        method succeeds, topic training cannot resume and the model can be
        used for new-document inference or saved as an inference artifact.

        Returns
        -------
        list of dict
            One online diagnostic record per inference-training epoch. ELBO
            values are corpus-token summaries observed while parameters are
            changing, rather than a separate final evaluation pass.

        Raises
        ------
        RuntimeError
            If discovery has not run, finalization is already running or has
            succeeded, or the instance is an inference-only checkpoint.

        Notes
        -----
        ``expected_log_beta`` and the word-to-topic evidence matrix are detached
        before optimization. Back-propagation therefore follows
        ``encoder -> gamma[0] -> gamma[1] -> gamma[2] -> local ELBO`` but has no
        route into the learned topics or structured word prior. If optimization
        fails, encoder parameters, random-generator state, and module mode are
        restored before the exception propagates.
        """
        if self._inference_only:
            raise RuntimeError("an inference artifact has no training documents")
        if self._inference_finalized:
            raise RuntimeError("document inference has already been finalized")
        if self._epochs < 1:
            raise RuntimeError(
                "fit at least one topic-discovery epoch before finalization"
            )
        if self._core is None or self._matrix is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        lambda_snapshot = core.lambda_posterior.detach().clone()
        prior_snapshot = [
            parameter.detach().clone() for parameter in core.prior_parameters()
        ]
        encoder_snapshot = [
            parameter.detach().clone() for parameter in core.encoder_parameters()
        ]
        rng_snapshot = copy.deepcopy(self._rng.bit_generator.state)
        previous_module_mode = core.training

        # On failure, restore every state changed by optimization so an explicit
        # retry starts from the same model and random-number stream.
        try:
            expected_log_beta = _expected_log_dirichlet(lambda_snapshot).detach()
            word_topic = torch.softmax(
                expected_log_beta.transpose(0, 1), dim=1
            ).detach()
            phase_history = self._optimize_inference_encoder(
                expected_log_beta=expected_log_beta,
                word_topic=word_topic,
            )
            if not torch.equal(lambda_snapshot, core.lambda_posterior):
                raise RuntimeError("inference finalization unexpectedly changed topics")
            if any(
                not torch.equal(previous, current)
                for previous, current in zip(
                    prior_snapshot,
                    core.prior_parameters(),
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "inference finalization unexpectedly changed the prior"
                )
        except BaseException:
            with torch.no_grad():
                core.lambda_posterior.copy_(lambda_snapshot)
                for parameter, snapshot in zip(
                    core.prior_parameters(),
                    prior_snapshot,
                    strict=True,
                ):
                    parameter.copy_(snapshot)
                for parameter, snapshot in zip(
                    core.encoder_parameters(),
                    encoder_snapshot,
                    strict=True,
                ):
                    parameter.copy_(snapshot)
            self._rng.bit_generator.state = copy.deepcopy(rng_snapshot)
            self._inference_finalized = False
            core.train(previous_module_mode)
            raise
        else:
            self._inference_finalized = True
            core.eval()
            return phase_history

    def _fit_epoch(self) -> dict[str, float]:
        """Run one classical local-VB and global expected-count cycle."""
        if self._core is None or self._matrix is None or self._gamma is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        core.eval()
        # E-step: refine gamma and phi against the current free topics. The
        # neural document encoder is completely absent from topic discovery.
        statistics = torch.zeros_like(core.lambda_posterior)
        updated_gamma = np.empty((len(self.docs), self.k), dtype=np.float32)
        with torch.no_grad():
            expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
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
                updated_gamma[indices] = gamma.cpu().numpy()
                statistics += _expected_topic_word_counts(
                    batch,
                    phi,
                    num_topics=self.k,
                    vocab_size=self.num_vocabs,
                )
        self._gamma = updated_gamma

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
            "lambda_relative_change": lambda_change,
            "prior_loss": prior_loss_value,
        }
        self.history.append(metrics)
        return metrics

    def train(self, iter: int | None = None) -> None:
        """Advance classical topic discovery by at most ``iter`` epochs.

        Each epoch refines training-document variational factors, accumulates
        full-corpus topic-word expected counts, and updates the bounded
        structured word prior during its declared window. The neural document
        encoder is not evaluated or optimized here. Convergence is counted
        only after the structured prior has stopped learning.

        Repeated calls are supported until :meth:`finalize_inference` succeeds.
        Finalization permanently declares the current topics frozen, after
        which this method raises instead of silently making the encoder stale.

        Parameters
        ----------
        iter
            Maximum additional discovery epochs. ``None`` uses
            ``config.max_epochs``; zero prepares the model but performs no
            global update.

        Raises
        ------
        RuntimeError
            If inference has already been finalized or this is an
            inference-only loaded artifact.
        ValueError
            If ``iter`` is negative.
        """
        if self._inference_only:
            raise RuntimeError("a loaded inference artifact cannot resume training")
        if self._inference_finalized:
            raise RuntimeError("topic discovery cannot resume after finalization")
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
        self._refresh_training_documents()

    @torch.no_grad()
    def _infer_matrix(
        self,
        matrix: sp.csr_matrix,
        documents: Sequence[HybridDocument],
        *,
        refinement_steps: int,
        tolerance: float | None = None,
    ) -> np.ndarray:
        """Run the neural initializer and optional local VB on a sparse matrix.

        When ``tolerance`` is supplied, ``refinement_steps`` is a maximum
        budget and a batch stops once every document satisfies the relative
        gamma-change threshold.  Fixed-step differentiable training never uses
        this adaptive path.
        """
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
                    tolerance=tolerance,
                )
            gamma_parts.append(gamma.cpu().numpy())
        gamma_matrix = np.vstack(gamma_parts).astype(np.float32, copy=False)
        return gamma_matrix / np.maximum(
            gamma_matrix.sum(axis=1, keepdims=True), EPSILON
        )

    def _refresh_training_documents(self) -> None:
        """Store LDA-refined topic means without using the neural initializer."""
        if self._matrix is None or self._gamma is None:
            raise RuntimeError("training matrix is not prepared")
        gamma = self._refine_training_posteriors(steps=self.config.training_local_steps)
        theta = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), EPSILON)
        for row, document in enumerate(self.docs):
            document._topic_dist = theta[row]

    def make_doc(
        self,
        words: Sequence[str],
        *,
        embedding: np.ndarray | Sequence[float],
    ) -> HybridDocument:
        """Create a vocabulary-indexed document for frozen-topic inference.

        Unknown words are retained in ``raw_words`` for provenance but omitted
        from the fitted vocabulary indices. The document is not added to the
        training corpus. Call :meth:`finalize_inference` before constructing
        inference documents from a model trained in this process.

        Parameters
        ----------
        words
            Spectral-word strings such as ``frag@100.1`` and ``loss@18.0``.
        embedding
            DreaMS spectrum embedding aligned with the supplied spectrum.

        Returns
        -------
        HybridDocument
            Detached query document indexed against the fitted vocabulary.
        """
        if self._core is None:
            raise RuntimeError("the model must be trained or loaded first")
        if not self._inference_finalized:
            raise RuntimeError("finalize document inference before making queries")
        document = self._new_document(words, embedding)
        self._index_document(document)
        return document

    def infer(
        self,
        doc: HybridDocument | Sequence[HybridDocument],
        *,
        iter: int = INFERENCE_REFINEMENT_STEPS,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, float] | tuple[list[np.ndarray], np.ndarray]:
        """Infer topic mixtures with fixed or adaptive local VB updates.

        The finalized encoder supplies ``gamma[0]``. With ``tolerance=None``,
        ``iter`` is the exact number of subsequent local coordinate updates.
        Otherwise, ``iter`` is a maximum and a sparse batch stops when every
        document meets the relative-gamma tolerance. ``iter=0`` returns the
        amortized prediction directly.

        A single input returns ``(theta, log_likelihood)``; a sequence returns
        ``(list[theta], log_likelihood_array)``. The likelihood is a
        posterior-mean plug-in score for the supplied counts, not a training
        ELBO or Tomotopy Gibbs likelihood.

        Parameters
        ----------
        doc
            One query document or a sequence created with :meth:`make_doc`.
        iter
            Exact local-update count, or the maximum when ``tolerance`` is set.
            The default is the same two-step budget used during finalization.
        tolerance
            Optional positive relative-gamma stopping threshold.

        Returns
        -------
        tuple
            Topic means and plug-in log likelihoods, with a scalar-shaped
            result for one document and batched containers for a sequence.
        """
        if not self._inference_finalized:
            raise RuntimeError("finalize document inference before calling infer")
        steps = int(iter)
        if steps < 0:
            raise ValueError("iter cannot be negative")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("tolerance must be positive")
        single = isinstance(doc, HybridDocument)
        documents = [doc] if single else list(doc)
        if not documents:
            return [], np.asarray([], dtype=np.float32)
        matrix = self._documents_to_matrix(documents)
        theta = self._infer_matrix(
            matrix,
            documents,
            refinement_steps=steps,
            tolerance=tolerance,
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
        """Save the finalized model as a weights-only inference artifact.

        Training documents, local gamma state, optimizer state, and random
        generators are deliberately omitted. Consequently, a loaded artifact
        can infer and expose topics but cannot resume either training stage.

        ``filename`` may be any path accepted by :class:`pathlib.Path`. The
        method refuses to serialize an unfinished model because checkpoint
        version 3 denotes the fixed semi-amortized objective explicitly.
        """
        if self._core is None:
            raise RuntimeError("the model must be trained before it can be saved")
        if not self._inference_finalized:
            raise RuntimeError("finalize document inference before saving")
        payload: dict[str, object] = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "inference_finalized": True,
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
        """Load a finalized, inference-only checkpoint without arbitrary pickle.

        ``torch.load(weights_only=True)`` restricts deserialization to tensors
        and primitive containers. The checkpoint version is intentionally
        strict because the inference objective and lifecycle are part of the
        serialized model definition.

        Parameters
        ----------
        filename
            Version-3 weights-only checkpoint created by :meth:`save`.
        device
            PyTorch device on which tensors should be reconstructed.

        Returns
        -------
        HybridLDAModel
            Finalized inference-only model with no attached training corpus.
        """
        payload = torch.load(Path(filename), map_location=device, weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("not an MS2LDA hybrid reference checkpoint")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        if payload.get("inference_finalized") is not True:
            raise ValueError("checkpoint does not contain finalized inference")
        model = cls(HybridLDAConfig(**payload["config"]), device=device)
        model._set_vocabulary(payload["vocabulary"])
        model._build_core(initialize_topics=False)
        if model._core is None:
            raise RuntimeError("checkpoint core was not constructed")
        model._core.load_state_dict(payload["core_state_dict"])
        model._inference_only = True
        model._inference_finalized = True
        return model
