"""DreaMS-conditioned LDA with classical discovery and neural local inference.

``HybridLDAModel`` has one explicit two-stage lifecycle. Classical
variational LDA first discovers topics from sparse expected counts and a
bounded DreaMS-conditioned word prior. The finalized topics are then frozen
while a DreaMS document encoder learns through two differentiable local
updates. The document encoder never contributes counts to topic discovery.
"""

from __future__ import annotations

import copy
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ._core import HybridLDACore as _HybridLDACore
from ._torch_safety import require_patched_torch
from ._variational import (
    EPSILON,
    estimate_dirichlet_alpha,
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
TRAINING_CHECKPOINT_FORMAT = "ms2lda-hybrid-training-state"
TRAINING_CHECKPOINT_VERSION = 1
INFERENCE_REFINEMENT_STEPS = 2
ZERO_STEP_ELBO_WEIGHT = 0.1
TopicWord = tuple[str, float]
TrainingCheckpointCallback = Callable[["HybridLDAModel", str, int], None]


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
        self._vocabulary: tuple[str, ...] = ()
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
        self._inference_epochs_completed = 0
        self._inference_history: list[dict[str, float]] = []
        self._inference_optimizer_state: dict[str, object] | None = None

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
        """Defensive copy of the retained vocabulary (Tomotopy-compatible)."""
        return list(self._vocabulary)

    @property
    def used_vocabs(self) -> list[str]:
        """Defensive copy of the retained spectral-word vocabulary."""
        return list(self._vocabulary)

    @property
    def num_vocabs(self) -> int:
        """Number of retained spectral words."""
        return len(self._vocabulary)

    @property
    def converged(self) -> bool:
        """Whether topic discovery met its global stopping rule.

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
        resolved = tuple(str(word) for word in vocabulary)
        if len(set(resolved)) != len(resolved):
            raise ValueError("vocabulary entries must be unique")
        self._vocabulary = resolved
        self._vocab_index = {word: index for index, word in enumerate(resolved)}
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
        for index, word in enumerate(self._vocabulary):
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
        checkpoint_callback: TrainingCheckpointCallback | None = None,
    ) -> list[dict[str, float]]:
        """Optimize the fixed two-step ELBO against frozen topics."""
        if self._core is None or self._matrix is None:
            raise RuntimeError("model is not prepared")
        core = self._core
        optimizer = torch.optim.Adam(
            core.encoder_parameters(),
            lr=self.config.encoder_learning_rate,
        )
        if self._inference_optimizer_state is not None:
            optimizer.load_state_dict(self._inference_optimizer_state)
        corpus_tokens = float(self._matrix.sum())
        corpus_documents = len(self.docs)
        phase_history = copy.deepcopy(self._inference_history)
        for phase_epoch in range(
            self._inference_epochs_completed + 1,
            self.config.inference_epochs + 1,
        ):
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
            self._inference_epochs_completed = phase_epoch
            self._inference_history = copy.deepcopy(phase_history)
            self._inference_optimizer_state = copy.deepcopy(optimizer.state_dict())
            if checkpoint_callback is not None:
                checkpoint_callback(self, "encoder", phase_epoch)
        return phase_history

    def finalize_inference(
        self,
        *,
        checkpoint_callback: TrainingCheckpointCallback | None = None,
    ) -> list[dict[str, float]]:
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
        alpha_snapshot = core.alpha.detach().clone()
        prior_snapshot = [
            parameter.detach().clone() for parameter in core.prior_parameters()
        ]
        encoder_snapshot = [
            parameter.detach().clone() for parameter in core.encoder_parameters()
        ]
        rng_snapshot = copy.deepcopy(self._rng.bit_generator.state)
        inference_epoch_snapshot = self._inference_epochs_completed
        inference_history_snapshot = copy.deepcopy(self._inference_history)
        inference_optimizer_snapshot = copy.deepcopy(self._inference_optimizer_state)
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
                checkpoint_callback=checkpoint_callback,
            )
            if not torch.equal(lambda_snapshot, core.lambda_posterior):
                raise RuntimeError("inference finalization unexpectedly changed topics")
            if not torch.equal(alpha_snapshot, core.alpha):
                raise RuntimeError("inference finalization unexpectedly changed alpha")
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
                core.alpha.copy_(alpha_snapshot)
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
            self._inference_epochs_completed = inference_epoch_snapshot
            self._inference_history = inference_history_snapshot
            self._inference_optimizer_state = inference_optimizer_snapshot
            self._inference_finalized = False
            core.train(previous_module_mode)
            raise
        else:
            self._inference_history = copy.deepcopy(phase_history)
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
        expected_log_theta_sum = np.zeros(self.k, dtype=np.float64)
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
                expected_log_theta_sum += (
                    _expected_log_dirichlet(gamma.double()).sum(dim=0).cpu().numpy()
                )
                statistics += _expected_topic_word_counts(
                    batch,
                    phi,
                    num_topics=self.k,
                    vocab_size=self.num_vocabs,
                )
        self._gamma = updated_gamma

        # Training-only empirical Bayes update for the document-topic prior.
        # This is independent of chemical identities and the neural encoder.
        previous_alpha = core.alpha.detach().clone()
        optimized_alpha = estimate_dirichlet_alpha(
            previous_alpha.cpu().numpy(),
            expected_log_theta_sum,
            len(self.docs),
        )
        core.alpha.copy_(
            torch.as_tensor(
                optimized_alpha,
                device=core.alpha.device,
                dtype=core.alpha.dtype,
            )
        )
        alpha_denominator = previous_alpha.abs().sum().clamp_min(EPSILON)
        alpha_change = float(
            ((core.alpha - previous_alpha).abs().sum() / alpha_denominator)
            .detach()
            .cpu()
        )

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
            "alpha_relative_change": alpha_change,
            "alpha_sum": float(core.alpha.sum().detach().cpu()),
            "alpha_min": float(core.alpha.min().detach().cpu()),
            "alpha_median": float(np.median(self.alpha)),
            "alpha_max": float(core.alpha.max().detach().cpu()),
            "prior_loss": prior_loss_value,
        }
        self.history.append(metrics)
        return metrics

    def train(
        self,
        iter: int | None = None,
        *,
        checkpoint_callback: TrainingCheckpointCallback | None = None,
    ) -> None:
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
        if self._inference_epochs_completed:
            raise RuntimeError(
                "topic discovery cannot resume after encoder training has started"
            )
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
                and metrics["alpha_relative_change"] < self.config.global_tolerance
            ):
                self._stable_epochs += 1
                self._converged = self._stable_epochs >= self.config.global_patience
            else:
                self._stable_epochs = 0
            if checkpoint_callback is not None:
                checkpoint_callback(self, "discovery", self._epochs)
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
            (self._vocabulary[index], float(distribution[index])) for index in indices
        ]

    def save_training_checkpoint(
        self,
        filename: str | Path,
        *,
        context_sha256: str,
    ) -> None:
        """Atomically save resumable discovery or encoder-training state.

        This format is intentionally separate from :meth:`save`. It contains
        the free-topic posterior, retained local variational factors, both
        optimizer states, random-generator state, convergence counters, and
        completed histories. Training documents and DreaMS feature arrays are
        not duplicated; callers must reconstruct those immutable inputs and
        supply their frozen context hash to :meth:`restore_training_checkpoint`.
        """
        if self._inference_only:
            raise RuntimeError("an inference artifact has no resumable training state")
        if self._inference_finalized:
            raise RuntimeError(
                "finalized inference does not need a training checkpoint"
            )
        if (
            self._core is None
            or self._matrix is None
            or self._gamma is None
            or self._prior_optimizer is None
        ):
            raise RuntimeError("fit at least one discovery epoch before checkpointing")
        context = str(context_sha256)
        if len(context) != 64 or any(
            character not in "0123456789abcdef" for character in context
        ):
            raise ValueError("context_sha256 must be a lowercase SHA-256 digest")
        phase = "encoder" if self._inference_epochs_completed else "discovery"
        phase_epoch = (
            self._inference_epochs_completed if phase == "encoder" else self._epochs
        )
        payload: dict[str, object] = {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "version": TRAINING_CHECKPOINT_VERSION,
            "context_sha256": context,
            "config": asdict(self.config),
            "phase": phase,
            "phase_epoch": phase_epoch,
            "vocabulary": list(self._vocabulary),
            "document_count": len(self.docs),
            "matrix_shape": tuple(int(value) for value in self._matrix.shape),
            "matrix_nnz": int(self._matrix.nnz),
            "matrix_token_sum": float(self._matrix.sum()),
            "core_state_dict": self._core.state_dict(),
            "prior_optimizer_state_dict": self._prior_optimizer.state_dict(),
            "gamma": torch.from_numpy(self._gamma.copy()),
            "discovery_epochs": self._epochs,
            "discovery_converged": self._converged,
            "stable_epochs": self._stable_epochs,
            "discovery_history": copy.deepcopy(self.history),
            "numpy_rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "inference_epochs_completed": self._inference_epochs_completed,
            "inference_history": copy.deepcopy(self._inference_history),
            "inference_optimizer_state_dict": copy.deepcopy(
                self._inference_optimizer_state
            ),
            "module_training": self._core.training,
        }
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def extend_training_checkpoint(
        source: str | Path,
        destination: str | Path,
        *,
        source_context_sha256: str,
        target_context_sha256: str,
        target_max_epochs: int,
    ) -> dict[str, int | str]:
        """Rebind a stopped discovery checkpoint to a larger epoch ceiling.

        This narrowly scoped migration exists for an audited continuation
        after a frozen run reaches its maximum discovery epoch without
        convergence. It preserves every tensor, optimizer state, variational
        factor, RNG state, stopping tolerance, and patience counter. The only
        accepted configuration change is a strictly larger ``max_epochs``.
        Encoder-phase or already-converged checkpoints are rejected.
        """
        require_patched_torch(torch, operation="hybrid training checkpoint loading")
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.resolve() == destination_path.resolve():
            raise ValueError("checkpoint continuation requires a new destination")
        for name, value in {
            "source_context_sha256": source_context_sha256,
            "target_context_sha256": target_context_sha256,
        }.items():
            context = str(value)
            if len(context) != 64 or any(
                character not in "0123456789abcdef" for character in context
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(target_max_epochs, bool)
            or not isinstance(target_max_epochs, int)
            or target_max_epochs < 1
        ):
            raise ValueError("target_max_epochs must be a positive integer")

        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        if payload.get("format") != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError("not an MS2LDA HybridLDA training checkpoint")
        if payload.get("version") != TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported training checkpoint version")
        if payload.get("context_sha256") != str(source_context_sha256):
            raise ValueError("source training checkpoint context hash mismatch")
        saved_config = payload.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("training checkpoint configuration is missing")
        source_max_epochs = saved_config.get("max_epochs")
        if (
            isinstance(source_max_epochs, bool)
            or not isinstance(source_max_epochs, int)
            or target_max_epochs <= source_max_epochs
        ):
            raise ValueError("continuation must strictly increase max_epochs")
        if payload.get("phase") != "discovery":
            raise ValueError("only a discovery checkpoint can be continued")
        discovery_epochs = int(payload.get("discovery_epochs", -1))
        if not 0 < discovery_epochs <= source_max_epochs:
            raise ValueError("source checkpoint discovery epoch is invalid")
        if bool(payload.get("discovery_converged")):
            raise ValueError("a converged checkpoint does not need continuation")
        if int(payload.get("inference_epochs_completed", -1)) != 0:
            raise ValueError("continuation cannot follow encoder training")
        if int(payload.get("phase_epoch", -1)) != discovery_epochs:
            raise ValueError("source checkpoint phase epoch is inconsistent")

        rebound_config = dict(saved_config)
        rebound_config["max_epochs"] = target_max_epochs
        payload["config"] = rebound_config
        payload["context_sha256"] = str(target_context_sha256)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "discovery_epochs": discovery_epochs,
            "source_max_epochs": source_max_epochs,
            "target_max_epochs": target_max_epochs,
            "source_context_sha256": str(source_context_sha256),
            "target_context_sha256": str(target_context_sha256),
        }

    def restore_training_checkpoint(
        self,
        filename: str | Path,
        *,
        context_sha256: str,
    ) -> dict[str, int | str]:
        """Restore a safe training checkpoint onto identical attached inputs.

        The caller must first attach the same word embeddings and training
        documents used by the saved run. The context hash, model configuration,
        vocabulary, document count, sparse-matrix shape, nonzero count, and
        token total are checked before any saved optimization state is accepted.
        """
        if self._core is not None or self._epochs:
            raise RuntimeError("restore into a fresh model before calling train")
        if self._inference_only or self._inference_finalized:
            raise RuntimeError("cannot restore training state into a finalized model")
        require_patched_torch(torch, operation="hybrid training checkpoint loading")
        payload = torch.load(
            Path(filename),
            map_location=self.device,
            weights_only=True,
        )
        if payload.get("format") != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError("not an MS2LDA HybridLDA training checkpoint")
        if payload.get("version") != TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported training checkpoint version")
        if payload.get("context_sha256") != str(context_sha256):
            raise ValueError("training checkpoint context hash mismatch")
        if payload.get("config") != asdict(self.config):
            raise ValueError("training checkpoint configuration mismatch")

        self._prepare()
        if self._core is None or self._matrix is None or self._prior_optimizer is None:
            raise RuntimeError("model preparation failed")
        if list(payload.get("vocabulary", [])) != list(self._vocabulary):
            raise ValueError("training checkpoint vocabulary mismatch")
        expected_matrix = {
            "document_count": len(self.docs),
            "matrix_shape": tuple(int(value) for value in self._matrix.shape),
            "matrix_nnz": int(self._matrix.nnz),
            "matrix_token_sum": float(self._matrix.sum()),
        }
        for name, expected in expected_matrix.items():
            observed = payload.get(name)
            if observed != expected:
                raise ValueError(f"training checkpoint {name} mismatch")

        phase = str(payload.get("phase"))
        if phase not in {"discovery", "encoder"}:
            raise ValueError("training checkpoint phase is invalid")
        discovery_epochs = int(payload.get("discovery_epochs", -1))
        inference_epochs = int(payload.get("inference_epochs_completed", -1))
        if not 0 <= discovery_epochs <= self.config.max_epochs:
            raise ValueError("training checkpoint discovery epoch is invalid")
        if not 0 <= inference_epochs <= self.config.inference_epochs:
            raise ValueError("training checkpoint inference epoch is invalid")
        if (phase == "discovery" and inference_epochs) or (
            phase == "encoder" and inference_epochs < 1
        ):
            raise ValueError("training checkpoint phase counters are inconsistent")
        phase_epoch = inference_epochs if phase == "encoder" else discovery_epochs
        if int(payload.get("phase_epoch", -1)) != phase_epoch:
            raise ValueError("training checkpoint phase epoch is inconsistent")

        gamma_value = payload.get("gamma")
        if not isinstance(gamma_value, torch.Tensor):
            raise ValueError("training checkpoint gamma is missing")
        gamma = gamma_value.detach().cpu().numpy().astype(np.float32, copy=True)
        if gamma.shape != (len(self.docs), self.k):
            raise ValueError("training checkpoint gamma shape mismatch")
        if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
            raise ValueError("training checkpoint gamma is invalid")

        self._core.load_state_dict(payload["core_state_dict"], strict=True)
        self._prior_optimizer.load_state_dict(payload["prior_optimizer_state_dict"])
        self._gamma = gamma
        self._epochs = discovery_epochs
        self._converged = bool(payload.get("discovery_converged"))
        self._stable_epochs = int(payload.get("stable_epochs", 0))
        self.history = copy.deepcopy(payload.get("discovery_history", []))
        self._rng.bit_generator.state = copy.deepcopy(payload["numpy_rng_state"])
        torch_rng_state = payload.get("torch_rng_state")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("training checkpoint torch RNG state is missing")
        torch.set_rng_state(torch_rng_state.detach().cpu())
        self._inference_epochs_completed = inference_epochs
        self._inference_history = copy.deepcopy(payload.get("inference_history", []))
        self._inference_optimizer_state = copy.deepcopy(
            payload.get("inference_optimizer_state_dict")
        )
        self._inference_finalized = False
        self._core.train(bool(payload.get("module_training", False)))
        return {
            "phase": phase,
            "phase_epoch": phase_epoch,
            "discovery_epochs": discovery_epochs,
            "inference_epochs_completed": inference_epochs,
        }

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
            "vocabulary": list(self._vocabulary),
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
        require_patched_torch(torch, operation="hybrid checkpoint loading")
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
