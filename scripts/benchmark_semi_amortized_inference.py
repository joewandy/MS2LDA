#!/usr/bin/env python
"""Compare the supported inference method with one synthetic-corpus baseline.

This benchmark intentionally removes topic-discovery variance.  Every method
receives the same known topic posterior, encoder initialization, documents, and
embeddings. Only the frozen-topic encoder objective changes: the fixed
two-step semi-amortized ELBO is compared with equal-epoch posterior regression.

The benchmark reports the local-ELBO gap to the per-document best long solve
from the symmetric and both trained encoder initializers, posterior KL, and
observed-token NLL at several inference budgets. It is an inference algorithm
check, not evidence about chemical Mass2Motif quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MS2LDA.hybrid_lda import (
    EPSILON,
    INFERENCE_REFINEMENT_STEPS,
    HybridLDAConfig,
    HybridLDAModel,
    _expected_log_dirichlet,
    _local_document_elbo,
    _local_vb,
    _make_sparse_batch,
    observed_token_nll,
)
from scripts.inference_baselines import fit_posterior_regression_baseline


@dataclass(frozen=True)
class SyntheticCorpus:
    """Known topics plus train/test spectral words and dense embeddings."""

    vocabulary: list[str]
    beta: np.ndarray
    train_words: list[list[str]]
    train_embeddings: np.ndarray
    test_words: list[list[str]]
    test_embeddings: np.ndarray


@dataclass(frozen=True)
class ReferenceSolution:
    """Per-document best long local solve across all compared initializers."""

    gamma: np.ndarray
    elbo: np.ndarray
    tail_gain_per_token: float
    source_counts: dict[str, int]


METHODS = ("posterior_regression_baseline", "semi_amortized")
REFERENCE_STEPS = 100
REFERENCE_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    """Parse reproducible benchmark options."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seeds", default="23,29,31")
    parser.add_argument("--train-documents", type=int, default=240)
    parser.add_argument("--test-documents", type=int, default=80)
    parser.add_argument("--tokens-per-document", type=int, default=40)
    parser.add_argument("--encoder-epochs", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_corpus(
    *,
    seed: int,
    train_documents: int,
    test_documents: int,
    tokens_per_document: int,
) -> SyntheticCorpus:
    """Generate three partially overlapping topics and noisy embeddings."""
    rng = np.random.default_rng(seed)
    num_topics = 3
    vocab_size = 18
    embedding_dim = 12
    vocabulary = [f"frag@{100 + index:.2f}" for index in range(vocab_size)]
    beta = np.full((num_topics, vocab_size), 0.005, dtype=np.float64)
    for topic in range(num_topics):
        start = 6 * topic
        beta[topic, start : start + 6] += np.asarray(
            [0.32, 0.24, 0.17, 0.12, 0.08, 0.04]
        )
    beta /= beta.sum(axis=1, keepdims=True)
    projection = rng.normal(size=(num_topics, embedding_dim))

    def sample(count: int) -> tuple[list[list[str]], np.ndarray]:
        documents: list[list[str]] = []
        embeddings = np.empty((count, embedding_dim), dtype=np.float32)
        for row in range(count):
            theta = rng.dirichlet(np.full(num_topics, 0.25))
            probabilities = theta @ beta
            counts = rng.multinomial(tokens_per_document, probabilities)
            documents.append(
                [
                    word
                    for word, repetitions in zip(vocabulary, counts, strict=True)
                    for _ in range(int(repetitions))
                ]
            )
            embeddings[row] = (
                theta @ projection + rng.normal(scale=0.20, size=embedding_dim)
            ).astype(np.float32)
        return documents, embeddings

    train_words, train_embeddings = sample(train_documents)
    test_words, test_embeddings = sample(test_documents)
    return SyntheticCorpus(
        vocabulary=vocabulary,
        beta=beta.astype(np.float32),
        train_words=train_words,
        train_embeddings=train_embeddings,
        test_words=test_words,
        test_embeddings=test_embeddings,
    )


def build_frozen_topic_model(
    corpus: SyntheticCorpus,
    *,
    seed: int,
    inference_epochs: int,
) -> HybridLDAModel:
    """Build the same seeded encoder and install the known frozen topics."""
    config = HybridLDAConfig(
        num_topics=corpus.beta.shape[0],
        embedding_dim=corpus.train_embeddings.shape[1],
        alpha=0.2,
        eta=0.01,
        hidden_size=64,
        feature_projection_dim=24,
        training_local_steps=30,
        batch_size=64,
        inference_epochs=inference_epochs,
        prior_mass_fraction=0.0,
        prior_warmup_epochs=1,
        prior_training_epochs=1,
        max_epochs=2,
        seed=seed,
    )
    model = HybridLDAModel(config)
    for words, embedding in zip(
        corpus.train_words,
        corpus.train_embeddings,
        strict=True,
    ):
        model.add_doc(words, embedding=embedding)
    model.train(0)
    if model._core is None:
        raise RuntimeError("model core was not prepared")
    source_columns = {word: index for index, word in enumerate(corpus.vocabulary)}
    aligned_beta = np.column_stack(
        [corpus.beta[:, source_columns[word]] for word in model.used_vocabs]
    )
    posterior_mass = 2_000.0
    model._core.lambda_posterior.copy_(
        torch.as_tensor(
            config.eta + posterior_mass * aligned_beta,
            dtype=model._core.lambda_posterior.dtype,
        )
    )
    # This harness installs an already learned posterior, so mark one completed
    # global epoch without running a discovery update that would alter it.
    model._epochs = 1
    return model


@torch.no_grad()
def build_common_reference(
    models: dict[str, HybridLDAModel],
    corpus: SyntheticCorpus,
) -> ReferenceSolution:
    """Solve from every compared basin and retain the best result per document.

    Local LDA inference can have initializer-dependent fixed points. A common
    reference must therefore include the symmetric start and the trained
    initializer from every benchmark arm; otherwise an arm can legitimately
    exceed the purported reference and produce a misleading negative gap.
    """
    if not models:
        raise ValueError("at least one model is required for the reference solve")
    template = next(iter(models.values()))
    if template._core is None:
        raise RuntimeError("template model core is not prepared")
    documents = [
        template._new_document(words, embedding)
        for words, embedding in zip(
            corpus.test_words,
            corpus.test_embeddings,
            strict=True,
        )
    ]
    matrix = template._documents_to_matrix(documents)
    indices = np.arange(len(documents))
    batch = _make_sparse_batch(matrix, indices, device=template.device)
    core = template._core
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    starts = [("symmetric", core.alpha.unsqueeze(0) + batch.totals / template.k)]
    for name, model in models.items():
        if model._core is None:
            raise RuntimeError(f"model core is not prepared for {name}")
        if model.device != template.device or model.used_vocabs != template.used_vocabs:
            raise RuntimeError("reference models do not share device and vocabulary")
        if not torch.equal(model._core.lambda_posterior, core.lambda_posterior):
            raise RuntimeError("reference models do not share frozen topics")
        model._core.eval()
        starts.append(
            (
                name,
                model._core.encode(
                    batch,
                    model._embedding_batch(documents, indices),
                    word_topic,
                ),
            )
        )

    best_gamma = torch.zeros_like(starts[0][1])
    best_elbo = torch.full(
        (len(documents),),
        -torch.inf,
        device=template.device,
        dtype=best_gamma.dtype,
    )
    best_source = torch.full(
        (len(documents),),
        -1,
        device=template.device,
        dtype=torch.long,
    )
    selected_tail_gain = torch.zeros_like(best_elbo)
    for source_index, (_, initial_gamma) in enumerate(starts):
        solved_gamma, _ = _local_vb(
            batch,
            initial_gamma,
            core.alpha,
            expected_log_beta,
            steps=REFERENCE_STEPS,
            tolerance=REFERENCE_TOLERANCE,
        )
        solved_elbo = _local_document_elbo(
            batch,
            solved_gamma,
            core.alpha,
            expected_log_beta,
        )
        tail_gamma, _ = _local_vb(
            batch,
            solved_gamma,
            core.alpha,
            expected_log_beta,
            steps=1,
            tolerance=None,
        )
        tail_elbo = _local_document_elbo(
            batch,
            tail_gamma,
            core.alpha,
            expected_log_beta,
        )
        tail_is_better = tail_elbo > solved_elbo
        candidate_gamma = torch.where(
            tail_is_better.unsqueeze(1),
            tail_gamma,
            solved_gamma,
        )
        candidate_elbo = torch.maximum(solved_elbo, tail_elbo)
        candidate_tail_gain = torch.clamp(tail_elbo - solved_elbo, min=0.0)
        candidate_is_better = candidate_elbo > best_elbo
        best_gamma = torch.where(
            candidate_is_better.unsqueeze(1),
            candidate_gamma,
            best_gamma,
        )
        best_elbo = torch.maximum(best_elbo, candidate_elbo)
        best_source = torch.where(
            candidate_is_better,
            torch.full_like(best_source, source_index),
            best_source,
        )
        selected_tail_gain = torch.where(
            candidate_is_better,
            candidate_tail_gain,
            selected_tail_gain,
        )

    total_tokens = float(batch.totals.sum().cpu())
    return ReferenceSolution(
        gamma=best_gamma.cpu().numpy(),
        elbo=best_elbo.cpu().numpy(),
        tail_gain_per_token=float(selected_tail_gain.sum().cpu()) / total_tokens,
        source_counts={
            source: int((best_source == source_index).sum().cpu())
            for source_index, (source, _) in enumerate(starts)
        },
    )


@torch.no_grad()
def evaluate_model(
    model: HybridLDAModel,
    corpus: SyntheticCorpus,
    reference: ReferenceSolution,
    *,
    training_refinement_steps: int,
) -> dict[str, Any]:
    """Evaluate fixed inference budgets against one common reference."""
    if model._core is None:
        raise RuntimeError("model core is not prepared")
    documents = [
        model._new_document(words, embedding)
        for words, embedding in zip(
            corpus.test_words,
            corpus.test_embeddings,
            strict=True,
        )
    ]
    matrix = model._documents_to_matrix(documents)
    indices = np.arange(len(documents))
    batch = _make_sparse_batch(matrix, indices, device=model.device)
    core = model._core
    expected_log_beta = _expected_log_dirichlet(core.lambda_posterior)
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1)
    test_embeddings = model._embedding_batch(documents, indices)
    reference_gamma = torch.as_tensor(
        reference.gamma,
        device=model.device,
        dtype=core.lambda_posterior.dtype,
    )
    reference_elbo = torch.as_tensor(
        reference.elbo,
        device=model.device,
        dtype=core.lambda_posterior.dtype,
    )
    if reference_gamma.shape != (len(documents), model.k):
        raise ValueError("reference gamma is not aligned to evaluation documents")
    reference_theta = reference_gamma / reference_gamma.sum(dim=1, keepdim=True)
    total_tokens = float(batch.totals.sum().cpu())
    beta = core.beta_mean().cpu().numpy()
    budgets: dict[str, dict[str, float]] = {}
    evaluation_budgets = sorted({0, 1, 2, 5, 20, training_refinement_steps})
    for steps in evaluation_budgets:
        timings: list[float] = []
        gamma: torch.Tensor | None = None
        for _ in range(3):
            started = time.perf_counter()
            gamma_zero = core.encode(batch, test_embeddings, word_topic)
            gamma = (
                _local_vb(
                    batch,
                    gamma_zero,
                    core.alpha,
                    expected_log_beta,
                    steps=steps,
                    tolerance=None,
                )[0]
                if steps
                else gamma_zero
            )
            timings.append(time.perf_counter() - started)
        if gamma is None:
            raise RuntimeError("inference timing did not run")
        elbo = _local_document_elbo(batch, gamma, core.alpha, expected_log_beta)
        theta = gamma / gamma.sum(dim=1, keepdim=True)
        posterior_kl = (
            reference_theta
            * (
                reference_theta.clamp_min(EPSILON).log()
                - theta.clamp_min(EPSILON).log()
            )
        ).sum(dim=1)
        gap = reference_elbo - elbo
        budgets[str(steps)] = {
            "elbo_gap_per_token": float(gap.sum().cpu()) / total_tokens,
            "reference_mean_kl": float(posterior_kl.mean().cpu()),
            "observed_token_nll": observed_token_nll(
                matrix,
                theta.cpu().numpy(),
                beta,
            ),
            "median_milliseconds_including_encoder": 1_000.0
            * float(np.median(timings)),
        }
    return {
        "reference_tail_gain_per_token": reference.tail_gain_per_token,
        "reference_source_counts": dict(reference.source_counts),
        "budgets": budgets,
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Train both inference methods from identical frozen-topic models."""
    corpus = make_corpus(
        seed=seed,
        train_documents=args.train_documents,
        test_documents=args.test_documents,
        tokens_per_document=args.tokens_per_document,
    )
    results: dict[str, Any] = {"seed": seed, "methods": {}}
    models: dict[str, HybridLDAModel] = {}
    training_records: dict[str, tuple[float, list[dict[str, float]]]] = {}
    for name in METHODS:
        model = build_frozen_topic_model(
            corpus,
            seed=seed,
            inference_epochs=args.encoder_epochs,
        )
        if model._core is None:
            raise RuntimeError("model core is not prepared")
        topics_before = model._core.lambda_posterior.detach().clone()
        prior_before = [
            parameter.detach().clone() for parameter in model._core.prior_parameters()
        ]
        started = time.perf_counter()
        if name == "semi_amortized":
            history = model.finalize_inference()
        else:
            history = fit_posterior_regression_baseline(
                model,
                epochs=args.encoder_epochs,
                target_steps=model.config.training_local_steps,
            )
        training_seconds = time.perf_counter() - started
        if not torch.equal(topics_before, model._core.lambda_posterior):
            raise RuntimeError("an inference objective changed the frozen topics")
        if any(
            not torch.equal(previous, current)
            for previous, current in zip(
                prior_before,
                model._core.prior_parameters(),
                strict=True,
            )
        ):
            raise RuntimeError("an inference objective changed the structured prior")
        models[name] = model
        training_records[name] = (training_seconds, history)

    reference = build_common_reference(models, corpus)
    for name in METHODS:
        model = models[name]
        training_seconds, history = training_records[name]
        results["methods"][name] = {
            "training_seconds": training_seconds,
            "final_training_metrics": history[-1],
            **evaluate_model(
                model,
                corpus,
                reference,
                training_refinement_steps=INFERENCE_REFINEMENT_STEPS,
            ),
        }
    return results


def aggregate(
    runs: list[dict[str, Any]],
    *,
    training_refinement_steps: int,
) -> dict[str, Any]:
    """Calculate mean metrics and compare the supported method to baseline."""
    methods: dict[str, Any] = {}
    for name in METHODS:
        budgets: dict[str, Any] = {}
        available_budgets = runs[0]["methods"][name]["budgets"]
        for budget in sorted(available_budgets, key=int):
            metrics = {
                key: [run["methods"][name]["budgets"][budget][key] for run in runs]
                for key in runs[0]["methods"][name]["budgets"][budget]
            }
            budgets[budget] = {
                f"mean_{key}": float(np.mean(values)) for key, values in metrics.items()
            }
        methods[name] = {"budgets": budgets}
    comparison_budget = str(training_refinement_steps)
    baseline_gap = methods["posterior_regression_baseline"]["budgets"][
        comparison_budget
    ]["mean_elbo_gap_per_token"]
    baseline_zero_nll = methods["posterior_regression_baseline"]["budgets"]["0"][
        "mean_observed_token_nll"
    ]
    candidate_gap = methods["semi_amortized"]["budgets"][comparison_budget][
        "mean_elbo_gap_per_token"
    ]
    candidate_zero_nll = methods["semi_amortized"]["budgets"]["0"][
        "mean_observed_token_nll"
    ]
    gap_reduction = (baseline_gap - candidate_gap) / max(baseline_gap, EPSILON)
    zero_nll_change = (candidate_zero_nll - baseline_zero_nll) / max(
        baseline_zero_nll,
        EPSILON,
    )
    paired_reductions = []
    for run in runs:
        paired_baseline = run["methods"]["posterior_regression_baseline"]["budgets"][
            comparison_budget
        ]["elbo_gap_per_token"]
        paired_candidate = run["methods"]["semi_amortized"]["budgets"][
            comparison_budget
        ]["elbo_gap_per_token"]
        paired_reductions.append(
            (paired_baseline - paired_candidate) / max(abs(paired_baseline), EPSILON)
        )
    return {
        "methods": methods,
        "comparison_to_posterior_regression": {
            "comparison_budget": training_refinement_steps,
            "relative_reduction_in_across_seed_mean_gap": gap_reduction,
            "paired_gap_reductions": paired_reductions,
            "fractional_change_in_across_seed_mean_zero_step_nll": zero_nll_change,
            "passes_exploratory_acceptance_criterion": gap_reduction >= 0.10
            and zero_nll_change <= 0.05,
        },
    }


def main() -> None:
    """Run the benchmark, print JSON, and optionally persist it."""
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    runs = [run_seed(args, seed) for seed in seeds]
    report = {
        "configuration": {
            "seeds": seeds,
            "train_documents": args.train_documents,
            "test_documents": args.test_documents,
            "tokens_per_document": args.tokens_per_document,
            "encoder_epochs": args.encoder_epochs,
            "refinement_steps": INFERENCE_REFINEMENT_STEPS,
            "reference_steps": REFERENCE_STEPS,
            "reference_tolerance": REFERENCE_TOLERANCE,
            "reference_initializers": ["symmetric", *METHODS],
        },
        "runs": runs,
        "aggregate": aggregate(
            runs,
            training_refinement_steps=INFERENCE_REFINEMENT_STEPS,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
