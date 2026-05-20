#!/usr/bin/env python
"""Train a no-tomotopy variational/EM LDA-like model for the MSn benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    BACKGROUND_WEIGHT,
    EPS,
    MEMBERSHIP_THRESHOLD,
    MODEL_OUTPUT_FILENAMES,
    bow_background_distribution,
    clear_named_outputs,
    entropy_rows,
    load_input_cache,
    membership_count_diagnostics,
    normalize_rows,
    resolve_path,
    run_kl_nmf,
    sharpen_theta,
    split_indices_json_payload,
    train_validation_test_split,
    write_json,
)


THETA_EXPORT_POWER = 3.0
ALPHA = 0.1
ETA = 0.01
EM_ITERATIONS = 100
THETA_INFER_ITERS = 50
VALIDATION_EVERY = 5
NMF_MAX_ITER = 100
EXTRA_OUTPUT_FILENAMES = {
    "theta_raw.npy",
    "theta_train_raw.npy",
    "split_indices.json",
    "validation_metrics.json",
    "model_checkpoint.npz",
}


@dataclass(frozen=True)
class VariationalLDAConfig:
    input_cache: str
    out_dir: str
    n_motifs: int
    em_iterations: int
    theta_infer_iters: int
    validation_every: int
    alpha: float
    eta: float
    background_weight: float
    init: str
    nmf_max_iter: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    theta_infer_init: str
    theta_export_power: float
    membership_threshold: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a split-aware LDA-like EM benchmark without tomotopy. "
            "The model learns beta on train spectra and infers theta locally "
            "for validation/test spectra."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-motifs", type=int, default=500)
    parser.add_argument("--em-iterations", type=int, default=EM_ITERATIONS)
    parser.add_argument("--theta-infer-iters", type=int, default=THETA_INFER_ITERS)
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--background-weight", type=float, default=BACKGROUND_WEIGHT)
    parser.add_argument("--init", choices=["nmf", "random"], default="nmf")
    parser.add_argument("--nmf-max-iter", type=int, default=NMF_MAX_ITER)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--theta-infer-init",
        choices=["projection", "uniform"],
        default="projection",
        help="Initialization for held-out theta inference.",
    )
    parser.add_argument("--theta-export-power", type=float, default=THETA_EXPORT_POWER)
    parser.add_argument(
        "--membership-threshold",
        type=float,
        default=MEMBERSHIP_THRESHOLD,
        help="Threshold used only for diagnostics and benchmark export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, out_dir: Path) -> VariationalLDAConfig:
    return VariationalLDAConfig(
        input_cache=str(resolve_path(args.input_cache)),
        out_dir=str(out_dir),
        n_motifs=int(args.n_motifs),
        em_iterations=int(args.em_iterations),
        theta_infer_iters=int(args.theta_infer_iters),
        validation_every=int(args.validation_every),
        alpha=float(args.alpha),
        eta=float(args.eta),
        background_weight=float(args.background_weight),
        init=str(args.init),
        nmf_max_iter=int(args.nmf_max_iter),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        test_fraction=float(args.test_fraction),
        theta_infer_init=str(args.theta_infer_init),
        theta_export_power=float(args.theta_export_power),
        membership_threshold=float(args.membership_threshold),
        seed=int(args.seed),
    )


def initialize_random(
    n_docs: int,
    vocab_size: int,
    n_topics: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    theta = rng.gamma(1.0, 1.0, size=(int(n_docs), int(n_topics)))
    beta = rng.gamma(1.0, 1.0, size=(int(n_topics), int(vocab_size)))
    return (
        normalize_rows(theta),
        normalize_rows(beta),
        {"init": "random", "seed": int(seed)},
    )


def initialize_model(
    x_train,
    *,
    n_topics: int,
    config: VariationalLDAConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config.init == "random":
        return initialize_random(
            x_train.shape[0],
            x_train.shape[1],
            n_topics,
            seed=config.seed,
        )
    theta, beta, metadata = run_kl_nmf(
        x_train,
        n_motifs=n_topics,
        max_iter=config.nmf_max_iter,
        seed=config.seed,
    )
    metadata = {"init": "nmf", **metadata}
    return theta, beta, metadata


def initialize_theta_for_inference(
    x,
    beta: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "uniform":
        return np.full((x.shape[0], beta.shape[0]), 1.0 / beta.shape[0], dtype=np.float32)
    projected = x @ beta.T
    return normalize_rows(np.asarray(projected, dtype=np.float64) + EPS)


def em_step(
    x,
    theta: np.ndarray,
    beta: np.ndarray,
    background: np.ndarray,
    *,
    alpha: float,
    eta: float,
    background_weight: float,
    update_beta: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int]]:
    x = x.tocsr()
    n_docs, vocab_size = x.shape
    n_topics = beta.shape[0]
    theta_counts = np.full((n_docs, n_topics), float(alpha), dtype=np.float64)
    beta_counts = (
        np.full((n_topics, vocab_size), float(eta), dtype=np.float64)
        if update_beta
        else None
    )
    total_loss = 0.0
    total_tokens = 0.0
    empty_documents = 0
    bg_weight = float(background_weight)
    motif_weight = 1.0 - bg_weight

    for doc_pos in range(n_docs):
        start = x.indptr[doc_pos]
        end = x.indptr[doc_pos + 1]
        if start == end:
            empty_documents += 1
            continue
        words = x.indices[start:end]
        counts = x.data[start:end].astype(np.float64, copy=False)
        beta_sub = beta[:, words]
        topic_word = theta[doc_pos, :, None] * beta_sub
        motif_prob = topic_word.sum(axis=0)
        if bg_weight > 0:
            numerator = motif_weight * topic_word
            dist = (motif_weight * motif_prob) + (bg_weight * background[words])
        else:
            numerator = topic_word
            dist = motif_prob
        dist = np.clip(dist, EPS, None)
        responsibilities = numerator / dist[None, :]
        weighted = responsibilities * counts[None, :]
        theta_counts[doc_pos] += weighted.sum(axis=1)
        if beta_counts is not None:
            beta_counts[:, words] += weighted
        total_loss += float(-(counts * np.log(dist)).sum())
        total_tokens += float(counts.sum())

    theta_new = normalize_rows(theta_counts)
    beta_new = normalize_rows(beta_counts) if beta_counts is not None else None
    metrics = {
        "reconstruction": total_loss / max(total_tokens, EPS),
        "total_reconstruction": total_loss,
        "tokens": total_tokens,
        "empty_documents": int(empty_documents),
    }
    return theta_new, beta_new, metrics


def infer_theta(
    x,
    beta: np.ndarray,
    background: np.ndarray,
    *,
    config: VariationalLDAConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    theta = initialize_theta_for_inference(x, beta, mode=config.theta_infer_init)
    metrics: dict[str, float | int] = {}
    for _ in range(int(config.theta_infer_iters)):
        theta, _beta_unused, metrics = em_step(
            x,
            theta,
            beta,
            background,
            alpha=config.alpha,
            eta=config.eta,
            background_weight=config.background_weight,
            update_beta=False,
        )
    return theta, metrics


def theta_summary(
    theta: np.ndarray,
    *,
    membership_threshold: float,
) -> dict[str, float | int]:
    counts = (theta >= float(membership_threshold)).sum(axis=0)
    entropy = entropy_rows(theta)
    return {
        "membership_rows_above_threshold": int(
            np.sum(theta >= float(membership_threshold))
        ),
        "active_topics_above_threshold": int(np.sum(counts > 0)),
        "topics_with_1_to_10_memberships": int(np.sum((counts >= 1) & (counts <= 10))),
        "mean_max_theta": float(np.max(theta, axis=1).mean()),
        "mean_theta_entropy": float(entropy.mean()),
    }


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["validation_topics_with_1_to_10_memberships"]),
        float(metrics["validation_active_topics_above_threshold"]),
        -float(metrics["validation_reconstruction"]),
    )


def evaluate_validation(
    x_validation,
    beta: np.ndarray,
    background: np.ndarray,
    *,
    config: VariationalLDAConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    theta_validation, inference_metrics = infer_theta(
        x_validation,
        beta,
        background,
        config=config,
    )
    theta_validation_export = sharpen_theta(theta_validation, config.theta_export_power)
    metrics = {
        **{
            f"validation_{key}": value
            for key, value in theta_summary(
                theta_validation_export,
                membership_threshold=config.membership_threshold,
            ).items()
        },
        "validation_reconstruction": float(inference_metrics["reconstruction"]),
        "validation_total_reconstruction": float(
            inference_metrics["total_reconstruction"]
        ),
        "validation_empty_documents": int(inference_metrics["empty_documents"]),
    }
    return theta_validation, metrics


def train_variational_lda(
    x_train,
    x_validation,
    *,
    background: np.ndarray,
    config: VariationalLDAConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    theta, beta, init_metadata = initialize_model(
        x_train,
        n_topics=config.n_motifs,
        config=config,
    )
    history: list[dict[str, Any]] = []
    best_theta = copy.deepcopy(theta)
    best_beta = copy.deepcopy(beta)
    best_metrics: dict[str, Any] = {}
    best_key: tuple[float, float, float] | None = None

    for iteration in range(1, int(config.em_iterations) + 1):
        theta, beta_new, train_metrics = em_step(
            x_train,
            theta,
            beta,
            background,
            alpha=config.alpha,
            eta=config.eta,
            background_weight=config.background_weight,
            update_beta=True,
        )
        if beta_new is None:
            raise RuntimeError("Expected beta update during training.")
        beta = beta_new

        row: dict[str, Any] = {
            "iteration": int(iteration),
            "train_reconstruction": float(train_metrics["reconstruction"]),
            "train_total_reconstruction": float(train_metrics["total_reconstruction"]),
            "train_empty_documents": int(train_metrics["empty_documents"]),
        }
        should_validate = (
            iteration == 1
            or iteration == int(config.em_iterations)
            or iteration % max(int(config.validation_every), 1) == 0
        )
        if should_validate:
            _theta_validation, validation_metrics = evaluate_validation(
                x_validation,
                beta,
                background,
                config=config,
            )
            row.update(validation_metrics)
            key = validation_selection_key(row)
            if best_key is None or key > best_key:
                best_key = key
                best_theta = copy.deepcopy(theta)
                best_beta = copy.deepcopy(beta)
                best_metrics = dict(row)
                row["best_checkpoint"] = True
        history.append(row)
        print(json.dumps(row))

    return best_theta, best_beta, history, {"initialization": init_metadata, **best_metrics}


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def clear_outputs(out_dir: Path) -> None:
    clear_named_outputs(out_dir, MODEL_OUTPUT_FILENAMES | EXTRA_OUTPUT_FILENAMES)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    input_cache = resolve_path(args.input_cache)
    out_dir = resolve_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_outputs(out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = load_input_cache(input_cache)
    x = cache["matrix"].tocsr()
    vocab = cache["vocab"]
    splits = train_validation_test_split(
        x.shape[0],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_json(out_dir / "split_indices.json", split_indices_json_payload(splits))
    config = build_config(args, out_dir)
    train_idx = splits["train_indices"]
    validation_idx = splits["validation_indices"]
    test_idx = splits["test_indices"]
    x_train = x[train_idx]
    x_validation = x[validation_idx]
    x_test = x[test_idx]
    background = bow_background_distribution(x_train)

    train_theta_raw, beta, history, training_metadata = train_variational_lda(
        x_train,
        x_validation,
        background=background,
        config=config,
    )
    validation_theta_raw, validation_inference_metrics = infer_theta(
        x_validation,
        beta,
        background,
        config=config,
    )
    test_theta_raw, test_inference_metrics = infer_theta(
        x_test,
        beta,
        background,
        config=config,
    )

    theta_raw = np.empty((x.shape[0], config.n_motifs), dtype=np.float32)
    theta_raw[train_idx] = train_theta_raw.astype(np.float32, copy=False)
    theta_raw[validation_idx] = validation_theta_raw.astype(np.float32, copy=False)
    theta_raw[test_idx] = test_theta_raw.astype(np.float32, copy=False)
    theta = sharpen_theta(theta_raw, config.theta_export_power)
    beta = normalize_rows(beta)
    beta_entropy = entropy_rows(beta)
    all_idx = np.arange(x.shape[0], dtype=np.int64)
    metrics = {
        **prefixed(
            "train",
            theta_summary(
                theta[train_idx],
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "validation",
            theta_summary(
                theta[validation_idx],
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "test",
            theta_summary(
                theta[test_idx],
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "all",
            theta_summary(
                theta[all_idx],
                membership_threshold=config.membership_threshold,
            ),
        ),
        "validation_inference_reconstruction": float(
            validation_inference_metrics["reconstruction"]
        ),
        "test_inference_reconstruction": float(test_inference_metrics["reconstruction"]),
        "mean_beta_entropy": float(beta_entropy.mean()),
        "mean_beta_effective_support": float(np.exp(beta_entropy).mean()),
        "mean_beta_max_probability": float(beta.max(axis=1).mean()),
        "membership_count_diagnostics": {
            "train": membership_count_diagnostics(theta[train_idx]),
            "validation": membership_count_diagnostics(theta[validation_idx]),
            "test": membership_count_diagnostics(theta[test_idx]),
            "all": membership_count_diagnostics(theta),
        },
        "training_metadata": training_metadata,
    }

    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "theta_raw.npy", theta_raw)
    np.save(out_dir / "theta_train_raw.npy", train_theta_raw)
    np.save(out_dir / "beta.npy", beta)
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    write_json(out_dir / "train_history.json", {"history": history})
    write_json(out_dir / "validation_metrics.json", metrics)
    np.savez_compressed(
        out_dir / "model_checkpoint.npz",
        beta=beta,
        train_theta_raw=train_theta_raw,
        background=background,
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_idx,
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "variational-lda-em",
        "config": asdict(config),
        "input": {
            "input_cache": str(input_cache),
            "documents": int(x.shape[0]),
            "vocab_size": int(x.shape[1]),
            "topics": int(beta.shape[0]),
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(validation_idx)),
            "test_documents": int(len(test_idx)),
        },
        "metrics": metrics,
        "outputs": {
            "theta_npy": str(out_dir / "theta.npy"),
            "theta_raw_npy": str(out_dir / "theta_raw.npy"),
            "theta_train_raw_npy": str(out_dir / "theta_train_raw.npy"),
            "beta_npy": str(out_dir / "beta.npy"),
            "vocab_json": str(out_dir / "vocab.json"),
            "train_history_json": str(out_dir / "train_history.json"),
            "validation_metrics_json": str(out_dir / "validation_metrics.json"),
            "split_indices_json": str(out_dir / "split_indices.json"),
            "model_checkpoint_npz": str(out_dir / "model_checkpoint.npz"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
