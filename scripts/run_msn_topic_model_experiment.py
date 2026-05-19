#!/usr/bin/env python
"""Train topic-model baselines and neural variants on cached MSn benchmark input."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    BACKGROUND_WEIGHT,
    OUTPUT_FILENAMES,
    THETA_EXPORT_POWER,
    TOPIC_OVERLAP_WEIGHT,
    TOPIC_USAGE_WEIGHT,
    build_bow_matrix,
    clear_named_outputs,
    entropy_rows,
    export_memberships,
    load_input_cache,
    membership_count_diagnostics,
    normalize_rows,
    prepare_msn_documents,
    resolve_path,
    run_kl_nmf,
    run_tomotopy_lda,
    select_eval_topic_ids,
    sharpen_theta,
    sparsemax,
    train_neural_lda,
    train_sparse_neural,
    write_checkpoint,
    write_json,
)


@dataclass(frozen=True)
class RunConfig:
    dataset: str | None
    input_cache: str | None
    model: str
    out_dir: str
    limit_spectra: int | None
    n_motifs: int
    lda_iterations: int
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    dropout: float
    min_df: int
    min_cf: float
    rm_top: int
    nmf_max_iter: int
    theta_entropy_weight: float
    beta_entropy_weight: float
    background_weight: float
    topic_overlap_weight: float
    topic_usage_weight: float
    theta_activation: str
    beta_activation: str
    topic_usage_mode: str
    beta_target_support: float
    beta_target_weight: float
    theta_init_strength: float
    theta_export_power: float
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LDA, NMF, or neural topic models for the MSn benchmark. "
            "Use prepare_msn_benchmark_input.py first for full-data runs."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path)
    inputs.add_argument("--input-cache", type=Path)
    parser.add_argument(
        "--model", required=True, choices=["lda", "nmf", "sparse-neural", "neural-lda"]
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit-spectra", type=int, default=500)
    parser.add_argument("--n-motifs", type=int, default=200)
    parser.add_argument("--lda-iterations", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--min-cf", type=float, default=0.0)
    parser.add_argument("--rm-top", type=int, default=0)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--theta-entropy-weight", type=float, default=0.1)
    parser.add_argument("--beta-entropy-weight", type=float, default=0.1)
    parser.add_argument("--background-weight", type=float, default=BACKGROUND_WEIGHT)
    parser.add_argument(
        "--topic-overlap-weight", type=float, default=TOPIC_OVERLAP_WEIGHT
    )
    parser.add_argument("--topic-usage-weight", type=float, default=TOPIC_USAGE_WEIGHT)
    parser.add_argument(
        "--theta-activation",
        choices=["sparsemax", "softmax"],
        default="sparsemax",
    )
    parser.add_argument(
        "--beta-activation",
        choices=["sparsemax", "softmax"],
        default="sparsemax",
    )
    parser.add_argument(
        "--topic-usage-mode",
        choices=["mse", "entropy"],
        default="mse",
    )
    parser.add_argument("--beta-target-support", type=float, default=0.0)
    parser.add_argument("--beta-target-weight", type=float, default=0.0)
    parser.add_argument("--theta-init-strength", type=float, default=8.0)
    parser.add_argument("--theta-export-power", type=float, default=THETA_EXPORT_POWER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-eval-motifs", type=int, default=50, help=argparse.SUPPRESS
    )
    parser.add_argument("--motif-top-n", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument(
        "--membership-threshold", type=float, default=0.5, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def load_training_input(
    args: argparse.Namespace,
) -> tuple[Any, list[str], list[list[str]] | None, dict]:
    if args.input_cache is not None:
        cache_dir = resolve_path(args.input_cache)
        cache = load_input_cache(cache_dir, require_documents=args.model == "lda")
        summary = cache["summary"]
        vocab_params = summary.get("vocabulary_parameters", {})
        return (
            cache["matrix"],
            cache["vocab"],
            cache.get("documents"),
            {
                "source": "input_cache",
                "cache_dir": str(cache_dir),
                "cache": summary.get("input", {}),
                "vocabulary_parameters": vocab_params,
            },
        )

    dataset = resolve_path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    spectra, documents, input_metadata = prepare_msn_documents(
        dataset,
        limit_spectra=args.limit_spectra,
    )
    x, vocab, bow_metadata = build_bow_matrix(
        documents,
        min_df=args.min_df,
        min_cf=args.min_cf,
        rm_top=args.rm_top,
    )
    return (
        x,
        vocab,
        documents,
        {
            "source": "dataset",
            "dataset": str(dataset),
            "cache": {**input_metadata, **bow_metadata},
            "vocabulary_parameters": {
                "min_df": int(args.min_df),
                "min_cf": float(args.min_cf),
                "rm_top": int(args.rm_top),
            },
        },
    )


def train_model(
    args: argparse.Namespace,
    x,
    vocab: list[str],
    documents: list[list[str]] | None,
    input_metadata: dict,
) -> tuple[
    np.ndarray, np.ndarray, list[str], list[dict[str, float]], dict, dict | None
]:
    vocab_params = input_metadata.get("vocabulary_parameters", {})
    history: list[dict[str, float]] = []
    checkpoint = None
    if args.model == "lda":
        if documents is None:
            raise ValueError(
                "LDA training requires documents.jsonl.gz in the input cache."
            )
        theta, beta, vocab, history, model_metadata = run_tomotopy_lda(
            documents,
            n_motifs=args.n_motifs,
            min_df=int(vocab_params.get("min_df", args.min_df)),
            min_cf=float(vocab_params.get("min_cf", args.min_cf)),
            rm_top=int(vocab_params.get("rm_top", args.rm_top)),
            lda_iterations=args.lda_iterations,
            seed=args.seed,
        )
    elif args.model == "nmf":
        theta, beta, model_metadata = run_kl_nmf(
            x,
            n_motifs=args.n_motifs,
            max_iter=args.nmf_max_iter,
            seed=args.seed,
        )
    elif args.model == "sparse-neural":
        theta, beta, history, model_metadata, checkpoint = train_sparse_neural(
            x,
            n_motifs=args.n_motifs,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            theta_entropy_weight=args.theta_entropy_weight,
            beta_entropy_weight=args.beta_entropy_weight,
            background_weight=args.background_weight,
            topic_overlap_weight=args.topic_overlap_weight,
            topic_usage_weight=args.topic_usage_weight,
            theta_activation=args.theta_activation,
            beta_activation=args.beta_activation,
            topic_usage_mode=args.topic_usage_mode,
            seed=args.seed,
            device=str(args.device),
        )
        theta = sharpen_theta(theta, args.theta_export_power)
        model_metadata["theta_export_power"] = float(args.theta_export_power)
    else:
        theta, beta, history, model_metadata, checkpoint = train_neural_lda(
            x,
            n_motifs=args.n_motifs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            theta_entropy_weight=args.theta_entropy_weight,
            topic_usage_weight=args.topic_usage_weight,
            beta_target_support=args.beta_target_support,
            beta_target_weight=args.beta_target_weight,
            background_weight=args.background_weight,
            theta_init_strength=args.theta_init_strength,
            seed=args.seed,
            device=str(args.device),
        )
        theta = sharpen_theta(theta, args.theta_export_power)
        model_metadata["theta_export_power"] = float(args.theta_export_power)
    return theta, beta, vocab, history, model_metadata, checkpoint


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_named_outputs(out_dir, OUTPUT_FILENAMES)

    random.seed(args.seed)
    np.random.seed(args.seed)

    x, vocab, documents, input_metadata = load_training_input(args)
    theta, beta, vocab, history, model_metadata, checkpoint = train_model(
        args,
        x,
        vocab,
        documents,
        input_metadata,
    )

    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "beta.npy", beta)
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    write_json(out_dir / "train_history.json", {"history": history})
    write_checkpoint(out_dir / "model_checkpoint.pt", checkpoint)

    config = RunConfig(
        dataset=str(resolve_path(args.dataset)) if args.dataset is not None else None,
        input_cache=(
            str(resolve_path(args.input_cache))
            if args.input_cache is not None
            else None
        ),
        model=str(args.model),
        out_dir=str(out_dir),
        limit_spectra=args.limit_spectra if args.dataset is not None else None,
        n_motifs=int(args.n_motifs),
        lda_iterations=int(args.lda_iterations),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
        min_df=int(args.min_df),
        min_cf=float(args.min_cf),
        rm_top=int(args.rm_top),
        nmf_max_iter=int(args.nmf_max_iter),
        theta_entropy_weight=float(args.theta_entropy_weight),
        beta_entropy_weight=float(args.beta_entropy_weight),
        background_weight=float(args.background_weight),
        topic_overlap_weight=float(args.topic_overlap_weight),
        topic_usage_weight=float(args.topic_usage_weight),
        theta_activation=str(args.theta_activation),
        beta_activation=str(args.beta_activation),
        topic_usage_mode=str(args.topic_usage_mode),
        beta_target_support=float(args.beta_target_support),
        beta_target_weight=float(args.beta_target_weight),
        theta_init_strength=float(args.theta_init_strength),
        theta_export_power=float(args.theta_export_power),
        seed=int(args.seed),
        device=str(args.device),
    )
    beta_entropy = entropy_rows(beta)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "input": input_metadata,
        "model": model_metadata,
        "metrics": {
            "documents": int(theta.shape[0]),
            "topics": int(theta.shape[1]),
            "vocab_size": int(beta.shape[1]),
            "mean_beta_entropy": float(beta_entropy.mean()),
            "mean_beta_effective_support": float(np.exp(beta_entropy).mean()),
            "mean_beta_max_probability": float(beta.max(axis=1).mean()),
            "membership_count_diagnostics": membership_count_diagnostics(theta),
        },
        "outputs": {
            "theta_npy": str(out_dir / "theta.npy"),
            "beta_npy": str(out_dir / "beta.npy"),
            "vocab_json": str(out_dir / "vocab.json"),
            "train_history_json": str(out_dir / "train_history.json"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
