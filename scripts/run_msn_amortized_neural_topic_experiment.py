#!/usr/bin/env python
"""Train a neural-only amortized topic model for the MSn benchmark."""

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

import torch  # noqa: E402
from torch import nn  # noqa: E402

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    BACKGROUND_WEIGHT,
    EPS,
    MEMBERSHIP_THRESHOLD,
    MODEL_OUTPUT_FILENAMES,
    beta_target_support_loss,
    bow_background_distribution,
    clear_named_outputs,
    entropy_rows,
    load_input_cache,
    membership_count_diagnostics,
    resolve_path,
    sharpen_theta,
    split_indices_json_payload,
    topic_usage_loss,
    train_validation_test_split,
    write_json,
)


THETA_EXPORT_POWER = 3.0
THETA_ENTROPY_WEIGHT = 0.05
TOPIC_USAGE_WEIGHT = 20.0
BETA_TARGET_SUPPORT = 64.0
BETA_TARGET_WEIGHT = 1.0
LOCAL_RECONSTRUCTION_WEIGHT = 1.0
ENCODER_RECONSTRUCTION_WEIGHT = 1.0
CONSISTENCY_WEIGHT = 1.0
ENCODER_TOPIC_USAGE_WEIGHT = 1.0
DROPOUT = 0.1
WEIGHT_DECAY = 0.0
EXTRA_OUTPUT_FILENAMES = {
    "theta_raw.npy",
    "theta_local_train.npy",
    "split_indices.json",
    "validation_metrics.json",
}


@dataclass(frozen=True)
class AmortizedTopicConfig:
    input_cache: str
    out_dir: str
    n_motifs: int
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    dropout: float
    weight_decay: float
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    local_reconstruction_weight: float
    encoder_reconstruction_weight: float
    consistency_weight: float
    theta_entropy_weight: float
    topic_usage_weight: float
    encoder_topic_usage_weight: float
    beta_target_support: float
    beta_target_weight: float
    background_weight: float
    theta_init_strength: float
    theta_export_power: float
    membership_threshold: float
    seed: int
    device: str


class AmortizedNeuralTopicModel(nn.Module):
    def __init__(
        self,
        *,
        n_train_docs: int,
        vocab_size: int,
        n_topics: int,
        hidden_size: int,
        dropout: float,
        theta_init_strength: float,
        seed: int,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(int(seed))
        theta_logits_init = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(int(n_train_docs), int(n_topics)),
        ).astype(np.float32)
        topic_init = rng.integers(0, int(n_topics), size=int(n_train_docs))
        theta_logits_init[np.arange(int(n_train_docs)), topic_init] += float(
            theta_init_strength
        )
        beta_logits_init = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(int(n_topics), int(vocab_size)),
        ).astype(np.float32)

        self.theta_logits = nn.Parameter(torch.from_numpy(theta_logits_init))
        self.beta_logits = nn.Parameter(torch.from_numpy(beta_logits_init))
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_topics),
        )

    def local_theta(self, train_positions: torch.Tensor | None = None) -> torch.Tensor:
        logits = (
            self.theta_logits
            if train_positions is None
            else self.theta_logits[train_positions]
        )
        return torch.softmax(logits, dim=1)

    def encoder_theta(self, x_norm: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.encoder(x_norm), dim=1)

    def beta(self) -> torch.Tensor:
        return torch.softmax(self.beta_logits, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a neural-only MSn topic model with learned beta, local train "
            "theta, and an encoder for held-out spectra."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-motifs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--local-reconstruction-weight",
        type=float,
        default=LOCAL_RECONSTRUCTION_WEIGHT,
    )
    parser.add_argument(
        "--encoder-reconstruction-weight",
        type=float,
        default=ENCODER_RECONSTRUCTION_WEIGHT,
    )
    parser.add_argument("--consistency-weight", type=float, default=CONSISTENCY_WEIGHT)
    parser.add_argument(
        "--theta-entropy-weight",
        type=float,
        default=THETA_ENTROPY_WEIGHT,
    )
    parser.add_argument("--topic-usage-weight", type=float, default=TOPIC_USAGE_WEIGHT)
    parser.add_argument(
        "--encoder-topic-usage-weight",
        type=float,
        default=ENCODER_TOPIC_USAGE_WEIGHT,
    )
    parser.add_argument(
        "--beta-target-support",
        type=float,
        default=BETA_TARGET_SUPPORT,
    )
    parser.add_argument("--beta-target-weight", type=float, default=BETA_TARGET_WEIGHT)
    parser.add_argument("--background-weight", type=float, default=BACKGROUND_WEIGHT)
    parser.add_argument("--theta-init-strength", type=float, default=8.0)
    parser.add_argument("--theta-export-power", type=float, default=THETA_EXPORT_POWER)
    parser.add_argument(
        "--membership-threshold",
        type=float,
        default=MEMBERSHIP_THRESHOLD,
        help="Threshold used only for diagnostics and benchmark export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense_normalized_batch(x, indices: np.ndarray) -> np.ndarray:
    dense = x[indices].toarray().astype(np.float32, copy=False)
    denom = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, denom + EPS).astype(np.float32, copy=False)


def reconstruction_loss(
    x_norm: torch.Tensor,
    theta: torch.Tensor,
    beta: torch.Tensor,
    background: torch.Tensor,
    *,
    background_weight: float,
) -> torch.Tensor:
    dist = (
        (1.0 - float(background_weight)) * (theta @ beta)
        + float(background_weight) * background.unsqueeze(0)
    ).clamp_min(EPS)
    dist = dist / dist.sum(dim=1, keepdim=True).clamp_min(EPS)
    return -(x_norm * dist.log()).sum(dim=1).mean()


def theta_kl_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=1, keepdim=True).clamp_min(EPS)
    return (
        (target * (target.clamp_min(EPS).log() - pred.clamp_min(EPS).log()))
        .sum(dim=1)
        .mean()
    )


def theta_entropy(theta: torch.Tensor) -> torch.Tensor:
    return -(theta * theta.clamp_min(EPS).log()).sum(dim=1).mean()


def build_config(args: argparse.Namespace, out_dir: Path) -> AmortizedTopicConfig:
    return AmortizedTopicConfig(
        input_cache=str(resolve_path(args.input_cache)),
        out_dir=str(out_dir),
        n_motifs=int(args.n_motifs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
        weight_decay=float(args.weight_decay),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        test_fraction=float(args.test_fraction),
        local_reconstruction_weight=float(args.local_reconstruction_weight),
        encoder_reconstruction_weight=float(args.encoder_reconstruction_weight),
        consistency_weight=float(args.consistency_weight),
        theta_entropy_weight=float(args.theta_entropy_weight),
        topic_usage_weight=float(args.topic_usage_weight),
        encoder_topic_usage_weight=float(args.encoder_topic_usage_weight),
        beta_target_support=float(args.beta_target_support),
        beta_target_weight=float(args.beta_target_weight),
        background_weight=float(args.background_weight),
        theta_init_strength=float(args.theta_init_strength),
        theta_export_power=float(args.theta_export_power),
        membership_threshold=float(args.membership_threshold),
        seed=int(args.seed),
        device=str(args.device),
    )


def evaluate_encoder_reconstruction(
    model: AmortizedNeuralTopicModel,
    x,
    indices: np.ndarray,
    background: torch.Tensor,
    *,
    config: AmortizedTopicConfig,
) -> dict[str, float]:
    model.eval()
    values = []
    max_values = []
    entropies = []
    with torch.no_grad():
        beta = model.beta()
        for start in range(0, len(indices), int(config.batch_size)):
            batch_indices = indices[start : start + int(config.batch_size)]
            x_norm = torch.from_numpy(dense_normalized_batch(x, batch_indices)).to(
                config.device
            )
            theta = model.encoder_theta(x_norm)
            loss = reconstruction_loss(
                x_norm,
                theta,
                beta,
                background,
                background_weight=config.background_weight,
            )
            values.append(float(loss.detach().cpu()))
            max_values.append(theta.max(dim=1).values.detach().cpu().numpy())
            entropies.append(
                (-(theta * theta.clamp_min(EPS).log()).sum(dim=1))
                .detach()
                .cpu()
                .numpy()
            )
    return {
        "encoder_reconstruction": float(np.mean(values)) if values else 0.0,
        "encoder_mean_max_theta": (
            float(np.concatenate(max_values).mean()) if max_values else 0.0
        ),
        "encoder_mean_theta_entropy": (
            float(np.concatenate(entropies).mean()) if entropies else 0.0
        ),
    }


def train_model(
    x,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    background: np.ndarray,
    config: AmortizedTopicConfig,
) -> tuple[AmortizedNeuralTopicModel, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = AmortizedNeuralTopicModel(
        n_train_docs=len(train_indices),
        vocab_size=x.shape[1],
        n_topics=config.n_motifs,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        theta_init_strength=config.theta_init_strength,
        seed=config.seed,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    background_tensor = torch.from_numpy(background).float().to(config.device)
    global_to_train_pos = {
        int(doc_index): int(train_pos)
        for train_pos, doc_index in enumerate(train_indices.tolist())
    }
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    usage_sample_size = min(len(train_indices), 4096)

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(train_indices)
        sums = {
            "loss": 0.0,
            "local_reconstruction": 0.0,
            "encoder_reconstruction": 0.0,
            "consistency": 0.0,
            "theta_entropy": 0.0,
            "topic_usage": 0.0,
            "encoder_topic_usage": 0.0,
            "beta_target": 0.0,
            "local_theta_support": 0.0,
            "encoder_theta_support": 0.0,
            "beta_effective_support": 0.0,
        }
        batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            batch_positions = np.asarray(
                [global_to_train_pos[int(index)] for index in batch_indices],
                dtype=np.int64,
            )
            x_norm = torch.from_numpy(dense_normalized_batch(x, batch_indices)).to(
                config.device
            )
            train_positions = torch.from_numpy(batch_positions).long().to(config.device)

            beta = model.beta()
            local_theta = model.local_theta(train_positions)
            encoder_theta = model.encoder_theta(x_norm)
            local_reconstruction = reconstruction_loss(
                x_norm,
                local_theta,
                beta,
                background_tensor,
                background_weight=config.background_weight,
            )
            encoder_reconstruction = reconstruction_loss(
                x_norm,
                encoder_theta,
                beta,
                background_tensor,
                background_weight=config.background_weight,
            )
            consistency = theta_kl_loss(local_theta.detach(), encoder_theta)
            sparsity = 0.5 * (theta_entropy(local_theta) + theta_entropy(encoder_theta))
            usage_positions_np = rng.choice(
                len(train_indices),
                size=usage_sample_size,
                replace=False,
            )
            usage_positions = (
                torch.from_numpy(usage_positions_np).long().to(config.device)
            )
            usage = topic_usage_loss(
                model.local_theta(usage_positions),
                mode="entropy",
            )
            encoder_usage = topic_usage_loss(encoder_theta, mode="entropy")
            beta_target = beta_target_support_loss(
                beta,
                target_support=config.beta_target_support,
            )
            loss = (
                (config.local_reconstruction_weight * local_reconstruction)
                + (config.encoder_reconstruction_weight * encoder_reconstruction)
                + (config.consistency_weight * consistency)
                + (config.theta_entropy_weight * sparsity)
                + (config.topic_usage_weight * usage)
                + (config.encoder_topic_usage_weight * encoder_usage)
                + (config.beta_target_weight * beta_target)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            beta_entropy = -(beta * beta.clamp_min(EPS).log()).sum(dim=1)
            sums["loss"] += float(loss.detach().cpu())
            sums["local_reconstruction"] += float(local_reconstruction.detach().cpu())
            sums["encoder_reconstruction"] += float(
                encoder_reconstruction.detach().cpu()
            )
            sums["consistency"] += float(consistency.detach().cpu())
            sums["theta_entropy"] += float(sparsity.detach().cpu())
            sums["topic_usage"] += float(usage.detach().cpu())
            sums["encoder_topic_usage"] += float(encoder_usage.detach().cpu())
            sums["beta_target"] += float(beta_target.detach().cpu())
            sums["local_theta_support"] += float(
                (local_theta >= 0.01).sum(dim=1).float().mean().detach().cpu()
            )
            sums["encoder_theta_support"] += float(
                (encoder_theta >= 0.01).sum(dim=1).float().mean().detach().cpu()
            )
            sums["beta_effective_support"] += float(
                beta_entropy.exp().mean().detach().cpu()
            )
            batches += 1

        train_row = {key: value / max(batches, 1) for key, value in sums.items()}
        validation_row = evaluate_encoder_reconstruction(
            model,
            x,
            validation_indices,
            background_tensor,
            config=config,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_row.items()},
            **{f"validation_{key}": value for key, value in validation_row.items()},
        }
        if validation_row["encoder_reconstruction"] < best_validation_loss:
            best_validation_loss = validation_row["encoder_reconstruction"]
            best_state = copy.deepcopy(model.state_dict())
            row["best_validation_encoder_reconstruction"] = best_validation_loss
        history.append(row)
        print(json.dumps(row))

    model.load_state_dict(best_state)
    return model, history


def infer_encoder_theta(
    model: AmortizedNeuralTopicModel,
    x,
    *,
    config: AmortizedTopicConfig,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(config.batch_size)):
            indices = np.arange(start, min(start + int(config.batch_size), x.shape[0]))
            x_norm = torch.from_numpy(dense_normalized_batch(x, indices)).to(
                config.device
            )
            rows.append(model.encoder_theta(x_norm).detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def local_train_theta(
    model: AmortizedNeuralTopicModel,
    *,
    batch_size: int,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        n_docs = model.theta_logits.shape[0]
        for start in range(0, n_docs, int(batch_size)):
            positions = torch.arange(
                start,
                min(start + int(batch_size), n_docs),
                device=next(model.parameters()).device,
            )
            rows.append(model.local_theta(positions).detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def model_beta(model: AmortizedNeuralTopicModel) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.beta().detach().cpu().numpy().astype(np.float32)


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def theta_summary(
    theta: np.ndarray,
    indices: np.ndarray,
    *,
    membership_threshold: float,
) -> dict[str, float | int]:
    subset = theta[indices]
    counts = (subset >= float(membership_threshold)).sum(axis=0)
    entropy = entropy_rows(subset)
    return {
        "membership_rows_above_threshold": int(
            np.sum(subset >= float(membership_threshold))
        ),
        "active_topics_above_threshold": int(np.sum(counts > 0)),
        "mean_max_theta": float(np.max(subset, axis=1).mean()),
        "mean_theta_entropy": float(entropy.mean()),
    }


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
    torch.manual_seed(args.seed)

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
    background = bow_background_distribution(x[train_idx])

    model, history = train_model(
        x,
        train_indices=train_idx,
        validation_indices=validation_idx,
        background=background,
        config=config,
    )
    theta_raw = infer_encoder_theta(model, x, config=config)
    theta = sharpen_theta(theta_raw, config.theta_export_power)
    theta_local_train = local_train_theta(model, batch_size=config.batch_size)
    beta = model_beta(model)
    all_idx = np.arange(x.shape[0], dtype=np.int64)
    beta_entropy = entropy_rows(beta)
    metrics = {
        **prefixed(
            "train",
            theta_summary(
                theta,
                train_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "validation",
            theta_summary(
                theta,
                validation_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "test",
            theta_summary(
                theta,
                test_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "all",
            theta_summary(
                theta,
                all_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        "mean_beta_entropy": float(beta_entropy.mean()),
        "mean_beta_effective_support": float(np.exp(beta_entropy).mean()),
        "mean_beta_max_probability": float(beta.max(axis=1).mean()),
        "membership_count_diagnostics": {
            "train": membership_count_diagnostics(theta[train_idx]),
            "validation": membership_count_diagnostics(theta[validation_idx]),
            "test": membership_count_diagnostics(theta[test_idx]),
            "all": membership_count_diagnostics(theta),
        },
    }

    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "theta_raw.npy", theta_raw)
    np.save(out_dir / "theta_local_train.npy", theta_local_train)
    np.save(out_dir / "beta.npy", beta)
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    write_json(out_dir / "train_history.json", {"history": history})
    write_json(out_dir / "validation_metrics.json", metrics)
    torch.save(
        {
            "model_type": "amortized_neural_topic",
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "background": background,
            "train_indices": train_idx,
        },
        out_dir / "model_checkpoint.pt",
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "amortized-neural-topic",
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
            "theta_local_train_npy": str(out_dir / "theta_local_train.npy"),
            "beta_npy": str(out_dir / "beta.npy"),
            "vocab_json": str(out_dir / "vocab.json"),
            "train_history_json": str(out_dir / "train_history.json"),
            "split_indices_json": str(out_dir / "split_indices.json"),
            "model_checkpoint": str(out_dir / "model_checkpoint.pt"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
