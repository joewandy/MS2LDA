#!/usr/bin/env python
"""Train a ProdLDA-style neural topic model for the MSn benchmark."""

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
DROPOUT = 0.2
WEIGHT_DECAY = 0.0
KL_WEIGHT = 1.0
KL_ANNEAL_EPOCHS = 50
BETA_INIT_NOISE = 0.1
THETA_ENTROPY_WEIGHT = 0.0
TOPIC_USAGE_WEIGHT = 0.0
BETA_TARGET_SUPPORT = 64.0
BETA_TARGET_WEIGHT = 0.0
BACKGROUND_WEIGHT = 0.0
EXTRA_OUTPUT_FILENAMES = {
    "theta_raw.npy",
    "split_indices.json",
    "validation_metrics.json",
}


@dataclass(frozen=True)
class ProdLDAConfig:
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
    kl_weight: float
    kl_anneal_epochs: int
    theta_entropy_weight: float
    topic_usage_weight: float
    beta_target_support: float
    beta_target_weight: float
    background_weight: float
    beta_init_noise: float
    theta_export_power: float
    membership_threshold: float
    seed: int
    device: str


class ProdLDAModel(nn.Module):
    """AVITM/ProdLDA encoder with a product-of-experts decoder."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_topics: int,
        hidden_size: int,
        dropout: float,
        beta_background: np.ndarray,
        beta_init_noise: float,
        seed: int,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(int(seed))
        background = np.asarray(beta_background, dtype=np.float32)
        if background.shape != (int(vocab_size),):
            raise ValueError("beta_background must match vocab_size.")
        background = background / max(float(background.sum()), EPS)
        beta_logits_init = np.log(np.clip(background, EPS, None))[None, :]
        beta_logits_init = np.repeat(beta_logits_init, int(n_topics), axis=0)
        beta_logits_init += rng.normal(
            loc=0.0,
            scale=float(beta_init_noise),
            size=(int(n_topics), int(vocab_size)),
        ).astype(np.float32)

        self.beta_logits = nn.Parameter(torch.from_numpy(beta_logits_init))
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.Softplus(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_size, n_topics)
        self.logvar = nn.Linear(hidden_size, n_topics)

    def encode(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x_norm)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def theta_from_mu(self, x_norm: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x_norm)
        return torch.softmax(mu, dim=1)

    def sample_theta(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_norm)
        std = torch.exp(0.5 * logvar)
        z = mu + (torch.randn_like(std) * std)
        theta = torch.softmax(z, dim=1)
        return theta, mu, logvar

    def beta(self) -> torch.Tensor:
        return torch.softmax(self.beta_logits, dim=1)

    def word_distribution(
        self,
        theta: torch.Tensor,
        background: torch.Tensor,
        *,
        background_weight: float,
    ) -> torch.Tensor:
        word_probs = torch.softmax(theta @ self.beta_logits, dim=1)
        if float(background_weight) > 0:
            word_probs = (
                (1.0 - float(background_weight)) * word_probs
                + float(background_weight) * background.unsqueeze(0)
            )
            word_probs = word_probs / word_probs.sum(dim=1, keepdim=True).clamp_min(EPS)
        return word_probs.clamp_min(EPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-LDA ProdLDA/AVITM-style neural topic model on an MSn "
            "benchmark input cache."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-motifs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--kl-anneal-epochs", type=int, default=KL_ANNEAL_EPOCHS)
    parser.add_argument(
        "--theta-entropy-weight",
        type=float,
        default=THETA_ENTROPY_WEIGHT,
        help="Optional sparsity penalty on per-document theta entropy.",
    )
    parser.add_argument(
        "--topic-usage-weight",
        type=float,
        default=TOPIC_USAGE_WEIGHT,
        help="Optional penalty for unused topics in the batch mean theta.",
    )
    parser.add_argument(
        "--beta-target-support",
        type=float,
        default=BETA_TARGET_SUPPORT,
    )
    parser.add_argument("--beta-target-weight", type=float, default=BETA_TARGET_WEIGHT)
    parser.add_argument(
        "--background-weight",
        type=float,
        default=BACKGROUND_WEIGHT,
        help="Optional corpus-background mixture weight in decoded word probabilities.",
    )
    parser.add_argument(
        "--beta-init-noise",
        type=float,
        default=BETA_INIT_NOISE,
        help="Noise added to corpus-background beta initialization.",
    )
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


def dense_count_batch(x, indices: np.ndarray) -> np.ndarray:
    return x[indices].toarray().astype(np.float32, copy=False)


def normalize_dense_counts(dense: np.ndarray) -> np.ndarray:
    denom = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, denom + EPS).astype(np.float32, copy=False)


def prodlda_reconstruction_loss(
    x_counts: torch.Tensor,
    word_probs: torch.Tensor,
) -> torch.Tensor:
    return -(x_counts * word_probs.clamp_min(EPS).log()).sum(dim=1).mean()


def logistic_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def theta_entropy(theta: torch.Tensor) -> torch.Tensor:
    return -(theta * theta.clamp_min(EPS).log()).sum(dim=1).mean()


def kl_scale_for_epoch(config: ProdLDAConfig, epoch: int) -> float:
    if int(config.kl_anneal_epochs) <= 0:
        return float(config.kl_weight)
    scale = min(1.0, float(epoch) / float(config.kl_anneal_epochs))
    return float(config.kl_weight) * scale


def build_config(args: argparse.Namespace, out_dir: Path) -> ProdLDAConfig:
    return ProdLDAConfig(
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
        kl_weight=float(args.kl_weight),
        kl_anneal_epochs=int(args.kl_anneal_epochs),
        theta_entropy_weight=float(args.theta_entropy_weight),
        topic_usage_weight=float(args.topic_usage_weight),
        beta_target_support=float(args.beta_target_support),
        beta_target_weight=float(args.beta_target_weight),
        background_weight=float(args.background_weight),
        beta_init_noise=float(args.beta_init_noise),
        theta_export_power=float(args.theta_export_power),
        membership_threshold=float(args.membership_threshold),
        seed=int(args.seed),
        device=str(args.device),
    )


def evaluate_model(
    model: ProdLDAModel,
    x,
    indices: np.ndarray,
    background: torch.Tensor,
    *,
    config: ProdLDAConfig,
) -> dict[str, float]:
    model.eval()
    losses = []
    reconstructions = []
    kls = []
    max_values = []
    entropies = []
    with torch.no_grad():
        for start in range(0, len(indices), int(config.batch_size)):
            batch_indices = indices[start : start + int(config.batch_size)]
            dense = dense_count_batch(x, batch_indices)
            x_counts = torch.from_numpy(dense).to(config.device)
            x_norm = torch.from_numpy(normalize_dense_counts(dense)).to(config.device)
            mu, logvar = model.encode(x_norm)
            theta = torch.softmax(mu, dim=1)
            word_probs = model.word_distribution(
                theta,
                background,
                background_weight=config.background_weight,
            )
            reconstruction = prodlda_reconstruction_loss(x_counts, word_probs)
            kl = logistic_normal_kl(mu, logvar)
            loss = reconstruction + (config.kl_weight * kl)
            losses.append(float(loss.detach().cpu()))
            reconstructions.append(float(reconstruction.detach().cpu()))
            kls.append(float(kl.detach().cpu()))
            max_values.append(theta.max(dim=1).values.detach().cpu().numpy())
            entropies.append(
                (-(theta * theta.clamp_min(EPS).log()).sum(dim=1))
                .detach()
                .cpu()
                .numpy()
            )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "reconstruction": float(np.mean(reconstructions)) if reconstructions else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "mean_max_theta": (
            float(np.concatenate(max_values).mean()) if max_values else 0.0
        ),
        "mean_theta_entropy": (
            float(np.concatenate(entropies).mean()) if entropies else 0.0
        ),
    }


def train_model(
    x,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    background: np.ndarray,
    config: ProdLDAConfig,
) -> tuple[ProdLDAModel, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = ProdLDAModel(
        vocab_size=x.shape[1],
        n_topics=config.n_motifs,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        beta_background=background,
        beta_init_noise=config.beta_init_noise,
        seed=config.seed,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    background_tensor = torch.from_numpy(background).float().to(config.device)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(train_indices)
        current_kl_scale = kl_scale_for_epoch(config, epoch)
        sums = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "kl_scale": 0.0,
            "theta_entropy": 0.0,
            "topic_usage": 0.0,
            "beta_target": 0.0,
            "theta_support": 0.0,
            "beta_effective_support": 0.0,
        }
        batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            dense = dense_count_batch(x, batch_indices)
            x_counts = torch.from_numpy(dense).to(config.device)
            x_norm = torch.from_numpy(normalize_dense_counts(dense)).to(config.device)

            theta, mu, logvar = model.sample_theta(x_norm)
            word_probs = model.word_distribution(
                theta,
                background_tensor,
                background_weight=config.background_weight,
            )
            reconstruction = prodlda_reconstruction_loss(x_counts, word_probs)
            kl = logistic_normal_kl(mu, logvar)
            entropy = theta_entropy(theta)
            usage = topic_usage_loss(theta, mode="entropy")
            beta = model.beta()
            beta_target = beta_target_support_loss(
                beta,
                target_support=config.beta_target_support,
            )
            loss = (
                reconstruction
                + (current_kl_scale * kl)
                + (config.theta_entropy_weight * entropy)
                + (config.topic_usage_weight * usage)
                + (config.beta_target_weight * beta_target)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            beta_entropy = -(beta * beta.clamp_min(EPS).log()).sum(dim=1)
            sums["loss"] += float(loss.detach().cpu())
            sums["reconstruction"] += float(reconstruction.detach().cpu())
            sums["kl"] += float(kl.detach().cpu())
            sums["kl_scale"] += float(current_kl_scale)
            sums["theta_entropy"] += float(entropy.detach().cpu())
            sums["topic_usage"] += float(usage.detach().cpu())
            sums["beta_target"] += float(beta_target.detach().cpu())
            sums["theta_support"] += float(
                (theta >= 0.01).sum(dim=1).float().mean().detach().cpu()
            )
            sums["beta_effective_support"] += float(
                beta_entropy.exp().mean().detach().cpu()
            )
            batches += 1

        train_row = {key: value / max(batches, 1) for key, value in sums.items()}
        validation_row = evaluate_model(
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
        if validation_row["loss"] < best_validation_loss:
            best_validation_loss = validation_row["loss"]
            best_state = copy.deepcopy(model.state_dict())
            row["best_validation_loss"] = best_validation_loss
        history.append(row)
        print(json.dumps(row))

    model.load_state_dict(best_state)
    return model, history


def infer_theta(
    model: ProdLDAModel,
    x,
    *,
    config: ProdLDAConfig,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(config.batch_size)):
            indices = np.arange(start, min(start + int(config.batch_size), x.shape[0]))
            dense = dense_count_batch(x, indices)
            x_norm = torch.from_numpy(normalize_dense_counts(dense)).to(config.device)
            rows.append(model.theta_from_mu(x_norm).detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def model_beta(model: ProdLDAModel) -> np.ndarray:
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
    theta_raw = infer_theta(model, x, config=config)
    theta = sharpen_theta(theta_raw, config.theta_export_power)
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
    np.save(out_dir / "beta.npy", beta)
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    write_json(out_dir / "train_history.json", {"history": history})
    write_json(out_dir / "validation_metrics.json", metrics)
    torch.save(
        {
            "model_type": "msn_prodlda",
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "background": background,
            "train_indices": train_idx,
        },
        out_dir / "model_checkpoint.pt",
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "prodlda",
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
