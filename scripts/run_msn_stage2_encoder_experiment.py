#!/usr/bin/env python
"""Train a Stage 2 encoder to imitate a Stage 1 MSn theta matrix."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    EPS,
    MEMBERSHIP_THRESHOLD,
    MODEL_OUTPUT_FILENAMES,
    bow_background_distribution,
    clear_named_outputs,
    load_input_cache,
    normalize_rows,
    resolve_path,
    sharpen_theta,
    write_json,
)


TOP1_LOSS_WEIGHT = 0.25
RECONSTRUCTION_LOSS_WEIGHT = 0.1
BACKGROUND_WEIGHT = 0.05
THETA_EXPORT_POWER = 1.0
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class Stage2Config:
    input_cache: str
    teacher_model_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    encoder: str
    token_embedding_size: int
    max_tokens: int
    train_fraction: float
    top1_loss_weight: float
    reconstruction_loss_weight: float
    background_weight: float
    dropout: float
    weight_decay: float
    theta_export_power: float
    membership_threshold: float
    seed: int
    device: str


class BowThetaEncoder(nn.Module):
    """Small MLP mapping normalized BoW spectra to motif mixtures."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_topics: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x_norm), dim=1)


class TokenSetThetaEncoder(nn.Module):
    """Set encoder over nonzero fragment/loss feature tokens."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_topics: int,
        token_embedding_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size + 1,
            token_embedding_size,
            padding_idx=0,
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(4, token_embedding_size),
            nn.ReLU(),
            nn.LayerNorm(token_embedding_size),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(token_embedding_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.theta_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_topics),
        )

    def forward(
        self,
        *,
        token_ids: torch.Tensor,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_repr = self.token_embedding(token_ids) + self.feature_projection(
            token_features
        )
        token_repr = self.token_projection(token_repr)
        scores = self.attention(token_repr).squeeze(-1)
        scores = scores.masked_fill(~token_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = (token_repr * weights.unsqueeze(-1)).sum(dim=1)
        return torch.softmax(self.theta_head(pooled), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Stage 2 BOW encoder that predicts theta from a Stage 1 "
            "MSn model directory containing theta.npy and beta.npy."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--teacher-model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--encoder", choices=["bow", "token-set"], default="token-set")
    parser.add_argument("--token-embedding-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--theta-export-power", type=float, default=THETA_EXPORT_POWER)
    parser.add_argument(
        "--membership-threshold",
        type=float,
        default=MEMBERSHIP_THRESHOLD,
        help="Threshold used only for summary diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_teacher_outputs(
    teacher_dir: Path, cache_vocab: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    theta_path = teacher_dir / "theta.npy"
    beta_path = teacher_dir / "beta.npy"
    if not theta_path.exists() or not beta_path.exists():
        raise FileNotFoundError(f"Expected theta.npy and beta.npy in {teacher_dir}")

    theta = normalize_rows(np.load(theta_path))
    beta = normalize_rows(np.load(beta_path))

    vocab_path = teacher_dir / "vocab.json"
    if vocab_path.exists():
        vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocab = [str(token) for token in vocab_payload["vocab"]]
    else:
        vocab = list(cache_vocab)
    return theta, beta, vocab


def validate_inputs(
    x: sparse.csr_matrix, teacher_theta: np.ndarray, beta: np.ndarray, vocab: list[str]
) -> None:
    if x.shape[0] != teacher_theta.shape[0]:
        raise ValueError(
            f"BoW rows ({x.shape[0]}) do not match teacher theta rows ({teacher_theta.shape[0]})."
        )
    if teacher_theta.shape[1] != beta.shape[0]:
        raise ValueError(
            f"Teacher theta topics ({teacher_theta.shape[1]}) do not match beta rows ({beta.shape[0]})."
        )
    if x.shape[1] != beta.shape[1]:
        raise ValueError(
            f"BoW columns ({x.shape[1]}) do not match beta columns ({beta.shape[1]})."
        )
    if len(vocab) != beta.shape[1]:
        raise ValueError(
            f"Vocabulary size ({len(vocab)}) does not match beta columns ({beta.shape[1]})."
        )


def train_validation_split(
    n_docs: int, *, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_docs < 2:
        raise ValueError("Need at least two spectra for a train/validation split.")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_docs)
    n_train = int(round(n_docs * float(train_fraction)))
    n_train = min(max(n_train, 1), n_docs - 1)
    train_idx = np.sort(order[:n_train])
    val_idx = np.sort(order[n_train:])
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def dense_normalized_batch(x: sparse.csr_matrix, indices: np.ndarray) -> np.ndarray:
    dense = x[indices].toarray().astype(np.float32, copy=False)
    denom = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, denom + EPS).astype(np.float32, copy=False)


def vocab_token_features(vocab: list[str]) -> np.ndarray:
    features = np.zeros((len(vocab), 2), dtype=np.float32)
    for index, token in enumerate(vocab):
        prefix, _, value = token.partition("@")
        try:
            mz = float(value)
        except ValueError:
            mz = 0.0
        features[index, 0] = mz / 1000.0
        features[index, 1] = 1.0 if prefix == "loss" else 0.0
    return features


def token_batch_from_csr(
    x: sparse.csr_matrix,
    indices: np.ndarray,
    *,
    max_tokens: int,
    vocab_features: np.ndarray,
) -> dict[str, np.ndarray]:
    token_ids = np.zeros((len(indices), int(max_tokens)), dtype=np.int64)
    token_features = np.zeros((len(indices), int(max_tokens), 4), dtype=np.float32)
    token_mask = np.zeros((len(indices), int(max_tokens)), dtype=bool)

    for batch_row, doc_index in enumerate(indices):
        start = x.indptr[int(doc_index)]
        end = x.indptr[int(doc_index) + 1]
        cols = x.indices[start:end]
        counts = x.data[start:end].astype(np.float32, copy=False)
        if len(cols) == 0:
            continue
        if len(cols) > int(max_tokens):
            keep = np.argpartition(-counts, int(max_tokens) - 1)[: int(max_tokens)]
            keep = keep[np.argsort(-counts[keep])]
            cols = cols[keep]
            counts = counts[keep]
        length = len(cols)
        total = float(counts.sum()) + EPS
        token_ids[batch_row, :length] = cols.astype(np.int64) + 1
        token_features[batch_row, :length, 0] = counts / total
        token_features[batch_row, :length, 1] = np.log1p(counts)
        token_features[batch_row, :length, 2:] = vocab_features[cols]
        token_mask[batch_row, :length] = True
    return {
        "token_ids": token_ids,
        "token_features": token_features,
        "token_mask": token_mask,
    }


def token_batch_to_torch(
    batch: dict[str, np.ndarray],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.from_numpy(batch["token_ids"]).long().to(device),
        "token_features": torch.from_numpy(batch["token_features"]).float().to(device),
        "token_mask": torch.from_numpy(batch["token_mask"]).bool().to(device),
    }


def theta_kl_loss(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(EPS)
    return (
        (teacher * (torch.log(teacher.clamp_min(EPS)) - torch.log(pred.clamp_min(EPS))))
        .sum(dim=1)
        .mean()
    )


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
    return -(x_norm * torch.log(dist)).sum(dim=1).mean()


def build_encoder(config: Stage2Config, *, vocab_size: int, n_topics: int) -> nn.Module:
    if config.encoder == "bow":
        return BowThetaEncoder(
            vocab_size=vocab_size,
            n_topics=n_topics,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )
    if config.encoder == "token-set":
        return TokenSetThetaEncoder(
            vocab_size=vocab_size,
            n_topics=n_topics,
            token_embedding_size=config.token_embedding_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported encoder: {config.encoder}")


def predict_batch(
    model: nn.Module,
    x: sparse.csr_matrix,
    indices: np.ndarray,
    *,
    config: Stage2Config,
    vocab_features: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm_np = dense_normalized_batch(x, indices)
    x_norm = torch.from_numpy(x_norm_np).to(config.device)
    if config.encoder == "bow":
        pred = model(x_norm)
    else:
        token_batch = token_batch_from_csr(
            x,
            indices,
            max_tokens=config.max_tokens,
            vocab_features=vocab_features,
        )
        pred = model(**token_batch_to_torch(token_batch, device=config.device))
    return pred, x_norm


def train_encoder(
    x: sparse.csr_matrix,
    teacher_theta: np.ndarray,
    beta: np.ndarray,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    vocab_features: np.ndarray,
    config: Stage2Config,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = build_encoder(
        config,
        vocab_size=x.shape[1],
        n_topics=teacher_theta.shape[1],
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    teacher_tensor = torch.from_numpy(teacher_theta).float().to(config.device)
    top1_targets = torch.argmax(teacher_tensor, dim=1)
    beta_tensor = torch.from_numpy(beta).float().to(config.device)
    background = (
        torch.from_numpy(bow_background_distribution(x)).float().to(config.device)
    )
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(train_indices)
        sums = {
            "loss": 0.0,
            "theta_kl": 0.0,
            "top1": 0.0,
            "reconstruction": 0.0,
        }
        batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            pred, x_norm = predict_batch(
                model,
                x,
                batch_indices,
                config=config,
                vocab_features=vocab_features,
            )
            batch_tensor = torch.from_numpy(batch_indices).long().to(config.device)
            kl = theta_kl_loss(pred, teacher_tensor[batch_tensor])
            top1 = F.nll_loss(
                torch.log(pred.clamp_min(EPS)),
                top1_targets[batch_tensor],
            )
            reconstruction = reconstruction_loss(
                x_norm,
                pred,
                beta_tensor,
                background,
                background_weight=config.background_weight,
            )
            loss = (
                kl
                + (float(config.top1_loss_weight) * top1)
                + (float(config.reconstruction_loss_weight) * reconstruction)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sums["loss"] += float(loss.detach().cpu())
            sums["theta_kl"] += float(kl.detach().cpu())
            sums["top1"] += float(top1.detach().cpu())
            sums["reconstruction"] += float(reconstruction.detach().cpu())
            batches += 1
        train_row = {key: value / max(batches, 1) for key, value in sums.items()}
        validation_row = evaluate_objective(
            model,
            x,
            teacher_tensor,
            top1_targets,
            beta_tensor,
            background,
            validation_indices,
            vocab_features,
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


def evaluate_objective(
    model: nn.Module,
    x: sparse.csr_matrix,
    teacher_tensor: torch.Tensor,
    top1_targets: torch.Tensor,
    beta_tensor: torch.Tensor,
    background: torch.Tensor,
    indices: np.ndarray,
    vocab_features: np.ndarray,
    *,
    config: Stage2Config,
) -> dict[str, float]:
    model.eval()
    sums = {
        "loss": 0.0,
        "theta_kl": 0.0,
        "top1": 0.0,
        "reconstruction": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for start in range(0, len(indices), int(config.batch_size)):
            batch_indices = indices[start : start + int(config.batch_size)]
            pred, x_norm = predict_batch(
                model,
                x,
                batch_indices,
                config=config,
                vocab_features=vocab_features,
            )
            batch_tensor = torch.from_numpy(batch_indices).long().to(config.device)
            kl = theta_kl_loss(pred, teacher_tensor[batch_tensor])
            top1 = F.nll_loss(
                torch.log(pred.clamp_min(EPS)),
                top1_targets[batch_tensor],
            )
            reconstruction = reconstruction_loss(
                x_norm,
                pred,
                beta_tensor,
                background,
                background_weight=config.background_weight,
            )
            loss = (
                kl
                + (float(config.top1_loss_weight) * top1)
                + (float(config.reconstruction_loss_weight) * reconstruction)
            )
            sums["loss"] += float(loss.detach().cpu())
            sums["theta_kl"] += float(kl.detach().cpu())
            sums["top1"] += float(top1.detach().cpu())
            sums["reconstruction"] += float(reconstruction.detach().cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


def infer_theta(
    model: nn.Module,
    x: sparse.csr_matrix,
    *,
    config: Stage2Config,
    vocab_features: np.ndarray,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(config.batch_size)):
            indices = np.arange(start, min(start + int(config.batch_size), x.shape[0]))
            pred, _x_norm = predict_batch(
                model,
                x,
                indices,
                config=config,
                vocab_features=vocab_features,
            )
            rows.append(pred.detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def numpy_reconstruction_nll(
    x: sparse.csr_matrix,
    theta: np.ndarray,
    beta: np.ndarray,
    background: np.ndarray,
    indices: np.ndarray,
    *,
    background_weight: float,
    batch_size: int = 256,
) -> float:
    values = []
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        x_norm = dense_normalized_batch(x, batch_indices)
        dist = (1.0 - float(background_weight)) * (theta[batch_indices] @ beta) + float(
            background_weight
        ) * background[None, :]
        dist = np.maximum(dist, EPS)
        dist = dist / (dist.sum(axis=1, keepdims=True) + EPS)
        values.append(-(x_norm * np.log(dist)).sum(axis=1))
    return float(np.concatenate(values).mean()) if values else 0.0


def theta_metrics(
    teacher_theta: np.ndarray,
    pred_theta: np.ndarray,
    indices: np.ndarray,
    *,
    membership_threshold: float,
) -> dict[str, float | int]:
    teacher = teacher_theta[indices]
    pred = pred_theta[indices]
    teacher_norm = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + EPS)
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + EPS)
    cosine = np.sum(teacher_norm * pred_norm, axis=1)
    pred_counts = (pred >= float(membership_threshold)).sum(axis=0)
    teacher_counts = (teacher >= float(membership_threshold)).sum(axis=0)
    return {
        "theta_cosine_mean": float(np.mean(cosine)),
        "theta_cosine_median": float(np.median(cosine)),
        "top1_agreement": float(
            np.mean(np.argmax(teacher, axis=1) == np.argmax(pred, axis=1))
        ),
        "pred_membership_rows_above_threshold": int(
            np.sum(pred >= float(membership_threshold))
        ),
        "teacher_membership_rows_above_threshold": int(
            np.sum(teacher >= float(membership_threshold))
        ),
        "pred_active_topics_above_threshold": int(np.sum(pred_counts > 0)),
        "teacher_active_topics_above_threshold": int(np.sum(teacher_counts > 0)),
    }


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    input_cache = resolve_path(args.input_cache)
    teacher_model_dir = resolve_path(args.teacher_model_dir)
    out_dir = resolve_path(args.out_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_named_outputs(out_dir, MODEL_OUTPUT_FILENAMES)
        for name in ["theta_raw.npy", "theta_teacher.npy", "split_indices.json"]:
            path = out_dir / name
            if path.exists():
                path.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = load_input_cache(input_cache)
    x = cache["matrix"].tocsr()
    cache_vocab = cache["vocab"]
    teacher_theta, beta, vocab = load_teacher_outputs(teacher_model_dir, cache_vocab)
    validate_inputs(x, teacher_theta, beta, vocab)
    train_idx, val_idx = train_validation_split(
        x.shape[0],
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    config = Stage2Config(
        input_cache=str(input_cache),
        teacher_model_dir=str(teacher_model_dir),
        out_dir=str(out_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_size=int(args.hidden_size),
        encoder=str(args.encoder),
        token_embedding_size=int(args.token_embedding_size),
        max_tokens=int(args.max_tokens),
        train_fraction=float(args.train_fraction),
        top1_loss_weight=TOP1_LOSS_WEIGHT,
        reconstruction_loss_weight=RECONSTRUCTION_LOSS_WEIGHT,
        background_weight=BACKGROUND_WEIGHT,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        theta_export_power=float(args.theta_export_power),
        membership_threshold=float(args.membership_threshold),
        seed=int(args.seed),
        device=str(args.device),
    )
    vocab_features = vocab_token_features(vocab)

    model, history = train_encoder(
        x,
        teacher_theta,
        beta,
        train_indices=train_idx,
        validation_indices=val_idx,
        vocab_features=vocab_features,
        config=config,
    )
    theta_raw = infer_theta(
        model,
        x,
        config=config,
        vocab_features=vocab_features,
    )
    theta = sharpen_theta(theta_raw, args.theta_export_power)
    background = bow_background_distribution(x)
    all_idx = np.arange(x.shape[0], dtype=np.int64)
    metrics = {
        **prefixed(
            "train",
            theta_metrics(
                teacher_theta,
                theta,
                train_idx,
                membership_threshold=args.membership_threshold,
            ),
        ),
        **prefixed(
            "validation",
            theta_metrics(
                teacher_theta,
                theta,
                val_idx,
                membership_threshold=args.membership_threshold,
            ),
        ),
        **prefixed(
            "all",
            theta_metrics(
                teacher_theta,
                theta,
                all_idx,
                membership_threshold=args.membership_threshold,
            ),
        ),
        "teacher_reconstruction_nll": numpy_reconstruction_nll(
            x,
            teacher_theta,
            beta,
            background,
            all_idx,
            background_weight=BACKGROUND_WEIGHT,
            batch_size=args.batch_size,
        ),
        "pred_reconstruction_nll": numpy_reconstruction_nll(
            x,
            theta,
            beta,
            background,
            all_idx,
            background_weight=BACKGROUND_WEIGHT,
            batch_size=args.batch_size,
        ),
    }

    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "theta_raw.npy", theta_raw)
    np.save(out_dir / "theta_teacher.npy", teacher_theta)
    np.save(out_dir / "beta.npy", beta.astype(np.float32, copy=False))
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    write_json(out_dir / "train_history.json", {"history": history})
    write_json(
        out_dir / "split_indices.json",
        {
            "train_indices": [int(value) for value in train_idx],
            "validation_indices": [int(value) for value in val_idx],
        },
    )
    torch.save(
        {
            "model_type": f"msn_stage2_{args.encoder}_theta_encoder",
            "config": asdict(config),
            "state_dict": model.state_dict(),
        },
        out_dir / "model_checkpoint.pt",
    )
    teacher_summary_path = teacher_model_dir / "run_summary.json"
    if teacher_summary_path.exists():
        shutil.copyfile(teacher_summary_path, out_dir / "teacher_run_summary.json")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"stage2-{args.encoder}-theta-encoder",
        "input": {
            "documents": int(x.shape[0]),
            "vocab_size": int(x.shape[1]),
            "topics": int(teacher_theta.shape[1]),
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(val_idx)),
        },
        "config": asdict(config),
        "metrics": metrics,
        "outputs": {
            "theta_npy": str(out_dir / "theta.npy"),
            "theta_raw_npy": str(out_dir / "theta_raw.npy"),
            "theta_teacher_npy": str(out_dir / "theta_teacher.npy"),
            "beta_npy": str(out_dir / "beta.npy"),
            "vocab_json": str(out_dir / "vocab.json"),
            "model_checkpoint": str(out_dir / "model_checkpoint.pt"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
