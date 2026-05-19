#!/usr/bin/env python
"""Train a neural encoder against tomotopy LDA held-out inference with fixed beta."""

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
    BACKGROUND_WEIGHT,
    EPS,
    MEMBERSHIP_THRESHOLD,
    MODEL_OUTPUT_FILENAMES,
    build_bow_matrix_for_vocabulary,
    bow_background_distribution,
    clear_named_outputs,
    entropy_rows,
    infer_tomotopy_lda_theta,
    load_input_cache,
    membership_count_diagnostics,
    resolve_path,
    sharpen_theta,
    split_indices_json_payload,
    tomotopy_lda_model_outputs,
    train_tomotopy_lda_model,
    train_validation_test_split,
    write_json,
)


TOP1_LOSS_WEIGHT = 0.25
RECONSTRUCTION_LOSS_WEIGHT = 0.1
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
THETA_EXPORT_POWER = 1.0
EXPERIMENT_OUTPUT_FILENAMES = {"split_indices.json", "run_summary.json"}
EXTRA_MODEL_FILENAMES = {
    "theta_raw.npy",
    "theta_teacher.npy",
    "validation_metrics.json",
}


@dataclass(frozen=True)
class FixedBetaConfig:
    input_cache: str
    out_dir: str
    n_motifs: int
    lda_iterations: int
    heldout_inference_iterations: int
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    dropout: float
    weight_decay: float
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    min_df: int
    min_cf: float
    rm_top: int
    top1_loss_weight: float
    reconstruction_loss_weight: float
    background_weight: float
    theta_export_power: float
    membership_threshold: float
    seed: int
    device: str


class FixedBetaThetaEncoder(nn.Module):
    """Dense MLP mapping normalized BoW spectra to motif mixtures."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LDA beta on train spectra, infer held-out tomotopy theta, then "
            "train a neural encoder with beta frozen."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-motifs", type=int, default=1000)
    parser.add_argument("--lda-iterations", type=int, default=500)
    parser.add_argument("--heldout-inference-iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--min-df", type=int)
    parser.add_argument("--min-cf", type=float)
    parser.add_argument("--rm-top", type=int)
    parser.add_argument("--top1-loss-weight", type=float, default=TOP1_LOSS_WEIGHT)
    parser.add_argument(
        "--reconstruction-loss-weight",
        type=float,
        default=RECONSTRUCTION_LOSS_WEIGHT,
    )
    parser.add_argument("--background-weight", type=float, default=BACKGROUND_WEIGHT)
    parser.add_argument("--theta-export-power", type=float, default=THETA_EXPORT_POWER)
    parser.add_argument(
        "--membership-threshold", type=float, default=MEMBERSHIP_THRESHOLD
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def documents_at_indices(
    documents: list[list[str]],
    indices: np.ndarray,
) -> list[list[str]]:
    return [documents[int(index)] for index in indices]


def vocabulary_parameters(
    args: argparse.Namespace, cache_summary: dict
) -> dict[str, Any]:
    cache_params = cache_summary.get("vocabulary_parameters", {})
    return {
        "min_df": int(
            args.min_df if args.min_df is not None else cache_params["min_df"]
        ),
        "min_cf": float(
            args.min_cf if args.min_cf is not None else cache_params.get("min_cf", 0.0)
        ),
        "rm_top": int(
            args.rm_top if args.rm_top is not None else cache_params["rm_top"]
        ),
    }


def dense_normalized_batch(x: sparse.csr_matrix, indices: np.ndarray) -> np.ndarray:
    dense = x[indices].toarray().astype(np.float32, copy=False)
    denom = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, denom + EPS).astype(np.float32, copy=False)


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


def evaluate_encoder_objective(
    model: nn.Module,
    x: sparse.csr_matrix,
    teacher_tensor: torch.Tensor,
    top1_targets: torch.Tensor,
    beta_tensor: torch.Tensor,
    background: torch.Tensor,
    indices: np.ndarray,
    *,
    config: FixedBetaConfig,
) -> dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "theta_kl": 0.0, "top1": 0.0, "reconstruction": 0.0}
    batches = 0
    with torch.no_grad():
        for start in range(0, len(indices), int(config.batch_size)):
            batch_indices = indices[start : start + int(config.batch_size)]
            x_norm_np = dense_normalized_batch(x, batch_indices)
            x_norm = torch.from_numpy(x_norm_np).to(config.device)
            pred = model(x_norm)
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


def train_encoder(
    x: sparse.csr_matrix,
    teacher_theta: np.ndarray,
    beta: np.ndarray,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    background: np.ndarray,
    config: FixedBetaConfig,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = FixedBetaThetaEncoder(
        vocab_size=x.shape[1],
        n_topics=teacher_theta.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    teacher_tensor = torch.from_numpy(teacher_theta).float().to(config.device)
    top1_targets = torch.argmax(teacher_tensor, dim=1)
    beta_tensor = torch.from_numpy(beta).float().to(config.device)
    background_tensor = torch.from_numpy(background).float().to(config.device)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(train_indices)
        sums = {"loss": 0.0, "theta_kl": 0.0, "top1": 0.0, "reconstruction": 0.0}
        batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            x_norm_np = dense_normalized_batch(x, batch_indices)
            x_norm = torch.from_numpy(x_norm_np).to(config.device)
            pred = model(x_norm)
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
                background_tensor,
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
        validation_row = evaluate_encoder_objective(
            model,
            x,
            teacher_tensor,
            top1_targets,
            beta_tensor,
            background_tensor,
            validation_indices,
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
    model: nn.Module,
    x: sparse.csr_matrix,
    *,
    config: FixedBetaConfig,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(config.batch_size)):
            indices = np.arange(start, min(start + int(config.batch_size), x.shape[0]))
            x_norm = torch.from_numpy(dense_normalized_batch(x, indices)).to(
                config.device
            )
            rows.append(model(x_norm).detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def numpy_reconstruction_nll(
    x: sparse.csr_matrix,
    theta: np.ndarray,
    beta: np.ndarray,
    background: np.ndarray,
    indices: np.ndarray,
    *,
    background_weight: float,
    batch_size: int,
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
    pred_entropy = entropy_rows(pred)
    teacher_entropy = entropy_rows(teacher)
    return {
        "theta_cosine_mean": float(np.mean(cosine)),
        "theta_cosine_median": float(np.median(cosine)),
        "top1_agreement": float(
            np.mean(np.argmax(teacher, axis=1) == np.argmax(pred, axis=1))
        ),
        "teacher_membership_rows_above_threshold": int(
            np.sum(teacher >= float(membership_threshold))
        ),
        "pred_membership_rows_above_threshold": int(
            np.sum(pred >= float(membership_threshold))
        ),
        "teacher_active_topics_above_threshold": int(np.sum(teacher_counts > 0)),
        "pred_active_topics_above_threshold": int(np.sum(pred_counts > 0)),
        "teacher_mean_max_theta": float(np.max(teacher, axis=1).mean()),
        "pred_mean_max_theta": float(np.max(pred, axis=1).mean()),
        "teacher_mean_theta_entropy": float(teacher_entropy.mean()),
        "pred_mean_theta_entropy": float(pred_entropy.mean()),
    }


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def split_diagnostics(
    theta: np.ndarray,
    splits: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, int | float]]]:
    return {
        split_name.replace("_indices", ""): membership_count_diagnostics(theta[indices])
        for split_name, indices in splits.items()
    }


def clear_model_outputs(model_dir: Path) -> None:
    clear_named_outputs(model_dir, MODEL_OUTPUT_FILENAMES | EXTRA_MODEL_FILENAMES)


def write_standard_model_outputs(
    model_dir: Path,
    *,
    theta: np.ndarray,
    beta: np.ndarray,
    vocab: list[str],
    history: list[dict[str, float]],
    summary: dict[str, Any],
    checkpoint: dict[str, Any] | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "theta.npy", theta.astype(np.float32, copy=False))
    np.save(model_dir / "beta.npy", beta.astype(np.float32, copy=False))
    write_json(model_dir / "vocab.json", {"vocab": vocab})
    write_json(model_dir / "train_history.json", {"history": history})
    write_json(model_dir / "run_summary.json", summary)
    if checkpoint is not None:
        torch.save(checkpoint, model_dir / "model_checkpoint.pt")


def build_config(
    args: argparse.Namespace, out_dir: Path, vocab_params: dict
) -> FixedBetaConfig:
    return FixedBetaConfig(
        input_cache=str(resolve_path(args.input_cache)),
        out_dir=str(out_dir),
        n_motifs=int(args.n_motifs),
        lda_iterations=int(args.lda_iterations),
        heldout_inference_iterations=int(args.heldout_inference_iterations),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
        weight_decay=float(args.weight_decay),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        test_fraction=float(args.test_fraction),
        min_df=int(vocab_params["min_df"]),
        min_cf=float(vocab_params["min_cf"]),
        rm_top=int(vocab_params["rm_top"]),
        top1_loss_weight=float(args.top1_loss_weight),
        reconstruction_loss_weight=float(args.reconstruction_loss_weight),
        background_weight=float(args.background_weight),
        theta_export_power=float(args.theta_export_power),
        membership_threshold=float(args.membership_threshold),
        seed=int(args.seed),
        device=str(args.device),
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    input_cache = resolve_path(args.input_cache)
    out_dir = resolve_path(args.out_dir)
    lda_dir = out_dir / "lda_inferred"
    neural_dir = out_dir / "neural_encoder"

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_named_outputs(out_dir, EXPERIMENT_OUTPUT_FILENAMES)
        clear_model_outputs(lda_dir)
        clear_model_outputs(neural_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = load_input_cache(input_cache, require_documents=True)
    documents = cache["documents"]
    splits = train_validation_test_split(
        len(documents),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_json(out_dir / "split_indices.json", split_indices_json_payload(splits))

    vocab_params = vocabulary_parameters(args, cache["summary"])
    config = build_config(args, out_dir, vocab_params)

    train_idx = splits["train_indices"]
    validation_idx = splits["validation_indices"]
    test_idx = splits["test_indices"]
    train_docs = documents_at_indices(documents, train_idx)
    validation_docs = documents_at_indices(documents, validation_idx)
    test_docs = documents_at_indices(documents, test_idx)

    lda_model, lda_history, lda_metadata = train_tomotopy_lda_model(
        train_docs,
        n_motifs=config.n_motifs,
        min_df=config.min_df,
        min_cf=config.min_cf,
        rm_top=config.rm_top,
        lda_iterations=config.lda_iterations,
        seed=config.seed,
    )
    lda_train_theta, beta, vocab = tomotopy_lda_model_outputs(lda_model)
    validation_theta, validation_inference_metadata = infer_tomotopy_lda_theta(
        lda_model,
        validation_docs,
        iterations=config.heldout_inference_iterations,
    )
    test_theta, test_inference_metadata = infer_tomotopy_lda_theta(
        lda_model,
        test_docs,
        iterations=config.heldout_inference_iterations,
    )
    lda_theta = np.zeros((len(documents), beta.shape[0]), dtype=np.float32)
    lda_theta[train_idx] = lda_train_theta
    lda_theta[validation_idx] = validation_theta
    lda_theta[test_idx] = test_theta

    x_model_vocab = build_bow_matrix_for_vocabulary(documents, vocab).tocsr()
    train_background = bow_background_distribution(x_model_vocab[train_idx])
    all_idx = np.arange(len(documents), dtype=np.int64)
    lda_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "tomotopy-lda-train-plus-heldout-inference",
        "config": asdict(config),
        "input": {
            "documents": int(len(documents)),
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(validation_idx)),
            "test_documents": int(len(test_idx)),
            "vocab_size": int(len(vocab)),
            "topics": int(beta.shape[0]),
        },
        "model_metadata": lda_metadata,
        "heldout_inference": {
            "validation": validation_inference_metadata,
            "test": test_inference_metadata,
        },
        "metrics": {
            "membership_count_diagnostics": split_diagnostics(lda_theta, splits),
            "all_reconstruction_nll": numpy_reconstruction_nll(
                x_model_vocab,
                lda_theta,
                beta,
                train_background,
                all_idx,
                background_weight=config.background_weight,
                batch_size=config.batch_size,
            ),
            "test_reconstruction_nll": numpy_reconstruction_nll(
                x_model_vocab,
                lda_theta,
                beta,
                train_background,
                test_idx,
                background_weight=config.background_weight,
                batch_size=config.batch_size,
            ),
        },
        "outputs": {
            "theta_npy": str(lda_dir / "theta.npy"),
            "beta_npy": str(lda_dir / "beta.npy"),
            "vocab_json": str(lda_dir / "vocab.json"),
            "train_history_json": str(lda_dir / "train_history.json"),
        },
    }
    write_standard_model_outputs(
        lda_dir,
        theta=lda_theta,
        beta=beta,
        vocab=vocab,
        history=lda_history,
        summary=lda_summary,
    )

    encoder, encoder_history = train_encoder(
        x_model_vocab,
        lda_theta,
        beta,
        train_indices=train_idx,
        validation_indices=validation_idx,
        background=train_background,
        config=config,
    )
    neural_theta_raw = infer_theta(encoder, x_model_vocab, config=config)
    neural_theta = sharpen_theta(neural_theta_raw, config.theta_export_power)
    neural_metrics = {
        **prefixed(
            "train",
            theta_metrics(
                lda_theta,
                neural_theta,
                train_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "validation",
            theta_metrics(
                lda_theta,
                neural_theta,
                validation_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "test",
            theta_metrics(
                lda_theta,
                neural_theta,
                test_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        **prefixed(
            "all",
            theta_metrics(
                lda_theta,
                neural_theta,
                all_idx,
                membership_threshold=config.membership_threshold,
            ),
        ),
        "lda_test_reconstruction_nll": numpy_reconstruction_nll(
            x_model_vocab,
            lda_theta,
            beta,
            train_background,
            test_idx,
            background_weight=config.background_weight,
            batch_size=config.batch_size,
        ),
        "neural_test_reconstruction_nll": numpy_reconstruction_nll(
            x_model_vocab,
            neural_theta,
            beta,
            train_background,
            test_idx,
            background_weight=config.background_weight,
            batch_size=config.batch_size,
        ),
        "membership_count_diagnostics": split_diagnostics(neural_theta, splits),
    }
    neural_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "fixed-beta-bow-encoder",
        "config": asdict(config),
        "input": {
            "documents": int(len(documents)),
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(validation_idx)),
            "test_documents": int(len(test_idx)),
            "vocab_size": int(len(vocab)),
            "topics": int(beta.shape[0]),
        },
        "metrics": neural_metrics,
        "outputs": {
            "theta_npy": str(neural_dir / "theta.npy"),
            "theta_raw_npy": str(neural_dir / "theta_raw.npy"),
            "theta_teacher_npy": str(neural_dir / "theta_teacher.npy"),
            "beta_npy": str(neural_dir / "beta.npy"),
            "vocab_json": str(neural_dir / "vocab.json"),
            "model_checkpoint": str(neural_dir / "model_checkpoint.pt"),
        },
    }
    write_standard_model_outputs(
        neural_dir,
        theta=neural_theta,
        beta=beta,
        vocab=vocab,
        history=encoder_history,
        summary=neural_summary,
        checkpoint={
            "model_type": "fixed_beta_bow_theta_encoder",
            "config": asdict(config),
            "state_dict": encoder.state_dict(),
        },
    )
    np.save(neural_dir / "theta_raw.npy", neural_theta_raw)
    np.save(neural_dir / "theta_teacher.npy", lda_theta)
    write_json(neural_dir / "validation_metrics.json", neural_metrics)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "fixed-beta-encoder-experiment",
        "config": asdict(config),
        "input": {
            "input_cache": str(input_cache),
            "documents": int(len(documents)),
            "cache_vocab_size": int(len(cache["vocab"])),
            "lda_vocab_size": int(len(vocab)),
        },
        "splits": {
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(validation_idx)),
            "test_documents": int(len(test_idx)),
        },
        "metrics": {
            "lda": lda_summary["metrics"],
            "neural_encoder": neural_metrics,
        },
        "outputs": {
            "split_indices_json": str(out_dir / "split_indices.json"),
            "lda_inferred_dir": str(lda_dir),
            "neural_encoder_dir": str(neural_dir),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
