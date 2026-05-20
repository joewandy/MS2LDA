#!/usr/bin/env python
"""Refine MSn benchmark theta outputs against a fixed learned beta matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
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

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    EPS,
    MEMBERSHIP_THRESHOLD,
    MODEL_OUTPUT_FILENAMES,
    clear_named_outputs,
    entropy_rows,
    load_input_cache,
    membership_count_diagnostics,
    normalize_rows,
    resolve_path,
    sharpen_theta,
    write_json,
)


EXTRA_OUTPUT_FILENAMES = {
    "theta_init.npy",
    "theta_refined_raw.npy",
    "theta_refinement_metrics.json",
    "split_indices.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a benchmark-compatible model directory by refining an "
            "existing theta matrix against fixed beta and the observed MSn BoW."
        )
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--theta-init",
        choices=["raw", "exported", "uniform"],
        default="raw",
        help=(
            "Initial theta. 'raw' uses theta_raw.npy when present and falls "
            "back to theta.npy."
        ),
    )
    parser.add_argument("--refine-iters", type=int, default=10)
    parser.add_argument(
        "--encoder-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight added to the initial theta after each multiplicative "
            "update before renormalization."
        ),
    )
    parser.add_argument("--min-theta", type=float, default=1e-12)
    parser.add_argument("--theta-export-power", type=float, default=3.0)
    parser.add_argument(
        "--membership-threshold",
        type=float,
        default=MEMBERSHIP_THRESHOLD,
        help="Threshold used only for diagnostics and benchmark export.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dense_normalized_batch(x, indices: np.ndarray) -> np.ndarray:
    dense = x[indices].toarray().astype(np.float32, copy=False)
    denom = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, denom + EPS).astype(np.float32, copy=False)


def normalize_theta_tensor(theta: torch.Tensor, *, min_theta: float) -> torch.Tensor:
    theta = theta.clamp_min(float(min_theta))
    return theta / theta.sum(dim=1, keepdim=True).clamp_min(EPS)


def reconstruction_values(
    x_norm: torch.Tensor,
    theta: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    dist = (theta @ beta).clamp_min(EPS)
    dist = dist / dist.sum(dim=1, keepdim=True).clamp_min(EPS)
    return -(x_norm * dist.log()).sum(dim=1)


def refine_theta_batch(
    x_norm: torch.Tensor,
    theta_init: torch.Tensor,
    beta: torch.Tensor,
    *,
    refine_iters: int,
    encoder_prior_weight: float,
    min_theta: float,
) -> torch.Tensor:
    theta = normalize_theta_tensor(theta_init, min_theta=min_theta)
    prior = theta.clone()
    non_empty = x_norm.sum(dim=1) > EPS
    if int(refine_iters) <= 0:
        return theta

    for _ in range(int(refine_iters)):
        dist = (theta @ beta).clamp_min(EPS)
        update = (x_norm / dist) @ beta.T
        theta = theta * update.clamp_min(float(min_theta))
        if float(encoder_prior_weight) > 0:
            theta = theta + (float(encoder_prior_weight) * prior)
        theta = normalize_theta_tensor(theta, min_theta=min_theta)
        theta = torch.where(non_empty[:, None], theta, prior)
    return theta


def load_theta_init(model_dir: Path, n_docs: int, n_topics: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.full((n_docs, n_topics), 1.0 / float(n_topics), dtype=np.float32)
    if mode == "raw":
        theta_path = model_dir / "theta_raw.npy"
        if not theta_path.exists():
            theta_path = model_dir / "theta.npy"
    else:
        theta_path = model_dir / "theta.npy"
    theta = np.load(theta_path).astype(np.float32, copy=False)
    if theta.shape != (n_docs, n_topics):
        raise ValueError(
            f"{theta_path} has shape {theta.shape}; expected {(n_docs, n_topics)}."
        )
    return normalize_rows(theta)


def refine_theta_matrix(
    x,
    theta_init: np.ndarray,
    beta: np.ndarray,
    *,
    refine_iters: int,
    encoder_prior_weight: float,
    min_theta: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, float]]:
    beta = normalize_rows(beta)
    beta_tensor = torch.from_numpy(beta).float().to(device)
    rows = []
    before_values = []
    after_values = []
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), x.shape[0]))
            x_norm = torch.from_numpy(dense_normalized_batch(x, indices)).to(device)
            theta_batch = torch.from_numpy(theta_init[indices]).float().to(device)
            theta_batch = normalize_theta_tensor(theta_batch, min_theta=min_theta)
            before_values.append(
                reconstruction_values(x_norm, theta_batch, beta_tensor)
                .detach()
                .cpu()
                .numpy()
            )
            refined = refine_theta_batch(
                x_norm,
                theta_batch,
                beta_tensor,
                refine_iters=refine_iters,
                encoder_prior_weight=encoder_prior_weight,
                min_theta=min_theta,
            )
            after_values.append(
                reconstruction_values(x_norm, refined, beta_tensor)
                .detach()
                .cpu()
                .numpy()
            )
            rows.append(refined.detach().cpu().numpy())
    theta = normalize_rows(np.vstack(rows).astype(np.float32, copy=False))
    before = np.concatenate(before_values) if before_values else np.asarray([])
    after = np.concatenate(after_values) if after_values else np.asarray([])
    metrics = {
        "mean_reconstruction_before": float(before.mean()) if before.size else 0.0,
        "mean_reconstruction_after": float(after.mean()) if after.size else 0.0,
        "median_reconstruction_before": float(np.median(before)) if before.size else 0.0,
        "median_reconstruction_after": float(np.median(after)) if after.size else 0.0,
    }
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
        "mean_max_theta": float(np.max(theta, axis=1).mean()),
        "mean_theta_entropy": float(entropy.mean()),
    }


def clear_outputs(out_dir: Path) -> None:
    clear_named_outputs(out_dir, MODEL_OUTPUT_FILENAMES | EXTRA_OUTPUT_FILENAMES)


def maybe_copy_split_indices(model_dir: Path, out_dir: Path) -> str | None:
    split_path = model_dir / "split_indices.json"
    if not split_path.exists():
        return None
    target = out_dir / "split_indices.json"
    shutil.copyfile(split_path, target)
    return str(target)


def run_refinement(args: argparse.Namespace) -> dict[str, Any]:
    input_cache = resolve_path(args.input_cache)
    model_dir = resolve_path(args.model_dir)
    out_dir = resolve_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_outputs(out_dir)

    cache = load_input_cache(input_cache)
    x = cache["matrix"].tocsr()
    beta = normalize_rows(np.load(model_dir / "beta.npy"))
    if beta.ndim != 2 or beta.shape[1] != x.shape[1]:
        raise ValueError(
            f"beta.npy has shape {beta.shape}; expected second dimension {x.shape[1]}."
        )
    theta_init = load_theta_init(
        model_dir,
        n_docs=x.shape[0],
        n_topics=beta.shape[0],
        mode=str(args.theta_init),
    )
    theta_refined_raw, refinement_metrics = refine_theta_matrix(
        x,
        theta_init,
        beta,
        refine_iters=args.refine_iters,
        encoder_prior_weight=args.encoder_prior_weight,
        min_theta=args.min_theta,
        batch_size=args.batch_size,
        device=args.device,
    )
    theta = sharpen_theta(theta_refined_raw, float(args.theta_export_power))
    vocab_payload = read_json(model_dir / "vocab.json")

    np.save(out_dir / "theta_init.npy", theta_init)
    np.save(out_dir / "theta_refined_raw.npy", theta_refined_raw)
    np.save(out_dir / "theta_raw.npy", theta_refined_raw)
    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "beta.npy", beta)
    write_json(out_dir / "vocab.json", vocab_payload)
    split_indices_path = maybe_copy_split_indices(model_dir, out_dir)

    metrics = {
        **refinement_metrics,
        **{f"init_{key}": value for key, value in theta_summary(
            theta_init,
            membership_threshold=float(args.membership_threshold),
        ).items()},
        **{f"refined_{key}": value for key, value in theta_summary(
            theta,
            membership_threshold=float(args.membership_threshold),
        ).items()},
        "membership_count_diagnostics": membership_count_diagnostics(theta),
    }
    write_json(out_dir / "theta_refinement_metrics.json", metrics)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "theta-refined",
        "input": {
            "input_cache": str(input_cache),
            "model_dir": str(model_dir),
            "documents": int(x.shape[0]),
            "vocab_size": int(x.shape[1]),
            "topics": int(beta.shape[0]),
        },
        "parameters": {
            "theta_init": str(args.theta_init),
            "refine_iters": int(args.refine_iters),
            "encoder_prior_weight": float(args.encoder_prior_weight),
            "min_theta": float(args.min_theta),
            "theta_export_power": float(args.theta_export_power),
            "membership_threshold": float(args.membership_threshold),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
        },
        "metrics": metrics,
        "outputs": {
            "theta_npy": str(out_dir / "theta.npy"),
            "theta_raw_npy": str(out_dir / "theta_raw.npy"),
            "theta_refined_raw_npy": str(out_dir / "theta_refined_raw.npy"),
            "theta_init_npy": str(out_dir / "theta_init.npy"),
            "beta_npy": str(out_dir / "beta.npy"),
            "vocab_json": str(out_dir / "vocab.json"),
            "split_indices_json": split_indices_path,
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_refinement(parse_args())


if __name__ == "__main__":
    main()
