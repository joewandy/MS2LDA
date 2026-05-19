#!/usr/bin/env python
"""Export MSn benchmark annotations and memberships from model theta/beta outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.msn_benchmark_pipeline import (  # noqa: E402
    EXPORT_OUTPUT_FILENAMES,
    MEMBERSHIP_THRESHOLD,
    annotate_topics_from_beta,
    clear_named_outputs,
    entropy_rows,
    export_memberships,
    load_input_cache,
    resolve_path,
    select_eval_topic_ids,
    write_json,
    write_topic_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotations.csv and memberships.csv for an MSn benchmark model run."
    )
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-eval-motifs", type=int, default=50)
    parser.add_argument("--motif-top-n", type=int, default=20)
    parser.add_argument(
        "--membership-threshold", type=float, default=MEMBERSHIP_THRESHOLD
    )
    parser.add_argument(
        "--split-indices",
        type=Path,
        help=(
            "Optional split_indices.json. When set, memberships are exported only "
            "for --split-name."
        ),
    )
    parser.add_argument(
        "--split-name",
        choices=["train", "validation", "test"],
        help="Split to export when --split-indices is provided.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model_vocab(model_dir: Path, cache_vocab: list[str]) -> list[str]:
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        return cache_vocab
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    return [str(token) for token in payload["vocab"]]


def load_split_indices(split_indices_path: Path, split_name: str) -> np.ndarray:
    payload = json.loads(split_indices_path.read_text(encoding="utf-8"))
    key = f"{split_name}_indices"
    if key not in payload and split_name in payload:
        key = split_name
    if key not in payload:
        raise KeyError(f"{split_indices_path} does not contain {split_name!r} indices.")
    return np.asarray([int(value) for value in payload[key]], dtype=np.int64)


def select_export_split(
    theta: np.ndarray,
    spectra_metadata,
    split_indices: np.ndarray | None,
) -> tuple[np.ndarray, object]:
    if split_indices is None:
        return theta, spectra_metadata
    if np.any(split_indices < 0) or np.any(split_indices >= theta.shape[0]):
        raise ValueError("Split indices are outside the theta row range.")
    return theta[split_indices], spectra_metadata.iloc[split_indices].reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    input_cache = resolve_path(args.input_cache)
    model_dir = resolve_path(args.model_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else model_dir

    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and out_dir != model_dir
        and not args.overwrite
    ):
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        out_dir / name for name in EXPORT_OUTPUT_FILENAMES if (out_dir / name).exists()
    ]
    if existing_outputs and not args.overwrite:
        existing_text = "\n".join(f"- {path}" for path in existing_outputs)
        raise FileExistsError(
            f"Export outputs already exist; pass --overwrite.\n{existing_text}"
        )
    if args.overwrite:
        clear_named_outputs(out_dir, EXPORT_OUTPUT_FILENAMES)

    theta_path = model_dir / "theta.npy"
    beta_path = model_dir / "beta.npy"
    if not theta_path.exists() or not beta_path.exists():
        raise FileNotFoundError(f"Expected theta.npy and beta.npy in {model_dir}")

    cache = load_input_cache(input_cache)
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    vocab = load_model_vocab(model_dir, cache["vocab"])
    if beta.shape[1] != len(vocab):
        raise ValueError(
            f"beta has {beta.shape[1]} columns but vocab has {len(vocab)} entries."
        )
    if theta.shape[0] != len(cache["spectra_metadata"]):
        raise ValueError(
            "theta row count does not match cached spectra_metadata.csv row count."
        )
    if (args.split_indices is None) != (args.split_name is None):
        raise ValueError("--split-indices and --split-name must be provided together.")

    split_indices = None
    if args.split_indices is not None:
        split_indices = load_split_indices(
            resolve_path(args.split_indices),
            str(args.split_name),
        )
    theta_export, metadata_export = select_export_split(
        theta,
        cache["spectra_metadata"],
        split_indices,
    )

    topic_ids = select_eval_topic_ids(
        theta_export,
        beta,
        max_eval_motifs=args.max_eval_motifs,
        membership_threshold=args.membership_threshold,
    )
    annotations, annotation_metadata = annotate_topics_from_beta(
        beta,
        vocab,
        topic_ids,
        motif_top_n=args.motif_top_n,
        motifset=f"msn_{model_dir.name}",
    )
    memberships = export_memberships(
        theta_export,
        metadata_export,
        topic_ids,
        membership_threshold=args.membership_threshold,
    )

    annotations.to_csv(out_dir / "annotations.csv", index=False)
    memberships.to_csv(out_dir / "memberships.csv", index=False)
    write_topic_diagnostics(
        out_dir / "topic_diagnostics.csv",
        theta_export,
        beta,
        vocab,
        membership_threshold=args.membership_threshold,
    )

    beta_entropy = entropy_rows(beta)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_cache": str(input_cache),
        "model_dir": str(model_dir),
        "parameters": {
            "max_eval_motifs": int(args.max_eval_motifs),
            "motif_top_n": int(args.motif_top_n),
            "membership_threshold": float(args.membership_threshold),
            "split_indices": (
                str(resolve_path(args.split_indices))
                if args.split_indices is not None
                else None
            ),
            "split_name": args.split_name,
        },
        "annotation": annotation_metadata,
        "metrics": {
            "selected_eval_motifs": int(len(topic_ids)),
            "membership_rows": int(len(memberships)),
            "exported_spectra": int(theta_export.shape[0]),
            "topics_with_memberships": int(
                memberships["motif_id"].nunique() if len(memberships) else 0
            ),
            "mean_beta_entropy": float(beta_entropy.mean()),
            "mean_beta_effective_support": float(np.exp(beta_entropy).mean()),
            "mean_beta_max_probability": float(beta.max(axis=1).mean()),
        },
        "outputs": {
            "annotations_csv": str(out_dir / "annotations.csv"),
            "memberships_csv": str(out_dir / "memberships.csv"),
            "topic_diagnostics_csv": str(out_dir / "topic_diagnostics.csv"),
        },
    }
    write_json(out_dir / "export_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
