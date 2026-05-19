#!/usr/bin/env python
"""Prepare reusable cached input for MSn benchmark model runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.msn_benchmark_pipeline import (
    prepare_input_cache,
    resolve_path,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the MSn MGF once and cache model-independent benchmark inputs."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit-spectra", type=int)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--min-cf", type=float, default=0.0)
    parser.add_argument("--rm-top", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    summary = prepare_input_cache(
        dataset=dataset,
        out_dir=resolve_path(args.out_dir),
        limit_spectra=args.limit_spectra,
        min_df=args.min_df,
        min_cf=args.min_cf,
        rm_top=args.rm_top,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
