"""Lightweight cache and process-resource helpers shared across environments."""

from __future__ import annotations

import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import read_json


def peak_rss_bytes() -> int:
    """Return current-process peak RSS in bytes on macOS and Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def load_feature_cache(
    run_dir: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load finalized row-aligned global and vocabulary feature arrays."""
    directory = Path(run_dir).expanduser().resolve() / "features"
    manifest = read_json(directory / "manifest.json")
    identifiers = list(
        map(str, read_json(directory / "identifiers.json")["identifiers"])
    )
    global_embeddings = np.load(directory / "global_embeddings.npy", mmap_mode="r")
    word_embeddings = np.load(directory / "word_embeddings.npy", mmap_mode="r")
    if global_embeddings.shape[0] != len(identifiers):
        raise ValueError("feature identifiers and global rows differ")
    return identifiers, global_embeddings, word_embeddings, manifest
