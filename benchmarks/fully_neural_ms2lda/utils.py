"""Small deterministic I/O helpers for the neural benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> bytes:
    """Serialize a value deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    """Return the SHA-256 of a JSON-compatible object."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write stable, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, value: Any) -> None:
    """Durably append one deterministic JSON event."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_save_numpy(path: str | Path, values: np.ndarray) -> None:
    """Atomically save a NumPy array."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, values)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    """Atomically save a PyTorch checkpoint."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)
