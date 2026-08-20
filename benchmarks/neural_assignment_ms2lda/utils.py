"""Small atomic I/O and provenance helpers for the bounded study."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

_MACOS_RSS_BYTES_THRESHOLD = 10_000_000


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write deterministic, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """Atomically write canonical JSON Lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_output_hashes(directory: str | Path, manifest: dict[str, Any]) -> None:
    """Verify every artifact in a standard stage output map."""
    root = Path(directory)
    for name, digest in manifest["output_sha256"].items():
        path = root / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"artifact changed: {path}")


def atomic_save_numpy(path: str | Path, value: np.ndarray) -> None:
    """Atomically save an uncompressed NumPy array."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    """Atomically save a PyTorch checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def peak_rss_bytes() -> int:
    """Return process peak resident memory in bytes on macOS or Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > _MACOS_RSS_BYTES_THRESHOLD else value * 1024
