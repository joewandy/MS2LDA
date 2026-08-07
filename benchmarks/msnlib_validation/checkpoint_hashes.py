"""Hash ledgers for resumable row-oriented embedding caches."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


def hash_array_rows(values: np.ndarray, start: int, end: int) -> str:
    """Return a stable SHA-256 for one contiguous row interval."""
    array = np.asarray(values)
    if array.ndim < 1 or start < 0 or end <= start or end > len(array):
        raise ValueError("embedding hash interval is invalid")
    selected = np.ascontiguousarray(array[start:end])
    digest = hashlib.sha256()
    digest.update(selected.dtype.str.encode("ascii"))
    digest.update(str(selected.shape).encode("ascii"))
    digest.update(memoryview(selected).cast("B"))
    return digest.hexdigest()


def extend_row_hash_ledger(
    ledger: Sequence[dict[str, Any]],
    values: np.ndarray,
    *,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Append one non-overlapping contiguous interval to a verified ledger."""
    validated = validate_row_hash_ledger(values, ledger, completed_rows=start)
    return [
        *validated,
        {"start": start, "end": end, "sha256": hash_array_rows(values, start, end)},
    ]


def validate_row_hash_ledger(
    values: np.ndarray,
    ledger: Any,
    *,
    completed_rows: int,
) -> list[dict[str, Any]]:
    """Validate contiguous row ranges and every recorded embedding digest."""
    array = np.asarray(values)
    if array.ndim < 1 or completed_rows < 0 or completed_rows > len(array):
        raise ValueError("embedding checkpoint row count is invalid")
    if not isinstance(ledger, list):
        raise ValueError("embedding checkpoint hash ledger is invalid")
    expected_start = 0
    validated: list[dict[str, Any]] = []
    for item in ledger:
        if not isinstance(item, dict) or set(item) != {"start", "end", "sha256"}:
            raise ValueError("embedding checkpoint hash ledger is invalid")
        start = item["start"]
        end = item["end"]
        digest = item["sha256"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start != expected_start
            or end <= start
            or end > completed_rows
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("embedding checkpoint hash ledger is invalid")
        if hash_array_rows(array, start, end) != digest:
            raise ValueError(f"embedding checkpoint rows {start}:{end} changed")
        validated.append({"start": start, "end": end, "sha256": digest})
        expected_start = end
    if expected_start != completed_rows:
        raise ValueError("embedding checkpoint hash ledger is incomplete")
    return validated
