"""Small functional utilities shared by reproducible neural experiments.

The functions here own deterministic execution, immutable validation inputs,
simple artifact I/O, probability checks, and runtime-memory measurements.
They intentionally use ordinary dictionaries and arrays rather than stateful
experiment classes.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import resource
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .utils import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

VALIDATION_DATA_FILES = (
    "train.npz",
    "validation_observed.npz",
    "validation_completion.npz",
    "validation_full.npz",
    "validation_records.jsonl",
    "vocabulary.json",
)
VALIDATION_MAG_INDEX_FILES = (
    "complete.json",
    "excluded_connectivity_keys.json",
    "kept_original_ids.npy",
    "spec2vec_filtered.faiss",
)
MemoryState = dict[str, int | None]


def configure_deterministic_execution(seed: int, threads: int) -> None:
    """Seed Python and PyTorch and request deterministic CPU/CUDA operations."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if threads <= 0:
        raise ValueError("threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(mode=True)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def resolve_torch_device(name: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` and fail if CUDA was unavailable."""
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if selected not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(selected)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object, rejecting arrays and scalar documents."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a deterministic small table using keys from its first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_file_link(destination: Path, source: Path) -> None:
    """Create one immutable-input symlink and reject conflicting destinations."""
    resolved_source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != resolved_source:
            raise ValueError(f"existing link has a different source: {destination}")
        return
    if destination.exists():
        raise FileExistsError(
            f"validation-view destination already exists: {destination}",
        )
    destination.symlink_to(resolved_source)


def prepare_validation_view(
    run_directory: Path,
    prepared_run: Path,
    *,
    expected_topics: int,
) -> dict[str, Any]:
    """Link only frozen training/validation inputs into a writable run directory.

    Candidate-test files are deliberately absent.  The returned manifest records
    every source path, byte count, and digest so a run can prove exactly which
    inputs were visible.
    """
    source = prepared_run.expanduser().resolve(strict=True)
    destination = run_directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = source / "protocol.json"
    protocol = read_json_object(protocol_path)
    if int(protocol["model"]["num_topics"]) != int(expected_topics):
        raise ValueError(
            f"validation protocol must specify K={expected_topics}",
        )
    destination_protocol = destination / "protocol.json"
    write_json(destination_protocol, protocol)

    visible_inputs: list[tuple[Path, Path]] = [(protocol_path, destination_protocol)]
    for name in VALIDATION_DATA_FILES:
        source_path = source / "data" / name
        linked_path = destination / "data" / name
        _ensure_file_link(linked_path, source_path)
        visible_inputs.append((source_path, linked_path))
    features_path = source / "token_features" / "features.npy"
    linked_features = destination / "token_features" / "features.npy"
    _ensure_file_link(linked_features, features_path)
    visible_inputs.append((features_path, linked_features))
    for name in VALIDATION_MAG_INDEX_FILES:
        source_path = source / "mag" / "index" / name
        linked_path = destination / "mag" / "index" / name
        _ensure_file_link(linked_path, source_path)
        visible_inputs.append((source_path, linked_path))

    manifest = {
        "evidence_boundary": "training plus validation only",
        "prepared_run": str(source),
        "candidate_test_artifacts_accessed": False,
        "candidate_test_metrics_inspected": False,
        "linked_inputs": [
            {
                "path": str(path),
                "linked_path": str(linked_path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, linked_path in visible_inputs
        ],
    }
    write_json(destination / "validation_input_manifest.json", manifest)
    return manifest


def _system_available_bytes() -> int | None:
    """Read Linux MemAvailable, or return None when the metric is unavailable."""
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def sample_runtime_memory(previous: MemoryState | None = None) -> MemoryState:
    """Return updated process and system-memory high-water measurements."""
    state = previous or {
        "peak_process_bytes": 0,
        "minimum_system_available_bytes": None,
    }
    process_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    available_bytes = _system_available_bytes()
    previous_available = state["minimum_system_available_bytes"]
    return {
        "peak_process_bytes": max(int(state["peak_process_bytes"] or 0), process_bytes),
        "minimum_system_available_bytes": (
            previous_available
            if available_bytes is None
            else (
                available_bytes
                if previous_available is None
                else min(previous_available, available_bytes)
            )
        ),
    }


def runtime_memory_metrics(
    state: MemoryState,
    device: torch.device,
) -> dict[str, Any]:
    """Return reportable CPU/system/CUDA memory measurements."""
    final_state = sample_runtime_memory(state)
    return {
        "measurement": (
            "sampled per epoch; process is Linux ru_maxrss and system is "
            "minimum /proc/meminfo MemAvailable"
        ),
        **final_state,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
    }


def validate_probability_matrix(probabilities: np.ndarray, *, name: str) -> None:
    """Fail closed unless every row is a finite non-negative probability vector."""
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty row matrix")
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError(f"{name} contains non-finite values")
    if np.any(probabilities < 0):
        raise FloatingPointError(f"{name} contains negative values")
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, atol=2e-6):
        maximum_deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise FloatingPointError(
            f"{name} rows do not sum to one: minimum={row_sums.min():.9g}, "
            f"maximum={row_sums.max():.9g}, "
            f"maximum_deviation={maximum_deviation:.9g}",
        )


def normalize_probability_rows(
    probabilities: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a float32 row-stochastic matrix for persisted evaluation.

    A wide float32 softmax is mathematically normalized, but rounding each of
    tens of thousands of vocabulary probabilities can leave its float64 row
    sum measurably different from one.  Re-normalizing the exported values in
    float64 makes the stored matrix match the probability-simplex equation;
    it does not alter training, model weights, or relative word probabilities.
    """
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty row matrix")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{name} contains non-finite values")
    if np.any(values < 0):
        raise FloatingPointError(f"{name} contains negative values")

    precise = values.astype(np.float64, copy=False)
    row_sums = precise.sum(axis=1, dtype=np.float64)
    if np.any(row_sums <= 0):
        raise FloatingPointError(f"{name} contains a zero-mass row")
    normalized = (precise / row_sums[:, None]).astype(np.float32)
    validate_probability_matrix(normalized, name=name)
    return normalized


def flatten_support_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Flatten support percentiles into one reviewable CSV row."""
    row = {
        key: value
        for key, value in summary.items()
        if key != "support_size_percentiles"
    }
    percentiles = summary["support_size_percentiles"]
    if not isinstance(percentiles, Mapping):
        raise TypeError("support_size_percentiles must be a mapping")
    row.update({f"support_p{key}": value for key, value in percentiles.items()})
    return row
