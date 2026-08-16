"""Lean MSnLib input configuration and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and immutable locks."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    """Return the SHA-256 digest of one canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
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
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Only the preprocessing settings consumed by the MSnLib parser."""

    expected_spectra: int
    min_mz: float
    max_mz: float
    max_fragments: int
    min_fragments: int
    min_intensity: float
    max_intensity: float
    significant_digits: int

    def __post_init__(self) -> None:
        if self.expected_spectra < 1:
            raise ValueError("expected_spectra must be positive")
        if not 0 <= self.min_mz < self.max_mz:
            raise ValueError("invalid m/z interval")
        if not 0 <= self.min_intensity <= self.max_intensity:
            raise ValueError("invalid intensity interval")
        if self.max_fragments < self.min_fragments or self.min_fragments < 1:
            raise ValueError("invalid fragment-count interval")
        if self.significant_digits < 0:
            raise ValueError("significant_digits cannot be negative")


def config_from_protocol(protocol: dict[str, Any]) -> BenchmarkConfig:
    """Build the parser configuration from the single neural protocol."""
    settings = protocol["preprocessing"]
    return BenchmarkConfig(
        **{name: settings[name] for name in BenchmarkConfig.__dataclass_fields__}
    )


def resolve_input_paths(
    protocol: dict[str, Any], data_root: str | Path
) -> dict[str, Path]:
    """Resolve every declared Zenodo-relative input below ``data_root``."""
    root = Path(data_root).expanduser().resolve()
    paths = {
        name: (root / str(spec["relative_path"])).resolve()
        for name, spec in protocol["input_files"].items()
    }
    for name, path in paths.items():
        if root not in path.parents:
            raise ValueError(f"input escapes data root: {name}")
    return paths


def verify_inputs(
    protocol: dict[str, Any],
    data_root: str | Path,
    *,
    names: set[str] | None = None,
    verify_hashes: bool = True,
) -> dict[str, dict[str, Any]]:
    """Verify existence, byte size, and optionally SHA-256 for frozen inputs."""
    paths = resolve_input_paths(protocol, data_root)
    selected = set(paths) if names is None else set(names)
    unknown = selected - set(paths)
    if unknown:
        raise ValueError(f"unknown input names: {sorted(unknown)}")
    results: dict[str, dict[str, Any]] = {}
    for name in sorted(selected):
        path = paths[name]
        spec = protocol["input_files"][name]
        if not path.is_file():
            raise FileNotFoundError(f"required MSnLib input is missing: {path}")
        size = path.stat().st_size
        if size != int(spec["bytes"]):
            raise ValueError(f"input byte size changed: {name}")
        digest = file_sha256(path) if verify_hashes else None
        if digest is not None and digest != str(spec["sha256"]):
            raise ValueError(f"input SHA-256 changed: {name}")
        results[name] = {
            "path": str(path),
            "bytes": size,
            "sha256": digest or str(spec["sha256"]),
            "hash_verified": verify_hashes,
        }
    return results
