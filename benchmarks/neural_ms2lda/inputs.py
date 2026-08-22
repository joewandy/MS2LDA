"""Lean MSnLib input configuration and provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import file_sha256


@dataclass(frozen=True)
class PreprocessingConfig:
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


def config_from_protocol(protocol: dict[str, Any]) -> PreprocessingConfig:
    """Build the parser configuration from the single neural protocol."""
    settings = protocol["preprocessing"]
    return PreprocessingConfig(
        **{name: settings[name] for name in PreprocessingConfig.__dataclass_fields__}
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
