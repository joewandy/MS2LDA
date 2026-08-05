"""Configuration, hashing, and protocol-lock helpers for the MSnLib study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "msnlib-validation/v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON-compatible value deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
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


def read_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fully resolved scientific settings for one immutable benchmark."""

    protocol_name: str
    evidence_scope: str
    seeds: tuple[int, ...]
    split_seed: int
    completion_seed: int
    split_fractions: tuple[float, float, float]
    num_topics: int
    min_mz: float
    max_mz: float
    max_fragments: int
    min_fragments: int
    min_intensity: float
    max_intensity: float
    significant_digits: int
    min_df: int
    min_cf: int
    rm_top: int
    alpha: float
    eta: float
    tomotopy_max_iterations: int
    tomotopy_step_size: int
    tomotopy_convergence_window: int
    tomotopy_convergence_threshold: float
    tomotopy_inference_iterations: int
    tomotopy_training_workers: int
    tomotopy_training_parallel: int
    hybrid_max_epochs: int
    hybrid_global_patience: int
    hybrid_inference_epochs: int
    hybrid_batch_size: int
    hybrid_training_cpu_threads: int
    hybrid_inference_cpu_threads: int
    hybrid_checkpoint_keep: int
    hybrid_reference_steps: int
    hybrid_reference_extension_steps: int
    reference_median_cosine: float
    reference_fifth_percentile_cosine: float
    completion_observed_fraction: float
    topic_top_n: int
    motif_spectrum_top_n: int
    document_active_threshold: float
    corpus_active_threshold: float
    membership_threshold: float
    mag_search_k: int
    mag_unique_molecules: int
    mag_cluster_cosine: float
    mag_fingerprint_threshold: float
    latency_subset_size: int
    latency_repeats: int
    expected_spectra: int
    input_files: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        """Reject incomplete or scientifically inconsistent configurations."""
        if not self.protocol_name:
            raise ValueError("protocol_name cannot be empty")
        if self.evidence_scope not in {"confirmatory", "indicative_single_seed"}:
            raise ValueError("unsupported evidence_scope")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds cannot be negative")
        if self.evidence_scope == "indicative_single_seed" and len(self.seeds) != 1:
            raise ValueError("indicative_single_seed requires exactly one seed")
        if self.evidence_scope == "confirmatory" and self.seeds != (
            42,
            43,
            44,
            45,
            46,
        ):
            raise ValueError("confirmatory evidence requires seeds 42 through 46")
        if len(self.split_fractions) != 3:
            raise ValueError("split_fractions must contain train, validation, test")
        if any(
            not math.isfinite(value) or value <= 0 for value in self.split_fractions
        ):
            raise ValueError("split fractions must be positive")
        if abs(sum(self.split_fractions) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to one")
        positive_integers = {
            "num_topics": self.num_topics,
            "max_fragments": self.max_fragments,
            "min_fragments": self.min_fragments,
            "min_df": self.min_df,
            "tomotopy_max_iterations": self.tomotopy_max_iterations,
            "tomotopy_step_size": self.tomotopy_step_size,
            "tomotopy_convergence_window": self.tomotopy_convergence_window,
            "tomotopy_inference_iterations": self.tomotopy_inference_iterations,
            "hybrid_max_epochs": self.hybrid_max_epochs,
            "hybrid_global_patience": self.hybrid_global_patience,
            "hybrid_inference_epochs": self.hybrid_inference_epochs,
            "hybrid_batch_size": self.hybrid_batch_size,
            "hybrid_training_cpu_threads": self.hybrid_training_cpu_threads,
            "hybrid_inference_cpu_threads": self.hybrid_inference_cpu_threads,
            "hybrid_checkpoint_keep": self.hybrid_checkpoint_keep,
            "hybrid_reference_steps": self.hybrid_reference_steps,
            "hybrid_reference_extension_steps": self.hybrid_reference_extension_steps,
            "topic_top_n": self.topic_top_n,
            "motif_spectrum_top_n": self.motif_spectrum_top_n,
            "mag_search_k": self.mag_search_k,
            "mag_unique_molecules": self.mag_unique_molecules,
            "latency_subset_size": self.latency_subset_size,
            "latency_repeats": self.latency_repeats,
            "expected_spectra": self.expected_spectra,
        }
        invalid = [
            name
            for name, value in positive_integers.items()
            if isinstance(value, bool) or not isinstance(value, int) or value < 1
        ]
        if invalid:
            raise ValueError(f"positive integers required for: {', '.join(invalid)}")
        if self.hybrid_inference_cpu_threads != 1:
            raise ValueError(
                "Hybrid held-out inference and latency must use exactly one CPU thread"
            )
        if self.hybrid_checkpoint_keep < 2:
            raise ValueError("retain at least two Hybrid training checkpoints")
        if self.hybrid_reference_steps < 2:
            raise ValueError("reference steps must permit a positive half-budget audit")
        if self.hybrid_reference_extension_steps <= self.hybrid_reference_steps:
            raise ValueError("reference extension must be longer than reference")
        if not 0 < self.completion_observed_fraction < 1:
            raise ValueError(
                "completion_observed_fraction must lie between zero and one"
            )
        if not 0 <= self.min_intensity <= self.max_intensity:
            raise ValueError("invalid intensity interval")
        if not 0 <= self.min_mz < self.max_mz:
            raise ValueError("invalid m/z interval")
        if (
            not math.isfinite(self.alpha)
            or not math.isfinite(self.eta)
            or self.alpha <= 0
            or self.eta <= 0
        ):
            raise ValueError("alpha and eta must be positive")
        unit_interval = {
            "reference_median_cosine": self.reference_median_cosine,
            "reference_fifth_percentile_cosine": self.reference_fifth_percentile_cosine,
            "document_active_threshold": self.document_active_threshold,
            "corpus_active_threshold": self.corpus_active_threshold,
            "membership_threshold": self.membership_threshold,
            "mag_cluster_cosine": self.mag_cluster_cosine,
            "mag_fingerprint_threshold": self.mag_fingerprint_threshold,
        }
        invalid_probabilities = [
            name for name, value in unit_interval.items() if not 0 <= value <= 1
        ]
        if invalid_probabilities:
            raise ValueError(
                f"unit-interval values required for: {', '.join(invalid_probabilities)}"
            )
        if (
            not math.isfinite(self.tomotopy_convergence_threshold)
            or self.tomotopy_convergence_threshold <= 0
        ):
            raise ValueError("Tomotopy convergence threshold must be positive")
        if (
            isinstance(self.tomotopy_training_workers, bool)
            or not isinstance(self.tomotopy_training_workers, int)
            or self.tomotopy_training_workers < 0
        ):
            raise ValueError("Tomotopy training workers cannot be negative")
        if (
            isinstance(self.tomotopy_training_parallel, bool)
            or not isinstance(self.tomotopy_training_parallel, int)
            or self.tomotopy_training_parallel not in {0, 1, 2, 3}
        ):
            raise ValueError("invalid Tomotopy parallel scheme")
        required_inputs = {
            "mgf",
            "spec2vec_model",
            "spec2vec_embeddings",
            "spec2vec_db",
        }
        missing = required_inputs - set(self.input_files)
        if missing:
            raise ValueError(f"missing input definitions: {sorted(missing)}")
        for name, spec in self.input_files.items():
            if not isinstance(spec, dict) or not spec.get("relative_path"):
                raise ValueError(f"input {name} requires relative_path")
            digest = str(spec.get("sha256", ""))
            if len(digest) != 64:
                raise ValueError(f"input {name} requires a SHA-256 digest")

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible configuration."""
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        value["split_fractions"] = list(self.split_fractions)
        value["schema_version"] = SCHEMA_VERSION
        return value


def load_config(path: str | Path) -> BenchmarkConfig:
    """Read and validate one benchmark JSON configuration."""
    payload = read_json(path)
    schema = payload.pop("schema_version", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema}")
    # Frozen pre-checkpoint protocols did not yet expose these execution-only
    # settings. Their recorded behavior was one training/inference thread and
    # no fewer than two recoverable checkpoint generations.
    payload.setdefault("hybrid_training_cpu_threads", 1)
    payload.setdefault("hybrid_inference_cpu_threads", 1)
    payload.setdefault("hybrid_checkpoint_keep", 2)
    payload["seeds"] = tuple(int(seed) for seed in payload["seeds"])
    payload["split_fractions"] = tuple(
        float(value) for value in payload["split_fractions"]
    )
    return BenchmarkConfig(**payload)


def resolve_input_paths(
    config: BenchmarkConfig, data_root: str | Path
) -> dict[str, Path]:
    """Resolve configured Zenodo-relative paths below ``data_root``."""
    root = Path(data_root).expanduser().resolve()
    return {
        name: (root / str(spec["relative_path"])).resolve()
        for name, spec in config.input_files.items()
    }


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Capture the exact Git state without changing it."""
    root = Path(repo_root)

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def environment_manifest() -> dict[str, Any]:
    """Capture lightweight dependency and hardware provenance."""
    distributions = {}
    for package in (
        "numpy",
        "scipy",
        "torch",
        "tomotopy",
        "rdkit",
        "matchms",
        "dreams",
        "spec2vec",
        "faiss-cpu",
    ):
        try:
            distributions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            distributions[package] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": distributions,
    }
