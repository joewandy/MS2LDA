# ruff: noqa: C901, PLR0912
"""Immutable study specification, source bindings, and integrity checks."""

from __future__ import annotations

import ast
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.msnlib_validation.config import (
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)
from benchmarks.msnlib_validation.protocol import (
    code_manifest as source_protocol_code_manifest,
)
from benchmarks.msnlib_validation.protocol import verify_protocol

SCHEMA_VERSION = "msnlib-hybrid-simplification/v1"
LOCK_NAME = "simplification.lock.json"
DISCOVERY_IDS = ("dreams_prior", "symmetric_prior")
INFERENCE_IDS = (
    "dreams_semi",
    "dreams_direct",
    "topic_semi",
    "topic_direct",
    "analytic",
)
ARM_IDS = tuple(
    f"{discovery}__{inference}"
    for discovery in DISCOVERY_IDS
    for inference in INFERENCE_IDS
)
BUDGETS = (0, 1, 2, 50)
FROZEN_SEED = 42
FROZEN_TOPICS = 1000
FROZEN_TRAINING_THREADS = 4
FROZEN_EVALUATION_THREADS = 1


@dataclass(frozen=True)
class SimplificationSpec:
    """Decision-complete settings for the single-seed factorial study."""

    schema_version: str = SCHEMA_VERSION
    study_name: str = "hybrid-lda-simplification-seed42"
    seed: int = FROZEN_SEED
    num_topics: int = FROZEN_TOPICS
    discoveries: tuple[str, ...] = DISCOVERY_IDS
    inference_modes: tuple[str, ...] = INFERENCE_IDS
    budgets: tuple[int, ...] = BUDGETS
    direct_target_steps: int = 50
    semi_refinement_steps: int = 2
    inference_epochs: int = 12
    batch_size: int = 32
    symmetric_max_epochs: int = 250
    symmetric_min_epochs: int = 20
    global_patience: int = 5
    bootstrap_replicates: int = 2000
    heartbeat_seconds: int = 300
    minimum_free_disk_gib: int = 20
    checkpoint_keep: int = 2
    training_cpu_threads: int = FROZEN_TRAINING_THREADS
    evaluation_cpu_threads: int = FROZEN_EVALUATION_THREADS
    membership_threshold: float = 0.5
    sos_margin: float = 0.02
    coverage_fraction: float = 0.90
    nll_relative_margin: float = 0.02
    npmi_margin: float = 0.02
    cosine_p05_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            msg = "unsupported simplification schema"
            raise ValueError(msg)
        if self.seed != FROZEN_SEED or self.num_topics != FROZEN_TOPICS:
            msg = "this frozen study requires seed 42 and K=1000"
            raise ValueError(msg)
        if self.discoveries != DISCOVERY_IDS:
            msg = "discovery matrix changed"
            raise ValueError(msg)
        if self.inference_modes != INFERENCE_IDS:
            msg = "inference matrix changed"
            raise ValueError(msg)
        if self.budgets != BUDGETS:
            msg = "inference budgets changed"
            raise ValueError(msg)
        positive = {
            "direct_target_steps": self.direct_target_steps,
            "semi_refinement_steps": self.semi_refinement_steps,
            "inference_epochs": self.inference_epochs,
            "batch_size": self.batch_size,
            "symmetric_max_epochs": self.symmetric_max_epochs,
            "symmetric_min_epochs": self.symmetric_min_epochs,
            "global_patience": self.global_patience,
            "bootstrap_replicates": self.bootstrap_replicates,
            "heartbeat_seconds": self.heartbeat_seconds,
            "minimum_free_disk_gib": self.minimum_free_disk_gib,
            "checkpoint_keep": self.checkpoint_keep,
            "training_cpu_threads": self.training_cpu_threads,
            "evaluation_cpu_threads": self.evaluation_cpu_threads,
        }
        if any(
            isinstance(value, bool) or int(value) < 1 for value in positive.values()
        ):
            msg = "positive integer study settings are required"
            raise ValueError(msg)
        if (
            self.training_cpu_threads != FROZEN_TRAINING_THREADS
            or self.evaluation_cpu_threads != FROZEN_EVALUATION_THREADS
        ):
            msg = (
                "this frozen study requires four training threads "
                "and one evaluation thread"
            )
            raise ValueError(msg)
        if self.direct_target_steps != max(self.budgets):
            msg = "direct targets must use the frozen long budget"
            raise ValueError(msg)
        if self.symmetric_min_epochs >= self.symmetric_max_epochs:
            msg = "symmetric discovery minimum must precede its maximum"
            raise ValueError(msg)
        for value in (
            self.membership_threshold,
            self.sos_margin,
            self.coverage_fraction,
            self.nll_relative_margin,
            self.npmi_margin,
            self.cosine_p05_margin,
        ):
            if not 0 <= value <= 1:
                msg = "study thresholds must lie in [0, 1]"
                raise ValueError(msg)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe canonical representation."""
        value = asdict(self)
        for name in ("discoveries", "inference_modes", "budgets"):
            value[name] = list(value[name])
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_paths() -> tuple[str, ...]:
    return (
        "protocol.lock.json",
        "code_manifest.json",
        "config.resolved.json",
        "environment.json",
        "input_manifest.json",
        "split_manifest.jsonl",
        "completion_manifest.jsonl",
        "vocabulary.json",
        "features/manifest.json",
        "features/identifiers.json",
        "features/global_embeddings.npy",
        "features/word_embeddings.npy",
        "core/seed_42/hybrid/complete.json",
        "core/seed_42/hybrid/model.pt",
        "core/seed_42/hybrid/beta.npy",
        "core/seed_42/hybrid/test_theta_0.npy",
        "core/seed_42/hybrid/test_theta_2.npy",
        "core/seed_42/hybrid/test_theta_50.npy",
        "core/seed_42/hybrid/discovery_history.json",
        "core/seed_42/hybrid/inference_history.json",
        "core/seed_42/tomotopy/complete.json",
        "core/seed_42/tomotopy/model.bin",
        "core/seed_42/tomotopy/beta.npy",
        "core/seed_42/tomotopy/test_theta.npy",
        "chemical_inference/features/manifest.json",
        "chemical_inference/features/identifiers.json",
        "chemical_inference/features/full_test_embeddings.npy",
        "chemical_inference/seed_42/hybrid/complete.json",
        "chemical_inference/seed_42/hybrid/test_full_theta_encoder.npy",
        "chemical_inference/seed_42/hybrid/test_full_theta_two_step.npy",
        "chemical_inference/seed_42/hybrid/test_full_theta_long.npy",
        "mag/index/manifest.json",
        "mag/index/kept_original_ids.npy",
        "mag/index/spec2vec_filtered.faiss",
        "mag/index/excluded_connectivity_keys.json",
        "mag/seed_42/hybrid/complete.json",
        "mag/seed_42/hybrid/topics.jsonl",
    )


def code_manifest(repo_root: str | Path) -> dict[str, str]:
    """Hash every source file capable of changing this benchmark."""
    root = Path(repo_root).expanduser().resolve()
    paths: list[Path] = []
    for relative in (
        "benchmarks/msnlib_simplification",
        "benchmarks/msnlib_validation",
        "ms2lda_hybrid",
    ):
        paths.extend(
            path
            for path in (root / relative).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json", ".md"}
        )
    paths.extend(
        path
        for path in (
            root / "benchmarks" / "inference_baselines.py",
            root / "scripts" / "run_hybrid_simplification_overnight.sh",
            root / "environment-hybrid.yml",
            root / "environment-msnlib-mag.yml",
            root / "pyproject.toml",
            root / "poetry.lock",
        )
        if path.is_file()
    )
    return {
        str(path.relative_to(root)): file_sha256(path) for path in sorted(set(paths))
    }


class _RemoveDocstrings(ast.NodeTransformer):
    """Normalize Python syntax while ignoring documentation-only changes."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        result = super().generic_visit(node)
        body = getattr(result, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            result.body = body[1:]
        return result


def _semantic_python_at_git_ref(
    repository: Path,
    reference: str,
    relative: str,
) -> str:
    result = subprocess.run(  # noqa: S603 - fixed Git executable and frozen ref/path
        ["/usr/bin/git", "show", f"{reference}:{relative}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    tree = _RemoveDocstrings().visit(ast.parse(result.stdout, filename=relative))
    return ast.dump(tree, include_attributes=False)


def _verify_source_code_drift(
    repository: Path,
    source: Path,
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    """Allow only the post-run prose/docstring publication changes."""
    historical = read_json(source / "code_manifest.json")
    if object_sha256(historical) != source_lock["source_manifest_sha256"]:
        msg = "corrected source code manifest does not match its protocol lock"
        raise ValueError(msg)
    current = source_protocol_code_manifest(repository)
    changed = sorted(
        relative
        for relative in set(historical) | set(current)
        if historical.get(relative) != current.get(relative)
    )
    allowed = {
        "benchmarks/msnlib_validation/README.md",
        "ms2lda_hybrid/dreams_features.py",
    }
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        msg = f"source implementation drifted after corrected run: {unexpected[0]}"
        raise ValueError(msg)
    source_commit = str(source_lock["git"]["commit"])
    python_path = "ms2lda_hybrid/dreams_features.py"
    historical_syntax = _semantic_python_at_git_ref(
        repository,
        source_commit,
        python_path,
    )
    current_tree = _RemoveDocstrings().visit(
        ast.parse((repository / python_path).read_text(encoding="utf-8")),
    )
    if ast.dump(current_tree, include_attributes=False) != historical_syntax:
        msg = "DreaMS source changed semantically after corrected run"
        raise ValueError(msg)
    return {
        "historical_manifest_sha256": object_sha256(historical),
        "current_manifest_sha256": object_sha256(current),
        "changed_files": changed,
        "allowed_changes": sorted(allowed),
        "python_change_is_docstring_only": True,
        "source_commit": source_commit,
    }


def freeze_study(
    *,
    run_dir: str | Path,
    source_run: str | Path,
    repo_root: str | Path,
    spec: SimplificationSpec | None = None,
) -> dict[str, Any]:
    """Freeze the full matrix against the corrected seed-42 source run."""
    directory = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    repository = Path(repo_root).expanduser().resolve()
    specification = SimplificationSpec() if spec is None else spec
    if directory.exists() and any(directory.iterdir()):
        if (directory / LOCK_NAME).is_file():
            return verify_study(directory)
        msg = "refusing to freeze into a non-empty run directory"
        raise FileExistsError(msg)

    source_lock = verify_protocol(source, verify_code=False)
    source_code_drift = _verify_source_code_drift(repository, source, source_lock)
    source_config = read_json(source / "config.resolved.json")
    if (
        source_config.get("seeds") != [FROZEN_SEED]
        or source_config.get("num_topics") != FROZEN_TOPICS
    ):
        msg = "source run is not the corrected seed-42 K=1000 study"
        raise ValueError(msg)
    source_files: dict[str, dict[str, Any]] = {}
    for relative in _source_paths():
        path = source / relative
        if not path.is_file():
            msg = f"required source artifact is missing: {path}"
            raise FileNotFoundError(msg)
        source_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    current_complete = read_json(source / "core/seed_42/hybrid/complete.json")
    if source_files["core/seed_42/hybrid/model.pt"]["sha256"] != current_complete.get(
        "model_sha256",
    ):
        msg = "source Hybrid model hash does not match its completion"
        raise ValueError(msg)
    if source_files["core/seed_42/hybrid/beta.npy"]["sha256"] != current_complete.get(
        "beta_sha256",
    ):
        msg = "source Hybrid beta hash does not match its completion"
        raise ValueError(msg)

    manifest = code_manifest(repository)
    git_status = subprocess.run(  # noqa: S603 - fixed Git executable and argv
        ["/usr/bin/git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "repo_root": str(repository),
        "source_run": str(source),
        "source_protocol_sha256": source_lock["protocol_sha256"],
        "source_code_drift_audit": source_code_drift,
        "source_files": source_files,
        "source_external_inputs": read_json(source / "input_manifest.json")["files"],
        "spec": specification.as_dict(),
        "spec_sha256": object_sha256(specification.as_dict()),
        "code_manifest": manifest,
        "code_manifest_sha256": object_sha256(manifest),
        "git": {
            "branch": subprocess.run(  # noqa: S603 - fixed Git executable and argv
                ["/usr/bin/git", "branch", "--show-current"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "commit": subprocess.run(  # noqa: S603 - fixed Git executable and argv
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "status_porcelain": git_status,
        },
        "scientific_status": "posthoc_single_seed_simplification_experiment",
        "test_results_previously_inspected": True,
        "selection_or_adoption_authorized": False,
    }
    payload["lock_sha256"] = object_sha256(payload)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / LOCK_NAME, payload)
    write_json(directory / "spec.resolved.json", specification.as_dict())
    write_json(directory / "code_manifest.json", manifest)
    return payload


def load_spec(run_dir: str | Path) -> SimplificationSpec:
    """Load and validate the frozen specification."""
    value = read_json(Path(run_dir).expanduser().resolve() / "spec.resolved.json")
    for name in ("discoveries", "inference_modes", "budgets"):
        value[name] = tuple(value[name])
    return SimplificationSpec(**value)


def verify_study(
    run_dir: str | Path,
    *,
    verify_source_files: bool = False,
    verify_external_inputs: bool = False,
) -> dict[str, Any]:
    """Verify the immutable lock, code, and optionally large source inputs."""
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / LOCK_NAME)
    digest = lock.pop("lock_sha256")
    if object_sha256(lock) != digest:
        msg = "simplification lock self-hash mismatch"
        raise ValueError(msg)
    lock["lock_sha256"] = digest
    spec = load_spec(directory)
    if object_sha256(spec.as_dict()) != lock.get("spec_sha256"):
        msg = "frozen simplification specification changed"
        raise ValueError(msg)
    manifest = code_manifest(lock["repo_root"])
    if object_sha256(manifest) != lock.get("code_manifest_sha256"):
        msg = "simplification benchmark source changed after freeze"
        raise ValueError(msg)
    if manifest != read_json(directory / "code_manifest.json"):
        msg = "stored code manifest changed"
        raise ValueError(msg)
    source = Path(lock["source_run"])
    source_protocol = read_json(source / "protocol.lock.json")
    if source_protocol.get("protocol_sha256") != lock.get("source_protocol_sha256"):
        msg = "corrected source protocol changed"
        raise ValueError(msg)
    if verify_source_files:
        for relative, expected in lock["source_files"].items():
            path = source / relative
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                msg = f"source artifact size changed: {relative}"
                raise ValueError(msg)
            if file_sha256(path) != expected["sha256"]:
                msg = f"source artifact hash changed: {relative}"
                raise ValueError(msg)
    if verify_external_inputs:
        for name, expected in lock["source_external_inputs"].items():
            path = Path(expected["path"])
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                msg = f"external input size changed: {name}"
                raise ValueError(msg)
            if file_sha256(path) != expected["sha256"]:
                msg = f"external input hash changed: {name}"
                raise ValueError(msg)
    return lock


def verify_archived_study(
    run_dir: str | Path,
    *,
    frozen_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an immutable completed run against its archived source snapshot.

    ``verify_study`` deliberately binds a resumable run to the live checkout.
    This companion is for historical result bundles after development has
    continued: it validates the lock and stored manifest against the exact
    ``frozen_source`` tree without weakening the live-run guard.
    """
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / LOCK_NAME)
    digest = lock.pop("lock_sha256")
    if object_sha256(lock) != digest:
        msg = "simplification lock self-hash mismatch"
        raise ValueError(msg)
    lock["lock_sha256"] = digest

    specification = load_spec(directory)
    if object_sha256(specification.as_dict()) != lock.get("spec_sha256"):
        msg = "frozen simplification specification changed"
        raise ValueError(msg)
    stored_manifest = read_json(directory / "code_manifest.json")
    if object_sha256(stored_manifest) != lock.get("code_manifest_sha256"):
        msg = "stored code manifest changed"
        raise ValueError(msg)

    source_root = (
        directory / "frozen_source"
        if frozen_source_root is None
        else Path(frozen_source_root).expanduser().resolve()
    )
    for relative, expected in stored_manifest.items():
        path = source_root / relative
        if not path.is_file():
            msg = f"archived source file is missing: {relative}"
            raise ValueError(msg)
        if file_sha256(path) != expected:
            msg = f"archived source file changed: {relative}"
            raise ValueError(msg)
    lock["verified_frozen_source_root"] = str(source_root)
    lock["verified_frozen_source_files"] = len(stored_manifest)
    return lock


def preflight(run_dir: str | Path) -> dict[str, Any]:
    """Perform all unattended-run safety and environment checks."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(
        directory,
        verify_source_files=True,
        verify_external_inputs=True,
    )
    spec = load_spec(directory)
    free_bytes = shutil.disk_usage(directory).free
    checks: dict[str, Any] = {
        "code_and_inputs_verified": True,
        "free_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": spec.minimum_free_disk_gib * 1024**3,
        "screen": shutil.which("screen"),
        "caffeinate": shutil.which("caffeinate"),
        "conda": shutil.which("conda"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
    }
    checks["disk_passed"] = free_bytes >= checks["minimum_free_disk_bytes"]
    checks["tools_passed"] = all(
        checks[name] for name in ("screen", "caffeinate", "conda")
    )
    power = subprocess.run(  # noqa: S603 - fixed pmset executable and argv
        ["/usr/bin/pmset", "-g", "batt"],
        check=False,
        capture_output=True,
        text=True,
    )
    checks["power_output"] = power.stdout.strip()
    checks["ac_power"] = "AC Power" in power.stdout
    checks["environment_passed"] = checks["conda_environment"] == "ms2lda-hybrid"
    environment_script = """import importlib.metadata as metadata
import json
import platform
import sys
names = ('numpy', 'scipy', 'torch', 'rdkit', 'faiss-cpu', 'matchms', 'dreams', 'ms2lda')
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
payload = {'python': sys.version, 'platform': platform.platform(), 'packages': versions}
print(json.dumps(payload, sort_keys=True))
"""
    environments: dict[str, Any] = {}
    for environment in ("ms2lda-hybrid", "ms2lda-msnlib-mag"):
        result = subprocess.run(  # noqa: S603 - frozen environment names and command
            [
                str(checks["conda"]),
                "run",
                "-n",
                environment,
                "python",
                "-c",
                environment_script,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            snapshot = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            snapshot = None
        environments[environment] = {
            "available": result.returncode == 0 and snapshot is not None,
            "returncode": result.returncode,
            "snapshot": snapshot,
            "stderr": result.stderr.strip(),
        }
    checks["environments"] = environments
    checks["worker_environments_passed"] = all(
        value["available"] for value in environments.values()
    )
    lock_path = directory / "runner.lock"
    lock_active = False
    if lock_path.exists():
        try:
            runner_pid = int(read_json(lock_path)["pid"])
            os.kill(runner_pid, 0)
        except (KeyError, OSError, ProcessLookupError, TypeError, ValueError):
            runner_pid = None
        else:
            lock_active = True
        checks["existing_runner_pid"] = runner_pid
    checks["runner_lock_inactive"] = not lock_active
    checks["passed"] = all(
        checks[name]
        for name in (
            "disk_passed",
            "tools_passed",
            "ac_power",
            "environment_passed",
            "worker_environments_passed",
            "runner_lock_inactive",
        )
    )
    checks["created_utc"] = _utc_now()
    checks["lock_sha256"] = lock["lock_sha256"]
    write_json(directory / "environment_manifest.json", environments)
    write_json(directory / "preflight.json", checks)
    if not checks["passed"]:
        failed = [
            name
            for name in (
                "disk_passed",
                "tools_passed",
                "ac_power",
                "environment_passed",
                "worker_environments_passed",
                "runner_lock_inactive",
            )
            if not checks[name]
        ]
        msg = f"overnight preflight failed: {', '.join(failed)}"
        raise RuntimeError(msg)
    return checks
