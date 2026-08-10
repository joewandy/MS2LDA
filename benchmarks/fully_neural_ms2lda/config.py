# ruff: noqa: C901, PERF401, PLR2004, S607
"""Protocol freezing and provenance checks for the bounded neural study."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .utils import file_sha256, object_sha256, read_json, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"

COUNT_FILES = (
    "complete.json",
    "heldout_records.jsonl",
    "identifiers.json",
    "train.npz",
    "validation_observed.npz",
    "validation_completion.npz",
    "validation_full.npz",
    "test_observed.npz",
    "test_completion.npz",
    "test_full.npz",
    "vocabulary.json",
)

REFERENCE_FILES = (
    "config.resolved.json",
    "core/seed_42/tomotopy/beta.npy",
    "core/seed_42/tomotopy/complete.json",
    "chemical_inference/seed_42/tomotopy/complete.json",
    "mag/seed_42/tomotopy/complete.json",
)

CANDIDATE_MODULES = (
    "data.py",
    "embeddings.py",
    "model.py",
    "training.py",
)


def load_protocol() -> dict[str, Any]:
    """Load and validate the committed frozen protocol."""
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("schema_version") != "fully-neural-ms2lda/protocol-v1":
        msg = "unexpected fully neural protocol schema"
        raise ValueError(msg)
    features = protocol["token_features"]
    dimensions = (
        protocol["sgns"]["dimensions"]
        + 2 * len(features["fourier_frequencies"])
        + features["type_dimensions"]
    )
    if dimensions != features["output_dimensions"]:
        msg = "token feature dimensions do not sum to the frozen output"
        raise ValueError(msg)
    if dimensions != protocol["model"]["embedding_dimensions"]:
        msg = "token and model embedding dimensions differ"
        raise ValueError(msg)
    if protocol["training_cpu_threads"] != 4:
        msg = "training must remain pinned to four CPU threads"
        raise ValueError(msg)
    if protocol["evaluation_cpu_threads"] != 1:
        msg = "evaluation must remain pinned to one CPU thread"
        raise ValueError(msg)
    return protocol


def _git_revision() -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_files() -> list[Path]:
    paths = sorted(PACKAGE_ROOT.glob("*.py"))
    paths.extend((PROTOCOL_PATH, REPO_ROOT / "scripts/run_fully_neural_ms2lda.sh"))
    return paths


def code_manifest() -> dict[str, str]:
    """Hash every maintained file that can affect the study."""
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in _source_files()
        if path.is_file()
    }


def static_candidate_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    """Reject forbidden imports from every candidate execution module."""
    forbidden = tuple(map(str.lower, protocol["forbidden_candidate_dependencies"]))
    imports: dict[str, list[str]] = {}
    violations: list[dict[str, str]] = []
    for name in CANDIDATE_MODULES:
        path = PACKAGE_ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_imports.append(node.module)
        imports[name] = sorted(set(module_imports))
        for imported in imports[name]:
            lowered = imported.lower()
            for blocked in forbidden:
                if blocked in lowered:
                    violations.append(
                        {"file": name, "import": imported, "forbidden": blocked},
                    )
    if violations:
        msg = f"forbidden candidate dependency: {violations[0]}"
        raise RuntimeError(msg)
    return {
        "schema_version": "fully-neural-ms2lda/candidate-audit-v1",
        "candidate_modules": list(CANDIDATE_MODULES),
        "forbidden_dependencies": list(forbidden),
        "imports": imports,
        "violations": [],
        "fully_neural": True,
        "local_vb_steps": 0,
        "iterative_test_inference_steps": 0,
        "dreams_used": False,
        "tomotopy_or_nmf_initialization_used": False,
        "conjugate_updates_used": False,
    }


def initialize_run(
    run_dir: str | Path,
    *,
    source_run: str | Path,
    reference_run: str | Path,
) -> dict[str, Any]:
    """Freeze code and input identities before any model fitting."""
    directory = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    reference = Path(reference_run).expanduser().resolve()
    lock_path = directory / "neural.lock.json"
    if lock_path.is_file():
        lock = verify_run(directory)
        if Path(lock["source_run"]) != source:
            msg = "requested source run differs from the frozen neural run"
            raise ValueError(msg)
        if Path(lock["reference_run"]) != reference:
            msg = "requested reference run differs from the frozen neural run"
            raise ValueError(msg)
        return lock
    counts = source / "shared/counts"
    missing = [counts / name for name in COUNT_FILES if not (counts / name).is_file()]
    missing.extend(
        reference / name for name in REFERENCE_FILES if not (reference / name).is_file()
    )
    if missing:
        msg = f"required neural input is missing: {missing[0]}"
        raise FileNotFoundError(msg)
    protocol = load_protocol()
    audit = static_candidate_audit(protocol)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.resolved.json", protocol)
    write_json(directory / "candidate_audit.json", audit)
    manifest = code_manifest()
    write_json(directory / "code_manifest.json", manifest)
    frozen = directory / "frozen_source"
    for relative in manifest:
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    lock: dict[str, Any] = {
        "schema_version": "fully-neural-ms2lda/run-lock-v1",
        "repo_root": str(REPO_ROOT),
        "git_revision": _git_revision(),
        "source_run": str(source),
        "reference_run": str(reference),
        "protocol_sha256": object_sha256(protocol),
        "code_manifest_sha256": object_sha256(manifest),
        "candidate_audit_sha256": object_sha256(audit),
        "count_inputs": {name: file_sha256(counts / name) for name in COUNT_FILES},
        "reference_inputs": {
            name: file_sha256(reference / name) for name in REFERENCE_FILES
        },
    }
    lock["lock_sha256"] = object_sha256(lock)
    write_json(lock_path, lock)
    return lock


def verify_run(
    run_dir: str | Path,
    *,
    require_live_code: bool = True,
    verify_large_inputs: bool = True,
) -> dict[str, Any]:
    """Verify the frozen protocol, code, audit, and immutable inputs."""
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / "neural.lock.json")
    digest = lock.pop("lock_sha256")
    if object_sha256(lock) != digest:
        msg = "fully neural run lock self-hash mismatch"
        raise ValueError(msg)
    lock["lock_sha256"] = digest
    protocol = read_json(directory / "protocol.resolved.json")
    if object_sha256(protocol) != lock["protocol_sha256"]:
        msg = "frozen fully neural protocol changed"
        raise ValueError(msg)
    manifest = read_json(directory / "code_manifest.json")
    if object_sha256(manifest) != lock["code_manifest_sha256"]:
        msg = "stored neural code manifest changed"
        raise ValueError(msg)
    root = REPO_ROOT if require_live_code else directory / "frozen_source"
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            msg = f"neural source changed: {relative}"
            raise ValueError(msg)
    audit = read_json(directory / "candidate_audit.json")
    if object_sha256(audit) != lock["candidate_audit_sha256"]:
        msg = "fully neural candidate audit changed"
        raise ValueError(msg)
    if verify_large_inputs:
        counts = Path(lock["source_run"]) / "shared/counts"
        for name, expected in lock["count_inputs"].items():
            if file_sha256(counts / name) != expected:
                msg = f"count input changed: {name}"
                raise ValueError(msg)
        reference = Path(lock["reference_run"])
        for name, expected in lock["reference_inputs"].items():
            if file_sha256(reference / name) != expected:
                msg = f"reference input changed: {name}"
                raise ValueError(msg)
    return lock
