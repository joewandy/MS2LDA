# ruff: noqa: C901, PERF203, PERF401, PLR2004, S603, S607
"""Protocol validation, static audit, and immutable run provenance."""

from __future__ import annotations

import ast
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .utils import file_sha256, object_sha256, read_json, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"
CONTINUATION_PROTOCOL_PATH = PACKAGE_ROOT / "protocol_k1000_continuation.json"
PROTOCOL_PATHS = (PROTOCOL_PATH, CONTINUATION_PROTOCOL_PATH)
RUNNER_PATH = REPO_ROOT / "scripts/run_neural_assignment_ms2lda.sh"
CONTINUATION_RUNNER_PATH = (
    REPO_ROOT / "scripts/run_neural_assignment_ms2lda_k1000_continuation.sh"
)
DESIGN_NOTE_PATH = REPO_ROOT / "docs/research/neural_assignment_ms2lda_protocol.md"
CONTINUATION_NOTE_PATH = (
    REPO_ROOT / "docs/research/neural_assignment_ms2lda_k1000_continuation.md"
)

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
    "protocol.lock.json",
    "model_assignments.json",
    "completion_manifest.jsonl",
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
    "synthetic.py",
)

TRANSITIVE_CANDIDATE_MODULES = {
    "shared_sgns_embeddings.py": (
        REPO_ROOT / "benchmarks/fully_neural_ms2lda/embeddings.py"
    ),
    "shared_sgns_data.py": REPO_ROOT / "benchmarks/fully_neural_ms2lda/data.py",
    "shared_sgns_utils.py": REPO_ROOT / "benchmarks/fully_neural_ms2lda/utils.py",
}

SHARED_SOURCE_FILES = (
    "benchmarks/fully_neural_ms2lda/chemical.py",
    "benchmarks/fully_neural_ms2lda/data.py",
    "benchmarks/fully_neural_ms2lda/embeddings.py",
    "benchmarks/fully_neural_ms2lda/metrics.py",
    "benchmarks/fully_neural_ms2lda/utils.py",
)


def resolve_protocol_path(path: str | Path | None = None) -> Path:
    """Resolve one of the two committed protocols, rejecting ad-hoc files."""
    selected = PROTOCOL_PATH if path is None else Path(path).expanduser().resolve()
    allowed = {candidate.resolve() for candidate in PROTOCOL_PATHS}
    if selected.resolve() not in allowed:
        msg = "the runner accepts only committed neural-assignment protocols"
        raise ValueError(msg)
    return selected.resolve()


def _validate_exploratory_amendment(
    protocol: dict[str, Any],
    *,
    schema: str,
) -> None:
    """Prove that protocol v2 changes only declared screening metadata."""
    amendment = protocol.get("exploratory_amendment")
    if schema != "neural-assignment-ms2lda/protocol-v2":
        if amendment is not None:
            msg = "the original protocol cannot contain a post-hoc amendment"
            raise ValueError(msg)
        return
    if amendment is None:
        msg = "protocol v2 requires an explicit exploratory amendment"
        raise ValueError(msg)
    if amendment.get("waived_k200_blocking_failures") != ["active_topics"]:
        msg = "only the K=200 active-topic screening stop may be waived"
        raise ValueError(msg)
    base = read_json(PROTOCOL_PATH)
    base_science = {
        key: value
        for key, value in base.items()
        if key not in {"schema_version", "evidence_scope"}
    }
    continuation_science = {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "schema_version",
            "evidence_scope",
            "exploratory_amendment",
        }
    }
    if continuation_science != base_science:
        msg = "the exploratory continuation changed a frozen scientific setting"
        raise ValueError(msg)


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a committed staged protocol."""
    selected = resolve_protocol_path(path)
    protocol = read_json(selected)
    schema = protocol.get("schema_version")
    if schema not in {
        "neural-assignment-ms2lda/protocol-v1",
        "neural-assignment-ms2lda/protocol-v2",
    }:
        msg = "unexpected neural-assignment protocol schema"
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
    if dimensions != protocol["model"]["input_dimensions"]:
        msg = "token and model input dimensions differ"
        raise ValueError(msg)
    if protocol["training_cpu_threads"] != 4:
        msg = "training must remain pinned to four CPU threads"
        raise ValueError(msg)
    if protocol["evaluation_cpu_threads"] != 1:
        msg = "evaluation must remain pinned to one CPU thread"
        raise ValueError(msg)
    if protocol["views"]["pairs"] != 4:
        msg = "the study requires four deterministic view pairs"
        raise ValueError(msg)
    if protocol["optimization"]["topic_updates_per_epoch"] != 4:
        msg = "the study requires four exact topic updates per epoch"
        raise ValueError(msg)
    if protocol["stages"]["k1000"]["num_topics"] != 1000:
        msg = "the final stage must retain K=1000"
        raise ValueError(msg)
    if protocol["stop_rule"]["maximum_k1000_attempts"] != 2:
        msg = "exactly one primary and at most one rescue are allowed"
        raise ValueError(msg)
    if not protocol["stop_rule"]["no_automatic_annotation_redirect"]:
        msg = "the accepted stop rule forbids an automatic redirect"
        raise ValueError(msg)
    occupation = float(
        protocol["synthetic_gates"]["occupation_usage_fraction_of_uniform"],
    )
    if not 0 < occupation < 1:
        msg = "synthetic occupation threshold must be a fraction of uniform"
        raise ValueError(msg)
    _validate_exploratory_amendment(protocol, schema=str(schema))
    return protocol


def _git_state() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def environment_manifest() -> dict[str, Any]:
    """Capture runtime versions without importing optional chemical packages."""
    packages = {}
    for name in ("numpy", "scipy", "torch", "rdkit", "faiss-cpu"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _source_files() -> list[Path]:
    files = list(PACKAGE_ROOT.glob("*.py"))
    files.extend(
        (
            *PROTOCOL_PATHS,
            PACKAGE_ROOT / "README.md",
            RUNNER_PATH,
            CONTINUATION_RUNNER_PATH,
            DESIGN_NOTE_PATH,
            CONTINUATION_NOTE_PATH,
        ),
    )
    files.extend(REPO_ROOT / relative for relative in SHARED_SOURCE_FILES)
    validation = REPO_ROOT / "benchmarks/msnlib_validation"
    files.extend(validation.glob("*.py"))
    files.extend((REPO_ROOT / "MS2LDA").rglob("*.py"))
    return sorted({path for path in files if path.is_file()})


def code_manifest() -> dict[str, str]:
    """Hash every maintained source file that can affect the staged study."""
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in _source_files()
    }


def static_candidate_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    """Reject forbidden candidate imports and record the one-pass contract."""
    forbidden = tuple(map(str.lower, protocol["forbidden_candidate_dependencies"]))
    imports: dict[str, list[str]] = {}
    violations: list[dict[str, str]] = []
    candidate_paths = {name: PACKAGE_ROOT / name for name in CANDIDATE_MODULES}
    candidate_paths.update(TRANSITIVE_CANDIDATE_MODULES)
    for name, path in candidate_paths.items():
        if not path.is_file():
            msg = f"candidate module is missing: {name}"
            raise FileNotFoundError(msg)
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
        "schema_version": "neural-assignment-ms2lda/candidate-audit-v1",
        "candidate_modules": list(candidate_paths),
        "forbidden_dependencies": list(forbidden),
        "imports": imports,
        "violations": [],
        "fully_neural": True,
        "discovery_model": "peak_to_topic_neural_assignment",
        "local_vb_steps": 0,
        "iterative_test_inference_steps": 0,
        "encoder_passes_per_representation": 1,
        "dreams_used": False,
        "classical_topic_teacher_used": False,
        "tomotopy_or_nmf_initialization_used": False,
        "conjugate_updates_used": False,
        "chemistry_fields_in_model_inputs": [],
        "allowed_safeguards": [
            "stop_gradient_sinkhorn_targets",
            "deterministic_dead_prototype_recycling",
        ],
    }


def initialize_run(
    run_dir: str | Path,
    *,
    source_run: str | Path,
    reference_run: str | Path,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze code and immutable input identities before fitting."""
    directory = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    reference = Path(reference_run).expanduser().resolve()
    selected_protocol = resolve_protocol_path(protocol_path)
    protocol_source = str(selected_protocol.relative_to(REPO_ROOT))
    lock_path = directory / "neural.lock.json"
    if lock_path.is_file():
        lock = verify_run(directory)
        if Path(lock["source_run"]) != source:
            msg = "requested source differs from the frozen run"
            raise ValueError(msg)
        if Path(lock["reference_run"]) != reference:
            msg = "requested reference differs from the frozen run"
            raise ValueError(msg)
        if lock.get("protocol_source", str(PROTOCOL_PATH.relative_to(REPO_ROOT))) != (
            protocol_source
        ):
            msg = "requested protocol differs from the frozen run"
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

    protocol = load_protocol(selected_protocol)
    audit = static_candidate_audit(protocol)
    manifest = code_manifest()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.resolved.json", protocol)
    write_json(directory / "candidate_audit.json", audit)
    write_json(directory / "code_manifest.json", manifest)
    write_json(directory / "environment.json", environment_manifest())
    frozen = directory / "frozen_source"
    for relative in manifest:
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    lock: dict[str, Any] = {
        "schema_version": "neural-assignment-ms2lda/run-lock-v1",
        "repo_root": str(REPO_ROOT),
        "git": _git_state(),
        "source_run": str(source),
        "reference_run": str(reference),
        "protocol_source": protocol_source,
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
    """Verify self-hashes, source closure, and immutable scientific inputs."""
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / "neural.lock.json")
    expected_lock = lock["lock_sha256"]
    unhashed = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if object_sha256(unhashed) != expected_lock:
        msg = "neural-assignment run lock self-hash mismatch"
        raise ValueError(msg)
    protocol = read_json(directory / "protocol.resolved.json")
    if object_sha256(protocol) != lock["protocol_sha256"]:
        msg = "frozen neural-assignment protocol changed"
        raise ValueError(msg)
    manifest = read_json(directory / "code_manifest.json")
    if object_sha256(manifest) != lock["code_manifest_sha256"]:
        msg = "stored neural-assignment code manifest changed"
        raise ValueError(msg)
    root = REPO_ROOT if require_live_code else directory / "frozen_source"
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            msg = f"neural-assignment source changed: {relative}"
            raise ValueError(msg)
    audit = read_json(directory / "candidate_audit.json")
    if object_sha256(audit) != lock["candidate_audit_sha256"]:
        msg = "neural-assignment candidate audit changed"
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
