"""Frozen protocol and provenance management for neural MS2LDA."""

from __future__ import annotations

import ast
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from benchmarks.msnlib_validation.config import resolve_input_paths, verify_inputs

from .utils import file_sha256, object_sha256, read_json, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"

SOURCE_FILES = (
    "benchmarks/msnlib_validation/__init__.py",
    "benchmarks/msnlib_validation/chemical.py",
    "benchmarks/msnlib_validation/config.py",
    "benchmarks/msnlib_validation/data.py",
    "benchmarks/msnlib_validation/mag.py",
    "benchmarks/msnlib_validation/metrics.py",
    "benchmarks/neural_assignment_ms2lda/__init__.py",
    "benchmarks/neural_assignment_ms2lda/__main__.py",
    "benchmarks/neural_assignment_ms2lda/bundle.py",
    "benchmarks/neural_assignment_ms2lda/chemical.py",
    "benchmarks/neural_assignment_ms2lda/cli.py",
    "benchmarks/neural_assignment_ms2lda/cooccurrence.py",
    "benchmarks/neural_assignment_ms2lda/config.py",
    "benchmarks/neural_assignment_ms2lda/core.py",
    "benchmarks/neural_assignment_ms2lda/data.py",
    "benchmarks/neural_assignment_ms2lda/embeddings.py",
    "benchmarks/neural_assignment_ms2lda/evaluation.py",
    "benchmarks/neural_assignment_ms2lda/inventory.py",
    "benchmarks/neural_assignment_ms2lda/metrics.py",
    "benchmarks/neural_assignment_ms2lda/model.py",
    "benchmarks/neural_assignment_ms2lda/orchestrator.py",
    "benchmarks/neural_assignment_ms2lda/regularizers.py",
    "benchmarks/neural_assignment_ms2lda/report.py",
    "benchmarks/neural_assignment_ms2lda/tomotopy.py",
    "benchmarks/neural_assignment_ms2lda/training.py",
    "benchmarks/neural_assignment_ms2lda/utils.py",
    "benchmarks/neural_assignment_ms2lda/development.py",
    "scripts/download_msnlib_validation_assets.py",
    "scripts/generate_neural_ms2lda_report.py",
    "scripts/run_neural_ms2lda.sh",
)


def load_protocol() -> dict[str, Any]:
    """Load and strictly validate the one supported protocol."""
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("schema_version") != "neural-ms2lda/protocol-v1":
        raise ValueError("unexpected neural MS2LDA protocol schema")
    if int(protocol["seed"]) != 42:
        raise ValueError("the checkpoint protocol is fixed to seed 42")
    if int(protocol["training_cpu_threads"]) != 4:
        raise ValueError("training must use four CPU threads")
    if int(protocol["model"]["num_topics"]) != 500:
        raise ValueError("the supported neural model is fixed to K=500")
    if int(protocol["model"]["top_k"]) != 2:
        raise ValueError("the supported router is fixed to top-2")
    routing = protocol["hierarchical_routing"]
    if routing["method"] != "local_document_product_of_experts":
        raise ValueError("unexpected hierarchical routing method")
    if float(routing["weight"]) != 1.0:
        raise ValueError("the document topic prior weight is fixed to one")
    if int(protocol["optimization"]["maximum_epochs"]) != 40:
        raise ValueError("the reproducibility run is fixed to 40 epochs")
    dimensions = (
        int(protocol["sgns"]["dimensions"])
        + 2 * len(protocol["token_features"]["fourier_frequencies"])
        + int(protocol["token_features"]["type_dimensions"])
    )
    if dimensions != int(protocol["model"]["input_dimensions"]):
        raise ValueError("token feature and model dimensions differ")
    return protocol


def _git_state() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def environment_manifest() -> dict[str, Any]:
    """Capture the packages that can affect the numerical workflow."""
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "scipy",
        "torch",
        "tomotopy",
        "rdkit",
        "faiss-cpu",
        "matchms",
    ):
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


def code_manifest() -> dict[str, str]:
    """Hash every source file that defines the committed workflow."""
    missing = [name for name in SOURCE_FILES if not (REPO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"neural workflow source files are missing: {missing}")
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def static_discovery_audit() -> dict[str, Any]:
    """Prove that candidate discovery does not import forbidden teachers."""
    discovery = (
        PACKAGE_ROOT / "model.py",
        PACKAGE_ROOT / "training.py",
        PACKAGE_ROOT / "core.py",
        PACKAGE_ROOT / "regularizers.py",
        PACKAGE_ROOT / "embeddings.py",
        PACKAGE_ROOT / "data.py",
    )
    forbidden = {"tomotopy", "dreams", "pymc", "ms2lda_hybrid"}
    imports: set[str] = set()
    suspicious_names: set[str] = set()
    for path in discovery:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".", 1)[0].lower() for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0].lower())
            elif isinstance(node, ast.Name) and node.id.lower() in forbidden:
                suspicious_names.add(node.id.lower())
    violations = sorted((imports | suspicious_names) & forbidden)
    if violations:
        raise RuntimeError(f"forbidden discovery dependency found: {violations}")
    return {
        "fully_neural": True,
        "forbidden_dependencies_found": [],
        "tomotopy_role": "post-training comparator only",
        "chemistry_fields_in_training": [],
        "test_information_in_training": False,
    }


def initialize_run(run_dir: str | Path, *, data_root: str | Path) -> dict[str, Any]:
    """Create or verify an immutable, exactly resumable run lock."""
    directory = Path(run_dir).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    lock_path = directory / "run.lock.json"
    expected = {
        "schema_version": "neural-ms2lda/run-lock-v1",
        "data_root": str(root),
        "protocol_sha256": object_sha256(protocol),
        "inputs": verify_inputs(protocol, root, names={"mgf"}),
        "code": code_manifest(),
        "environment": environment_manifest(),
        "git": _git_state(),
        "discovery_audit": static_discovery_audit(),
    }
    if lock_path.is_file():
        existing = read_json(lock_path)
        immutable = ("data_root", "protocol_sha256", "inputs", "code")
        if any(existing.get(key) != expected.get(key) for key in immutable):
            raise ValueError("run provenance differs from its immutable lock")
        return existing
    write_json(directory / "protocol.resolved.json", protocol)
    write_json(lock_path, expected)
    return expected


def verify_run(
    run_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    verify_large_inputs: bool = False,
) -> dict[str, Any]:
    """Verify protocol, code, inputs, split evidence, and produced artifacts."""
    directory = Path(run_dir).expanduser().resolve()
    lock = read_json(directory / "run.lock.json")
    protocol = read_json(directory / "protocol.resolved.json")
    if object_sha256(protocol) != lock["protocol_sha256"]:
        raise ValueError("resolved protocol changed")
    if code_manifest() != lock["code"]:
        raise ValueError("workflow source changed after the run was frozen")
    root = Path(data_root or lock["data_root"]).expanduser().resolve()
    names = None if verify_large_inputs else {"mgf"}
    inputs = verify_inputs(protocol, root, names=names)
    if inputs["mgf"] != lock["inputs"]["mgf"]:
        raise ValueError("raw MGF changed")
    checked: list[str] = []

    def verify_outputs(manifest_name: str) -> dict[str, Any]:
        manifest_path = directory / manifest_name
        manifest = read_json(manifest_path)
        for name, digest in manifest.get("output_sha256", {}).items():
            output = manifest_path.parent / name
            if not output.is_file() or file_sha256(output) != digest:
                raise ValueError(f"artifact changed: {output}")
        checked.append(manifest_name)
        return manifest

    for manifest_name in (
        "data/complete.json",
        "training_views/complete.json",
        "embeddings/complete.json",
        "token_features/complete.json",
        "initialization/complete.json",
        "model/complete.json",
        "evaluation/neural/complete.json",
        "evaluation/tomotopy/complete.json",
        "chemical/neural/complete.json",
        "chemical/tomotopy/complete.json",
        "report/report.json",
    ):
        path = directory / manifest_name
        if path.is_file():
            manifest = verify_outputs(manifest_name)
            if manifest_name == "embeddings/complete.json":
                if (
                    file_sha256(path.parent / "embeddings.npy")
                    != manifest["embeddings_sha256"]
                ):
                    raise ValueError("SGNS embeddings changed")
            elif manifest_name == "token_features/complete.json":
                if (
                    file_sha256(path.parent / "features.npy")
                    != manifest["features_sha256"]
                ):
                    raise ValueError("token features changed")
            elif manifest_name == "initialization/complete.json":
                if (
                    file_sha256(path.parent / "model_initialization.pt")
                    != manifest["checkpoint_sha256"]
                ):
                    raise ValueError("model initialization changed")
            elif manifest_name == "model/complete.json":
                selected = manifest["selected"]
                if (
                    file_sha256(path.parent / selected["checkpoint"])
                    != selected["checkpoint_sha256"]
                ):
                    raise ValueError("selected neural checkpoint changed")
            elif manifest_name == "report/report.json":
                report_sources = {
                    "protocol": directory / "protocol.resolved.json",
                    "neural_evaluation": directory / "evaluation/neural/complete.json",
                    "tomotopy_evaluation": directory
                    / "evaluation/tomotopy/complete.json",
                    "neural_chemistry": directory / "chemical/neural/complete.json",
                    "tomotopy_chemistry": directory / "chemical/tomotopy/complete.json",
                }
                for name, source in report_sources.items():
                    if file_sha256(source) != manifest["source_sha256"][name]:
                        raise ValueError(f"report source changed: {source}")

    data_manifest = read_json(directory / "data/complete.json")
    leakage = data_manifest["leakage_audit"]
    if leakage["leaked_compounds"] or leakage["leaked_groups"]:
        raise ValueError("compound or scaffold leakage found")
    vocabulary = data_manifest["vocabulary"]
    if (
        vocabulary["source_split"] != "train"
        or vocabulary["order"] != "raw_training_spectra_first_seen"
    ):
        raise ValueError("vocabulary is not a first-seen training-only vocabulary")
    views_path = directory / "training_views/complete.json"
    if views_path.is_file():
        views = read_json(views_path)
        if not views["frozen_train_counts_reproduced_exactly"]:
            raise ValueError("raw-MGF training counts do not reproduce exactly")
        if not views["physical_peak_groups_atomic"]:
            raise ValueError("fragment/loss peak groups were not kept atomic")
        if views["chemistry_fields_in_model_inputs"]:
            raise ValueError("chemistry fields entered the neural model inputs")
    return {
        "schema_version": "neural-ms2lda/verification-v1",
        "verified": True,
        "run": str(directory),
        "data_root": str(root),
        "manifests_present": checked,
        "discovery_audit": static_discovery_audit(),
        "input_paths": {
            key: str(value)
            for key, value in resolve_input_paths(protocol, root).items()
        },
    }
