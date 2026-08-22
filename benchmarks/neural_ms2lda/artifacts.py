"""Protocol, provenance, and artifact verification for neural MS2LDA."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .inputs import resolve_input_paths, verify_inputs
from .utils import (
    file_sha256,
    object_sha256,
    read_json,
    verify_output_hashes,
    write_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"

MANIFEST_PATHS = (
    "data/complete.json",
    "training_views/complete.json",
    "embeddings/complete.json",
    "token_features/complete.json",
    "initialization/complete.json",
    "cooccurrence_graph/complete.json",
    "model/complete.json",
    "tomotopy/complete.json",
    "validation_evaluation/neural/complete.json",
    "validation_evaluation/tomotopy/complete.json",
    "validation_chemical/neural/complete.json",
    "validation_chemical/tomotopy/complete.json",
    "mag/annotations/neural/complete.json",
    "mag/annotations/tomotopy/complete.json",
    "evaluation/neural/complete.json",
    "evaluation/tomotopy/complete.json",
    "chemical/neural/complete.json",
    "chemical/tomotopy/complete.json",
)


def load_protocol() -> dict[str, Any]:
    """Load the sole protocol used by the reproducibility workflow."""
    protocol = read_json(PROTOCOL_PATH)
    model = protocol["model"]
    required = {
        "top_k": 2,
        "token_type_balance": 0.25,
        "document_mixture_weight": 0.75,
    }
    allowed = {
        "num_topics",
        "projection_dimensions",
        "router_hidden_dimensions",
        "beta_temperature",
        "sinkhorn_epsilon",
        "sinkhorn_iterations",
        "gradient_clip_norm",
        *required,
    }
    if set(model) != allowed or any(
        model.get(name) != value for name, value in required.items()
    ):
        raise ValueError("protocol differs from the supported neural equations")
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
    """Hash the package and scripts that define this workflow."""
    sources = [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "results" not in path.parts
        and path.suffix in {".py", ".json", ".md"}
    ]
    sources.extend(
        REPO_ROOT / name
        for name in (
            "scripts/download_msnlib_validation_assets.py",
            "scripts/generate_neural_ms2lda_report.py",
        )
    )
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sorted(sources)
    }


def initialize_run(
    run_dir: str | Path,
    *,
    data_root: str | Path,
) -> dict[str, Any]:
    """Create or verify an immutable, exactly resumable run lock."""
    directory = Path(run_dir).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    lock_path = directory / "run.lock.json"
    expected = {
        "data_root": str(root),
        "protocol_sha256": object_sha256(protocol),
        "inputs": verify_inputs(protocol, root, names={"mgf"}),
        "code": code_manifest(),
        "environment": environment_manifest(),
        "git": _git_state(),
    }
    if lock_path.is_file():
        existing = read_json(lock_path)
        immutable = (
            "data_root",
            "protocol_sha256",
            "inputs",
            "code",
        )
        if any(existing.get(key) != expected.get(key) for key in immutable):
            raise ValueError("run provenance differs from its immutable lock")
        return existing
    write_json(directory / "protocol.resolved.json", protocol)
    write_json(lock_path, expected)
    return expected


def _verify_model_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    graph_manifest_path: Path,
) -> None:
    """Cross-check selected neural weights and their train-only graph source."""
    if manifest["cooccurrence_graph"] != read_json(graph_manifest_path):
        raise ValueError("model co-occurrence graph provenance changed")
    selected = manifest["selected"]
    selected_path = path.parent / "selected.json"
    if not selected_path.is_file() or read_json(selected_path) != selected:
        raise ValueError("selected model manifest changed")
    if (
        file_sha256(path.parent / selected["checkpoint"])
        != selected["checkpoint_sha256"]
    ):
        raise ValueError("selected neural checkpoint changed")
    if selected.get("selection_rule") != "fixed_final_epoch":
        raise ValueError("neural checkpoint was not selected by final epoch")
    if int(selected["epoch"]) != int(protocol["optimization"]["maximum_epochs"]):
        raise ValueError("selected neural checkpoint is not the final epoch")


def _verify_tomotopy_training(
    path: Path, manifest: dict[str, Any], *, cpu_threads: int
) -> None:
    """Verify the comparator binary and the shared fitting-worker contract."""
    if file_sha256(path.parent / "model.bin") != manifest["model_sha256"]:
        raise ValueError("Tomotopy model changed")
    if int(manifest.get("training_workers", 0)) != cpu_threads:
        raise ValueError("Tomotopy training did not use six workers")


def _verify_results(directory: Path) -> None:
    """Verify that canonical results name the exact evaluated artifacts."""
    results_path = directory / "results.json"
    bundle_path = directory / "model_bundle/manifest.json"
    if not results_path.is_file() and not bundle_path.is_file():
        return
    if not results_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("results and model bundle must be produced together")
    from .bundle import load_bundle

    load_bundle(bundle_path.parent)
    results = read_json(results_path)
    provenance = results["provenance"]
    if provenance["model_bundle_manifest_sha256"] != file_sha256(bundle_path):
        raise ValueError("results name a different model bundle")
    sources = {
        "neural_validation_evaluation": directory
        / "validation_evaluation/neural/complete.json",
        "tomotopy_validation_evaluation": directory
        / "validation_evaluation/tomotopy/complete.json",
        "neural_validation_chemistry": directory
        / "validation_chemical/neural/complete.json",
        "tomotopy_validation_chemistry": directory
        / "validation_chemical/tomotopy/complete.json",
        "neural_test_evaluation": directory / "evaluation/neural/complete.json",
        "tomotopy_test_evaluation": directory / "evaluation/tomotopy/complete.json",
        "neural_test_chemistry": directory / "chemical/neural/complete.json",
        "tomotopy_test_chemistry": directory / "chemical/tomotopy/complete.json",
    }
    for name, source in sources.items():
        if file_sha256(source) != provenance["recorded_source_manifest_sha256"][name]:
            raise ValueError(f"results source changed: {source}")


def _verify_manifest_semantics(  # noqa: PLR0913
    manifest_name: str,
    path: Path,
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    graph_manifest_path: Path,
) -> None:
    """Apply the extra semantic checks required by stateful artifacts."""
    if manifest_name == "cooccurrence_graph/complete.json":
        if set(manifest.get("output_sha256", {})) != {"positive_npmi_graph.npz"}:
            raise ValueError("co-occurrence graph manifest is incomplete")
    elif manifest_name == "model/complete.json":
        _verify_model_manifest(
            path,
            manifest,
            protocol=protocol,
            graph_manifest_path=graph_manifest_path,
        )
    elif manifest_name == "tomotopy/complete.json":
        _verify_tomotopy_training(
            path, manifest, cpu_threads=int(protocol["cpu_threads"])
        )
    elif manifest_name == "evaluation/tomotopy/complete.json":
        if int(manifest.get("inference_workers", 0)) != int(protocol["cpu_threads"]):
            raise ValueError("Tomotopy inference did not use six workers")


def _verify_completed_manifests(directory: Path, protocol: dict[str, Any]) -> list[str]:
    """Verify every completed stage that is present in a resumable run."""
    model_manifest_path = directory / "model/complete.json"
    graph_manifest_path = directory / "cooccurrence_graph/complete.json"
    if model_manifest_path.is_file() and not graph_manifest_path.is_file():
        raise FileNotFoundError("trained model requires a co-occurrence graph manifest")

    checked: list[str] = []
    for manifest_name in MANIFEST_PATHS:
        path = directory / manifest_name
        if not path.is_file():
            continue
        manifest = read_json(path)
        if manifest_name != "model/complete.json":
            verify_output_hashes(path.parent, manifest)
        _verify_manifest_semantics(
            manifest_name,
            path,
            manifest,
            protocol=protocol,
            graph_manifest_path=graph_manifest_path,
        )
        checked.append(manifest_name)
    return checked


def _verify_data_contract(directory: Path) -> None:
    """Prove split isolation, vocabulary provenance, and frozen-view fidelity."""
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
    checked = _verify_completed_manifests(directory, protocol)
    _verify_data_contract(directory)
    _verify_results(directory)
    return {
        "verified": True,
        "run": str(directory),
        "data_root": str(root),
        "manifests_present": checked,
        "input_paths": {
            key: str(value)
            for key, value in resolve_input_paths(protocol, root).items()
        },
    }
