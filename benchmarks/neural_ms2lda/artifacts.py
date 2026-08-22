"""Protocol, provenance, and artifact verification for neural MS2LDA."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from .inputs import resolve_input_paths, verify_inputs
from .model import (
    DOCUMENT_MIXTURE_EXPONENT,
    TOKEN_TYPE_BALANCE,
    TOPICS_PER_TOKEN,
    NeuralMS2LDA,
)
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
BUNDLE_FILES = ("model.pt", "vocabulary.json", "protocol.json", "provenance.json")
REQUIRED_MODEL_CONSTANTS = {
    "top_k": TOPICS_PER_TOKEN,
    "token_type_balance": TOKEN_TYPE_BALANCE,
    "document_mixture_weight": DOCUMENT_MIXTURE_EXPONENT,
}
MODEL_PROTOCOL_FIELDS = {
    "num_topics",
    "projection_dimensions",
    "router_hidden_dimensions",
    "beta_temperature",
    "sinkhorn_epsilon",
    "sinkhorn_iterations",
    "gradient_clip_norm",
    *REQUIRED_MODEL_CONSTANTS,
}

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


def _validate_model_protocol(protocol: dict[str, Any]) -> None:
    """Reject fields or constants outside the single supported model."""
    model = protocol["model"]
    if set(model) != MODEL_PROTOCOL_FIELDS or any(
        model.get(name) != value for name, value in REQUIRED_MODEL_CONSTANTS.items()
    ):
        raise ValueError("protocol differs from the supported neural equations")


def load_protocol() -> dict[str, Any]:
    """Load and validate the sole reproducibility protocol."""
    protocol = read_json(PROTOCOL_PATH)
    _validate_model_protocol(protocol)
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
    load_bundle(bundle_path.parent)
    results = read_json(results_path)
    provenance = results["provenance"]
    if provenance["model_bundle_manifest_sha256"] != file_sha256(bundle_path):
        raise ValueError("results name a different model bundle")
    sources = _result_source_paths(directory)
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


def _portable_provenance(run: Path, selected: dict[str, Any]) -> dict[str, Any]:
    """Retain only provenance needed to identify and reproduce a checkpoint."""
    lock = read_json(run / "run.lock.json")
    environment = lock.get("environment", {})
    return {
        "protocol_sha256": lock["protocol_sha256"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "inputs": {
            name: {
                key: details[key]
                for key in ("bytes", "sha256", "hash_verified")
                if key in details
            }
            for name, details in lock["inputs"].items()
        },
        "source_sha256": lock["code"],
        "environment": {
            "python": str(environment.get("python", "")).split(" | ", 1)[0],
            "machine": environment.get("machine"),
            "packages": environment.get("packages", {}),
        },
    }


def package_bundle(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write the minimal portable bundle for the selected final checkpoint."""
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = read_json(run / "model/selected.json")
    shutil.copy2(run / "model" / selected["checkpoint"], output / "model.pt")
    shutil.copy2(run / "data/vocabulary.json", output / "vocabulary.json")
    shutil.copy2(run / "protocol.resolved.json", output / "protocol.json")
    write_json(output / "provenance.json", _portable_provenance(run, selected))
    manifest = {
        "selected_epoch": int(selected["epoch"]),
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": file_sha256(output / name),
            }
            for name in BUNDLE_FILES
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _verified_bundle_manifest(directory: Path) -> dict[str, Any]:
    """Verify the minimal bundle inventory and every declared digest."""
    manifest = read_json(directory / "manifest.json")
    if set(manifest.get("files", {})) != set(BUNDLE_FILES):
        raise ValueError("model bundle has an unexpected file set")
    present = {path.name for path in directory.iterdir() if path.is_file()}
    if present != {*BUNDLE_FILES, "manifest.json"}:
        raise ValueError("model bundle directory contains undeclared files")
    for name, details in manifest["files"].items():
        path = directory / name
        if (
            path.stat().st_size != int(details["bytes"])
            or file_sha256(path) != details["sha256"]
        ):
            raise ValueError(f"bundle file changed: {name}")
    return manifest


def _verified_bundle_protocol(
    directory: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Cross-check the sole architecture protocol against provenance."""
    protocol = read_json(directory / "protocol.json")
    _validate_model_protocol(protocol)
    if int(manifest["selected_epoch"]) != int(
        protocol["optimization"]["maximum_epochs"]
    ):
        raise ValueError("bundle checkpoint is not the fixed final epoch")
    provenance = read_json(directory / "provenance.json")
    if provenance["protocol_sha256"] != object_sha256(protocol):
        raise ValueError("bundle protocol differs from recorded provenance")
    if (
        provenance["selected_checkpoint_sha256"]
        != manifest["files"]["model.pt"]["sha256"]
    ):
        raise ValueError("bundle checkpoint differs from recorded provenance")
    return protocol


def _model_from_checkpoint(
    protocol: dict[str, Any], checkpoint: dict[str, Any]
) -> NeuralMS2LDA:
    """Reconstruct the module shape, then restore the selected state."""
    features = checkpoint["model"]["token_features"].detach().clone()
    config = protocol["model"]
    model = NeuralMS2LDA(
        features,
        num_topics=int(config["num_topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        router_hidden_dimensions=int(config["router_hidden_dimensions"]),
        beta_temperature=float(config["beta_temperature"]),
        topic_initial_indices=torch.arange(
            int(config["num_topics"]), dtype=torch.int64
        ),
        seed=int(protocol["seed"]) + int(config["num_topics"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_bundle(
    bundle_dir: str | Path,
) -> tuple[NeuralMS2LDA, list[str], dict[str, Any]]:
    """Verify and load a portable checkpoint and its vocabulary."""
    directory = Path(bundle_dir).expanduser().resolve()
    manifest = _verified_bundle_manifest(directory)
    protocol = _verified_bundle_protocol(directory, manifest)
    checkpoint = torch.load(
        directory / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["epoch"]) != int(manifest["selected_epoch"]):
        raise ValueError("bundled checkpoint epoch differs from its manifest")
    model = _model_from_checkpoint(protocol, checkpoint)
    vocabulary = list(map(str, read_json(directory / "vocabulary.json")["vocabulary"]))
    if len(vocabulary) != model.vocabulary_size:
        raise ValueError("bundled vocabulary does not match the model")
    return model, vocabulary, manifest


def _chemistry_summary(chemistry: dict[str, Any]) -> dict[str, Any]:
    """Keep only probability-thresholded SOS quantities used in the paper."""
    scored = chemistry["high_confidence_chemistry"]
    bands = scored["sos_bands"]
    return {
        "optimized_motifs": int(
            round(float(chemistry["annotation_coverage"]) * chemistry["topics"])
        ),
        "annotation_coverage": float(chemistry["annotation_coverage"]),
        "high_confidence_evaluable_motifs": int(scored["eligible_topics"]),
        "useful_high_confidence_motifs": int(
            bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
        ),
        "sos_bands": bands,
        "mean_sos": float(scored["mean_sos"]),
        "median_sos": float(scored["median_sos"]),
    }


def _method_result(
    *,
    method: str,
    validation_chemistry: dict[str, Any],
    test_chemistry: dict[str, Any],
    fitting_seconds: float,
    fitting_workers: int,
) -> dict[str, Any]:
    """Return one paper-facing method comparison row."""
    return {
        "method": method,
        "fitting_seconds": float(fitting_seconds),
        "fitting_workers": int(fitting_workers),
        "validation": _chemistry_summary(validation_chemistry),
        "test": _chemistry_summary(test_chemistry),
    }


def _result_source_paths(directory: Path) -> dict[str, Path]:
    """Name the exact stage manifests consumed by canonical results."""
    return {
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


def build_results(run_dir: str | Path) -> dict[str, Any]:  # noqa: PLR0915
    """Write the sole comparison result from verified stage outputs."""
    directory = Path(run_dir).expanduser().resolve()
    protocol = read_json(directory / "protocol.resolved.json")
    neural_training = read_json(directory / "model/complete.json")
    tomotopy_training = read_json(directory / "tomotopy/complete.json")
    paths = _result_source_paths(directory)
    evidence = {name: read_json(path) for name, path in paths.items()}
    neural_test = evidence["neural_test_evaluation"]
    tomotopy_test = evidence["tomotopy_test_evaluation"]
    neural_warm = neural_test["metrics"]["warm_in_memory_batch_inference"]
    tomotopy_warm = tomotopy_test["metrics"]["warm_in_memory_batch_inference"]
    methods = [
        _method_result(
            method="neural",
            validation_chemistry=evidence["neural_validation_chemistry"],
            test_chemistry=evidence["neural_test_chemistry"],
            fitting_seconds=float(neural_training["elapsed_seconds"]),
            fitting_workers=int(protocol["cpu_threads"]),
        ),
        _method_result(
            method="tomotopy",
            validation_chemistry=evidence["tomotopy_validation_chemistry"],
            test_chemistry=evidence["tomotopy_test_chemistry"],
            fitting_seconds=float(tomotopy_training["training_seconds_total"]),
            fitting_workers=int(tomotopy_training["training_workers"]),
        ),
    ]
    nll = {
        "neural": {
            "validation": float(
                evidence["neural_validation_evaluation"]["metrics"][
                    "validation_document_completion"
                ]["nll_per_token"]
            ),
            "test": float(
                neural_test["metrics"]["test_document_completion"]["nll_per_token"]
            ),
        },
        "tomotopy": {
            "validation": float(
                evidence["tomotopy_validation_evaluation"]["metrics"][
                    "validation_document_completion"
                ]["nll_per_token"]
            ),
            "test": float(
                tomotopy_test["metrics"]["test_document_completion"]["nll_per_token"]
            ),
        },
    }
    result = {
        "comparison_contract": {
            "association_probability_threshold": float(
                protocol["chemistry"]["membership_threshold"]
            ),
            "cpu_threads": int(protocol["cpu_threads"]),
            "neural_selected_epoch": int(neural_test["selected_epoch"]),
            "seed": int(protocol["seed"]),
            "selection_split": "validation",
            "topics": int(protocol["model"]["num_topics"]),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
        },
        "methods": methods,
        "provenance": {
            "model_bundle_manifest_sha256": file_sha256(
                directory / "model_bundle/manifest.json"
            ),
            "recorded_source_manifest_sha256": {
                name: file_sha256(path) for name, path in paths.items()
            },
            "selected_checkpoint_sha256": neural_training["selected"][
                "checkpoint_sha256"
            ],
            "test_opened_after_validation_selection": True,
        },
        "secondary_diagnostics": {
            "completion_nll_per_token": nll,
            "neural_recycled_topics_during_training": int(
                neural_training["recycle_count_total"]
            ),
            "neural_test_corpus_active_topics": int(
                neural_test["metrics"]["active_topics"]["corpus_active_topics"]
            ),
            "neural_test_median_effective_topics_per_spectrum": float(
                neural_test["metrics"]["full_spectrum_mixture"][
                    "effective_topic_count_median"
                ]
            ),
        },
        "secondary_warm_in_memory_batch_inference": {
            "batch_size": int(neural_warm["documents"]),
            "cpu_threads": int(neural_warm["cpu_threads"]),
            "neural_routing_passes": 1,
            "neural_spectra_per_second": float(
                neural_warm["median_spectra_per_second"]
            ),
            "speedup_over_tomotopy": float(
                neural_warm["median_spectra_per_second"]
                / tomotopy_warm["median_spectra_per_second"]
            ),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
            "tomotopy_spectra_per_second": float(
                tomotopy_warm["median_spectra_per_second"]
            ),
        },
    }
    write_json(directory / "results.json", result)
    return result
