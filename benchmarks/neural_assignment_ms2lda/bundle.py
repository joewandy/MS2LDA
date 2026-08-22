"""Hash-verified portable neural MS2LDA model bundles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .utils import file_sha256, read_json, write_json

BUNDLE_FILES = (
    "model.pt",
    "vocabulary.json",
    "protocol.json",
    "evaluation.json",
    "chemistry.json",
    "provenance.json",
)


def _portable_provenance(run: Path, selected: dict[str, Any]) -> dict[str, Any]:
    """Retain scientific provenance without local paths or user metadata."""
    development_path = run / "development.lock.json"
    if development_path.is_file():
        development = read_json(development_path)
        source = Path(development["source_run"])
        visited = {run.resolve()}
        while not (source / "run.lock.json").is_file():
            resolved = source.resolve()
            if resolved in visited:
                raise ValueError("development provenance contains a cycle")
            visited.add(resolved)
            source = Path(read_json(source / "development.lock.json")["source_run"])
        lock = read_json(source / "run.lock.json")
        protocol_sha256 = development["protocol_sha256"]
        source_sha256 = {"development_code_manifest": development["code_sha256"]}
        discovery_audit = {
            **lock.get("discovery_audit", {}),
            "development_hypothesis": development["hypothesis"],
            "frozen_source_artifact_sha256": development["source_artifact_sha256"],
        }
    else:
        lock = read_json(run / "run.lock.json")
        protocol_sha256 = lock["protocol_sha256"]
        source_sha256 = lock["code"]
        discovery_audit = lock.get("discovery_audit", {})
    inputs = {
        name: {
            key: details[key]
            for key in ("bytes", "sha256", "hash_verified")
            if key in details
        }
        for name, details in lock["inputs"].items()
    }
    environment = lock.get("environment", {})
    return {
        "schema_version": "neural-ms2lda/portable-provenance-v1",
        "protocol_sha256": protocol_sha256,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "inputs": inputs,
        "run_source_sha256": source_sha256,
        "packaging_source_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": str(environment.get("python", "")).split(" | ", 1)[0],
            "machine": environment.get("machine"),
            "packages": environment.get("packages", {}),
        },
        "discovery_audit": discovery_audit,
    }


def package_bundle(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Package the selected weights and sufficient beta-derivation metadata."""
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = read_json(run / "model/selected.json")
    checkpoint = run / "model" / selected["checkpoint"]
    protocol_path = run / "protocol.resolved.json"
    shutil.copy2(checkpoint, output / "model.pt")
    shutil.copy2(run / "data/vocabulary.json", output / "vocabulary.json")
    shutil.copy2(protocol_path, output / "protocol.json")
    shutil.copy2(run / "evaluation/neural/complete.json", output / "evaluation.json")
    shutil.copy2(run / "chemical/neural/complete.json", output / "chemistry.json")
    write_json(output / "provenance.json", _portable_provenance(run, selected))
    model_config = read_json(protocol_path)["model"]
    balance = float(model_config.get("token_type_balance", 0.0))
    normalized_type_evidence = bool(
        model_config.get("normalize_token_type_evidence", False)
    )
    unbalanced_derivation = (
        "softmax(2 * normalized_topics @ normalized_projected_tokens.T "
        "/ beta_temperature)"
    )
    if not balance:
        beta_derivation = unbalanced_derivation
    elif normalized_type_evidence:
        beta_derivation = (
            f"mean_type_evidence_with_{balance:g}_pull_to_equal_fragment_loss_mass"
        )
    else:
        beta_derivation = (
            f"summed_type_evidence_with_{balance:g}_pull_to_equal_fragment_loss_mass"
        )
    manifest = {
        "schema_version": "neural-ms2lda/model-bundle-v1",
        "selected_epoch": int(selected["epoch"]),
        "beta_derivation": beta_derivation,
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


def load_bundle(
    bundle_dir: str | Path,
) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    """Verify and load a portable bundle, including derived topic probabilities."""
    directory = Path(bundle_dir).expanduser().resolve()
    manifest = read_json(directory / "manifest.json")
    for name, details in manifest["files"].items():
        path = directory / name
        if (
            path.stat().st_size != int(details["bytes"])
            or file_sha256(path) != details["sha256"]
        ):
            raise ValueError(f"bundle file changed: {name}")
    protocol = read_json(directory / "protocol.json")
    checkpoint = torch.load(
        directory / "model.pt", map_location="cpu", weights_only=False
    )
    features = checkpoint["model"]["token_features"].detach().clone()
    from .model import NeuralAssignmentMS2LDA

    config = protocol["model"]
    topic_indices = torch.arange(int(config["num_topics"]), dtype=torch.int64)
    model = NeuralAssignmentMS2LDA(
        features,
        num_topics=int(config["num_topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        router_hidden_dimensions=int(config["router_hidden_dimensions"]),
        beta_temperature=float(config["beta_temperature"]),
        token_type_balance=float(config.get("token_type_balance", 0.0)),
        normalize_token_type_evidence=bool(
            config.get("normalize_token_type_evidence", False)
        ),
        document_mixture_weight=float(config.get("document_mixture_weight", 0.0)),
        document_topic_prior_weight=float(
            config.get(
                "document_topic_prior_weight",
                protocol.get("hierarchical_routing", {}).get("weight", 0.0),
            )
        ),
        topic_initial_indices=topic_indices,
        seed=int(protocol["seed"]) + int(config["num_topics"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    vocabulary = list(map(str, read_json(directory / "vocabulary.json")["vocabulary"]))
    return model, vocabulary, manifest
