"""Hash-verified portable neural MS2LDA model bundles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .model import NeuralAssignmentMS2LDA
from .utils import file_sha256, object_sha256, read_json, write_json

BUNDLE_FILES = (
    "model.pt",
    "vocabulary.json",
    "protocol.json",
    "evaluation.json",
    "chemistry.json",
    "provenance.json",
)


def _beta_derivation(model_config: dict[str, Any]) -> str:
    """Describe the exact deterministic map from prototypes to ``beta``."""
    balance = float(model_config["token_type_balance"])
    if not balance:
        return (
            "softmax(2 * normalized_topics @ normalized_projected_tokens.T "
            "/ beta_temperature)"
        )
    return f"mean_type_evidence_with_{balance:g}_pull_to_equal_fragment_loss_mass"


def _portable_provenance(run: Path, selected: dict[str, Any]) -> dict[str, Any]:
    """Retain scientific provenance without local paths or user metadata."""
    lock = read_json(run / "run.lock.json")
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
        "protocol_sha256": lock["protocol_sha256"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "inputs": inputs,
        "run_source_sha256": lock["code"],
        "packaging_source_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": str(environment.get("python", "")).split(" | ", 1)[0],
            "machine": environment.get("machine"),
            "packages": environment.get("packages", {}),
        },
        "discovery_audit": lock["discovery_audit"],
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
    manifest = {
        "schema_version": "neural-ms2lda/model-bundle-v1",
        "selected_epoch": int(selected["epoch"]),
        "beta_derivation": _beta_derivation(model_config),
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


def _verified_manifest(directory: Path) -> dict[str, Any]:
    """Verify the fixed bundle inventory and every declared file digest."""
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "neural-ms2lda/model-bundle-v1":
        raise ValueError("unexpected model bundle schema")
    if set(manifest.get("files", {})) != set(BUNDLE_FILES):
        raise ValueError("model bundle manifest has an unexpected file set")
    for name, details in manifest["files"].items():
        path = directory / name
        if (
            path.stat().st_size != int(details["bytes"])
            or file_sha256(path) != details["sha256"]
        ):
            raise ValueError(f"bundle file changed: {name}")
    return manifest


def _verified_protocol(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Cross-check architecture metadata against portable provenance."""
    protocol = read_json(directory / "protocol.json")
    if protocol.get("schema_version") != "neural-ms2lda/protocol-v1":
        raise ValueError("unexpected bundled protocol schema")
    if "normalize_token_type_evidence" in protocol["model"]:
        raise ValueError("bundle contains the removed decoder experiment switch")
    if manifest["beta_derivation"] != _beta_derivation(protocol["model"]):
        raise ValueError("bundle beta derivation differs from its protocol")
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


def _verified_checkpoint(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load weights only after their file, schema, and epoch have been verified."""
    checkpoint = torch.load(
        directory / "model.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != "neural-ms2lda/selected-model-v1":
        raise ValueError("unexpected bundled checkpoint schema")
    if int(checkpoint["epoch"]) != int(manifest["selected_epoch"]):
        raise ValueError("bundled checkpoint epoch differs from its manifest")
    return checkpoint


def _model_from_checkpoint(
    protocol: dict[str, Any], checkpoint: dict[str, Any]
) -> NeuralAssignmentMS2LDA:
    """Reconstruct module shape from protocol, then restore the selected state."""
    features = checkpoint["model"]["token_features"].detach().clone()
    config = protocol["model"]
    topic_indices = torch.arange(int(config["num_topics"]), dtype=torch.int64)
    model = NeuralAssignmentMS2LDA(
        features,
        num_topics=int(config["num_topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        router_hidden_dimensions=int(config["router_hidden_dimensions"]),
        beta_temperature=float(config["beta_temperature"]),
        token_type_balance=float(config["token_type_balance"]),
        document_mixture_weight=float(config["document_mixture_weight"]),
        document_topic_prior_weight=float(config["document_topic_prior_weight"]),
        topic_initial_indices=topic_indices,
        seed=int(protocol["seed"]) + int(config["num_topics"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_bundle(
    bundle_dir: str | Path,
) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    """Verify and load a portable bundle, including derived topic probabilities."""
    directory = Path(bundle_dir).expanduser().resolve()
    manifest = _verified_manifest(directory)
    protocol = _verified_protocol(directory, manifest)
    checkpoint = _verified_checkpoint(directory, manifest)
    model = _model_from_checkpoint(protocol, checkpoint)
    vocabulary = list(map(str, read_json(directory / "vocabulary.json")["vocabulary"]))
    if len(vocabulary) != model.vocabulary_size:
        raise ValueError("bundled vocabulary does not match the model")
    return model, vocabulary, manifest
