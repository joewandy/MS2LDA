"""Minimal, hash-verified neural MS2LDA model bundles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .model import (
    DOCUMENT_MIXTURE_EXPONENT,
    TOKEN_TYPE_BALANCE,
    TOPICS_PER_TOKEN,
    NeuralMS2LDA,
)
from .utils import file_sha256, object_sha256, read_json, write_json

BUNDLE_FILES = ("model.pt", "vocabulary.json", "protocol.json", "provenance.json")


def _portable_provenance(run: Path, selected: dict[str, Any]) -> dict[str, Any]:
    """Retain only provenance needed to identify and reproduce the checkpoint."""
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
    """Copy the selected weights and their complete reconstruction inputs."""
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


def _verified_manifest(directory: Path) -> dict[str, Any]:
    """Verify the minimal inventory and every declared file digest."""
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


def _verified_protocol(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Cross-check the sole architecture protocol against provenance."""
    protocol = read_json(directory / "protocol.json")
    required = {
        "top_k": TOPICS_PER_TOKEN,
        "token_type_balance": TOKEN_TYPE_BALANCE,
        "document_mixture_weight": DOCUMENT_MIXTURE_EXPONENT,
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
    if set(protocol["model"]) != allowed or any(
        protocol["model"].get(name) != value for name, value in required.items()
    ):
        raise ValueError("bundle protocol differs from the supported equations")
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
    manifest = _verified_manifest(directory)
    protocol = _verified_protocol(directory, manifest)
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
