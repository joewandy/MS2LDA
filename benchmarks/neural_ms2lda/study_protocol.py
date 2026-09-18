"""Load and bind the single Contextual Sparse ETM study protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"
METHOD = "contextual_sparse_etm"
MODEL_DISPLAY_NAME = "Contextual Sparse ETM"
NEURAL_DEVICE = "cuda"
TRAINING_SEEDS = (7043, 23, 37)
SYNTHETIC_SEEDS = (11, 23, 37)
SYNTHETIC_FORMULATIONS = (
    "balanced_softmax",
    "balanced_entmax",
    "contextual_softmax",
    "contextual_entmax",
)
SYNTHETIC_ARTIFACT_LABELS = {
    "balanced_softmax": "balanced_etm_softmax_raw_counts",
    "balanced_entmax": "balanced_etm_entmax15_raw_counts",
    "contextual_softmax": "contextual_etm_softmax_raw_counts",
    "contextual_entmax": "contextual_sparse_etm_entmax15_raw_counts",
}
SYNTHETIC_DISPLAY_LABELS = {
    "balanced_etm_softmax_raw_counts": "channel-balanced ETM + softmax",
    "balanced_etm_entmax15_raw_counts": "channel-balanced ETM + 1.5-entmax",
    "contextual_etm_softmax_raw_counts": (
        "channel-balanced ETM + contextual evidence + softmax"
    ),
    "contextual_sparse_etm_entmax15_raw_counts": MODEL_DISPLAY_NAME,
}
FINAL_SYNTHETIC_LABEL = SYNTHETIC_DISPLAY_LABELS[
    SYNTHETIC_ARTIFACT_LABELS["contextual_entmax"]
]
TRAINING_ACCESS_AUDIT_FILENAME = "training_access_audit.json"
VALIDATION_ACCESS_AUDIT_FILENAME = "validation_access_audit.json"


def load_protocol() -> dict[str, Any]:
    """Load the only supported study configuration."""
    return read_json(PROTOCOL_PATH)


def initialize_run(run_dir: str | Path, *, data_root: str | Path) -> dict[str, Any]:
    """Create a run directory bound to one protocol and one data location."""
    directory = Path(run_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    resolved_protocol = directory / "protocol.json"
    if resolved_protocol.is_file() and read_json(resolved_protocol) != protocol:
        raise ValueError("run protocol differs from the current study")
    write_json(resolved_protocol, protocol)

    data = Path(data_root).expanduser().resolve()
    mgf = data / protocol["input_files"]["mgf"]
    if not mgf.is_file():
        raise FileNotFoundError(f"required MGF is missing: {mgf}")
    binding = directory / "data_root.txt"
    if binding.is_file():
        if binding.read_text(encoding="utf-8").strip() != str(data):
            raise ValueError("run data root differs from the original location")
    else:
        stage_outputs = [
            path for path in directory.iterdir() if path != resolved_protocol
        ]
        if stage_outputs:
            raise ValueError("existing run lacks a data-root binding; start a new run")
        binding.write_text(f"{data}\n", encoding="utf-8")
    return protocol
