"""Configuration, trained-model I/O, and canonical paper results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import NeuralMS2LDA
from .utils import atomic_torch_save, read_json, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"


def load_protocol() -> dict[str, Any]:
    """Load the only supported study configuration."""
    return read_json(PROTOCOL_PATH)


def initialize_run(run_dir: str | Path, *, data_root: str | Path) -> dict[str, Any]:
    """Create a run directory bound to the current protocol and data location.

    The resolved JSON is deliberately human-readable, and an existing run
    directory cannot silently mix configurations.
    """
    directory = Path(run_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    resolved = directory / "protocol.json"
    if resolved.is_file() and read_json(resolved) != protocol:
        raise ValueError("run protocol differs from the current study")
    write_json(resolved, protocol)
    data = Path(data_root).expanduser().resolve()
    mgf = data / protocol["input_files"]["mgf"]
    if not mgf.is_file():
        raise FileNotFoundError(f"required MGF is missing: {mgf}")
    binding = directory / "data_root.txt"
    if binding.is_file():
        if binding.read_text(encoding="utf-8").strip() != str(data):
            raise ValueError("run data root differs from the original location")
    else:
        stage_outputs = [path for path in directory.iterdir() if path != resolved]
        if stage_outputs:
            raise ValueError("existing run lacks a data-root binding; start a new run")
        binding.write_text(f"{data}\n", encoding="utf-8")
    return protocol


def save_trained_model(
    directory: str | Path,
    model: NeuralMS2LDA,
    vocabulary: list[str],
    *,
    routing_temperature: float,
) -> None:
    """Write the three files needed for one-pass inference.

    Tensor shapes encode the architecture dimensions. The tiny JSON file stores
    only the two non-tensor temperatures that cannot be inferred from state.
    """
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(output / "weights.pt", model.state_dict())
    write_json(output / "vocabulary.json", list(vocabulary))
    write_json(
        output / "model.json",
        {
            "beta_temperature": model.beta_temperature,
            "routing_temperature": float(routing_temperature),
        },
    )


def load_trained_model(
    directory: str | Path,
) -> tuple[NeuralMS2LDA, list[str], float]:
    """Load the current three-file model artifact."""
    root = Path(directory).expanduser().resolve()
    state = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
    settings = read_json(root / "model.json")
    vocabulary = list(map(str, read_json(root / "vocabulary.json")))
    features = state["token_features"]
    topics, projection_dimensions = state["topic_prototypes"].shape
    if len(vocabulary) != features.shape[0]:
        raise ValueError("vocabulary and token features differ in length")
    model = NeuralMS2LDA(
        features,
        num_topics=int(topics),
        projection_dimensions=int(projection_dimensions),
        beta_temperature=float(settings["beta_temperature"]),
        topic_initial_indices=torch.arange(int(topics), dtype=torch.int64),
        seed=0,
    )
    model.load_state_dict(state)
    model.eval()
    return model, vocabulary, float(settings["routing_temperature"])


def _chemistry_summary(chemistry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the probability-thresholded SOS quantities used in the paper."""
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
    method: str,
    validation: dict[str, Any],
    test: dict[str, Any],
    fitting_seconds: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "fitting_seconds": float(fitting_seconds),
        "validation": _chemistry_summary(validation),
        "test": _chemistry_summary(test),
    }


def build_results(run_dir: str | Path) -> dict[str, Any]:
    """Build the single paper-facing ``results.json`` from completed stages."""
    run = Path(run_dir).expanduser().resolve()
    protocol = read_json(run / "protocol.json")

    def stage(path: str) -> dict[str, Any]:
        return read_json(run / path / "complete.json")

    neural_training = stage("trained_model")
    tomotopy_training = stage("tomotopy")
    data = stage("data")
    evaluations = {
        (method, split): stage(f"{group}/{method}")
        for method in ("neural", "tomotopy")
        for split, group in (
            ("validation", "validation_evaluation"),
            ("test", "evaluation"),
        )
    }
    chemistry = {
        (method, split): stage(
            f"{'validation_chemical' if split == 'validation' else 'chemical'}/{method}"
        )
        for method in ("neural", "tomotopy")
        for split in ("validation", "test")
    }
    methods = [
        _method_result(
            "neural",
            chemistry[("neural", "validation")],
            chemistry[("neural", "test")],
            neural_training["fitting_seconds"],
        ),
        _method_result(
            "tomotopy",
            chemistry[("tomotopy", "validation")],
            chemistry[("tomotopy", "test")],
            tomotopy_training["training_seconds_total"],
        ),
    ]
    neural_test = evaluations[("neural", "test")]["metrics"]
    tomotopy_test = evaluations[("tomotopy", "test")]["metrics"]
    neural_warm = neural_test["warm_in_memory_batch_inference"]
    tomotopy_warm = tomotopy_test["warm_in_memory_batch_inference"]
    result = {
        "study": {
            "seed": int(protocol["seed"]),
            "topics": int(protocol["model"]["num_topics"]),
            "cpu_threads": int(protocol["cpu_threads"]),
            "final_epoch": int(protocol["optimization"]["maximum_epochs"]),
            "association_probability_threshold": float(
                protocol["chemistry"]["membership_threshold"]
            ),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
            "source_spectra": int(protocol["preprocessing"]["expected_spectra"]),
            "retained_spectra": int(data["parsing"]["retained_spectra"]),
            "split_spectra": {
                name: int(count)
                for name, count in data["split"]["spectrum_counts"].items()
            },
            "vocabulary_size": int(data["vocabulary"]["vocabulary_size"]),
            "tomotopy_training_iterations": int(
                tomotopy_training["training_iterations"]
            ),
        },
        "methods": methods,
        "secondary": {
            "completion_nll_per_token": {
                method: {
                    split: float(
                        evaluations[(method, split)]["metrics"][
                            f"{split}_document_completion"
                        ]["nll_per_token"]
                    )
                    for split in ("validation", "test")
                }
                for method in ("neural", "tomotopy")
            },
            "neural_recycled_topics_during_training": 0,
            "neural_test_corpus_active_topics": int(
                neural_test["active_topics"]["corpus_active_topics"]
            ),
            "neural_test_median_effective_topics_per_spectrum": float(
                neural_test["full_spectrum_mixture"]["effective_topic_count_median"]
            ),
            "warm_in_memory_batch_inference": {
                "batch_size": int(neural_warm["documents"]),
                "neural_spectra_per_second": float(
                    neural_warm["median_spectra_per_second"]
                ),
                "tomotopy_spectra_per_second": float(
                    tomotopy_warm["median_spectra_per_second"]
                ),
            },
        },
    }
    write_json(run / "results.json", result)
    return result
