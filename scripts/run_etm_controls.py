"""Train and evaluate the two published ETM controls on frozen MSnLib views."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.etm_baselines import (
    CanonicalETM,
    ChannelBalancedETM,
    load_sgns_embeddings,
)
from benchmarks.neural_ms2lda.model_evaluation import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
    completion_metrics,
    mixture_distribution_summary,
    save_validation,
    score_chemical_validation,
    topic_word_diagnostics,
)
from benchmarks.neural_ms2lda.reproducibility import (
    MemoryState,
    configure_deterministic_execution,
    normalize_probability_rows,
    read_json_object,
    resolve_torch_device,
    runtime_memory_metrics,
    sample_runtime_memory,
    write_csv_rows,
)
from benchmarks.neural_ms2lda.study_protocol import TRAINING_ACCESS_AUDIT_FILENAME
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    raw_count_reconstruction_loss,
)
from benchmarks.neural_ms2lda.utils import atomic_torch_save, write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

METHODS = ("etm", "etm_balanced")
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1.2e-6
HIDDEN_WIDTH = 800
EPSILON = 1e-12
ControlModel = CanonicalETM | ChannelBalancedETM


class ControlData(NamedTuple):
    """The train and validation artifacts visible to an ETM control."""

    train: sp.csr_matrix
    observed: sp.csr_matrix
    completion: sp.csr_matrix
    full: sp.csr_matrix
    records: list[dict[str, Any]]
    vocabulary: list[str]
    embeddings: np.ndarray
    protocol: dict[str, Any]
    input_manifest: dict[str, Any]


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before reading a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_control_data(run: Path) -> ControlData:
    """Load only the sealed training and validation view."""
    data = run / "data"
    protocol = read_json_object(run / "protocol.json")
    input_manifest = read_json_object(run / "validation_input_manifest.json")
    if input_manifest.get("candidate_test_artifacts_accessed") is not False:
        raise RuntimeError("validation view does not preserve the test boundary")
    forbidden = sorted(path.name for path in data.glob("test*") if path.is_file())
    if forbidden:
        raise RuntimeError(f"sealed training view exposes test files: {forbidden}")
    vocabulary = load_vocabulary(data)
    embeddings = load_sgns_embeddings(run / "token_features/features.npy")
    train = load_csr(data / "train.npz")
    observed = load_csr(data / "validation_observed.npz")
    completion = load_csr(data / "validation_completion.npz")
    full = load_csr(data / "validation_full.npz")
    records = load_heldout_records(data, "validation")
    if train.shape[1] != len(vocabulary) or embeddings.shape[0] != len(vocabulary):
        raise ValueError("train matrix, vocabulary and SGNS features do not align")
    if observed.shape != completion.shape or observed.shape != full.shape:
        raise ValueError("validation matrices do not have identical shapes")
    if full.shape[0] != len(records):
        raise ValueError("validation records and matrices do not align")
    return ControlData(
        train=train,
        observed=observed,
        completion=completion,
        full=full,
        records=records,
        vocabulary=vocabulary,
        embeddings=embeddings,
        protocol=protocol,
        input_manifest=input_manifest,
    )


def build_control_model(data: ControlData, method: str) -> ControlModel:
    """Construct exactly one of the two predeclared published controls."""
    topics = int(data.protocol["model"]["num_topics"])
    if method == "etm":
        return CanonicalETM(data.embeddings, topics, hidden=HIDDEN_WIDTH)
    if method == "etm_balanced":
        fragment_mask = np.asarray(
            [word.startswith("frag@") for word in data.vocabulary],
            dtype=bool,
        )
        return ChannelBalancedETM(
            data.embeddings,
            topics,
            fragment_mask,
            hidden=HIDDEN_WIDTH,
        )
    raise ValueError(f"control method must be one of {METHODS}")


@torch.inference_mode()
def infer_document_topics(
    model: ControlModel,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Evaluate deterministic ETM inference and return spectra per second."""
    model.eval()
    _synchronize(device)
    started = time.perf_counter()
    values: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        theta, _ = model.document_topic_mixture(
            dense_normalized(matrix, rows, device),
            sample=False,
        )
        values.append(theta.cpu().numpy().astype(np.float32))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPSILON)


def _validation_metrics(
    data: ControlData,
    theta_observed: np.ndarray,
    theta_full: np.ndarray,
    beta: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the common completion and model-quality diagnostics."""
    _, top_words = topic_word_diagnostics(beta, data.vocabulary)
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        data.vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    return (
        {
            "document_completion": completion_metrics(
                theta_observed,
                beta,
                data.completion,
                data.records,
            ),
            **diagnostics,
            "theta_distribution": mixture_distribution_summary(theta_full),
            "finite_stable": bool(
                np.all(np.isfinite(beta))
                and np.all(np.isfinite(theta_observed))
                and np.all(np.isfinite(theta_full))
            ),
        },
        top_words,
    )


def train_control(
    run: Path,
    *,
    method: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Fit one ETM control and persist its immutable validation artifacts."""
    if method not in METHODS:
        raise ValueError(f"control method must be one of {METHODS}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    output = run / "models" / method
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json_object(result_path)

    data = load_control_data(run)
    seed = int(data.protocol["seed"]) + 7001
    threads = int(data.protocol["cpu_threads"])
    configure_deterministic_execution(seed, threads)
    model = build_control_model(data, method).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    random_generator = np.random.default_rng(int(data.protocol["seed"]) + 7019)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / TRAINING_ACCESS_AUDIT_FILENAME, data.input_manifest)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory_state: MemoryState = sample_runtime_memory()
    history: list[dict[str, Any]] = []
    _synchronize(device)
    started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        model.train()
        order = random_generator.permutation(data.train.shape[0])
        reconstruction_values: list[float] = []
        kl_values: list[float] = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            normalized = dense_normalized(data.train, rows, device)
            theta, kl = model.document_topic_mixture(normalized, sample=True)
            beta = model.topic_word_distribution()
            reconstruction, _ = raw_count_reconstruction_loss(
                theta,
                beta,
                data.train[rows],
                device,
            )
            objective = reconstruction + kl.mean()
            if not torch.isfinite(objective):
                raise FloatingPointError(f"{method} produced a non-finite objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if not all(
                parameter.grad is None
                or bool(torch.all(torch.isfinite(parameter.grad)).item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{method} produced non-finite gradients")
            optimizer.step()
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.mean().detach().cpu()))
        _synchronize(device)
        memory_state = sample_runtime_memory(memory_state)
        row = {
            "epoch": epoch_index + 1,
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv_rows(output / "training_history.csv", history)
        print(
            "ETM_CONTROL_EPOCH",
            json.dumps({"method": method, **row}, sort_keys=True),
            flush=True,
        )
    _synchronize(device)
    training_seconds = time.perf_counter() - started

    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.topic_word_distribution().cpu().numpy(),
            name=f"{method} validation beta",
        )
    theta_observed, observed_throughput = infer_document_topics(
        model,
        data.observed,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = infer_document_topics(
        model,
        data.full,
        batch_size=batch_size,
        device=device,
    )
    metrics, top_words = _validation_metrics(
        data,
        theta_observed,
        theta_full,
        beta,
    )
    metrics["runtime"] = {
        "training_wall_seconds": training_seconds,
        "validation_observed_spectra_per_second": observed_throughput,
        "validation_full_spectra_per_second": full_throughput,
        "memory": runtime_memory_metrics(memory_state, device),
    }
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    config = {
        "architecture": (
            "canonical fixed-SGNS ETM"
            if method == "etm"
            else "fixed-SGNS ETM with fragment/loss-balanced decoder"
        ),
        "embedding_dimensions": int(data.embeddings.shape[1]),
        "hidden_dimensions": HIDDEN_WIDTH,
        "topics": int(data.protocol["model"]["num_topics"]),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "seed": seed,
        "decoder_normalization": (
            "global topic-word softmax"
            if method == "etm"
            else "independent fragment/loss softmaxes at 0.5 each"
        ),
        "paired_reference_method": "etm" if method == "etm_balanced" else None,
        "only_scientific_change": (
            "beta normalization: global softmax to fixed 0.5 fragment and 0.5 loss"
            if method == "etm_balanced"
            else None
        ),
        "trained_separately": True,
    }
    write_json(output / "config.json", config)
    write_csv_rows(output / "top_words.csv", top_words)
    write_json(
        output / "fragment_mass_summary.json", metrics["fragment_probability_mass"]
    )
    write_json(
        output / "duplicate_component_summary.json",
        {
            "duplicate_components": metrics["topic_inventory"]["duplicate_components"],
            "largest_strict_duplicate_component": metrics["topic_inventory"][
                "largest_strict_duplicate_component"
            ],
            "catastrophic_duplicate_component": metrics["topic_inventory"][
                "catastrophic_duplicate_component"
            ],
        },
    )
    result = {
        "method": method,
        "architecture_method": method,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
        "metrics": metrics,
    }
    write_json(result_path, result)
    save_validation(run, method, beta, theta_full, metrics)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch control training or validation chemistry scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--run", required=True, type=Path)
    train.add_argument("--method", required=True, choices=METHODS)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=256)
    chemical = commands.add_parser("chemical")
    chemical.add_argument("--run", required=True, type=Path)
    chemical.add_argument("--data-root", required=True, type=Path)
    chemical.add_argument("--method", required=True, choices=METHODS)
    arguments = parser.parse_args(argv)
    run = arguments.run.expanduser().resolve()
    if arguments.command == "train":
        result = train_control(
            run,
            method=arguments.method,
            device=resolve_torch_device(arguments.device),
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
        )
    else:
        result = score_chemical_validation(
            run,
            arguments.data_root.expanduser().resolve(),
            arguments.method,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
