"""Train and evaluate the two published ETM controls on frozen MSnLib views."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.neural_ms2lda.baseline_repeats import (
    baseline_fit_identity,
    require_cached_fit,
    resolve_training_seed,
)
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.etm_baselines import (
    CanonicalETM,
    ChannelBalancedETM,
)
from benchmarks.neural_ms2lda.model_evaluation import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
    completion_metrics,
    mixture_distribution_summary,
    save_validation,
    score_chemical_validation,
    topic_word_diagnostics,
)
from benchmarks.neural_ms2lda.model_inference import (
    infer_document_topics as infer_document_topics,
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
from benchmarks.neural_ms2lda.validation_data import (
    ValidationInputs as ControlData,
)
from benchmarks.neural_ms2lda.validation_data import (
    load_validation_inputs as load_control_data,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


METHODS = ("etm", "etm_balanced")
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1.2e-6
HIDDEN_WIDTH = 800
EPSILON = 1e-12
ControlModel = CanonicalETM | ChannelBalancedETM


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before reading a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    training_seed: int | None = None,
) -> dict[str, Any]:
    """Fit one ETM control and persist its immutable validation artifacts."""
    if method not in METHODS:
        raise ValueError(f"control method must be one of {METHODS}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    output = run / "models" / method
    result_path = output / "result.json"
    data = load_control_data(run)
    seed = resolve_training_seed(data.protocol, training_seed, offset=7001)
    # The original runner used protocol.seed + 7019 for shuffling, eighteen
    # above its initialization seed. Preserve this relationship for every fit.
    shuffle_seed = seed + 18
    threads = int(data.protocol["cpu_threads"])
    identity = baseline_fit_identity(
        run,
        seed,
        {
            "method": method,
            "topics": int(data.protocol["model"]["num_topics"]),
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden": HIDDEN_WIDTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_betas": [0.9, 0.999],
            "threads": threads,
            "device": str(device),
            "torch": str(torch.__version__),
            "shuffle_seed": shuffle_seed,
        },
    )
    if result_path.is_file():
        cached = read_json_object(result_path)
        require_cached_fit(cached, identity)
        validation = run / "validation_evaluation" / method
        required = (
            output / "weights.pt",
            validation / "beta.npy",
            validation / "validation_full_theta.npy",
            validation / "complete.json",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(
                "cached baseline is missing required validation artifacts"
            )
        return cached
    configure_deterministic_execution(seed, threads)
    model = build_control_model(data, method).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    random_generator = np.random.default_rng(shuffle_seed)
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
        "training_seed": seed,
        "shuffle_seed": shuffle_seed,
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
        "fit_identity": identity,
    }
    save_validation(run, method, beta, theta_full, metrics)
    # The result is the completion marker. Write it only after its checkpoint,
    # validation arrays and evaluation metadata exist, never before them.
    write_json(result_path, result)
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
    train.add_argument("--training-seed", type=int)
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
            training_seed=arguments.training_seed,
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
