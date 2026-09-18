"""Validation-only screen of ETM with published prior/normalization additions.

Every run is immutable, records the exact input/code hashes, and fits only the
training matrix. Synthetic truth is loaded for evaluation, never optimization.
Real inputs must be an existing sealed train/validation view.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.torch_version import TorchVersion

from benchmarks.neural_ms2lda.batchnorm_calibration import calibrate_posterior_batchnorm
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.model_evaluation import (
    completion_metrics,
    save_validation,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.model_inference import infer_document_topics
from benchmarks.neural_ms2lda.model_variants import VARIANTS, build_model
from benchmarks.neural_ms2lda.prior_etm import PriorETM
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
    normalize_probability_rows,
    resolve_torch_device,
    sha256_file,
    write_csv_rows,
)
from benchmarks.neural_ms2lda.synthetic_msms import (
    EVALUATION_PROTOCOL,
    load_prepared_synthetic_seed,
    matched_truth_metrics,
)
from benchmarks.neural_ms2lda.topic_model_training import (
    dense_normalized,
    sparse_reconstruction_loss,
    training_batches,
)
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    input_identity,
    write_json,
)
from benchmarks.neural_ms2lda.validation_data import load_validation_inputs

# Explicit scientific dependencies: changes to input construction, equations,
# optimization OR metrics must be visible in a new run's source provenance.
SOURCE_FILES = (
    "scripts/run_minimal_etm.py",
    *(
        f"benchmarks/neural_ms2lda/{name}.py"
        for name in (
            "model_variants",
            "validation_data",
            "reproducibility",
            "utils",
            "prior_etm",
            "contextual_reductions",
            "simplified_etm",
            "batchnorm_calibration",
            "etm_baselines",
            "contextual_sparse_etm",
            "topic_model_training",
            "model_inference",
            "model_evaluation",
            "diagnostics",
            "synthetic_msms",
            "data",
            "spectra",
            "reproduction_audit",
            "reproduction_plan",
            "study_protocol",
        )
    ),
)


@torch.inference_mode()
def infer(model, matrix, *, batch_size: int, device: torch.device) -> np.ndarray:
    """Use the posterior mean and frozen BatchNorm statistics on validation."""
    theta, _ = infer_document_topics(
        model, matrix, batch_size=batch_size, device=device
    )
    return normalize_probability_rows(theta, name="validation theta")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Fit one explicit configuration and report validation evidence."""
    source_path = getattr(arguments, "recalibrate_from", None)
    source = None
    if source_path is not None:
        source = json.loads((source_path / "result.json").read_text())
        if (
            source["config"]["test_matrices_loaded"]
            or source["config"]["selection_split"] != "validation"
        ):
            raise ValueError("source model must be validation-only")
        if arguments.variant != source["config"]["variant"]:
            raise ValueError("recalibration cannot change the source architecture")
        # A recalibration is not retraining: retain the original training recipe.
        for name in (
            "seed",
            "topics",
            "hidden",
            "concentration",
            "epochs",
            "batch_size",
            "learning_rate",
            "momentum",
            "count_scaling",
        ):
            setattr(arguments, name, source["config"][name])
    arguments.normalization_statistics = getattr(
        arguments, "normalization_statistics", "training"
    )
    if arguments.epochs <= 0 or arguments.topics < 2:
        raise ValueError("epochs must be positive and topics must be at least two")
    if getattr(arguments, "checkpoint_interval", 20) < 0:
        raise ValueError("checkpoint interval cannot be negative")
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"use a fresh output directory: {output}")
    synthetic = None
    if arguments.synthetic_root is not None:
        synthetic, embeddings, input_directory = load_prepared_synthetic_seed(
            arguments.synthetic_root.resolve(), seed=arguments.seed
        )
        train, observed, full, completion = (
            synthetic.train,
            synthetic.validation_observed,
            synthetic.validation_full,
            synthetic.validation_completion,
        )
        records, vocabulary = synthetic.validation_records, synthetic.vocabulary
        input_paths = sorted(input_directory.glob("*.np*")) + [
            input_directory / "vocabulary.json",
            input_directory / "validation_records.jsonl",
            input_directory / "token_features/features.npy",
        ]
    else:
        data = load_validation_inputs(arguments.validation_run.resolve())
        train, observed, full, completion = (
            data.train,
            data.observed,
            data.full,
            data.completion,
        )
        records, vocabulary, embeddings = data.records, data.vocabulary, data.embeddings
        input_directory = arguments.validation_run.resolve()
        if output != input_directory / "models/minimal_etm":
            raise ValueError("real output must be <validation-run>/models/minimal_etm")
        if (input_directory / "validation_evaluation/minimal_etm").exists():
            raise FileExistsError("validation evidence already exists; use a fresh run")
        input_paths = [
            input_directory / "data" / name
            for name in (
                "train.npz",
                "validation_observed.npz",
                "validation_full.npz",
                "validation_completion.npz",
                "validation_records.jsonl",
                "vocabulary.json",
            )
        ] + [input_directory / "token_features/features.npy"]
        if arguments.topics != int(data.protocol["model"]["num_topics"]):
            raise ValueError("topics must match the sealed real-data protocol")
    training_batches(np.arange(train.shape[0]), arguments.batch_size)
    configure_deterministic_execution(arguments.seed + 7001, arguments.threads)
    device = resolve_torch_device(arguments.device)
    model = build_model(
        arguments.variant,
        embeddings,
        vocabulary,
        arguments.topics,
        hidden=arguments.hidden,
        concentration=arguments.concentration,
    ).to(device)
    has_batchnorm = isinstance(model, PriorETM) and isinstance(
        model.mean_normalization, torch.nn.BatchNorm1d
    )
    if source is not None and not has_batchnorm:
        raise ValueError("recalibration requires a BatchNorm model")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }
    config.update(
        {
            "training_seed": arguments.seed + 7001,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "torch": str(torch.__version__),
            "input_shape": list(train.shape),
            "test_matrices_loaded": False,
            "selection_split": "validation",
            "input_sha256": {str(path): sha256_file(path) for path in input_paths},
            "code_sha256": {
                name: sha256_file(Path(__file__).parents[1] / name)
                for name in SOURCE_FILES
            },
        }
    )
    if source is not None:
        if input_identity(config["input_sha256"]) != input_identity(
            source["config"]["input_sha256"]
        ):
            raise ValueError("recalibration inputs must match the source run exactly")
        checkpoint_path = source_path / "checkpoint.pt"
        # Early local pilots stored TorchVersion, a str subclass, as metadata.
        # Allow only this harmless type; never use weights_only=False.
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(
                checkpoint_path, weights_only=True, map_location=device
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        config["parent_result_sha256"] = sha256_file(source_path / "result.json")
        config["parent_checkpoint_sha256"] = sha256_file(checkpoint_path)
        config["parent_code_sha256"] = source["config"]["code_sha256"]
    if not has_batchnorm:
        config["normalization_statistics"] = "not_applicable"
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    if synthetic is None:
        write_json(output / "training_access_audit.json", data.input_manifest)
    if source is None:
        resume_path = getattr(arguments, "resume_training_from", None)
        if resume_path is not None:
            recovery = torch.load(resume_path, weights_only=True, map_location="cpu")
            previous = recovery["config"]
            for key in (
                "variant",
                "seed",
                "topics",
                "hidden",
                "concentration",
                "epochs",
                "batch_size",
                "learning_rate",
                "momentum",
                "count_scaling",
                # Numerical continuation is only exact on the same runtime.
                "threads",
                "device",
                "torch",
            ):
                if previous[key] != config[key]:
                    raise ValueError(
                        f"resuming cannot alter the training recipe: {key}"
                    )
            if input_identity(previous["input_sha256"]) != input_identity(
                config["input_sha256"]
            ):
                raise ValueError("resuming cannot change training/validation inputs")
            if input_identity(previous["code_sha256"]) != input_identity(
                config["code_sha256"]
            ):
                raise ValueError(
                    "exact resumption requires unchanged scientific source code"
                )
            config["parent_code_sha256"] = previous["code_sha256"]
            config["resumed_training_checkpoint_sha256"] = sha256_file(resume_path)
            config["resumed_completed_epochs"] = recovery["completed_epochs"]
            write_json(output / "config.json", config)
        else:
            recovery = None
        history, training_seconds = fit(
            model, train, arguments, device, recovery=recovery
        )
    else:
        with (source_path / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        training_seconds = source["training_seconds"]
    if has_batchnorm and arguments.normalization_statistics == "training":
        calibration = calibrate_posterior_batchnorm(
            model, train, batch_size=arguments.batch_size, device=device
        )
        write_json(output / "normalization_calibration.json", calibration)
    model.eval()
    with torch.inference_mode():
        beta = normalize_probability_rows(
            model.topic_word_distribution().cpu().numpy(), name="beta"
        )
    theta = infer(model, full, batch_size=arguments.batch_size, device=device)
    observed_theta = infer(
        model, observed, batch_size=arguments.batch_size, device=device
    )
    metrics = {
        "completion": completion_metrics(
            observed_theta, beta, completion, list(records)
        ),
        "support": theta_support_diagnostics(theta),
        **model_selection_diagnostics(theta, beta, vocabulary, EVALUATION_PROTOCOL),
    }
    if synthetic is not None:
        metrics["truth"] = matched_truth_metrics(
            beta, theta, synthetic.true_beta, synthetic.validation_true_theta
        )
    else:
        save_validation(input_directory, "minimal_etm", beta, theta, metrics)
    atomic_torch_save(
        output / "checkpoint.pt", {"state_dict": model.state_dict(), "config": config}
    )
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "validation_theta.npy", theta)
    write_csv_rows(output / "history.csv", history)
    result = {
        "config": config,
        "training_seconds": training_seconds,
        "metrics": metrics,
    }
    write_json(output / "result.json", result)
    inventory = metrics["topic_inventory"]
    print(
        json.dumps(
            {
                "output": str(output),
                "nll": metrics["completion"]["nll_per_token"],
                "effective_topics": metrics["support"][
                    "median_effective_topics_per_spectrum"
                ],
                "unique_top1": inventory["unique_top1_topics"],
                "nearest_beta_cosine": inventory["mean_nearest_topic_beta_cosine"],
                "truth": {
                    key: value
                    for key, value in metrics.get("truth", {}).items()
                    if key != "matching"
                },
            }
        ),
        flush=True,
    )
    return result


def fit(model, train, arguments, device, *, recovery=None) -> tuple[list[dict], float]:
    """Optimize the negative Gaussian ELBO on training counts only.

    One reparameterized draw supplies the reconstruction expectation in eq:elbo;
    mean(KL_d) is analytic and has coefficient one. The sparse likelihood sums
    counts within spectra and averages spectra, with no document-length scaling
    in the chosen raw_counts recipe. Validation/truth never enters this loop.
    Recovery restores optimizer and shuffle/sampling RNG states. Exact replay
    additionally requires unchanged code, numerical libraries and hardware;
    run() checks recorded source, torch version, device kind and thread count.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(arguments.momentum, 0.999),
        weight_decay=1.2e-6,
    )
    rng = np.random.default_rng(arguments.seed + 7019)
    history = []
    completed_epochs = 0
    previous_seconds = 0.0
    if recovery is not None:
        model.load_state_dict(recovery["state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        rng.bit_generator.state = recovery["shuffle_rng_state"]
        torch.set_rng_state(recovery["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(recovery["cuda_rng_state_all"])
        history = recovery["history"]
        completed_epochs = recovery["completed_epochs"]
        previous_seconds = recovery["training_seconds"]
    started = time.perf_counter()
    for epoch in range(completed_epochs, arguments.epochs):
        model.train()
        reconstruction_sum = kl_sum = 0.0
        for rows in training_batches(
            rng.permutation(train.shape[0]), arguments.batch_size
        ):
            theta, kl = model.document_topic_mixture(
                dense_normalized(train, rows, device), sample=True
            )
            reconstruction, _ = sparse_reconstruction_loss(
                theta,
                model.topic_word_distribution(),
                train[rows],
                device,
                scaling=arguments.count_scaling,
            )
            loss = reconstruction + kl.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("non-finite training gradient")
            optimizer.step()
            reconstruction_sum += float(reconstruction.detach()) * len(rows)
            kl_sum += float(kl.mean().detach()) * len(rows)
        row = {
            "epoch": epoch + 1,
            "reconstruction": reconstruction_sum / train.shape[0],
            "kl": kl_sum / train.shape[0],
        }
        history.append(row)
        interval = getattr(arguments, "checkpoint_interval", 20)
        if interval and ((epoch + 1) % interval == 0 or epoch + 1 == arguments.epochs):
            atomic_torch_save(
                arguments.output / f"training_recovery_epoch{epoch + 1:03d}.pt",
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_epochs": epoch + 1,
                    "history": history,
                    "shuffle_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                    ),
                    "training_seconds": previous_seconds
                    + time.perf_counter()
                    - started,
                    "config": json.loads(
                        (arguments.output / "config.json").read_text()
                    ),
                },
            )
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(json.dumps({"variant": arguments.variant, **row}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return history, previous_seconds + time.perf_counter() - started


def main() -> None:
    """Run one small, explicitly labelled experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--synthetic-root", type=Path)
    inputs.add_argument("--validation-run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument(
        "--recalibrate-from",
        type=Path,
        help="reuse a source model's weights and recipe without retraining",
    )
    reuse.add_argument("--resume-training-from", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument(
        "--normalization-statistics", choices=("ema", "training"), default="training"
    )
    parser.add_argument("--topics", type=int, default=36)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=800)
    parser.add_argument("--concentration", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument(
        "--count-scaling",
        choices=("raw_counts", "distinct_words", "unit_mass"),
        default="raw_counts",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
