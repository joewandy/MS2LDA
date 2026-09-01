"""Run bounded NSTM experiments on synthetic MS/MS and frozen validation data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.nstm import (
    DOCUMENT_INPUT_MODES,
    DocumentInputMode,
    NeuralSinkhornTopicModel,
)
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
    read_json_object,
    resolve_torch_device,
    sha256_file,
    write_csv_rows,
)
from benchmarks.neural_ms2lda.sparse_etm import theta_support_diagnostics
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    write_json,
)
from scripts.run_sparse_etm_campaign import (
    SYNTHETIC_ACTIVE_USAGE_THRESHOLD,
    SYNTHETIC_EVALUATION_PROTOCOL,
    _matched_truth_metrics,
    prepare_synthetic_seed,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

EPS = 1e-12
OFFICIAL_NSTM_SHA = "610d1604d5467289028714ed0ce684dfb5ef8a7b"
TOPMOST_NSTM_SHA = "ef24433859b2e283959ddef7f95020a40abb104f"


def dense_counts(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Materialize one bounded dense NSTM batch on the selected device."""
    values = matrix[rows].toarray().astype(np.float32, copy=False)
    return torch.from_numpy(values).to(device)


@torch.inference_mode()
def infer_theta(
    model: NeuralSinkhornTopicModel,
    matrix: sp.csr_matrix,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Infer deterministic NSTM document-topic distributions."""
    model.eval()
    values = []
    started = time.perf_counter()
    for start in range(0, matrix.shape[0], int(batch_size)):
        rows = np.arange(
            start,
            min(start + int(batch_size), matrix.shape[0]),
            dtype=np.int64,
        )
        values.append(
            model.theta(dense_counts(matrix, rows, device))
            .cpu()
            .numpy()
            .astype(np.float32),
        )
    elapsed = time.perf_counter() - started
    return np.concatenate(values), matrix.shape[0] / max(elapsed, EPS)


@torch.inference_mode()
def decoder_completion_metrics(  # noqa: PLR0913
    model: NeuralSinkhornTopicModel,
    theta: np.ndarray,
    completion: sp.csr_matrix,
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Score pseudo-count completion under NSTM's exact virtual decoder."""
    if theta.shape != (completion.shape[0], model.num_topics):
        message = "theta and completion shapes do not match NSTM"
        raise ValueError(message)
    if len(records) != completion.shape[0]:
        message = "completion records do not align with the matrix"
        raise ValueError(message)
    total_loss = 0.0
    in_vocabulary = 0
    out_of_vocabulary = 0
    eligible = 0
    for start in range(0, completion.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), completion.shape[0])
        local_theta = torch.from_numpy(theta[start:stop]).to(device)
        log_probability = model.decode_log_probabilities(local_theta).cpu().numpy()
        for local_row, row in enumerate(range(start, stop)):
            left, right = completion.indptr[row], completion.indptr[row + 1]
            words = completion.indices[left:right]
            counts = completion.data[left:right]
            token_count = int(counts.sum())
            out_of_vocabulary += int(records[row]["completion_oov_tokens"])
            if not token_count:
                continue
            total_loss -= float(np.sum(counts * log_probability[local_row, words]))
            in_vocabulary += token_count
            eligible += 1
    total = in_vocabulary + out_of_vocabulary
    return {
        "nll_per_token": total_loss / in_vocabulary,
        "in_vocabulary_tokens": in_vocabulary,
        "out_of_vocabulary_tokens": out_of_vocabulary,
        "oov_fraction": out_of_vocabulary / total,
        "eligible_documents": eligible,
        "total_documents": completion.shape[0],
        "decoder": "softmax(theta @ cosine(topic_embedding, word_embedding))",
    }


def method_label(input_mode: DocumentInputMode) -> str:
    """Return a stable artifact label for one reference interpretation."""
    return f"nstm_{input_mode}"


def run_synthetic(  # noqa: PLR0913, PLR0915
    output_root: Path,
    *,
    seed: int,
    fitted_topics: int,
    input_mode: DocumentInputMode,
    epochs: int,
    batch_size: int,
    device: torch.device,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> dict[str, Any]:
    """Train and evaluate one reference NSTM on truth-known MS/MS data."""
    label = method_label(input_mode)
    output = output_root / "synthetic_runs" / f"seed_{seed}_K_{fitted_topics}_{label}"
    result_path = output / "result.json"
    if result_path.is_file():
        return read_json_object(result_path)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "evidence": "truth-known synthetic train and validation only",
        "reference_paper": (
            "Zhao et al., Neural Topic Model via Optimal Transport, ICLR 2021"
        ),
        "official_reference_repository": (
            "https://github.com/ethanhezhao/NeuralSinkhornTopicModel"
        ),
        "official_reference_sha": OFFICIAL_NSTM_SHA,
        "modern_cross_check_repository": "https://github.com/bobxwu/TopMost",
        "modern_cross_check_sha": TOPMOST_NSTM_SHA,
        "seed": int(seed),
        "true_topics": 18,
        "fitted_topics": int(fitted_topics),
        "document_input_mode": input_mode,
        "fixed_train_only_sgns_dimensions": 48,
        "word_embeddings_trainable": False,
        "hidden_dimensions": 200,
        "dropout_probability": 0.25,
        "reconstruction_weight_epsilon": 0.07,
        "sinkhorn_alpha": 20.0,
        "sinkhorn_maximum_iterations": 1000,
        "sinkhorn_stop_tolerance": 0.005,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "device": str(device),
        "threads": int(threads),
        "training_documents": int(training_documents),
        "validation_documents": int(validation_documents),
        "stopping_rule": "fixed epochs or immediate non-finite loss/gradient",
    }
    write_json(output / "config.json", config)
    dataset, embeddings, seed_directory = prepare_synthetic_seed(
        output_root,
        seed=seed,
        threads=threads,
        training_documents=training_documents,
        validation_documents=validation_documents,
    )
    configure_deterministic_execution(seed + 8101, threads)
    model = NeuralSinkhornTopicModel(
        embeddings,
        fitted_topics,
        hidden=200,
        dropout=0.25,
        reconstruction_weight=0.07,
        sinkhorn_alpha=20.0,
        sinkhorn_maximum_iterations=1000,
        sinkhorn_stop_tolerance=0.005,
        input_mode=input_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed + 8119)
    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(dataset.train.shape[0])
        joint_values = []
        reconstruction_values = []
        sinkhorn_values = []
        gradient_values = []
        iteration_values = []
        marginal_error_values = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            if len(rows) == 1:
                rows = np.concatenate((rows, order[:1]))
            output_values = model(dense_counts(dataset.train, rows, device))
            if not torch.isfinite(output_values.loss):
                message = "NSTM produced a non-finite objective"
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            output_values.loss.backward()
            trainable = (
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            finite_gradients = all(
                parameter.grad is None
                or torch.all(torch.isfinite(parameter.grad)).item()
                for parameter in trainable
            )
            if not finite_gradients:
                message = "NSTM produced non-finite gradients"
                raise FloatingPointError(message)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            optimizer.step()
            joint_values.append(float(output_values.loss.detach().cpu()))
            reconstruction_values.append(
                float(output_values.reconstruction_loss.detach().cpu()),
            )
            sinkhorn_values.append(float(output_values.sinkhorn_loss.detach().cpu()))
            gradient_values.append(float(gradient_norm.detach().cpu()))
            iteration_values.append(output_values.sinkhorn_iterations)
            marginal_error_values.append(output_values.sinkhorn_marginal_error)
        row = {
            "epoch": epoch + 1,
            "joint_loss": float(np.mean(joint_values)),
            "reconstruction": float(np.mean(reconstruction_values)),
            "sinkhorn": float(np.mean(sinkhorn_values)),
            "mean_gradient_norm": float(np.mean(gradient_values)),
            "mean_sinkhorn_iterations": float(np.mean(iteration_values)),
            "maximum_sinkhorn_iterations": int(np.max(iteration_values)),
            "maximum_sinkhorn_marginal_error": float(
                np.max(marginal_error_values),
            ),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv_rows(output / "training_history.csv", history)
        print(  # noqa: T201
            "NSTM_EPOCH",
            json.dumps({"run": label, **row}, sort_keys=True),
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    model.eval()
    with torch.inference_mode():
        beta = model.beta().cpu().numpy().astype(np.float32)
    theta_observed, observed_throughput = infer_theta(
        model,
        dataset.validation_observed,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = infer_theta(
        model,
        dataset.validation_full,
        batch_size=batch_size,
        device=device,
    )
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        dataset.vocabulary,
        SYNTHETIC_EVALUATION_PROTOCOL,
    )
    usage = theta_full.astype(np.float64).mean(axis=0)
    metrics = {
        "dataset": dataset.summary,
        "heldout_completion": decoder_completion_metrics(
            model,
            theta_observed,
            dataset.validation_completion,
            dataset.validation_records,
            batch_size=batch_size,
            device=device,
        ),
        "truth_recovery": _matched_truth_metrics(
            beta,
            theta_full,
            dataset.true_beta,
            dataset.validation_true_theta,
        ),
        "theta_support": theta_support_diagnostics(theta_full),
        **diagnostics,
        "active_topics_mean_usage_gt_0_005": int(
            np.sum(usage > SYNTHETIC_ACTIVE_USAGE_THRESHOLD),
        ),
        "finite_stable": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(theta_observed))
            and np.all(np.isfinite(theta_full)),
        ),
        "runtime": {
            "training_wall_seconds": training_seconds,
            "validation_observed_spectra_per_second": observed_throughput,
            "validation_full_spectra_per_second": full_throughput,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_torch_save(
        output / "weights.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "validation_observed_theta.npy", theta_observed)
    atomic_save_numpy(output / "validation_full_theta.npy", theta_full)
    provenance = {
        "synthetic_artifact_manifest": str(
            seed_directory / "artifact_manifest.json",
        ),
        "token_features": {
            "path": str(seed_directory / "token_features/features.npy"),
            "bytes": (seed_directory / "token_features/features.npy").stat().st_size,
            "sha256": sha256_file(seed_directory / "token_features/features.npy"),
        },
        "weights": {
            "path": str(output / "weights.pt"),
            "bytes": (output / "weights.pt").stat().st_size,
            "sha256": sha256_file(output / "weights.pt"),
        },
        "candidate_test_artifacts_accessed": False,
    }
    write_json(output / "provenance.json", provenance)
    result = {
        "method": label,
        "config": config,
        "metrics": metrics,
        "provenance": provenance,
    }
    write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded NSTM campaign command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic-run")
    synthetic.add_argument("--output-root", required=True, type=Path)
    synthetic.add_argument("--seed", required=True, type=int)
    synthetic.add_argument("--fitted-topics", required=True, type=int)
    synthetic.add_argument(
        "--input-mode",
        required=True,
        choices=DOCUMENT_INPUT_MODES,
    )
    synthetic.add_argument("--epochs", type=int, default=50)
    synthetic.add_argument("--batch-size", type=int, default=200)
    synthetic.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    synthetic.add_argument("--threads", type=int, default=6)
    synthetic.add_argument("--training-documents", type=int, default=800)
    synthetic.add_argument("--validation-documents", type=int, default=160)
    args = parser.parse_args(argv)
    if args.command != "synthetic-run":
        message = "unknown NSTM campaign command"
        raise RuntimeError(message)
    result = run_synthetic(
        args.output_root.expanduser().resolve(),
        seed=args.seed,
        fitted_topics=args.fitted_topics,
        input_mode=args.input_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=resolve_torch_device(args.device),
        threads=args.threads,
        training_documents=args.training_documents,
        validation_documents=args.validation_documents,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
