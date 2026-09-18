"""Validation-only prototypes of ProdLDA and direct-Dirichlet topic models.

This is a port/adaptation study, not a reproduction of text-corpus paper scores.
The fixed protocol is research/minimal_neural_etm/review_20260907/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.model_evaluation import (
    save_validation,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.published_evaluation import (
    dense_counts,
    infer_published,
    published_completion,
)
from benchmarks.neural_ms2lda.published_models import VARIANTS, build_published_model
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
from benchmarks.neural_ms2lda.topic_model_training import training_batches
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    atomic_torch_save,
    write_json,
)
from benchmarks.neural_ms2lda.validation_data import load_validation_inputs


def fit(model, train, args, device):
    """Train raw-count reconstruction and analytic KL on training rows only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    rng = np.random.default_rng(args.seed + 7019)
    history, started = [], time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        warmup = (
            1.0 if args.variant.startswith("prodlda") else min(1.0, (epoch + 1) / 100)
        )
        reconstruction_sum = kl_sum = 0.0
        for rows in training_batches(rng.permutation(train.shape[0]), args.batch_size):
            counts = dense_counts(train, rows, device)
            logp, _, kl = model(counts, sample=True)
            reconstruction = -(counts * logp).sum(dim=1).mean()
            loss = reconstruction + warmup * kl.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite published-model objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("non-finite published-model gradient")
            optimizer.step()
            reconstruction_sum += float(reconstruction.detach()) * len(rows)
            kl_sum += float(kl.mean().detach()) * len(rows)
        row = {
            "epoch": epoch + 1,
            "reconstruction": reconstruction_sum / train.shape[0],
            "kl": kl_sum / train.shape[0],
            "kl_weight": warmup,
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(json.dumps({"variant": args.variant, **row}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return history, time.perf_counter() - started


def run(args):
    """Write immutable input/code provenance and model-specific validation."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"use a fresh output directory: {output}")
    if args.epochs <= 0 or args.batch_size < 2:
        raise ValueError("positive epochs and batch size >=2 are required")
    synthetic = None
    if args.synthetic_root is not None:
        synthetic, embeddings, directory = load_prepared_synthetic_seed(
            args.synthetic_root.resolve(), seed=args.seed
        )
        train, full, observed, completion = (
            synthetic.train,
            synthetic.validation_full,
            synthetic.validation_observed,
            synthetic.validation_completion,
        )
        records, vocabulary = synthetic.validation_records, synthetic.vocabulary
        inputs = sorted(directory.glob("*.np*")) + [
            directory / "vocabulary.json",
            directory / "validation_records.jsonl",
            directory / "token_features/features.npy",
        ]
    else:
        directory = args.validation_run.resolve()
        data = load_validation_inputs(directory)
        train, full, observed, completion = (
            data.train,
            data.full,
            data.observed,
            data.completion,
        )
        records, vocabulary, embeddings = data.records, data.vocabulary, data.embeddings
        if output != directory / "models/minimal_etm":
            raise ValueError("real output must be <validation-run>/models/minimal_etm")
        if args.topics != int(data.protocol["model"]["num_topics"]):
            raise ValueError("topics must match the sealed real-data protocol")
        inputs = [
            directory / "data" / name
            for name in (
                "train.npz",
                "validation_full.npz",
                "validation_observed.npz",
                "validation_completion.npz",
                "validation_records.jsonl",
                "vocabulary.json",
            )
        ] + [directory / "token_features/features.npy"]
    configure_deterministic_execution(args.seed + 7001, args.threads)
    device = resolve_torch_device(args.device)
    model = build_published_model(args.variant, embeddings, args.topics).to(device)
    source_root = Path(__file__).resolve().parents[1]
    sources = [
        "scripts/run_published_neural.py",
        "benchmarks/neural_ms2lda/validation_data.py",
        "benchmarks/neural_ms2lda/topic_model_training.py",
        "benchmarks/neural_ms2lda/utils.py",
        "benchmarks/neural_ms2lda/published_models.py",
        "benchmarks/neural_ms2lda/published_evaluation.py",
        "benchmarks/neural_ms2lda/reproducibility.py",
        "benchmarks/neural_ms2lda/contextual_sparse_etm.py",
        "benchmarks/neural_ms2lda/diagnostics.py",
        "benchmarks/neural_ms2lda/model_evaluation.py",
        "benchmarks/neural_ms2lda/synthetic_msms.py",
    ]
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training_seed": args.seed + 7001,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "hidden": 100,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "count_scaling": "raw_counts",
        "concentration": None if args.variant.startswith("prodlda") else 0.02,
        "kl_warmup_epochs": 0 if args.variant.startswith("prodlda") else 100,
        "normalization_statistics": "ema",
        "torch": str(torch.__version__),
        "input_shape": list(train.shape),
        "test_matrices_loaded": False,
        "selection_split": "validation",
        "framework": "pytorch_port_not_pyro_svi",
        "decoder": "product_of_experts" if model.product else "additive_mixture",
        "beta_meaning": (
            "one_hot_conditional_prototypes" if model.product else "emissions"
        ),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "code_sha256": {name: sha256_file(source_root / name) for name in sources},
    }
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    if synthetic is None:
        write_json(output / "training_access_audit.json", data.input_manifest)
    history, seconds = fit(model, train, args, device)
    model.eval()
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    beta = normalize_probability_rows(
        model.topic_prototypes().cpu().numpy(), name="beta"
    )
    theta = infer_published(model, full, batch_size=args.batch_size, device=device)
    observed_theta = infer_published(
        model, observed, batch_size=args.batch_size, device=device
    )
    metrics = {
        "completion": published_completion(
            model,
            observed_theta,
            completion,
            records,
            batch_size=args.batch_size,
            device=device,
        ),
        "support": theta_support_diagnostics(theta),
        **model_selection_diagnostics(theta, beta, vocabulary, EVALUATION_PROTOCOL),
    }
    if any(
        not torch.equal(value, frozen[name]) for name, value in model.named_buffers()
    ):
        raise RuntimeError("validation mutated model buffers")
    if synthetic is not None:
        metrics["truth"] = matched_truth_metrics(
            beta, theta, synthetic.true_beta, synthetic.validation_true_theta
        )
    else:
        save_validation(directory, "minimal_etm", beta, theta, metrics)
    atomic_torch_save(
        output / "checkpoint.pt", {"state_dict": model.state_dict(), "config": config}
    )
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "validation_theta.npy", theta)
    write_csv_rows(output / "history.csv", history)
    result = {"config": config, "training_seconds": seconds, "metrics": metrics}
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "output": str(output),
                "nll": metrics["completion"]["nll_per_token"],
                "effective_topics": metrics["support"][
                    "median_effective_topics_per_spectrum"
                ],
                "winners": metrics["topic_inventory"]["unique_top1_topics"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--synthetic-root", type=Path)
    inputs.add_argument("--validation-run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--topics", type=int, default=36)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
