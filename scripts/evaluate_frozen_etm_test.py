"""Evaluate frozen ETM-family models on the fixed MSnLib test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.neural_ms2lda.contextual_sparse_etm import ContextualSparseETM
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
    entropy_diagnostics,
    mixture_distribution_summary,
    theta_support_diagnostics,
)
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
    normalize_probability_rows,
    resolve_torch_device,
    sha256_file,
    validate_probability_matrix,
)
from benchmarks.neural_ms2lda.study_protocol import METHOD
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    read_json,
    write_json,
)
from scripts.prepare_msnlib_test_view import verify_released_model
from scripts.run_contextual_sparse_etm import infer_document_topic_mixtures
from scripts.run_etm_controls import infer_document_topics

if TYPE_CHECKING:
    from collections.abc import Sequence

SUPPORTED_METHODS = ("etm", "etm_balanced", METHOD)


def _load_model(
    run: Path,
    method: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    """Reconstruct one fitted architecture and load only its frozen weights."""
    output = run / "models" / method
    config = read_json(output / "config.json")
    protocol = read_json(run / "protocol.json")
    vocabulary = load_vocabulary(run / "data")
    embeddings = load_sgns_embeddings(run / "token_features/features.npy")
    topics = int(protocol["model"]["num_topics"])
    hidden = int(config["hidden_dimensions"])
    if method == "etm":
        model: torch.nn.Module = CanonicalETM(embeddings, topics, hidden=hidden)
    else:
        fragment_mask = np.asarray(
            [word.startswith("frag@") for word in vocabulary],
            dtype=bool,
        )
        model = (
            ContextualSparseETM(
                embeddings,
                topics,
                fragment_mask,
                hidden=hidden,
            )
            if method == METHOD
            else ChannelBalancedETM(
                embeddings,
                topics,
                fragment_mask,
                hidden=hidden,
            )
        )
    weights_path = output / "weights.pt"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config, vocabulary


def _infer(
    model: torch.nn.Module,
    method: str,
    matrix: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Apply the deterministic inference equation for the fitted architecture."""
    if method == METHOD:
        return infer_document_topic_mixtures(
            model,  # type: ignore[arg-type]
            matrix,
            batch_size=batch_size,
        )
    return infer_document_topics(
        model,  # type: ignore[arg-type]
        matrix,
        batch_size=batch_size,
        device=device,
    )


def _beta(model: torch.nn.Module, method: str) -> np.ndarray:
    """Evaluate and canonically normalize the fitted topic-word equation."""
    with torch.inference_mode():
        values = model.topic_word_distribution()  # type: ignore[attr-defined]
    return normalize_probability_rows(
        values.detach().cpu().numpy(),
        name=f"{method} beta",
    )


def _load_test_inputs(
    run: Path,
    vocabulary: Sequence[str],
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    """Load and cross-check the fixed observed, withheld, and full test views."""
    data = run / "data"
    observed = load_csr(data / "test_observed.npz")
    completion = load_csr(data / "test_completion.npz")
    full = load_csr(data / "test_full.npz")
    records = load_heldout_records(data, "test")
    if observed.shape != completion.shape or observed.shape != full.shape:
        msg = "test matrices differ in shape"
        raise ValueError(msg)
    if full.shape[0] != len(records) or full.shape[1] != len(vocabulary):
        msg = "test records, vocabulary, and matrices do not align"
        raise ValueError(msg)
    return observed, completion, full, records


def evaluate_test(
    run_directory: Path,
    *,
    method: str,
    device: torch.device,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    """Load frozen weights and compute completion plus diagnostic test metrics."""
    if method not in SUPPORTED_METHODS:
        msg = f"method must be one of {SUPPORTED_METHODS}"
        raise ValueError(msg)
    run = run_directory.expanduser().resolve(strict=True)
    if not (run / "test_input_manifest.json").is_file():
        msg = "test inputs have not been released for this run"
        raise FileNotFoundError(msg)
    model_output = run / "models" / method
    weights_path = model_output / "weights.pt"
    release_record = verify_released_model(
        run,
        method=method,
        model_path=weights_path,
    )
    output = run / "evaluation" / method
    complete = output / "complete.json"
    if complete.is_file():
        result = read_json(complete)
        if (
            result.get("weights_sha256") != release_record["sha256"]
            or result.get("weights_unchanged_after_evaluation") is not True
        ):
            msg = f"cached test result is not bound to frozen model: {method}"
            raise RuntimeError(msg)
        return result

    weights_before = sha256_file(weights_path)
    config = read_json(model_output / "config.json")
    seed = int(config.get("training_seed", config.get("seed", 0)))
    configure_deterministic_execution(seed, threads)
    model, config, vocabulary = _load_model(run, method, device)
    observed, completion_matrix, full, records = _load_test_inputs(run, vocabulary)

    theta_observed, observed_throughput = _infer(
        model,
        method,
        observed,
        batch_size=batch_size,
        device=device,
    )
    theta_full, full_throughput = _infer(
        model,
        method,
        full,
        batch_size=batch_size,
        device=device,
    )
    beta = _beta(model, method)
    validate_probability_matrix(beta, name=f"{method} test beta")
    validate_probability_matrix(theta_observed, name=f"{method} test observed theta")
    validate_probability_matrix(theta_full, name=f"{method} test full theta")
    diagnostics = model_selection_diagnostics(
        theta_full,
        beta,
        vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    metrics = {
        "document_completion": completion_metrics(
            theta_observed,
            beta,
            completion_matrix,
            records,
        ),
        **diagnostics,
        "theta_support": theta_support_diagnostics(theta_full),
        "theta_distribution": mixture_distribution_summary(theta_full),
        "theta_information": entropy_diagnostics(theta_full),
        "finite_stable": True,
        "runtime": {
            "test_observed_spectra_per_second": observed_throughput,
            "test_full_spectra_per_second": full_throughput,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / "beta.npy", beta)
    atomic_save_numpy(output / "test_observed_theta.npy", theta_observed)
    atomic_save_numpy(output / "test_full_theta.npy", theta_full)
    result = {
        "method": method,
        "split": "test",
        "device": str(device),
        "model_config": config,
        "weights_sha256": weights_before,
        "weights_unchanged_after_evaluation": sha256_file(weights_path)
        == weights_before,
        "evaluated_utc_unix": time.time(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
    }
    if not result["weights_unchanged_after_evaluation"]:
        msg = "frozen weights changed during test evaluation"
        raise RuntimeError(msg)
    write_json(complete, result)
    write_json(
        output / "test_access_audit.json",
        {
            "split": "test",
            "test_input_manifest": str(run / "test_input_manifest.json"),
            "frozen_weights_sha256": weights_before,
            "training_or_optimization_performed": False,
            "device": str(device),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one frozen ETM-family model on test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)
    result = evaluate_test(
        args.run,
        method=args.method,
        device=resolve_torch_device(args.device),
        batch_size=args.batch_size,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
