"""Hash-verified evaluation of the established Tomotopy comparator."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import load_csr, load_heldout_records, load_vocabulary
from .inventory import topic_inventory_summary
from .metrics import (
    active_topic_metrics,
    completion_metrics,
    effective_topic_summary,
    sparse_npmi,
    top_word_diversity,
)
from .utils import (
    atomic_save_numpy,
    file_sha256,
    object_sha256,
    peak_rss_bytes,
    read_json,
    verify_output_hashes,
    write_json,
)

REFERENCE_DATA_FILES = (
    "train.npz",
    "test_observed.npz",
    "test_completion.npz",
    "test_full.npz",
    "vocabulary.json",
    "identifiers.json",
)
ALPHA_OPTIMIZATION_INTERVAL = 10


def _documents(matrix: Any, vocabulary: Sequence[str]) -> list[list[str]]:
    documents: list[list[str]] = []
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        words: list[str] = []
        for column, count in zip(
            matrix.indices[start:stop], matrix.data[start:stop], strict=True
        ):
            words.extend([vocabulary[int(column)]] * int(round(float(count))))
        documents.append(words)
    return documents


def _align_beta(
    beta: np.ndarray, model_vocabulary: Sequence[str], vocabulary: Sequence[str]
) -> np.ndarray:
    columns = {str(word): index for index, word in enumerate(model_vocabulary)}
    if set(columns) != set(vocabulary):
        raise ValueError("Tomotopy vocabulary differs from the frozen vocabulary")
    aligned = np.column_stack([beta[:, columns[word]] for word in vocabulary])
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32, copy=False)


def _alpha_evidence(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the learned alpha vector and record its scalar initializer."""
    alpha = np.asarray(model.alpha, dtype=np.float64)
    if alpha.shape != (int(model.k),) or not np.isfinite(alpha).all():
        raise ValueError("loaded Tomotopy alpha vector is invalid")
    if np.any(alpha <= 0):
        raise ValueError("loaded Tomotopy alpha vector is not positive")
    if int(model.optim_interval) != ALPHA_OPTIMIZATION_INTERVAL:
        raise ValueError("loaded Tomotopy alpha optimization interval differs")
    return {
        "initial_value": float(config["alpha"]),
        "optimization_interval": int(model.optim_interval),
        "learned_vector_sha256": object_sha256(alpha.tolist()),
        "learned_minimum": float(alpha.min()),
        "learned_maximum": float(alpha.max()),
    }


def tomotopy_reference_evidence(
    reference_run: str | Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Validate and fingerprint the frozen six-worker Tomotopy training run."""
    root = Path(reference_run).expanduser().resolve()
    source_protocol_path = root / "protocol.resolved.json"
    source_data_path = root / "data/complete.json"
    source_evaluation_path = root / "evaluation/tomotopy/complete.json"
    model_path = root / "evaluation/tomotopy/model.bin"
    required = (
        source_protocol_path,
        source_data_path,
        source_evaluation_path,
        model_path,
        *(root / "data" / name for name in REFERENCE_DATA_FILES),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tomotopy reference is incomplete: {missing}")

    source_protocol = read_json(source_protocol_path)
    source_config = source_protocol["tomotopy"]
    current_config = protocol["tomotopy"]
    compared = (
        "num_topics",
        "alpha",
        "eta",
        "maximum_iterations",
        "step_size",
        "convergence_window",
        "convergence_threshold",
        "inference_iterations",
        "parallel",
    )
    changed = [
        key for key in compared if source_config.get(key) != current_config.get(key)
    ]
    if int(source_protocol["seed"]) != int(protocol["seed"]):
        changed.append("seed")
    workers = int(protocol["cpu_threads"])
    if int(source_config.get("workers", 0)) != workers:
        changed.append("workers")
    if changed:
        raise ValueError(f"Tomotopy reference protocol differs: {sorted(changed)}")

    source_data = read_json(source_data_path)
    data_sha256 = {
        name: file_sha256(root / "data" / name) for name in REFERENCE_DATA_FILES
    }
    for name, digest in data_sha256.items():
        if source_data["output_sha256"].get(name) != digest:
            raise ValueError(f"Tomotopy reference data changed: {name}")

    source_evaluation = read_json(source_evaluation_path)
    model_sha256 = file_sha256(model_path)
    if source_evaluation.get("model_sha256") != model_sha256:
        raise ValueError("Tomotopy reference model changed")
    expected_training = {
        "topic_count": int(current_config["num_topics"]),
        "training_workers": workers,
        "training_parallel": int(current_config["parallel"]),
    }
    for key, expected in expected_training.items():
        if int(source_evaluation.get(key, -1)) != expected:
            raise ValueError(f"Tomotopy reference {key} differs")
    if source_evaluation.get("converged") is not True:
        raise ValueError("Tomotopy reference did not converge")

    return {
        "run": str(root),
        "model_sha256": model_sha256,
        "source_protocol_sha256": file_sha256(source_protocol_path),
        "source_data_manifest_sha256": file_sha256(source_data_path),
        "source_evaluation_sha256": file_sha256(source_evaluation_path),
        "data_sha256": data_sha256,
        "training_iterations": int(source_evaluation["training_iterations"]),
        "training_seconds_total": float(source_evaluation["training_seconds_total"]),
        "training_workers": int(source_evaluation["training_workers"]),
        "training_parallel": int(source_evaluation["training_parallel"]),
        "source_peak_rss_bytes": int(source_evaluation["peak_rss_bytes"]),
    }


def _latency(
    model: Any,
    words: list[list[str]],
    protocol: dict[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    config = protocol["evaluation"]
    iterations = int(protocol["tomotopy"]["inference_iterations"])
    subset = [row for row in words if row][: int(config["latency_subset_size"])]
    if not subset:
        raise ValueError("No non-empty documents are available for latency measurement")
    documents = [model.make_doc(row) for row in subset]
    model.infer(
        documents,
        iter=iterations,
        workers=workers,
        parallel=1,
        together=False,
    )
    durations = []
    for _ in range(int(config["latency_repeats"])):
        started = time.perf_counter()
        model.infer(
            documents,
            iter=iterations,
            workers=workers,
            parallel=1,
            together=False,
        )
        durations.append(time.perf_counter() - started)
    per_spectrum = np.asarray(durations, dtype=np.float64) / len(documents)
    median = float(statistics.median(per_spectrum))
    return {
        "documents": len(documents),
        "repeats": len(durations),
        "median_seconds_per_spectrum": median,
        "median_spectra_per_second": 1.0 / median,
        "p95_seconds_per_spectrum": float(np.percentile(per_spectrum, 95)),
        "inference_iterations": iterations,
        "workers": workers,
        "parallel": 1,
    }


def _infer_theta(
    model: Any,
    words: list[list[str]],
    *,
    iterations: int,
    workers: int,
) -> np.ndarray:
    """Infer mixtures while assigning prior-only mixtures to empty documents."""
    alpha = np.asarray(model.alpha, dtype=np.float32)
    prior = alpha / np.maximum(alpha.sum(), 1e-12)
    theta = np.broadcast_to(prior, (len(words), model.k)).copy()
    nonempty_indices = [index for index, row in enumerate(words) if row]
    if nonempty_indices:
        documents = [model.make_doc(words[index]) for index in nonempty_indices]
        inferred, _ = model.infer(
            documents,
            iter=iterations,
            workers=workers,
            parallel=1,
            together=False,
        )
        theta[nonempty_indices] = np.asarray(inferred, dtype=np.float32)
    return theta


def evaluate_tomotopy_reference(
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    reference_run: str | Path,
    heartbeat: Any = None,
) -> dict[str, Any]:
    """Evaluate a frozen K=1000 model without retraining or modifying its run."""
    try:
        import tomotopy as tp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tomotopy==0.13.0 is required for the comparator") from exc

    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation/tomotopy"
    complete_path = output / "complete.json"
    reference = tomotopy_reference_evidence(reference_run, protocol)
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        if result.get("reference") != reference:
            raise ValueError("Tomotopy reference provenance changed")
        return result

    data = directory / "data"
    current_data = read_json(data / "complete.json")
    for name, expected in reference["data_sha256"].items():
        actual = file_sha256(data / name)
        if actual != expected or current_data["output_sha256"].get(name) != actual:
            raise ValueError(f"current data differs from Tomotopy reference: {name}")

    model_path = Path(reference["run"]) / "evaluation/tomotopy/model.bin"
    model = tp.LDAModel.load(str(model_path))
    config = protocol["tomotopy"]
    if int(model.k) != int(config["num_topics"]):
        raise ValueError("loaded Tomotopy topic count differs")
    if not np.isclose(float(model.eta), float(config["eta"])):
        raise ValueError("loaded Tomotopy eta differs")
    alpha_evidence = _alpha_evidence(model, config)

    vocabulary = load_vocabulary(data)
    train = load_csr(data / "train.npz")
    test_observed = load_csr(data / "test_observed.npz")
    test_completion = load_csr(data / "test_completion.npz")
    test_full = load_csr(data / "test_full.npz")
    test_records = load_heldout_records(data, "test")
    observed_words = _documents(test_observed, vocabulary)
    full_words = _documents(test_full, vocabulary)
    workers = int(protocol["cpu_threads"])
    iterations = int(config["inference_iterations"])
    if heartbeat is not None:
        heartbeat(stage="evaluate_tomotopy_reference", workers=workers)

    started = time.perf_counter()
    raw_beta = np.vstack(
        [
            np.asarray(model.get_topic_word_dist(topic), dtype=np.float32)
            for topic in range(model.k)
        ]
    )
    beta = _align_beta(raw_beta, list(model.used_vocabs), vocabulary)
    theta = _infer_theta(
        model,
        observed_words,
        iterations=iterations,
        workers=workers,
    )
    full_theta = _infer_theta(
        model,
        full_words,
        iterations=iterations,
        workers=workers,
    )
    completion, losses = completion_metrics(theta, beta, test_completion, test_records)
    usage = full_theta.mean(axis=0)
    metrics = {
        "test_document_completion": completion,
        "active_topics": active_topic_metrics(
            theta,
            document_threshold=float(
                protocol["evaluation"]["document_active_threshold"]
            ),
            corpus_threshold=1.0 / int(config["num_topics"]),
        ),
        "full_spectrum_mixture": effective_topic_summary(full_theta),
        "top_word_diversity": top_word_diversity(
            beta, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "word_cooccurrence_npmi": sparse_npmi(
            beta, train, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "topic_inventory": topic_inventory_summary(
            beta, usage, top_n=int(protocol["evaluation"]["topic_top_n"])
        ),
        "cached_latency": _latency(
            model,
            observed_words,
            protocol,
            workers=workers,
        ),
    }
    arrays = {
        "beta.npy": beta,
        "test_theta.npy": theta,
        "test_full_theta.npy": full_theta,
        "test_completion_nll.npy": losses,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "schema_version": "neural-ms2lda/tomotopy-evaluation-v1",
        "method": "tomotopy",
        "topic_count": int(model.k),
        "training_reused": True,
        "training_iterations": reference["training_iterations"],
        "training_seconds_total": reference["training_seconds_total"],
        "training_workers": reference["training_workers"],
        "training_parallel": reference["training_parallel"],
        "inference_workers": workers,
        "inference_parallel": 1,
        "inference_iterations": iterations,
        "alpha": alpha_evidence,
        "evaluation_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "peak_rss_bytes": peak_rss_bytes(),
        "source_peak_rss_bytes": reference["source_peak_rss_bytes"],
        "reference": reference,
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result


def evaluate_tomotopy_validation_reference(
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    reference_run: str | Path,
) -> dict[str, Any]:
    """Evaluate the frozen comparator on validation, without candidate test access."""
    try:
        import tomotopy as tp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tomotopy==0.13.0 is required for the comparator") from exc

    directory = Path(run_dir).expanduser().resolve()
    output = directory / "validation_evaluation/tomotopy"
    complete_path = output / "complete.json"
    reference = tomotopy_reference_evidence(reference_run, protocol)
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        if result.get("reference") != reference:
            raise ValueError("Tomotopy validation reference provenance changed")
        return result
    data = directory / "data"
    for name in ("train.npz", "vocabulary.json"):
        if file_sha256(data / name) != reference["data_sha256"][name]:
            raise ValueError(f"validation data differs from Tomotopy reference: {name}")

    model_path = Path(reference["run"]) / "evaluation/tomotopy/model.bin"
    model = tp.LDAModel.load(str(model_path))
    config = protocol["tomotopy"]
    if int(model.k) != int(config["num_topics"]):
        raise ValueError("loaded Tomotopy topic count differs")
    _alpha_evidence(model, config)
    vocabulary = load_vocabulary(data)
    observed = load_csr(data / "validation_observed.npz")
    completion = load_csr(data / "validation_completion.npz")
    full = load_csr(data / "validation_full.npz")
    records = load_heldout_records(data, "validation")
    workers = int(protocol["cpu_threads"])
    iterations = int(config["inference_iterations"])
    started = time.perf_counter()
    raw_beta = np.vstack(
        [
            np.asarray(model.get_topic_word_dist(topic), dtype=np.float32)
            for topic in range(model.k)
        ]
    )
    beta = _align_beta(raw_beta, list(model.used_vocabs), vocabulary)
    theta = _infer_theta(
        model,
        _documents(observed, vocabulary),
        iterations=iterations,
        workers=workers,
    )
    full_theta = _infer_theta(
        model,
        _documents(full, vocabulary),
        iterations=iterations,
        workers=workers,
    )
    document_completion, losses = completion_metrics(theta, beta, completion, records)
    maximum = full_theta.max(axis=1)
    confident = maximum >= 0.5
    arrays = {
        "beta.npy": beta,
        "validation_observed_theta.npy": theta,
        "validation_full_theta.npy": full_theta,
        "validation_completion_nll.npy": losses,
    }
    if not all(np.isfinite(values).all() for values in (beta, theta, full_theta)):
        raise FloatingPointError("Tomotopy produced non-finite validation evidence")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "schema_version": "neural-ms2lda/validation-evaluation-v1",
        "method": "tomotopy",
        "split": "validation",
        "topic_count": int(model.k),
        "source_sha256": reference["model_sha256"],
        "stable": True,
        "metrics": {
            "validation_document_completion": document_completion,
            "high_confidence_spectra": int(confident.sum()),
            "distinct_high_confidence_topics": int(
                np.unique(full_theta[confident].argmax(axis=1)).size
            ),
            "full_spectrum_mixture": effective_topic_summary(full_theta),
        },
        "evaluation_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "reference": reference,
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result
