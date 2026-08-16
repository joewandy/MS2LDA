"""Established Tomotopy K=1000 comparator, isolated from neural discovery."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import load_csr, load_heldout_records, load_vocabulary
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
    write_json,
)


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


def _converged(
    history: list[dict[str, float]], *, window: int, threshold: float
) -> bool:
    if len(history) <= window:
        return False
    perplexities = [row["perplexity"] for row in history]
    changes = [
        abs(perplexities[index] - perplexities[index - 1])
        / max(abs(perplexities[index - 1]), 1e-12)
        for index in range(1, len(perplexities))
    ]
    return all(change < threshold for change in changes[-window:])


def _align_beta(
    beta: np.ndarray, model_vocabulary: Sequence[str], vocabulary: Sequence[str]
) -> np.ndarray:
    columns = {str(word): index for index, word in enumerate(model_vocabulary)}
    if set(columns) != set(vocabulary):
        raise ValueError("Tomotopy vocabulary differs from the frozen vocabulary")
    aligned = np.column_stack([beta[:, columns[word]] for word in vocabulary])
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32, copy=False)


def _latency(
    model: Any, words: list[list[str]], protocol: dict[str, Any]
) -> dict[str, Any]:
    config = protocol["evaluation"]
    iterations = int(protocol["tomotopy"]["inference_iterations"])
    subset = [row for row in words if row][: int(config["latency_subset_size"])]
    if not subset:
        raise ValueError("No non-empty documents are available for latency measurement")
    documents = [model.make_doc(row) for row in subset]
    model.infer(documents, iter=iterations, workers=1, parallel=1, together=False)
    durations = []
    for _ in range(int(config["latency_repeats"])):
        started = time.perf_counter()
        model.infer(documents, iter=iterations, workers=1, parallel=1, together=False)
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
    }


def _infer_theta(model: Any, words: list[list[str]], *, iterations: int) -> np.ndarray:
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
            workers=1,
            parallel=1,
            together=False,
        )
        theta[nonempty_indices] = np.asarray(inferred, dtype=np.float32)
    return theta


def train_and_evaluate_tomotopy(
    run_dir: str | Path, protocol: dict[str, Any], *, heartbeat: Any = None
) -> dict[str, Any]:
    """Train, resume, and evaluate the post-hoc K=1000 comparator."""
    try:
        import tomotopy as tp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tomotopy==0.13.0 is required for the comparator") from exc
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation/tomotopy"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Tomotopy artifact changed: {name}")
        return result
    output.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    vocabulary = load_vocabulary(data)
    train = load_csr(data / "train.npz")
    test_observed = load_csr(data / "test_observed.npz")
    test_completion = load_csr(data / "test_completion.npz")
    test_full = load_csr(data / "test_full.npz")
    test_records = load_heldout_records(data, "test")
    train_words = _documents(train, vocabulary)
    observed_words = _documents(test_observed, vocabulary)
    full_words = _documents(test_full, vocabulary)
    config = protocol["tomotopy"]
    context_hash = object_sha256(
        {
            "protocol": protocol,
            "train_sha256": file_sha256(data / "train.npz"),
            "vocabulary_sha256": file_sha256(data / "vocabulary.json"),
        }
    )
    checkpoint_binary = output / "checkpoint.bin"
    checkpoint_metadata = output / "checkpoint.json"
    if checkpoint_metadata.is_file():
        metadata = read_json(checkpoint_metadata)
        if metadata["context_sha256"] != context_hash:
            raise ValueError("Tomotopy checkpoint context changed")
        if file_sha256(checkpoint_binary) != metadata["model_sha256"]:
            raise ValueError("Tomotopy checkpoint binary changed")
        model = tp.LDAModel.load(str(checkpoint_binary))
        history = list(metadata["history"])
        trained = int(history[-1]["iteration"])
        elapsed_before = float(history[-1]["cumulative_training_seconds"])
    else:
        model = tp.LDAModel(
            k=int(config["num_topics"]),
            min_df=1,
            min_cf=0,
            rm_top=0,
            alpha=float(config["alpha"]),
            eta=float(config["eta"]),
            seed=int(protocol["seed"]),
        )
        for words in train_words:
            model.add_doc(words)
        history = []
        trained = 0
        elapsed_before = 0.0
    started = time.perf_counter()
    while trained < int(config["maximum_iterations"]):
        step = min(
            int(config["step_size"]), int(config["maximum_iterations"]) - trained
        )
        model.train(
            step,
            workers=int(config["workers"]),
            parallel=int(config["parallel"]),
        )
        trained += step
        elapsed = elapsed_before + time.perf_counter() - started
        history.append(
            {
                "iteration": trained,
                "ll_per_word": float(model.ll_per_word),
                "perplexity": float(model.perplexity),
                "cumulative_training_seconds": elapsed,
            }
        )
        temporary = output / f".checkpoint.{os.getpid()}.tmp"
        model.save(str(temporary))
        os.replace(temporary, checkpoint_binary)
        write_json(
            checkpoint_metadata,
            {
                "schema_version": "neural-ms2lda/tomotopy-checkpoint-v1",
                "context_sha256": context_hash,
                "model_sha256": file_sha256(checkpoint_binary),
                "history": history,
            },
        )
        if heartbeat is not None:
            heartbeat(
                stage="train_tomotopy",
                iteration=trained,
                maximum_iterations=int(config["maximum_iterations"]),
                perplexity=float(model.perplexity),
                elapsed_seconds=elapsed,
            )
        if _converged(
            history,
            window=int(config["convergence_window"]),
            threshold=float(config["convergence_threshold"]),
        ):
            break
    raw_beta = np.vstack(
        [
            np.asarray(model.get_topic_word_dist(topic), dtype=np.float32)
            for topic in range(model.k)
        ]
    )
    beta = _align_beta(raw_beta, list(model.used_vocabs), vocabulary)
    theta = _infer_theta(
        model, observed_words, iterations=int(config["inference_iterations"])
    )
    full_theta = _infer_theta(
        model, full_words, iterations=int(config["inference_iterations"])
    )
    completion, losses = completion_metrics(theta, beta, test_completion, test_records)
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
        "cached_latency": _latency(model, observed_words, protocol),
    }
    arrays = {
        "beta.npy": beta,
        "test_theta.npy": theta,
        "test_full_theta.npy": full_theta,
        "test_completion_nll.npy": losses,
    }
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    final_model = output / "model.bin"
    model.save(str(final_model))
    result = {
        "schema_version": "neural-ms2lda/tomotopy-evaluation-v1",
        "method": "tomotopy_k1000_comparator",
        "topic_count": int(model.k),
        "training_iterations": trained,
        "converged": _converged(
            history,
            window=int(config["convergence_window"]),
            threshold=float(config["convergence_threshold"]),
        ),
        "training_workers": int(config["workers"]),
        "training_parallel": int(config["parallel"]),
        "training_seconds_total": float(history[-1]["cumulative_training_seconds"]),
        "metrics": metrics,
        "peak_rss_bytes": peak_rss_bytes(),
        "model_sha256": file_sha256(final_model),
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result
