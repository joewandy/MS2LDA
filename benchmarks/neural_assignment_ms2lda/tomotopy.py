"""Self-contained training and held-out evaluation of the Tomotopy comparator."""

from __future__ import annotations

import os
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

ALPHA_OPTIMIZATION_INTERVAL = 10


def _tomotopy() -> Any:
    """Import the pinned optional comparator dependency with a useful error."""
    try:
        import tomotopy as tp
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError("tomotopy==0.13.0 is required for the comparator") from exc
    return tp


def _documents(matrix: Any, vocabulary: Sequence[str]) -> list[list[str]]:
    """Expand a sparse count matrix into Tomotopy token documents."""
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
    """Return whether every recent relative perplexity change is small."""
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
    """Reorder Tomotopy's topic-word matrix to the frozen training vocabulary."""
    columns = {str(word): index for index, word in enumerate(model_vocabulary)}
    if set(columns) != set(vocabulary):
        raise ValueError("Tomotopy vocabulary differs from the frozen vocabulary")
    aligned = np.column_stack([beta[:, columns[word]] for word in vocabulary])
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32, copy=False)


def _alpha_evidence(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the learned asymmetric alpha and record its initialization."""
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


def _model_beta(model: Any, vocabulary: Sequence[str]) -> np.ndarray:
    """Extract and align the dense topic-word probability matrix."""
    raw = np.vstack(
        [
            np.asarray(model.get_topic_word_dist(topic), dtype=np.float32)
            for topic in range(model.k)
        ]
    )
    return _align_beta(raw, list(model.used_vocabs), vocabulary)


def _infer_theta(
    model: Any,
    words: list[list[str]],
    *,
    iterations: int,
    workers: int,
) -> np.ndarray:
    """Infer mixtures, using the learned prior only for empty spectra."""
    alpha = np.asarray(model.alpha, dtype=np.float32)
    prior = alpha / np.maximum(alpha.sum(), 1e-12)
    theta = np.broadcast_to(prior, (len(words), model.k)).copy()
    nonempty = [index for index, row in enumerate(words) if row]
    if nonempty:
        documents = [model.make_doc(words[index]) for index in nonempty]
        inferred, _ = model.infer(
            documents,
            iter=iterations,
            workers=workers,
            parallel=1,
            together=False,
        )
        theta[nonempty] = np.asarray(inferred, dtype=np.float32)
    return theta


def _latency(
    model: Any,
    words: list[list[str]],
    protocol: dict[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    """Measure warm resident-model batch inference after one warm-up call."""
    config = protocol["evaluation"]
    iterations = int(protocol["tomotopy"]["inference_iterations"])
    subset = [row for row in words if row][: int(config["latency_subset_size"])]
    if not subset:
        raise ValueError("no non-empty documents are available for latency measurement")
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


def train_tomotopy(  # noqa: PLR0915
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    heartbeat: Any = None,
) -> dict[str, Any]:
    """Train or exactly resume the paper-matched K=1000 LDA comparator."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "tomotopy"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        if file_sha256(output / "model.bin") != result["model_sha256"]:
            raise ValueError("Tomotopy model changed")
        return result

    tp = _tomotopy()
    output.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    vocabulary = load_vocabulary(data)
    train_words = _documents(load_csr(data / "train.npz"), vocabulary)
    config = protocol["tomotopy"]
    topics = int(protocol["model"]["num_topics"])
    workers = int(protocol["cpu_threads"])
    context_sha256 = object_sha256(
        {
            "seed": int(protocol["seed"]),
            "cpu_threads": workers,
            "tomotopy": config,
            "train_sha256": file_sha256(data / "train.npz"),
            "vocabulary_sha256": file_sha256(data / "vocabulary.json"),
        }
    )
    checkpoint_binary = output / "checkpoint.bin"
    checkpoint_metadata = output / "checkpoint.json"
    if checkpoint_metadata.is_file():
        metadata = read_json(checkpoint_metadata)
        if metadata["context_sha256"] != context_sha256:
            raise ValueError("Tomotopy checkpoint context changed")
        if file_sha256(checkpoint_binary) != metadata["model_sha256"]:
            raise ValueError("Tomotopy checkpoint binary changed")
        model = tp.LDAModel.load(str(checkpoint_binary))
        history = list(metadata["history"])
        trained = int(history[-1]["iteration"])
        elapsed_before = float(history[-1]["cumulative_training_seconds"])
    else:
        model = tp.LDAModel(
            k=topics,
            # The shared vocabulary has already applied the paper's min_df=3;
            # retaining every supplied column avoids filtering it a second time.
            min_df=1,
            min_cf=0,
            rm_top=0,
            alpha=float(config["alpha"]),
            eta=float(config["eta"]),
            seed=int(protocol["seed"]),
        )
        model.optim_interval = ALPHA_OPTIMIZATION_INTERVAL
        for words in train_words:
            model.add_doc(words)
        history = []
        trained = 0
        elapsed_before = 0.0

    started = time.perf_counter()
    maximum = int(config["maximum_iterations"])
    while trained < maximum:
        step = min(int(config["step_size"]), maximum - trained)
        model.train(step, workers=workers, parallel=int(config["parallel"]))
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
                "context_sha256": context_sha256,
                "model_sha256": file_sha256(checkpoint_binary),
                "history": history,
            },
        )
        if heartbeat is not None:
            heartbeat(
                stage="train_tomotopy",
                iteration=trained,
                maximum_iterations=maximum,
                perplexity=float(model.perplexity),
                elapsed_seconds=elapsed,
            )
        if _converged(
            history,
            window=int(config["convergence_window"]),
            threshold=float(config["convergence_threshold"]),
        ):
            break

    final_model = output / "model.bin"
    model.save(str(final_model))
    result = {
        "schema_version": "neural-ms2lda/tomotopy-training-v1",
        "topic_count": int(model.k),
        "training_iterations": trained,
        "converged": _converged(
            history,
            window=int(config["convergence_window"]),
            threshold=float(config["convergence_threshold"]),
        ),
        "training_workers": workers,
        "training_parallel": int(config["parallel"]),
        "training_seconds_total": float(history[-1]["cumulative_training_seconds"]),
        "alpha": _alpha_evidence(model, config),
        "peak_rss_bytes": peak_rss_bytes(),
        "model_sha256": file_sha256(final_model),
        "output_sha256": {"model.bin": file_sha256(final_model)},
    }
    write_json(complete_path, result)
    return result


def evaluate_tomotopy(  # noqa: PLR0915
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Evaluate the trained comparator on validation or on the locked test split."""
    if split not in {"validation", "test"}:
        raise ValueError("Tomotopy split must be validation or test")
    directory = Path(run_dir).expanduser().resolve()
    group = "validation_evaluation" if split == "validation" else "evaluation"
    output = directory / group / "tomotopy"
    complete_path = output / "complete.json"
    training = read_json(directory / "tomotopy/complete.json")
    model_path = directory / "tomotopy/model.bin"
    if file_sha256(model_path) != training["model_sha256"]:
        raise ValueError("Tomotopy model changed after training")
    if complete_path.is_file():
        result = read_json(complete_path)
        verify_output_hashes(output, result)
        if result["model_sha256"] != training["model_sha256"]:
            raise ValueError("Tomotopy evaluation source changed")
        return result

    tp = _tomotopy()
    model = tp.LDAModel.load(str(model_path))
    config = protocol["tomotopy"]
    topics = int(protocol["model"]["num_topics"])
    if int(model.k) != topics:
        raise ValueError("loaded Tomotopy topic count differs")
    if not np.isclose(float(model.eta), float(config["eta"])):
        raise ValueError("loaded Tomotopy eta differs")
    alpha = _alpha_evidence(model, config)
    workers = int(protocol["cpu_threads"])
    iterations = int(config["inference_iterations"])
    data = directory / "data"
    vocabulary = load_vocabulary(data)
    train = load_csr(data / "train.npz")
    observed = load_csr(data / f"{split}_observed.npz")
    completion = load_csr(data / f"{split}_completion.npz")
    full = load_csr(data / f"{split}_full.npz")
    records = load_heldout_records(data, split)
    observed_words = _documents(observed, vocabulary)
    full_words = _documents(full, vocabulary)
    started = time.perf_counter()
    beta = _model_beta(model, vocabulary)
    theta = _infer_theta(model, observed_words, iterations=iterations, workers=workers)
    full_theta = _infer_theta(model, full_words, iterations=iterations, workers=workers)
    completion_summary, losses = completion_metrics(theta, beta, completion, records)
    stable = all(np.isfinite(values).all() for values in (beta, theta, full_theta))
    stable = stable and np.isfinite(completion_summary["nll_per_token"])
    if not stable:
        raise FloatingPointError("Tomotopy produced non-finite evaluation evidence")

    arrays = {
        "beta.npy": beta,
        f"{split}_observed_theta.npy": theta,
        f"{split}_full_theta.npy": full_theta,
        f"{split}_completion_nll.npy": losses,
    }
    maximum = full_theta.max(axis=1)
    confident = maximum >= float(protocol["chemistry"]["membership_threshold"])
    if split == "validation":
        metrics = {
            "validation_document_completion": completion_summary,
            "high_confidence_spectra": int(confident.sum()),
            "distinct_high_confidence_topics": int(
                np.unique(full_theta[confident].argmax(axis=1)).size
            ),
            "full_spectrum_mixture": effective_topic_summary(full_theta),
        }
        schema = "neural-ms2lda/validation-evaluation-v1"
    else:
        usage = full_theta.mean(axis=0)
        metrics = {
            "test_document_completion": completion_summary,
            "active_topics": active_topic_metrics(
                theta,
                document_threshold=float(
                    protocol["evaluation"]["document_active_threshold"]
                ),
                corpus_threshold=1.0 / topics,
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
            "warm_in_memory_batch_inference": _latency(
                model, observed_words, protocol, workers=workers
            ),
        }
        schema = "neural-ms2lda/tomotopy-evaluation-v1"

    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "schema_version": schema,
        "method": "tomotopy",
        "split": split,
        "topic_count": int(model.k),
        "model_sha256": training["model_sha256"],
        "stable": True,
        "training_iterations": int(training["training_iterations"]),
        "training_seconds_total": float(training["training_seconds_total"]),
        "training_workers": int(training["training_workers"]),
        "training_parallel": int(training["training_parallel"]),
        "inference_workers": workers,
        "inference_parallel": 1,
        "inference_iterations": iterations,
        "alpha": alpha,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {name: file_sha256(output / name) for name in arrays},
    }
    write_json(complete_path, result)
    return result
