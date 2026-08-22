"""Self-contained training and held-out evaluation of the Tomotopy comparator."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import load_csr, load_heldout_records, load_vocabulary
from .objectives import completion_metrics
from .utils import (
    atomic_save_numpy,
    read_json,
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


def _validate_alpha(model: Any) -> None:
    """Check the learned asymmetric alpha used for held-out inference."""
    alpha = np.asarray(model.alpha, dtype=np.float64)
    if alpha.shape != (int(model.k),) or not np.isfinite(alpha).all():
        raise ValueError("loaded Tomotopy alpha vector is invalid")
    if np.any(alpha <= 0):
        raise ValueError("loaded Tomotopy alpha vector is not positive")
    if int(model.optim_interval) != ALPHA_OPTIMIZATION_INTERVAL:
        raise ValueError("loaded Tomotopy alpha optimization interval differs")


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
        "median_spectra_per_second": 1.0 / median,
    }


def train_tomotopy(  # noqa: PLR0915
    run_dir: str | Path,
    protocol: dict[str, Any],
    *,
    heartbeat: Any = None,
) -> dict[str, Any]:
    """Train the paper-matched K=1000 LDA comparator."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "tomotopy"
    complete_path = output / "complete.json"
    if complete_path.is_file() and (output / "model.bin").is_file():
        return read_json(complete_path)

    tp = _tomotopy()
    output.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    vocabulary = load_vocabulary(data)
    train_words = _documents(load_csr(data / "train.npz"), vocabulary)
    config = protocol["tomotopy"]
    topics = int(protocol["model"]["num_topics"])
    workers = int(protocol["cpu_threads"])
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

    started = time.perf_counter()
    maximum = int(config["maximum_iterations"])
    while trained < maximum:
        step = min(int(config["step_size"]), maximum - trained)
        model.train(step, workers=workers, parallel=int(config["parallel"]))
        trained += step
        elapsed = time.perf_counter() - started
        history.append(
            {
                "iteration": trained,
                "ll_per_word": float(model.ll_per_word),
                "perplexity": float(model.perplexity),
                "cumulative_training_seconds": elapsed,
            }
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
        "training_iterations": trained,
        "converged": _converged(
            history,
            window=int(config["convergence_window"]),
            threshold=float(config["convergence_threshold"]),
        ),
        "training_seconds_total": float(history[-1]["cumulative_training_seconds"]),
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
    model_path = directory / "tomotopy/model.bin"
    if complete_path.is_file():
        return read_json(complete_path)

    tp = _tomotopy()
    model = tp.LDAModel.load(str(model_path))
    config = protocol["tomotopy"]
    topics = int(protocol["model"]["num_topics"])
    if int(model.k) != topics:
        raise ValueError("loaded Tomotopy topic count differs")
    if not np.isclose(float(model.eta), float(config["eta"])):
        raise ValueError("loaded Tomotopy eta differs")
    _validate_alpha(model)
    workers = int(protocol["cpu_threads"])
    iterations = int(config["inference_iterations"])
    data = directory / "data"
    vocabulary = load_vocabulary(data)
    observed = load_csr(data / f"{split}_observed.npz")
    completion = load_csr(data / f"{split}_completion.npz")
    full = load_csr(data / f"{split}_full.npz")
    records = load_heldout_records(data, split)
    observed_words = _documents(observed, vocabulary)
    full_words = _documents(full, vocabulary)
    beta = _model_beta(model, vocabulary)
    theta = _infer_theta(model, observed_words, iterations=iterations, workers=workers)
    full_theta = _infer_theta(model, full_words, iterations=iterations, workers=workers)
    completion_summary = completion_metrics(theta, beta, completion, records)
    stable = all(np.isfinite(values).all() for values in (beta, theta, full_theta))
    stable = stable and np.isfinite(completion_summary["nll_per_token"])
    if not stable:
        raise FloatingPointError("Tomotopy produced non-finite evaluation evidence")

    arrays = {f"{split}_full_theta.npy": full_theta}
    if split == "validation":
        arrays["beta.npy"] = beta
        metrics = {"validation_document_completion": completion_summary}
    else:
        metrics = {
            "test_document_completion": completion_summary,
            "warm_in_memory_batch_inference": _latency(
                model, observed_words, protocol, workers=workers
            ),
        }

    output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_save_numpy(output / name, values)
    result = {
        "method": "tomotopy",
        "split": split,
        "metrics": metrics,
    }
    write_json(complete_path, result)
    return result
