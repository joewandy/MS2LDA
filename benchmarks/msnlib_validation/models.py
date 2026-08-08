"""Deterministic Tomotopy and HybridLDA benchmark workers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import (
    BenchmarkConfig,
    file_sha256,
    load_config,
    object_sha256,
    read_json,
    write_json,
)
from .data import (
    SpectrumRecord,
    load_records,
    renormalize_peak_groups,
    split_records,
    to_matchms_spectrum,
)
from .features import prepare_features
from .metrics import (
    active_topic_metrics,
    convergence_metrics,
    document_completion_nll,
    top_word_diversity,
    word_cooccurrence_npmi,
)
from .protocol import (
    load_assignments,
    load_completion_rows,
    load_vocabulary,
    verify_protocol,
)
from .runtime import load_feature_cache, peak_rss_bytes


@dataclass(frozen=True)
class PreparedDocuments:
    """Chemistry-free model inputs aligned with cached embeddings."""

    train_ids: tuple[str, ...]
    train_words: tuple[list[str], ...]
    train_embeddings: np.ndarray
    validation_ids: tuple[str, ...]
    validation_observed_words: tuple[list[str], ...]
    validation_embeddings: np.ndarray
    test_ids: tuple[str, ...]
    test_observed_words: tuple[list[str], ...]
    test_completion_words: tuple[list[str], ...]
    test_embeddings: np.ndarray
    test_observed_records: tuple[SpectrumRecord, ...]


def _verify_completed_core_result(
    output: Path,
    *,
    method: str,
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Verify a completed core result before it is skipped or consumed."""
    result = read_json(output / "complete.json")
    expected = {
        "evidence_scope": config.evidence_scope,
        "method": method,
        "seed": seed,
        "topic_count": config.num_topics,
    }
    if method == "tomotopy":
        expected.update(
            {
                "training_parallel_scheme_value": config.tomotopy_training_parallel,
                "training_workers_requested": config.tomotopy_training_workers,
            }
        )
    elif method == "hybrid":
        expected.update(
            {
                "inference_cpu_threads": config.hybrid_inference_cpu_threads,
                "training_cpu_threads": config.hybrid_training_cpu_threads,
            }
        )
    else:
        raise ValueError(f"unsupported core method: {method}")
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"completed {method} result has changed field: {name}")

    declared = {
        "beta.npy": result["beta_sha256"],
        "model.bin" if method == "tomotopy" else "model.pt": result["model_sha256"],
    }
    if method == "tomotopy":
        declared["test_theta.npy"] = result["theta_sha256"]
    else:
        reference_steps = str(result["reference_steps"])
        expected_steps = {"0", "2", reference_steps}
        theta_hashes = result.get("theta_sha256")
        if not isinstance(theta_hashes, dict) or set(theta_hashes) != expected_steps:
            raise ValueError("completed Hybrid result has invalid theta hashes")
        declared.update(
            {f"test_theta_{steps}.npy": theta_hashes[steps] for steps in expected_steps}
        )
    for name, digest in declared.items():
        path = output / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"completed {method} artifact changed: {name}")
    return result


def _completion_groups(
    record: SpectrumRecord, row: dict[str, Any], *, significant_digits: int
) -> tuple[list[str], list[str], SpectrumRecord]:
    if not row.get("eligible"):
        raise ValueError(f"test spectrum is ineligible: {record.spectrum_id}")
    observed_indices = set(map(int, row["observed_peak_indices"]))
    completion_indices = set(map(int, row["completion_peak_indices"]))
    all_indices = {group.original_index for group in record.peak_groups}
    if (
        observed_indices & completion_indices
        or (observed_indices | completion_indices) != all_indices
    ):
        raise ValueError(f"completion groups changed: {record.spectrum_id}")
    observed_groups = tuple(
        group
        for group in record.peak_groups
        if group.original_index in observed_indices
    )
    completion_groups = tuple(
        group
        for group in record.peak_groups
        if group.original_index in completion_indices
    )
    observed_groups = renormalize_peak_groups(
        observed_groups,
        precursor_mz=record.precursor_mz,
        significant_digits=significant_digits,
    )
    from dataclasses import replace

    observed_record = replace(record, peak_groups=observed_groups)
    observed_words = [token for group in observed_groups for token in group.tokens]
    completion_words = [token for group in completion_groups for token in group.tokens]
    return observed_words, completion_words, observed_record


def prepare_documents(run_dir: str | Path) -> PreparedDocuments:
    """Build model inputs while deliberately omitting all chemical labels."""
    lock = verify_protocol(run_dir)
    directory = Path(run_dir).expanduser().resolve()
    config = load_config(directory / "config.resolved.json")
    mgf_path = Path(lock["data_root"]) / config.input_files["mgf"]["relative_path"]
    records, _ = load_records(mgf_path, config)
    assignments = load_assignments(directory)
    completion_rows = load_completion_rows(directory)
    vocabulary = set(load_vocabulary(directory))
    feature_ids, global_embeddings, _, _ = load_feature_cache(directory)
    row_by_id = {identifier: row for row, identifier in enumerate(feature_ids)}
    if len(row_by_id) != len(feature_ids):
        raise ValueError("feature identifiers are not unique")
    train = split_records(records, assignments, "train")
    validation = split_records(records, assignments, "validation")
    test = split_records(records, assignments, "test")
    train_words = tuple(
        [word for word in record.words if word in vocabulary] for record in train
    )
    if any(not document for document in train_words):
        raise ValueError("training-only vocabulary created an empty training document")
    observed_by_split: dict[str, list[list[str]]] = {
        "validation": [],
        "test": [],
    }
    completion_by_split: dict[str, list[list[str]]] = {
        "validation": [],
        "test": [],
    }
    observed_records_by_split: dict[str, list[SpectrumRecord]] = {
        "validation": [],
        "test": [],
    }
    for split, split_rows in (("validation", validation), ("test", test)):
        for record in split_rows:
            observed, completion, observed_record = _completion_groups(
                record,
                completion_rows[record.spectrum_id],
                significant_digits=config.significant_digits,
            )
            # Preserve raw OOV words so both backends can construct a valid
            # document even when every observed word is outside the frozen
            # vocabulary. Each backend excludes those words internally.
            observed_by_split[split].append(observed)
            completion_by_split[split].append(completion)
            observed_records_by_split[split].append(observed_record)
    validation_embeddings = np.asarray(
        global_embeddings[[row_by_id[record.spectrum_id] for record in validation]],
        dtype=np.float32,
    )
    test_embeddings = np.asarray(
        global_embeddings[[row_by_id[record.spectrum_id] for record in test]],
        dtype=np.float32,
    )
    train_embeddings = np.asarray(
        global_embeddings[[row_by_id[record.spectrum_id] for record in train]],
        dtype=np.float32,
    )
    return PreparedDocuments(
        train_ids=tuple(record.spectrum_id for record in train),
        train_words=train_words,
        train_embeddings=train_embeddings,
        validation_ids=tuple(record.spectrum_id for record in validation),
        validation_observed_words=tuple(observed_by_split["validation"]),
        validation_embeddings=validation_embeddings,
        test_ids=tuple(record.spectrum_id for record in test),
        test_observed_words=tuple(observed_by_split["test"]),
        test_completion_words=tuple(completion_by_split["test"]),
        test_embeddings=test_embeddings,
        test_observed_records=tuple(observed_records_by_split["test"]),
    )


def _aligned_beta(
    beta: np.ndarray,
    model_vocabulary: Sequence[str],
    frozen_vocabulary: Sequence[str],
) -> np.ndarray:
    columns = {str(word): index for index, word in enumerate(model_vocabulary)}
    missing = [word for word in frozen_vocabulary if word not in columns]
    extra = set(columns) - set(frozen_vocabulary)
    if missing or extra:
        raise ValueError(
            f"model vocabulary mismatch: missing={len(missing)} extra={len(extra)}"
        )
    aligned = np.column_stack([beta[:, columns[word]] for word in frozen_vocabulary])
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32, copy=False)


def _converged(
    history: Sequence[dict[str, float]], *, window: int, threshold: float
) -> bool:
    if len(history) <= window:
        return False
    values = [row["perplexity"] for row in history]
    changes = [
        abs(values[index] - values[index - 1]) / max(abs(values[index - 1]), 1e-12)
        for index in range(1, len(values))
    ]
    return all(change < threshold for change in changes[-window:])


def _alpha_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Return compact diagnostics for one document-topic prior vector."""
    alpha = np.asarray(values, dtype=np.float64)
    if alpha.ndim != 1 or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
        raise ValueError("alpha must be a positive finite vector")
    return {
        "sum": float(alpha.sum()),
        "minimum": float(alpha.min()),
        "median": float(np.median(alpha)),
        "maximum": float(alpha.max()),
    }


def _latency_summary(durations: Sequence[float], count: int) -> dict[str, float | int]:
    values = np.asarray(durations, dtype=np.float64) / max(count, 1)
    return {
        "documents": count,
        "repeats": len(values),
        "median_seconds_per_spectrum": float(np.median(values)),
        "p95_seconds_per_spectrum": float(np.percentile(values, 95)),
        "median_spectra_per_second": float(1.0 / max(np.median(values), 1e-12)),
    }


def _measure(call: Callable[[], Any], *, repeats: int) -> list[float]:
    call()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
    return durations


def _common_metrics(
    *,
    theta: np.ndarray,
    beta: np.ndarray,
    heldout_words: Sequence[Sequence[str]],
    vocabulary: Sequence[str],
    train_words: Sequence[Sequence[str]],
    config,
) -> dict[str, Any]:
    return {
        "document_completion": document_completion_nll(
            theta, beta, heldout_words, vocabulary
        ),
        "active_topics": active_topic_metrics(
            theta,
            document_threshold=config.document_active_threshold,
            corpus_threshold=config.corpus_active_threshold,
        ),
        "top_word_diversity": top_word_diversity(beta, top_n=config.topic_top_n),
        "word_cooccurrence_npmi": word_cooccurrence_npmi(
            beta,
            train_words,
            vocabulary,
            top_n=config.topic_top_n,
        ),
    }


def _tomotopy_checkpoint_sidecars(output: Path) -> list[Path]:
    """Return Tomotopy checkpoint generations newest first."""
    return sorted((output / "checkpoints").glob("checkpoint-*.json"), reverse=True)


def _verified_tomotopy_checkpoint(
    sidecar: Path, *, context_sha256: str
) -> dict[str, Any]:
    """Verify one self-contained Tomotopy checkpoint generation."""
    metadata = read_json(sidecar)
    if metadata.get("context_sha256") != context_sha256:
        raise ValueError("Tomotopy checkpoint context hash mismatch")
    binary = sidecar.parent / str(metadata.get("file", ""))
    if binary.parent != sidecar.parent or binary.suffix != ".bin":
        raise ValueError("Tomotopy checkpoint filename escapes its directory")
    if not binary.is_file():
        raise FileNotFoundError(f"Tomotopy checkpoint payload is missing: {binary}")
    if binary.stat().st_size != metadata.get("bytes"):
        raise ValueError("Tomotopy checkpoint byte size mismatch")
    if file_sha256(binary) != metadata.get("sha256"):
        raise ValueError("Tomotopy checkpoint SHA-256 mismatch")
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Tomotopy checkpoint history is empty")
    if int(history[-1].get("iteration", -1)) != int(metadata.get("iteration", -2)):
        raise ValueError("Tomotopy checkpoint history iteration mismatch")
    return metadata


def _save_tomotopy_checkpoint(
    model: Any,
    output: Path,
    *,
    context_sha256: str,
    history: Sequence[dict[str, Any]],
    keep: int = 2,
) -> dict[str, Any]:
    """Publish one atomic, hash-verified Tomotopy checkpoint generation."""
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sequences = []
    for path in _tomotopy_checkpoint_sidecars(output):
        try:
            sequences.append(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    sequence = max(sequences, default=0) + 1
    stem = f"checkpoint-{sequence:06d}"
    binary = checkpoint_dir / f"{stem}.bin"
    temporary = checkpoint_dir / f".{stem}.{os.getpid()}.tmp"
    try:
        model.save(str(temporary))
        os.replace(temporary, binary)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "schema_version": "tomotopy-checkpoint/v1",
        "sequence": sequence,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context_sha256": context_sha256,
        "file": binary.name,
        "bytes": binary.stat().st_size,
        "sha256": file_sha256(binary),
        "iteration": int(history[-1]["iteration"]),
        "cumulative_training_seconds": float(
            history[-1]["cumulative_training_seconds"]
        ),
        "history": list(history),
    }
    sidecar = checkpoint_dir / f"{stem}.json"
    write_json(sidecar, metadata)
    write_json(checkpoint_dir / "latest.json", metadata)

    valid: list[tuple[Path, dict[str, Any]]] = []
    for candidate in _tomotopy_checkpoint_sidecars(output):
        try:
            candidate_metadata = _verified_tomotopy_checkpoint(
                candidate, context_sha256=context_sha256
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        valid.append((candidate, candidate_metadata))
    for obsolete, obsolete_metadata in valid[keep:]:
        (obsolete.parent / obsolete_metadata["file"]).unlink(missing_ok=True)
        obsolete.unlink(missing_ok=True)
    return metadata


def _restore_tomotopy_checkpoint(
    tp: Any, output: Path, *, context_sha256: str
) -> tuple[Any | None, list[dict[str, Any]], dict[str, Any]]:
    """Load the newest valid generation, falling back after corruption."""
    candidates = _tomotopy_checkpoint_sidecars(output)
    rejected = []
    for sidecar in candidates:
        try:
            metadata = _verified_tomotopy_checkpoint(
                sidecar, context_sha256=context_sha256
            )
            model = tp.LDAModel.load(str(sidecar.parent / metadata["file"]))
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            rejected.append(
                {"sidecar": sidecar.name, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        audit = {
            "selected_checkpoint": metadata,
            "rejected_newer_checkpoints": rejected,
        }
        write_json(output / "checkpoint_resume_audit.json", audit)
        return model, list(metadata["history"]), audit
    if candidates:
        raise RuntimeError(
            "no valid Tomotopy checkpoint remains: "
            + "; ".join(row["reason"] for row in rejected)
        )
    legacy = (output / "model.bin.partial", output / "history.json")
    if any(path.exists() for path in legacy):
        raise RuntimeError(
            "legacy Tomotopy checkpoint cannot be resumed safely; remove it and restart"
        )
    audit = {"selected_checkpoint": None, "rejected_newer_checkpoints": []}
    write_json(output / "checkpoint_resume_audit.json", audit)
    return None, [], audit


def run_tomotopy_seed(run_dir: str | Path, seed: int) -> dict[str, Any]:
    """Train and evaluate one deterministic ordinary Tomotopy LDA seed."""
    try:
        import tomotopy as tp
    except ImportError as exc:  # pragma: no cover - environment validation
        raise ImportError("tomotopy==0.13.0 is required in ms2lda-hybrid") from exc
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not frozen")
    output = directory / "core" / f"seed_{seed}" / "tomotopy"
    complete_path = output / "complete.json"
    if complete_path.exists():
        return _verify_completed_core_result(
            output,
            method="tomotopy",
            seed=seed,
            config=config,
        )
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_documents(directory)
    vocabulary = load_vocabulary(directory)
    checkpoint_context = object_sha256(
        {
            "protocol_sha256": lock["protocol_sha256"],
            "method": "tomotopy",
            "seed": seed,
            "topic_count": config.num_topics,
            "alpha": config.alpha,
            "eta": config.eta,
            "step_size": config.tomotopy_step_size,
            "training_workers": config.tomotopy_training_workers,
            "training_parallel": config.tomotopy_training_parallel,
            "train_ids": list(data.train_ids),
            "vocabulary": list(vocabulary),
        }
    )
    model, history, checkpoint_audit = _restore_tomotopy_checkpoint(
        tp, output, context_sha256=checkpoint_context
    )
    if model is None:
        model = tp.LDAModel(
            k=config.num_topics,
            min_df=1,
            min_cf=0,
            rm_top=0,
            alpha=config.alpha,
            eta=config.eta,
            seed=seed,
        )
        for words in data.train_words:
            model.add_doc(words)
        history = []
        trained = 0
    else:
        trained = int(history[-1]["iteration"])
    training_started = time.perf_counter()
    previous_training_seconds = (
        float(history[-1].get("cumulative_training_seconds", 0.0)) if history else 0.0
    )
    while trained < config.tomotopy_max_iterations:
        step = min(config.tomotopy_step_size, config.tomotopy_max_iterations - trained)
        step_started = time.perf_counter()
        model.train(
            step,
            workers=config.tomotopy_training_workers,
            parallel=config.tomotopy_training_parallel,
        )
        previous_training_seconds += time.perf_counter() - step_started
        trained += step
        history.append(
            {
                "iteration": trained,
                "ll_per_word": float(model.ll_per_word),
                "perplexity": float(model.perplexity),
                "cumulative_training_seconds": previous_training_seconds,
            }
        )
        _save_tomotopy_checkpoint(
            model,
            output,
            context_sha256=checkpoint_context,
            history=history,
        )
        if _converged(
            history,
            window=config.tomotopy_convergence_window,
            threshold=config.tomotopy_convergence_threshold,
        ):
            break
    training_seconds = time.perf_counter() - training_started
    model_vocabulary = list(model.used_vocabs)
    beta = np.vstack(
        [
            np.asarray(model.get_topic_word_dist(topic), dtype=np.float32)
            for topic in range(model.k)
        ]
    )
    beta = _aligned_beta(beta, model_vocabulary, vocabulary)
    query_documents = [model.make_doc(words) for words in data.test_observed_words]
    inference_started = time.perf_counter()
    theta, _ = model.infer(
        query_documents,
        iter=config.tomotopy_inference_iterations,
        workers=1,
        parallel=1,
        together=False,
    )
    inference_seconds = time.perf_counter() - inference_started
    theta = np.asarray(theta, dtype=np.float32)
    latency_count = min(config.latency_subset_size, len(query_documents))
    latency_words = data.test_observed_words[:latency_count]
    cached_documents = [model.make_doc(words) for words in latency_words]
    cached_durations = _measure(
        lambda: model.infer(
            cached_documents,
            iter=config.tomotopy_inference_iterations,
            workers=1,
            parallel=1,
            together=False,
        ),
        repeats=config.latency_repeats,
    )

    def end_to_end() -> Any:
        documents = [model.make_doc(words) for words in latency_words]
        return model.infer(
            documents,
            iter=config.tomotopy_inference_iterations,
            workers=1,
            parallel=1,
            together=False,
        )

    end_to_end_durations = _measure(end_to_end, repeats=config.latency_repeats)
    metrics = _common_metrics(
        theta=theta,
        beta=beta,
        heldout_words=data.test_completion_words,
        vocabulary=vocabulary,
        train_words=data.train_words,
        config=config,
    )
    np.save(output / "beta.npy", beta)
    np.save(output / "test_theta.npy", theta)
    write_json(output / "vocabulary.json", {"vocabulary": list(vocabulary)})
    final_model_path = output / "model.bin"
    model.save(str(final_model_path))
    result = {
        "method": "tomotopy",
        "seed": seed,
        "topic_count": config.num_topics,
        "evidence_scope": config.evidence_scope,
        "training_workers_requested": config.tomotopy_training_workers,
        "training_parallel_scheme": tp.ParallelScheme(
            config.tomotopy_training_parallel
        ).name,
        "training_parallel_scheme_value": config.tomotopy_training_parallel,
        "training_bitwise_reproducible": bool(
            config.tomotopy_training_workers == 1
            and config.tomotopy_training_parallel == 1
        ),
        "training_iterations": trained,
        "alpha": {
            "initial": _alpha_summary(
                np.full(config.num_topics, config.alpha, dtype=np.float64)
            ),
            "final": _alpha_summary(np.asarray(model.alpha, dtype=np.float64)),
        },
        "converged": _converged(
            history,
            window=config.tomotopy_convergence_window,
            threshold=config.tomotopy_convergence_threshold,
        ),
        "training_seconds_this_process": training_seconds,
        "training_seconds_total": previous_training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "cached_latency": _latency_summary(cached_durations, latency_count),
        "end_to_end_latency": _latency_summary(end_to_end_durations, latency_count),
        "metrics": metrics,
        "checkpointing": {
            "schema_version": "tomotopy-checkpoint/v1",
            "keep": 2,
            "context_sha256": checkpoint_context,
            "resume_audit": checkpoint_audit,
            "retained": [
                path.name for path in _tomotopy_checkpoint_sidecars(output)[:2]
            ],
        },
        "beta_sha256": file_sha256(output / "beta.npy"),
        "theta_sha256": file_sha256(output / "test_theta.npy"),
        "model_sha256": file_sha256(final_model_path),
    }
    write_json(complete_path, result)
    return result


def _hybrid_infer(model, documents, steps: int) -> np.ndarray:
    theta, _ = model.infer(documents, iter=steps, tolerance=None)
    return np.asarray(theta, dtype=np.float32)


def _measure_hybrid_end_to_end(
    *,
    model,
    records: Sequence[SpectrumRecord],
    steps: Sequence[int],
    repeats: int,
    extraction_batch_size: int = 16,
) -> dict[str, dict[str, float | int]]:
    from ms2lda_hybrid.dreams_features import DreaMSFeatureExtractor

    extractor = DreaMSFeatureExtractor(device="cpu")
    spectra = [to_matchms_spectrum(record) for record in records]
    identifiers = [record.spectrum_id for record in records]
    output: dict[str, list[float]] = {str(value): [] for value in steps}
    for repeat in range(repeats + 1):
        started = time.perf_counter()
        features = extractor.extract(
            spectra,
            identifiers=identifiers,
            batch_size=extraction_batch_size,
        )
        extraction_seconds = time.perf_counter() - started
        documents = [
            model.make_doc(
                record.words,
                embedding=features.spectrum_embeddings[row],
            )
            for row, record in enumerate(records)
        ]
        for value in steps:
            inference_started = time.perf_counter()
            _hybrid_infer(model, documents, value)
            duration = extraction_seconds + (time.perf_counter() - inference_started)
            if repeat:
                output[str(value)].append(duration)
    return {
        key: _latency_summary(durations, len(records))
        for key, durations in output.items()
    }


def _new_hybrid_model(
    *,
    config: BenchmarkConfig,
    data: PreparedDocuments,
    vocabulary: Sequence[str],
    pooled_word_embeddings: np.ndarray,
    seed: int,
) -> Any:
    """Construct one fresh HybridLDA model from frozen, chemistry-free inputs."""
    from ms2lda_hybrid import HybridLDAConfig, HybridLDAModel

    hybrid_config = HybridLDAConfig(
        num_topics=config.num_topics,
        embedding_dim=data.train_embeddings.shape[1],
        alpha=config.alpha,
        eta=config.eta,
        batch_size=config.hybrid_batch_size,
        inference_epochs=config.hybrid_inference_epochs,
        global_patience=config.hybrid_global_patience,
        max_epochs=config.hybrid_max_epochs,
        seed=seed,
    )
    model = HybridLDAModel(hybrid_config, device="cpu")
    model.set_word_embeddings(
        {
            word: np.asarray(pooled_word_embeddings[index], dtype=np.float32)
            for index, word in enumerate(vocabulary)
            if np.any(pooled_word_embeddings[index])
        }
    )
    for words, embedding in zip(data.train_words, data.train_embeddings, strict=True):
        model.add_doc(words, embedding=embedding)
    return model


def _hybrid_checkpoint_context(
    *,
    directory: Path,
    lock: dict[str, Any],
    config: BenchmarkConfig,
    seed: int,
    data: PreparedDocuments,
    feature_manifest: dict[str, Any],
) -> str:
    """Bind resumable state to the exact protocol, inputs, features, and seed."""
    return object_sha256(
        {
            "protocol_sha256": lock["protocol_sha256"],
            "config_sha256": object_sha256(config.as_dict()),
            "feature_manifest_sha256": object_sha256(feature_manifest),
            "vocabulary_sha256": file_sha256(directory / "vocabulary.json"),
            "train_ids_sha256": object_sha256(list(data.train_ids)),
            "seed": seed,
        }
    )


def _checkpoint_sidecars(checkpoint_dir: Path) -> list[Path]:
    """Return versioned checkpoint metadata files newest first."""
    return sorted(checkpoint_dir.glob("checkpoint-*.json"), reverse=True)


def _verified_checkpoint_metadata(
    path: Path,
    *,
    context_sha256: str,
) -> dict[str, Any]:
    """Verify one checkpoint sidecar and its binary payload."""
    metadata = read_json(path)
    if metadata.get("context_sha256") != context_sha256:
        raise ValueError("checkpoint context hash mismatch")
    filename = str(metadata.get("file", ""))
    binary = (path.parent / filename).resolve()
    if not filename or binary.parent != path.parent.resolve():
        raise ValueError("checkpoint filename escapes its directory")
    if not binary.is_file():
        raise FileNotFoundError(f"checkpoint payload is missing: {binary}")
    if binary.stat().st_size != int(metadata.get("bytes", -1)):
        raise ValueError("checkpoint byte size mismatch")
    if file_sha256(binary) != metadata.get("sha256"):
        raise ValueError("checkpoint SHA-256 mismatch")
    return metadata


def _discard_checkpoint_generation(sidecar: Path) -> None:
    """Remove one unusable or obsolete generated checkpoint safely."""
    try:
        metadata = read_json(sidecar)
        filename = str(metadata.get("file", ""))
        binary = (sidecar.parent / filename).resolve()
        if filename and binary.parent == sidecar.parent.resolve():
            binary.unlink(missing_ok=True)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    sidecar.unlink(missing_ok=True)


def _prune_hybrid_checkpoints(
    checkpoint_dir: Path,
    *,
    context_sha256: str,
    keep: int,
) -> None:
    """Retain the newest ``keep`` verified generations, not corrupt files."""
    retained = 0
    for sidecar in _checkpoint_sidecars(checkpoint_dir):
        try:
            _verified_checkpoint_metadata(
                sidecar,
                context_sha256=context_sha256,
            )
        except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError):
            _discard_checkpoint_generation(sidecar)
            continue
        retained += 1
        if retained > keep:
            _discard_checkpoint_generation(sidecar)


def _save_hybrid_checkpoint(
    model: Any,
    *,
    output: Path,
    context_sha256: str,
    phase: str,
    phase_epoch: int,
    keep: int,
    training_cpu_threads: int,
    cumulative_discovery_seconds: float,
    cumulative_finalization_seconds: float,
) -> dict[str, Any]:
    """Write one atomic Hybrid checkpoint and retain the newest two or more."""
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sequence = (
        phase_epoch if phase == "discovery" else model.config.max_epochs + phase_epoch
    )
    stem = f"checkpoint-{sequence:04d}-{phase}-{phase_epoch:04d}"
    binary = checkpoint_dir / f"{stem}.pt"
    sidecar = checkpoint_dir / f"{stem}.json"
    model.save_training_checkpoint(binary, context_sha256=context_sha256)
    metadata = {
        "bytes": binary.stat().st_size,
        "context_sha256": context_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cumulative_discovery_seconds": cumulative_discovery_seconds,
        "cumulative_finalization_seconds": cumulative_finalization_seconds,
        "file": binary.name,
        "phase": phase,
        "phase_epoch": phase_epoch,
        "sequence": sequence,
        "sha256": file_sha256(binary),
        "training_cpu_threads": training_cpu_threads,
    }
    write_json(sidecar, metadata)
    write_json(checkpoint_dir / "latest.json", metadata)
    with (checkpoint_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")

    _prune_hybrid_checkpoints(
        checkpoint_dir,
        context_sha256=context_sha256,
        keep=keep,
    )
    return metadata


def _restore_hybrid_checkpoint(
    *,
    factory: Callable[[], Any],
    output: Path,
    context_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    """Restore the newest valid checkpoint, falling back to its predecessor."""
    checkpoint_dir = output / "checkpoints"
    candidates = _checkpoint_sidecars(checkpoint_dir)
    rejected = []
    for sidecar in candidates:
        try:
            metadata = _verified_checkpoint_metadata(
                sidecar,
                context_sha256=context_sha256,
            )
            model = factory()
            progress = model.restore_training_checkpoint(
                checkpoint_dir / metadata["file"],
                context_sha256=context_sha256,
            )
        except Exception as exc:
            rejected.append(
                {
                    "metadata": sidecar.name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        audit = {
            "context_sha256": context_sha256,
            "rejected_newer_checkpoints": rejected,
            "resumed": True,
            "selected_checkpoint": metadata,
            "selected_progress": progress,
        }
        write_json(output / "checkpoint_resume_audit.json", audit)
        return model, audit
    if candidates:
        raise RuntimeError(
            "no valid Hybrid checkpoint remains: "
            + "; ".join(row["reason"] for row in rejected)
        )
    model = factory()
    audit = {
        "context_sha256": context_sha256,
        "rejected_newer_checkpoints": [],
        "resumed": False,
        "selected_checkpoint": None,
        "selected_progress": None,
    }
    write_json(output / "checkpoint_resume_audit.json", audit)
    return model, audit


def run_hybrid_seed(run_dir: str | Path, seed: int) -> dict[str, Any]:
    """Train and evaluate one full HybridLDA seed."""
    import torch

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not frozen")
    torch.set_num_threads(config.hybrid_training_cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    output = directory / "core" / f"seed_{seed}" / "hybrid"
    complete_path = output / "complete.json"
    if complete_path.exists():
        return _verify_completed_core_result(
            output,
            method="hybrid",
            seed=seed,
            config=config,
        )
    output.mkdir(parents=True, exist_ok=True)
    for stale in (
        output / "model.pt",
        output / "beta.npy",
        output / "vocabulary.json",
        output / "discovery_history.json",
        output / "inference_history.json",
    ):
        stale.unlink(missing_ok=True)
    for stale in output.glob("test_theta_*.npy"):
        stale.unlink(missing_ok=True)
    data = prepare_documents(directory)
    vocabulary = load_vocabulary(directory)
    _, _, pooled_word_embeddings, feature_manifest = load_feature_cache(directory)
    context_sha256 = _hybrid_checkpoint_context(
        directory=directory,
        lock=lock,
        config=config,
        seed=seed,
        data=data,
        feature_manifest=feature_manifest,
    )

    def factory() -> Any:
        return _new_hybrid_model(
            config=config,
            data=data,
            vocabulary=vocabulary,
            pooled_word_embeddings=pooled_word_embeddings,
            seed=seed,
        )

    model, checkpoint_audit = _restore_hybrid_checkpoint(
        factory=factory,
        output=output,
        context_sha256=context_sha256,
    )
    selected_checkpoint = checkpoint_audit.get("selected_checkpoint") or {}
    selected_progress = checkpoint_audit.get("selected_progress") or {}
    discovery_base = float(selected_checkpoint.get("cumulative_discovery_seconds", 0.0))
    finalization_base = float(
        selected_checkpoint.get("cumulative_finalization_seconds", 0.0)
    )
    timing: dict[str, float | None] = {
        "discovery_started": None,
        "discovery_total": discovery_base,
        "finalization_started": None,
    }

    def checkpoint_callback(
        checkpoint_model: Any,
        phase: str,
        phase_epoch: int,
    ) -> None:
        now = time.perf_counter()
        if phase == "discovery":
            started = timing["discovery_started"]
            if started is None:
                raise RuntimeError("discovery checkpoint has no active timer")
            discovery_total = discovery_base + (now - started)
            finalization_total = finalization_base
        else:
            started = timing["finalization_started"]
            if started is None:
                raise RuntimeError("encoder checkpoint has no active timer")
            discovery_total = float(timing["discovery_total"])
            finalization_total = finalization_base + (now - started)
        _save_hybrid_checkpoint(
            checkpoint_model,
            output=output,
            context_sha256=context_sha256,
            phase=phase,
            phase_epoch=phase_epoch,
            keep=config.hybrid_checkpoint_keep,
            training_cpu_threads=config.hybrid_training_cpu_threads,
            cumulative_discovery_seconds=discovery_total,
            cumulative_finalization_seconds=finalization_total,
        )

    resumed_phase = selected_progress.get("phase")
    discovery_seconds_this_process = 0.0
    if resumed_phase != "encoder":
        timing["discovery_started"] = time.perf_counter()
        model.train(
            config.hybrid_max_epochs,
            checkpoint_callback=checkpoint_callback,
        )
        discovery_seconds_this_process = time.perf_counter() - float(
            timing["discovery_started"]
        )
        timing["discovery_total"] = discovery_base + discovery_seconds_this_process
    if not model.converged:
        raise RuntimeError(
            "Hybrid topic discovery reached its frozen maximum without "
            "lambda and alpha convergence; the latest checkpoint was retained"
        )
    finalization_started = time.perf_counter()
    timing["finalization_started"] = finalization_started
    inference_history = model.finalize_inference(
        checkpoint_callback=checkpoint_callback
    )
    finalization_seconds_this_process = time.perf_counter() - finalization_started
    finalization_seconds_total = finalization_base + finalization_seconds_this_process
    discovery_seconds_total = float(timing["discovery_total"])
    torch.set_num_threads(config.hybrid_inference_cpu_threads)
    beta = _aligned_beta(
        np.vstack([model.get_topic_word_dist(topic) for topic in range(model.k)]),
        model.used_vocabs,
        vocabulary,
    )
    validation_documents = [
        model.make_doc(words, embedding=embedding)
        for words, embedding in zip(
            data.validation_observed_words,
            data.validation_embeddings,
            strict=True,
        )
    ]
    validation_theta = {}
    validation_inference_seconds = {}
    for steps in (config.hybrid_reference_steps // 2, config.hybrid_reference_steps):
        started = time.perf_counter()
        validation_theta[steps] = _hybrid_infer(model, validation_documents, steps)
        validation_inference_seconds[str(steps)] = time.perf_counter() - started
    selection_audit = convergence_metrics(
        validation_theta[config.hybrid_reference_steps // 2],
        validation_theta[config.hybrid_reference_steps],
    )
    reference_steps = config.hybrid_reference_steps
    if (
        selection_audit["cosine_median"] < config.reference_median_cosine
        or selection_audit["cosine_p05"] < config.reference_fifth_percentile_cosine
    ):
        reference_steps = config.hybrid_reference_extension_steps
        started = time.perf_counter()
        validation_theta[reference_steps] = _hybrid_infer(
            model, validation_documents, reference_steps
        )
        validation_inference_seconds[str(reference_steps)] = (
            time.perf_counter() - started
        )
        selection_audit = convergence_metrics(
            validation_theta[config.hybrid_reference_steps],
            validation_theta[reference_steps],
        )
    reference_converged = bool(
        selection_audit["cosine_median"] >= config.reference_median_cosine
        and selection_audit["cosine_p05"] >= config.reference_fifth_percentile_cosine
    )
    if not reference_converged:
        raise RuntimeError(
            "the frozen maximum local-refinement budget did not meet the "
            "validation-set near-convergence criteria"
        )

    documents = [
        model.make_doc(words, embedding=embedding)
        for words, embedding in zip(
            data.test_observed_words, data.test_embeddings, strict=True
        )
    ]
    theta_by_steps = {}
    inference_seconds = {}
    for steps in (0, 2, reference_steps):
        started = time.perf_counter()
        theta_by_steps[steps] = _hybrid_infer(model, documents, steps)
        inference_seconds[str(steps)] = time.perf_counter() - started
    reference = theta_by_steps[reference_steps]
    metrics = {}
    for steps in (0, 2, reference_steps):
        label = "long" if steps == reference_steps else f"iter_{steps}"
        metrics[label] = _common_metrics(
            theta=theta_by_steps[steps],
            beta=beta,
            heldout_words=data.test_completion_words,
            vocabulary=vocabulary,
            train_words=data.train_words,
            config=config,
        )
        if steps != reference_steps:
            metrics[label]["convergence_to_long"] = convergence_metrics(
                theta_by_steps[steps], reference
            )
            metrics[label]["nll_gap_fraction"] = (
                metrics[label]["document_completion"]["nll_per_token"]
                / metrics["long"]["document_completion"]["nll_per_token"]
                - 1.0
                if "long" in metrics
                else None
            )
    # The loop creates ``long`` last. Fill the NLL gaps after its value exists.
    long_nll = metrics["long"]["document_completion"]["nll_per_token"]
    for label in ("iter_0", "iter_2"):
        metrics[label]["nll_gap_fraction"] = (
            metrics[label]["document_completion"]["nll_per_token"] / long_nll - 1.0
        )

    latency_count = min(config.latency_subset_size, len(documents))
    latency_documents = documents[:latency_count]
    cached_latency = {}
    for steps in (0, 2, reference_steps):
        durations = _measure(
            lambda selected=steps: _hybrid_infer(model, latency_documents, selected),
            repeats=config.latency_repeats,
        )
        cached_latency[str(steps)] = _latency_summary(durations, latency_count)
    end_to_end_latency = _measure_hybrid_end_to_end(
        model=model,
        records=data.test_observed_records[:latency_count],
        steps=(0, 2, reference_steps),
        repeats=config.latency_repeats,
    )
    model_path = output / "model.pt"
    model.save(model_path)
    np.save(output / "beta.npy", beta)
    for steps, theta in theta_by_steps.items():
        np.save(output / f"test_theta_{steps}.npy", theta)
    write_json(output / "vocabulary.json", {"vocabulary": list(vocabulary)})
    write_json(output / "discovery_history.json", model.history)
    write_json(output / "inference_history.json", inference_history)
    retained_checkpoints = _checkpoint_sidecars(output / "checkpoints")
    result = {
        "method": "hybrid",
        "seed": seed,
        "topic_count": config.num_topics,
        "evidence_scope": config.evidence_scope,
        "training_cpu_threads": config.hybrid_training_cpu_threads,
        "inference_cpu_threads": config.hybrid_inference_cpu_threads,
        "training_bitwise_reproducible": bool(config.hybrid_training_cpu_threads == 1),
        "checkpointing": {
            "context_sha256": context_sha256,
            "enabled": True,
            "keep": config.hybrid_checkpoint_keep,
            "resume_audit": checkpoint_audit,
            "retained": [path.name for path in retained_checkpoints],
        },
        "discovery_epochs": len(model.history),
        "discovery_converged": model.converged,
        "alpha": {
            "initial": _alpha_summary(
                np.full(config.num_topics, config.alpha, dtype=np.float64)
            ),
            "final": _alpha_summary(model.alpha),
        },
        "reference_steps": reference_steps,
        "reference_selection_split": "validation",
        "reference_converged": reference_converged,
        "reference_audit": selection_audit,
        "validation_inference_seconds": validation_inference_seconds,
        "discovery_seconds": discovery_seconds_total,
        "discovery_seconds_this_process": discovery_seconds_this_process,
        "discovery_seconds_total": discovery_seconds_total,
        "finalization_seconds": finalization_seconds_total,
        "finalization_seconds_this_process": finalization_seconds_this_process,
        "finalization_seconds_total": finalization_seconds_total,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "cached_latency": cached_latency,
        "end_to_end_latency": end_to_end_latency,
        "metrics": metrics,
        "beta_sha256": file_sha256(output / "beta.npy"),
        "theta_sha256": {
            str(steps): file_sha256(output / f"test_theta_{steps}.npy")
            for steps in theta_by_steps
        },
        "model_sha256": file_sha256(model_path),
    }
    write_json(complete_path, result)
    return result


def _worker_command(run_dir: Path, method: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.msnlib_validation",
        "_run-model",
        "--run",
        str(run_dir),
        "--method",
        method,
        "--seed",
        str(seed),
    ]


def run_all_core_models(run_dir: str | Path) -> dict[str, Any]:
    """Prepare features and execute every frozen seed in isolated processes."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    feature_manifest = prepare_features(directory)
    environment = dict(os.environ)
    completed = []
    for seed in config.seeds:
        for method in ("tomotopy", "hybrid"):
            result_path = directory / "core" / f"seed_{seed}" / method / "complete.json"
            if not result_path.exists():
                worker_environment = dict(environment)
                worker_threads = (
                    config.hybrid_training_cpu_threads if method == "hybrid" else 1
                )
                worker_environment.update(
                    {
                        "OMP_NUM_THREADS": str(worker_threads),
                        "MKL_NUM_THREADS": str(worker_threads),
                        "OPENBLAS_NUM_THREADS": str(worker_threads),
                        "NUMEXPR_NUM_THREADS": str(worker_threads),
                    }
                )
                subprocess.run(
                    _worker_command(directory, method, seed),
                    cwd=lock["repo_root"],
                    env=worker_environment,
                    check=True,
                )
            completed.append(
                _verify_completed_core_result(
                    result_path.parent,
                    method=method,
                    seed=seed,
                    config=config,
                )
            )
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "feature_manifest_sha256": object_sha256(feature_manifest),
        "evidence_scope": config.evidence_scope,
        "tomotopy_training_workers_requested": config.tomotopy_training_workers,
        "tomotopy_training_parallel": config.tomotopy_training_parallel,
        "hybrid_training_cpu_threads": config.hybrid_training_cpu_threads,
        "hybrid_inference_cpu_threads": config.hybrid_inference_cpu_threads,
        "required_seeds": list(config.seeds),
        "completed": [
            {"method": row["method"], "seed": row["seed"]} for row in completed
        ],
    }
    write_json(directory / "core" / "complete.json", manifest)
    return manifest
