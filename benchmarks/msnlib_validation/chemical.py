"""Full-spectrum held-out inference for chemical-coherence evaluation.

Document-completion likelihood deliberately observes only one frozen half of a
test spectrum.  Chemical association is a different endpoint: it must use the
whole held-out spectrum, while still withholding every chemical label from
model inference.  This module keeps those two representations physically and
provenance-wise separate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import file_sha256, load_config, read_json, resolve_input_paths, write_json
from .data import SpectrumRecord, load_records, split_records, to_matchms_spectrum
from .protocol import (
    load_assignments,
    load_vocabulary,
    verify_frozen_input_files,
    verify_protocol,
)
from .runtime import peak_rss_bytes

ASSOCIATION_MODES = ("dominant_topic", "probability_ge_frozen_threshold")


def full_spectrum_documents(
    records: Sequence[SpectrumRecord], vocabulary: Sequence[str]
) -> tuple[list[list[str]], dict[str, int]]:
    """Return full-spectrum words without consulting chemical metadata."""
    retained = set(map(str, vocabulary))
    documents = [list(record.words) for record in records]
    in_vocabulary = sum(word in retained for words in documents for word in words)
    return documents, {
        "documents": len(documents),
        "tokens": sum(map(len, documents)),
        "in_vocabulary_tokens": in_vocabulary,
        "out_of_vocabulary_tokens": sum(map(len, documents)) - in_vocabulary,
    }


def associated_record_indices(
    theta: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> dict[int, list[int]]:
    """Associate spectra to topics without assuming calibrated probability scale."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("topic mixtures must be a nonempty matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("topic mixtures must be finite and nonnegative")
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("topic mixtures must have positive row sums")
    values = values / row_sums[:, None]
    associated: dict[int, list[int]] = defaultdict(list)
    if mode == "dominant_topic":
        for row, topic in enumerate(np.argmax(values, axis=1)):
            associated[int(topic)].append(row)
    elif mode == "probability_ge_frozen_threshold":
        rows, topics = np.where(values >= threshold)
        for row, topic in zip(rows, topics, strict=True):
            associated[int(topic)].append(int(row))
    else:
        raise ValueError(f"unsupported chemical association mode: {mode}")
    return associated


def _test_records(directory: Path, lock: dict[str, Any]):
    config = load_config(directory / "config.resolved.json")
    inputs = resolve_input_paths(config, lock["data_root"])
    records, _ = load_records(inputs["mgf"], config)
    return split_records(records, load_assignments(directory), "test")


def prepare_full_test_features(
    run_dir: str | Path,
    *,
    extraction_batch_size: int = 16,
    checkpoint_every_chunks: int = 25,
) -> dict[str, Any]:
    """Extract a resumable full-spectrum DreaMS cache for held-out chemistry."""
    import torch

    from ms2lda_hybrid.dreams_features import (
        DREAMS_EMBEDDING_DIM,
        DreaMSFeatureExtractor,
    )

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    verify_frozen_input_files(directory, names={"mgf"}, lock=lock)
    output = directory / "chemical_inference" / "features"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("chemical feature cache belongs to another protocol")
        for name, digest in manifest["output_sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"chemical feature cache changed: {name}")
        return manifest

    config = load_config(directory / "config.resolved.json")
    torch.set_num_threads(config.hybrid_inference_cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    test_records = _test_records(directory, lock)
    identifiers = [record.spectrum_id for record in test_records]
    identifiers_path = output / "identifiers.json"
    embeddings_path = output / "full_test_embeddings.npy"
    completed_path = output / "completed.npy"
    progress_path = output / "progress.json"
    expected_shape = (len(test_records), DREAMS_EMBEDDING_DIM)
    partial = [
        path.exists() for path in (embeddings_path, completed_path, progress_path)
    ]
    if any(partial) and not all(partial):
        raise RuntimeError("chemical feature checkpoint is incomplete")
    if identifiers_path.exists():
        if read_json(identifiers_path).get("identifiers") != identifiers:
            raise ValueError("chemical feature identifiers changed")
    else:
        write_json(identifiers_path, {"identifiers": identifiers})

    if all(partial):
        embeddings = np.lib.format.open_memmap(
            embeddings_path, mode="r+", dtype=np.float32, shape=expected_shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="r+", dtype=np.bool_, shape=(len(test_records),)
        )
        progress = read_json(progress_path)
        previous_seconds = float(progress.get("cumulative_extraction_seconds", 0.0))
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=(len(test_records),)
        )
        completed[:] = False
        completed.flush()
        previous_seconds = 0.0
        write_json(
            progress_path,
            {"completed_rows": 0, "cumulative_extraction_seconds": 0.0},
        )
    incomplete = np.flatnonzero(~np.asarray(completed))
    start_row = int(incomplete[0]) if len(incomplete) else len(test_records)
    if np.any(np.asarray(completed)[start_row:]):
        raise ValueError("chemical feature progress is non-contiguous")

    initialized = time.perf_counter()
    extractor = DreaMSFeatureExtractor(device="cpu")
    initialization_seconds = time.perf_counter() - initialized
    extraction_seconds = 0.0
    checkpoint_start = start_row
    chunks = 0
    for start in range(start_row, len(test_records), extraction_batch_size):
        stop = min(start + extraction_batch_size, len(test_records))
        selected = test_records[start:stop]
        started = time.perf_counter()
        batch = extractor.extract(
            [to_matchms_spectrum(record) for record in selected],
            identifiers=[record.spectrum_id for record in selected],
            batch_size=extraction_batch_size,
        )
        extraction_seconds += time.perf_counter() - started
        if list(batch.identifiers) != [record.spectrum_id for record in selected]:
            raise ValueError("chemical DreaMS features are not row aligned")
        embeddings[start:stop] = batch.spectrum_embeddings
        embeddings.flush()
        chunks += 1
        if chunks % checkpoint_every_chunks == 0 or stop == len(test_records):
            completed[checkpoint_start:stop] = True
            completed.flush()
            checkpoint_start = stop
            write_json(
                progress_path,
                {
                    "completed_rows": stop,
                    "total_rows": len(test_records),
                    "cumulative_extraction_seconds": previous_seconds
                    + extraction_seconds,
                    "extraction_seconds_this_process": extraction_seconds,
                    "peak_rss_bytes": peak_rss_bytes(),
                },
            )
    if not bool(np.all(np.asarray(completed))):
        raise RuntimeError("chemical feature extraction ended with incomplete rows")
    outputs = (identifiers_path, embeddings_path, completed_path)
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "rows": len(test_records),
        "embedding_dim": DREAMS_EMBEDDING_DIM,
        "split": "test",
        "full_spectrum_peak_groups": True,
        "document_completion_representation_used": False,
        "chemical_labels_used_for_extraction": False,
        "extraction_cpu_threads": config.hybrid_inference_cpu_threads,
        "extractor_initialization_seconds": initialization_seconds,
        "extraction_seconds_this_process": extraction_seconds,
        "cumulative_extraction_seconds": previous_seconds + extraction_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "extractor_provenance": extractor.provenance,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(manifest_path, manifest)
    return manifest


def _validate_theta(theta: np.ndarray, *, rows: int, topics: int) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float32)
    if values.shape != (rows, topics):
        raise ValueError(f"unexpected chemical mixture shape: {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("chemical mixtures must be finite and nonnegative")
    sums = values.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("chemical mixtures have zero-mass rows")
    return values / sums


def _mixture_diagnostics(theta: np.ndarray, *, threshold: float) -> dict[str, Any]:
    """Describe calibration and topic use without chemical labels."""
    values = np.asarray(theta, dtype=np.float64)
    maximum = values.max(axis=1)
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, None)), axis=1)
    threshold_rows, threshold_topics = np.where(values >= threshold)
    return {
        "dominant_topics_used": int(np.unique(np.argmax(values, axis=1)).size),
        "maximum_topic_probability_median": float(np.median(maximum)),
        "maximum_topic_probability_p95": float(np.percentile(maximum, 95)),
        "maximum_topic_probability_maximum": float(np.max(maximum)),
        "effective_topic_count_median": float(np.median(np.exp(entropy))),
        "frozen_threshold": threshold,
        "spectra_crossing_frozen_threshold": int(np.unique(threshold_rows).size),
        "topics_crossing_frozen_threshold": int(np.unique(threshold_topics).size),
    }


def run_full_spectrum_inference_for_model(
    run_dir: str | Path, *, seed: int, method: str
) -> dict[str, Any]:
    """Infer full-test mixtures from a completed, unchanged core model."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    if seed not in config.seeds or method not in {"tomotopy", "hybrid"}:
        raise ValueError("method or seed is not frozen")
    output = directory / "chemical_inference" / f"seed_{seed}" / method
    complete_path = output / "complete.json"
    if complete_path.exists():
        result = read_json(complete_path)
        if result.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("chemical inference belongs to another protocol")
        for name, digest in result["theta_sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"chemical mixture changed: {name}")
        return result
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest = prepare_full_test_features(directory)
    test_records = _test_records(directory, lock)
    vocabulary = load_vocabulary(directory)
    words, document_summary = full_spectrum_documents(test_records, vocabulary)
    identifiers = read_json(
        directory / "chemical_inference" / "features" / "identifiers.json"
    )["identifiers"]
    if identifiers != [record.spectrum_id for record in test_records]:
        raise ValueError("chemical feature cache and test rows differ")
    embeddings = np.load(
        directory / "chemical_inference" / "features" / "full_test_embeddings.npy",
        mmap_mode="r",
    )
    core = directory / "core" / f"seed_{seed}" / method
    core_result = read_json(core / "complete.json")
    theta_paths: dict[str, Path] = {}
    inference_seconds: dict[str, float] = {}
    mixture_diagnostics: dict[str, dict[str, Any]] = {}
    if method == "tomotopy":
        import tomotopy as tp

        if file_sha256(core / "model.bin") != core_result["model_sha256"]:
            raise ValueError("Tomotopy model changed before chemical inference")
        model = tp.LDAModel.load(str(core / "model.bin"))
        documents = [model.make_doc(document) for document in words]
        started = time.perf_counter()
        theta, _ = model.infer(
            documents,
            iter=config.tomotopy_inference_iterations,
            workers=1,
            parallel=1,
            together=False,
        )
        inference_seconds["standard"] = time.perf_counter() - started
        theta_paths["standard"] = output / "test_full_theta_standard.npy"
        values = _validate_theta(
            theta, rows=len(test_records), topics=config.num_topics
        )
        np.save(theta_paths["standard"], values)
        mixture_diagnostics["standard"] = _mixture_diagnostics(
            values, threshold=config.membership_threshold
        )
        reference_steps: int | None = None
    else:
        import torch

        from ms2lda_hybrid import HybridLDAModel

        torch.set_num_threads(config.hybrid_inference_cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        if file_sha256(core / "model.pt") != core_result["model_sha256"]:
            raise ValueError("Hybrid model changed before chemical inference")
        model = HybridLDAModel.load(core / "model.pt", device="cpu")
        documents = [
            model.make_doc(document, embedding=embeddings[row])
            for row, document in enumerate(words)
        ]
        reference_steps = int(core_result["reference_steps"])
        for arm, steps in (("encoder", 0), ("two_step", 2), ("long", reference_steps)):
            started = time.perf_counter()
            theta, _ = model.infer(documents, iter=steps, tolerance=None)
            inference_seconds[arm] = time.perf_counter() - started
            theta_paths[arm] = output / f"test_full_theta_{arm}.npy"
            values = _validate_theta(
                theta, rows=len(test_records), topics=config.num_topics
            )
            np.save(theta_paths[arm], values)
            mixture_diagnostics[arm] = _mixture_diagnostics(
                values, threshold=config.membership_threshold
            )
    result = {
        "method": method,
        "seed": seed,
        "protocol_sha256": lock["protocol_sha256"],
        "topic_count": config.num_topics,
        "test_spectra": len(test_records),
        "full_spectrum_peak_groups": True,
        "document_completion_representation_used": False,
        "chemical_labels_used_for_inference": False,
        "feature_manifest_sha256": file_sha256(
            directory / "chemical_inference" / "features" / "manifest.json"
        ),
        "feature_protocol_sha256": feature_manifest["protocol_sha256"],
        "document_summary": document_summary,
        "reference_steps": reference_steps,
        "inference_seconds": inference_seconds,
        "mixture_diagnostics": mixture_diagnostics,
        "peak_rss_bytes": peak_rss_bytes(),
        "theta_sha256": {path.name: file_sha256(path) for path in theta_paths.values()},
    }
    write_json(complete_path, result)
    return result


def run_all_chemical_inference(run_dir: str | Path) -> dict[str, Any]:
    """Create full-spectrum test mixtures for every frozen seed and arm."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    if not (directory / "core" / "complete.json").is_file():
        raise RuntimeError("all core models must complete before chemical inference")
    feature_manifest = prepare_full_test_features(directory)
    config = load_config(directory / "config.resolved.json")
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed: list[tuple[Path, dict[str, Any]]] = []
    for seed in config.seeds:
        for method in ("tomotopy", "hybrid"):
            path = (
                directory
                / "chemical_inference"
                / f"seed_{seed}"
                / method
                / "complete.json"
            )
            if not path.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.msnlib_validation",
                        "_run-chemical-model",
                        "--run",
                        str(directory),
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                    ],
                    cwd=lock["repo_root"],
                    env=environment,
                    check=True,
                )
            completed.append(
                (
                    path,
                    run_full_spectrum_inference_for_model(
                        directory, seed=seed, method=method
                    ),
                )
            )
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "features": feature_manifest,
        "full_spectrum_peak_groups": True,
        "document_completion_representation_used": False,
        "chemical_labels_used_for_inference": False,
        "completed": [
            {
                "method": row["method"],
                "seed": row["seed"],
                "complete_sha256": file_sha256(path),
                "theta_sha256": row["theta_sha256"],
            }
            for path, row in completed
        ],
    }
    write_json(directory / "chemical_inference" / "complete.json", manifest)
    return manifest
