# ruff: noqa: C901, PLR0912, PLR0915
"""Frozen validation-first evaluation and rowwise common references."""

from __future__ import annotations

import csv
import os
import time
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.msnlib_validation.config import (
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)
from benchmarks.msnlib_validation.metrics import (
    active_topic_metrics,
    cosine_rows,
    jensen_shannon_rows,
    optimal_topic_matching,
    top_word_diversity,
)
from benchmarks.msnlib_validation.runtime import peak_rss_bytes

from .data import (
    heldout_metadata,
    load_count_matrix,
    load_full_dreams_embeddings,
    load_observed_dreams_embeddings,
)
from .discovery import load_discovery
from .encoders import InferenceSession, infer_theta, local_elbo_rows, train_encoder
from .spec import ARM_IDS, BUDGETS, INFERENCE_IDS, load_spec, verify_study

if TYPE_CHECKING:
    import scipy.sparse as sp


def _atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_models(run_dir: str | Path) -> dict[str, Any]:
    """Freeze all discoveries and inference artifacts before evaluation."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    path = directory / "models_frozen.json"
    if path.is_file():
        result = read_json(path)
        current = _model_inventory(directory)
        if current != result["inventory"]:
            msg = "model inventory changed after evaluation freeze"
            raise ValueError(msg)
        return result
    for arm_id in ARM_IDS:
        train_encoder(directory, arm_id)
    inventory = _model_inventory(directory)
    result = {
        "schema_version": "msnlib-simplification/models-frozen-v1",
        "spec_sha256": lock["spec_sha256"],
        "arm_count": len(ARM_IDS),
        "inventory": inventory,
        "inventory_sha256": object_sha256(inventory),
        "validation_evaluated_before_freeze": False,
        "test_evaluated_before_freeze": False,
    }
    write_json(path, result)
    return result


def _model_inventory(directory: Path) -> dict[str, str]:
    paths: list[Path] = []
    for discovery in ("dreams_prior", "symmetric_prior"):
        paths.extend(
            (
                directory / "discoveries" / discovery / "complete.json",
                directory / "discoveries" / discovery / "snapshot.npz",
            ),
        )
    for arm_id in ARM_IDS:
        root = directory / "encoders" / arm_id
        paths.append(root / "complete.json")
        if (root / "model.pt").is_file():
            paths.append(root / "model.pt")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        msg = f"model freeze is incomplete: {missing[0]}"
        raise FileNotFoundError(msg)
    return {
        str(path.relative_to(directory)): file_sha256(path) for path in sorted(paths)
    }


def _verify_model_freeze(directory: Path) -> dict[str, Any]:
    result = read_json(directory / "models_frozen.json")
    if _model_inventory(directory) != result["inventory"]:
        msg = "a frozen model artifact changed"
        raise ValueError(msg)
    return result


def _arm_output(directory: Path, split: str, representation: str, arm_id: str) -> Path:
    return directory / "evaluation" / split / representation / "arms" / arm_id


def _verify_inference_complete(output: Path, arm_id: str) -> dict[str, Any]:
    result = read_json(output / "inference_complete.json")
    if result.get("arm_id") != arm_id:
        msg = "inference completion identity changed"
        raise ValueError(msg)
    for name, digest in result["theta_sha256"].items():
        if file_sha256(output / name) != digest:
            msg = f"theta artifact changed: {name}"
            raise ValueError(msg)
    return result


def evaluate_arm(
    run_dir: str | Path,
    *,
    split: str,
    representation: str,
    arm_id: str,
) -> dict[str, Any]:
    """Generate all four frozen-budget mixtures for one arm and split."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    _verify_model_freeze(directory)
    if split not in {"validation", "test"}:
        msg = "evaluation split must be validation or test"
        raise ValueError(msg)
    if representation not in {"observed", "full"}:
        msg = "representation must be observed or full"
        raise ValueError(msg)
    if arm_id not in ARM_IDS:
        msg = "arm is not in the frozen matrix"
        raise ValueError(msg)
    if (
        split == "test"
        and not (directory / "evaluation/validation/frozen.json").is_file()
    ):
        msg = "test evaluation requires the frozen validation pass"
        raise RuntimeError(msg)
    output = _arm_output(directory, split, representation, arm_id)
    if (output / "inference_complete.json").is_file():
        return _verify_inference_complete(output, arm_id)
    discovery, inference = arm_id.split("__", 1)
    matrix = load_count_matrix(directory, f"{split}_{representation}")
    uses_dreams = inference.startswith("dreams_")
    embeddings = None
    if uses_dreams:
        embeddings = (
            load_observed_dreams_embeddings(directory, split)
            if representation == "observed"
            else load_full_dreams_embeddings(directory, split)
        )
    output.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    theta_paths: dict[str, Path] = {}
    session = InferenceSession(directory, arm_id)
    for budget in BUDGETS:
        started = time.perf_counter()
        theta = session.infer(
            matrix,
            budget=budget,
            embeddings=embeddings,
        )
        if representation == "full":
            theta = theta / theta.sum(axis=1, keepdims=True)
        durations[str(budget)] = time.perf_counter() - started
        path = output / f"theta_{budget}.npy"
        _atomic_save(path, theta)
        theta_paths[str(budget)] = path
    source_lock = read_json(directory / "simplification.lock.json")
    source_config = read_json(Path(source_lock["source_run"]) / "config.resolved.json")
    latency_rows = min(int(source_config["latency_subset_size"]), matrix.shape[0])
    latency_repeats = int(source_config["latency_repeats"])
    latency_matrix = matrix[:latency_rows].tocsr()
    latency_embeddings = embeddings[:latency_rows] if embeddings is not None else None
    warm_latency: dict[str, Any] = {}
    for budget in BUDGETS:
        session.infer(
            latency_matrix,
            budget=budget,
            embeddings=latency_embeddings,
        )
        repeats = []
        for _ in range(latency_repeats):
            started = time.perf_counter()
            session.infer(
                latency_matrix,
                budget=budget,
                embeddings=latency_embeddings,
            )
            repeats.append(time.perf_counter() - started)
        per_spectrum = np.asarray(repeats) / latency_rows
        warm_latency[str(budget)] = {
            "documents": latency_rows,
            "repeats": latency_repeats,
            "median_seconds_per_spectrum": float(np.median(per_spectrum)),
            "p95_seconds_per_spectrum": float(np.percentile(per_spectrum, 95)),
            "median_spectra_per_second": float(
                1.0 / max(float(np.median(per_spectrum)), 1e-12),
            ),
        }
    baseline_parity: dict[str, Any] | None = None
    if arm_id == "dreams_prior__dreams_semi" and split == "test":
        source = Path(read_json(directory / "simplification.lock.json")["source_run"])
        if representation == "observed":
            source_complete = read_json(source / "core/seed_42/hybrid/complete.json")
            source_paths = {
                budget: source / f"core/seed_42/hybrid/test_theta_{budget}.npy"
                for budget in (0, 2, 50)
            }
            source_hashes = source_complete["theta_sha256"]
        else:
            source_complete = read_json(
                source / "chemical_inference/seed_42/hybrid/complete.json",
            )
            source_paths = {
                0: source
                / "chemical_inference/seed_42/hybrid/test_full_theta_encoder.npy",
                2: source
                / "chemical_inference/seed_42/hybrid/test_full_theta_two_step.npy",
                50: source
                / "chemical_inference/seed_42/hybrid/test_full_theta_long.npy",
            }
            source_hashes = source_complete["theta_sha256"]
        exact = {}
        for budget in (0, 2, 50):
            source_theta = np.load(source_paths[budget], mmap_mode="r")
            exact[str(budget)] = bool(
                np.array_equal(np.load(theta_paths[str(budget)]), source_theta),
            )
        if not all(exact.values()):
            msg = "current baseline did not exactly reproduce published theta"
            raise RuntimeError(msg)
        baseline_parity = {
            "exact_array_equality": exact,
            "source_theta_sha256": source_hashes,
        }
    encoder_complete = read_json(directory / "encoders" / arm_id / "complete.json")
    result = {
        "schema_version": "msnlib-simplification/inference-complete-v1",
        "arm_id": arm_id,
        "discovery": discovery,
        "inference": inference,
        "split": split,
        "representation": representation,
        "budgets": list(BUDGETS),
        "rows": matrix.shape[0],
        "topics": load_spec(directory).num_topics,
        "uses_dreams": uses_dreams,
        "dreams_cache_loaded": uses_dreams,
        "chemical_labels_used_for_inference": False,
        "full_spectrum_protocol_renormalization": representation == "full",
        "inference_seconds": durations,
        "warm_model_only_latency": warm_latency,
        "peak_rss_bytes": peak_rss_bytes(),
        "parameter_count": encoder_complete["parameter_count"],
        "checkpoint_bytes": sum(
            path.stat().st_size
            for path in (directory / "encoders" / arm_id).glob("model.pt")
        ),
        "theta_sha256": {path.name: file_sha256(path) for path in theta_paths.values()},
        "corrected_current_baseline_parity": baseline_parity,
    }
    write_json(output / "inference_complete.json", result)
    return result


def _npmi_sparse(
    beta: np.ndarray,
    matrix: sp.csr_matrix,
    *,
    top_n: int,
) -> dict[str, float | int]:
    count = min(top_n, beta.shape[1])
    top_indices = np.argsort(-beta, axis=1, kind="stable")[:, :count]
    requested = set(map(int, top_indices.ravel()))
    requested_pairs = {
        pair
        for topic in top_indices
        for pair in combinations(sorted(map(int, topic)), 2)
    }
    neighbours: dict[int, set[int]] = {}
    for first, second in requested_pairs:
        neighbours.setdefault(first, set()).add(second)
    document_frequency: dict[int, int] = {}
    pair_frequency: dict[tuple[int, int], int] = {}
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        present = {
            int(value) for value in matrix.indices[start:stop] if value in requested
        }
        for first in present:
            document_frequency[first] = document_frequency.get(first, 0) + 1
            for second in neighbours.get(first, ()):
                if second in present:
                    pair = (first, second)
                    pair_frequency[pair] = pair_frequency.get(pair, 0) + 1
    topic_scores: list[float] = []
    undefined = 0
    for topic in top_indices:
        scores = []
        for first, second in combinations(sorted(map(int, topic)), 2):
            joint_count = pair_frequency.get((first, second), 0)
            if not joint_count:
                scores.append(-1.0)
                undefined += 1
                continue
            joint = joint_count / matrix.shape[0]
            first_p = document_frequency[first] / matrix.shape[0]
            second_p = document_frequency[second] / matrix.shape[0]
            scores.append(
                (
                    1.0
                    if joint == 1.0
                    else float(np.log(joint / (first_p * second_p)) / -np.log(joint))
                ),
            )
        topic_scores.append(float(np.mean(scores)))
    return {
        "mean_npmi": float(np.mean(topic_scores)),
        "median_topic_npmi": float(np.median(topic_scores)),
        "undefined_pairs_scored_as_minus_one": undefined,
    }


def _completion_rows(
    theta: np.ndarray,
    beta: np.ndarray,
    completion: sp.csr_matrix,
    metadata: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    losses = np.full(completion.shape[0], np.nan, dtype=np.float64)
    in_tokens = 0
    out_tokens = 0
    total_loss = 0.0
    eligible = 0
    for row in range(completion.shape[0]):
        start, stop = completion.indptr[row], completion.indptr[row + 1]
        words = completion.indices[start:stop]
        counts = completion.data[start:stop]
        count = int(counts.sum())
        out_tokens += int(metadata[row]["completion_oov_tokens"])
        if count:
            probabilities = theta[row] @ beta[:, words]
            loss = -float(np.sum(counts * np.log(np.clip(probabilities, 1e-12, None))))
            losses[row] = loss / count
            total_loss += loss
            in_tokens += count
            eligible += 1
    total = in_tokens + out_tokens
    return (
        {
            "nll_per_token": total_loss / in_tokens if in_tokens else None,
            "in_vocabulary_tokens": in_tokens,
            "out_of_vocabulary_tokens": out_tokens,
            "oov_fraction": out_tokens / total if total else None,
            "eligible_documents": eligible,
            "total_documents": completion.shape[0],
        },
        losses,
    )


def _discovery_metrics(run_dir: Path, discovery: str) -> dict[str, Any]:
    """Compute topic-only metrics once and verify them for every inference arm."""
    output = run_dir / "evaluation" / "discovery_metrics" / f"{discovery}.json"
    if output.is_file():
        return read_json(output)
    _, _, beta, complete = load_discovery(run_dir, discovery)
    train = load_count_matrix(run_dir, "train")
    source_lock = read_json(run_dir / "simplification.lock.json")
    source_config = read_json(Path(source_lock["source_run"]) / "config.resolved.json")
    result = {
        "schema_version": "msnlib-simplification/discovery-metrics-v1",
        "discovery": discovery,
        "discovery_sha256": object_sha256(complete),
        "top_word_diversity": top_word_diversity(
            beta,
            top_n=int(source_config["topic_top_n"]),
        ),
        "word_cooccurrence_npmi": _npmi_sparse(
            beta,
            train,
            top_n=int(source_config["topic_top_n"]),
        ),
    }
    write_json(output, result)
    return result


def build_common_reference(
    run_dir: str | Path,
    *,
    split: str,
    discovery: str,
) -> dict[str, Any]:
    """Select the highest-ELBO 50-step candidate independently per document."""
    directory = Path(run_dir).expanduser().resolve()
    spec = load_spec(directory)
    output = (
        directory / "evaluation" / split / "observed" / discovery / "common_reference"
    )
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"common-reference artifact changed: {name}"
                raise ValueError(msg)
        return result
    matrix = load_count_matrix(directory, f"{split}_observed")
    candidates: list[tuple[str, np.ndarray]] = []
    for inference in INFERENCE_IDS:
        arm_id = f"{discovery}__{inference}"
        evaluate_arm(directory, split=split, representation="observed", arm_id=arm_id)
        path = _arm_output(directory, split, "observed", arm_id) / "theta_50.npy"
        candidates.append((arm_id, np.load(path, mmap_mode="r")))
    analytic_arm = f"{discovery}__analytic"
    uniform = infer_theta(
        directory,
        analytic_arm,
        matrix,
        budget=spec.direct_target_steps,
        embeddings=None,
        initializer="uniform",
    )
    candidates.append((f"{discovery}__uniform", uniform))
    elbos = np.column_stack(
        [
            local_elbo_rows(directory, discovery, matrix, theta)
            for _, theta in candidates
        ],
    )
    choices = np.argmax(elbos, axis=1).astype(np.int16)
    reference = np.empty((matrix.shape[0], spec.num_topics), dtype=np.float32)
    for candidate_index, (_, theta) in enumerate(candidates):
        selected = choices == candidate_index
        reference[selected] = theta[selected]
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "theta.npy"
    choices_path = output / "choices.npy"
    elbo_path = output / "elbo.npy"
    _atomic_save(reference_path, reference)
    _atomic_save(choices_path, choices)
    _atomic_save(elbo_path, elbos)
    result = {
        "schema_version": "msnlib-simplification/common-reference-v1",
        "split": split,
        "discovery": discovery,
        "steps": spec.direct_target_steps,
        "selection": "rowwise_highest_local_elbo",
        "candidates": [name for name, _ in candidates],
        "choice_counts": {
            name: int(np.sum(choices == index))
            for index, (name, _) in enumerate(candidates)
        },
        "output_sha256": {
            path.name: file_sha256(path)
            for path in (reference_path, choices_path, elbo_path)
        },
    }
    write_json(complete_path, result)
    return result


def score_observed_arm(
    run_dir: str | Path,
    *,
    split: str,
    arm_id: str,
) -> dict[str, Any]:
    """Score one observed-spectrum arm against completion and common reference."""
    directory = Path(run_dir).expanduser().resolve()
    load_spec(directory)
    discovery, _ = arm_id.split("__", 1)
    output = _arm_output(directory, split, "observed", arm_id)
    metrics_path = output / "metrics_complete.json"
    if metrics_path.is_file():
        result = read_json(metrics_path)
        for name, digest in result["per_document_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"per-document result changed: {name}"
                raise ValueError(msg)
        return result
    inference = evaluate_arm(
        directory,
        split=split,
        representation="observed",
        arm_id=arm_id,
    )
    build_common_reference(directory, split=split, discovery=discovery)
    reference = np.load(
        directory
        / "evaluation"
        / split
        / "observed"
        / discovery
        / "common_reference"
        / "theta.npy",
        mmap_mode="r",
    )
    _, _, beta, _ = load_discovery(directory, discovery)
    completion = load_count_matrix(directory, f"{split}_completion")
    metadata = heldout_metadata(directory, split)
    source_lock = read_json(directory / "simplification.lock.json")
    source_config = read_json(Path(source_lock["source_run"]) / "config.resolved.json")
    discovery_metrics = _discovery_metrics(directory, discovery)
    metrics: dict[str, Any] = {}
    per_document_paths: dict[str, Path] = {}
    oov_fraction = np.asarray(
        [
            row["observed_oov_tokens"] / max(row["observed_tokens"], 1)
            for row in metadata
        ],
        dtype=np.float32,
    )
    quartile_edges = np.quantile(oov_fraction, [0.25, 0.5, 0.75]).tolist()
    quartiles = np.digitize(oov_fraction, quartile_edges, right=True)
    for budget in BUDGETS:
        theta = np.load(output / f"theta_{budget}.npy", mmap_mode="r")
        completion_metrics, nll_rows = _completion_rows(
            theta,
            beta,
            completion,
            metadata,
        )
        cosine = cosine_rows(theta, reference)
        js = jensen_shannon_rows(theta, reference)
        oov_strata = []
        for quartile in range(4):
            selected = quartiles == quartile
            finite_nll = selected & np.isfinite(nll_rows)
            oov_strata.append(
                {
                    "quartile": quartile + 1,
                    "documents": int(np.sum(selected)),
                    "eligible_nll_documents": int(np.sum(finite_nll)),
                    "nll_per_document_mean": (
                        float(np.mean(nll_rows[finite_nll]))
                        if np.any(finite_nll)
                        else None
                    ),
                    "cosine_mean": (
                        float(np.mean(cosine[selected])) if np.any(selected) else None
                    ),
                    "js_mean": (
                        float(np.mean(js[selected])) if np.any(selected) else None
                    ),
                },
            )
        metrics[str(budget)] = {
            "document_completion": completion_metrics,
            "active_topics": active_topic_metrics(
                theta,
                document_threshold=float(source_config["document_active_threshold"]),
                corpus_threshold=float(source_config["corpus_active_threshold"]),
            ),
            "convergence_to_common_reference": {
                "cosine_mean": float(np.mean(cosine)),
                "cosine_median": float(np.median(cosine)),
                "cosine_p05": float(np.percentile(cosine, 5)),
                "js_mean": float(np.mean(js)),
                "js_median": float(np.median(js)),
                "js_p95": float(np.percentile(js, 95)),
            },
            "oov_quartile_edges": quartile_edges,
            "oov_strata": oov_strata,
        }
        path = output / f"per_document_{budget}.npz"
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(
            temporary,
            nll_per_token=nll_rows,
            oov_fraction=oov_fraction,
            cosine_to_reference=cosine,
            js_to_reference=js,
        )
        temporary.replace(path)
        per_document_paths[path.name] = path
    result = {
        "schema_version": "msnlib-simplification/observed-metrics-v1",
        "arm_id": arm_id,
        "split": split,
        "inference_complete_sha256": file_sha256(output / "inference_complete.json"),
        "discovery_metrics": discovery_metrics,
        "metrics": metrics,
        "per_document_sha256": {
            name: file_sha256(path) for name, path in per_document_paths.items()
        },
        "inference_seconds": inference["inference_seconds"],
    }
    write_json(metrics_path, result)
    return result


def finalize_validation(run_dir: str | Path) -> dict[str, Any]:
    """Freeze all validation outputs before any test mixture is generated."""
    directory = Path(run_dir).expanduser().resolve()
    path = directory / "evaluation" / "validation" / "frozen.json"
    if path.is_file():
        result = read_json(path)
        for relative, digest in result["inventory"].items():
            if file_sha256(directory / relative) != digest:
                msg = f"frozen validation artifact changed: {relative}"
                raise ValueError(msg)
        return result
    for arm_id in ARM_IDS:
        evaluate_arm(
            directory,
            split="validation",
            representation="observed",
            arm_id=arm_id,
        )
        score_observed_arm(directory, split="validation", arm_id=arm_id)
        evaluate_arm(
            directory,
            split="validation",
            representation="full",
            arm_id=arm_id,
        )
    inventory: dict[str, str] = {}
    root = directory / "evaluation" / "validation"
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact != path:
            inventory[str(artifact.relative_to(directory))] = file_sha256(artifact)
    result = {
        "schema_version": "msnlib-simplification/validation-frozen-v1",
        "arms": list(ARM_IDS),
        "budgets": list(BUDGETS),
        "observed_and_full_spectrum_complete": True,
        "test_outputs_present_before_freeze": (directory / "evaluation/test").exists(),
        "inventory": inventory,
        "inventory_sha256": object_sha256(inventory),
    }
    if result["test_outputs_present_before_freeze"]:
        msg = "test outputs exist before validation was frozen"
        raise RuntimeError(msg)
    write_json(path, result)
    return result


def finalize_test(run_dir: str | Path) -> dict[str, Any]:
    """Perform the single frozen posthoc test pass for every arm."""
    directory = Path(run_dir).expanduser().resolve()
    finalize_validation(directory)
    path = directory / "evaluation" / "test" / "complete.json"
    if path.is_file():
        return read_json(path)
    for arm_id in ARM_IDS:
        evaluate_arm(directory, split="test", representation="observed", arm_id=arm_id)
        score_observed_arm(directory, split="test", arm_id=arm_id)
        evaluate_arm(directory, split="test", representation="full", arm_id=arm_id)
    inventory: dict[str, str] = {}
    root = directory / "evaluation" / "test"
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact != path:
            inventory[str(artifact.relative_to(directory))] = file_sha256(artifact)
    result = {
        "schema_version": "msnlib-simplification/test-complete-v1",
        "arms": list(ARM_IDS),
        "budgets": list(BUDGETS),
        "single_frozen_posthoc_pass": True,
        "validation_frozen_sha256": file_sha256(
            directory / "evaluation/validation/frozen.json",
        ),
        "inventory": inventory,
        "inventory_sha256": object_sha256(inventory),
    }
    write_json(path, result)
    return result


def topic_matching(run_dir: str | Path) -> dict[str, Any]:
    """Match the two discovery topic matrices at the same frozen seed."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "evaluation" / "topic_matching.json"
    if output.is_file():
        return read_json(output)
    _, _, current, _ = load_discovery(directory, "dreams_prior")
    _, _, symmetric, _ = load_discovery(directory, "symmetric_prior")
    result = {
        "schema_version": "msnlib-simplification/topic-matching-v1",
        "seed": load_spec(directory).seed,
        "left": "dreams_prior",
        "right": "symmetric_prior",
        "metrics": optimal_topic_matching(current, symmetric, top_n=10),
    }
    write_json(output, result)
    return result


def metric_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return flat factual rows for the neutral collection report."""
    directory = Path(run_dir).expanduser().resolve()
    rows = []
    for split in ("validation", "test"):
        for arm_id in ARM_IDS:
            metrics = read_json(
                _arm_output(directory, split, "observed", arm_id)
                / "metrics_complete.json",
            )
            inference = read_json(
                _arm_output(directory, split, "observed", arm_id)
                / "inference_complete.json",
            )
            for budget in BUDGETS:
                value = metrics["metrics"][str(budget)]
                rows.append(
                    {
                        "split": split,
                        "arm_id": arm_id,
                        "discovery": arm_id.split("__", 1)[0],
                        "inference": arm_id.split("__", 1)[1],
                        "budget": budget,
                        "nll_per_token": value["document_completion"]["nll_per_token"],
                        "eligible_documents": value["document_completion"][
                            "eligible_documents"
                        ],
                        "corpus_active_topics": value["active_topics"][
                            "corpus_active_topics"
                        ],
                        "cosine_mean": value["convergence_to_common_reference"][
                            "cosine_mean"
                        ],
                        "cosine_p05": value["convergence_to_common_reference"][
                            "cosine_p05"
                        ],
                        "js_mean": value["convergence_to_common_reference"]["js_mean"],
                        "inference_seconds": inference["inference_seconds"][
                            str(budget)
                        ],
                        "warm_seconds_per_spectrum_median": inference[
                            "warm_model_only_latency"
                        ][str(budget)]["median_seconds_per_spectrum"],
                        "warm_seconds_per_spectrum_p95": inference[
                            "warm_model_only_latency"
                        ][str(budget)]["p95_seconds_per_spectrum"],
                        "parameter_count": inference["parameter_count"],
                        "checkpoint_bytes": inference["checkpoint_bytes"],
                        "peak_rss_bytes": inference["peak_rss_bytes"],
                        "npmi": metrics["discovery_metrics"]["word_cooccurrence_npmi"][
                            "mean_npmi"
                        ],
                        "top_word_diversity": metrics["discovery_metrics"][
                            "top_word_diversity"
                        ],
                    },
                )
    return rows


def write_metric_csv(run_dir: str | Path, path: str | Path) -> None:
    """Write deterministic flat metrics without interpreting them."""
    rows = metric_rows(run_dir)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
