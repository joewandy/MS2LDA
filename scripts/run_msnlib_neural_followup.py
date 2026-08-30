"""Validation-only diagnostics for the real-MSnLib neural follow-up campaign.

The script never resolves, opens, or scores candidate test matrices.  It
reconstructs validation inference from saved weights, reuses fixed MAG
annotations for temperature studies, and writes only validation diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy import stats

from benchmarks.neural_ms2lda.chemical import score_precomputed_annotations
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
)
from benchmarks.neural_ms2lda.followup import (
    retemperature_theta,
    theta_distribution,
    top_rank_stability,
)
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.pooled import (
    PooledProjectedMS2LDA,
    infer_pooled_theta,
)
from benchmarks.neural_ms2lda.utils import (
    atomic_save_numpy,
    read_json,
    write_json,
)
from scripts.run_msnlib_model_comparison import (
    FragmentLossBalancedETM,
    GatedFragmentLossBalancedETM,
    configure,
    dense_normalized,
    infer_etm,
    mixture_diagnostics,
    resolve_device,
    sparse_reconstruction,
    topic_word_diagnostics,
    write_csv,
)
from scripts.run_published_topic_models_msnlib import FixedETM, sgns_only

POOLED_TEMPERATURES = (
    0.24,
    0.20,
    0.18,
    0.16,
    0.14,
    0.12,
    0.11,
    0.10,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
    0.04,
    0.03,
)
ETM_TEMPERATURES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)
GATES = {
    "optimized_motifs": 840,
    "evaluable_motifs": 388,
    "useful_motifs": 252,
    "mean_sos": 0.651498,
    "maximum_completion_nll": 9.422847,
}


def _completion_nll(metrics: dict[str, Any]) -> float:
    """Read the locked completion scorer's canonical per-token field."""
    return float(metrics["nll_per_token"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _annotations(run: Path, method: str) -> list[dict[str, Any]]:
    path = run / "mag/annotations" / method / "annotations.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"fixed MAG annotations are missing: {path}")
    return _read_jsonl(path)


def _pooled_model(
    run: Path,
    *,
    method: str,
    device: torch.device,
) -> tuple[PooledProjectedMS2LDA, dict[str, Any]]:
    config = read_json(run / "models" / method / "config.json")
    features = torch.from_numpy(
        np.load(run / "token_features/features.npy").astype(np.float32)
    )
    topic_indices = torch.as_tensor(config["topic_initial_indices"], dtype=torch.long)
    model = PooledProjectedMS2LDA(
        features,
        num_topics=int(config["topics"]),
        projection_dimensions=int(config["projection_dimensions"]),
        theta_temperature=float(config["theta_temperature"]),
        beta_temperature=float(config["beta_temperature"]),
        topic_initial_indices=topic_indices,
        seed=int(config["seed"]) + int(config["topics"]),
    )
    state = torch.load(
        run / "models" / method / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.to(device).eval(), config


def _etm_model(
    run: Path,
    *,
    method: str,
    device: torch.device,
) -> tuple[FixedETM, dict[str, Any]]:
    config = read_json(run / "models" / method / "config.json")
    embeddings = sgns_only(run / "token_features/features.npy")
    topics = int(config["topics"])
    if method == "etm_balanced":
        vocabulary = load_vocabulary(run / "data")
        fragment_mask = np.asarray(
            [word.startswith("frag@") for word in vocabulary], dtype=bool
        )
        model: FixedETM = FragmentLossBalancedETM(
            embeddings,
            topics,
            fragment_mask,
            hidden=int(config["hidden_dimensions"]),
        )
    elif method.startswith("etm_balanced_gated_"):
        vocabulary = load_vocabulary(run / "data")
        fragment_mask = np.asarray(
            [word.startswith("frag@") for word in vocabulary], dtype=bool
        )
        model = GatedFragmentLossBalancedETM(
            embeddings,
            topics,
            fragment_mask,
            gate_temperature=float(config["gate_temperature"]),
            gate_gamma=float(config["gate_gamma"]),
            hidden=int(config["hidden_dimensions"]),
        )
    elif method == "etm":
        model = FixedETM(
            embeddings,
            topics,
            hidden=int(config["hidden_dimensions"]),
        )
    else:
        raise ValueError(
            "ETM diagnostics support etm, etm_balanced, or a gated balanced ETM"
        )
    state = torch.load(
        run / "models" / method / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.to(device).eval(), config


def _method_inference(
    run: Path,
    *,
    method: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    observed = load_csr(run / "data/validation_observed.npz")
    full = load_csr(run / "data/validation_full.npz")
    saved_full = np.load(
        run / "validation_evaluation" / method / "validation_full_theta.npy",
        mmap_mode="r",
    )
    saved_beta = np.load(
        run / "validation_evaluation" / method / "beta.npy", mmap_mode="r"
    )
    if method.startswith("pooled_"):
        model, config = _pooled_model(run, method=method, device=device)
        batch_size = int(config["batch_size"])
        theta_observed = infer_pooled_theta(
            model, observed, batch_size=batch_size, device=device
        )
        reconstructed_full = infer_pooled_theta(
            model, full, batch_size=batch_size, device=device
        )
        with torch.inference_mode():
            reconstructed_beta = (
                model.topic_word_distribution().cpu().numpy().astype(np.float32)
            )
        source_temperature = float(config["theta_temperature"])
    else:
        model, config = _etm_model(run, method=method, device=device)
        batch_size = int(config["batch_size"])
        theta_observed, _ = infer_etm(
            model, observed, batch_size=batch_size, device=device
        )
        reconstructed_full, _ = infer_etm(
            model, full, batch_size=batch_size, device=device
        )
        with torch.inference_mode():
            reconstructed_beta = model.beta().cpu().numpy().astype(np.float32)
        source_temperature = 1.0
    checks = {
        "method": method,
        "validation_only": True,
        "saved_full_theta_shape": list(saved_full.shape),
        "max_abs_reconstructed_full_theta_difference": float(
            np.max(np.abs(reconstructed_full - np.asarray(saved_full)))
        ),
        "max_abs_reconstructed_beta_difference": float(
            np.max(np.abs(reconstructed_beta - np.asarray(saved_beta)))
        ),
    }
    return (
        theta_observed,
        np.asarray(saved_full),
        np.asarray(saved_beta),
        source_temperature,
        checks,
    )


def _chemistry_fields(
    summary: dict[str, Any],
    *,
    optimized_motifs: int,
) -> dict[str, Any]:
    rows = summary["topic_scores"]
    useful = sum(bool(row["eligible"]) and float(row["sos"]) >= 0.6 for row in rows)
    mean_sos = summary["mean_sos"]
    result = {
        "optimized_motifs": int(optimized_motifs),
        "evaluable_motifs": int(summary["eligible_topics"]),
        "useful_motifs": int(useful),
        "mean_sos": mean_sos,
        "median_sos": summary["median_sos"],
        "sos_high_gt_0_8": int(summary["sos_bands"]["high_gt_0_8"]),
        "sos_intermediate_0_6_to_0_8": int(
            summary["sos_bands"]["intermediate_0_6_to_0_8"]
        ),
        "sos_low_lt_0_6": int(summary["sos_bands"]["low_lt_0_6"]),
        "associated_spectra": int(summary["associated_spectra"]),
        "associated_molecules": int(summary["associated_molecules"]),
    }
    result.update(
        {
            "gate_optimized": optimized_motifs >= GATES["optimized_motifs"],
            "gate_evaluable": summary["eligible_topics"] >= GATES["evaluable_motifs"],
            "gate_useful": useful >= GATES["useful_motifs"],
            "gate_mean_sos": mean_sos is not None and mean_sos >= GATES["mean_sos"],
        }
    )
    return result


def _temperature_rows(
    *,
    theta_observed: np.ndarray,
    theta_full: np.ndarray,
    beta: np.ndarray,
    source_temperature: float,
    temperatures: Sequence[float],
    completion: Any,
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    optimized = sum(int(row["optimized_feature_count"]) > 0 for row in annotations)
    rows = []
    for temperature in temperatures:
        calibrated_observed = retemperature_theta(
            theta_observed,
            source_temperature=source_temperature,
            target_temperature=float(temperature),
        )
        calibrated_full = retemperature_theta(
            theta_full,
            source_temperature=source_temperature,
            target_temperature=float(temperature),
        )
        completion_result = completion_metrics(
            calibrated_observed, beta, completion, records
        )
        chemical = score_precomputed_annotations(
            theta=calibrated_full,
            records=records,
            annotations=annotations,
            membership_threshold=float(protocol["chemistry"]["membership_threshold"]),
            fingerprint_threshold=float(
                protocol["chemistry"]["mag_fingerprint_threshold"]
            ),
        )
        distribution = theta_distribution(calibrated_full)
        row = {
            "theta_temperature": float(temperature),
            **distribution,
            **top_rank_stability(theta_full, calibrated_full),
            **_chemistry_fields(chemical, optimized_motifs=optimized),
            "completion_nll": _completion_nll(completion_result),
            "completion_oov_fraction": float(completion_result["oov_fraction"]),
            "finite_stable": bool(
                np.all(np.isfinite(calibrated_observed))
                and np.all(np.isfinite(calibrated_full))
            ),
        }
        row["gate_completion_nll"] = (
            row["completion_nll"] <= GATES["maximum_completion_nll"]
        )
        row["passed_all_numeric_gates"] = bool(
            row["gate_optimized"]
            and row["gate_evaluable"]
            and row["gate_useful"]
            and row["gate_mean_sos"]
            and row["gate_completion_nll"]
            and row["finite_stable"]
        )
        rows.append(row)
    return rows


def _components(similarity: np.ndarray, threshold: float) -> dict[str, Any]:
    topics = similarity.shape[0]
    parent = np.arange(topics)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_rows = np.argwhere(np.triu(similarity, k=1) >= float(threshold))
    for left, right in pair_rows:
        union(int(left), int(right))
    sizes = Counter(find(topic) for topic in range(topics))
    duplicate_sizes = sorted(
        (size for size in sizes.values() if size > 1), reverse=True
    )
    return {
        "threshold": threshold,
        "pair_count": int(len(pair_rows)),
        "topics_in_duplicate_components": int(sum(duplicate_sizes)),
        "duplicate_component_count": int(len(duplicate_sizes)),
        "largest_component_size": int(max(duplicate_sizes, default=1)),
    }


def _largest_component_members(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Return topic ids in the largest thresholded cosine component."""
    topics = similarity.shape[0]
    parent = np.arange(topics)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for left, right in np.argwhere(np.triu(similarity, k=1) >= float(threshold)):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    roots = np.asarray([find(topic) for topic in range(topics)])
    values, counts = np.unique(roots, return_counts=True)
    return np.flatnonzero(roots == values[int(np.argmax(counts))])


def _redundancy_diagnostics(
    *,
    theta: np.ndarray,
    beta: np.ndarray,
    topic_prototypes: np.ndarray,
    annotations: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    values = np.asarray(theta, dtype=np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    normalized = np.array(beta, dtype=np.float32, copy=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, -1.0)
    nearest = np.argmax(similarity, axis=1)
    nearest_cosine = similarity[np.arange(len(similarity)), nearest]
    top1 = np.argmax(values, axis=1)
    top1_counts = np.bincount(top1, minlength=values.shape[1])
    usage = values.mean(axis=0)
    top3 = np.argpartition(-values, 2, axis=1)[:, :3]
    row_ids = np.arange(len(values))
    top_neighbour = nearest[top1]
    neighbour_probability = values[row_ids, top_neighbour]
    top_probability = values[row_ids, top1]
    neighbour_in_top3 = np.any(top3 == top_neighbour[:, None], axis=1)
    top20 = np.argpartition(-beta, 19, axis=1)[:, :20]
    optimized = np.asarray(
        [int(row["optimized_feature_count"]) > 0 for row in annotations], dtype=bool
    )
    largest_component = _largest_component_members(similarity, 0.999)
    in_largest_component = np.zeros(values.shape[1], dtype=bool)
    in_largest_component[largest_component] = True
    topic_rows = []
    for topic in range(values.shape[1]):
        neighbour = int(nearest[topic])
        overlap = len(set(top20[topic]).intersection(top20[neighbour])) / 20.0
        topic_rows.append(
            {
                "topic_id": topic,
                "nearest_topic_id": neighbour,
                "nearest_beta_cosine": float(nearest_cosine[topic]),
                "nearest_top20_word_overlap": float(overlap),
                "mean_theta_usage": float(usage[topic]),
                "top1_spectrum_count": int(top1_counts[topic]),
                "optimized": bool(optimized[topic]),
                "nearest_topic_optimized": bool(optimized[neighbour]),
                "in_largest_beta_component_ge_0_999": bool(in_largest_component[topic]),
            }
        )
    write_csv(output / "pooled_redundancy_diagnostics.csv", topic_rows)

    upper = np.triu_indices(len(similarity), k=1)
    pair_values = similarity[upper]
    keep = np.flatnonzero(pair_values >= 0.95)
    order = keep[np.argsort(-pair_values[keep], kind="stable")]
    stored_order = order[:1000]
    duplicate_rows = [
        {
            "topic_a": int(upper[0][index]),
            "topic_b": int(upper[1][index]),
            "beta_cosine": float(pair_values[index]),
            "top20_word_overlap": float(
                len(set(top20[upper[0][index]]).intersection(top20[upper[1][index]]))
                / 20.0
            ),
            "combined_mean_theta_usage": float(
                usage[upper[0][index]] + usage[upper[1][index]]
            ),
            "combined_top1_spectrum_count": int(
                top1_counts[upper[0][index]] + top1_counts[upper[1][index]]
            ),
        }
        for index in stored_order
    ]
    write_csv(output / "pooled_duplicate_topic_pairs.csv", duplicate_rows)
    correlation = stats.spearmanr(
        nearest_cosine[top1], top_probability, nan_policy="omit"
    )
    prototypes = np.asarray(topic_prototypes, dtype=np.float64)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    component_prototype_similarity = (
        prototypes[largest_component] @ prototypes[largest_component].T
    )
    component_triangle = component_prototype_similarity[
        np.triu_indices(len(largest_component), k=1)
    ]
    beta_entropy = -np.sum(
        np.asarray(beta, dtype=np.float64)
        * np.log(np.clip(np.asarray(beta, dtype=np.float64), 1e-300, None)),
        axis=1,
    )
    top20_mass = np.sort(np.asarray(beta, dtype=np.float64), axis=1)[:, -20:].sum(
        axis=1
    )
    summary = {
        "topics": int(values.shape[1]),
        "validation_spectra": int(values.shape[0]),
        "unique_top1_topics": int(np.count_nonzero(top1_counts)),
        "maximum_possible_evaluable_topics_under_rank_preserving_sharpening": int(
            np.count_nonzero(top1_counts)
        ),
        "topics_never_top1": int(np.sum(top1_counts == 0)),
        "optimized_topics_never_top1": int(np.sum(optimized & (top1_counts == 0))),
        "mean_nearest_beta_cosine": float(nearest_cosine.mean()),
        "nearest_beta_cosine_percentiles": {
            str(value): float(np.percentile(nearest_cosine, value))
            for value in (50, 75, 90, 95, 99)
        },
        "mean_nearest_neighbour_to_top_probability_ratio": float(
            np.mean(neighbour_probability / np.maximum(top_probability, 1e-12))
        ),
        "median_nearest_neighbour_to_top_probability_ratio": float(
            np.median(neighbour_probability / np.maximum(top_probability, 1e-12))
        ),
        "fraction_nearest_beta_neighbour_in_document_top3": float(
            np.mean(neighbour_in_top3)
        ),
        "spearman_nearest_beta_cosine_vs_document_max_theta": {
            "statistic": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        },
        "duplicate_components": [
            _components(similarity, threshold) for threshold in (0.95, 0.99, 0.999)
        ],
        "duplicate_pair_rows_saved": int(len(stored_order)),
        "duplicate_pair_rows_sort": "top beta cosine among pairs >=0.95",
        "largest_near_exact_component": {
            "threshold": 0.999,
            "topics": int(len(largest_component)),
            "optimized_topics": int(np.sum(optimized[largest_component])),
            "top1_spectra": int(np.sum(top1_counts[largest_component])),
            "topics_ever_top1": int(np.count_nonzero(top1_counts[largest_component])),
            "prototype_cosine_minimum": float(component_triangle.min()),
            "prototype_cosine_median": float(np.median(component_triangle)),
            "prototype_cosine_maximum": float(component_triangle.max()),
            "median_beta_effective_words": float(
                np.median(np.exp(beta_entropy[largest_component]))
            ),
            "median_beta_max_probability": float(
                np.median(np.max(beta, axis=1)[largest_component])
            ),
            "median_beta_top20_mass": float(np.median(top20_mass[largest_component])),
        },
    }
    write_json(output / "pooled_redundancy_summary.json", summary)
    return summary


def _pooled_redundancy_headline(run: Path, method: str) -> dict[str, Any]:
    """Compare collapse counts across already-trained pooled objectives."""
    beta = np.load(run / "validation_evaluation" / method / "beta.npy")
    normalized_beta = beta.astype(np.float64)
    normalized_beta /= np.maximum(
        np.linalg.norm(normalized_beta, axis=1, keepdims=True), 1e-12
    )
    beta_similarity = normalized_beta @ normalized_beta.T
    np.fill_diagonal(beta_similarity, -1.0)
    state = torch.load(
        run / "models" / method / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    prototypes = state["topic_prototypes"].numpy().astype(np.float64)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    prototype_similarity = prototypes @ prototypes.T
    np.fill_diagonal(prototype_similarity, -1.0)
    theta = np.load(
        run / "validation_evaluation" / method / "validation_full_theta.npy",
        mmap_mode="r",
    )
    return {
        "method": method,
        "beta_pairs_cosine_ge_0_999": int(
            np.sum(np.triu(beta_similarity, k=1) >= 0.999)
        ),
        "topics_with_beta_neighbour_cosine_ge_0_999": int(
            np.sum(np.max(beta_similarity, axis=1) >= 0.999)
        ),
        "prototype_pairs_cosine_ge_0_999": int(
            np.sum(np.triu(prototype_similarity, k=1) >= 0.999)
        ),
        "unique_top1_topics": int(len(np.unique(np.argmax(theta, axis=1)))),
    }


def diagnose_existing(
    run: Path,
    *,
    m1_run: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(run / "protocol.json")
    records = load_heldout_records(run / "data", "validation")
    completion = load_csr(run / "data/validation_completion.npz")
    run_records = run / "data/validation_records.jsonl"
    m1_records = m1_run / "data/validation_records.jsonl"
    record_hashes = {
        "followup_validation_records_sha256": _sha256(run_records),
        "m1_validation_records_sha256": _sha256(m1_records),
    }
    if len(set(record_hashes.values())) != 1:
        raise ValueError("M1 and follow-up validation records do not match")

    pooled = _method_inference(run, method="pooled_likelihood", device=device)
    pooled_rows = _temperature_rows(
        theta_observed=pooled[0],
        theta_full=pooled[1],
        beta=pooled[2],
        source_temperature=pooled[3],
        temperatures=POOLED_TEMPERATURES,
        completion=completion,
        records=records,
        annotations=_annotations(run, "pooled_likelihood"),
        protocol=protocol,
    )
    write_csv(output / "pooled_temperature_sweep.csv", pooled_rows)
    distribution_fields = [
        key
        for key in pooled_rows[0]
        if key.startswith("max_theta_")
        or key.startswith("fraction_max_theta_")
        or key
        in {
            "theta_temperature",
            "median_effective_topics_per_spectrum",
            "mean_effective_topics_per_spectrum",
            "median_max_theta",
            "unique_top1_topics",
        }
    ]
    write_csv(
        output / "pooled_theta_distribution.csv",
        [{key: row[key] for key in distribution_fields} for row in pooled_rows],
    )
    redundancy = _redundancy_diagnostics(
        theta=pooled[1],
        beta=pooled[2],
        topic_prototypes=torch.load(
            run / "models/pooled_likelihood/weights.pt",
            map_location="cpu",
            weights_only=True,
        )["topic_prototypes"].numpy(),
        annotations=_annotations(run, "pooled_likelihood"),
        output=output,
    )
    pooled_objective_comparison = [
        _pooled_redundancy_headline(run, method)
        for method in ("pooled_likelihood", "pooled_mi005")
    ]
    write_json(
        output / "pooled_mi_redundancy_comparison.json",
        {"comparison": pooled_objective_comparison},
    )

    etm = _method_inference(run, method="etm", device=device)
    etm_rows = _temperature_rows(
        theta_observed=etm[0],
        theta_full=etm[1],
        beta=etm[2],
        source_temperature=etm[3],
        temperatures=ETM_TEMPERATURES,
        completion=completion,
        records=records,
        annotations=_annotations(run, "etm"),
        protocol=protocol,
    )
    write_csv(output / "etm_temperature_diagnostic.csv", etm_rows)

    m1_theta = np.load(
        m1_run / "validation_evaluation/neural/validation_full_theta.npy",
        mmap_mode="r",
    )
    if m1_theta.shape != pooled[1].shape:
        raise ValueError("M1 and candidate validation theta shapes do not match")
    m1_distribution = theta_distribution(m1_theta)
    write_csv(output / "m1_theta_distribution.csv", [m1_distribution])
    result = {
        "evidence_boundary": "validation only; candidate test artifacts untouched",
        "validation_record_compatibility": record_hashes,
        "reconstruction_checks": [pooled[4], etm[4]],
        "pooled_top1_upper_bound": redundancy[
            "maximum_possible_evaluable_topics_under_rank_preserving_sharpening"
        ],
        "pooled_objective_redundancy": pooled_objective_comparison,
        "m1_theta_distribution": m1_distribution,
    }
    write_json(output / "existing_diagnostics.json", result)
    return result


def balanced_smoke(
    run: Path,
    *,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 7001, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    vocabulary = load_vocabulary(run / "data")
    embeddings = sgns_only(run / "token_features/features.npy")
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary], dtype=bool
    )
    model = FragmentLossBalancedETM(
        embeddings,
        int(protocol["model"]["num_topics"]),
        fragment_mask,
        hidden=800,
    ).to(device)
    rows = np.arange(min(8, train.shape[0]), dtype=np.int64)
    theta, kl = model.theta(dense_normalized(train, rows, device), sample=True)
    beta = model.beta()
    reconstruction = sparse_reconstruction(theta, beta, train, rows, device)
    objective = reconstruction + kl.mean()
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "validation_only": True,
        "objective": float(objective.detach().cpu()),
        "reconstruction": float(reconstruction.detach().cpu()),
        "kl": float(kl.mean().detach().cpu()),
        "finite_objective": bool(torch.isfinite(objective)),
        "finite_gradients": bool(
            gradients and all(torch.all(torch.isfinite(value)) for value in gradients)
        ),
        "fragment_mass_min": float(beta[:, fragment_mask].sum(1).min().detach().cpu()),
        "fragment_mass_max": float(beta[:, fragment_mask].sum(1).max().detach().cpu()),
        "loss_mass_min": float(beta[:, ~fragment_mask].sum(1).min().detach().cpu()),
        "loss_mass_max": float(beta[:, ~fragment_mask].sum(1).max().detach().cpu()),
    }
    write_json(output, result)
    return result


def diagnose_etm_temperature(
    run: Path,
    *,
    method: str,
    temperatures: Sequence[float],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Sweep one trained ETM variant after its fixed beta has been annotated."""
    if method not in {"etm", "etm_balanced"} and not method.startswith(
        "etm_balanced_gated_"
    ):
        raise ValueError("temperature sweep requires an ETM method")
    protocol = read_json(run / "protocol.json")
    records = load_heldout_records(run / "data", "validation")
    completion = load_csr(run / "data/validation_completion.npz")
    observed, full, beta, source_temperature, checks = _method_inference(
        run, method=method, device=device
    )
    rows = _temperature_rows(
        theta_observed=observed,
        theta_full=full,
        beta=beta,
        source_temperature=source_temperature,
        temperatures=temperatures,
        completion=completion,
        records=records,
        annotations=_annotations(run, method),
        protocol=protocol,
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{method}_temperature_sweep.csv"
    write_csv(path, rows)
    result = {
        "method": method,
        "temperatures": list(map(float, temperatures)),
        "rows": len(rows),
        "output": str(path),
        "reconstruction_check": checks,
        "maximum_optimized_motifs": max(row["optimized_motifs"] for row in rows),
        "maximum_evaluable_motifs": max(row["evaluable_motifs"] for row in rows),
        "maximum_useful_motifs": max(row["useful_motifs"] for row in rows),
    }
    write_json(output / f"{method}_temperature_sweep_summary.json", result)
    return result


def select_temperature(
    run: Path,
    *,
    method: str,
    temperature: float,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    protocol = read_json(run / "protocol.json")
    records = load_heldout_records(run / "data", "validation")
    completion = load_csr(run / "data/validation_completion.npz")
    observed, full, beta, source_temperature, checks = _method_inference(
        run, method=method, device=device
    )
    calibrated_observed = retemperature_theta(
        observed,
        source_temperature=source_temperature,
        target_temperature=temperature,
    )
    calibrated_full = retemperature_theta(
        full,
        source_temperature=source_temperature,
        target_temperature=temperature,
    )
    annotations = _annotations(run, method)
    chemical = score_precomputed_annotations(
        theta=calibrated_full,
        records=records,
        annotations=annotations,
        membership_threshold=float(protocol["chemistry"]["membership_threshold"]),
        fingerprint_threshold=float(protocol["chemistry"]["mag_fingerprint_threshold"]),
    )
    optimized = sum(int(row["optimized_feature_count"]) > 0 for row in annotations)
    completion_result = completion_metrics(
        calibrated_observed, beta, completion, records
    )
    metrics = {
        "method": method,
        "post_hoc_inference_temperature": float(temperature),
        "source_temperature": float(source_temperature),
        "beta_unchanged": True,
        "validation_only": True,
        "reconstruction_check": checks,
        "theta_distribution": theta_distribution(calibrated_full),
        "topic_inventory": mixture_diagnostics(calibrated_full, beta),
        "topic_words": topic_word_diagnostics(beta, load_vocabulary(run / "data"))[0],
        "completion": completion_result,
        "chemistry": _chemistry_fields(chemical, optimized_motifs=optimized),
        "rank_stability": top_rank_stability(full, calibrated_full),
    }
    metrics["chemistry"]["gate_completion_nll"] = (
        _completion_nll(completion_result) <= GATES["maximum_completion_nll"]
    )
    label = f"{method}_tau_{temperature:.3f}".replace(".", "p")
    local = run / "followup_validation" / label
    local.mkdir(parents=True, exist_ok=True)
    full_path = local / "validation_full_theta.npy"
    observed_path = local / "validation_observed_theta.npy"
    atomic_save_numpy(full_path, calibrated_full)
    atomic_save_numpy(observed_path, calibrated_observed)
    write_json(local / "metrics.json", metrics)
    metrics["large_local_artifacts"] = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (full_path, observed_path)
    ]
    output.mkdir(parents=True, exist_ok=True)
    output_name = (
        "pooled_temperature_selected_metrics.json"
        if method == "pooled_likelihood"
        else f"{method}_temperature_selected_metrics.json"
    )
    write_json(output / output_name, metrics)
    score_name = (
        "pooled_temperature_selected_chemical_scores.csv"
        if method == "pooled_likelihood"
        else f"{method}_temperature_selected_chemical_scores.csv"
    )
    write_csv(output / score_name, chemical["topic_scores"])
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose = commands.add_parser("diagnose-existing")
    diagnose.add_argument("--run", required=True, type=Path)
    diagnose.add_argument("--m1-run", required=True, type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    diagnose.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu"
    )

    smoke = commands.add_parser("balanced-smoke")
    smoke.add_argument("--run", required=True, type=Path)
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu"
    )

    select = commands.add_parser("select-temperature")
    select.add_argument("--run", required=True, type=Path)
    select.add_argument("--method", required=True)
    select.add_argument("--temperature", required=True, type=float)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu"
    )

    sweep = commands.add_parser("sweep-etm-temperature")
    sweep.add_argument("--run", required=True, type=Path)
    sweep.add_argument("--method", required=True)
    sweep.add_argument(
        "--temperatures", nargs="+", type=float, default=ETM_TEMPERATURES
    )
    sweep.add_argument("--output", required=True, type=Path)
    sweep.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu"
    )

    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    device = resolve_device(args.device)
    if args.command == "diagnose-existing":
        result = diagnose_existing(
            run,
            m1_run=args.m1_run.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            device=device,
        )
    elif args.command == "balanced-smoke":
        result = balanced_smoke(
            run,
            output=args.output.expanduser().resolve(),
            device=device,
        )
    elif args.command == "sweep-etm-temperature":
        result = diagnose_etm_temperature(
            run,
            method=args.method,
            temperatures=args.temperatures,
            output=args.output.expanduser().resolve(),
            device=device,
        )
    else:
        result = select_temperature(
            run,
            method=args.method,
            temperature=args.temperature,
            output=args.output.expanduser().resolve(),
            device=device,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
