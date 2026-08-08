"""Strict result collation for the frozen MSnLib benchmark."""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np

from .config import file_sha256, load_config, read_json, resolve_input_paths, write_json
from .mag import _consensus_fingerprint
from .metrics import optimal_topic_matching
from .protocol import load_vocabulary, verify_frozen_input_files, verify_protocol


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _require_completed_runs(directory: Path, seeds: Sequence[int]) -> None:
    if not (directory / "core" / "complete.json").is_file():
        raise RuntimeError("core benchmark is incomplete")
    if not (directory / "mag" / "complete.json").is_file():
        raise RuntimeError("MAG benchmark is incomplete")
    if not (directory / "chemical_inference" / "complete.json").is_file():
        raise RuntimeError("full-spectrum chemical inference is incomplete")
    missing = []
    for seed in seeds:
        for method in ("tomotopy", "hybrid"):
            for phase in ("core", "mag"):
                path = directory / phase / f"seed_{seed}" / method / "complete.json"
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise RuntimeError(f"required seed outputs are missing: {missing}")
    if not (directory / "mag" / "raw_dreams" / "complete.json").is_file():
        raise RuntimeError("raw-DreaMS baseline is incomplete")


def _metric_row(result: dict[str, Any], *, arm: str, config) -> dict[str, Any]:
    method = str(result["method"])
    metrics = result["metrics"] if method == "tomotopy" else result["metrics"][arm]
    document = metrics["document_completion"]
    active = metrics["active_topics"]
    npmi = metrics["word_cooccurrence_npmi"]
    if method == "tomotopy":
        inference_key = "standard"
        training_seconds = result["training_seconds_total"]
        inference_seconds = result["inference_seconds"]
        cached = result["cached_latency"]
        end_to_end = result["end_to_end_latency"]
        reference_steps = ""
        convergence = {}
        includes_dreams = False
        training_worker_request = result["training_workers_requested"]
        training_parallel_scheme = result["training_parallel_scheme"]
        inference_cpu_threads = 1
        checkpoint_resumed = False
    else:
        reference_steps = int(result["reference_steps"])
        inference_key = {
            "iter_0": "0",
            "iter_2": "2",
            "long": str(reference_steps),
        }[arm]
        training_seconds = result["discovery_seconds"] + result["finalization_seconds"]
        inference_seconds = result["inference_seconds"][inference_key]
        cached = result["cached_latency"][inference_key]
        end_to_end = result["end_to_end_latency"][inference_key]
        convergence = metrics.get("convergence_to_long", {})
        includes_dreams = True
        training_worker_request = result["training_cpu_threads"]
        training_parallel_scheme = "PYTORCH_INTRAOP"
        inference_cpu_threads = result["inference_cpu_threads"]
        checkpoint_resumed = result["checkpointing"]["resume_audit"]["resumed"]
    return {
        "seed": int(result["seed"]),
        "method": method,
        "inference_arm": arm,
        "inference_iterations": inference_key,
        "reference_iterations": reference_steps,
        "nll_per_token": document["nll_per_token"],
        "completion_in_vocabulary_tokens": document["in_vocabulary_tokens"],
        "completion_out_of_vocabulary_tokens": document["out_of_vocabulary_tokens"],
        "completion_oov_fraction": document["oov_fraction"],
        "completion_eligible_documents": document["eligible_documents"],
        "active_topics_document_mean": active["document_active_mean"],
        "active_topics_document_median": active["document_active_median"],
        "active_topics_document_p95": active["document_active_p95"],
        "active_topics_corpus": active["corpus_active_topics"],
        "top_word_diversity": metrics["top_word_diversity"],
        "word_cooccurrence_npmi_mean": npmi["mean_npmi"],
        "word_cooccurrence_npmi_median": npmi["median_topic_npmi"],
        "convergence_cosine_mean": convergence.get("cosine_mean"),
        "convergence_cosine_median": convergence.get("cosine_median"),
        "convergence_cosine_p05": convergence.get("cosine_p05"),
        "convergence_js_mean": convergence.get("js_mean"),
        "nll_gap_fraction_to_long": metrics.get("nll_gap_fraction"),
        "training_seconds": training_seconds,
        "training_worker_request": training_worker_request,
        "training_parallel_scheme": training_parallel_scheme,
        "training_bitwise_reproducible": result["training_bitwise_reproducible"],
        "inference_cpu_threads": inference_cpu_threads,
        "checkpoint_resumed": checkpoint_resumed,
        "alpha_initial_sum": result["alpha"]["initial"]["sum"],
        "alpha_final_sum": result["alpha"]["final"]["sum"],
        "alpha_final_minimum": result["alpha"]["final"]["minimum"],
        "alpha_final_median": result["alpha"]["final"]["median"],
        "alpha_final_maximum": result["alpha"]["final"]["maximum"],
        "inference_seconds_all_test_spectra": inference_seconds,
        "peak_rss_bytes": result["peak_rss_bytes"],
        "cached_seconds_per_spectrum_median": cached["median_seconds_per_spectrum"],
        "cached_spectra_per_second_median": cached["median_spectra_per_second"],
        "end_to_end_seconds_per_spectrum_median": end_to_end[
            "median_seconds_per_spectrum"
        ],
        "end_to_end_spectra_per_second_median": end_to_end["median_spectra_per_second"],
        "end_to_end_includes_dreams_extraction": includes_dreams,
        "latency_subset_size": config.latency_subset_size,
        "latency_repeats": config.latency_repeats,
    }


def _core_rows(directory: Path, config) -> list[dict[str, Any]]:
    rows = []
    for seed in config.seeds:
        tomotopy = read_json(
            directory / "core" / f"seed_{seed}" / "tomotopy" / "complete.json"
        )
        hybrid = read_json(
            directory / "core" / f"seed_{seed}" / "hybrid" / "complete.json"
        )
        if not tomotopy.get("converged"):
            raise RuntimeError(
                f"Tomotopy reached its frozen maximum without convergence for seed {seed}"
            )
        if not hybrid.get("reference_converged"):
            raise RuntimeError(
                f"Hybrid reference was not near-converged for seed {seed}"
            )
        if not hybrid.get("discovery_converged"):
            raise RuntimeError(f"Hybrid discovery did not converge for seed {seed}")
        rows.append(_metric_row(tomotopy, arm="standard", config=config))
        for arm in ("iter_0", "iter_2", "long"):
            rows.append(_metric_row(hybrid, arm=arm, config=config))
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows have inconsistent schemas")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_beta(directory: Path, seed: int, method: str) -> np.ndarray:
    path = directory / "core" / f"seed_{seed}" / method / "beta.npy"
    result = read_json(path.parent / "complete.json")
    if file_sha256(path) != result["beta_sha256"]:
        raise ValueError(f"topic matrix changed: {path}")
    return np.load(path, mmap_mode="r")


def _topic_similarity_rows(directory: Path, config) -> list[dict[str, Any]]:
    rows = []
    for method in ("tomotopy", "hybrid"):
        for left_seed, right_seed in combinations(config.seeds, 2):
            metrics = optimal_topic_matching(
                _load_beta(directory, left_seed, method),
                _load_beta(directory, right_seed, method),
                top_n=config.topic_top_n,
            )
            rows.append(
                {
                    "comparison": "cross_seed_topic_stability",
                    "method": method,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "interpretation": "similarity_not_correctness",
                    **metrics,
                }
            )
    for seed in config.seeds:
        metrics = optimal_topic_matching(
            _load_beta(directory, seed, "tomotopy"),
            _load_beta(directory, seed, "hybrid"),
            top_n=config.topic_top_n,
        )
        rows.append(
            {
                "comparison": "same_seed_between_methods",
                "method": "tomotopy_vs_hybrid",
                "left_seed": seed,
                "right_seed": seed,
                "interpretation": "similarity_not_correctness",
                **metrics,
            }
        )
    return rows


def _fingerprint_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _mag_summary(
    directory: Path,
    config,
    topic_similarity: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    agreement = []
    matching_by_seed = {
        int(row["left_seed"]): row
        for row in topic_similarity
        if row["comparison"] == "same_seed_between_methods"
    }
    for seed in config.seeds:
        model_rows = {}
        topic_rows = {}
        for method in ("tomotopy", "hybrid"):
            path = directory / "mag" / f"seed_{seed}" / method
            model_rows[method] = read_json(path / "complete.json")
            chemical_complete = (
                directory
                / "chemical_inference"
                / f"seed_{seed}"
                / method
                / "complete.json"
            )
            if (
                file_sha256(chemical_complete)
                != model_rows[method]["chemical_inference_complete_sha256"]
            ):
                raise ValueError(
                    f"chemical inference changed after MAG for {method} seed {seed}"
                )
            if (
                file_sha256(path / "topics.jsonl")
                != model_rows[method]["topics_sha256"]
            ):
                raise ValueError(f"MAG topic rows changed for {method} seed {seed}")
            all_topic_rows = _jsonl_rows(path / "topics.jsonl")
            if len(all_topic_rows) != model_rows[method]["topic_rows"]:
                raise RuntimeError(
                    f"incomplete MAG topic rows for {method} seed {seed}"
                )
            comparison_arm = "standard" if method == "tomotopy" else "long"
            topic_rows[method] = [
                row
                for row in all_topic_rows
                if row["inference_arm"] == comparison_arm
                and row["association_mode"] == "dominant_topic"
            ]
            if len(topic_rows[method]) != config.num_topics:
                raise RuntimeError(
                    f"incomplete primary MAG rows for {method} seed {seed}"
                )
            common = {
                key: model_rows[method][key]
                for key in (
                    "annotation_coverage",
                    "cluster_failures",
                    "clustered_topics",
                    "chemical_inference_complete_sha256",
                    "index_exclusion_audit",
                    "mag_annotation_available_topics",
                    "mag_seconds",
                    "optimization_failures",
                    "peak_rss_bytes",
                    "sos_definitions",
                )
            }
            per_seed.extend(
                {**common, **association}
                for association in model_rows[method]["association_results"]
            )
        matched = matching_by_seed[int(seed)]
        values = []
        both_annotated = 0
        for left_topic, right_topic in zip(
            matched["left_topic_ids"], matched["right_topic_ids"], strict=True
        ):
            left_row = topic_rows["tomotopy"][int(left_topic)]
            right_row = topic_rows["hybrid"][int(right_topic)]
            if (
                not left_row["mag_annotation_available"]
                or not right_row["mag_annotation_available"]
            ):
                continue
            left = _consensus_fingerprint(
                left_row["clustered_smiles"],
                config.mag_fingerprint_threshold,
            )
            right = _consensus_fingerprint(
                right_row["clustered_smiles"],
                config.mag_fingerprint_threshold,
            )
            if left is None or right is None:
                continue
            both_annotated += 1
            values.append(_fingerprint_jaccard(left, right))
        agreement.append(
            {
                "seed": seed,
                "matched_topics": config.num_topics,
                "both_annotated_topics": both_annotated,
                "both_annotated_fraction": both_annotated / config.num_topics,
                "maccs_consensus_jaccard_mean": (
                    float(np.mean(values)) if values else None
                ),
                "maccs_consensus_jaccard_median": (
                    float(np.median(values)) if values else None
                ),
                "interpretation": "MAG_annotation_similarity_not_correctness",
            }
        )
    return per_seed, agreement


def _published_beta(
    path: Path, vocabulary: Sequence[str], *, digits: int, expected_topics: int
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=lambda _: math.nan
    )
    motifs = payload.get("ms2", [])
    if len(motifs) != expected_topics:
        raise ValueError("deposited MSnLib motif count differs from frozen topic count")
    columns = {word: index for index, word in enumerate(vocabulary)}
    beta = np.zeros((expected_topics, len(vocabulary)), dtype=np.float32)
    retained_features = 0
    omitted_features = 0
    for row, motif in enumerate(motifs):
        for kind, value_key, weight_key in (
            ("frag", "frag_mz", "frag_intens"),
            ("loss", "loss_mz", "loss_intens"),
        ):
            for value, weight in zip(
                motif.get(value_key, []), motif.get(weight_key, []), strict=True
            ):
                if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
                    continue
                word = f"{kind}@{round(float(value), digits)}"
                column = columns.get(word)
                if column is None:
                    omitted_features += 1
                    continue
                beta[row, column] += float(weight)
                retained_features += 1
    empty = int((beta.sum(axis=1) == 0).sum())
    return beta, {
        "source_topics": len(motifs),
        "retained_top_features": retained_features,
        "features_outside_training_vocabulary": omitted_features,
        "empty_topics_after_alignment": empty,
        "scope": "same_dataset_full_corpus_deposit_context_only",
        "interpretation": "truncated_top_feature_similarity_not_correctness",
    }


def _published_similarity(directory: Path, config, data_root: Path) -> dict[str, Any]:
    inputs = resolve_input_paths(config, data_root)
    vocabulary = load_vocabulary(directory)
    published, audit = _published_beta(
        inputs["published_motifset"],
        vocabulary,
        digits=config.significant_digits,
        expected_topics=config.num_topics,
    )
    rows = []
    for seed in config.seeds:
        for method in ("tomotopy", "hybrid"):
            metrics = optimal_topic_matching(
                _load_beta(directory, seed, method),
                published,
                top_n=config.topic_top_n,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "interpretation": "same_dataset_similarity_not_correctness",
                    **metrics,
                }
            )
    return {"audit": audit, "per_seed": rows}


def _aggregate(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['method']}:{row['inference_arm']}"
        groups.setdefault(key, []).append(row)
    output = {}
    for key, group in groups.items():
        output[key] = {}
        for field in fields:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            if values:
                output[key][field] = {
                    "median": median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "n_seeds": len(values),
                }
    return output


def _manuscript_text(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate_core"]
    if summary["evidence_scope"] == "indicative_single_seed":
        caption = (
            "Single-seed indicative held-out and latency results; "
            "these are not cross-seed confirmatory evidence."
        )
    else:
        caption = (
            "Seed-median held-out and latency results; every seed is in the "
            "machine-readable table."
        )
    lines = [
        "% Generated by benchmarks.msnlib_validation; do not edit by hand.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "Method & inference & NLL/token & active topics & cached ms & end-to-end ms \\\\",
        "\\hline",
    ]
    labels = {
        "tomotopy:standard": ("Tomotopy", "100 Gibbs"),
        "hybrid:iter_0": ("HybridLDA", "encoder"),
        "hybrid:iter_2": ("HybridLDA", "encoder + 2 VB"),
        "hybrid:long": ("HybridLDA", "long VB"),
    }
    for key, (method, arm) in labels.items():
        values = aggregate[key]
        lines.append(
            f"{method} & {arm} & "
            f"{values['nll_per_token']['median']:.4f} & "
            f"{values['active_topics_corpus']['median']:.0f} & "
            f"{1000 * values['cached_seconds_per_spectrum_median']['median']:.3f} & "
            f"{1000 * values['end_to_end_seconds_per_spectrum_median']['median']:.3f} \\\\"  # noqa: E501
        )
    lines.extend(
        [
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\end{table}",
            "",
        ]
    )
    lines.extend(
        [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrr}",
            "Hybrid inference & cosine mean & cosine median & cosine p05 & JS mean \\\\",
            "\\hline",
        ]
    )
    for key, arm in (
        ("hybrid:iter_0", "encoder"),
        ("hybrid:iter_2", "encoder + 2 VB"),
    ):
        values = aggregate[key]
        lines.append(
            f"{arm} & "
            f"{values['convergence_cosine_mean']['median']:.4f} & "
            f"{values['convergence_cosine_median']['median']:.4f} & "
            f"{values['convergence_cosine_p05']['median']:.4f} & "
            f"{values['convergence_js_mean']['median']:.4f} \\\\"
        )
    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Convergence toward the validated long-refinement reference. The mean and fifth percentile expose tail behaviour that a median alone can hide.}",  # noqa: E501
            "\\end{table}",
            "",
        ]
    )
    sos_rows = [
        row
        for row in summary["mag_per_seed"]
        if row["association_mode"] == "dominant_topic"
    ]
    derivation_kind = (summary.get("protocol_derivation") or {}).get("kind")
    evaluation_timing = summary.get("evaluation_timing", "prespecified")
    if (
        derivation_kind == "implementation_correction"
        or evaluation_timing == "posthoc_implementation_correction"
    ):
        sos_caption = (
            "Post-hoc implementation-corrected single-seed SOS diagnostic "
            "using full held-out spectra and dominant-topic association. "
            "This is not confirmatory."
        )
    elif derivation_kind == "chemical_evaluation_correction":
        sos_caption = (
            "Post-hoc corrected single-seed SOS diagnostic using full held-out "
            "spectra and dominant-topic association. This is not confirmatory."
        )
    elif summary["evidence_scope"] == "indicative_single_seed":
        sos_caption = (
            "Prespecified single-seed indicative SOS using full held-out spectra "
            "and dominant-topic association. This is not confirmatory."
        )
    else:
        sos_caption = (
            "Five-seed confirmatory SOS using full held-out spectra and "
            "dominant-topic association; every seed is reported."
        )
    sos_caption += (
        " Means are compound-balanced within topic. The annotation-containment "
        "and smaller-fingerprint denominators are both shown. Fingerprint "
        "settings are frozen equally across methods and do not reproduce the "
        "paper's downstream RDKit/0.9 analysis."
    )

    def optional(value: float | None) -> str:
        return "--" if value is None else f"{value:.4f}"

    labels = {
        ("tomotopy", "standard"): ("Tomotopy", "standard"),
        ("hybrid", "encoder"): ("HybridLDA", "encoder"),
        ("hybrid", "two_step"): ("HybridLDA", "encoder + 2 VB"),
        ("hybrid", "long"): ("HybridLDA", "long VB"),
    }
    lines.extend(
        [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{rllrrrr}",
            "Seed & Method & inference & MAG topics & SOS topics & containment SOS & smaller-FP SOS \\\\",
            "\\hline",
        ]
    )
    for row in sos_rows:
        method, arm = labels[(row["method"], row["inference_arm"])]
        lines.append(
            f"{row['seed']} & {method} & {arm} & "
            f"{row['mag_annotation_available_topics']} & "
            f"{row['eligible_topics']} & "
            f"{optional(row['mean_sos_notebook_annotation_containment'])} & "
            f"{optional(row['mean_sos_supplement_smaller_fingerprint'])} \\\\"
        )
    lines.extend(
        [
            "\\end{tabular}",
            f"\\caption{{{sos_caption}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(run_dir: str | Path) -> dict[str, Any]:
    """Require every frozen seed and write final machine-readable outputs."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    verify_frozen_input_files(
        directory,
        names={"published_motifset"},
        lock=lock,
    )
    config = load_config(directory / "config.resolved.json")
    _require_completed_runs(directory, config.seeds)
    report_dir = directory / "report"
    complete_path = report_dir / "complete.json"
    if complete_path.exists():
        result = read_json(complete_path)
        if result.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("report belongs to another frozen protocol")
        for name, digest in result["output_sha256"].items():
            if file_sha256(report_dir / name) != digest:
                raise ValueError(f"report artifact changed: {name}")
        return result
    report_dir.mkdir(parents=True, exist_ok=True)
    core_rows = _core_rows(directory, config)
    topic_similarity = _topic_similarity_rows(directory, config)
    mag_rows, mag_agreement = _mag_summary(directory, config, topic_similarity)
    published = _published_similarity(directory, config, Path(lock["data_root"]))
    aggregate = _aggregate(
        core_rows,
        fields=(
            "nll_per_token",
            "active_topics_corpus",
            "top_word_diversity",
            "word_cooccurrence_npmi_mean",
            "convergence_cosine_mean",
            "convergence_cosine_median",
            "convergence_cosine_p05",
            "convergence_js_mean",
            "nll_gap_fraction_to_long",
            "training_seconds",
            "peak_rss_bytes",
            "cached_seconds_per_spectrum_median",
            "end_to_end_seconds_per_spectrum_median",
        ),
    )
    raw_dreams = read_json(directory / "mag" / "raw_dreams" / "complete.json")
    if (
        file_sha256(directory / "mag" / "raw_dreams" / "nearest_neighbors.jsonl")
        != raw_dreams["rows_sha256"]
    ):
        raise ValueError("raw-DreaMS result rows changed before reporting")
    summary = {
        "schema_version": "msnlib-validation/report-v2",
        "protocol_sha256": lock["protocol_sha256"],
        "protocol_derivation": lock.get("derivation"),
        "evidence_scope": config.evidence_scope,
        "evaluation_timing": config.evaluation_timing,
        "required_seeds": list(config.seeds),
        "all_required_seeds_reported": True,
        "cross_seed_topic_stability_available": len(config.seeds) > 1,
        "core_per_seed": core_rows,
        "aggregate_core": aggregate,
        "topic_similarity": topic_similarity,
        "mag_per_seed": mag_rows,
        "mag_same_seed_agreement": mag_agreement,
        "raw_dreams": raw_dreams,
        "published_msnlib_context": published,
        "published_paper_context": {
            "source_doi": "10.1038/s41467-026-75038-0",
            "generated_topics": 1000,
            "removed_by_mag_optimization": 432,
            "without_probability_gt_0_5_associated_molecules": 145,
            "stated_remaining_topics": 423,
            "maccs_high_sos_gt_0_8": 158,
            "maccs_intermediate_sos_0_6_to_0_8": 176,
            "maccs_low_sos_lt_0_6": 78,
            "reported_bin_sum": 412,
            "unreconciled_difference_from_stated_remaining": 11,
            "comparison_scope": "published_full_corpus_context_only_not_a_leakage_safe_baseline",
        },
        "manual_motif_spectrum_endpoint": {
            "status": "unavailable",
            "reason": "No independent manual motif-spectrum annotation file is present in the frozen Zenodo inputs.",  # noqa: E501
            "substitute_ground_truth_used": False,
        },
        "claim_boundary": {
            "confirmatory_evidence_scope": config.evidence_scope == "confirmatory",
            "tomotopy_training_bitwise_reproducible": bool(
                config.tomotopy_training_workers == 1
                and config.tomotopy_training_parallel == 1
            ),
            "hybrid_training_bitwise_reproducible": bool(
                config.hybrid_training_cpu_threads == 1
            ),
            "hybrid_training_cpu_threads": config.hybrid_training_cpu_threads,
            "hybrid_inference_cpu_threads": config.hybrid_inference_cpu_threads,
            "prior_test_results_inspected": bool(lock.get("test_results_inspected")),
            "chemical_evaluation_posthoc_correction": bool(
                (lock.get("derivation") or {}).get("kind")
                == "chemical_evaluation_correction"
                or config.evaluation_timing == "posthoc_implementation_correction"
            ),
            "implementation_posthoc_correction": bool(
                (lock.get("derivation") or {}).get("kind")
                == "implementation_correction"
                or config.evaluation_timing == "posthoc_implementation_correction"
            ),
            "dominant_topic_sos_uses_no_absolute_probability_cutoff": True,
            "dominant_topic_sos_is_invariant_to_all_calibration_changes": False,
            "primary_sos_is_compound_balanced": True,
            "probability_ge_frozen_threshold_is_sensitivity_only": True,
            "chemical_association_uses_full_test_spectra": True,
            "document_completion_uses_observed_peak_groups_only": True,
            "software_validation_is_chemical_evidence": False,
            "msnlib_peaks_are_ground_truth_fragment_assignments": False,
            "background_component_is_proven_experimental_background": False,
            "same_seed_topic_matching_establishes_correctness": False,
        },
    }
    core_csv = report_dir / "per_seed_metrics.csv"
    similarity_path = report_dir / "topic_similarity.jsonl"
    mag_path = report_dir / "mag_per_seed.jsonl"
    agreement_path = report_dir / "mag_agreement.jsonl"
    summary_path = report_dir / "summary.json"
    manuscript_path = report_dir / "manuscript_results.tex"
    _write_csv(core_csv, core_rows)
    _write_jsonl(similarity_path, topic_similarity)
    _write_jsonl(mag_path, mag_rows)
    _write_jsonl(agreement_path, mag_agreement)
    write_json(summary_path, summary)
    manuscript_path.write_text(_manuscript_text(summary), encoding="utf-8")
    outputs = (
        core_csv,
        similarity_path,
        mag_path,
        agreement_path,
        summary_path,
        manuscript_path,
    )
    result = {
        "schema_version": "msnlib-validation/report-complete-v2",
        "protocol_sha256": lock["protocol_sha256"],
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(complete_path, result)
    return result
