# ruff: noqa: C901, PLR0913, PLR0915
"""Shared MAG annotations and chemical SOS scoring for every frozen arm."""

from __future__ import annotations

import json
import os
import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.msnlib_validation.chemical import (
    ASSOCIATION_MODES,
    associated_record_indices,
)
from benchmarks.msnlib_validation.config import (
    file_sha256,
    load_config,
    read_json,
    resolve_input_paths,
    write_json,
)
from benchmarks.msnlib_validation.mag import (
    _consensus_fingerprint,
    _library_matches,
    _maccs_fingerprint,
    _normalize,
    _optimized_feature_count,
    _topic_spectra,
)
from benchmarks.msnlib_validation.metrics import (
    calculate_sos,
    calculate_sos_smaller_fingerprint,
)

from .data import heldout_metadata, load_vocabulary_copy
from .spec import ARM_IDS, BUDGETS, load_spec, verify_study

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _annotation_output(directory: Path, discovery: str) -> Path:
    return directory / "chemical" / "annotations" / discovery


def _verify_annotations(output: Path, discovery: str) -> dict[str, Any]:
    result = read_json(output / "complete.json")
    if result.get("discovery") != discovery:
        msg = "annotation discovery identity changed"
        raise ValueError(msg)
    if file_sha256(output / "topics.jsonl") != result["topics_sha256"]:
        msg = "annotation topic rows changed"
        raise ValueError(msg)
    return result


def import_current_annotations(run_dir: str | Path) -> dict[str, Any]:
    """Reuse the exact MAG annotation generated for the current beta."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    discovery = "dreams_prior"
    output = _annotation_output(directory, discovery)
    if (output / "complete.json").is_file():
        return _verify_annotations(output, discovery)
    source = Path(lock["source_run"])
    source_complete = read_json(source / "mag/seed_42/hybrid/complete.json")
    source_rows = _jsonl(source / "mag/seed_42/hybrid/topics.jsonl")
    selected = [
        row
        for row in source_rows
        if row["inference_arm"] == "encoder"
        and row["association_mode"] == "dominant_topic"
    ]
    selected.sort(key=lambda row: int(row["topic_id"]))
    spec = load_spec(directory)
    if [row["topic_id"] for row in selected] != list(range(spec.num_topics)):
        msg = "current MAG source does not contain one row per topic"
        raise ValueError(msg)
    annotations = [
        {
            "topic_id": int(row["topic_id"]),
            "clustered_smiles": row["clustered_smiles"],
            "optimized_feature_count": int(row["optimized_feature_count"]),
            "cluster_error": row["cluster_error"],
            "optimization_error": row["optimization_error"],
            "retrieved_smiles": row["retrieved_smiles"],
            "retrieved_scores": row["retrieved_scores"],
        }
        for row in selected
    ]
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "topics.jsonl"
    _write_jsonl(rows_path, annotations)
    result = {
        "schema_version": "msnlib-simplification/mag-annotations-v1",
        "discovery": discovery,
        "topics": spec.num_topics,
        "imported_from_corrected_source": True,
        "source_complete_sha256": file_sha256(
            source / "mag/seed_42/hybrid/complete.json",
        ),
        "source_topics_sha256": source_complete["topics_sha256"],
        "annotation_coverage": sum(
            row["optimized_feature_count"] > 0 for row in annotations
        )
        / spec.num_topics,
        "topics_sha256": file_sha256(rows_path),
    }
    write_json(output / "complete.json", result)
    return result


def annotate_symmetric_discovery(run_dir: str | Path) -> dict[str, Any]:
    """Run MAG once for the new symmetric-prior topic matrix."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    discovery = "symmetric_prior"
    output = _annotation_output(directory, discovery)
    if (output / "complete.json").is_file():
        return _verify_annotations(output, discovery)

    import faiss

    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_model
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
        hit_clustering,
        motif_optimization,
    )

    discovery_root = directory / "discoveries" / discovery
    discovery_complete = read_json(discovery_root / "complete.json")
    snapshot = discovery_root / "snapshot.npz"
    if file_sha256(snapshot) != discovery_complete["output_sha256"]["snapshot.npz"]:
        msg = "symmetric discovery changed before MAG"
        raise ValueError(msg)
    beta = np.load(snapshot)["beta"]
    source = Path(lock["source_run"])
    source_config = load_config(source / "config.resolved.json")
    source_lock = read_json(source / "protocol.lock.json")
    inputs = resolve_input_paths(source_config, source_lock["data_root"])
    vocabulary = load_vocabulary_copy(directory)
    motif_spectra = _topic_spectra(
        beta,
        vocabulary,
        source_config.motif_spectrum_top_n,
        significant_digits=source_config.significant_digits,
    )
    spec2vec = load_s2v_model(str(inputs["spec2vec_model"]))
    query_embeddings = calc_embeddings(spec2vec, motif_spectra).astype(np.float32)
    index_root = source / "mag" / "index"
    index_manifest = read_json(index_root / "manifest.json")
    for name, digest in index_manifest["output_sha256"].items():
        if file_sha256(index_root / name) != digest:
            msg = f"source leakage-filtered MAG index changed: {name}"
            raise ValueError(msg)
    index = faiss.read_index(str(index_root / "spec2vec_filtered.faiss"))
    kept_ids = np.load(index_root / "kept_original_ids.npy", mmap_mode="r")
    excluded = set(
        map(
            str,
            read_json(index_root / "excluded_connectivity_keys.json")[
                "connectivity_keys"
            ],
        ),
    )
    started = time.perf_counter()
    similarities, indices = index.search(
        _normalize(query_embeddings),
        min(source_config.mag_search_k, index.ntotal),
    )
    matches = _library_matches(
        similarities=similarities,
        filtered_indices=indices,
        kept_original_ids=kept_ids,
        db_path=inputs["spec2vec_db"],
        unique_molecules=source_config.mag_unique_molecules,
        excluded_connectivity=excluded,
    )
    rows = []
    for topic_id, (motif, match) in enumerate(zip(motif_spectra, matches, strict=True)):
        cluster_error = ""
        optimization_error = ""
        clustered_smiles: list[str] = []
        clustered_spectra: list[Any] = []
        try:
            spectra, smiles, _ = hit_clustering(
                s2v_similarity=spec2vec,
                motif_spectra=[motif],
                library_matches=[match],
                criterium="best",
                cosine_similarity=source_config.mag_cluster_cosine,
            )
            clustered_spectra = spectra[0] if spectra else []
            clustered_smiles = smiles[0] if smiles else []
        except Exception as exc:  # noqa: BLE001 - preserve per-topic failures
            cluster_error = f"{type(exc).__name__}: {exc}"
        optimized_count = 0
        if clustered_spectra and clustered_smiles:
            try:
                optimized = motif_optimization(
                    [motif],
                    [clustered_spectra],
                    [clustered_smiles],
                    loss_err=1,
                )
                optimized_count = _optimized_feature_count(
                    optimized[0] if optimized else None,
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-topic failures
                optimization_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "topic_id": topic_id,
                "clustered_smiles": clustered_smiles,
                "optimized_feature_count": optimized_count,
                "cluster_error": cluster_error,
                "optimization_error": optimization_error,
                "retrieved_smiles": match[0],
                "retrieved_scores": match[2],
            },
        )
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "topics.jsonl"
    _write_jsonl(rows_path, rows)
    result = {
        "schema_version": "msnlib-simplification/mag-annotations-v1",
        "discovery": discovery,
        "topics": len(rows),
        "imported_from_corrected_source": False,
        "mag_seconds": time.perf_counter() - started,
        "annotation_coverage": sum(row["optimized_feature_count"] > 0 for row in rows)
        / len(rows),
        "cluster_failures": sum(bool(row["cluster_error"]) for row in rows),
        "optimization_failures": sum(bool(row["optimization_error"]) for row in rows),
        "leakage_filtered_index_manifest_sha256": file_sha256(
            index_root / "manifest.json",
        ),
        "topics_sha256": file_sha256(rows_path),
    }
    write_json(output / "complete.json", result)
    return result


def annotate_discovery(run_dir: str | Path, discovery: str) -> dict[str, Any]:
    """Create or verify discovery-level annotations."""
    if discovery == "dreams_prior":
        return import_current_annotations(run_dir)
    if discovery == "symmetric_prior":
        return annotate_symmetric_discovery(run_dir)
    msg = f"unknown discovery: {discovery}"
    raise ValueError(msg)


def _topic_scores(
    *,
    theta: np.ndarray,
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    mode: str,
    threshold: float,
    fingerprint_threshold: float,
    fingerprint_fn: Callable[[str], np.ndarray | None] = _maccs_fingerprint,
    consensus_fn: Callable[
        [Sequence[str], float],
        np.ndarray | None,
    ] = _consensus_fingerprint,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    associated = associated_record_indices(theta, mode=mode, threshold=threshold)
    fingerprints: dict[str, np.ndarray | None] = {}
    compound_metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        fingerprints.setdefault(
            record["connectivity_key"],
            fingerprint_fn(record["smiles"]),
        )
        compound_metadata.setdefault(record["connectivity_key"], record)
    rows = []
    compound_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        topic_id = int(annotation["topic_id"])
        annotation_fp = (
            consensus_fn(
                annotation["clustered_smiles"],
                fingerprint_threshold,
            )
            if int(annotation["optimized_feature_count"]) > 0
            else None
        )
        unique: dict[str, np.ndarray] = {}
        spectrum_scores = []
        for row_index in associated.get(topic_id, []):
            record = records[row_index]
            fingerprint = fingerprints[record["connectivity_key"]]
            if fingerprint is None:
                continue
            unique.setdefault(record["connectivity_key"], fingerprint)
            if annotation_fp is not None:
                spectrum_scores.append(calculate_sos(annotation_fp, fingerprint))
        compound_scores = (
            [calculate_sos(annotation_fp, fp) for fp in unique.values()]
            if annotation_fp is not None
            else []
        )
        supplement = (
            [
                calculate_sos_smaller_fingerprint(annotation_fp, fp)
                for fp in unique.values()
            ]
            if annotation_fp is not None
            else []
        )
        if annotation_fp is not None:
            for connectivity, fingerprint in unique.items():
                record = compound_metadata[connectivity]
                compound_rows.append(
                    {
                        "topic_id": topic_id,
                        "connectivity_key": connectivity,
                        "scaffold_key": record["scaffold_key"],
                        "sos": calculate_sos(annotation_fp, fingerprint),
                        "sos_supplement": calculate_sos_smaller_fingerprint(
                            annotation_fp,
                            fingerprint,
                        ),
                    },
                )
        rows.append(
            {
                "topic_id": topic_id,
                "eligible": bool(compound_scores),
                "associated_spectra": len(associated.get(topic_id, [])),
                "associated_molecules": len(unique),
                "sos": float(np.mean(compound_scores)) if compound_scores else None,
                "sos_supplement": float(np.mean(supplement)) if supplement else None,
                "sos_spectrum_weighted": (
                    float(np.mean(spectrum_scores)) if spectrum_scores else None
                ),
            },
        )
    eligible = [row for row in rows if row["eligible"]]
    values = [float(row["sos"]) for row in eligible]
    summary = {
        "association_mode": mode,
        "membership_threshold": threshold if mode != "dominant_topic" else None,
        "eligible_topics": len(eligible),
        "sos_evaluable_coverage": len(eligible) / len(annotations),
        "associated_spectra": sum(row["associated_spectra"] for row in rows),
        "topic_compound_associations": sum(row["associated_molecules"] for row in rows),
        "mean_sos": float(np.mean(values)) if values else None,
        "median_sos": float(np.median(values)) if values else None,
    }
    return rows, compound_rows, summary


def score_all_chemical_results(run_dir: str | Path) -> dict[str, Any]:
    """Score full-spectrum validation and test mixtures without retraining."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    spec = load_spec(directory)
    output = directory / "chemical" / "scores"
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        for name, digest in result["output_sha256"].items():
            if file_sha256(output / name) != digest:
                msg = f"chemical score artifact changed: {name}"
                raise ValueError(msg)
        return result
    annotations_by_discovery = {}
    lock = verify_study(directory)
    source_config = load_config(Path(lock["source_run"]) / "config.resolved.json")
    cached_fingerprint = cache(_maccs_fingerprint)

    @cache
    def cached_consensus(
        smiles_values: tuple[str, ...],
        threshold: float,
    ) -> np.ndarray | None:
        return _consensus_fingerprint(smiles_values, threshold)

    def consensus_fingerprint(
        smiles_values: Sequence[str],
        threshold: float,
    ) -> np.ndarray | None:
        return cached_consensus(tuple(smiles_values), threshold)

    for discovery in ("dreams_prior", "symmetric_prior"):
        annotate_discovery(directory, discovery)
        annotations_by_discovery[discovery] = _jsonl(
            _annotation_output(directory, discovery) / "topics.jsonl",
        )
    summary_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    summaries_path = output / "summaries.jsonl"
    topics_path = output / "topics.jsonl"
    compounds_path = output / "compound_scores.jsonl"
    compounds_temporary = compounds_path.with_name(
        f".{compounds_path.name}.{os.getpid()}.tmp",
    )
    compound_count = 0
    try:
        with compounds_temporary.open("w", encoding="utf-8") as compound_handle:
            for split in ("validation", "test"):
                records = heldout_metadata(directory, split)
                for arm_id in ARM_IDS:
                    discovery = arm_id.split("__", 1)[0]
                    arm_root = (
                        directory / "evaluation" / split / "full" / "arms" / arm_id
                    )
                    inference_complete = read_json(
                        arm_root / "inference_complete.json",
                    )
                    for budget in BUDGETS:
                        theta_path = arm_root / f"theta_{budget}.npy"
                        if (
                            file_sha256(theta_path)
                            != inference_complete["theta_sha256"][theta_path.name]
                        ):
                            msg = "full-spectrum theta changed before SOS"
                            raise ValueError(msg)
                        theta = np.load(theta_path, mmap_mode="r")
                        for mode in ASSOCIATION_MODES:
                            rows, compounds, summary = _topic_scores(
                                theta=theta,
                                records=records,
                                annotations=annotations_by_discovery[discovery],
                                mode=mode,
                                threshold=spec.membership_threshold,
                                fingerprint_threshold=(
                                    source_config.mag_fingerprint_threshold
                                ),
                                fingerprint_fn=cached_fingerprint,
                                consensus_fn=consensus_fingerprint,
                            )
                            identity = {
                                "split": split,
                                "arm_id": arm_id,
                                "discovery": discovery,
                                "inference": arm_id.split("__", 1)[1],
                                "budget": budget,
                            }
                            summary_rows.append({**identity, **summary})
                            topic_rows.extend(
                                {
                                    **identity,
                                    "association_mode": mode,
                                    "membership_threshold": (
                                        spec.membership_threshold
                                        if mode != "dominant_topic"
                                        else None
                                    ),
                                    **row,
                                }
                                for row in rows
                            )
                            for row in compounds:
                                payload = {
                                    **identity,
                                    "association_mode": mode,
                                    "membership_threshold": (
                                        spec.membership_threshold
                                        if mode != "dominant_topic"
                                        else None
                                    ),
                                    **row,
                                }
                                compound_handle.write(
                                    json.dumps(
                                        payload,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                    + "\n",
                                )
                                compound_count += 1
            compound_handle.flush()
            os.fsync(compound_handle.fileno())
        compounds_temporary.replace(compounds_path)
    finally:
        compounds_temporary.unlink(missing_ok=True)
    _write_jsonl(summaries_path, summary_rows)
    _write_jsonl(topics_path, topic_rows)
    result = {
        "schema_version": "msnlib-simplification/chemical-scores-v1",
        "summary_rows": len(summary_rows),
        "topic_rows": len(topic_rows),
        "compound_rows": compound_count,
        "primary_association_mode": "dominant_topic",
        "threshold_mode_role": "sensitivity_only",
        "full_spectrum_inference": True,
        "chemical_labels_used_for_training": False,
        "fingerprint_cache": cached_fingerprint.cache_info()._asdict(),
        "consensus_cache": cached_consensus.cache_info()._asdict(),
        "output_sha256": {
            path.name: file_sha256(path)
            for path in (summaries_path, topics_path, compounds_path)
        },
    }
    write_json(complete_path, result)
    return result
