# ruff: noqa: C901, PLR0912, PLR0913, PLR0915
"""Leakage-controlled MAG annotation and chemical scoring for neural topics."""

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
from benchmarks.msnlib_validation.config import load_config, resolve_input_paths
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

from .utils import file_sha256, peak_rss_bytes, read_json, write_json

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_complete(output: Path) -> dict[str, Any]:
    result = read_json(output / "complete.json")
    for name, digest in result["output_sha256"].items():
        if file_sha256(output / name) != digest:
            msg = f"neural chemical artifact changed: {name}"
            raise ValueError(msg)
    return result


def _load_vocabulary(counts: Path) -> list[str]:
    payload = read_json(counts / "vocabulary.json")
    return list(map(str, payload["vocabulary"]))


def _load_test_records(counts: Path) -> list[dict[str, Any]]:
    records = []
    with (counts / "heldout_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == "test":
                    records.append(row)
    return records


def _topic_scores(
    *,
    theta: np.ndarray,
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    mode: str,
    threshold: float,
    fingerprint_threshold: float,
    fingerprint_fn: Callable[[str], np.ndarray | None],
    consensus_fn: Callable[[Sequence[str], float], np.ndarray | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the corrected compound-balanced SOS definition."""
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
    compound_rows = []
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
            [calculate_sos(annotation_fp, value) for value in unique.values()]
            if annotation_fp is not None
            else []
        )
        supplements = (
            [
                calculate_sos_smaller_fingerprint(annotation_fp, value)
                for value in unique.values()
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
                "sos_supplement": (
                    float(np.mean(supplements)) if supplements else None
                ),
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


def run_chemical_scoring(
    run_dir: str | Path,
    *,
    attempt: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Annotate neural topics and score full held-out spectra with corrected SOS."""
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "chemical" / attempt
    if (output / "complete.json").is_file():
        return _verify_complete(output)

    import faiss

    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_model
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
        hit_clustering,
        motif_optimization,
    )

    lock = read_json(directory / "neural.lock.json")
    reference = Path(lock["reference_run"])
    source_config = load_config(reference / "config.resolved.json")
    source_lock = read_json(reference / "protocol.lock.json")
    inputs = resolve_input_paths(source_config, source_lock["data_root"])
    for name in ("spec2vec_model", "spec2vec_db"):
        expected = source_config.input_files[name]["sha256"]
        if file_sha256(inputs[name]) != expected:
            msg = f"frozen chemical input changed: {name}"
            raise ValueError(msg)
    counts = Path(lock["source_run"]) / "shared/counts"
    evaluation = directory / "evaluation" / attempt
    evaluation_complete = read_json(evaluation / "complete.json")
    beta_path = evaluation / "beta.npy"
    theta_path = evaluation / "test_full_theta.npy"
    for path in (beta_path, theta_path):
        expected = evaluation_complete["output_sha256"][path.name]
        if file_sha256(path) != expected:
            msg = f"neural evaluation changed before chemistry: {path.name}"
            raise ValueError(msg)
    beta = np.load(beta_path, mmap_mode="r")
    theta = np.load(theta_path, mmap_mode="r")
    vocabulary = _load_vocabulary(counts)
    records = _load_test_records(counts)
    if theta.shape[0] != len(records):
        msg = "full neural mixtures and held-out records differ"
        raise ValueError(msg)

    motif_spectra = _topic_spectra(
        beta,
        vocabulary,
        source_config.motif_spectrum_top_n,
        significant_digits=source_config.significant_digits,
    )
    spec2vec = load_s2v_model(str(inputs["spec2vec_model"]))
    query_embeddings = calc_embeddings(spec2vec, motif_spectra).astype(np.float32)
    index_root = reference / "mag" / "index"
    index_manifest = read_json(index_root / "manifest.json")
    reference_mag = read_json(reference / "mag/seed_42/tomotopy/complete.json")
    expected_index = reference_mag["index_exclusion_audit"]["output_sha256"]
    if index_manifest["output_sha256"] != expected_index:
        msg = "MAG index manifest differs from the frozen reference result"
        raise ValueError(msg)
    for name, digest in index_manifest["output_sha256"].items():
        if file_sha256(index_root / name) != digest:
            msg = f"reference leakage-filtered MAG index changed: {name}"
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
    annotations = []
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
        except Exception as exc:  # noqa: BLE001 - retain every per-topic failure
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
            except Exception as exc:  # noqa: BLE001 - retain every per-topic failure
                optimization_error = f"{type(exc).__name__}: {exc}"
        annotations.append(
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

    summary_rows = []
    topic_rows = []
    compound_rows = []
    for mode in ASSOCIATION_MODES:
        rows, compounds, summary = _topic_scores(
            theta=theta,
            records=records,
            annotations=annotations,
            mode=mode,
            threshold=float(protocol["evaluation"]["membership_threshold"]),
            fingerprint_threshold=float(
                protocol["evaluation"]["mag_fingerprint_threshold"],
            ),
            fingerprint_fn=cached_fingerprint,
            consensus_fn=consensus_fingerprint,
        )
        summary_rows.append(summary)
        topic_rows.extend({"association_mode": mode, **row} for row in rows)
        compound_rows.extend({"association_mode": mode, **row} for row in compounds)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotations.jsonl": annotations,
        "topic_scores.jsonl": topic_rows,
        "compound_scores.jsonl": compound_rows,
    }
    for name, rows in paths.items():
        _write_jsonl(output / name, rows)
    summaries_path = output / "summaries.json"
    write_json(summaries_path, summary_rows)
    summary_by_mode = {row["association_mode"]: row for row in summary_rows}
    result = {
        "schema_version": "fully-neural-ms2lda/chemical-complete-v1",
        "attempt": attempt,
        "topics": len(annotations),
        "annotation_coverage": sum(
            row["optimized_feature_count"] > 0 for row in annotations
        )
        / len(annotations),
        "cluster_failures": sum(bool(row["cluster_error"]) for row in annotations),
        "optimization_failures": sum(
            bool(row["optimization_error"]) for row in annotations
        ),
        "association_results": summary_rows,
        "dominant_topic_chemistry": summary_by_mode["dominant_topic"],
        "high_confidence_chemistry": summary_by_mode["probability_ge_frozen_threshold"],
        "full_spectrum_inference": True,
        "chemical_labels_used_for_training": False,
        "leakage_filtered_index": True,
        "mag_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "fingerprint_cache": cached_fingerprint.cache_info()._asdict(),
        "consensus_cache": cached_consensus.cache_info()._asdict(),
        "input_bindings": {
            "evaluation_complete_sha256": file_sha256(evaluation / "complete.json"),
            "beta_sha256": file_sha256(beta_path),
            "test_full_theta_sha256": file_sha256(theta_path),
            "reference_index_manifest_sha256": file_sha256(
                index_root / "manifest.json",
            ),
        },
        "output_sha256": {
            name: file_sha256(output / name) for name in (*paths, summaries_path.name)
        },
    }
    write_json(output / "complete.json", result)
    return result
