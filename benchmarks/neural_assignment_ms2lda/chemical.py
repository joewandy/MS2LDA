"""Leakage-controlled MAG annotation and compound-balanced chemical scoring."""

from __future__ import annotations

import argparse
import json
import time
from functools import cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from benchmarks.msnlib_validation.chemical import (
    ASSOCIATION_MODES,
    associated_record_indices,
)
from benchmarks.msnlib_validation.config import resolve_input_paths
from benchmarks.msnlib_validation.mag import (
    build_filtered_mag_index,
    consensus_fingerprint,
    library_matches,
    maccs_fingerprint,
    optimized_feature_count,
    topic_spectra,
)
from benchmarks.msnlib_validation.metrics import (
    calculate_sos,
    calculate_sos_smaller_fingerprint,
)

from .data import load_heldout_records
from .utils import (
    file_sha256,
    peak_rss_bytes,
    read_json,
    verify_output_hashes,
    write_json,
    write_jsonl,
)


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
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        fingerprints.setdefault(
            record["connectivity_key"], fingerprint_fn(record["smiles"])
        )
        metadata.setdefault(record["connectivity_key"], record)
    rows: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        topic_id = int(annotation["topic_id"])
        annotation_fp = (
            consensus_fn(annotation["clustered_smiles"], fingerprint_threshold)
            if int(annotation["optimized_feature_count"]) > 0
            else None
        )
        unique: dict[str, np.ndarray] = {}
        spectrum_scores: list[float] = []
        for row_index in associated.get(topic_id, []):
            record = records[row_index]
            fingerprint = fingerprints[record["connectivity_key"]]
            if fingerprint is None:
                continue
            unique.setdefault(record["connectivity_key"], fingerprint)
            if annotation_fp is not None:
                spectrum_scores.append(calculate_sos(annotation_fp, fingerprint))
        scores = (
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
                record = metadata[connectivity]
                compound_rows.append(
                    {
                        "topic_id": topic_id,
                        "connectivity_key": connectivity,
                        "scaffold_key": record["scaffold_key"],
                        "sos": calculate_sos(annotation_fp, fingerprint),
                        "sos_supplement": calculate_sos_smaller_fingerprint(
                            annotation_fp, fingerprint
                        ),
                    }
                )
        rows.append(
            {
                "topic_id": topic_id,
                "eligible": bool(scores),
                "associated_spectra": len(associated.get(topic_id, [])),
                "associated_molecules": len(unique),
                "sos": float(np.mean(scores)) if scores else None,
                "sos_supplement": float(np.mean(supplements)) if supplements else None,
                "sos_spectrum_weighted": (
                    float(np.mean(spectrum_scores)) if spectrum_scores else None
                ),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    values = [float(row["sos"]) for row in eligible]
    return (
        rows,
        compound_rows,
        {
            "association_mode": mode,
            "membership_threshold": threshold if mode != "dominant_topic" else None,
            "eligible_topics": len(eligible),
            "total_topics": len(annotations),
            "sos_evaluable_coverage": len(eligible) / len(annotations),
            "associated_spectra": sum(row["associated_spectra"] for row in rows),
            "topic_compound_associations": sum(
                row["associated_molecules"] for row in rows
            ),
            "mean_sos": float(np.mean(values)) if values else None,
            "median_sos": float(np.median(values)) if values else None,
        },
    )


def run_chemical_scoring(
    run_dir: str | Path,
    *,
    method: str,
    data_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Annotate neural or comparator topics and score held-out compounds."""
    if method not in {"neural", "tomotopy"}:
        raise ValueError("chemical method must be neural or tomotopy")
    directory = Path(run_dir).expanduser().resolve()
    output = directory / "chemical" / method
    if (output / "complete.json").is_file():
        result = read_json(output / "complete.json")
        verify_output_hashes(output, result)
        return result
    import faiss

    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_model
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
        hit_clustering,
        motif_optimization,
    )

    inputs = resolve_input_paths(protocol, data_root)
    for name in ("spec2vec_model", "spec2vec_db", "spec2vec_embeddings"):
        if file_sha256(inputs[name]) != protocol["input_files"][name]["sha256"]:
            raise ValueError(f"frozen chemical input changed: {name}")
    index_manifest = build_filtered_mag_index(
        directory, data_root=data_root, protocol=protocol
    )
    data = directory / "data"
    evaluation = directory / "evaluation" / method
    verify_output_hashes(evaluation, read_json(evaluation / "complete.json"))
    beta_path = evaluation / "beta.npy"
    theta_path = evaluation / "test_full_theta.npy"
    beta = np.load(beta_path, mmap_mode="r")
    theta = np.load(theta_path, mmap_mode="r")
    vocabulary = list(map(str, read_json(data / "vocabulary.json")["vocabulary"]))
    records = load_heldout_records(data, "test")
    if theta.shape[0] != len(records):
        raise ValueError("full mixtures and held-out records differ")
    spectra = topic_spectra(
        beta,
        vocabulary,
        int(protocol["chemistry"]["motif_spectrum_top_n"]),
        significant_digits=int(protocol["preprocessing"]["significant_digits"]),
    )
    spec2vec = load_s2v_model(str(inputs["spec2vec_model"]))
    query_embeddings = calc_embeddings(spec2vec, spectra).astype(np.float32)
    index_root = directory / "mag/index"
    index = faiss.read_index(str(index_root / "spec2vec_filtered.faiss"))
    kept_ids = np.load(index_root / "kept_original_ids.npy", mmap_mode="r")
    excluded = set(
        read_json(index_root / "excluded_connectivity_keys.json")["connectivity_keys"]
    )
    normalized = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    started = time.perf_counter()
    similarities, indices = index.search(
        normalized,
        min(int(protocol["chemistry"]["mag_search_k"]), index.ntotal),
    )
    matches = library_matches(
        similarities=similarities,
        filtered_indices=indices,
        kept_original_ids=kept_ids,
        db_path=inputs["spec2vec_db"],
        unique_molecules=int(protocol["chemistry"]["mag_unique_molecules"]),
        excluded_connectivity=excluded,
    )
    annotations: list[dict[str, Any]] = []
    for topic_id, (motif, match) in enumerate(zip(spectra, matches, strict=True)):
        cluster_error = ""
        optimization_error = ""
        clustered_smiles: list[str] = []
        clustered_spectra: list[Any] = []
        try:
            clustered_spectra_rows, clustered_smiles_rows, _ = hit_clustering(
                s2v_similarity=spec2vec,
                motif_spectra=[motif],
                library_matches=[match],
                criterium="best",
                cosine_similarity=float(protocol["chemistry"]["mag_cluster_cosine"]),
            )
            clustered_spectra = (
                clustered_spectra_rows[0] if clustered_spectra_rows else []
            )
            clustered_smiles = clustered_smiles_rows[0] if clustered_smiles_rows else []
        except Exception as exc:  # noqa: BLE001
            cluster_error = f"{type(exc).__name__}: {exc}"
        optimized_count = 0
        if clustered_spectra and clustered_smiles:
            try:
                optimized = motif_optimization(
                    [motif], [clustered_spectra], [clustered_smiles], loss_err=1
                )
                optimized_count = optimized_feature_count(
                    optimized[0] if optimized else None
                )
            except Exception as exc:  # noqa: BLE001
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
            }
        )
    cached_fingerprint = cache(maccs_fingerprint)

    @cache
    def cached_consensus(
        values: tuple[str, ...], threshold: float
    ) -> np.ndarray | None:
        return consensus_fingerprint(values, threshold)

    summaries: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    for mode in ASSOCIATION_MODES:
        rows, compounds, summary = _topic_scores(
            theta=theta,
            records=records,
            annotations=annotations,
            mode=mode,
            threshold=float(protocol["chemistry"]["membership_threshold"]),
            fingerprint_threshold=float(
                protocol["chemistry"]["mag_fingerprint_threshold"]
            ),
            fingerprint_fn=cached_fingerprint,
            consensus_fn=lambda values, threshold: cached_consensus(
                tuple(values), threshold
            ),
        )
        summaries.append(summary)
        topic_rows.extend({"association_mode": mode, **row} for row in rows)
        compound_rows.extend({"association_mode": mode, **row} for row in compounds)
    output.mkdir(parents=True, exist_ok=True)
    rows_by_name = {
        "annotations.jsonl": annotations,
        "topic_scores.jsonl": topic_rows,
        "compound_scores.jsonl": compound_rows,
    }
    for name, rows in rows_by_name.items():
        write_jsonl(output / name, rows)
    write_json(output / "summaries.json", summaries)
    by_mode = {row["association_mode"]: row for row in summaries}
    result = {
        "schema_version": "neural-ms2lda/chemical-evaluation-v1",
        "method": method,
        "topics": len(annotations),
        "annotation_coverage": sum(
            row["optimized_feature_count"] > 0 for row in annotations
        )
        / len(annotations),
        "cluster_failures": sum(bool(row["cluster_error"]) for row in annotations),
        "optimization_failures": sum(
            bool(row["optimization_error"]) for row in annotations
        ),
        "association_results": summaries,
        "dominant_topic_chemistry": by_mode["dominant_topic"],
        "high_confidence_chemistry": by_mode["probability_ge_frozen_threshold"],
        "heldout_compounds_excluded_from_mag": index_manifest["retained_leak_rows"]
        == 0,
        "mag_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {
            name: file_sha256(output / name)
            for name in (*rows_by_name, "summaries.json")
        },
    }
    write_json(output / "complete.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated MAG worker used by the top-level orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--method", choices=("neural", "tomotopy"), required=True)
    args = parser.parse_args(argv)
    result = run_chemical_scoring(
        args.run,
        method=args.method,
        data_root=args.data_root,
        protocol=read_json(args.run / "protocol.resolved.json"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
