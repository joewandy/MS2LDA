"""Leakage-controlled MAG annotation and compound-balanced chemical scoring."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_heldout_records
from .mag import (
    build_filtered_mag_index,
    consensus_fingerprint,
    library_matches,
    maccs_fingerprint,
    optimized_feature_count,
    topic_spectra,
)
from .spectra import input_paths
from .utils import (
    read_json,
    write_json,
    write_jsonl,
)

SOS_USEFUL_THRESHOLD = 0.6
SOS_HIGH_THRESHOLD = 0.8


def _associated_record_indices(
    theta: np.ndarray,
) -> dict[int, list[int]]:
    """Assign every spectrum to its single dominant topic.

    Topic-mixture values are comparable within a spectrum, so ``argmax`` does
    not require row normalization.  NumPy resolves an exact tie in favour of
    the lowest topic index, which makes the assignment deterministic.  The
    resulting comparison gives every model exactly one topic association per
    spectrum and therefore does not depend on cross-model probability
    calibration.
    """
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("theta must be a non-empty document-topic matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta must contain finite non-negative values")
    if np.any(values.sum(axis=1) <= 0):
        raise ValueError("every theta row must have positive probability mass")
    associated: defaultdict[int, list[int]] = defaultdict(list)
    for row, topic in enumerate(np.argmax(values, axis=1)):
        associated[int(topic)].append(int(row))
    return dict(associated)


def _calculate_sos(annotation: np.ndarray, molecule: np.ndarray) -> float:
    """Return annotation containment: fingerprint overlap over annotation bits."""
    annotation = np.asarray(annotation, dtype=bool)
    molecule = np.asarray(molecule, dtype=bool)
    if annotation.shape != molecule.shape:
        raise ValueError("fingerprints must have identical shapes")
    denominator = int(annotation.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(annotation, molecule).sum() / denominator)


def _sos_bands(values: Sequence[float]) -> dict[str, int]:
    """Count eligible topic SOS values in the paper's fixed bands."""
    return {
        "high_gt_0_8": sum(value > SOS_HIGH_THRESHOLD for value in values),
        "intermediate_0_6_to_0_8": sum(
            SOS_USEFUL_THRESHOLD <= value <= SOS_HIGH_THRESHOLD for value in values
        ),
        "low_lt_0_6": sum(value < SOS_USEFUL_THRESHOLD for value in values),
    }


def _topic_scores(
    *,
    theta: np.ndarray,
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    fingerprint_threshold: float,
) -> dict[str, Any]:
    """Apply the corrected compound-balanced SOS definition."""
    associated = _associated_record_indices(theta)
    fingerprint_fn = cache(maccs_fingerprint)

    @cache
    def consensus_fn(values: tuple[str, ...], cutoff: float) -> np.ndarray | None:
        return consensus_fingerprint(values, cutoff)

    fingerprints: dict[str, np.ndarray | None] = {}
    for record in records:
        fingerprints.setdefault(
            record["connectivity_key"], fingerprint_fn(record["smiles"])
        )
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        topic_id = int(annotation["topic_id"])
        annotation_fp = (
            consensus_fn(tuple(annotation["clustered_smiles"]), fingerprint_threshold)
            if int(annotation["optimized_feature_count"]) > 0
            else None
        )
        unique: dict[str, np.ndarray] = {}
        for row_index in associated.get(topic_id, []):
            record = records[row_index]
            fingerprint = fingerprints[record["connectivity_key"]]
            if fingerprint is None:
                continue
            unique.setdefault(record["connectivity_key"], fingerprint)
        scores = (
            [_calculate_sos(annotation_fp, value) for value in unique.values()]
            if annotation_fp is not None
            else []
        )
        rows.append(
            {
                "topic_id": topic_id,
                "eligible": bool(scores),
                "associated_spectra": len(associated.get(topic_id, [])),
                "associated_molecules": len(unique),
                "sos": float(np.mean(scores)) if scores else None,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    values = [float(row["sos"]) for row in eligible]
    return {
        "association_rule": "dominant_topic",
        "eligible_topics": len(eligible),
        "total_topics": len(annotations),
        "associated_spectra": sum(row["associated_spectra"] for row in rows),
        "associated_molecules": sum(row["associated_molecules"] for row in rows),
        "mean_sos": float(np.mean(values)) if values else None,
        "median_sos": float(np.median(values)) if values else None,
        "sos_bands": _sos_bands(values),
        "topic_scores": rows,
    }


def score_precomputed_annotations(
    *,
    theta: np.ndarray,
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    fingerprint_threshold: float,
) -> dict[str, Any]:
    """Score fixed MAG annotations using one dominant topic per spectrum.

    This inexpensive path deliberately reuses the locked MAG annotations and
    compound-balanced SOS implementation instead of rerunning beta-dependent
    annotation.
    """
    return _topic_scores(
        theta=theta,
        records=records,
        annotations=annotations,
        fingerprint_threshold=fingerprint_threshold,
    )


def _chemical_inputs(
    protocol: dict[str, Any], data_root: str | Path
) -> dict[str, Path]:
    """Resolve the three Spec2Vec/MAG inputs required by annotation."""
    inputs = input_paths(protocol, data_root)
    for name in ("spec2vec_model", "spec2vec_db", "spec2vec_embeddings"):
        if not inputs[name].is_file():
            raise FileNotFoundError(
                f"required chemical input is missing: {inputs[name]}"
            )
    return inputs


def _mag_matches(
    directory: Path,
    inputs: dict[str, Path],
    spectra: list[Any],
    protocol: dict[str, Any],
) -> tuple[Any, list[Any]]:
    """Embed motif spectra and retrieve leakage-filtered library neighbours."""
    import faiss
    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_model

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
    return spec2vec, matches


def _annotate_topics(
    spectra: list[Any],
    matches: list[Any],
    spec2vec: Any,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cluster retrieved neighbours and optimize every motif independently."""
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
        hit_clustering,
        motif_optimization,
    )

    annotations: list[dict[str, Any]] = []
    for topic_id, (motif, match) in enumerate(zip(spectra, matches, strict=True)):
        clustered_smiles: list[str] = []
        clustered_spectra: list[Any] = []
        clustering_failure: dict[str, str] | None = None
        optimization_failure: dict[str, str] | None = None
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
            clustering_failure = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"MAG clustering failed for topic {topic_id}: {exc}", file=sys.stderr)
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
                optimization_failure = {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                print(
                    f"MAG optimization failed for topic {topic_id}: {exc}",
                    file=sys.stderr,
                )
        annotations.append(
            {
                "topic_id": topic_id,
                "clustered_smiles": clustered_smiles,
                "optimized_feature_count": optimized_count,
                "clustering_failure": clustering_failure,
                "optimization_failure": optimization_failure,
            }
        )
    return annotations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one deterministic JSON-lines artifact."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _shared_annotations(
    directory: Path,
    *,
    method: str,
    data_root: str | Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate each model's fixed beta once for both held-out splits."""
    output = directory / "mag/annotations" / method
    annotations_path = output / "annotations.jsonl"
    complete_path = output / "complete.json"
    beta_path = directory / "validation_evaluation" / method / "beta.npy"
    if complete_path.is_file() and annotations_path.is_file():
        return _read_jsonl(annotations_path), read_json(complete_path)

    inputs = _chemical_inputs(protocol, data_root)
    index_summary = build_filtered_mag_index(
        directory,
        data_root=data_root,
        protocol=protocol,
    )
    data = directory / "data"
    vocabulary = list(map(str, read_json(data / "vocabulary.json")["vocabulary"]))
    beta = np.load(beta_path, mmap_mode="r")
    spectra = topic_spectra(
        beta,
        vocabulary,
        int(protocol["chemistry"]["motif_spectrum_top_n"]),
        significant_digits=int(protocol["preprocessing"]["significant_digits"]),
    )
    spec2vec, matches = _mag_matches(directory, inputs, spectra, protocol)
    annotations = _annotate_topics(spectra, matches, spec2vec, protocol)
    write_jsonl(annotations_path, annotations)
    clustering_failures = [
        int(row["topic_id"])
        for row in annotations
        if row.get("clustering_failure") is not None
    ]
    optimization_failures = [
        int(row["topic_id"])
        for row in annotations
        if row.get("optimization_failure") is not None
    ]
    result = {
        "method": method,
        "topics": len(annotations),
        "annotation_coverage": sum(
            row["optimized_feature_count"] > 0 for row in annotations
        )
        / len(annotations),
        "heldout_compounds_excluded_from_mag": index_summary["retained_leak_rows"] == 0,
        "mag_failures": {
            "clustering_count": len(clustering_failures),
            "clustering_topic_ids": clustering_failures,
            "optimization_count": len(optimization_failures),
            "optimization_topic_ids": optimization_failures,
        },
    }
    write_json(complete_path, result)
    return annotations, result


def run_chemical_scoring(
    run_dir: str | Path,
    *,
    method: str,
    data_root: str | Path,
    protocol: dict[str, Any],
    split: str = "test",
    annotation_method: str | None = None,
) -> dict[str, Any]:
    """Annotate one registered topic model and score held-out compounds."""
    allowed = {
        "etm",
        "etm_balanced",
        "contextual_sparse_etm",
        "tomotopy",
    }
    if method not in allowed:
        raise ValueError(f"chemical method must be one of {sorted(allowed)}")
    if split not in {"validation", "test"}:
        raise ValueError("chemical split must be validation or test")
    if protocol["chemistry"].get("spectrum_topic_assignment") != "dominant_topic":
        raise ValueError("chemical protocol must use dominant-topic assignment")
    directory = Path(run_dir).expanduser().resolve()
    group = "chemical" if split == "test" else "validation_chemical"
    evaluation_group = "evaluation" if split == "test" else "validation_evaluation"
    output = directory / group / method
    if (output / "complete.json").is_file():
        return read_json(output / "complete.json")
    data = directory / "data"
    evaluation = directory / evaluation_group / method
    theta_path = evaluation / f"{split}_full_theta.npy"
    theta = np.load(theta_path, mmap_mode="r")
    records = load_heldout_records(data, split)
    if theta.shape[0] != len(records):
        raise ValueError("full mixtures and held-out records differ")
    annotations, annotation = _shared_annotations(
        directory,
        method=annotation_method or method,
        data_root=data_root,
        protocol=protocol,
    )
    if "mag_failures" not in annotation:
        raise RuntimeError("MAG annotation evidence lacks explicit failure accounting")
    summary = _topic_scores(
        theta=theta,
        records=records,
        annotations=annotations,
        fingerprint_threshold=float(protocol["chemistry"]["mag_fingerprint_threshold"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "method": method,
        "annotation_method": annotation_method or method,
        "split": split,
        "topics": len(annotations),
        "annotation_coverage": annotation["annotation_coverage"],
        "chemical_evaluation": summary,
        "heldout_compounds_excluded_from_mag": annotation[
            "heldout_compounds_excluded_from_mag"
        ],
        "mag_failures": annotation["mag_failures"],
    }
    write_json(output / "complete.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated MAG worker used by the top-level pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    result = run_chemical_scoring(
        args.run,
        method=args.method,
        data_root=args.data_root,
        protocol=read_json(args.run / "protocol.json"),
        split=args.split,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
