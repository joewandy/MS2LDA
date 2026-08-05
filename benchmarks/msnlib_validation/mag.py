"""Leakage-filtered MAG and raw-DreaMS chemical evaluation."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import (
    environment_manifest,
    file_sha256,
    load_config,
    read_json,
    resolve_input_paths,
    write_json,
)
from .data import SpectrumRecord, load_records, split_records
from .protocol import (
    load_assignments,
    load_vocabulary,
    verify_frozen_input_files,
    verify_protocol,
)
from .runtime import load_feature_cache, peak_rss_bytes


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _connectivity_key(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid MAG-library SMILES: {smiles!r}")
    return MolToInchiKey(mol).split("-", 1)[0]


def _full_inchikey(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MolToInchiKey(mol)


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def audit_mag_exclusion(
    excluded_connectivity: set[str], retained_connectivity: Iterable[str]
) -> dict[str, int]:
    """Assert that no retained MAG row belongs to a held-out compound."""
    retained = list(retained_connectivity)
    leaked = sum(key in excluded_connectivity for key in retained)
    if leaked:
        raise RuntimeError(f"MAG exclusion audit failed for {leaked} rows")
    return {"retained_rows": len(retained), "retained_leak_rows": leaked}


def _require_frozen_data_root(lock: dict[str, Any], data_root: str | Path) -> Path:
    """Reject MAG assets from a location other than the frozen input root."""
    resolved = Path(data_root).expanduser().resolve()
    frozen = Path(lock["data_root"]).expanduser().resolve()
    if resolved != frozen:
        raise ValueError(f"MAG data root differs from frozen protocol: {resolved}")
    return resolved


def build_filtered_mag_index(
    run_dir: str | Path, *, data_root: str | Path
) -> dict[str, Any]:
    """Build a FAISS index after excluding every nontraining compound."""
    import faiss

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    frozen_data_root = _require_frozen_data_root(lock, data_root)
    inputs = resolve_input_paths(config, frozen_data_root)
    index_dir = directory / "mag" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("MAG index belongs to another frozen protocol")
        for name, digest in manifest["output_sha256"].items():
            if file_sha256(index_dir / name) != digest:
                raise ValueError(f"MAG index artifact changed: {name}")
        return manifest
    excluded_keys = {
        str(row["connectivity_key"])
        for row in _jsonl_rows(directory / "split_manifest.jsonl")
        if row["split"] in {"validation", "test"}
    }
    embeddings = np.load(inputs["spec2vec_embeddings"], mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != 300:
        raise ValueError("unexpected Spec2Vec embedding shape")
    connection = sqlite3.connect(inputs["spec2vec_db"])
    try:
        rows = connection.execute(
            "SELECT id, smiles FROM spectra ORDER BY CAST(id AS INTEGER)"
        )
        keep_ids = []
        kept_connectivity = []
        excluded_rows = 0
        smiles_cache: dict[str, str] = {}
        database_rows = 0
        for expected_id, (raw_id, smiles) in enumerate(rows):
            database_rows += 1
            original_id = int(raw_id)
            if original_id != expected_id:
                raise ValueError("MAG database IDs are not contiguous and row aligned")
            smiles = str(smiles)
            connectivity = smiles_cache.get(smiles)
            if connectivity is None:
                connectivity = _connectivity_key(smiles)
                smiles_cache[smiles] = connectivity
            if connectivity in excluded_keys:
                excluded_rows += 1
            else:
                keep_ids.append(original_id)
                kept_connectivity.append(connectivity)
    finally:
        connection.close()
    if embeddings.shape[0] != database_rows:
        raise ValueError("MAG database and embedding row counts differ")
    exclusion_audit = audit_mag_exclusion(excluded_keys, kept_connectivity)
    original_ids = np.asarray(keep_ids, dtype=np.int64)
    original_ids_path = index_dir / "kept_original_ids.npy"
    np.save(original_ids_path, original_ids)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    chunk_size = 16_384
    for start in range(0, len(original_ids), chunk_size):
        selected = original_ids[start : start + chunk_size]
        index.add(_normalize(np.asarray(embeddings[selected], dtype=np.float32)))
    index_path = index_dir / "spec2vec_filtered.faiss"
    faiss.write_index(index, str(index_path))
    excluded_keys_path = index_dir / "excluded_connectivity_keys.json"
    write_json(excluded_keys_path, {"connectivity_keys": sorted(excluded_keys)})
    outputs = (original_ids_path, index_path, excluded_keys_path)
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "database_rows": int(embeddings.shape[0]),
        "excluded_connectivity_keys": len(excluded_keys),
        "excluded_reference_rows": excluded_rows,
        "retained_reference_rows": len(original_ids),
        "retained_leak_rows": exclusion_audit["retained_leak_rows"],
        "embedding_database_alignment": True,
        "peak_rss_bytes": peak_rss_bytes(),
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    write_json(manifest_path, manifest)
    return manifest


def _topic_spectra(beta: np.ndarray, vocabulary: Sequence[str], top_n: int):
    from MS2LDA.utils import create_spectrum

    spectra = []
    for topic_id, row in enumerate(beta):
        count = min(top_n, row.shape[0])
        indices = np.argsort(-row, kind="stable")[:count]
        words = [(vocabulary[index], float(row[index])) for index in indices]
        spectra.append(
            create_spectrum(
                words,
                topic_id,
                charge=1,
                motifset="msnlib_validation",
                significant_digits=2,
            )
        )
    return spectra


def _library_matches(
    *,
    similarities: np.ndarray,
    filtered_indices: np.ndarray,
    kept_original_ids: np.ndarray,
    db_path: Path,
    unique_molecules: int,
    excluded_connectivity: set[str],
) -> list[tuple[list[str], list[Any], list[float]]]:
    connection = sqlite3.connect(db_path)
    output = []
    try:
        for topic in range(filtered_indices.shape[0]):
            smiles_values = []
            spectra = []
            scores = []
            seen_inchikeys = set()
            for rank, filtered_index in enumerate(filtered_indices[topic]):
                if filtered_index < 0:
                    continue
                original_id = int(kept_original_ids[int(filtered_index)])
                row = connection.execute(
                    "SELECT smiles, spectrum FROM spectra WHERE id = ?", (original_id,)
                ).fetchone()
                if row is None:
                    continue
                smiles = str(row[0])
                if _connectivity_key(smiles) in excluded_connectivity:
                    raise RuntimeError(
                        "MAG search returned a held-out compound after filtering"
                    )
                inchikey = _full_inchikey(smiles)
                if not inchikey or inchikey in seen_inchikeys:
                    continue
                seen_inchikeys.add(inchikey)
                smiles_values.append(smiles)
                spectra.append(pickle.loads(row[1]))
                scores.append(float(similarities[topic, rank]))
                if len(smiles_values) == unique_molecules:
                    break
            output.append((smiles_values, spectra, scores))
    finally:
        connection.close()
    return output


def _maccs_fingerprint(smiles: str) -> np.ndarray | None:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    values = np.zeros(fingerprint.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values.astype(bool)


def _consensus_fingerprint(
    smiles_values: Sequence[str], threshold: float
) -> np.ndarray | None:
    fingerprints = [
        fingerprint
        for smiles in smiles_values
        if (fingerprint := _maccs_fingerprint(smiles)) is not None
    ]
    if not fingerprints:
        return None
    return np.mean(np.asarray(fingerprints, dtype=np.float32), axis=0) >= threshold


def _associated_smiles(
    theta: np.ndarray,
    records: Sequence[SpectrumRecord],
    threshold: float,
) -> dict[int, list[str]]:
    if theta.shape[0] != len(records):
        raise ValueError("topic memberships and test records differ")
    associated: dict[int, list[str]] = defaultdict(list)
    rows, topics = np.where(theta >= threshold)
    for row, topic in zip(rows, topics, strict=True):
        associated[int(topic)].append(records[int(row)].smiles)
    return associated


def _score_mag_topic(
    *,
    topic_id: int,
    clustered_smiles: Sequence[str],
    associated_smiles: Sequence[str],
    fingerprint_threshold: float,
    calculate_sos_fn: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    annotation = _consensus_fingerprint(clustered_smiles, fingerprint_threshold)
    molecule_fps = [
        fingerprint
        for smiles in associated_smiles
        if (fingerprint := _maccs_fingerprint(smiles)) is not None
    ]
    scores = (
        [calculate_sos_fn(annotation, fingerprint) for fingerprint in molecule_fps]
        if annotation is not None
        else []
    )
    mean_sos = float(np.mean(scores)) if scores else None
    if mean_sos is None:
        quality = "unavailable"
    elif mean_sos > 0.8:
        quality = "high"
    elif mean_sos > 0.6:
        quality = "intermediate"
    else:
        quality = "low"
    return {
        "topic_id": topic_id,
        "clustered_smiles": list(clustered_smiles),
        "cluster_size": len(clustered_smiles),
        "associated_test_molecules": len(molecule_fps),
        "sos": mean_sos,
        "quality_bin": quality,
        "eligible": bool(scores),
    }


def run_mag_for_model(
    run_dir: str | Path,
    *,
    data_root: str | Path,
    seed: int,
    method: str,
) -> dict[str, Any]:
    """Annotate and score all topics for one completed core model."""
    # SciPy's optimizer stack and FAISS ship separate OpenMP runtimes in the
    # legacy environment. MAG needs both, so retain its historically working
    # import order here while keeping the raw-DreaMS worker free of SciPy.
    calculate_sos = import_module(f"{__package__}.metrics").calculate_sos

    import faiss

    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_model
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import hit_clustering

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    config = load_config(directory / "config.resolved.json")
    frozen_data_root = _require_frozen_data_root(lock, data_root)
    if seed not in config.seeds or method not in {"tomotopy", "hybrid"}:
        raise ValueError("method or seed is not frozen")
    output = directory / "mag" / f"seed_{seed}" / method
    complete_path = output / "complete.json"
    if complete_path.exists():
        result = read_json(complete_path)
        if result.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("MAG result belongs to another frozen protocol")
        if file_sha256(output / "topics.jsonl") != result["topics_sha256"]:
            raise ValueError("MAG topic rows changed after completion")
        return result
    output.mkdir(parents=True, exist_ok=True)
    index_manifest = build_filtered_mag_index(directory, data_root=data_root)
    inputs = resolve_input_paths(config, frozen_data_root)
    core_dir = directory / "core" / f"seed_{seed}" / method
    core_result = read_json(core_dir / "complete.json")
    beta_path = core_dir / "beta.npy"
    if file_sha256(beta_path) != core_result["beta_sha256"]:
        raise ValueError("core topic matrix changed before MAG evaluation")
    beta = np.load(beta_path, mmap_mode="r")
    vocabulary = load_vocabulary(directory)
    motif_spectra = _topic_spectra(beta, vocabulary, config.motif_spectrum_top_n)
    spec2vec = load_s2v_model(str(inputs["spec2vec_model"]))
    query_embeddings = calc_embeddings(spec2vec, motif_spectra).astype(np.float32)
    index_dir = directory / "mag" / "index"
    index = faiss.read_index(str(index_dir / "spec2vec_filtered.faiss"))
    kept_original_ids = np.load(index_dir / "kept_original_ids.npy", mmap_mode="r")
    excluded_connectivity = set(
        map(
            str,
            read_json(index_dir / "excluded_connectivity_keys.json")[
                "connectivity_keys"
            ],
        )
    )
    started = time.perf_counter()
    similarities, filtered_indices = index.search(
        _normalize(query_embeddings), min(config.mag_search_k, index.ntotal)
    )
    matches = _library_matches(
        similarities=similarities,
        filtered_indices=filtered_indices,
        kept_original_ids=kept_original_ids,
        db_path=inputs["spec2vec_db"],
        unique_molecules=config.mag_unique_molecules,
        excluded_connectivity=excluded_connectivity,
    )
    clustered = []
    cluster_errors = []
    for topic_id, (motif, match) in enumerate(zip(motif_spectra, matches, strict=True)):
        try:
            _, smiles, _ = hit_clustering(
                s2v_similarity=spec2vec,
                motif_spectra=[motif],
                library_matches=[match],
                criterium="best",
                cosine_similarity=config.mag_cluster_cosine,
            )
            clustered.append(smiles[0] if smiles else [])
            cluster_errors.append("")
        except Exception as exc:  # record per-topic failure; never omit it
            clustered.append([])
            cluster_errors.append(f"{type(exc).__name__}: {exc}")
    mag_seconds = time.perf_counter() - started
    mgf_path = Path(lock["data_root"]) / config.input_files["mgf"]["relative_path"]
    records, _ = load_records(mgf_path, config)
    assignments = load_assignments(directory)
    test_records = split_records(records, assignments, "test")
    if method == "tomotopy":
        theta_path = core_dir / "test_theta.npy"
        expected_theta_sha256 = core_result["theta_sha256"]
    else:
        reference_steps = str(core_result["reference_steps"])
        theta_path = core_dir / f"test_theta_{reference_steps}.npy"
        expected_theta_sha256 = core_result["theta_sha256"][reference_steps]
    if file_sha256(theta_path) != expected_theta_sha256:
        raise ValueError("core test mixtures changed before MAG evaluation")
    theta = np.load(theta_path, mmap_mode="r")
    associated = _associated_smiles(theta, test_records, config.membership_threshold)
    rows = []
    for topic_id in range(config.num_topics):
        row = _score_mag_topic(
            topic_id=topic_id,
            clustered_smiles=clustered[topic_id],
            associated_smiles=associated.get(topic_id, []),
            fingerprint_threshold=config.mag_fingerprint_threshold,
            calculate_sos_fn=calculate_sos,
        )
        row.update(
            {
                "seed": seed,
                "method": method,
                "cluster_error": cluster_errors[topic_id],
                "retrieved_smiles": matches[topic_id][0],
                "retrieved_scores": matches[topic_id][2],
            }
        )
        rows.append(row)
    rows_path = output / "topics.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    eligible = [row for row in rows if row["eligible"]]
    sos_values = [float(row["sos"]) for row in eligible]
    clustered_topics = sum(bool(row["clustered_smiles"]) for row in rows)
    result = {
        "method": method,
        "protocol_sha256": lock["protocol_sha256"],
        "seed": seed,
        "topics": config.num_topics,
        "mag_seconds": mag_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "annotation_coverage": clustered_topics / config.num_topics,
        "sos_evaluable_coverage": len(eligible) / config.num_topics,
        "eligible_topics": len(eligible),
        "clustered_topics": clustered_topics,
        "cluster_failures": sum(bool(row["cluster_error"]) for row in rows),
        "mean_sos": float(np.mean(sos_values)) if sos_values else None,
        "median_sos": float(np.median(sos_values)) if sos_values else None,
        "quality_counts": {
            quality: sum(row["quality_bin"] == quality for row in rows)
            for quality in ("high", "intermediate", "low", "unavailable")
        },
        "index_exclusion_audit": index_manifest,
        "topics_sha256": file_sha256(rows_path),
    }
    write_json(complete_path, result)
    return result


def run_raw_dreams_baseline(run_dir: str | Path) -> dict[str, Any]:
    """Evaluate exact DreaMS nearest-neighbour structural similarity."""
    import faiss
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    output = directory / "mag" / "raw_dreams"
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "complete.json"
    if complete_path.exists():
        result = read_json(complete_path)
        if result.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("raw-DreaMS result belongs to another frozen protocol")
        if file_sha256(output / "nearest_neighbors.jsonl") != result["rows_sha256"]:
            raise ValueError("raw-DreaMS rows changed after completion")
        return result
    config = load_config(directory / "config.resolved.json")
    mgf_path = Path(lock["data_root"]) / config.input_files["mgf"]["relative_path"]
    records, _ = load_records(mgf_path, config)
    assignments = load_assignments(directory)
    train = split_records(records, assignments, "train")
    test = split_records(records, assignments, "test")
    feature_ids, embeddings, _, _ = load_feature_cache(directory)
    row_by_id = {identifier: row for row, identifier in enumerate(feature_ids)}
    train_embeddings = _normalize(
        np.asarray(embeddings[[row_by_id[row.spectrum_id] for row in train]])
    )
    test_embeddings = _normalize(
        np.asarray(embeddings[[row_by_id[row.spectrum_id] for row in test]])
    )
    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_embeddings)
    started = time.perf_counter()
    similarities, neighbours = index.search(test_embeddings, 1)
    search_seconds = time.perf_counter() - started

    def fingerprint(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    train_fps = {}
    test_fps = {}
    rows = []
    for test_index, neighbour in enumerate(neighbours[:, 0]):
        query = test[test_index]
        match = train[int(neighbour)]
        query_fp = test_fps.setdefault(query.smiles, fingerprint(query.smiles))
        match_fp = train_fps.setdefault(match.smiles, fingerprint(match.smiles))
        rows.append(
            {
                "spectrum_id": query.spectrum_id,
                "nearest_train_spectrum_id": match.spectrum_id,
                "dreams_cosine": float(similarities[test_index, 0]),
                "connectivity_match": query.connectivity_key == match.connectivity_key,
                "scaffold_match": query.split_group == match.split_group,
                "morgan_tanimoto": float(
                    DataStructs.TanimotoSimilarity(query_fp, match_fp)
                ),
            }
        )
    rows_path = output / "nearest_neighbors.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "method": "raw_dreams",
        "protocol_sha256": lock["protocol_sha256"],
        "test_spectra": len(rows),
        "search_seconds": search_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "connectivity_match_fraction": float(
            np.mean([row["connectivity_match"] for row in rows])
        ),
        "scaffold_match_fraction": float(
            np.mean([row["scaffold_match"] for row in rows])
        ),
        "mean_morgan_tanimoto": float(
            np.mean([row["morgan_tanimoto"] for row in rows])
        ),
        "median_morgan_tanimoto": float(
            np.median([row["morgan_tanimoto"] for row in rows])
        ),
        "rows_sha256": file_sha256(rows_path),
    }
    write_json(complete_path, result)
    return result


def run_all_mag(run_dir: str | Path, *, data_root: str | Path) -> dict[str, Any]:
    """Run leakage-filtered MAG for all seeds and methods, then DreaMS."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_protocol(directory)
    _require_frozen_data_root(lock, data_root)
    verify_frozen_input_files(
        directory,
        names={"mgf", "spec2vec_db", "spec2vec_embeddings", "spec2vec_model"},
        lock=lock,
    )
    if not (directory / "core" / "complete.json").is_file():
        raise RuntimeError("all core models must complete before MAG")
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

    def worker(*arguments: str) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.msnlib_validation",
                *arguments,
                "--run",
                str(directory),
            ],
            cwd=lock["repo_root"],
            env=environment,
            check=True,
        )

    index_path = directory / "mag" / "index" / "manifest.json"
    if not index_path.exists():
        worker("_build-mag-index", "--data-root", str(data_root))
    index_manifest = read_json(index_path)
    results = []
    for seed in config.seeds:
        for method in ("tomotopy", "hybrid"):
            result_path = directory / "mag" / f"seed_{seed}" / method / "complete.json"
            if not result_path.exists():
                worker(
                    "_run-mag-model",
                    "--data-root",
                    str(data_root),
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                )
            results.append(read_json(result_path))
    dreams_path = directory / "mag" / "raw_dreams" / "complete.json"
    if not dreams_path.exists():
        worker("_run-raw-dreams")
    dreams = read_json(dreams_path)
    manifest = {
        "protocol_sha256": lock["protocol_sha256"],
        "environment": environment_manifest(),
        "index": index_manifest,
        "completed": [
            {"method": row["method"], "seed": row["seed"]} for row in results
        ],
        "raw_dreams": dreams,
    }
    write_json(directory / "mag" / "complete.json", manifest)
    return manifest
