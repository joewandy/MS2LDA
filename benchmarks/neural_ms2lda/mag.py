"""Leakage-filtered MAG index and motif-annotation primitives."""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import load_heldout_records
from .spectra import input_paths
from .utils import read_json, write_json


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
    return "" if mol is None else MolToInchiKey(mol)


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def build_filtered_mag_index(
    run_dir: str | Path,
    *,
    data_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build a FAISS index after excluding every held-out compound."""
    import faiss

    directory = Path(run_dir).expanduser().resolve()
    inputs = input_paths(protocol, data_root)
    index_dir = directory / "mag" / "index"
    complete_path = index_dir / "complete.json"
    if complete_path.is_file():
        return read_json(complete_path)

    excluded_keys = {
        str(row["connectivity_key"])
        for split in ("validation", "test")
        for row in load_heldout_records(directory / "data", split)
    }
    embeddings = np.load(inputs["spec2vec_embeddings"], mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != 300:
        raise ValueError("unexpected Spec2Vec embedding shape")

    connection = sqlite3.connect(inputs["spec2vec_db"])
    try:
        rows = connection.execute(
            "SELECT id, smiles FROM spectra ORDER BY CAST(id AS INTEGER)"
        )
        keep_ids: list[int] = []
        kept_connectivity: list[str] = []
        excluded_rows = 0
        database_rows = 0
        smiles_cache: dict[str, str] = {}
        for expected_id, (raw_id, raw_smiles) in enumerate(rows):
            database_rows += 1
            original_id = int(raw_id)
            if original_id != expected_id:
                raise ValueError("MAG database IDs are not contiguous and aligned")
            smiles = str(raw_smiles)
            connectivity = smiles_cache.setdefault(smiles, _connectivity_key(smiles))
            if connectivity in excluded_keys:
                excluded_rows += 1
            else:
                keep_ids.append(original_id)
                kept_connectivity.append(connectivity)
    finally:
        connection.close()
    if embeddings.shape[0] != database_rows:
        raise ValueError("MAG database and embedding row counts differ")
    leaked = sum(key in excluded_keys for key in kept_connectivity)
    if leaked:
        raise RuntimeError(f"MAG exclusion audit failed for {leaked} rows")

    index_dir.mkdir(parents=True, exist_ok=True)
    original_ids = np.asarray(keep_ids, dtype=np.int64)
    original_ids_path = index_dir / "kept_original_ids.npy"
    np.save(original_ids_path, original_ids)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    for start in range(0, len(original_ids), 16_384):
        selected = original_ids[start : start + 16_384]
        index.add(_normalize(np.asarray(embeddings[selected], dtype=np.float32)))
    index_path = index_dir / "spec2vec_filtered.faiss"
    faiss.write_index(index, str(index_path))
    excluded_path = index_dir / "excluded_connectivity_keys.json"
    write_json(excluded_path, {"connectivity_keys": sorted(excluded_keys)})
    result = {
        "database_rows": int(embeddings.shape[0]),
        "excluded_connectivity_keys": len(excluded_keys),
        "excluded_reference_rows": excluded_rows,
        "retained_reference_rows": len(original_ids),
        "retained_leak_rows": leaked,
    }
    write_json(complete_path, result)
    return result


def topic_spectra(
    beta: np.ndarray,
    vocabulary: Sequence[str],
    top_n: int,
    *,
    significant_digits: int,
):
    """Convert topic-word rows into Mass2Motif spectra for MAG."""
    from MS2LDA.utils import create_spectrum

    spectra = []
    for topic_id, row in enumerate(beta):
        count = min(int(top_n), row.shape[0])
        indices = np.argsort(-row, kind="stable")[:count]
        words = [(vocabulary[index], float(row[index])) for index in indices]
        spectra.append(
            create_spectrum(
                words,
                topic_id,
                charge=1,
                motifset="neural_ms2lda_msnlib",
                significant_digits=significant_digits,
            )
        )
    return spectra


def library_matches(
    *,
    similarities: np.ndarray,
    filtered_indices: np.ndarray,
    kept_original_ids: np.ndarray,
    db_path: Path,
    unique_molecules: int,
    excluded_connectivity: set[str],
) -> list[tuple[list[str], list[Any], list[float]]]:
    """Load unique, leakage-audited library hits for every topic."""
    connection = sqlite3.connect(db_path)
    output = []
    try:
        for topic in range(filtered_indices.shape[0]):
            smiles_values: list[str] = []
            spectra: list[Any] = []
            scores: list[float] = []
            seen_inchikeys: set[str] = set()
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
                    raise RuntimeError("MAG returned a held-out compound")
                inchikey = _full_inchikey(smiles)
                if not inchikey or inchikey in seen_inchikeys:
                    continue
                seen_inchikeys.add(inchikey)
                smiles_values.append(smiles)
                spectra.append(pickle.loads(row[1]))
                scores.append(float(similarities[topic, rank]))
                if len(smiles_values) == int(unique_molecules):
                    break
            output.append((smiles_values, spectra, scores))
    finally:
        connection.close()
    return output


def maccs_fingerprint(smiles: str) -> np.ndarray | None:
    """Return a boolean MACCS fingerprint, or ``None`` for invalid SMILES."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    values = np.zeros(fingerprint.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values.astype(bool)


def consensus_fingerprint(
    smiles_values: Sequence[str], threshold: float
) -> np.ndarray | None:
    """Return the thresholded consensus MACCS fingerprint for MAG hits."""
    fingerprints = [
        fingerprint
        for smiles in smiles_values
        if (fingerprint := maccs_fingerprint(smiles)) is not None
    ]
    if not fingerprints:
        return None
    return np.mean(np.asarray(fingerprints, dtype=np.float32), axis=0) >= threshold


def optimized_feature_count(spectrum: Any | None) -> int:
    """Count fragments and losses retained by MAG motif optimization."""
    if spectrum is None:
        return 0
    peaks = getattr(getattr(spectrum, "peaks", None), "mz", ())
    losses = getattr(getattr(spectrum, "losses", None), "mz", ())
    return len(peaks) + len(losses)
