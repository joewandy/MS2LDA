"""Input identity and compound-level joins for post-hoc chemical assessment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mag import maccs_fingerprint
from .spectra import iter_mgf
from .utils import read_json, sha256_file, write_json


def read_jsonl(path: Path) -> list[dict]:
    """Read a saved line-oriented record stream without changing its order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fit_paths(fit: dict) -> dict[str, Path]:
    """Name the exact saved artifacts used, keeping full-validation theta."""
    root, method = Path(fit["path"]), fit["method"]
    return {
        "beta": root / f"validation_evaluation/{method}/beta.npy",
        "theta": root / f"validation_evaluation/{method}/validation_full_theta.npy",
        "annotations": root / f"mag/annotations/{method}/annotations.jsonl",
        "chemical": root / f"validation_chemical/{method}/complete.json",
        "evaluation": root / f"validation_evaluation/{method}/complete.json",
        "records": root / "data/validation_records.jsonl",
        "vocabulary": root / "data/vocabulary.json",
        "protocol": root / "protocol.json",
        "validation_manifest": root / "validation_input_manifest.json",
    }


def seal_inputs(protocol_path: Path, output: Path) -> None:
    """Write an immutable-by-convention inventory BEFORE outcome calculation.

    A second seal never overwrites the first. The digest is content-based, so
    all later stages abort if either scientific settings or input bytes change.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("sealing requires a new, empty output directory")
    protocol = read_json(protocol_path)
    paths = {protocol_path, Path(protocol["mgf"])}
    paths.update(Path(p) for p in protocol["references"])
    model = Path(protocol["spec2vec_model"])
    paths.update(model.parent.glob(model.name + "*"))
    paths.add(model)
    for fit in protocol["fits"]:
        paths.update(fit_paths(fit).values())
    paths.update(
        Path(protocol["prepared"]) / "data" / name
        for name in ("validation_records.jsonl", "vocabulary.json")
    )
    entries = {
        str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(paths)
    }
    versions = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "rdkit", "numba", "gensim", "spec2vec", "matchms")
    }
    write_json(
        output / "input_seal.json",
        {
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "inputs": entries,
            "versions": versions,
            "outcomes_calculated_before_seal": False,
        },
    )
    # Preserve the exact sealed bytes, not just an equivalent JSON object.
    # Report reproduction can then verify settings without the raw inputs or
    # assumptions about where this output directory is stored.
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    print(
        f"Sealed {len(entries)} input paths before new outcome calculation.", flush=True
    )


def verify_seal(protocol_path: Path, output: Path) -> dict:
    """Fail closed on modified protocol, files, dependency versions or alignment."""
    seal = read_json(output / "input_seal.json")
    if sha256_file(protocol_path) != seal["protocol_sha256"]:
        raise ValueError("analysis protocol changed after sealing")
    for name, expected in seal["inputs"].items():
        path = Path(name)
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"sealed input changed: {name}")
    for package, version in seal["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"sealed dependency changed: {package}")
    protocol = read_json(protocol_path)
    for role in ("records", "vocabulary"):
        if (
            len(
                {
                    seal["inputs"][str(fit_paths(f)[role])]["sha256"]
                    for f in protocol["fits"]
                }
            )
            != 1
        ):
            raise ValueError(f"fit {role} are not identically ordered")
    return protocol


def compound_cohort(
    records: list[dict], mgf_path: Path
) -> tuple[list[dict], np.ndarray]:
    """Join raw acquisition metadata by exact USI and deduplicate compounds.

    The returned mapping has one compound index per saved spectrum row. A
    compound fingerprint is shared across all its spectra, even when spectra
    have different dominant topics. Acquisition profiles are sets of pairs,
    not independently resampled spectrum-level covariates.
    """
    ids = [r["spectrum_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("validation spectrum IDs must be unique")
    wanted, metadata = set(ids), {}
    for meta, _, _ in iter_mgf(mgf_path):
        usi = meta.get("usi")
        if usi not in wanted:
            continue
        if usi in metadata:
            raise ValueError("ambiguous duplicate raw USI")
        metadata[usi] = meta
    if set(metadata) != wanted:
        raise ValueError(
            f"missing raw acquisition metadata: {len(wanted - set(metadata))}"
        )
    grouped = defaultdict(list)
    for r in records:
        grouped[r["connectivity_key"]].append(r)
    compounds = []
    for key, rows in sorted(grouped.items()):
        # Follow the original evaluator's first stored connectivity fingerprint;
        # reject disagreements rather than choosing one after seeing results.
        fps = [maccs_fingerprint(r["smiles"]) for r in rows]
        if any(fp is None for fp in fps) or any(
            not np.array_equal(fps[0], fp) for fp in fps[1:]
        ):
            raise ValueError(f"invalid or inconsistent connectivity fingerprint: {key}")
        scaffolds = {r["scaffold_key"] for r in rows}
        if len(scaffolds) != 1:
            raise ValueError(f"inconsistent compound scaffold: {key}")
        meta = [metadata[r["spectrum_id"]] for r in rows]
        masses = [float(m["precursor_mz"]) for m in meta]
        if not np.all(np.isfinite(masses)) or min(masses) <= 0:
            raise ValueError("invalid precursor mass")
        profiles = sorted({(m["adduct"], m["collision_energy"]) for m in meta})
        compounds.append(
            {
                "connectivity_key": key,
                "scaffold_key": rows[0]["scaffold_key"] or f"acyclic:{key}",
                "smiles": rows[0]["smiles"],
                "spectrum_ids": [r["spectrum_id"] for r in rows],
                "median_precursor_mz": float(np.median(masses)),
                "acquisition_profile": profiles,
                "maccs": fps[0].astype(int).tolist(),
                "instrument_profiles": sorted(
                    {
                        (
                            m["ms_mass_analyzer"],
                            m["fragmentation_method"],
                            m["ms_level"],
                        )
                        for m in meta
                    }
                ),
            }
        )
    lookup = {r["connectivity_key"]: i for i, r in enumerate(compounds)}
    return compounds, np.array([lookup[r["connectivity_key"]] for r in records])


def cohort_strata(
    compounds: list[dict], width_da: float, balanced: bool
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return retained global compound IDs and disjoint local permutation blocks."""
    if not np.isfinite(width_da) or width_da <= 0:
        raise ValueError("mass-bin width must be positive")
    selected = np.arange(len(compounds))
    if balanced:
        groups = defaultdict(list)
        for i, row in enumerate(compounds):
            groups[row["scaffold_key"]].append(i)
        selected = np.array(
            sorted(
                min(
                    values,
                    key=lambda i: hashlib.sha256(
                        compounds[i]["connectivity_key"].encode()
                    ).hexdigest(),
                )
                for values in groups.values()
            )
        )
    blocks = defaultdict(list)
    for local, global_id in enumerate(selected):
        row = compounds[global_id]
        key = (
            tuple(map(tuple, row["acquisition_profile"])),
            int(np.floor(row["median_precursor_mz"] / width_da)),
        )
        blocks[key].append(local)
    return selected, [np.array(v, dtype=np.int64) for _, v in sorted(blocks.items())]
