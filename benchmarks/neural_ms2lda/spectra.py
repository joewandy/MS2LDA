"""MSnLib parsing, preprocessing, scaffold splitting, and document completion."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .inputs import PreprocessingConfig


@dataclass(frozen=True)
class PeakGroup:
    """One physical peak and all vocabulary tokens derived from it."""

    original_index: int
    mz: float
    intensity: float
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class SpectrumRecord:
    """Validated spectrum plus evaluation-only chemical metadata."""

    spectrum_id: str
    feature_id: str
    smiles: str
    supplied_inchikey: str
    connectivity_key: str
    scaffold_key: str
    split_group: str
    precursor_mz: float
    peak_groups: tuple[PeakGroup, ...]
    declared_num_peaks: int | None
    parsed_num_peaks: int
    compound_name: str
    metadata: dict[str, str]

    @property
    def words(self) -> list[str]:
        """Return the full intensity-weighted document."""
        return [token for group in self.peak_groups for token in group.tokens]


@dataclass(frozen=True)
class CompletionDocument:
    """Observed and completion peak groups for one held-out spectrum."""

    spectrum_id: str
    observed_groups: tuple[PeakGroup, ...]
    completion_groups: tuple[PeakGroup, ...]

    @property
    def observed_words(self) -> list[str]:
        return [token for group in self.observed_groups for token in group.tokens]

    @property
    def completion_words(self) -> list[str]:
        return [token for group in self.completion_groups for token in group.tokens]


def _metadata_key(key: str) -> str:
    return key.strip().lower()


def _finish_raw_record(
    metadata: dict[str, str], mz: list[float], intensities: list[float]
) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    if not metadata:
        raise ValueError("MGF record has no metadata")
    return (
        dict(metadata),
        np.asarray(mz, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
    )


def iter_mgf(
    path: str | Path,
) -> Iterable[tuple[dict[str, str], np.ndarray, np.ndarray]]:
    """Yield MGF blocks while trusting parsed peaks over ``NUM_PEAKS``."""
    in_block = False
    metadata: dict[str, str] = {}
    mz: list[float] = []
    intensities: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                if in_block:
                    raise ValueError(f"nested BEGIN IONS at line {line_number}")
                in_block = True
                metadata = {}
                mz = []
                intensities = []
                continue
            if line == "END IONS":
                if not in_block:
                    raise ValueError(f"END IONS without a block at line {line_number}")
                yield _finish_raw_record(metadata, mz, intensities)
                in_block = False
                continue
            if not in_block:
                raise ValueError(f"content outside MGF block at line {line_number}")
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[_metadata_key(key)] = value.strip()
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"malformed peak at line {line_number}: {line!r}")
            try:
                mz.append(float(fields[0]))
                intensities.append(float(fields[1]))
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric peak at line {line_number}: {line!r}"
                ) from exc
    if in_block:
        raise ValueError("unterminated MGF block")


def _spectral_word(kind: str, value: float, digits: int) -> str:
    return f"{kind}@{round(float(value), digits)}"


def _clean_peaks(
    mz: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    config: PreprocessingConfig,
) -> tuple[PeakGroup, ...]:
    if mz.ndim != 1 or mz.shape != intensities.shape or not len(mz):
        return ()
    valid = np.isfinite(mz) & np.isfinite(intensities) & (mz > 0) & (intensities >= 0)
    mz = mz[valid]
    intensities = intensities[valid]
    if not len(mz) or float(np.max(intensities)) <= 0:
        return ()
    intensities = intensities / float(np.max(intensities))
    retained = (
        (mz >= config.min_mz)
        & (mz <= config.max_mz)
        & (intensities >= config.min_intensity)
        & (intensities <= config.max_intensity)
    )
    original_indices = np.flatnonzero(valid)[retained]
    mz = mz[retained]
    intensities = intensities[retained]
    if len(mz) > config.max_fragments:
        top = np.argsort(-intensities, kind="stable")[: config.max_fragments]
        top = top[np.argsort(mz[top], kind="stable")]
        mz = mz[top]
        intensities = intensities[top]
        original_indices = original_indices[top]
    if len(mz) < config.min_fragments:
        return ()
    groups = []
    for original_index, peak_mz, intensity in zip(
        original_indices, mz, intensities, strict=True
    ):
        repetitions = int(np.rint(float(intensity) * 100.0))
        if repetitions < 1:
            continue
        fragment = _spectral_word("frag", peak_mz, config.significant_digits)
        tokens = [fragment] * repetitions
        loss = precursor_mz - float(peak_mz)
        if loss > 0.01:
            tokens.extend(
                [_spectral_word("loss", loss, config.significant_digits)] * repetitions
            )
        groups.append(
            PeakGroup(
                original_index=int(original_index),
                mz=float(peak_mz),
                intensity=float(intensity),
                tokens=tuple(tokens),
            )
        )
    return tuple(groups)


def _structure_keys(smiles: str, supplied_inchikey: str) -> tuple[str, str, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - environment validation
        raise ImportError(
            "RDKit is required for leakage-safe structural splits"
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    derived_inchikey = MolToInchiKey(mol)
    connectivity = derived_inchikey.split("-", 1)[0]
    supplied_connectivity = supplied_inchikey.split("-", 1)[0]
    if supplied_connectivity != connectivity:
        raise ValueError(
            "SMILES/InChIKey connectivity mismatch: "
            f"{connectivity} != {supplied_connectivity}"
        )
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    group = f"scaffold:{scaffold}" if scaffold else f"acyclic:{connectivity}"
    return connectivity, scaffold, group


def load_records(
    path: str | Path,
    config: PreprocessingConfig,
    *,
    require_expected_count: bool = True,
) -> tuple[list[SpectrumRecord], dict[str, Any]]:
    """Parse, clean, structurally validate, and identify the full MSnLib MGF."""
    records: list[SpectrumRecord] = []
    seen_ids: set[str] = set()
    malformed_num_peaks: list[dict[str, Any]] = []
    dropped_too_few_peaks = 0
    missing_optional_name = 0
    for index, (metadata, mz, intensities) in enumerate(iter_mgf(path)):
        missing = [
            key
            for key in ("usi", "feature_id", "smiles", "inchikey", "precursor_mz")
            if not metadata.get(key)
        ]
        if missing:
            raise ValueError(f"spectrum {index} is missing metadata: {missing}")
        spectrum_id = metadata["usi"]
        if spectrum_id in seen_ids:
            raise ValueError(f"duplicate USI: {spectrum_id}")
        seen_ids.add(spectrum_id)
        try:
            precursor_mz = float(metadata["precursor_mz"])
        except ValueError as exc:
            raise ValueError(f"invalid precursor_mz for {spectrum_id}") from exc
        if not math.isfinite(precursor_mz) or precursor_mz <= 0:
            raise ValueError(f"invalid precursor_mz for {spectrum_id}")
        declared_raw = metadata.get("num_peaks")
        try:
            declared = int(declared_raw) if declared_raw else None
        except ValueError as exc:
            raise ValueError(f"invalid num_peaks for {spectrum_id}") from exc
        if declared is not None and declared != len(mz):
            malformed_num_peaks.append(
                {
                    "spectrum_index": index,
                    "spectrum_id": spectrum_id,
                    "feature_id": metadata["feature_id"],
                    "declared": declared,
                    "parsed": len(mz),
                }
            )
        groups = _clean_peaks(mz, intensities, precursor_mz, config)
        if not groups:
            dropped_too_few_peaks += 1
            continue
        connectivity, scaffold, split_group = _structure_keys(
            metadata["smiles"], metadata["inchikey"]
        )
        compound_name = metadata.get("compound_name", "")
        if not compound_name:
            missing_optional_name += 1
        records.append(
            SpectrumRecord(
                spectrum_id=spectrum_id,
                feature_id=metadata["feature_id"],
                smiles=metadata["smiles"],
                supplied_inchikey=metadata["inchikey"],
                connectivity_key=connectivity,
                scaffold_key=scaffold,
                split_group=split_group,
                precursor_mz=precursor_mz,
                peak_groups=groups,
                declared_num_peaks=declared,
                parsed_num_peaks=len(mz),
                compound_name=compound_name,
                metadata=metadata,
            )
        )
    if require_expected_count and len(seen_ids) != config.expected_spectra:
        raise ValueError(
            f"expected {config.expected_spectra} input spectra, found {len(seen_ids)}"
        )
    summary = {
        "parsed_blocks": len(seen_ids),
        "retained_spectra": len(records),
        "dropped_too_few_peaks": dropped_too_few_peaks,
        "unique_connectivity_keys": len(
            {record.connectivity_key for record in records}
        ),
        "unique_split_groups": len({record.split_group for record in records}),
        "acyclic_spectra": sum(not record.scaffold_key for record in records),
        "missing_optional_compound_name": missing_optional_name,
        "num_peaks_mismatches": malformed_num_peaks,
    }
    return records, summary


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def assign_scaffold_splits(
    records: Sequence[SpectrumRecord],
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign whole scaffold groups while minimizing target-count deficits."""
    names = ("train", "validation", "test")
    grouped: dict[str, list[SpectrumRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split_group].append(record)
    targets = {
        name: len(records) * fraction
        for name, fraction in zip(names, fractions, strict=True)
    }
    counts = dict.fromkeys(names, 0)
    assignments: dict[str, str] = {}
    group_order = sorted(
        grouped,
        key=lambda group: (-len(grouped[group]), _stable_digest(seed, group)),
    )
    for group in group_order:
        size = len(grouped[group])

        def score(split: str) -> tuple[float, float, str]:
            target = targets[split]
            projected = counts[split] + size
            overflow = max(projected - target, 0.0) / max(target, 1.0)
            deficit = (target - counts[split]) / max(target, 1.0)
            return overflow, -deficit, _stable_digest(seed, f"{group}\0{split}")

        selected = min(names, key=score)
        assignments[group] = selected
        counts[selected] += size
    by_id = {record.spectrum_id: assignments[record.split_group] for record in records}
    audit_split_disjointness(records, by_id)
    summary = {
        "algorithm": "descending-group-size deterministic deficit minimization",
        "seed": seed,
        "targets": targets,
        "spectrum_counts": counts,
        "group_counts": {
            name: sum(split == name for split in assignments.values()) for name in names
        },
    }
    return by_id, summary


def audit_split_disjointness(
    records: Sequence[SpectrumRecord], assignments: dict[str, str]
) -> dict[str, Any]:
    """Assert zero compound and scaffold leakage across partitions."""
    expected = {record.spectrum_id for record in records}
    if set(assignments) != expected:
        missing = expected - set(assignments)
        extra = set(assignments) - expected
        raise ValueError(
            f"split manifest mismatch; missing={len(missing)} extra={len(extra)}"
        )
    compound_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = assignments[record.spectrum_id]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        compound_splits[record.connectivity_key].add(split)
        group_splits[record.split_group].add(split)
    leaked_compounds = [
        key for key, values in compound_splits.items() if len(values) > 1
    ]
    leaked_groups = [key for key, values in group_splits.items() if len(values) > 1]
    if leaked_compounds or leaked_groups:
        raise ValueError(
            f"split leakage: compounds={len(leaked_compounds)} groups={len(leaked_groups)}"
        )
    return {
        "leaked_compounds": 0,
        "leaked_groups": 0,
        "connectivity_groups": len(compound_splits),
        "split_groups": len(group_splits),
    }


def build_training_vocabulary(
    records: Sequence[SpectrumRecord],
    assignments: dict[str, str],
    *,
    min_df: int,
    min_cf: int,
    rm_top: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Construct the only allowed vocabulary from training documents."""
    document_frequency: Counter[str] = Counter()
    corpus_frequency: Counter[str] = Counter()
    train_documents = 0
    for record in records:
        if assignments[record.spectrum_id] != "train":
            continue
        train_documents += 1
        words = record.words
        document_frequency.update(set(words))
        corpus_frequency.update(words)
    eligible = {
        word
        for word, count in corpus_frequency.items()
        if count >= min_cf and document_frequency[word] >= min_df
    }
    if rm_top:
        removed = {
            word
            for word, _ in sorted(
                ((word, corpus_frequency[word]) for word in eligible),
                key=lambda item: (-item[1], item[0]),
            )[:rm_top]
        }
        eligible -= removed
    # Preserve MS2LDA's historical first-seen column order while deriving the
    # eligible set exclusively from training spectra.
    vocabulary_list: list[str] = []
    seen: set[str] = set()
    for record in records:
        if assignments[record.spectrum_id] != "train":
            continue
        for word in record.words:
            if word in eligible and word not in seen:
                seen.add(word)
                vocabulary_list.append(word)
    vocabulary = tuple(vocabulary_list)
    if not vocabulary:
        raise ValueError("training vocabulary is empty")
    return vocabulary, {
        "source_split": "train",
        "training_documents": train_documents,
        "vocabulary_size": len(vocabulary),
        "min_df": min_df,
        "min_cf": min_cf,
        "rm_top": rm_top,
        "order": "raw_training_spectra_first_seen",
    }


def filtered_words(record: SpectrumRecord, vocabulary: set[str]) -> list[str]:
    """Return one document using the frozen training vocabulary only."""
    return [word for word in record.words if word in vocabulary]


def completion_document(
    record: SpectrumRecord,
    *,
    observed_fraction: float,
    seed: int,
) -> CompletionDocument:
    """Split physical peak groups deterministically without fragment/loss leakage."""
    if len(record.peak_groups) < 2:
        raise ValueError("document completion requires at least two retained peaks")
    ranked = sorted(
        record.peak_groups,
        key=lambda group: _stable_digest(
            seed, f"{record.spectrum_id}\0{group.original_index}"
        ),
    )
    observed_count = int(round(len(ranked) * observed_fraction))
    observed_count = min(max(observed_count, 1), len(ranked) - 1)
    observed_ids = {group.original_index for group in ranked[:observed_count]}
    observed = tuple(
        group for group in record.peak_groups if group.original_index in observed_ids
    )
    completion = tuple(
        group
        for group in record.peak_groups
        if group.original_index not in observed_ids
    )
    return CompletionDocument(record.spectrum_id, observed, completion)


def renormalize_peak_groups(
    groups: Sequence[PeakGroup],
    *,
    precursor_mz: float,
    significant_digits: int,
) -> tuple[PeakGroup, ...]:
    """Rebuild observed counts without using a held-out intensity maximum."""
    if not groups:
        raise ValueError("cannot normalize an empty peak-group sequence")
    maximum = max(group.intensity for group in groups)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("observed peak groups require positive intensity")
    normalized = []
    for group in groups:
        intensity = group.intensity / maximum
        repetitions = max(int(np.rint(intensity * 100.0)), 1)
        tokens = [_spectral_word("frag", group.mz, significant_digits)] * repetitions
        loss = precursor_mz - group.mz
        if loss > 0.01:
            tokens.extend(
                [_spectral_word("loss", loss, significant_digits)] * repetitions
            )
        normalized.append(
            replace(group, intensity=float(intensity), tokens=tuple(tokens))
        )
    return tuple(normalized)


def split_records(
    records: Sequence[SpectrumRecord], assignments: dict[str, str], split: str
) -> list[SpectrumRecord]:
    """Select records from one named frozen partition in input order."""
    return [record for record in records if assignments[record.spectrum_id] == split]
