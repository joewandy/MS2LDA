"""Small deterministic fixtures shared by the focused neural tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from benchmarks.neural_ms2lda.artifacts import load_protocol
from benchmarks.neural_ms2lda.spectra import PeakGroup, SpectrumRecord
from benchmarks.neural_ms2lda.utils import file_sha256


def spectrum_record(identifier: str, words: list[str]) -> SpectrumRecord:
    """Build a minimal spectrum record whose peak groups each hold one token."""
    groups = tuple(
        PeakGroup(index, 100.0 + index, 1.0, (word,))
        for index, word in enumerate(words)
    )
    return SpectrumRecord(
        spectrum_id=identifier,
        feature_id=identifier,
        smiles="CCO",
        supplied_inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        connectivity_key=identifier,
        scaffold_key="",
        split_group=identifier,
        precursor_mz=300.0,
        peak_groups=groups,
        declared_num_peaks=len(groups),
        parsed_num_peaks=len(groups),
        compound_name=identifier,
        metadata={},
    )


def token_features(
    tokens: int,
    dimensions: int = 64,
    *,
    fragments: int | None = None,
) -> torch.Tensor:
    """Return normalized features with explicit fragment/loss indicator columns."""
    generator = torch.Generator().manual_seed(42 + tokens + dimensions)
    features = torch.randn(tokens, dimensions, generator=generator)
    features[:, -2:] = 0.0
    fragment_count = tokens // 2 if fragments is None else fragments
    features[:fragment_count, -2] = 1.0
    features[fragment_count:, -1] = 1.0
    return torch.nn.functional.normalize(features, dim=1)


def mini_protocol(mgf: Path) -> dict[str, Any]:
    """Shrink only capacities and iteration counts for a fast end-to-end test."""
    protocol = copy.deepcopy(load_protocol())
    protocol["input_files"] = {
        "mgf": {
            "relative_path": mgf.name,
            "bytes": mgf.stat().st_size,
            "sha256": file_sha256(mgf),
        }
    }
    protocol["preprocessing"].update(
        {
            "expected_spectra": 18,
            "min_fragments": 3,
            "min_df": 1,
            "split_fractions": [0.6, 0.2, 0.2],
        }
    )
    protocol["sgns"].update(
        {
            "dimensions": 4,
            "epochs": 1,
            "positive_pairs_per_document": 2,
            "batch_size": 32,
        }
    )
    protocol["token_features"]["fourier_frequencies"] = [1]
    protocol["model"].update(
        {
            "num_topics": 4,
            "projection_dimensions": 8,
            "router_hidden_dimensions": 8,
            "sinkhorn_iterations": 10,
        }
    )
    protocol["views"]["pairs"] = 2
    protocol["optimization"].update(
        {
            "batch_size": 4,
            "topic_update_batch_size": 4,
            "topic_updates_per_epoch": 1,
            "maximum_epochs": 2,
            "validation_interval": 1,
        }
    )
    protocol["anti_collapse"].update(
        {
            "routing_temperature_anneal_epochs": 2,
            "sinkhorn_weight_hold_epochs": 0,
            "sinkhorn_weight_end_epoch": 2,
            "recycle_patience_validations": 10,
            "recycle_through_epoch": 2,
        }
    )
    protocol["cooccurrence_regularization"].update(
        {
            "minimum_document_frequency": 1,
            "minimum_pair_frequency": 1,
            "maximum_neighbors": 2,
        }
    )
    protocol["topic_separation"]["neighbors"] = 2
    protocol["evaluation"].update({"latency_subset_size": 2, "latency_repeats": 1})
    return protocol


def write_mini_mgf(path: Path) -> None:
    """Write 18 chemically distinct spectra suitable for scaffold splitting."""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    smiles_values = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CCF",
        "COC",
        "CNC",
        "CCS",
        "CC=O",
        "CC#N",
        "C=CO",
        "C1CC1",
        "C1CCC1",
        "c1ccccc1",
        "c1ccncc1",
        "O=C=O",
        "N#N",
    ]
    blocks = []
    for index, smiles in enumerate(smiles_values):
        inchikey = MolToInchiKey(Chem.MolFromSmiles(smiles))
        peaks = [
            f"{50 + offset * 10 + index % 3}.0 {100 - offset * 10}.0"
            for offset in range(5)
        ]
        blocks.append(
            "\n".join(
                [
                    "BEGIN IONS",
                    f"USI=mini:{index}",
                    f"FEATURE_ID=feature:{index}",
                    f"SMILES={smiles}",
                    f"INCHIKEY={inchikey}",
                    "PRECURSOR_MZ=250.0",
                    "NUM_PEAKS=5",
                    *peaks,
                    "END IONS",
                ]
            )
        )
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def chemistry_result(topics: int = 4) -> dict[str, Any]:
    """Return the smallest valid paper-facing chemistry manifest."""
    return {
        "topics": topics,
        "annotation_coverage": 0.5,
        "high_confidence_chemistry": {
            "eligible_topics": 2,
            "associated_spectra": 2,
            "mean_sos": 0.7,
            "median_sos": 0.7,
            "sos_bands": {
                "high_gt_0_8": 0,
                "intermediate_0_6_to_0_8": 1,
                "low_lt_0_6": 1,
            },
        },
    }
