"""Recover frozen v8 scoring after an unnecessary annotation dependency import."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmarks.msnlib_simplification import chemical

_original_maccs_fingerprint = chemical._maccs_fingerprint  # noqa: SLF001
_original_consensus_fingerprint = chemical._consensus_fingerprint  # noqa: SLF001


@lru_cache(maxsize=None)
def _cached_maccs_fingerprint(smiles: str) -> np.ndarray | None:
    return _original_maccs_fingerprint(smiles)


@lru_cache(maxsize=None)
def _cached_consensus_fingerprint(
    smiles_values: tuple[str, ...],
    threshold: float,
) -> np.ndarray | None:
    return _original_consensus_fingerprint(smiles_values, threshold)


def _consensus_fingerprint(
    smiles_values: Sequence[str],
    threshold: float,
) -> np.ndarray | None:
    return _cached_consensus_fingerprint(tuple(smiles_values), threshold)


def _verify_existing_annotation(
    run_dir: str | Path,
    discovery: str,
) -> dict[str, object]:
    directory = Path(run_dir).expanduser().resolve()
    return chemical._verify_annotations(  # noqa: SLF001 - recovery verification
        chemical._annotation_output(directory, discovery),  # noqa: SLF001
        discovery,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: recover_chemical_scores.py RUN_DIR")
    chemical.annotate_discovery = _verify_existing_annotation
    chemical._maccs_fingerprint = _cached_maccs_fingerprint  # noqa: SLF001
    chemical._consensus_fingerprint = _consensus_fingerprint  # noqa: SLF001
    result = chemical.score_all_chemical_results(sys.argv[1])
    print(result)
    print(
        {
            "maccs_cache": _cached_maccs_fingerprint.cache_info()._asdict(),
            "consensus_cache": _cached_consensus_fingerprint.cache_info()._asdict(),
        },
    )
