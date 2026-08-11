"""Corrected leakage-controlled MAG scoring for neural-assignment artifacts."""

from benchmarks.fully_neural_ms2lda.chemical import (
    run_chemical_scoring as _run_chemical_scoring,
)

run_chemical_scoring = _run_chemical_scoring

__all__ = ("run_chemical_scoring",)
