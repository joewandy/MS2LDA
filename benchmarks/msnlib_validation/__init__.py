"""Leakage-safe MSnLib validation for the HybridLDA reference model.

This package is benchmark-only.  It deliberately does not alter the production
MS2LDA backend selection or make HybridLDA available through ``MS2LDA.run``.
"""

from .config import BenchmarkConfig, load_config

__all__ = ["BenchmarkConfig", "load_config"]
