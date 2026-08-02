# Copyright (c) WUR
# This software may be used and distributed in accordance with the terms of
# the MS2LDA Community License Agreement.

"""Public MS2LDA entry points.

The full workflow has optional UI, annotation, and plotting dependencies.
Loading it only when one of its functions is requested keeps lightweight
submodules (including the hybrid reference model) independently importable.
"""

from importlib import import_module

from .__version__ import __version__

__all__ = ["__version__", "run", "screen_spectra", "screen_structure"]


def __getattr__(name: str):
    """Load the established workflow functions on first access."""
    if name not in {"run", "screen_spectra", "screen_structure"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".run", __name__), name)
    globals()[name] = value
    return value
