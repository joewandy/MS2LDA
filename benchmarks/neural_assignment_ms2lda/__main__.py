"""Module entry point for the neural-assignment MS2LDA benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_process_threads() -> None:
    """Set the frozen CPU allowance before importing numerical libraries.

    PyTorch can change its own pool later, but BLAS libraries commonly read
    their limits at import time. Reading the JSON with the standard library
    here keeps the documented ``python -m`` entry point equivalent to the
    unattended shell wrapper.
    """
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    count = str(int(protocol["cpu_threads"]))
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = count


def _main() -> int:
    """Configure the process, then import and dispatch the lightweight CLI."""
    _configure_process_threads()
    from .cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
