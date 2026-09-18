"""Prepare the shared MSnLib split and train-only SGNS representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from benchmarks.neural_ms2lda.reproducibility import (
    configure_deterministic_execution,
)
from benchmarks.neural_ms2lda.study_protocol import initialize_run
from benchmarks.neural_ms2lda.utils import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence


def prepare_shared_inputs(run: Path, data_root: Path) -> dict[str, Any]:
    """Build the deterministic split and SGNS table used by every method."""
    protocol = initialize_run(run, data_root=data_root)
    configure_deterministic_execution(
        int(protocol["seed"]),
        int(protocol["cpu_threads"]),
    )
    data_result = prepare_data(run, data_root=data_root, protocol=protocol)
    train = load_csr(run / "data/train.npz")
    vocabulary = load_vocabulary(run / "data")
    feature_result = train_token_features(
        run / "token_features",
        train,
        vocabulary,
        protocol,
        seed=int(protocol["seed"]),
    )
    result = {
        "data": data_result,
        "token_features": feature_result,
        "train_shape": list(train.shape),
        "vocabulary_size": len(vocabulary),
        "evidence_boundary": "shared train and validation preparation",
    }
    write_json(run / "preparation_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one clean-room shared-input directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = prepare_shared_inputs(
        arguments.run.expanduser().resolve(),
        arguments.data_root.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
