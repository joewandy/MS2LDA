"""Generate and seal one synthetic MS/MS dataset and its SGNS table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.neural_ms2lda.reproducibility import read_json_object
from benchmarks.neural_ms2lda.synthetic_msms import prepare_synthetic_seed

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a single seed before any ablation model is fitted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--training-documents", type=int, default=800)
    parser.add_argument("--validation-documents", type=int, default=160)
    arguments = parser.parse_args(argv)
    output_root = arguments.output_root.expanduser().resolve()
    dataset, embeddings, seed_directory = prepare_synthetic_seed(
        output_root,
        seed=arguments.seed,
        threads=arguments.threads,
        training_documents=arguments.training_documents,
        validation_documents=arguments.validation_documents,
    )
    result = {
        "seed": arguments.seed,
        "dataset": dataset.summary,
        "embedding_shape": list(embeddings.shape),
        "dataset_manifest": read_json_object(
            seed_directory / "artifact_manifest.json",
        ),
        "sgns_manifest": read_json_object(
            seed_directory / "token_features/complete.json",
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
