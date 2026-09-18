"""Run sealed directed overlap assessment on existing neural/Tomotopy fits."""

import argparse
from pathlib import Path

from benchmarks.neural_ms2lda.cross_model_overlap_evidence import (
    run_overlap,
    seal_overlap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("research/cross_model_overlap_20260908/protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal_overlap(args.protocol, args.output)
    else:
        run_overlap(args.output)


if __name__ == "__main__":
    main()
