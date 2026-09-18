"""Run the frozen, research-only chemical assessment of saved topic models.

First use --seal to record the protocol, input bytes and dependency versions.
Then choose --stage specificity, stability, references, or all. No stage trains
a model, changes a MAG annotation, or reads the independent test partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.neural_ms2lda.chemical_assessment_io import seal_inputs, verify_seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "specificity", "stability", "references"),
        default="all",
    )
    args = parser.parse_args()
    if args.seal:
        seal_inputs(args.protocol, args.output)
        return
    protocol = verify_seal(args.protocol, args.output)
    # Heavy dependencies are not loaded for sealing / hash verification.
    from benchmarks.neural_ms2lda.chemical_assessment import run_assessment

    run_assessment(protocol, args.output, args.stage)


if __name__ == "__main__":
    main()
