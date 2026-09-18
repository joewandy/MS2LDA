"""Regenerate manuscript tables and figures from verified assessment evidence."""

import argparse
from pathlib import Path

from benchmarks.neural_ms2lda.chemical_assessment_report import generate_chemical_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("research/motif_chemical_assessment_20260908/evidence"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/research/generated"))
    parser.add_argument("--figures", type=Path, default=Path("docs/research/figures"))
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    generate_chemical_report(args.evidence, args.output)
    if not args.tables_only:
        from benchmarks.neural_ms2lda.chemical_assessment_figures import (
            generate_figures,
        )

        generate_figures(args.evidence, args.figures)


if __name__ == "__main__":
    main()
