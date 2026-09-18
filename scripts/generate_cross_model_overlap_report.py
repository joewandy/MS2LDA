"""Generate overlap manuscript fragments and static figures from sealed evidence."""

import argparse
from pathlib import Path

from benchmarks.neural_ms2lda.cross_model_overlap_report import generate_overlap_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("research/cross_model_overlap_20260908/evidence"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/research/generated"))
    parser.add_argument("--figures", type=Path, default=Path("docs/research/figures"))
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    generate_overlap_report(args.evidence, args.output)
    if not args.tables_only:
        from benchmarks.neural_ms2lda.cross_model_overlap_figures import (
            generate_overlap_figures,
        )

        generate_overlap_figures(args.evidence, args.figures)


if __name__ == "__main__":
    main()
