"""Generate validation-only simplification tables and explicit decision gates.

This generator is separate from the sealed historical test-report generator.
It consumes the complete compact run inventories, including negative results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from benchmarks.neural_ms2lda.review_tables import (
    FROZEN,
    REVIEW,
    ROOT,
    boolean,
    number,
    read_rows,
    real_gate,
    synthetic_gate,
    validation_line,
)
from benchmarks.neural_ms2lda.utils import sha256_file, write_json

LABELS = {
    "contextual": "Contextual (paired)",
    "batchnorm_entmax": "BN--entmax, learned scale",
    "batchnorm_fixed_entmax": "BN--entmax, fixed scale",
    "contextual_unbalanced": "Contextual, no channel balance",
    "batchnorm": "BN, learned scale",
    "prior_batchnorm": "BN + prior, learned scale",
    "batchnorm_fixed": "BN, fixed scale",
    "prior_batchnorm_fixed": "BN + prior, fixed scale",
}
SYNTHETIC_VARIANTS = (
    "contextual",
    "batchnorm_entmax",
    "batchnorm_fixed_entmax",
    "contextual_unbalanced",
)
PRIOR_VARIANTS = (
    "batchnorm",
    "prior_batchnorm",
    "batchnorm_fixed",
    "prior_batchnorm_fixed",
)


def synthetic_groups(rows: list[dict]) -> dict:
    """Require the complete paired 4-model x 2-K x 3-seed experiment."""
    groups = defaultdict(list)
    for row in rows:
        if row["data"] != "synthetic":
            continue
        if (
            number(row, "epochs") != 120
            or number(row, "batch_size") != 200
            or number(row, "momentum") != 0.9
            or row["count_scaling"] != "raw_counts"
            or boolean(row, "reused_training_weights")
        ):
            raise ValueError("synthetic recipe differs from the paired protocol")
        groups[(int(row["topics"]), row["variant"])].append(row)
    expected = {(k, v) for k in (36, 128) for v in SYNTHETIC_VARIANTS}
    if set(groups) != expected:
        raise ValueError("missing or unexpected synthetic comparison group")
    for values in groups.values():
        if sorted(int(row["seed"]) for row in values) != [11, 23, 37]:
            raise ValueError("synthetic seeds must be exactly 11, 23, 37")
    return dict(groups)


def generate(evidence: Path, frozen: Path, output: Path) -> dict:
    rows = read_rows(evidence / "runs.csv")
    groups = synthetic_groups(rows)
    gates, synthetic_table = {}, []
    fields = (
        "beta_recovery",
        "theta_recovery",
        "recovered_motifs",
        "completion_nll",
        "median_effective_topics",
        "nearest_beta_cosine",
    )
    group_summaries = []
    for k in (36, 128):
        for variant in SYNTHETIC_VARIANTS:
            group = groups[k, variant]
            summary = {
                key: {
                    "mean": mean(number(r, key) for r in group),
                    "sample_sd": stdev(number(r, key) for r in group),
                }
                for key in fields
            }
            group_summaries.append({"topics": k, "variant": variant, **summary})
            cells = [str(k), LABELS[variant]]
            for field in fields:
                precision = (
                    2 if field in {"recovered_motifs", "median_effective_topics"} else 3
                )
                cells.append(f"{summary[field]['mean']:.{precision}f}")
            synthetic_table.append(" & ".join(cells) + r" \\")
            gates[f"{k}_{variant}"] = synthetic_gate(group, groups[k, "contextual"], k)
        if k == 36:
            synthetic_table.append(r"\midrule")

    frozen_rows = {row["model"]: row for row in read_rows(frozen)}
    prior_path = evidence / "prior_round/runs.csv"
    prior_rows = [
        r
        for r in read_rows(prior_path)
        if r["data"] == "MSnLib validation"
        and r["normalization_statistics"] == "training"
    ]
    if sorted(r["variant"] for r in prior_rows) != sorted(PRIOR_VARIANTS):
        raise ValueError("prior-round table needs all four calibrated real models")
    prior_lookup = {row["variant"]: row for row in prior_rows}
    prior_table = [validation_line(LABELS[v], prior_lookup[v]) for v in PRIOR_VARIANTS]
    real = [r for r in rows if r["data"] == "MSnLib validation"]
    initial = {r["variant"]: r for r in real if r["seed"] == "42"}
    required = {"contextual", "contextual_unbalanced", "batchnorm_fixed_entmax"}
    if set(initial) != required or len(real) != len(initial):
        raise ValueError("initial real review needs one matched run for each model")
    for row in real:
        if (
            number(row, "batch_size") != 200
            or number(row, "training_seed") != 7043
            or number(row, "topics") != 1000
            or number(row, "epochs") != 120
            or number(row, "momentum") != 0.9
            or row["count_scaling"] != "raw_counts"
            or row["normalization_statistics"]
            != (
                "training"
                if row["variant"] == "batchnorm_fixed_entmax"
                else "not_applicable"
            )
            or boolean(row, "reused_training_weights")
        ):
            raise ValueError("real recipe differs from the paired protocol")
        for field in ("evaluable_motifs", "useful_motifs", "mean_sos"):
            number(row, field)  # Chemical scoring is required even for weak runs.
    reference = frozen_rows["Contextual Sparse ETM"]
    validation_table = [
        validation_line("ETM (historical)", frozen_rows["canonical ETM"]),
        validation_line("Balanced ETM (historical)", frozen_rows["balanced ETM"]),
        validation_line("Contextual (historical)", reference),
        validation_line("Tomotopy (historical)", frozen_rows["Tomotopy LDA"]),
        r"\midrule",
        validation_line(LABELS["contextual"], initial["contextual"]),
        validation_line(
            LABELS["batchnorm_fixed_entmax"], initial["batchnorm_fixed_entmax"]
        ),
        validation_line(
            LABELS["contextual_unbalanced"], initial["contextual_unbalanced"]
        ),
    ]
    real_gates = {
        v: real_gate(initial[v], [reference, initial["contextual"]])
        for v in ("batchnorm_fixed_entmax", "contextual_unbalanced")
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, lines in (
        ("synthetic", synthetic_table),
        ("prior", prior_table),
        ("validation", validation_table),
    ):
        (output / f"minimal_etm_review_{name}_table.tex").write_text(
            "% Generated by scripts.generate_minimal_etm_review; validation only.\n"
            + "\n".join([*lines, r"\bottomrule"])
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "stage": "validation development after historical test results were available",
        "new_training_runs": len(rows),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in (
                evidence / "runs.csv",
                evidence / "manifest.json",
                prior_path,
                prior_path.with_name("manifest.json"),
                frozen,
                Path(__file__),
                ROOT / "benchmarks/neural_ms2lda/review_tables.py",
                ROOT / "benchmarks/neural_ms2lda/utils.py",
            )
        },
        "synthetic_groups": group_summaries,
        "synthetic_gates": gates,
        "initial_real_gates": real_gates,
        "replacement_confirmed": False,
        "confirmation_rule": "Any passing candidate still needs two further real seeds",
    }
    write_json(evidence / "review_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=REVIEW)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/generated")
    args = parser.parse_args()
    generate(args.evidence.resolve(), args.frozen.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
