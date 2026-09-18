"""Generate complete evidence tables for the independently expanded model screen."""

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
    "prodlda": "ProdLDA (Pyro-reference port)",
    "dvae_poe": "Direct Dirichlet, PoE",
    "dirichlet_lda": "Direct Dirichlet, additive LDA",
    "dirichlet_etm": "Direct Dirichlet, additive ETM",
    "prodlda_entmax": "ProdLDA + entmax",
}
FIELDS = (
    "beta_recovery",
    "theta_recovery",
    "recovered_motifs",
    "completion_nll",
    "median_effective_topics",
    "nearest_beta_cosine",
)


def validate_recipe(row):
    """Fail closed on an unreported model, recipe change, or incomplete score."""
    variant = row["variant"]
    if variant not in LABELS:
        raise ValueError("unknown published model")
    expected = {
        "epochs": 120,
        "batch_size": 200,
        "hidden": 100,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0,
        "kl_warmup_epochs": 0 if variant.startswith("prodlda") else 100,
    }
    if any(number(row, field) != value for field, value in expected.items()):
        raise ValueError("published model recipe mismatch")
    if not variant.startswith("prodlda") and number(row, "concentration") != 0.02:
        raise ValueError("Dirichlet prior mismatch")
    product = variant in ("prodlda", "prodlda_entmax", "dvae_poe")
    for field, expected_value in {
        "normalization_statistics": "ema",
        "count_scaling": "raw_counts",
        "framework": "pytorch_port_not_pyro_svi",
        "decoder": "product_of_experts" if product else "additive_mixture",
        "beta_meaning": "one_hot_conditional_prototypes" if product else "emissions",
    }.items():
        if row[field] != expected_value:
            raise ValueError(f"published evidence meaning mismatch: {field}")
    if boolean(row, "reused_training_weights"):
        raise ValueError("unexpected repeated inference instead of training")


def synthetic_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        validate_recipe(row)
        if row["data"] == "synthetic":
            groups[int(row["topics"]), row["variant"]].append(row)
    if set(groups) != {(k, v) for k in (36, 128) for v in LABELS}:
        raise ValueError("incomplete expanded synthetic screen")
    if any(
        sorted(int(r["seed"]) for r in group) != [11, 23, 37]
        for group in groups.values()
    ):
        raise ValueError("expanded synthetic screen needs exactly three paired seeds")
    return groups


def generate(evidence=REVIEW, output=ROOT / "docs/research/generated"):
    """Use every completed prototype, preserving negative and follow-on results."""
    published = evidence / "published"
    rows = read_rows(published / "runs.csv")
    controls = read_rows(evidence / "runs.csv")
    groups = synthetic_groups(rows)
    summaries, gates, lines = [], {}, []
    for k in (36, 128):
        current = [
            r
            for r in controls
            if r["data"] == "synthetic"
            and int(r["topics"]) == k
            and r["variant"] == "contextual"
        ]
        if sorted(int(r["seed"]) for r in current) != [11, 23, 37]:
            raise ValueError("paired current controls are missing")
        for variant in LABELS:
            group = groups[k, variant]
            summary = {
                key: {
                    "mean": mean(number(r, key) for r in group),
                    "sample_sd": stdev(number(r, key) for r in group),
                }
                for key in FIELDS
            }
            summaries.append({"topics": k, "variant": variant, **summary})
            gates[f"{k}_{variant}"] = synthetic_gate(group, current, k)
            cells = [str(k), LABELS[variant]]
            for key in FIELDS:
                precision = (
                    2 if key in ("recovered_motifs", "median_effective_topics") else 3
                )
                cells.append(f"{summary[key]['mean']:.{precision}f}")
            lines.append(" & ".join(cells) + r" \\")
        if k == 36:
            lines.append(r"\midrule")
    real = [r for r in rows if r["data"] == "MSnLib validation"]
    expected = {"prodlda"}
    if all(gates[f"{k}_prodlda_entmax"]["passed"] for k in (36, 128)):
        expected.add("prodlda_entmax")
    if sorted(r["variant"] for r in real) != sorted(expected):
        raise ValueError("missing or unexpected published real-validation run")
    reference = next(
        r for r in read_rows(FROZEN) if r["model"] == "Contextual Sparse ETM"
    )
    paired = next(
        r
        for r in controls
        if r["data"] == "MSnLib validation" and r["variant"] == "contextual"
    )
    real_lines = [validation_line("Contextual (paired)", paired)]
    real_gates = {}
    for row in sorted(real, key=lambda r: r["variant"]):
        if number(row, "topics") != 1000 or number(row, "training_seed") != 7043:
            raise ValueError("unexpected real topic count or seed")
        for field in ("evaluable_motifs", "useful_motifs", "mean_sos"):
            number(row, field)
        real_lines.append(validation_line(LABELS[row["variant"]], row))
        real_gates[row["variant"]] = real_gate(row, [reference, paired])
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("synthetic", lines), ("validation", real_lines)):
        (output / f"published_neural_review_{name}_table.tex").write_text(
            "% Generated: published neural review. Validation only.\n"
            + "\n".join([*contents, r"\bottomrule"])
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "stage": "expanded validation development after historical test reporting",
        "new_training_runs": len(rows),
        "synthetic_groups": summaries,
        "synthetic_gates": gates,
        "real_gates": real_gates,
        "replacement_confirmed": False,
        "confirmation_rule": "Passing candidates still require two further real seeds",
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                published / "runs.csv",
                published / "manifest.json",
                evidence / "runs.csv",
                evidence / "manifest.json",
                FROZEN,
                Path(__file__),
                ROOT / "benchmarks/neural_ms2lda/review_tables.py",
                ROOT / "benchmarks/neural_ms2lda/utils.py",
                ROOT / "scripts/generate_minimal_etm_review.py",
            )
        },
    }
    write_json(published / "review_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=REVIEW)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/generated")
    args = parser.parse_args()
    generate(args.evidence.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
