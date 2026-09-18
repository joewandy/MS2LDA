"""Audit full replay, coverage denominators and correspondence invariants.

The independent coverage check uses sorted-score search, rather than the
runner's threshold comparison matrix. Existing Hungarian matches give another
oracle: a free nearest neighbour cannot score below an available forced mate.
No validation command changes a protocol, seal, fitted model or evidence file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.neural_ms2lda.cross_model_overlap_evidence import read_overlap
from benchmarks.neural_ms2lda.utils import sha256_file, write_json


def check_pair_curves(data: dict, rows: np.ndarray) -> dict:
    """Check every source denominator and curve using an independent algorithm."""
    fits = {f["id"]: i for i, f in enumerate(data["inventory"]["fits"])}
    t = np.array(data["summary"]["thresholds"])
    errors, total_rows = [], 0
    for pair in data["summary"]["pairs"]:
        source, target = fits[pair["source_fit"]], fits[pair["target_fit"]]
        cohort = data["protocol"]["cohorts"].index(pair["cohort"])
        metric = data["protocol"]["similarities"].index(pair["similarity"])
        subset = rows[
            (rows["source_fit"] == source)
            & (rows["target_fit"] == target)
            & (rows["cohort"] == cohort)
            & (rows["similarity"] == metric)
        ]
        expected_n = data["inventory"]["cohort_sizes"][pair["source_fit"]][
            pair["cohort"]
        ]
        if len(subset) != expected_n or len(set(subset["source_topic"])) != expected_n:
            raise ValueError("source cohort dropped or duplicated in directed matches")
        best = np.sort(subset["best_score"])
        second = np.sort(subset["second_score"][np.isfinite(subset["second_score"])])
        c = (len(best) - np.searchsorted(best, t, side="left")) / expected_n
        m = (len(second) - np.searchsorted(second, t, side="left")) / expected_n
        for name, values in (
            ("coverage", c),
            ("multiple", m),
            ("single", c - m),
            ("unmatched", 1 - c),
        ):
            errors.append(float(np.max(abs(values - pair["curves"][name]))))
        total_rows += len(subset)
    if max(errors) > 1e-14 or total_rows != len(rows):
        raise ValueError("saved coverage or complete pair inventory differs")
    return {
        "checked_directed_rows": total_rows,
        "fit_pair_cohort_metric_groups": len(errors) // 4,
        "maximum_coverage_error": max(errors),
    }


def check_counterpart_invariants(data: dict, rows: np.ndarray) -> dict:
    """Check reciprocal flags and both directions against saved forced matches."""
    lookup = {
        (
            int(r["source_fit"]),
            int(r["target_fit"]),
            int(r["cohort"]),
            int(r["similarity"]),
            int(r["source_topic"]),
        ): r
        for r in rows
    }
    for row in rows:
        reverse = lookup[
            (
                int(row["target_fit"]),
                int(row["source_fit"]),
                int(row["cohort"]),
                int(row["similarity"]),
                int(row["target_topic"]),
            )
        ]
        if bool(row["reciprocal"]) != (reverse["target_topic"] == row["source_topic"]):
            raise ValueError("reciprocal nearest-neighbour flag differs")
    fits = {f["id"]: i for i, f in enumerate(data["inventory"]["fits"])}
    previous = Path(data["protocol"]["chemical_evidence"]) / "topic_matches.jsonl"
    checked = {"all": 0, "recurring_evaluable": 0}
    for line in previous.read_text().splitlines():
        row = json.loads(line)
        a, b = fits[row["left_fit"]], fits[row["right_fit"]]
        k, ell = row["left_topic"], row["right_topic"]
        for i, j, x, y in ((a, b, k, ell), (b, a, ell, k)):
            for cohort, name in enumerate(("all", "recurring_evaluable")):
                key, reverse_key = (i, j, cohort, 0, x), (j, i, cohort, 0, y)
                if key not in lookup or reverse_key not in lookup:
                    continue
                if lookup[key]["best_score"] + 1e-12 < row["beta_cosine"]:
                    raise ValueError(
                        "free nearest neighbour scores below an eligible forced match"
                    )
                checked[name] += 1
    return {
        "reciprocal_flags_checked": len(rows),
        "nearest_ge_previous_forced_comparisons": checked,
        "previous_matches_sha256": sha256_file(previous),
    }


def validate(evidence: Path, replay: Path) -> dict:
    data, other = read_overlap(evidence), read_overlap(replay)
    names = ("directed_matches.npy", "summary.json", "inventory.json", "examples.json")
    for name in names:
        if sha256_file(evidence / name) != sha256_file(replay / name):
            raise ValueError(f"full replay is not byte-identical: {name}")
    if data != other:
        raise ValueError("full replay changed scientific summaries")
    rows = np.load(evidence / "directed_matches.npy", allow_pickle=False)
    if len(rows) != data["inventory"]["directed_match_rows"]:
        raise ValueError("row inventory changed")
    source_paths = list(
        Path("benchmarks/neural_ms2lda").glob("cross_model_overlap*.py")
    )
    source_paths += list(Path("scripts").glob("*cross_model_overlap*.py"))
    return {
        "byte_identical_scientific_payloads": list(names),
        "pair_curves": check_pair_curves(data, rows),
        "counterpart_invariants": check_counterpart_invariants(data, rows),
        "validated_source_sha256": {
            str(p): sha256_file(p) for p in sorted(source_paths)
        },
        "input_seals": {
            "original": sha256_file(evidence / "input_seal.json"),
            "replay": sha256_file(replay / "input_seal.json"),
        },
        "interpretation": (
            "Descriptive cross-model recovery on fixed development-validation fits; "
            "no chemical-equivalence test, retraining or model selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, validate(args.evidence, args.replay))
    print("Full replay, all coverage denominators and counterpart invariants passed.")


if __name__ == "__main__":
    main()
