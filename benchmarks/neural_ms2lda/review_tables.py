"""Dependency-free table formatting and archived screening criteria.

These gates describe historical decisions; the current paper uses descriptive
tradeoffs. Shared helpers are independent of every command-line generator.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "research/minimal_neural_etm/review_20260907/evidence"
FROZEN = (
    ROOT
    / "research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room"
    / "validation_comparison.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def boolean(row: dict, field: str) -> bool:
    value = str(row[field]).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid Boolean {field}")
    return value == "true"


def synthetic_gate(candidate: list[dict], current: list[dict], topics: int) -> dict:
    def average(rows: list[dict], key: str) -> float:
        return mean(number(row, key) for row in rows)

    checks = {
        "beta_recovery": average(candidate, "beta_recovery")
        >= average(current, "beta_recovery") - 0.05,
        "theta_recovery": average(candidate, "theta_recovery")
        >= average(current, "theta_recovery") - 0.10,
        "median_effective_topics": average(candidate, "median_effective_topics") <= 5,
        "no_catastrophic_duplicate": not any(
            boolean(r, "catastrophic_duplicate") for r in candidate
        ),
    }
    if topics == 128:
        recovered = average(candidate, "recovered_motifs")
        threshold = 0.9 * average(current, "recovered_motifs")
        # Equality at the predeclared 90% boundary must pass: 0.9*(50/3)
        # rounds to 15.000000000000002 in binary floating-point arithmetic.
        checks["recovered_motif_retention"] = recovered >= threshold or math.isclose(
            recovered, threshold, rel_tol=1e-12, abs_tol=1e-12
        )
    return {"passed": all(checks.values()), "checks": checks}


def real_gate(candidate: dict, references: list[dict]) -> dict:
    """Apply the recorded joint requirements; no favorable-metric substitution."""
    checks = {
        "sparse_mixtures": number(candidate, "median_effective_topics") <= 5,
        "broad_winner_inventory": number(candidate, "unique_top1_topics") >= 800,
        "no_catastrophic_duplicate": not boolean(candidate, "catastrophic_duplicate"),
    }
    for i, reference in enumerate(references):
        for field in ("evaluable_motifs", "useful_motifs"):
            checks[f"reference_{i}_{field}"] = number(candidate, field) >= (
                0.95 * number(reference, field)
            )
        checks[f"reference_{i}_mean_sos"] = number(candidate, "mean_sos") >= (
            number(reference, "mean_sos") - 0.02
        )
        checks[f"reference_{i}_completion_nll"] = number(
            candidate, "completion_nll"
        ) <= (1.02 * number(reference, "completion_nll"))
    return {"passed": all(checks.values()), "checks": checks}


def latex_number(row: dict, field: str, precision: int) -> str:
    return f"{number(row, field):.{precision}f}" if row.get(field) != "" else "--"


def validation_line(label: str, row: dict) -> str:
    fields = (
        ("evaluable_motifs", 0),
        ("useful_motifs", 0),
        ("mean_sos", 3),
        ("completion_nll", 3),
        ("median_effective_topics", 2),
        ("unique_top1_topics", 0),
    )
    return " & ".join([label, *(latex_number(row, f, p) for f, p in fields)]) + r" \\"
