"""Generate the validation-only model-selection tables used by the report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO
    / "research/etm_ecrtm_msnlib/local_results/20260827_followup/comparison.csv"
)
GENERATED = REPO / "docs/research/generated"

ORDER = (
    "m1_reference",
    "etm",
    "pooled_likelihood",
    "pooled_likelihood_tau011",
    "pooled_mi005",
    "etm_balanced",
    "ecrtm_canonical",
)

LABELS = {
    "m1_reference": "M1 locked reference",
    "etm": "Canonical fixed-SGNS ETM",
    "pooled_likelihood": "Pooled projected",
    "pooled_likelihood_tau011": r"Pooled projected, $\tau=0.11$",
    "pooled_mi005": "Pooled projected + MI 0.05",
    "etm_balanced": "Fragment/loss-balanced ETM",
    "ecrtm_canonical": "Canonical ECRTM",
}

OUTCOMES = {
    "m1_reference": "Pass",
    "etm": "Fail chemistry",
    "pooled_likelihood": "Fail breadth/collapse",
    "pooled_likelihood_tau011": "Fail breadth/collapse",
    "pooled_mi005": "Fail breadth/collapse",
    "etm_balanced": "Fail breadth/SOS",
    "ecrtm_canonical": "Infeasible at epoch 22",
}


def _read() -> dict[str, dict[str, str]]:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    missing = set(ORDER) - set(rows)
    if missing:
        raise ValueError(f"model-selection comparison is missing: {sorted(missing)}")
    if rows["m1_reference"]["passed_all_frozen_gates"] != "True":
        raise ValueError("locked M1 reference must pass the frozen gates")
    for method in ORDER[1:6]:
        if rows[method]["passed_all_frozen_gates"] != "False":
            raise ValueError(f"unexpected gate result for {method}")
    if rows["ecrtm_canonical"]["execution_status"] != "failed_sinkhorn_nonconvergence":
        raise ValueError("canonical ECRTM failure status changed")
    return rows


def _value(row: dict[str, str], key: str, digits: int = 6) -> str:
    value = row.get(key, "")
    if not value:
        return "--"
    return f"{float(value):.{digits}f}"


def _integer(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return str(int(float(value))) if value else "--"


def _write_primary(rows: dict[str, dict[str, str]]) -> None:
    lines = []
    for method in ORDER:
        row = rows[method]
        lines.append(
            " & ".join(
                (
                    LABELS[method],
                    _integer(row, "optimized_motifs"),
                    _integer(row, "evaluable_motifs"),
                    _integer(row, "useful_motifs"),
                    _value(row, "mean_sos"),
                    _value(row, "completion_nll"),
                    OUTCOMES[method],
                )
            )
            + r" \\"
        )
    (GENERATED / "model_selection_validation_table.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n",
        encoding="utf-8",
    )


def _write_diagnostics(rows: dict[str, dict[str, str]]) -> None:
    methods = (
        "etm",
        "pooled_likelihood",
        "pooled_likelihood_tau011",
        "etm_balanced",
    )
    lines = []
    for method in methods:
        row = rows[method]
        lines.append(
            " & ".join(
                (
                    LABELS[method],
                    _value(row, "median_effective_topics_per_spectrum", 2),
                    _integer(row, "unique_top1_topics"),
                    _integer(row, "largest_beta_component_cosine_ge_0_999"),
                    _value(row, "mean_nearest_topic_beta_cosine", 3),
                    _value(row, "maximum_pairwise_beta_cosine", 4),
                )
            )
            + r" \\"
        )
    (GENERATED / "model_selection_diagnostics_table.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n",
        encoding="utf-8",
    )


def generate() -> dict[str, Any]:
    """Validate the frozen evidence and write both report fragments."""
    rows = _read()
    GENERATED.mkdir(parents=True, exist_ok=True)
    _write_primary(rows)
    _write_diagnostics(rows)
    return {
        "source": str(SOURCE.relative_to(REPO)),
        "methods": list(ORDER),
        "outputs": [
            "docs/research/generated/model_selection_validation_table.tex",
            "docs/research/generated/model_selection_diagnostics_table.tex",
        ],
    }


if __name__ == "__main__":
    print(generate())
