"""Small, standard-library-only summaries for the current paper's comparison.

Each input row describes one completed fit, not one spectrum or one motif.
We average fit-level statistics and use sample SD (denominator n - 1).
In particular, median_effective_topics is already a median over spectra;
its reported mean and SD describe variation of that median across fits.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from .review_tables import number, read_rows
from .utils import sha256_file

BASELINE_SEEDS = {
    "Tomotopy LDA": (11, 23, 42),
    "canonical ETM": (7012, 7024, 7043),
}
BASELINE_LABELS = {"Tomotopy LDA": "Tomotopy LDA", "canonical ETM": "Plain ETM"}
COMPARISON_FIELDS = (
    ("evaluable_motifs", 1),
    ("useful_motifs", 1),
    ("mean_sos", 3),
    ("completion_nll", 3),
    ("median_effective_topics", 2),
)


def read_baseline_evidence(directory: Path) -> list[dict]:
    """Read the portable evidence only after checking its complete hash manifest.

    Report generation needs no ML dependencies or local training artifacts. The
    packager validates those artifacts first; this boundary verifies the public
    copies, their fit identities, and their declared validation-only scope.
    """
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("reported_split") != "validation"
        or manifest.get("historical_tomotopy_included") is not False
        or manifest.get("selected_model_refitted") is not False
    ):
        raise ValueError("baseline manifest has a different comparison scope")
    sources = manifest["sources"]
    paths = [record["path"] for record in sources]
    required = {
        "runs.csv",
        "summary.json",
        "protocol.json",
        "execution_source_manifest.json",
        "execution_source.tar.gz",
        *(
            f"{record['run']}/{name}"
            for record in manifest["records"]
            for name in (
                "result.json",
                "chemical.json",
                "validation_input_manifest.json",
            )
        ),
    }
    if not required.issubset(paths) or len(paths) != len(set(paths)):
        raise ValueError(
            "baseline sources must uniquely include core and per-fit evidence"
        )
    for record in sources:
        relative = Path(record["path"])
        path = (directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(directory):
            raise ValueError("baseline source must stay within its evidence directory")
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"baseline source integrity mismatch: {relative}")
    rows = read_rows(directory / "runs.csv")

    def identities(records):
        return Counter(
            (row["model"], number(row, "training_seed"), row["run"]) for row in records
        )

    if identities(rows) != identities(manifest["records"]):
        raise ValueError("baseline manifest and table identify different fits")
    return rows


def baseline_summaries(rows: list[dict]) -> dict[str, dict]:
    """Require the six declared baseline fits before producing uncertainty.

    These checks guard the public report inventory. The evidence packager also
    verifies raw artifacts, scientific input identities and fixed model recipes.
    No partial inventory, synthetic run or historical fourth Tomotopy fit can
    silently contribute to the current comparison.
    """
    expected = Counter(
        (model, seed) for model, seeds in BASELINE_SEEDS.items() for seed in seeds
    )
    actual = Counter((row["model"], number(row, "training_seed")) for row in rows)
    if actual != expected:
        raise ValueError("comparison needs exactly three declared fits per baseline")
    if len({row["run"] for row in rows}) != len(rows) or any(
        not row["run"] for row in rows
    ):
        raise ValueError("baseline runs must have distinct nonempty identities")
    for row in rows:
        for field, expected_value in (
            ("spectrum_topic_associations", 3889),
            ("completion_documents", 3888),
            ("completion_tokens", 1277983),
        ):
            if number(row, field) != expected_value:
                raise ValueError(f"baseline validation population differs: {field}")
        optimized, evaluable, useful = (
            number(row, key)
            for key in ("optimized_motifs", "evaluable_motifs", "useful_motifs")
        )
        if not (
            0 <= useful <= evaluable <= optimized <= 1000
            and all(value.is_integer() for value in (optimized, evaluable, useful))
        ):
            raise ValueError("baseline motif counts are inconsistent")
        if evaluable == 0:
            raise ValueError("mean SOS is undefined without an evaluable motif")
        if not 0 <= number(row, "mean_sos") <= 1:
            raise ValueError("baseline mean SOS must be in [0, 1]")
        if number(row, "completion_nll") < 0:
            raise ValueError("baseline completion NLL must be nonnegative")
        if not 1 <= number(row, "median_effective_topics") <= 1000:
            raise ValueError("baseline effective topic count must be in [1, K]")
    summaries = {}
    for model, seeds in BASELINE_SEEDS.items():
        group = [row for row in rows if row["model"] == model]
        summaries[model] = {
            "fits": 3,
            "training_seeds": list(seeds),
            **{
                field: {
                    "mean": mean(number(row, field) for row in group),
                    "sample_sd": stdev(number(row, field) for row in group),
                }
                for field, _ in COMPARISON_FIELDS
            },
        }
    return summaries


def comparison_cell(stat: dict, precision: int, *, highlight: bool = False) -> str:
    """Format mean +/- sample SD; bold is descriptive, not a significance test."""
    value = (
        f"{stat['mean']:.{precision}f}" + r"\pm" + f"{stat['sample_sd']:.{precision}f}"
    )
    return "$" + (r"\boldsymbol{" + value + "}" if highlight else value) + "$"


def comparison_table(baselines: dict[str, dict], selected: dict) -> list[str]:
    """Highlight observed extrema only for metrics with a stated direction.

    Effective topic count is descriptive: a smaller value is not necessarily
    better. Fit counts and this sparsity diagnostic are never ranked. Exact ties
    at full numeric precision receive the same emphasis before display rounding.
    """
    entries = [(BASELINE_LABELS[name], baselines[name]) for name in BASELINE_SEEDS]
    entries.append(("Contextual Sparse ETM", selected))
    best = {
        field: select(stat[field]["mean"] for _, stat in entries)
        for field, select in (
            ("evaluable_motifs", max),
            ("useful_motifs", max),
            ("mean_sos", max),
            ("completion_nll", min),
        )
    }
    lines = []
    for label, stats in entries:
        if label == "Contextual Sparse ETM":
            lines.append(r"\midrule")
        cells = [label, "3"]
        cells.extend(
            comparison_cell(
                stats[field],
                precision,
                highlight=field in best and stats[field]["mean"] == best[field],
            )
            for field, precision in COMPARISON_FIELDS
        )
        lines.append(" & ".join(cells) + r" \\")
    return lines
