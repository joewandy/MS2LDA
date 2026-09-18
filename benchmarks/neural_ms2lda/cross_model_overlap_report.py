"""Small manuscript tables and macros from verified overlap evidence only."""

from __future__ import annotations

from pathlib import Path

from .chemical_assessment_report import cell
from .cross_model_overlap_evidence import read_overlap
from .utils import write_json

DIRECTIONS = {
    "selected_to_selected": ("NN", r"Neural $\to$ neural"),
    "selected_to_tomotopy": ("NL", r"Neural $\to$ Tomotopy"),
    "tomotopy_to_selected": ("LN", r"Tomotopy $\to$ neural"),
    "tomotopy_to_tomotopy": ("LL", r"Tomotopy $\to$ Tomotopy"),
}


def group(data: dict, direction: str, cohort: str, similarity="full_beta") -> dict:
    """Require one complete summary for the explicit comparison population."""
    matches = [
        g
        for g in data["summary"]["groups"]
        if g["direction"] == direction
        and g["cohort"] == cohort
        and g["similarity"] == similarity
    ]
    if len(matches) != 1 or matches[0]["source_fits"] != 3:
        raise ValueError("overlap table needs one complete three-source summary")
    if any(m["n"] != 3 for m in matches[0]["metrics"].values()):
        raise ValueError("overlap table cannot silently omit an unscorable source fit")
    return matches[0]["metrics"]


def generate_overlap_report(evidence: Path, output: Path) -> dict:
    """Emit compact aggregate evidence; do not hand-copy or rank descriptive data."""
    data = read_overlap(evidence)
    output.mkdir(parents=True, exist_ok=True)
    macros, table = [], []
    for direction, (prefix, label) in DIRECTIONS.items():
        all_topics = group(data, direction, "all")
        recurring = group(data, direction, "recurring_evaluable")
        # These are plotted fit-pair/topic records, not independent motifs.
        # Report the few undefined chemical profiles without dropping them
        # from the spectral recovery denominator.
        pairs = [
            p
            for p in data["summary"]["pairs"]
            if p["direction"] == direction
            and p["cohort"] == "recurring_evaluable"
            and p["similarity"] == "full_beta"
        ]
        for suffix, field in (
            ("ChemicalDefined", "n"),
            ("ChemicalUndefined", "undefined"),
        ):
            count = sum(p["metrics"]["feature_effect_cosine"][field] for p in pairs)
            macros.append(rf"\newcommand{{\Overlap{prefix}{suffix}}}{{{count:,}}}")
        for suffix, value in (
            ("AllCos", all_topics["best_score"]["mean"]),
            ("RecurringCos", recurring["best_score"]["mean"]),
            ("Compound", recurring["compound_jaccard"]["mean"]),
            ("Chemical", recurring["feature_effect_cosine"]["mean"]),
            ("Fragment", recurring["fragment_cosine"]["mean"]),
            ("Loss", recurring["loss_cosine"]["mean"]),
            (
                "ChannelCos",
                group(data, direction, "recurring_evaluable", "balanced_channels")[
                    "best_score"
                ]["mean"],
            ),
            (
                "NoMAGFilterCos",
                group(data, direction, "recurring")["best_score"]["mean"],
            ),
        ):
            macros.append(rf"\newcommand{{\Overlap{prefix}{suffix}}}{{{value:.3f}}}")
        table.append(
            " & ".join(
                [
                    label,
                    cell(all_topics["best_score"], 3),
                    cell(recurring["best_score"], 3),
                    cell(recurring["compound_jaccard"], 3),
                    cell(recurring["feature_effect_cosine"], 3),
                ]
            )
            + r" \\"
        )
    examples = [
        e
        for e in data["examples"]["examples"]
        if e["direction"] == "selected_to_tomotopy"
    ]
    for label, example in zip(("Median", "Upper", "Multiple"), examples, strict=True):
        for suffix, value in (
            ("Cos", example["best_score"]),
            ("SecondCos", example["second_score"]),
        ):
            macros.append(rf"\newcommand{{\Overlap{label}{suffix}}}{{{value:.3f}}}")
        for target_index, overlaps in enumerate(example["direct_overlaps"]):
            suffix = "" if target_index == 0 else "Second"
            for overlap in overlaps:
                if overlap["tolerance_da"] == 0.01:
                    channel = "Fragment" if overlap["channel"] == "frag" else "Loss"
                    macros.append(
                        rf"\newcommand{{\Overlap{label}{suffix}{channel}Shared}}{{{overlap['matched']}}}"
                    )
    (output / "cross_model_overlap_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )
    (output / "cross_model_overlap_table.tex").write_text(
        "\n".join(table) + "\n" + r"\bottomrule" + "\n", encoding="utf-8"
    )
    summary = {
        "groups": data["summary"]["groups"],
        "inventory": data["inventory"],
        "examples": data["examples"],
    }
    write_json(output / "cross_model_overlap_summary.json", summary)
    return summary
