"""Generate manuscript numbers only from verified chemical-assessment evidence.

This reader uses the standard library so the report can be regenerated without
loading checkpoints, RDKit, NumPy, MAG or any model-training dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from .utils import read_json, sha256_file, write_json


def read_assessment_evidence(evidence: Path) -> dict:
    """Validate all payloads, declared fit inventory and fixed analysis families."""
    protocol = read_json(evidence / "protocol.json")
    seal = read_json(evidence / "input_seal.json")
    if sha256_file(evidence / "protocol.json") != seal["protocol_sha256"]:
        raise ValueError("chemical assessment protocol identity differs")
    stages = {}
    for stage in ("specificity", "stability", "references"):
        complete = read_json(evidence / f"{stage}_complete.json")
        for name, expected in complete["payloads"].items():
            path = evidence / name
            if (
                path.stat().st_size != expected["bytes"]
                or sha256_file(path) != expected["sha256"]
            ):
                raise ValueError(f"chemical assessment payload changed: {path}")
        stages[stage] = complete["summary"]
    expected_fits = {f["id"] for f in protocol["fits"]}
    if (
        len(expected_fits) != 6
        or sorted(f["model"] for f in protocol["fits"])
        != ["selected"] * 3 + ["tomotopy"] * 3
    ):
        raise ValueError("chemical comparison requires exactly three fits per model")
    for config in protocol["configurations"]:
        summary = stages["specificity"][config["id"]]
        if {f["fit"] for f in summary["fits"]} != expected_fits or len(
            summary["fits"]
        ) != 6:
            raise ValueError("incomplete chemical fit inventory")
        if (
            summary["sos_family_size"] != 6000
            or summary["feature_family_size"] != 996000
            or summary["permutations"] != 99999
        ):
            raise ValueError("changed null-analysis family or Monte Carlo budget")
    if len(stages["stability"]["pairs"]) != 15:
        raise ValueError("missing fit-pair comparisons")
    if {f["fit"] for f in stages["references"]["fits"]} != expected_fits:
        raise ValueError("missing reference comparison fit")
    return {"protocol": protocol, **stages}


def latex_escape(value: str) -> str:
    """Escape reference annotations as text, never as LaTeX instructions."""
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(substitutions.get(c, c) for c in value)


def reference_display_label(value: str) -> str:
    """Keep the leading manual description; full original text stays in evidence.

    Repair the export's observed misdecoded UTF-8 en dash for display only.
    Parenthetical source/formula explanations remain in the full saved label.
    """
    return value.replace("â€“", "-").split(" (", 1)[0]


def fit_stats(rows: list[dict], model: str, field: str) -> dict:
    """Three-fit sample SD, not a topic-level standard error."""
    values = [r[field] for r in rows if r["model"] == model]
    if len(values) != 3:
        raise ValueError("fit-level summary needs exactly three values")
    return {"mean": mean(values), "sample_sd": stdev(values)}


def cell(stats: dict, digits: int) -> str:
    """Fixed display precision, preserving full precision in evidence summaries."""
    return f"${stats['mean']:.{digits}f}" + r"\pm" + f"{stats['sample_sd']:.{digits}f}$"


def selected_reference_examples(evidence: Path) -> list[dict]:
    """Choose distinct high-ranked recurring examples, explicitly illustrative.

    This deterministic display rule was documented before reference matching:
    selected-model top-one hits, >=2 supporting compounds and at least one
    directly shared fragment/loss at 0.01 Da, sorted by embedding cosine.
    It is not a correctness threshold or a representative sample of motifs.
    """
    hits = [
        json.loads(line)
        for line in (evidence / "reference_hits.jsonl").read_text().splitlines()
    ]
    eligible = [
        r
        for r in hits
        if r["model"] == "selected"
        and r["rank"] == 1
        and r["support_compounds"] >= 2
        and any(
            o["matched"] > 0 for o in r["direct_overlaps"] if o["tolerance_da"] == 0.01
        )
    ]
    chosen, seen = [], set()
    for row in sorted(
        eligible, key=lambda r: (-r["spec2vec_cosine"], r["fit"], r["topic_id"])
    ):
        if row["reference_id"] not in seen:
            chosen.append(row)
            seen.add(row["reference_id"])
        if len(chosen) == 3:
            break
    return chosen


def generate_chemical_report(evidence: Path, output: Path) -> dict:
    """Write one compact table, numeric macros and reference examples with hashes."""
    data = read_assessment_evidence(evidence)
    output.mkdir(parents=True, exist_ok=True)
    macros, comparisons, summaries = [], [], {}

    def macro(name: str, value: float, digits: int = 3) -> None:
        macros.append(rf"\newcommand{{\Chem{name}}}{{{value:.{digits}f}}}")

    primary = data["specificity"]["primary_50"]
    macro("PrimaryStrata", primary["strata"], 0)
    macro("FixedCompounds", primary["singleton_compounds"], 0)
    for model in ("selected", "tomotopy"):
        prefix = "Selected" if model == "selected" else "LDA"
        rows = [r for r in primary["fits"] if r["model"] == model]
        for field, suffix in (
            ("eligible_topics", "Eligible"),
            ("supported_topics", "Supported"),
        ):
            macro(prefix + suffix + "Min", min(r[field] for r in rows), 0)
            macro(prefix + suffix + "Max", max(r[field] for r in rows), 0)

    for config in data["protocol"]["configurations"]:
        rows = data["specificity"][config["id"]]["fits"]
        flat = [
            {
                **r,
                "background": r["background_sos"]["mean"],
                "excess": r["excess_sos"]["mean"],
                "recurring_excess": r["recurring_excess_sos"]["mean"],
            }
            for r in rows
        ]
        summaries[config["id"]] = {}
        for model in ("tomotopy", "selected"):
            prefix = "Selected" if model == "selected" else "LDA"
            fields = (
                ("recurring_eligible_topics", "Recurring", 1),
                ("single_compound_topics", "Singleton", 1),
                ("background", "Background", 3),
                ("excess", "Excess", 3),
                ("sos_q05_topics", "SOSTopics", 1),
                ("feature_q05_topics", "FeatureTopics", 1),
            )
            stats = {field: fit_stats(flat, model, field) for field, _, _ in fields}
            summaries[config["id"]][model] = stats
            if config["id"] == "primary_50":
                macro(
                    prefix + "SingletonEligible",
                    mean(
                        r["eligible_topics"] - r["recurring_eligible_topics"]
                        for r in rows
                        if r["model"] == model
                    ),
                    1,
                )
                for field, suffix, digits in fields:
                    macro(prefix + suffix, stats[field]["mean"], digits)
                    macro(prefix + suffix + "SD", stats[field]["sample_sd"], digits)
                comparisons.append(
                    " & ".join(
                        [
                            (
                                "Tomotopy LDA"
                                if model == "tomotopy"
                                else "Contextual Sparse ETM"
                            ),
                            *(
                                cell(stats[field], digits)
                                for field, _, digits in fields
                                if field != "single_compound_topics"
                            ),
                        ]
                    )
                    + r" \\"
                )
            elif config["id"] == "scaffold_50":
                macro(prefix + "ScaffoldExcess", stats["excess"]["mean"])
                macro(
                    prefix + "ScaffoldFeatureTopics",
                    stats["feature_q05_topics"]["mean"],
                    1,
                )
    for model in ("selected", "tomotopy"):
        prefix = "Selected" if model == "selected" else "LDA"
        within = [
            r for r in data["stability"]["pairs"] if r["kind"] == f"within_{model}"
        ]
        macro(prefix + "MatchMedian", mean(r["beta_cosine"]["median"] for r in within))
        macro(prefix + "Reciprocal", mean(r["reciprocal_fraction"] for r in within))
        macro(
            prefix + "CompoundMatch",
            mean(r["compound_jaccard"]["median"] for r in within),
        )
        redundant = [r for r in data["stability"]["redundancy"] if r["model"] == model]
        macro(
            prefix + "RedundantMedian",
            mean(r["nearest_beta_cosine"]["median"] for r in redundant),
        )
        macro(
            prefix + "RepeatedMAG",
            mean(r["topics_sharing_MAG_profile"] for r in redundant),
            1,
        )
        macro(
            prefix + "RepeatedMAGPercent",
            mean(
                100 * r["topics_sharing_MAG_profile"] / r["available_MAG_profiles"]
                for r in redundant
            ),
            1,
        )
        reference = [r for r in data["references"]["fits"] if r["model"] == model]
        macro(
            prefix + "ReferenceMedian",
            mean(r["top_reference_cosine"]["median"] for r in reference),
        )
        macro(
            prefix + "ReferenceOverlap",
            mean(r["top_reference_query_coverage"]["mean"] for r in reference),
        )
        macro(
            prefix + "ReferenceAnyOverlap",
            mean(r["top_reference_with_any_direct_overlap"] for r in reference),
            1,
        )
    cross = [r for r in data["stability"]["pairs"] if r["kind"] == "cross_model"]
    macro("CrossMatchMedian", mean(r["beta_cosine"]["median"] for r in cross))
    for field, name in (
        ("raw_reference_records", "RawReferences"),
        ("unique_reference_signatures", "UniqueReferences"),
        ("scorable_references", "ScorableReferences"),
        ("excluded_records", "ExcludedReferences"),
        ("conflicting_label_signatures", "ConflictingReferences"),
    ):
        macro(name, data["references"][field], 0)
    references = {
        r["reference_id"]: r
        for r in read_json(evidence / "references.json")["references"]
    }
    example_rows = []
    examples = selected_reference_examples(evidence)
    for index, row in enumerate(examples, start=1):
        reference = references[row["reference_id"]]
        labels = sorted(
            {
                s["short_annotation"] or s["annotation"] or "Unlabelled"
                for s in reference["sources"]
            }
        )
        source_names = sorted(
            {
                str(s["motifset"]) + ": " + str(s["motif_id"])
                for s in reference["sources"]
            }
        )
        overlaps = [o for o in row["direct_overlaps"] if o["tolerance_da"] == 0.01]
        typed = [f"{o['matched']}/{o['query_features']}" for o in overlaps]
        example_rows.append(
            " & ".join(
                [
                    str(index),
                    latex_escape("; ".join(reference_display_label(v) for v in labels)),
                    str(row["support_compounds"]),
                    f"{row['spec2vec_cosine']:.3f}",
                    *typed,
                ]
            )
            + r" \\"
        )
        row["display_example"] = index
        row["reference_labels"] = labels
        row["reference_sources"] = source_names
    for name, lines in (
        ("macros", macros),
        ("table", comparisons),
        ("reference_examples", example_rows),
    ):
        if name != "macros":
            lines = [*lines, r"\bottomrule"]
        (output / f"chemical_assessment_{name}.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    summary = {
        "fit_aggregates": summaries,
        "reference_examples": examples,
        "evidence_stage_sha256": {
            stage: sha256_file(evidence / f"{stage}_complete.json")
            for stage in ("specificity", "stability", "references")
        },
    }
    write_json(output / "chemical_assessment_summary.json", summary)
    return summary
