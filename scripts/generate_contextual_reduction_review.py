"""Validate the staged within-model edit inventory and generate exact tables."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from benchmarks.neural_ms2lda.report_comparison import (
    baseline_summaries,
    comparison_cell,
    comparison_table,
    read_baseline_evidence,
)
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

BASELINE_EVIDENCE = ROOT / "research/baseline_repeats_20260908/evidence"

LABELS = {
    "reduced_no_loo": "No context, $r=xA$",
    "reduced_document_context": "Full-document context",
    "reduced_fixed_context": "Fixed context weight $c=1$",
    "reduced_full_routing": "All-topic routing",
    "reduced_linear_evidence": "Linear evidence offset",
    "reduced_softmax": "Softmax instead of entmax",
    "reduced_shallow": "One hidden layer",
    "reduced_evidence_only": "Evidence-only encoder",
    "reduced_no_loo_shallow": "No context + one layer",
    "reduced_no_loo_top1": "No context + top-1",
    "reduced_no_loo_shallow_top1": "No context + one layer + top-1",
    "reduced_document_context_shallow": "Document context + one layer",
    "reduced_document_context_fixed": "Document context + fixed $c$",
    "reduced_document_context_fixed_shallow": "Document + fixed $c$ + one layer",
    "reduced_scaled_full_routing": "Scaled all-topic routing",
    "reduced_fixed_context_top1": "Fixed $c$ + hard top-1",
    "reduced_shallow_narrow": "One layer, width 100",
    "reduced_attention_document_fixed": "Attention-only + document + fixed $c$",
}
ATOMIC = tuple(LABELS)[:8]
DESCRIPTIVE_SYNTHETIC = {"reduced_no_loo", "reduced_softmax"}
DESCRIPTIVE_REAL = DESCRIPTIVE_SYNTHETIC | {
    "reduced_document_context_fixed_shallow",
    "reduced_shallow_narrow",
    "reduced_attention_document_fixed",
}
REPEAT_VARIANTS = {
    "reduced_shallow",
    "reduced_document_context",
    "reduced_evidence_only",
}
FIELDS = (
    "beta_recovery",
    "theta_recovery",
    "recovered_motifs",
    "completion_nll",
    "median_effective_topics",
)
SOS_FIELDS = tuple(f"sos_count_ge_{cutoff:.1f}" for cutoff in (0.5, 0.6, 0.7, 0.8, 0.9))


def failed_inventory(payload):
    """Retain explicit numerical failures without inventing completed scores."""
    failures = {}
    for record in payload["runs"]:
        key = record["variant"], record["seed"]
        if key in failures or key[0] not in LABELS:
            raise ValueError("duplicate or unknown failed reduction")
        expected = {
            "seed": 42,
            "training_seed": 7043,
            "topics": 1000,
            "selection_split": "validation",
            "test_matrices_loaded": False,
            "planned_epochs": 120,
            "status": "non_finite_training_loss",
            "metrics": None,
        }
        if any(
            record.get(field, "missing") != value for field, value in expected.items()
        ):
            raise ValueError("invalid failed-run boundary or status")
        if not 0 < record["last_completed_checkpoint_epoch"] < record["planned_epochs"]:
            raise ValueError("invalid failed-run checkpoint epoch")
        sources = record["source_sha256"]
        if not sources or any(
            len(digest) != 64 or set(digest) - set("0123456789abcdef")
            for digest in sources.values()
        ):
            raise ValueError("failed run needs source hashes")
        failures[key] = record
    return failures


def check_real_inventory(completed, failed, expected):
    if set(completed) & set(failed):
        raise ValueError("same reduction recorded as complete and failed")
    if set(completed) | set(failed) != expected:
        raise ValueError("missing or unexpected real reduction record")


def paired_tradeoffs(candidate, reference):
    """Effect sizes, not a replacement score or arbitrary acceptance cutoff."""
    nll_delta = number(candidate, "completion_nll") - number(
        reference, "completion_nll"
    )
    return {
        "nll_difference_nats_per_token": nll_delta,
        "perplexity_change_percent": 100 * math.expm1(nll_delta),
        "mean_sos_difference": number(candidate, "mean_sos")
        - number(reference, "mean_sos"),
        "relative_changes_percent": {
            key: 100 * (number(candidate, key) / number(reference, key) - 1)
            for key in (
                "evaluable_motifs",
                "useful_motifs",
                "unique_top1_topics",
                "median_effective_topics",
                "parameters",
            )
        },
        "sos_cutoff_counts_difference": {
            key: int(number(candidate, key) - number(reference, key))
            for key in SOS_FIELDS
        },
    }


def sos_line(label, row):
    return " & ".join([label, *(str(int(number(row, f))) for f in SOS_FIELDS)]) + r" \\"


def repeat_summary(rows):
    """Descriptive same-split training stability, not external confidence bounds."""
    if sorted(int(r["seed"]) for r in rows) != [11, 23, 42]:
        raise ValueError("repeat summary needs exactly seeds 11, 23 and 42")
    if len({number(r, "parameters") for r in rows}) != 1:
        raise ValueError("repeat parameter counts differ")
    return {
        "parameters": int(number(rows[0], "parameters")),
        **{
            key: {
                "mean": mean(number(r, key) for r in rows),
                "sample_sd": stdev(number(r, key) for r in rows),
            }
            for key in (
                "evaluable_motifs",
                "useful_motifs",
                "mean_sos",
                "completion_nll",
                "median_effective_topics",
                "unique_top1_topics",
            )
        },
    }


def repeat_line(label, summary):
    fields = (
        ("evaluable_motifs", 1),
        ("useful_motifs", 1),
        ("mean_sos", 3),
        ("completion_nll", 3),
        ("median_effective_topics", 2),
    )
    cells = [label, f"{summary['parameters']:,}"]
    for field, precision in fields:
        value = summary[field]
        cells.append(
            "$"
            + f"{value['mean']:.{precision}f}"
            + r"\pm"
            + f"{value['sample_sd']:.{precision}f}$"
        )
    return " & ".join(cells) + r" \\"


def current_model_report(rows, baseline_rows):
    """Report the chosen model only, without borrowing its predecessor's test data."""
    selected = [r for r in rows if r["variant"] == "reduced_document_context"]
    keys = [(r["data"], int(r["topics"]), int(r["seed"])) for r in selected]
    expected = {("MSnLib validation", 1000, s) for s in (11, 23, 42)} | {
        ("synthetic", k, s) for k in (36, 128) for s in (11, 23, 37)
    }
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("current-model report needs exactly its nine completed fits")
    for row in selected:
        validate_recipe(row)
    real = sorted(
        (r for r in selected if r["data"] == "MSnLib validation"),
        key=lambda r: int(r["seed"]),
    )
    summary = repeat_summary(real)
    baselines = baseline_summaries(baseline_rows)
    comparison = comparison_table(baselines, summary)
    stability = [validation_line(str(int(r["training_seed"])), r) for r in real]
    synthetic, synthetic_summaries = [], {}
    for k in (36, 128):
        group = [r for r in selected if int(r["topics"]) == k]
        stats = {
            field: {
                "mean": mean(number(r, field) for r in group),
                "sample_sd": stdev(number(r, field) for r in group),
            }
            for field in FIELDS
        }
        synthetic_summaries[str(k)] = stats
        synthetic.append(
            " & ".join(
                [
                    str(k),
                    *(
                        comparison_cell(stats[f], 2 if f == "recovered_motifs" else 3)
                        for f in FIELDS
                    ),
                ]
            )
            + r" \\",
        )
    macros = []
    for key, suffix, precision in (
        ("evaluable_motifs", "Evaluable", 1),
        ("useful_motifs", "Useful", 1),
        ("mean_sos", "SOS", 3),
        ("completion_nll", "NLL", 3),
        ("median_effective_topics", "Effective", 2),
        ("unique_top1_topics", "Winners", 1),
    ):
        for value, tail in (("mean", "Mean"), ("sample_sd", "SD")):
            macros.append(
                rf"\newcommand{{\Current{suffix}{tail}}}"
                + "{"
                + f"{summary[key][value]:.{precision}f}"
                + "}"
            )
    for model, stats in baselines.items():
        prefix = {
            "canonical ETM": "Canonical",
            "Tomotopy LDA": "LDA",
        }[model]
        for key, suffix, precision in (
            ("evaluable_motifs", "Evaluable", 1),
            ("useful_motifs", "Useful", 1),
            ("mean_sos", "SOS", 3),
            ("completion_nll", "NLL", 3),
            ("median_effective_topics", "Effective", 2),
        ):
            for statistic, tail in (("mean", ""), ("sample_sd", "SD")):
                macros.append(
                    rf"\newcommand{{\Reference{prefix}{suffix}{tail}}}"
                    + "{"
                    + f"{stats[key][statistic]:.{precision}f}"
                    + "}"
                )
    useful_difference = (
        summary["useful_motifs"]["mean"]
        - baselines["Tomotopy LDA"]["useful_motifs"]["mean"]
    )
    macros.append(
        rf"\newcommand{{\CurrentUsefulDifference}}{{{useful_difference:+.1f}}}"
    )
    return (
        {
            "validation_table": comparison,
            "stability_table": stability,
            "synthetic_table": synthetic,
            "macros": macros,
        },
        {
            "variant": "reduced_document_context",
            "reported_split": "validation",
            "real": summary,
            "synthetic": synthetic_summaries,
            "historical_test_results_used": False,
            "primary_comparator": "Tomotopy LDA",
            "secondary_base_check": "canonical ETM",
            "baselines": baselines,
            "baseline_training_caveat": (
                "Neural reference batch size 256; chosen-model batch size 200. "
                "Every comparison row summarizes three fits on fixed data; "
                "sample SD is between-fit variation, not a confidence interval. "
                "The retained plain ETM fit used WSL2; its two new fits used "
                "native Linux with matching scientific package versions. "
                "This is not a seed-only variance estimate."
            ),
        },
    )


def validate_recipe(row):
    """Forbid incomplete inference-only, altered-loss or altered-budget records."""
    if row["data"] not in {"synthetic", "MSnLib validation"}:
        raise ValueError("unexpected evidence split")
    if row["variant"] not in {*LABELS, "contextual"}:
        raise ValueError("unknown reduction")
    for field, expected in {
        "epochs": 120,
        "batch_size": 200,
        "hidden": 100 if row["variant"] == "reduced_shallow_narrow" else 800,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 1.2e-6,
    }.items():
        if number(row, field) != expected:
            raise ValueError(f"reduction recipe mismatch: {field}")
    if boolean(row, "reused_training_weights"):
        raise ValueError("reduction needs a fresh training run")
    for field, expected in {
        "normalization_statistics": "not_applicable",
        "count_scaling": "raw_counts",
        "decoder": "additive_mixture",
        "beta_meaning": "emissions",
        "framework": "pytorch",
    }.items():
        if row[field] != expected:
            raise ValueError(f"reduction meaning mismatch: {field}")
    if number(row, "training_seed") != number(row, "seed") + 7001:
        raise ValueError("unexpected training seed")


def synthetic_inventory(rows, controls):
    """Reconstruct the predeclared advancement tree without hiding failures."""
    groups = defaultdict(list)
    for row in rows:
        validate_recipe(row)
        if row["data"] == "synthetic":
            groups[int(row["topics"]), row["variant"]].append(row)
    for group in groups.values():
        if sorted(int(r["seed"]) for r in group) != [11, 23, 37]:
            raise ValueError("each reduction group needs exactly seeds 11/23/37")
    current = {
        k: [
            r
            for r in controls
            if r["data"] == "synthetic"
            and int(r["topics"]) == k
            and r["variant"] == "contextual"
        ]
        for k in (36, 128)
    }
    if any(sorted(int(r["seed"]) for r in g) != [11, 23, 37] for g in current.values()):
        raise ValueError("paired controls missing")
    expected, gates = set(), {}

    def require(k, variant):
        key = k, variant
        expected.add(key)
        if key not in groups:
            raise ValueError(f"missing required reduction: {key}")
        gates[key] = synthetic_gate(groups[key], current[k], k)
        return gates[key]["passed"]

    passed = set()
    for variant in ATOMIC:
        high_pass = require(128, variant)
        low_pass = (
            require(36, variant)
            if high_pass or variant in DESCRIPTIVE_SYNTHETIC
            else False
        )
        if high_pass and low_pass:
            passed.add(variant)
    smooth = "reduced_scaled_full_routing"
    if all([require(k, smooth) for k in (128, 36)]):
        passed.add(smooth)
    narrow = "reduced_shallow_narrow"
    if all([require(k, narrow) for k in (128, 36)]):
        passed.add(narrow)
    attention = "reduced_attention_document_fixed"
    if all([require(k, attention) for k in (128, 36)]):
        passed.add(attention)
    if "reduced_fixed_context" in passed:
        hard = "reduced_fixed_context_top1"
        if all([require(k, hard) for k in (128, 36)]):
            passed.add(hard)
    followups = []
    if "reduced_no_loo" in passed:
        followups.append("reduced_no_loo_top1")
        if "reduced_shallow" in passed:
            followups.extend(["reduced_no_loo_shallow", "reduced_no_loo_shallow_top1"])
    if {"reduced_document_context", "reduced_shallow"} <= passed:
        followups.append("reduced_document_context_shallow")
    if {"reduced_document_context", "reduced_fixed_context"} <= passed:
        followups.append("reduced_document_context_fixed")
    for variant in followups:
        passes = [require(k, variant) for k in (128, 36)]
        if all(passes):
            passed.add(variant)
    if {"reduced_document_context_fixed", "reduced_shallow"} <= passed:
        variant = "reduced_document_context_fixed_shallow"
        if all([require(k, variant) for k in (128, 36)]):
            passed.add(variant)
    if set(groups) != expected:
        raise ValueError("unexpected, unreported synthetic reduction group")
    return groups, gates, passed


def generate(
    evidence=REVIEW,
    output=ROOT / "docs/research/generated",
    baseline_evidence=BASELINE_EVIDENCE,
):
    """Produce complete synthetic/real tables, all gates and source hashes."""
    evidence = evidence.resolve()
    output = output.resolve()
    baseline_evidence = baseline_evidence.resolve()
    reduction_root = evidence / "within_model"
    rows = read_rows(reduction_root / "runs.csv")
    # Require the complete current comparison before writing any report input.
    current_parts, current_summary = current_model_report(
        rows, read_baseline_evidence(baseline_evidence)
    )
    failures = failed_inventory(
        json.loads((reduction_root / "failed_runs.json").read_text())
    )
    controls = read_rows(evidence / "runs.csv")
    groups, gates, passed = synthetic_inventory(rows, controls)
    lines, summaries = [], []
    for k in (128, 36):
        for variant in LABELS:
            if (k, variant) not in groups:
                continue
            group = groups[k, variant]
            summary = {
                key: {
                    "mean": mean(number(r, key) for r in group),
                    "sample_sd": stdev(number(r, key) for r in group),
                }
                for key in FIELDS
            }
            summaries.append({"topics": k, "variant": variant, **summary})
            cells = [str(k), LABELS[variant]]
            for field in FIELDS:
                precision = (
                    2 if field in {"recovered_motifs", "median_effective_topics"} else 3
                )
                cells.append(f"{summary[field]['mean']:.{precision}f}")
            cells.append("Pass" if gates[k, variant]["passed"] else "Fail")
            lines.append(" & ".join(cells) + r" \\")
        if k == 128:
            lines.append(r"\midrule")
    real = [r for r in rows if r["data"] == "MSnLib validation"]
    lookup = {(r["variant"], int(r["seed"])): r for r in real}
    if len(lookup) != len(real) or any(int(r["topics"]) != 1000 for r in real):
        raise ValueError("duplicate or wrong-K real reduction")
    frozen = next(r for r in read_rows(FROZEN) if r["model"] == "Contextual Sparse ETM")
    paired = next(
        r
        for r in controls
        if r["data"] == "MSnLib validation" and r["variant"] == "contextual"
    )
    real_lines = [validation_line("Current, seed 42", paired)]
    sos_lines = [sos_line("Current, seed 42", paired)]
    advanced = passed | DESCRIPTIVE_REAL
    expected, real_gates, initial_passed = {(v, 42) for v in advanced}, {}, set()
    for variant in sorted(advanced):
        if (variant, 42) in failures:
            real_lines.append(
                LABELS[variant]
                + r", 42 & \multicolumn{6}{c}{Training failed: non-finite loss} \\"
            )
            sos_lines.append(
                LABELS[variant]
                + r", 42 & \multicolumn{5}{c}{No completed evaluation} \\"
            )
            continue
        if (variant, 42) not in lookup:
            raise ValueError("missing advanced real reduction")
        row = lookup[variant, 42]
        real_gates[variant, 42] = real_gate(row, [frozen, paired])
        real_lines.append(validation_line(LABELS[variant] + ", 42", row))
        sos_lines.append(sos_line(LABELS[variant] + ", 42", row))
        if real_gates[variant, 42]["passed"]:
            initial_passed.add(variant)
    # Repeats follow the recorded scientific shortlist, not a discontinuous
    # flag. Near misses can be shortlisted; a flag alone does not recommend one.
    repeated = REPEAT_VARIANTS
    if repeated:
        for seed in (11, 23):
            expected.add(("contextual", seed))
            if ("contextual", seed) not in lookup:
                raise ValueError("missing same-seed confirmation control")
            control = lookup["contextual", seed]
            real_lines.append(validation_line(f"Current, seed {seed}", control))
            sos_lines.append(sos_line(f"Current, seed {seed}", control))
            for variant in sorted(repeated):
                expected.add((variant, seed))
                if (variant, seed) not in lookup:
                    raise ValueError("missing confirmation seed")
                row = lookup[variant, seed]
                real_gates[variant, seed] = real_gate(row, [frozen, control])
                real_lines.append(validation_line(LABELS[variant] + f", {seed}", row))
                sos_lines.append(sos_line(LABELS[variant] + f", {seed}", row))
    check_real_inventory(lookup, failures, expected)
    repeat_summaries = {
        v: repeat_summary(
            [
                paired if v == "contextual" and seed == 42 else lookup[v, seed]
                for seed in (42, 11, 23)
            ]
        )
        for v in ["contextual", *sorted(repeated)]
    }
    repeat_lines = [
        repeat_line("Current" if v == "contextual" else LABELS[v], summary)
        for v, summary in repeat_summaries.items()
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("synthetic", lines),
        ("validation", real_lines),
        ("sos", sos_lines),
        ("repeats", repeat_lines),
    ):
        (output / f"contextual_reduction_{name}_table.tex").write_text(
            "% Generated: staged within-model review, validation only.\n"
            + "\n".join([*content, r"\bottomrule"])
            + "\n",
            encoding="utf-8",
        )
    for name, content in current_parts.items():
        end = [] if name == "macros" else [r"\bottomrule"]
        (output / f"current_contextual_etm_{name}.tex").write_text(
            "% Generated: chosen full-document model; validation only.\n"
            + "\n".join([*content, *end])
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "stage": "within-model validation development after phases 1/2",
        "new_training_runs": len(rows),
        "successful_training_runs": len(rows),
        "failed_training_runs": len(failures),
        "attempted_model_configurations": len(rows) + len(failures),
        "failed_runs": list(failures.values()),
        "current_model_report": current_summary,
        "synthetic_groups": summaries,
        "synthetic_gates": {f"{k}_{v}": g for (k, v), g in gates.items()},
        "real_gates": {f"{v}_{s}": g for (v, s), g in real_gates.items()},
        "real_repeat_summaries": repeat_summaries,
        "strictly_confirmed_reductions": sorted(
            v
            for v in repeated & initial_passed
            if all(real_gates[v, s]["passed"] for s in (42, 11, 23))
        ),
        "global_minimality_proved": False,
        "recorded_repeat_variants": sorted(REPEAT_VARIANTS),
        "descriptive_real_variants": sorted(DESCRIPTIVE_REAL),
        "strict_gates_unchanged": True,
        "thresholds_determine_recommendation": False,
        "recommended_variant": "reduced_document_context",
        "recommendation_scope": (
            "Validation-development recommendation, not a production promotion "
            "or unbiased test confirmation. Retain the published ETM MLP "
            "Gaussian recognition backbone and use the simpler full-document "
            "context. Three paired seeds preserve chemical usefulness and "
            "slightly improve NLL; a winner count of 799 is not a scientific "
            "failure at an arbitrary 800 cutoff. One-layer width 100 remains "
            "promising but has only one real seed. Global minimality is not proved."
        ),
        "paired_effect_sizes": {
            f"{v}_{s}": paired_tradeoffs(
                lookup[v, s], paired if s == 42 else lookup["contextual", s]
            )
            for v, s in real_gates
        },
        "user_decision_clarification": (
            "The user accepted the observed approximately 2.6% predictive-fit "
            "trade-off before repeat results and then clarified that all "
            "arbitrary thresholds should be reference points, not rigid "
            "decision laws. Report actual trade-offs and repeat-seed "
            "stability rather than choosing a new threshold to force a pass."
        ),
        "chemical_inventory_checks": {
            f"{v}_{s}": {
                "passed": all(
                    value
                    for key, value in g["checks"].items()
                    if not key.endswith("completion_nll")
                ),
                "checks": {
                    key: value
                    for key, value in g["checks"].items()
                    if not key.endswith("completion_nll")
                },
            }
            for (v, s), g in real_gates.items()
        },
        "completion_nll_change_pct": {
            f"{v}_{s}": {
                "versus_frozen": 100
                * (
                    number(lookup[v, s], "completion_nll")
                    / number(frozen, "completion_nll")
                    - 1
                ),
                "versus_paired": 100
                * (
                    number(lookup[v, s], "completion_nll")
                    / number(
                        paired if s == 42 else lookup["contextual", s], "completion_nll"
                    )
                    - 1
                ),
            }
            for v, s in real_gates
        },
        "source_sha256": {
            # Repository inputs stay portable. Explicit external inputs retain
            # their full identity instead of failing after table publication.
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256_file(p)
            for p in (
                reduction_root / "runs.csv",
                reduction_root / "manifest.json",
                evidence / "runs.csv",
                evidence / "manifest.json",
                FROZEN,
                evidence.parent / "within_model_protocol.md",
                reduction_root / "mathematical_audit.json",
                reduction_root / "interrupted_attempts.json",
                reduction_root / "failed_runs.json",
                reduction_root / "attention_interface_audit_seed42.json",
                reduction_root / "checkpoint_reload_audit.json",
                reduction_root / "recommended_checkpoint_reload_audit.json",
                reduction_root / "thread_count_replay_audit.json",
                Path(__file__),
                ROOT / "benchmarks/neural_ms2lda/review_tables.py",
                ROOT / "benchmarks/neural_ms2lda/report_comparison.py",
                baseline_evidence / "runs.csv",
                baseline_evidence / "manifest.json",
                ROOT / "benchmarks/neural_ms2lda/utils.py",
                ROOT / "scripts/generate_minimal_etm_review.py",
            )
        },
    }
    write_json(reduction_root / "review_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=REVIEW)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/generated")
    parser.add_argument("--baseline-evidence", type=Path, default=BASELINE_EVIDENCE)
    args = parser.parse_args()
    generate(args.evidence, args.output, args.baseline_evidence)


if __name__ == "__main__":
    main()
