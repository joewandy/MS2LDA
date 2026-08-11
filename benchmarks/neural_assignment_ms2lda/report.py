# ruff: noqa: PLR0913
"""Compact decision artifacts for the staged neural-assignment study."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .utils import read_json, write_json


def chemical_gate_checks(
    chemical: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the two frozen chemical viability thresholds."""
    config = protocol["chemical_gates"]
    dominant = float(chemical["dominant_topic_chemistry"]["mean_sos"])
    coverage = float(
        chemical["high_confidence_chemistry"]["sos_evaluable_coverage"],
    )
    checks = {
        "dominant_mean_sos": {
            "pass": dominant >= float(config["minimum_dominant_mean_sos"]),
            "actual": dominant,
            "minimum": float(config["minimum_dominant_mean_sos"]),
        },
        "high_confidence_coverage": {
            "pass": coverage >= float(config["minimum_high_confidence_coverage"]),
            "actual": coverage,
            "minimum": float(config["minimum_high_confidence_coverage"]),
        },
    }
    return {
        "checks": checks,
        "pass": all(row["pass"] for row in checks.values()),
        "failed": [name for name, row in checks.items() if not row["pass"]],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Neural-assignment MS2LDA bounded experiment",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Furthest stage: **{report['furthest_stage']}**",
        "",
        f"Validation-selected K=1000 attempt: "
        f"**{report.get('selected_attempt') or 'none'}**",
        "",
        "This is a staged viability result on the corrected fixed split. "
        "A failed gate stops the bounded experiment; it does not silently "
        "redirect the project to motif annotation.",
        "",
    ]
    amendment = report.get("exploratory_amendment")
    if amendment is not None:
        lines.extend(
            [
                "This continuation was declared after the v1 K=200 validation "
                "result. Only the K=200 active-topic screening stop is waived; "
                "all final K=1000, test, and chemical gates are unchanged.",
                "",
            ],
        )
    for stage in ("synthetic", "k200", "k1000_validation", "k1000_test", "chemical"):
        value = report["gates"].get(stage)
        if value is not None:
            label = "PASS" if value["pass"] else "FAIL"
            waived = value.get("waived_failures", [])
            if value["pass"] and waived:
                label = f"PASS WITH EXPLORATORY WAIVER ({', '.join(waived)})"
            lines.append(f"- {stage}: {label}")
    lines.extend(
        [
            "",
            "The model uses one deterministic routing pass, no local VB, "
            "no DreaMS input, and no classical topic teacher.",
            "",
        ],
    )
    return "\n".join(lines)


def _scorecard_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage, gate in report["gates"].items():
        if gate is None:
            continue
        if "checks" in gate:
            for metric, check in gate["checks"].items():
                rows.append(
                    {
                        "stage": stage,
                        "metric": metric,
                        "pass": check["pass"],
                        "actual": check.get("actual", ""),
                        "minimum": check.get("minimum", ""),
                        "maximum": check.get("maximum", ""),
                        "waived": metric in gate.get("waived_failures", []),
                        "blocking": metric in gate.get("blocking_failures", []),
                    },
                )
        elif stage == "synthetic":
            for scenario in gate["scenarios"]:
                for metric, check in scenario["gate"]["checks"].items():
                    rows.append(
                        {
                            "stage": f"synthetic:{scenario['scenario']}",
                            "metric": metric,
                            "pass": check["pass"],
                            "actual": check.get("actual", ""),
                            "minimum": check.get("minimum", ""),
                            "maximum": check.get("maximum", ""),
                            "waived": False,
                            "blocking": not check["pass"],
                        },
                    )
    return rows


def _latex(report: dict[str, Any]) -> str:
    decision = str(report["decision"]).replace("_", r"\_")
    selected = str(report.get("selected_attempt") or "none").replace("_", r"\_")
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\begin{document}",
            r"\section*{Neural-assignment MS2LDA bounded result}",
            f"Decision: \\texttt{{{decision}}}.\\\\",
            f"Selected attempt: \\texttt{{{selected}}}.\\\\",
            "The candidate uses one-pass sparse neural peak routing and no "
            "variational-Bayes refinement.",
            r"\end{document}",
            "",
        ],
    )


def build_report(
    run_dir: str | Path,
    *,
    decision: str,
    furthest_stage: str,
    selected_attempt: str | None,
    protocol: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build machine-readable, tabular, prose, and LaTeX artifacts."""
    directory = Path(run_dir).expanduser().resolve()
    synthetic_path = directory / "stages/synthetic/complete.json"
    k200_path = directory / "stages/k200/attempts/primary/complete.json"
    selection_path = directory / "validation_selection.json"
    evaluation_path = (
        directory / "evaluation" / selected_attempt / "complete.json"
        if selected_attempt
        else None
    )
    nonchemical_path = (
        directory / "evaluation" / selected_attempt / "nonchemical_gates.json"
        if selected_attempt
        else None
    )
    chemical_path = (
        directory / "chemical" / selected_attempt / "complete.json"
        if selected_attempt
        else None
    )
    synthetic = read_json(synthetic_path) if synthetic_path.is_file() else None
    k200 = read_json(k200_path) if k200_path.is_file() else None
    selection = read_json(selection_path) if selection_path.is_file() else None
    evaluation = (
        read_json(evaluation_path)
        if evaluation_path is not None and evaluation_path.is_file()
        else None
    )
    nonchemical = (
        read_json(nonchemical_path)
        if nonchemical_path is not None and nonchemical_path.is_file()
        else None
    )
    chemical = (
        read_json(chemical_path)
        if chemical_path is not None and chemical_path.is_file()
        else None
    )
    chemical_gate = (
        chemical_gate_checks(chemical, protocol) if chemical is not None else None
    )
    gates = {
        "synthetic": synthetic,
        "k200": k200["validation_gate"] if k200 is not None else None,
        "k1000_validation": (
            selection["selected_validation_gate"] if selection is not None else None
        ),
        "k1000_test": nonchemical,
        "chemical": chemical_gate,
    }
    report = {
        "schema_version": "neural-assignment-ms2lda/final-report-v1",
        "tracking_issue": protocol["tracking_issue"],
        "evidence_scope": protocol["evidence_scope"],
        "decision": decision,
        "furthest_stage": furthest_stage,
        "selected_attempt": selected_attempt,
        "gates": gates,
        "evaluation": evaluation,
        "chemical": chemical,
        "reference": reference,
        "exploratory_amendment": protocol.get("exploratory_amendment"),
        "fully_neural_contract": {
            "routing_passes_per_representation": 1,
            "local_vb_steps": 0,
            "iterative_test_inference_steps": 0,
            "dreams_used": False,
            "classical_topic_teacher_used": False,
        },
        "no_automatic_annotation_redirect": True,
    }
    output = directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    (output / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    (output / "manuscript_results.tex").write_text(
        _latex(report),
        encoding="utf-8",
    )
    issue = (
        "# Review the viable neural-assignment MS2LDA model\n\n"
        if decision == "retain_viable_neural_assignment_model"
        else "# Revisit fully neural MS2LDA model design\n\n"
    )
    issue += (
        "Follow-up to #10. The bounded result is preserved in the results "
        "checkpoint. Do not automatically redirect to downstream annotation; "
        "use the failed stage and continuous metrics to decide the next "
        "fully neural design.\n"
    )
    (output / "outcome_issue.md").write_text(issue, encoding="utf-8")
    rows = _scorecard_rows(report)
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (output / "scorecard.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return report
