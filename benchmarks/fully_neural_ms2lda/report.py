"""Decision report for the bounded fully neural experiment."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .evaluation import nonchemical_hard_gates
from .utils import read_json, write_json


def _check(
    *,
    actual: float,
    reference: float,
    limit: float,
    direction: str,
) -> dict[str, Any]:
    if direction == "minimum":
        passed = actual >= limit
    elif direction == "maximum":
        passed = actual <= limit
    else:
        msg = f"unknown gate direction: {direction}"
        raise ValueError(msg)
    return {
        "pass": bool(passed),
        "actual": float(actual),
        direction: float(limit),
        "reference": float(reference),
    }


def attempt_scorecard(
    evaluation: dict[str, Any],
    *,
    chemical: dict[str, Any] | None,
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Compute hard viability and the nonfatal competitive scorecard."""
    hard = nonchemical_hard_gates(evaluation, reference, protocol)
    hard_thresholds = protocol["hard_viability_gates"]
    if chemical is None:
        chemical_checks = {
            "dominant_topic_sos": {
                "pass": False,
                "status": "not_run_because_nonchemical_hard_gate_failed",
            },
            "high_confidence_chemical_coverage": {
                "pass": False,
                "status": "not_run_because_nonchemical_hard_gate_failed",
            },
        }
    else:
        dominant = float(chemical["dominant_topic_chemistry"]["mean_sos"])
        reference_dominant = float(
            reference["dominant_topic_chemistry"]["mean_sos"],
        )
        coverage = float(
            chemical["high_confidence_chemistry"]["sos_evaluable_coverage"],
        )
        reference_coverage = float(
            reference["high_confidence_chemistry"]["sos_evaluable_coverage"],
        )
        chemical_checks = {
            "dominant_topic_sos": _check(
                actual=dominant,
                reference=reference_dominant,
                limit=reference_dominant
                - hard_thresholds["dominant_sos_absolute_drop"],
                direction="minimum",
            ),
            "high_confidence_chemical_coverage": _check(
                actual=coverage,
                reference=reference_coverage,
                limit=reference_coverage
                * hard_thresholds[
                    "high_confidence_chemical_coverage_reference_fraction"
                ],
                direction="minimum",
            ),
        }
    hard["checks"].update(chemical_checks)
    hard["pass"] = all(row["pass"] for row in hard["checks"].values())

    metrics = evaluation["metrics"]
    competitive_thresholds = protocol["competitive_scorecard"]
    nll = float(metrics["test_document_completion"]["nll_per_token"])
    reference_nll = float(reference["document_completion"]["nll_per_token"])
    diversity = float(metrics["top_word_diversity"])
    reference_diversity = float(reference["top_word_diversity"])
    npmi = float(metrics["word_cooccurrence_npmi"]["mean_npmi"])
    reference_npmi = float(reference["word_cooccurrence_npmi"]["mean_npmi"])
    candidate_latency = float(
        metrics["cached_latency"]["median_seconds_per_spectrum"],
    )
    reference_latency = float(
        reference["cached_latency"]["median_seconds_per_spectrum"],
    )
    speedup = reference_latency / candidate_latency
    competitive_checks = {
        "heldout_nll": _check(
            actual=nll,
            reference=reference_nll,
            limit=reference_nll
            * competitive_thresholds["nll_reference_maximum_fraction"],
            direction="maximum",
        ),
        "top_word_diversity": _check(
            actual=diversity,
            reference=reference_diversity,
            limit=reference_diversity
            - competitive_thresholds["diversity_absolute_drop"],
            direction="minimum",
        ),
        "word_cooccurrence_npmi": _check(
            actual=npmi,
            reference=reference_npmi,
            limit=reference_npmi - competitive_thresholds["npmi_absolute_drop"],
            direction="minimum",
        ),
        "inference_speedup": {
            "pass": speedup >= competitive_thresholds["inference_speedup_minimum"],
            "actual": speedup,
            "minimum": competitive_thresholds["inference_speedup_minimum"],
            "stretch": competitive_thresholds["inference_speedup_stretch"],
            "stretch_pass": speedup
            >= competitive_thresholds["inference_speedup_stretch"],
            "reference_seconds_per_spectrum": reference_latency,
            "candidate_seconds_per_spectrum": candidate_latency,
        },
    }
    if chemical is None:
        competitive_checks.update(
            {
                "dominant_topic_sos": {
                    "pass": False,
                    "status": "not_run",
                },
                "high_confidence_chemical_coverage": {
                    "pass": False,
                    "status": "not_run",
                },
            },
        )
    else:
        dominant = float(chemical["dominant_topic_chemistry"]["mean_sos"])
        reference_dominant = float(
            reference["dominant_topic_chemistry"]["mean_sos"],
        )
        coverage = float(
            chemical["high_confidence_chemistry"]["sos_evaluable_coverage"],
        )
        reference_coverage = float(
            reference["high_confidence_chemistry"]["sos_evaluable_coverage"],
        )
        competitive_checks.update(
            {
                "dominant_topic_sos": _check(
                    actual=dominant,
                    reference=reference_dominant,
                    limit=reference_dominant
                    - competitive_thresholds["dominant_sos_absolute_drop"],
                    direction="minimum",
                ),
                "high_confidence_chemical_coverage": _check(
                    actual=coverage,
                    reference=reference_coverage,
                    limit=reference_coverage
                    * competitive_thresholds["chemical_coverage_reference_fraction"],
                    direction="minimum",
                ),
            },
        )
    competitive = {
        "checks": competitive_checks,
        "passed": sum(row["pass"] for row in competitive_checks.values()),
        "total": len(competitive_checks),
        "all_pass": all(row["pass"] for row in competitive_checks.values()),
        "role": "nonfatal_scorecard",
    }
    return {
        "attempt": evaluation["attempt"],
        "hard_viability": hard,
        "competitive_scorecard": competitive,
    }


def _gate_rows(scorecards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scorecard in scorecards:
        for tier in ("hard_viability", "competitive_scorecard"):
            for name, result in scorecard[tier]["checks"].items():
                rows.append(
                    {
                        "attempt": scorecard["attempt"],
                        "tier": tier,
                        "metric": name,
                        "pass": result["pass"],
                        "actual": result.get("actual", ""),
                        "reference": result.get("reference", ""),
                        "minimum": result.get("minimum", ""),
                        "maximum": result.get("maximum", ""),
                        "status": result.get("status", ""),
                    },
                )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    selected = report.get("selected_attempt") or "none"
    lines = [
        "# Fully neural MS2LDA bounded experiment",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Selected attempt: **{selected}**",
        "",
        "The hard viability gates decide whether a working neural discovery "
        "model exists. "
        "The competitive scorecard is descriptive and cannot veto a viable model.",
        "",
        "| Attempt | Hard viable | Competitive checks |",
        "| --- | ---: | ---: |",
    ]
    for row in report["attempts"]:
        competitive = row["competitive_scorecard"]
        lines.append(
            f"| {row['attempt']} | {row['hard_viability']['pass']} | "
            f"{competitive['passed']}/{competitive['total']} |",
        )
    lines.extend(["", "Evidence scope: corrected fixed split, seed 42.", ""])
    return "\n".join(lines)


def _outcome_issue(report: dict[str, Any]) -> str:
    if report["decision"] == "redirect_neural_work_to_motif_annotation":
        title = "Redirect neural work from motif discovery to motif interpretation"
        body = (
            "Both eligible fully neural attempts failed the predefined hard viability "
            "gates. Preserve this negative result and redirect the neural contribution "
            "to Mass2Motif annotation, cross-dataset matching, and substructure "
            "retrieval."
        )
    else:
        title = "Follow up the viable fully neural MS2LDA research model"
        body = (
            f"The `{report['selected_attempt']}` attempt passed every hard "
            "viability gate. "
            "Treat the competitive misses as focused neural-model research questions, "
            "not as grounds to discard the working model."
        )
    return (
        f"# {title}\n\n"
        f"Follow-up to #6.\n\n{body}\n\n"
        "See the committed bounded-experiment report and artifacts for the "
        "exact scorecard.\n"
    )


def build_report(
    run_dir: str | Path,
    *,
    attempts: list[str],
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build the final machine-readable, tabular, and prose decision artifacts."""
    directory = Path(run_dir).expanduser().resolve()
    scorecards = []
    for attempt in attempts:
        evaluation = read_json(directory / "evaluation" / attempt / "complete.json")
        chemical_path = directory / "chemical" / attempt / "complete.json"
        chemical = read_json(chemical_path) if chemical_path.is_file() else None
        scorecards.append(
            attempt_scorecard(
                evaluation,
                chemical=chemical,
                reference=reference,
                protocol=protocol,
            ),
        )
    viable = [row for row in scorecards if row["hard_viability"]["pass"]]
    viable.sort(
        key=lambda row: (
            -row["competitive_scorecard"]["passed"],
            read_json(
                directory / "evaluation" / row["attempt"] / "complete.json",
            )[
                "metrics"
            ]["test_document_completion"]["nll_per_token"],
        ),
    )
    selected = viable[0]["attempt"] if viable else None
    decision = (
        "retain_viable_fully_neural_research_model"
        if selected is not None
        else "redirect_neural_work_to_motif_annotation"
    )
    report = {
        "schema_version": "fully-neural-ms2lda/final-report-v1",
        "tracking_issue": protocol["tracking_issue"],
        "evidence_scope": protocol["evidence_scope"],
        "attempts": scorecards,
        "selected_attempt": selected,
        "decision": decision,
        "fallback_triggered": selected is None,
        "competitive_scorecard_is_nonfatal": True,
    }
    output = directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    (output / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    (output / "outcome_issue.md").write_text(_outcome_issue(report), encoding="utf-8")
    rows = _gate_rows(scorecards)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / "scorecard.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return report
