from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
RUN_DIR = ANALYSIS_DIR.parent
REPORT_DIR = RUN_DIR / "report"
GENERATED_AT = "2026-08-10T02:25:31.002265+00:00"
TITLE = "A simpler HybridLDA that keeps the useful parts"


shortlist = pd.read_csv(ANALYSIS_DIR / "shortlist_comparison.csv")
gate_results = pd.read_csv(ANALYSIS_DIR / "configuration_gate_results.csv")
summary = json.loads((ANALYSIS_DIR / "analysis_summary.json").read_text())


def selected(form: str, split: str) -> pd.Series:
    rows = shortlist[(shortlist["form"] == form) & (shortlist["split"] == split)]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {form!r} / {split!r}")
    return rows.iloc[0]


current_validation = selected("Current Hybrid", "validation")
current_test = selected("Current Hybrid", "test")
recommended_validation = selected("Recommended: topic-direct + 1 VB", "validation")
recommended_test = selected("Recommended: topic-direct + 1 VB", "test")
analytic_validation = selected("Zero-encoder: analytic + 2 VB", "validation")
analytic_test = selected("Zero-encoder: analytic + 2 VB", "test")


form_order = [
    "Current Hybrid",
    "Recommended: topic-direct + 1 VB",
    "Zero-encoder: analytic + 2 VB",
    "Neural, no VB: topic-direct",
    "Neural, no VB: DreaMS-direct",
    "No DreaMS anywhere",
]
short_labels = {
    "Current Hybrid": "Current: DreaMS encoder + 2 VB",
    "Recommended: topic-direct + 1 VB": "Recommended: topic-only encoder + 1 VB",
    "Zero-encoder: analytic + 2 VB": "Analytic initializer + 2 VB",
    "Neural, no VB: topic-direct": "Topic-only neural, 0 VB",
    "Neural, no VB: DreaMS-direct": "DreaMS neural, 0 VB",
    "No DreaMS anywhere": "Symmetric discovery + topic-only + 1 VB",
}


form_rows = []
for rank, form in enumerate(form_order, start=1):
    validation = selected(form, "validation")
    test = selected(form, "test")
    form_rows.append(
        {
            "rank": rank,
            "form": short_labels[form],
            "full_form": form,
            "discovery": validation["discovery"],
            "inference": validation["inference"],
            "vb_steps": int(validation["vb_steps"]),
            "uses_dreams_inference": bool(validation["uses_dreams_inference"]),
            "learned_encoder": bool(validation["learned_encoder"]),
            "parameter_count": int(validation["parameter_count"]),
            "checkpoint_mb": float(validation["checkpoint_bytes"] / 1_000_000),
            "validation_nll_change_pct": float(validation["nll_relative_pct"]),
            "test_nll_change_pct": float(test["nll_relative_pct"]),
            "validation_cosine_p05": float(validation["cosine_p05"]),
            "test_cosine_p05": float(test["cosine_p05"]),
            "validation_dominant_sos_delta": float(validation["dominant_sos_delta"]),
            "test_dominant_sos_delta": float(test["dominant_sos_delta"]),
            "validation_threshold_sos_delta": float(validation["threshold_sos_delta"]),
            "test_threshold_sos_delta": float(test["threshold_sos_delta"]),
            "validation_threshold_eligible_topics": int(validation["threshold_eligible_topics"]),
            "test_threshold_eligible_topics": int(test["threshold_eligible_topics"]),
            "validation_active_topics": int(validation["active_topics"]),
            "test_active_topics": int(test["active_topics"]),
            "validation_warm_latency_ms": float(validation["warm_latency_ms"]),
            "test_warm_latency_ms": float(test["warm_latency_ms"]),
            "validation_all_checks_pass": bool(validation["all_frozen_checks_pass"]),
            "validation_failed_checks": (
                str(validation["failed_checks"])
                if pd.notna(validation["failed_checks"])
                else ""
            ),
        }
    )


recommended_gate_rows = []
gate_definitions = [
    (
        "1 · Completion NLL",
        "No more than 2% worse",
        "nll_relative_change",
        lambda row: f"{100 * row['nll_relative_change']:+.3f}%",
    ),
    (
        "2 · Tail posterior agreement",
        "5th-percentile cosine delta ≥ -0.020",
        "cosine_p05_delta",
        lambda row: f"{row['cosine_p05_delta']:+.4f}",
    ),
    (
        "3 · Topic coherence",
        "NPMI delta ≥ -0.020",
        "npmi_delta",
        lambda row: f"{row['npmi_delta']:+.4f}",
    ),
    (
        "4 · Dominant-topic chemistry",
        "Mean SOS delta ≥ -0.020",
        "dominant_sos_delta",
        lambda row: f"{row['dominant_sos_delta']:+.4f}",
    ),
    (
        "5 · Dominant-topic coverage",
        "Coverage ratio ≥ 0.900",
        "dominant_coverage_ratio",
        lambda row: f"{row['dominant_coverage_ratio']:.3f}",
    ),
    (
        "6 · High-confidence chemistry",
        "Mean SOS delta ≥ -0.020",
        "threshold_sos_delta",
        lambda row: f"{row['threshold_sos_delta']:+.4f}",
    ),
    (
        "7 · High-confidence coverage",
        "Coverage ratio ≥ 0.900",
        "threshold_coverage_ratio",
        lambda row: f"{row['threshold_coverage_ratio']:.3f}",
    ),
]
for gate, floor, field, formatter in gate_definitions:
    validation = gate_results[
        (gate_results["split"] == "validation")
        & (gate_results["arm_id"] == "dreams_prior__topic_direct")
        & (gate_results["budget"] == 1)
    ].iloc[0]
    test = gate_results[
        (gate_results["split"] == "test")
        & (gate_results["arm_id"] == "dreams_prior__topic_direct")
        & (gate_results["budget"] == 1)
    ].iloc[0]
    recommended_gate_rows.append(
        {
            "gate": gate,
            "frozen_floor": floor,
            "validation_result": formatter(validation),
            "test_result": formatter(test),
            "status": "Pass",
            "validation_numeric": float(validation[field]),
            "test_numeric": float(test[field]),
            "baseline_arm": "dreams_prior__dreams_semi",
            "baseline_vb_steps": 2,
            "candidate_arm": "dreams_prior__topic_direct",
            "candidate_vb_steps": 1,
        }
    )


architecture_rows = [
    {
        "rank": 1,
        "model_form": "Recommended",
        "discovery_prior": "DreaMS-informed",
        "inference_initializer": "Topic-only direct encoder",
        "learned_parameters": int(recommended_validation["parameter_count"]),
        "local_vb_steps": 1,
        "dreams_needed_for_new_spectra": "No",
        "validation_result": "7 / 7 checks pass",
    },
    {
        "rank": 2,
        "model_form": "Zero-encoder runner-up",
        "discovery_prior": "DreaMS-informed",
        "inference_initializer": "Analytic topic evidence",
        "learned_parameters": 0,
        "local_vb_steps": 2,
        "dreams_needed_for_new_spectra": "No",
        "validation_result": "6 / 7 checks pass",
    },
    {
        "rank": 3,
        "model_form": "Current Hybrid",
        "discovery_prior": "DreaMS-informed",
        "inference_initializer": "DreaMS + topic semi-amortized encoder",
        "learned_parameters": int(current_validation["parameter_count"]),
        "local_vb_steps": 2,
        "dreams_needed_for_new_spectra": "Yes",
        "validation_result": "Reference",
    },
]


connection = sqlite3.connect(":memory:")
pd.DataFrame(form_rows).to_sql("model_forms", connection, index=False)
pd.DataFrame(recommended_gate_rows).to_sql(
    "recommended_gates",
    connection,
    index=False,
)
pd.DataFrame(architecture_rows).to_sql(
    "architecture_options",
    connection,
    index=False,
)

model_forms_sql = "SELECT * FROM model_forms ORDER BY rank"
posterior_forms_sql = (
    "SELECT * FROM model_forms WHERE discovery = 'dreams_prior' ORDER BY rank"
)
recommended_gates_sql = "SELECT * FROM recommended_gates ORDER BY gate"
architecture_options_sql = "SELECT * FROM architecture_options ORDER BY rank"

model_form_rows = pd.read_sql_query(model_forms_sql, connection).to_dict(
    orient="records",
)
posterior_form_rows = pd.read_sql_query(
    posterior_forms_sql,
    connection,
).to_dict(orient="records")
recommended_gate_rows = pd.read_sql_query(
    recommended_gates_sql,
    connection,
).to_dict(orient="records")
architecture_rows = pd.read_sql_query(
    architecture_options_sql,
    connection,
).to_dict(orient="records")
connection.close()


sources = [
    {
        "id": "frozen_bundle",
        "label": "Frozen HybridLDA simplification v8 result bundle",
        "query": {
            "description": "Mechanically collected factorial metrics and exact result manifests from the completed overnight run.",
            "language": "files",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "report/metrics.csv",
                "report/chemical_metrics.csv",
                "report/bootstrap.jsonl",
                "report/collection_summary.json",
                "verification.json",
            ],
            "filters": [
                "K=1000",
                "seed=42",
                "validation selected before the test pass",
                "current baseline=dreams_prior__dreams_semi at 2 VB steps",
            ],
            "metric_definitions": [
                "Completion NLL is token-weighted held-out negative log likelihood; lower is better.",
                "Posterior agreement is cosine similarity to the best 50-step local reference within the same discovery.",
                "SOS is averaged within eligible topics and then across eligible topics.",
                "Coverage is the fraction of 1000 annotated topics with at least one evaluable associated compound.",
            ],
        },
    },
    {
        "id": "analysis_notebook",
        "label": "Executed HybridLDA simplification analysis notebook",
        "path": "hybrid_lda_simplification_analysis.ipynb",
    },
    {
        "id": "prior_neural_evidence",
        "label": "Preserved amortized-LDA benchmarks at git commit b12278a",
        "query": {
            "description": "Historical random-start neural discovery and anchored semi-amortized benchmark results used to interpret collapse risk.",
            "language": "git",
            "id": "b12278a72ed18594bfede0b6c82f6cab212e48f9",
            "tables_used": [
                "docs/model/amortized_lda_benchmark.md",
                "docs/model/semi_amortized_lda_benchmark.md",
            ],
            "metric_definitions": [
                "The random-start neural LDA control used 1 of 200 topics; the anchored semi-amortized model used all 200 but did not match topic-identity stability.",
            ],
        },
    },
    {
        "id": "model_forms_query",
        "label": "Analyzed model-form comparison",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": model_forms_sql,
            "description": "Select the six reviewed model forms in the prespecified report order.",
            "executed_at": "2026-08-10T09:52:00+07:00",
            "tables_used": ["model_forms"],
            "filters": ["validation-selected shortlist", "current baseline at 2 VB steps"],
            "metric_definitions": [
                "validation_nll_change_pct is the candidate completion NLL relative to the current two-step Hybrid.",
                "validation_cosine_p05 is the fifth percentile of spectrum-level cosine to the discovery-specific 50-step reference.",
            ],
        },
    },
    {
        "id": "posterior_forms_query",
        "label": "DreaMS-prior posterior-fidelity comparison",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": posterior_forms_sql,
            "description": "Select only forms that share DreaMS-prior discovery and the same local-reference definition.",
            "executed_at": "2026-08-10T09:52:00+07:00",
            "tables_used": ["model_forms"],
            "filters": ["discovery=dreams_prior", "validation-selected shortlist"],
            "metric_definitions": [
                "validation_cosine_p05 is the fifth percentile of spectrum-level cosine to the common DreaMS-prior 50-step reference.",
            ],
        },
    },
    {
        "id": "recommended_gates_query",
        "label": "Recommended-form preservation checks",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": recommended_gates_sql,
            "description": "Select the seven frozen validation and test checks for the recommended form.",
            "executed_at": "2026-08-10T09:52:00+07:00",
            "tables_used": ["recommended_gates"],
            "filters": ["candidate=dreams_prior__topic_direct", "candidate VB steps=1"],
            "metric_definitions": [
                "Pass requires all seven frozen preservation checks to clear their original floors.",
            ],
        },
    },
    {
        "id": "architecture_options_query",
        "label": "Final architecture options",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": architecture_options_sql,
            "description": "Select the recommended form, zero-encoder runner-up, and current reference in decision order.",
            "executed_at": "2026-08-10T09:52:00+07:00",
            "tables_used": ["architecture_options"],
            "filters": ["decision-relevant architecture forms only"],
            "metric_definitions": [
                "Learned parameters count only the spectrum inference initializer, not the frozen global topic state.",
            ],
        },
    },
]


blocks = [
    {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
    {
        "id": "executive-summary",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """## Executive Summary

- **Use a simpler hybrid, not a fully neural replacement.** Keep the DreaMS-informed topic discovery, switch to a topic-evidence-only direct encoder, and use one local VB correction step.
- **The proposed form preserves the measured benefits.** It passes all seven frozen checks on validation and test, with completion NLL 0.78% and 0.82% lower than the current two-step Hybrid and essentially unchanged chemical scores.
- **The runtime becomes materially cleaner.** DreaMS is no longer needed for new-spectrum inference; learned inference parameters and checkpoint size fall 22%, while warm model-only latency falls 32% on validation and 36% on test.
- **Do not remove the final VB correction or the DreaMS discovery prior.** The zero-VB neural arms lose tail assignment fidelity, while DreaMS-free discovery gives up roughly 3.3% completion NLL and high-confidence chemical quality.""",
    },
    {
        "id": "zero-vb-finding",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """## The zero-VB route gives up the stabilized assignments

**No-VB neural inference looks attractive on likelihood, but it no longer reproduces the stabilized LDA assignment.** Against the current two-step model, the fifth-percentile cosine falls from 0.937 to 0.835 for the topic-only direct encoder and to 0.787 for the DreaMS direct encoder on validation. Both miss the frozen 0.917 preservation floor by a wide margin.

The corpus still uses about 350 topics because topic discovery was frozen in this study. That rules out collapse in local inference, but it does not make fully neural topic discovery safe: the earlier random-start neural discovery experiment collapsed to one active topic.""",
    },
    {"id": "posterior-chart", "type": "chart", "chartId": "posterior-fidelity", "layout": "full"},
    {
        "id": "posterior-interpretation",
        "type": "markdown",
        "layout": "full",
        "body": """The chart isolates the property the VB correction protects: the weakest tail of spectrum-level assignments. One correction step after the topic-only encoder stays just 0.0027 below the current model, comfortably inside the 0.020 allowance; removing VB loses 0.10–0.15 instead. **The final correction is earning its complexity.**""",
    },
    {
        "id": "dreams-finding",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """## DreaMS belongs in discovery, not in the inference hot path

**The structured prior has a measurable discovery benefit even though the learned topics look very similar.** Symmetric discovery matches DreaMS-prior topics at 0.999 mean cosine and 0.964 mean top-word Jaccard, yet its best simple count-only form is 3.34% worse on validation completion NLL and 3.12% worse on test. Its high-confidence SOS is also lower by 0.0469 on validation and 0.0364 on test.

This supports a cleaner division of labor: pay for DreaMS once when fitting the global topic model, then infer new spectra from their topic-word evidence without computing DreaMS embeddings.""",
    },
    {"id": "nll-chart", "type": "chart", "chartId": "nll-comparison", "layout": "full"},
    {
        "id": "nll-interpretation",
        "type": "markdown",
        "layout": "full",
        "body": """The no-DreaMS-discovery arm is the only candidate shown that breaches the frozen 2% likelihood allowance. The recommended form instead improves completion NLL while retaining the DreaMS-shaped global topics. **DreaMS can be removed from routine inference, but not from discovery if we want to preserve the complete performance profile.**""",
    },
    {
        "id": "recommended-finding",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """## One topic-only encoder plus one correction clears every check

**`dreams_prior__topic_direct` with one VB step is the strongest simplification that meets the rules set before looking at the results.** On validation, its completion NLL is 0.78% lower, fifth-percentile cosine is only 0.0027 lower, dominant-topic SOS is 0.0001 higher, and high-confidence SOS is 0.0140 higher. It retains 36 high-confidence eligible topics versus 37 for the current model, above the 90% floor.

The other one-step count-only passer, `topic_semi`, sits only 0.0010 above the validation tail-cosine floor; `topic_direct` has 0.0173 of headroom and retains 36 rather than 34 high-confidence eligible topics. The held-out test tells the same practical story for the selected form: NLL is 0.82% lower, the tail-cosine change is -0.0019, dominant-topic SOS changes by -0.0014, high-confidence SOS changes by -0.0054, and high-confidence coverage is unchanged at 55 topics.""",
    },
    {"id": "gate-table", "type": "table", "tableId": "recommended-gates", "layout": "full"},
    {
        "id": "footprint-finding",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """**The simplification is architectural as well as numerical.** The inference network shrinks from 745,320 to 579,048 parameters and the checkpoint from 2.99 MB to 2.32 MB. Warm model-only latency drops from 0.985 to 0.673 ms per spectrum on validation and from 1.592 to 1.015 ms on test. More importantly, the new-spectrum path no longer requires the DreaMS extractor or its feature cache.""",
    },
    {"id": "architecture-table", "type": "table", "tableId": "architecture-options", "layout": "full"},
    {
        "id": "analytic-runner-up",
        "type": "markdown",
        "sourceId": "analysis_notebook",
        "layout": "full",
        "body": """## The zero-encoder form is close, but not the defensible default

**The analytic initializer plus two VB steps is the cleanest conceptual model:** no inference network, no inference checkpoint, and no DreaMS embeddings for new spectra. It preserves NLL, posterior agreement, dominant-topic chemistry, and activity. However, its validation high-confidence coverage is 33 eligible topics versus 37 for the current model—89.2% against the frozen 90% floor. It passes that check on test at 52 versus 55 topics, but choosing it now would mean relaxing a rule after seeing the result.

Keep it as the parsimony benchmark and fallback. If a later focused check clears that one narrow coverage issue, it would be the next worthwhile simplification; there is no reason to rerun the whole factorial grid.""",
    },
    {
        "id": "next-steps",
        "type": "markdown",
        "layout": "full",
        "body": """## Recommended next steps

1. **Define the canonical model as DreaMS-prior discovery + topic-direct initializer + one local VB step.** Keep the current two-step DreaMS-conditioned model as a reproducibility reference, not the default inference path.
2. **Make inference explicitly count-only.** The production-facing interface should accept the spectrum count representation and frozen topic state; DreaMS extraction should exist only in the discovery/training workflow.
3. **Describe the encoder honestly as posterior distillation.** It is trained against 50-step VB targets and followed by one classical correction; it is not a VB-free neural topic model.
4. **Retain analytic + two VB steps as the zero-parameter ablation.** It is the right comparator for demonstrating whether the learned initializer earns its remaining complexity.""",
    },
    {
        "id": "further-questions",
        "type": "markdown",
        "layout": "full",
        "body": """## Further questions

- Can the analytic initializer recover the missing high-confidence topic with a very small, predeclared adjustment such as one additional refinement step? This is the only simplification question still worth a focused follow-up.
- Once implemented, does the count-only path deliver the expected end-to-end gain when preprocessing and serialization are timed together? The completed benchmark measured model-only latency for the simplification arms.""",
    },
    {
        "id": "caveats",
        "type": "markdown",
        "layout": "full",
        "body": """## Caveats and assumptions

- Chemical SOS is indirect evidence and should be read together with its eligible-topic coverage.
- Warm latency excludes feature generation; removing DreaMS from inference should remove its extractor cost, but that end-to-end count-only path was not separately timed here.
- This report interprets the completed seed-42 factorial bundle; the scope is recorded once here and not repeated throughout the findings.""",
    },
]


artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Decision report for the completed HybridLDA simplification factorial study.",
        "generatedAt": GENERATED_AT,
        "cards": [],
        "charts": [
            {
                "id": "posterior-fidelity",
                "title": "Validation tail posterior agreement",
                "subtitle": "Fifth-percentile cosine to each DreaMS-prior arm's 50-step local reference; frozen preservation floor shown",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which inference forms preserve the weakest tail of stabilized spectrum assignments?",
                "rationale": "Horizontal bars make the long model-form labels and distance from the frozen floor easy to compare.",
                "comparisonContext": {
                    "baseline": "Current DreaMS-conditioned Hybrid with 2 VB steps",
                    "denominator": "validation spectra with observed-count inference",
                    "grain": "model form",
                    "normalization": "cosine to the discovery-specific best 50-step local reference",
                    "unit": "cosine similarity",
                },
                "type": "horizontalBar",
                "dataset": "posterior_forms",
                "sourceId": "posterior_forms_query",
                "encodings": {
                    "x": {"field": "form", "type": "nominal", "label": "Model form"},
                    "y": {"field": "validation_cosine_p05", "type": "quantitative", "label": "5th-percentile cosine", "format": "number"},
                    "tooltip": [
                        {"field": "validation_nll_change_pct", "type": "quantitative", "label": "Validation NLL change", "unit": "%"},
                        {"field": "vb_steps", "type": "quantitative", "label": "VB steps"},
                        {"field": "validation_active_topics", "type": "quantitative", "label": "Active topics"},
                    ],
                },
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 5,
                "palette": {"kind": "sequential", "name": "blue"},
                "referenceLines": [
                    {"axis": "y", "value": float(current_validation["cosine_p05"] - 0.02), "label": "Preservation floor", "color": "neutral", "lineStyle": "dashed"},
                    {"axis": "y", "value": float(current_validation["cosine_p05"]), "label": "Current", "color": "neutral", "lineStyle": "dotted"},
                ],
                "labels": {"values": "all"},
                "settings": {"orientation": "horizontal", "sort": "custom", "showValues": True},
                "surface": {"surface": "card", "viewMode": "both", "showControls": False},
            },
            {
                "id": "nll-comparison",
                "title": "Validation completion NLL change",
                "subtitle": "Relative to the current two-step Hybrid; lower is better and the frozen worsening allowance is +2%",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which simplifications preserve held-out token prediction?",
                "rationale": "A signed horizontal bar chart separates improvements from the DreaMS-free discovery regression and keeps the 2% boundary visible.",
                "comparisonContext": {
                    "baseline": "Current DreaMS-conditioned Hybrid with 2 VB steps",
                    "denominator": "in-vocabulary completion tokens from 3,888 eligible validation spectra",
                    "grain": "model form",
                    "normalization": "candidate NLL divided by current NLL minus one",
                    "unit": "%",
                },
                "type": "horizontalBar",
                "dataset": "model_forms",
                "sourceId": "model_forms_query",
                "encodings": {
                    "x": {"field": "form", "type": "nominal", "label": "Model form"},
                    "y": {"field": "validation_nll_change_pct", "type": "quantitative", "label": "Relative NLL change", "format": "number", "unit": "%"},
                    "tooltip": [
                        {"field": "validation_cosine_p05", "type": "quantitative", "label": "5th-percentile cosine"},
                        {"field": "validation_threshold_sos_delta", "type": "quantitative", "label": "High-confidence SOS delta"},
                        {"field": "parameter_count", "type": "quantitative", "label": "Learned parameters"},
                    ],
                },
                "valueFormat": "number",
                "unit": "%",
                "layout": "full",
                "maxRows": 6,
                "palette": {"kind": "diverging", "name": "blue-orange", "midpoint": 0},
                "referenceLines": [
                    {"axis": "y", "value": 0, "label": "Current", "color": "neutral", "lineStyle": "dotted"},
                    {"axis": "y", "value": 2, "label": "Worsening allowance", "color": "neutral", "lineStyle": "dashed"},
                ],
                "labels": {"values": "all"},
                "settings": {"orientation": "horizontal", "sort": "custom", "showValues": True},
                "surface": {"surface": "card", "viewMode": "both", "showControls": False},
            },
        ],
        "tables": [
            {
                "id": "recommended-gates",
                "title": "Preservation checks for the recommended form",
                "subtitle": "Validation selects the form; test confirms the same seven checks against the current two-step Hybrid",
                "showDescription": True,
                "dataset": "recommended_gates",
                "defaultSort": {"field": "gate", "direction": "asc"},
                "density": "spacious",
                "sourceId": "recommended_gates_query",
                "layout": "full",
                "columns": [
                    {"field": "gate", "label": "Check", "type": "text"},
                    {"field": "frozen_floor", "label": "Frozen floor", "type": "text"},
                    {"field": "validation_result", "label": "Validation", "type": "text"},
                    {"field": "test_result", "label": "Test", "type": "text"},
                    {"field": "status", "label": "Result", "type": "text"},
                ],
            },
            {
                "id": "architecture-options",
                "title": "Final architecture choices",
                "subtitle": "The recommended form is the simplest tested architecture that clears every frozen validation check",
                "showDescription": True,
                "dataset": "architecture_options",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "spacious",
                "sourceId": "architecture_options_query",
                "layout": "full",
                "columns": [
                    {"field": "rank", "label": "Order", "format": "number", "type": "number"},
                    {"field": "model_form", "label": "Form", "type": "text"},
                    {"field": "discovery_prior", "label": "Discovery prior", "type": "text"},
                    {"field": "inference_initializer", "label": "Inference initializer", "type": "text"},
                    {"field": "learned_parameters", "label": "Learned parameters", "format": "compact", "type": "compact"},
                    {"field": "local_vb_steps", "label": "VB steps", "format": "number", "type": "number"},
                    {"field": "dreams_needed_for_new_spectra", "label": "DreaMS at inference", "type": "text"},
                    {"field": "validation_result", "label": "Validation", "type": "text"},
                ],
            },
        ],
        "sources": sources,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "model_forms": model_form_rows,
            "posterior_forms": posterior_form_rows,
            "recommended_gates": recommended_gate_rows,
            "architecture_options": architecture_rows,
        },
    },
    "sources": sources,
}


(ANALYSIS_DIR / "artifact.json").write_text(
    json.dumps(artifact, indent=2, sort_keys=False) + "\n",
    encoding="utf-8",
)
print(ANALYSIS_DIR / "artifact.json")
