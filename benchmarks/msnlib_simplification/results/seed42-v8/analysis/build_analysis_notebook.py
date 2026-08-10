from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ANALYSIS_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = ANALYSIS_DIR / "hybrid_lda_simplification_analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# HybridLDA simplification analysis

## tl;dr

- **Recommended final form:** keep the DreaMS-informed prior for topic discovery, replace the DreaMS-conditioned semi-amortized inference network with the topic-evidence-only direct-regression encoder, and run **one** local variational-Bayes (VB) correction step.
- This form clears every frozen preservation check on validation and again on test. Relative to the current two-step Hybrid inference, it removes DreaMS from the inference path, cuts the learned inference parameters and checkpoint by 22%, and reduces measured warm model-only latency by roughly one third.
- A completely neural, zero-VB inference path is not supported: its lower completion NLL comes with a material loss of agreement with the stabilized local posterior, especially in the weakest 5% of spectra.
- The zero-parameter analytic initializer plus two VB steps is the cleanest runner-up, but it narrowly misses the frozen high-confidence chemical-coverage check on validation (33 versus 37 eligible topics; 89.2% of the current coverage against a 90% floor).
""",
    ),
    markdown(
        """
## Context & Methods

The decision is whether the current HybridLDA can be simplified without giving up the properties that motivated it. The current comparison point is `dreams_prior__dreams_semi` with two local VB steps.

The frozen preservation checks are: completion NLL no more than 2% worse; mean NPMI no more than 0.02 worse; fifth-percentile cosine to the discovery-specific 50-step reference no more than 0.02 worse; mean SOS no more than 0.02 worse in each chemical association view; and SOS-evaluable coverage at least 90% of the current model in each view. Topic activity is inspected separately because earlier fully neural discovery runs collapsed.

### Key Assumptions

- Lower completion NLL is better; higher cosine, SOS, NPMI, diversity, and coverage are better.
- Validation determines the recommendation; test is used as the held-out confirmation.
- Cross-discovery cosine values are not treated as directly comparable because each discovery has its own local reference. The DreaMS-free discovery arm is judged on NLL, chemistry, diversity, topic matching, and activity.
- The chemical score is averaged first within each eligible topic and then across eligible topics. Both the score and its exact eligible-topic denominator are retained.
""",
    ),
    markdown("## Data"),
    code(
        """
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

analysis_dir = Path.cwd()
run_dir = analysis_dir.parent
report_dir = run_dir / "report"

baseline_arm = "dreams_prior__dreams_semi"
baseline_budget = 2

verification = json.loads((run_dir / "verification.json").read_text())
overnight = json.loads((run_dir / "overnight_complete.json").read_text())
report_complete = json.loads((report_dir / "complete.json").read_text())
recovery = json.loads((run_dir / "recovery_provenance.json").read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_checks = {
    name: sha256(report_dir / name) == expected
    for name, expected in report_complete["output_sha256"].items()
}

assert verification["complete"] is True
assert verification["missing_artifacts"] == 0
assert verification["required_artifacts"] == 226
assert overnight["complete"] is True and not overnight["failed"] and not overnight["skipped"]
assert all(hash_checks.values())
assert recovery["retraining_performed"] is False
assert recovery["inference_repeated"] is False
assert recovery["frozen_source"]["all_files_match_frozen_manifest"] is True

metrics = pd.read_csv(report_dir / "metrics.csv")
chemical = pd.read_csv(report_dir / "chemical_metrics.csv")
bootstrap = pd.read_json(report_dir / "bootstrap.jsonl", lines=True)

assert len(metrics) == 80
assert len(chemical) == 160
assert len(bootstrap) == 480

integrity = pd.DataFrame(
    [
        {"check": "Required result artifacts", "value": verification["required_artifacts"], "status": "pass"},
        {"check": "Missing result artifacts", "value": verification["missing_artifacts"], "status": "pass"},
        {"check": "Factorial metric rows", "value": len(metrics), "status": "pass"},
        {"check": "Chemical summary rows", "value": len(chemical), "status": "pass"},
        {"check": "Scaffold-bootstrap rows", "value": len(bootstrap), "status": "pass"},
        {"check": "Frozen source files verified", "value": recovery["frozen_source"]["files_verified"], "status": "pass"},
    ]
)
display(integrity)
""",
    ),
    markdown("## Results"),
    code(
        """
def row_for(frame: pd.DataFrame, split: str, arm_id: str, budget: int) -> pd.Series:
    selected = frame[(frame["split"] == split) & (frame["arm_id"] == arm_id) & (frame["budget"] == budget)]
    assert len(selected) == 1
    return selected.iloc[0]

def chemical_rows(split: str, arm_id: str, budget: int) -> pd.DataFrame:
    selected = chemical[
        (chemical["split"] == split)
        & (chemical["arm_id"] == arm_id)
        & (chemical["budget"] == budget)
    ].set_index("association_mode")
    assert set(selected.index) == {"dominant_topic", "probability_ge_frozen_threshold"}
    return selected

gate_rows = []
for split in ("validation", "test"):
    baseline = row_for(metrics, split, baseline_arm, baseline_budget)
    baseline_chemical = chemical_rows(split, baseline_arm, baseline_budget)
    for _, candidate in metrics[metrics["split"] == split].iterrows():
        candidate_chemical = chemical_rows(split, candidate["arm_id"], int(candidate["budget"]))
        dominant = candidate_chemical.loc["dominant_topic"]
        dominant_base = baseline_chemical.loc["dominant_topic"]
        threshold = candidate_chemical.loc["probability_ge_frozen_threshold"]
        threshold_base = baseline_chemical.loc["probability_ge_frozen_threshold"]
        values = {
            "nll_relative_change": candidate["nll_per_token"] / baseline["nll_per_token"] - 1.0,
            "cosine_p05_delta": candidate["cosine_p05"] - baseline["cosine_p05"],
            "npmi_delta": candidate["npmi"] - baseline["npmi"],
            "dominant_sos_delta": dominant["mean_sos"] - dominant_base["mean_sos"],
            "dominant_coverage_ratio": dominant["sos_evaluable_coverage"] / dominant_base["sos_evaluable_coverage"],
            "threshold_sos_delta": threshold["mean_sos"] - threshold_base["mean_sos"],
            "threshold_coverage_ratio": threshold["sos_evaluable_coverage"] / threshold_base["sos_evaluable_coverage"],
        }
        checks = {
            "nll": values["nll_relative_change"] <= 0.02,
            "p05_cosine": values["cosine_p05_delta"] >= -0.02,
            "npmi": values["npmi_delta"] >= -0.02,
            "dominant_sos": values["dominant_sos_delta"] >= -0.02,
            "dominant_coverage": values["dominant_coverage_ratio"] >= 0.90,
            "threshold_sos": values["threshold_sos_delta"] >= -0.02,
            "threshold_coverage": values["threshold_coverage_ratio"] >= 0.90,
        }
        gate_rows.append(
            {
                "split": split,
                "arm_id": candidate["arm_id"],
                "discovery": candidate["discovery"],
                "inference": candidate["inference"],
                "budget": int(candidate["budget"]),
                **values,
                "checks_passed": sum(checks.values()),
                "all_checks_pass": all(checks.values()),
                "failed_checks": ", ".join(key for key, passed in checks.items() if not passed),
                "active_topics": int(candidate["corpus_active_topics"]),
                "parameter_count": int(candidate["parameter_count"]),
                "checkpoint_bytes": int(candidate["checkpoint_bytes"]),
                "warm_latency_ms": 1000.0 * candidate["warm_seconds_per_spectrum_median"],
            }
        )

gate_results = pd.DataFrame(gate_rows)
gate_results.to_csv(analysis_dir / "configuration_gate_results.csv", index=False)

validation_passers = gate_results[
    (gate_results["split"] == "validation") & gate_results["all_checks_pass"]
].sort_values(["budget", "parameter_count", "warm_latency_ms"])
display(
    validation_passers[
        ["arm_id", "budget", "nll_relative_change", "cosine_p05_delta", "dominant_sos_delta", "threshold_sos_delta", "parameter_count", "warm_latency_ms"]
    ].style.format(
        {
            "nll_relative_change": "{:+.3%}",
            "cosine_p05_delta": "{:+.4f}",
            "dominant_sos_delta": "{:+.4f}",
            "threshold_sos_delta": "{:+.4f}",
            "warm_latency_ms": "{:.3f}",
        }
    )
)
""",
    ),
    code(
        """
forms = {
    "Current Hybrid": ("dreams_prior__dreams_semi", 2),
    "Recommended: topic-direct + 1 VB": ("dreams_prior__topic_direct", 1),
    "Zero-encoder: analytic + 2 VB": ("dreams_prior__analytic", 2),
    "Neural, no VB: topic-direct": ("dreams_prior__topic_direct", 0),
    "Neural, no VB: DreaMS-direct": ("dreams_prior__dreams_direct", 0),
    "No DreaMS anywhere": ("symmetric_prior__topic_direct", 1),
}

uses_dreams_inference = {
    "dreams_semi": True,
    "dreams_direct": True,
    "topic_semi": False,
    "topic_direct": False,
    "analytic": False,
}

shortlist_rows = []
for form, (arm_id, budget) in forms.items():
    for split in ("validation", "test"):
        metric = row_for(metrics, split, arm_id, budget)
        baseline = row_for(metrics, split, baseline_arm, baseline_budget)
        chem = chemical_rows(split, arm_id, budget)
        base_chem = chemical_rows(split, baseline_arm, baseline_budget)
        gate = gate_results[
            (gate_results["split"] == split)
            & (gate_results["arm_id"] == arm_id)
            & (gate_results["budget"] == budget)
        ].iloc[0]
        shortlist_rows.append(
            {
                "form": form,
                "split": split,
                "arm_id": arm_id,
                "discovery": metric["discovery"],
                "inference": metric["inference"],
                "vb_steps": budget,
                "uses_dreams_inference": uses_dreams_inference[metric["inference"]],
                "learned_encoder": metric["inference"] != "analytic",
                "parameter_count": int(metric["parameter_count"]),
                "checkpoint_bytes": int(metric["checkpoint_bytes"]),
                "nll_per_token": metric["nll_per_token"],
                "nll_relative_pct": 100.0 * (metric["nll_per_token"] / baseline["nll_per_token"] - 1.0),
                "cosine_mean": metric["cosine_mean"],
                "cosine_p05": metric["cosine_p05"],
                "js_mean": metric["js_mean"],
                "active_topics": int(metric["corpus_active_topics"]),
                "npmi": metric["npmi"],
                "top_word_diversity": metric["top_word_diversity"],
                "dominant_sos": chem.loc["dominant_topic", "mean_sos"],
                "dominant_eligible_topics": int(chem.loc["dominant_topic", "eligible_topics"]),
                "dominant_sos_delta": chem.loc["dominant_topic", "mean_sos"] - base_chem.loc["dominant_topic", "mean_sos"],
                "threshold_sos": chem.loc["probability_ge_frozen_threshold", "mean_sos"],
                "threshold_eligible_topics": int(chem.loc["probability_ge_frozen_threshold", "eligible_topics"]),
                "threshold_sos_delta": chem.loc["probability_ge_frozen_threshold", "mean_sos"] - base_chem.loc["probability_ge_frozen_threshold", "mean_sos"],
                "warm_latency_ms": 1000.0 * metric["warm_seconds_per_spectrum_median"],
                "all_frozen_checks_pass": bool(gate["all_checks_pass"]),
                "failed_checks": gate["failed_checks"],
            }
        )

shortlist = pd.DataFrame(shortlist_rows)
shortlist.to_csv(analysis_dir / "shortlist_comparison.csv", index=False)

display(
    shortlist[shortlist["split"] == "validation"]
    [["form", "nll_relative_pct", "cosine_p05", "dominant_sos_delta", "threshold_sos_delta", "threshold_eligible_topics", "active_topics", "parameter_count", "warm_latency_ms", "all_frozen_checks_pass", "failed_checks"]]
    .style.format(
        {
            "nll_relative_pct": "{:+.3f}%",
            "cosine_p05": "{:.4f}",
            "dominant_sos_delta": "{:+.4f}",
            "threshold_sos_delta": "{:+.4f}",
            "warm_latency_ms": "{:.3f}",
        }
    )
)
""",
    ),
    code(
        """
heldout_records = [
    json.loads(line)
    for line in (run_dir / "shared/counts/heldout_records.jsonl").read_text().splitlines()
    if line
]

def grouped_mean_interval(values: np.ndarray, groups: list[str], *, seed: int, replicates: int = 2000):
    values = np.asarray(values, dtype=np.float64)
    unique = sorted(set(groups))
    group_index = {group: index for index, group in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for value, group in zip(values, groups, strict=True):
        if np.isfinite(value):
            index = group_index[group]
            sums[index] += value
            counts[index] += 1
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        denominator = counts[selected].sum()
        draws[replicate] = sums[selected].sum() / denominator
    finite = np.isfinite(values)
    return {
        "estimate": float(values[finite].mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "groups": len(unique),
        "observations": int(finite.sum()),
        "replicates": replicates,
    }

paired_rows = []
for split_index, split in enumerate(("validation", "test")):
    groups = [str(row["scaffold_key"]) for row in heldout_records if row["split"] == split]
    baseline_values = np.load(
        run_dir / "evaluation" / split / "observed" / "arms" / baseline_arm / f"per_document_{baseline_budget}.npz"
    )
    for form, (arm_id, budget) in forms.items():
        candidate_values = np.load(
            run_dir / "evaluation" / split / "observed" / "arms" / arm_id / f"per_document_{budget}.npz"
        )
        for metric_name in ("nll_per_token", "cosine_to_reference", "js_to_reference"):
            delta = np.asarray(candidate_values[metric_name]) - np.asarray(baseline_values[metric_name])
            interval = grouped_mean_interval(
                delta,
                groups,
                seed=42 + split_index * 100_000 + budget * 1_000,
            )
            paired_rows.append(
                {
                    "form": form,
                    "split": split,
                    "arm_id": arm_id,
                    "budget": budget,
                    "metric": metric_name,
                    **interval,
                }
            )

paired = pd.DataFrame(paired_rows)
paired.to_csv(analysis_dir / "paired_bootstrap_vs_current.csv", index=False)

display(
    paired[(paired["form"] == "Recommended: topic-direct + 1 VB")]
    .pivot(index="metric", columns="split", values=["estimate", "ci_low", "ci_high"])
    .style.format("{:+.6f}")
)
""",
    ),
    code(
        """
activity_rows = []
for form, (arm_id, budget) in forms.items():
    for split in ("validation", "test"):
        payload = json.loads(
            (
                run_dir
                / "evaluation"
                / split
                / "observed"
                / "arms"
                / arm_id
                / "metrics_complete.json"
            ).read_text()
        )
        activity = payload["metrics"][str(budget)]["active_topics"]
        activity_rows.append({"form": form, "split": split, **activity})

activity = pd.DataFrame(activity_rows)
activity.to_csv(analysis_dir / "topic_activity.csv", index=False)
display(activity)

recommended = shortlist[shortlist["form"] == "Recommended: topic-direct + 1 VB"].set_index("split")
current = shortlist[shortlist["form"] == "Current Hybrid"].set_index("split")

footprint = pd.DataFrame(
    [
        {
            "measure": "Learned inference parameters",
            "current": int(current.loc["validation", "parameter_count"]),
            "recommended": int(recommended.loc["validation", "parameter_count"]),
            "relative_change": recommended.loc["validation", "parameter_count"] / current.loc["validation", "parameter_count"] - 1,
        },
        {
            "measure": "Checkpoint bytes",
            "current": int(current.loc["validation", "checkpoint_bytes"]),
            "recommended": int(recommended.loc["validation", "checkpoint_bytes"]),
            "relative_change": recommended.loc["validation", "checkpoint_bytes"] / current.loc["validation", "checkpoint_bytes"] - 1,
        },
        {
            "measure": "Validation warm model-only latency, ms/spectrum",
            "current": current.loc["validation", "warm_latency_ms"],
            "recommended": recommended.loc["validation", "warm_latency_ms"],
            "relative_change": recommended.loc["validation", "warm_latency_ms"] / current.loc["validation", "warm_latency_ms"] - 1,
        },
        {
            "measure": "Test warm model-only latency, ms/spectrum",
            "current": current.loc["test", "warm_latency_ms"],
            "recommended": recommended.loc["test", "warm_latency_ms"],
            "relative_change": recommended.loc["test", "warm_latency_ms"] / current.loc["test", "warm_latency_ms"] - 1,
        },
    ]
)
footprint.to_csv(analysis_dir / "recommended_footprint.csv", index=False)
display(footprint.style.format({"current": "{:.4f}", "recommended": "{:.4f}", "relative_change": "{:+.1%}"}))
""",
    ),
    code(
        """
topic_matching = json.loads((run_dir / "evaluation/topic_matching.json").read_text())
matching_metrics = topic_matching["metrics"]

recommended_validation = shortlist[
    (shortlist["form"] == "Recommended: topic-direct + 1 VB") & (shortlist["split"] == "validation")
].iloc[0]
recommended_test = shortlist[
    (shortlist["form"] == "Recommended: topic-direct + 1 VB") & (shortlist["split"] == "test")
].iloc[0]
analytic_validation = shortlist[
    (shortlist["form"] == "Zero-encoder: analytic + 2 VB") & (shortlist["split"] == "validation")
].iloc[0]

analysis_summary = {
    "decision": "recommend_simplified_hybrid",
    "recommended_form": {
        "discovery": "dreams_prior",
        "inference": "topic_direct",
        "local_vb_steps": 1,
        "uses_dreams_at_inference": False,
        "learned_inference_parameters": int(recommended_validation["parameter_count"]),
    },
    "validation": recommended_validation.to_dict(),
    "test": recommended_test.to_dict(),
    "parameter_reduction_fraction": float(1 - recommended_validation["parameter_count"] / current.loc["validation", "parameter_count"]),
    "validation_latency_reduction_fraction": float(1 - recommended_validation["warm_latency_ms"] / current.loc["validation", "warm_latency_ms"]),
    "test_latency_reduction_fraction": float(1 - recommended_test["warm_latency_ms"] / current.loc["test", "warm_latency_ms"]),
    "analytic_runner_up_validation": analytic_validation.to_dict(),
    "topic_matching": {
        "matched_cosine_mean": matching_metrics["matched_cosine_mean"],
        "top_word_jaccard_mean": matching_metrics["top_word_jaccard_mean"],
    },
    "bundle_integrity": {
        "required_artifacts": verification["required_artifacts"],
        "missing_artifacts": verification["missing_artifacts"],
        "frozen_source_files_verified": recovery["frozen_source"]["files_verified"],
    },
}

(analysis_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2, sort_keys=True, default=str) + "\\n")
display(pd.json_normalize(analysis_summary["recommended_form"]))
""",
    ),
    markdown(
        """
## Takeaways

1. **Do not replace classical topic discovery with a fully neural objective.** The controlled no-VB arms only changed local inference because their topics were frozen. They therefore do not overturn the earlier random-start collapse result for neural topic discovery.
2. **Do not remove DreaMS from discovery.** The symmetric-prior discovery is extremely close in top words, but it gives up about 3.3% completion NLL and materially lowers high-confidence SOS, so it fails the frozen preservation rule.
3. **Remove DreaMS from routine inference and halve the VB budget.** `dreams_prior__topic_direct` with one VB step is the simplest tested form that clears every frozen check on both splits. It retains the stabilized LDA correction while making inference count-only.
4. **Keep analytic + two VB steps as the parsimony benchmark, not the default.** It removes the learned encoder entirely and otherwise performs very well, but the validation high-confidence coverage is 33 eligible topics versus 37 for the current model, just below the frozen 90% floor.

The resulting model is still a hybrid, but the division of labor is cleaner: DreaMS contributes only a structured prior during topic discovery; a small count-conditioned network initializes a spectrum; and one classical update makes the final assignment conform to the learned LDA topics.
""",
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
