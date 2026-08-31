"""Generate the Contextual Sparse ETM paper tables from frozen evidence.

The report deliberately consumes only training, synthetic, and validation
artifacts.  The current model's test partition is outside this evidence set.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "docs/research/generated"

PREPARATION = (
    REPO
    / "research/etm_ecrtm_msnlib/local_results/20260827_seed42_validation"
    / "preparation_summary.json"
)
PROTOCOL = REPO / "benchmarks/neural_ms2lda/protocol.json"
LOCKED_COMPARATOR = REPO / "benchmarks/neural_ms2lda/results/seed42/results.json"
ROUTING = REPO / "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm"
STABILITY = (
    REPO / "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_stability"
)

EXPECTED_METHOD = "etm_balanced_routing_top2_entmax15_raw_counts"
EXPECTED_TRAINING_SEEDS = [7043, 23, 37]
EXPECTED_SYNTHETIC_FORMULATIONS = 4
EXPECTED_HIGH_K_ROWS = 3


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        msg = f"missing {key!r} in {row}"
        raise ValueError(msg)
    return float(value)


def _integer(row: dict[str, str], key: str) -> int:
    return round(_float(row, key))


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        msg = f"{name} changed: {actual} != {expected}"
        raise ValueError(msg)


def _command(name: str, value: str | int) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _tex_integer(value: int) -> str:
    """Format a large integer with non-breaking mathematical digit grouping."""
    return f"{int(value):,}".replace(",", r"{,}")


def _write(name: str, lines: list[str]) -> str:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.relative_to(REPO))


def _bold_if(value: str, *, condition: bool) -> str:
    return rf"\textbf{{{value}}}" if condition else value


def _validate_and_load() -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    preparation = _json(PREPARATION)
    protocol = _json(PROTOCOL)
    comparator = _json(LOCKED_COMPARATOR)
    checkpoint = _json(ROUTING / "checkpoint_manifest.json")
    config = _json(ROUTING / "config.json")
    metrics = _json(ROUTING / "metrics.json")
    comparison = {row["model"]: row for row in _csv(ROUTING / "comparison.csv")}
    synthetic = _csv(ROUTING / "synthetic_summary.csv")
    high_k = _csv(ROUTING / "high_k_stress.csv")
    stability = _json(STABILITY / "stability_summary.json")

    if checkpoint["method"] != EXPECTED_METHOD or config["method"] != EXPECTED_METHOD:
        msg = "Contextual Sparse ETM method identity changed"
        raise ValueError(msg)
    if checkpoint["evidence_boundary"] != (
        "training plus validation only; candidate test remains locked"
    ):
        msg = "checkpoint evidence boundary changed"
        raise ValueError(msg)
    if config["candidate_test_artifacts_accessed"] is not False:
        msg = "current-model test artifacts must remain untouched"
        raise ValueError(msg)
    if stability["direction_checks"]["candidate_test_remained_locked"] is not True:
        msg = "stability evidence no longer preserves the test boundary"
        raise ValueError(msg)
    if (
        stability["runs"] != len(EXPECTED_TRAINING_SEEDS)
        or stability["training_seeds"] != EXPECTED_TRAINING_SEEDS
    ):
        msg = "expected exactly the three frozen training seeds"
        raise ValueError(msg)

    data = preparation["data"]
    if data["leakage_audit"]["leaked_compounds"] != 0:
        msg = "compound leakage detected"
        raise ValueError(msg)
    if data["leakage_audit"]["leaked_groups"] != 0:
        msg = "split-group leakage detected"
        raise ValueError(msg)
    if data["split"]["seed"] != protocol["seed"]:
        msg = "preparation and protocol split seeds differ"
        raise ValueError(msg)
    if data["vocabulary"]["vocabulary_size"] != preparation["vocabulary_size"]:
        msg = "vocabulary size mismatch"
        raise ValueError(msg)

    # Frozen comparison artifacts retain the experiment-time method key; all
    # rendered report labels use the final public model name.
    required_models = {
        "canonical ETM",
        "balanced ETM",
        "routing-informed sparse ETM",
    }
    if not required_models.issubset(comparison):
        msg = "comparison is missing a required ETM baseline"
        raise ValueError(msg)
    proposed_row = comparison["routing-informed sparse ETM"]
    if proposed_row["finite_stable"] != "True":
        msg = "Contextual Sparse ETM is not marked finite and stable"
        raise ValueError(msg)
    chemistry = metrics["validation_chemistry"]
    completion = metrics["document_completion"]
    for key, source_key in (
        ("optimized_motifs", "optimized_motifs"),
        ("evaluable_motifs", "eligible_topics"),
        ("useful_motifs", "useful_motifs"),
    ):
        if _integer(proposed_row, key) != int(chemistry[source_key]):
            msg = f"Contextual Sparse ETM {key} disagrees with metrics.json"
            raise ValueError(msg)
    _close(_float(proposed_row, "mean_sos"), chemistry["mean_sos"], name="mean SOS")
    _close(
        _float(proposed_row, "completion_nll"),
        completion["nll_per_token"],
        name="completion NLL",
    )
    if metrics["parameters"] != config["context_parameters"] + 19_278_000:
        msg = "unexpected Contextual Sparse ETM parameter count"
        raise ValueError(msg)

    methods = {row["method"]: row for row in comparator["methods"]}
    if "tomotopy" not in methods:
        msg = "locked Tomotopy comparator is missing"
        raise ValueError(msg)
    tomotopy = methods["tomotopy"]
    tomotopy_validation = tomotopy["validation"]
    tomotopy_nll = comparator["secondary"]["completion_nll_per_token"]["tomotopy"]
    expected_tomotopy = checkpoint["expected_validation_metrics"]["tomotopy"]
    for key, source_key in (
        ("optimized_motifs", "optimized_motifs"),
        ("evaluable_motifs", "high_confidence_evaluable_motifs"),
        ("useful_motifs", "useful_high_confidence_motifs"),
    ):
        if int(expected_tomotopy[key]) != int(tomotopy_validation[source_key]):
            msg = f"Tomotopy {key} changed"
            raise ValueError(msg)
    _close(
        expected_tomotopy["completion_nll"],
        tomotopy_nll["validation"],
        name="Tomotopy completion NLL",
    )

    if len(synthetic) != EXPECTED_SYNTHETIC_FORMULATIONS or {
        int(row["seeds"]) for row in synthetic
    } != {len(EXPECTED_TRAINING_SEEDS)}:
        msg = "expected four three-seed K=36 synthetic formulations"
        raise ValueError(msg)
    if {int(row["k"]) for row in synthetic} != {36}:
        msg = "synthetic summary K changed"
        raise ValueError(msg)
    if len(high_k) != EXPECTED_HIGH_K_ROWS or {
        int(row["fitted_topics"]) for row in high_k
    } != {128}:
        msg = "expected three K=128 rows"
        raise ValueError(msg)
    if {int(row["true_topics"]) for row in high_k} != {18}:
        msg = "high-K planted topic count changed"
        raise ValueError(msg)

    return {
        "preparation": preparation,
        "protocol": protocol,
        "config": config,
        "metrics": metrics,
        "comparison": comparison,
        "synthetic": synthetic,
        "high_k": high_k,
        "stability": stability,
        "tomotopy": tomotopy,
        "tomotopy_nll": float(tomotopy_nll["validation"]),
    }


def _generate_macros(evidence: dict[str, Any]) -> str:
    preparation = evidence["preparation"]
    data = preparation["data"]
    metrics = evidence["metrics"]
    config = evidence["config"]
    canonical = evidence["comparison"]["canonical ETM"]
    balanced = evidence["comparison"]["balanced ETM"]
    tomotopy = evidence["tomotopy"]["validation"]
    stability = evidence["stability"]
    chemistry = metrics["validation_chemistry"]
    completion = metrics["document_completion"]
    theta = metrics["theta_support"]
    inventory = metrics["topic_inventory"]
    runtime = metrics["runtime"]
    memory = runtime["memory"]
    aggregate = stability["aggregate"]
    training_wall = aggregate["training_wall_seconds"]

    def pct(numerator: float, denominator: float) -> str:
        return f"{100.0 * numerator / denominator:.1f}\\%"

    values: dict[str, str | int] = {
        "SourceSpectra": data["parsing"]["parsed_blocks"],
        "RetainedSpectra": data["parsing"]["retained_spectra"],
        "TrainingSpectra": data["split"]["spectrum_counts"]["train"],
        "ValidationSpectra": data["split"]["spectrum_counts"]["validation"],
        "TestSpectra": data["split"]["spectrum_counts"]["test"],
        "ConnectivityGroups": data["leakage_audit"]["connectivity_groups"],
        "SplitGroups": data["leakage_audit"]["split_groups"],
        "VocabularySize": data["vocabulary"]["vocabulary_size"],
        "StudyTopics": config["topics"],
        "StudyThreads": config["threads"],
        "ContextualParameters": _tex_integer(metrics["parameters"]),
        "BaseParameters": _tex_integer(
            metrics["parameters"] - config["context_parameters"],
        ),
        "LearnedContextScale": f"{metrics['learned_context_scale']:.4f}",
        "ContextualOptimized": chemistry["optimized_motifs"],
        "ContextualEvaluable": chemistry["eligible_topics"],
        "ContextualUseful": chemistry["useful_motifs"],
        "ContextualMeanSOS": f"{chemistry['mean_sos']:.4f}",
        "ContextualMedianSOS": f"{chemistry['median_sos']:.4f}",
        "ContextualNLL": f"{completion['nll_per_token']:.4f}",
        "ContextualCoverage": f"{100 * chemistry['annotation_coverage']:.1f}\\%",
        "ContextualEvaluableConversion": pct(
            chemistry["eligible_topics"],
            chemistry["optimized_motifs"],
        ),
        "ContextualUsefulConversion": pct(
            chemistry["useful_motifs"],
            chemistry["optimized_motifs"],
        ),
        "TomotopyOptimized": tomotopy["optimized_motifs"],
        "TomotopyEvaluable": tomotopy["high_confidence_evaluable_motifs"],
        "TomotopyUseful": tomotopy["useful_high_confidence_motifs"],
        "TomotopyMeanSOS": f"{tomotopy['mean_sos']:.4f}",
        "TomotopyMedianSOS": f"{tomotopy['median_sos']:.4f}",
        "TomotopyNLL": f"{evidence['tomotopy_nll']:.4f}",
        "CanonicalOptimized": _integer(canonical, "optimized_motifs"),
        "CanonicalEvaluable": _integer(canonical, "evaluable_motifs"),
        "CanonicalUseful": _integer(canonical, "useful_motifs"),
        "CanonicalMeanSOS": f"{_float(canonical, 'mean_sos'):.4f}",
        "CanonicalNLL": f"{_float(canonical, 'completion_nll'):.4f}",
        "BalancedOptimized": _integer(balanced, "optimized_motifs"),
        "BalancedEvaluable": _integer(balanced, "evaluable_motifs"),
        "BalancedUseful": _integer(balanced, "useful_motifs"),
        "BalancedMeanSOS": f"{_float(balanced, 'mean_sos'):.4f}",
        "BalancedNLL": f"{_float(balanced, 'completion_nll'):.4f}",
        "MedianEffectiveTopics": f"{theta['median_effective_topics_per_spectrum']:.2f}",
        "MedianExactSupport": f"{theta['median_exact_support']:.0f}",
        "NinetyFifthExactSupport": (f"{theta['support_size_percentiles']['95']:.0f}"),
        "UniqueTopOneTopics": inventory["unique_top1_topics"],
        "CorpusEffectiveTopics": f"{inventory['corpus_effective_topic_count']:.1f}",
        "ActiveTopics": inventory["active_topics_above_usage_threshold"],
        "MaximumMeanTopicUsage": f"{inventory['maximum_mean_topic_usage']:.4f}",
        "MeanNearestBetaCosine": f"{inventory['mean_nearest_topic_beta_cosine']:.4f}",
        "LargestStrictDuplicateComponent": inventory[
            "largest_strict_duplicate_component"
        ],
        "CompletionDocuments": completion["eligible_documents"],
        "CompletionTokens": f"{completion['in_vocabulary_tokens']:,}",
        "CompletionOOV": f"{100 * completion['oov_fraction']:.2f}\\%",
        "StabilityOptimizedRange": (
            f"{int(aggregate['optimized_motifs']['minimum'])}--"
            f"{int(aggregate['optimized_motifs']['maximum'])}"
        ),
        "StabilityEvaluableRange": (
            f"{int(aggregate['evaluable_motifs']['minimum'])}--"
            f"{int(aggregate['evaluable_motifs']['maximum'])}"
        ),
        "StabilityUsefulRange": (
            f"{int(aggregate['useful_motifs']['minimum'])}--"
            f"{int(aggregate['useful_motifs']['maximum'])}"
        ),
        "StabilitySOSRange": (
            f"{aggregate['mean_sos']['minimum']:.4f}--"
            f"{aggregate['mean_sos']['maximum']:.4f}"
        ),
        "StabilityNLLRange": (
            f"{aggregate['completion_nll']['minimum']:.4f}--"
            f"{aggregate['completion_nll']['maximum']:.4f}"
        ),
        "StabilityUniqueRange": (
            f"{int(aggregate['unique_top1_topics']['minimum'])}--"
            f"{int(aggregate['unique_top1_topics']['maximum'])}"
        ),
        "TrainingMinutesMean": f"{training_wall['mean'] / 60:.1f}",
        "TrainingMinutesSD": (f"{training_wall['sample_standard_deviation'] / 60:.1f}"),
        "ValidationThroughput": f"{runtime['validation_full_spectra_per_second']:,.0f}",
        "PeakCudaAllocatedGB": f"{memory['peak_cuda_allocated_bytes'] / 1e9:.3f}",
        "PeakCudaReservedGB": f"{memory['peak_cuda_reserved_bytes'] / 1e9:.3f}",
        "PeakProcessGB": f"{memory['peak_process_bytes'] / 1e9:.3f}",
        "MinimumSystemAvailableGB": (
            f"{memory['minimum_system_available_bytes'] / 1e9:.2f}"
        ),
    }
    return _write(
        "routing_etm_macros.tex",
        [_command(name, value) for name, value in values.items()],
    )


def _generate_synthetic_table(evidence: dict[str, Any]) -> str:
    labels = {
        "balanced ETM softmax raw": "Balanced ETM + softmax",
        "balanced ETM plus entmax15": r"Balanced ETM + $1.5$-entmax",
        "balanced ETM plus top-2-context routing and softmax": (
            "Contextual top-2 evidence + softmax"
        ),
        "balanced ETM plus top-2-context routing and entmax15": (
            r"\textbf{Contextual Sparse ETM}"
        ),
    }
    lines = []
    for row in evidence["synthetic"]:
        label = labels[row["formulation"]]
        lines.append(
            " & ".join(
                (
                    label,
                    f"{_float(row, 'mean_nll'):.4f}",
                    f"{_float(row, 'mean_true_beta_cosine'):.4f}",
                    f"{_float(row, 'mean_true_theta_cosine'):.4f}",
                    f"{_float(row, 'mean_median_effective_topics'):.2f}",
                    f"{_float(row, 'mean_active_topics_gt_0_005'):.2f}",
                    f"{_float(row, 'mean_unique_top1_topics'):.2f}",
                ),
            )
            + r" \\",
        )
    lines.append(r"\bottomrule")
    return _write("routing_etm_synthetic_table.tex", lines)


def _generate_high_k_table(evidence: dict[str, Any]) -> str:
    labels = {
        "balanced ETM softmax raw": "Balanced ETM + softmax",
        "balanced ETM plus entmax15": r"Balanced ETM + $1.5$-entmax",
        "balanced ETM plus top-2-context routing and entmax15": (
            r"\textbf{Contextual Sparse ETM}"
        ),
    }
    lines = []
    for row in evidence["high_k"]:
        support = row.get("median_exact_support") or "--"
        lines.append(
            " & ".join(
                (
                    labels[row["formulation"]],
                    f"{_float(row, 'nll'):.4f}",
                    f"{_float(row, 'true_beta_cosine'):.4f}",
                    f"{_float(row, 'true_theta_cosine'):.4f}",
                    f"{_float(row, 'top_motif_accuracy'):.4f}",
                    f"{_float(row, 'median_effective_topics'):.2f}",
                    support,
                    str(_integer(row, "unique_top1_topics")),
                ),
            )
            + r" \\",
        )
    lines.append(r"\bottomrule")
    return _write("routing_etm_high_k_table.tex", lines)


def _generate_validation_table(evidence: dict[str, Any]) -> str:
    comparison = evidence["comparison"]
    tomotopy = evidence["tomotopy"]["validation"]
    rows = [
        {
            **comparison["canonical ETM"],
            "model": "Canonical ETM",
        },
        {
            **comparison["balanced ETM"],
            "model": "Fragment/loss-balanced ETM",
        },
        {
            **comparison["routing-informed sparse ETM"],
            "model": "Contextual Sparse ETM",
        },
        {
            "model": "Tomotopy LDA",
            "optimized_motifs": str(tomotopy["optimized_motifs"]),
            "evaluable_motifs": str(tomotopy["high_confidence_evaluable_motifs"]),
            "useful_motifs": str(tomotopy["useful_high_confidence_motifs"]),
            "mean_sos": str(tomotopy["mean_sos"]),
            "median_sos": str(tomotopy["median_sos"]),
            "completion_nll": str(evidence["tomotopy_nll"]),
        },
    ]
    best = {
        "optimized_motifs": max(_integer(row, "optimized_motifs") for row in rows),
        "evaluable_motifs": max(_integer(row, "evaluable_motifs") for row in rows),
        "useful_motifs": max(_integer(row, "useful_motifs") for row in rows),
        "mean_sos": max(_float(row, "mean_sos") for row in rows),
        "median_sos": max(_float(row, "median_sos") for row in rows),
        "completion_nll": min(_float(row, "completion_nll") for row in rows),
    }
    lines = []
    for row in rows:
        model = row["model"]
        if model == "Contextual Sparse ETM":
            model = rf"\textbf{{{model}}}"
        rendered = [model]
        for key in ("optimized_motifs", "evaluable_motifs", "useful_motifs"):
            value = _integer(row, key)
            rendered.append(_bold_if(str(value), condition=value == best[key]))
        for key in ("mean_sos", "median_sos", "completion_nll"):
            value = _float(row, key)
            rendered.append(
                _bold_if(f"{value:.4f}", condition=math.isclose(value, best[key])),
            )
        lines.append(" & ".join(rendered) + r" \\")
    lines.append(r"\bottomrule")
    return _write("routing_etm_validation_table.tex", lines)


def _generate_stability_table(evidence: dict[str, Any]) -> str:
    stability = evidence["stability"]
    lines = [
        (
            " & ".join(
                (
                    str(row["training_seed"]),
                    str(row["optimized_motifs"]),
                    str(row["evaluable_motifs"]),
                    str(row["useful_motifs"]),
                    f"{row['mean_sos']:.4f}",
                    f"{row['median_sos']:.4f}",
                    f"{row['completion_nll']:.4f}",
                    f"{row['median_effective_topics']:.3f}",
                    str(row["unique_top1_topics"]),
                ),
            )
            + r" \\"
        )
        for row in stability["by_seed"]
    ]
    aggregate = stability["aggregate"]
    lines.append(r"\midrule")
    lines.append(
        " & ".join(
            (
                "Mean $\\pm$ SD",
                f"{aggregate['optimized_motifs']['mean']:.1f} $\\pm$ "
                f"{aggregate['optimized_motifs']['sample_standard_deviation']:.1f}",
                f"{aggregate['evaluable_motifs']['mean']:.1f} $\\pm$ "
                f"{aggregate['evaluable_motifs']['sample_standard_deviation']:.1f}",
                f"{aggregate['useful_motifs']['mean']:.1f} $\\pm$ "
                f"{aggregate['useful_motifs']['sample_standard_deviation']:.1f}",
                f"{aggregate['mean_sos']['mean']:.4f} $\\pm$ "
                f"{aggregate['mean_sos']['sample_standard_deviation']:.4f}",
                f"{aggregate['median_sos']['mean']:.4f} $\\pm$ "
                f"{aggregate['median_sos']['sample_standard_deviation']:.4f}",
                f"{aggregate['completion_nll']['mean']:.4f} $\\pm$ "
                f"{aggregate['completion_nll']['sample_standard_deviation']:.4f}",
                f"{aggregate['median_effective_topics']['mean']:.3f} $\\pm$ "
                f"{aggregate['median_effective_topics']['sample_standard_deviation']:.3f}",
                f"{aggregate['unique_top1_topics']['mean']:.1f} $\\pm$ "
                f"{aggregate['unique_top1_topics']['sample_standard_deviation']:.1f}",
            ),
        )
        + r" \\",
    )
    lines.append(r"\bottomrule")
    return _write("routing_etm_stability_table.tex", lines)


def _generate_diagnostics_table(evidence: dict[str, Any]) -> str:
    comparison = evidence["comparison"]
    labels = (
        ("canonical ETM", "Canonical ETM"),
        ("balanced ETM", "Fragment/loss-balanced ETM"),
        ("routing-informed sparse ETM", r"\textbf{Contextual Sparse ETM}"),
    )
    lines = []
    for key, label in labels:
        row = comparison[key]
        support = row.get("median_exact_support") or "--"
        lines.append(
            " & ".join(
                (
                    label,
                    f"{_float(row, 'median_effective_topics'):.2f}",
                    support,
                    str(_integer(row, "unique_top1_topics")),
                    str(_integer(row, "active_topics_gt_0_0005")),
                    f"{_float(row, 'corpus_effective_topics'):.1f}",
                    f"{_float(row, 'mean_nearest_beta_cosine'):.3f}",
                ),
            )
            + r" \\",
        )
    lines.append(r"\bottomrule")
    return _write("routing_etm_diagnostics_table.tex", lines)


def _generate_hyperparameters(evidence: dict[str, Any]) -> str:
    config = evidence["config"]
    protocol = evidence["protocol"]
    chemistry = protocol["chemistry"]
    rows = (
        ("Data split seed", str(protocol["seed"])),
        ("Training seeds", "7043, 23, 37"),
        ("Topics", str(config["topics"])),
        ("Vocabulary", r"21,233 train-only fragment/loss tokens"),
        (
            "Token coordinates",
            "48-dimensional train-only skip-gram negative-sampling (SGNS); fixed",
        ),
        (
            "Encoder",
            r"21,233 $\rightarrow$ 800 $\rightarrow$ 800; rectified linear unit (ReLU)",
        ),
        ("Variational outputs", r"1,000-dimensional $\mu$ and $\log\sigma^2$"),
        (
            "Topic-word decoder",
            "Embedded Topic Model (ETM) inner products; 50/50 fragment/loss mass",
        ),
        ("Contextual evidence", r"leave-one-out context; top-2; temperature 1.0"),
        ("Evidence pseudocount", r"fixed $1/K$"),
        ("Numerical normalization floor", r"$10^{-12}$"),
        ("Additional learned parameters", "one context scalar"),
        ("Document-topic probability transform", r"$1.5$-entmax"),
        ("Reconstruction", "raw intensity pseudo-count multinomial"),
        (
            "Prior and KL divergence",
            "standard-normal analytic Gaussian Kullback--Leibler divergence",
        ),
        ("Optimizer", "Adam"),
        ("Learning rate; weight decay", "0.005; $1.2\\times10^{-6}$"),
        ("Batch size; epochs", "256; 120"),
        ("Device; CPU threads", f"NVIDIA CUDA GPU; {config['threads']}"),
        (
            "Mass2Motif Annotation Guidance (MAG) query",
            f"top {chemistry['motif_spectrum_top_n']} words; search "
            f"{chemistry['mag_search_k']}; "
            f"{chemistry['mag_unique_molecules']} unique molecules",
        ),
        (
            "MAG and substructure overlap score (SOS) thresholds",
            f"membership $\\geq{chemistry['membership_threshold']}$; "
            f"cluster cosine {chemistry['mag_cluster_cosine']}; "
            f"consensus {chemistry['mag_fingerprint_threshold']}",
        ),
    )
    return _write(
        "routing_etm_hyperparameters_table.tex",
        [f"{name} & {value} \\\\" for name, value in rows] + [r"\bottomrule"],
    )


def _generate_code_map() -> str:
    rows = (
        (r"Balanced ETM and $\beta$", r"\texttt{sparse\_etm.py: BalancedSparseETM}"),
        ("Contextual token evidence", r"\texttt{routing\_etm.py: routing\_evidence}"),
        (
            "Posterior offset and KL divergence",
            r"\texttt{routing\_etm.py: encode}",
        ),
        (r"$1.5$-entmax $\theta$", r"\texttt{sparse\_etm.py: transform\_theta}"),
        (
            "Reconstruction objective",
            r"\texttt{sparse\_etm.py: sparse\_reconstruction\_loss}",
        ),
        ("Real training and inference", r"\texttt{scripts/run\_routing\_etm\_real.py}"),
        ("Synthetic design", r"\texttt{scripts/run\_routing\_etm\_campaign.py}"),
        ("Document completion", r"\texttt{objectives.py: completion\_metrics}"),
        ("MAG and SOS", r"\texttt{mag.py; chemical.py}"),
        (
            "Equation--code correspondence test",
            r"\texttt{tests/test\_routing\_etm.py}",
        ),
        (
            "Checkpoint verification",
            r"\texttt{scripts/verify\_routing\_etm\_checkpoint.py}",
        ),
        (
            "Stability verification",
            r"\texttt{scripts/verify\_routing\_etm\_stability.py}",
        ),
    )
    return _write(
        "routing_etm_code_table.tex",
        [f"{name} & {location} \\\\" for name, location in rows] + [r"\bottomrule"],
    )


def generate() -> dict[str, Any]:
    """Validate frozen evidence and regenerate every canonical paper fragment."""
    evidence = _validate_and_load()
    outputs = [
        _generate_macros(evidence),
        _generate_synthetic_table(evidence),
        _generate_high_k_table(evidence),
        _generate_validation_table(evidence),
        _generate_stability_table(evidence),
        _generate_diagnostics_table(evidence),
        _generate_hyperparameters(evidence),
        _generate_code_map(),
    ]
    return {
        "status": "generated",
        "method": EXPECTED_METHOD,
        "evidence_boundary": "training, synthetic, and validation only; test untouched",
        "outputs": outputs,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))  # noqa: T201
