"""Generate the Contextual Sparse ETM paper tables from frozen evidence.

The report consumes synthetic ablations, validation development evidence, and
the final frozen-model test evaluation from one sealed reproduction bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "docs/research/generated"
DEFAULT_EVIDENCE = (
    REPO
    / "research/etm_ecrtm_msnlib/local_results"
    / "20260831_contextual_sparse_etm_reproduction"
)

EXPECTED_METHOD = "contextual_sparse_etm"
EXPECTED_TRAINING_SEEDS = [7043, 23, 37]
EXPECTED_SYNTHETIC_FORMULATIONS = 4
EXPECTED_HIGH_K_ROWS = 3
BALANCED_SOFTMAX = "balanced ETM softmax raw"
BALANCED_ENTMAX = "balanced ETM plus entmax15"
CONTEXT_SOFTMAX = "balanced ETM plus top-2-context routing and softmax"
CONTEXT_ENTMAX = "balanced ETM plus top-2-context routing and entmax15"
EXPECTED_OUTPUTS = {
    "routing_etm_macros.tex",
    "routing_etm_synthetic_table.tex",
    "routing_etm_high_k_table.tex",
    "routing_etm_test_table.tex",
    "routing_etm_stability_table.tex",
    "routing_etm_diagnostics_table.tex",
    "routing_etm_hyperparameters_table.tex",
    "routing_etm_code_table.tex",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    """Hash one evidence artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_package_integrity(root: Path, checkpoint: dict[str, Any]) -> None:
    """Verify the package seal and every compact evidence artifact it owns."""
    manifest_path = root / "fresh_evidence_manifest.json"
    if _sha256_file(manifest_path) != checkpoint["fresh_evidence_manifest_sha256"]:
        msg = "fresh evidence manifest changed after checkpointing"
        raise ValueError(msg)
    manifest = _json(manifest_path)
    for row in manifest["packaged_files"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            msg = f"packaged evidence changed after sealing: {path}"
            raise ValueError(msg)


def _require_reportable_claims(acceptance: dict[str, Any]) -> None:
    """Stop stale affirmative prose when a predeclared direction does not hold.

    The evidence package remains a valid record of a failed scientific
    hypothesis.  The current manuscript, however, contains prose interpreting
    the predeclared directions as observed, so it must not be regenerated until
    that prose has been revised to match the evidence.
    """
    if acceptance.get("all_passed") is not True:
        failed = sorted(
            name for name, passed in acceptance.get("checks", {}).items() if not passed
        )
        msg = (
            "directional claims failed; revise the report before rendering: "
            + ", ".join(
                failed,
            )
        )
        raise ValueError(msg)


def _require_manuscript_claims(evidence: dict[str, Any]) -> None:
    """Verify every data-dependent comparison stated by the static manuscript."""
    comparison = evidence["comparison"]
    canonical = comparison["canonical ETM"]
    balanced = comparison["balanced ETM"]
    contextual = comparison["Contextual Sparse ETM"]
    tomotopy = comparison["Tomotopy LDA"]
    synthetic = {row["formulation"]: row for row in evidence["synthetic"]}
    high_k = {row["formulation"]: row for row in evidence["high_k"]}
    base = synthetic[BALANCED_SOFTMAX]
    sparse = synthetic[BALANCED_ENTMAX]
    context_dense = synthetic[CONTEXT_SOFTMAX]
    complete = synthetic[CONTEXT_ENTMAX]
    high_complete = high_k[CONTEXT_ENTMAX]

    checks = {
        "k36_context_improves_beta": _float(context_dense, "mean_true_beta_cosine")
        > _float(base, "mean_true_beta_cosine"),
        "k36_context_improves_theta": _float(context_dense, "mean_true_theta_cosine")
        > _float(base, "mean_true_theta_cosine"),
        "k36_context_improves_nll": _float(context_dense, "mean_nll")
        < _float(base, "mean_nll"),
        "k36_context_uses_more_winners": _float(
            context_dense,
            "mean_unique_top1_topics",
        )
        > _float(base, "mean_unique_top1_topics"),
        "k36_entmax_alone_reduces_beta": _float(sparse, "mean_true_beta_cosine")
        < _float(base, "mean_true_beta_cosine"),
        "k36_entmax_alone_reduces_theta": _float(sparse, "mean_true_theta_cosine")
        < _float(base, "mean_true_theta_cosine"),
        "k36_entmax_alone_uses_fewer_winners": _float(
            sparse,
            "mean_unique_top1_topics",
        )
        < _float(base, "mean_unique_top1_topics"),
        "k36_complete_has_strongest_beta": _float(
            complete,
            "mean_true_beta_cosine",
        )
        == max(_float(row, "mean_true_beta_cosine") for row in synthetic.values()),
        "k36_complete_has_strongest_theta": _float(
            complete,
            "mean_true_theta_cosine",
        )
        == max(_float(row, "mean_true_theta_cosine") for row in synthetic.values()),
        "k36_complete_nll_is_worse_than_context_softmax": _float(complete, "mean_nll")
        > _float(context_dense, "mean_nll"),
        "high_k_complete_has_lowest_nll": _float(high_complete, "nll")
        == min(_float(row, "nll") for row in high_k.values()),
        "balancing_raises_optimized_motifs": _integer(balanced, "optimized_motifs")
        > _integer(canonical, "optimized_motifs"),
        "balancing_raises_evaluable_motifs": _integer(balanced, "evaluable_motifs")
        > _integer(canonical, "evaluable_motifs"),
        "balancing_raises_useful_motifs": _integer(balanced, "useful_motifs")
        > _integer(canonical, "useful_motifs"),
        "contextual_mean_sos_exceeds_both_etm_controls": _float(
            contextual,
            "mean_sos",
        )
        > max(_float(canonical, "mean_sos"), _float(balanced, "mean_sos")),
        "contextual_has_fewer_optimized_motifs_than_balanced": _integer(
            contextual,
            "optimized_motifs",
        )
        < _integer(balanced, "optimized_motifs"),
        "contextual_nll_is_lower_than_tomotopy": _float(contextual, "completion_nll")
        < _float(tomotopy, "completion_nll"),
        "etm_controls_have_more_effective_topics": min(
            _float(canonical, "median_effective_topics"),
            _float(balanced, "median_effective_topics"),
        )
        > _float(contextual, "median_effective_topics"),
        "etm_controls_have_fewer_unique_winners": max(
            _integer(canonical, "unique_top1_topics"),
            _integer(balanced, "unique_top1_topics"),
        )
        < _integer(contextual, "unique_top1_topics"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        msg = (
            "manuscript result prose is not supported by fresh evidence: "
            + ", ".join(
                failed,
            )
        )
        raise ValueError(msg)


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


def _tex_scientific(value: float) -> str:
    """Format a positive scientific-notation value as valid LaTeX math."""
    if not math.isfinite(value) or value <= 0:
        msg = f"expected a finite positive value, got {value}"
        raise ValueError(msg)
    exponent = math.floor(math.log10(value))
    coefficient = value / (10**exponent)
    if math.isclose(coefficient, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return rf"10^{{{exponent}}}"
    return rf"{coefficient:.3g}\times10^{{{exponent}}}"


def _render(name: str, lines: list[str]) -> tuple[str, str]:
    """Render one complete TeX fragment in memory without touching the output."""
    return name, "\n".join(lines) + "\n"


def _publish_artifacts(output: Path, artifacts: list[tuple[str, str]]) -> list[str]:
    """Publish a fully rendered, complete fragment set through atomic file moves."""
    names = [name for name, _ in artifacts]
    if len(names) != len(set(names)) or set(names) != EXPECTED_OUTPUTS:
        msg = "rendered report fragment set is incomplete or ambiguous"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for name, content in artifacts:
            target = output / name
            temporary = output / f".{name}.tmp"
            temporary.write_text(content, encoding="utf-8")
            temporary_paths.append((temporary, target))
        for temporary, target in temporary_paths:
            temporary.replace(target)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)
    return [str((output / name).resolve()) for name in names]


def _bold_if(value: str, *, condition: bool) -> str:
    return rf"\textbf{{{value}}}" if condition else value


def _validate_and_load(  # noqa: C901, PLR0912, PLR0915
    evidence_root: Path,
) -> dict[str, Any]:
    root = evidence_root.expanduser().resolve(strict=True)
    preparation = _json(root / "preparation_summary.json")
    protocol = _json(root / "protocol.json")
    checkpoint = _json(root / "checkpoint_manifest.json")
    _validate_package_integrity(root, checkpoint)
    config = _json(root / "config.json")
    metrics = _json(root / "metrics.json")
    comparison = {row["model"]: row for row in _csv(root / "comparison.csv")}
    synthetic = _csv(root / "synthetic_summary.csv")
    high_k = _csv(root / "high_k_stress.csv")
    stability = _json(root / "stability_summary.json")
    tomotopy = _json(root / "tomotopy.json")
    acceptance = _json(root / "acceptance.json")
    data_quality = _json(root / "data_quality.json")

    config_method = config.get("artifact_method_id", config.get("method"))
    if checkpoint["method"] != EXPECTED_METHOD or config_method != EXPECTED_METHOD:
        msg = "Contextual Sparse ETM method identity changed"
        raise ValueError(msg)
    if checkpoint["test_released_after_model_and_validation_freeze"] is not True:
        msg = "test split was not released after model and validation freeze"
        raise ValueError(msg)
    if checkpoint["acceptance_all_passed"] is not acceptance["all_passed"]:
        msg = "checkpoint and detailed claim checks disagree"
        raise ValueError(msg)
    _require_reportable_claims(acceptance)
    if data_quality["status"] != "pass":
        msg = "fresh evidence did not pass data-quality checks"
        raise ValueError(msg)
    if config["candidate_test_artifacts_accessed"] is not False:
        msg = "training configuration indicates test access"
        raise ValueError(msg)
    if (
        stability["direction_checks"]["test_released_only_after_model_freeze"]
        is not True
    ):
        msg = "stability evidence does not preserve split ordering"
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

    required_models = {
        "canonical ETM",
        "balanced ETM",
        "Contextual Sparse ETM",
    }
    if not required_models.issubset(comparison):
        msg = "comparison is missing a required ETM baseline"
        raise ValueError(msg)
    proposed_row = comparison["Contextual Sparse ETM"]
    if proposed_row["finite_stable"] != "True":
        msg = "Contextual Sparse ETM is not marked finite and stable"
        raise ValueError(msg)
    chemistry = metrics["test_chemistry"]
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
    if int(metrics["parameters"]) != _integer(proposed_row, "parameters"):
        msg = "unexpected Contextual Sparse ETM parameter count"
        raise ValueError(msg)
    if int(metrics["parameters"]) - _integer(
        comparison["canonical ETM"],
        "parameters",
    ) != int(config["context_parameters"]):
        msg = "Contextual Sparse ETM parameter increment changed"
        raise ValueError(msg)

    if tomotopy.get("method") != "tomotopy":
        msg = "fresh Tomotopy comparator is missing"
        raise ValueError(msg)
    tomotopy_test = tomotopy["test"]
    tomotopy_nll = float(tomotopy_test["completion_nll"])
    tomotopy_row = comparison["Tomotopy LDA"]
    _close(
        _float(tomotopy_row, "completion_nll"),
        tomotopy_nll,
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

    evidence = {
        "preparation": preparation,
        "protocol": protocol,
        "config": config,
        "metrics": metrics,
        "comparison": comparison,
        "synthetic": synthetic,
        "high_k": high_k,
        "stability": stability,
        "tomotopy": tomotopy,
        "tomotopy_nll": tomotopy_nll,
        "evidence_root": root,
    }
    _require_manuscript_claims(evidence)
    return evidence


def _generate_macros(evidence: dict[str, Any]) -> tuple[str, str]:
    preparation = evidence["preparation"]
    data = preparation["data"]
    metrics = evidence["metrics"]
    config = evidence["config"]
    canonical = evidence["comparison"]["canonical ETM"]
    balanced = evidence["comparison"]["balanced ETM"]
    tomotopy = evidence["tomotopy"]["test"]
    stability = evidence["stability"]
    chemistry = metrics["test_chemistry"]
    completion = metrics["document_completion"]
    theta = metrics["theta_support"]
    inventory = metrics["topic_inventory"]
    runtime = metrics["training_runtime"]
    memory = runtime["memory"]
    aggregate = stability["aggregate"]
    training_wall = aggregate["training_wall_seconds"]
    synthetic = {row["formulation"]: row for row in evidence["synthetic"]}
    high_k = {row["formulation"]: row for row in evidence["high_k"]}
    synthetic_complete = synthetic[CONTEXT_ENTMAX]
    high_k_sparse = high_k[BALANCED_ENTMAX]
    high_k_complete = high_k[CONTEXT_ENTMAX]
    tomotopy_training = evidence["tomotopy"]["training"]
    tomotopy_config = evidence["protocol"]["tomotopy"]

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
        "TestThroughput": f"{metrics['runtime']['test_full_spectra_per_second']:,.0f}",
        "PeakCudaAllocatedGB": f"{memory['peak_cuda_allocated_bytes'] / 1e9:.3f}",
        "PeakCudaReservedGB": f"{memory['peak_cuda_reserved_bytes'] / 1e9:.3f}",
        "PeakProcessGB": f"{memory['peak_process_bytes'] / 1e9:.3f}",
        "MinimumSystemAvailableGB": (
            f"{memory['minimum_system_available_bytes'] / 1e9:.2f}"
        ),
        "TomotopyTrainingIterations": tomotopy_training["training_iterations"],
        "TomotopyMaximumIterations": tomotopy_config["maximum_iterations"],
        "TomotopyInferenceIterations": tomotopy_config["inference_iterations"],
        "TomotopyParallelScheme": tomotopy_config["parallel"],
        "TomotopyWorkers": evidence["protocol"]["cpu_threads"],
        "SyntheticContextualMedianEffective": (
            f"{_float(synthetic_complete, 'mean_median_effective_topics'):.2f}"
        ),
        "SyntheticContextualUniqueTopOne": (
            f"{_float(synthetic_complete, 'mean_unique_top1_topics'):.2f}"
        ),
        "HighKEntmaxMedianSupport": (
            f"{_float(high_k_sparse, 'median_exact_support'):.0f}"
        ),
        "HighKEntmaxUniqueTopOne": _integer(high_k_sparse, "unique_top1_topics"),
        "HighKContextualMedianSupport": (
            f"{_float(high_k_complete, 'median_exact_support'):.0f}"
        ),
        "HighKContextualUniqueTopOne": _integer(
            high_k_complete,
            "unique_top1_topics",
        ),
        "HighKContextualBetaRecovery": (
            f"{_float(high_k_complete, 'true_beta_cosine'):.4f}"
        ),
        "HighKContextualThetaRecovery": (
            f"{_float(high_k_complete, 'true_theta_cosine'):.4f}"
        ),
        "HighKContextualDominantAccuracy": (
            f"{_float(high_k_complete, 'top_motif_accuracy'):.4f}"
        ),
    }
    return _render(
        "routing_etm_macros.tex",
        [_command(name, value) for name, value in values.items()],
    )


def _generate_synthetic_table(evidence: dict[str, Any]) -> tuple[str, str]:
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
    return _render("routing_etm_synthetic_table.tex", lines)


def _generate_high_k_table(evidence: dict[str, Any]) -> tuple[str, str]:
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
    return _render("routing_etm_high_k_table.tex", lines)


def _generate_test_table(evidence: dict[str, Any]) -> tuple[str, str]:
    comparison = evidence["comparison"]
    tomotopy = evidence["tomotopy"]["test"]
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
            **comparison["Contextual Sparse ETM"],
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
    return _render("routing_etm_test_table.tex", lines)


def _generate_stability_table(evidence: dict[str, Any]) -> tuple[str, str]:
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
    return _render("routing_etm_stability_table.tex", lines)


def _generate_diagnostics_table(evidence: dict[str, Any]) -> tuple[str, str]:
    comparison = evidence["comparison"]
    labels = (
        ("canonical ETM", "Canonical ETM"),
        ("balanced ETM", "Fragment/loss-balanced ETM"),
        ("Contextual Sparse ETM", r"\textbf{Contextual Sparse ETM}"),
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
    return _render("routing_etm_diagnostics_table.tex", lines)


def _generate_hyperparameters(evidence: dict[str, Any]) -> tuple[str, str]:
    config = evidence["config"]
    protocol = evidence["protocol"]
    preparation = evidence["preparation"]
    chemistry = protocol["chemistry"]
    vocabulary_size = int(preparation["vocabulary_size"])
    embedding_dimensions = int(config["fixed_train_only_sgns_dimensions"])
    hidden = int(config["hidden_dimensions"])
    topics = int(config["topics"])
    training_seeds = ", ".join(
        str(seed) for seed in evidence["stability"]["training_seeds"]
    )
    rows = (
        ("Data split seed", str(protocol["seed"])),
        ("Training seeds", training_seeds),
        ("Topics", str(topics)),
        ("Vocabulary", f"{vocabulary_size:,} train-only fragment/loss tokens"),
        (
            "Token coordinates",
            f"{embedding_dimensions}-dimensional train-only skip-gram "
            "negative-sampling (SGNS); fixed",
        ),
        (
            "Encoder",
            f"{vocabulary_size:,} $\\rightarrow$ {hidden} $\\rightarrow$ "
            f"{hidden}; rectified linear unit (ReLU)",
        ),
        (
            "Variational outputs",
            f"{topics:,}-dimensional $\\mu$ and $\\log\\sigma^2$",
        ),
        (
            "Topic-word decoder",
            "Embedded Topic Model (ETM) inner products; 50/50 fragment/loss mass",
        ),
        (
            "Contextual evidence",
            f"leave-one-out context; top-{config['context_top_k']}; "
            f"temperature {config['context_temperature']}",
        ),
        ("Evidence pseudocount", f"fixed ${config['evidence_pseudocount']}$"),
        (
            "Numerical normalization floor",
            f"${_tex_scientific(float(config['numerical_probability_floor']))}$",
        ),
        (
            "Additional learned parameters",
            f"{config['context_parameters']} context scalar",
        ),
        (
            "Document-topic probability transform",
            f"${config['entmax_alpha']}$-entmax",
        ),
        ("Reconstruction", str(config["reconstruction"])),
        (
            "Prior and KL divergence",
            "standard-normal analytic Gaussian Kullback--Leibler divergence",
        ),
        ("Optimizer", str(config["optimizer"])),
        (
            "Learning rate; weight decay",
            f"{config['learning_rate']}; "
            f"${_tex_scientific(float(config['weight_decay']))}$",
        ),
        ("Batch size; epochs", f"{config['batch_size']}; {config['epochs']}"),
        ("Device; CPU threads", f"{config['device']}; {config['threads']}"),
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
    return _render(
        "routing_etm_hyperparameters_table.tex",
        [f"{name} & {value} \\\\" for name, value in rows] + [r"\bottomrule"],
    )


def _generate_code_map() -> tuple[str, str]:
    rows = (
        (
            r"Channel-balanced $\beta$ (Eq.~\ref{eq:beta})",
            r"\texttt{contextual\_sparse\_etm.py: "
            r"channel\_balanced\_topic\_word\_distribution}",
        ),
        (
            r"Leave-one-out context (Eq.~\ref{eq:loo-context})",
            r"\texttt{contextual\_sparse\_etm.py: leave\_one\_out\_context}",
        ),
        (
            r"Contextual top-2 evidence (Eqs.~\ref{eq:contextual-word}--"
            r"\ref{eq:document-evidence})",
            r"\texttt{contextual\_sparse\_etm.py: contextual\_top2\_evidence}",
        ),
        (
            r"Posterior offset (Eq.~\ref{eq:posterior-offset})",
            r"\texttt{contextual\_sparse\_etm.py: centered\_log\_evidence\_offset}",
        ),
        (
            r"Gaussian posterior and KL (Eqs.~\ref{eq:encoder}, \ref{eq:kl})",
            r"\texttt{contextual\_sparse\_etm.py: posterior; "
            r"diagonal\_gaussian\_kl}",
        ),
        (
            r"$1.5$-entmax $\theta$ (Eqs.~\ref{eq:theta}, "
            r"\ref{eq:entmax-closed-form})",
            (r"\texttt{contextual\_sparse\_etm.py: " + r"entmax15\_document\_mixture}"),
        ),
        (
            r"Reconstruction loss (Eq.~\ref{eq:reconstruction})",
            r"\texttt{topic\_model\_training.py: raw\_count\_reconstruction\_loss}",
        ),
        (
            "Real training and deterministic inference",
            r"\texttt{scripts/run\_contextual\_sparse\_etm.py}",
        ),
        ("Synthetic design", r"\texttt{scripts/run\_routing\_etm\_campaign.py}"),
        ("Document completion", r"\texttt{objectives.py: completion\_metrics}"),
        ("MAG and SOS", r"\texttt{mag.py; chemical.py}"),
        (
            "Equation--code correspondence test",
            r"\texttt{tests/test\_contextual\_sparse\_etm.py}",
        ),
        (
            "Clean-room orchestration",
            r"\texttt{reproduction\_plan.py}; "
            r"\texttt{run\_contextual\_sparse\_etm\_reproduction.py}",
        ),
        (
            "Evidence verification and packaging",
            r"\texttt{reproduction\_audit.py}; "
            r"\texttt{package\_contextual\_sparse\_etm\_reproduction.py}",
        ),
    )
    return _render(
        "routing_etm_code_table.tex",
        [f"{name} & {location} \\\\" for name, location in rows] + [r"\bottomrule"],
    )


def generate(
    evidence_root: Path = DEFAULT_EVIDENCE,
    output: Path = GENERATED,
) -> dict[str, Any]:
    """Validate frozen evidence and regenerate every canonical paper fragment."""
    evidence = _validate_and_load(evidence_root)
    artifacts = [
        _generate_macros(evidence),
        _generate_synthetic_table(evidence),
        _generate_high_k_table(evidence),
        _generate_test_table(evidence),
        _generate_stability_table(evidence),
        _generate_diagnostics_table(evidence),
        _generate_hyperparameters(evidence),
        _generate_code_map(),
    ]
    outputs = _publish_artifacts(output, artifacts)
    return {
        "status": "generated",
        "method": EXPECTED_METHOD,
        "reported_split": "test",
        "evidence_root": str(evidence["evidence_root"]),
        "outputs": outputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=GENERATED)
    arguments = parser.parse_args()
    print(  # noqa: T201
        json.dumps(
            generate(arguments.evidence_root, arguments.output),
            indent=2,
            sort_keys=True,
        ),
    )
