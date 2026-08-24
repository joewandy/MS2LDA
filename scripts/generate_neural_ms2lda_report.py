"""Generate all numerical LaTeX fragments from ``results.json``."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "benchmarks/neural_ms2lda"
RESULTS = PACKAGE / "results/seed42/results.json"
PROTOCOL = PACKAGE / "protocol.json"
DOCS = REPO / "docs/research"
GENERATED = DOCS / "generated"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and semantically validate the report's two source files."""
    evidence = _json(RESULTS)
    protocol = _json(PROTOCOL)
    study = evidence["study"]
    if [row["method"] for row in evidence["methods"]] != ["neural", "tomotopy"]:
        raise ValueError("results must list neural then Tomotopy")
    for key, expected in (
        ("seed", protocol["seed"]),
        ("topics", protocol["model"]["num_topics"]),
        ("cpu_threads", protocol["cpu_threads"]),
        ("final_epoch", protocol["optimization"]["maximum_epochs"]),
    ):
        if int(study[key]) != int(expected):
            raise ValueError(f"results and protocol disagree on {key}")
    for method in evidence["methods"]:
        for split in ("validation", "test"):
            row = method[split]
            bands = row["sos_bands"]
            if sum(bands.values()) != row["high_confidence_evaluable_motifs"]:
                raise ValueError("SOS bands must partition evaluable motifs")
            if (
                bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
                != row["useful_high_confidence_motifs"]
            ):
                raise ValueError("useful motif count must equal the two upper bands")
    return evidence, protocol


def _table_rows(split: str):
    return (
        ("Optimized motifs", lambda row: str(row[split]["optimized_motifs"])),
        (
            "MAG annotation coverage",
            lambda row: f"{100 * row[split]['annotation_coverage']:.1f}\\%",
        ),
        (
            "High-confidence SOS-evaluable motifs",
            lambda row: str(row[split]["high_confidence_evaluable_motifs"]),
        ),
        ("SOS high ($>0.8$)", lambda row: str(row[split]["sos_bands"]["high_gt_0_8"])),
        (
            "SOS intermediate ($0.6$--$0.8$)",
            lambda row: str(row[split]["sos_bands"]["intermediate_0_6_to_0_8"]),
        ),
        ("SOS low ($<0.6$)", lambda row: str(row[split]["sos_bands"]["low_lt_0_6"])),
        (
            "Useful motifs (high-confidence; SOS $\\geq0.6$)",
            lambda row: str(row[split]["useful_high_confidence_motifs"]),
        ),
        ("Mean SOS (high-confidence)", lambda row: f"{row[split]['mean_sos']:.4f}"),
        ("Median SOS (high-confidence)", lambda row: f"{row[split]['median_sos']:.4f}"),
        (
            "Model-fitting time (minutes)",
            lambda row: f"{row['fitting_seconds'] / 60:.1f}",
        ),
    )


def _write_two_method_table(
    path: Path,
    methods: list[dict[str, Any]],
    rows: Sequence[tuple[str, Callable[[dict[str, Any]], str]]],
) -> None:
    body = "\n".join(
        f"{name} & " + " & ".join(render(method) for method in methods) + r" \\"
        for name, render in rows
    )
    path.write_text(body + "\n\\bottomrule\n", encoding="utf-8")


def _command(name: str, value: str | int | float) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def generated_text(evidence: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Write tables and every numerical macro used by prose."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    methods = evidence["methods"]
    by_method = {row["method"]: row for row in methods}
    neural = by_method["neural"]
    tomotopy = by_method["tomotopy"]
    sgns = protocol["sgns"]
    model = protocol["model"]
    optimization = protocol["optimization"]
    anti_collapse = protocol["anti_collapse"]
    cooccurrence = protocol["cooccurrence_regularization"]
    separation = protocol["topic_separation"]
    comparator = protocol["tomotopy"]
    chemistry = protocol["chemistry"]
    feature_dimensions = int(sgns["dimensions"]) + 2
    for split in ("validation", "test"):
        _write_two_method_table(
            GENERATED / f"primary_{split}_table.tex", methods, _table_rows(split)
        )

    parameters = (
        (
            "Seed; topics; CPU threads",
            f"{protocol['seed']}; {model['num_topics']}; {protocol['cpu_threads']}",
        ),
        (
            "Token features",
            f"{sgns['dimensions']} SGNS + 2 type " f"= {feature_dimensions} dimensions",
        ),
        (
            "SGNS",
            f"SparseAdam; {sgns['epochs']} epochs; "
            f"{sgns['positive_pairs_per_document']} pairs/document; "
            f"{sgns['negative_samples']} negatives; power {sgns['negative_power']}; "
            f"batch {sgns['batch_size']}; lr {sgns['learning_rate']}",
        ),
        (
            "Token projection; context projection",
            f"{model['projection_dimensions']}; identity-initialized bias-free linear "
            f"{model['projection_dimensions']} by {model['projection_dimensions']}",
        ),
        (
            "Decoder",
            f"temperature {model['beta_temperature']}; separate channel softmax; "
            "fixed 50/50 fragment/loss mass",
        ),
        ("Router", "top-2; additive document evidence; gate exponent 0.75"),
        ("Training spectra", "one full-spectrum path; no corrupted views"),
        (
            "AdamW",
            f"lr {optimization['learning_rate']}; "
            f"weight decay {optimization['weight_decay']}; "
            f"gradient clip {optimization['gradient_clip_norm']}; "
            "deterministic PyTorch algorithms",
        ),
        (
            "Batches and epochs",
            f"router {optimization['batch_size']}; "
            f"topic {optimization['topic_update_batch_size']}; "
            f"{optimization['topic_updates_per_epoch']} topic updates/epoch; "
            f"{optimization['maximum_epochs']} epochs",
        ),
        (
            "Loss weights",
            f"NPMI {cooccurrence['weight']}; separation {separation['weight']}",
        ),
        (
            "Routing temperature",
            f"{anti_collapse['routing_temperature_start']} to "
            f"{anti_collapse['routing_temperature_end']} over "
            f"{anti_collapse['routing_temperature_anneal_epochs']} epochs",
        ),
        (
            "Sinkhorn",
            f"epsilon {anti_collapse['sinkhorn_epsilon']}; "
            f"{anti_collapse['sinkhorn_iterations']} iterations; weight "
            f"{anti_collapse['sinkhorn_weight_start']} to "
            f"{anti_collapse['sinkhorn_weight_end']}",
        ),
        (
            "NPMI graph",
            f"min DF {cooccurrence['minimum_document_frequency']}; "
            f"min pair {cooccurrence['minimum_pair_frequency']}; "
            f"{cooccurrence['maximum_neighbors']} neighbours; positive NPMI",
        ),
        (
            "Prototype separation",
            f"{separation['neighbors']} neighbours; margin {separation['margin']}",
        ),
        (
            "Tomotopy",
            f"alpha {comparator['alpha']}; eta {comparator['eta']}; "
            f"parallel {comparator['parallel']}; "
            f"step {comparator['step_size']}; "
            f"max {comparator['maximum_iterations']}; "
            f"stop {comparator['convergence_window']} changes "
            f"$<{comparator['convergence_threshold']}$; "
            f"inference {comparator['inference_iterations']}",
        ),
        (
            "MAG/SOS",
            f"top {chemistry['motif_spectrum_top_n']} motif tokens; search "
            f"{chemistry['mag_search_k']}; "
            f"{chemistry['mag_unique_molecules']} unique molecules; "
            f"membership $\\geq {chemistry['membership_threshold']}$; "
            f"cluster cosine {chemistry['mag_cluster_cosine']}; "
            f"fingerprint consensus {chemistry['mag_fingerprint_threshold']}",
        ),
    )
    (GENERATED / "hyperparameters_table.tex").write_text(
        "\n".join(f"{name} & {value} \\\\" for name, value in parameters)
        + "\n\\bottomrule\n",
        encoding="utf-8",
    )
    code_rows = (
        ("Token features $x_w$", r"\texttt{data.py: build\_token\_features}"),
        ("Projected tokens $z_w$", r"\texttt{model.py: projected\_tokens}"),
        (r"Decoder $\beta$", r"\texttt{model.py: topic\_word\_distribution}"),
        ("Leave-one-out context", r"\texttt{model.py: \_route\_embeddings}"),
        ("Top-2 routing", r"\texttt{model.py: route}"),
        (r"Gated $\theta_d$", r"\texttt{model.py: aggregate\_theta}"),
        ("Likelihood and regularizers", r"\texttt{objectives.py}"),
        ("Alternating optimization", r"\texttt{training.py; optimization.py}"),
        ("Held-out inference", r"\texttt{evaluation.py}"),
        ("MAG and SOS", r"\texttt{mag.py; chemical.py}"),
    )
    (GENERATED / "equation_code_table.tex").write_text(
        "\n".join(f"{name} & {location} \\\\" for name, location in code_rows)
        + "\n\\bottomrule\n",
        encoding="utf-8",
    )

    secondary = evidence["secondary"]
    nll = secondary["completion_nll_per_token"]
    warm = secondary["warm_in_memory_batch_inference"]
    neural_validation = neural["validation"]
    tomotopy_validation = tomotopy["validation"]
    effective_topics = secondary["neural_test_median_effective_topics_per_spectrum"]
    warm_ratio = warm["neural_spectra_per_second"] / warm["tomotopy_spectra_per_second"]
    neural_coverage = 100 * neural_validation["annotation_coverage"]
    tomotopy_coverage = 100 * tomotopy_validation["annotation_coverage"]
    values = {
        "StudySeed": evidence["study"]["seed"],
        "StudyTopics": evidence["study"]["topics"],
        "StudyThreads": evidence["study"]["cpu_threads"],
        "FinalEpoch": evidence["study"]["final_epoch"],
        "SourceSpectra": evidence["study"]["source_spectra"],
        "RetainedSpectra": evidence["study"]["retained_spectra"],
        "TrainingSpectra": evidence["study"]["split_spectra"]["train"],
        "ValidationSpectra": evidence["study"]["split_spectra"]["validation"],
        "TestSpectra": evidence["study"]["split_spectra"]["test"],
        "VocabularySize": evidence["study"]["vocabulary_size"],
        "TomotopyTrainingIterations": evidence["study"]["tomotopy_training_iterations"],
        "NeuralValidationOptimized": neural["validation"]["optimized_motifs"],
        "TomotopyValidationOptimized": tomotopy["validation"]["optimized_motifs"],
        "NeuralValidationEvaluable": neural["validation"][
            "high_confidence_evaluable_motifs"
        ],
        "TomotopyValidationEvaluable": tomotopy["validation"][
            "high_confidence_evaluable_motifs"
        ],
        "NeuralValidationUseful": neural["validation"]["useful_high_confidence_motifs"],
        "TomotopyValidationUseful": tomotopy["validation"][
            "useful_high_confidence_motifs"
        ],
        "NeuralValidationSOS": f"{neural['validation']['mean_sos']:.4f}",
        "TomotopyValidationSOS": f"{tomotopy['validation']['mean_sos']:.4f}",
        "NeuralTestOptimized": neural["test"]["optimized_motifs"],
        "TomotopyTestOptimized": tomotopy["test"]["optimized_motifs"],
        "NeuralTestEvaluable": neural["test"]["high_confidence_evaluable_motifs"],
        "TomotopyTestEvaluable": tomotopy["test"]["high_confidence_evaluable_motifs"],
        "NeuralTestUseful": neural["test"]["useful_high_confidence_motifs"],
        "TomotopyTestUseful": tomotopy["test"]["useful_high_confidence_motifs"],
        "NeuralTestSOS": f"{neural['test']['mean_sos']:.4f}",
        "TomotopyTestSOS": f"{tomotopy['test']['mean_sos']:.4f}",
        "NeuralCoverage": f"{neural_coverage:.1f}\\%",
        "TomotopyCoverage": f"{tomotopy_coverage:.1f}\\%",
        "NeuralFitMinutes": f"{neural['fitting_seconds'] / 60:.1f}",
        "TomotopyFitMinutes": f"{tomotopy['fitting_seconds'] / 60:.1f}",
        "NeuralActiveTopics": secondary["neural_test_corpus_active_topics"],
        "NeuralMedianEffectiveTopics": f"{effective_topics:.2f}",
        "NeuralValidationNLL": f"{nll['neural']['validation']:.4f}",
        "TomotopyValidationNLL": f"{nll['tomotopy']['validation']:.4f}",
        "NeuralTestNLL": f"{nll['neural']['test']:.4f}",
        "TomotopyTestNLL": f"{nll['tomotopy']['test']:.4f}",
        "WarmBatchSize": warm["batch_size"],
        "NeuralWarmRate": f"{warm['neural_spectra_per_second']:.1f}",
        "TomotopyWarmRate": f"{warm['tomotopy_spectra_per_second']:.1f}",
        "WarmSpeedRatio": f"{warm_ratio:.1f}",
    }
    (GENERATED / "results_macros.tex").write_text(
        "\n".join(_command(name, value) for name, value in values.items()) + "\n",
        encoding="utf-8",
    )


def generate() -> None:
    evidence, protocol = _evidence()
    generated_text(evidence, protocol)


if __name__ == "__main__":
    generate()
