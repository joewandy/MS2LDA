"""Generate the final neural MS2LDA report from one compact evidence file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "benchmarks/neural_ms2lda"
RESULTS = PACKAGE / "results/seed42"
EVIDENCE = RESULTS / "results.json"
PROTOCOL = PACKAGE / "protocol.json"
BUNDLE_MANIFEST = RESULTS / "model_bundle/manifest.json"
DOCS = REPO / "docs/research"
FIGURES = DOCS / "figures"
GENERATED = DOCS / "generated"
MANIFEST = DOCS / "report_manifest.json"
TEX = DOCS / "neural_ms2lda_checkpoint.tex"
PDF = DOCS / "neural_ms2lda_checkpoint.pdf"
COLORS = {"neural": "#2563eb", "tomotopy": "#64748b"}
LABELS = {"neural": "Neural", "tomotopy": "Tomotopy"}


def _json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Hash a potentially large artifact without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence() -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: C901
    """Load and cross-check the final results against protocol and bundle."""
    evidence = _json(EVIDENCE)
    protocol = _json(PROTOCOL)
    contract = evidence["comparison_contract"]
    if [row["method"] for row in evidence["methods"]] != ["neural", "tomotopy"]:
        raise ValueError("final report must compare neural then Tomotopy")
    if int(contract["topics"]) != int(protocol["model"]["num_topics"]):
        raise ValueError("result and protocol topic counts differ")
    if int(contract["cpu_threads"]) != int(protocol["cpu_threads"]):
        raise ValueError("result and protocol thread counts differ")
    if set(protocol["model"]) != {
        "num_topics",
        "projection_dimensions",
        "router_hidden_dimensions",
        "top_k",
        "beta_temperature",
        "token_type_balance",
        "document_mixture_weight",
        "sinkhorn_epsilon",
        "sinkhorn_iterations",
        "gradient_clip_norm",
    }:
        raise ValueError("final protocol does not describe the supported model")
    expected_bundle = evidence["provenance"]["model_bundle_manifest_sha256"]
    if _sha256(BUNDLE_MANIFEST) != expected_bundle:
        raise ValueError("model bundle manifest differs from final results")
    for method in evidence["methods"]:
        for split in ("validation", "test"):
            row = method[split]
            bands = row["sos_bands"]
            useful = bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
            if useful != row["useful_high_confidence_motifs"]:
                raise ValueError("useful motif count differs from SOS bands")
            if sum(bands.values()) != row["high_confidence_evaluable_motifs"]:
                raise ValueError("SOS bands do not cover every eligible motif")
    return evidence, protocol


def _save(figure: plt.Figure, name: str) -> None:
    """Write one publication-resolution figure with a white background."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def model_overview() -> None:
    """Draw the exact forward path and the training-only safeguards."""
    figure, axis = plt.subplots(figsize=(12.0, 5.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    forward = (
        (0.2, "Token features\nSGNS + mass + type", "#dbeafe"),
        (2.65, "Shared geometry\n$z_w$, $q_k$", "#bfdbfe"),
        (5.1, "Contextual top-2\ntoken routes", "#bfdbfe"),
        (7.55, "Token mass $M_d$\n$\\times\\,g_d^{0.75}$", "#fde68a"),
        (10.0, "One-pass $\\theta_d$\nand $p(w|d)$", "#d1fae5"),
    )
    for x, label, color in forward:
        axis.add_patch(
            plt.Rectangle((x, 3.15), 1.85, 0.95, facecolor=color, edgecolor="#334155")
        )
        axis.text(x + 0.925, 3.625, label, ha="center", va="center", fontsize=9)
    for start, end in ((2.05, 2.65), (4.5, 5.1), (6.95, 7.55), (9.4, 10.0)):
        axis.annotate(
            "",
            xy=(end, 3.625),
            xytext=(start, 3.625),
            arrowprops={"arrowstyle": "->", "color": "#334155"},
        )

    axis.add_patch(
        plt.Rectangle((2.65, 1.55), 4.3, 0.95, facecolor="#ede9fe", edgecolor="#334155")
    )
    axis.text(
        4.8,
        2.025,
        "Shared decoder $\\beta$: cosine logits $\\to$ mean fragment/loss evidence\n"
        "$\\to$ 25% pull toward equal channel mass",
        ha="center",
        va="center",
        fontsize=9,
    )
    axis.annotate(
        "",
        xy=(3.55, 3.15),
        xytext=(3.55, 2.5),
        arrowprops={"arrowstyle": "<->", "color": "#334155"},
    )
    axis.annotate(
        "",
        xy=(10.0, 2.025),
        xytext=(6.95, 2.025),
        arrowprops={"arrowstyle": "->", "color": "#334155"},
    )

    safeguards = (
        "weighted k-means++\ninitialization",
        "temperature annealing\n+ Sinkhorn usage",
        "positive-NPMI graph\n+ prototype margin",
        "hard-context\ndead-topic recycling",
    )
    axis.text(
        6.0,
        1.28,
        "Training-only anti-collapse",
        ha="center",
        weight="bold",
        color="#334155",
    )
    for index, label in enumerate(safeguards):
        x = 0.35 + index * 2.9
        axis.add_patch(
            plt.Rectangle(
                (x, 0.25), 2.55, 0.75, facecolor="#f1f5f9", edgecolor="#94a3b8"
            )
        )
        axis.text(x + 1.275, 0.625, label, ha="center", va="center", fontsize=8)
    axis.text(
        6.0,
        4.7,
        "Forward model used identically in training and inference",
        ha="center",
        weight="bold",
        color="#1e3a8a",
    )
    _save(figure, "model_overview.png")


def primary_results(evidence: dict[str, Any]) -> None:
    """Plot four paper-facing chemical outcomes across both held-out splits."""
    methods = evidence["methods"]
    x = np.arange(2)
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    specifications: tuple[
        tuple[str, Callable[[dict[str, Any], str], float], str], ...
    ] = (
        (
            "MAG-optimized motifs",
            lambda row, split: row[split]["optimized_motifs"],
            "count",
        ),
        (
            "High-confidence evaluable motifs",
            lambda row, split: row[split]["high_confidence_evaluable_motifs"],
            "count",
        ),
        (
            "Useful high-confidence motifs",
            lambda row, split: row[split]["useful_high_confidence_motifs"],
            "count",
        ),
        (
            "Mean compound-balanced SOS",
            lambda row, split: row[split]["mean_sos"],
            "score",
        ),
    )
    for axis, (title, extract, kind) in zip(axes.ravel(), specifications, strict=True):
        for offset, method in zip((-width / 2, width / 2), methods, strict=True):
            values = [extract(method, split) for split in ("validation", "test")]
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=LABELS[method["method"]],
                color=COLORS[method["method"]],
            )
            for bar, value in zip(bars, values, strict=True):
                label = f"{value:.3f}" if kind == "score" else f"{value:.0f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(title, fontsize=10)
        axis.set_xticks(x, ("Validation", "Test"))
        axis.grid(axis="y", alpha=0.2)
        if kind == "score":
            axis.set_ylim(0.55, 0.72)
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    _save(figure, "primary_results.png")


def inference_speed(evidence: dict[str, Any]) -> None:
    """Plot the explicitly secondary matched-thread warm inference result."""
    result = evidence["secondary_warm_in_memory_batch_inference"]
    values = (
        result["neural_spectra_per_second"],
        result["tomotopy_spectra_per_second"],
    )
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    bars = axis.bar(
        ("Neural", "Tomotopy"), values, color=(COLORS["neural"], COLORS["tomotopy"])
    )
    axis.set_yscale("log")
    axis.set_ylabel("Spectra/s (warm in-memory batch)")
    axis.grid(axis="y", which="both", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:,.1f}",
            ha="center",
        )
    axis.set_title("Six CPU threads; batch size 128")
    _save(figure, "inference_speed.png")


def _table_rows(
    split: str,
) -> tuple[tuple[str, Callable[[dict[str, Any]], str]], ...]:
    """Define the primary table once for validation and test."""
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
        (
            "SOS high ($>0.8$)",
            lambda row: str(row[split]["sos_bands"]["high_gt_0_8"]),
        ),
        (
            "SOS intermediate ($0.6$--$0.8$)",
            lambda row: str(row[split]["sos_bands"]["intermediate_0_6_to_0_8"]),
        ),
        (
            "SOS low ($<0.6$)",
            lambda row: str(row[split]["sos_bands"]["low_lt_0_6"]),
        ),
        (
            "Useful high-confidence motifs (SOS $\\geq0.6$)",
            lambda row: str(row[split]["useful_high_confidence_motifs"]),
        ),
        ("Mean high-confidence SOS", lambda row: f"{row[split]['mean_sos']:.4f}"),
        ("Median high-confidence SOS", lambda row: f"{row[split]['median_sos']:.4f}"),
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


def tables(evidence: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Generate the primary, protocol, code-map, and safety tables."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    methods = evidence["methods"]
    for split in ("validation", "test"):
        _write_two_method_table(
            GENERATED / f"primary_{split}_table.tex",
            methods,
            _table_rows(split),
        )

    parameter_rows = (
        (
            "Seed; topics; CPU threads",
            f"{protocol['seed']}; {protocol['model']['num_topics']}; {protocol['cpu_threads']}",
        ),
        ("Token features", "48 SGNS + 14 Fourier + 2 type = 64 dimensions"),
        ("SGNS", "5 epochs; 8 pairs/document; 5 negatives; power 0.75; lr 0.01"),
        (
            "Projection; router hidden",
            f"{protocol['model']['projection_dimensions']}; {protocol['model']['router_hidden_dimensions']}",
        ),
        (
            "Decoder",
            f"temperature {protocol['model']['beta_temperature']}; type pull {protocol['model']['token_type_balance']}",
        ),
        (
            "Router",
            f"top-{protocol['model']['top_k']}; additive document evidence; gate exponent {protocol['model']['document_mixture_weight']}",
        ),
        (
            "Views",
            f"{protocol['views']['pairs']} pairs; {100 * protocol['views']['retained_peak_group_fraction']:.0f}\\% physical peak groups/view",
        ),
        (
            "AdamW",
            f"lr {protocol['optimization']['learning_rate']}; weight decay {protocol['optimization']['weight_decay']}; gradient clip {protocol['model']['gradient_clip_norm']}",
        ),
        (
            "Batches and epochs",
            f"router {protocol['optimization']['batch_size']}; topic {protocol['optimization']['topic_update_batch_size']}; {protocol['optimization']['topic_updates_per_epoch']} topic updates/epoch; {protocol['optimization']['maximum_epochs']} epochs",
        ),
        (
            "Loss weights",
            f"local {protocol['optimization']['local_decoder_weight']}; view consistency {protocol['optimization']['theta_consistency_weight']}; NPMI {protocol['cooccurrence_regularization']['weight']}; separation {protocol['topic_separation']['weight']}",
        ),
        (
            "Routing temperature",
            f"{protocol['anti_collapse']['routing_temperature_start']} to {protocol['anti_collapse']['routing_temperature_end']} over {protocol['anti_collapse']['routing_temperature_anneal_epochs']} epochs",
        ),
        (
            "Sinkhorn",
            f"epsilon {protocol['model']['sinkhorn_epsilon']}; {protocol['model']['sinkhorn_iterations']} iterations; weight {protocol['anti_collapse']['sinkhorn_weight_start']} to {protocol['anti_collapse']['sinkhorn_weight_end']}",
        ),
        (
            "NPMI graph",
            f"min DF {protocol['cooccurrence_regularization']['minimum_document_frequency']}; min pair {protocol['cooccurrence_regularization']['minimum_pair_frequency']}; {protocol['cooccurrence_regularization']['maximum_neighbors']} neighbours; min NPMI {protocol['cooccurrence_regularization']['minimum_npmi']}",
        ),
        (
            "Prototype separation",
            f"{protocol['topic_separation']['neighbors']} neighbours; margin {protocol['topic_separation']['margin']}",
        ),
        (
            "Recycling",
            f"usage $<{protocol['anti_collapse']['recycle_usage_fraction_of_uniform']}$ of uniform for {protocol['anti_collapse']['recycle_patience_validations']} validations; max {protocol['anti_collapse']['maximum_recycles_per_topic']}/topic",
        ),
        (
            "Tomotopy",
            f"alpha {protocol['tomotopy']['alpha']}; eta {protocol['tomotopy']['eta']}; step {protocol['tomotopy']['step_size']}; max {protocol['tomotopy']['maximum_iterations']}; inference {protocol['tomotopy']['inference_iterations']}",
        ),
        (
            "MAG/SOS",
            f"top {protocol['chemistry']['motif_spectrum_top_n']} motif tokens; membership $\\geq {protocol['chemistry']['membership_threshold']}$; cluster cosine {protocol['chemistry']['mag_cluster_cosine']}",
        ),
    )
    (GENERATED / "hyperparameters_table.tex").write_text(
        "\n".join(f"{name} & {value} \\\\" for name, value in parameter_rows)
        + "\n\\bottomrule\n",
        encoding="utf-8",
    )

    code_rows = (
        ("Token feature $x_w$", r"\texttt{data.py: build\_token\_features}"),
        ("Projected token $z_w$", r"\texttt{model.py: projected\_tokens}"),
        (
            "Decoder $\\beta$ and mean type evidence",
            r"\texttt{model.py: topic\_word\_distribution}",
        ),
        ("Leave-one-out $h_{dw}$ and $u_d$", r"\texttt{model.py: \_route\_embeddings}"),
        ("Top-2 $r_{dwk}$", r"\texttt{model.py: route}"),
        ("Gated mixture $\\theta_d$", r"\texttt{model.py: aggregate\_theta}"),
        ("Completion likelihood", r"\texttt{model.py: sparse\_completion\_nll}"),
        ("Router and topic objectives", r"\texttt{objectives.py}"),
        ("Positive-NPMI graph", r"\texttt{cooccurrence.py}"),
        ("NPMI and separation losses", r"\texttt{objectives.py}"),
        (
            "Alternating optimization and recycling",
            r"\texttt{optimization.py; training.py}",
        ),
        ("One-pass held-out inference", r"\texttt{model.py: infer\_theta}"),
        ("Tomotopy fit and inference", r"\texttt{tomotopy.py}"),
        ("MAG/SOS evaluation", r"\texttt{chemical.py}"),
    )
    (GENERATED / "equation_code_table.tex").write_text(
        "\n".join(f"{name} & {location} \\\\" for name, location in code_rows)
        + "\n\\bottomrule\n",
        encoding="utf-8",
    )

    diagnostics = evidence["secondary_diagnostics"]
    nll = diagnostics["completion_nll_per_token"]
    (GENERATED / "secondary_diagnostics.tex").write_text(
        "\n".join(
            (
                rf"\newcommand{{\NeuralRecycledTopics}}{{{diagnostics['neural_recycled_topics_during_training']}}}",
                rf"\newcommand{{\NeuralActiveTopics}}{{{diagnostics['neural_test_corpus_active_topics']}}}",
                rf"\newcommand{{\NeuralMedianEffectiveTopics}}{{{diagnostics['neural_test_median_effective_topics_per_spectrum']:.2f}}}",
                rf"\newcommand{{\NeuralValidationNLL}}{{{nll['neural']['validation']:.4f}}}",
                rf"\newcommand{{\TomotopyValidationNLL}}{{{nll['tomotopy']['validation']:.4f}}}",
                rf"\newcommand{{\NeuralTestNLL}}{{{nll['neural']['test']:.4f}}}",
                rf"\newcommand{{\TomotopyTestNLL}}{{{nll['tomotopy']['test']:.4f}}}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def generate() -> None:
    """Regenerate every report-derived figure and table."""
    evidence, protocol = _evidence()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    model_overview()
    primary_results(evidence)
    inference_speed(evidence)
    tables(evidence, protocol)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _manifest_files() -> tuple[list[Path], list[Path]]:
    """Return the minimal evidence set and every report output."""
    _evidence()
    evidence = [EVIDENCE, PROTOCOL, BUNDLE_MANIFEST, Path(__file__).resolve()]
    outputs = [
        FIGURES / "model_overview.png",
        FIGURES / "primary_results.png",
        FIGURES / "inference_speed.png",
        GENERATED / "primary_validation_table.tex",
        GENERATED / "primary_test_table.tex",
        GENERATED / "hyperparameters_table.tex",
        GENERATED / "equation_code_table.tex",
        GENERATED / "secondary_diagnostics.tex",
        TEX,
        PDF,
    ]
    return evidence, outputs


def write_manifest() -> None:
    """Hash the final report and the evidence that generated it."""
    evidence, outputs = _manifest_files()
    missing = [path for path in (*evidence, *outputs) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"report artifact is missing: {missing}")
    payload = {
        "evidence_sha256": {_relative(path): _sha256(path) for path in evidence},
        "output_sha256": {_relative(path): _sha256(path) for path in outputs},
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_manifest() -> None:
    """Re-hash every declared input and output."""
    _manifest_files()
    payload = _json(MANIFEST)
    for section in ("evidence_sha256", "output_sha256"):
        for name, expected in payload[section].items():
            path = REPO / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"report manifest mismatch: {name}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run generation or one of the two manifest actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.write_manifest and args.verify_manifest:
        parser.error("choose at most one manifest action")
    if args.write_manifest:
        write_manifest()
    elif args.verify_manifest:
        verify_manifest()
    else:
        generate()


if __name__ == "__main__":
    main()
