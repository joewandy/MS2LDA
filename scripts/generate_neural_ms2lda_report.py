"""Generate neural MS2LDA paper figures and tables from committed evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks/neural_assignment_ms2lda/results/seed42"
DOCS = REPO / "docs/research"
FIGURES = DOCS / "figures"
GENERATED = DOCS / "generated"
MANIFEST = DOCS / "report_manifest.json"

COLORS = {"Neural ERNTM": "#2563eb", "Tomotopy": "#64748b"}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _neural_summary() -> dict[str, object]:
    """Normalize the committed selected bundle evidence for plotting."""
    evaluation = _json(RESULTS / "model_bundle/evaluation.json")
    chemistry = _json(RESULTS / "model_bundle/chemistry.json")
    metrics = evaluation["metrics"]
    return {
        "test": {
            "nll_per_token": metrics["test_document_completion"]["nll_per_token"],
            "top_word_diversity": metrics["top_word_diversity"],
            "mean_npmi": metrics["word_cooccurrence_npmi"]["mean_npmi"],
            "corpus_active_topics": metrics["active_topics"]["corpus_active_topics"],
            "mass99_distinct_topic_equivalents": metrics["topic_inventory"][
                "mass_coverages"
            ]["mass_99"]["distinct_topic_equivalents"],
            "cached_spectra_per_second": metrics["cached_latency"][
                "median_spectra_per_second"
            ],
        },
        "chemistry": {
            "annotation_coverage": chemistry["annotation_coverage"],
            "topic_count": chemistry["topics"],
            "dominant_eligible_topics": chemistry["dominant_topic_chemistry"][
                "eligible_topics"
            ],
            "dominant_mean_sos": chemistry["dominant_topic_chemistry"]["mean_sos"],
            "high_confidence_eligible_topics": chemistry["high_confidence_chemistry"][
                "eligible_topics"
            ],
            "high_confidence_associated_spectra": chemistry[
                "high_confidence_chemistry"
            ]["associated_spectra"],
            "high_confidence_mean_sos": chemistry["high_confidence_chemistry"][
                "mean_sos"
            ],
        },
    }


def architecture() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 5.8))
    for axis in axes:
        axis.set_xlim(0, 11)
        axis.set_ylim(0, 3.25)
        axis.axis("off")

    axes[0].text(
        0.2,
        2.9,
        "One-pass inference and shared topic geometry",
        fontsize=11,
        weight="bold",
    )
    inference_boxes = [
        (0.2, 1.15, 1.5, 1.0, "Sparse spectrum\nfragment/loss weights", "#dbeafe"),
        (2.0, 1.15, 1.7, 1.0, "64-D train-only\ntoken features", "#dbeafe"),
        (4.0, 1.15, 1.7, 1.0, "128-D contextual\ntop-2 router", "#bfdbfe"),
        (6.0, 1.15, 1.65, 1.0, "Token-weighted\nmixture θ", "#bfdbfe"),
        (8.0, 1.15, 1.55, 1.0, "Prototype-derived\ntopic matrix β", "#bfdbfe"),
        (9.85, 1.15, 0.95, 1.0, "θβ\nword model", "#d1fae5"),
    ]
    for x, y, width, height, label, color in inference_boxes:
        axes[0].add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor="#1e3a8a",
                linewidth=1.4,
            )
        )
        axes[0].text(
            x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9
        )
    for left, right in (
        (1.7, 2.0),
        (3.7, 4.0),
        (5.7, 6.0),
        (7.65, 8.0),
        (9.55, 9.85),
    ):
        axes[0].annotate(
            "",
            xy=(right, 1.65),
            xytext=(left, 1.65),
            arrowprops={"arrowstyle": "->", "lw": 1.6},
        )
    axes[0].text(
        5.5,
        0.45,
        "No local optimization: θ is produced directly; the same prototypes route tokens and define β",
        ha="center",
        fontsize=9,
        color="#1e3a8a",
    )

    axes[1].text(
        0.2,
        2.9,
        "Training-only collapse controls with distinct roles",
        fontsize=11,
        weight="bold",
    )
    control_boxes = [
        (
            0.25,
            1.05,
            3.1,
            1.15,
            "Balanced Sinkhorn targets\nGive weak topics early routing mass\n(weight 0.25 to 0.0567 executed)",
            "#fef3c7",
        ),
        (
            3.95,
            1.05,
            3.1,
            1.15,
            "ERNTM separation\nPush normalized prototypes apart\n(weight 1.0)",
            "#ede9fe",
        ),
        (
            7.65,
            1.05,
            3.1,
            1.15,
            "Deterministic recycling\nReplace persistently dead topics\nwith high-loss contexts",
            "#fee2e2",
        ),
    ]
    for x, y, width, height, label, color in control_boxes:
        axes[1].add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.3,
            )
        )
        axes[1].text(
            x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9
        )
    axes[1].text(1.8, 0.45, "usage collapse", ha="center", fontsize=9, color="#92400e")
    axes[1].text(
        5.5, 0.45, "semantic duplication", ha="center", fontsize=9, color="#5b21b6"
    )
    axes[1].text(
        9.2,
        0.45,
        "dead-component recovery",
        ha="center",
        fontsize=9,
        color="#991b1b",
    )
    figure.subplots_adjust(hspace=0.18)
    _save(figure, "architecture.png")


def capacity_pareto() -> None:
    with (RESULTS / "capacity_screen.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    markers = {"baseline": "o", "ecr": "s", "erntm": "^"}
    colors = {"baseline": "#94a3b8", "ecr": "#f59e0b", "erntm": "#2563eb"}
    for method in markers:
        selected = [row for row in rows if row["method"] == method]
        axis.scatter(
            [float(row["mass99_distinct_topic_equivalents"]) for row in selected],
            [float(row["top_word_diversity"]) for row in selected],
            s=[35 + int(row["K"]) / 12 for row in selected],
            marker=markers[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.7,
            label=method.upper(),
            alpha=0.9,
        )
    winner = next(row for row in rows if row["arm"] == "erntm_k500")
    axis.annotate(
        "selected ERNTM\nK=500",
        (
            float(winner["mass99_distinct_topic_equivalents"]),
            float(winner["top_word_diversity"]),
        ),
        xytext=(-85, -32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#1e3a8a"},
        fontsize=9,
    )
    axis.set_xlabel("Validation mass-99 distinct-topic equivalents")
    axis.set_ylabel("Top-10 word diversity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    _save(figure, "capacity_pareto.png")


def comparisons() -> None:
    neural = _neural_summary()
    comparator = _json(RESULTS / "comparator.json")
    labels = ["Neural ERNTM\nK=500", "Tomotopy\nK=1000"]
    colors = [COLORS["Neural ERNTM"], COLORS["Tomotopy"]]
    metrics = [
        (
            "Held-out NLL\n(lower is better)",
            neural["test"]["nll_per_token"],
            comparator["test"]["nll_per_token"],
        ),
        (
            "Top-word diversity\n(higher is better)",
            neural["test"]["top_word_diversity"],
            comparator["test"]["top_word_diversity"],
        ),
        (
            "Mean NPMI\n(higher is better)",
            neural["test"]["mean_npmi"],
            comparator["test"]["mean_npmi"],
        ),
        (
            "Active topics",
            neural["test"]["corpus_active_topics"],
            comparator["test"]["corpus_active_topics"],
        ),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.6))
    for axis, (title, left, right) in zip(axes, metrics, strict=True):
        bars = axis.bar(labels, [left, right], color=colors, width=0.65)
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, (left, right), strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    figure.tight_layout()
    _save(figure, "final_comparison.png")


def chemistry() -> None:
    neural = _neural_summary()["chemistry"]
    comparator = _json(RESULTS / "comparator.json")["chemistry"]
    labels = ["Neural ERNTM K=500", "Tomotopy K=1000"]
    colors = [COLORS["Neural ERNTM"], COLORS["Tomotopy"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = np.arange(2)
    width = 0.34
    neural_topics = neural["topic_count"]
    comparator_topics = 1000
    dominant_coverage = [
        neural["dominant_eligible_topics"] / neural_topics,
        comparator["dominant_eligible_topics"] / comparator_topics,
    ]
    confident_coverage = [
        neural["high_confidence_eligible_topics"] / neural_topics,
        comparator["high_confidence_eligible_topics"] / comparator_topics,
    ]
    dominant_bars = axes[0].bar(
        x - width / 2,
        dominant_coverage,
        width,
        label="dominant",
        color="#2563eb",
    )
    confident_bars = axes[0].bar(
        x + width / 2,
        confident_coverage,
        width,
        label="probability ≥ 0.5",
        color="#93c5fd",
    )
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("SOS-evaluable topic proportion")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    raw_counts = [
        (neural["dominant_eligible_topics"], neural_topics),
        (comparator["dominant_eligible_topics"], comparator_topics),
        (neural["high_confidence_eligible_topics"], neural_topics),
        (comparator["high_confidence_eligible_topics"], comparator_topics),
    ]
    for bar, (numerator, denominator) in zip(
        [*dominant_bars, *confident_bars], raw_counts, strict=True
    ):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{numerator}/{denominator}",
            ha="center",
            fontsize=8,
        )
    dominant = [neural["dominant_mean_sos"], comparator["dominant_mean_sos"]]
    confident = [
        neural["high_confidence_mean_sos"],
        comparator["high_confidence_mean_sos"],
    ]
    axes[1].bar(x - width / 2, dominant, width, color="#2563eb", label="dominant")
    axes[1].bar(
        x + width / 2, confident, width, color="#93c5fd", label="probability ≥ 0.5"
    )
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Compound-balanced mean SOS")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(axes[1].patches, [*dominant, *confident], strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout()
    _save(figure, "chemistry_coverage.png")


def speed() -> None:
    neural = _neural_summary()
    comparator = _json(RESULTS / "comparator.json")
    values = [
        neural["test"]["cached_spectra_per_second"],
        comparator["test"]["cached_spectra_per_second"],
    ]
    figure, axis = plt.subplots(figsize=(5.7, 3.7))
    bars = axis.bar(
        ["Neural ERNTM\none pass", "Tomotopy\n100 iterations"],
        values,
        color=[COLORS["Neural ERNTM"], COLORS["Tomotopy"]],
    )
    axis.set_yscale("log")
    axis.set_ylabel("Cached inference throughput (spectra/s, log scale)")
    axis.grid(axis="y", which="both", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:,.1f}",
            ha="center",
            fontsize=9,
        )
    _save(figure, "inference_speed.png")


def tables() -> None:
    neural = _neural_summary()
    comparator = _json(RESULTS / "comparator.json")
    fresh = _json(RESULTS / "fresh_reproducibility.json")
    comparator_report = next(
        method
        for method in fresh["methods"]
        if method["method"] == "tomotopy_k1000_comparator"
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Topics requested", "500", "1000"),
        (
            "Held-out NLL (lower is better)",
            f"{neural['test']['nll_per_token']:.4f}",
            f"{comparator['test']['nll_per_token']:.4f}",
        ),
        (
            "Top-word diversity (higher is less repetition)",
            f"{neural['test']['top_word_diversity']:.4f}",
            f"{comparator['test']['top_word_diversity']:.4f}",
        ),
        (
            "Mean NPMI (higher is better)",
            f"{neural['test']['mean_npmi']:.4f}",
            f"{comparator['test']['mean_npmi']:.4f}",
        ),
        (
            "Corpus-active topics (method-relative)",
            str(neural["test"]["corpus_active_topics"]),
            str(comparator["test"]["corpus_active_topics"]),
        ),
        (
            "Annotation coverage",
            f"{neural['chemistry']['annotation_coverage']:.1%} (136/500)".replace(
                "%", "\\%"
            ),
            f"{comparator_report['annotation_coverage']:.1%} (607/1000)".replace(
                "%", "\\%"
            ),
        ),
        (
            "Dominant SOS-evaluable coverage",
            f"{neural['chemistry']['dominant_eligible_topics'] / 500:.1%} (105/500)".replace(
                "%", "\\%"
            ),
            f"{comparator['chemistry']['dominant_eligible_topics'] / 1000:.1%} (598/1000)".replace(
                "%", "\\%"
            ),
        ),
        (
            "Dominant mean SOS",
            f"{neural['chemistry']['dominant_mean_sos']:.4f}",
            f"{comparator['chemistry']['dominant_mean_sos']:.4f}",
        ),
        (
            "High-confidence topic--spectrum associations",
            str(neural["chemistry"]["high_confidence_associated_spectra"]),
            str(comparator["chemistry"]["high_confidence_associated_spectra"]),
        ),
    ]
    body = "\n".join(f"{name} & {left} & {right} \\\\" for name, left, right in rows)
    (GENERATED / "final_comparison_table.tex").write_text(
        body + "\n\\bottomrule\n", encoding="utf-8"
    )


def _generate() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    architecture()
    capacity_pareto()
    comparisons()
    chemistry()
    speed()
    tables()


def _manifest_files() -> tuple[list[Path], list[Path]]:
    evidence = [
        RESULTS / "capacity_screen.csv",
        RESULTS / "comparator.json",
        RESULTS / "fresh_reproducibility.json",
        RESULTS / "historical_replay.json",
        RESULTS / "model_bundle/evaluation.json",
        RESULTS / "model_bundle/chemistry.json",
        Path(__file__).resolve(),
    ]
    outputs = [
        FIGURES / "architecture.png",
        FIGURES / "capacity_pareto.png",
        FIGURES / "chemistry_coverage.png",
        FIGURES / "final_comparison.png",
        FIGURES / "inference_speed.png",
        GENERATED / "final_comparison_table.tex",
        DOCS / "neural_ms2lda_checkpoint.tex",
        DOCS / "neural_ms2lda_checkpoint.pdf",
    ]
    return evidence, outputs


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _write_manifest() -> None:
    evidence, outputs = _manifest_files()
    missing = [path for path in (*evidence, *outputs) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"report artifact is missing: {missing}")
    payload = {
        "schema_version": "neural-ms2lda/report-manifest-v1",
        "evidence_sha256": {_relative(path): _sha256(path) for path in evidence},
        "output_sha256": {_relative(path): _sha256(path) for path in outputs},
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _verify_manifest() -> None:
    payload = _json(MANIFEST)
    for section in ("evidence_sha256", "output_sha256"):
        for name, expected in payload[section].items():
            path = REPO / str(name)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"report manifest mismatch: {name}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.write_manifest and args.verify_manifest:
        parser.error("choose at most one manifest action")
    if args.write_manifest:
        _write_manifest()
    elif args.verify_manifest:
        _verify_manifest()
    else:
        _generate()


if __name__ == "__main__":
    main()
