"""Generate the neural MS2LDA comparison figures and table from evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks/neural_assignment_ms2lda/results/seed42"
DOCS = REPO / "docs/research"
FIGURES = DOCS / "figures"
GENERATED = DOCS / "generated"
MANIFEST = DOCS / "report_manifest.json"
COLORS = ("#2563eb", "#64748b")


def _json(path: Path) -> dict[str, Any]:
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


def _methods() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _json(RESULTS / "comparison.json")["methods"]
    neural = next(row for row in rows if str(row["method"]).startswith("neural_"))
    tomotopy = next(row for row in rows if row["method"] == "tomotopy")
    return neural, tomotopy


def architecture() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 5.6))
    for axis in axes:
        axis.set_xlim(0, 11)
        axis.set_ylim(0, 3.2)
        axis.axis("off")

    axes[0].text(0.2, 2.85, "One-pass neural inference", fontsize=11, weight="bold")
    boxes = [
        (0.2, 1.15, 1.6, "Sparse spectrum", "#dbeafe"),
        (2.2, 1.15, 1.6, "Train-only\ntoken features", "#dbeafe"),
        (4.2, 1.15, 1.6, "Hierarchical\ntop-2 router", "#bfdbfe"),
        (6.2, 1.15, 1.6, "Topic mixture", "#bfdbfe"),
        (8.2, 1.15, 1.6, "Topic-word\nmatrix", "#bfdbfe"),
    ]
    for x, y, width, label, color in boxes:
        axes[0].add_patch(
            plt.Rectangle(
                (x, y), width, 1.0, facecolor=color, edgecolor="#1e3a8a", lw=1.4
            )
        )
        axes[0].text(x + width / 2, y + 0.5, label, ha="center", va="center")
    for x in (1.8, 3.8, 5.8, 7.8):
        axes[0].annotate(
            "", xy=(x + 0.4, 1.65), xytext=(x, 1.65), arrowprops={"arrowstyle": "->"}
        )
    axes[0].text(
        5.0,
        0.45,
        "The same learned prototypes route tokens and define the topic-word distribution",
        ha="center",
        color="#1e3a8a",
    )

    axes[1].text(0.2, 2.85, "Training-only topic controls", fontsize=11, weight="bold")
    controls = [
        (0.3, "Balanced routing\nlimits usage collapse", "#fef3c7"),
        (3.9, "Positive NPMI graph\nshapes topic words", "#d1fae5"),
        (7.5, "Nearest-topic margin\nlimits duplication", "#ede9fe"),
    ]
    for x, label, color in controls:
        axes[1].add_patch(
            plt.Rectangle(
                (x, 1.0), 3.0, 1.15, facecolor=color, edgecolor="#334155", lw=1.3
            )
        )
        axes[1].text(x + 1.5, 1.575, label, ha="center", va="center")
    axes[1].text(
        5.5,
        0.35,
        "Persistently underused topics are recycled from high-loss contexts",
        ha="center",
        color="#334155",
    )
    figure.subplots_adjust(hspace=0.15)
    _save(figure, "architecture.png")


def comparison() -> None:
    neural, tomotopy = _methods()
    labels = [f"Neural\nK={neural['topics']}", f"Tomotopy\nK={tomotopy['topics']}"]
    metrics = (
        ("Held-out NLL\n(lower is better)", "test_nll"),
        ("Top-word diversity\n(higher is better)", "top_word_diversity"),
        ("Mean NPMI\n(higher is better)", "mean_npmi"),
        ("Active topics", "active_topics"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.6))
    for axis, (title, key) in zip(axes, metrics, strict=True):
        values = [float(neural[key]), float(tomotopy[key])]
        bars = axis.bar(labels, values, color=COLORS, width=0.65)
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
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
    neural, tomotopy = _methods()
    rows = (neural, tomotopy)
    labels = [f"Neural K={neural['topics']}", f"Tomotopy K={tomotopy['topics']}"]
    x = np.arange(2)
    width = 0.34
    dominant = [row["dominant_eligible_topics"] / row["topics"] for row in rows]
    confident = [row["high_confidence_eligible_topics"] / row["topics"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    first = axes[0].bar(
        x - width / 2, dominant, width, label="dominant", color="#2563eb"
    )
    second = axes[0].bar(
        x + width / 2,
        confident,
        width,
        label="probability >= 0.5",
        color="#93c5fd",
    )
    axes[0].set_xticks(x, labels, rotation=10, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("SOS-evaluable topic proportion")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    counts = [
        *((row["dominant_eligible_topics"], row["topics"]) for row in rows),
        *((row["high_confidence_eligible_topics"], row["topics"]) for row in rows),
    ]
    for bar, (numerator, denominator) in zip([*first, *second], counts, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{numerator}/{denominator}",
            ha="center",
            fontsize=8,
        )

    dominant_sos = [row["dominant_mean_sos"] for row in rows]
    confident_sos = [row["high_confidence_mean_sos"] for row in rows]
    axes[1].bar(x - width / 2, dominant_sos, width, color="#2563eb")
    axes[1].bar(x + width / 2, confident_sos, width, color="#93c5fd")
    axes[1].set_xticks(x, labels, rotation=10, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Compound-balanced mean SOS")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(
        axes[1].patches, [*dominant_sos, *confident_sos], strict=True
    ):
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
    neural, tomotopy = _methods()
    values = [neural["spectra_per_second"], tomotopy["spectra_per_second"]]
    figure, axis = plt.subplots(figsize=(5.7, 3.7))
    bars = axis.bar(
        ["Neural\none pass", "Tomotopy\n100 iterations"], values, color=COLORS
    )
    axis.set_yscale("log")
    axis.set_ylabel("Six-thread cached throughput (spectra/s, log scale)")
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


def resources() -> None:
    neural, tomotopy = _methods()
    rows = (neural, tomotopy)
    labels = ["Neural", "Tomotopy"]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    minutes = [row["training_seconds"] / 60 for row in rows]
    memory = [row["pipeline_peak_rss_bytes"] / (1024**3) for row in rows]
    for axis, values, title, unit in (
        (axes[0], minutes, "Training wall time", "minutes"),
        (axes[1], memory, "Cumulative pipeline peak RSS", "GiB"),
    ):
        bars = axis.bar(labels, values, color=COLORS)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
    figure.tight_layout()
    _save(figure, "resource_use.png")


def _percent(value: float, numerator: int, denominator: int) -> str:
    return f"{value:.1%} ({numerator}/{denominator})".replace("%", "\\%")


def tables() -> None:
    neural, tomotopy = _methods()
    rows = (
        ("Topics requested", f"{neural['topics']}", f"{tomotopy['topics']}"),
        (
            "Held-out NLL (lower is better)",
            f"{neural['test_nll']:.4f}",
            f"{tomotopy['test_nll']:.4f}",
        ),
        (
            "Top-word diversity",
            f"{neural['top_word_diversity']:.4f}",
            f"{tomotopy['top_word_diversity']:.4f}",
        ),
        ("Mean NPMI", f"{neural['mean_npmi']:.4f}", f"{tomotopy['mean_npmi']:.4f}"),
        (
            "Undefined top-word pairs",
            f"{neural['undefined_pair_fraction']:.1%}".replace("%", "\\%"),
            f"{tomotopy['undefined_pair_fraction']:.1%}".replace("%", "\\%"),
        ),
        (
            "Corpus-active topics",
            str(neural["active_topics"]),
            str(tomotopy["active_topics"]),
        ),
        (
            "Median effective topics per spectrum",
            f"{neural['effective_topics_median']:.2f}",
            f"{tomotopy['effective_topics_median']:.2f}",
        ),
        (
            "Mass-99 distinct-topic equivalents",
            f"{neural['mass99_distinct_topic_equivalents']:.1f}",
            f"{tomotopy['mass99_distinct_topic_equivalents']:.1f}",
        ),
        (
            "Annotation coverage",
            _percent(
                neural["annotation_coverage"],
                neural["annotated_topics"],
                neural["topics"],
            ),
            _percent(
                tomotopy["annotation_coverage"],
                tomotopy["annotated_topics"],
                tomotopy["topics"],
            ),
        ),
        (
            "Dominant SOS-evaluable topics",
            f"{neural['dominant_eligible_topics']}/{neural['topics']}",
            f"{tomotopy['dominant_eligible_topics']}/{tomotopy['topics']}",
        ),
        (
            "Dominant mean SOS",
            f"{neural['dominant_mean_sos']:.4f}",
            f"{tomotopy['dominant_mean_sos']:.4f}",
        ),
        (
            "High-confidence associations",
            str(neural["high_confidence_associated_spectra"]),
            str(tomotopy["high_confidence_associated_spectra"]),
        ),
        (
            "Six-thread cached spectra/s",
            f"{neural['spectra_per_second']:,.1f}",
            f"{tomotopy['spectra_per_second']:,.1f}",
        ),
        (
            "Training time (minutes)",
            f"{neural['training_seconds'] / 60:.1f}",
            f"{tomotopy['training_seconds'] / 60:.1f}",
        ),
        (
            "Cumulative pipeline peak RSS (GiB)",
            f"{neural['pipeline_peak_rss_bytes'] / (1024**3):.2f}",
            f"{tomotopy['pipeline_peak_rss_bytes'] / (1024**3):.2f}",
        ),
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{name} & {left} & {right} \\\\" for name, left, right in rows)
    (GENERATED / "final_comparison_table.tex").write_text(
        body + "\n\\bottomrule\n", encoding="utf-8"
    )


def _generate() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    architecture()
    comparison()
    chemistry()
    speed()
    resources()
    tables()


def _manifest_files() -> tuple[list[Path], list[Path]]:
    source_manifests = RESULTS / "source_manifests"
    comparison_sources = {
        "protocol": RESULTS / "model_bundle/protocol.json",
        "neural_training": source_manifests / "neural_training.json",
        "neural_evaluation": RESULTS / "model_bundle/evaluation.json",
        "tomotopy_evaluation": source_manifests / "tomotopy_evaluation.json",
        "neural_chemistry": RESULTS / "model_bundle/chemistry.json",
        "tomotopy_chemistry": source_manifests / "tomotopy_chemistry.json",
    }
    declared = _json(RESULTS / "comparison.json")["source_sha256"]
    for name, path in comparison_sources.items():
        if path.is_file() and _sha256(path) != declared.get(name):
            raise ValueError(f"comparison source mismatch: {name}")
    evidence = [
        RESULTS / "comparison.json",
        *comparison_sources.values(),
        RESULTS / "model_bundle/provenance.json",
        Path(__file__).resolve(),
    ]
    outputs = [
        FIGURES / "architecture.png",
        FIGURES / "chemistry_coverage.png",
        FIGURES / "final_comparison.png",
        FIGURES / "inference_speed.png",
        FIGURES / "resource_use.png",
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
    _manifest_files()
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
