"""Generate the paper-focused mean-evidence comparison from hashed evidence."""

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
SOURCES = RESULTS / "source_manifests"
DOCS = REPO / "docs/research"
FIGURES = DOCS / "figures"
GENERATED = DOCS / "generated"
MANIFEST = DOCS / "report_manifest.json"
COLORS = ("#2563eb", "#f59e0b", "#64748b")
LABELS = ("Current neural", "Mean-evidence neural", "Tomotopy")


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


def _comparison() -> dict[str, Any]:
    return _json(RESULTS / "comparison.json")


def _methods() -> list[dict[str, Any]]:
    rows = _comparison()["methods"]
    expected = ("current_neural", "candidate_neural", "tomotopy")
    if tuple(row["method"] for row in rows) != expected:
        raise ValueError("unexpected report method order")
    return rows


def architecture() -> None:
    figure, axis = plt.subplots(figsize=(11.0, 4.4))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4.4)
    axis.axis("off")
    boxes = (
        (0.1, "Sparse\nspectrum", "#dbeafe"),
        (2.5, "Top-2 token\nrouting", "#bfdbfe"),
        (4.9, "Token topic\nmass", "#bfdbfe"),
        (7.3, "Fixed document\ngate $g_d^{0.75}$", "#fde68a"),
        (9.7, "Normalized\n$\\theta_d$", "#d1fae5"),
    )
    for x, label, color in boxes:
        axis.add_patch(
            plt.Rectangle((x, 2.25), 1.9, 0.9, facecolor=color, edgecolor="#334155")
        )
        axis.text(x + 0.95, 2.7, label, ha="center", va="center")
    for start, end in ((2.0, 2.5), (4.4, 4.9), (6.8, 7.3), (9.2, 9.7)):
        axis.annotate(
            "",
            xy=(end, 2.7),
            xytext=(start, 2.7),
            arrowprops={"arrowstyle": "->"},
        )
    axis.text(
        6,
        3.85,
        r"$g_d=\mathrm{softmax}(\mathrm{document\ logits}/T)$, "
        r"$\theta_d=\mathrm{normalize}(\mathrm{token\ mass}\,g_d^{0.75})$",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    decoder_boxes = (
        (0.1, "Neural topic-word\nlogits", "#ede9fe"),
        (3.0, "Mean log evidence per\nfragment/loss vocabulary", "#ddd6fe"),
        (5.9, "Pull type mass 25%\ntoward 50:50", "#c4b5fd"),
        (8.8, "Joint fragment-loss\nmotifs", "#a78bfa"),
    )
    for x, label, color in decoder_boxes:
        axis.add_patch(
            plt.Rectangle((x, 0.45), 2.3, 0.9, facecolor=color, edgecolor="#334155")
        )
        axis.text(x + 1.15, 0.9, label, ha="center", va="center")
    for start, end in ((2.4, 3.0), (5.3, 5.9), (8.2, 8.8)):
        axis.annotate(
            "",
            xy=(end, 0.9),
            xytext=(start, 0.9),
            arrowprops={"arrowstyle": "->"},
        )
    _save(figure, "architecture.png")


def comparison() -> None:
    rows = _methods()
    metrics = (
        ("Optimized motifs", "optimized_motifs"),
        ("High-confidence\nevaluable motifs", "high_confidence_eligible_topics"),
        (
            "Useful high-confidence motifs\n(SOS >= 0.6)",
            "useful_high_confidence_motifs",
        ),
        ("Model-fitting time\n(minutes)", "training_seconds"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 3.7))
    for axis, (title, key) in zip(axes, metrics, strict=True):
        values = [float(row[key]) for row in rows]
        if key == "training_seconds":
            values = [value / 60 for value in values]
        bars = axis.bar(LABELS, values, color=COLORS, width=0.68)
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", labelrotation=18, labelsize=7)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}" if key == "training_seconds" else f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout()
    _save(figure, "final_comparison.png")


def chemistry() -> None:
    rows = _methods()
    x = np.arange(len(rows))
    high = np.asarray([row["sos_bands"]["high_gt_0_8"] for row in rows])
    intermediate = np.asarray(
        [row["sos_bands"]["intermediate_0_6_to_0_8"] for row in rows]
    )
    low = np.asarray([row["sos_bands"]["low_lt_0_6"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(x, low, color="#cbd5e1", label="Low < 0.6")
    axes[0].bar(x, intermediate, bottom=low, color="#60a5fa", label="0.6–0.8")
    axes[0].bar(
        x,
        high,
        bottom=low + intermediate,
        color="#1d4ed8",
        label="High > 0.8",
    )
    axes[0].set_xticks(x, LABELS, rotation=12, ha="right")
    axes[0].set_ylabel("High-confidence SOS-evaluable motifs")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.34
    means = [row["high_confidence_mean_sos"] for row in rows]
    medians = [row["high_confidence_median_sos"] for row in rows]
    axes[1].bar(x - width / 2, means, width, color="#2563eb", label="Mean")
    axes[1].bar(x + width / 2, medians, width, color="#93c5fd", label="Median")
    axes[1].set_xticks(x, LABELS, rotation=12, ha="right")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel("Compound-balanced SOS")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save(figure, "chemistry_coverage.png")


def speed() -> None:
    evidence = _comparison()["secondary_warm_in_memory_batch_inference"]
    values = (
        evidence["candidate_neural_spectra_per_second"],
        evidence["tomotopy_spectra_per_second"],
    )
    figure, axis = plt.subplots(figsize=(5.7, 3.7))
    bars = axis.bar(
        ("Mean-evidence neural", "Tomotopy"),
        values,
        color=(COLORS[1], COLORS[2]),
    )
    axis.set_yscale("log")
    axis.set_ylabel("Warm in-memory batch inference (spectra/s)")
    axis.grid(axis="y", which="both", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:,.1f}",
            ha="center",
        )
    _save(figure, "inference_speed.png")


def tables() -> None:
    rows = _methods()
    metrics = (
        ("Optimized motifs", lambda row: str(row["optimized_motifs"])),
        (
            "High-confidence SOS-evaluable motifs",
            lambda row: str(row["high_confidence_eligible_topics"]),
        ),
        ("SOS high ($>0.8$)", lambda row: str(row["sos_bands"]["high_gt_0_8"])),
        (
            "SOS intermediate ($0.6$--$0.8$)",
            lambda row: str(row["sos_bands"]["intermediate_0_6_to_0_8"]),
        ),
        ("SOS low ($<0.6$)", lambda row: str(row["sos_bands"]["low_lt_0_6"])),
        (
            "Useful high-confidence motifs (SOS $\\geq0.6$)",
            lambda row: str(row["useful_high_confidence_motifs"]),
        ),
        (
            "Mean high-confidence SOS",
            lambda row: f"{row['high_confidence_mean_sos']:.4f}",
        ),
        (
            "Median high-confidence SOS",
            lambda row: f"{row['high_confidence_median_sos']:.4f}",
        ),
        (
            "Model-fitting time (minutes)",
            lambda row: f"{row['training_seconds'] / 60:.1f}",
        ),
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"{name} & " + " & ".join(render(row) for row in rows) + " \\\\"
        for name, render in metrics
    )
    (GENERATED / "final_comparison_table.tex").write_text(
        body + "\n\\bottomrule\n", encoding="utf-8"
    )
    confirmation = _comparison()["test_confirmation"]
    test_rows = (confirmation["candidate_neural"], confirmation["tomotopy"])
    test_metrics = (
        ("Optimized motifs", lambda row: str(row["optimized_motifs"])),
        (
            "High-confidence SOS-evaluable motifs",
            lambda row: str(row["high_confidence_eligible_topics"]),
        ),
        ("SOS high ($>0.8$)", lambda row: str(row["sos_bands"]["high_gt_0_8"])),
        (
            "SOS intermediate ($0.6$--$0.8$)",
            lambda row: str(row["sos_bands"]["intermediate_0_6_to_0_8"]),
        ),
        ("SOS low ($<0.6$)", lambda row: str(row["sos_bands"]["low_lt_0_6"])),
        (
            "Useful high-confidence motifs (SOS $\\geq0.6$)",
            lambda row: str(row["useful_high_confidence_motifs"]),
        ),
        (
            "Mean high-confidence SOS",
            lambda row: f"{row['high_confidence_mean_sos']:.4f}",
        ),
        (
            "Median high-confidence SOS",
            lambda row: f"{row['high_confidence_median_sos']:.4f}",
        ),
        ("Held-out completion NLL", lambda row: f"{row['test_nll']:.4f}"),
    )
    test_body = "\n".join(
        f"{name} & " + " & ".join(render(row) for row in test_rows) + " \\\\"
        for name, render in test_metrics
    )
    (GENERATED / "test_confirmation_table.tex").write_text(
        test_body + "\n\\bottomrule\n", encoding="utf-8"
    )


def _generate() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    architecture()
    comparison()
    chemistry()
    speed()
    tables()


def _manifest_files() -> tuple[list[Path], list[Path]]:
    sources = {
        "candidate_training": SOURCES / "candidate_training.json",
        "candidate_evaluation": SOURCES / "candidate_validation.json",
        "current_evaluation": SOURCES / "current_validation.json",
        "tomotopy_evaluation": SOURCES / "tomotopy_validation.json",
        "candidate_chemistry": SOURCES / "candidate_validation_chemistry.json",
        "current_chemistry": SOURCES / "current_validation_chemistry.json",
        "tomotopy_chemistry": SOURCES / "tomotopy_validation_chemistry.json",
        "validation_gate": SOURCES / "validation_gate.json",
        "candidate_test_evaluation": SOURCES / "candidate_test.json",
        "candidate_test_access": SOURCES / "candidate_test_access.json",
        "candidate_test_chemistry": SOURCES / "candidate_test_chemistry.json",
        "tomotopy_test_evaluation": SOURCES / "tomotopy_evaluation.json",
        "tomotopy_test_chemistry": SOURCES / "tomotopy_chemistry.json",
    }
    declared = _comparison()["source_sha256"]
    for name, path in sources.items():
        if _sha256(path) != declared[name]:
            raise ValueError(f"comparison source mismatch: {name}")
    evidence = [
        RESULTS / "comparison.json",
        *sources.values(),
        Path(__file__).resolve(),
    ]
    outputs = [
        FIGURES / "architecture.png",
        FIGURES / "chemistry_coverage.png",
        FIGURES / "final_comparison.png",
        FIGURES / "inference_speed.png",
        GENERATED / "final_comparison_table.tex",
        GENERATED / "test_confirmation_table.tex",
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
            path = REPO / name
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
