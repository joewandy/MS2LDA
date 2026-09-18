"""Two compact scientific figures from verified, fixed-assessment evidence.

Chart contracts are in research/motif_chemical_assessment_20260908/README.md.
Thin curves preserve fit/pair variation without treating topics or fit pairs
as independent replicates. No smoothing, inferred confidence band or chemical
correctness threshold is added. Matplotlib is confined to artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .chemical_assessment_report import read_assessment_evidence

STYLES = {
    "selected": {"color": "#226699", "linestyle": "-"},
    "tomotopy": {"color": "#C47B28", "linestyle": "--"},
    "cross_model": {"color": "#777777", "linestyle": ":"},
}
LABELS = {
    "selected": "Contextual Sparse ETM",
    "tomotopy": "Tomotopy LDA",
    "cross_model": "Across models",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _ecdf(ax, values, style: dict, bounds: tuple[float, float]) -> None:
    """Plot each observed order statistic against its empirical cumulative mass."""
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("ECDF needs a nonempty finite sample")
    ax.step(
        np.r_[bounds[0], values, bounds[1]],
        np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
        where="post",
        linewidth=1.05,
        alpha=0.8,
        **style,
    )


def _axes():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "text.color": "#252525",
            "axes.labelcolor": "#252525",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "svg.hashsalt": "motif-chemical-assessment-v1",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    for ax in axes:
        ax.set_ylim(0, 1.01)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.24, top=0.79, wspace=0.32)
    return fig, axes


def _save(fig, path: Path) -> None:
    """Fixed PDF metadata makes exact regeneration meaningful for version control."""
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "MS2LDA chemical assessment",
        },
    )
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def generate_figures(evidence: Path, output: Path) -> None:
    """Draw whole distributions; report captions carry cohort and sample context."""
    data = read_assessment_evidence(evidence)
    output.mkdir(parents=True, exist_ok=True)
    primary = _rows(evidence / "primary_50_topics.jsonl")
    fig, axes = _axes()
    for fit in data["protocol"]["fits"]:
        rows = [r for r in primary if r["fit"] == fit["id"]]
        _ecdf(
            axes[0],
            [r["sos_excess"] for r in rows if r["sos_eligible"]],
            STYLES[fit["model"]],
            (-1, 1),
        )
        _ecdf(
            axes[1],
            [r["compounds"] for r in rows if r["compounds"] > 0],
            STYLES[fit["model"]],
            (1, max(r["compounds"] for r in primary)),
        )
    axes[0].axvline(0, color="#444444", linewidth=0.7)
    axes[0].set(
        xlim=(-1, 1),
        xlabel="Excess SOS, observed - background",
        ylabel="Cumulative fraction",
        title="A  Chemical specificity",
    )
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 2, 5, 10, 20, 50], labels=["1", "2", "5", "10", "20", "50"])
    axes[1].set(
        xlim=(1, max(r["compounds"] for r in primary)),
        xlabel="Supporting compounds (log scale)",
        ylabel="Cumulative fraction",
        title="B  Compound support",
    )
    handles = [
        Line2D([0], [0], label=LABELS[m], **STYLES[m]) for m in ("selected", "tomotopy")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
    )
    _save(fig, output / "chemical_specificity")

    matches = _rows(evidence / "topic_matches.jsonl")
    scores = np.load(evidence / "reference_similarities.npz", allow_pickle=False)[
        "cosine"
    ]
    fig, axes = _axes()
    for pair in data["stability"]["pairs"]:
        rows = [
            r
            for r in matches
            if r["left_fit"] == pair["left_fit"] and r["right_fit"] == pair["right_fit"]
        ]
        model = pair["kind"].removeprefix("within_")
        _ecdf(axes[0], [r["beta_cosine"] for r in rows], STYLES[model], (0, 1))
    thresholds = np.linspace(0, 1, 201)
    for i, fit in enumerate(data["protocol"]["fits"]):
        matrix = scores[i * 1000 : (i + 1) * 1000]
        # Invalid embeddings never acquire a positive hit. The denominator
        # remains ALL 1,000 topics, not only the high-scoring subset.
        best = np.max(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
        coverage = (best[:, None] >= thresholds).mean(axis=0)
        axes[1].plot(
            thresholds, coverage, linewidth=1.05, alpha=0.8, **STYLES[fit["model"]]
        )
    axes[0].set(
        xlim=(0, 1),
        xlabel="Matched full-beta cosine",
        ylabel="Cumulative fraction",
        title="A  Spectral repeatability",
    )
    axes[1].set(
        xlim=(0, 1),
        xlabel="Best-reference cosine threshold",
        ylabel="Topic coverage",
        title="B  MotifDB correspondence",
    )
    handles = [Line2D([0], [0], label=LABELS[m], **STYLES[m]) for m in STYLES]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        fontsize=8,
    )
    _save(fig, output / "motif_correspondence")
