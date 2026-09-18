"""Publication figures for directed recovery, corroboration and actual spectra.

Chart contracts are in research/cross_model_overlap_20260908/README.md.
Three source-fit curves show variation, not a confidence band. Counts in the
relationship panels are reused fit-pair/topic records, not independent motifs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .cross_model_overlap_evidence import read_overlap

BLUE, ORANGE, GREY = "#226699", "#C47B28", "#777777"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "text.color": "#252525",
            "axes.labelcolor": "#252525",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
        }
    )


def save_figure(fig, path: Path) -> None:
    """Fixed metadata and stable paths permit exact artifact-regeneration checks."""
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "MS2LDA overlap assessment",
        },
    )
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def source_curves(
    data: dict, direction: str, cohort: str, similarity: str, field: str
) -> np.ndarray:
    sources = [
        s
        for s in data["summary"]["sources"]
        if s["direction"] == direction
        and s["cohort"] == cohort
        and s["similarity"] == similarity
    ]
    if len(sources) != 3:
        raise ValueError("plot needs all three source-fit summaries")
    return np.array([s["curves"][field] for s in sources])


def draw_curves(
    ax,
    data,
    direction,
    cohort,
    *,
    color,
    linestyle,
    field="coverage",
    similarity="full_beta",
    variation=True,
):
    """Thin source-fit traces plus a thick mean; never an inferred CI ribbon."""
    x = data["summary"]["thresholds"]
    curves = source_curves(data, direction, cohort, similarity, field)
    if variation:
        for curve in curves:
            ax.plot(
                x, curve, color=color, linestyle=linestyle, linewidth=0.7, alpha=0.5
            )
    ax.plot(x, curves.mean(axis=0), color=color, linestyle=linestyle, linewidth=1.7)


def coverage_axes(ax):
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="Spectral similarity threshold",
        ylabel="Fraction with counterpart",
    )
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)


def recovery_figure(data: dict, output: Path) -> None:
    """Main result: both directions, complete and recurring/evaluable cohorts."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for row, cohort in enumerate(("all", "recurring_evaluable")):
        for col, (source, target, color) in enumerate(
            (("selected", "tomotopy", BLUE), ("tomotopy", "selected", ORANGE))
        ):
            ax = axes[row, col]
            draw_curves(
                ax, data, f"{source}_to_{target}", cohort, color=color, linestyle="-"
            )
            draw_curves(
                ax, data, f"{source}_to_{source}", cohort, color=GREY, linestyle="--"
            )
            coverage_axes(ax)
            population = "All" if row == 0 else "Recurring, evaluable"
            model = "neural" if source == "selected" else "Tomotopy"
            ax.set_title(f"{'ABCD'[row*2+col]}  {population} {model}", loc="left")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="Neural to Tomotopy"),
            Line2D([0], [0], color=ORANGE, label="Tomotopy to neural"),
            Line2D([0], [0], color=GREY, linestyle="--", label="Repeat-run benchmark"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.10, top=0.87, wspace=0.28, hspace=0.48
    )
    save_figure(fig, output / "cross_model_recovery")


def agreement_figure(data: dict, rows: np.ndarray, output: Path) -> None:
    """Chemical concordance, second neighbours and channel-weight sensitivity."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    cohort = data["protocol"]["cohorts"].index("recurring_evaluable")
    models = np.array([f["model"] for f in data["inventory"]["fits"]])
    recurring = rows[(rows["cohort"] == cohort) & (rows["similarity"] == 0)]
    densities = []
    for col, source in enumerate(("selected", "tomotopy")):
        ax = axes[0, col]
        chosen = recurring[
            (models[recurring["source_fit"]] == source)
            & (models[recurring["target_fit"]] != source)
        ]
        valid = np.isfinite(chosen["feature_effect_cosine"])
        plot = ax.hexbin(
            chosen["best_score"][valid],
            chosen["feature_effect_cosine"][valid],
            gridsize=(24, 14),
            extent=(0, 1, -1, 1),
            mincnt=1,
            bins="log",
            cmap="Blues",
            linewidths=0,
        )
        ax.axhline(0, color="#555555", linestyle=":", linewidth=0.6)
        ax.set(
            xlim=(0, 1),
            ylim=(-1, 1),
            xlabel="Full-beta spectral cosine",
            ylabel="Chemical-effect cosine",
        )
        direction = (
            "Neural to Tomotopy" if source == "selected" else "Tomotopy to neural"
        )
        ax.set_title(f"{'AB'[col]}  {direction}", loc="left")
        densities.append(plot)
    norm = LogNorm(vmin=1, vmax=max(p.get_array().max() for p in densities))
    for plot in densities:
        plot.set_norm(norm)
    colorbar = fig.colorbar(
        densities[0],
        cax=fig.add_axes([0.36, 0.525, 0.31, 0.015]),
        orientation="horizontal",
    )
    colorbar.set_label("Matched records per hexagon", fontsize=7)
    ax = axes[1, 0]
    for direction, color, style in (
        ("selected_to_tomotopy", BLUE, "-"),
        ("tomotopy_to_selected", ORANGE, "--"),
    ):
        draw_curves(
            ax,
            data,
            direction,
            "recurring_evaluable",
            color=color,
            linestyle=style,
            field="multiple",
        )
    coverage_axes(ax)
    ax.set(
        ylabel="Fraction with 2+ counterparts",
        title="C  Multiple spectral counterparts",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="Neural to Tomotopy"),
            Line2D([0], [0], color=ORANGE, linestyle="--", label="Tomotopy to neural"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )
    ax = axes[1, 1]
    for direction, color in (
        ("selected_to_tomotopy", BLUE),
        ("tomotopy_to_selected", ORANGE),
    ):
        for similarity, style in (("full_beta", "--"), ("balanced_channels", "-")):
            draw_curves(
                ax,
                data,
                direction,
                "recurring_evaluable",
                color=color,
                linestyle=style,
                similarity=similarity,
                variation=False,
            )
    coverage_axes(ax)
    ax.set_title("D  Equal-channel sensitivity", loc="left")
    ax.legend(
        handles=[
            Line2D([0], [0], color=GREY, linestyle="--", label="Full beta"),
            Line2D([0], [0], color=GREY, label="Equal channels"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )
    fig.subplots_adjust(
        left=0.095, right=0.98, bottom=0.09, top=0.94, wspace=0.32, hspace=0.77
    )
    save_figure(fig, output / "cross_model_agreement")


def stick_pair(
    ax, source: dict, target: dict, channel: str, maximum_mass: float
) -> None:
    """Mirror top-word probabilities, scaled to each topic's joint maximum."""
    for spectrum, sign, color in ((source, 1, BLUE), (target, -1, ORANGE)):
        words = spectrum["words"]
        scale = max(w["probability"] for w in words)
        selected = [w for w in words if w["token"].startswith(channel + "@")]
        masses = [float(w["token"].split("@")[1]) for w in selected]
        values = [sign * w["probability"] / scale for w in selected]
        ax.vlines(masses, 0, values, color=color, linewidth=1.15)
        if values:
            strongest = int(np.argmax(np.abs(values)))
            ax.text(
                masses[strongest] + maximum_mass * 0.01,
                sign * max(0.28, abs(values[strongest]) - 0.18),
                f"{masses[strongest]:.2f}",
                fontsize=6.5,
                color=color,
            )
    ax.axhline(0, color="#777777", linewidth=0.55)
    ax.set(xlim=(0, max(20, maximum_mass * 1.04)), ylim=(-1.05, 1.05))
    ax.set_yticks([-1, 0, 1], labels=["1", "0", "1"])
    ax.set_xlabel("Fragment m/z" if channel == "frag" else "Neutral-loss mass (Da)")
    ax.tick_params(labelsize=7)


def examples_figure(data: dict, output: Path) -> None:
    """Show two score-quantile pairs and one source with its two best targets."""
    examples = [
        e
        for e in data["examples"]["examples"]
        if e["direction"] == "selected_to_tomotopy"
    ]
    panels = [
        ("A  Median-score pair", examples[0], 0),
        ("B  Upper-decile pair", examples[1], 0),
        ("C  One source, first counterpart", examples[2], 0),
        ("D  Same source, second counterpart", examples[2], 1),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 6.4))
    maximum_masses = {
        channel: max(
            float(w["token"].split("@")[1])
            for e in examples
            for spectrum in [e["source"]] + e["targets"]
            for w in spectrum["words"]
            if w["token"].startswith(channel + "@")
        )
        for channel in ("frag", "loss")
    }
    for row, (title, example, target_index) in enumerate(panels):
        source, target = example["source"], example["targets"][target_index]
        score = example["best_score"] if target_index == 0 else example["second_score"]
        for col, channel in enumerate(("frag", "loss")):
            stick_pair(axes[row, col], source, target, channel, maximum_masses[channel])
        axes[row, 0].set_ylabel("Relative probability")
        axes[row, 0].text(
            0,
            1.19,
            f"{title}: cosine {score:.3f}; support "
            f"{source['support_compounds']} / {target['support_compounds']} compounds",
            transform=axes[row, 0].transAxes,
            fontsize=9,
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="Neural motif (above)"),
            Line2D([0], [0], color=ORANGE, label="Tomotopy counterpart (below)"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.065, top=0.89, wspace=0.23, hspace=0.95
    )
    save_figure(fig, output / "cross_model_examples")


def generate_overlap_figures(evidence: Path, output: Path) -> None:
    data = read_overlap(evidence)
    rows = np.load(evidence / "directed_matches.npy", allow_pickle=False)
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    recovery_figure(data, output)
    agreement_figure(data, rows, output)
    examples_figure(data, output)
