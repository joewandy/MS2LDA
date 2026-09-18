"""Collect compact validation evidence without promoting an experimental model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.neural_ms2lda.reproducibility import sha256_file, write_csv_rows
from benchmarks.neural_ms2lda.utils import write_json

SOS_CUTOFFS = (0.5, 0.6, 0.7, 0.8, 0.9)


def sos_sensitivity(evaluation: dict) -> dict[str, int]:
    """Count eligible motifs above fixed inclusive SOS cutoffs, without refitting."""
    values = [
        float(row["sos"]) for row in evaluation["topic_scores"] if row["eligible"]
    ]
    if len(values) != evaluation["eligible_topics"] or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in values
    ):
        raise ValueError("invalid eligible SOS distribution")
    counts = {
        f"sos_count_ge_{cutoff:.1f}": sum(v >= cutoff for v in values)
        for cutoff in SOS_CUTOFFS
    }
    bands = evaluation["sos_bands"]
    if counts["sos_count_ge_0.6"] != (
        bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
    ):
        raise ValueError("SOS sensitivity disagrees with the existing useful count")
    return counts


def repository_code_path(name: str) -> str:
    """Make archived code hashes portable across the user's Linux/Mac clones."""
    path = Path(name)
    if not path.is_absolute():
        return str(path)
    for anchor in ("benchmarks", "scripts"):
        if anchor in path.parts:
            return str(Path(*path.parts[path.parts.index(anchor) :]))
    raise ValueError(f"unexpected implementation path in evidence: {name}")


def summarize(root: Path, output: Path) -> list[dict]:
    """Retain every completed run, including weak candidates and ablations."""
    rows, sources = [], []
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        config = result.get("config", {})
        if "variant" not in config or "test_matrices_loaded" not in config:
            continue
        if config["test_matrices_loaded"] or config["selection_split"] != "validation":
            raise ValueError(f"unexpected evidence boundary: {path}")
        metrics = result["metrics"]
        inventory = metrics["topic_inventory"]
        truth = metrics.get("truth", {})
        row = {
            "run": str(path.parent.relative_to(root)),
            "data": "synthetic" if truth else "MSnLib validation",
            "seed": config["seed"],
            "training_seed": config["training_seed"],
            "topics": config["topics"],
            "variant": config["variant"],
            "normalization_statistics": config.get(
                "normalization_statistics",
                "ema" if "batchnorm" in config["variant"] else "not_applicable",
            ),
            "reused_training_weights": "parent_checkpoint_sha256" in config,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "hidden": config.get("hidden", ""),
            "learning_rate": config.get("learning_rate", ""),
            "momentum": config["momentum"],
            "weight_decay": config.get("weight_decay", 1.2e-6),
            "count_scaling": config["count_scaling"],
            "concentration": (
                config["concentration"]
                if config["variant"].startswith(("prior", "dirichlet"))
                or config["variant"] == "dvae_poe"
                else ""
            ),
            "kl_warmup_epochs": config.get("kl_warmup_epochs", 0),
            "decoder": config.get("decoder", "additive_mixture"),
            "beta_meaning": config.get("beta_meaning", "emissions"),
            "framework": config.get("framework", "pytorch"),
            "completion_nll": metrics["completion"]["nll_per_token"],
            "median_effective_topics": metrics["support"][
                "median_effective_topics_per_spectrum"
            ],
            "median_exact_support": metrics["support"]["median_exact_support"],
            "p95_exact_support": metrics["support"]["support_size_percentiles"]["95"],
            "unique_top1_topics": inventory["unique_top1_topics"],
            "corpus_effective_topics": inventory["corpus_effective_topic_count"],
            "nearest_beta_cosine": inventory["mean_nearest_topic_beta_cosine"],
            "catastrophic_duplicate": inventory["catastrophic_duplicate_component"],
            "beta_recovery": truth.get("true_beta_matched_cosine_mean", ""),
            "theta_recovery": truth.get("true_theta_cosine_mean", ""),
            "recovered_motifs": truth.get(
                "planted_motifs_recovered_cosine_ge_0_50", ""
            ),
            "training_seconds": result["training_seconds"],
            "parameters": config["parameters"],
            "optimized_motifs": "",
            "evaluable_motifs": "",
            "useful_motifs": "",
            "mean_sos": "",
            **{f"sos_count_ge_{cutoff:.1f}": "" for cutoff in SOS_CUTOFFS},
        }
        # Real outputs are <run>/models/minimal_etm/result.json.
        chemistry_path = (
            path.parents[2] / "validation_chemical/minimal_etm/complete.json"
        )
        if not truth and chemistry_path.is_file():
            chemistry = json.loads(chemistry_path.read_text())
            if chemistry["split"] != "validation":
                raise ValueError("chemistry must be validation only")
            if not chemistry["heldout_compounds_excluded_from_mag"]:
                raise ValueError(
                    "chemistry reference index contains held-out compounds"
                )
            failures = chemistry["mag_failures"]
            if failures["clustering_count"] or failures["optimization_count"]:
                raise ValueError("resolve MAG failures before summarizing usefulness")
            sources.append(
                {
                    "path": str(chemistry_path.relative_to(root)),
                    "sha256": sha256_file(chemistry_path),
                }
            )
            evaluation = chemistry["chemical_evaluation"]
            bands = evaluation["sos_bands"]
            row.update(
                {
                    "optimized_motifs": round(
                        chemistry["annotation_coverage"] * chemistry["topics"]
                    ),
                    "evaluable_motifs": evaluation["eligible_topics"],
                    "useful_motifs": bands["high_gt_0_8"]
                    + bands["intermediate_0_6_to_0_8"],
                    "mean_sos": evaluation["mean_sos"],
                    **sos_sensitivity(evaluation),
                }
            )
        rows.append(row)
        sources.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "code_sha256": {
                    repository_code_path(name): digest
                    for name, digest in config["code_sha256"].items()
                },
            }
        )
    if not rows:
        raise ValueError("no minimal ETM runs found")
    write_csv_rows(output / "runs.csv", rows)
    write_json(
        output / "manifest.json",
        {"stage": "validation development", "sources": sources},
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(args.root.resolve(), args.output.resolve())
    print(f"Wrote {len(rows)} completed runs to {args.output}")


if __name__ == "__main__":
    main()
