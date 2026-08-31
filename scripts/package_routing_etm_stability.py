"""Package the validation-only Routing ETM training-seed stability study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

METHOD = "etm_balanced_routing_top2_entmax15_raw_counts"
DEFAULT_OUTPUT = Path(
    "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_stability",
)
BASELINE_RESULTS = Path(
    "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm",
)
COMPACT_FILES = (
    "config.json",
    "metrics.json",
    "chemical_validation.json",
    "chemical_scores.csv",
    "training_history.csv",
    "theta_support_summary.csv",
    "routing_evidence_support_summary.csv",
    "duplicate_component_summary.json",
    "fragment_mass_summary.json",
    "top_words.csv",
    "validation_access_audit.json",
    "provenance.json",
)
SUMMARY_FIELDS = (
    "optimized_motifs",
    "evaluable_motifs",
    "useful_motifs",
    "mean_sos",
    "median_sos",
    "completion_nll",
    "median_effective_topics",
    "median_exact_support",
    "unique_top1_topics",
    "corpus_effective_topics",
    "learned_context_scale",
    "training_wall_seconds",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        message = f"expected a JSON object: {path}"
        raise TypeError(message)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_file(path: Path, repo_root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        message = f"{label}: expected {expected!r}, found {actual!r}"
        raise ValueError(message)


def seed_row(
    metrics: Mapping[str, Any],
    *,
    training_seed: int,
    data_split_seed: int = 42,
) -> dict[str, object]:
    """Extract the audit table row from one complete metrics object."""
    chemistry = metrics["validation_chemistry"]
    support = metrics["theta_support"]
    inventory = metrics["topic_inventory"]
    bands = chemistry["sos_bands"]
    _assert_equal(
        "SOS bands",
        sum(int(value) for value in bands.values()),
        int(chemistry["eligible_topics"]),
    )
    _assert_equal(
        "useful motifs",
        int(bands["high_gt_0_8"]) + int(bands["intermediate_0_6_to_0_8"]),
        int(chemistry["useful_motifs"]),
    )
    return {
        "training_seed": int(training_seed),
        "data_split_seed": int(data_split_seed),
        "optimized_motifs": int(chemistry["optimized_motifs"]),
        "evaluable_motifs": int(chemistry["eligible_topics"]),
        "useful_motifs": int(chemistry["useful_motifs"]),
        "mean_sos": float(chemistry["mean_sos"]),
        "median_sos": float(chemistry["median_sos"]),
        "completion_nll": float(metrics["document_completion"]["nll_per_token"]),
        "median_effective_topics": float(
            inventory["median_effective_topics_per_spectrum"],
        ),
        "median_exact_support": float(support["median_exact_support"]),
        "unique_top1_topics": int(inventory["unique_top1_topics"]),
        "corpus_effective_topics": float(inventory["corpus_effective_topic_count"]),
        "learned_context_scale": float(metrics["learned_context_scale"]),
        "training_wall_seconds": float(metrics["runtime"]["training_wall_seconds"]),
        "finite_stable": bool(metrics["finite_stable"]),
        "catastrophic_duplicate_component": bool(
            inventory["catastrophic_duplicate_component"],
        ),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    comparators: Mapping[str, Any],
) -> dict[str, Any]:
    """Return descriptive stability statistics and direction checks."""
    aggregates: dict[str, dict[str, float]] = {}
    for field in SUMMARY_FIELDS:
        values = [float(row[field]) for row in rows]
        aggregates[field] = {
            "mean": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
            "sample_standard_deviation": statistics.stdev(values),
        }
    m1 = comparators["m1"]
    tomotopy = comparators["tomotopy"]
    return {
        "training_seeds": [int(row["training_seed"]) for row in rows],
        "runs": len(rows),
        "aggregate": aggregates,
        "direction_checks": {
            "all_finite_stable": all(bool(row["finite_stable"]) for row in rows),
            "no_catastrophic_duplicate_component_on_any_seed": all(
                not bool(row["catastrophic_duplicate_component"]) for row in rows
            ),
            "all_exceed_m1_evaluable_motifs": all(
                int(row["evaluable_motifs"]) > int(m1["evaluable_motifs"])
                for row in rows
            ),
            "all_exceed_m1_useful_motifs": all(
                int(row["useful_motifs"]) > int(m1["useful_motifs"]) for row in rows
            ),
            "all_exceed_tomotopy_evaluable_motifs": all(
                int(row["evaluable_motifs"]) > int(tomotopy["evaluable_motifs"])
                for row in rows
            ),
            "all_exceed_tomotopy_useful_motifs": all(
                int(row["useful_motifs"]) > int(tomotopy["useful_motifs"])
                for row in rows
            ),
            "all_below_m1_optimized_motifs": all(
                int(row["optimized_motifs"]) < int(m1["optimized_motifs"])
                for row in rows
            ),
            "all_below_m1_mean_sos": all(
                float(row["mean_sos"]) < float(m1["mean_sos"]) for row in rows
            ),
            "all_worse_than_m1_completion_nll": all(
                float(row["completion_nll"]) > float(m1["completion_nll"])
                for row in rows
            ),
            "candidate_test_remained_locked": True,
        },
        "interpretation": (
            "Discovery breadth is stable across training seeds; optimized coverage, "
            "mean SOS, and likelihood remain reproducible trade-offs."
        ),
    }


def _copy_seed_artifacts(source: Path, destination: Path, seed: int) -> None:
    model_dir = source.expanduser().resolve(strict=True) / "models" / METHOD
    config = _read_json(model_dir / "config.json")
    _assert_equal("training seed", int(config["seed"]), seed)
    _assert_equal("data split seed", int(config["data_split_seed"]), 42)
    for name in COMPACT_FILES:
        source_path = model_dir / name
        if not source_path.is_file():
            message = f"missing seed-{seed} artifact: {source_path}"
            raise FileNotFoundError(message)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination / name)


def package_stability(
    *,
    repo_root: Path,
    output_dir: Path,
    seed_runs: Mapping[int, Path],
) -> dict[str, Any]:
    """Copy compact evidence and write the multiseed summary and manifest."""
    root = repo_root.resolve()
    output = (root / output_dir).resolve()
    if not output.is_relative_to(root):
        message = "stability output must remain inside the repository"
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)

    for seed, run in sorted(seed_runs.items()):
        _copy_seed_artifacts(run, output / f"seed_{seed}", seed)

    baseline_metrics = _read_json(root / BASELINE_RESULTS / "metrics.json")
    baseline_config = _read_json(root / BASELINE_RESULTS / "config.json")
    rows = [
        seed_row(
            baseline_metrics,
            training_seed=int(baseline_config["seed"]),
        ),
    ]
    for seed in sorted(seed_runs):
        metrics = _read_json(output / f"seed_{seed}" / "metrics.json")
        rows.append(seed_row(metrics, training_seed=seed))

    frozen_manifest = _read_json(root / BASELINE_RESULTS / "checkpoint_manifest.json")
    comparators = frozen_manifest["expected_validation_metrics"]
    summary = {
        "schema_version": 1,
        "method": METHOD,
        "evidence_boundary": "same frozen training/validation split; test locked",
        "only_changed_factor": "model initialization and minibatch-order seed",
        "comparators": {
            "m1": comparators["m1"],
            "tomotopy": comparators["tomotopy"],
        },
        "by_seed": rows,
        **aggregate_rows(rows, comparators),
    }
    _write_json(output / "stability_summary.json", summary)
    with (output / "stability_by_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    implementation_paths = (
        Path("benchmarks/neural_ms2lda/contextual_sparse_etm.py"),
        Path("benchmarks/neural_ms2lda/model_evaluation.py"),
        Path("benchmarks/neural_ms2lda/reproducibility.py"),
        Path("benchmarks/neural_ms2lda/topic_model_training.py"),
        Path("scripts/run_contextual_sparse_etm.py"),
        Path("scripts/package_routing_etm_stability.py"),
        Path("scripts/verify_routing_etm_stability.py"),
        Path("environment.yml"),
    )
    packaged_paths = [
        output / "stability_summary.json",
        output / "stability_by_seed.csv",
    ]
    for seed in sorted(seed_runs):
        packaged_paths.extend(output / f"seed_{seed}" / name for name in COMPACT_FILES)
    external_artifacts: dict[str, Any] = {}
    validation_inputs: list[dict[str, Any]] | None = None
    for seed in sorted(seed_runs):
        provenance = _read_json(output / f"seed_{seed}" / "provenance.json")
        _assert_equal(
            f"seed {seed} candidate test access",
            actual=provenance["candidate_test_artifacts_accessed"],
            expected=False,
        )
        _assert_equal(
            f"seed {seed} candidate test metrics",
            actual=provenance["candidate_test_metrics_inspected"],
            expected=False,
        )
        linked = provenance["validation_inputs"]["linked_inputs"]
        if validation_inputs is None:
            validation_inputs = linked
        else:
            _assert_equal(f"seed {seed} validation inputs", linked, validation_inputs)
        external_artifacts[str(seed)] = provenance["local_artifacts"]

    source_implementation_commit = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "checkpoint_id": "routing-etm-real-training-seed-stability-20260830",
        "baseline_checkpoint_commit": "1bfac58ec7efc6720839a7c65c9e98ade3536027",
        "source_implementation_commit": source_implementation_commit,
        "method": METHOD,
        "data_split_seed": 42,
        "training_seeds": [int(row["training_seed"]) for row in rows],
        "candidate_test_artifacts_accessed": False,
        "candidate_test_metrics_inspected": False,
        "implementation_files": [
            _hashed_file(root / path, root) for path in implementation_paths
        ],
        "packaged_files": [_hashed_file(path, root) for path in packaged_paths],
        "source_runs": {
            str(seed): str(path.resolve()) for seed, path in seed_runs.items()
        },
        "external_local_artifacts": external_artifacts,
        "validation_inputs": validation_inputs,
    }
    _write_json(output / "checkpoint_manifest.json", manifest)
    return summary


def _parse_seed_run(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", maxsplit=1)
        return int(seed_text), Path(path_text)
    except (TypeError, ValueError) as exc:
        message = "--seed-run must have the form SEED=/absolute/run/path"
        raise argparse.ArgumentTypeError(message) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Package two or more completed stability runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-run",
        action="append",
        required=True,
        type=_parse_seed_run,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seed_runs = dict(args.seed_run)
    if len(seed_runs) != len(args.seed_run):
        parser.error("training seeds must be unique")
    result = package_stability(
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=args.output_dir,
        seed_runs=seed_runs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
