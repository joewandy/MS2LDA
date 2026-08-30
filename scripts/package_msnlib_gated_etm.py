"""Package reviewable validation-only evidence for the gated-ETM campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import scipy
import torch

from benchmarks.neural_ms2lda.data import load_vocabulary
from benchmarks.neural_ms2lda.diagnostics import model_selection_diagnostics
from benchmarks.neural_ms2lda.followup import theta_distribution
from benchmarks.neural_ms2lda.utils import read_json, write_json
from scripts.run_msnlib_model_comparison import (
    MODEL_SELECTION_EVALUATION_PROTOCOL,
)

BASE_COMMIT = "a2cd7e201ac49a5f5f95aff40889a186fe4757ae"
USEFUL_SOS_THRESHOLD = 0.6
MEMBERSHIP_THRESHOLD = 0.5
M1_TRAINING_WALL_SECONDS = 1764.874561500037
M1_PARAMETERS = 167_168
GATES = {
    "optimized_motifs": 840,
    "evaluable_motifs": 388,
    "useful_motifs": 252,
    "mean_sos": 0.651498,
    "maximum_completion_nll": 9.422847,
}
BALANCED_LOCKED_HASHES = {
    "weights.pt": "20b8e183d39615ec0452314f7bd225ba7f2920641a1619e049bb757eff87c8df",
    "beta.npy": "4d6d1d533cff3bfe05e14fa3fa9c7b5999358616a1afcccd7a175fe7a52b61b2",
    "validation_full_theta.npy": (
        "f0a764cb8e8f17e9f0c32ceb9e33a89df8613a3f637a3512422915a452cbdcde"
    ),
}
REVIEWABLE_MODEL_FILES = (
    "config.json",
    "metrics.json",
    "training_history.csv",
    "chemical_scores.csv",
    "chemical_validation.json",
    "top_words.csv",
    "fragment_mass_summary.json",
    "duplicate_component_summary.json",
    "theta_distribution.csv",
    "validation_access_audit.json",
)


def sha256(path: Path) -> str:
    """Hash one local artifact without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path) -> dict[str, Any]:
    """Return absolute provenance for one existing file."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic small CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def chemistry_headline(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the frozen chemistry fields from one validation result."""
    summary = result["high_confidence_chemistry"]
    valid_contract = (
        result["split"] == "validation"
        and float(summary["membership_threshold"]) == MEMBERSHIP_THRESHOLD
        and bool(result["heldout_compounds_excluded_from_mag"])
    )
    if not valid_contract:
        message = "chemical result violates the validation-only locked contract"
        raise ValueError(message)
    rows = summary.get("topic_scores", [])
    useful = (
        sum(
            bool(row["eligible"])
            and row["sos"] is not None
            and float(row["sos"]) >= USEFUL_SOS_THRESHOLD
            for row in rows
        )
        if rows
        else int(summary["sos_bands"]["high_gt_0_8"])
        + int(summary["sos_bands"]["intermediate_0_6_to_0_8"])
    )
    return {
        "optimized_motifs": round(
            float(result["annotation_coverage"]) * int(result["topics"]),
        ),
        "evaluable_motifs": int(summary["eligible_topics"]),
        "useful_motifs": int(useful),
        "mean_sos": summary["mean_sos"],
        "median_sos": summary["median_sos"],
        "sos_high_gt_0_8": int(summary["sos_bands"]["high_gt_0_8"]),
        "sos_intermediate_0_6_to_0_8": int(
            summary["sos_bands"]["intermediate_0_6_to_0_8"],
        ),
        "sos_low_lt_0_6": int(summary["sos_bands"]["low_lt_0_6"]),
        "associated_spectra": int(summary["associated_spectra"]),
        "associated_molecules": summary.get("associated_molecules"),
        "membership_threshold": float(summary["membership_threshold"]),
        "leakage_audit_passed": bool(result["heldout_compounds_excluded_from_mag"]),
    }


def _completion(result: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "m1":
        return result["metrics"]["validation_document_completion"]
    return result["metrics"]["document_completion"]


def _runtime(result: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "m1":
        return {"training_wall_seconds": M1_TRAINING_WALL_SECONDS, "memory": {}}
    return result["metrics"].get("runtime", {})


def comparison_row(  # noqa: PLR0913
    *,
    label: str,
    method: str,
    model_result: dict[str, Any],
    chemical_result: dict[str, Any],
    theta: np.ndarray,
    beta: np.ndarray,
    vocabulary: list[str],
) -> dict[str, Any]:
    """Build one baseline/candidate row under the expanded diagnostic contract."""
    chemistry = chemistry_headline(chemical_result)
    completion = _completion(model_result, method)
    diagnostics = model_selection_diagnostics(
        theta,
        beta,
        vocabulary,
        MODEL_SELECTION_EVALUATION_PROTOCOL,
    )
    inventory = diagnostics["topic_inventory"]
    distribution = theta_distribution(theta)
    runtime = _runtime(model_result, method)
    memory = runtime.get("memory", {})
    config = model_result.get("config", {})
    duplicates = {
        float(item["threshold"]): item for item in inventory["duplicate_components"]
    }
    row = {
        "model": label,
        "gate_temperature": config.get("gate_temperature"),
        "gate_gamma": config.get("gate_gamma"),
        "trained_separately": config.get("trained_separately"),
        **chemistry,
        "completion_nll": float(completion["nll_per_token"]),
        "completion_oov_fraction": float(completion["oov_fraction"]),
        "median_effective_topics_per_spectrum": inventory[
            "median_effective_topics_per_spectrum"
        ],
        "mean_effective_topics_per_spectrum": inventory[
            "mean_effective_topics_per_spectrum"
        ],
        "median_max_theta": distribution["median_max_theta"],
        "fraction_max_theta_ge_0_5": distribution["fraction_max_theta_ge_0_5"],
        "fraction_max_theta_ge_0_3": distribution["fraction_max_theta_ge_0_3"],
        "fraction_max_theta_ge_0_2": distribution["fraction_max_theta_ge_0_2"],
        "unique_top1_topics": inventory["unique_top1_topics"],
        "topics_never_top1": inventory["topics_never_top1"],
        "corpus_effective_topic_count": inventory["corpus_effective_topic_count"],
        "active_topics_gt_0_0005": inventory["active_topics_above_usage_threshold"],
        "active_topics_ge_1_over_k": inventory["active_topics_mean_usage_ge_1_over_k"],
        "maximum_mean_topic_usage": inventory["maximum_mean_topic_usage"],
        "mean_nearest_beta_cosine": inventory["mean_nearest_topic_beta_cosine"],
        "median_nearest_beta_cosine": inventory["median_nearest_topic_beta_cosine"],
        "maximum_beta_cosine": inventory["maximum_pairwise_beta_cosine"],
        "duplicate_components_0_95": duplicates[0.95]["duplicate_component_count"],
        "duplicate_topics_0_95": duplicates[0.95]["topics_in_duplicate_components"],
        "largest_duplicate_component_0_95": duplicates[0.95]["largest_component_size"],
        "duplicate_components_0_99": duplicates[0.99]["duplicate_component_count"],
        "duplicate_topics_0_99": duplicates[0.99]["topics_in_duplicate_components"],
        "largest_duplicate_component_0_99": duplicates[0.99]["largest_component_size"],
        "duplicate_components_0_999": duplicates[0.999]["duplicate_component_count"],
        "duplicate_topics_0_999": duplicates[0.999]["topics_in_duplicate_components"],
        "largest_duplicate_component_0_999": inventory[
            "largest_strict_duplicate_component"
        ],
        "catastrophic_duplicate_component": inventory[
            "catastrophic_duplicate_component"
        ],
        "median_beta_effective_words": inventory["median_beta_effective_words"],
        "median_beta_max_probability": inventory["median_beta_max_probability"],
        "median_beta_top20_mass": inventory["median_beta_top_n_mass"],
        "top_word_uniqueness": inventory["top_word_uniqueness"],
        "fragment_mass_median": diagnostics["fragment_probability_mass"]["median"],
        "fragment_mass_fraction_extreme": diagnostics["fragment_probability_mass"][
            "fraction_extreme_skew"
        ],
        "training_wall_seconds": runtime.get("training_wall_seconds"),
        "validation_full_spectra_per_second": runtime.get(
            "validation_full_spectra_per_second",
        ),
        "peak_process_bytes": memory.get("peak_process_bytes"),
        "parameters": (
            M1_PARAMETERS if method == "m1" else model_result.get("parameters")
        ),
        "finite_stable": bool(model_result["metrics"].get("finite_stable", True)),
    }
    row.update(
        {
            "gate_optimized": row["optimized_motifs"] >= GATES["optimized_motifs"],
            "gate_evaluable": row["evaluable_motifs"] >= GATES["evaluable_motifs"],
            "gate_useful": row["useful_motifs"] >= GATES["useful_motifs"],
            "gate_mean_sos": (
                row["mean_sos"] is not None and row["mean_sos"] >= GATES["mean_sos"]
            ),
            "gate_completion_nll": (
                row["completion_nll"] <= GATES["maximum_completion_nll"]
            ),
            "gate_finite_stable": row["finite_stable"],
            "gate_no_catastrophic_duplicate_component": not row[
                "catastrophic_duplicate_component"
            ],
        },
    )
    row["passed_all_frozen_gates"] = all(
        row[key]
        for key in (
            "gate_optimized",
            "gate_evaluable",
            "gate_useful",
            "gate_mean_sos",
            "gate_completion_nll",
            "gate_finite_stable",
            "gate_no_catastrophic_duplicate_component",
        )
    )
    return row


def _copy_model(run: Path, output: Path, method: str) -> None:
    source = run / "models" / method
    destination = output / method
    destination.mkdir(parents=True, exist_ok=True)
    for name in REVIEWABLE_MODEL_FILES:
        path = source / name
        if not path.is_file():
            message = f"required reviewable model artifact is missing: {path}"
            raise FileNotFoundError(message)
        shutil.copy2(path, destination / name)


def _assert_validation_only_candidate(run: Path, method: str) -> None:
    audit = read_json(run / "models" / method / "validation_access_audit.json")
    opened_names = [Path(item["path"]).name for item in audit["loaded_artifacts"]]
    valid = (
        not audit["candidate_test_artifacts_loaded"]
        and not audit["candidate_test_chemistry_loaded"]
        and not audit["candidate_test_mag_or_sos_computed"]
        and not audit["candidate_test_metrics_inspected"]
        and all("test" not in name.lower() for name in opened_names)
    )
    if not valid:
        message = f"candidate access audit is not validation-only: {method}"
        raise RuntimeError(message)


def _candidate_consistency(
    row: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    completion = metrics["document_completion"]
    chemistry = metrics["validation_chemistry"]
    theta = metrics["theta_distribution"]
    inventory = metrics["topic_inventory"]
    fragment = metrics["fragment_probability_mass"]
    expected = {
        "completion_nll": completion["nll_per_token"],
        "completion_oov_fraction": completion["oov_fraction"],
        "optimized_motifs": chemistry["optimized_motifs"],
        "evaluable_motifs": chemistry["eligible_topics"],
        "useful_motifs": chemistry["useful_motifs"],
        "mean_sos": chemistry["mean_sos"],
        "median_sos": chemistry["median_sos"],
        "associated_spectra": chemistry["associated_spectra"],
        "associated_molecules": chemistry["associated_molecules"],
        "median_effective_topics_per_spectrum": theta[
            "median_effective_topics_per_spectrum"
        ],
        "mean_effective_topics_per_spectrum": theta[
            "mean_effective_topics_per_spectrum"
        ],
        "median_max_theta": theta["median_max_theta"],
        "fraction_max_theta_ge_0_5": theta["fraction_max_theta_ge_0_5"],
        "fraction_max_theta_ge_0_3": theta["fraction_max_theta_ge_0_3"],
        "fraction_max_theta_ge_0_2": theta["fraction_max_theta_ge_0_2"],
        "unique_top1_topics": inventory["unique_top1_topics"],
        "topics_never_top1": inventory["topics_never_top1"],
        "corpus_effective_topic_count": inventory["corpus_effective_topic_count"],
        "maximum_beta_cosine": inventory["maximum_pairwise_beta_cosine"],
        "fragment_mass_median": fragment["median"],
    }
    checks = {}
    for field, metric_value in expected.items():
        comparison_value = row[field]
        if isinstance(metric_value, float):
            matches = bool(
                np.isclose(
                    comparison_value,
                    metric_value,
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                ),
            )
        else:
            matches = comparison_value == metric_value
        checks[field] = {
            "comparison_csv_value": comparison_value,
            "metrics_json_value": metric_value,
            "matches": matches,
        }
    return {
        "all_fields_match": all(item["matches"] for item in checks.values()),
        "checks": checks,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["/usr/bin/git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sysctl(name: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["/usr/sbin/sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _conda_export(environment: str) -> str:
    conda = shutil.which("conda")
    if conda is None:
        return "conda executable unavailable"
    completed = subprocess.run(  # noqa: S603
        [conda, "list", "--export", "-n", environment],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr


def _environment() -> str:
    lines = [
        f"generated_utc={datetime.now(UTC).isoformat()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
        f"cpu_brand={_sysctl('machdep.cpu.brand_string')}",
        f"physical_cores={_sysctl('hw.physicalcpu')}",
        f"logical_cores={_sysctl('hw.logicalcpu')}",
        f"memory_bytes={_sysctl('hw.memsize')}",
        f"conda_environment={os.environ.get('CONDA_DEFAULT_ENV', 'unavailable')}",
        f"python_executable={sys.executable}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"mps_available={torch.backends.mps.is_available()}",
        f"torch_threads={torch.get_num_threads()}",
        "",
        "# conda list --export: ms2lda-neural",
        _conda_export("ms2lda-neural"),
    ]
    return "\n".join(lines) + "\n"


def package(  # noqa: PLR0913
    *,
    repo: Path,
    run: Path,
    m1_run: Path,
    data_root: Path,
    output: Path,
    source_commit: str,
    methods: Sequence[str],
) -> dict[str, Any]:
    """Copy compact evidence and recompute one internally consistent table."""
    output.mkdir(parents=True, exist_ok=True)
    vocabulary = load_vocabulary(run / "data")
    rows = []
    baselines = (
        (
            "M1",
            "m1",
            m1_run / "validation_evaluation/neural/complete.json",
            m1_run / "validation_chemical/neural/complete.json",
            m1_run / "validation_evaluation/neural/validation_full_theta.npy",
            m1_run / "validation_evaluation/neural/beta.npy",
        ),
        (
            "canonical ETM",
            "etm",
            run / "models/etm/result.json",
            run / "validation_chemical/etm/complete.json",
            run / "validation_evaluation/etm/validation_full_theta.npy",
            run / "validation_evaluation/etm/beta.npy",
        ),
        (
            "balanced ETM",
            "etm_balanced",
            run / "models/etm_balanced/result.json",
            run / "validation_chemical/etm_balanced/complete.json",
            run / "validation_evaluation/etm_balanced/validation_full_theta.npy",
            run / "validation_evaluation/etm_balanced/beta.npy",
        ),
    )
    for label, method, result_path, chemistry_path, theta_path, beta_path in baselines:
        rows.append(
            comparison_row(
                label=label,
                method=method,
                model_result=read_json(result_path),
                chemical_result=read_json(chemistry_path),
                theta=np.load(theta_path, mmap_mode="r"),
                beta=np.load(beta_path, mmap_mode="r"),
                vocabulary=vocabulary,
            ),
        )
    for method in methods:
        _assert_validation_only_candidate(run, method)
        _copy_model(run, output, method)
        rows.append(
            comparison_row(
                label=method,
                method=method,
                model_result=read_json(run / "models" / method / "result.json"),
                chemical_result=read_json(
                    run / "validation_chemical" / method / "complete.json",
                ),
                theta=np.load(
                    run
                    / "validation_evaluation"
                    / method
                    / "validation_full_theta.npy",
                    mmap_mode="r",
                ),
                beta=np.load(
                    run / "validation_evaluation" / method / "beta.npy",
                    mmap_mode="r",
                ),
                vocabulary=vocabulary,
            ),
        )
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "gate_strength_study.csv", rows[3:])
    write_csv(
        output / "runtime_memory.csv",
        [
            {
                "model": row["model"],
                "training_wall_seconds": row["training_wall_seconds"],
                "validation_full_spectra_per_second": row[
                    "validation_full_spectra_per_second"
                ],
                "peak_process_bytes": row["peak_process_bytes"],
                "parameters": row["parameters"],
                "finite_stable": row["finite_stable"],
            }
            for row in rows
        ],
    )
    consistency = {}
    for method in methods:
        row = next(item for item in rows if item["model"] == method)
        consistency[method] = _candidate_consistency(
            row,
            read_json(run / "models" / method / "metrics.json"),
        )
        if not consistency[method]["all_fields_match"]:
            message = f"comparison/metrics inconsistency for {method}"
            raise RuntimeError(message)
    write_json(output / "consistency_audit.json", consistency)
    write_json(output / "frozen_gates.json", GATES)
    (output / "environment.txt").write_text(_environment(), encoding="utf-8")

    important_paths = [
        run / "data/train.npz",
        run / "data/validation_records.jsonl",
        run / "data/validation_observed.npz",
        run / "data/validation_completion.npz",
        run / "data/validation_full.npz",
        run / "data/vocabulary.json",
        run / "token_features/features.npy",
        run / "mag/index/spec2vec_filtered.faiss",
        run / "models/etm/weights.pt",
        run / "validation_evaluation/etm/beta.npy",
        run / "validation_evaluation/etm/validation_full_theta.npy",
        run / "models/etm_balanced/weights.pt",
        run / "validation_evaluation/etm_balanced/beta.npy",
        run / "validation_evaluation/etm_balanced/validation_full_theta.npy",
        m1_run / "trained_model/weights.pt",
        m1_run / "validation_evaluation/neural/beta.npy",
        m1_run / "validation_evaluation/neural/validation_full_theta.npy",
    ]
    for method in methods:
        important_paths.extend(
            [
                run / "models" / method / "weights.pt",
                run / "validation_evaluation" / method / "beta.npy",
                run / "validation_evaluation" / method / "validation_full_theta.npy",
            ],
        )
    command_path = output / "exact_commands.txt"
    commands = (
        command_path.read_text(encoding="utf-8").splitlines()
        if command_path.is_file()
        else []
    )
    provenance = {
        "evidence_boundary": "validation only; candidate test artifacts not accessed",
        "base_commit": BASE_COMMIT,
        "experiment_source_commit": source_commit,
        "final_result_commit": (
            "SELF: resolve the commit containing this provenance.json; exact SHA is "
            "reported in the final handoff"
        ),
        "packaged_from_commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "data_root": str(data_root),
        "run_root": str(run),
        "m1_reference_run_root": str(m1_run),
        "seeds": {"data_split": 42, "etm_optimization": 7043},
        "methods": list(methods),
        "all_candidate_methods_trained": True,
        "post_hoc_candidate_methods": [],
        "exact_commands": commands,
        "important_local_artifacts": [artifact_entry(path) for path in important_paths],
        "balanced_etm_unchanged_audit": {
            "expected_sha256": BALANCED_LOCKED_HASHES,
            "actual_sha256": {
                "weights.pt": sha256(run / "models/etm_balanced/weights.pt"),
                "beta.npy": sha256(run / "validation_evaluation/etm_balanced/beta.npy"),
                "validation_full_theta.npy": sha256(
                    run
                    / "validation_evaluation/etm_balanced/validation_full_theta.npy",
                ),
            },
        },
        "candidate_test_data_loaded": False,
        "candidate_test_completion_computed": False,
        "candidate_test_mag_sos_computed": False,
        "candidate_test_metrics_inspected": False,
    }
    provenance["balanced_etm_unchanged_audit"]["all_hashes_match"] = (
        provenance["balanced_etm_unchanged_audit"]["actual_sha256"]
        == BALANCED_LOCKED_HASHES
    )
    if not provenance["balanced_etm_unchanged_audit"]["all_hashes_match"]:
        message = "balanced ETM local artifacts do not match the locked hashes"
        raise RuntimeError(message)
    write_json(output / "provenance.json", provenance)
    return {"rows": rows, "provenance": provenance}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--m1-run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args(argv)
    result = package(
        repo=args.repo.expanduser().resolve(),
        run=args.run.expanduser().resolve(),
        m1_run=args.m1_run.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        source_commit=args.source_commit,
        methods=args.methods,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
