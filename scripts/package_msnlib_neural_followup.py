"""Package the validation-only MSnLib neural follow-up for independent review."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.neural_ms2lda.utils import read_json, write_json
from scripts.package_msnlib_model_comparison import (
    GATES,
    M1,
    candidate_row,
    chemistry_summary,
    environment_text,
    file_record,
    sha256,
    write_csv,
)
from scripts.run_msnlib_model_comparison import entropy_diagnostics
from scripts.run_msnlib_neural_followup import _largest_component_members

HANDOFF_SHA = "9baec8aa62f684480eba35d4fc7f626c46f7b804"
PREVIOUS_RESULT_SHA = "ecb09251de94093e345584ed53f87ed799e88dc4"
MAIN_SHA = "20de0e45aec25203e6bc38770a795b25cc18bff7"
BALANCED_ETM_LAUNCH_SHA = "6f0f2dff45b5bcd29347c76969d76a14ccbb2581"
BALANCED_ETM_RUNNER_BLOB = "b79e1d2d0008eab8091964ff220e6b9c2c799e79"
ECRTM_LAUNCH_SHA = "d3ff2e85f0e4ef08e653c9ac917afa055ed1edc4"
ECRTM_RUNNER_BLOB = "ea5ed49e42ba9e902ed154aad83b4357c6c792fe"
ECRTM_QUEUE_LOG = Path(tempfile.gettempdir()) / "ms2lda_ecrtm_overnight_20260828.log"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git executable is unavailable")
    result = subprocess.run(  # noqa: S603
        [executable, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _collapse_evidence(beta_path: Path, theta_path: Path) -> dict[str, Any]:
    beta = np.load(beta_path, mmap_mode="r")
    normalized = np.array(beta, dtype=np.float32, copy=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -1.0)
    largest = _largest_component_members(similarity, 0.999)
    theta = np.load(theta_path, mmap_mode="r")
    return {
        "largest_beta_component_cosine_ge_0_999": int(len(largest)),
        "topics_with_nearest_beta_cosine_ge_0_999": int(
            np.sum(np.max(similarity, axis=1) >= 0.999)
        ),
        "unique_top1_topics": int(len(np.unique(np.argmax(theta, axis=1)))),
    }


def _apply_gates(row: dict[str, Any]) -> dict[str, Any]:
    catastrophic = bool(
        row.get("largest_beta_component_cosine_ge_0_999", 0) >= 100
        or row.get("active_topics_usage_gt_0_0005", 1000) < 100
        or row.get("corpus_effective_topic_count", 1000) < 100
        or row.get("maximum_mean_topic_usage", 0.0) > 0.1
    )
    row.update(
        {
            "catastrophic_inventory_collapse": catastrophic,
            "gate_optimized": row["optimized_motifs"] >= GATES["optimized_motifs"],
            "gate_evaluable": row["evaluable_motifs"] >= GATES["evaluable_motifs"],
            "gate_useful": row["useful_motifs"] >= GATES["useful_motifs"],
            "gate_mean_sos": row["mean_sos"] >= GATES["mean_sos"],
            "gate_completion_nll": row["completion_nll"] <= GATES["completion_nll"],
            "gate_finite_stable": bool(row["finite_stable"]),
            "gate_no_catastrophic_inventory_collapse": not catastrophic,
        }
    )
    row["passed_all_frozen_gates"] = all(
        row[name]
        for name in (
            "gate_optimized",
            "gate_evaluable",
            "gate_useful",
            "gate_mean_sos",
            "gate_completion_nll",
            "gate_finite_stable",
            "gate_no_catastrophic_inventory_collapse",
        )
    )
    return row


def _standard_row(run: Path, method: str, *, kind: str) -> dict[str, Any]:
    row = candidate_row(run, method)
    row["model_kind"] = kind
    row.update(
        _collapse_evidence(
            run / "validation_evaluation" / method / "beta.npy",
            run / "validation_evaluation" / method / "validation_full_theta.npy",
        )
    )
    return _apply_gates(row)


def _pooled_calibrated_row(run: Path, output: Path) -> dict[str, Any]:
    payload = read_json(output / "pooled_temperature_selected_metrics.json")
    source = read_json(run / "models/pooled_likelihood/result.json")
    chemistry = payload["chemistry"]
    inventory = payload["topic_inventory"]
    entropy = entropy_diagnostics(
        np.load(
            run
            / "followup_validation/pooled_likelihood_tau_0p110"
            / "validation_full_theta.npy",
            mmap_mode="r",
        )
    )
    row = {
        "method": "pooled_likelihood_tau011",
        "model_kind": "post-hoc calibrated model",
        **{
            key: chemistry[key]
            for key in (
                "optimized_motifs",
                "evaluable_motifs",
                "useful_motifs",
                "mean_sos",
                "median_sos",
                "sos_high_gt_0_8",
                "sos_intermediate_0_6_to_0_8",
                "sos_low_lt_0_6",
                "associated_spectra",
                "associated_molecules",
            )
        },
        "leakage_audit_passed": True,
        "completion_nll": payload["completion"]["nll_per_token"],
        "completion_oov_fraction": payload["completion"]["oov_fraction"],
        "median_effective_topics_per_spectrum": inventory[
            "median_effective_topics_per_spectrum"
        ],
        "corpus_effective_topic_count": inventory["corpus_effective_topic_count"],
        "active_topics_usage_gt_0_0005": inventory[
            "active_topics_mean_usage_gt_0_0005"
        ],
        "active_topics_usage_ge_1_over_k": inventory[
            "active_topics_mean_usage_ge_1_over_k"
        ],
        "maximum_mean_topic_usage": inventory["maximum_mean_topic_usage"],
        "mean_nearest_topic_beta_cosine": inventory["mean_nearest_topic_beta_cosine"],
        "maximum_pairwise_beta_cosine": inventory["maximum_pairwise_beta_cosine"],
        "top_word_uniqueness": payload["topic_words"]["top_word_uniqueness"],
        "mean_conditional_theta_entropy": entropy["mean_conditional_theta_entropy"],
        "marginal_theta_entropy": entropy["marginal_theta_entropy"],
        "theta_mutual_information": entropy["mutual_information"],
        "training_wall_seconds": source["metrics"]["runtime"]["training_wall_seconds"],
        "validation_full_spectra_per_second": source["metrics"]["runtime"][
            "validation_full_spectra_per_second"
        ],
        "peak_process_bytes": source["metrics"]["runtime"]["memory"][
            "peak_process_bytes"
        ],
        "parameters": source["parameters"],
        "finite_stable": True,
        "post_hoc_theta_temperature": payload["post_hoc_inference_temperature"],
    }
    row.update(
        _collapse_evidence(
            run / "validation_evaluation/pooled_likelihood/beta.npy",
            run
            / "followup_validation/pooled_likelihood_tau_0p110"
            / "validation_full_theta.npy",
        )
    )
    return _apply_gates(row)


def _ecrtm_calibrated_row(run: Path) -> dict[str, Any]:
    method = "ecrtm_canonical_tau030"
    model = read_json(run / "models" / method / "result.json")
    source = read_json(run / "models/ecrtm_canonical/result.json")
    chemistry = chemistry_summary(
        read_json(run / "validation_chemical" / method / "complete.json")
    )
    metrics = model["metrics"]
    inventory = metrics["topic_inventory"]
    row = {
        "method": method,
        "model_kind": "post-hoc calibrated model",
        **chemistry,
        "completion_nll": metrics["document_completion"]["nll_per_token"],
        "completion_oov_fraction": metrics["document_completion"]["oov_fraction"],
        "median_effective_topics_per_spectrum": inventory[
            "median_effective_topics_per_spectrum"
        ],
        "corpus_effective_topic_count": inventory["corpus_effective_topic_count"],
        "active_topics_usage_gt_0_0005": inventory[
            "active_topics_mean_usage_gt_0_0005"
        ],
        "active_topics_usage_ge_1_over_k": inventory[
            "active_topics_mean_usage_ge_1_over_k"
        ],
        "maximum_mean_topic_usage": inventory["maximum_mean_topic_usage"],
        "mean_nearest_topic_beta_cosine": inventory["mean_nearest_topic_beta_cosine"],
        "maximum_pairwise_beta_cosine": inventory["maximum_pairwise_beta_cosine"],
        "top_word_uniqueness": metrics["top_word_uniqueness"],
        "training_wall_seconds": source["metrics"]["runtime"]["training_wall_seconds"],
        "validation_full_spectra_per_second": source["metrics"]["runtime"][
            "validation_full_spectra_per_second"
        ],
        "peak_process_bytes": source["metrics"]["runtime"]["memory"][
            "peak_process_bytes"
        ],
        "parameters": source["parameters"],
        "finite_stable": metrics["finite_stable"],
        "post_hoc_theta_temperature": 0.30,
    }
    row.update(
        _collapse_evidence(
            run / "validation_evaluation" / method / "beta.npy",
            run / "validation_evaluation" / method / "validation_full_theta.npy",
        )
    )
    return _apply_gates(row)


def _ecrtm_failure_payload(run: Path) -> dict[str, Any]:
    """Summarize the fail-closed canonical run without evaluating a partial model."""
    history_path = run / "models/ecrtm_canonical/training_history.csv"
    with history_path.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError("canonical ECRTM failure history is empty")
    last = history[-1]
    return {
        "status": "failed_sinkhorn_nonconvergence",
        "epochs_requested": 40,
        "epochs_completed": len(history),
        "failed_epoch": len(history) + 1,
        "resume_attempts": 1,
        "failure": (
            "canonical Sinkhorn residual did not reach 0.005 within 1000 "
            "iterations; the deterministic checkpoint resume failed again"
        ),
        "completed_epoch_seconds": float(sum(float(row["seconds"]) for row in history)),
        "last_completed_epoch": {
            "epoch": int(last["epoch"]),
            "topic_model_loss": float(last["topic_model_loss"]),
            "ecr_loss": float(last["ecr_loss"]),
            "mean_sinkhorn_iterations": float(last["mean_sinkhorn_iterations"]),
            "maximum_sinkhorn_iterations": int(last["maximum_sinkhorn_iterations"]),
            "mean_final_checked_residual": float(last["mean_final_checked_residual"]),
            "maximum_final_checked_residual": float(
                last["maximum_final_checked_residual"]
            ),
            "seconds": float(last["seconds"]),
        },
        "partial_model_scored": False,
        "numerical_approximation_used": False,
    }


def _ecrtm_failure_row(run: Path) -> dict[str, Any]:
    failure = _ecrtm_failure_payload(run)
    return {
        "method": "ecrtm_canonical",
        "model_kind": "incomplete published-model feasibility diagnostic",
        "execution_status": failure["status"],
        "optimized_motifs": None,
        "evaluable_motifs": None,
        "useful_motifs": None,
        "mean_sos": None,
        "median_sos": None,
        "completion_nll": None,
        "finite_stable": False,
        "training_wall_seconds": failure["completed_epoch_seconds"],
        "epochs_completed": failure["epochs_completed"],
        "failed_epoch": failure["failed_epoch"],
        "final_mean_sinkhorn_iterations": failure["last_completed_epoch"][
            "mean_sinkhorn_iterations"
        ],
        "final_maximum_sinkhorn_iterations": failure["last_completed_epoch"][
            "maximum_sinkhorn_iterations"
        ],
        "gate_optimized": False,
        "gate_evaluable": False,
        "gate_useful": False,
        "gate_mean_sos": False,
        "gate_completion_nll": False,
        "gate_finite_stable": False,
        "gate_no_catastrophic_inventory_collapse": False,
        "passed_all_frozen_gates": False,
    }


def _m1_row(fieldnames: list[str]) -> dict[str, Any]:
    row = dict.fromkeys(fieldnames)
    row.update(M1)
    row.update(
        {
            "model_kind": "locked incumbent reference",
            "leakage_audit_passed": True,
            "finite_stable": True,
            "catastrophic_inventory_collapse": False,
            "gate_optimized": True,
            "gate_evaluable": True,
            "gate_useful": True,
            "gate_mean_sos": True,
            "gate_completion_nll": True,
            "gate_finite_stable": True,
            "gate_no_catastrophic_inventory_collapse": True,
            "passed_all_frozen_gates": True,
        }
    )
    return row


def _copy_small_evidence(run: Path, output: Path, method: str) -> None:
    destination = output / method
    destination.mkdir(parents=True, exist_ok=True)
    source = run / "models" / method
    for name in (
        "config.json",
        "result.json",
        "training_history.csv",
        "chemical_scores.csv",
        "top_words.csv",
        "fragment_mass_summary.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    chemistry = run / "validation_chemical" / method / "complete.json"
    if chemistry.is_file():
        shutil.copy2(chemistry, destination / "chemical_validation.json")


def _exact_commands(
    run: Path, data_root: Path, output: Path, source_sha: str
) -> list[str]:
    module = "conda run -n ms2lda-neural python -m"
    commands = [
        "git status",
        "git fetch origin --prune",
        (
            "git switch -c experiment/msnlib-neural-followup-20260827 "
            + PREVIOUS_RESULT_SHA
        ),
        (
            f"{module} scripts.run_msnlib_neural_followup diagnose-existing "
            f"--run {run} --m1-run "
            "/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/"
            f"neural-minimality-seed42/m1-lock --output {output} --device cpu"
        ),
        (
            f"{module} scripts.run_msnlib_neural_followup select-temperature "
            f"--run {run} --method pooled_likelihood --temperature 0.11 "
            f"--output {output} --device cpu"
        ),
        (
            f"{module} scripts.run_msnlib_model_comparison train --run {run} "
            "--method etm_balanced --device cpu --etm-epochs 120 "
            "--etm-batch-size 256"
        ),
        (
            "conda run -n ms2lda-neural python -m "
            f"scripts.run_msnlib_model_comparison chemical --run {run} "
            f"--data-root {data_root} --method etm_balanced"
        ),
        (
            "conda run --no-capture-output -n ms2lda-neural python -m "
            f"scripts.run_msnlib_neural_followup sweep-etm-temperature --run {run} "
            f"--method etm_balanced --output {output} --device cpu"
        ),
        (
            "conda run --no-capture-output -n ms2lda-neural python -m "
            "scripts.run_msnlib_model_comparison "
            f"train-ecrtm-canonical --run {run} --device cpu --epochs 40 "
            "--batch-size 200 --max-iter 1000"
        ),
        ("caffeinate -dimsu /bin/zsh " f"{REPO_ROOT / 'ecrtm_overnight_queue.sh'}"),
    ]
    if (run / "validation_chemical/ecrtm_canonical/complete.json").is_file():
        commands.append(
            "conda run -n ms2lda-neural python -m "
            f"scripts.run_msnlib_model_comparison chemical --run {run} "
            f"--data-root {data_root} --method ecrtm_canonical"
        )
    if (run / "validation_chemical/ecrtm_canonical_tau030/complete.json").is_file():
        commands.append(
            "conda run -n ms2lda-neural python -m "
            f"scripts.run_msnlib_model_comparison chemical --run {run} "
            f"--data-root {data_root} --method ecrtm_canonical_tau030"
        )
    commands.extend(
        [
            (
                "conda run -n ms2lda-neural python -m pytest -q "
                "benchmarks/neural_ms2lda/tests"
            ),
            (
                "conda run -n ms2lda-neural black --check benchmarks/neural_ms2lda "
                "scripts/run_msnlib_model_comparison.py "
                "scripts/run_msnlib_neural_followup.py "
                "scripts/package_msnlib_neural_followup.py"
            ),
            (
                "conda run -n ms2lda-neural ruff check --config "
                "benchmarks/neural_ms2lda/ruff.toml benchmarks/neural_ms2lda "
                "scripts/run_msnlib_model_comparison.py "
                "scripts/run_msnlib_neural_followup.py "
                "scripts/package_msnlib_neural_followup.py"
            ),
            (
                f"{module} scripts.package_msnlib_neural_followup --run {run} "
                f"--data-root {data_root} --output {output} "
                f"--source-sha {source_sha}"
            ),
        ]
    )
    return commands


def package(
    run: Path,
    data_root: Path,
    output: Path,
    *,
    source_sha: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        _standard_row(run, "etm", kind="previous newly trained model"),
        _standard_row(run, "pooled_likelihood", kind="previous newly trained model"),
        _pooled_calibrated_row(run, output),
        _standard_row(run, "pooled_mi005", kind="previous diagnostic model"),
        _standard_row(run, "etm_balanced", kind="newly trained model"),
    ]
    if (run / "models/ecrtm_canonical/result.json").is_file():
        rows.extend(
            (
                _standard_row(
                    run, "ecrtm_canonical", kind="newly trained published model"
                ),
                _ecrtm_calibrated_row(run),
            )
        )
    elif (run / "models/ecrtm_canonical/checkpoint.pt").is_file():
        rows.append(_ecrtm_failure_row(run))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    comparison = [_m1_row(fieldnames)] + [
        {key: row.get(key) for key in fieldnames} for row in rows
    ]
    write_csv(output / "comparison.csv", comparison)
    write_csv(
        output / "runtime_memory.csv",
        [
            {
                "method": row["method"],
                "training_wall_seconds": row.get("training_wall_seconds"),
                "validation_full_spectra_per_second": row.get(
                    "validation_full_spectra_per_second"
                ),
                "peak_process_bytes": row.get("peak_process_bytes"),
                "parameters": row.get("parameters"),
                "finite_stable": row.get("finite_stable"),
            }
            for row in comparison
        ],
    )

    shutil.copy2(run / "protocol.json", output / "locked_protocol.json")
    preparation = run / "comparison_preparation.json"
    if preparation.is_file():
        shutil.copy2(preparation, output / "preparation_summary.json")
    feasibility_output = output / "ecrtm_feasibility"
    feasibility_output.mkdir(parents=True, exist_ok=True)
    for name in ("canonical.json", "bounded_50.json"):
        source = run / "ecrtm_feasibility" / name
        if source.is_file():
            shutil.copy2(source, feasibility_output / name)

    for method in ("etm_balanced", "ecrtm_canonical", "ecrtm_canonical_tau030"):
        if (run / "models" / method).is_dir():
            _copy_small_evidence(run, output, method)
    if (run / "models/ecrtm_canonical/checkpoint.pt").is_file():
        failure_output = output / "ecrtm_canonical"
        failure_output.mkdir(parents=True, exist_ok=True)
        write_json(failure_output / "failure_metrics.json", _ecrtm_failure_payload(run))
        if ECRTM_QUEUE_LOG.is_file():
            shutil.copy2(ECRTM_QUEUE_LOG, failure_output / "failure.log")
        queue_script = REPO_ROOT / "ecrtm_overnight_queue.sh"
        if queue_script.is_file():
            shutil.copy2(queue_script, failure_output / queue_script.name)

    important_paths = [
        run / "models/etm_balanced/weights.pt",
        run / "validation_evaluation/etm_balanced/beta.npy",
        run / "validation_evaluation/etm_balanced/validation_full_theta.npy",
        run
        / "followup_validation/pooled_likelihood_tau_0p110/validation_full_theta.npy",
        run
        / (
            "followup_validation/pooled_likelihood_tau_0p110/"
            "validation_observed_theta.npy"
        ),
    ]
    if (run / "models/ecrtm_canonical/checkpoint.pt").is_file():
        important_paths.append(run / "models/ecrtm_canonical/checkpoint.pt")
    if (run / "models/ecrtm_canonical/result.json").is_file():
        important_paths.extend(
            (
                run / "models/ecrtm_canonical/weights.pt",
                run / "models/ecrtm_canonical/validation_observed_theta.npy",
                run / "validation_evaluation/ecrtm_canonical/beta.npy",
                run / "validation_evaluation/ecrtm_canonical/validation_full_theta.npy",
                run / "models/ecrtm_canonical_tau030/validation_observed_theta.npy",
                run / "validation_evaluation/ecrtm_canonical_tau030/beta.npy",
                run
                / (
                    "validation_evaluation/ecrtm_canonical_tau030/"
                    "validation_full_theta.npy"
                ),
            )
        )
    important = [file_record(path) for path in important_paths if path.is_file()]
    commands = _exact_commands(run, data_root, output, source_sha)
    asset_manifest = read_json(data_root / "acquisition_manifest.json")
    provenance = {
        "evidence_boundary": (
            "validation only; candidate test arrays, theta, completion, MAG, SOS, "
            "and result artifacts were not opened, loaded, scored, or summarized"
        ),
        "branch": "experiment/msnlib-neural-followup-20260827",
        "origin_main_sha": MAIN_SHA,
        "research_handoff_sha": HANDOFF_SHA,
        "previous_result_sha": PREVIOUS_RESULT_SHA,
        "source_and_diagnostic_sha": source_sha,
        "packaging_head_sha": _git("rev-parse", "HEAD"),
        "model_runner_blob_sha": _git(
            "hash-object", "scripts/run_msnlib_model_comparison.py"
        ),
        "balanced_etm_execution": {
            "launch_head_sha": BALANCED_ETM_LAUNCH_SHA,
            "model_runner_blob_sha": BALANCED_ETM_RUNNER_BLOB,
            "later_runner_difference": (
                "canonical ECRTM support and execution hardening only; balanced ETM "
                "implementation unchanged"
            ),
        },
        "canonical_ecrtm_execution": {
            "launch_head_sha": ECRTM_LAUNCH_SHA,
            "model_runner_blob_sha": ECRTM_RUNNER_BLOB,
            "later_runner_difference": "report packaging only; model code unchanged",
            "failure": (
                _ecrtm_failure_payload(run)
                if not (run / "models/ecrtm_canonical/result.json").is_file()
                else None
            ),
        },
        "execution_run_directory": str(run),
        "data_root": str(data_root),
        "data_asset_manifest": asset_manifest,
        "data_asset_manifest_sha256": sha256(data_root / "acquisition_manifest.json"),
        "validation_records_sha256": sha256(run / "data/validation_records.jsonl"),
        "random_seeds": {
            "locked_split_sgns_pooled": 42,
            "etm_and_balanced_etm": 7043,
            "canonical_ecrtm": 8043,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "software": {
            "python": sys.version,
            "environment_file": "environment.txt",
        },
        "validation_boundary_notes": {
            "shared_mag_index_reused": bool(
                (run / "mag/index/complete.json").is_file()
            ),
            "candidate_test_results_accessed": False,
            "candidate_test_matrices_accessed": False,
            "mag_index_semantics": (
                "the existing leakage-filtered index excludes compound identifiers "
                "from both held-out splits; follow-up candidate selection and scoring "
                "use validation theta and validation records only"
            ),
        },
        "execution_hardening": {
            "canonical_ecrtm_nonfinite_residuals_fail_closed": True,
            "canonical_ecrtm_checkpoint_contract_binds": [
                "train matrix SHA-256",
                "token features SHA-256",
                "protocol SHA-256",
                "optimizer and learning rate",
                "ECR and Sinkhorn settings",
            ],
            "synthetic_checkpoint_resume_max_abs_parameter_difference": 0.0,
        },
        "exact_commands": commands,
        "important_large_uncommitted_artifacts": important,
        "operational_failures": [
            {
                "stage": "first follow-up diagnostic invocation",
                "effect": "stopped before sweep output",
                "error": "KeyError: nll_per_in_vocab_token",
                "resolution": "use locked completion key nll_per_token; add test",
            },
            {
                "stage": "second follow-up diagnostic invocation",
                "effect": "stopped after pooled sweep before redundancy output",
                "error": "read-only memory-mapped beta normalized in place",
                "resolution": "make an explicit working copy; add deterministic rerun",
            },
            {
                "stage": "canonical ECRTM full training",
                "effect": (
                    "21/40 epochs completed; no partial model was inferred, chemically "
                    "scored, or considered for selection"
                ),
                "error": (
                    "Sinkhorn residual did not reach 0.005 within 1000 iterations "
                    "during epoch 22; one deterministic checkpoint resume failed again"
                ),
                "resolution": (
                    "stop the published comparator as operationally infeasible; do not "
                    "use the known-unconverged 50-step numerical approximation"
                ),
            },
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "provenance.json", provenance)
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")
    (output / "exact_commands.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    return {
        "comparison_rows": len(comparison),
        "important_large_artifacts": len(important),
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)
    result = package(
        args.run.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        source_sha=args.source_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
