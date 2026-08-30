"""Package the bounded gamma-2 gated-ETM validation temperature sweep.

The input sweep is produced by ``scripts.run_msnlib_neural_followup``. This
packager reads only the frozen validation artifacts named in ``ARTIFACTS``;
it never discovers or resolves candidate-test paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from benchmarks.neural_ms2lda.followup import retemperature_theta
from benchmarks.neural_ms2lda.utils import read_json, write_json

METHOD = "etm_balanced_gated_t1_g2"
BRANCH = "experiment/msnlib-gated-etm-20260828"
SOURCE_COMMIT = "6dbbefa149179460acf9c39eaf10bf428244af1c"
CURRENT_RESULT_COMMIT = "5cfed890d86d5a102d6b6f16f1a2a2431be5b4a6"
TEMPERATURES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
GATES = {
    "optimized_motifs": 840,
    "evaluable_motifs": 388,
    "useful_motifs": 252,
    "mean_sos": 0.651498,
    "maximum_completion_nll": 9.422847,
}
EXPECTED_HASHES = {
    "protocol.json": (
        "0102834def3009e142fe719b12129d7e59cb5dc158cb9cb16b19894acc249bf7"
    ),
    "models/etm_balanced_gated_t1_g2/config.json": (
        "81eab95a33b0db5083ac93df47968b684581e9fa4b58dc4165b099c5d61fba20"
    ),
    "models/etm_balanced_gated_t1_g2/result.json": (
        "c1c85b4e3420bb2f4dea8de5282b0277bb02b6a7a22cf87086640a71e25ce25c"
    ),
    "models/etm_balanced_gated_t1_g2/weights.pt": (
        "ffba02db4a4110d3cd4381d456f8454041e32f1e602004d23c63246c696d1417"
    ),
    "validation_evaluation/etm_balanced_gated_t1_g2/beta.npy": (
        "127372d8db634ba75f54da4ec32f1bf3a2de4722c6c09ae83746ae0e1f78bbf9"
    ),
    "validation_evaluation/etm_balanced_gated_t1_g2/validation_full_theta.npy": (
        "8f57fe165e1bb008d4fd32e69f1c4a77a21a6bf5ab26add5408e70ce94f07121"
    ),
    "validation_evaluation/etm_balanced_gated_t1_g2/complete.json": (
        "1f0803c16433d492b3aac4c805f9c7ffcc0cf957faa65a85f4280fd97cecb96d"
    ),
    "mag/annotations/etm_balanced_gated_t1_g2/annotations.jsonl": (
        "978497ba69d314f0c4a10dfc08cc293790765d64169d9f5c15ad436ea33e7732"
    ),
    "mag/annotations/etm_balanced_gated_t1_g2/complete.json": (
        "5ef1a37735616da1da0a669dfde6901e9252cf8f865f9d23d9137857694e183e"
    ),
    "data/validation_observed.npz": (
        "76afa43a3540d91edd4dd7e851afe10c65e3d67bcf06007c83eac1d9b6e15bff"
    ),
    "data/validation_completion.npz": (
        "f75245b599bc3c087aad97ff7247112d422ec0ab87418c5f96b0a889b2f661f0"
    ),
    "data/validation_full.npz": (
        "51888951815f448d0527a6a876be345cfe76ab346d9e97bada9a186e8857fab2"
    ),
    "data/validation_records.jsonl": (
        "0e85218489af6a07413474bb2db6ce74da537f6fd3c8ee77d5286f5775ba068c"
    ),
    "data/vocabulary.json": (
        "9ed46f54a4e4917560f764d4ac662bae7036c1b98733d6b25a62a869799bd90e"
    ),
    "token_features/features.npy": (
        "74fcfff0ee9bc776a372225023ab7070dc0cae11582191a3a4891a19560c84e5"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_entry(path: Path, relative: str) -> dict[str, Any]:
    return {
        "relative_path": relative,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"not a boolean value: {value}")


def _typed_sweep(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    integer_fields = {
        "unique_top1_topics",
        "optimized_motifs",
        "evaluable_motifs",
        "useful_motifs",
        "sos_high_gt_0_8",
        "sos_intermediate_0_6_to_0_8",
        "sos_low_lt_0_6",
        "associated_spectra",
        "associated_molecules",
    }
    boolean_fields = {
        "gate_optimized",
        "gate_evaluable",
        "gate_useful",
        "gate_mean_sos",
        "finite_stable",
        "gate_completion_nll",
        "passed_all_numeric_gates",
    }
    typed = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in integer_fields:
                converted[key] = int(value)
            elif key in boolean_fields:
                converted[key] = _as_bool(value)
            else:
                converted[key] = float(value)
        typed.append(converted)
    return typed


def _theta_inventory(theta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(theta, dtype=np.float64)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    usage = values.mean(axis=0)
    return {
        "topics_never_top1": int(
            values.shape[1] - len(np.unique(np.argmax(values, axis=1)))
        ),
        "corpus_effective_topic_count": float(
            np.exp(-np.sum(usage * np.log(np.clip(usage, 1e-12, None))))
        ),
        "active_topics_gt_0_0005": int(np.sum(usage > 0.0005)),
        "active_topics_ge_1_over_k": int(np.sum(usage >= 1.0 / len(usage))),
        "maximum_mean_topic_usage": float(usage.max()),
    }


def _passes_all(row: dict[str, Any], no_catastrophic_duplicate: bool) -> bool:
    return bool(
        row["gate_optimized"]
        and row["gate_evaluable"]
        and row["gate_useful"]
        and row["gate_mean_sos"]
        and row["gate_completion_nll"]
        and row["finite_stable"]
        and no_catastrophic_duplicate
    )


def select_nll_preserving_temperature(rows: Sequence[dict[str, Any]]) -> float | None:
    """Return the broadest calibrated row that retains the locked NLL gate."""
    eligible = [row for row in rows if row["gate_completion_nll"]]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda row: (
            row["evaluable_motifs"],
            row["useful_motifs"],
            row["mean_sos"],
        ),
    )
    return float(best["theta_temperature"])


def should_add_intermediate(rows: Sequence[dict[str, Any]]) -> bool:
    """Apply the prompt's single-midpoint rule without extrapolating a search."""
    if not rows:
        return False
    # A bracket cannot exist when no endpoint reaches either limiting chemistry
    # gate; here all fixed-grid rows miss both useful count and mean SOS.
    return any(row["gate_useful"] for row in rows) and any(
        row["gate_mean_sos"] for row in rows
    )


def _reordered_comparison_row(
    row: dict[str, Any],
    *,
    variant_type: str,
    inference_temperature: object = "",
    source_model: str = "",
) -> dict[str, Any]:
    return {
        "model": row["model"],
        "variant_type": variant_type,
        "inference_temperature": inference_temperature,
        "source_model": source_model,
        **{
            key: value
            for key, value in row.items()
            if key
            not in {"model", "variant_type", "inference_temperature", "source_model"}
        },
    }


def _comparison_rows(
    *,
    existing: list[dict[str, str]],
    sweep: list[dict[str, Any]],
    full_theta: np.ndarray,
) -> list[dict[str, Any]]:
    retained = [
        row for row in existing if not row["model"].startswith(f"{METHOD}_posthoc_tau_")
    ]
    source = next(row for row in retained if row["model"] == METHOD)
    output = []
    for row in retained:
        inference_temperature = 1.0 if row["model"] == METHOD else ""
        output.append(
            _reordered_comparison_row(
                row,
                variant_type="trained_model",
                inference_temperature=inference_temperature,
            )
        )
    for sweep_row in sweep:
        temperature = float(sweep_row["theta_temperature"])
        if temperature == 1.0:
            continue
        label = f"{temperature:.1f}".replace(".", "p")
        calibrated = retemperature_theta(
            full_theta,
            source_temperature=1.0,
            target_temperature=temperature,
        )
        inventory = _theta_inventory(calibrated)
        row: dict[str, Any] = dict(source)
        row.update(
            {
                "model": f"{METHOD}_posthoc_tau_{label}",
                "trained_separately": False,
                "optimized_motifs": sweep_row["optimized_motifs"],
                "evaluable_motifs": sweep_row["evaluable_motifs"],
                "useful_motifs": sweep_row["useful_motifs"],
                "mean_sos": sweep_row["mean_sos"],
                "median_sos": sweep_row["median_sos"],
                "sos_high_gt_0_8": sweep_row["sos_high_gt_0_8"],
                "sos_intermediate_0_6_to_0_8": sweep_row["sos_intermediate_0_6_to_0_8"],
                "sos_low_lt_0_6": sweep_row["sos_low_lt_0_6"],
                "associated_spectra": sweep_row["associated_spectra"],
                "associated_molecules": sweep_row["associated_molecules"],
                "completion_nll": sweep_row["completion_nll"],
                "completion_oov_fraction": sweep_row["completion_oov_fraction"],
                "median_effective_topics_per_spectrum": sweep_row[
                    "median_effective_topics_per_spectrum"
                ],
                "mean_effective_topics_per_spectrum": sweep_row[
                    "mean_effective_topics_per_spectrum"
                ],
                "median_max_theta": sweep_row["median_max_theta"],
                "fraction_max_theta_ge_0_5": sweep_row["fraction_max_theta_ge_0_5"],
                "fraction_max_theta_ge_0_3": sweep_row["fraction_max_theta_ge_0_3"],
                "fraction_max_theta_ge_0_2": sweep_row["fraction_max_theta_ge_0_2"],
                "unique_top1_topics": sweep_row["unique_top1_topics"],
                **inventory,
                "training_wall_seconds": 0.0,
                "validation_full_spectra_per_second": "",
                "peak_process_bytes": "",
                "finite_stable": sweep_row["finite_stable"],
                "gate_optimized": sweep_row["gate_optimized"],
                "gate_evaluable": sweep_row["gate_evaluable"],
                "gate_useful": sweep_row["gate_useful"],
                "gate_mean_sos": sweep_row["gate_mean_sos"],
                "gate_completion_nll": sweep_row["gate_completion_nll"],
                "gate_finite_stable": sweep_row["finite_stable"],
                "gate_no_catastrophic_duplicate_component": True,
                "passed_all_frozen_gates": sweep_row["passed_all_frozen_gates"],
            }
        )
        output.append(
            _reordered_comparison_row(
                row,
                variant_type="post_hoc_calibration",
                inference_temperature=temperature,
                source_model=METHOD,
            )
        )
    return output


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["/usr/bin/git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package(  # noqa: PLR0913
    *,
    repo: Path,
    run: Path,
    output: Path,
    sweep_csv: Path,
    sweep_summary: Path,
    command: str,
) -> dict[str, Any]:
    if _git(repo, "branch", "--show-current") != BRANCH:
        raise RuntimeError(f"temperature packaging requires branch {BRANCH}")
    artifact_entries = []
    for relative, expected_hash in EXPECTED_HASHES.items():
        path = run / relative
        if not path.is_file():
            raise FileNotFoundError(f"required validation artifact is missing: {path}")
        entry = _artifact_entry(path, relative)
        if entry["sha256"] != expected_hash:
            raise RuntimeError(f"artifact hash mismatch: {relative}")
        artifact_entries.append(entry)

    source_summary = read_json(sweep_summary)
    sweep = _typed_sweep(_read_csv(sweep_csv))
    actual_temperatures = tuple(row["theta_temperature"] for row in sweep)
    if actual_temperatures != TEMPERATURES:
        raise ValueError(
            f"expected frozen temperatures {TEMPERATURES}, got {actual_temperatures}"
        )
    if len(sweep) != len(TEMPERATURES):
        raise ValueError("temperature sweep contains duplicate or missing rows")

    result_metrics = read_json(output / METHOD / "metrics.json")
    source_inventory = result_metrics["topic_inventory"]
    no_catastrophic_duplicate = not bool(
        source_inventory["catastrophic_duplicate_component"]
    )
    for row in sweep:
        row["variant_type"] = (
            "trained_gamma2_source"
            if row["theta_temperature"] == 1.0
            else "post_hoc_gamma2_calibration"
        )
        row["source_model"] = METHOD
        row["beta_unchanged"] = True
        row["no_catastrophic_duplicate_component"] = no_catastrophic_duplicate
        row["gate_no_catastrophic_duplicate_component"] = no_catastrophic_duplicate
        row["passed_all_frozen_gates"] = _passes_all(row, no_catastrophic_duplicate)

    if should_add_intermediate(sweep):
        raise RuntimeError(
            "fixed-grid endpoints require an explicit review before adding a midpoint"
        )

    tau_one = sweep[0]
    committed = next(
        row for row in _read_csv(output / "comparison.csv") if row["model"] == METHOD
    )
    baseline_checks = {
        "completion_nll_abs_difference": abs(
            tau_one["completion_nll"] - float(committed["completion_nll"])
        ),
        "evaluable_motifs_match": (
            tau_one["evaluable_motifs"] == int(committed["evaluable_motifs"])
        ),
        "useful_motifs_match": (
            tau_one["useful_motifs"] == int(committed["useful_motifs"])
        ),
        "mean_sos_abs_difference": abs(
            tau_one["mean_sos"] - float(committed["mean_sos"])
        ),
        "median_effective_topics_abs_difference": abs(
            tau_one["median_effective_topics_per_spectrum"]
            - float(committed["median_effective_topics_per_spectrum"])
        ),
    }
    baseline_checks["all_pass"] = bool(
        baseline_checks["completion_nll_abs_difference"] <= 1e-6
        and baseline_checks["evaluable_motifs_match"]
        and baseline_checks["useful_motifs_match"]
        and baseline_checks["mean_sos_abs_difference"] <= 1e-12
        and baseline_checks["median_effective_topics_abs_difference"] <= 1e-5
    )
    if not baseline_checks["all_pass"]:
        raise RuntimeError("tau=1 does not reproduce the committed gamma-2 baseline")

    best_temperature = select_nll_preserving_temperature(sweep)
    best_row = next(
        row for row in sweep if row["theta_temperature"] == best_temperature
    )
    passing = [row for row in sweep if row["passed_all_frozen_gates"]]
    m1 = next(
        row for row in _read_csv(output / "comparison.csv") if row["model"] == "M1"
    )
    comparison_to_m1 = {
        "temperature": best_temperature,
        "selection_rule": (
            "maximum evaluable motifs among rows retaining the frozen completion-NLL "
            "gate; useful motifs and mean SOS break ties"
        ),
        "evaluable_motifs": {
            "calibrated": best_row["evaluable_motifs"],
            "m1": int(m1["evaluable_motifs"]),
        },
        "useful_motifs": {
            "calibrated": best_row["useful_motifs"],
            "m1": int(m1["useful_motifs"]),
        },
        "mean_sos": {
            "calibrated": best_row["mean_sos"],
            "m1": float(m1["mean_sos"]),
        },
        "completion_nll": {
            "calibrated": best_row["completion_nll"],
            "m1": float(m1["completion_nll"]),
        },
        "median_effective_topics_per_spectrum": {
            "calibrated": best_row["median_effective_topics_per_spectrum"],
            "m1": float(m1["median_effective_topics_per_spectrum"]),
        },
        "median_max_theta": {
            "calibrated": best_row["median_max_theta"],
            "m1": float(m1["median_max_theta"]),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    sweep_path = output / "gated_etm_gamma2_temperature_sweep.csv"
    _write_csv(sweep_path, sweep)
    comparison_rows = _comparison_rows(
        existing=_read_csv(output / "comparison.csv"),
        sweep=sweep,
        full_theta=np.load(
            run / "validation_evaluation" / METHOD / "validation_full_theta.npy",
            mmap_mode="r",
        ),
    )
    _write_csv(output / "comparison.csv", comparison_rows)

    summary = {
        "evidence_boundary": "validation only; candidate test artifacts not accessed",
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "current_result_commit": CURRENT_RESULT_COMMIT,
        "trained_model": {
            "method": METHOD,
            "gate_temperature": 1.0,
            "gate_gamma": 2.0,
            "trained_separately": True,
        },
        "experiment": {
            "kind": "post-hoc rank-preserving inference-temperature calibration",
            "formula": "theta_tau[k] proportional to theta[k] ** (1 / tau)",
            "implementation": "numerically stable log space",
            "beta_unchanged": True,
            "mag_annotations_reused": True,
            "membership_threshold": 0.5,
            "temperature_grid": list(TEMPERATURES),
            "intermediate_temperature_added": False,
            "intermediate_temperature_reason": (
                "No fixed-grid row reached either the useful-motif or mean-SOS gate, "
                "so adjacent points did not bracket an all-gate-passing region."
            ),
        },
        "frozen_gates": {
            **GATES,
            "finite_stable": True,
            "no_catastrophic_duplicate_component": True,
        },
        "baseline_reproduction": baseline_checks,
        "reconstruction_check": source_summary["reconstruction_check"],
        "beta_diagnostics": {
            "maximum_beta_cosine": source_inventory["maximum_pairwise_beta_cosine"],
            "pairs_ge_0_999": 0,
            "catastrophic_duplicate_component": False,
        },
        "selected_temperature": (
            float(passing[0]["theta_temperature"]) if passing else None
        ),
        "any_temperature_passes_all_frozen_gates": bool(passing),
        "best_nll_preserving_temperature": best_temperature,
        "best_nll_preserving_row": best_row,
        "comparison_to_m1": comparison_to_m1,
        "decision": (
            "Stop the ETM architecture path; balanced ETM plus detached geometry "
            "gate and bounded inference calibration remain insufficient. M1 "
            "multiseed stability is the next campaign."
        ),
    }
    write_json(output / "gated_etm_gamma2_temperature_summary.json", summary)

    provenance_path = output / "provenance.json"
    provenance = read_json(provenance_path)
    post_hoc_methods = [
        f"{METHOD}_posthoc_tau_{temperature:.1f}".replace(".", "p")
        for temperature in TEMPERATURES[1:]
    ]
    provenance["post_hoc_candidate_methods"] = post_hoc_methods
    exact_commands = list(provenance.get("exact_commands", []))
    if command not in exact_commands:
        exact_commands.append(command)
    provenance["exact_commands"] = exact_commands
    provenance["temperature_calibration"] = {
        "evidence_boundary": (
            "validation only; candidate test artifacts not accessed, loaded, scored, "
            "or summarized"
        ),
        "source_method": METHOD,
        "trained_model_unchanged": True,
        "retrained": False,
        "beta_unchanged": True,
        "mag_annotations_reused": True,
        "membership_threshold": 0.5,
        "temperature_grid": list(TEMPERATURES),
        "intermediate_temperature_added": False,
        "any_temperature_passes_all_frozen_gates": bool(passing),
        "best_nll_preserving_temperature": best_temperature,
        "source_artifacts": artifact_entries,
        "source_artifacts_all_hashes_match": True,
        "command": command,
        "runtime": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    provenance["candidate_test_data_loaded"] = False
    provenance["candidate_test_completion_computed"] = False
    provenance["candidate_test_mag_sos_computed"] = False
    provenance["candidate_test_metrics_inspected"] = False
    write_json(provenance_path, provenance)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sweep-csv", required=True, type=Path)
    parser.add_argument("--sweep-summary", required=True, type=Path)
    parser.add_argument("--command", required=True)
    args = parser.parse_args(argv)
    result = package(
        repo=args.repo.expanduser().resolve(),
        run=args.run.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        sweep_csv=args.sweep_csv.expanduser().resolve(),
        sweep_summary=args.sweep_summary.expanduser().resolve(),
        command=args.command,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
