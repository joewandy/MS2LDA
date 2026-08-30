"""Package a completed validation-only MSnLib comparison for review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmarks.neural_ms2lda.utils import read_json, write_json

METHODS = ("etm", "pooled_likelihood", "pooled_mi005")
HANDOFF_SHA = "9baec8aa62f684480eba35d4fc7f626c46f7b804"
MAIN_SHA = "20de0e45aec25203e6bc38770a795b25cc18bff7"
M1 = {
    "method": "m1_reference",
    "optimized_motifs": 884,
    "evaluable_motifs": 408,
    "useful_motifs": 265,
    "mean_sos": 0.6580793714074608,
    "median_sos": 0.6488636363636364,
    "completion_nll": 8.974139925584877,
    "training_wall_seconds": 1764.874561500037,
    "parameters": 167168,
}
GATES = {
    "optimized_motifs": 840,
    "evaluable_motifs": 388,
    "useful_motifs": 252,
    "mean_sos": 0.651498,
    "completion_nll": 9.422847,
}


def sha256(path: Path) -> str:
    """Hash one local artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Return absolute path, size, and SHA-256 for one local artifact."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one deterministic small CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def command_output(command: list[str]) -> str:
    """Capture a read-only environment command for provenance."""
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    return result.stdout.strip()


def safe_hardware_profile() -> str:
    """Return useful Mac hardware fields without unique device identifiers."""
    executable = shutil.which("system_profiler") or "system_profiler"
    raw = command_output([executable, "SPHardwareDataType"])
    allowed = (
        "Model Name:",
        "Model Identifier:",
        "Chip:",
        "Total Number of Cores:",
        "Memory:",
    )
    return "\n".join(
        line.strip() for line in raw.splitlines() if line.strip().startswith(allowed)
    )


def chemistry_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the headline values from the shared scorer output."""
    scored = payload["high_confidence_chemistry"]
    bands = scored["sos_bands"]
    return {
        "optimized_motifs": int(round(payload["annotation_coverage"] * 1000)),
        "evaluable_motifs": int(scored["eligible_topics"]),
        "useful_motifs": int(bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]),
        "mean_sos": float(scored["mean_sos"]),
        "median_sos": float(scored["median_sos"]),
        "sos_high_gt_0_8": int(bands["high_gt_0_8"]),
        "sos_intermediate_0_6_to_0_8": int(bands["intermediate_0_6_to_0_8"]),
        "sos_low_lt_0_6": int(bands["low_lt_0_6"]),
        "associated_spectra": int(scored["associated_spectra"]),
        "associated_molecules": int(scored["associated_molecules"]),
        "leakage_audit_passed": bool(payload["heldout_compounds_excluded_from_mag"]),
    }


def candidate_row(run: Path, method: str) -> dict[str, Any]:
    """Build one complete comparison row from local result artifacts."""
    model = read_json(run / "models" / method / "result.json")
    chemistry = read_json(run / "validation_chemical" / method / "complete.json")
    row = {"method": method, **chemistry_summary(chemistry)}
    metrics = model["metrics"]
    inventory = metrics["topic_inventory"]
    runtime = metrics["runtime"]
    entropy = metrics.get("theta_entropy") or {}
    row.update(
        {
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
            "mean_nearest_topic_beta_cosine": inventory[
                "mean_nearest_topic_beta_cosine"
            ],
            "maximum_pairwise_beta_cosine": inventory["maximum_pairwise_beta_cosine"],
            "top_word_uniqueness": metrics["top_word_uniqueness"],
            "mean_conditional_theta_entropy": entropy.get(
                "mean_conditional_theta_entropy"
            ),
            "marginal_theta_entropy": entropy.get("marginal_theta_entropy"),
            "theta_mutual_information": entropy.get("mutual_information"),
            "training_wall_seconds": runtime["training_wall_seconds"],
            "validation_full_spectra_per_second": runtime[
                "validation_full_spectra_per_second"
            ],
            "peak_process_bytes": runtime["memory"]["peak_process_bytes"],
            "parameters": model["parameters"],
            "finite_stable": metrics["finite_stable"],
        }
    )
    catastrophic = bool(
        row["active_topics_usage_gt_0_0005"] < 100
        or row["corpus_effective_topic_count"] < 100
        or row["maximum_mean_topic_usage"] > 0.1
    )
    row["catastrophic_inventory_collapse"] = catastrophic
    row["gate_optimized"] = row["optimized_motifs"] >= GATES["optimized_motifs"]
    row["gate_evaluable"] = row["evaluable_motifs"] >= GATES["evaluable_motifs"]
    row["gate_useful"] = row["useful_motifs"] >= GATES["useful_motifs"]
    row["gate_mean_sos"] = row["mean_sos"] >= GATES["mean_sos"]
    row["gate_completion_nll"] = row["completion_nll"] <= GATES["completion_nll"]
    row["gate_finite_stable"] = bool(row["finite_stable"])
    row["gate_no_catastrophic_inventory_collapse"] = not catastrophic
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


def m1_row(fieldnames: list[str]) -> dict[str, Any]:
    """Align the locked incumbent with the candidate comparison schema."""
    row = dict.fromkeys(fieldnames)
    row.update(M1)
    row.update(
        {
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


def copy_model_evidence(run: Path, output: Path, method: str) -> None:
    """Copy small reviewer-facing model evidence, never large arrays/checkpoints."""
    destination = output / method
    destination.mkdir(parents=True, exist_ok=True)
    source = run / "models" / method
    for source_name, target_name in (
        ("config.json", "protocol.json"),
        ("result.json", "metrics.json"),
        ("training_history.csv", "training_history.csv"),
        ("chemical_scores.csv", "chemical_scores.csv"),
        ("top_words.csv", "top_words.csv"),
    ):
        shutil.copy2(source / source_name, destination / target_name)
    fragment = source / "fragment_mass_summary.json"
    if fragment.is_file():
        shutil.copy2(fragment, destination / fragment.name)
    chemistry = read_json(run / "validation_chemical" / method / "complete.json")
    write_json(destination / "chemical_validation.json", chemistry)


def environment_text() -> str:
    """Capture system and both pinned environment package inventories."""
    conda = shutil.which("conda")
    sections = [
        "# System",
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
    ]
    for command in (
        ["sw_vers"],
        ["uname", "-a"],
        ["sysctl", "-n", "hw.memsize"],
    ):
        executable = shutil.which(command[0])
        sections.extend(
            (
                f"\n$ {' '.join(command)}",
                command_output([executable or command[0], *command[1:]]),
            )
        )
    sections.extend(
        (
            "\n$ system_profiler SPHardwareDataType (sanitized)",
            safe_hardware_profile(),
        )
    )
    if conda:
        for environment in ("ms2lda-neural",):
            sections.extend(
                (
                    f"\n# conda list --export: {environment}",
                    command_output([conda, "list", "-n", environment, "--export"]),
                )
            )
    return "\n".join(sections) + "\n"


def exact_commands(
    run: Path,
    data_root: Path,
    output: Path,
    execution_sha: str,
) -> list[str]:
    """Return the commands actually used for the campaign."""
    prefix = "conda run --no-capture-output -n"
    module = "python -m scripts.run_msnlib_model_comparison"
    commands = [
        (
            "python scripts/download_msnlib_validation_assets.py "
            f"--data-root {data_root} --verify-only"
        ),
        (
            "conda run --no-capture-output -n ms2lda-neural python "
            f"scripts/run_msnlib_model_comparison.py prepare --run {run} "
            f"--data-root {data_root}"
        ),
        f"{prefix} ms2lda-neural {module} prepare --run {run} --data-root {data_root}",
        f"{prefix} ms2lda-neural {module} smoke --run {run} --device mps",
        f"{prefix} ms2lda-neural {module} smoke --run {run} --device cpu",
        (
            f"{prefix} ms2lda-neural {module} train --run {run} --method etm "
            "--device cpu --etm-epochs 120 --etm-batch-size 256"
        ),
        (
            f"{prefix} ms2lda-neural {module} train --run {run} "
            "--method pooled_likelihood --device cpu"
        ),
        (
            f"{prefix} ms2lda-neural {module} train --run {run} "
            "--method pooled_mi005 --device cpu"
        ),
    ]
    commands.extend(
        (
            f"{prefix} ms2lda-neural {module} chemical --run {run} "
            f"--data-root {data_root} --method {method}"
        )
        for method in METHODS
    )
    commands.extend(
        (
            (
                f"{prefix} ms2lda-neural {module} ecrtm-probe --run {run} "
                "--device cpu --max-iter 1000 --batch-size 8 "
                "--wall-cap-seconds 900"
            ),
            (
                f"{prefix} ms2lda-neural {module} ecrtm-probe --run {run} "
                "--device cpu --max-iter 1000 --batch-size 200 "
                "--wall-cap-seconds 900"
            ),
            (
                f"{prefix} ms2lda-neural {module} ecrtm-probe --run {run} "
                "--device cpu --max-iter 50 --batch-size 200 "
                "--wall-cap-seconds 900"
            ),
            (
                "conda run -n ms2lda-neural python -m pytest -q "
                "benchmarks/neural_ms2lda/tests"
            ),
            (
                "conda run -n ms2lda-neural ruff check --config "
                "benchmarks/neural_ms2lda/ruff.toml benchmarks/neural_ms2lda "
                "scripts/run_msnlib_model_comparison.py "
                "scripts/package_msnlib_model_comparison.py"
            ),
            (
                "conda run -n ms2lda-neural black --check "
                "benchmarks/neural_ms2lda scripts/run_msnlib_model_comparison.py "
                "scripts/package_msnlib_model_comparison.py"
            ),
            (
                "conda run -n ms2lda-neural python -m "
                f"scripts.package_msnlib_model_comparison --run {run} "
                f"--data-root {data_root} --output {output} "
                f"--execution-sha {execution_sha}"
            ),
        )
    )
    return commands


def package(
    run: Path,
    data_root: Path,
    output: Path,
    *,
    execution_sha: str,
) -> dict[str, Any]:
    """Build the complete small-text review surface."""
    output.mkdir(parents=True, exist_ok=True)
    candidates = [candidate_row(run, method) for method in METHODS]
    fieldnames = list(candidates[0])
    comparison = [m1_row(fieldnames), *candidates]
    write_csv(output / "comparison.csv", comparison)
    write_csv(
        output / "runtime_memory.csv",
        [
            {
                "method": row["method"],
                "training_wall_seconds": row["training_wall_seconds"],
                "validation_full_spectra_per_second": row[
                    "validation_full_spectra_per_second"
                ],
                "peak_process_bytes": row["peak_process_bytes"],
                "parameters": row["parameters"],
                "finite_stable": row["finite_stable"],
            }
            for row in candidates
        ],
    )
    for method in METHODS:
        copy_model_evidence(run, output, method)
    shutil.copy2(run / "protocol.json", output / "locked_protocol.json")
    shutil.copy2(
        run / "comparison_preparation.json", output / "preparation_summary.json"
    )
    shutil.copy2(run / "real_batch_smoke_mps.json", output / "smoke_mps.json")
    shutil.copy2(run / "real_batch_smoke.json", output / "smoke_cpu.json")
    ecr_output = output / "ecrtm_feasibility"
    ecr_output.mkdir(parents=True, exist_ok=True)
    for name in ("canonical.json", "bounded_50.json"):
        shutil.copy2(run / "ecrtm_feasibility" / name, ecr_output / name)
    canonical = read_json(ecr_output / "canonical.json")
    bounded = read_json(ecr_output / "bounded_50.json")
    batches = math.ceil(27222 / 200)
    feasibility = {
        "batches_per_epoch": batches,
        "canonical_projected_epoch_seconds": canonical["total_seconds"] * batches,
        "canonical_projected_40_epoch_seconds": canonical["total_seconds"]
        * batches
        * 40,
        "bounded_50_projected_epoch_seconds": bounded["total_seconds"] * batches,
        "bounded_50_projected_40_epoch_seconds": bounded["total_seconds"]
        * batches
        * 40,
        "decision": (
            "Full ECRTM not run: ETM did not show catastrophic inventory collapse; "
            "its decisive failure was chemical breadth. Canonical ECRTM would add "
            "about seven hours locally, while the 50-step approximation was not "
            "numerically converged at real K/V."
        ),
    }
    write_json(ecr_output / "summary.json", feasibility)

    asset_manifest = read_json(data_root / "acquisition_manifest.json")
    important = []
    for method in METHODS:
        important.extend(
            file_record(path)
            for path in (
                run / "models" / method / "weights.pt",
                run / "validation_evaluation" / method / "beta.npy",
                run / "validation_evaluation" / method / "validation_full_theta.npy",
            )
        )
    important.extend(
        file_record(path)
        for path in (
            run / "mag/index/spec2vec_filtered.faiss",
            run / "mag/index/kept_original_ids.npy",
            run / "token_features/features.npy",
        )
    )
    provenance = {
        "evidence_boundary": (
            "validation candidates only; no candidate test theta, completion, "
            "chemistry, MAG, or SOS was opened"
        ),
        "branch": "experiment/msnlib-etm-pooled-local-20260827",
        "origin_main_sha": MAIN_SHA,
        "research_handoff_sha": HANDOFF_SHA,
        "execution_branch_sha": execution_sha,
        "execution_run_directory": str(run),
        "data_root": str(data_root),
        "data_asset_manifest": asset_manifest,
        "data_asset_manifest_sha256": sha256(data_root / "acquisition_manifest.json"),
        "prepared_data_summary": read_json(run / "data/complete.json"),
        "random_seeds": {
            "locked_split_sgns_pooled": 42,
            "etm": 7043,
        },
        "model_configurations": {
            method: read_json(run / "models" / method / "config.json")
            for method in METHODS
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "memory_bytes": command_output(
                [shutil.which("sysctl") or "sysctl", "-n", "hw.memsize"]
            ),
            "hardware_profile": safe_hardware_profile(),
        },
        "software": {
            "python": sys.version,
            "environment_file": "environment.txt",
        },
        "exact_commands": exact_commands(run, data_root, output, execution_sha),
        "important_large_uncommitted_artifacts": important,
        "acquisition_verification_caveat": (
            "Both archives passed frozen Zenodo MD5 and ZIP checks and all required "
            "extracted files matched exact sizes. The current verify-only command "
            "then rejected the older richer manifest solely because its extracted "
            "entries include additional SHA-256 fields. The manifest was preserved."
        ),
        "operational_failures": [
            {
                "stage": "first direct-script preparation invocation",
                "effect": "failed before run initialization or data access",
                "error": "ModuleNotFoundError: No module named 'benchmarks'",
                "resolution": (
                    "Used the repository-root module invocation python -m "
                    "scripts.run_msnlib_model_comparison."
                ),
            },
            {
                "stage": "asset verify-only manifest comparison",
                "effect": (
                    "archives and extracted assets verified before the manifest "
                    "schema comparison returned exit 1"
                ),
                "error": "acquisition manifest extracted-input evidence differs",
                "resolution": (
                    "Confirmed the difference is the older manifest's additional "
                    "per-file SHA-256 fields; preserved the richer manifest."
                ),
            },
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "provenance.json", provenance)
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")
    (output / "exact_commands.txt").write_text(
        "\n".join(provenance["exact_commands"]) + "\n", encoding="utf-8"
    )

    def table_row(label: str, row: dict[str, Any], gate: str) -> str:
        return (
            f"| {label} | {row['optimized_motifs']} | {row['evaluable_motifs']} "
            f"| {row['useful_motifs']} | {row['mean_sos']:.6f} "
            f"| {row['median_sos']:.6f} | {row['completion_nll']:.6f} "
            f"| {gate} |"
        )

    table = "\n".join(
        (
            (
                "| method | optimized | evaluable | useful | mean SOS | "
                "median SOS | completion NLL | frozen gate |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---|",
            table_row("M1 reference", M1, "pass"),
            table_row("canonical ETM", candidates[0], "fail"),
            table_row("pooled likelihood", candidates[1], "fail"),
            table_row("pooled + MI 0.05", candidates[2], "fail"),
        )
    )
    vocabulary_size = provenance["prepared_data_summary"]["vocabulary"][
        "vocabulary_size"
    ]
    readme = f"""# Real MSnLib validation-only comparison (seed 42)

No candidate test output was opened. All candidates used the locked split,
train-only V={vocabulary_size} vocabulary,
train-only 48D SGNS, completion views, leakage-filtered MAG, and SOS
implementation from M1.

## Result

{table}

All candidates passed the completion-NLL and finite-execution gates, but none
preserved M1's chemical breadth. ETM retained a non-catastrophic inventory (269
topics above 0.0005 usage; corpus effective count 344.6), yet produced only 79
useful validation motifs. The pooled models annotated many topic spectra but
their diffuse document mixtures (median 130.0/124.6 effective topics) almost
never crossed the locked 0.5 association threshold, leaving only 14 evaluable
motifs each. Weak MI did not materially help and was slightly worse on NLL and
annotation coverage.

## ETM channel and collapse diagnostics

ETM fragment mass was materially asymmetric but not uniformly one-sided:
minimum 0.0134, median 0.3420, maximum 0.9973, with 15.5% of topics below 0.1
or above 0.9. No forced-50/50 ETM was run because the predeclared simulation
found no consistent benefit and the real failure was broad chemical retention,
not channel mass alone. ETM did not show catastrophic topic-inventory collapse:
269 topics exceeded 0.0005 usage, corpus effective count was 344.6, and mean
nearest-topic beta cosine was 0.321, although one near-duplicate pair reached
0.9995 and document mixtures were diffuse (median 43.8 effective topics).

## ECRTM feasibility and decision

The canonical full-K/V probe converged in 201 Sinkhorn iterations at residual
0.00475 and used about 1.08 GB peak process memory. At batch 200 it took 4.66
seconds per forward/backward/step, projecting to about 10.6 minutes per epoch
and 7.1 hours for 40 epochs. The labelled 50-step approximation took 1.34
seconds but remained unconverged (residual 1.24). Full ECRTM was not run because
ETM's decisive failure was chemical breadth rather than the topic collapse ECR
is designed to repair; the long comparator was therefore not scientifically
warranted for this first campaign.

## Review surface

`comparison.csv` contains all gates and headline diagnostics. Each model
directory contains its exact protocol, metrics, training history, motif-level
chemical scores, and top words. `provenance.json` records commands, SHAs, asset
identifiers, hardware/software, seeds, and SHA-256 values for uncommitted large
artifacts. `ecrtm_feasibility/` contains the exact and approximate probe evidence.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output": str(output),
        "comparison_rows": len(comparison),
        "large_artifacts_recorded": len(important),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Package one completed local run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-sha", required=True)
    args = parser.parse_args(argv)
    result = package(
        args.run.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        execution_sha=args.execution_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
