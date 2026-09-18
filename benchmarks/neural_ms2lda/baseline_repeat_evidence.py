"""Package exactly the six agreed baseline fits, without rerunning any model.

Rows represent fits, not spectra. The old plain ETM fit remains eligible because
the new fits use its batch-256 recipe. The old Tomotopy fit is historical only:
the three new LDA fits each have genuine full-validation mixture arrays.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from .baseline_repeats import baseline_fit_identity, require_cached_fit
from .evidence_bundle import portable_value
from .model_evaluation import theta_support_diagnostics
from .report_comparison import baseline_summaries
from .reproducibility import normalize_probability_rows
from .reproduction_audit import file_record, verify_linked_inputs, write_csv
from .reproduction_comparison import chemistry_fields, chemistry_summary
from .utils import read_json_object, sha256_file, write_json

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room"
PLAN = (("tomotopy", (11, 23, 42)), ("etm", (7012, 7024)))


def scheduling_handoff(root: Path) -> dict | None:
    """Require a completed, unchanged audit if execution scheduling was handed off.

    The original experiment started with a sequential LDA queue. Its supervisor
    was then paused, while two additional LDA fits ran from the same immutable
    source snapshot. The supervisor resumed only after those complete caches
    were checked. That one-off operational change is not another model recipe.
    """
    audit = root / "parallel_handoff"
    if not audit.exists():
        return None
    record = read_json_object(audit / "handoff.json")
    if (
        record["status"] != "supervisor_resumed"
        or record["model_recipe_changed"]
        or record["source_snapshot_changed"]
        or sha256_file(audit / "handoff_runner.py") != record["runner_sha256"]
        or {item["training_seed"] for item in record["completed"]} != {23, 42}
    ):
        raise RuntimeError("scheduling handoff is incomplete or its audit differs")
    return record


def stage_provenance(root: Path, identity: str, handoff: dict | None):
    """Keep actual launches distinct from later validated cache rechecks.

    Stage ``finished_utc`` records when the supervisor observed the child's exit.
    For a child preserved while that supervisor was paused, this can be later
    than the actual exit. Use the model's measured training duration for timing
    comparisons; do not misrepresent cache recheck duration as training time.
    """
    handed_off = handoff is not None and identity in {
        "tomotopy_seed23_attempt1",
        "tomotopy_seed42_attempt1",
    }
    groups = {"execution": root / "parallel_handoff" if handed_off else root}
    if handed_off:
        groups["cache_recheck"] = root
    payloads, artifacts, records = {}, {}, {}
    for label, origin in groups.items():
        records[label] = []
        for name in ("seal", "fit_validation", "chemistry"):
            source = origin / "stages" / f"{identity}.{name}.json"
            stage = read_json_object(source)
            if (
                stage["status"] != "complete"
                or stage["return_code"] != 0
                or Path(stage["run"]).name != identity
                or stage["stage"] != name
            ):
                raise RuntimeError("baseline stage is incomplete, failed or mismatched")
            destination = f"{identity}/{label}/stage_{name}.json"
            payloads[destination] = stage
            log_name = f"{identity}/{label}/{name}.log"
            # Preserve original raw log identity as well as a portable text copy.
            log_source = origin / "logs" / f"{identity}.{name}.log"
            artifacts[log_name] = log_source
            records[label].append(
                {
                    "stage": name,
                    "record": destination,
                    "log": log_name,
                    "raw_record": file_record(source, relative_to=REPO),
                    "raw_log": file_record(log_source, relative_to=REPO),
                }
            )
    return payloads, artifacts, records


def scientific_inputs(manifest: dict) -> dict:
    """Compare sealed data roles, excluding path/JSON formatting differences.

    The historical protocol was exported as canonical JSON. Its semantic
    settings are checked separately; scientific arrays must match byte-for-byte.
    MAG index identities are included because chemistry depends on that library.
    """
    values = {}
    for row in manifest["linked_inputs"]:
        path = Path(row["linked_path"])
        if path.name == "protocol.json":
            continue
        if path.parent.name == "data":
            role = "data/" + path.name
        elif path.parent.name == "token_features":
            role = "token_features/" + path.name
        elif path.parent.name == "index":
            role = "mag/index/" + path.name
        else:
            raise ValueError(f"unknown sealed scientific input role: {path}")
        if role in values:
            raise ValueError("duplicate sealed input role")
        values[role] = row["sha256"]
    return values


def comparison_row(label, seed, run, completion, chemistry, support, parameters):
    """Use the same chemical summary functions and column names as the old study."""
    row = {
        "model": label,
        "training_seed": seed,
        "run": run,
        **chemistry_fields(label, chemistry_summary(chemistry)),
        "completion_nll": float(completion["nll_per_token"]),
        "completion_documents": int(completion["eligible_documents"]),
        "completion_tokens": int(completion["in_vocabulary_tokens"]),
        "median_effective_topics": float(
            support["median_effective_topics_per_spectrum"]
        ),
        "median_exact_support": support.get("median_exact_support", ""),
        "unique_top1_topics": int(support["unique_top1_topics"]),
        "finite_stable": True,
        "parameters": parameters,
    }
    if completion["total_documents"] != 3889:
        raise ValueError("completion population differs from the frozen validation set")
    return row


def check_chemical_associations(chemistry: dict, theta: np.ndarray):
    """Trace every full-validation mixture to its reported dominant-topic count.

    Equal grand totals alone cannot detect chemistry accidentally copied from
    another seed. Its per-topic spectrum counts must also match argmax(theta).
    """
    scores = chemistry["chemical_evaluation"]["topic_scores"]
    counts = {row["topic_id"]: row["associated_spectra"] for row in scores}
    topics = theta.shape[1]
    expected = np.bincount(theta.argmax(axis=1), minlength=topics)
    if (
        len(scores) != topics
        or set(counts) != set(range(topics))
        or not np.array_equal(expected, [counts[topic] for topic in range(topics)])
    ):
        raise ValueError("chemical associations do not match this fit's mixtures")


def check_etm_recipe(result: dict, seed: int):
    """Do not pool a different batch size, budget, architecture or seed."""
    expected = {
        "seed": seed,
        "topics": 1000,
        "epochs": 120,
        "batch_size": 256,
        "hidden_dimensions": 800,
        "embedding_dimensions": 48,
        "learning_rate": 0.005,
        "weight_decay": 1.2e-6,
        "optimizer": "Adam",
        "device": "cuda",
        "decoder_normalization": "global topic-word softmax",
    }
    if result.get("architecture_method") != "etm" or any(
        result["config"].get(key) != value for key, value in expected.items()
    ):
        raise ValueError("plain ETM repeat recipe differs")
    if result["parameters"] != 19278000:
        raise ValueError("plain ETM parameter count differs")
    if "fit_identity" in result:
        # New fits explicitly record optimizer defaults and shuffle randomness;
        # the retained historical fit predates that stronger cache contract.
        recipe = result["fit_identity"]["recipe"]
        expected_recipe = {
            "method": "etm",
            "topics": 1000,
            "epochs": 120,
            "batch_size": 256,
            "hidden": 800,
            "learning_rate": 0.005,
            "weight_decay": 1.2e-6,
            "adam_betas": [0.9, 0.999],
            "device": "cuda",
            "shuffle_seed": seed + 18,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise ValueError("plain ETM optimizer or shuffle recipe differs")


def publish_evidence(
    output: Path,
    *,
    root: Path,
    payloads: dict,
    text_artifacts: dict,
    rows: list,
    manifest: dict,
    handoff: dict | None,
):
    """Copy verified evidence, seal every published file, then rename atomically.

    Only the task-owned temporary copies are cleaned on failure. Original run
    directories and frozen evidence are never modified, even if copying or
    archiving fails. Keeping this mechanical transaction separate makes failure
    behavior testable without fitting models or recreating a scientific study.
    """
    if output.exists():
        raise FileExistsError("use a fresh evidence destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    replacements = [(str(root), "<repeat-root>"), (str(REPO), "<repository>")]
    with tempfile.TemporaryDirectory(
        prefix=".baseline-evidence-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "evidence"
        staging.mkdir()
        for name, value in payloads.items():
            write_json(staging / name, portable_value(value, replacements))
        for name, path in text_artifacts.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                portable_value(path.read_text(encoding="utf-8"), replacements),
                encoding="utf-8",
            )
        if handoff is not None:
            write_json(
                staging / "scheduling_handoff/audit.json",
                portable_value(handoff, replacements),
            )
            # An exact audit copy, not a supported experiment API. Preserve bytes
            # (including original paths) so its original runner hash verifies.
            shutil.copy2(
                root / "parallel_handoff/handoff_runner.py",
                staging / "scheduling_handoff/handoff_runner.py",
            )
        write_csv(staging / "runs.csv", rows)
        with tarfile.open(staging / "postprocessing_source.tar.gz", "w:gz") as archive:
            for record in payloads["postprocessing_source_manifest.json"]:
                path = REPO / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise RuntimeError("postprocessing source changed during packaging")
                archive.add(path, arcname=record["path"])
        with tarfile.open(staging / "execution_source.tar.gz", "w:gz") as archive:
            for record in payloads["execution_source_manifest.json"]["sources"]:
                path = root / "source_snapshot" / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise RuntimeError("executed source changed during packaging")
                archive.add(path, arcname=f"source_snapshot/{record['path']}")
        sealed = {
            **portable_value(manifest, replacements),
            "sources": [
                file_record(path, relative_to=staging)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        }
        write_json(staging / "manifest.json", sealed)
        staging.rename(output)


def package(root: Path, output: Path):
    """Validate all runs first, then atomically publish a portable evidence bundle."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(
            "use a fresh evidence destination; do not replace frozen results"
        )
    if read_json_object(root / "driver.json")["status"] != "complete":
        raise RuntimeError("all five new baseline fits must finish before packaging")
    source = read_json_object(root / "source_manifest.json")
    handoff = scheduling_handoff(root)
    # Post-hoc derivation happens after model fitting. Freeze its own source
    # identity rather than attributing later report/audit edits to the fits.
    postprocessing = [
        file_record(path, relative_to=REPO)
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    ]
    for record in source["sources"]:
        if sha256_file(root / "source_snapshot" / record["path"]) != record["sha256"]:
            raise RuntimeError("executed source snapshot changed")
    old = FROZEN / "controls/etm"
    expected_inputs = scientific_inputs(
        read_json_object(old / "validation_input_manifest.json")
    )
    protocol = read_json_object(FROZEN / "protocol.json")
    payloads, text_artifacts, records, rows = {}, {}, [], []

    def remember(name, path):
        value = read_json_object(path)
        payloads[name] = value
        return value

    if (root / "runtime_observation.json").is_file():
        remember("runtime_observation.json", root / "runtime_observation.json")

    for kind, seeds in PLAN:
        for seed in seeds:
            identity = f"{kind}_seed{seed}_attempt1"
            run = root / "real" / identity
            launches, logs, stages = stage_provenance(root, identity, handoff)
            payloads.update(launches)
            text_artifacts.update(logs)
            seal = remember(
                f"{identity}/validation_input_manifest.json",
                run / "validation_input_manifest.json",
            )
            verify_linked_inputs(seal, field="linked_inputs")
            if scientific_inputs(seal) != expected_inputs:
                raise ValueError(
                    "baseline scientific inputs differ from the original fit"
                )
            if read_json_object(run / "protocol.json") != protocol:
                raise ValueError("baseline protocol settings differ")
            result_path = (
                run / "tomotopy/validation_only_result.json"
                if kind == "tomotopy"
                else run / "models/etm/result.json"
            )
            result = remember(f"{identity}/result.json", result_path)
            chemistry = remember(
                f"{identity}/chemical.json",
                run / "validation_chemical" / kind / "complete.json",
            )
            if (
                chemistry["split"] != "validation"
                or chemistry["method"] != kind
                or chemistry["topics"] != 1000
                or chemistry["heldout_compounds_excluded_from_mag"] is not True
            ):
                raise ValueError("baseline chemistry violates the comparison scope")
            theta_path = (
                run / "validation_evaluation" / kind / "validation_full_theta.npy"
            )
            theta = normalize_probability_rows(
                np.load(theta_path), name="baseline full theta"
            )
            if theta.shape != (3889, 1000):
                raise ValueError("baseline mixture shape differs")
            check_chemical_associations(chemistry, theta)
            support = {
                **theta_support_diagnostics(theta),
                "unique_top1_topics": int(len(np.unique(theta.argmax(axis=1)))),
            }
            payloads[f"{identity}/derived_support.json"] = {
                "metrics": support,
                "theta": file_record(theta_path, relative_to=REPO),
                "implementation": file_record(
                    Path(__file__).with_name("model_evaluation.py"), relative_to=REPO
                ),
                "split": "validation",
                "model_fitting_performed": False,
            }
            if kind == "etm":
                check_etm_recipe(result, seed)
                cached = result
                if (
                    cached["fit_identity"]["recipe"]["threads"]
                    != protocol["cpu_threads"]
                ):
                    raise ValueError("plain ETM CPU-thread recipe differs")
                completion = result["metrics"]["document_completion"]
                parameters = result["parameters"]
                label = "canonical ETM"
                weights = run / "models/etm/weights.pt"
            else:
                cached = result["training"]
                completion = result["validation"]["metrics"][
                    "validation_document_completion"
                ]
                parameters, label = "", "Tomotopy LDA"
                weights = run / "tomotopy/model.bin"
                recipe = cached["fit_identity"]["recipe"]
                if recipe != {
                    "topics": 1000,
                    "tomotopy": protocol["tomotopy"],
                    "version": "0.13.0",
                    "alpha_optimization_interval": 10,
                }:
                    raise ValueError("Tomotopy repeat recipe differs")
            expected = baseline_fit_identity(
                run, seed, cached["fit_identity"]["recipe"]
            )
            require_cached_fit(cached, expected)
            rows.append(
                comparison_row(
                    label, seed, identity, completion, chemistry, support, parameters
                )
            )
            records.append(
                {
                    "model": label,
                    "training_seed": seed,
                    "run": identity,
                    "origin": "new_fit",
                    "fit_identity": expected,
                    "stages": stages,
                    "raw_artifacts": [
                        file_record(path, relative_to=REPO)
                        for path in (result_path, theta_path, weights)
                    ],
                }
            )

    identity = "frozen_etm_seed7043"
    result = remember(f"{identity}/result.json", old / "result.json")
    check_etm_recipe(result, 7043)
    chemistry = remember(f"{identity}/chemical.json", old / "validation_chemical.json")
    remember(
        f"{identity}/validation_input_manifest.json",
        old / "validation_input_manifest.json",
    )
    historical_path = FROZEN / "reproduction_manifest.json"
    historical = read_json_object(historical_path)
    payloads[f"{identity}/execution_provenance.json"] = {
        "source": historical["source"],
        "source_manifest": file_record(historical_path, relative_to=REPO),
        "environment": {
            name: historical["environment"][name]
            for name in ("python", "platform", "conda_packages")
        },
        "neural_execution_device": historical["neural_execution_device"],
        "interpretation": (
            "Retained ETM fit used the same scientific recipe and input bytes. "
            "It ran under WSL2; new fits ran under native Linux. Across-fit SD "
            "therefore describes run-to-run variation, not exclusively seed effects."
        ),
    }
    rows.append(
        comparison_row(
            "canonical ETM",
            7043,
            identity,
            result["metrics"]["document_completion"],
            chemistry,
            result["metrics"]["theta_distribution"],
            result["parameters"],
        )
    )
    records.append(
        {
            "model": "canonical ETM",
            "training_seed": 7043,
            "run": identity,
            "origin": "retained_historical_fit",
            "config": result["config"],
            "execution_provenance": f"{identity}/execution_provenance.json",
            "raw_artifacts": [
                file_record(old / name, relative_to=REPO)
                for name in (
                    "result.json",
                    "validation_chemical.json",
                    "validation_input_manifest.json",
                )
            ],
        }
    )
    rows.sort(key=lambda row: (row["model"] != "Tomotopy LDA", row["training_seed"]))
    summary = baseline_summaries(rows)
    payloads.update(
        {
            "execution_source_manifest.json": source,
            "postprocessing_source_manifest.json": postprocessing,
            "protocol.json": protocol,
            "summary.json": summary,
        }
    )
    publish_evidence(
        output,
        root=root,
        payloads=payloads,
        text_artifacts=text_artifacts,
        rows=rows,
        handoff=handoff,
        manifest={
            "schema_version": 1,
            "reported_split": "validation",
            "records": records,
            "scientific_input_sha256": expected_inputs,
            "historical_tomotopy_included": False,
            "selected_model_refitted": False,
            "postprocessing_source": (
                "postprocessing_source.tar.gz and its manifest preserve the "
                "separate source used to derive support metrics and package "
                "results after fitting. Training used execution_source.tar.gz."
            ),
            "scheduling": (
                "Initial sequential-LDA driver plus audited parallel handoff; "
                "one worker per LDA fit, serialized ETM fitting and chemistry."
                if handoff is not None
                else "See exact executed source and per-fit launch records."
            ),
            "stage_timestamp_semantics": (
                "finished_utc is when the supervisor observed child exit. "
                "It may be delayed for children preserved while the original "
                "supervisor was paused. Actual parallel launch records and "
                "later validated cache rechecks are separate; fit durations "
                "come from model results, never cache recheck wall time."
            ),
            "inference_policy": (
                "Tomotopy 0.13.0 default per-document RNG; "
                "100 iterations; one worker"
            ),
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
