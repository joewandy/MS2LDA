"""Declarative execution plan for the Contextual Sparse ETM reproduction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

METHOD = "contextual_sparse_etm"
TRAINING_SEEDS = (7043, 23, 37)
SYNTHETIC_SEEDS = (11, 23, 37)


class ReproductionPaths(NamedTuple):
    """Filesystem layout rooted in one clean-room reproduction."""

    root: Path
    assets: Path
    prepared: Path
    synthetic: Path
    controls: Path
    tomotopy: Path
    contextual: dict[int, Path]
    smoke: Path
    logs: Path
    stages: Path


class Stage(NamedTuple):
    """One deterministic command and the files that prove it completed."""

    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    requires_idle_system: bool = False


def reproduction_paths(root: Path) -> ReproductionPaths:
    """Resolve every path under the requested clean-room root."""
    resolved = root.expanduser().resolve()
    return ReproductionPaths(
        root=resolved,
        assets=resolved / "assets",
        prepared=resolved / "prepared",
        synthetic=resolved / "synthetic",
        controls=resolved / "real" / "controls",
        tomotopy=resolved / "real" / "tomotopy",
        contextual={
            seed: resolved / "real" / f"contextual_seed_{seed}"
            for seed in TRAINING_SEEDS
        },
        smoke=resolved / "real" / "contextual_smoke_epoch1",
        logs=resolved / "logs",
        stages=resolved / "stages",
    )


def acceptance_policy() -> dict[str, object]:
    """Return the scientific acceptance rules frozen before result access."""
    return {
        "exact_gates": {
            "source_spectra": 41_568,
            "retained_spectra": 38_888,
            "connectivity_groups": 38_465,
            "split_groups": 28_572,
            "split_spectra": {"train": 27_222, "validation": 3_889, "test": 7_777},
            "vocabulary_size": 21_233,
            "leaked_compounds": 0,
            "leaked_split_groups": 0,
            "topics": 1000,
            "training_seeds": list(TRAINING_SEEDS),
            "synthetic_seeds": list(SYNTHETIC_SEEDS),
        },
        "identity_gates": [
            "all methods use identical matrix, vocabulary, SGNS, and MAG-index hashes",
            (
                "test inputs are released only after every fitted model and validation "
                "result is frozen"
            ),
            "all probability matrices are finite, non-negative, and row-normalized",
            "MAG reports zero clustering and optimization exceptions for every model",
            (
                "SOS bands sum to evaluable motifs and useful equals high plus "
                "intermediate"
            ),
        ],
        "claim_gates": [
            (
                "Contextual Sparse ETM has more evaluable and useful test "
                "motifs than every comparator"
            ),
            "Tomotopy retains higher conditional mean and median SOS",
            "dense ETM controls retain lower completion NLL than Contextual Sparse ETM",
            (
                "Contextual Sparse ETM retains median effective topics at most five "
                "and at least 800 unique winners"
            ),
            "all three Contextual seeds avoid catastrophic duplicate components",
            (
                "the high-K treatment recovers all 18 planted motifs as winners with "
                "median support at most three"
            ),
        ],
        "interpretation": (
            "Data/configuration identities are exact. Stochastic scientific results "
            "must preserve the predeclared directional claims."
        ),
    }


def _module(*arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", *arguments)


def _synthetic_output_name(
    seed: int,
    topics: int,
    routing_variant: str,
    theta_transform: str,
) -> str:
    if routing_variant == "etm":
        method = f"balanced_etm_{theta_transform}_raw_counts"
    else:
        suffix = "" if theta_transform == "softmax" else f"_{theta_transform}"
        method = f"balanced_etm_routing_{routing_variant}{suffix}_raw_counts"
    return f"seed_{seed}_K_{topics}_{method}"


def _synthetic_stage(
    paths: ReproductionPaths,
    *,
    seed: int,
    topics: int,
    routing_variant: str,
    theta_transform: str,
) -> Stage:
    name = _synthetic_output_name(seed, topics, routing_variant, theta_transform)
    command = _module(
        "scripts.run_routing_etm_campaign",
        "--output-root",
        str(paths.synthetic),
        "--seed",
        str(seed),
        "--fitted-topics",
        str(topics),
        "--routing-variant",
        routing_variant,
        "--theta-transform",
        theta_transform,
        "--reconstruction-scaling",
        "raw_counts",
        "--epochs",
        "120",
        "--batch-size",
        "200",
        "--device",
        "cuda",
        "--threads",
        "6",
        "--training-documents",
        "800",
        "--validation-documents",
        "160",
    )
    result = paths.synthetic / "synthetic_runs" / name / "result.json"
    return Stage(f"synthetic_{name}", command, (result,))


def _validated_stage_plan(stages: list[Stage]) -> list[Stage]:
    """Reject ambiguous stage names or multiply owned output artifacts."""
    names = [stage.name for stage in stages]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        msg = f"stage plan contains duplicate names: {duplicate_names}"
        raise RuntimeError(msg)
    output_paths = [str(path) for stage in stages for path in stage.outputs]
    duplicate_outputs = sorted(
        {path for path in output_paths if output_paths.count(path) > 1},
    )
    if duplicate_outputs:
        msg = f"stage plan contains multiply owned outputs: {duplicate_outputs}"
        raise RuntimeError(msg)
    return stages


def stage_plan(paths: ReproductionPaths) -> list[Stage]:
    """Return the dependency-ordered train/validation/test execution plan."""
    stages = [
        Stage(
            "preflight_tests",
            _module("pytest", "-q", "benchmarks/neural_ms2lda/tests"),
            (),
        ),
        Stage(
            "acquire_public_assets",
            _module(
                "scripts.download_msnlib_validation_assets",
                "--data-root",
                str(paths.assets),
            ),
            (paths.assets / "acquisition_manifest.json",),
        ),
        Stage(
            "prepare_data_and_sgns",
            _module(
                "scripts.run_msnlib_model_comparison",
                "prepare",
                "--run",
                str(paths.prepared),
                "--data-root",
                str(paths.assets),
            ),
            (
                paths.prepared / "comparison_preparation.json",
                paths.prepared / "data/complete.json",
                paths.prepared / "token_features/complete.json",
            ),
        ),
        Stage(
            "build_leakage_filtered_mag_index",
            _module(
                "benchmarks.neural_ms2lda.campaign",
                "prepare-shared-index",
                "--run",
                str(paths.prepared),
                "--data-root",
                str(paths.assets),
            ),
            (paths.prepared / "mag/index/complete.json",),
        ),
    ]
    for name, run in (
        ("controls", paths.controls),
        ("tomotopy", paths.tomotopy),
        ("contextual_smoke", paths.smoke),
        *(
            (f"contextual_seed_{seed}", paths.contextual[seed])
            for seed in TRAINING_SEEDS
        ),
    ):
        stages.append(
            Stage(
                f"seal_validation_view_{name}",
                _module(
                    "scripts.prepare_msnlib_validation_view",
                    "--run",
                    str(run),
                    "--prepared-run",
                    str(paths.prepared),
                ),
                (run / "validation_input_manifest.json",),
            ),
        )
    stages.append(
        Stage(
            "contextual_one_epoch_smoke",
            _module(
                "scripts.run_contextual_sparse_etm",
                "train",
                "--real-run",
                str(paths.smoke),
                "--epochs",
                "1",
                "--batch-size",
                "256",
                "--device",
                "cuda",
                "--threads",
                "6",
                "--training-seed",
                "7043",
            ),
            (paths.smoke / "models" / METHOD / "result.json",),
            requires_idle_system=True,
        ),
    )
    for topics, seeds, formulations in (
        (
            36,
            SYNTHETIC_SEEDS,
            (
                ("etm", "softmax"),
                ("etm", "entmax15"),
                ("top2_context", "softmax"),
                ("top2_context", "entmax15"),
            ),
        ),
        (
            128,
            (11,),
            (
                ("etm", "softmax"),
                ("etm", "entmax15"),
                ("top2_context", "entmax15"),
            ),
        ),
    ):
        for seed in seeds:
            for routing_variant, theta_transform in formulations:
                stages.append(
                    _synthetic_stage(
                        paths,
                        seed=seed,
                        topics=topics,
                        routing_variant=routing_variant,
                        theta_transform=theta_transform,
                    ),
                )
    stages.extend(
        (
            Stage(
                "tomotopy_validation",
                _module(
                    "scripts.run_tomotopy_validation",
                    "--run",
                    str(paths.tomotopy),
                ),
                (paths.tomotopy / "tomotopy/validation_only_result.json",),
                requires_idle_system=True,
            ),
            Stage(
                "tomotopy_validation_chemistry",
                _module(
                    "benchmarks.neural_ms2lda.chemical",
                    "--run",
                    str(paths.tomotopy),
                    "--data-root",
                    str(paths.assets),
                    "--method",
                    "tomotopy",
                    "--split",
                    "validation",
                ),
                (paths.tomotopy / "validation_chemical/tomotopy/complete.json",),
            ),
            Stage(
                "canonical_etm_train",
                _module(
                    "scripts.run_msnlib_model_comparison",
                    "train",
                    "--run",
                    str(paths.controls),
                    "--method",
                    "etm",
                    "--device",
                    "cpu",
                    "--etm-epochs",
                    "120",
                    "--etm-batch-size",
                    "256",
                ),
                (paths.controls / "models/etm/result.json",),
                requires_idle_system=True,
            ),
            Stage(
                "canonical_etm_chemistry",
                _module(
                    "scripts.run_msnlib_model_comparison",
                    "chemical",
                    "--run",
                    str(paths.controls),
                    "--data-root",
                    str(paths.assets),
                    "--method",
                    "etm",
                ),
                (paths.controls / "validation_chemical/etm/complete.json",),
            ),
            Stage(
                "balanced_etm_train",
                _module(
                    "scripts.run_msnlib_model_comparison",
                    "train",
                    "--run",
                    str(paths.controls),
                    "--method",
                    "etm_balanced",
                    "--device",
                    "cpu",
                    "--etm-epochs",
                    "120",
                    "--etm-batch-size",
                    "256",
                ),
                (paths.controls / "models/etm_balanced/result.json",),
                requires_idle_system=True,
            ),
            Stage(
                "balanced_etm_chemistry",
                _module(
                    "scripts.run_msnlib_model_comparison",
                    "chemical",
                    "--run",
                    str(paths.controls),
                    "--data-root",
                    str(paths.assets),
                    "--method",
                    "etm_balanced",
                ),
                (paths.controls / "validation_chemical/etm_balanced/complete.json",),
            ),
        ),
    )
    for seed in TRAINING_SEEDS:
        run = paths.contextual[seed]
        stages.extend(
            (
                Stage(
                    f"contextual_seed_{seed}_train",
                    _module(
                        "scripts.run_contextual_sparse_etm",
                        "train",
                        "--real-run",
                        str(run),
                        "--epochs",
                        "120",
                        "--batch-size",
                        "256",
                        "--device",
                        "cuda",
                        "--threads",
                        "6",
                        "--training-seed",
                        str(seed),
                    ),
                    (run / "models" / METHOD / "result.json",),
                    requires_idle_system=True,
                ),
                Stage(
                    f"contextual_seed_{seed}_chemistry",
                    _module(
                        "scripts.run_contextual_sparse_etm",
                        "chemical",
                        "--real-run",
                        str(run),
                        "--data-root",
                        str(paths.assets),
                    ),
                    (run / "validation_chemical" / METHOD / "complete.json",),
                ),
            ),
        )

    # Only now, after every model and validation result is frozen, expose the
    # fixed test split.  The exposure manifests hash both the fitted weights and
    # the completed validation artifacts before any test file is linked.
    stages.extend(
        (
            Stage(
                "release_test_view_controls",
                _module(
                    "scripts.prepare_msnlib_test_view",
                    "--run",
                    str(paths.controls),
                    "--prepared-run",
                    str(paths.prepared),
                    "--method",
                    "etm",
                    "--method",
                    "etm_balanced",
                ),
                (paths.controls / "test_input_manifest.json",),
            ),
            Stage(
                "release_test_view_tomotopy",
                _module(
                    "scripts.prepare_msnlib_test_view",
                    "--run",
                    str(paths.tomotopy),
                    "--prepared-run",
                    str(paths.prepared),
                    "--method",
                    "tomotopy",
                ),
                (paths.tomotopy / "test_input_manifest.json",),
            ),
        ),
    )
    stages.extend(
        Stage(
            f"release_test_view_contextual_seed_{seed}",
            _module(
                "scripts.prepare_msnlib_test_view",
                "--run",
                str(paths.contextual[seed]),
                "--prepared-run",
                str(paths.prepared),
                "--method",
                METHOD,
            ),
            (paths.contextual[seed] / "test_input_manifest.json",),
        )
        for seed in TRAINING_SEEDS
    )

    for method in ("etm", "etm_balanced"):
        stages.extend(
            (
                Stage(
                    f"{method}_test_evaluation",
                    _module(
                        "scripts.evaluate_frozen_etm_test",
                        "--run",
                        str(paths.controls),
                        "--method",
                        method,
                        "--device",
                        "cpu",
                        "--batch-size",
                        "256",
                        "--threads",
                        "6",
                    ),
                    (paths.controls / "evaluation" / method / "complete.json",),
                    requires_idle_system=True,
                ),
                Stage(
                    f"{method}_test_chemistry",
                    _module(
                        "benchmarks.neural_ms2lda.chemical",
                        "--run",
                        str(paths.controls),
                        "--data-root",
                        str(paths.assets),
                        "--method",
                        method,
                        "--split",
                        "test",
                    ),
                    (paths.controls / "chemical" / method / "complete.json",),
                ),
            ),
        )
    stages.append(
        Stage(
            "tomotopy_test_evaluation_and_chemistry",
            _module(
                "scripts.evaluate_frozen_tomotopy_test",
                "--run",
                str(paths.tomotopy),
                "--data-root",
                str(paths.assets),
            ),
            (paths.tomotopy / "tomotopy/test_result.json",),
            requires_idle_system=True,
        ),
    )
    for seed in TRAINING_SEEDS:
        run = paths.contextual[seed]
        stages.extend(
            (
                Stage(
                    f"contextual_seed_{seed}_test_evaluation",
                    _module(
                        "scripts.evaluate_frozen_etm_test",
                        "--run",
                        str(run),
                        "--method",
                        METHOD,
                        "--device",
                        "cuda",
                        "--batch-size",
                        "256",
                        "--threads",
                        "6",
                    ),
                    (run / "evaluation" / METHOD / "complete.json",),
                    requires_idle_system=True,
                ),
                Stage(
                    f"contextual_seed_{seed}_test_chemistry",
                    _module(
                        "benchmarks.neural_ms2lda.chemical",
                        "--run",
                        str(run),
                        "--data-root",
                        str(paths.assets),
                        "--method",
                        METHOD,
                        "--split",
                        "test",
                    ),
                    (run / "chemical" / METHOD / "complete.json",),
                ),
            ),
        )
    return _validated_stage_plan(stages)
