"""Declarative execution plan for the Contextual Sparse ETM reproduction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from .study_protocol import (
    METHOD,
    NEURAL_DEVICE,
    SYNTHETIC_ARTIFACT_LABELS,
    SYNTHETIC_SEEDS,
    TRAINING_ACCESS_AUDIT_FILENAME,
    TRAINING_SEEDS,
)


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
    """Return the study's scientific integrity and directional checks."""
    return {
        "exact_gates": {
            "neural_execution_device": NEURAL_DEVICE,
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
            "every test spectrum is associated with exactly one dominant topic",
        ],
        "claim_gates": [
            (
                "Contextual Sparse ETM has more evaluable and useful test "
                "motifs than every comparator"
            ),
            "dense ETM controls retain lower completion NLL than Contextual Sparse ETM",
            (
                "Contextual Sparse ETM retains median effective topics at most five "
                "and at least 800 unique winners"
            ),
            "all three Contextual seeds avoid catastrophic duplicate components",
            (
                "the high-K treatment truth-matches all 18 planted motifs at beta "
                "cosine at least 0.50 with median support at most three"
            ),
        ],
        "interpretation": (
            "Data/configuration identities are exact. Stochastic scientific results "
            "must pass the reported directional checks."
        ),
    }


def _module(*arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", *arguments)


def _synthetic_output_name(
    seed: int,
    topics: int,
    formulation: str,
) -> str:
    try:
        label = SYNTHETIC_ARTIFACT_LABELS[formulation]
    except KeyError as error:
        raise ValueError(f"unknown synthetic formulation: {formulation}") from error
    return f"seed_{seed}_K_{topics}_{label}"


def _synthetic_stage(
    paths: ReproductionPaths,
    *,
    seed: int,
    topics: int,
    formulation: str,
) -> Stage:
    name = _synthetic_output_name(seed, topics, formulation)
    command = _module(
        "scripts.run_contextual_sparse_etm_synthetic",
        "--output-root",
        str(paths.synthetic),
        "--seed",
        str(seed),
        "--fitted-topics",
        str(topics),
        "--formulation",
        formulation,
        "--epochs",
        "120",
        "--batch-size",
        "200",
        "--device",
        NEURAL_DEVICE,
        "--threads",
        "6",
        "--training-documents",
        "800",
        "--validation-documents",
        "160",
    )
    result = paths.synthetic / "synthetic_runs" / name / "result.json"
    return Stage(f"synthetic_{name}", command, (result,))


def probability_artifact_paths(run: Path, method: str) -> tuple[Path, Path, Path]:
    """Return the beta, validation-theta, and test-theta files used by the paper."""
    return (
        run / "validation_evaluation" / method / "beta.npy",
        run / "validation_evaluation" / method / "validation_full_theta.npy",
        run / "evaluation" / method / "test_full_theta.npy",
    )


def _etm_training_outputs(run: Path, method: str) -> tuple[Path, ...]:
    """Return immutable ETM training and validation artifacts used downstream."""
    model = run / "models" / method
    beta, validation_theta, _ = probability_artifact_paths(run, method)
    return (
        model / "result.json",
        model / "weights.pt",
        model / "config.json",
        model / "training_history.csv",
        model / "top_words.csv",
        model / "fragment_mass_summary.json",
        model / "duplicate_component_summary.json",
        model / TRAINING_ACCESS_AUDIT_FILENAME,
        run / "validation_evaluation" / method / "complete.json",
        beta,
        validation_theta,
    )


def _contextual_training_outputs(run: Path) -> tuple[Path, ...]:
    """Return immutable Contextual Sparse ETM artifacts used downstream."""
    model = run / "models" / METHOD
    return (
        *_etm_training_outputs(run, METHOD),
        model / "theta_support_summary.csv",
        model / "context_evidence_support_summary.csv",
        model / "provenance.json",
    )


def _etm_test_outputs(run: Path, method: str) -> tuple[Path, ...]:
    """Return every frozen-test ETM artifact used by audit or packaging."""
    output = run / "evaluation" / method
    _, _, test_theta = probability_artifact_paths(run, method)
    return (
        output / "complete.json",
        output / "test_access_audit.json",
        output / "beta.npy",
        output / "test_observed_theta.npy",
        test_theta,
    )


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


def _preparation_stages(paths: ReproductionPaths) -> list[Stage]:
    """Return source checks plus real and synthetic input preparation."""
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
                "scripts.prepare_contextual_sparse_etm_data",
                "--run",
                str(paths.prepared),
                "--data-root",
                str(paths.assets),
            ),
            (
                paths.prepared / "preparation_summary.json",
                paths.prepared / "protocol.json",
                paths.prepared / "data_root.txt",
                paths.prepared / "data/complete.json",
                paths.prepared / "token_features/complete.json",
            ),
        ),
        Stage(
            "build_leakage_filtered_mag_index",
            _module(
                "benchmarks.neural_ms2lda.mag",
                "--run",
                str(paths.prepared),
                "--data-root",
                str(paths.assets),
            ),
            (paths.prepared / "mag/index/complete.json",),
        ),
    ]
    for seed in SYNTHETIC_SEEDS:
        seed_directory = paths.synthetic / "synthetic_artifacts" / f"seed_{seed}"
        stages.append(
            Stage(
                f"prepare_synthetic_seed_{seed}",
                _module(
                    "scripts.prepare_contextual_sparse_etm_synthetic",
                    "--output-root",
                    str(paths.synthetic),
                    "--seed",
                    str(seed),
                    "--threads",
                    "6",
                    "--training-documents",
                    "800",
                    "--validation-documents",
                    "160",
                ),
                (
                    seed_directory / "artifact_manifest.json",
                    seed_directory / "token_features/complete.json",
                ),
            ),
        )
    return stages


def _validation_view_stages(paths: ReproductionPaths) -> list[Stage]:
    """Return stages that expose train/validation data to isolated run roots."""
    stages = []
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
    return stages


def _smoke_and_synthetic_stages(paths: ReproductionPaths) -> list[Stage]:
    """Return the bounded CUDA smoke fit and truth-known ablation matrix."""
    stages = []
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
                NEURAL_DEVICE,
                "--threads",
                "6",
                "--training-seed",
                "7043",
            ),
            _contextual_training_outputs(paths.smoke),
            requires_idle_system=True,
        ),
    )
    for topics, seeds, formulations in (
        (
            36,
            SYNTHETIC_SEEDS,
            (
                "balanced_softmax",
                "balanced_entmax",
                "contextual_softmax",
                "contextual_entmax",
            ),
        ),
        (
            128,
            (11,),
            (
                "balanced_softmax",
                "balanced_entmax",
                "contextual_entmax",
            ),
        ),
    ):
        for seed in seeds:
            for formulation in formulations:
                stages.append(
                    _synthetic_stage(
                        paths,
                        seed=seed,
                        topics=topics,
                        formulation=formulation,
                    ),
                )
    return stages


def _validation_fit_stages(paths: ReproductionPaths) -> list[Stage]:
    """Return all frozen fits and validation-only evaluations."""
    stages = []
    stages.extend(
        (
            Stage(
                "tomotopy_validation",
                _module(
                    "scripts.run_tomotopy_validation",
                    "--run",
                    str(paths.tomotopy),
                ),
                (
                    paths.tomotopy / "tomotopy/validation_only_result.json",
                    paths.tomotopy / "tomotopy/model.bin",
                    paths.tomotopy / "tomotopy/complete.json",
                    paths.tomotopy / "tomotopy/validation_access_audit.json",
                    paths.tomotopy / "validation_evaluation/tomotopy/complete.json",
                    *probability_artifact_paths(paths.tomotopy, "tomotopy")[:2],
                ),
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
                    "scripts.run_etm_controls",
                    "train",
                    "--run",
                    str(paths.controls),
                    "--method",
                    "etm",
                    "--device",
                    NEURAL_DEVICE,
                    "--epochs",
                    "120",
                    "--batch-size",
                    "256",
                ),
                _etm_training_outputs(paths.controls, "etm"),
                requires_idle_system=True,
            ),
            Stage(
                "canonical_etm_chemistry",
                _module(
                    "scripts.run_etm_controls",
                    "chemical",
                    "--run",
                    str(paths.controls),
                    "--data-root",
                    str(paths.assets),
                    "--method",
                    "etm",
                ),
                (
                    paths.controls / "validation_chemical/etm/complete.json",
                    paths.controls / "models/etm/validation_access_audit.json",
                ),
            ),
            Stage(
                "balanced_etm_train",
                _module(
                    "scripts.run_etm_controls",
                    "train",
                    "--run",
                    str(paths.controls),
                    "--method",
                    "etm_balanced",
                    "--device",
                    NEURAL_DEVICE,
                    "--epochs",
                    "120",
                    "--batch-size",
                    "256",
                ),
                _etm_training_outputs(paths.controls, "etm_balanced"),
                requires_idle_system=True,
            ),
            Stage(
                "balanced_etm_chemistry",
                _module(
                    "scripts.run_etm_controls",
                    "chemical",
                    "--run",
                    str(paths.controls),
                    "--data-root",
                    str(paths.assets),
                    "--method",
                    "etm_balanced",
                ),
                (
                    paths.controls / "validation_chemical/etm_balanced/complete.json",
                    paths.controls / "models/etm_balanced/validation_access_audit.json",
                ),
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
                        NEURAL_DEVICE,
                        "--threads",
                        "6",
                        "--training-seed",
                        str(seed),
                    ),
                    _contextual_training_outputs(run),
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
                    (
                        run / "validation_chemical" / METHOD / "complete.json",
                        run / "models" / METHOD / "validation_access_audit.json",
                    ),
                ),
            ),
        )
    return stages


def _test_release_stages(paths: ReproductionPaths) -> list[Stage]:
    """Expose the held-out test split only after validation is frozen."""
    stages = []

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
    return stages


def _test_evaluation_stages(paths: ReproductionPaths) -> list[Stage]:
    """Return frozen-model test inference and chemistry evaluation stages."""
    stages = []

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
                        NEURAL_DEVICE,
                        "--batch-size",
                        "256",
                        "--threads",
                        "6",
                    ),
                    _etm_test_outputs(paths.controls, method),
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
            (
                paths.tomotopy / "tomotopy/test_result.json",
                paths.tomotopy / "evaluation/tomotopy/complete.json",
                probability_artifact_paths(paths.tomotopy, "tomotopy")[2],
                paths.tomotopy / "chemical/tomotopy/complete.json",
            ),
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
                        NEURAL_DEVICE,
                        "--batch-size",
                        "256",
                        "--threads",
                        "6",
                    ),
                    _etm_test_outputs(run, METHOD),
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
    return stages


def stage_plan(paths: ReproductionPaths) -> list[Stage]:
    """Return the dependency-ordered train/validation/test execution plan."""
    stages = [
        *_preparation_stages(paths),
        *_validation_view_stages(paths),
        *_smoke_and_synthetic_stages(paths),
        *_validation_fit_stages(paths),
        *_test_release_stages(paths),
        *_test_evaluation_stages(paths),
    ]
    return _validated_stage_plan(stages)
