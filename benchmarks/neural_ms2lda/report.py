"""Build the one canonical machine-readable result used by the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import file_sha256, read_json, write_json


def _chemistry_summary(chemistry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the probability-thresholded SOS quantities in the paper."""
    scored = chemistry["high_confidence_chemistry"]
    bands = scored["sos_bands"]
    return {
        "optimized_motifs": int(
            round(float(chemistry["annotation_coverage"]) * chemistry["topics"])
        ),
        "annotation_coverage": float(chemistry["annotation_coverage"]),
        "high_confidence_evaluable_motifs": int(scored["eligible_topics"]),
        "useful_high_confidence_motifs": int(
            bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"]
        ),
        "sos_bands": bands,
        "mean_sos": float(scored["mean_sos"]),
        "median_sos": float(scored["median_sos"]),
    }


def _method_result(
    *,
    method: str,
    validation_chemistry: dict[str, Any],
    test_chemistry: dict[str, Any],
    fitting_seconds: float,
    fitting_workers: int,
) -> dict[str, Any]:
    """Return one paper-facing method comparison row."""
    return {
        "method": method,
        "fitting_seconds": float(fitting_seconds),
        "fitting_workers": int(fitting_workers),
        "validation": _chemistry_summary(validation_chemistry),
        "test": _chemistry_summary(test_chemistry),
    }


def build_results(run_dir: str | Path) -> dict[str, Any]:
    """Write the canonical comparison assembled from verified stage outputs."""
    directory = Path(run_dir).expanduser().resolve()
    protocol = read_json(directory / "protocol.resolved.json")
    neural_training = read_json(directory / "model/complete.json")
    tomotopy_training = read_json(directory / "tomotopy/complete.json")
    paths = {
        "neural_validation_evaluation": directory
        / "validation_evaluation/neural/complete.json",
        "tomotopy_validation_evaluation": directory
        / "validation_evaluation/tomotopy/complete.json",
        "neural_validation_chemistry": directory
        / "validation_chemical/neural/complete.json",
        "tomotopy_validation_chemistry": directory
        / "validation_chemical/tomotopy/complete.json",
        "neural_test_evaluation": directory / "evaluation/neural/complete.json",
        "tomotopy_test_evaluation": directory / "evaluation/tomotopy/complete.json",
        "neural_test_chemistry": directory / "chemical/neural/complete.json",
        "tomotopy_test_chemistry": directory / "chemical/tomotopy/complete.json",
    }
    evidence = {name: read_json(path) for name, path in paths.items()}
    neural_test = evidence["neural_test_evaluation"]
    tomotopy_test = evidence["tomotopy_test_evaluation"]
    neural_warm = neural_test["metrics"]["warm_in_memory_batch_inference"]
    tomotopy_warm = tomotopy_test["metrics"]["warm_in_memory_batch_inference"]
    methods = [
        _method_result(
            method="neural",
            validation_chemistry=evidence["neural_validation_chemistry"],
            test_chemistry=evidence["neural_test_chemistry"],
            fitting_seconds=float(neural_training["elapsed_seconds"]),
            fitting_workers=int(protocol["cpu_threads"]),
        ),
        _method_result(
            method="tomotopy",
            validation_chemistry=evidence["tomotopy_validation_chemistry"],
            test_chemistry=evidence["tomotopy_test_chemistry"],
            fitting_seconds=float(tomotopy_training["training_seconds_total"]),
            fitting_workers=int(tomotopy_training["training_workers"]),
        ),
    ]
    result = {
        "comparison_contract": {
            "association_probability_threshold": float(
                protocol["chemistry"]["membership_threshold"]
            ),
            "cpu_threads": int(protocol["cpu_threads"]),
            "neural_selected_epoch": int(neural_test["selected_epoch"]),
            "seed": int(protocol["seed"]),
            "selection_split": "validation",
            "topics": int(protocol["model"]["num_topics"]),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
        },
        "methods": methods,
        "provenance": {
            "model_bundle_manifest_sha256": file_sha256(
                directory / "model_bundle/manifest.json"
            ),
            "recorded_source_manifest_sha256": {
                name: file_sha256(path) for name, path in paths.items()
            },
            "selected_checkpoint_sha256": neural_training["selected"][
                "checkpoint_sha256"
            ],
            "test_opened_after_validation_selection": True,
        },
        "secondary_diagnostics": {
            "completion_nll_per_token": {
                "neural": {
                    "validation": float(
                        evidence["neural_validation_evaluation"]["metrics"][
                            "validation_document_completion"
                        ]["nll_per_token"]
                    ),
                    "test": float(
                        neural_test["metrics"]["test_document_completion"][
                            "nll_per_token"
                        ]
                    ),
                },
                "tomotopy": {
                    "validation": float(
                        evidence["tomotopy_validation_evaluation"]["metrics"][
                            "validation_document_completion"
                        ]["nll_per_token"]
                    ),
                    "test": float(
                        tomotopy_test["metrics"]["test_document_completion"][
                            "nll_per_token"
                        ]
                    ),
                },
            },
            "neural_recycled_topics_during_training": int(
                neural_training["recycle_count_total"]
            ),
            "neural_test_corpus_active_topics": int(
                neural_test["metrics"]["active_topics"]["corpus_active_topics"]
            ),
            "neural_test_median_effective_topics_per_spectrum": float(
                neural_test["metrics"]["full_spectrum_mixture"][
                    "effective_topic_count_median"
                ]
            ),
        },
        "secondary_warm_in_memory_batch_inference": {
            "batch_size": int(neural_warm["documents"]),
            "cpu_threads": int(neural_warm["cpu_threads"]),
            "neural_routing_passes": 1,
            "neural_spectra_per_second": float(
                neural_warm["median_spectra_per_second"]
            ),
            "speedup_over_tomotopy": float(
                neural_warm["median_spectra_per_second"]
                / tomotopy_warm["median_spectra_per_second"]
            ),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
            "tomotopy_spectra_per_second": float(
                tomotopy_warm["median_spectra_per_second"]
            ),
        },
    }
    write_json(directory / "results.json", result)
    return result
