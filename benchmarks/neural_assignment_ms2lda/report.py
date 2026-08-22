"""Assemble the paper-facing comparison from verified run manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import file_sha256, read_json, write_json


def _chemistry_summary(chemistry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the motif-inventory and SOS quantities used in the report."""
    high_confidence = chemistry["high_confidence_chemistry"]
    bands = high_confidence["sos_bands"]
    useful = int(bands["high_gt_0_8"] + bands["intermediate_0_6_to_0_8"])
    return {
        "optimized_motifs": int(
            round(float(chemistry["annotation_coverage"]) * chemistry["topics"])
        ),
        "annotation_coverage": float(chemistry["annotation_coverage"]),
        "high_confidence_evaluable_motifs": int(high_confidence["eligible_topics"]),
        "useful_high_confidence_motifs": useful,
        "sos_bands": bands,
        "mean_sos": high_confidence["mean_sos"],
        "median_sos": high_confidence["median_sos"],
    }


def _split_summary(
    evaluation: dict[str, Any], chemistry: dict[str, Any], *, split: str
) -> dict[str, Any]:
    """Combine held-out completion and chemistry for one named split."""
    completion = evaluation["metrics"][f"{split}_document_completion"]
    return {
        "completion_nll_per_token": float(completion["nll_per_token"]),
        **_chemistry_summary(chemistry),
    }


def _method_row(  # noqa: PLR0913
    *,
    method: str,
    validation: dict[str, Any],
    validation_chemistry: dict[str, Any],
    test: dict[str, Any],
    test_chemistry: dict[str, Any],
    fitting_seconds: float,
    fitting_workers: int,
) -> dict[str, Any]:
    """Return one concise, method-agnostic result row."""
    return {
        "method": method,
        "topics": int(test["topic_count"]),
        "fitting_seconds": float(fitting_seconds),
        "fitting_workers": int(fitting_workers),
        "validation": _split_summary(
            validation, validation_chemistry, split="validation"
        ),
        "test": _split_summary(test, test_chemistry, split="test"),
        "warm_in_memory_batch_inference": test["metrics"][
            "warm_in_memory_batch_inference"
        ],
    }


def build_machine_report(run_dir: str | Path) -> dict[str, Any]:
    """Build the final comparison from the complete self-contained run."""
    directory = Path(run_dir).expanduser().resolve()
    protocol = read_json(directory / "protocol.resolved.json")
    neural_training = read_json(directory / "model/complete.json")
    tomotopy_training = read_json(directory / "tomotopy/complete.json")

    paths = {
        "neural_validation": directory / "validation_evaluation/neural/complete.json",
        "tomotopy_validation": directory
        / "validation_evaluation/tomotopy/complete.json",
        "neural_validation_chemistry": directory
        / "validation_chemical/neural/complete.json",
        "tomotopy_validation_chemistry": directory
        / "validation_chemical/tomotopy/complete.json",
        "neural_test": directory / "evaluation/neural/complete.json",
        "tomotopy_test": directory / "evaluation/tomotopy/complete.json",
        "neural_test_chemistry": directory / "chemical/neural/complete.json",
        "tomotopy_test_chemistry": directory / "chemical/tomotopy/complete.json",
    }
    evidence = {name: read_json(path) for name, path in paths.items()}
    methods = [
        _method_row(
            method="neural",
            validation=evidence["neural_validation"],
            validation_chemistry=evidence["neural_validation_chemistry"],
            test=evidence["neural_test"],
            test_chemistry=evidence["neural_test_chemistry"],
            fitting_seconds=float(neural_training["elapsed_seconds"]),
            fitting_workers=int(protocol["cpu_threads"]),
        ),
        _method_row(
            method="tomotopy",
            validation=evidence["tomotopy_validation"],
            validation_chemistry=evidence["tomotopy_validation_chemistry"],
            test=evidence["tomotopy_test"],
            test_chemistry=evidence["tomotopy_test_chemistry"],
            fitting_seconds=float(tomotopy_training["training_seconds_total"]),
            fitting_workers=int(tomotopy_training["training_workers"]),
        ),
    ]
    source_paths = {
        "protocol": directory / "protocol.resolved.json",
        "neural_training": directory / "model/complete.json",
        "tomotopy_training": directory / "tomotopy/complete.json",
        **paths,
    }
    result = {
        "schema_version": "neural-ms2lda/research-report-v1",
        "title": "Neural MS2LDA on MSnLib",
        "evidence_scope": "single-seed applied-method reproducibility checkpoint",
        "comparison_contract": {
            "seed": int(protocol["seed"]),
            "topics": int(protocol["model"]["num_topics"]),
            "cpu_threads": int(protocol["cpu_threads"]),
            "neural_selected_epoch": int(evidence["neural_test"]["selected_epoch"]),
            "tomotopy_inference_iterations": int(
                protocol["tomotopy"]["inference_iterations"]
            ),
            "tomotopy_training_reused": False,
        },
        "methods": methods,
        "training_contract": {
            "unsupervised": True,
            "tomotopy_teacher": False,
            "pretrained_spectrum_encoder": False,
            "variational_bayes": False,
            "chemistry_labels": False,
            "test_information": False,
        },
        "limitations": [
            "single fixed seed",
            "research checkpoint rather than production replacement",
            "warm resident-model throughput compares different native inference algorithms",
        ],
        "source_sha256": {
            name: file_sha256(path) for name, path in source_paths.items()
        },
    }
    write_json(directory / "report/report.json", result)
    return result
