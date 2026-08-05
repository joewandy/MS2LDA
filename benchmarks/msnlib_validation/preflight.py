"""Paper-scale resource preflight without chemical result evaluation."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    BenchmarkConfig,
    environment_manifest,
    object_sha256,
    resolve_input_paths,
    write_json,
)
from .data import assign_scaffold_splits, build_training_vocabulary, load_records


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _allocation_probe(
    num_topics: int, vocabulary_size: int, padded_words: int, batch: int
) -> float:
    """Allocate representative float32 workspaces, then release them."""
    import torch

    started = time.perf_counter()
    matrices = [
        torch.empty((num_topics, vocabulary_size), dtype=torch.float32)
        for _ in range(5)
    ]
    local = torch.empty((batch, padded_words, num_topics), dtype=torch.float32)
    gamma = torch.empty((batch, num_topics), dtype=torch.float32)
    matrices[0].zero_()
    local.zero_()
    gamma.zero_()
    del gamma, local, matrices
    return time.perf_counter() - started


def run_preflight(
    config: BenchmarkConfig,
    *,
    data_root: str | Path,
    output_path: str | Path | None = None,
    allocation_probe: bool = True,
) -> dict[str, Any]:
    """Inspect the exact full corpus and test K=1000 tensor allocation only."""
    inputs = resolve_input_paths(config, data_root)
    records, data_summary = load_records(inputs["mgf"], config)
    assignments, split_summary = assign_scaffold_splits(
        records, fractions=config.split_fractions, seed=config.split_seed
    )
    vocabulary, vocabulary_summary = build_training_vocabulary(
        records,
        assignments,
        min_df=config.min_df,
        min_cf=config.min_cf,
        rm_top=config.rm_top,
    )
    vocabulary_set = set(vocabulary)
    train_widths = [
        len({word for word in record.words if word in vocabulary_set})
        for record in records
        if assignments[record.spectrum_id] == "train"
    ]
    padded_words = max(train_widths)
    batch = min(config.hybrid_batch_size, len(train_widths))
    float_bytes = np.dtype(np.float32).itemsize
    global_workspace = 5 * config.num_topics * len(vocabulary) * float_bytes
    local_workspace = batch * padded_words * config.num_topics * float_bytes
    conservative_peak = 2 * global_workspace + 3 * local_workspace
    physical_memory = _physical_memory_bytes()
    free_disk = shutil.disk_usage(Path(data_root).expanduser().resolve()).free
    checkpoint_bytes = config.hybrid_checkpoint_keep * (
        global_workspace
        + split_summary["spectrum_counts"]["train"] * config.num_topics * float_bytes
    )
    disk_estimate = (
        len(records) * 1024 * float_bytes
        + len(vocabulary) * 1024 * float_bytes
        + len(config.seeds) * 2 * config.num_topics * len(vocabulary) * float_bytes
        + len(config.seeds) * checkpoint_bytes
    )
    probe_seconds = None
    if allocation_probe:
        probe_seconds = _allocation_probe(
            config.num_topics, len(vocabulary), padded_words, batch
        )
    memory_headroom = (
        None if physical_memory is None else conservative_peak / max(physical_memory, 1)
    )
    result = {
        "mode": "resource_preflight",
        "config_sha256": object_sha256(config.as_dict()),
        "software_validation_only": True,
        "chemical_evidence": False,
        "test_set_metrics_inspected": False,
        "full_input_spectra": config.expected_spectra,
        "retained_spectra": len(records),
        "training_spectra": split_summary["spectrum_counts"]["train"],
        "num_topics": config.num_topics,
        "vocabulary_size": len(vocabulary),
        "maximum_training_unique_words": padded_words,
        "hybrid_batch_size": batch,
        "hybrid_training_cpu_threads": config.hybrid_training_cpu_threads,
        "hybrid_inference_cpu_threads": config.hybrid_inference_cpu_threads,
        "hybrid_checkpoint_keep": config.hybrid_checkpoint_keep,
        "estimated_hybrid_checkpoint_bytes_per_seed": checkpoint_bytes,
        "estimated_global_workspace_bytes": global_workspace,
        "estimated_local_workspace_bytes": local_workspace,
        "conservative_peak_workspace_bytes": conservative_peak,
        "physical_memory_bytes": physical_memory,
        "estimated_peak_fraction_of_physical_memory": memory_headroom,
        "estimated_generated_disk_bytes": disk_estimate,
        "free_disk_bytes": free_disk,
        "allocation_probe_performed": allocation_probe,
        "allocation_probe_seconds": probe_seconds,
        "memory_preflight_passed": memory_headroom is None or memory_headroom < 0.8,
        "disk_preflight_passed": disk_estimate < 0.8 * free_disk,
        "data_summary": data_summary,
        "split_summary": split_summary,
        "vocabulary_summary": vocabulary_summary,
        "environment": environment_manifest(),
    }
    result["passed"] = bool(
        result["memory_preflight_passed"] and result["disk_preflight_passed"]
    )
    if output_path is not None:
        write_json(output_path, result)
    return result
