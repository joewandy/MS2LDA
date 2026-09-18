"""Audit exact model reductions without fitting or accessing test matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.neural_ms2lda.contextual_sparse_etm import (
    ContextualSparseETM,
    centered_log_evidence_offset,
    diagonal_gaussian_kl,
    entmax15_document_mixture,
    leave_one_out_context,
    unit_normalize_rows,
)
from benchmarks.neural_ms2lda.reproducibility import sha256_file
from benchmarks.neural_ms2lda.topic_model_training import dense_normalized
from benchmarks.neural_ms2lda.utils import write_json
from scripts.run_etm_controls import load_control_data


@torch.inference_mode()
def audit(run: Path, output: Path):
    """Check formula identities on every validation row of a frozen checkpoint."""
    torch.set_num_threads(4)
    data = load_control_data(run.resolve())
    checkpoint_path = run / "models/minimal_etm/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["variant"] != "contextual" or config["test_matrices_loaded"]:
        raise ValueError("audit requires the paired validation-only current model")
    for name, expected in config["input_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("frozen audit inputs changed")
    model = ContextualSparseETM(
        data.embeddings,
        config["topics"],
        np.asarray([w.startswith("frag@") for w in data.vocabulary]),
        hidden=config["hidden"],
    ).double()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    errors = {
        key: 0.0
        for key in (
            "log_rewrite",
            "theta_gauge",
            "kl_identity",
            "channel_identity",
            "compact_loo_direction",
        )
    }
    penalties = []
    original_kl = []
    k = config["topics"]
    mask = model.fragment_mask
    logits = model.alphas.weight @ model.rho.T
    conditional = torch.empty_like(logits)
    for selected in (mask, ~mask):
        conditional[:, selected] = torch.softmax(logits[:, selected], dim=1)
    beta = model.topic_word_distribution()
    for start in range(0, data.full.shape[0], 200):
        rows = np.arange(start, min(start + 200, data.full.shape[0]))
        x = dense_normalized(data.full, rows, torch.device("cpu")).double()
        rho_hat = unit_normalize_rows(model.rho)
        documents, words = torch.nonzero(x, as_tuple=True)
        weights = x[documents, words]
        original_h = unit_normalize_rows(
            rho_hat[words]
            + model.context_scale * leave_one_out_context(x, rho_hat, documents, words)
        )
        compact_h = unit_normalize_rows(
            (1 - (1 + model.context_scale) * weights[:, None]) * rho_hat[words]
            + model.context_scale * (x @ rho_hat)[documents]
        )
        compact_h = torch.where((weights == 1)[:, None], rho_hat[words], compact_h)
        errors["compact_loo_direction"] = max(
            errors["compact_loo_direction"], (original_h - compact_h).abs().max().item()
        )
        r = model.contextual_evidence(x)
        offset = centered_log_evidence_offset(r)
        simple = torch.log1p(k * r)
        simple -= simple.mean(1, keepdim=True)
        errors["log_rewrite"] = max(
            errors["log_rewrite"], (simple - offset).abs().max().item()
        )
        mean, logvar, kl = model.posterior(x)
        centred_mean = mean - mean.mean(1, keepdim=True)
        theta = entmax15_document_mixture(mean)
        projected = entmax15_document_mixture(centred_mean)
        errors["theta_gauge"] = max(
            errors["theta_gauge"], (theta - projected).abs().max().item()
        )
        penalty = 0.5 * k * mean.mean(1).square()
        difference = kl - diagonal_gaussian_kl(centred_mean, logvar)
        errors["kl_identity"] = max(
            errors["kl_identity"], (difference - penalty).abs().max().item()
        )
        errors["channel_identity"] = max(
            errors["channel_identity"],
            ((theta @ beta).log() - ((theta @ conditional).log() - np.log(2)))
            .abs()
            .max()
            .item(),
        )
        penalties.extend(penalty.tolist())
        original_kl.extend(kl.tolist())
    if max(errors.values()) > 1e-8:
        raise AssertionError(f"mathematical identity failed: {errors}")
    result = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "script_sha256": sha256_file(Path(__file__)),
        "source_code_sha256": config["code_sha256"],
        "input_sha256": config["input_sha256"],
        "validation_rows": data.full.shape[0],
        "test_matrices_loaded": False,
        "precision": "float64 audit of float32-trained weights; no new fit",
        "maximum_absolute_errors": errors,
        "mean_gaussian_kl": float(np.mean(original_kl)),
        "mean_removable_common_mode_kl": float(np.mean(penalties)),
        "maximum_removable_common_mode_kl": float(np.max(penalties)),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.run, args.output)


if __name__ == "__main__":
    main()
