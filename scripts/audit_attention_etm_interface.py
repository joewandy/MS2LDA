"""Verify MLP-free checkpoint compatibility on validation data, without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.neural_ms2lda.attention_etm import AttentionETM
from benchmarks.neural_ms2lda.contextual_reductions import ReducedContextualETM
from benchmarks.neural_ms2lda.model_evaluation import completion_metrics
from benchmarks.neural_ms2lda.reproducibility import sha256_file
from benchmarks.neural_ms2lda.utils import write_json
from scripts.run_etm_controls import load_control_data
from scripts.run_minimal_etm import infer


@torch.inference_mode()
def audit(run: Path, output: Path) -> dict:
    """Check the whole frozen validation set, never loading a test matrix."""
    torch.set_num_threads(4)
    run = run.resolve()
    data = load_control_data(run)
    path = run / "models/minimal_etm/checkpoint.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    if (
        config["variant"] != "reduced_evidence_only"
        or config["test_matrices_loaded"]
        or config["selection_split"] != "validation"
    ):
        raise ValueError("need a frozen evidence-only validation checkpoint")
    for name, expected in config["input_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("frozen checkpoint inputs changed")
    mask = np.asarray([word.startswith("frag@") for word in data.vocabulary])
    old = ReducedContextualETM(
        data.embeddings,
        config["topics"],
        mask,
        hidden=config["hidden"],
        variant="reduced_evidence_only",
    ).eval()
    old.load_state_dict(saved["state_dict"], strict=True)
    new = AttentionETM.from_evidence_only_state(saved["state_dict"], mask).eval()
    state = {key: value.clone() for key, value in new.state_dict().items()}
    theta = {}
    for name, matrix in (("full", data.full), ("observed", data.observed)):
        previous = infer(old, matrix, batch_size=200, device=torch.device("cpu"))
        theta[name] = infer(new, matrix, batch_size=200, device=torch.device("cpu"))
        np.testing.assert_array_equal(previous, theta[name])
    beta = new.topic_word_distribution().numpy()
    np.testing.assert_array_equal(old.topic_word_distribution().numpy(), beta)
    nll = completion_metrics(theta["observed"], beta, data.completion, data.records)[
        "nll_per_token"
    ]
    raw = json.loads((run / "models/minimal_etm/result.json").read_text())
    saved_theta = np.load(
        run / "validation_evaluation/minimal_etm/validation_full_theta.npy"
    )
    theta_error = float(np.max(np.abs(theta["full"] - saved_theta)))
    nll_error = abs(nll - raw["metrics"]["completion"]["nll_per_token"])
    if theta_error > 1e-5 or nll_error > 1e-6:
        raise ValueError("CPU shell disagrees with frozen GPU evaluation")
    for key, value in state.items():
        torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
    result = {
        "variant": config["variant"],
        "seed": config["seed"],
        "checkpoint_sha256": sha256_file(path),
        "validation_rows": len(data.records),
        "test_matrices_loaded": False,
        "fitted_new_model": False,
        "foundation_model_used": False,
        "same_device_theta_and_beta_bit_identical": True,
        "cpu_vs_saved_gpu_max_theta_error": theta_error,
        "cpu_vs_saved_gpu_nll_absolute_error": nll_error,
        "frozen_state_unchanged": True,
        "trainable_parameters": sum(p.numel() for p in new.parameters()),
        "fixed_embedding_coordinates": new.rho.numel(),
        "channel_mask_serialized": "fragment_mask" in new.state_dict(),
        "encoder_contract": "opaque observations -> Gaussian mean and log variance",
        "decoder": "unchanged channel-balanced token-emission ETM",
        "source_sha256": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__).relative_to(Path(__file__).resolve().parents[1]),
                Path("benchmarks/neural_ms2lda/attention_etm.py"),
                Path("benchmarks/neural_ms2lda/tests/test_attention_etm.py"),
            )
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.run, arguments.output), indent=2))


if __name__ == "__main__":
    main()
