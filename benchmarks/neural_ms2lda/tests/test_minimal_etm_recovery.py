"""Recovery restores the exact optimizer, shuffle and sampling trajectory."""

import argparse

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.neural_ms2lda.contextual_sparse_etm import ContextualSparseETM
from benchmarks.neural_ms2lda.reproducibility import configure_deterministic_execution
from benchmarks.neural_ms2lda.utils import write_json
from scripts.run_minimal_etm import fit


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_recovery_and_checkpoint_io_are_numerically_transparent(tmp_path, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    configure_deterministic_execution(321, 1)
    device = torch.device(device_name)
    rng = np.random.default_rng(9)
    rho = rng.normal(size=(17, 6)).astype(np.float32)
    rho /= np.linalg.norm(rho, axis=1, keepdims=True)
    train = sp.csr_matrix(rng.integers(0, 6, size=(12, 17)).astype(np.float32))
    mask = np.arange(17) < 8
    args = argparse.Namespace(
        learning_rate=0.005,
        momentum=0.9,
        seed=11,
        epochs=6,
        batch_size=4,
        count_scaling="raw_counts",
        variant="contextual",
        checkpoint_interval=2,
        output=tmp_path,
    )
    write_json(tmp_path / "config.json", {"test_recipe": "six epochs"})
    torch.manual_seed(321)
    complete = ContextualSparseETM(rho, 9, mask, hidden=11).to(device)
    full_history, _ = fit(complete, train, args, device)
    recovery = torch.load(tmp_path / "training_recovery_epoch002.pt", weights_only=True)
    torch.manual_seed(999)
    resumed = ContextualSparseETM(rho, 9, mask, hidden=11).to(device)
    resumed_history, _ = fit(resumed, train, args, device, recovery=recovery)
    assert full_history == resumed_history
    args.checkpoint_interval = 0
    torch.manual_seed(321)
    unsaved = ContextualSparseETM(rho, 9, mask, hidden=11).to(device)
    unsaved_history, _ = fit(unsaved, train, args, device)
    assert full_history == unsaved_history
    for name, value in complete.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(value, unsaved.state_dict()[name], atol=0, rtol=0)
