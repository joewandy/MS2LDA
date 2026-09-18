"""Fit posterior normalization statistics using training spectra only."""

from __future__ import annotations

import numpy as np
import torch

from .prior_etm import PriorETM
from .topic_model_training import dense_normalized


@torch.inference_mode()
def calibrate_posterior_batchnorm(
    model: PriorETM, train, *, batch_size: int, device: torch.device
) -> dict:
    """Replace lagging EMAs with full-training moments at the final weights.

    The ETM encoder contains no internal BatchNorm/dropout, so raw posterior
    head values are independent of chunk composition. Parallel Welford moments
    in float64 avoid cancellation when a head has a large offset and small
    variance. No gradient, model weight or held-out spectrum is used here.
    Evaluation subsequently keeps these statistics frozen.
    """
    layers = (model.mean_normalization, model.variance_normalization)
    if not all(isinstance(layer, torch.nn.BatchNorm1d) for layer in layers):
        raise ValueError("calibration requires two posterior BatchNorm heads")
    if train.shape[0] < 2 or batch_size < 1:
        raise ValueError("calibration needs at least two training rows")
    model.eval()
    means = [
        torch.zeros_like(layer.running_mean, dtype=torch.float64) for layer in layers
    ]
    sums = [torch.zeros_like(mean) for mean in means]
    count = batches = 0
    for start in range(0, train.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, train.shape[0]))
        encoded = model.encoder(dense_normalized(train, rows, device))
        for i, values in enumerate((model.mu(encoded), model.logvar(encoded))):
            variance, mean = torch.var_mean(values.double(), dim=0, correction=0)
            delta = mean - means[i]
            sums[i] += variance * len(rows) + delta.square() * count * len(rows) / (
                count + len(rows)
            )
            means[i] += delta * len(rows) / (count + len(rows))
        count += len(rows)
        batches += 1
    shifts = []
    for layer, mean, squared_sum in zip(layers, means, sums, strict=True):
        variance = squared_sum / (count - 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("non-finite training normalization statistics")
        shifts.append(float((mean - layer.running_mean).abs().max()))
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(variance)
        layer.num_batches_tracked.fill_(batches)
    return {
        "statistics": "full_training_sample_moments",
        "training_rows": count,
        "maximum_head_mean_shifts": shifts,
        "validation_rows_used": 0,
        "test_rows_used": 0,
    }
