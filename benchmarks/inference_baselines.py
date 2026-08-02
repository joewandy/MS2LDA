"""Benchmark-only inference baselines kept outside the production model.

The supported :class:`ms2lda_hybrid.HybridLDAModel` exposes only its
semi-amortized ELBO finalization. This module contains the posterior-regression
comparator needed to reproduce objective studies without expanding the public
model API or allowing a comparator checkpoint to masquerade as a finalized
production model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ms2lda_hybrid import HybridLDAModel
from ms2lda_hybrid._variational import (
    EPSILON,
    expected_log_dirichlet,
    make_sparse_batch,
)


def fit_posterior_regression_baseline(
    model: HybridLDAModel,
    *,
    epochs: int,
    target_steps: int,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Fit an equal-epoch normalized-posterior regression comparator.

    This function is deliberately benchmark-only. It computes fixed local-VB
    targets from the frozen topic posterior, minimizes categorical KL between
    the target and encoder posterior means, and verifies that neither topics
    nor structured-prior parameters change. It does not mark ``model`` as
    finalized and must not be used to create production inference artifacts.

    Parameters
    ----------
    model
        Prepared model whose topic posterior is treated as immutable.
    epochs
        Number of full random-reshuffling passes over training documents.
    target_steps
        Number of classical local-VB updates used to construct the fixed
        target posterior means.
    progress_callback
        Optional callback receiving a defensive copy of each epoch record.

    Returns
    -------
    list of dict
        Per-epoch categorical-KL and gradient-norm diagnostics.
    """
    if epochs < 1 or target_steps < 1:
        raise ValueError("baseline epochs and target steps must be positive")
    if model._core is None or model._matrix is None or model._epochs < 1:
        raise RuntimeError("baseline requires a prepared frozen-topic model")
    if model.inference_finalized:
        raise RuntimeError("baseline must start before supported finalization")

    core = model._core
    lambda_snapshot = core.lambda_posterior.detach().clone()
    prior_snapshot = [
        parameter.detach().clone() for parameter in core.prior_parameters()
    ]
    expected_log_beta = expected_log_dirichlet(lambda_snapshot).detach()
    word_topic = torch.softmax(expected_log_beta.transpose(0, 1), dim=1).detach()
    target_posteriors = model._refine_training_posteriors(steps=target_steps)
    optimizer = torch.optim.Adam(
        core.encoder_parameters(),
        lr=model.config.encoder_learning_rate,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        core.train()
        shuffled = model._rng.permutation(len(model.docs))
        kl_sum = 0.0
        gradient_norm_sum = 0.0
        document_sum = 0
        batches = 0
        for start in range(0, len(model.docs), model.config.batch_size):
            indices = shuffled[start : start + model.config.batch_size]
            batch = make_sparse_batch(model._matrix, indices, device=model.device)
            predicted_gamma = core.encode(
                batch,
                model._embedding_batch(model.docs, indices),
                word_topic,
            )
            target_gamma = torch.as_tensor(
                target_posteriors[indices],
                device=model.device,
                dtype=predicted_gamma.dtype,
            )
            target_theta = target_gamma / target_gamma.sum(dim=1, keepdim=True)
            predicted_theta = predicted_gamma / predicted_gamma.sum(
                dim=1,
                keepdim=True,
            )
            document_kl = (
                target_theta
                * (
                    target_theta.clamp_min(EPSILON).log()
                    - predicted_theta.clamp_min(EPSILON).log()
                )
            ).sum(dim=1)
            loss = document_kl.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                core.encoder_parameters(),
                10.0,
            )
            optimizer.step()

            kl_sum += float(document_kl.sum().detach().cpu())
            gradient_norm_sum += float(gradient_norm.detach().cpu())
            document_sum += len(indices)
            batches += 1
        metrics = {
            "inference_epoch": float(epoch),
            "loss": kl_sum / max(document_sum, 1),
            "posterior_mean_kl": kl_sum / max(document_sum, 1),
            "encoder_gradient_norm": gradient_norm_sum / max(batches, 1),
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(dict(metrics))

    if not torch.equal(lambda_snapshot, core.lambda_posterior):
        raise RuntimeError("posterior-regression baseline changed frozen topics")
    if any(
        not torch.equal(previous, current)
        for previous, current in zip(
            prior_snapshot,
            core.prior_parameters(),
            strict=True,
        )
    ):
        raise RuntimeError("posterior-regression baseline changed the frozen prior")
    return history
