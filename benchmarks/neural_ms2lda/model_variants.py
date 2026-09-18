"""Explicit ETM ablation construction, independent of command-line runners."""

from __future__ import annotations

import numpy as np

from .contextual_reductions import REDUCTION_VARIANTS, ReducedContextualETM
from .contextual_sparse_etm import ContextualSparseETM
from .etm_baselines import CanonicalETM, ChannelBalancedETM
from .prior_etm import PriorETM
from .simplified_etm import BatchNormSparseETM, UnbalancedContextualETM

VARIANTS = (
    "etm",
    "balanced",
    "contextual",
    "batchnorm",
    "prior",
    "prior_batchnorm",
    "batchnorm_fixed",
    "prior_batchnorm_fixed",
    "batchnorm_entmax",
    "batchnorm_fixed_entmax",
    "contextual_unbalanced",
    *REDUCTION_VARIANTS,
)


def build_model(
    variant: str,
    embeddings: np.ndarray,
    vocabulary: list[str] | tuple[str, ...],
    topics: int,
    *,
    hidden: int,
    concentration: float,
) -> CanonicalETM | ContextualSparseETM:
    """Construct a named ablation without changing the historical models."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    mask = np.asarray([word.startswith("frag@") for word in vocabulary])
    if variant in REDUCTION_VARIANTS:
        return ReducedContextualETM(
            embeddings, topics, mask, hidden=hidden, variant=variant
        )
    if variant == "etm":
        return CanonicalETM(embeddings, topics, hidden=hidden)
    if variant == "balanced":
        return ChannelBalancedETM(embeddings, topics, mask, hidden=hidden)
    if variant == "contextual":
        return ContextualSparseETM(embeddings, topics, mask, hidden=hidden)
    if variant == "contextual_unbalanced":
        return UnbalancedContextualETM(embeddings, topics, mask, hidden=hidden)
    if variant in ("batchnorm_entmax", "batchnorm_fixed_entmax"):
        return BatchNormSparseETM(
            embeddings,
            topics,
            hidden=hidden,
            learn_normalization_scale=variant == "batchnorm_entmax",
        )
    return PriorETM(
        embeddings,
        topics,
        hidden=hidden,
        concentration=concentration if variant.startswith("prior") else None,
        batch_normalize=variant != "prior",
        learn_normalization_scale=not variant.endswith("_fixed"),
    )
