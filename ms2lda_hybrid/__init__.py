"""Isolated research reference for DreaMS-conditioned variational LDA."""

from .config import HybridLDAConfig
from .dreams_features import (
    DreaMSFeatureBatch,
    DreaMSFeatureExtractor,
    pool_word_embeddings,
)
from .model import HybridDocument, HybridLDAModel

__all__ = [
    "DreaMSFeatureBatch",
    "DreaMSFeatureExtractor",
    "HybridDocument",
    "HybridLDAConfig",
    "HybridLDAModel",
    "pool_word_embeddings",
]
