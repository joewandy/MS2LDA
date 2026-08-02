"""Validated configuration for the hybrid LDA reference model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class HybridLDAConfig:
    """Scientific and optimization settings for the reference model.

    The neural architecture is intentionally fixed to the two-hidden-layer
    network described in the method paper. Only settings that are useful for
    fitting or controlled experiments remain configurable.
    """

    # LDA and input dimensions.
    num_topics: int
    embedding_dim: int
    alpha: float | tuple[float, ...] = 0.1
    eta: float = 0.01

    # Local VB and the final semi-amortized document encoder.
    hidden_size: int = 256
    feature_projection_dim: int = 128
    training_local_steps: int = 50
    batch_size: int = 128
    encoder_learning_rate: float = 1e-3
    inference_epochs: int = 12

    # Empirical-Bayes topic-word prior.
    prior_mass_fraction: float = 0.05
    prior_warmup_epochs: int = 15
    prior_training_epochs: int = 20
    prior_temperature: float = 0.5
    prior_learning_rate: float = 1e-3
    topic_diversity_weight: float = 1e-3

    # Local and global stopping rules.
    local_tolerance: float = 1e-4
    global_tolerance: float = 1e-3
    global_patience: int = 3
    max_epochs: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject configurations that would make the algorithm ill-defined."""
        positive_integers = {
            "num_topics": self.num_topics,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "feature_projection_dim": self.feature_projection_dim,
            "training_local_steps": self.training_local_steps,
            "batch_size": self.batch_size,
            "inference_epochs": self.inference_epochs,
            "prior_warmup_epochs": self.prior_warmup_epochs,
            "prior_training_epochs": self.prior_training_epochs,
            "global_patience": self.global_patience,
            "max_epochs": self.max_epochs,
        }
        invalid_integers = [
            name
            for name, value in positive_integers.items()
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ]
        if invalid_integers:
            names = ", ".join(invalid_integers)
            raise ValueError(f"positive integers required for: {names}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("seed must be an integer between 0 and 2**64 - 1")

        finite_settings = {
            "eta": self.eta,
            "encoder_learning_rate": self.encoder_learning_rate,
            "prior_mass_fraction": self.prior_mass_fraction,
            "prior_temperature": self.prior_temperature,
            "prior_learning_rate": self.prior_learning_rate,
            "topic_diversity_weight": self.topic_diversity_weight,
            "local_tolerance": self.local_tolerance,
            "global_tolerance": self.global_tolerance,
        }
        nonfinite = [
            name for name, value in finite_settings.items() if not np.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(f"finite values required for: {', '.join(nonfinite)}")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.encoder_learning_rate <= 0 or self.prior_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive")
        if self.local_tolerance <= 0 or self.global_tolerance <= 0:
            raise ValueError("convergence tolerances must be positive")
        if not 0 <= self.prior_mass_fraction <= 1:
            raise ValueError("prior_mass_fraction must lie between zero and one")
        if self.topic_diversity_weight < 0:
            raise ValueError("topic_diversity_weight cannot be negative")
        if self.prior_training_epochs < self.prior_warmup_epochs:
            raise ValueError("prior training must cover the prior warmup")
        if self.max_epochs <= self.prior_training_epochs:
            raise ValueError("max_epochs must include at least one fixed-prior epoch")
        self.alpha_vector()

    def alpha_vector(self) -> np.ndarray:
        """Return one positive alpha value per topic."""
        values = np.asarray(self.alpha, dtype=np.float32)
        if values.ndim == 0:
            values = np.repeat(values, self.num_topics)
        if (
            values.shape != (self.num_topics,)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError("alpha must be positive and scalar or one value per topic")
        return values
