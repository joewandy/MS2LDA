"""Reject incomplete or silently reinterpreted published-model evidence."""

from copy import deepcopy

import pytest

from scripts.generate_published_neural_review import (
    LABELS,
    synthetic_groups,
    validate_recipe,
)


def row(variant="prodlda", topics=36, seed=11):
    product = variant in ("prodlda", "prodlda_entmax", "dvae_poe")
    return {
        "variant": variant,
        "topics": topics,
        "seed": seed,
        "data": "synthetic",
        "epochs": 120,
        "batch_size": 200,
        "hidden": 100,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0,
        "kl_warmup_epochs": 0 if variant.startswith("prodlda") else 100,
        "concentration": 0.02,
        "normalization_statistics": "ema",
        "count_scaling": "raw_counts",
        "framework": "pytorch_port_not_pyro_svi",
        "decoder": "product_of_experts" if product else "additive_mixture",
        "beta_meaning": "one_hot_conditional_prototypes" if product else "emissions",
        "reused_training_weights": False,
    }


@pytest.mark.parametrize("variant", LABELS)
def test_expected_recipe_accepted(variant):
    validate_recipe(row(variant))


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden", 800),
        ("learning_rate", 0.005),
        ("epochs", 100),
        ("normalization_statistics", "validation"),
        ("decoder", "additive_mixture"),
        ("beta_meaning", "emissions"),
        ("framework", "pyro_svi"),
        ("reused_training_weights", True),
    ],
)
def test_recipe_and_meaning_drift_rejected(field, value):
    candidate = row()
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_recipe(candidate)


def test_all_models_all_seeds_including_negative_results_required():
    rows = [row(v, k, s) for v in LABELS for k in (36, 128) for s in (11, 23, 37)]
    assert len(synthetic_groups(rows)) == 10
    with pytest.raises(ValueError):
        synthetic_groups(rows[:-1])
    duplicate = deepcopy(rows)
    duplicate[-1] = deepcopy(duplicate[-2])
    with pytest.raises(ValueError):
        synthetic_groups(duplicate)
