"""Integrity checks for the committed Routing ETM validation checkpoint."""

from pathlib import Path

from scripts.verify_routing_etm_checkpoint import DEFAULT_MANIFEST, verify_checkpoint
from scripts.verify_routing_etm_npmi import verify_npmi_experiment
from scripts.verify_routing_etm_stability import (
    DEFAULT_MANIFEST as DEFAULT_STABILITY_MANIFEST,
)
from scripts.verify_routing_etm_stability import verify_stability_package


def test_committed_routing_etm_checkpoint_is_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = verify_checkpoint(DEFAULT_MANIFEST, repo_root=repo_root)
    assert result["status"] == "passed", result["errors"]
    assert result["checks_completed"] >= 50


def test_committed_routing_etm_stability_package_is_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = verify_stability_package(
        DEFAULT_STABILITY_MANIFEST,
        repo_root=repo_root,
    )
    assert result["status"] == "passed", result["errors"]
    assert result["checks_completed"] >= 90


def test_committed_routing_etm_npmi_stopping_decision_is_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = verify_npmi_experiment(repo_root=repo_root)
    assert result["status"] == "passed", result["errors"]
    assert result["stopping_decision"] == "failed synthetic triage; no real promotion"
