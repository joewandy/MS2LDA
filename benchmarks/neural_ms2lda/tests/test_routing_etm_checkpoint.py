"""Integrity checks for the committed Routing ETM validation checkpoint."""

from pathlib import Path

from scripts.verify_routing_etm_checkpoint import DEFAULT_MANIFEST, verify_checkpoint


def test_committed_routing_etm_checkpoint_is_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = verify_checkpoint(DEFAULT_MANIFEST, repo_root=repo_root)
    assert result["status"] == "passed", result["errors"]
    assert result["checks_completed"] >= 50
