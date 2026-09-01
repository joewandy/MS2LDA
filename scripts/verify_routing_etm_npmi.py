"""Verify the stopped Routing ETM positive-NPMI synthetic experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

PACKAGE = Path(
    "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_npmi",
)
BETA_PROMOTION_DELTA = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        message = f"expected JSON object: {path}"
        raise TypeError(message)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_npmi_experiment(
    *,
    repo_root: Path | None = None,
    require_external: bool = False,
) -> dict[str, Any]:
    """Return integrity and stopping-rule checks for the negative result."""
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    package = root / PACKAGE
    config = _read_json(package / "config.json")
    provenance = _read_json(package / "provenance.json")
    with (package / "synthetic_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = {row["model"]: row for row in csv.DictReader(handle)}

    errors: list[str] = []
    checks = 0

    def check(label: str, actual: object, *, expected: object) -> None:
        nonlocal checks
        checks += 1
        if isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{label}: expected {expected!r}, found {actual!r}")

    check(
        "summary models",
        set(rows),
        expected={"routing_etm", "routing_etm_positive_npmi"},
    )
    control = rows["routing_etm"]
    candidate = rows["routing_etm_positive_npmi"]
    check("NPMI weight", config["positive_npmi"]["weight"], expected=1.0)
    check(
        "config test access",
        config["candidate_test_artifacts_accessed"],
        expected=False,
    )
    check(
        "provenance test access",
        provenance["candidate_test_artifacts_accessed"],
        expected=False,
    )
    check(
        "provenance test metrics",
        provenance["candidate_test_metrics_inspected"],
        expected=False,
    )
    check(
        "control beta",
        control["true_beta_cosine"],
        expected=0.4984537661075592,
    )
    check(
        "candidate beta",
        candidate["true_beta_cosine"],
        expected=0.49157607555389404,
    )
    check(
        "control graph loss",
        control["train_graph_loss"],
        expected=5.526061534881592,
    )
    check(
        "candidate graph loss",
        candidate["train_graph_loss"],
        expected=5.499502182006836,
    )
    check(
        "candidate improves graph objective",
        float(candidate["train_graph_loss"]) < float(control["train_graph_loss"]),
        expected=True,
    )
    check(
        "candidate fails beta promotion",
        float(candidate["true_beta_cosine"]) - float(control["true_beta_cosine"])
        < BETA_PROMOTION_DELTA,
        expected=True,
    )
    check("candidate promoted", candidate["promoted"], expected="false")

    external_checked = 0
    for group in ("candidate", "control"):
        run = Path(str(provenance[f"{group}_run"]))
        for artifact in provenance[f"{group}_artifacts"]:
            path = run / str(artifact["path"])
            if not path.is_file():
                if require_external:
                    errors.append(f"missing {group} artifact: {path}")
                continue
            external_checked += 1
            check(
                f"{group} {path.name} bytes",
                path.stat().st_size,
                expected=artifact["bytes"],
            )
            check(
                f"{group} {path.name} sha256",
                _sha256(path),
                expected=artifact["sha256"],
            )

    return {
        "candidate_test_remained_locked": True,
        "checks_completed": checks,
        "errors": errors,
        "external_artifacts_checked": external_checked,
        "external_artifacts_required": require_external,
        "status": "passed" if not errors else "failed",
        "stopping_decision": "failed synthetic triage; no real promotion",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the committed summary and optionally require local artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-external", action="store_true")
    args = parser.parse_args(argv)
    result = verify_npmi_experiment(require_external=args.require_external)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
