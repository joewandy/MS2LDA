"""Verify the stopped zero-parameter top-2 token Routing ETM experiment."""

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
    "research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_top2_token",
)
PREDECLARATION_COMMIT = "18290f2614c928240b5aeccd33e101bba604e81b"


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


def verify_top2_token_experiment(  # noqa: PLR0915
    *,
    repo_root: Path | None = None,
    require_external: bool = False,
) -> dict[str, Any]:
    """Return integrity and stopping-rule checks for the simplification."""
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
        expected={"entmax_etm", "top2_token_entmax", "top2_context_entmax"},
    )
    candidate = rows["top2_token_entmax"]
    context = rows["top2_context_entmax"]
    entmax = rows["entmax_etm"]

    check(
        "config predeclaration",
        config["predeclaration_commit"],
        expected=PREDECLARATION_COMMIT,
    )
    check(
        "provenance predeclaration",
        provenance["predeclaration_commit"],
        expected=PREDECLARATION_COMMIT,
    )
    check("routing variant", config["routing_variant"], expected="top2_token")
    check(
        "zero-parameter requirement",
        config["zero_parameter_requirement"],
        expected=True,
    )
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

    expected_candidate = {
        "heldout_nll": 6.343999825860643,
        "true_beta_cosine": 0.41035377979278564,
        "true_theta_cosine": 0.6614250649456055,
        "planted_recovered_ge_0_50": "6",
        "active_topics_gt_0_005": "11",
        "unique_top1_topics": "10",
        "median_effective_topics": 2.005119164831524,
        "median_exact_support": 4.0,
        "parameters": "2167400",
        "learned_context_scale": "",
        "finite_stable": "true",
        "catastrophic_duplicate_component": "false",
        "triage_status": "failed",
    }
    for field, expected in expected_candidate.items():
        check(f"candidate {field}", candidate[field], expected=expected)

    check("context beta", context["true_beta_cosine"], expected=0.4984537661075592)
    check("context theta", context["true_theta_cosine"], expected=0.7648745037831431)
    check("context NLL", context["heldout_nll"], expected=6.278416276690836)
    check("context parameters", context["parameters"], expected="2167401")
    check("entmax parameters", entmax["parameters"], expected="2167400")
    check(
        "candidate matches ETM parameter count",
        int(candidate["parameters"]) == int(entmax["parameters"]),
        expected=True,
    )
    check(
        "context adds one parameter",
        int(context["parameters"]) - int(candidate["parameters"]),
        expected=1,
    )
    check(
        "top-2 token improves beta over entmax",
        float(candidate["true_beta_cosine"]) > float(entmax["true_beta_cosine"]),
        expected=True,
    )
    check(
        "top-2 token improves theta over entmax",
        float(candidate["true_theta_cosine"]) > float(entmax["true_theta_cosine"]),
        expected=True,
    )
    check(
        "candidate fails beta non-inferiority",
        float(context["true_beta_cosine"]) - float(candidate["true_beta_cosine"])
        > float(config["triage_thresholds"]["beta_maximum_loss"]),
        expected=True,
    )
    check(
        "candidate fails theta non-inferiority",
        float(context["true_theta_cosine"]) - float(candidate["true_theta_cosine"])
        > float(config["triage_thresholds"]["theta_maximum_loss"]),
        expected=True,
    )
    check(
        "candidate fails NLL non-inferiority",
        float(candidate["heldout_nll"]) / float(context["heldout_nll"])
        > 1.0 + float(config["triage_thresholds"]["nll_maximum_relative_worsening"]),
        expected=True,
    )
    check(
        "candidate fails recovered-motif minimum",
        int(candidate["planted_recovered_ge_0_50"])
        < int(config["triage_thresholds"]["minimum_recovered_motifs"]),
        expected=True,
    )

    external_checked = 0
    for group in ("candidate", "context_control", "entmax_control"):
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
        "stopping_decision": "failed synthetic triage; context retained",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the committed summary and optionally require local artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-external", action="store_true")
    args = parser.parse_args(argv)
    result = verify_top2_token_experiment(require_external=args.require_external)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
