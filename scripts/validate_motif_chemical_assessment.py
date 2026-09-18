"""Audit the saved assessment against a full replay and independent BY adjustment.

This is a validation command, not a model-fitting or result-selection step.
The original input seals are never rewritten. The protocol mirror is made
byte-identical to its already sealed source only after semantic/hash checks;
this corrects the initial mirror's JSON formatting, not any scientific setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control

from benchmarks.neural_ms2lda.chemical_assessment import source_hashes, verify_stage
from benchmarks.neural_ms2lda.chemical_assessment_io import read_jsonl
from benchmarks.neural_ms2lda.utils import read_json, sha256_file, write_json

ROUNDOFF_TOLERANCE = 1e-12


def verify_protocol_mirrors(evidence: Path, replay: Path, protocol: Path) -> None:
    """Restore byte-preserving mirrors without changing either original seal."""
    for directory in (evidence, replay):
        seal = read_json(directory / "input_seal.json")
        if sha256_file(protocol) != seal["protocol_sha256"]:
            msg = "original sealed protocol changed"
            raise ValueError(msg)
        if read_json(protocol) != read_json(directory / "protocol.json"):
            msg = "protocol mirror settings differ"
            raise ValueError(msg)
        (directory / "protocol.json").write_bytes(protocol.read_bytes())


def compare_summaries(evidence: Path, replay: Path) -> dict:
    """No displayed scientific summary may change in the full replay."""
    stages = ("specificity", "stability", "references")
    summaries = {}
    for stage in stages:
        original = verify_stage(evidence, stage)
        reproduced = verify_stage(replay, stage)
        if original != reproduced:
            msg = f"replay changed {stage} summary"
            raise ValueError(msg)
        summaries[stage] = True
    return summaries


def reference_hit_roundoff(source: Path, other: Path) -> float:
    """Permit rounding in cosine only, never in hit identities, ranks or overlap."""
    differences = []
    for left, right in zip(read_jsonl(source), read_jsonl(other), strict=True):
        differences.append(
            abs(left.pop("spec2vec_cosine") - right.pop("spec2vec_cosine"))
        )
        if left != right:
            msg = "replay changed reference ranks, identities or overlap"
            raise ValueError(msg)
    if max(differences) > ROUNDOFF_TOLERANCE:
        msg = "reference score discrepancy exceeds roundoff tolerance"
        raise ValueError(msg)
    return max(differences)


def compare_payloads(evidence: Path, replay: Path) -> tuple[list[str], dict]:
    """Classify exact replay files separately from bounded BLAS rounding."""
    exact, roundoff = [], {}
    for source in sorted(evidence.rglob("*")):
        if (
            not source.is_file()
            or source.name == "input_seal.json"
            or source.name.endswith("_complete.json")
        ):
            continue
        relative = source.relative_to(evidence)
        other = replay / relative
        if sha256_file(source) == sha256_file(other):
            exact.append(str(relative))
        elif source.name == "reference_similarities.npz":
            a, b = np.load(source)["cosine"], np.load(other)["cosine"]
            np.testing.assert_allclose(
                a, b, atol=ROUNDOFF_TOLERANCE, rtol=0, equal_nan=True
            )
            roundoff[str(relative)] = float(np.nanmax(abs(a - b)))
        elif source.name == "reference_hits.jsonl":
            roundoff[str(relative)] = reference_hit_roundoff(source, other)
        else:
            msg = f"nonidentical scientific payload: {relative}"
            raise ValueError(msg)
    return exact, roundoff


def check_multiplicity(evidence: Path, protocol: Path) -> dict:
    """Recompute every declared BY family with SciPy's independent routine."""
    adjustments = {}
    for config in read_json(protocol)["configurations"]:
        name = config["id"]
        feature = np.load(evidence / f"{name}_enrichment.npz", allow_pickle=False)
        independent = false_discovery_control(feature["p"].ravel(), method="by")
        error = float(
            np.max(abs(independent.reshape(feature["q"].shape) - feature["q"])),
        )
        if error > ROUNDOFF_TOLERANCE:
            msg = "feature multiplicity adjustment differs from SciPy"
            raise ValueError(msg)
        if np.any(feature["count"] > feature["support"][:, None]):
            msg = "feature-positive count exceeds distinct compound support"
            raise ValueError(msg)
        rows = read_jsonl(evidence / f"{name}_topics.jsonl")
        p, q = np.array([r["sos_p"] for r in rows]), np.array(
            [r["sos_q"] for r in rows],
        )
        np.testing.assert_allclose(
            q,
            false_discovery_control(p, method="by"),
            atol=ROUNDOFF_TOLERANCE,
            rtol=0,
        )
        adjustments[name] = {
            "sos_tests": len(rows),
            "feature_tests": feature["p"].size,
            "feature_by_max_abs_difference": error,
            "sos_min_p": float(p.min()),
            "feature_min_p": float(feature["p"].min()),
        }
    return adjustments


def validate(evidence: Path, replay: Path, protocol: Path) -> dict:
    """Require exact scientific inventories and document harmless BLAS roundoff."""
    verify_protocol_mirrors(evidence, replay, protocol)
    summaries = compare_summaries(evidence, replay)
    exact, roundoff = compare_payloads(evidence, replay)
    adjustments = check_multiplicity(evidence, protocol)
    return {
        "all_stage_summaries_identical": summaries,
        "bitwise_identical_payloads": exact,
        "reference_roundoff_max_abs": roundoff,
        "independent_by_checks": adjustments,
        "replay_input_seal_sha256": sha256_file(replay / "input_seal.json"),
        "original_input_seal_sha256": sha256_file(evidence / "input_seal.json"),
        "validated_source_sha256": source_hashes(),
        "interpretation": (
            "No retraining; no changed ranks, supports, overlaps, p/q values or "
            "conclusions. Reference BLAS thread counts differed (2 versus 4)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.evidence, args.replay, args.protocol)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
