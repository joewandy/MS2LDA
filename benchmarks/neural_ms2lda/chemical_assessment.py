"""Orchestrate saved-fit assessments; no training or production-path changes.

This module owns scientific data preparation and evidence output. Statistical
equations live in chemical_nulls; spectral comparisons in motif_correspondence.
All 6,000 topics remain in the inventory, including unsupported/unannotated
ones. Main results and sensitivity results are separate frozen configurations.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

from .chemical import score_precomputed_annotations
from .chemical_assessment_io import (
    cohort_strata,
    compound_cohort,
    fit_paths,
    read_jsonl,
)
from .chemical_nulls import feature_enrichment, sos_permutations
from .mag import consensus_fingerprint
from .utils import read_json, sha256_file, write_json, write_jsonl


def source_hashes() -> dict[str, str]:
    """Record analysis implementation separately from pre-outcome input sealing."""
    paths = sorted(Path(__file__).parent.glob("chemical_*.py"))
    paths += [
        Path(__file__).with_name("motif_correspondence.py"),
        Path(__file__).with_name("motif_reference.py"),
        Path("scripts/run_motif_chemical_assessment.py"),
    ]
    return {str(p): sha256_file(p) for p in paths if p.is_file()}


def finish_stage(output: Path, stage: str, payloads: list[Path], summary: dict) -> None:
    """Write a completion marker only after every evidence payload is present."""
    write_json(
        output / f"{stage}_complete.json",
        {
            "summary": summary,
            "source_sha256": source_hashes(),
            "payloads": {
                str(p.relative_to(output)): {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in payloads
            },
        },
    )


def verify_stage(output: Path, stage: str) -> dict:
    """Hash-check saved outputs before resume, downstream analysis or reporting."""
    complete = read_json(output / f"{stage}_complete.json")
    for name, expected in complete["payloads"].items():
        path = output / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"assessment evidence changed: {path}")
    return complete["summary"]


def prepare_assessment(protocol: dict, output: Path) -> tuple[list[dict], list[dict]]:
    """Load all six fits, independently reconstruct and reproduce original SOS."""
    records = read_jsonl(Path(protocol["prepared"]) / "data/validation_records.jsonl")
    compounds, spectrum_compounds = compound_cohort(records, Path(protocol["mgf"]))
    fingerprints = np.array([c["maccs"] for c in compounds], dtype=bool)
    consensus = cache(consensus_fingerprint)
    fits, verification = [], []
    for specification in protocol["fits"]:
        paths = fit_paths(specification)
        theta = np.load(paths["theta"], allow_pickle=False)
        annotations = read_jsonl(paths["annotations"])
        original = read_json(paths["chemical"])["chemical_evaluation"]
        recalculated = score_precomputed_annotations(
            theta=theta,
            records=records,
            annotations=annotations,
            fingerprint_threshold=protocol["consensus_threshold"],
        )
        if recalculated != original:
            raise ValueError(
                f"original SOS did not reproduce exactly: {specification['id']}"
            )
        winner = np.argmax(theta, axis=1)
        members, vectors, available, rows = [], [], [], []
        for k, annotation in enumerate(
            sorted(annotations, key=lambda r: r["topic_id"])
        ):
            if annotation["topic_id"] != k:
                raise ValueError("non-contiguous annotation IDs")
            ids = np.unique(spectrum_compounds[winner == k])
            fp = (
                consensus(
                    tuple(annotation["clustered_smiles"]),
                    protocol["consensus_threshold"],
                )
                if annotation["optimized_feature_count"] > 0
                else None
            )
            members.append(ids)
            available.append(fp is not None)
            vectors.append(
                np.zeros(fingerprints.shape[1], dtype=bool) if fp is None else fp
            )
            direct = (
                float((fingerprints[ids] & fp).sum() / max(1, len(ids) * fp.sum()))
                if fp is not None and len(ids)
                else None
            )
            saved = original["topic_scores"][k]["sos"]
            if (direct is None) != (saved is None) or (
                direct is not None and not np.isclose(direct, saved, atol=1e-14, rtol=0)
            ):
                raise ValueError(
                    "independent integer-intersection SOS reconstruction differs"
                )
            rows.append(
                {
                    "fit": specification["id"],
                    "topic_id": k,
                    "compound_ids": ids.tolist(),
                    "associated_spectra": int(np.sum(winner == k)),
                    "annotation_available": fp is not None,
                    "annotation_maccs": None if fp is None else fp.astype(int).tolist(),
                }
            )
        fits.append(
            {
                **specification,
                "members": members,
                "annotations": np.array(vectors),
                "available": np.array(available),
            }
        )
        write_jsonl(output / f"inventory/{specification['id']}.jsonl", rows)
        verification.append(
            {
                "fit": specification["id"],
                "exact_original_reproduction": True,
                "independent_sos_max_abs_tolerance": 1e-14,
                "eligible_topics": original["eligible_topics"],
                "mean_sos": original["mean_sos"],
            }
        )
        print(
            f"Verified original and independent SOS: {specification['id']}",
            flush=True,
        )
    write_jsonl(output / "compounds.jsonl", compounds)
    write_json(output / "original_sos_verification.json", verification)
    return compounds, fits


def restrict_members(
    fits: list[dict], selected: np.ndarray, total: int
) -> list[np.ndarray]:
    """Keep exactly the same hash-selected compound cohort for all model fits."""
    local = np.full(total, -1, dtype=int)
    local[selected] = np.arange(len(selected))
    return [local[ids][local[ids] >= 0] for fit in fits for ids in fit["members"]]


def describe(values: np.ndarray) -> dict:
    """Descriptive distribution; empty denominators are explicit, not zero."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q10": float(np.quantile(a, 0.1)),
        "q90": float(np.quantile(a, 0.9)),
    }


def specificity_rows(
    fits: list[dict],
    compounds: list[dict],
    selected: np.ndarray,
    members: list[np.ndarray],
    sos: dict,
    enrichment: dict,
) -> list[dict]:
    """One auditable record per topic, with support and explicit missing reasons."""
    rows, offset = [], 0
    for fit in fits:
        for k in range(len(fit["members"])):
            i = offset + k
            ids = selected[members[i]]
            eligible = bool(sos["eligible"][i])
            reason = (
                "no_compound_support"
                if not len(ids)
                else (
                    "missing_MAG_consensus"
                    if not fit["available"][k]
                    else (
                        "constant_conditional_null"
                        if not sos["informative"][i]
                        else None
                    )
                )
            )
            row = {
                "fit": fit["id"],
                "model": fit["model"],
                "topic_id": k,
                "compounds": len(ids),
                "scaffolds": len({compounds[j]["scaffold_key"] for j in ids}),
                "movable_compounds": int(sos["movable"][i]),
                "unavailable_or_uninformative_reason": reason,
                "sos_eligible": eligible,
                "sos_informative": bool(sos["informative"][i]),
                "enriched_features_q05": int(np.sum(enrichment["q"][i] <= 0.05)),
            }
            for field in (
                "observed",
                "expected",
                "excess",
                "null_sd",
                "mc_mean",
                "p",
                "q",
                "tail_mc_lower",
                "tail_mc_upper",
            ):
                row[f"sos_{field}"] = (
                    float(sos[field][i]) if eligible or field in ("p", "q") else None
                )
            row["sos_tail_hits"] = int(sos["tail_hits"][i]) if eligible else None
            rows.append(row)
        offset += len(fit["members"])
    return rows


def summarize_specificity(rows: list[dict], fits: list[dict]) -> list[dict]:
    """Aggregate fits separately; do not treat topics as independent replicates."""
    result = []
    for fit in fits:
        fit_rows = [r for r in rows if r["fit"] == fit["id"]]
        eligible = [r for r in fit_rows if r["sos_eligible"]]
        recurring = [r for r in eligible if r["compounds"] >= 2]
        result.append(
            {
                "fit": fit["id"],
                "model": fit["model"],
                "total_topics": len(fit_rows),
                "supported_topics": sum(r["compounds"] > 0 for r in fit_rows),
                "single_compound_topics": sum(r["compounds"] == 1 for r in fit_rows),
                "eligible_topics": len(eligible),
                "recurring_eligible_topics": len(recurring),
                "informative_sos_topics": sum(r["sos_informative"] for r in fit_rows),
                "sos_q05_topics": sum(r["sos_q"] <= 0.05 for r in fit_rows),
                "feature_q05_topics": sum(
                    r["enriched_features_q05"] > 0 for r in fit_rows
                ),
                "feature_q05_tests": sum(r["enriched_features_q05"] for r in fit_rows),
                "mean_sos": describe([r["sos_observed"] for r in eligible]),
                "background_sos": describe([r["sos_expected"] for r in eligible]),
                "excess_sos": describe([r["sos_excess"] for r in eligible]),
                "recurring_excess_sos": describe([r["sos_excess"] for r in recurring]),
                "positive_excess_topics": sum(r["sos_excess"] > 0 for r in eligible),
                "reasons": dict(
                    Counter(
                        r["unavailable_or_uninformative_reason"] or "informative"
                        for r in fit_rows
                    )
                ),
            }
        )
    return result


def write_maccs_definitions(output: Path) -> None:
    """Export the pinned native key definitions, including non-SMARTS exceptions."""
    from rdkit.Chem import MACCSkeys

    special = {
        1: "Isotope key: undefined and always unset in native RDKit.",
        125: "More than one aromatic ring (native procedural implementation).",
        166: "More than one disconnected molecular fragment (native procedure).",
    }
    write_json(
        output / "maccs_definitions.json",
        {
            "source": "Sealed rdkit.Chem.MACCSkeys.smartsPatts; native GenMACCSKeys",
            "position_zero": "unused and excluded from enrichment family",
            "source_file_sha256": sha256_file(Path(MACCSkeys.__file__)),
            "keys": [
                {
                    "position": bit,
                    "smarts": pattern,
                    "count_threshold_exclusive": threshold,
                    "special_definition": special.get(bit),
                }
                for bit, (pattern, threshold) in sorted(MACCSkeys.smartsPatts.items())
            ],
        },
    )


def run_specificity(protocol: dict, output: Path) -> None:
    """Execute primary and prespecified sensitivity nulls with identical fits."""
    compounds, fits = prepare_assessment(protocol, output)
    fingerprints = np.array([c["maccs"] for c in compounds], dtype=bool)
    annotations = np.concatenate([f["annotations"] for f in fits])
    available = np.concatenate([f["available"] for f in fits])
    summaries = {}
    for configuration in protocol["configurations"]:
        name = configuration["id"]
        if (output / f"{name}_complete.json").exists():
            summaries[name] = verify_stage(output, name)
            continue
        selected, blocks = cohort_strata(
            compounds, configuration["width_da"], configuration["scaffold_balanced"]
        )
        members = restrict_members(fits, selected, len(compounds))
        print(
            f"{name}: {len(selected)} compounds, {len(blocks)} strata; feature tests",
            flush=True,
        )
        enrichment = feature_enrichment(fingerprints[selected, 1:], members, blocks)
        print(
            f"{name}: {protocol['permutations']:,} shared fingerprint permutations",
            flush=True,
        )
        sos = sos_permutations(
            fingerprints[selected],
            annotations,
            available,
            members,
            blocks,
            permutations=protocol["permutations"],
            seed=protocol["permutation_seed"],
            batch_size=protocol["permutation_batch_size"],
        )
        rows = specificity_rows(fits, compounds, selected, members, sos, enrichment)
        row_path, array_path = (
            output / f"{name}_topics.jsonl",
            output / f"{name}_enrichment.npz",
        )
        write_jsonl(row_path, rows)
        # Store compact sufficient statistics; prevalence/effect/ratio are
        # reconstructed exactly from count, expected_count and support.
        np.savez_compressed(
            array_path,
            **{
                k: enrichment[k]
                for k in ("count", "support", "expected_count", "informative", "p", "q")
            },
        )
        cohort_path = output / f"{name}_cohort.json"
        write_json(
            cohort_path,
            {
                "compound_ids": selected.tolist(),
                "strata_local_ids": [b.tolist() for b in blocks],
            },
        )
        summary = {
            "configuration": configuration,
            "compounds": len(selected),
            "strata": len(blocks),
            "singleton_compounds": sum(len(b) == 1 for b in blocks),
            "permutations": protocol["permutations"],
            "sos_family_size": len(rows),
            "feature_family_size": int(enrichment["p"].size),
            "fits": summarize_specificity(rows, fits),
            "max_mc_mean_minus_analytic_in_mc_se": float(
                np.max(
                    np.abs(sos["mc_mean"] - sos["expected"])
                    / np.maximum(
                        sos["null_sd"] / np.sqrt(protocol["permutations"]), 1e-14
                    )
                )
            ),
        }
        finish_stage(output, name, [row_path, array_path, cohort_path], summary)
        summaries[name] = summary
        print(f"{name}: completed, evidence saved", flush=True)
    write_maccs_definitions(output)
    payloads = [
        output / "compounds.jsonl",
        output / "original_sos_verification.json",
        output / "maccs_definitions.json",
    ]
    payloads += sorted((output / "inventory").glob("*.jsonl"))
    payloads += [
        p
        for c in protocol["configurations"]
        for p in sorted(output.glob(c["id"] + "*"))
    ]
    finish_stage(output, "specificity", payloads, summaries)


def run_assessment(protocol: dict, output: Path, stage: str) -> None:
    """Run requested stages, resuming only verified completed scientific outputs."""
    from .motif_correspondence import run_stability
    from .motif_reference import run_reference_assessment

    actions = {
        "specificity": run_specificity,
        "stability": run_stability,
        "references": run_reference_assessment,
    }
    for name, action in actions.items():
        if stage not in ("all", name):
            continue
        if (output / f"{name}_complete.json").exists():
            verify_stage(output, name)
            print(f"Verified existing completed {name} evidence.", flush=True)
        else:
            action(protocol, output)
