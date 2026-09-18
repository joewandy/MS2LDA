"""Sealed, research-only orchestration for the fixed cross-model comparison.

The report reader uses only the standard library. NumPy and scientific model
helpers are imported only when recalculating the evidence, not for LaTeX tables.
An input seal precedes calculation; a completion marker follows all payloads.
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

from .chemical_assessment_report import read_assessment_evidence
from .utils import read_json, sha256_file, write_json


def validate_protocol(protocol: dict) -> None:
    """Fixed array codes and display grid must retain their declared meanings."""
    if protocol["cohorts"] != ["all", "recurring_evaluable", "recurring"]:
        raise ValueError("overlap cohort codes or order changed")
    if protocol["similarities"] != ["full_beta", "balanced_channels"]:
        raise ValueError("overlap similarity codes or order changed")
    if protocol["threshold_grid"] != {"minimum": 0.0, "maximum": 1.0, "points": 201}:
        raise ValueError("overlap threshold display grid changed")


def input_paths(protocol_path: Path) -> tuple[dict, dict, list[Path]]:
    """Resolve the exact preceding evidence and identically ordered beta columns."""
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    evidence = Path(protocol["chemical_evidence"])
    parent = read_assessment_evidence(evidence)
    seal = read_json(evidence / "input_seal.json")
    if (
        sha256_file(evidence / "input_seal.json")
        != protocol["chemical_input_seal_sha256"]
    ):
        raise ValueError("preceding chemical input seal differs")
    paths = [
        protocol_path,
        evidence / "input_seal.json",
        evidence / "protocol.json",
        evidence / "primary_50_enrichment.npz",
        evidence / "primary_50_topics.jsonl",
    ]
    vocabulary_hashes = set()
    for fit in parent["protocol"]["fits"]:
        root = Path(fit["path"])
        beta = root / f"validation_evaluation/{fit['method']}/beta.npy"
        vocabulary = root / "data/vocabulary.json"
        for path in (beta, vocabulary):
            if sha256_file(path) != seal["inputs"][str(path)]["sha256"]:
                raise ValueError(f"original fitted input changed: {path}")
        vocabulary_hashes.add(sha256_file(vocabulary))
        paths += [beta, vocabulary, evidence / f"inventory/{fit['id']}.jsonl"]
    if len(vocabulary_hashes) != 1:
        raise ValueError("fit vocabularies differ in content or order")
    return protocol, parent["protocol"], sorted(set(paths))


def seal_overlap(protocol_path: Path, output: Path) -> None:
    """Seal one new output directory; never overwrite a previous experiment."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("overlap sealing requires a new empty directory")
    _, parent, paths = input_paths(protocol_path)
    write_json(
        output / "input_seal.json",
        {
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "inputs": {
                str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in paths
            },
            "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
            "fits": parent["fits"],
        },
    )
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    print(
        f"Sealed {len(paths)} exact input files before extended calculations.",
        flush=True,
    )


def verify_inputs(output: Path) -> tuple[dict, dict]:
    """Verify the saved protocol and every source payload before recalculation."""
    seal = read_json(output / "input_seal.json")
    if sha256_file(output / "protocol.json") != seal["protocol_sha256"]:
        raise ValueError("overlap protocol changed after sealing")
    for name, expected in seal["inputs"].items():
        p = Path(name)
        if (
            p.stat().st_size != expected["bytes"]
            or sha256_file(p) != expected["sha256"]
        ):
            raise ValueError(f"overlap input changed: {p}")
    for package, version in seal["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"overlap dependency changed: {package}")
    protocol = read_json(output / "protocol.json")
    validate_protocol(protocol)
    return protocol, seal


def load_fits(protocol: dict, specifications: list[dict]) -> tuple[list, list[str]]:
    """Attach spectral vectors and fixed chemical effects to each topic index."""
    import numpy as np

    from .chemical_assessment_io import read_jsonl
    from .cross_model_overlap import spectral_vectors
    from .data import load_vocabulary, token_types
    from .motif_correspondence import load_beta, normalize_rows

    evidence = Path(protocol["chemical_evidence"])
    vocabulary = load_vocabulary(Path(specifications[0]["path"]) / "data")
    fragment = token_types(vocabulary)[:, 0].astype(bool)
    with np.load(evidence / "primary_50_enrichment.npz", allow_pickle=False) as data:
        effects = (data["count"] - data["expected_count"]) / np.maximum(
            1, data["support"]
        )[:, None]
    normalized_effects, effect_valid = normalize_rows(effects)
    fits, offset = [], 0
    for specification in specifications:
        beta = load_beta(specification, len(vocabulary))
        inventory = read_jsonl(evidence / f"inventory/{specification['id']}.jsonl")
        if (
            len(beta) != 1000
            or len(inventory) != len(beta)
            or any(r["topic_id"] != k for k, r in enumerate(inventory))
        ):
            raise ValueError("overlap requires ordered complete 1000-topic inventories")
        stop = offset + len(beta)
        fits.append(
            {
                "specification": specification,
                "beta": beta,
                "inventory": inventory,
                "vectors": spectral_vectors(beta, fragment),
                "top_words": np.argsort(-beta, axis=1, kind="stable")[:, :20],
                "effects": normalized_effects[offset:stop],
                "effect_valid": effect_valid[offset:stop],
            }
        )
        offset = stop
    if offset != len(effects):
        raise ValueError("chemical-feature and fitted-topic axes disagree")
    return fits, vocabulary


def run_overlap(output: Path) -> dict:
    """Evaluate six fixed fits, then seal complete evidence. No training occurs."""
    from itertools import combinations

    import numpy as np

    from .cross_model_overlap import (
        cohort_topics,
        directed_records,
        similarity_matrices,
        summarize_matches,
    )
    from .cross_model_overlap_examples import select_examples
    from .utils import atomic_save_numpy

    protocol, seal = verify_inputs(output)
    if (output / "complete.json").exists():
        return read_overlap(output)["summary"]
    fits, vocabulary = load_fits(protocol, seal["fits"])
    cohort_sizes = {
        fit["specification"]["id"]: {
            c: len(cohort_topics(fit["inventory"], c)) for c in protocol["cohorts"]
        }
        for fit in fits
    }
    records = []
    for i, j in combinations(range(len(fits)), 2):
        matrices = similarity_matrices(fits[i]["vectors"], fits[j]["vectors"])
        for source, target, scores in (
            (i, j, matrices),
            (j, i, {k: v.T for k, v in matrices.items()}),
        ):
            for c in range(len(protocol["cohorts"])):
                for s in range(len(protocol["similarities"])):
                    records.append(
                        directed_records(
                            fits[source],
                            fits[target],
                            scores,
                            source_id=source,
                            target_id=target,
                            cohort_id=c,
                            similarity_id=s,
                            cohorts=protocol["cohorts"],
                        )
                    )
        print(f"Compared {seal['fits'][i]['id']} / {seal['fits'][j]['id']}", flush=True)
    rows = np.concatenate(records)
    summary = summarize_matches(
        rows, {**protocol, "cohort_sizes": cohort_sizes}, seal["fits"]
    )
    examples = select_examples(rows, fits, vocabulary, protocol)
    atomic_save_numpy(output / "directed_matches.npy", rows)
    write_json(output / "summary.json", summary)
    write_json(output / "examples.json", examples)
    write_json(
        output / "inventory.json",
        {
            "fits": seal["fits"],
            "cohort_sizes": cohort_sizes,
            "vocabulary_size": len(vocabulary),
            "directed_match_rows": len(rows),
            "cohorts": protocol["cohorts"],
            "similarities": protocol["similarities"],
            "dtype": rows.dtype.descr,
        },
    )
    # Recheck source identity before declaring completion; partial outputs have
    # no completion marker and can never be consumed as publication evidence.
    verify_inputs(output)
    source_paths = list(Path(__file__).parent.glob("cross_model_overlap*.py"))
    source_paths += [
        Path(__file__).with_name(name)
        for name in ("motif_correspondence.py", "motif_reference.py")
    ]
    write_json(
        output / "complete.json",
        {
            "input_seal_sha256": sha256_file(output / "input_seal.json"),
            "protocol_sha256": sha256_file(output / "protocol.json"),
            "source_sha256": {str(p): sha256_file(p) for p in sorted(source_paths)},
            "payloads": {
                name: {
                    "sha256": sha256_file(output / name),
                    "bytes": (output / name).stat().st_size,
                }
                for name in (
                    "directed_matches.npy",
                    "summary.json",
                    "examples.json",
                    "inventory.json",
                )
            },
        },
    )
    return summary


def read_overlap(output: Path) -> dict:
    """Read report evidence without raw fits; reject altered or incomplete data."""
    complete = read_json(output / "complete.json")
    if set(complete["payloads"]) != {
        "directed_matches.npy",
        "summary.json",
        "examples.json",
        "inventory.json",
    }:
        raise ValueError("overlap evidence manifest is incomplete")
    for name, key in (
        ("input_seal.json", "input_seal_sha256"),
        ("protocol.json", "protocol_sha256"),
    ):
        if sha256_file(output / name) != complete[key]:
            raise ValueError(f"overlap {name} identity changed")
    for name, expected in complete["payloads"].items():
        p = output / name
        if (
            p.stat().st_size != expected["bytes"]
            or sha256_file(p) != expected["sha256"]
        ):
            raise ValueError(f"overlap payload changed: {name}")
    result = {
        name: read_json(output / f"{name}.json")
        for name in ("protocol", "summary", "examples", "inventory")
    }
    validate_protocol(result["protocol"])
    if (
        len(result["summary"]["pairs"]) != 180
        or len(result["summary"]["sources"]) != 72
    ):
        raise ValueError("incomplete ordered-fit/cohort/similarity inventory")
    if (
        sorted(f["model"] for f in result["inventory"]["fits"])
        != ["selected"] * 3 + ["tomotopy"] * 3
    ):
        raise ValueError("overlap requires three selected and three Tomotopy fits")
    return result
