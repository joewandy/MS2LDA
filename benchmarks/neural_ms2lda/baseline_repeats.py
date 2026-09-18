"""Small provenance boundaries for repeated fits on one frozen data split.

Training randomness is separate from the protocol seed that created the split,
word embeddings and completion mask. Repeats never rewrite that shared protocol.
"""

import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .reproducibility import VALIDATION_DATA_FILES
from .reproduction_audit import verify_linked_inputs
from .utils import read_json_object, sha256_file


def runtime_versions() -> dict:
    """Record actual scientific dependencies without importing GPU libraries.

    The timestamp states when versions were observed. An observation collected
    after job launch must not be described as a launch-time environment snapshot.
    Exact required versions remain owned by the canonical environment.yml.
    """
    packages = (
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "tomotopy",
        "gensim",
        "matchms",
        "spec2vec",
        "rdkit",
    )
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def resolve_training_seed(protocol: dict, requested: int | None, *, offset=0) -> int:
    """Preserve the old protocol-derived default; override only fit randomness."""
    seed = int(protocol["seed"]) + offset if requested is None else requested
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("training seed must be an integer in [0, 2**31)")
    return seed


def frozen_training_input_hashes(run: Path) -> dict:
    """Rehash the input roles that were visible when the model was fitted.

    Relative input roles, rather than absolute paths, permit relocating an
    unchanged run. A swapped matrix or changed recipe must never hit the cache.
    This does not authorize test use: evaluation must separately verify release,
    while fitting must reject test exposure in ``baseline_fit_identity`` below.
    """
    run = run.absolute()
    manifest_path = run / "validation_input_manifest.json"
    if not manifest_path.exists():
        # The generic Tomotopy backend also supports the original full study
        # runner. Its training cache depends only on these training inputs;
        # validation-only repeat jobs always take the sealed branch below.
        roles = ["protocol.json", "data/train.npz", "data/vocabulary.json"]
        return {role: sha256_file(run / role) for role in roles}
    manifest = read_json_object(manifest_path)
    if manifest.get("candidate_test_artifacts_accessed") is not False:
        raise RuntimeError("baseline repeat must be validation-only")
    roles = [
        "protocol.json",
        "token_features/features.npy",
        *(f"data/{name}" for name in VALIDATION_DATA_FILES),
    ]
    declared = {
        Path(row["linked_path"]).absolute(): row
        for row in manifest.get("linked_inputs", [])
    }
    if not {run / role for role in roles}.issubset(declared):
        raise RuntimeError("baseline manifest does not seal every required input")
    records = [declared[run / role] for role in roles]
    verify_linked_inputs({"linked_inputs": records}, field="linked_inputs")
    return {role: declared[run / role]["sha256"] for role in roles}


def baseline_fit_identity(run: Path, training_seed: int, recipe: dict) -> dict:
    """Identify one training-only fit, refusing test exposure in a sealed view."""
    if (run / "validation_input_manifest.json").is_file() and any(
        (run / "data").glob("test*")
    ):
        raise RuntimeError("baseline repeat exposes test inputs")
    return {
        "training_seed": training_seed,
        "recipe": recipe,
        "input_sha256": frozen_training_input_hashes(run),
    }


def require_cached_fit(result: dict, expected: dict) -> None:
    """Reject stale/legacy caches whose recipe and consumed bytes cannot be proved."""
    if result.get("fit_identity") != expected:
        raise RuntimeError("cached baseline seed, recipe or input identity differs")
