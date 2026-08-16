"""Download and verify the public assets used by the MSnLib benchmark.

Only the two positive-mode archives required by the benchmark are acquired.
Downloads resume through HTTP Range requests, archives are checked against the
versioned Zenodo record, and extraction is staged until every benchmark input
has its expected SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

RECORD_ID = 20179680
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    """One immutable Zenodo archive."""

    name: str
    size_bytes: int
    md5: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{RECORD_API}/files/{self.name}/content"


@dataclass(frozen=True)
class RequiredFile:
    """One extracted file required by the frozen benchmark configuration."""

    relative_path: str
    size_bytes: int
    sha256: str


ASSETS = (
    Asset(
        name="Data.zip",
        size_bytes=1_641_318_401,
        md5="4ae210759ced93a2e673f0e5daf17d3e",
        sha256="c5f5855e99a09fa6d96b9b116b6a492e05561e5c7d033e696ea45f04d7ec5737",
    ),
    Asset(
        name="model_positive_mode.zip",
        size_bytes=1_991_689_484,
        md5="83d7ac4ff546653a03c3132be69d3d0f",
        sha256="d419d919d8ef4507f3ec362a9996117d17e2c5ad8cc16d966811271c28403bac",
    ),
)

REQUIRED_FILES = (
    RequiredFile(
        relative_path="Data/Benchmark_MSn_Lib/Corinna_Library_filtered_positive.mgf",
        size_bytes=88_159_030,
        sha256="c1a2389754e630590111bde8f50403c9d90081d3f3427bc63defef0b9714682f",
    ),
    RequiredFile(
        relative_path=(
            "Data/Benchmark_MSn_Lib/"
            "CaseStudy_Corinna_Library_filtered_1000motifs_output_100625/"
            "motifset.json"
        ),
        size_bytes=1_537_391,
        sha256="76cc12fc12b9160a5871d4bf1f5c373d6ca5b520bbafcd078ae0d1b5e542587d",
    ),
    RequiredFile(
        relative_path=(
            "model_positive_mode/positive_train_data/" "150225_CombLibraries_spectra.db"
        ),
        size_bytes=4_372_279_296,
        sha256="f4d8c6af067479f1082cad8c4fa8958d975506eed89d6026fa10ee89121912f1",
    ),
    RequiredFile(
        relative_path=(
            "model_positive_mode/positive_train_data/"
            "150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy"
        ),
        size_bytes=1_023_072_128,
        sha256="30ac8b287b02e53cd3e72bcfc2083e608aa0dcfcda8f73f5e237dfd40754b4d3",
    ),
    RequiredFile(
        relative_path=(
            "model_positive_mode/positive_train_data/"
            "150225_Spec2Vec_pos_CleanedLibraries.model"
        ),
        size_bytes=9_943_191,
        sha256="322a81fcd4f1f77a12007735d71f0a4f509f87c76703e2fef8f954eb909a2b82",
    ),
)


def file_digests(path: Path) -> tuple[str, str]:
    """Return MD5 and SHA-256 without loading a large file into memory."""

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def verify_archive(path: Path, asset: Asset) -> dict[str, object]:
    """Validate an acquired archive against the frozen Zenodo metadata."""

    if not path.is_file():
        raise FileNotFoundError(f"missing archive: {path}")
    size = path.stat().st_size
    if size != asset.size_bytes:
        raise ValueError(
            f"archive size mismatch for {asset.name}: {size} != {asset.size_bytes}"
        )
    md5, sha256 = file_digests(path)
    if md5 != asset.md5:
        raise ValueError(f"archive MD5 mismatch for {asset.name}")
    if sha256 != asset.sha256:
        raise ValueError(f"archive SHA-256 mismatch for {asset.name}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"archive CRC failure in {asset.name}: {bad_member}")
    return {
        "bytes": size,
        "md5": md5,
        "sha256": sha256,
        "zip_test": "ok",
    }


def safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject paths or links that could escape the extraction directory."""

    members = archive.infolist()
    for member in members:
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in ZIP archive: {member.filename!r}")
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ValueError(f"symbolic link in ZIP archive: {member.filename!r}")
    return members


def validate_extracted(
    extracted_root: Path,
    required_files: Sequence[RequiredFile] = REQUIRED_FILES,
) -> dict[str, dict[str, object]]:
    """Validate all benchmark-facing extracted inputs."""

    verified: dict[str, dict[str, object]] = {}
    for required in required_files:
        path = extracted_root / required.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing extracted benchmark input: {path}")
        size = path.stat().st_size
        if size != required.size_bytes:
            raise ValueError(
                f"extracted size mismatch for {required.relative_path}: "
                f"{size} != {required.size_bytes}"
            )
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        if digest != required.sha256:
            raise ValueError(f"extracted SHA-256 mismatch for {required.relative_path}")
        verified[required.relative_path] = {"bytes": size, "sha256": digest}
    return verified


def fetch_record_metadata() -> dict[str, object]:
    """Fetch and validate the immutable Zenodo record metadata."""

    with urllib.request.urlopen(RECORD_API) as response:  # noqa: S310
        record = json.load(response)
    if int(record.get("id", -1)) != RECORD_ID:
        raise ValueError("Zenodo returned the wrong record")
    deposited = {row["key"]: row for row in record.get("files", [])}
    for asset in ASSETS:
        row = deposited.get(asset.name)
        if row is None:
            raise ValueError(f"Zenodo record no longer lists {asset.name}")
        if int(row.get("size", -1)) != asset.size_bytes:
            raise ValueError(f"Zenodo size changed for {asset.name}")
        if row.get("checksum") != f"md5:{asset.md5}":
            raise ValueError(f"Zenodo checksum changed for {asset.name}")
    return record


def download_asset(asset: Asset, archive_dir: Path) -> Path:
    """Resume one download and atomically publish it after verification."""

    destination = archive_dir / asset.name
    if destination.exists():
        verify_archive(destination, asset)
        print(f"verified existing {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset.size_bytes:
        raise ValueError(f"partial download is larger than expected: {partial}")
    if offset == asset.size_bytes:
        verify_archive(partial, asset)
        os.replace(partial, destination)
        return destination

    request = urllib.request.Request(asset.url)  # noqa: S310
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:  # noqa: S310
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        if offset and not append:
            print(f"server did not resume {asset.name}; restarting partial download")
            offset = 0
        mode = "ab" if append else "wb"
        downloaded = offset
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                print(
                    f"\r{asset.name}: {downloaded / asset.size_bytes:.1%}",
                    end="",
                    flush=True,
                )
    print()
    verify_archive(partial, asset)
    os.replace(partial, destination)
    return destination


def extract_archives(archive_paths: Iterable[Path], data_root: Path) -> Path:
    """Extract through a resumable staging directory and publish atomically."""

    extracted = data_root / "extracted"
    if extracted.exists():
        validate_extracted(extracted)
        print(f"verified existing {extracted}")
        return extracted

    staging = data_root / ".extracted.partial"
    staging.mkdir(parents=True, exist_ok=True)
    for path in archive_paths:
        print(f"extracting {path.name}")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(staging, members=safe_zip_members(archive))
    validate_extracted(staging)
    os.replace(staging, extracted)
    return extracted


def write_acquisition_manifest(
    data_root: Path,
    archives: dict[str, dict[str, object]],
    extracted: dict[str, dict[str, object]],
) -> Path:
    """Record the public acquisition without machine-specific absolute paths."""

    manifest = {
        "schema_version": "msnlib-validation-acquisition/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "zenodo_record": RECORD_ID,
        "zenodo_api": RECORD_API,
        "archives": archives,
        "extracted_inputs": extracted,
        "construction_command": (
            "python scripts/download_msnlib_validation_assets.py "
            "--data-root <DATA_ROOT>"
        ),
    }
    path = data_root / "acquisition_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def validate_acquisition_manifest(
    data_root: Path,
    archives: dict[str, dict[str, object]],
    extracted: dict[str, dict[str, object]],
) -> Path:
    """Verify the existing acquisition manifest without rewriting evidence."""

    path = data_root / "acquisition_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing acquisition manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "msnlib-validation-acquisition/v1":
        raise ValueError("unexpected acquisition manifest schema")
    if int(manifest.get("zenodo_record", -1)) != RECORD_ID:
        raise ValueError("acquisition manifest records the wrong Zenodo record")
    if manifest.get("zenodo_api") != RECORD_API:
        raise ValueError("acquisition manifest records the wrong Zenodo API URL")
    if manifest.get("archives") != archives:
        raise ValueError("acquisition manifest archive evidence differs")
    if manifest.get("extracted_inputs") != extracted:
        raise ValueError("acquisition manifest extracted-input evidence differs")
    return path


def acquire(data_root: Path, *, verify_only: bool = False) -> Path:
    """Acquire or verify the complete public benchmark input surface."""

    data_root = data_root.resolve()
    archive_dir = data_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if verify_only:
        archive_paths = [archive_dir / asset.name for asset in ASSETS]
    else:
        record = fetch_record_metadata()
        (data_root / f"record-{RECORD_ID}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        archive_paths = [download_asset(asset, archive_dir) for asset in ASSETS]

    archive_results = {
        asset.name: verify_archive(path, asset)
        for asset, path in zip(ASSETS, archive_paths, strict=True)
    }
    extracted_root = extract_archives(archive_paths, data_root)
    extracted_results = validate_extracted(extracted_root)
    if verify_only:
        return validate_acquisition_manifest(
            data_root, archive_results, extracted_results
        )
    return write_acquisition_manifest(data_root, archive_results, extracted_results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and checksum the public MSnLib validation assets."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Output root containing archives/, extracted/, and the manifest.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not use the network; verify existing archives and extraction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = acquire(args.data_root, verify_only=args.verify_only)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"asset acquisition failed: {error}", file=sys.stderr)
        return 1
    print(f"validated acquisition manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
