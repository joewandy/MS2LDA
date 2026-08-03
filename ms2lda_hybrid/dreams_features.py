"""Extract frozen DreaMS features and align them with spectral words.

DreaMS is imported only when ``DreaMSFeatureExtractor`` is constructed.  The
rest of MS2LDA therefore remains usable without the optional dependency. This
module owns the shared ``frag@mass``/``loss@mass`` grammar used by both feature
pooling and the hybrid topic-word prior.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DREAMS_GIT_COMMIT = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
DREAMS_MODEL_NAME = "DreaMS_embedding"
DREAMS_EMBEDDING_DIM = 1024
DREAMS_MAX_PEAKS = 100


def _sha256(path: Path) -> str:
    """Hash a checkpoint without reading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_dreams_commit() -> str | None:
    """Return the PEP 610 commit recorded for the installed DreaMS package."""
    try:
        distribution = importlib.metadata.distribution("dreams")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, Mapping):
        return None
    vcs_info = metadata.get("vcs_info")
    if not isinstance(vcs_info, Mapping):
        return None
    commit = vcs_info.get("commit_id")
    return str(commit) if commit else None


def _metadata(spectrum: Any, key: str, default: Any = None) -> Any:
    """Read metadata from a matchms-like getter or mapping."""
    getter = getattr(spectrum, "get", None)
    if callable(getter):
        value = getter(key, default)
        if value is not None:
            return value
    values = getattr(spectrum, "metadata", {})
    return values.get(key, default) if isinstance(values, Mapping) else default


def spectrum_arrays(spectrum: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return positive m/z, nonnegative intensities, and precursor m/z."""

    peaks = getattr(spectrum, "peaks", None)
    if peaks is None:
        raise TypeError("each spectrum must expose a matchms-like .peaks object")
    mz = np.asarray(peaks.mz, dtype=np.float32)
    intensities = np.asarray(peaks.intensities, dtype=np.float32)
    if mz.ndim != 1 or mz.shape != intensities.shape or not len(mz):
        raise ValueError("m/z and intensity must be equal-length non-empty vectors")
    if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensities)):
        raise ValueError("spectrum peaks contain non-finite values")
    if np.any(mz <= 0) or np.any(intensities < 0) or not np.any(intensities > 0):
        raise ValueError(
            "spectrum peaks require positive m/z, nonnegative intensity, "
            "and at least one positive intensity"
        )
    precursor = float(_metadata(spectrum, "precursor_mz", np.nan))
    if not np.isfinite(precursor) or precursor <= 0:
        raise ValueError("each spectrum requires a positive precursor_mz")
    return mz, intensities, precursor


def parse_spectral_word(word: str) -> tuple[str, float] | None:
    """Parse a finite nonnegative ``frag@mass`` or ``loss@mass`` token.

    Prefix matching is case-insensitive. ``None`` denotes an unrelated or
    malformed vocabulary item, which the hybrid model treats as an ``other``
    token with no numeric mass feature.
    """
    prefix, separator, raw_value = str(word).partition("@")
    if not separator or prefix.lower() not in {"frag", "loss"}:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return prefix.lower(), value


@dataclass(frozen=True)
class DreaMSFeatureBatch:
    """Row-aligned global and contextual embeddings for a spectrum collection.

    For ``D`` spectra, ``spectrum_embeddings`` has shape ``D x 1024`` and
    ``peak_embeddings`` has shape ``D x P x 1024``. ``peak_mz`` and
    ``peak_mask`` have shape ``D x P``. Row order is identical to
    ``identifiers`` and to the input order passed to :meth:`extract`.
    """

    identifiers: tuple[str, ...]
    spectrum_embeddings: np.ndarray
    peak_embeddings: np.ndarray
    peak_mz: np.ndarray
    peak_mask: np.ndarray
    precursor_mz: np.ndarray
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate row counts, tensor shapes, and finite embeddings."""
        count = len(self.identifiers)
        if self.spectrum_embeddings.ndim != 2:
            raise ValueError("spectrum_embeddings must be a matrix")
        if self.peak_embeddings.ndim != 3:
            raise ValueError("peak_embeddings must be a rank-three array")
        if self.spectrum_embeddings.shape[0] != count:
            raise ValueError("identifier and spectrum embedding counts differ")
        if self.peak_embeddings.shape[0] != count:
            raise ValueError("identifier and peak embedding counts differ")
        peak_shape = self.peak_embeddings.shape[:2]
        if any(values.shape != peak_shape for values in (self.peak_mz, self.peak_mask)):
            raise ValueError("peak metadata is not aligned to peak embeddings")
        if not np.issubdtype(self.peak_mask.dtype, np.bool_):
            raise ValueError("peak_mask must be boolean")
        if self.precursor_mz.shape != (count,):
            raise ValueError("precursor_mz must contain one value per spectrum")
        if self.peak_embeddings.shape[2] != self.spectrum_embeddings.shape[1]:
            raise ValueError("spectrum and peak embedding dimensions differ")
        if not np.all(np.isfinite(self.spectrum_embeddings)):
            raise ValueError("spectrum embeddings contain non-finite values")
        if not np.all(np.isfinite(self.peak_embeddings)):
            raise ValueError("peak embeddings contain non-finite values")
        observed_mz = self.peak_mz[self.peak_mask]
        if not np.all(np.isfinite(observed_mz)) or np.any(observed_mz <= 0):
            raise ValueError("observed peak m/z values must be finite and positive")
        if not np.all(np.isfinite(self.precursor_mz)) or np.any(self.precursor_mz <= 0):
            raise ValueError("precursor m/z values must be finite and positive")

    def save(self, path: str | Path) -> None:
        """Write a compressed, self-describing HDF5 cache."""

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("DreaMS feature caching requires h5py") from exc
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "w") as handle:
            text = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "identifiers",
                data=np.asarray(self.identifiers, dtype=object),
                dtype=text,
            )
            for name, values in (
                ("spectrum_embeddings", self.spectrum_embeddings),
                ("peak_mz", self.peak_mz),
                ("peak_mask", self.peak_mask),
            ):
                handle.create_dataset(name, data=values, compression="gzip")
            # Contextual states dominate cache size; float16 is sufficient for
            # the downstream pooled word prior and matches the validated cache.
            handle.create_dataset(
                "peak_embeddings",
                data=self.peak_embeddings.astype(np.float16, copy=False),
                compression="gzip",
            )
            handle.create_dataset("precursor_mz", data=self.precursor_mz)
            handle.attrs["provenance_json"] = json.dumps(
                self.provenance,
                sort_keys=True,
            )

    @classmethod
    def load(cls, path: str | Path) -> DreaMSFeatureBatch:
        """Read a cache written by :meth:`save` into memory."""
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("DreaMS feature caching requires h5py") from exc
        with h5py.File(Path(path), "r") as handle:
            identifiers = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["identifiers"][:]
            )
            return cls(
                identifiers=identifiers,
                spectrum_embeddings=handle["spectrum_embeddings"][:],
                peak_embeddings=handle["peak_embeddings"][:],
                peak_mz=handle["peak_mz"][:],
                peak_mask=handle["peak_mask"][:].astype(bool),
                precursor_mz=handle["precursor_mz"][:],
                provenance=json.loads(handle.attrs.get("provenance_json", "{}")),
            )


class DreaMSFeatureExtractor:
    """Load the official frozen top-100 DreaMS embedding model once.

    Construction may download the public checkpoint through the official
    DreaMS API. The model is never fine-tuned; repeated :meth:`extract` calls
    reuse the same frozen backbone.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
    ) -> None:
        """Initialize the validated public checkpoint on ``device``."""
        try:
            import torch
            from dreams.api import PreTrainedModel
            from dreams.definitions import DREAMS_EMBEDDING, PRETRAINED
            from dreams.utils.data import SpectrumPreprocessor
            from dreams.utils.dformats import DataFormatA
        except ImportError as exc:
            raise ImportError(
                "install the pinned DreaMS dependency described in the README"
            ) from exc
        installed_commit = _installed_dreams_commit()
        if installed_commit != DREAMS_GIT_COMMIT:
            raise RuntimeError(
                "DreaMS must be installed from the pinned commit "
                f"{DREAMS_GIT_COMMIT}; found {installed_commit or 'no commit metadata'}"
            )
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        wrapped = PreTrainedModel.from_name(DREAMS_EMBEDDING)
        head_checkpoint = Path(PRETRAINED) / "embedding_model.ckpt"
        backbone_checkpoint = Path(PRETRAINED) / "ssl_model.ckpt"
        self._torch = torch
        self._model = wrapped.model.to(requested_device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(requires_grad=False)
        if not hasattr(self._model, "backbone") or not hasattr(self._model, "head"):
            raise TypeError("checkpoint is not the public DreaMS embedding model")
        self._device = requested_device
        self._dtype = next(self._model.parameters()).dtype
        self._preprocessor = SpectrumPreprocessor(
            dformat=DataFormatA(),
            n_highest_peaks=DREAMS_MAX_PEAKS,
        )
        self._n_peaks = DREAMS_MAX_PEAKS
        try:
            version = importlib.metadata.version("dreams")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            version = None
        self._provenance: dict[str, Any] = {
            "expected_dreams_git_commit": DREAMS_GIT_COMMIT,
            "installed_dreams_git_commit": installed_commit,
            "dreams_package_version": version,
            "model_name": DREAMS_MODEL_NAME,
            "head_checkpoint_sha256": (
                _sha256(head_checkpoint) if head_checkpoint.is_file() else None
            ),
            "backbone_checkpoint_sha256": (
                _sha256(backbone_checkpoint) if backbone_checkpoint.is_file() else None
            ),
            "embedding_dim": DREAMS_EMBEDDING_DIM,
            "n_highest_peaks": DREAMS_MAX_PEAKS,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a copy of the expected package and checkpoint provenance."""
        return dict(self._provenance)

    def extract(
        self,
        spectra: Sequence[Any],
        *,
        identifiers: Sequence[str] | None = None,
        batch_size: int = 32,
    ) -> DreaMSFeatureBatch:
        """Extract row-aligned global and per-peak embeddings.

        ``spectra`` must expose matchms-like peaks and ``precursor_mz``. The
        returned rows preserve input order. ``identifiers``, when supplied,
        must contain exactly one identifier per spectrum.
        """
        if not spectra or batch_size < 1:
            raise ValueError("spectra must be non-empty and batch_size positive")
        if identifiers is not None and len(identifiers) != len(spectra):
            raise ValueError("identifiers must align with spectra")
        processed: list[np.ndarray] = []
        shape = (len(spectra), self._n_peaks)
        peak_mz = np.zeros(shape, dtype=np.float32)
        peak_mask = np.zeros(shape, dtype=bool)
        precursor_mz = np.empty(len(spectra), dtype=np.float32)
        resolved_ids: list[str] = []
        for index, spectrum in enumerate(spectra):
            mz, intensities, precursor = spectrum_arrays(spectrum)
            values = self._preprocessor(
                np.column_stack([mz, intensities]),
                prec_mz=precursor,
                high_form=True,
                augment=False,
            )
            if values.shape != (self._n_peaks + 1, 2):
                raise RuntimeError("DreaMS preprocessing returned an unexpected shape")
            peaks = values[1:]
            peak_mz[index] = peaks[:, 0]
            peak_mask[index] = peaks[:, 0] > 0
            precursor_mz[index] = precursor
            processed.append(values)
            fallback = _metadata(spectrum, "feature_id", str(index))
            resolved_ids.append(
                str(
                    identifiers[index]
                    if identifiers is not None
                    else _metadata(spectrum, "id", fallback)
                )
            )
        global_parts: list[np.ndarray] = []
        peak_parts: list[np.ndarray] = []
        for start in range(0, len(processed), batch_size):
            tensor = self._torch.from_numpy(
                np.stack(processed[start : start + batch_size])
            ).to(device=self._device, dtype=self._dtype)
            with self._torch.inference_mode():
                # Call backbone and head separately: the public convenience API
                # returns only global embeddings, while pooling also needs the
                # contextual state corresponding to every retained peak.
                states = self._model.backbone(tensor, charge=None)
                global_embeddings = self._model.head(states[:, 0])
            if (
                states.shape[-1] != DREAMS_EMBEDDING_DIM
                or global_embeddings.shape[-1] != DREAMS_EMBEDDING_DIM
            ):
                raise RuntimeError("public DreaMS embedding dimension changed")
            global_parts.append(global_embeddings.cpu().float().numpy())
            peak_parts.append(states[:, 1 : self._n_peaks + 1].cpu().half().numpy())
        return DreaMSFeatureBatch(
            identifiers=tuple(resolved_ids),
            spectrum_embeddings=np.concatenate(global_parts),
            peak_embeddings=np.concatenate(peak_parts),
            peak_mz=peak_mz,
            peak_mask=peak_mask,
            precursor_mz=precursor_mz,
            provenance=self.provenance,
        )


def pool_word_embeddings(
    documents: Sequence[Sequence[str]],
    features: DreaMSFeatureBatch,
    *,
    mz_tolerance: float = 0.02,
) -> dict[str, np.ndarray]:
    """Pool train-only contextual peak states into spectral-word features.

    The vocabulary is the first-occurrence order of words in ``documents``,
    matching :class:`HybridLDAModel`. A ``frag@m`` word maps to the nearest
    peak at ``m``; ``loss@l`` maps to the nearest peak at
    ``precursor_mz - l``. Matches must fall within ``mz_tolerance``. Repeated
    words are count-weighted and unmatched words are omitted from the result.
    Documents and feature rows must describe the same spectra in the same
    order; callers should pass training rows only to avoid leakage.
    """

    if len(documents) != len(features.identifiers):
        raise ValueError("documents must align with feature rows")
    if not np.isfinite(mz_tolerance) or mz_tolerance <= 0:
        raise ValueError("mz_tolerance must be finite and positive")
    vocabulary = list(dict.fromkeys(str(word) for doc in documents for word in doc))
    columns = {str(word): index for index, word in enumerate(vocabulary)}
    sums = np.zeros(
        (len(vocabulary), features.peak_embeddings.shape[2]),
        dtype=np.float32,
    )
    weights = np.zeros(len(vocabulary), dtype=np.float64)
    for row, document in enumerate(documents):
        mask = features.peak_mask[row]
        mz_values = features.peak_mz[row, mask]
        states = features.peak_embeddings[row, mask].astype(np.float32)
        if not len(mz_values):
            continue
        for word, count in Counter(map(str, document)).items():
            parsed = parse_spectral_word(word)
            column = columns.get(word)
            if parsed is None or column is None:
                continue
            word_type, value = parsed
            target = (
                value if word_type == "frag" else features.precursor_mz[row] - value
            )
            if target <= 0:
                continue
            peak = int(np.argmin(np.abs(mz_values - target)))
            if abs(float(mz_values[peak]) - target) <= mz_tolerance:
                sums[column] += float(count) * states[peak]
                weights[column] += float(count)
    return {
        str(word): sums[index] / np.float32(weights[index])
        for index, word in enumerate(vocabulary)
        if weights[index] > 0
    }
