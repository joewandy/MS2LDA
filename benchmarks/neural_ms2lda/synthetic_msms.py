"""Truth-known short, sparse MS/MS simulator for neural topic-model screens."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from .data import train_token_features
from .reproducibility import (
    configure_deterministic_execution,
    read_json_object,
    sha256_file,
)
from .utils import atomic_save_numpy, read_json, write_json

LOOSE_RECOVERY_COSINE = 0.20
PRIMARY_RECOVERY_COSINE = 0.50
ACTIVE_TOPIC_USAGE_THRESHOLD = 0.005
EVALUATION_PROTOCOL = {
    "active_topic_usage_threshold": 0.0005,
    "duplicate_cosine_thresholds": (0.95, 0.99, 0.999),
    "catastrophic_duplicate_component_fraction": 0.5,
    "top_word_count": 20,
    "channel_extreme_lower": 0.1,
    "channel_extreme_upper": 0.9,
}
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


@dataclass(frozen=True)
class SyntheticPeak:
    """One paired fragment/loss observation after intensity discretization."""

    fragment: str
    loss: str
    count: int
    topic: int | None


SyntheticDocument: TypeAlias = tuple[SyntheticPeak, ...]


@dataclass(frozen=True)
class SyntheticMsmsDataset:
    """One train/validation simulation and its planted topic truth."""

    train: sp.csr_matrix
    validation_observed: sp.csr_matrix
    validation_completion: sp.csr_matrix
    validation_full: sp.csr_matrix
    vocabulary: tuple[str, ...]
    true_beta: np.ndarray
    train_true_theta: np.ndarray
    validation_true_theta: np.ndarray
    validation_records: tuple[dict[str, int | str], ...]
    summary: dict[str, float | int]


def _token(prefix: str, mass: float) -> str:
    return f"{prefix}@{mass:.2f}"


def _motif_anchors(
    true_topics: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    shared_fragments = np.asarray(
        [73.03, 91.05, 105.07, 119.09, 133.10, 147.12],
        dtype=np.float64,
    )
    shared_losses = np.asarray(
        [18.01, 28.03, 44.03, 60.02, 79.04, 97.05],
        dtype=np.float64,
    )
    motifs: list[tuple[np.ndarray, np.ndarray]] = []
    for topic in range(true_topics):
        fragment_base = 52.0 + 13.1 * topic
        loss_base = 21.0 + 7.35 * topic
        fragments = np.asarray(
            [
                fragment_base + offset
                for offset in (0.11, 1.37, 2.63, 4.19, 5.43, 6.71, 8.03, 9.29)
            ]
            + [
                shared_fragments[topic % len(shared_fragments)],
                shared_fragments[(topic // 3 + 1) % len(shared_fragments)],
            ],
            dtype=np.float64,
        )
        losses = np.asarray(
            [
                loss_base + offset
                for offset in (0.23, 1.11, 2.47, 3.89, 5.17, 6.43, 7.79, 9.13)
            ]
            + [
                shared_losses[topic % len(shared_losses)],
                shared_losses[(topic // 3 + 2) % len(shared_losses)],
            ],
            dtype=np.float64,
        )
        motifs.append((fragments, losses))
    return motifs, shared_fragments, shared_losses


def _draw_peak_mass(
    rng: np.random.Generator,
    *,
    precursor: float,
    fragments: np.ndarray,
    losses: np.ndarray,
) -> tuple[float, float]:
    if rng.random() < 0.5:
        fragment = float(rng.choice(fragments))
        loss = precursor - fragment
    else:
        loss = float(rng.choice(losses))
        fragment = precursor - loss
    return fragment, loss


def _draw_document(
    rng: np.random.Generator,
    *,
    motif_anchors: list[tuple[np.ndarray, np.ndarray]],
    background_fragments: np.ndarray,
    background_losses: np.ndarray,
    prevalence: np.ndarray,
) -> SyntheticDocument:
    # A finite precursor grid represents recurring molecular standards while
    # still making complementary fragment/loss words substantially rarer than
    # the planted anchors. This is what allows train-only min-DF filtering to
    # retain only a minority of complementary observations.
    precursor = float(rng.choice(np.linspace(420.0, 720.0, 61)))
    active_count = int(rng.choice((1, 2, 3), p=(0.46, 0.39, 0.15)))
    active = rng.choice(
        len(motif_anchors),
        size=active_count,
        replace=False,
        p=prevalence,
    )
    mixture = rng.dirichlet(np.full(active_count, 0.45, dtype=np.float64))
    physical_peaks = int(rng.integers(18, 42))
    proposals: list[tuple[float, float, int | None, float]] = []

    # Guarantee direct evidence for every planted active motif before drawing
    # the remaining heterogeneous signal/background/noise peaks.
    for topic in active:
        fragments, losses = motif_anchors[int(topic)]
        fragment, loss = _draw_peak_mass(
            rng,
            precursor=precursor,
            fragments=fragments,
            losses=losses,
        )
        intensity = float(rng.lognormal(mean=0.0, sigma=0.60))
        proposals.append((fragment, loss, int(topic), intensity))

    while len(proposals) < physical_peaks:
        category = int(rng.choice((0, 1, 2), p=(0.78, 0.14, 0.08)))
        if category == 0:
            local = int(rng.choice(active_count, p=mixture))
            topic = int(active[local])
            fragments, losses = motif_anchors[topic]
            fragment, loss = _draw_peak_mass(
                rng,
                precursor=precursor,
                fragments=fragments,
                losses=losses,
            )
            scale = 1.0
        elif category == 1:
            topic = None
            fragment, loss = _draw_peak_mass(
                rng,
                precursor=precursor,
                fragments=background_fragments,
                losses=background_losses,
            )
            scale = 0.65
        else:
            topic = None
            fragment = float(rng.uniform(45.0, precursor - 20.0))
            loss = precursor - fragment
            scale = 0.35
        intensity = scale * float(rng.lognormal(mean=0.0, sigma=0.60))
        proposals.append((fragment, loss, topic, intensity))

    maximum = max(row[3] for row in proposals)
    peaks = []
    for fragment, loss, topic, intensity in proposals:
        count = max(1, int(np.rint(100.0 * intensity / maximum)))
        peaks.append(
            SyntheticPeak(
                fragment=_token("frag", fragment),
                loss=_token("loss", loss),
                count=count,
                topic=topic,
            ),
        )
    return tuple(peaks)


def _training_vocabulary(
    documents: list[SyntheticDocument],
    *,
    minimum_document_frequency: int,
) -> tuple[str, ...]:
    document_frequency: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for document in documents:
        words = {word for peak in document for word in (peak.fragment, peak.loss)}
        document_frequency.update(words)
        for peak in document:
            for word in (peak.fragment, peak.loss):
                first_seen.setdefault(word, len(first_seen))
    return tuple(
        sorted(
            (
                word
                for word, frequency in document_frequency.items()
                if frequency >= int(minimum_document_frequency)
            ),
            key=first_seen.__getitem__,
        ),
    )


def _matrix(
    documents: list[SyntheticDocument],
    vocabulary: tuple[str, ...],
) -> sp.csr_matrix:
    index = {word: column for column, word in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, document in enumerate(documents):
        counts: Counter[int] = Counter()
        for peak in document:
            for word in (peak.fragment, peak.loss):
                column = index.get(word)
                if column is not None:
                    counts[column] += peak.count
        rows.extend([row] * len(counts))
        columns.extend(counts)
        values.extend(map(float, counts.values()))
    matrix = sp.csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix


def _true_theta(
    documents: list[SyntheticDocument],
    true_topics: int,
) -> np.ndarray:
    theta = np.zeros((len(documents), true_topics), dtype=np.float64)
    for row, document in enumerate(documents):
        for peak in document:
            if peak.topic is not None:
                theta[row, peak.topic] += 2.0 * peak.count
    totals = theta.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise RuntimeError("synthetic document contains no planted motif evidence")
    return (theta / totals).astype(np.float32)


def _true_beta(
    documents: list[SyntheticDocument],
    vocabulary: tuple[str, ...],
    true_topics: int,
) -> np.ndarray:
    index = {word: column for column, word in enumerate(vocabulary)}
    beta = np.zeros((true_topics, len(vocabulary)), dtype=np.float64)
    for document in documents:
        for peak in document:
            if peak.topic is None:
                continue
            for word in (peak.fragment, peak.loss):
                column = index.get(word)
                if column is not None:
                    beta[peak.topic, column] += peak.count
    totals = beta.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise RuntimeError("a planted motif has no in-vocabulary training evidence")
    return (beta / totals).astype(np.float32)


def _completion_views(
    documents: list[SyntheticDocument],
    vocabulary: tuple[str, ...],
    *,
    seed: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix, tuple[dict[str, int | str], ...]]:
    observed: list[SyntheticDocument] = []
    completion: list[SyntheticDocument] = []
    vocabulary_set = set(vocabulary)
    records: list[dict[str, int | str]] = []
    for row, document in enumerate(documents):
        rng = np.random.default_rng(seed + 104729 * row)
        order = rng.permutation(len(document))
        cut = max(1, min(len(document) - 1, len(document) // 2))
        observed_indices = set(map(int, order[:cut]))
        left = tuple(
            peak for index, peak in enumerate(document) if index in observed_indices
        )
        right = tuple(
            peak for index, peak in enumerate(document) if index not in observed_indices
        )
        observed.append(left)
        completion.append(right)
        oov = sum(
            peak.count
            for peak in right
            for word in (peak.fragment, peak.loss)
            if word not in vocabulary_set
        )
        records.append(
            {
                "split": "validation",
                "completion_oov_tokens": int(oov),
            },
        )
    return (
        _matrix(observed, vocabulary),
        _matrix(completion, vocabulary),
        tuple(records),
    )


def generate_synthetic_msms(
    *,
    seed: int,
    true_topics: int = 18,
    training_documents: int = 800,
    validation_documents: int = 160,
    minimum_document_frequency: int = 3,
) -> SyntheticMsmsDataset:
    """Generate the frozen paired fragment/loss sparse-spectrum protocol."""
    if int(true_topics) <= 1:
        raise ValueError("true_topics must exceed one")
    if int(training_documents) <= 0 or int(validation_documents) <= 0:
        raise ValueError("synthetic document counts must be positive")
    rng = np.random.default_rng(int(seed))
    motif_anchors, shared_fragments, shared_losses = _motif_anchors(true_topics)
    background_fragments = np.concatenate(
        (shared_fragments, np.asarray([57.03, 69.04, 85.06, 101.07])),
    )
    background_losses = np.concatenate(
        (shared_losses, np.asarray([17.03, 31.02, 46.04, 63.03])),
    )
    prevalence = 1.0 / np.power(np.arange(1, true_topics + 1), 0.72)
    prevalence /= prevalence.sum()
    train_documents = [
        _draw_document(
            rng,
            motif_anchors=motif_anchors,
            background_fragments=background_fragments,
            background_losses=background_losses,
            prevalence=prevalence,
        )
        for _ in range(int(training_documents))
    ]
    validation = [
        _draw_document(
            rng,
            motif_anchors=motif_anchors,
            background_fragments=background_fragments,
            background_losses=background_losses,
            prevalence=prevalence,
        )
        for _ in range(int(validation_documents))
    ]
    vocabulary = _training_vocabulary(
        train_documents,
        minimum_document_frequency=minimum_document_frequency,
    )
    train = _matrix(train_documents, vocabulary)
    validation_full = _matrix(validation, vocabulary)
    observed, completion, records = _completion_views(
        validation,
        vocabulary,
        seed=int(seed) + 50021,
    )
    true_beta = _true_beta(train_documents, vocabulary, true_topics)
    train_theta = _true_theta(train_documents, true_topics)
    validation_theta = _true_theta(validation, true_topics)
    train_nonzero = np.diff(train.indptr)
    train_mass = np.asarray(train.sum(axis=1)).ravel()
    completion_invocab = float(completion.sum())
    completion_oov = float(
        sum(int(record["completion_oov_tokens"]) for record in records),
    )
    fragment_mask = np.asarray(
        [word.startswith("frag@") for word in vocabulary],
        dtype=bool,
    )
    summary: dict[str, float | int] = {
        "seed": int(seed),
        "true_topics": int(true_topics),
        "training_documents": int(training_documents),
        "validation_documents": int(validation_documents),
        "vocabulary_size": len(vocabulary),
        "median_train_physical_peaks": float(
            np.median([len(document) for document in train_documents]),
        ),
        "median_train_nonzero_words": float(np.median(train_nonzero)),
        "median_train_pseudocount_mass": float(np.median(train_mass)),
        "mean_train_pseudocount_mass": float(np.mean(train_mass)),
        "validation_completion_oov_fraction": float(
            completion_oov / max(completion_invocab + completion_oov, 1.0),
        ),
        "true_beta_mean_fragment_mass": float(
            true_beta[:, fragment_mask].sum(axis=1).mean(),
        ),
    }
    return SyntheticMsmsDataset(
        train=train,
        validation_observed=observed,
        validation_completion=completion,
        validation_full=validation_full,
        vocabulary=vocabulary,
        true_beta=true_beta,
        train_true_theta=train_theta,
        validation_true_theta=validation_theta,
        validation_records=records,
        summary=summary,
    )


def _save_dataset_artifacts(
    directory: Path,
    dataset: SyntheticMsmsDataset,
) -> dict[str, dict[str, Any]]:
    """Persist one truth-known dataset and hash every scientific input."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, matrix in {
        "train": dataset.train,
        "validation_observed": dataset.validation_observed,
        "validation_completion": dataset.validation_completion,
        "validation_full": dataset.validation_full,
    }.items():
        path = directory / f"{name}.npz"
        if not path.is_file():
            sp.save_npz(path, matrix, compressed=False)
    for name, array in {
        "true_beta": dataset.true_beta,
        "train_true_theta": dataset.train_true_theta,
        "validation_true_theta": dataset.validation_true_theta,
    }.items():
        path = directory / f"{name}.npy"
        if not path.is_file():
            atomic_save_numpy(path, array.astype(np.float32, copy=False))
    vocabulary_path = directory / "vocabulary.json"
    if not vocabulary_path.is_file():
        write_json(vocabulary_path, {"vocabulary": list(dataset.vocabulary)})
    records_path = directory / "validation_records.jsonl"
    if not records_path.is_file():
        with records_path.open("w", encoding="utf-8") as handle:
            for record in dataset.validation_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(directory / "summary.json", dataset.summary)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        path.name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    write_json(directory / "artifact_manifest.json", manifest)
    return manifest


def prepare_synthetic_seed(
    output_root: Path,
    *,
    seed: int,
    threads: int,
    training_documents: int,
    validation_documents: int,
) -> tuple[SyntheticMsmsDataset, np.ndarray, Path]:
    """Generate one seed and train or reuse its single train-only SGNS table."""
    dataset = generate_synthetic_msms(
        seed=seed,
        training_documents=training_documents,
        validation_documents=validation_documents,
    )
    seed_directory = output_root / "synthetic_artifacts" / f"seed_{seed}"
    _save_dataset_artifacts(seed_directory, dataset)
    protocol = read_json_object(PROTOCOL_PATH)
    configure_deterministic_execution(seed, threads)
    train_token_features(
        seed_directory / "token_features",
        dataset.train,
        list(dataset.vocabulary),
        protocol,
        seed=seed,
    )
    return load_prepared_synthetic_seed(output_root, seed=seed)


def load_prepared_synthetic_seed(
    output_root: Path,
    *,
    seed: int,
) -> tuple[SyntheticMsmsDataset, np.ndarray, Path]:
    """Verify and load one explicitly prepared synthetic dataset and SGNS table."""
    seed_directory = output_root / "synthetic_artifacts" / f"seed_{seed}"
    manifest_path = seed_directory / "artifact_manifest.json"
    feature_complete = seed_directory / "token_features/complete.json"
    if not manifest_path.is_file() or not feature_complete.is_file():
        raise FileNotFoundError(
            f"synthetic seed {seed} must be prepared before model training",
        )
    manifest = read_json(manifest_path)
    for name, record in manifest.items():
        path = seed_directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing synthetic input: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"synthetic input size changed: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"synthetic input hash changed: {path}")

    vocabulary = tuple(
        map(str, read_json(seed_directory / "vocabulary.json")["vocabulary"])
    )
    with (seed_directory / "validation_records.jsonl").open(encoding="utf-8") as handle:
        records = tuple(json.loads(line) for line in handle if line.strip())
    dataset = SyntheticMsmsDataset(
        train=sp.load_npz(seed_directory / "train.npz").tocsr(),
        validation_observed=sp.load_npz(
            seed_directory / "validation_observed.npz",
        ).tocsr(),
        validation_completion=sp.load_npz(
            seed_directory / "validation_completion.npz",
        ).tocsr(),
        validation_full=sp.load_npz(
            seed_directory / "validation_full.npz",
        ).tocsr(),
        vocabulary=vocabulary,
        true_beta=np.load(seed_directory / "true_beta.npy"),
        train_true_theta=np.load(seed_directory / "train_true_theta.npy"),
        validation_true_theta=np.load(seed_directory / "validation_true_theta.npy"),
        validation_records=records,
        summary=read_json(seed_directory / "summary.json"),
    )
    if int(dataset.summary["seed"]) != int(seed):
        raise ValueError("prepared synthetic seed does not match the requested seed")
    features_path = seed_directory / "token_features/features.npy"
    features = np.load(features_path).astype(np.float32, copy=False)
    embeddings = np.array(features[:, :-2], dtype=np.float32, copy=True)
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-8,
    )
    if embeddings.shape[0] != len(vocabulary):
        raise ValueError("synthetic SGNS rows do not match the vocabulary")
    return dataset, embeddings, seed_directory


def matched_truth_metrics(
    learned_beta: np.ndarray,
    learned_theta: np.ndarray,
    true_beta: np.ndarray,
    true_theta: np.ndarray,
) -> dict[str, Any]:
    """Align planted motifs one-to-one and report beta and theta recovery."""
    epsilon = 1e-12
    learned_norm = learned_beta / np.maximum(
        np.linalg.norm(learned_beta, axis=1, keepdims=True),
        epsilon,
    )
    true_norm = true_beta / np.maximum(
        np.linalg.norm(true_beta, axis=1, keepdims=True),
        epsilon,
    )
    similarity = true_norm @ learned_norm.T
    true_rows, learned_rows = linear_sum_assignment(-similarity)
    matched = similarity[true_rows, learned_rows]
    aligned = np.zeros_like(true_theta, dtype=np.float64)
    aligned[:, true_rows] = learned_theta[:, learned_rows]
    numerator = np.sum(true_theta * aligned, axis=1)
    denominator = np.linalg.norm(true_theta, axis=1) * np.linalg.norm(aligned, axis=1)
    theta_cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > epsilon,
    )
    top_n = min(20, learned_beta.shape[1])
    jaccards = []
    for truth, learned in zip(true_rows, learned_rows, strict=True):
        truth_top = set(np.argsort(-true_beta[truth], kind="stable")[:top_n])
        learned_top = set(np.argsort(-learned_beta[learned], kind="stable")[:top_n])
        jaccards.append(len(truth_top & learned_top) / len(truth_top | learned_top))
    return {
        "true_beta_matched_cosine_mean": float(matched.mean()),
        "true_beta_matched_cosine_median": float(np.median(matched)),
        "true_beta_matched_cosine_minimum": float(matched.min()),
        "true_beta_top20_jaccard_mean": float(np.mean(jaccards)),
        "true_theta_cosine_mean": float(theta_cosine.mean()),
        "true_theta_cosine_median": float(np.median(theta_cosine)),
        "top_planted_motif_accuracy": float(
            np.mean(np.argmax(true_theta, axis=1) == np.argmax(aligned, axis=1)),
        ),
        "planted_motifs_recovered_cosine_gt_0_20": int(
            np.sum(matched > LOOSE_RECOVERY_COSINE),
        ),
        "planted_motifs_recovered_cosine_ge_0_50": int(
            np.sum(matched >= PRIMARY_RECOVERY_COSINE),
        ),
        "matching": [
            {
                "true_topic": int(truth),
                "learned_topic": int(learned),
                "beta_cosine": float(similarity[truth, learned]),
            }
            for truth, learned in zip(true_rows, learned_rows, strict=True)
        ],
    }
