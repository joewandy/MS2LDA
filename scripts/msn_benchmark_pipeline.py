"""Shared MSn benchmark preparation, modelling, and export utilities."""

from __future__ import annotations

import gzip
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF

if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC2VEC_DIR = REPO_ROOT / "MS2LDA" / "Add_On" / "Spec2Vec" / "model_positive_mode"
SPEC2VEC_MODEL = SPEC2VEC_DIR / "150225_Spec2Vec_pos_CleanedLibraries.model"
SPEC2VEC_EMBEDDINGS = (
    SPEC2VEC_DIR / "150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy"
)
SPEC2VEC_DB = SPEC2VEC_DIR / "150225_CombLibraries_spectra.db"

PREPROCESSING_PARAMETERS = {
    "min_mz": 0,
    "max_mz": 2000,
    "max_frags": 1000,
    "min_frags": 5,
    "min_intensity": 0.01,
    "max_intensity": 1,
}
DATASET_PARAMETERS = {
    "acquisition_type": "DDA",
    "charge": 1,
    "significant_digits": 2,
}
ANNOTATION_PARAMETERS = {
    "criterium": "best",
    "cosine_similarity": 0.90,
    "n_mols_retrieved": 10,
    "spec2vec_search_k": 1000,
}
LDA_ALPHA = 0.6
LDA_ETA = 0.1
LDA_TRAIN_STEP_SIZE = 50
BACKGROUND_WEIGHT = 0.05
TOPIC_OVERLAP_WEIGHT = 0.05
TOPIC_USAGE_WEIGHT = 0.1
THETA_EXPORT_POWER = 1.7
EPS = 1e-12
MEMBERSHIP_THRESHOLD = 0.5
MODEL_OUTPUT_FILENAMES = {
    "theta.npy",
    "beta.npy",
    "vocab.json",
    "train_history.json",
    "model_checkpoint.pt",
    "run_summary.json",
}
EXPORT_OUTPUT_FILENAMES = {
    "annotations.csv",
    "memberships.csv",
    "topic_diagnostics.csv",
    "export_summary.json",
}
CACHE_FILENAMES = {
    "bow.npz",
    "vocab.json",
    "spectra_metadata.csv",
    "documents.jsonl.gz",
    "cache_summary.json",
}
OUTPUT_FILENAMES = MODEL_OUTPUT_FILENAMES | EXPORT_OUTPUT_FILENAMES


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_spectra(path: Path) -> list:
    from MS2LDA.Preprocessing.load_and_clean import load_mgf, load_msp, load_mzml

    suffix = path.suffix.lower()
    if suffix == ".mgf":
        return list(load_mgf(str(path)))
    if suffix == ".mzml":
        return list(load_mzml(str(path)))
    if suffix == ".msp":
        return list(load_msp(str(path)))
    raise ValueError("Unsupported dataset format. Expected .mgf, .mzML, or .msp.")


def prepare_msn_documents(
    dataset: Path, limit_spectra: int | None
) -> tuple[list, list[list[str]], dict]:
    from MS2LDA.Preprocessing.generate_corpus import features_to_words
    from MS2LDA.Preprocessing.load_and_clean import clean_spectra

    loaded = load_spectra(dataset)
    cleaned = clean_spectra(loaded, PREPROCESSING_PARAMETERS.copy())
    documents = features_to_words(
        cleaned,
        significant_figures=DATASET_PARAMETERS["significant_digits"],
        acquisition_type=DATASET_PARAMETERS["acquisition_type"],
    )
    non_empty = [
        (spectrum, document)
        for spectrum, document in zip(cleaned, documents, strict=True)
        if document
    ]
    if limit_spectra is not None:
        non_empty = non_empty[: int(limit_spectra)]
    if not non_empty:
        raise ValueError("No non-empty spectra remained after preprocessing.")
    spectra, documents = zip(*non_empty, strict=True)
    metadata = {
        "loaded_spectra": len(loaded),
        "cleaned_spectra": len(cleaned),
        "documents": len(documents),
        "limit_spectra": limit_spectra,
    }
    return list(spectra), list(documents), metadata


def spectrum_metadata_value(spectrum, keys: list[str]) -> Any:
    for key in keys:
        try:
            value = spectrum.get(key)
        except AttributeError:
            value = None
        if value not in (None, ""):
            return value
    return ""


def spectra_to_metadata_frame(spectra: list) -> pd.DataFrame:
    rows = []
    for doc_index, spectrum in enumerate(spectra):
        rows.append(
            {
                "doc_index": int(doc_index),
                "smiles": spectrum_metadata_value(
                    spectrum,
                    ["smiles", "canonical_smiles", "inchikey_smiles"],
                ),
                "spectrum_id": spectrum_metadata_value(
                    spectrum,
                    ["spectrum_id", "feature_id", "scan", "scans"],
                ),
                "name": spectrum_metadata_value(
                    spectrum,
                    ["compound_name", "name", "title"],
                ),
                "inchikey": spectrum_metadata_value(
                    spectrum, ["inchikey", "inchikey_inchi"]
                ),
                "precursor_mz": spectrum_metadata_value(
                    spectrum,
                    ["precursor_mz", "pepmass", "parentmass"],
                ),
            }
        )
    return pd.DataFrame(rows)


def write_documents_jsonl(path: Path, documents: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for doc_index, document in enumerate(documents):
            handle.write(
                json.dumps(
                    {"doc_index": int(doc_index), "tokens": list(document)},
                    separators=(",", ":"),
                )
                + "\n"
            )


def read_documents_jsonl(path: Path) -> list[list[str]]:
    documents = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append([str(token) for token in row["tokens"]])
    return documents


def train_validation_test_split(
    n_items: int,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create deterministic non-overlapping train/validation/test indices."""
    split_names = ("train", "validation", "test")
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction],
        dtype=np.float64,
    )
    if int(n_items) < len(split_names):
        raise ValueError(
            "Need at least three spectra for a train/validation/test split."
        )
    if np.any(fractions <= 0):
        raise ValueError("All split fractions must be greater than zero.")
    fraction_sum = float(fractions.sum())
    if fraction_sum <= 0:
        raise ValueError("Split fractions must sum to a positive value.")

    raw_counts = int(n_items) * fractions / fraction_sum
    counts = np.floor(raw_counts).astype(int)
    remainders = raw_counts - counts
    order_for_extra = np.argsort(-remainders)
    for offset in range(int(n_items) - int(counts.sum())):
        counts[int(order_for_extra[offset % len(counts)])] += 1
    while np.any(counts == 0):
        zero_index = int(np.where(counts == 0)[0][0])
        donor_index = int(np.argmax(counts))
        if counts[donor_index] <= 1:
            raise ValueError("Cannot allocate non-empty train/validation/test splits.")
        counts[zero_index] += 1
        counts[donor_index] -= 1

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_items)).astype(np.int64)
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    return {
        "train_indices": np.sort(order[:train_end]).astype(np.int64),
        "validation_indices": np.sort(order[train_end:validation_end]).astype(np.int64),
        "test_indices": np.sort(order[validation_end:]).astype(np.int64),
    }


def split_indices_json_payload(splits: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: [int(value) for value in values] for key, values in splits.items()}


def build_bow_matrix_for_vocabulary(
    documents: list[list[str]],
    vocab: list[str],
) -> sparse.csr_matrix:
    """Build a BoW matrix using an already ordered vocabulary."""
    if not vocab:
        raise ValueError("Cannot build a BoW matrix with an empty vocabulary.")

    token_to_idx = {str(token): idx for idx, token in enumerate(vocab)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for doc_index, document in enumerate(documents):
        counts = Counter(str(token) for token in document)
        for token, count in counts.items():
            vocab_index = token_to_idx.get(token)
            if vocab_index is None:
                continue
            rows.append(doc_index)
            cols.append(vocab_index)
            data.append(float(count))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(documents), len(vocab)),
        dtype=np.float32,
    )


def prepare_input_cache(
    *,
    dataset: Path,
    out_dir: Path,
    limit_spectra: int | None,
    min_df: int,
    min_cf: float,
    rm_top: int,
    overwrite: bool,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clear_named_outputs(out_dir, CACHE_FILENAMES)

    spectra, documents, input_metadata = prepare_msn_documents(
        dataset,
        limit_spectra=limit_spectra,
    )
    x, vocab, bow_metadata = build_bow_matrix(
        documents,
        min_df=min_df,
        min_cf=min_cf,
        rm_top=rm_top,
    )
    metadata = spectra_to_metadata_frame(spectra)

    sparse.save_npz(out_dir / "bow.npz", x)
    write_json(out_dir / "vocab.json", {"vocab": vocab})
    metadata.to_csv(out_dir / "spectra_metadata.csv", index=False)
    write_documents_jsonl(out_dir / "documents.jsonl.gz", documents)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "preprocessing_parameters": PREPROCESSING_PARAMETERS,
        "dataset_parameters": DATASET_PARAMETERS,
        "vocabulary_parameters": {
            "min_df": int(min_df),
            "min_cf": float(min_cf),
            "rm_top": int(rm_top),
        },
        "input": {**input_metadata, **bow_metadata},
        "outputs": {
            "bow_npz": str(out_dir / "bow.npz"),
            "vocab_json": str(out_dir / "vocab.json"),
            "spectra_metadata_csv": str(out_dir / "spectra_metadata.csv"),
            "documents_jsonl_gz": str(out_dir / "documents.jsonl.gz"),
        },
    }
    write_json(out_dir / "cache_summary.json", summary)
    return summary


def load_input_cache(
    cache_dir: Path, *, require_documents: bool = False
) -> dict[str, Any]:
    bow_path = cache_dir / "bow.npz"
    vocab_path = cache_dir / "vocab.json"
    metadata_path = cache_dir / "spectra_metadata.csv"
    documents_path = cache_dir / "documents.jsonl.gz"
    summary_path = cache_dir / "cache_summary.json"
    missing = [
        path
        for path in [bow_path, vocab_path, metadata_path, summary_path]
        if not path.exists()
    ]
    if require_documents and not documents_path.exists():
        missing.append(documents_path)
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Input cache is incomplete:\n{missing_text}")

    vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    cache = {
        "matrix": sparse.load_npz(bow_path).tocsr(),
        "vocab": [str(token) for token in vocab_payload["vocab"]],
        "spectra_metadata": pd.read_csv(metadata_path, keep_default_na=False),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }
    if documents_path.exists() and require_documents:
        cache["documents"] = read_documents_jsonl(documents_path)
    return cache


def build_bow_matrix(
    documents: list[list[str]],
    *,
    min_df: int,
    min_cf: float,
    rm_top: int,
) -> tuple[sparse.csr_matrix, list[str], dict]:
    corpus_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    for document in documents:
        corpus_frequency.update(document)
        document_frequency.update(set(document))

    keep = [
        token
        for token, cf in corpus_frequency.items()
        if cf >= float(min_cf) and document_frequency[token] >= int(min_df)
    ]
    if rm_top > 0 and keep:
        top_tokens = {
            token
            for token, _ in sorted(
                ((token, corpus_frequency[token]) for token in keep),
                key=lambda item: (-item[1], item[0]),
            )[: int(rm_top)]
        }
        keep = [token for token in keep if token not in top_tokens]

    vocab = sorted(keep)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    nnz_before = 0
    for doc_index, document in enumerate(documents):
        counts = Counter(document)
        nnz_before += len(counts)
        for token, count in counts.items():
            vocab_index = token_to_idx.get(token)
            if vocab_index is None:
                continue
            rows.append(doc_index)
            cols.append(vocab_index)
            data.append(float(count))

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(documents), len(vocab)),
        dtype=np.float32,
    )
    if matrix.shape[1] == 0 or matrix.nnz == 0:
        raise ValueError("BoW matrix is empty after vocabulary filtering.")

    metadata = {
        "documents": len(documents),
        "vocab_size": len(vocab),
        "nnz_before_filtering": int(nnz_before),
        "nnz_after_filtering": int(matrix.nnz),
        "min_df": int(min_df),
        "min_cf": float(min_cf),
        "rm_top": int(rm_top),
    }
    return matrix, vocab, metadata


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    denom = matrix.sum(axis=1, keepdims=True)
    out = np.divide(matrix, denom + EPS)
    zero_rows = np.where(denom.ravel() <= EPS)[0]
    if zero_rows.size:
        out[zero_rows, :] = 1.0 / max(matrix.shape[1], 1)
    return out.astype(np.float32, copy=False)


def sharpen_theta(theta: np.ndarray, power: float) -> np.ndarray:
    if float(power) == 1.0:
        return np.asarray(theta, dtype=np.float32)
    if float(power) <= 0:
        raise ValueError("--theta-export-power must be greater than 0.")
    return normalize_rows(np.power(np.asarray(theta, dtype=np.float64), float(power)))


def bow_background_distribution(x: sparse.csr_matrix) -> np.ndarray:
    counts = np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
    counts = counts + EPS
    return normalize_rows(counts[None, :])[0]


def sparsemax(logits, dim: int = -1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted = z.sort(dim=dim, descending=True).values
    z_cumsum = z_sorted.cumsum(dim)
    range_shape = [1] * z.dim()
    range_shape[dim] = z.size(dim)
    support_range = logits.new_tensor(
        np.arange(1, z.size(dim) + 1),
        dtype=logits.dtype,
    ).view(range_shape)
    support = 1 + support_range * z_sorted > z_cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_index = support_size - 1
    tau = (z_cumsum.gather(dim, tau_index) - 1) / support_size.to(logits.dtype)
    return (z - tau).clamp_min(0)


def activate_topic_distribution(logits, *, activation: str, dim: int = -1):
    if activation == "sparsemax":
        return sparsemax(logits, dim=dim)
    if activation == "softmax":
        return logits.softmax(dim=dim)
    raise ValueError(f"Unsupported topic activation: {activation}")


def run_kl_nmf(
    x: sparse.csr_matrix,
    *,
    n_motifs: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dense = x.toarray().astype(np.float32, copy=False)
    init = "nndsvda" if int(n_motifs) <= min(dense.shape) else "random"
    model = NMF(
        n_components=int(n_motifs),
        init=init,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    w = model.fit_transform(dense)
    h = model.components_
    metadata = {
        "nmf_init": init,
        "nmf_max_iter": int(max_iter),
        "nmf_n_iter": int(getattr(model, "n_iter_", 0)),
        "nmf_reconstruction_err": float(getattr(model, "reconstruction_err_", np.nan)),
    }
    return normalize_rows(w), normalize_rows(h), metadata


def run_tomotopy_lda(
    documents: list[list[str]],
    *,
    n_motifs: int,
    min_df: int,
    min_cf: float,
    rm_top: int,
    lda_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, float]], dict]:
    model, history, metadata = train_tomotopy_lda_model(
        documents,
        n_motifs=n_motifs,
        min_df=min_df,
        min_cf=min_cf,
        rm_top=rm_top,
        lda_iterations=lda_iterations,
        seed=seed,
    )
    theta, beta, vocab = tomotopy_lda_model_outputs(model)
    return theta, beta, vocab, history, metadata


def train_tomotopy_lda_model(
    documents: list[list[str]],
    *,
    n_motifs: int,
    min_df: int,
    min_cf: float,
    rm_top: int,
    lda_iterations: int,
    seed: int,
):
    try:
        import tomotopy as tp
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "tomotopy is required for --model lda. Use the MS2LDA conda environment."
        ) from err

    model = tp.LDAModel(
        k=int(n_motifs),
        min_df=int(min_df),
        min_cf=int(min_cf),
        rm_top=int(rm_top),
        alpha=LDA_ALPHA,
        eta=LDA_ETA,
        seed=int(seed),
    )
    for document in documents:
        model.add_doc(document)

    history = []
    trained = 0
    while trained < int(lda_iterations):
        step = min(LDA_TRAIN_STEP_SIZE, int(lda_iterations) - trained)
        model.train(step, workers=1, parallel=1)
        trained += step
        history.append(
            {
                "iteration": float(trained),
                "ll_per_word": float(model.ll_per_word),
                "perplexity": float(model.perplexity),
            }
        )

    if not list(model.used_vocabs):
        raise ValueError("Tomotopy LDA vocabulary is empty after filtering.")

    metadata = {
        "lda_iterations": int(lda_iterations),
        "lda_k": int(model.k),
        "lda_alpha": LDA_ALPHA,
        "lda_eta": LDA_ETA,
        "lda_min_df": int(min_df),
        "lda_min_cf": int(min_cf),
        "lda_rm_top": int(rm_top),
        "lda_ll_per_word": float(model.ll_per_word),
        "lda_perplexity": float(model.perplexity),
    }
    return model, history, metadata


def tomotopy_lda_model_outputs(model) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocab = list(model.used_vocabs)
    if not vocab:
        raise ValueError("Tomotopy LDA vocabulary is empty after filtering.")
    theta = normalize_rows(
        np.vstack(
            [np.asarray(doc.get_topic_dist(), dtype=np.float32) for doc in model.docs]
        )
    )
    beta = normalize_rows(
        np.vstack(
            [
                np.asarray(model.get_topic_word_dist(topic_id), dtype=np.float32)
                for topic_id in range(int(model.k))
            ]
        )
    )
    return theta, beta, vocab


def infer_tomotopy_lda_theta(
    model,
    documents: list[list[str]],
    *,
    iterations: int = 100,
) -> tuple[np.ndarray, dict[str, int | float]]:
    vocab = set(str(token) for token in model.used_vocabs)
    rows = []
    log_likelihoods = []
    empty_documents = 0
    for document in documents:
        filtered = [str(token) for token in document if str(token) in vocab]
        if not filtered:
            empty_documents += 1
            rows.append(np.full(int(model.k), 1.0 / float(model.k), dtype=np.float32))
            continue
        doc = model.make_doc(filtered)
        topic_dist, log_likelihood = model.infer(
            doc,
            iter=int(iterations),
            workers=1,
            parallel=1,
        )
        rows.append(np.asarray(topic_dist, dtype=np.float32))
        log_likelihoods.append(float(log_likelihood))
    theta = (
        normalize_rows(np.vstack(rows))
        if rows
        else np.empty((0, int(model.k)), dtype=np.float32)
    )
    metadata = {
        "documents": int(len(documents)),
        "empty_documents": int(empty_documents),
        "inference_iterations": int(iterations),
        "mean_log_likelihood": (
            float(np.mean(log_likelihoods)) if log_likelihoods else 0.0
        ),
    }
    return theta, metadata


def topic_usage_loss(theta, *, mode: str = "mse"):
    mean_theta = theta.mean(dim=0)
    if mode == "mse":
        return ((mean_theta - (1.0 / theta.shape[1])) ** 2).mean()
    if mode == "entropy":
        if theta.shape[1] <= 1:
            return mean_theta.new_tensor(0.0)
        entropy = -(mean_theta * mean_theta.clamp_min(EPS).log()).sum()
        return 1.0 - (entropy / mean_theta.new_tensor(np.log(theta.shape[1])))
    raise ValueError(f"Unsupported topic usage mode: {mode}")


def membership_count_diagnostics(
    theta: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.99),
) -> dict[str, dict[str, int | float]]:
    diagnostics = {}
    for threshold in thresholds:
        counts = (theta >= float(threshold)).sum(axis=0)
        nonzero = counts[counts > 0]
        diagnostics[f"{threshold:g}"] = {
            "active_topics": int((counts > 0).sum()),
            "total_memberships": int(counts.sum()),
            "topics_0": int((counts == 0).sum()),
            "topics_1": int((counts == 1).sum()),
            "topics_2_4": int(((counts >= 2) & (counts <= 4)).sum()),
            "topics_5_7": int(((counts >= 5) & (counts <= 7)).sum()),
            "topics_8_10": int(((counts >= 8) & (counts <= 10)).sum()),
            "topics_11_plus": int((counts > 10).sum()),
            "mean_nonzero_memberships": float(nonzero.mean()) if len(nonzero) else 0.0,
            "median_nonzero_memberships": (
                float(np.median(nonzero)) if len(nonzero) else 0.0
            ),
        }
    return diagnostics


def beta_target_support_loss(beta, *, target_support: float):
    if float(target_support) <= 0:
        return beta.new_tensor(0.0)
    target_entropy = beta.new_tensor(np.log(float(target_support)))
    beta_entropy = -(beta * beta.clamp_min(EPS).log()).sum(dim=1)
    return ((beta_entropy - target_entropy) ** 2).mean()


class SparseNeuralMS2LDA:
    def __init__(
        self,
        *,
        vocab_size: int,
        n_motifs: int,
        hidden_size: int,
        dropout: float,
        beta_logits_init: np.ndarray,
        background: np.ndarray,
        theta_activation: str,
        beta_activation: str,
        seed: int,
        device: str,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "PyTorch is required for --model sparse-neural."
            ) from err

        torch.manual_seed(int(seed))
        self.torch = torch
        self.device = device
        self.theta_activation = theta_activation
        self.beta_activation = beta_activation

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(vocab_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, n_motifs),
                )
                self.beta_logits = nn.Parameter(
                    torch.from_numpy(np.asarray(beta_logits_init, dtype=np.float32))
                )
                self.register_buffer(
                    "background",
                    torch.from_numpy(np.asarray(background, dtype=np.float32)),
                )

            def theta(self, x_norm):
                return activate_topic_distribution(
                    self.encoder(x_norm),
                    activation=theta_activation,
                    dim=1,
                )

            def beta(self):
                return activate_topic_distribution(
                    self.beta_logits,
                    activation=beta_activation,
                    dim=1,
                )

        self.model = _Model().to(device)

    def train(
        self,
        x: sparse.csr_matrix,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        theta_entropy_weight: float,
        beta_entropy_weight: float,
        background_weight: float,
        topic_overlap_weight: float,
        topic_usage_weight: float,
        topic_usage_mode: str,
        seed: int,
    ) -> list[dict[str, float]]:
        torch = self.torch
        rng = np.random.default_rng(int(seed))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        history = []
        n_docs = x.shape[0]

        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(n_docs)
            sums = {
                "loss": 0.0,
                "reconstruction": 0.0,
                "theta_entropy": 0.0,
                "beta_entropy": 0.0,
                "topic_overlap": 0.0,
                "topic_usage": 0.0,
                "theta_support": 0.0,
                "beta_support": 0.0,
            }
            batches = 0
            for start in range(0, n_docs, int(batch_size)):
                indices = order[start : start + int(batch_size)]
                xb_np = x[indices].toarray().astype(np.float32, copy=False)
                xb = torch.from_numpy(xb_np).to(self.device)
                x_norm = xb / xb.sum(dim=1, keepdim=True).clamp_min(EPS)

                theta = self.model.theta(x_norm)
                beta = self.model.beta()
                reconstruction = self._reconstruction_loss(
                    xb,
                    theta,
                    beta,
                    background_weight=background_weight,
                )
                theta_entropy = (
                    -(theta * torch.log(theta.clamp_min(EPS))).sum(dim=1).mean()
                )
                beta_entropy = (
                    -(beta * torch.log(beta.clamp_min(EPS))).sum(dim=1).mean()
                )
                topic_overlap = (
                    self._topic_overlap(beta)
                    if float(topic_overlap_weight) != 0.0
                    else beta.new_tensor(0.0)
                )
                usage = topic_usage_loss(theta, mode=topic_usage_mode)
                loss = (
                    reconstruction
                    + (float(theta_entropy_weight) * theta_entropy)
                    + (float(beta_entropy_weight) * beta_entropy)
                    + (float(topic_overlap_weight) * topic_overlap)
                    + (float(topic_usage_weight) * usage)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                sums["loss"] += float(loss.detach().cpu())
                sums["reconstruction"] += float(reconstruction.detach().cpu())
                sums["theta_entropy"] += float(theta_entropy.detach().cpu())
                sums["beta_entropy"] += float(beta_entropy.detach().cpu())
                sums["topic_overlap"] += float(topic_overlap.detach().cpu())
                sums["topic_usage"] += float(usage.detach().cpu())
                sums["theta_support"] += float(
                    (theta > 1e-8).sum(dim=1).float().mean().cpu()
                )
                sums["beta_support"] += float(
                    (beta > 1e-8).sum(dim=1).float().mean().cpu()
                )
                batches += 1

            row = {
                "epoch": epoch,
                **{key: value / batches for key, value in sums.items()},
            }
            history.append(row)
            print(json.dumps(row))
        return history

    def _reconstruction_loss(self, xb, theta, beta, *, background_weight: float):
        motif_dist = theta @ beta
        background = self.model.background.unsqueeze(0)
        mixed_dist = (
            (1.0 - float(background_weight)) * motif_dist
            + float(background_weight) * background
        ).clamp_min(EPS)
        mixed_dist = mixed_dist / mixed_dist.sum(dim=1, keepdim=True).clamp_min(EPS)
        doc_total = xb.sum(dim=1, keepdim=True).clamp_min(EPS)
        rate = (doc_total * mixed_dist).clamp_min(EPS)
        return (rate - (xb * self.torch.log(rate))).sum(dim=1).mean()

    def _topic_overlap(self, beta):
        gram = beta @ beta.T
        if gram.shape[0] <= 1:
            return gram.new_tensor(0.0)
        off_diagonal = gram - self.torch.diag(self.torch.diag(gram))
        return off_diagonal.sum() / (gram.shape[0] * (gram.shape[0] - 1))

    def infer_theta(self, x: sparse.csr_matrix, *, batch_size: int) -> np.ndarray:
        torch = self.torch
        rows = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], int(batch_size)):
                xb_np = (
                    x[start : start + int(batch_size)]
                    .toarray()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )
                xb = torch.from_numpy(xb_np).to(self.device)
                x_norm = xb / xb.sum(dim=1, keepdim=True).clamp_min(EPS)
                rows.append(self.model.theta(x_norm).detach().cpu().numpy())
        return np.vstack(rows).astype(np.float32, copy=False)

    def beta(self) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            return self.model.beta().detach().cpu().numpy().astype(np.float32)

    def state_dict(self) -> dict:
        return self.model.state_dict()


def train_sparse_neural(
    x: sparse.csr_matrix,
    *,
    n_motifs: int,
    hidden_size: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    theta_entropy_weight: float,
    beta_entropy_weight: float,
    background_weight: float,
    topic_overlap_weight: float,
    topic_usage_weight: float,
    seed: int = 42,
    device: str = "cpu",
    theta_activation: str = "sparsemax",
    beta_activation: str = "sparsemax",
    topic_usage_mode: str = "mse",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict, dict]:
    rng = np.random.default_rng(int(seed))
    beta_logits_init = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(int(n_motifs), int(x.shape[1])),
    ).astype(np.float32)
    background = bow_background_distribution(x)
    model = SparseNeuralMS2LDA(
        vocab_size=x.shape[1],
        n_motifs=n_motifs,
        hidden_size=hidden_size,
        dropout=dropout,
        beta_logits_init=beta_logits_init,
        background=background,
        theta_activation=theta_activation,
        beta_activation=beta_activation,
        seed=seed,
        device=device,
    )
    history = model.train(
        x,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        theta_entropy_weight=theta_entropy_weight,
        beta_entropy_weight=beta_entropy_weight,
        background_weight=background_weight,
        topic_overlap_weight=topic_overlap_weight,
        topic_usage_weight=topic_usage_weight,
        topic_usage_mode=topic_usage_mode,
        seed=seed,
    )
    theta = model.infer_theta(x, batch_size=batch_size)
    beta = normalize_rows(model.beta())
    checkpoint = {
        "model_type": "sparse_neural_ms2lda",
        "state_dict": model.state_dict(),
        "background": background,
    }
    metadata = {
        "sparse_neural_init": "random",
        "beta_logits_init": "normal(loc=0.0, scale=1.0)",
        "background_weight": float(background_weight),
        "topic_overlap_weight": float(topic_overlap_weight),
        "topic_usage_weight": float(topic_usage_weight),
        "theta_activation": theta_activation,
        "beta_activation": beta_activation,
        "topic_usage_mode": topic_usage_mode,
    }
    return theta, beta, history, metadata, checkpoint


class NeuralLDAModel:
    def __init__(
        self,
        *,
        n_docs: int,
        vocab_size: int,
        n_motifs: int,
        beta_logits_init: np.ndarray,
        background: np.ndarray,
        theta_init_strength: float,
        seed: int,
        device: str,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "PyTorch is required for --model neural-lda."
            ) from err

        torch.manual_seed(int(seed))
        self.torch = torch
        self.device = device

        rng = np.random.default_rng(int(seed))
        theta_logits_init = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(int(n_docs), int(n_motifs)),
        ).astype(np.float32)
        topic_init = rng.integers(0, int(n_motifs), size=int(n_docs))
        theta_logits_init[np.arange(int(n_docs)), topic_init] += float(
            theta_init_strength
        )

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta_logits = nn.Parameter(torch.from_numpy(theta_logits_init))
                self.beta_logits = nn.Parameter(
                    torch.from_numpy(np.asarray(beta_logits_init, dtype=np.float32))
                )
                self.register_buffer(
                    "background",
                    torch.from_numpy(np.asarray(background, dtype=np.float32)),
                )

            def theta(self, indices=None):
                logits = (
                    self.theta_logits if indices is None else self.theta_logits[indices]
                )
                return logits.softmax(dim=1)

            def beta(self):
                return self.beta_logits.softmax(dim=1)

        self.model = _Model().to(device)

    def train(
        self,
        x: sparse.csr_matrix,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        theta_entropy_weight: float,
        topic_usage_weight: float,
        beta_target_support: float,
        beta_target_weight: float,
        background_weight: float,
        seed: int,
    ) -> list[dict[str, float]]:
        torch = self.torch
        rng = np.random.default_rng(int(seed))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        history = []
        n_docs = x.shape[0]
        usage_sample_size = min(n_docs, 4096)

        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(n_docs)
            sums = {
                "loss": 0.0,
                "reconstruction": 0.0,
                "theta_entropy": 0.0,
                "topic_usage": 0.0,
                "beta_target": 0.0,
                "theta_support": 0.0,
                "beta_effective_support": 0.0,
            }
            batches = 0
            for start in range(0, n_docs, int(batch_size)):
                indices_np = order[start : start + int(batch_size)]
                xb_np = x[indices_np].toarray().astype(np.float32, copy=False)
                xb = torch.from_numpy(xb_np).to(self.device)
                x_norm = xb / xb.sum(dim=1, keepdim=True).clamp_min(EPS)
                indices = torch.from_numpy(indices_np).to(self.device)

                theta = self.model.theta(indices)
                beta = self.model.beta()
                reconstruction = self._reconstruction_loss(
                    x_norm,
                    theta,
                    beta,
                    background_weight=background_weight,
                )
                theta_entropy = -(theta * theta.clamp_min(EPS).log()).sum(dim=1).mean()
                if usage_sample_size == n_docs:
                    usage_indices = None
                else:
                    usage_indices_np = rng.choice(
                        n_docs,
                        size=usage_sample_size,
                        replace=False,
                    )
                    usage_indices = torch.from_numpy(usage_indices_np).to(self.device)
                usage_theta = self.model.theta(usage_indices)
                usage = topic_usage_loss(usage_theta, mode="entropy")
                beta_target = beta_target_support_loss(
                    beta,
                    target_support=beta_target_support,
                )
                loss = (
                    reconstruction
                    + (float(theta_entropy_weight) * theta_entropy)
                    + (float(topic_usage_weight) * usage)
                    + (float(beta_target_weight) * beta_target)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                beta_entropy = -(beta * beta.clamp_min(EPS).log()).sum(dim=1)
                sums["loss"] += float(loss.detach().cpu())
                sums["reconstruction"] += float(reconstruction.detach().cpu())
                sums["theta_entropy"] += float(theta_entropy.detach().cpu())
                sums["topic_usage"] += float(usage.detach().cpu())
                sums["beta_target"] += float(beta_target.detach().cpu())
                sums["theta_support"] += float(
                    (theta >= 0.01).sum(dim=1).float().mean().cpu()
                )
                sums["beta_effective_support"] += float(
                    beta_entropy.exp().mean().detach().cpu()
                )
                batches += 1

            row = {
                "epoch": epoch,
                **{key: value / batches for key, value in sums.items()},
            }
            history.append(row)
            print(json.dumps(row))
        return history

    def _reconstruction_loss(self, x_norm, theta, beta, *, background_weight: float):
        motif_dist = theta @ beta
        mixed_dist = (
            (1.0 - float(background_weight)) * motif_dist
            + float(background_weight) * self.model.background.unsqueeze(0)
        ).clamp_min(EPS)
        mixed_dist = mixed_dist / mixed_dist.sum(dim=1, keepdim=True).clamp_min(EPS)
        return -(x_norm * mixed_dist.log()).sum(dim=1).mean()

    def theta(self, *, batch_size: int) -> np.ndarray:
        rows = []
        self.model.eval()
        with self.torch.no_grad():
            n_docs = self.model.theta_logits.shape[0]
            for start in range(0, n_docs, int(batch_size)):
                indices = self.torch.arange(
                    start,
                    min(start + int(batch_size), n_docs),
                    device=self.device,
                )
                rows.append(self.model.theta(indices).detach().cpu().numpy())
        return np.vstack(rows).astype(np.float32, copy=False)

    def beta(self) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            return self.model.beta().detach().cpu().numpy().astype(np.float32)

    def state_dict(self) -> dict:
        return self.model.state_dict()


def train_neural_lda(
    x: sparse.csr_matrix,
    *,
    n_motifs: int,
    epochs: int,
    batch_size: int,
    lr: float,
    theta_entropy_weight: float,
    topic_usage_weight: float,
    beta_target_support: float,
    beta_target_weight: float,
    background_weight: float,
    theta_init_strength: float,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict, dict]:
    rng = np.random.default_rng(int(seed))
    beta_logits_init = rng.normal(
        loc=0.0,
        scale=0.5,
        size=(int(n_motifs), int(x.shape[1])),
    ).astype(np.float32)
    background = bow_background_distribution(x)
    model = NeuralLDAModel(
        n_docs=x.shape[0],
        vocab_size=x.shape[1],
        n_motifs=n_motifs,
        beta_logits_init=beta_logits_init,
        background=background,
        theta_init_strength=theta_init_strength,
        seed=seed,
        device=device,
    )
    history = model.train(
        x,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        theta_entropy_weight=theta_entropy_weight,
        topic_usage_weight=topic_usage_weight,
        beta_target_support=beta_target_support,
        beta_target_weight=beta_target_weight,
        background_weight=background_weight,
        seed=seed,
    )
    theta = model.theta(batch_size=batch_size)
    beta = normalize_rows(model.beta())
    checkpoint = {
        "model_type": "neural_lda",
        "state_dict": model.state_dict(),
        "background": background,
    }
    metadata = {
        "neural_lda_init": "document_topic_bias",
        "theta_init_strength": float(theta_init_strength),
        "beta_logits_init": "normal(loc=0.0, scale=0.5)",
        "theta_activation": "softmax",
        "beta_activation": "softmax",
        "background_weight": float(background_weight),
        "topic_usage_mode": "entropy",
        "topic_usage_weight": float(topic_usage_weight),
        "topic_usage_sample_size": int(min(x.shape[0], 4096)),
        "theta_entropy_weight": float(theta_entropy_weight),
        "beta_target_support": float(beta_target_support),
        "beta_target_weight": float(beta_target_weight),
    }
    return theta, beta, history, metadata, checkpoint


def spectrum_smiles(spectrum) -> str:
    for key in ["smiles", "canonical_smiles", "inchikey_smiles"]:
        value = spectrum.get(key)
        if value:
            return str(value)
    return ""


def select_eval_topic_ids(
    theta: np.ndarray,
    beta: np.ndarray,
    *,
    max_eval_motifs: int,
    membership_threshold: float,
) -> list[int]:
    counts = (theta >= float(membership_threshold)).sum(axis=0)
    maxima = theta.max(axis=0)
    beta_maxima = beta.max(axis=1)
    topic_ids = list(range(theta.shape[1]))
    topic_ids.sort(
        key=lambda topic: (
            -int(counts[topic]),
            -float(maxima[topic]),
            -float(beta_maxima[topic]),
            topic,
        )
    )
    if max_eval_motifs > 0:
        topic_ids = topic_ids[: int(max_eval_motifs)]
    return [int(topic) for topic in topic_ids if maxima[topic] > 0.0]


def export_memberships(
    theta: np.ndarray,
    spectra_or_metadata: list | pd.DataFrame,
    topic_ids: list[int],
    *,
    membership_threshold: float,
) -> pd.DataFrame:
    topic_set = set(int(topic) for topic in topic_ids)
    rows = []
    if isinstance(spectra_or_metadata, pd.DataFrame):
        smiles_values = spectra_or_metadata["smiles"].tolist()
    else:
        smiles_values = [spectrum_smiles(spectrum) for spectrum in spectra_or_metadata]

    for doc_index, smiles in enumerate(smiles_values):
        if not smiles:
            continue
        for topic_id, score in enumerate(theta[doc_index]):
            if topic_id not in topic_set or float(score) < float(membership_threshold):
                continue
            rows.append(
                {
                    "motif_id": f"motif_{topic_id}",
                    "smiles": smiles,
                    "membership_score": float(score),
                }
            )
    return pd.DataFrame(rows, columns=["motif_id", "smiles", "membership_score"])


def require_spec2vec_assets() -> None:
    missing = [
        path
        for path in [SPEC2VEC_MODEL, SPEC2VEC_EMBEDDINGS, SPEC2VEC_DB]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing positive-mode Spec2Vec assets. Run `ms2lda --only-download` first.\n"
            f"{missing_text}"
        )


def annotate_topics_from_beta(
    beta: np.ndarray,
    vocab: list[str],
    topic_ids: list[int],
    *,
    motif_top_n: int,
    motifset: str,
) -> tuple[pd.DataFrame, dict]:
    from MS2LDA.Add_On.Spec2Vec.annotation import (
        calc_embeddings,
        calc_similarity_faiss,
        get_library_matches,
        load_s2v_model,
    )
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
        hit_clustering,
        motif_optimization,
    )
    from MS2LDA.utils import create_spectrum

    require_spec2vec_assets()
    motif_spectra = []
    motif_topic_ids = []
    for topic_id in topic_ids:
        row = beta[int(topic_id)]
        top_n = min(int(motif_top_n), row.shape[0])
        if top_n <= 0:
            continue
        top_idx = np.argpartition(-row, top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        words = [(vocab[int(idx)], float(row[int(idx)])) for idx in top_idx]
        motif_spectra.append(
            create_spectrum(
                words,
                int(topic_id),
                charge=DATASET_PARAMETERS["charge"],
                motifset=motifset,
                significant_digits=DATASET_PARAMETERS["significant_digits"],
            )
        )
        motif_topic_ids.append(int(topic_id))

    clustered_by_topic = {int(topic_id): [] for topic_id in topic_ids}
    optimized_motif_count = 0
    if motif_spectra:
        s2v_similarity = load_s2v_model(str(SPEC2VEC_MODEL))
        embeddings = np.load(SPEC2VEC_EMBEDDINGS)
        motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
        search_k = min(
            int(ANNOTATION_PARAMETERS["spec2vec_search_k"]), embeddings.shape[0]
        )
        similarities, indices = calc_similarity_faiss(
            motif_embeddings,
            embeddings,
            k=search_k,
        )
        library_matches = get_library_matches(
            similarities=similarities,
            indices=indices,
            db_path=str(SPEC2VEC_DB),
            top_n=ANNOTATION_PARAMETERS["n_mols_retrieved"],
            unique_mols=True,
        )
        clustered_spectra, clustered_smiles, _ = hit_clustering(
            s2v_similarity=s2v_similarity,
            motif_spectra=motif_spectra,
            library_matches=library_matches,
            criterium=ANNOTATION_PARAMETERS["criterium"],
            cosine_similarity=ANNOTATION_PARAMETERS["cosine_similarity"],
        )
        optimized_motifs = motif_optimization(
            motif_spectra,
            clustered_spectra,
            clustered_smiles,
            loss_err=1,
        )
        optimized_motif_count = len(optimized_motifs)
        for topic_id, smiles in zip(motif_topic_ids, clustered_smiles, strict=True):
            clustered_by_topic[int(topic_id)] = [str(value) for value in smiles]

    annotations = pd.DataFrame(
        {
            "motif_id": [f"motif_{topic_id}" for topic_id in topic_ids],
            "annotation_smiles": [
                "|".join(clustered_by_topic[int(topic_id)]) for topic_id in topic_ids
            ],
        }
    )
    metadata = {
        "requested_motifs": len(topic_ids),
        "motif_spectra": len(motif_spectra),
        "optimized_motif_count": optimized_motif_count,
    }
    return annotations, metadata


def entropy_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return -(matrix * np.log(matrix + EPS)).sum(axis=1)


def write_topic_diagnostics(
    path: Path,
    theta: np.ndarray,
    beta: np.ndarray,
    vocab: list[str],
    *,
    membership_threshold: float,
    top_n: int = 10,
) -> None:
    beta_entropy = entropy_rows(beta)
    theta_counts = (theta >= float(membership_threshold)).sum(axis=0)
    rows = []
    for topic_id in range(beta.shape[0]):
        row = beta[topic_id]
        n = min(int(top_n), row.shape[0])
        top_idx = np.argpartition(-row, n - 1)[:n]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        rows.append(
            {
                "motif_id": f"motif_{topic_id}",
                "topic_id": topic_id,
                "beta_entropy": float(beta_entropy[topic_id]),
                "beta_effective_support": float(np.exp(beta_entropy[topic_id])),
                "beta_max_probability": float(row.max()),
                "theta_memberships_above_threshold": int(theta_counts[topic_id]),
                "theta_max_probability": float(theta[:, topic_id].max()),
                "top_words": "|".join(vocab[int(idx)] for idx in top_idx),
                "top_word_probabilities": "|".join(
                    f"{float(row[int(idx)]):.6g}" for idx in top_idx
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_checkpoint(path: Path, checkpoint: dict | None) -> None:
    if checkpoint is None:
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.save(checkpoint, path)


def clear_named_outputs(out_dir: Path, filenames: set[str]) -> None:
    for name in filenames:
        path = out_dir / name
        if path.exists():
            path.unlink()


def clear_expected_outputs(out_dir: Path) -> None:
    clear_named_outputs(out_dir, OUTPUT_FILENAMES)
