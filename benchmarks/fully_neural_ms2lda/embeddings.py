# ruff: noqa: N812, PLR0915, PLR2004
"""Training-only skip-gram negative-sampling token embeddings."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import corpus_frequencies
from .utils import (
    atomic_save_numpy,
    atomic_torch_save,
    file_sha256,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    import scipy.sparse as sp


class _Sgns(nn.Module):
    def __init__(self, vocabulary_size: int, dimensions: int) -> None:
        super().__init__()
        self.source = nn.Embedding(vocabulary_size, dimensions, sparse=True)
        self.context = nn.Embedding(vocabulary_size, dimensions, sparse=True)
        bound = 0.5 / dimensions
        nn.init.uniform_(self.source.weight, -bound, bound)
        nn.init.zeros_(self.context.weight)

    def forward(
        self,
        source: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        source_values = self.source(source)
        context_values = self.context(context)
        positive = torch.sum(source_values * context_values, dim=1)
        negative_values = self.context(negatives)
        negative = torch.einsum("bd,bnd->bn", source_values, negative_values)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negative).sum(dim=1)).mean()


def _positive_pairs(
    matrix: sp.csr_matrix,
    *,
    pairs_per_document: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = matrix.shape[0] * pairs_per_document
    sources = np.empty(total, dtype=np.int64)
    contexts = np.empty(total, dtype=np.int64)
    cursor = 0
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        words = matrix.indices[start:stop]
        counts = matrix.data[start:stop].astype(np.float64, copy=False)
        if not len(words):
            continue
        probability = counts / counts.sum()
        left = rng.choice(words, size=pairs_per_document, p=probability)
        right = rng.choice(words, size=pairs_per_document, p=probability)
        if len(words) > 1:
            same = left == right
            attempts = 0
            while np.any(same) and attempts < 8:
                right[same] = rng.choice(words, size=int(same.sum()), p=probability)
                same = left == right
                attempts += 1
            if np.any(same):
                positions = {int(word): index for index, word in enumerate(words)}
                right[same] = [
                    words[(positions[int(word)] + 1) % len(words)]
                    for word in left[same]
                ]
        count = len(left)
        sources[cursor : cursor + count] = left
        contexts[cursor : cursor + count] = right
        cursor += count
    return sources[:cursor], contexts[:cursor]


def _combined_embeddings(model: _Sgns) -> np.ndarray:
    values = 0.5 * (model.source.weight.detach() + model.context.weight.detach())
    values = F.normalize(values, dim=1)
    return values.cpu().numpy().astype(np.float32)


def train_sgns(
    output_dir: str | Path,
    matrix: sp.csr_matrix,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Train or resume the frozen five-epoch SGNS feature initializer."""
    output = Path(output_dir)
    complete_path = output / "complete.json"
    embeddings_path = output / "embeddings.npy"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if file_sha256(embeddings_path) != complete["embeddings_sha256"]:
            msg = "SGNS embeddings changed after completion"
            raise ValueError(msg)
        return complete
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = _Sgns(matrix.shape[1], int(config["dimensions"]))
    optimizer = torch.optim.SparseAdam(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )
    latest_path = output / "latest.pt"
    start_epoch = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
    frequencies = corpus_frequencies(matrix)
    negative = np.power(frequencies, float(config["negative_power"]))
    negative /= negative.sum()
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"])):
        sources, contexts = _positive_pairs(
            matrix,
            pairs_per_document=int(config["positive_pairs_per_document"]),
            seed=seed + 1009 * epoch,
        )
        rng = np.random.default_rng(seed + 2027 * epoch)
        order = rng.permutation(len(sources))
        total_loss = 0.0
        observations = 0
        for begin in range(0, len(order), int(config["batch_size"])):
            selected = order[begin : begin + int(config["batch_size"])]
            negatives = rng.choice(
                matrix.shape[1],
                size=(len(selected), int(config["negative_samples"])),
                p=negative,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                torch.from_numpy(sources[selected]),
                torch.from_numpy(contexts[selected]),
                torch.from_numpy(negatives),
            )
            if not torch.isfinite(loss):
                msg = "SGNS training produced a non-finite loss"
                raise FloatingPointError(msg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(selected)
            observations += len(selected)
        elapsed = elapsed_before + time.perf_counter() - started
        history.append(
            {
                "epoch": epoch + 1,
                "positive_pairs": observations,
                "mean_loss": total_loss / max(observations, 1),
                "elapsed_seconds": elapsed,
            },
        )
        atomic_torch_save(
            latest_path,
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "elapsed_seconds": elapsed,
            },
        )
        write_json(output / "history.json", history)
    embeddings = _combined_embeddings(model)
    atomic_save_numpy(embeddings_path, embeddings)
    result = {
        "schema_version": "fully-neural-ms2lda/sgns-v1",
        "training_split_only": True,
        "documents": matrix.shape[0],
        "vocabulary_size": matrix.shape[1],
        "dimensions": embeddings.shape[1],
        "epochs": int(config["epochs"]),
        "positive_pairs_per_document_per_epoch": int(
            config["positive_pairs_per_document"],
        ),
        "negative_samples": int(config["negative_samples"]),
        "negative_power": float(config["negative_power"]),
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "embeddings_sha256": file_sha256(embeddings_path),
    }
    write_json(complete_path, result)
    return result
