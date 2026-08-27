"""Validation-only published-topic-model experiment on the locked MSnLib split.

This research runner is isolated from the production M1 implementation.  It
uses the repository's exact data split, vocabulary, document-completion data,
and train-only SGNS features.  Test data are never evaluated here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.neural_ms2lda.artifacts import initialize_run
from benchmarks.neural_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_vocabulary,
    prepare_data,
    train_token_features,
)
from benchmarks.neural_ms2lda.objectives import completion_metrics
from benchmarks.neural_ms2lda.utils import atomic_save_numpy, read_json, write_json

EPS = 1e-12


def configure(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def sgns_only(path: Path) -> np.ndarray:
    features = np.load(path).astype(np.float32, copy=False)
    embeddings = np.array(features[:, :-2], dtype=np.float32, copy=True)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    return embeddings


def dense_normalized(matrix: sp.csr_matrix, rows: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(matrix[rows].toarray().astype(np.float32, copy=False))
    return x / x.sum(1, keepdim=True).clamp_min(1.0)


def sparse_reconstruction(theta: torch.Tensor, beta: torch.Tensor, matrix: sp.csr_matrix, rows: np.ndarray) -> torch.Tensor:
    """Exact multinomial reconstruction, materializing only observed words."""
    batch = matrix[rows].tocsr()
    if batch.nnz == 0:
        return theta.new_zeros(())
    row_ids = torch.from_numpy(np.repeat(np.arange(len(rows), dtype=np.int64), np.diff(batch.indptr)))
    word_ids = torch.from_numpy(batch.indices.astype(np.int64, copy=False))
    weights = torch.from_numpy(batch.data.astype(np.float32, copy=False))
    probability = torch.sum(theta[row_ids] * beta[:, word_ids].T, dim=1).clamp_min(EPS)
    per_doc = theta.new_zeros((len(rows),))
    per_doc.index_add_(0, row_ids, -weights * torch.log(probability))
    return per_doc.mean()


class FixedETM(nn.Module):
    """Dieng/Ruiz/Blei ETM with fixed pretrained SGNS embeddings."""

    def __init__(self, embeddings: np.ndarray, topics: int, hidden: int = 800) -> None:
        super().__init__()
        self.register_buffer("rho", torch.as_tensor(embeddings, dtype=torch.float32))
        vocab, dim = embeddings.shape
        self.alphas = nn.Linear(dim, topics, bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(vocab, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, topics)
        self.logvar = nn.Linear(hidden, topics)

    def beta(self) -> torch.Tensor:
        return F.softmax(self.alphas(self.rho), dim=0).T

    def theta(self, normalized_bows: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(normalized_bows)
        mu = self.mu(h)
        logvar = self.logvar(h)
        kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return F.softmax(z, dim=1), kl


class ECR(nn.Module):
    """Published ECR objective with the predeclared 50-step numerical cap."""

    def __init__(self, weight: float = 100.0, alpha: float = 20.0, max_iter: int = 50) -> None:
        super().__init__()
        self.weight = float(weight)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        a = torch.ones((cost.shape[0], 1), dtype=cost.dtype) / cost.shape[0]
        b = torch.ones((cost.shape[1], 1), dtype=cost.dtype) / cost.shape[1]
        u = torch.ones_like(a) / a.shape[0]
        kernel = torch.exp(-cost * self.alpha)
        eps = 1e-16
        for iteration in range(self.max_iter):
            v = b / (kernel.T @ u + eps)
            u = a / (kernel @ v + eps)
            if iteration % 50 == 0:
                residual = torch.max(torch.sum(torch.abs(v * (kernel.T @ u) - b), dim=0))
                if float(residual.detach()) <= 0.005:
                    break
        transport = u * (kernel * v.T)
        return self.weight * torch.sum(transport * cost)


class TopMostECRTM(nn.Module):
    """Maintained TopMost ECRTM equations with SGNS initialization."""

    def __init__(self, embeddings: np.ndarray, topics: int) -> None:
        super().__init__()
        emb = F.normalize(torch.as_tensor(embeddings, dtype=torch.float32), dim=1)
        vocab, dim = embeddings.shape
        self.topics = int(topics)
        self.fc11 = nn.Linear(vocab, 200)
        self.fc12 = nn.Linear(200, 200)
        self.fc21 = nn.Linear(200, topics)
        self.fc22 = nn.Linear(200, topics)
        self.mean_bn = nn.BatchNorm1d(topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab, affine=True)
        self.decoder_bn.weight.requires_grad = False
        self.word_embeddings = nn.Parameter(emb.clone())
        topic = torch.empty((topics, dim), dtype=torch.float32)
        nn.init.trunc_normal_(topic, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(topic, dim=1))
        self.ecr = ECR(weight=100.0, alpha=20.0, max_iter=50)
        concentration = np.ones((1, topics), dtype=np.float32)
        mu2 = (np.log(concentration).T - np.mean(np.log(concentration), 1)).T
        var2 = (((1.0 / concentration) * (1 - 2.0 / topics)).T + (1.0 / topics**2) * np.sum(1.0 / concentration, 1)).T
        self.register_buffer("mu2", torch.from_numpy(mu2))
        self.register_buffer("var2", torch.from_numpy(var2))

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.square(), dim=1, keepdim=True) + torch.sum(y.square(), dim=1) - 2 * (x @ y.T)

    def beta(self) -> torch.Tensor:
        return F.softmax(-self.distance(self.topic_embeddings, self.word_embeddings) / 0.2, dim=0)

    def theta(self, bows: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.softplus(self.fc11(bows))
        h = F.softplus(self.fc12(h))
        mu = self.mean_bn(self.fc21(h))
        logvar = self.logvar_bn(self.fc22(h))
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        theta = F.softmax(z, dim=1)
        var = logvar.exp()
        kl = 0.5 * ((var / self.var2 + (mu - self.mu2).square() / self.var2 + self.var2.log() - logvar).sum(dim=1) - self.topics)
        return theta, kl

    def ecr_loss(self) -> torch.Tensor:
        return self.ecr(self.distance(self.topic_embeddings, self.word_embeddings))


def infer_etm(model: FixedETM, matrix: sp.csr_matrix, batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            rows = np.arange(start, min(start + batch_size, matrix.shape[0]), dtype=np.int64)
            theta, _ = model.theta(dense_normalized(matrix, rows), sample=False)
            values.append(theta.numpy().astype(np.float32))
    return np.concatenate(values)


def infer_ecrtm(model: TopMostECRTM, matrix: sp.csr_matrix, batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            rows = np.arange(start, min(start + batch_size, matrix.shape[0]), dtype=np.int64)
            bows = torch.from_numpy(matrix[rows].toarray().astype(np.float32, copy=False))
            theta, _ = model.theta(bows, sample=False)
            values.append(theta.numpy().astype(np.float32))
    return np.concatenate(values)


def sharpen(theta: np.ndarray, temperature: float = 0.30) -> np.ndarray:
    logits = np.log(np.clip(theta.astype(np.float64), EPS, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    return values.astype(np.float32)


def collapse_metrics(theta: np.ndarray, beta: np.ndarray) -> dict[str, Any]:
    usage = theta.astype(np.float64).mean(axis=0)
    entropy = -np.sum(theta * np.log(np.clip(theta, EPS, None)), axis=1)
    normalized = beta / np.maximum(np.linalg.norm(beta, axis=1, keepdims=True), EPS)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)
    return {
        "median_effective_topics_per_spectrum": float(np.median(np.exp(entropy))),
        "corpus_effective_topics": float(np.exp(-np.sum(usage * np.log(np.clip(usage, EPS, None))))),
        "active_topics_usage_ge_1_over_k": int(np.sum(usage >= 1.0 / len(usage))),
        "active_topics_usage_gt_0_0005": int(np.sum(usage > 0.0005)),
        "max_mean_topic_usage": float(usage.max()),
        "mean_nearest_beta_cosine": float(np.max(similarity, axis=1).mean()),
        "max_beta_pair_cosine": float(similarity.max()),
    }


def ecrtm_completion(model: TopMostECRTM, theta: np.ndarray, completion: sp.csr_matrix, records: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    model.eval()
    loss = 0.0
    invocab = oov = eligible = 0
    with torch.no_grad():
        beta = model.beta()
        for start in range(0, completion.shape[0], batch_size):
            stop = min(start + batch_size, completion.shape[0])
            th = torch.from_numpy(theta[start:stop].astype(np.float32, copy=False))
            probs = F.softmax(model.decoder_bn(th @ beta), dim=-1).numpy()
            for local, row in enumerate(range(start, stop)):
                a, b = completion.indptr[row], completion.indptr[row + 1]
                words, counts = completion.indices[a:b], completion.data[a:b]
                oov += int(records[row]["completion_oov_tokens"])
                if not len(words):
                    continue
                loss -= float(np.sum(counts * np.log(np.clip(probs[local, words], EPS, None))))
                invocab += int(counts.sum())
                eligible += 1
    return {
        "nll_per_token": loss / max(invocab, 1),
        "in_vocabulary_tokens": invocab,
        "out_of_vocabulary_tokens": oov,
        "oov_fraction": oov / max(invocab + oov, 1),
        "eligible_documents": eligible,
        "total_documents": completion.shape[0],
    }


def save_validation(run: Path, method: str, beta: np.ndarray, theta: np.ndarray, metrics: dict[str, Any]) -> None:
    output = run / "validation_evaluation" / method
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output / "beta.npy", beta.astype(np.float32, copy=False))
    atomic_save_numpy(output / "validation_full_theta.npy", theta.astype(np.float32, copy=False))
    write_json(output / "complete.json", {"method": method, "split": "validation", "metrics": metrics})


def prepare(run: Path, data_root: Path) -> None:
    protocol = initialize_run(run, data_root=data_root)
    configure(int(protocol["seed"]), int(protocol["cpu_threads"]))
    prepare_data(run, data_root=data_root, protocol=protocol)
    train = load_csr(run / "data/train.npz")
    vocabulary = load_vocabulary(run / "data")
    train_token_features(run / "token_features", train, vocabulary, protocol, seed=int(protocol["seed"]))
    print(json.dumps({"prepared": True, "train_shape": list(train.shape), "vocabulary": len(vocabulary)}), flush=True)


def train_etm(run: Path, epochs: int = 120, batch_size: int = 256) -> dict[str, Any]:
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 7001, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    observed = load_csr(run / "data/validation_observed.npz")
    completion = load_csr(run / "data/validation_completion.npz")
    full = load_csr(run / "data/validation_full.npz")
    records = load_heldout_records(run / "data", "validation")
    model = FixedETM(sgns_only(run / "token_features/features.npy"), int(protocol["model"]["num_topics"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1.2e-6)
    rng = np.random.default_rng(seed + 7019)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train.shape[0])
        rec = kl_value = 0.0
        batches = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            theta, kl = model.theta(dense_normalized(train, rows), sample=True)
            beta = model.beta()
            reconstruction = sparse_reconstruction(theta, beta, train, rows)
            objective = reconstruction + kl.mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            rec += float(reconstruction.detach())
            kl_value += float(kl.mean().detach())
            batches += 1
        row = {"epoch": epoch + 1, "reconstruction": rec / batches, "kl": kl_value / batches, "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("ETM_EPOCH", json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        beta = model.beta().numpy().astype(np.float32)
    theta_obs = infer_etm(model, observed, batch_size)
    theta_full = infer_etm(model, full, batch_size)
    metrics = {
        "document_completion": completion_metrics(theta_obs, beta, completion, records),
        "full_spectrum_mixture": collapse_metrics(theta_full, beta),
    }
    result = {"method": "etm", "epochs": epochs, "fitting_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters()), "metrics": metrics}
    output = run / "published_models/etm"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "weights.pt")
    write_json(output / "history.json", history)
    write_json(output / "result.json", result)
    save_validation(run, "etm", beta, theta_full, metrics)
    print("ETM_RESULT", json.dumps(result), flush=True)
    return result


def train_ecrtm(run: Path, epochs: int = 40, batch_size: int = 200, wall_cap: float = 14400.0) -> dict[str, Any]:
    protocol = read_json(run / "protocol.json")
    seed = int(protocol["seed"])
    configure(seed + 8001, int(protocol["cpu_threads"]))
    train = load_csr(run / "data/train.npz")
    observed = load_csr(run / "data/validation_observed.npz")
    completion = load_csr(run / "data/validation_completion.npz")
    full = load_csr(run / "data/validation_full.npz")
    records = load_heldout_records(run / "data", "validation")
    model = TopMostECRTM(sgns_only(run / "token_features/features.npy"), int(protocol["model"]["num_topics"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rng = np.random.default_rng(seed + 8017)
    history = []
    started = time.perf_counter()
    status = "complete"
    stop_reason = None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train.shape[0])
        tm_total = ecr_total = 0.0
        batches = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            bows = torch.from_numpy(train[rows].toarray().astype(np.float32, copy=False))
            theta, kl = model.theta(bows, sample=True)
            beta = model.beta()
            reconstruction = F.softmax(model.decoder_bn(theta @ beta), dim=-1)
            reconstruction_loss = -(bows * torch.log(reconstruction.clamp_min(EPS))).sum(dim=1).mean()
            topic_loss = reconstruction_loss + kl.mean()
            ecr_loss = model.ecr_loss()
            objective = topic_loss + ecr_loss
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            tm_total += float(topic_loss.detach())
            ecr_total += float(ecr_loss.detach())
            batches += 1
        epoch_seconds = time.perf_counter() - epoch_start
        projection = epoch_seconds * epochs
        row = {"epoch": epoch + 1, "topic_model_loss": tm_total / batches, "ecr_loss": ecr_total / batches, "seconds": epoch_seconds, "projected_total_seconds": projection}
        history.append(row)
        print("ECRTM_EPOCH", json.dumps(row), flush=True)
        if epoch == 0 and projection > wall_cap:
            status = "stopped_operationally_infeasible"
            stop_reason = f"first-epoch projection {projection:.1f}s exceeds {wall_cap:.1f}s cap"
            break
    output = run / "published_models/ecrtm"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "history.json", history)
    if status != "complete":
        result = {"method": "ecrtm", "status": status, "stop_reason": stop_reason, "epochs_completed": len(history), "fitting_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters())}
        write_json(output / "result.json", result)
        print("ECRTM_RESULT", json.dumps(result), flush=True)
        return result
    model.eval()
    with torch.no_grad():
        beta_internal = model.beta().numpy().astype(np.float32)
    beta = beta_internal / np.maximum(beta_internal.sum(axis=1, keepdims=True), EPS)
    theta_obs = infer_ecrtm(model, observed, batch_size)
    theta_full = infer_ecrtm(model, full, batch_size)
    metrics = {
        "document_completion": ecrtm_completion(model, theta_obs, completion, records, batch_size),
        "full_spectrum_mixture": collapse_metrics(theta_full, beta),
    }
    result = {"method": "ecrtm", "status": status, "epochs": epochs, "fitting_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters()), "metrics": metrics}
    torch.save(model.state_dict(), output / "weights.pt")
    write_json(output / "result.json", result)
    save_validation(run, "ecrtm", beta, theta_full, metrics)
    theta_obs_tau = sharpen(theta_obs, 0.30)
    theta_full_tau = sharpen(theta_full, 0.30)
    tau_metrics = {
        "document_completion": ecrtm_completion(model, theta_obs_tau, completion, records, batch_size),
        "full_spectrum_mixture": collapse_metrics(theta_full_tau, beta),
    }
    tau_result = {"method": "ecrtm_tau030", "source": "ecrtm", "theta_temperature": 0.30, "metrics": tau_metrics}
    write_json(run / "published_models/ecrtm_tau030_result.json", tau_result)
    save_validation(run, "ecrtm_tau030", beta, theta_full_tau, tau_metrics)
    print("ECRTM_RESULT", json.dumps(result), flush=True)
    print("ECRTM_TAU030_RESULT", json.dumps(tau_result), flush=True)
    return result


def compare(run: Path, etm_epochs: int, ecrtm_epochs: int) -> None:
    etm = train_etm(run, etm_epochs)
    ecrtm = train_ecrtm(run, ecrtm_epochs)
    summary: dict[str, Any] = {
        "evidence_boundary": "validation only; no candidate test evaluation",
        "etm": etm,
        "ecrtm": ecrtm,
        "historical_m1_validation": {
            "completion_nll_per_token": 8.974139925584877,
            "optimized_motifs": 884,
            "evaluable_motifs": 408,
            "useful_motifs": 265,
            "mean_sos": 0.6580793714074608,
            "median_sos": 0.6488636363636364,
        },
    }
    tau = run / "published_models/ecrtm_tau030_result.json"
    if tau.is_file():
        summary["ecrtm_tau030"] = read_json(tau)
    write_json(run / "published_models/summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run", required=True, type=Path)
    prep.add_argument("--data-root", required=True, type=Path)
    run = commands.add_parser("compare")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--etm-epochs", type=int, default=120)
    run.add_argument("--ecrtm-epochs", type=int, default=40)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run.resolve(), args.data_root.resolve())
    else:
        compare(args.run.resolve(), args.etm_epochs, args.ecrtm_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
