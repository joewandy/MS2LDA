# ruff: noqa: C901, PLR0913, PLR0915, TRY301
"""Benchmark-only inference variants with frozen topic discovery."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from benchmarks.msnlib_validation.config import (
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)
from ms2lda_hybrid._variational import (
    EPSILON,
    corpus_elbo_minibatch_scale,
    expected_log_dirichlet,
    local_document_elbo,
    local_vb,
    make_sparse_batch,
)

from .data import load_count_matrix, load_observed_dreams_embeddings
from .discovery import load_discovery
from .runtime import configure_cpu_threads
from .spec import INFERENCE_IDS, load_spec, verify_study

if TYPE_CHECKING:
    import scipy.sparse as sp


class EvidenceEncoder(nn.Module):
    """Residual initializer using topic evidence with optional DreaMS context."""

    def __init__(
        self,
        *,
        num_topics: int,
        embedding_dim: int,
        hidden_size: int,
        feature_projection_dim: int,
        use_dreams: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.num_topics = int(num_topics)
        self.embedding_dim = int(embedding_dim)
        self.use_dreams = bool(use_dreams)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.use_dreams:
                self.document_projector: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, feature_projection_dim),
                    nn.GELU(),
                    nn.LayerNorm(feature_projection_dim),
                )
                input_size = num_topics + feature_projection_dim
            else:
                self.document_projector = None
                input_size = num_topics
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_topics),
            )
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)

    def forward(
        self,
        evidence: torch.Tensor,
        totals: torch.Tensor,
        alpha: torch.Tensor,
        embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return positive document-topic Dirichlet parameters."""
        if self.use_dreams:
            if embeddings is None or self.document_projector is None:
                msg = "DreaMS encoder requires document embeddings"
                raise ValueError(msg)
            if tuple(embeddings.shape) != (evidence.shape[0], self.embedding_dim):
                msg = "DreaMS embedding shape changed"
                raise ValueError(msg)
            inputs = torch.cat([evidence, self.document_projector(embeddings)], dim=1)
        else:
            if embeddings is not None:
                msg = "topic-only encoder must not receive DreaMS embeddings"
                raise ValueError(
                    msg,
                )
            inputs = evidence
        residual = self.encoder(inputs)
        mean = torch.softmax(evidence.clamp_min(EPSILON).log() + residual, dim=1)
        return alpha.unsqueeze(0) + totals * mean


def _arm_parts(arm_id: str) -> tuple[str, str]:
    try:
        discovery, inference = arm_id.split("__", 1)
    except ValueError as exc:
        msg = f"invalid arm identifier: {arm_id}"
        raise ValueError(msg) from exc
    if inference not in INFERENCE_IDS:
        msg = f"unknown inference variant: {inference}"
        raise ValueError(msg)
    return discovery, inference


def _inference_properties(inference: str) -> tuple[bool, str]:
    if inference == "analytic":
        return False, "analytic"
    use_dreams = inference.startswith("dreams_")
    objective = "semi" if inference.endswith("_semi") else "direct"
    return use_dreams, objective


def _word_topic(lambda_posterior: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(np.asarray(lambda_posterior, dtype=np.float32))
    expected = expected_log_dirichlet(values)
    return expected, torch.softmax(expected.transpose(0, 1), dim=1)


def _evidence(batch: Any, word_topic: torch.Tensor) -> torch.Tensor:
    counts = batch.word_counts * batch.word_mask
    evidence = (counts.unsqueeze(-1) * word_topic[batch.word_ids]).sum(dim=1)
    evidence = evidence / batch.totals.clamp_min(1.0)
    empty = batch.totals <= 0
    if torch.any(empty):
        evidence = torch.where(
            empty,
            torch.full_like(evidence, 1.0 / word_topic.shape[1]),
            evidence,
        )
    return evidence / evidence.sum(dim=1, keepdim=True).clamp_min(EPSILON)


def analytic_gamma(
    batch: Any,
    alpha: torch.Tensor,
    word_topic: torch.Tensor,
) -> torch.Tensor:
    """Use count-derived topic evidence without a learned encoder."""
    return alpha.unsqueeze(0) + batch.totals * _evidence(batch, word_topic)


def uniform_gamma(batch: Any, alpha: torch.Tensor) -> torch.Tensor:
    """Return the symmetric initializer used as a common-reference candidate."""
    return alpha.unsqueeze(0) + batch.totals / alpha.numel()


def _encoder_settings(run_dir: Path) -> dict[str, int | float]:
    lock = verify_study(run_dir)
    payload = torch.load(
        Path(lock["source_run"]) / "core/seed_42/hybrid/model.pt",
        map_location="cpu",
        weights_only=True,
    )
    return payload["config"]


def new_encoder(run_dir: str | Path, inference: str) -> EvidenceEncoder:
    """Construct one deterministic benchmark-only neural initializer."""
    directory = Path(run_dir).expanduser().resolve()
    settings = _encoder_settings(directory)
    use_dreams, objective = _inference_properties(inference)
    if objective == "analytic":
        msg = "analytic inference has no encoder"
        raise ValueError(msg)
    spec = load_spec(directory)
    return EvidenceEncoder(
        num_topics=spec.num_topics,
        embedding_dim=int(settings["embedding_dim"]),
        hidden_size=int(settings["hidden_size"]),
        feature_projection_dim=int(settings["feature_projection_dim"]),
        use_dreams=use_dreams,
        seed=spec.seed,
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _encoder_context(
    run_dir: Path,
    arm_id: str,
    discovery_complete: dict[str, Any],
) -> str:
    spec = load_spec(run_dir)
    return object_sha256(
        {
            "spec": spec.as_dict(),
            "arm_id": arm_id,
            "discovery_outputs": discovery_complete["output_sha256"],
        },
    )


def _model_output(run_dir: Path, arm_id: str) -> Path:
    return run_dir / "encoders" / arm_id


def _verify_encoder_complete(output: Path, arm_id: str) -> dict[str, Any]:
    result = read_json(output / "complete.json")
    if result.get("arm_id") != arm_id:
        msg = "encoder completion identity changed"
        raise ValueError(msg)
    for name, digest in result.get("output_sha256", {}).items():
        if file_sha256(output / name) != digest:
            msg = f"encoder artifact changed: {name}"
            raise ValueError(msg)
    return result


def build_direct_targets(run_dir: str | Path, discovery: str) -> dict[str, Any]:
    """Create fixed analytic-plus-50-step posterior targets once per discovery."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    cpu_threads = configure_cpu_threads(directory, "training")
    spec = load_spec(directory)
    output = directory / "targets" / discovery
    complete_path = output / "complete.json"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(output / "gamma.npy") != result["gamma_sha256"]:
            msg = "direct-regression targets changed"
            raise ValueError(msg)
        return result
    alpha_values, lambda_values, _, discovery_complete = load_discovery(
        directory,
        discovery,
    )
    matrix = load_count_matrix(directory, "train")
    alpha = torch.from_numpy(alpha_values)
    expected_log_beta, word_topic = _word_topic(lambda_values)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.lib.format.open_memmap(
        output / "gamma.npy",
        mode="w+",
        dtype=np.float32,
        shape=(matrix.shape[0], spec.num_topics),
    )
    started = time.perf_counter()
    for start in range(0, matrix.shape[0], spec.batch_size):
        indices = np.arange(start, min(start + spec.batch_size, matrix.shape[0]))
        batch = make_sparse_batch(matrix, indices, device=torch.device("cpu"))
        initial = analytic_gamma(batch, alpha, word_topic)
        refined, _ = local_vb(
            batch,
            initial,
            alpha,
            expected_log_beta,
            steps=spec.direct_target_steps,
            tolerance=None,
        )
        targets[start : start + len(indices)] = refined.numpy()
        targets.flush()
    result = {
        "schema_version": "msnlib-simplification/direct-targets-v1",
        "discovery": discovery,
        "discovery_sha256": object_sha256(discovery_complete),
        "initializer": "analytic_topic_evidence",
        "refinement_steps": spec.direct_target_steps,
        "rows": matrix.shape[0],
        "topics": spec.num_topics,
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_threads": cpu_threads,
        "dreams_cache_loaded": False,
        "gamma_sha256": file_sha256(output / "gamma.npy"),
    }
    write_json(complete_path, result)
    return result


def _checkpoint_sidecars(output: Path) -> list[Path]:
    return sorted((output / "checkpoints").glob("checkpoint-*.json"), reverse=True)


def _save_encoder_checkpoint(
    output: Path,
    *,
    context: str,
    epoch: int,
    encoder: EvidenceEncoder,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    history: list[dict[str, float]],
    elapsed_seconds: float,
    keep: int,
) -> None:
    checkpoint_dir = output / "checkpoints"
    stem = f"checkpoint-{epoch:04d}"
    path = checkpoint_dir / f"{stem}.pt"
    _atomic_torch_save(
        path,
        {
            "schema_version": "simplification-encoder-checkpoint/v1",
            "context_sha256": context,
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
            "history": copy.deepcopy(history),
            "elapsed_seconds": elapsed_seconds,
        },
    )
    metadata = {
        "context_sha256": context,
        "epoch": epoch,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(checkpoint_dir / f"{stem}.json", metadata)
    write_json(checkpoint_dir / "latest.json", metadata)
    for sidecar in _checkpoint_sidecars(output)[keep:]:
        old = read_json(sidecar)
        (sidecar.parent / str(old.get("file", ""))).unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def _restore_encoder_checkpoint(
    output: Path,
    *,
    context: str,
    encoder: EvidenceEncoder,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> tuple[int, np.random.Generator, list[dict[str, float]], float]:
    rejected: list[dict[str, str]] = []
    for sidecar in _checkpoint_sidecars(output):
        try:
            metadata = read_json(sidecar)
            if metadata.get("context_sha256") != context:
                msg = "context mismatch"
                raise ValueError(msg)
            path = sidecar.parent / str(metadata["file"])
            if path.stat().st_size != int(metadata["bytes"]):
                msg = "byte-size mismatch"
                raise ValueError(msg)
            if file_sha256(path) != metadata["sha256"]:
                msg = "hash mismatch"
                raise ValueError(msg)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("context_sha256") != context:
                msg = "payload context mismatch"
                raise ValueError(msg)
            encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except Exception as exc:  # noqa: BLE001 - fall back to older checkpoint
            rejected.append(
                {"sidecar": sidecar.name, "reason": f"{type(exc).__name__}: {exc}"},
            )
            continue
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = copy.deepcopy(payload["numpy_rng_state"])
        write_json(
            output / "checkpoint_resume_audit.json",
            {"selected": metadata, "rejected_newer": rejected, "resumed": True},
        )
        return (
            int(payload["epoch"]),
            rng,
            copy.deepcopy(payload["history"]),
            float(payload["elapsed_seconds"]),
        )
    if _checkpoint_sidecars(output):
        msg = "no valid encoder checkpoint remains"
        raise RuntimeError(msg)
    write_json(
        output / "checkpoint_resume_audit.json",
        {"selected": None, "rejected_newer": [], "resumed": False},
    )
    return 0, np.random.default_rng(seed), [], 0.0


def import_current_encoder(run_dir: str | Path) -> dict[str, Any]:
    """Import the exact corrected current encoder as the factorial baseline."""
    directory = Path(run_dir).expanduser().resolve()
    lock = verify_study(directory)
    arm_id = "dreams_prior__dreams_semi"
    output = _model_output(directory, arm_id)
    if (output / "complete.json").is_file():
        return _verify_encoder_complete(output, arm_id)
    _, _, _, discovery_complete = load_discovery(directory, "dreams_prior")
    source_model = Path(lock["source_run"]) / "core/seed_42/hybrid/model.pt"
    payload = torch.load(source_model, map_location="cpu", weights_only=True)
    encoder = new_encoder(directory, "dreams_semi")
    state = payload["core_state_dict"]
    imported = {
        name: value
        for name, value in state.items()
        if name.startswith(("document_projector.", "encoder."))
    }
    encoder.load_state_dict(imported, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pt"
    _atomic_torch_save(
        model_path,
        {
            "schema_version": "msnlib-simplification/encoder-v1",
            "arm_id": arm_id,
            "encoder_state_dict": encoder.state_dict(),
        },
    )
    history_path = output / "history.json"
    write_json(
        history_path,
        read_json(
            Path(lock["source_run"]) / "core/seed_42/hybrid/inference_history.json",
        ),
    )
    result = {
        "schema_version": "msnlib-simplification/encoder-complete-v1",
        "arm_id": arm_id,
        "discovery": "dreams_prior",
        "inference": "dreams_semi",
        "objective": "semi",
        "uses_dreams": True,
        "imported_current_baseline": True,
        "source_model_sha256": file_sha256(source_model),
        "discovery_sha256": object_sha256(discovery_complete),
        "epochs": len(read_json(history_path)),
        "parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "output_sha256": {
            path.name: file_sha256(path) for path in (model_path, history_path)
        },
    }
    write_json(output / "complete.json", result)
    return result


def mark_analytic_arm(run_dir: str | Path, discovery: str) -> dict[str, Any]:
    """Publish an immutable zero-parameter analytic arm manifest."""
    directory = Path(run_dir).expanduser().resolve()
    arm_id = f"{discovery}__analytic"
    output = _model_output(directory, arm_id)
    if (output / "complete.json").is_file():
        return _verify_encoder_complete(output, arm_id)
    _, _, _, discovery_complete = load_discovery(directory, discovery)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "msnlib-simplification/encoder-complete-v1",
        "arm_id": arm_id,
        "discovery": discovery,
        "inference": "analytic",
        "objective": "analytic",
        "uses_dreams": False,
        "dreams_cache_loaded": False,
        "parameter_count": 0,
        "epochs": 0,
        "discovery_sha256": object_sha256(discovery_complete),
        "output_sha256": {},
    }
    write_json(output / "complete.json", result)
    return result


def train_encoder(run_dir: str | Path, arm_id: str) -> dict[str, Any]:
    """Train one frozen-topic semi-amortized or direct-regression initializer."""
    directory = Path(run_dir).expanduser().resolve()
    verify_study(directory)
    cpu_threads = configure_cpu_threads(directory, "training")
    spec = load_spec(directory)
    discovery, inference = _arm_parts(arm_id)
    use_dreams, objective = _inference_properties(inference)
    if objective == "analytic":
        return mark_analytic_arm(directory, discovery)
    if arm_id == "dreams_prior__dreams_semi":
        return import_current_encoder(directory)
    output = _model_output(directory, arm_id)
    if (output / "complete.json").is_file():
        return _verify_encoder_complete(output, arm_id)
    alpha_values, lambda_values, _, discovery_complete = load_discovery(
        directory,
        discovery,
    )
    matrix = load_count_matrix(directory, "train")
    embeddings = (
        load_observed_dreams_embeddings(directory, "train") if use_dreams else None
    )
    target_values: np.ndarray | None = None
    if objective == "direct":
        targets = build_direct_targets(directory, discovery)
        target_path = directory / "targets" / discovery / "gamma.npy"
        if file_sha256(target_path) != targets["gamma_sha256"]:
            msg = "direct target changed before encoder training"
            raise ValueError(msg)
        target_values = np.load(target_path, mmap_mode="r")
    alpha = torch.from_numpy(alpha_values)
    expected_log_beta, word_topic = _word_topic(lambda_values)
    encoder = new_encoder(directory, inference)
    settings = _encoder_settings(directory)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=float(settings["encoder_learning_rate"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    context = _encoder_context(directory, arm_id, discovery_complete)
    epoch, rng, history, elapsed_base = _restore_encoder_checkpoint(
        output,
        context=context,
        encoder=encoder,
        optimizer=optimizer,
        seed=spec.seed,
    )
    lambda_snapshot = lambda_values.copy()
    started = time.perf_counter()
    corpus_tokens = float(matrix.sum())
    for phase_epoch in range(epoch + 1, spec.inference_epochs + 1):
        shuffled = rng.permutation(matrix.shape[0])
        loss_sum = 0.0
        objective_sum = 0.0
        gradient_sum = 0.0
        batches = 0
        documents = 0
        encoder.train()
        for start in range(0, matrix.shape[0], spec.batch_size):
            indices = shuffled[start : start + spec.batch_size]
            batch = make_sparse_batch(matrix, indices, device=torch.device("cpu"))
            evidence = _evidence(batch, word_topic)
            embedding_batch = (
                torch.from_numpy(np.asarray(embeddings[indices], dtype=np.float32))
                if embeddings is not None
                else None
            )
            predicted = encoder(evidence, batch.totals, alpha, embedding_batch)
            if objective == "semi":
                refined, _ = local_vb(
                    batch,
                    predicted,
                    alpha,
                    expected_log_beta,
                    steps=spec.semi_refinement_steps,
                    tolerance=None,
                )
                refined_elbo = local_document_elbo(
                    batch,
                    refined,
                    alpha,
                    expected_log_beta,
                )
                zero_elbo = local_document_elbo(
                    batch,
                    predicted,
                    alpha,
                    expected_log_beta,
                )
                scale = corpus_elbo_minibatch_scale(
                    corpus_documents=matrix.shape[0],
                    batch_documents=len(indices),
                    corpus_tokens=corpus_tokens,
                )
                loss = -(refined_elbo.sum() + 0.1 * zero_elbo.sum()) * scale
                objective_sum += float(refined_elbo.sum().detach())
            else:
                if target_values is None:
                    msg = "direct regression target is unavailable"
                    raise RuntimeError(msg)
                target = torch.from_numpy(
                    np.asarray(target_values[indices], dtype=np.float32),
                )
                target_theta = target / target.sum(dim=1, keepdim=True)
                predicted_theta = predicted / predicted.sum(dim=1, keepdim=True)
                document_kl = (
                    target_theta
                    * (
                        target_theta.clamp_min(EPSILON).log()
                        - predicted_theta.clamp_min(EPSILON).log()
                    )
                ).sum(dim=1)
                loss = document_kl.mean()
                objective_sum += float(document_kl.sum().detach())
            if not bool(torch.isfinite(loss)):
                msg = "non-finite encoder loss"
                raise FloatingPointError(msg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                10.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach())
            gradient_sum += float(gradient.detach())
            batches += 1
            documents += len(indices)
        history.append(
            {
                "inference_epoch": float(phase_epoch),
                "loss": loss_sum / max(batches, 1),
                "objective_per_document": objective_sum / max(documents, 1),
                "encoder_gradient_norm": gradient_sum / max(batches, 1),
            },
        )
        elapsed = elapsed_base + time.perf_counter() - started
        _save_encoder_checkpoint(
            output,
            context=context,
            epoch=phase_epoch,
            encoder=encoder,
            optimizer=optimizer,
            rng=rng,
            history=history,
            elapsed_seconds=elapsed,
            keep=spec.checkpoint_keep,
        )
    if not np.array_equal(lambda_values, lambda_snapshot):
        msg = "encoder training changed frozen topics"
        raise RuntimeError(msg)
    model_path = output / "model.pt"
    _atomic_torch_save(
        model_path,
        {
            "schema_version": "msnlib-simplification/encoder-v1",
            "arm_id": arm_id,
            "encoder_state_dict": encoder.state_dict(),
        },
    )
    history_path = output / "history.json"
    write_json(history_path, history)
    result = {
        "schema_version": "msnlib-simplification/encoder-complete-v1",
        "arm_id": arm_id,
        "discovery": discovery,
        "inference": inference,
        "objective": objective,
        "uses_dreams": use_dreams,
        "dreams_cache_loaded": use_dreams,
        "imported_current_baseline": False,
        "epochs": spec.inference_epochs,
        "elapsed_seconds": elapsed_base + time.perf_counter() - started,
        "cpu_threads": cpu_threads,
        "parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "discovery_sha256": object_sha256(discovery_complete),
        "context_sha256": context,
        "output_sha256": {
            path.name: file_sha256(path) for path in (model_path, history_path)
        },
    }
    write_json(output / "complete.json", result)
    return result


def load_encoder(run_dir: str | Path, arm_id: str) -> EvidenceEncoder | None:
    """Load one verified encoder, returning None for the analytic arm."""
    directory = Path(run_dir).expanduser().resolve()
    discovery, inference = _arm_parts(arm_id)
    result = train_encoder(directory, arm_id)
    if result["inference"] == "analytic":
        return None
    encoder = new_encoder(directory, inference)
    path = _model_output(directory, arm_id) / "model.pt"
    if file_sha256(path) != result["output_sha256"]["model.pt"]:
        msg = "encoder model changed"
        raise ValueError(msg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("arm_id") != f"{discovery}__{inference}":
        msg = "encoder artifact arm identity changed"
        raise ValueError(msg)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    encoder.eval()
    return encoder


class InferenceSession:
    """One preloaded arm used for fair warm inference timing."""

    def __init__(self, run_dir: str | Path, arm_id: str) -> None:
        self.directory = Path(run_dir).expanduser().resolve()
        configure_cpu_threads(self.directory, "evaluation")
        self.spec = load_spec(self.directory)
        self.arm_id = arm_id
        self.discovery, self.inference = _arm_parts(arm_id)
        self.use_dreams, _ = _inference_properties(self.inference)
        alpha_values, lambda_values, _, _ = load_discovery(
            self.directory,
            self.discovery,
        )
        self.alpha = torch.from_numpy(alpha_values)
        self.expected_log_beta, self.word_topic = _word_topic(lambda_values)
        self.encoder = load_encoder(self.directory, arm_id)
        # Encoder verification can apply the training policy. Restore the
        # single-thread policy before final or timed inference.
        configure_cpu_threads(self.directory, "evaluation")

    @torch.no_grad()
    def infer(
        self,
        matrix: sp.csr_matrix,
        *,
        budget: int,
        embeddings: np.ndarray | None,
        initializer: str = "arm",
    ) -> np.ndarray:
        """Infer one matrix without reloading model artifacts."""
        if budget < 0:
            msg = "refinement budget cannot be negative"
            raise ValueError(msg)
        if self.use_dreams and embeddings is None:
            msg = "DreaMS arm requires embeddings"
            raise ValueError(msg)
        if not self.use_dreams and embeddings is not None:
            msg = "DreaMS-free arm received embeddings"
            raise ValueError(msg)
        output = np.empty((matrix.shape[0], self.spec.num_topics), dtype=np.float32)
        for start in range(0, matrix.shape[0], self.spec.batch_size):
            indices = np.arange(
                start,
                min(start + self.spec.batch_size, matrix.shape[0]),
            )
            batch = make_sparse_batch(matrix, indices, device=torch.device("cpu"))
            if initializer == "uniform":
                gamma = uniform_gamma(batch, self.alpha)
            elif initializer == "analytic" or self.inference == "analytic":
                gamma = analytic_gamma(batch, self.alpha, self.word_topic)
            else:
                if self.encoder is None:
                    msg = "trained arm has no encoder"
                    raise RuntimeError(msg)
                evidence = _evidence(batch, self.word_topic)
                embedding_batch = (
                    torch.from_numpy(np.asarray(embeddings[indices], dtype=np.float32))
                    if embeddings is not None
                    else None
                )
                gamma = self.encoder(
                    evidence,
                    batch.totals,
                    self.alpha,
                    embedding_batch,
                )
            if budget:
                gamma, _ = local_vb(
                    batch,
                    gamma,
                    self.alpha,
                    self.expected_log_beta,
                    steps=budget,
                    tolerance=None,
                )
            gamma_values = gamma.cpu().numpy()
            theta = gamma_values / np.maximum(
                gamma_values.sum(axis=1, keepdims=True),
                EPSILON,
            )
            output[start : start + len(indices)] = theta
        return output


def infer_theta(
    run_dir: str | Path,
    arm_id: str,
    matrix: sp.csr_matrix,
    *,
    budget: int,
    embeddings: np.ndarray | None,
    initializer: str = "arm",
) -> np.ndarray:
    """Infer one split with an arm, analytic, or uniform initializer."""
    return InferenceSession(run_dir, arm_id).infer(
        matrix,
        budget=budget,
        embeddings=embeddings,
        initializer=initializer,
    )


@torch.no_grad()
def local_elbo_rows(
    run_dir: str | Path,
    discovery: str,
    matrix: sp.csr_matrix,
    theta: np.ndarray,
) -> np.ndarray:
    """Evaluate rowwise local ELBO for common-reference selection."""
    directory = Path(run_dir).expanduser().resolve()
    spec = load_spec(directory)
    alpha_values, lambda_values, _, _ = load_discovery(directory, discovery)
    alpha = torch.from_numpy(alpha_values)
    expected_log_beta, _ = _word_topic(lambda_values)
    output = np.empty(matrix.shape[0], dtype=np.float64)
    totals = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    gamma_sums = totals + float(alpha.sum())
    for start in range(0, matrix.shape[0], spec.batch_size):
        indices = np.arange(start, min(start + spec.batch_size, matrix.shape[0]))
        batch = make_sparse_batch(matrix, indices, device=torch.device("cpu"))
        gamma = torch.from_numpy(
            np.asarray(theta[indices], dtype=np.float32) * gamma_sums[indices, None],
        )
        output[indices] = (
            local_document_elbo(batch, gamma, alpha, expected_log_beta).double().numpy()
        )
    return output
