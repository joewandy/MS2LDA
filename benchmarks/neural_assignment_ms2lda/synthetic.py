# ruff: noqa: C901, PERF401, PLR0912, PLR0913, PLR0915, PLR2004
"""Two deterministic synthetic recovery gates for the assignment model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment

from .data import (
    ViewPair,
    iter_row_batches,
    prototype_seeding_weights,
    sparse_batch,
)
from .metrics import completion_metrics
from .model import initialize_model, router_block_loss, topic_block_loss
from .training import infer_theta, routing_temperature, sinkhorn_weight
from .utils import (
    atomic_save_numpy,
    atomic_torch_save,
    file_sha256,
    object_sha256,
    read_json,
    write_json,
)


@dataclass(frozen=True)
class SyntheticDataset:
    """One generated recovery problem and its known topics."""

    train: sp.csr_matrix
    views: list[ViewPair]
    validation_observed: sp.csr_matrix
    validation_completion: sp.csr_matrix
    validation_full: sp.csr_matrix
    true_beta: np.ndarray
    token_features: np.ndarray


def _draw_documents(
    rng: np.random.Generator,
    *,
    beta: np.ndarray,
    documents: int,
    tokens_per_document: int,
    prevalence: np.ndarray,
) -> sp.csr_matrix:
    rows = []
    for _ in range(documents):
        active = int(rng.integers(1, 4))
        topics = rng.choice(
            len(beta),
            size=active,
            replace=False,
            p=prevalence,
        )
        mixture = rng.dirichlet(np.full(active, 0.35))
        distribution = mixture @ beta[topics]
        rows.append(rng.multinomial(tokens_per_document, distribution))
    return sp.csr_matrix(np.asarray(rows, dtype=np.float32))


def _masked(
    matrix: sp.csr_matrix,
    *,
    probability: float,
    seed: int,
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = matrix.toarray().astype(np.int64)
    masked = rng.binomial(dense, probability)
    for row in range(len(masked)):
        if masked[row].sum() == 0:
            available = np.flatnonzero(dense[row])
            selected = int(rng.choice(available))
            masked[row, selected] = 1
    return sp.csr_matrix(masked.astype(np.float32))


def generate_synthetic(
    scenario: str,
    *,
    seed: int,
    num_topics: int = 32,
) -> SyntheticDataset:
    """Generate a separable or long-tail/shared-background recovery problem."""
    if scenario not in {"separable", "long_tail_shared_background"}:
        msg = "unknown synthetic scenario"
        raise ValueError(msg)
    rng = np.random.default_rng(seed + (0 if scenario == "separable" else 10_000))
    anchor_words = 24
    background_words = 64
    vocabulary_size = num_topics * anchor_words + background_words
    true_beta = np.zeros((num_topics, vocabulary_size), dtype=np.float64)
    anchor_mass = 0.95 if scenario == "separable" else 0.72
    background_mass = 1.0 - anchor_mass
    for topic in range(num_topics):
        start = topic * anchor_words
        true_beta[topic, start : start + anchor_words] = anchor_mass / anchor_words
        true_beta[topic, -background_words:] = background_mass / background_words
        if scenario == "long_tail_shared_background":
            neighbour = ((topic + 1) % num_topics) * anchor_words
            true_beta[topic] *= 0.93
            true_beta[topic, neighbour : neighbour + anchor_words] += (
                0.07 / anchor_words
            )
    true_beta /= true_beta.sum(axis=1, keepdims=True)

    features = np.zeros((vocabulary_size, 64), dtype=np.float32)
    centers = rng.normal(size=(num_topics, 64))
    centers[:, :num_topics] += 5.0 * np.eye(num_topics)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for topic in range(num_topics):
        start = topic * anchor_words
        features[start : start + anchor_words] = centers[topic] + rng.normal(
            scale=0.035,
            size=(anchor_words, 64),
        )
    features[-background_words:] = rng.normal(
        scale=0.35,
        size=(background_words, 64),
    )
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)

    if scenario == "separable":
        prevalence = np.full(num_topics, 1.0 / num_topics)
    else:
        prevalence = 1.0 / np.power(np.arange(1, num_topics + 1), 0.65)
        prevalence /= prevalence.sum()
    train = _draw_documents(
        rng,
        beta=true_beta,
        documents=768,
        tokens_per_document=96,
        prevalence=prevalence,
    )
    validation_full = _draw_documents(
        rng,
        beta=true_beta,
        documents=192,
        tokens_per_document=96,
        prevalence=prevalence,
    )
    views = []
    for pair_index in range(4):
        views.append(
            ViewPair(
                left=_masked(
                    train,
                    probability=0.8,
                    seed=seed + pair_index * 2 + 1,
                ),
                right=_masked(
                    train,
                    probability=0.8,
                    seed=seed + pair_index * 2 + 2,
                ),
                pair_index=pair_index,
            ),
        )
    validation_observed = _masked(
        validation_full,
        probability=0.8,
        seed=seed + 909,
    )
    validation_completion = validation_full - validation_observed
    validation_completion.eliminate_zeros()
    return SyntheticDataset(
        train=train,
        views=views,
        validation_observed=validation_observed,
        validation_completion=validation_completion,
        validation_full=validation_full,
        true_beta=true_beta.astype(np.float32),
        token_features=features,
    )


def _matched_topic_metrics(
    learned_beta: np.ndarray,
    true_beta: np.ndarray,
) -> dict[str, Any]:
    learned_norm = learned_beta / np.maximum(
        np.linalg.norm(learned_beta, axis=1, keepdims=True),
        1e-12,
    )
    true_norm = true_beta / np.maximum(
        np.linalg.norm(true_beta, axis=1, keepdims=True),
        1e-12,
    )
    similarities = learned_norm @ true_norm.T
    learned_rows, true_rows = linear_sum_assignment(-similarities)
    cosines = similarities[learned_rows, true_rows]
    jaccards = []
    top_n = min(20, learned_beta.shape[1])
    for learned, truth in zip(learned_rows, true_rows, strict=True):
        learned_top = set(
            np.argsort(-learned_beta[learned], kind="stable")[:top_n].tolist(),
        )
        true_top = set(
            np.argsort(-true_beta[truth], kind="stable")[:top_n].tolist(),
        )
        jaccards.append(len(learned_top & true_top) / len(learned_top | true_top))
    return {
        "matched_beta_cosine_mean": float(np.mean(cosines)),
        "matched_beta_cosine_median": float(np.median(cosines)),
        "matched_top20_jaccard_mean": float(np.mean(jaccards)),
        "matched_top20_jaccard_median": float(np.median(jaccards)),
        "assignment": [
            {"learned_topic": int(left), "true_topic": int(right)}
            for left, right in zip(learned_rows, true_rows, strict=True)
        ],
    }


@torch.inference_mode()
def _evaluate(
    model: Any,
    dataset: SyntheticDataset,
    *,
    temperature: float,
    top_k: int,
    batch_size: int,
    occupation_usage_fraction_of_uniform: float,
) -> dict[str, Any]:
    beta = model.topic_word_distribution().cpu().numpy().astype(np.float32)
    observed_theta = infer_theta(
        model,
        dataset.validation_observed,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    train_theta = infer_theta(
        model,
        dataset.train,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    usage = train_theta.mean(axis=0)
    occupation_threshold = (
        float(occupation_usage_fraction_of_uniform) / model.num_topics
    )
    records = [
        {"completion_oov_tokens": 0}
        for _ in range(dataset.validation_completion.shape[0])
    ]
    completion, _ = completion_metrics(
        observed_theta,
        beta,
        dataset.validation_completion,
        records,
    )
    frequencies = np.asarray(dataset.train.sum(axis=0)).ravel().astype(np.float64)
    unigram = frequencies / frequencies.sum()
    total_loss = 0.0
    total_tokens = 0.0
    for row in range(dataset.validation_completion.shape[0]):
        start = dataset.validation_completion.indptr[row]
        stop = dataset.validation_completion.indptr[row + 1]
        words = dataset.validation_completion.indices[start:stop]
        counts = dataset.validation_completion.data[start:stop]
        total_loss -= float(
            np.sum(counts * np.log(np.clip(unigram[words], 1e-12, None))),
        )
        total_tokens += float(counts.sum())
    unigram_nll = total_loss / total_tokens
    return {
        **_matched_topic_metrics(beta, dataset.true_beta),
        "occupied_topics": int(np.sum(usage >= occupation_threshold)),
        "occupied_fraction": float(np.mean(usage >= occupation_threshold)),
        "occupation_usage_fraction_of_uniform": float(
            occupation_usage_fraction_of_uniform,
        ),
        "completion_nll": float(completion["nll_per_token"]),
        "unigram_completion_nll": float(unigram_nll),
        "completion_nll_fraction_of_unigram": float(
            completion["nll_per_token"] / unigram_nll,
        ),
    }


def synthetic_gate_checks(
    metrics: dict[str, Any],
    *,
    stable: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply every frozen synthetic recovery gate."""
    gates = protocol["synthetic_gates"]
    checks = {
        "stable": {"pass": stable, "actual": stable, "required": True},
        "occupied_fraction": {
            "pass": metrics["occupied_fraction"] >= gates["minimum_occupied_fraction"],
            "actual": metrics["occupied_fraction"],
            "minimum": gates["minimum_occupied_fraction"],
        },
        "matched_beta_cosine": {
            "pass": metrics["matched_beta_cosine_mean"]
            >= gates["minimum_matched_beta_cosine"],
            "actual": metrics["matched_beta_cosine_mean"],
            "minimum": gates["minimum_matched_beta_cosine"],
        },
        "top20_jaccard": {
            "pass": metrics["matched_top20_jaccard_mean"]
            >= gates["minimum_top20_jaccard"],
            "actual": metrics["matched_top20_jaccard_mean"],
            "minimum": gates["minimum_top20_jaccard"],
        },
        "completion_vs_unigram": {
            "pass": metrics["completion_nll_fraction_of_unigram"]
            <= gates["maximum_nll_fraction_of_unigram"],
            "actual": metrics["completion_nll_fraction_of_unigram"],
            "maximum": gates["maximum_nll_fraction_of_unigram"],
        },
    }
    return {
        "checks": checks,
        "pass": all(row["pass"] for row in checks.values()),
        "failed": [name for name, row in checks.items() if not row["pass"]],
    }


def train_synthetic_scenario(
    run_dir: str | Path,
    *,
    scenario: str,
    protocol: dict[str, Any],
    maximum_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train or exactly resume one synthetic scenario."""
    directory = Path(run_dir) / "stages" / "synthetic" / scenario
    complete_path = directory / "complete.json"
    model_path = directory / "model.pt"
    if complete_path.is_file():
        result = read_json(complete_path)
        if file_sha256(model_path) != result["model_sha256"]:
            msg = f"synthetic {scenario} model changed"
            raise ValueError(msg)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    stage = protocol["stages"]["synthetic"]
    num_topics = int(stage["num_topics"])
    seed = int(protocol["seed"]) + (0 if scenario == "separable" else 10_000)
    dataset = generate_synthetic(scenario, seed=seed, num_topics=num_topics)
    data_identity = {
        "scenario": scenario,
        "seed": seed,
        "train_shape": list(dataset.train.shape),
        "train_sum": float(dataset.train.sum()),
        "true_beta_sha256": object_sha256(dataset.true_beta.tolist()),
        "features_sha256": object_sha256(dataset.token_features.tolist()),
    }
    write_json(directory / "data_manifest.json", data_identity)
    atomic_save_numpy(directory / "true_beta.npy", dataset.true_beta)
    atomic_save_numpy(directory / "token_features.npy", dataset.token_features)

    model, initial_indices = initialize_model(
        torch.from_numpy(dataset.token_features),
        num_topics=num_topics,
        protocol=protocol,
        seeding_weights=prototype_seeding_weights(dataset.train),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["optimization"]["learning_rate"]),
        weight_decay=float(protocol["optimization"]["weight_decay"]),
    )
    latest_path = directory / "checkpoint_latest.pt"
    best_path = directory / "checkpoint_best.pt"
    epoch_start = 0
    history: list[dict[str, Any]] = []
    best_key: tuple[int, float] | None = None
    elapsed_before = 0.0
    stable = True
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        best_key = (
            tuple(checkpoint["best_key"])
            if checkpoint["best_key"] is not None
            else None
        )
        elapsed_before = float(checkpoint["elapsed_seconds"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
    started = time.perf_counter()
    maximum_seconds = float(stage["maximum_hours"]) * 3600.0
    maximum_epochs = (
        int(maximum_epochs_override)
        if maximum_epochs_override is not None
        else int(stage["maximum_epochs"])
    )
    best_epoch = None
    stop_reason = "maximum_epochs"
    batch_size = 128
    model_config = protocol["model"]
    optimization = protocol["optimization"]
    for epoch in range(epoch_start, maximum_epochs):
        model.train()
        pair = dataset.views[epoch % len(dataset.views)]
        temperature = routing_temperature(
            epoch,
            attempt="primary",
            rescue_mode=None,
            protocol=protocol,
        )
        balance = sinkhorn_weight(epoch, attempt="primary", protocol=protocol)
        with torch.no_grad():
            cached_beta = model.topic_word_distribution().detach()
        losses = []
        for rows in iter_row_batches(
            dataset.train.shape[0],
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            terms = router_block_loss(
                model,
                sparse_batch(pair.left, rows),
                sparse_batch(pair.right, rows),
                cached_beta=cached_beta,
                temperature=temperature,
                top_k=int(model_config["top_k"]),
                sinkhorn_weight=balance,
                consistency_weight=float(
                    optimization["theta_consistency_weight"],
                ),
                sinkhorn_epsilon=float(model_config["sinkhorn_epsilon"]),
                sinkhorn_iterations=int(model_config["sinkhorn_iterations"]),
            )
            if not torch.isfinite(terms.total):
                stable = False
                stop_reason = "non_finite_router_loss"
                break
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(model_config["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(terms.total.detach()))
        if stable:
            topic_rows = list(
                iter_row_batches(
                    dataset.train.shape[0],
                    batch_size=int(optimization["topic_update_batch_size"]),
                    shuffle=True,
                    seed=seed + 100_003 + epoch,
                ),
            )
            for update in range(int(optimization["topic_updates_per_epoch"])):
                rows = topic_rows[update % len(topic_rows)]
                optimizer.zero_grad(set_to_none=True)
                terms = topic_block_loss(
                    model,
                    sparse_batch(pair.left, rows),
                    sparse_batch(pair.right, rows),
                    temperature=temperature,
                    top_k=int(model_config["top_k"]),
                    local_decoder_weight=float(
                        optimization["local_decoder_weight"],
                    ),
                )
                if not torch.isfinite(terms.total):
                    stable = False
                    stop_reason = "non_finite_topic_loss"
                    break
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(model_config["gradient_clip_norm"]),
                )
                optimizer.step()
        elapsed = elapsed_before + time.perf_counter() - started
        metrics = None
        gate = None
        if (epoch + 1) % int(stage["validation_interval"]) == 0 and stable:
            model.eval()
            final_temperature = routing_temperature(
                epoch + 1,
                attempt="primary",
                rescue_mode=None,
                protocol=protocol,
            )
            metrics = _evaluate(
                model,
                dataset,
                temperature=final_temperature,
                top_k=int(model_config["top_k"]),
                batch_size=batch_size,
                occupation_usage_fraction_of_uniform=float(
                    protocol["synthetic_gates"]["occupation_usage_fraction_of_uniform"],
                ),
            )
            gate = synthetic_gate_checks(metrics, stable=stable, protocol=protocol)
            candidate_key = (
                len(gate["failed"]),
                -float(metrics["matched_beta_cosine_mean"]),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_epoch = epoch + 1
                atomic_torch_save(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "metrics": metrics,
                        "gate": gate,
                    },
                )
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else None,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "gate": gate,
            "stable": stable,
        }
        history.append(row)
        atomic_torch_save(
            latest_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_key": list(best_key) if best_key is not None else None,
                "elapsed_seconds": elapsed,
                "torch_rng_state": torch.get_rng_state(),
            },
        )
        write_json(directory / "history.json", history)
        if not stable:
            break
        if elapsed >= maximum_seconds:
            stop_reason = "wall_clock_cap"
            break
        if (
            gate is not None
            and gate["pass"]
            and epoch + 1 >= int(stage["minimum_epochs"])
        ):
            stop_reason = "synthetic_gates_passed"
            break
    if not best_path.is_file():
        msg = f"synthetic {scenario} produced no validation checkpoint"
        raise RuntimeError(msg)
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    atomic_torch_save(
        model_path,
        {
            "schema_version": "neural-assignment-ms2lda/synthetic-model-v1",
            "scenario": scenario,
            "model": model.state_dict(),
            "initial_indices": initial_indices,
            "selected_epoch": selected["epoch"],
            "single_routing_pass": True,
        },
    )
    result = {
        "schema_version": "neural-assignment-ms2lda/synthetic-complete-v1",
        "scenario": scenario,
        "stable": stable,
        "stop_reason": stop_reason,
        "selected_epoch": int(selected["epoch"]),
        "metrics": selected["metrics"],
        "gate": selected["gate"],
        "single_routing_pass": True,
        "local_vb_steps": 0,
        "model_sha256": file_sha256(model_path),
    }
    write_json(complete_path, result)
    return result


def run_synthetic_gate(
    run_dir: str | Path,
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Run both synthetic scenarios and require both to pass."""
    directory = Path(run_dir)
    complete_path = directory / "stages" / "synthetic" / "complete.json"
    if complete_path.is_file():
        return read_json(complete_path)
    results = [
        train_synthetic_scenario(directory, scenario=scenario, protocol=protocol)
        for scenario in ("separable", "long_tail_shared_background")
    ]
    result = {
        "schema_version": "neural-assignment-ms2lda/synthetic-gate-v1",
        "scenarios": results,
        "pass": all(row["gate"]["pass"] for row in results),
        "failed_scenarios": [
            row["scenario"] for row in results if not row["gate"]["pass"]
        ],
    }
    write_json(complete_path, result)
    return result
