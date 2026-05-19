import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import msn_benchmark_pipeline as pipeline
from scripts import run_msn_stage2_encoder_experiment as stage2


def test_train_validation_split_is_reproducible_and_nonempty():
    train_a, val_a = stage2.train_validation_split(10, train_fraction=0.8, seed=7)
    train_b, val_b = stage2.train_validation_split(10, train_fraction=0.8, seed=7)

    assert train_a.tolist() == train_b.tolist()
    assert val_a.tolist() == val_b.tolist()
    assert len(train_a) == 8
    assert len(val_a) == 2
    assert set(train_a).isdisjoint(set(val_a))


def test_theta_metrics_reports_membership_and_top1():
    teacher = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    pred = np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)

    metrics = stage2.theta_metrics(
        teacher,
        pred,
        np.array([0, 1], dtype=np.int64),
        membership_threshold=0.5,
    )

    assert metrics["top1_agreement"] == pytest.approx(0.5)
    assert metrics["pred_membership_rows_above_threshold"] == 2
    assert metrics["teacher_membership_rows_above_threshold"] == 2
    assert metrics["pred_active_topics_above_threshold"] == 1


def test_token_batch_from_csr_uses_shifted_token_ids():
    x = sparse.csr_matrix(
        np.array(
            [
                [0.0, 3.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    vocab_features = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 1.0],
        ],
        dtype=np.float32,
    )

    batch = stage2.token_batch_from_csr(
        x,
        np.array([0, 1], dtype=np.int64),
        max_tokens=2,
        vocab_features=vocab_features,
    )

    assert batch["token_ids"].tolist() == [[2, 3], [1, 0]]
    assert batch["token_mask"].tolist() == [[True, True], [True, False]]
    assert batch["token_features"][0, 0, 0] == pytest.approx(0.75)
    assert batch["token_features"][0, 1, 3] == pytest.approx(1.0)


def test_stage2_token_set_encoder_smoke_shapes(tmp_path):
    pytest.importorskip("torch")
    x = sparse.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    teacher = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    beta = np.array(
        [
            [0.45, 0.45, 0.05, 0.05],
            [0.05, 0.05, 0.45, 0.45],
        ],
        dtype=np.float32,
    )
    config = stage2.Stage2Config(
        input_cache="cache",
        teacher_model_dir="teacher",
        out_dir=str(tmp_path),
        epochs=2,
        batch_size=2,
        lr=1e-2,
        hidden_size=8,
        encoder="token-set",
        token_embedding_size=6,
        max_tokens=3,
        train_fraction=0.75,
        top1_loss_weight=0.25,
        reconstruction_loss_weight=0.1,
        background_weight=0.05,
        dropout=0.2,
        weight_decay=1e-4,
        theta_export_power=1.0,
        membership_threshold=0.5,
        seed=1,
        device="cpu",
    )

    model, history = stage2.train_encoder(
        x,
        teacher,
        beta,
        train_indices=np.array([0, 1, 2], dtype=np.int64),
        validation_indices=np.array([3], dtype=np.int64),
        vocab_features=np.array(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
                [0.4, 1.0],
            ],
            dtype=np.float32,
        ),
        config=config,
    )
    pred = stage2.infer_theta(
        model,
        x,
        config=config,
        vocab_features=np.array(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
                [0.4, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    assert len(history) == 2
    assert pred.shape == teacher.shape
    assert pred.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_fixed_beta_encoder_experiment_smoke_outputs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("tomotopy")
    from scripts import run_msn_fixed_beta_encoder_experiment as fixed_beta

    documents = [
        ["a", "a", "b"],
        ["a", "b", "b"],
        ["a", "a", "c"],
        ["d", "d", "e"],
        ["d", "e", "e"],
        ["d", "d", "f"],
        ["g", "g", "h"],
        ["g", "h", "h"],
        ["g", "g", "i"],
        ["a", "d", "g"],
    ]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    matrix, vocab, bow_metadata = pipeline.build_bow_matrix(
        documents,
        min_df=1,
        min_cf=0,
        rm_top=0,
    )
    sparse.save_npz(cache_dir / "bow.npz", matrix)
    pipeline.write_json(cache_dir / "vocab.json", {"vocab": vocab})
    pd.DataFrame(
        [
            {"doc_index": index, "smiles": f"C{index}", "spectrum_id": str(index)}
            for index in range(len(documents))
        ]
    ).to_csv(cache_dir / "spectra_metadata.csv", index=False)
    pipeline.write_documents_jsonl(cache_dir / "documents.jsonl.gz", documents)
    pipeline.write_json(
        cache_dir / "cache_summary.json",
        {
            "input": bow_metadata,
            "vocabulary_parameters": {"min_df": 1, "min_cf": 0.0, "rm_top": 0},
        },
    )

    out_dir = tmp_path / "fixed_beta"
    summary = fixed_beta.run_experiment(
        SimpleNamespace(
            input_cache=cache_dir,
            out_dir=out_dir,
            n_motifs=3,
            lda_iterations=2,
            heldout_inference_iterations=5,
            epochs=2,
            batch_size=3,
            lr=1e-2,
            hidden_size=12,
            dropout=0.0,
            weight_decay=0.0,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            min_df=1,
            min_cf=0.0,
            rm_top=0,
            top1_loss_weight=0.25,
            reconstruction_loss_weight=0.1,
            background_weight=0.05,
            theta_export_power=1.0,
            membership_threshold=0.5,
            seed=1,
            device="cpu",
            overwrite=True,
        )
    )

    split_payload = json.loads((out_dir / "split_indices.json").read_text())
    assert set(split_payload) == {
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    for model_name in ["lda_inferred", "neural_encoder"]:
        theta = np.load(out_dir / model_name / "theta.npy")
        beta = np.load(out_dir / model_name / "beta.npy")
        assert theta.shape == (len(documents), 3)
        assert beta.shape[0] == 3
        assert theta.sum(axis=1).tolist() == pytest.approx([1.0] * len(documents))
        assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert (out_dir / "neural_encoder" / "validation_metrics.json").exists()
    assert "validation_theta_cosine_mean" in summary["metrics"]["neural_encoder"]
