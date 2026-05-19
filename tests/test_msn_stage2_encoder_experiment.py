import numpy as np
import pytest
from scipy import sparse

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
