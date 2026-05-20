import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import msn_benchmark_pipeline as pipeline
from scripts import evaluate_motif_substructure_quality as quality
from scripts import export_msn_model_outputs as exporter
from scripts import run_msn_topic_model_experiment as experiment


def test_train_validation_test_split_is_deterministic_and_nonoverlapping():
    split_a = pipeline.train_validation_test_split(10, seed=7)
    split_b = pipeline.train_validation_test_split(10, seed=7)

    assert split_a.keys() == split_b.keys()
    for key in split_a:
        assert split_a[key].tolist() == split_b[key].tolist()
    assert len(split_a["train_indices"]) == 8
    assert len(split_a["validation_indices"]) == 1
    assert len(split_a["test_indices"]) == 1
    combined = np.concatenate(list(split_a.values()))
    assert sorted(combined.tolist()) == list(range(10))
    assert len(set(combined.tolist())) == 10


def test_build_bow_matrix_for_vocabulary_uses_explicit_order():
    docs = [
        ["frag@100", "frag@100", "loss@50", "ignored"],
        ["loss@50", "frag@200"],
    ]

    matrix = pipeline.build_bow_matrix_for_vocabulary(
        docs,
        ["loss@50", "frag@100"],
    )

    assert matrix.shape == (2, 2)
    assert matrix.toarray().tolist() == [[1.0, 2.0], [1.0, 0.0]]


def test_build_bow_matrix_filters_and_counts():
    docs = [
        ["frag@100", "frag@100", "loss@50"],
        ["frag@100", "frag@200"],
        ["frag@300"],
    ]

    matrix, vocab, metadata = experiment.build_bow_matrix(
        docs,
        min_df=2,
        min_cf=0,
        rm_top=0,
    )

    assert vocab == ["frag@100"]
    assert matrix.toarray().tolist() == [[2.0], [1.0], [0.0]]
    assert metadata["vocab_size"] == 1


def test_normalize_rows_handles_zero_rows():
    out = experiment.normalize_rows(np.array([[1.0, 3.0], [0.0, 0.0]]))

    assert out[0].tolist() == pytest.approx([0.25, 0.75])
    assert out[1].tolist() == pytest.approx([0.5, 0.5])


def test_sharpen_theta_renormalizes_rows():
    theta = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)

    out = experiment.sharpen_theta(theta, 2.0)

    assert out.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert out[0, 0] > theta[0, 0]
    assert out[1, 1] > theta[1, 1]


def test_score_summary_uses_requested_membership_threshold(tmp_path):
    scores = pd.DataFrame(
        [
            {
                "motif_id": "motif_1",
                "included_in_range": True,
                "candidate_range": "1",
                "sos": 0.75,
                "quality_bin": "intermediate",
            }
        ]
    )

    summary = quality.summarize_scores(
        scores,
        {"motif_count": 1},
        tmp_path,
        membership_threshold=0.1,
    )

    assert summary["parameters"]["membership_threshold"] == pytest.approx(0.1)
    assert summary["metrics"]["quality_adjusted_coverage"] == pytest.approx(0.75)


def test_sparsemax_outputs_sparse_probability_rows():
    torch = pytest.importorskip("torch")

    out = experiment.sparsemax(
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, -1.0]]),
        dim=1,
    )

    assert out.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert out.tolist()[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert out.tolist()[1] == pytest.approx([1.0, 0.0, 0.0])
    assert (out >= 0).all()


def test_activate_topic_distribution_supports_softmax_and_sparsemax():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0, -1.0]])

    sparse_out = pipeline.activate_topic_distribution(
        logits,
        activation="sparsemax",
        dim=1,
    )
    soft_out = pipeline.activate_topic_distribution(
        logits,
        activation="softmax",
        dim=1,
    )

    assert sparse_out.sum(dim=1).tolist() == pytest.approx([1.0])
    assert soft_out.sum(dim=1).tolist() == pytest.approx([1.0])
    assert sparse_out.tolist()[0][-1] == pytest.approx(0.0)
    assert soft_out.tolist()[0][-1] > 0.0


def test_entropy_topic_usage_loss_penalizes_collapsed_usage():
    torch = pytest.importorskip("torch")
    uniform = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    assert pipeline.topic_usage_loss(uniform, mode="entropy").item() == pytest.approx(
        0.0
    )
    assert pipeline.topic_usage_loss(collapsed, mode="entropy").item() > 0.99


def test_beta_target_support_loss_prefers_target_entropy():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
    too_broad = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)

    target_loss = pipeline.beta_target_support_loss(target, target_support=2)
    broad_loss = pipeline.beta_target_support_loss(too_broad, target_support=2)

    assert target_loss.item() == pytest.approx(0.0)
    assert broad_loss.item() > target_loss.item()


def test_membership_count_diagnostics_bins_topics():
    theta = np.array(
        [
            [0.6, 0.6, 0.1, 0.0],
            [0.6, 0.1, 0.1, 0.0],
            [0.1, 0.1, 0.6, 0.0],
            [0.1, 0.1, 0.6, 0.0],
            [0.1, 0.1, 0.6, 0.0],
        ],
        dtype=np.float32,
    )

    diagnostics = pipeline.membership_count_diagnostics(theta, thresholds=(0.5,))

    assert diagnostics["0.5"] == {
        "active_topics": 3,
        "total_memberships": 6,
        "topics_0": 1,
        "topics_1": 1,
        "topics_2_4": 2,
        "topics_5_7": 0,
        "topics_8_10": 0,
        "topics_11_plus": 0,
        "mean_nonzero_memberships": pytest.approx(2.0),
        "median_nonzero_memberships": pytest.approx(2.0),
    }


def test_run_kl_nmf_returns_normalized_theta_beta():
    matrix, _vocab, _metadata = experiment.build_bow_matrix(
        [
            ["frag@100", "frag@100", "loss@50"],
            ["frag@100", "frag@200", "frag@200"],
            ["loss@50", "loss@50", "frag@200"],
        ],
        min_df=1,
        min_cf=0,
        rm_top=0,
    )

    theta, beta, metadata = experiment.run_kl_nmf(
        matrix,
        n_motifs=2,
        max_iter=20,
        seed=1,
    )

    assert theta.shape == (3, 2)
    assert beta.shape == (2, 3)
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert metadata["nmf_n_iter"] > 0


def test_run_tomotopy_lda_returns_normalized_outputs():
    pytest.importorskip("tomotopy")

    theta, beta, vocab, history, metadata = experiment.run_tomotopy_lda(
        [
            ["frag@100", "frag@100", "loss@50"],
            ["frag@100", "frag@200", "frag@200"],
            ["loss@50", "loss@50", "frag@200"],
        ],
        n_motifs=2,
        min_df=1,
        min_cf=0,
        rm_top=0,
        lda_iterations=2,
        seed=1,
    )

    assert theta.shape == (3, 2)
    assert beta.shape == (2, len(vocab))
    assert len(history) == 1
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert metadata["lda_k"] == 2


def test_select_eval_topic_ids_uses_membership_counts_then_strength():
    theta = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float32,
    )
    beta = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.5, 0.1],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float32,
    )

    assert experiment.select_eval_topic_ids(
        theta,
        beta,
        max_eval_motifs=2,
        membership_threshold=0.5,
    ) == [0, 1]


def test_export_memberships_filters_topics(sample_spectra_list):
    theta = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
    sample_spectra_list[0].set("smiles", "CCO")
    sample_spectra_list[1].set("smiles", "CCC")

    memberships = experiment.export_memberships(
        theta,
        sample_spectra_list[:2],
        [0, 1],
        membership_threshold=0.5,
    )

    assert memberships.to_dict("records") == [
        {
            "motif_id": "motif_0",
            "smiles": "CCO",
            "membership_score": pytest.approx(0.6),
        },
        {
            "motif_id": "motif_1",
            "smiles": "CCC",
            "membership_score": pytest.approx(0.8),
        },
    ]


def test_input_cache_round_trip(tmp_path):
    matrix = sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))
    sparse.save_npz(tmp_path / "bow.npz", matrix)
    pipeline.write_json(tmp_path / "vocab.json", {"vocab": ["frag@100", "loss@50"]})
    pd.DataFrame(
        [
            {"doc_index": 0, "smiles": "CCO"},
            {"doc_index": 1, "smiles": "CCC"},
        ]
    ).to_csv(tmp_path / "spectra_metadata.csv", index=False)
    pipeline.write_documents_jsonl(
        tmp_path / "documents.jsonl.gz",
        [["frag@100", "frag@100"], ["loss@50", "loss@50", "loss@50"]],
    )
    pipeline.write_json(
        tmp_path / "cache_summary.json",
        {
            "input": {"documents": 2, "vocab_size": 2},
            "vocabulary_parameters": {"min_df": 1, "min_cf": 0.0, "rm_top": 0},
        },
    )

    cache = pipeline.load_input_cache(tmp_path, require_documents=True)

    assert cache["matrix"].toarray().tolist() == [[2.0, 0.0], [0.0, 3.0]]
    assert cache["vocab"] == ["frag@100", "loss@50"]
    assert cache["documents"] == [
        ["frag@100", "frag@100"],
        ["loss@50", "loss@50", "loss@50"],
    ]
    assert cache["spectra_metadata"]["smiles"].tolist() == ["CCO", "CCC"]


def test_export_memberships_accepts_cached_metadata():
    theta = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
    metadata = pd.DataFrame(
        [
            {"doc_index": 0, "smiles": "CCO"},
            {"doc_index": 1, "smiles": "CCC"},
        ]
    )

    memberships = pipeline.export_memberships(
        theta,
        metadata,
        [0, 1],
        membership_threshold=0.5,
    )

    assert memberships.to_dict("records") == [
        {
            "motif_id": "motif_0",
            "smiles": "CCO",
            "membership_score": pytest.approx(0.6),
        },
        {
            "motif_id": "motif_1",
            "smiles": "CCC",
            "membership_score": pytest.approx(0.8),
        },
    ]


def test_split_aware_export_selects_only_requested_metadata_rows(tmp_path):
    split_path = tmp_path / "split_indices.json"
    pipeline.write_json(
        split_path,
        {
            "train_indices": [0, 2],
            "validation_indices": [1],
            "test_indices": [3],
        },
    )
    theta = np.array(
        [
            [0.7, 0.3],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        [
            {"doc_index": 0, "smiles": "CCO"},
            {"doc_index": 1, "smiles": "CCC"},
            {"doc_index": 2, "smiles": "CCN"},
            {"doc_index": 3, "smiles": "CNC"},
        ]
    )

    test_indices = exporter.load_split_indices(split_path, "test")
    theta_export, metadata_export = exporter.select_export_split(
        theta,
        metadata,
        test_indices,
    )
    memberships = pipeline.export_memberships(
        theta_export,
        metadata_export,
        [0, 1],
        membership_threshold=0.5,
    )

    assert metadata_export["smiles"].tolist() == ["CNC"]
    assert memberships.to_dict("records") == [
        {
            "motif_id": "motif_1",
            "smiles": "CNC",
            "membership_score": pytest.approx(0.9),
        }
    ]


def test_sparse_neural_smoke_shapes():
    pytest.importorskip("torch")
    matrix, _vocab, _metadata = experiment.build_bow_matrix(
        [
            ["frag@100", "frag@100", "loss@50"],
            ["frag@100", "frag@200", "frag@200"],
            ["loss@50", "loss@50", "frag@200"],
            ["frag@300", "frag@300", "loss@50"],
        ],
        min_df=1,
        min_cf=0,
        rm_top=0,
    )

    theta, beta, history, metadata, _checkpoint = experiment.train_sparse_neural(
        matrix,
        n_motifs=2,
        hidden_size=8,
        dropout=0.0,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        theta_entropy_weight=0.1,
        beta_entropy_weight=0.1,
        background_weight=0.05,
        topic_overlap_weight=0.05,
        topic_usage_weight=0.1,
        seed=1,
        device="cpu",
    )

    assert theta.shape == (4, 2)
    assert beta.shape == (2, 4)
    assert len(history) == 2
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert metadata["sparse_neural_init"] == "random"


def test_sparse_neural_softmax_entropy_usage_smoke_shapes():
    pytest.importorskip("torch")
    matrix, _vocab, _metadata = experiment.build_bow_matrix(
        [
            ["frag@100", "frag@100", "loss@50"],
            ["frag@100", "frag@200", "frag@200"],
            ["loss@50", "loss@50", "frag@200"],
            ["frag@300", "frag@300", "loss@50"],
        ],
        min_df=1,
        min_cf=0,
        rm_top=0,
    )

    theta, beta, history, metadata, _checkpoint = experiment.train_sparse_neural(
        matrix,
        n_motifs=2,
        hidden_size=8,
        dropout=0.0,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        theta_entropy_weight=0.0,
        beta_entropy_weight=-0.01,
        background_weight=0.05,
        topic_overlap_weight=0.05,
        topic_usage_weight=1.0,
        theta_activation="sparsemax",
        beta_activation="softmax",
        topic_usage_mode="entropy",
        seed=1,
        device="cpu",
    )

    assert theta.shape == (4, 2)
    assert beta.shape == (2, 4)
    assert len(history) == 2
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert metadata["theta_activation"] == "sparsemax"
    assert metadata["beta_activation"] == "softmax"
    assert metadata["topic_usage_mode"] == "entropy"


def test_neural_lda_smoke_shapes():
    pytest.importorskip("torch")
    matrix, _vocab, _metadata = experiment.build_bow_matrix(
        [
            ["frag@100", "frag@100", "loss@50"],
            ["frag@100", "frag@200", "frag@200"],
            ["loss@50", "loss@50", "frag@200"],
            ["frag@300", "frag@300", "loss@50"],
        ],
        min_df=1,
        min_cf=0,
        rm_top=0,
    )

    theta, beta, history, metadata, _checkpoint = experiment.train_neural_lda(
        matrix,
        n_motifs=2,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        theta_entropy_weight=0.05,
        topic_usage_weight=1.0,
        beta_target_support=2.0,
        beta_target_weight=0.1,
        background_weight=0.05,
        theta_init_strength=4.0,
        seed=1,
        device="cpu",
    )

    assert theta.shape == (4, 2)
    assert beta.shape == (2, 4)
    assert len(history) == 2
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert metadata["neural_lda_init"] == "document_topic_bias"


def test_amortized_neural_topic_experiment_smoke_without_documents_or_tomotopy(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from scripts import run_msn_amortized_neural_topic_experiment as amortized

    monkeypatch.setitem(__import__("sys").modules, "tomotopy", None)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    matrix = sparse.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    sparse.save_npz(cache_dir / "bow.npz", matrix)
    pipeline.write_json(
        cache_dir / "vocab.json",
        {"vocab": ["frag@100", "frag@101", "loss@50", "loss@51"]},
    )
    pd.DataFrame(
        [
            {"doc_index": index, "smiles": f"C{index}"}
            for index in range(matrix.shape[0])
        ]
    ).to_csv(cache_dir / "spectra_metadata.csv", index=False)
    pipeline.write_json(
        cache_dir / "cache_summary.json",
        {
            "input": {"documents": int(matrix.shape[0]), "vocab_size": matrix.shape[1]},
            "vocabulary_parameters": {"min_df": 1, "min_cf": 0.0, "rm_top": 0},
        },
    )

    out_dir = tmp_path / "amortized"
    summary = amortized.run_experiment(
        SimpleNamespace(
            input_cache=cache_dir,
            out_dir=out_dir,
            n_motifs=3,
            epochs=2,
            batch_size=3,
            lr=1e-2,
            hidden_size=8,
            dropout=0.0,
            weight_decay=0.0,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            local_reconstruction_weight=1.0,
            encoder_reconstruction_weight=1.0,
            consistency_weight=1.0,
            theta_entropy_weight=0.05,
            topic_usage_weight=1.0,
            encoder_topic_usage_weight=0.1,
            beta_target_support=2.0,
            beta_target_weight=0.1,
            background_weight=0.05,
            theta_init_strength=4.0,
            theta_export_power=1.5,
            membership_threshold=0.5,
            seed=1,
            device="cpu",
            overwrite=True,
        )
    )

    theta = np.load(out_dir / "theta.npy")
    theta_raw = np.load(out_dir / "theta_raw.npy")
    beta = np.load(out_dir / "beta.npy")
    split_payload = pd.read_json(out_dir / "split_indices.json", typ="series")

    assert summary["model"] == "amortized-neural-topic"
    assert theta.shape == (matrix.shape[0], 3)
    assert theta_raw.shape == theta.shape
    assert beta.shape == (3, matrix.shape[1])
    assert np.isfinite(theta).all()
    assert np.isfinite(beta).all()
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0] * matrix.shape[0])
    assert theta_raw.sum(axis=1).tolist() == pytest.approx([1.0] * matrix.shape[0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert set(split_payload.index) == {
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    assert (out_dir / "model_checkpoint.pt").exists()
    assert (out_dir / "validation_metrics.json").exists()


def test_prodlda_experiment_smoke_without_documents_tomotopy_or_pyro(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from scripts import run_msn_prodlda_experiment as prodlda

    monkeypatch.setitem(__import__("sys").modules, "tomotopy", None)
    monkeypatch.setitem(__import__("sys").modules, "pyro", None)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    matrix = sparse.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    sparse.save_npz(cache_dir / "bow.npz", matrix)
    pipeline.write_json(
        cache_dir / "vocab.json",
        {"vocab": ["frag@100", "frag@101", "loss@50", "loss@51"]},
    )
    pd.DataFrame(
        [
            {"doc_index": index, "smiles": f"C{index}"}
            for index in range(matrix.shape[0])
        ]
    ).to_csv(cache_dir / "spectra_metadata.csv", index=False)
    pipeline.write_json(
        cache_dir / "cache_summary.json",
        {
            "input": {"documents": int(matrix.shape[0]), "vocab_size": matrix.shape[1]},
            "vocabulary_parameters": {"min_df": 1, "min_cf": 0.0, "rm_top": 0},
        },
    )

    out_dir = tmp_path / "prodlda"
    summary = prodlda.run_experiment(
        SimpleNamespace(
            input_cache=cache_dir,
            out_dir=out_dir,
            n_motifs=3,
            epochs=2,
            batch_size=3,
            lr=1e-3,
            hidden_size=8,
            dropout=0.0,
            weight_decay=0.0,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            kl_weight=0.1,
            kl_anneal_epochs=2,
            theta_entropy_weight=0.0,
            topic_usage_weight=0.0,
            beta_target_support=2.0,
            beta_target_weight=0.0,
            background_weight=0.0,
            beta_init_noise=0.01,
            theta_export_power=1.5,
            membership_threshold=0.5,
            seed=1,
            device="cpu",
            overwrite=True,
        )
    )

    theta = np.load(out_dir / "theta.npy")
    theta_raw = np.load(out_dir / "theta_raw.npy")
    beta = np.load(out_dir / "beta.npy")
    split_payload = pd.read_json(out_dir / "split_indices.json", typ="series")

    assert summary["model"] == "prodlda"
    assert theta.shape == (matrix.shape[0], 3)
    assert theta_raw.shape == theta.shape
    assert beta.shape == (3, matrix.shape[1])
    assert np.isfinite(theta).all()
    assert np.isfinite(theta_raw).all()
    assert np.isfinite(beta).all()
    assert theta.sum(axis=1).tolist() == pytest.approx([1.0] * matrix.shape[0])
    assert theta_raw.sum(axis=1).tolist() == pytest.approx([1.0] * matrix.shape[0])
    assert beta.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert set(split_payload.index) == {
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    assert (out_dir / "model_checkpoint.pt").exists()
    assert (out_dir / "validation_metrics.json").exists()
