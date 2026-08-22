"""Focused software and miniature end-to-end tests for neural MS2LDA."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmarks.msnlib_validation.data import (
    PeakGroup,
    SpectrumRecord,
    audit_split_disjointness,
    build_training_vocabulary,
)
from benchmarks.neural_assignment_ms2lda import config as neural_config
from benchmarks.neural_assignment_ms2lda.bundle import (
    _portable_provenance,
    load_bundle,
    package_bundle,
)
from benchmarks.neural_assignment_ms2lda.chemical import _sos_bands
from benchmarks.neural_assignment_ms2lda.config import load_protocol
from benchmarks.neural_assignment_ms2lda.cooccurrence import (
    positive_npmi_graph,
    torch_sparse_graph,
)
from benchmarks.neural_assignment_ms2lda.core import (
    prepare_initialization,
    prepare_token_features,
)
from benchmarks.neural_assignment_ms2lda.data import (
    load_csr,
    load_heldout_records,
    load_view_pairs,
    prepare_data,
    prepare_training_views,
    select_view_peak_groups,
    sparse_batch,
)
from benchmarks.neural_assignment_ms2lda.development import (
    DEVELOPMENT_DATA_FILES,
    READ_ONLY_STAGES,
    _development_protocol,
    _link_read_only_inputs,
    _validate_frozen_protocol,
    _verify_source_artifacts,
)
from benchmarks.neural_assignment_ms2lda.embeddings import train_sgns
from benchmarks.neural_assignment_ms2lda.evaluation import evaluate_neural
from benchmarks.neural_assignment_ms2lda.gates import evaluate_validation_gate
from benchmarks.neural_assignment_ms2lda.model import (
    balanced_sinkhorn_targets,
    recycle_dead_prototypes,
    router_block_loss,
    topic_block_loss,
)
from benchmarks.neural_assignment_ms2lda.regularizers import (
    cooccurrence_topic_constraint,
    nearest_neighbor_topic_constraint,
)
from benchmarks.neural_assignment_ms2lda.report import build_machine_report
from benchmarks.neural_assignment_ms2lda.tomotopy import (
    REFERENCE_DATA_FILES,
    _alpha_evidence,
    _infer_theta,
    tomotopy_reference_evidence,
)
from benchmarks.neural_assignment_ms2lda.training import (
    _selection,
    _weighted_topic_separation,
    train_model,
)
from benchmarks.neural_assignment_ms2lda.utils import (
    file_sha256,
    write_json,
    write_jsonl,
)
from scripts.download_msnlib_validation_assets import (
    RECORD_API,
    RECORD_ID,
    safe_zip_members,
    validate_acquisition_manifest,
)


def _record(identifier: str, words: list[str]) -> SpectrumRecord:
    groups = tuple(
        PeakGroup(index, 100.0 + index, 1.0, (word,))
        for index, word in enumerate(words)
    )
    return SpectrumRecord(
        spectrum_id=identifier,
        feature_id=identifier,
        smiles="CCO",
        supplied_inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        connectivity_key=identifier,
        scaffold_key="",
        split_group=identifier,
        precursor_mz=300.0,
        peak_groups=groups,
        declared_num_peaks=len(groups),
        parsed_num_peaks=len(groups),
        compound_name=identifier,
        metadata={},
    )


def _token_features(tokens: int, dimensions: int = 64) -> torch.Tensor:
    """Return synthetic features with the production fragment/loss contract."""
    features = torch.randn(tokens, dimensions)
    features[:, -2:] = 0.0
    features[::2, -2] = 1.0
    features[1::2, -1] = 1.0
    return torch.nn.functional.normalize(features, dim=1)


def test_first_seen_training_vocabulary() -> None:
    records = [
        _record("a", ["frag@2.0", "frag@1.0", "frag@2.0"]),
        _record("b", ["loss@3.0", "frag@1.0"]),
        _record("c", ["frag@9.0"]),
    ]
    assignments = {"a": "train", "b": "train", "c": "test"}
    vocabulary, summary = build_training_vocabulary(
        records, assignments, min_df=1, min_cf=0, rm_top=0
    )
    assert vocabulary == ("frag@2.0", "frag@1.0", "loss@3.0")
    assert summary["order"] == "raw_training_spectra_first_seen"
    assert "frag@9.0" not in vocabulary


def test_split_audit_rejects_compound_leakage() -> None:
    records = [_record("a", ["frag@1.0"]), _record("b", ["frag@2.0"])]
    records[1] = SpectrumRecord(
        **{**records[1].__dict__, "connectivity_key": records[0].connectivity_key}
    )
    with pytest.raises(ValueError, match="split leakage"):
        audit_split_disjointness(records, {"a": "train", "b": "test"})


def test_training_views_keep_physical_peak_groups_atomic() -> None:
    groups = (
        PeakGroup(0, 100.0, 1.0, ("frag@100.0", "loss@200.0")),
        PeakGroup(1, 110.0, 0.5, ("frag@110.0", "loss@190.0")),
    )
    selected = select_view_peak_groups(
        groups,
        spectrum_id="spectrum",
        seed=42,
        pair_index=0,
        side="left",
        retained_fraction=0.5,
    )
    assert len(selected) == 1
    assert selected[0] in groups
    assert len(selected[0].tokens) == 2


def test_sinkhorn_top2_topic_gradients_and_recycling() -> None:
    protocol = load_protocol()
    features = _token_features(20)
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    matrix = sp.csr_matrix(np.eye(4, 20, dtype=np.float32) + 1.0)
    batch = sparse_batch(matrix, np.arange(4, dtype=np.int64))
    beta = model.topic_word_distribution().detach()
    router = router_block_loss(
        model,
        batch,
        batch,
        cached_beta=beta,
        temperature=0.5,
        top_k=2,
        sinkhorn_weight=0.25,
        consistency_weight=0.1,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=20,
    )
    assert torch.all((router.left.assignments > 0).sum(dim=1) == 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    topic = topic_block_loss(
        model, batch, batch, temperature=0.5, top_k=2, local_decoder_weight=0.25
    )
    topic.total.backward()
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()
    before = model.topic_prototypes.detach().clone()
    recycle_dead_prototypes(
        model,
        optimizer,
        topic_indices=torch.tensor([0]),
        replacements=torch.randn(1, model.projection_dimensions),
    )
    assert not torch.equal(before[0], model.topic_prototypes.detach()[0])
    generator = torch.Generator().manual_seed(42)
    targets = balanced_sinkhorn_targets(
        torch.randn(40, 4, generator=generator), epsilon=0.2, iterations=100
    )
    assert torch.allclose(targets.sum(dim=1), torch.ones(40), atol=1e-5)
    assert torch.allclose(targets.sum(dim=0), torch.full((4,), 10.0), atol=1e-5)


def test_decoder_softly_balances_token_types_with_gradients() -> None:
    protocol = load_protocol()
    summed_protocol = copy.deepcopy(protocol)
    summed_protocol["model"]["normalize_token_type_evidence"] = False
    legacy_protocol = copy.deepcopy(protocol)
    legacy_protocol["model"]["token_type_balance"] = 0.0
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(10)
    features[:, -2:] = 0.0
    features[:2, -2] = 1.0
    features[2:, -1] = 1.0
    features = torch.nn.functional.normalize(features, dim=1)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    summed, _ = initialize_model(features, num_topics=4, protocol=summed_protocol)
    legacy, _ = initialize_model(features, num_topics=4, protocol=legacy_protocol)
    beta = model.topic_word_distribution()
    summed_beta = summed.topic_word_distribution()
    legacy_beta = legacy.topic_word_distribution()
    assert torch.allclose(beta.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(summed_beta.sum(dim=1), torch.ones(4), atol=1e-6)
    (-torch.log(beta[:, 0]).mean()).backward()
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()

    zero_logits = torch.zeros(10, model.projection_dimensions)
    neutral_beta = model.topic_word_distribution(zero_logits)
    legacy_summed_beta = summed.topic_word_distribution(zero_logits)
    assert torch.allclose(neutral_beta[:, :2].sum(dim=1), torch.full((4,), 0.5))
    assert torch.allclose(
        legacy_summed_beta[:, :2].sum(dim=1),
        torch.full((4,), 0.275),
    )

    tokens = legacy.projected_tokens()
    topics = torch.nn.functional.normalize(legacy.topic_prototypes, dim=1)
    expected_legacy = torch.softmax(
        2.0 * topics @ tokens.T / legacy.beta_temperature,
        dim=1,
    )
    assert torch.equal(legacy_beta, expected_legacy)


def test_positive_npmi_graph_supplies_topic_gradients() -> None:
    protocol = load_protocol()
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    graph, diagnostics = positive_npmi_graph(
        matrix,
        minimum_document_frequency=1,
        minimum_pair_frequency=1,
        maximum_neighbors=2,
        minimum_npmi=0.0,
    )
    assert graph.nnz > 0
    assert (graph != graph.T).nnz == 0
    assert np.allclose(graph.diagonal(), 0.0)
    assert diagnostics["words_with_neighbors"] >= 4

    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(6)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    result = cooccurrence_topic_constraint(model, torch_sparse_graph(graph))
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()


def test_checkpoint_selection_is_always_the_fixed_final_epoch() -> None:
    protocol = load_protocol()
    history = [
        {"epoch": 2, "validation": {"nll": 1.0}},
        {"epoch": 40, "validation": {"nll": 100.0}},
    ]
    selected = _selection(history, protocol)
    assert selected["selection_rule"] == "fixed_final_epoch"
    assert selected["epoch"] == 40
    assert selected["validation"] == {"nll": 100.0}


def test_nearest_neighbor_constraint_penalizes_close_topics() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(20)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    with torch.no_grad():
        model.topic_prototypes[1] = model.topic_prototypes[0] + 0.01
    result = nearest_neighbor_topic_constraint(model, neighbors=2, margin=0.3)
    result.loss.backward()
    assert float(result.loss) > 0
    assert result.diagnostics["nearest_topic_cosine_maximum"] > 0.9
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()


def test_hierarchical_router_adds_one_shared_document_score() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(20)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    matrix = sp.csr_matrix(np.eye(2, 20, dtype=np.float32) + 1.0)
    batch = sparse_batch(matrix, np.arange(2, dtype=np.int64))
    model.document_topic_prior_weight = 0.0
    local = model.route(batch, temperature=0.5, top_k=2, straight_through=False)
    model.document_topic_prior_weight = 1.0
    hierarchical = model.route(batch, temperature=0.5, top_k=2, straight_through=False)
    added = hierarchical.logits - local.logits
    for row in range(batch.documents):
        document_rows = added[batch.row_ids == row]
        assert torch.allclose(document_rows, document_rows[:1].expand_as(document_rows))
    assert torch.isfinite(hierarchical.theta).all()


def test_document_gate_normalization_support_and_fixed_evidence() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(8)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    assignments = torch.tensor(
        [[0.6, 0.4, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
        requires_grad=True,
    )
    row_ids = torch.tensor([0, 0])
    weights = torch.tensor([1.0, 2.0])
    logits = torch.tensor(
        [[1.0, -1.0, 4.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        requires_grad=True,
    )
    gated = model.aggregate_theta(
        assignments=assignments,
        row_ids=row_ids,
        weights=weights,
        documents=2,
        document_logits=logits,
        temperature=0.5,
        document_mixture_weight=0.5,
    )
    assert torch.allclose(gated.sum(dim=1), torch.ones(2))
    assert torch.equal(gated[0, 2:], torch.zeros(2))
    assert torch.allclose(gated[1], torch.full((4,), 0.25))
    gated[0, 0].backward()
    assert assignments.grad is not None and torch.isfinite(assignments.grad).all()
    assert logits.grad is None


def test_zero_document_gate_weight_exactly_recovers_token_mixture() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(8)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    assignments = torch.softmax(torch.randn(5, 4), dim=1)
    row_ids = torch.tensor([0, 0, 1, 1, 1])
    weights = torch.rand(5)
    baseline = model.aggregate_theta(
        assignments=assignments,
        row_ids=row_ids,
        weights=weights,
        documents=2,
    )
    recovered = model.aggregate_theta(
        assignments=assignments,
        row_ids=row_ids,
        weights=weights,
        documents=2,
        document_logits=torch.randn(2, 4),
        temperature=0.1,
        document_mixture_weight=0.0,
    )
    assert torch.equal(recovered, baseline)


def test_sos_bands_include_fixed_boundaries_and_cover_all_values() -> None:
    bands = _sos_bands([0.0, 0.5999, 0.6, 0.7, 0.8, 0.8001, 1.0])
    assert bands == {
        "high_gt_0_8": 2,
        "intermediate_0_6_to_0_8": 3,
        "low_lt_0_6": 2,
    }
    assert sum(bands.values()) == 7


def test_validation_gate_uses_only_predeclared_validation_evidence(
    tmp_path: Path,
) -> None:
    def chemistry(useful: int, *, coverage: float, mean_sos: float) -> dict:
        return {
            "annotation_coverage": coverage,
            "high_confidence_chemistry": {
                "mean_sos": mean_sos,
                "sos_bands": {
                    "high_gt_0_8": 0,
                    "intermediate_0_6_to_0_8": useful,
                    "low_lt_0_6": 1,
                },
            },
        }

    payloads = {
        "validation_chemical/current_neural/complete.json": chemistry(
            148, coverage=0.515, mean_sos=0.6418
        ),
        "validation_chemical/candidate_neural/complete.json": chemistry(
            148, coverage=0.607, mean_sos=0.6318
        ),
        "validation_chemical/tomotopy/complete.json": chemistry(
            138, coverage=0.607, mean_sos=0.6761
        ),
        "validation_evaluation/candidate_neural/complete.json": {
            "stable": True,
            "metrics": {"validation_document_completion": {"nll_per_token": 8.5}},
        },
        "model/complete.json": {"stable": True, "elapsed_seconds": 9000.0},
        "model/selected.json": {"checkpoint_sha256": "fixed-checkpoint"},
    }
    for name, payload in payloads.items():
        write_json(tmp_path / name, payload)
    result = evaluate_validation_gate(tmp_path)
    assert result["decision"] == "accepted"
    assert result["test_evaluation_authorized"] is True
    assert result["reported_context"]["training_within_prior_10_percent"] is False
    assert result["paper_outcome"]["candidate_change_from_current"] == 0


def test_committed_bundle_matches_supported_architecture() -> None:
    bundle = Path(__file__).parents[1] / "results/seed42/model_bundle"
    model, vocabulary, manifest = load_bundle(bundle)
    assert manifest["schema_version"] == "neural-ms2lda/model-bundle-v1"
    assert model.num_topics == 1000
    assert model.document_topic_prior_weight == 1.0
    assert model.document_mixture_weight == 0.75
    assert model.token_type_balance == 0.25
    assert model.normalize_token_type_evidence is False
    assert len(vocabulary) == model.vocabulary_size


def test_protocol_exposes_one_active_topic_architecture() -> None:
    protocol = load_protocol()
    assert protocol["cpu_threads"] == 6
    assert protocol["model"]["num_topics"] == 1000
    assert protocol["model"]["document_mixture_weight"] == 0.75
    assert protocol["model"]["beta_temperature"] == 0.18
    assert protocol["model"]["token_type_balance"] == 0.25
    assert protocol["model"]["normalize_token_type_evidence"] is True
    assert "development_gates" not in protocol
    assert "workers" not in protocol["tomotopy"]
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = _token_features(20)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    config = copy.deepcopy(protocol["topic_separation"])
    config["neighbors"] = 2
    result, weighted = _weighted_topic_separation(model, config)
    assert torch.allclose(weighted, float(config["weight"]) * result.loss)
    assert "erntm_weight" not in protocol["optimization"]
    assert "enabled" not in protocol["cooccurrence_regularization"]
    assert set(protocol["topic_separation"]) == {
        "neighbors",
        "margin",
        "weight",
    }
    assert "protocol_name" not in protocol
    assert "evidence_scope" not in protocol
    assert "training_exclusions" not in protocol
    assert "published_motifset" not in protocol["input_files"]


def test_split_records_are_loaded_without_opening_the_other_split(
    tmp_path: Path,
) -> None:
    validation = {"split": "validation", "spectrum_id": "validation-1"}
    (tmp_path / "validation_records.jsonl").write_text(
        json.dumps(validation) + "\n", encoding="utf-8"
    )
    (tmp_path / "test_records.jsonl").write_text("not-json\n", encoding="utf-8")
    assert load_heldout_records(tmp_path, "validation") == [validation]


def test_development_inputs_exclude_the_test_partition(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    for name in DEVELOPMENT_DATA_FILES:
        path = source / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
    (source / "data/test_full.npz").write_bytes(b"sealed")
    for name in READ_ONLY_STAGES:
        (source / name).mkdir(parents=True)
    (source / "mag").mkdir(parents=True)

    _link_read_only_inputs(source, output)

    assert {path.name for path in (output / "data").iterdir()} == set(
        DEVELOPMENT_DATA_FILES
    )
    assert not (output / "data/test_full.npz").exists()
    assert (output / "mag").is_symlink()

    (source / "data/validation_records.jsonl").unlink()
    validation = {"split": "validation", "spectrum_id": "validation-1"}
    test = {"split": "test", "spectrum_id": "test-1"}
    write_jsonl(
        source / "data/heldout_records.jsonl",
        (validation, test),
    )
    legacy_output = tmp_path / "legacy-output"
    _link_read_only_inputs(source, legacy_output)
    assert load_heldout_records(legacy_output / "data", "validation") == [validation]
    assert not (legacy_output / "data/test_records.jsonl").exists()


def test_development_verifies_frozen_source_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data_hashes = {}
    for name in DEVELOPMENT_DATA_FILES:
        path = source / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        data_hashes[name] = file_sha256(path)
    split_manifest = source / "data/split_manifest.jsonl"
    write_jsonl(
        split_manifest,
        (
            {"split": "train", "connectivity_key": "train"},
            {"split": "validation", "connectivity_key": "validation"},
            {"split": "test", "connectivity_key": "test"},
        ),
    )
    data_hashes[split_manifest.name] = file_sha256(split_manifest)
    write_json(source / "data/complete.json", {"output_sha256": data_hashes})

    view = source / "training_views/pair.npz"
    view.parent.mkdir(parents=True)
    view.write_bytes(b"paired training view")
    write_json(
        view.parent / "complete.json",
        {"output_sha256": {view.name: file_sha256(view)}},
    )

    features = source / "token_features/features.npy"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"token features")
    write_json(
        features.parent / "complete.json",
        {"features_sha256": file_sha256(features)},
    )

    mag = source / "mag/index"
    mag.mkdir(parents=True)
    mag_hashes = {}
    excluded = mag / "excluded_connectivity_keys.json"
    write_json(excluded, {"connectivity_keys": ["test", "validation"]})
    mag_hashes[excluded.name] = file_sha256(excluded)
    for name in ("kept_original_ids.npy", "spec2vec_filtered.faiss"):
        path = mag / name
        path.write_bytes(name.encode("utf-8"))
        mag_hashes[name] = file_sha256(path)
    write_json(
        mag / "manifest.json",
        {"output_sha256": mag_hashes},
    )

    _verify_source_artifacts(source, source / "data/validation_records.jsonl")
    features.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source artifact changed"):
        _verify_source_artifacts(source, source / "data/validation_records.jsonl")


def test_development_accepts_removed_legacy_protocol_metadata() -> None:
    current = load_protocol()
    source = copy.deepcopy(current)
    source["input_files"]["published_motifset"] = {"unused": True}
    source["token_features"].update({"type_dimensions": 2, "output_dimensions": 64})
    source["model"].update({"input_dimensions": 64})
    source["model"].pop("document_topic_prior_weight")
    source["model"].pop("document_mixture_weight")
    source["model"].pop("normalize_token_type_evidence")
    source["views"]["fragment_loss_group_atomic"] = True
    source["evaluation"].update(
        {
            "membership_threshold": 0.5,
            "mag_fingerprint_threshold": 0.8,
            "motif_spectrum_top_n": 20,
        }
    )
    source["development_gates"] = {"legacy": True}
    _validate_frozen_protocol(source, current)
    source["seed"] = 43
    with pytest.raises(ValueError, match="seed"):
        _validate_frozen_protocol(source, current)


def test_development_uses_the_single_supported_topic_capacity() -> None:
    source = copy.deepcopy(load_protocol())
    source["model"]["num_topics"] = 500
    protocol = _development_protocol(source)
    assert protocol["model"]["num_topics"] == 1000
    assert protocol["cpu_threads"] == 6
    _validate_frozen_protocol(source, protocol)


def test_development_run_has_portable_bundle_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = tmp_path / "candidate"
    write_json(
        source / "run.lock.json",
        {
            "inputs": {"mgf": {"bytes": 3, "sha256": "input"}},
            "environment": {"python": "3.11", "machine": "arm64"},
            "discovery_audit": {"forbidden_dependencies_found": []},
        },
    )
    write_json(
        run / "development.lock.json",
        {
            "source_run": str(source),
            "protocol_sha256": "protocol",
            "code_sha256": "code",
            "hypothesis": "soft balance",
            "source_artifact_sha256": {"data/train.npz": "train"},
        },
    )
    result = _portable_provenance(run, {"checkpoint_sha256": "checkpoint"})
    assert result["protocol_sha256"] == "protocol"
    assert result["run_source_sha256"] == {"development_code_manifest": "code"}
    assert result["inputs"]["mgf"]["sha256"] == "input"
    assert result["discovery_audit"]["development_hypothesis"] == "soft balance"
    assert str(source) not in json.dumps(result)


def test_verify_run_checks_cooccurrence_graph(tmp_path: Path) -> None:
    mgf = tmp_path / "input.mgf"
    mgf.write_text("BEGIN IONS\nEND IONS\n", encoding="utf-8")
    protocol = copy.deepcopy(load_protocol())
    protocol["input_files"] = {
        "mgf": {
            "relative_path": mgf.name,
            "bytes": mgf.stat().st_size,
            "sha256": file_sha256(mgf),
        }
    }
    run = tmp_path / "run"
    write_json(run / "protocol.resolved.json", protocol)
    inputs = neural_config.verify_inputs(protocol, tmp_path, names={"mgf"})
    write_json(
        run / "run.lock.json",
        {
            "data_root": str(tmp_path),
            "protocol_sha256": neural_config.object_sha256(protocol),
            "inputs": inputs,
            "code": neural_config.code_manifest(),
        },
    )
    data_artifact = run / "data/train.npz"
    data_artifact.parent.mkdir(parents=True)
    data_artifact.write_bytes(b"frozen training data")
    data_manifest = {
        "leakage_audit": {"leaked_compounds": 0, "leaked_groups": 0},
        "vocabulary": {
            "source_split": "train",
            "order": "raw_training_spectra_first_seen",
        },
        "output_sha256": {data_artifact.name: file_sha256(data_artifact)},
    }
    write_json(run / "data/complete.json", data_manifest)
    graph = run / "cooccurrence_graph/positive_npmi_graph.npz"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"frozen train-only graph")
    graph_manifest = {
        "output_sha256": {graph.name: file_sha256(graph)},
    }
    write_json(
        graph.parent / "complete.json",
        graph_manifest,
    )
    selected_checkpoint = run / "model/selected.pt"
    selected_checkpoint.parent.mkdir(parents=True)
    selected_checkpoint.write_bytes(b"selected model")
    selected_manifest = {
        "checkpoint": selected_checkpoint.name,
        "checkpoint_sha256": file_sha256(selected_checkpoint),
        "selection_rule": "fixed_final_epoch",
        "epoch": int(protocol["optimization"]["maximum_epochs"]),
    }
    write_json(
        run / "model/complete.json",
        {
            "selected": selected_manifest,
            "cooccurrence_graph": graph_manifest,
        },
    )
    write_json(run / "model/selected.json", selected_manifest)
    result = neural_config.verify_run(run, data_root=tmp_path)
    assert "cooccurrence_graph/complete.json" in result["manifests_present"]
    unhashed_data_manifest = data_manifest.copy()
    unhashed_data_manifest.pop("output_sha256")
    write_json(run / "data/complete.json", unhashed_data_manifest)
    with pytest.raises(ValueError, match="no output hashes"):
        neural_config.verify_run(run, data_root=tmp_path)
    write_json(run / "data/complete.json", data_manifest)
    write_json(graph.parent / "complete.json", {"output_sha256": {}})
    with pytest.raises(ValueError, match="no output hashes"):
        neural_config.verify_run(run, data_root=tmp_path)
    write_json(graph.parent / "complete.json", graph_manifest)
    graph.write_bytes(b"tampered graph")
    with pytest.raises(ValueError, match="artifact changed"):
        neural_config.verify_run(run, data_root=tmp_path)
    graph.write_bytes(b"frozen train-only graph")
    (graph.parent / "complete.json").unlink()
    with pytest.raises(FileNotFoundError, match="requires a co-occurrence graph"):
        neural_config.verify_run(run, data_root=tmp_path)


def test_neural_evaluation_rejects_nonfinal_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    protocol = load_protocol()
    selected = {
        "selection_rule": "fixed_final_epoch",
        "epoch": int(protocol["optimization"]["maximum_epochs"]) - 2,
    }
    write_json(run / "model/selected.json", selected)
    write_json(run / "model/complete.json", {"selected": selected})
    with pytest.raises(ValueError, match="fixed final epoch"):
        evaluate_neural(run, protocol)
    assert not (run / "evaluation/neural/test_access.json").exists()


def test_tomotopy_empty_document_uses_topic_prior() -> None:
    calls = []

    class FakeModel:
        k = 2
        alpha = np.asarray([0.3, 0.7], dtype=np.float32)

        @staticmethod
        def make_doc(words: list[str]) -> list[str]:
            assert words
            return words

        @staticmethod
        def infer(
            documents: list[list[str]], **kwargs: object
        ) -> tuple[list[list[float]], None]:
            calls.append(kwargs)
            return [[0.8, 0.2] for _ in documents], None

    theta = _infer_theta(FakeModel(), [[], ["frag@100.0"], []], iterations=5, workers=6)
    assert np.allclose(theta[0], [0.3, 0.7])
    assert np.allclose(theta[1], [0.8, 0.2])
    assert np.allclose(theta[2], [0.3, 0.7])
    assert calls == [{"iter": 5, "workers": 6, "parallel": 1, "together": False}]


def test_tomotopy_alpha_evidence_allows_a_learned_vector() -> None:
    class FakeModel:
        k = 2
        alpha = np.asarray([0.1, 0.2])
        optim_interval = 10

    evidence = _alpha_evidence(FakeModel(), {"alpha": 0.6})
    assert evidence["initial_value"] == 0.6
    assert evidence["optimization_interval"] == 10
    assert evidence["learned_minimum"] == 0.1

    FakeModel.alpha = np.asarray([0.1, 0.0])
    with pytest.raises(ValueError, match="not positive"):
        _alpha_evidence(FakeModel(), {"alpha": 0.6})


def test_tomotopy_reference_requires_matching_protocol_and_hashes(
    tmp_path: Path,
) -> None:
    protocol = load_protocol()
    reference = tmp_path / "reference"
    source_protocol = copy.deepcopy(protocol)
    source_protocol["tomotopy"]["workers"] = 6
    source_protocol["training_cpu_threads"] = 4
    write_json(reference / "protocol.resolved.json", source_protocol)

    output_sha256 = {}
    for name in REFERENCE_DATA_FILES:
        path = reference / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        output_sha256[name] = file_sha256(path)
    write_json(reference / "data/complete.json", {"output_sha256": output_sha256})

    model = reference / "evaluation/tomotopy/model.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"frozen tomotopy model")
    write_json(
        model.parent / "complete.json",
        {
            "model_sha256": file_sha256(model),
            "topic_count": 1000,
            "training_workers": 6,
            "training_parallel": 3,
            "training_iterations": 750,
            "training_seconds_total": 4561.0,
            "peak_rss_bytes": 1024,
            "converged": True,
        },
    )

    evidence = tomotopy_reference_evidence(reference, protocol)
    assert evidence["model_sha256"] == file_sha256(model)
    assert evidence["training_workers"] == 6

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="reference model changed"):
        tomotopy_reference_evidence(reference, protocol)


def _mini_protocol(mgf: Path) -> dict[str, object]:
    protocol = copy.deepcopy(load_protocol())
    protocol["input_files"] = {
        "mgf": {
            "relative_path": mgf.name,
            "bytes": mgf.stat().st_size,
            "sha256": file_sha256(mgf),
        }
    }
    protocol["preprocessing"].update(
        {
            "expected_spectra": 18,
            "min_fragments": 3,
            "min_df": 1,
            "split_fractions": [0.6, 0.2, 0.2],
        }
    )
    protocol["sgns"].update(
        {
            "dimensions": 4,
            "epochs": 1,
            "positive_pairs_per_document": 2,
            "batch_size": 32,
        }
    )
    protocol["token_features"]["fourier_frequencies"] = [1]
    protocol["model"].update(
        {
            "num_topics": 4,
            "projection_dimensions": 8,
            "router_hidden_dimensions": 8,
            "sinkhorn_iterations": 10,
        }
    )
    protocol["views"]["pairs"] = 2
    protocol["optimization"].update(
        {
            "batch_size": 4,
            "topic_update_batch_size": 4,
            "topic_updates_per_epoch": 1,
            "maximum_epochs": 2,
            "validation_interval": 1,
        }
    )
    protocol["anti_collapse"].update(
        {
            "routing_temperature_anneal_epochs": 2,
            "sinkhorn_weight_hold_epochs": 0,
            "sinkhorn_weight_end_epoch": 2,
            "recycle_patience_validations": 10,
            "recycle_through_epoch": 2,
        }
    )
    protocol["cooccurrence_regularization"].update(
        {
            "minimum_document_frequency": 1,
            "minimum_pair_frequency": 1,
            "maximum_neighbors": 2,
        }
    )
    protocol["topic_separation"]["neighbors"] = 2
    protocol["evaluation"].update({"latency_subset_size": 2, "latency_repeats": 1})
    return protocol


def _write_mini_mgf(path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    smiles_values = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CCF",
        "COC",
        "CNC",
        "CCS",
        "CC=O",
        "CC#N",
        "C=CO",
        "C1CC1",
        "C1CCC1",
        "c1ccccc1",
        "c1ccncc1",
        "O=C=O",
        "N#N",
    ]
    blocks = []
    for index, smiles in enumerate(smiles_values):
        inchikey = MolToInchiKey(Chem.MolFromSmiles(smiles))
        peaks = [
            f"{50 + offset * 10 + index % 3}.0 {100 - offset * 10}.0"
            for offset in range(5)
        ]
        blocks.append(
            "\n".join(
                [
                    "BEGIN IONS",
                    f"USI=mini:{index}",
                    f"FEATURE_ID=feature:{index}",
                    f"SMILES={smiles}",
                    f"INCHIKEY={inchikey}",
                    "PRECURSOR_MZ=250.0",
                    "NUM_PEAKS=5",
                    *peaks,
                    "END IONS",
                ]
            )
        )
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def test_miniature_mgf_through_report_and_bundle(tmp_path: Path) -> None:
    mgf = tmp_path / "mini.mgf"
    _write_mini_mgf(mgf)
    protocol = _mini_protocol(mgf)
    run = tmp_path / "run"
    prepare_data(run, data_root=tmp_path, protocol=protocol)
    data = run / "data"
    complete = json.loads((data / "complete.json").read_text())
    assert complete["leakage_audit"] == {
        "connectivity_groups": 18,
        "leaked_compounds": 0,
        "leaked_groups": 0,
        "split_groups": 18,
    }
    prepare_training_views(run, counts_dir=data, data_root=tmp_path, protocol=protocol)
    train = load_csr(data / "train.npz")
    embeddings = train_sgns(run / "embeddings", train, protocol["sgns"], seed=42)
    assert set(embeddings["output_sha256"]) == {"embeddings.npy"}
    assert "embeddings_sha256" not in embeddings
    features = prepare_token_features(run, counts_dir=data, protocol=protocol)
    assert set(features["output_sha256"]) == {"features.npy"}
    assert "features_sha256" not in features
    initialization = prepare_initialization(run, train=train, protocol=protocol)
    assert set(initialization["output_sha256"]) == {"model_initialization.pt"}
    assert "checkpoint_sha256" not in initialization
    result = train_model(
        run,
        train=train,
        views=load_view_pairs(run, protocol),
        validation_observed=load_csr(data / "validation_observed.npz"),
        validation_completion=load_csr(data / "validation_completion.npz"),
        validation_full=load_csr(data / "validation_full.npz"),
        validation_records=load_heldout_records(data, "validation"),
        protocol=protocol,
    )
    selected_hash = result["selected"]["checkpoint_sha256"]
    assert "graph_sha256" not in result["cooccurrence_graph"]
    resumed = train_model(
        run,
        train=train,
        views=load_view_pairs(run, protocol),
        validation_observed=load_csr(data / "validation_observed.npz"),
        validation_completion=load_csr(data / "validation_completion.npz"),
        validation_full=load_csr(data / "validation_full.npz"),
        validation_records=load_heldout_records(data, "validation"),
        protocol=protocol,
    )
    assert resumed["selected"]["checkpoint_sha256"] == selected_hash
    write_json(run / "protocol.resolved.json", protocol)
    neural = evaluate_neural(run, protocol)
    assert "test_access.json" in neural["output_sha256"]
    comparator = copy.deepcopy(neural)
    comparator.update(
        {
            "method": "tomotopy",
            "training_reused": True,
            "training_seconds_total": 10.0,
            "training_workers": 6,
            "inference_iterations": 100,
        }
    )
    write_json(run / "evaluation/tomotopy/complete.json", comparator)
    chemistry = {
        "topics": 4,
        "annotation_coverage": 0.5,
        "dominant_topic_chemistry": {"eligible_topics": 2, "mean_sos": 0.6},
        "high_confidence_chemistry": {
            "eligible_topics": 1,
            "associated_spectra": 2,
            "mean_sos": 0.7,
        },
    }
    write_json(run / "chemical/neural/complete.json", chemistry)
    write_json(run / "chemical/tomotopy/complete.json", chemistry)
    write_json(
        run / "run.lock.json",
        {
            "schema_version": "neural-ms2lda/run-lock-v1",
            "protocol_sha256": "miniature-test",
            "inputs": {"mgf": {"sha256": file_sha256(mgf)}},
            "code": {},
        },
    )
    report = build_machine_report(run)
    assert len(report["methods"]) == 2
    assert "pipeline_peak_rss_bytes" in report["methods"][0]
    assert report["source_sha256"]["neural_training"] == file_sha256(
        run / "model/complete.json"
    )
    assert neural["method"] == "neural_cooccurrence_margin_hierarchical"
    assert report["title"] == "Neural MS2LDA on MSnLib"
    assert "ERNTM" not in report["headline"]
    bundle = tmp_path / "bundle"
    packaged = package_bundle(run, bundle)
    assert "bundle_version" not in packaged
    loaded, vocabulary, manifest = load_bundle(bundle)
    assert loaded.num_topics == 4
    assert loaded.document_topic_prior_weight == 1.0
    assert loaded.document_mixture_weight == 0.75
    assert loaded.normalize_token_type_evidence is True
    assert len(vocabulary) == train.shape[1]
    assert manifest["selected_epoch"] == 2
    provenance_text = (bundle / "provenance.json").read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert provenance["inputs"]["mgf"]["sha256"] == file_sha256(mgf)
    assert "path" not in provenance_text
    legacy_protocol = json.loads((bundle / "protocol.json").read_text())
    legacy_protocol["model"].pop("document_topic_prior_weight")
    legacy_protocol["model"].pop("document_mixture_weight")
    legacy_protocol["model"].pop("normalize_token_type_evidence")
    legacy_protocol["hierarchical_routing"] = {"weight": 1.0}
    write_json(bundle / "protocol.json", legacy_protocol)
    manifest["files"]["protocol.json"] = {
        "bytes": (bundle / "protocol.json").stat().st_size,
        "sha256": file_sha256(bundle / "protocol.json"),
    }
    write_json(bundle / "manifest.json", manifest)
    legacy_loaded, _, _ = load_bundle(bundle)
    assert legacy_loaded.document_topic_prior_weight == 1.0
    assert legacy_loaded.document_mixture_weight == 0.0
    assert legacy_loaded.normalize_token_type_evidence is False


def test_zip_safety_rejects_parent_traversal(tmp_path: Path) -> None:
    import zipfile

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_zip_members(archive)


def test_acquisition_manifest_is_validated_without_rewrite(tmp_path: Path) -> None:
    archives = {"Data.zip": {"bytes": 1, "md5": "a", "sha256": "b"}}
    extracted = {"input.mgf": {"bytes": 2, "sha256": "c"}}
    manifest = {
        "schema_version": "msnlib-validation-acquisition/v1",
        "zenodo_record": RECORD_ID,
        "zenodo_api": RECORD_API,
        "archives": archives,
        "extracted_inputs": extracted,
    }
    path = tmp_path / "acquisition_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    assert validate_acquisition_manifest(tmp_path, archives, extracted) == path
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="archive evidence differs"):
        validate_acquisition_manifest(tmp_path, {}, extracted)
