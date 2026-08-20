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
    _bundle_version,
    load_bundle,
    package_bundle,
)
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
    sparse_batch,
)
from benchmarks.neural_assignment_ms2lda.embeddings import train_sgns
from benchmarks.neural_assignment_ms2lda.evaluation import evaluate_neural
from benchmarks.neural_assignment_ms2lda.model import (
    balanced_sinkhorn_targets,
    recycle_dead_prototypes,
    router_block_loss,
    topic_block_loss,
)
from benchmarks.neural_assignment_ms2lda.regularizers import (
    cooccurrence_topic_constraint,
    erntm_topic_constraint,
    nearest_neighbor_topic_constraint,
)
from benchmarks.neural_assignment_ms2lda.report import build_machine_report
from benchmarks.neural_assignment_ms2lda.tomotopy import _infer_theta
from benchmarks.neural_assignment_ms2lda.training import (
    _weighted_topic_separation,
    train_model,
    validation_gate_summary,
)
from benchmarks.neural_assignment_ms2lda.utils import file_sha256, write_json
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


def test_sinkhorn_top2_erntm_gradients_and_recycling() -> None:
    protocol = load_protocol()
    features = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
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
    regularizer = erntm_topic_constraint(model)
    (topic.total + regularizer.loss).backward()
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
    targets = balanced_sinkhorn_targets(torch.randn(40, 4), epsilon=0.2, iterations=100)
    assert torch.allclose(targets.sum(dim=1), torch.ones(40), atol=1e-5)
    assert torch.allclose(targets.sum(dim=0), torch.full((4,), 10.0), atol=1e-5)


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

    features = torch.nn.functional.normalize(torch.randn(6, 64), dim=1)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    result = cooccurrence_topic_constraint(model, torch_sparse_graph(graph))
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.topic_prototypes.grad is not None
    assert torch.isfinite(model.topic_prototypes.grad).all()


def test_validation_gate_summary_uses_predeclared_thresholds() -> None:
    protocol = load_protocol()
    targets = protocol["development_gates"]
    validation = {
        "word_cooccurrence_npmi": {
            "mean_npmi": targets["minimum_mean_npmi"],
            "undefined_pair_fraction": targets["maximum_undefined_pair_fraction"],
        },
        "top_word_diversity": targets["minimum_top_word_diversity"],
        "mixture_diagnostics": {
            "effective_topic_count_median": targets["maximum_effective_topics_median"]
        },
        "document_completion": {"nll_per_token": targets["maximum_validation_nll"]},
    }
    summary = validation_gate_summary(validation, protocol)
    assert summary["all_gates_met"]
    assert summary["gates_met"] == 5
    validation["top_word_diversity"] = (
        float(targets["minimum_top_word_diversity"]) - 1e-6
    )
    summary = validation_gate_summary(validation, protocol)
    assert not summary["all_gates_met"]
    assert summary["gates_met"] == 4


def test_nearest_neighbor_constraint_penalizes_close_topics() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
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

    features = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
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


def test_committed_v1_bundle_keeps_zero_document_prior() -> None:
    bundle = Path(__file__).parents[1] / "results/seed42/model_bundle"
    model, vocabulary, manifest = load_bundle(bundle)
    assert manifest["bundle_version"] == "neural-ms2lda-msnlib-k500-v1"
    assert model.document_topic_prior_weight == 0.0
    assert len(vocabulary) == model.vocabulary_size


def test_bundle_version_follows_packaged_protocol() -> None:
    protocol = load_protocol()
    assert _bundle_version(protocol) == "neural-ms2lda-msnlib-k500-v2"
    legacy = copy.deepcopy(protocol)
    del legacy["hierarchical_routing"]
    assert _bundle_version(legacy) == "neural-ms2lda-msnlib-k500-v1"


def test_topic_separation_honors_update_placement_flags() -> None:
    protocol = load_protocol()
    from benchmarks.neural_assignment_ms2lda.model import initialize_model

    features = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
    model, _ = initialize_model(features, num_topics=4, protocol=protocol)
    config = copy.deepcopy(protocol["topic_separation"])
    config["neighbors"] = 2
    reference = model.topic_prototypes.sum()
    for placement_key in (
        "apply_during_router_updates",
        "apply_during_topic_updates",
    ):
        config[placement_key] = False
        result, weighted = _weighted_topic_separation(
            model,
            config,
            placement_key=placement_key,
            reference=reference,
        )
        assert result is None
        assert float(weighted) == 0.0


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
    write_json(
        run / "data/complete.json",
        {
            "leakage_audit": {"leaked_compounds": 0, "leaked_groups": 0},
            "vocabulary": {
                "source_split": "train",
                "order": "raw_training_spectra_first_seen",
            },
        },
    )
    graph = run / "cooccurrence_graph/positive_npmi_graph.npz"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"frozen train-only graph")
    write_json(
        graph.parent / "complete.json",
        {"graph_sha256": file_sha256(graph)},
    )
    result = neural_config.verify_run(run, data_root=tmp_path)
    assert "cooccurrence_graph/complete.json" in result["manifests_present"]
    graph.write_bytes(b"tampered graph")
    with pytest.raises(ValueError, match="co-occurrence graph changed"):
        neural_config.verify_run(run, data_root=tmp_path)


def test_tomotopy_empty_document_uses_topic_prior() -> None:
    class FakeModel:
        k = 2
        alpha = np.asarray([0.3, 0.7], dtype=np.float32)

        @staticmethod
        def make_doc(words: list[str]) -> list[str]:
            assert words
            return words

        @staticmethod
        def infer(
            documents: list[list[str]], **_: object
        ) -> tuple[list[list[float]], None]:
            return [[0.8, 0.2] for _ in documents], None

    theta = _infer_theta(FakeModel(), [[], ["frag@100.0"], []], iterations=5)
    assert np.allclose(theta[0], [0.3, 0.7])
    assert np.allclose(theta[1], [0.8, 0.2])
    assert np.allclose(theta[2], [0.3, 0.7])


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
    protocol["token_features"].update(
        {"fourier_frequencies": [1], "output_dimensions": 8}
    )
    protocol["model"].update(
        {
            "num_topics": 4,
            "input_dimensions": 8,
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
    train_sgns(run / "embeddings", train, protocol["sgns"], seed=42)
    prepare_token_features(run, counts_dir=data, protocol=protocol)
    prepare_initialization(run, train=train, protocol=protocol)
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
    comparator = copy.deepcopy(neural)
    comparator.update({"method": "tomotopy_k1000_comparator", "topic_count": 1000})
    write_json(run / "evaluation/tomotopy/complete.json", comparator)
    chemistry = {
        "annotation_coverage": 0.5,
        "dominant_topic_chemistry": {"eligible_topics": 2, "mean_sos": 0.6},
        "high_confidence_chemistry": {"eligible_topics": 1, "mean_sos": 0.7},
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
    assert neural["method"] == "neural_cooccurrence_margin_hierarchical_k500"
    assert "ERNTM" not in report["headline"]
    bundle = tmp_path / "bundle"
    packaged = package_bundle(run, bundle)
    assert packaged["bundle_version"] == "neural-ms2lda-msnlib-k500-v2"
    loaded, vocabulary, manifest = load_bundle(bundle)
    assert loaded.num_topics == 4
    assert len(vocabulary) == train.shape[1]
    assert manifest["selected_epoch"] in {1, 2}
    provenance_text = (bundle / "provenance.json").read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert provenance["inputs"]["mgf"]["sha256"] == file_sha256(mgf)
    assert "path" not in provenance_text


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
