# Contextual Sparse ETM research evidence

This directory contains the compact, committed evidence for the final
Contextual Sparse ETM study. The canonical package is:

`local_results/20260901_contextual_sparse_etm_reproduction/`

It was produced from clean-room reproduction
`c8e26d27-861e-42ed-a04f-ca5a9ecfedae`, whose raw execution source was clean
commit `379ab80ab6f0916e0c31a036bab229dda1d4727a`.

## Study design

- Public acquisition: MSnLib positive-mode spectra and the positive-mode
  Spec2Vec model.
- Split: deterministic connectivity/scaffold-group-disjoint 70/10/20
  train/validation/test partition.
- Counts: 27,222 train, 3,889 validation and 7,777 test spectra.
- Vocabulary: 21,233 fragment/loss words built on training spectra only.
- Development boundary: fit on train, select and ablate on validation, freeze
  every model and validation output, then expose test exactly once.
- Neural execution: CUDA for canonical ETM, balanced ETM and all three
  Contextual Sparse ETM fits and inference stages.
- Chemical evaluation: leakage-filtered Mass2Motif Annotation Guidance (MAG)
  and substructure overlap score (SOS), with zero clustering or optimization
  exceptions.

## Evidence package map

| File | Purpose |
| --- | --- |
| `README.md` | Short package identity and acceptance status |
| `comparison.csv` | Final held-out test comparison |
| `validation_comparison.csv` | Development-only comparison |
| `stability_by_seed.csv` | Three-seed final-model test results |
| `synthetic_summary.csv` | K=36 component study |
| `high_k_stress.csv` | K=128 overcompleteness study |
| `metrics.json` and `config.json` | Primary model test metrics and frozen configuration |
| `acceptance.json` | Predeclared directional scientific checks |
| `data_quality.json` | Split, probability, MAG and accounting checks |
| `checkpoint_manifest.json` | Compact final checkpoint identity |
| `fresh_evidence_manifest.json` | Hash inventory for the package and raw stage outputs |
| `reproduction_manifest.json` | Immutable raw controller metadata |
| `stage_records/` | Commands, times and output hashes for all 54 stages |
| `contextual/`, `controls/`, `tomotopy/` | Compact method-specific metrics, diagnostics and access audits |

`reproduction_manifest.json` preserves the raw controller declaration exactly.
The audited interpretation of high-K truth recovery and every final claim is in
`acceptance.json`; report generation reads the audited compact tables rather
than free-form manifest wording.

## Machine checks

Packaging a completed raw run re-verifies stage chronology and hashes,
validation/test view identity, release order, probability matrices, split
invariants, MAG failure counts, SOS accounting and scientific claims:

```bash
python -m scripts.package_contextual_sparse_etm_reproduction \
  --root /path/to/completed-run \
  --output /path/to/new-evidence-package
```

The packager is atomic and refuses an existing output directory. Use a new
location for every reproduction; never edit a sealed package in place.

Generate the manuscript fragments from one verified package with:

```bash
python -m scripts.generate_routing_etm_report \
  --evidence-root /path/to/evidence-package \
  --output docs/research/generated
```

The scientific report and reviewed PDF are in `docs/research/`.

## Interpretation

Contextual Sparse ETM has the broadest held-out evaluable and useful motif
inventory among the compared methods while preserving sparse per-spectrum
mixtures and broad global topic use. Dense ETM controls retain better
completion likelihood, and Tomotopy retains higher conditional SOS over a
smaller evaluable inventory. The paper reports these axes separately.

`HANDOFF.md` summarizes the final state. `NEXT_AGENT.md` lists the rules for any
future continuation.
