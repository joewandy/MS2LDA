# Contextual Sparse ETM clean-room reproduction

This bundle was generated from reproduction `f141cfa6-ce05-4cc6-bbad-424ae4afd52b` at
source commit `102d59c07e2e6bd0a3cfc17ccfe153cbb34afec2`. Models were fitted on training
spectra, selected and ablated on validation spectra, frozen, and then evaluated
on the fixed test split. `validation_comparison.csv` records development-split
evidence; `comparison.csv` and `stability_by_seed.csv` contain final test results.
Chemical scores are computed from the sealed full-spectrum topic mixtures and
MAG annotations using one dominant-topic association per spectrum.

## Evidence checks

Scientific integrity and directional checks passed: **True**. Inspect
`acceptance.json`, `data_quality.json`, `fresh_evidence_manifest.json`, and the
CSV/JSON result tables for the complete evidence trail.

`reproduction_manifest.json` records the frozen fit and inference provenance.
`acceptance.json` is the authoritative audited interpretation of its scientific
gates. In particular, high-K planted-motif recovery is counted by one-to-one
truth matching at topic-word cosine at least 0.50; it is not the same quantity
as the number of fitted topics that become a document-level top-1 winner.
