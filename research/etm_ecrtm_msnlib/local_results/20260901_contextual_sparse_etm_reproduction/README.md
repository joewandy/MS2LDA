# Contextual Sparse ETM clean-room reproduction

This bundle was generated from reproduction `c8e26d27-861e-42ed-a04f-ca5a9ecfedae` at
source commit `379ab80ab6f0916e0c31a036bab229dda1d4727a`. Models were fitted on training
spectra, selected and ablated on validation spectra, frozen, and then evaluated
on the fixed test split. `validation_comparison.csv` records development-split
evidence; `comparison.csv` and `stability_by_seed.csv` contain final test results.

## Acceptance status

Predeclared directional claims passed: **True**. Inspect
`acceptance.json`, `data_quality.json`, `fresh_evidence_manifest.json`, and the
CSV/JSON result tables for the complete evidence trail.

`reproduction_manifest.json` is the immutable raw controller declaration.
`acceptance.json` is the authoritative audited interpretation of its scientific
gates. In particular, high-K planted-motif recovery is counted by one-to-one
truth matching at topic-word cosine at least 0.50; it is not the same quantity
as the number of fitted topics that become a document-level top-1 winner.
