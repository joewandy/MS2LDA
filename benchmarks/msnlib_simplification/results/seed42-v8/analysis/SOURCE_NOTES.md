# HybridLDA simplification report notes

## Reporting job

- Question: can HybridLDA be simplified, or made fully neural without local VB, while preserving the current model's measured benefits?
- Audience: product stakeholders, using the executive-report specification.
- Delivery mode: portable HTML because this runtime has no callable Data Analytics MCP report renderer or full Sites lifecycle.
- Baseline: `dreams_prior__dreams_semi` with two local VB steps.
- Selection: validation first; test is the held-out confirmation.

## Required-structure mapping

- Title: `A simpler HybridLDA that keeps the useful parts`.
- Executive summary: visible immediately after the title.
- Key findings with visual evidence: zero-VB fidelity, DreaMS discovery value, and the seven-gate result.
- Recommended next steps: canonical form and implementation boundaries.
- Further questions: only the focused analytic-initializer follow-up and end-to-end timing.
- Caveats and assumptions: one short final section.

No KPI strip is used because the report is a model-selection strategy memo and the metrics have different meanings and scales.

## Source inventory

- Frozen bundle: `report/metrics.csv`, `report/chemical_metrics.csv`, `report/bootstrap.jsonl`, `report/collection_summary.json`, and `verification.json`.
- Reproducible analysis: executed notebook plus derived gate, shortlist, bootstrap, activity, and footprint tables in this directory.
- Historical collapse context: preserved git commit `b12278a72ed18594bfede0b6c82f6cab212e48f9`, particularly `docs/model/amortized_lda_benchmark.md` and `docs/model/semi_amortized_lda_benchmark.md`.

## Chart map

| Segment | Analytical question | Family / type | Fields | Supported claim | Palette policy |
| --- | --- | --- | --- | --- | --- |
| Zero-VB fidelity | Which same-discovery inference forms preserve the weakest assignment tail? | Comparison / horizontal bar | five DreaMS-prior model forms, validation fifth-percentile cosine, VB steps, active topics | One VB correction preserves the tail; zero-VB neural forms do not | Single blue root plus neutral reference lines |
| DreaMS discovery value | Which simplifications preserve held-out token prediction? | Signed comparison / horizontal bar | model form, relative validation NLL, threshold SOS, parameters | Symmetric discovery breaches the 2% NLL allowance; the recommended form improves NLL | Blue-orange diverging around zero plus neutral limit lines |

The posterior chart uses the five forms with DreaMS-prior discovery so every cosine uses the same reference definition; the NLL chart uses all six forms, including symmetric discovery. Both are full-width report blocks with direct value labels and richer tooltip datasets. The repeated horizontal-bar family is intentional because both questions compare the same long-labeled architecture family, while the second chart uses a signed scale and different reference lines.

## Validation notes

- Bundle completeness: 226 required artifacts, zero missing, all report hashes match, and all 53 frozen source files match the frozen manifest.
- Recommendation gate: `dreams_prior__topic_direct` at one VB step passes all seven frozen checks on validation and test.
- Collapse check: recommended form uses 351 active topics on validation and 352 on test versus 354 for the current form; median active topics per spectrum remains 12.
- Cross-budget document contrasts use the same scaffold-group resampling method and 2,000 replicates as the bundle's bootstrap.
- Chemical recommendations use the precomputed topic-level point estimates and exact eligible-topic denominators. No custom chemical interval is shown in the executive report because coverage is sparse and the frozen decision rule is denominator-based.
- The zero-parameter analytic form fails only validation high-confidence coverage: 33 eligible topics versus 37, or 89.2% against a 90% floor.

## Omitted or bounded material

- Full 80-row factorial and 160-row chemical tables remain in the source bundle and notebook outputs rather than the reader-facing report.
- Warm latency is model-only for the simplification arms; the report does not claim a directly measured count-only end-to-end speedup.
- The study scope is stated once in the final caveat section, following the user's request not to repeat it throughout the analysis.
