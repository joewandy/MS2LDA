# Main-paper and supplement clarity review - 18 September 2026

The user requested a shorter, less mathematical main paper, a separately
referenced supplementary document, and an independent editor agent. The
`editor_review` agent reviewed the allocation before restructuring, reviewed
both revised sources independently, and verified the resulting corrections.
Its final editorial outcome was ready for handoff after PDF QA, with no
outstanding blocking editorial issues. This is an editorial and preservation
review, not a new independent validation of the experimental results.

## Exposition references

- [ETM, Dieng et al. (2020)](https://aclanthology.org/2020.tacl-1.29.pdf):
  establish the probabilistic model before explaining inference and estimation.
- [Dynamic ETM, Dieng et al. (2019)](https://arxiv.org/pdf/1907.05545):
  distinguish inherited components from the extension and its inference method.
- [Original MS2LDA, van der Hooft et al. (2016)](https://www.pure.ed.ac.uk/ws/portalfiles/portal/178956661/130100.pdf):
  retain explanatory spectral examples in the main text and place detailed
  workflow, inference and supporting analyses in supplementary information.

The papers guide organization; their wording and artwork are not copied.

## Material retained for comprehension

The main paper retains the ETM decoder and generator, the Gaussian posterior,
the meaning of reconstruction-minus-KL training, three motivated enhancement
sections, and the assembled model. All three teaching figures and numerical
examples remain. The training row in Figure 3 now describes drawing Gaussian
scores in words; its reparameterization formula remains in the supplement.

The main paper also retains the primary performance results, actual motif
counterpart spectra, chemical-specificity findings and metric meanings. It
still states the conditional mean-SOS denominator, unmatched-topic limitation,
batch-size difference, dependence among fit summaries, lack of factorial
ablations of the final model, and reuse of the validation data for selection.

## Material moved and referenced

| Supplement | Content | Main-paper point of use |
| --- | --- | --- |
| S1 | Notation table, exact filtering/tokenization, split and embedding protocol, synthetic construction | Spectral input and synthetic-data paragraphs |
| S2 | Full base/final objectives, reparameterization, analytic KL, balanced emissions, attention and offset formulas, entmax threshold, numerical conventions | Base ETM, each enhancement, assembled model |
| S3 | LDA/ETM component table and full comparator training protocols | Comparators and experimental design |
| S4 | Full recovery, completion, entropy, MAG and SOS definitions and edge cases | Metric explanations |
| S5 | Matched background, permutation nulls, feature enrichment, multiplicity, repeatability and reference-matching rules | Chemical methods and background-sensitivity results |
| S6 | Directed recovery equations, cohort selection, aggregation and example selection | Recovery methods and results |
| S7 | Chemical-agreement/multiplicity/channel-sensitivity figure and within-fit redundancy details | Correspondence and repeatability results |
| S8 | Implementation/evidence paths, environment, random states and scope of component evidence | Design, discussion and availability statement |

Numbered PDF links connect these references to their actual destinations.
The main text preserves the findings supported by the moved diagnostics.

## Editor findings resolved

- Expanded MACCS and explained fingerprints as predefined molecular features.
- Expanded KL and explained departure from the prior.
- Replaced undefined “exact support” with the number of strictly positive weights.
- Explained sparse weights using nonnegativity, unit sum and exact zeros.
- Defined spectral/mixture cosine similarity in plain language.
- Named 25 and 100 Da sensitivity bins and linked the exact S5.1 protocol.
- Shortened the abstract while retaining the primary trade-off and evidence limits.

The editor verified that all previous source labels survive across the two
documents and that the moved technical equation blocks are unchanged. The
parent agent separately checks the complete equation inventory, generated
result inputs, compiled links, LaTeX logs and rendered pages before replacing
the canonical PDFs. The final page counts and PDF hashes are recorded in the
[source inventory](contextual_sparse_etm_report_sources.md).
