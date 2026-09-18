# Final editorial, scientific and implementation review — 18 September 2026

## Assessment: share with caveats

The user requested a complete editorial review, code-to-paper and results
checks, and an experienced ML researcher's assessment before publication of
the revised main paper and supplement. Four separate agent reviewers covered
editorial clarity, mathematical implementation, numerical evidence and ML
methodology. The parent agent integrated their findings and checked packaging,
documentation, PDF links and rendering. These are agent reviews, not external
peer review or independent laboratory validation.

No new executable model defect or invalid generative/variational construction
was found. The evidence supports an explicitly qualified development study:
broader chemically screened candidate inventories and compact mixtures under
the reported conditions. It does not establish general chemical superiority,
confirmed new substructures, independent generalization or necessity of every
enhancement. No models were retrained and no saved results were replaced.

## Scientific assessment

- Separate fragment/loss normalization gives a valid topic-word distribution.
  Gaussian scores mapped through entmax define a valid generator. The ELBO
  remains in Gaussian latent space, so an inverse entmax map or Jacobian is
  unnecessary. Shared topic parameters in the encoder and decoder do not
  invalidate the bound. The implementation uses the corrected posterior mean
  in the Gaussian KL.
- The executable objective floors probabilities and adds optimizer weight
  decay. Both qualifications are already distinguished from the ideal ELBO.
  New text explains that the pseudo-count scale changes reconstruction
  relative to KL regularization, even if normalized encoder inputs are fixed.
- Hard top-two routing can switch at rank ties. The supplement now states
  this explicitly; optimization remains nonconvex and topic identifiability
  is not established. Sparse weights are not chemical certainty.
- Completion uses a plug-in inferred mixture, not integration over posterior
  uncertainty. This is now explicit beside the main metric definition and
  supplementary equation. The original ETM also uses a plug-in completion
  approximation.

Primary scientific comparators were the [original ETM paper, including its
completion procedure](https://aclanthology.org/2020.tacl-1.29.pdf) and the
[entmax formulation](https://aclanthology.org/P19-1146.pdf). The editor also
compared the exposition with [Dynamic ETM](https://arxiv.org/pdf/1907.05545)
and [original MS2LDA](https://www.pure.ed.ac.uk/ws/portalfiles/portal/178956661/130100.pdf).
These comparisons guide interpretation and organization, not borrowed prose.

## Findings corrected

1. The first data-split description now discloses earlier use of the reserved
   test data. Those data are absent from the reported comparisons, but are
   not an untouched confirmation set.
2. The synthetic comparison now points readers to the historical K=36
   gradient qualification. The uniform-evidence branch is provably unreachable
   throughout real K=1,000 and synthetic K=128 training: the independently
   checked maximum numbers of observed word types are 494 and 51/50/52,
   respectively, and each word contributes to at most two topics. This proof
   does not cover the historical K=36 trajectories. Reloading their final
   checkpoints is a different check from replaying those training trajectories.
3. Abstract and conclusion distinguish training runs from validation
   evaluation. The ETM comparison heading no longer suggests a controlled
   causal attribution to the enhancements.
4. Main-text repeatability now says that neural fits share fixed embeddings;
   sensitivity to relearning them was not measured. Directed-match aggregation
   is explained as successive averages of within-pair medians.
5. Supplementary metric definitions now report synthetic completion OOV
   exclusions of 35.42–37.33% of withheld pseudo-tokens. Word-recovery truth is
   vocabulary-restricted; mixture truth still includes all motif-labeled
   physical peaks and excludes background/noise.
6. Both PDFs now identify the public research branch. Equation-location
   docstrings and the research README point to the supplement. Two stale
   manuscript contract tests now check the two-document structure, preserved
   equations, bibliography and cross-references.
7. CI now builds both documents in alternating passes and rejects unresolved
   references. Previously it compiled only the main paper after the split.

## Verification

The [detailed results audit](final_results_audit_20260918.md) records the frozen
inputs, denominators, independent calculations and replay boundaries.

| Check | Result |
| --- | --- |
| Canonical scientific suite, `pytest -q benchmarks/neural_ms2lda/tests` | 367 passed |
| Production suite, `NUMBA_DISABLE_JIT=1 pytest -q tests` | 125 passed; two dependency deprecation warnings |
| Black and Ruff, exact research workflow scope | Pass, 102 Python files |
| All six report-input generators, exact workflow commands | Pass; no generated-input drift |
| Nine selected checkpoint reloads | Exported topic distributions and full validation mixtures bitwise identical to saved arrays |
| Independent selected completion calculation | NLL agrees within 4.48e-8; denominators exact |
| Available baseline completion replay | All three Tomotopy and two plain-ETM runs agree within 1.38e-7; denominators exact |
| Empirical figure regeneration | All five figures, in PDF and PNG, are byte-identical |
| Chemical and cross-model calculations | All four multiplicity families, 24,000 chemical topic/configuration records and 133,220 directed matches verified |
| Wheel build, payload inspection and isolated production import | Pass; production CLI present, research modules/runners excluded |
| MkDocs build in a separate temporary environment | Pass; existing production docstring/link warnings remain |
| Main/supplement LaTeX | 17 and 14 pages; no warnings or over/underfull boxes |
| PDF destination checks | 25 main-to-supplement, 7 supplement-to-main and 91 internal links resolve |
| Rendered visual review | Every page inspected; no visible layout defect |
| Final manuscript contracts and whitespace check | Pass |

The scientific environment was Python 3.11.16 and PyTorch 2.10.0+cu128 in
`ms2lda-neural`. The main-paper editor inspected all 17 rendered pages
individually. The parent agent inspected all 14 supplementary pages, with
full-page checks of changed equations, metric definitions and availability
text. The main retains five numbered equations and all three teaching figures.

One historical plain-ETM run (seed 7043) has sealed summary and per-topic
evidence but lacks its full local model arrays. Its chemical summaries and
reported values were checked against those records; its mixture and completion
inference could not be independently replayed. This boundary is separate from
the successful replay of all nine selected-model checkpoints.

## Remaining evidence limits

Validation informed model selection, and earlier test use prevents an
independent confirmation claim. Three repetitions on fixed data describe
run variation, not population uncertainty or statistical significance.
Training budgets and batch sizes are not fully matched, and the final model
has no complete factorial ablation. Larger synthetic topic inventories also
offer more candidate matches; recovery is not an identifiability proof.

Pseudo-count assumptions, conditional MAG/SOS populations, singleton support,
related molecular structures, exploratory multiplicity-adjusted screens and
correlated reference resources limit chemical interpretation. Replaying
annotation evidence does not independently validate its chemistry. Structural
follow-up and new evaluation data remain necessary. These limitations are
stated in the paper and do not change the reproduced numerical results.

There is no unresolved editorial or scientific blocker to sharing this
qualified study. Full retraining, a new untouched test cohort and new wet-lab
confirmation were not performed in this audit.
