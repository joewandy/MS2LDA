# Real-MSnLib Neural MS2LDA follow-up

This is a validation-only follow-up to
`20260827_seed42_validation`. Candidate test theta, completion, chemistry,
MAG/SOS, and result artifacts were not opened, loaded, scored, or summarized.

## Current answer

The pooled model is not merely miscalibrated. It contains a useful specialized
sub-inventory, and inference-only sharpening recovers substantially more
chemistry, but 614/1000 prototypes form one near-exact, almost-uniform beta
component that never wins a validation spectrum. Rank-preserving calibration
therefore has a hard practical ceiling of 374 evaluable topics and cannot pass
the frozen breadth gates.

Tau 0.11 is the selected pooled diagnostic calibration. It lies in a robust
0.10-0.12 region and most closely matches M1's mixture sparsity without choosing
temperature solely for useful-motif count. It produces 293 evaluable and 194
useful motifs, mean SOS 0.665436, median SOS 0.657143, completion NLL 8.994812,
and median 7.35 effective topics/spectrum. It passes optimized, mean-SOS, NLL,
and finite/stability gates, but fails evaluable, useful, and no-catastrophic-
component-collapse gates.

Canonical ETM sharpening is diagnostic only. Tau 0.8 raises evaluable/useful
motifs from 130/79 to 163/100 while retaining the NLL gate, but beta remains
limited to 609 optimized motifs. Stronger sharpening reaches only 186/114 and
breaks the NLL gate. Calibration cannot repair ETM's topic-word deficit.

Fragment/loss-balanced ETM repairs annotatable beta coverage (911 optimized)
and eliminates channel skew, but does not rescue chemistry: raw validation has
166 evaluable / 104 useful motifs, mean SOS 0.629819, and NLL 8.766069. A
diagnostic temperature sweep reaches at most 240 / 148, never passes mean SOS,
and breaks the NLL gate below tau 0.8. No balanced temperature is selected.

Canonical ECRTM was scientifically justified but operationally failed closed.
It completed 21/40 epochs before Sinkhorn demand rose to 721 mean / 951 maximum
iterations per batch. Epoch 22 could not meet residual 0.005 within the locked
1,000-iteration cap, and one exact checkpoint resume failed again. No partial
ECRTM model was inferred or chemically scored; the unconverged 50-step
approximation was not substituted.

M1 therefore remains the simplest scientifically defensible model and the only
gate-passing neural model. None of the new candidates should advance to test.

## Validation headline

| Model | Optimized | Evaluable | Useful | Mean SOS | Completion NLL | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M1 locked reference | 884 | 408 | 265 | 0.658079 | 8.974140 | Pass |
| Canonical ETM | 609 | 130 | 79 | 0.638796 | 8.690730 | Fail breadth/SOS |
| Pooled likelihood | 967 | 14 | 11 | 0.751621 | 8.385787 | Fail breadth/collapse |
| Pooled likelihood, tau 0.11 | 967 | 293 | 194 | 0.665436 | 8.994812 | Fail breadth/collapse |
| Pooled MI 0.05 | 799 | 14 | 11 | 0.696540 | 8.393690 | Fail breadth/collapse |
| Balanced ETM | 911 | 166 | 104 | 0.629819 | 8.766069 | Fail breadth/SOS |
| Balanced ETM, tau 0.8 diagnostic | 911 | 209 | 130 | 0.620555 | 9.294272 | Fail breadth/SOS |
| Canonical ECRTM | - | - | - | - | - | Infeasible at epoch 22 |

## Evidence classes

### Previous campaign findings

- Locked M1: 884 optimized, 408 evaluable, 265 useful, mean SOS 0.658079,
  completion NLL 8.974140.
- Canonical fixed-SGNS ETM: 609/130/79, mean SOS 0.638796, NLL 8.690730.
- Pooled likelihood: 967/14/11, mean SOS 0.751621, NLL 8.385787.
- Pooled MI=0.05: 799/14/11, mean SOS 0.696540, NLL 8.393690.

### Exploratory diagnostics

- Dense pooled temperature sweep with fixed beta/MAG annotations.
- Canonical ETM temperature diagnostic with fixed beta/MAG annotations.
- Locked M1 theta-shape comparison on byte-identical validation records.
- Per-topic pooled beta/prototype redundancy and top-1 competition analysis.
- Paired full-K/V optimizer smoke for balanced ETM.

### Newly trained models

- Fragment/loss-balanced fixed-SGNS ETM: complete, fails chemistry breadth and
  mean-SOS gates despite passing optimized coverage and completion.
- Canonical ECRTM: failed closed after 21/40 epochs because the canonical
  Sinkhorn solver no longer converged within 1,000 iterations. No partial model
  was evaluated.

### Post-hoc calibrated models

- Pooled likelihood + tau 0.11: complete, fails breadth/collapse gates.
- Canonical ETM temperature sweep: diagnostic only, no selected candidate.
- Balanced ETM temperature sweep: diagnostic only, no selected candidate.
- ECRTM tau 0.30: not produced because base training did not complete.

### Candidates passing all frozen validation gates

None. No candidate is authorized for test.

## Explicit decision questions

1. **Is pooled fundamentally bad or mainly miscalibrated?** Both effects are
   present. The 386-topic specialized sub-inventory has useful chemistry, but
   614 near-identical, almost-uniform topics constitute severe component
   collapse.
2. **Can theta sharpening recover broad high-confidence chemistry?** It recovers
   substantial chemistry but not enough: selected tau 0.11 reaches 293
   evaluable and 194 useful motifs versus gates of 388 and 252.
3. **What tau region is stable/useful?** Approximately 0.10-0.12. Tau 0.11 is
   selected because its median effective-topic count (7.35) closely matches
   M1 (7.20).
4. **Does redundancy explain diffuse pooled theta?** It explains a major part
   of the inventory failure. Exactly 614 topics form the dominant near-exact
   component and none wins top-1; only 374 topics ever win top-1.
5. **Does channel balancing rescue ETM?** No. It raises optimized motifs from
   609 to 911 and removes fragment/loss skew, but raw evaluable/useful motifs
   are only 166/104 and mean SOS falls to 0.629819.
6. **Can balanced-ETM theta calibration rescue it?** No. Tau 0.8 retains the
   NLL gate but reaches only 209/130 and mean SOS 0.620555. Harder settings
   asymptote at 240/148, become nearly one-topic-per-spectrum, and fail NLL.
7. **Is ECRTM justified?** The scientific question was justified, but this
   implementation is not operationally viable at real K/V: it failed the
   canonical convergence contract in epoch 22 even after one exact resume.
   Raising the cap would continue an already severe runtime escalation; using
   50 steps would be an unconverged numerical approximation.
8. **Is one coherence/diversity mechanism justified?** A diversity/anti-collapse
   mechanism remains justified; a coherence/NPMI mechanism is not supported by
   the measured defect. Because canonical ECRTM was numerically impractical, a
   future campaign may test exactly one cheaper, published diversity mechanism
   on the pooled model. That is a new validation experiment, not a reason to
   reinterpret the current failures.
9. **Simplest defensible candidate?** M1 remains the only gate-passing neural
   model. Pooled tau 0.11 is simpler but not defensible as a replacement because
   it fails breadth and collapse gates.
10. **Should anything advance to test?** No. Test remains locked.

See `EXPERIMENT_LOG.md` for hypothesis-by-hypothesis decisions and the CSV/JSON
files in this directory for exact metrics.
