# MSn Neural Replacement Conclusion

## 2026-05-20 decision

The unsupervised no-tomotopy neural replacement path should stop for now. The
experiments consistently fail to approach the MSn LDA benchmark, and the failure
mode is not limited to a single architecture or hyperparameter.

Recommended production/paper model:

```text
tomotopy LDA
```

Recommended use of the neural/no-LDA work:

```text
negative evidence and future R&D reference
```

## Reference results

| Model / experiment | Held-out? | Coverage | Mean SoS | QAC | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| LDA full | no | 0.832 | 0.6235 | 0.5188 | Strong production/reference model |
| LDA 2k | no | 0.592 | 0.5459 | 0.3232 | 2k reference target |
| neural-lda 2k | no | 0.604 | 0.5344 | 0.3228 | Shows local `theta @ beta` can work on train/all data |
| Fixed-beta tomotopy inference | yes | 0.083 | 0.5252 | 0.0436 | Held-out split is much harder |
| Fixed-beta neural encoder | yes | 0.056 | 0.5397 | 0.0302 | Encoder did not replace inference |
| Amortized neural K500 | yes | 0.238 | 0.5194 | 0.1236 | Best neural encoder-style no-LDA result before EM |
| Amortized neural K1000 | yes | 0.124 | 0.5498 | 0.0682 | More topics did not help |
| Refined amortized K500, best | yes | 0.178 | 0.5518 | 0.0982 | Better reconstruction worsened QAC |
| ProdLDA K500, best | yes | 0.016 | 0.5481 | 0.0088 | Collapsed / near-collapsed |
| Variational LDA EM K500 NMF | yes | 0.192 | 0.5685 | 0.1092 | Sparse beta was not enough |
| Variational LDA EM K500 random | yes | 0.262 | 0.5487 | 0.1438 | Best no-LDA held-out result, still far below LDA |

## What failed

The following directions are not worth further tuning for the current benchmark:

- one-shot neural encoders for `theta`;
- fixed-beta encoder imitation;
- vanilla ProdLDA / AVITM;
- theta refinement against the learned neural beta;
- reconstruction-only checkpoint selection;
- full-cache no-LDA scaling before a strong 2k held-out result.

These approaches can produce normalized `theta.npy` and `beta.npy`, but they do
not produce enough useful held-out Mass2Motifs.

## What the results imply

The core `theta @ beta` factorization is not dead. The all-2k `neural-lda`
result roughly matched 2k LDA when every spectrum had a directly optimized local
theta. Local theta optimization and tensorized factorization are therefore still
valid mechanisms.

The blocker is beta/objective alignment. Reconstructing BoW spectra is not the
same as learning sparse, reusable, chemically coherent Mass2Motifs. The clearest
evidence is theta refinement: reconstruction improved, but QAC dropped. The EM
experiments repeat the same pattern: later iterations improve reconstruction but
reduce held-out motif utility.

The best no-LDA held-out result, variational LDA EM with random initialization,
scored QAC `0.1438`, which beats the previous no-LDA K500 result (`0.1236`) but
is still less than half of the 2k LDA reference (`0.3232`). Its best checkpoint
was iteration 1 with very broad beta rows, so it is not a credible LDA-quality
motif learner.

## Recommendation

Stop treating no-tomotopy unsupervised neural topic modelling as a near-term LDA
replacement. Use tomotopy LDA for the MSn benchmark and paper-facing results.

If this line is revisited later, change the framing. Plausible future work would
need additional signal or a different objective, for example:

- neural encoders only as speed-ups or initializers after LDA-quality beta exists;
- fixed tomotopy motifs as features for downstream differentiable models;
- supervised or weakly supervised motif learning with chemical labels;
- objectives that directly reward motif support, diversity, and chemical
  coherence rather than only BoW reconstruction.
