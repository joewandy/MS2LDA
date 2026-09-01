# Routing ETM positive-NPMI experiment log

Status: predeclared before implementation or training. Validation only; candidate
test remains locked.

## Experiment 1: one isolated coherence term

- **Question:** can the smallest literature-supported coherence intervention
  recover Routing ETM's optimized-coverage / mean-SOS deficit without sacrificing
  its evaluable/useful breadth, sparse mixtures or broad topic inventory?
- **Published base:** the frozen ETM generator with fixed train-only SGNS,
  fragment/loss balancing, Gaussian variational posterior, standard-normal KL,
  raw-count multinomial reconstruction, top-2 contextual posterior evidence and
  alpha-entmax 1.5.
- **Only change:** add the existing train-derived positive-NPMI topic loss to the
  ordinary minibatch objective:

  `loss = reconstruction + mean(KL) + 1.0 * mean_k[-log(beta_k^T G beta_k)]`

  `G` is built from binary training-document presence only. It retains mutual
  positive-NPMI neighbours with minimum document frequency 10, minimum pair
  frequency 3, at most 16 neighbours per word and NPMI greater than zero.
- **Coefficient rule:** weight 1.0 is frozen before training because it is the
  already-audited value in the locked M1 protocol. There is no coefficient,
  threshold or schedule search.
- **Excluded changes:** no document gate, Sinkhorn target, prototype separation,
  alternating optimizer, temperature schedule, new prior, decoder change,
  reconstruction rescaling or additional learned parameter.
- **Synthetic control:** compare to the exact saved Routing ETM runs with the
  same seeds, data, initialization, batch order, K, epochs and evaluation.
- **Initial triage:** seed 11, K=36, 120 epochs.
- **Triage gate:** finite loss/gradients; higher train-graph affinity; true-beta
  cosine improves by at least 0.01; true-theta cosine falls by no more than 0.02;
  held-out NLL is no more than 5% worse; active and unique-top-1 topics fall by no
  more than 2; median effective topics rises by no more than 0.5; and there is no
  catastrophic duplicate component.
- **Multi-seed gate:** only after triage passes, repeat K=36 at seeds 23 and 37.
  All seeds must be finite and non-collapsed, no seed may lose more than 0.02
  true-beta cosine or 0.03 true-theta cosine, NLL must remain within 5%, active
  and unique-top-1 counts may fall by no more than 2, and mean true-beta recovery
  must improve by at least 0.01.
- **High-K gate:** only after the multi-seed gate passes, run seed 11 at K=128.
  It must retain all 18 planted motifs as distinct top-1 topics, finite execution,
  no catastrophic duplicate component and NLL within 5% of the exact control.
- **Real promotion:** at most this one formulation may advance. Reuse the frozen
  seed-42 train/validation split, K=1000, seed 7043 and unchanged MAG/SOS
  evaluation. Do not access candidate test.
- **Real retention rule:** keep it only if it improves mean SOS or optimized
  coverage over the seed-7043 Routing ETM while retaining at least 439 evaluable,
  274 useful and 800 unique-top-1 topics, sparse median support no greater than 8,
  completion NLL no more than 1% worse, finite execution and no catastrophic
  duplicate component. The original all-gates Boolean remains separately frozen
  and is not relaxed.
- **Stopping rule:** stop immediately at the first failed promotion stage. Do not
  compensate with a larger coefficient or another M1 component.

### Result

The seed-11 K=36 run completed with finite loss and gradients. It reduced the
saved train-graph coherence loss from 5.526062 to 5.499502 (lower is better), so
the implementation demonstrably optimized the intended NPMI objective. It did
not improve recovery of the planted topic-word distributions.

| metric | exact Routing ETM control | + positive-NPMI | change |
|---|---:|---:|---:|
| held-out NLL | 6.278416 | 6.287552 | +0.009136 |
| true-beta cosine | 0.498454 | 0.491576 | -0.006878 |
| true-theta cosine | 0.764875 | 0.765219 | +0.000344 |
| planted motifs recovered at cosine >=0.50 | 10 | 10 | 0 |
| active topics above 0.5% | 14 | 14 | 0 |
| unique top-1 topics | 14 | 14 | 0 |
| median effective topics | 1.971 | 1.955 | -0.017 |
| median exact support | 4 | 4 | 0 |
| mean nearest beta cosine | 0.740197 | 0.747304 | +0.007107 |
| catastrophic duplicate component | no | no | unchanged |

The NLL change is only +0.15%, theta behavior and inventory are effectively
unchanged, and the model remains finite and sparse. However, the predeclared
triage rule required at least +0.01 true-beta cosine. The observed change is
-0.006878. The intervention therefore fails Experiment 1.

### Decision

Stop. Do not run seeds 23/37, K=128 or real MSnLib validation, and do not tune a
larger NPMI coefficient. The simple weight-1 coherence add-on slightly improves
its own train-graph statistic but does not improve planted motif recovery. It is
not justified as an addition to the paper-facing model. The frozen Routing ETM
remains the simpler and better-supported formulation; candidate test remains
locked.
