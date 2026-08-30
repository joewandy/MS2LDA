# Routing-informed ETM checkpoint handoff

## Current decision

The routing-informed sparse ETM is frozen as the current paper-facing validation
baseline. Do not describe it simply as a failed model: it exceeds M1 and
Tomotopy on evaluable/useful discovery breadth, has sparse per-spectrum mixtures
and preserves a broad global topic inventory.

Across three unchanged real training seeds, the model produces 439--453
evaluable and 274--289 useful motifs. The discovery advantage over M1 and
Tomotopy is therefore not a one-initialization result. The predeclared all-gates
Boolean remains false because optimized coverage, mean SOS and NLL consistently
miss their M1-relative thresholds. Candidate test remains locked. M1 is private
donor evidence only and is not the publication model.

## Verify before doing anything

Run the committed integrity and consistency check:

```bash
conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_checkpoint

conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_stability
```

On the original host, verify all retained local artifacts and immutable inputs:

```bash
conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_checkpoint \
  --verify-inputs --verify-local-artifacts --require-external

conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_stability \
  --verify-inputs --verify-local-artifacts --require-external
```

Do not mutate the frozen baseline evidence listed in
`local_results/20260830_routing_etm/checkpoint_manifest.json`. Its source
implementation is verified from the recorded Git commit. New experiments must
use a new result directory and compare back to that checkpoint.

## Frozen result

| metric | M1 | **Routing ETM** | Tomotopy |
|---|---:|---:|---:|
| optimized motifs | 884 | 803 | 607 |
| evaluable motifs | 408 | **445** | 206 |
| useful motifs | 265 | **289** | 138 |
| mean SOS | 0.658079 | 0.647153 | 0.676149 |
| median SOS | 0.648864 | 0.657895 | 0.685450 |
| completion NLL | 8.974140 | 9.542924 | 9.662228 |

Routing ETM also has median 3.70 effective topics, median exact support 6, 828
unique top-1 topics and 538.40 corpus-effective topics, with no catastrophic
duplicate component.

## Completed real training-seed stability

Seeds 23 and 37 changed only initialization and minibatch-order RNG. They reused
the exact frozen seed-42 train/validation split, vocabulary, SGNS, K=1000,
configuration, epochs and evaluation. Candidate test remained inaccessible.

| training seed | optimized | evaluable | useful | mean SOS | NLL | median effective | unique top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7043 (original) | 803 | 445 | 289 | 0.647153 | 9.542924 | 3.699 | 828 |
| 23 | 791 | 453 | 274 | 0.637558 | 9.546012 | 3.702 | 816 |
| 37 | 787 | 439 | 275 | 0.647350 | 9.539388 | 3.714 | 813 |

Every seed exceeds M1's 408 evaluable / 265 useful motifs and Tomotopy's
206/138. Every seed also reproduces the lower optimized coverage, lower mean SOS
and worse completion NLL relative to M1. Median exact support is 6 in every run,
and no run has a catastrophic duplicate component. This is descriptive
same-split initialization stability at n=3, not split or external generalization.

## Optional targeted model improvement

The residual optimized-coverage and mean-SOS deficit has now reproduced, so one
train-derived positive-NPMI coherence intervention may be considered as the next
bounded experiment. Predeclare one formulation and coefficient rule before
training. Screen synthetically first, then promote at most one configuration to
validation.

It must preserve or improve the current evaluable/useful breadth, exact support,
unique top-1 breadth and non-collapse diagnostics. Stop if it merely trades away
the discovery advantage to satisfy the old Boolean gate.

Do not add the M1 document gate, Sinkhorn balancing, prototype separation,
alternating optimizer, temperature schedule or another custom prior. Do not run
an unrestricted coefficient or architecture search.

## Shared immutable inputs

- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/sparse-etm-campaign-20260830`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-campaign-20260830`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed23`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed37`

Use the existing `ms2lda-neural` Conda environment and `environment.yml`. Do not
redownload or rebuild MSnLib, SGNS, Spec2Vec, MAG or FAISS assets. Do not commit
weights, NumPy arrays, databases or indexes.

## Authoritative sources

- Current technical report:
  `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md`
- Detailed campaign report and replay commands:
  `local_results/20260830_routing_etm/README.md`
- Real training-seed stability package:
  `local_results/20260830_routing_etm_stability/README.md`
- Chronological decisions:
  `local_results/20260830_routing_etm/EXPERIMENT_LOG.md`
- Frozen manifest:
  `local_results/20260830_routing_etm/checkpoint_manifest.json`
- Stability manifest:
  `local_results/20260830_routing_etm_stability/checkpoint_manifest.json`
- Model implementation: `benchmarks/neural_ms2lda/routing_etm.py`
- Real runner: `scripts/run_routing_etm_real.py`
- Synthetic runner: `scripts/run_routing_etm_campaign.py`
