# Routing-informed ETM checkpoint handoff

## Current decision

The routing-informed sparse ETM is frozen as the current paper-facing validation
baseline. Do not describe it simply as a failed model: it exceeds M1 and
Tomotopy on evaluable/useful discovery breadth, has sparse per-spectrum mixtures
and preserves a broad global topic inventory.

The predeclared all-gates Boolean remains false because optimized coverage, mean
SOS and NLL miss their M1-relative thresholds. Candidate test remains locked.
M1 is private donor evidence only and is not the publication model.

## Verify before doing anything

Run the committed integrity and consistency check:

```bash
conda run -n ms2lda-neural python \
  scripts/verify_routing_etm_checkpoint.py
```

On the original host, verify all retained local artifacts and immutable inputs:

```bash
conda run -n ms2lda-neural python \
  scripts/verify_routing_etm_checkpoint.py \
  --verify-inputs --verify-local-artifacts --require-external
```

Do not edit the files listed in
`local_results/20260830_routing_etm/checkpoint_manifest.json`. New experiments
must use a new result directory and compare back to this checkpoint.

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

## Evidence priority before model changes

No further architecture experiment is automatically authorized by this handoff.
The highest-value next evidence is real-training stability:

1. Keep the same train/validation split, vocabulary, SGNS, K=1000, MAG index,
   model configuration, epochs and evaluation.
2. Repeat only the training seed twice, without tuning against either outcome.
3. Report the distribution of optimized, evaluable, useful, mean/median SOS,
   NLL, sparse-support and inventory metrics.
4. Keep candidate test inaccessible.

This is a robustness study of the frozen paper-facing model, not M1 multiseed.

## Optional targeted model improvement

Only if additional real seeds reproduce the optimized-coverage and mean-SOS
deficit should one train-derived positive-NPMI coherence intervention be
considered. Predeclare one formulation and a bounded coefficient rule before
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

Use the existing `ms2lda-neural` Conda environment and `environment.yml`. Do not
redownload or rebuild MSnLib, SGNS, Spec2Vec, MAG or FAISS assets. Do not commit
weights, NumPy arrays, databases or indexes.

## Authoritative sources

- Current technical report:
  `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md`
- Detailed campaign report and replay commands:
  `local_results/20260830_routing_etm/README.md`
- Chronological decisions:
  `local_results/20260830_routing_etm/EXPERIMENT_LOG.md`
- Frozen manifest:
  `local_results/20260830_routing_etm/checkpoint_manifest.json`
- Model implementation: `benchmarks/neural_ms2lda/routing_etm.py`
- Real runner: `scripts/run_routing_etm_real.py`
- Synthetic runner: `scripts/run_routing_etm_campaign.py`
