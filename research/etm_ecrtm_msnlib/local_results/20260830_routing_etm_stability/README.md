# Routing ETM real training-seed stability

Status: **completed validation-only robustness study**. Candidate test remained
locked throughout.

## Decision

The Routing ETM formulation is stable enough to retain as the paper-facing
baseline. Across three model-training seeds on the identical frozen seed-42
MSnLib train/validation split, every run exceeds both M1 and Tomotopy on
evaluable and useful motif counts. Sparse per-spectrum mixtures, broad global
topic use, completion NLL and the learned context scale are also tightly stable.

The study also confirms the trade-off rather than removing it: every seed has
lower optimized coverage and mean SOS and worse completion NLL than M1. The
result supports an honest breadth-first claim, not uniform model dominance.

## Exact results

Only model initialization and minibatch order changed. Seed 7043 is the original
frozen run; seeds 23 and 37 are the new independent repetitions.

| training seed | optimized | evaluable | useful | mean SOS | median SOS | completion NLL | median effective | median support | unique top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7043 | 803 | 445 | 289 | 0.647153 | 0.657895 | 9.542924 | 3.699 | 6 | 828 |
| 23 | 791 | 453 | 274 | 0.637558 | 0.639535 | 9.546012 | 3.702 | 6 | 816 |
| 37 | 787 | 439 | 275 | 0.647350 | 0.647727 | 9.539388 | 3.714 | 6 | 813 |
| **mean** | **793.7** | **445.7** | **279.3** | **0.644020** | **0.648386** | **9.542775** | **3.705** | **6** | **819.0** |

Comparator values on the same split:

| model | optimized | evaluable | useful | mean SOS | median SOS | completion NLL |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 884 | 408 | 265 | 0.658079 | 0.648864 | 8.974140 |
| Routing ETM, three-seed mean | 793.7 | **445.7** | **279.3** | 0.644020 | 0.648386 | 9.542775 |
| Tomotopy | 607 | 206 | 138 | 0.676149 | 0.685450 | 9.662228 |

The range is more important than the mean for only three repetitions:

- evaluable motifs: **439-453**, always 31-45 above M1;
- useful motifs: **274-289**, always 9-24 above M1;
- optimized motifs: **787-803**, always below M1's 884;
- mean SOS: **0.637558-0.647350**, always below M1's 0.658079;
- completion NLL: **9.539388-9.546012**, extremely tight but always worse than
  M1's 8.974140;
- median effective topics: **3.699-3.714**, with median exact support exactly 6
  on every seed; and
- unique top-1 topics: **813-828**, with no catastrophic duplicate component on
  any seed.

## Interpretation

The discovery-breadth advantage is not a seed-7043 accident. All three runs
produce more evaluable and useful motifs than M1, and far more than Tomotopy.
The model's central short-spectrum behavior is also stable: a spectrum uses
about 3.7 entropy-effective topics, exact median support remains 6, and more than
800 of 1,000 topics win at least one validation spectrum.

Chemical quality is the variable part. Seed 23 increases evaluable breadth to
453 but shifts more topics into the low-SOS band, reducing mean SOS to 0.637558.
This is consistent with a breadth-quality trade-off rather than collapse. The
quality shortfall now reproduces often enough to justify one bounded coherence
experiment if further model work is desired.

This is descriptive robustness evidence on one frozen data split, not an
independent-dataset replication and not a test-set result. With n=3, the reported
sample standard deviations are audit descriptors, not population uncertainty
estimates.

## Reproduction

The current runner keeps the original seed behavior when `--training-seed` is
omitted and allows an explicit training-only override. Run it as a module from
the repository root:

```bash
conda run --no-capture-output -n ms2lda-neural \
python -m scripts.run_routing_etm_real train \
  --real-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed23 \
  --prepared-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 \
  --epochs 120 --batch-size 256 --device cuda --threads 6 \
  --training-seed 23

conda run --no-capture-output -n ms2lda-neural \
python -m scripts.run_routing_etm_real chemical \
  --real-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed23 \
  --data-root /home/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680
```

Replace both occurrences of `seed23` and `--training-seed 23` with seed 37 for
the second repetition.

Rebuild the compact package from the retained runs:

```bash
conda run -n ms2lda-neural \
python -m scripts.package_routing_etm_stability \
  --seed-run 23=/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed23 \
  --seed-run 37=/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-stability-seed37
```

Verify committed evidence only:

```bash
conda run -n ms2lda-neural python -m scripts.verify_routing_etm_stability
```

On the original host, also hash every large retained artifact and immutable
validation input:

```bash
conda run -n ms2lda-neural python -m scripts.verify_routing_etm_stability \
  --verify-local-artifacts --verify-inputs --require-external
```

The committed-only verifier performs 93 checks. The full local verification
performs 117 checks and passed without discrepancy.

## Evidence map

- `stability_summary.json`: exact rows, descriptive aggregates, comparator
  values and direction checks.
- `stability_by_seed.csv`: compact three-run audit table.
- `checkpoint_manifest.json`: implementation, package, retained artifact and
  immutable input hashes.
- `seed_23/` and `seed_37/`: compact config, metrics, full per-topic chemistry,
  training history, topic words, diagnostics, access audit and provenance.
- `scripts/package_routing_etm_stability.py`: deterministic packager.
- `scripts/verify_routing_etm_stability.py`: integrity and cross-file checker.

Large weights, beta/theta arrays and checkpoints remain outside Git. Their exact
paths, sizes and SHA-256 hashes are retained in the manifest and per-seed
provenance.

## Next bounded decision

The stability question is answered for initialization on this split. The one
predeclared positive-NPMI add-on was subsequently screened and failed its first
synthetic promotion gate, so it was not run on real validation. Do not repeat
identical seeds, tune the NPMI coefficient or add another donor mechanism.
Further robustness evidence should vary the split or external library.
The subsequent zero-parameter top-2 token simplification also failed its first
synthetic non-inferiority gate, so the one-scalar contextual route is retained.

Candidate test remains locked until an independent review accepts the method,
checkpoint and interpretation. Do not add M1's document gate, Sinkhorn target,
prototype separation, alternating optimizer or an unrestricted parameter sweep.
