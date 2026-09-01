# Routing-informed sparse ETM campaign

## Decision

Freeze this model as the current paper-facing validation baseline. It is already
the strongest discovery result among the explainable published-base candidates:
it produces more evaluable and useful validation motifs than both the locked M1
reference and Tomotopy, while retaining sparse per-spectrum mixtures and a broad
global topic inventory.

The right description is **a strong near-pass and the new baseline**, not simply
"failed." The predeclared all-gates Boolean remains false because optimized
coverage, mean SOS and completion NLL miss their conservative M1-relative
thresholds. That formal result keeps candidate test locked; it does not erase the
candidate's discovery gains.

All figures below are from the same seed-42 validation split:

| metric | M1 | **Routing ETM** | Tomotopy |
|---|---:|---:|---:|
| optimized motifs | **884** | 803 | 607 |
| evaluable motifs | 408 | **445** | 206 |
| useful motifs | 265 | **289** | 138 |
| mean SOS | 0.658079 | 0.647153 | **0.676149** |
| median SOS | 0.648864 | 0.657895 | **0.685450** |
| completion NLL (lower is better) | **8.974140** | 9.542924 | 9.662228 |

Routing ETM is therefore better on discovery breadth, not on every scalar. It
beats M1 by 37 evaluable and 24 useful motifs and beats Tomotopy by 239 and 151,
respectively. Tomotopy has the highest SOS among its much smaller evaluable set;
M1 has the best completion NLL and optimized coverage. Routing ETM makes the
strongest breadth/sparsity trade-off while remaining explainable.

The tested model is a balanced fixed-SGNS Embedded Topic Model with two small,
auditable inference changes:

1. observed fragment/loss words provide top-2 contextual evidence to the ETM
   Gaussian posterior mean; and
2. published 1.5-entmax maps the latent vector to an exactly sparse topic
   mixture.

This is the first ETM-family result in this repository to fix both previously
observed real failure modes at once. Its median spectrum uses 3.70 effective
topics with exact support 6, while 828 of 1,000 topics win at least one
validation spectrum. Chemistry reaches 445 evaluable and 289 useful motifs,
exceeding the frozen targets of 388 and 252.

It is not a full frozen-gate pass. Optimized coverage is 803 rather than 840,
mean SOS is 0.647153 rather than 0.651498, and completion NLL is 9.542924 rather
than at most 9.422847. Candidate test therefore remains locked.

The result is qualitatively different from earlier near misses: the remaining
deficits are chemical-quality/generalization trade-offs, not diffuse document
mixtures or a collapsed global topic inventory. Further architecture changes
are optional improvements to a viable model, not rescue work.

## What the model actually does

The paper-facing generator is unchanged ETM: fixed train-only SGNS word
coordinates, learned topic coordinates, an embedding topic-word decoder,
multinomial reconstruction, a Gaussian variational latent, and analytic
standard-normal KL. The fragment and neutral-loss decoder channels each receive
half of every topic's probability mass.

For inference, each nonzero spectrum word follows a short calculation:

1. take its fixed SGNS vector;
2. add one learned scalar times the mean vector of the other words in the same
   spectrum;
3. compare that vector with ETM's own topic vectors;
4. retain its two strongest topic matches and normalize those two scores;
5. average this evidence over the observed spectrum; and
6. add a centered, bounded log-evidence offset to the ordinary ETM posterior
   mean before applying 1.5-entmax.

The uniform pseudocount is fixed at `1/K`, routing temperature is fixed at 1.0,
and routing strength is fixed at 1.0. The only additional learned parameter is
the context scalar; it finished at 0.1772. Total parameters are 19,278,001,
versus 19,278,000 for the balanced ETM base.

There is no nonlinear donor router, document gate, Sinkhorn target, positive-NPMI
loss, prototype-separation loss, alternating optimizer, temperature schedule or
custom prior. The private model is donor evidence only, not a proposed base or
publication target.

## Synthetic mechanism screen

The paired fragment/loss simulator used 18 planted motifs, 1--3 motifs per
spectrum, train-only vocabulary and SGNS, 800 training / 160 validation spectra,
raw intensity pseudo-count reconstruction, 120 epochs, and fixed seeds 11, 23
and 37.

K=36 means over all three seeds:

| formulation | NLL | beta cosine | theta cosine | median effective | active >0.5% | unique top-1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| balanced ETM + softmax | 6.4710 | 0.3052 | 0.4868 | 3.59 | 8.0 | 6.7 | control |
| balanced ETM + entmax | 6.5977 | 0.2566 | 0.3973 | 1.88 | 5.0 | 4.7 | topic starvation |
| top-2 context + softmax | **6.2438** | 0.4423 | 0.7066 | 3.22 | 13.3 | 11.7 | recovery gain, insufficient diffuse-stress sparsity |
| **top-2 context + entmax** | 6.2892 | **0.4648** | **0.7513** | **2.00** | **14.0** | **13.0** | promote |

The component ladder was informative:

- soft token evidence produced only a small theta gain;
- adding leave-one-out context helped further;
- top-2 contextual evidence supplied the large recovery and inventory gain;
- top-2 routing with dense softmax still failed a deliberately diffuse stress;
- entmax alone was sparse but starved topics; and
- top-2 evidence plus entmax supplied the complementary combination.

One auxiliary seed-23 comparison was recorded as a miss rather than hidden:
routing+entmax beta cosine was 0.4215 versus 0.4594 for routing+softmax, a loss
of 0.0379 rather than the predeclared 0.0300 allowance. Relative to the actual
ETM base, however, the combination improved beta, theta, likelihood, sparsity,
active topics and top-1 breadth on every seed. High-K adjudication was therefore
required before real promotion.

Full rows are in `synthetic_by_seed.csv`; compact means are in
`synthetic_summary.csv`.

## K=128 adjudication

Seed 11 changed only fitted K from 36 to 128.

| formulation | NLL | beta cosine | theta cosine | top accuracy | median effective / support | active >0.5% | unique top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced ETM + softmax | 6.4175 | 0.3696 | 0.5782 | 0.5438 | 3.67 / 128 | 7 | 7 |
| balanced ETM + entmax | 6.7375 | 0.2344 | 0.2965 | 0.2625 | 1.72 / 2 | 3 | 3 |
| **top-2 context + entmax** | **6.1276** | **0.6638** | **0.9556** | **0.9313** | **1.45 / 2** | **20** | **18** |

The selected model recovered all 18 planted motifs as distinct top-1 topics,
retained 20 materially active topics, improved likelihood and recovery, and had
no catastrophic duplicate component. This was a clear high-K promotion pass.

## Real MSnLib validation

The sole real candidate reused the frozen seed-42 training/validation split,
vocabulary, train-only 48D SGNS, K=1000, leakage-filtered MAG index, membership
threshold, completion metric and SOS calculation. Training used the unchanged
raw pseudo-count objective, Adam 0.005, weight decay 1.2e-6, hidden size 800,
batch 256, 120 fixed epochs, deterministic CUDA and six CPU threads.

| model | optimized | evaluable | useful | mean SOS | NLL | median effective | unique top-1 | outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| canonical ETM | 609 | 130 | 79 | 0.638796 | 8.690730 | 43.79 | 260 | fail chemistry/diffuseness |
| balanced ETM | 911 | 166 | 104 | 0.629819 | 8.766069 | 46.72 | 260 | fail chemistry/diffuseness |
| balanced gated ETM gamma 2 | 889 | 220 | 138 | 0.645148 | 8.686967 | 65.57 | 471 | fail chemistry/diffuseness |
| prior sparse ETM | 993 | 7 | 6 | 0.663853 | 9.577829 | 1.80 | 20 | fail inventory/NLL |
| **routing-informed sparse ETM** | **803** | **445** | **289** | **0.647153** | **9.542924** | **3.70** | **828** | strong near-pass; freeze baseline |

The locked private donor reference produced 884 / 408 / 265 optimized,
evaluable and useful motifs. It is retained in `comparison.csv` solely as a
historical validation comparator. The routing-informed ETM exceeds its evaluable
and useful counts but does not satisfy the complete frozen contract. These two
statements are both true and should always be reported together.

Frozen gate detail:

| gate | threshold | candidate | outcome |
|---|---:|---:|:---:|
| optimized | >=840 | 803 | **fail** |
| evaluable | >=388 | 445 | pass |
| useful | >=252 | 289 | pass |
| mean SOS | >=0.651498 | 0.647153 | **fail** |
| completion NLL | <=9.422847 | 9.542924 | **fail** |
| finite/stable | required | yes | pass |
| no catastrophic duplicate component | required | yes | pass |

Median SOS was 0.657895. SOS bands were 67 above 0.8, 222 from 0.6 to
0.8, and 156 below 0.6. Membership associated 1,400 validation spectra and
1,393 molecules. Held-out compounds were excluded from MAG.

### Theta and global inventory

- Exact support: minimum 1, median 6, mean 6.89, p75 9, p95 13, p99 17,
  maximum 23.
- Effective topics: median 3.70, mean 4.09.
- Maximum theta: median 0.444; fractions >=0.5/0.3/0.2 were
  0.360/0.850/0.989.
- Inventory: 828 unique top-1 topics, 535 active above 0.0005, 273 active at
  least `1/K`, and 538.40 corpus-effective topics.
- Maximum mean topic use was only 0.0143.
- Beta geometry: mean nearest-topic cosine 0.1529 and maximum cosine 0.99965.
  At the frozen strict 0.999 threshold, one two-topic pair exists; there is no
  catastrophic component.
- Beta sharpness: median 367.75 effective words, median maximum word probability
  0.1894, median top-20 mass 0.4388, and top-word uniqueness 0.31045.
- Fragment/loss balance remained exact at 0.5 with no extreme channel skew.
- Completion OOV fraction was 0.03136.

## Runtime and system load

The full run trained in 876.8 seconds (14.6 minutes) on the RTX 5070; including
final inference and file hashing it completed in about 16 minutes. One-epoch
smoke and steady-state measurements showed roughly 7--8 seconds per epoch,
about 83% GPU utilization, 2.7 of 12.2 GiB total GPU memory in use, 53 C, and
116 W of a 250 W limit.

PyTorch peak allocated/reserved CUDA memory was 0.821/1.053 GB. Process high-water
memory was 2.879 GB, and at least 12.43 GB system memory remained available.
Deterministic inference processed 23,426 full validation spectra/second. The
load was sustained but comfortably below memory, thermal and power limits.

## Direct answers

1. **Is dense softmax the main cause of ETM diffuseness?** It is a contributor,
   not the complete cause. Removing it without routing produced sparse topic
   starvation. Routing without removing it remained too diffuse under stress.
2. **Does sparsemax/entmax solve the problem?** Not alone. Entmax becomes useful
   when routing supplies a broad, recoverable posterior signal.
3. **Does pseudo-count scaling matter?** Yes synthetically, but it is not the
   selected solution. The earlier distinct-word candidate failed real inventory
   and NLL. This successful repair keeps raw pseudo-count reconstruction.
4. **Does a sparse prior add anything beyond sparse theta?** Unknown and not yet
   needed. No sparse-prior claim is made.
5. **Which mechanism survives K=128?** Top-2 contextual evidence plus entmax 1.5
   with raw counts.
6. **Which formulation was promoted?** Balanced fixed-SGNS ETM with a bounded
   top-2 contextual posterior offset, entmax 1.5 theta and raw counts.
7. **Does real theta approach the desired short-document scale?** Yes. Median
   effective topics are 3.70 and median exact support is 6, with a small tail.
8. **Does topic inventory remain broad?** Yes: 828 unique top-1 and 538
   corpus-effective topics, without catastrophic duplication.
9. **Does chemistry reach every frozen gate?** No. Evaluable and useful pass;
   optimized, mean SOS and NLL fail.
10. **Is the candidate recognizably ETM?** Yes. The ETM generator, likelihood,
    Gaussian variational family and KL are unchanged.
11. **Is it scientifically simpler than the private donor?** Yes mechanically:
    one extra learned scalar and a short posterior-evidence calculation, although
    ETM's embedding decoder has many more raw parameters.
12. **What exact failure remains?** A 37-motif optimized-coverage shortfall,
    0.004345 mean-SOS shortfall, and 0.120077 NLL excess.
13. **Is further ETM work justified?** The model is already viable. Same-split
    stability is now established across three training seeds; any model change
    must be optional, bounded and justified by the reproduced residual.
14. **Is M1 multiseed next?** No. The private model remains donor evidence only.

## What should improve next

This checkpoint has now been followed by two unchanged validation-only training
seeds. The compact evidence and machine-checkable summary are in
`../20260830_routing_etm_stability/README.md`. Across all three runs, Routing ETM
has 439--453 evaluable and 274--289 useful motifs, median 3.70--3.71 effective
topics and 813--828 unique top-1 topics. The discovery advantage, sparsity and
broad inventory are stable on this split; the coverage/SOS/NLL trade-off also
reproduces.

The priorities are now:

1. **Keep the candidate test locked.** Test becomes appropriate only after the
   method, metrics and acceptance interpretation are independently reviewed and
   frozen.
2. **Do not tune coherence regularization.** The predeclared weight-1
   positive-NPMI experiment subsequently improved its graph loss but reduced
   true-beta recovery and stopped before real validation.
3. **Retain the one-scalar context.** A zero-parameter top-2 token route improved
   over entmax alone but failed direct non-inferiority to contextual routing.
4. **Do not repeat identical seeds indefinitely.** The current n=3 result is
   adequate for initialization stability; future robustness evidence should vary
   the split or dataset.
5. **Do not reconstruct M1.** The donor document gate, Sinkhorn balancing,
   prototype separation, alternating optimizer and temperature schedule remain
   out of scope.

The completion gap is already smaller than Tomotopy's but worse than M1's. It
should be reported as a trade-off rather than optimized away at the cost of the
discovery gains.

## Exact verification and replay

The committed checkpoint is machine-checkable without training or opening test:

```bash
conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_checkpoint
```

On the original host, also verify every immutable validation input and retained
large artifact against its recorded SHA-256:

```bash
conda run -n ms2lda-neural python \
  -m scripts.verify_routing_etm_checkpoint \
  --verify-inputs --verify-local-artifacts --require-external
```

Replay the frozen real training run into a new validation-only directory:

```bash
conda run -n ms2lda-neural python -m scripts.run_routing_etm_real train \
  --real-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-checkpoint-replay-seed42 \
  --prepared-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 \
  --epochs 120 --batch-size 256 --device cuda --threads 6

conda run -n ms2lda-neural python -m scripts.run_routing_etm_real chemical \
  --real-run /home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-checkpoint-replay-seed42 \
  --data-root /home/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680
```

The exact synthetic decision configs are in `configs/`. For the promoted
formulation, replay seeds 11, 23 and 37 at K=36 and seed 11 at K=128 with
`scripts/run_routing_etm_campaign.py`, `top2_context`, `entmax15` and
`raw_counts`. `checkpoint_manifest.json` freezes the implementation, evidence,
comparator values and replay locations.

## Reproducibility and evidence boundary

All selection was synthetic or real validation only. Candidate test theta,
completion, MAG, SOS and result files were not loaded, computed or inspected.
The explicit audit is in `validation_access_audit.json`.

Large local artifacts are retained outside Git:

| artifact | bytes | SHA-256 |
|---|---:|---|
| weights | 81,193,085 | `bd18778f2fa8e3e20280f778386306764b8ff22957ed85307225783681941259` |
| checkpoint | 235,446,608 | `6a57fd11b9a94dd065fcbfcc4f8e75f0c9cf6a2e3e683fddbff78c22ac7bed47` |
| beta | 84,932,128 | `c912be5600b7883d169c7be249226d1433c7dc7752ad53da0c3d15c1afa743ce` |
| validation full theta | 15,556,128 | `ce922df7acdae58318e0033bba3967ea8350e73109e92b35e1748408821bf51b` |
| validation routing evidence | 15,556,128 | `209d74871b98c717e90eba366388164d604cbb9bd8eb0c7c50fb5b81db9e9bf0` |
| training log | local text | `1f467e4987e833742efd1083595afa1c20b57de8c1b7fe3129c6080fb5427057` |
| chemistry log | local text | `4bce4fad3d2880c36cb9e1ac962acf64ab6cb43d1c9940b952d1d114f3f58471` |

No `.pt`, `.npy`, FAISS index, database, raw MSnLib asset or candidate-test
artifact is committed.

## Quality control

- Routing/sparse focused tests: 24 passed.
- Full neural suite: 93 passed.
- CI-equivalent production regression suite: 87 passed, with the two existing
  empty-document NumPy warnings.
- Black and Ruff: passed on every changed Python file.
- Repository-wide Black/Ruff sweeps still identify formatting in four untouched
  pre-existing files; those unrelated files were not changed in this commit.
- Finite loss/gradient, theta non-negativity/simplex, exact sparse support,
  deterministic inference, validation isolation and checkpoint advancement:
  passed.

## Evidence map

- `EXPERIMENT_LOG.md`: chronological predeclarations, results and decisions.
- `synthetic_by_seed.csv`, `synthetic_summary.csv`, `high_k_stress.csv`: compact
  synthetic evidence.
- `comparison.csv`: real baseline/candidate metrics and frozen gate booleans.
- `metrics.json`, `chemical_scores.csv`, `theta_support_summary.csv`,
  `duplicate_component_summary.json`, `fragment_mass_summary.json` and
  `top_words.csv`: promoted-candidate validation outputs.
- `validation_access_audit.json`: train/validation-only and chemistry audit.
- `provenance.json`: immutable input hashes and local large-artifact hashes.
- `checkpoint_manifest.json`: machine-readable frozen implementation, metrics,
  evidence hashes, comparator values and replay locations.
- `configs/`: every decision-bearing synthetic config; `config.json` is the
  frozen real configuration.
- `scripts/verify_routing_etm_checkpoint.py`: one-command integrity and
  cross-file consistency check.
