# Routing-informed ETM for Neural MS2LDA

Status: **frozen validation checkpoint and current paper-facing baseline**.
Candidate test remains locked.

Branch: `codex/unified-ms2lda-environment`

## Bottom line

Routing-informed sparse ETM is already a successful discovery model. Across
three model-training seeds on the locked seed-42 validation split, every run
produces more evaluable and useful motifs than both the private M1 reference and
the production Tomotopy comparator, while document mixtures remain sparse and
the global topic inventory remains broad.

| metric | M1 | **Routing ETM** | Tomotopy |
|---|---:|---:|---:|
| optimized motifs | **884** | 803 | 607 |
| evaluable motifs | 408 | **445** | 206 |
| useful motifs | 265 | **289** | 138 |
| mean SOS | 0.658079 | 0.647153 | **0.676149** |
| median SOS | 0.648864 | 0.657895 | **0.685450** |
| completion NLL (lower is better) | **8.974140** | 9.542924 | 9.662228 |

The conclusion is not that Routing ETM wins every metric. It wins the main
Mass2Motif discovery-breadth measures, beats Tomotopy on completion, and fixes
the two earlier ETM failures: diffuse document mixtures and collapsed/starved
topic inventories. M1 retains better optimized coverage, mean SOS and completion
NLL; Tomotopy has higher SOS over a much smaller evaluable set.

The three Routing ETM runs span 787-803 optimized, 439-453 evaluable and 274-289
useful motifs, mean SOS 0.637558-0.647350 and NLL 9.539388-9.546012. The
predeclared all-gates Boolean remains false on every seed, but the discovery
advantage also holds on every seed. The correct status is **stable breadth-first
baseline; freeze and build from here**.

| training seed | optimized | evaluable | useful | mean SOS | NLL | median effective | unique top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7043 | 803 | 445 | 289 | 0.647153 | 9.542924 | 3.699 | 828 |
| 23 | 791 | 453 | 274 | 0.637558 | 9.546012 | 3.702 | 816 |
| 37 | 787 | 439 | 275 | 0.647350 | 9.539388 | 3.714 | 813 |

## What the model is

The generator remains a recognizable Embedded Topic Model:

- fixed 48-dimensional train-only spectral SGNS word coordinates;
- learned ETM topic coordinates and embedding-space topic-word decoder;
- separate fragment and neutral-loss softmaxes with 50/50 probability mass;
- ordinary multinomial reconstruction from raw intensity pseudo-counts;
- a Gaussian variational latent and analytic standard-normal KL; and
- the existing two-layer ETM document encoder.

Two auditable inference changes adapt ETM to very short spectra:

1. Each observed word is combined with a leave-one-out mean of the other words
   using one learned scalar. The contextual word is compared with ETM's own
   topic vectors, and only its two strongest topic matches are retained.
2. Aggregated top-2 evidence is converted into a bounded, row-centred log offset
   and added to the ordinary ETM posterior mean. Published alpha-entmax 1.5 then
   maps the sampled latent vector to an exactly sparse topic mixture.

In compact form, for normalized spectrum `x`, ETM posterior mean `mu(x)`,
aggregated top-2 evidence `r(x)` and K topics:

```text
mu_routed(x) = mu(x) + center(log(r(x) + 1/K))
z            = mu_routed(x) + sigma(x) * epsilon
theta        = entmax_1.5(z)
prediction   = theta @ beta_ETM
```

The `1/K` pseudocount bounds absent-topic evidence; centring makes uniform
evidence an exact no-op. Routing temperature and strength are fixed at 1.0. The
only additional learned parameter is the context scalar, which finished at
0.1772. The model has 19,278,001 parameters versus 19,278,000 for balanced ETM.

There is no M1 nonlinear router, document gate, Sinkhorn target, positive-NPMI
loss, prototype-separation loss, alternating optimizer or temperature schedule.
ETM and entmax are published components; top-2 contextual posterior evidence is
the small domain adaptation supported by the measured short-spectrum failure.

## Why the result matters

Balanced softmax ETM had 46.72 median effective topics per spectrum and only
166 evaluable / 104 useful motifs. Entmax alone made mixtures sparse but starved
the inventory, leaving only 20 unique top-1 topics and 7 evaluable / 6 useful
motifs. Routing plus entmax supplies the complementary repair:

- median 3.70 effective topics per spectrum;
- median exact support 6 and p95 support 13;
- 828 of 1,000 topics win at least one validation spectrum;
- 538.40 corpus-effective topics;
- no catastrophic duplicate component; and
- 445 evaluable / 289 useful motifs.

The candidate also has a higher median SOS than M1. Its lower mean arises while
evaluating 37 more topics: it has 67 high-SOS, 222 intermediate-SOS and 156
low-SOS topics, compared with M1's 79, 186 and 143. Useful motifs are the high
plus intermediate bands, so the candidate gains breadth mainly through 36 extra
intermediate motifs rather than by inflating a small high-scoring subset.

## Evidence scope and limitations

- Real evidence is one deterministic MSnLib split, K=1000 and three Routing ETM
  training seeds. It demonstrates initialization stability on this split, not
  split stability or independent-dataset generalization.
- Model selection used training plus validation only. Candidate test theta,
  completion, MAG, SOS and result files were not loaded, calculated or inspected.
- M1 is a locked private comparator and donor-ablation source, not the proposed
  publication architecture.
- Tomotopy is an independently trained production comparator, not a teacher.
- SOS and motif counts depend on the frozen membership threshold 0.5,
  leakage-filtered MAG index and compound-balanced validation calculation.
- The M1-relative gates are intentionally conservative. They should remain in
  the audit trail even when the scientific interpretation uses the full metric
  profile.

## What to improve next

Initialization stability is complete, and the one bounded positive-NPMI add-on
failed synthetic promotion. The next work should improve evidence rather than
add more mechanisms:

1. Verify and preserve this checkpoint.
2. Do not tune the NPMI coefficient or add another donor component. Test split
   or external-library robustness if more evidence is needed.
3. Authorize test only after an independent review freezes the method and the
   interpretation of success.

Do not add M1's document gate, Sinkhorn balancing, prototype separation,
alternating optimization or schedule. Additional complexity now carries a real
risk of making a strong explainable result harder to publish.

## Reproduce and verify

The committed result package, implementation and environment hashes are frozen
in `local_results/20260830_routing_etm/checkpoint_manifest.json`.

Verify all committed evidence without training:

```bash
conda run -n ms2lda-neural python \
  scripts/verify_routing_etm_checkpoint.py
```

On the original host, verify the retained large artifacts and immutable input
assets as well:

```bash
conda run -n ms2lda-neural python \
  scripts/verify_routing_etm_checkpoint.py \
  --verify-inputs --verify-local-artifacts --require-external
```

The original replay commands and synthetic decision configs are in
`local_results/20260830_routing_etm/README.md`. The multiseed commands, compact
results, runtime evidence and hashes are in
`local_results/20260830_routing_etm_stability/README.md`.

## Evidence map

- `local_results/20260830_routing_etm/README.md` — authoritative technical
  report, exact results, interpretation and replay commands.
- `local_results/20260830_routing_etm/EXPERIMENT_LOG.md` — chronological
  hypotheses, stopping rules, results and decisions.
- `local_results/20260830_routing_etm/checkpoint_manifest.json` — frozen hashes,
  comparator values, expected metrics and replay locations.
- `local_results/20260830_routing_etm_stability/` — three-seed real stability
  report, compact per-seed evidence and machine-verifiable manifest.
- `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md` — report-level comparison
  and current publication decision.
- `benchmarks/neural_ms2lda/routing_etm.py` — model implementation.
- `scripts/run_routing_etm_campaign.py` — truth-known synthetic runner.
- `scripts/run_routing_etm_real.py` — validation-only real runner.
- `scripts/verify_routing_etm_checkpoint.py` — integrity and consistency checker.
- `scripts/package_routing_etm_stability.py` and
  `scripts/verify_routing_etm_stability.py` — deterministic multiseed packaging
  and verification.
- `LITERATURE_SURVEY.md` and `REFERENCES.bib` — published-method grounding.
- `HANDOFF.md` — historical published-model campaign and failure diagnosis.
- `NEXT_AGENT.md` — rules for any work after this frozen checkpoint.
