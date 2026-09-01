# Principled sparse-ETM campaign

## Decision

Stop the current ETM line and make M1 multiseed stability the next campaign.

The sole promoted formulation was a balanced fixed-SGNS ETM with a Gaussian
posterior and standard-normal KL, `theta=entmax15(z)`, and training
reconstruction mass normalized to each spectrum's number of distinct observed
words. It is recognizably ETM and is mechanically simpler than M1. It does not,
however, pass real MSnLib validation.

At K=1000, the candidate made the median spectrum very sparse (median exact
support 2; median effective topics 1.80), but not the distribution as a whole.
Exact support jumped to 789 at the 75th percentile and 994 at the 95th;
mean effective topics were 140.6. Global usage then collapsed to only 20 unique
top-1 topics, 98 topics active above 0.0005, and 57.0 corpus-effective topics.
Only 7 motifs were chemically evaluable and 6 useful. Completion NLL was
9.577829. The candidate therefore failed the evaluable, useful, and NLL gates.

All model selection in this campaign was validation-only. Candidate test theta,
completion, MAG, SOS, and result artifacts were not loaded, computed, or
inspected.

## Why these interventions

[Sparsemax](https://proceedings.mlr.press/v48/martins16.html) is a projection
onto the probability simplex that produces exact zeros. Lin, Hu and Guo's
[sparsemax topic model](https://doi.org/10.1145/3289600.3290957) applies the
Gaussian-sparsemax idea to topic sparsity, although its full formulation also
uses a relaxed-Wasserstein objective. [1.5-entmax](https://doi.org/10.18653/v1/P19-1146)
offers an exact-zero transform between softmax and sparsemax. The campaign used
the authors' tested `entmax==1.3` implementation.

The first mechanism screen changed only `softmax(z)` to 1.5-entmax or
sparsemax. A separate screen kept softmax but changed the effective
reconstruction mass from arbitrary intensity-derived pseudo-count total to the
number of distinct observed words. The latter preserves within-spectrum
relative intensity weights and uses a directly measured document length rather
than a tuned constant. The two changes were combined only after they showed
complementary independent effects.

No sparse prior, MMD/WAE objective, gate, Sinkhorn term, NPMI term, contextual
routing, or prototype separation was added. In particular, this is a
Gaussian-sparse-transform ETM ablation, not a reproduction of Lin et al.'s full
relaxed-Wasserstein model.

## Synthetic mechanism screen

The screen used a deterministic reconstruction of the documented paired
fragment/loss protocol: 18 planted motifs, 1--3 motifs per spectrum, intensity
pseudo-counts, train-only vocabulary/SGNS, 800 training spectra, 160 validation
spectra, and fixed 120-epoch runs. The historical exploratory simulator source
was not retained. Corpus summary statistics were close, but the new baseline's
recovery was not an exact reproduction of the old CSV. These results are
therefore valid paired comparisons within the auditable reconstructed harness,
not claims of historical metric reproduction.

K=36 results are means over seeds 11/23/37 where `n=3`; the two raw sparse
transforms were rejected after their predeclared seed-11 triage (`n=1`).

| Formulation | n | NLL | beta cosine | theta cosine | median effective | median exact support | mean exact support | active >0.0005 | unique top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax + raw counts (A) | 3 | 6.4710 | 0.3052 | 0.4869 | 3.59 | 36 | 36.0 | 36 | 6.7 |
| 1.5-entmax + raw counts (B1) | 1 | 6.6167 | 0.2503 | 0.4031 | 1.82 | 3 | 2.84 | 5 | 4 |
| sparsemax + raw counts (B2) | 1 | 6.6465 | 0.2574 | 0.3822 | 1.90 | 2 | 2.38 | 5 | 4 |
| softmax + distinct-word mass (C) | 3 | 6.4473 | 0.4176 | 0.5961 | 31.00 | 36 | 36.0 | 36 | 14.0 |
| 1.5-entmax + distinct-word mass (D) | 3 | **6.3863** | **0.4546** | **0.7422** | **1.00** | **1** | 9.50 | 36 | 11.7 |

Raw sparse transforms supplied exact support but harmed recovery and starved
the topic inventory. Distinct-word scaling independently improved likelihood,
beta recovery, theta recovery, and inventory breadth, while making softmax
theta far more diffuse. The combined formulation recovered the benefit of both:
all three seeds were finite, all 36 topics remained active above 0.0005, median
effective/exact support was 1/1, and recovery improved over both controls on
every seed. Its support tail was already bimodal: mean exact support 9.5 and
mean 95th percentile 35.7.

The full seed rows are in `synthetic_by_seed.csv`; the compact aggregation is in
`synthetic_summary.csv`.

## K=128 stress

Seed 11 used the same 18 planted motifs and changed only fitted K.

| Formulation | NLL | beta cosine | theta cosine | median effective | median / mean exact support | p95 support | active >0.0005 | unique top-1 | mean nearest-beta cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax + raw counts | 6.4175 | 0.3696 | 0.5782 | 3.67 | 128 / 128.0 | 128 | 128 | 7 | 0.9637 |
| softmax + distinct-word mass | 6.6158 | 0.4202 | 0.4876 | 127.29 | 128 / 128.0 | 128 | 128 | 14 | 0.9623 |
| 1.5-entmax + distinct-word mass | 6.5786 | **0.4374** | **0.6896** | 3.85 | **15.5 / 62.9** | 128 | 128 | 11 | 0.9766 |

The combination survived high K in beta/theta recovery and median effective
sparsity. It did not preserve uniformly small exact support: almost half of
spectra had support above 3, the 95th percentile reached all 128 topics, and
beta redundancy was high. This warning was carried into real validation rather
than hidden. The exact row is in `high_k_stress.csv`.

## Real MSnLib validation

Only formulation D was promoted. It reused the frozen seed-42 split,
vocabulary, train-only 48D SGNS, K=1000, 0.5 membership threshold,
leakage-filtered MAG index, completion evaluation, and SOS evaluation. Training
used Adam at 0.005 with weight decay 1.2e-6, hidden size 800, batch size 256,
120 epochs, CUDA, and six CPU threads.

| Model | Optimized | Evaluable | Useful | Mean SOS | NLL | Median effective topics | Pass all gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 locked reference | 884 | 408 | 265 | 0.658079 | 8.974140 | 7.20 | Yes |
| canonical ETM | 609 | 130 | 79 | 0.638796 | 8.690730 | 43.79 | No |
| balanced ETM | 911 | 166 | 104 | 0.629819 | 8.766069 | 46.72 | No |
| balanced gated ETM, gamma=2 | 889 | 220 | 138 | 0.645148 | 8.686967 | 65.57 | No |
| **1.5-entmax + distinct-word mass** | **993** | **7** | **6** | **0.663853** | **9.577829** | **1.80** | **No** |

Frozen gate detail:

| Gate | Threshold | Candidate | Outcome |
|---|---:|---:|---|
| optimized | >=840 | 993 | pass |
| evaluable | >=388 | 7 | **fail** |
| useful | >=252 | 6 | **fail** |
| mean SOS | >=0.651498 | 0.663853 | pass |
| completion NLL | <=9.422847 | 9.577829 | **fail** |
| finite/stable | required | yes | pass |
| no catastrophic duplicate component at 0.999 | required | yes | pass |

The apparently strong mean SOS is based on only seven eligible motifs: six in
the 0.6--0.8 band, one below 0.6, and none above 0.8. It does not compensate for
the inventory failure. Membership associated 2,464 validation spectra and
2,436 molecules; held-out compounds were excluded from MAG as required.

### Theta and inventory diagnostics

- Exact support: minimum 1, median 2, mean 295.3, maximum 1000; p75 789,
  p95 994, p99 997; 55.4% of spectra had support <=3.
- Effective topics: median 1.80, mean 140.60.
- Maximum theta: median 0.772; fractions >=0.5/0.3/0.2 were
  0.634/0.683/0.712.
- Inventory: 20 unique top-1 topics, 98 topics active above 0.0005, 39 at or
  above `1/K`, and 56.98 corpus-effective topics. Maximum mean topic use was
  0.1765.
- Beta geometry: mean nearest-topic cosine 0.9764 and maximum cosine 0.9973.
  At 0.95, 971 topics entered duplicate components and the largest component
  contained 941 topics. At the frozen strict 0.999 threshold there was no pair,
  so the formal catastrophic-duplicate gate passed despite severe looser-scale
  redundancy.
- Beta sharpness: median effective words 6,085.7, median maximum word
  probability 0.00295, median top-20 mass 0.0381, and top-word uniqueness
  0.03885.
- Balanced beta remained exact: median fragment mass 0.5 and no extreme
  fragment/loss skew.
- Completion OOV fraction was 0.03136.

This is not the old dense-softmax failure in a corrected form. It is a new
failure mode: hard topic starvation for many documents/topics combined with a
large near-uniform support regime for the remaining documents, severe global
beta similarity, weak chemically evaluable inventory, and degraded completion.

## Runtime and numerical audit

The successful real run trained in 791.3 seconds (13.2 minutes) on the RTX
5070. Deterministic inference measured 32,677 observed and 31,242 full
validation spectra/second. PyTorch peak allocated/reserved CUDA memory was
0.811/1.028 GB; the process peak was 2.839 GB and the minimum sampled system
available memory was 12.875 GB. The model has 19,278,000 parameters.

The first 120-epoch real attempt remained finite but reporting failed closed:
the float32 entmax kernel's row-sum error grew to 0.000478 at K=1000. Every
transform output is now divided by its positive row total. This is a numerical
contract correction, not a scientific hyperparameter; it preserves exact
zeros, ranks, and relative nonzero mass. K=1000 tests and fresh K=36/K=128
rechecks passed, and the real run was restarted from initialization rather than
resuming weights trained under the prior numerics. Both checkpoints and logs
are retained and hashed in `provenance.json`.

## Explicit answers

1. **Is dense softmax the main cause of real ETM diffuseness?** No, not by
   itself. Softmax guarantees dense support, but removing it exposed a bimodal
   support and inventory-collapse problem rather than producing uniformly
   sparse, useful topics.
2. **Does sparsemax/entmax solve it?** No. Raw sparse transforms solved exact
   support on synthetic seed 11 but harmed recovery and starved topics. The
   combined entmax candidate fixed the real median but not the tail, inventory,
   completion, or chemistry.
3. **Does pseudo-count scaling matter?** Yes. It was the strongest independent
   synthetic recovery intervention, improving beta/theta recovery and
   inventory. With softmax it also made theta much more diffuse, so it is not a
   standalone sparsity solution.
4. **Does a sparse prior add anything beyond sparse theta?** Unknown: no sparse
   prior was tested. The bounded ladder did not authorize that larger model
   change before the simpler candidate's real result, so this campaign makes no
   positive or negative causal claim about it.
5. **Which mechanism survives K=128/high-K stress?** Entmax plus distinct-word
   mass survived in recovery and median effective sparsity. Uniform exact
   support and beta distinctness did not survive.
6. **Which formulation was promoted?** Balanced fixed-SGNS Gaussian ETM with
   1.5-entmax theta and distinct-word reconstruction mass; no comparator was
   promoted.
7. **Does real theta support approach M1-like scale?** Only at the median:
   1.80 effective topics and exact support 2 are below M1's median 7.20. The
   p75/p95 supports of 789/994 and mean effective count 140.6 make the overall
   answer no.
8. **Does topic inventory remain broad?** No. It fell to 20 unique top-1, 98
   active, and 57 corpus-effective topics, with high beta redundancy.
9. **Does chemistry reach the frozen gates?** No. Optimized and mean-SOS gates
   passed, but evaluable was 7/388 required, useful was 6/252, and NLL was
   9.577829/9.422847 maximum.
10. **Is the candidate recognizably ETM?** Yes. It retains the Gaussian
    variational encoder, standard-normal KL, embedding decoder, and topic-word
    mixture; only the published simplex transform and evidence-mass scaling
    change.
11. **Is it scientifically simpler than M1?** Mechanistically yes, but its
    simplicity does not overcome the failed validation evidence.
12. **What exact failure remains?** Bimodal document support, global topic
    starvation, high beta similarity, only seven evaluable/six useful motifs,
    and completion NLL above threshold.
13. **Is further ETM work justified?** Not as the next campaign. A sparse prior
    remains logically untested, but it would be a materially larger neural
    topic formulation and lacks evidence strong enough to displace the validated
    incumbent now.
14. **Should M1 multiseed stability be next?** Yes. This campaign supplies the
    stopping evidence requested before starting it.

## Quality control

- Focused sparse-transform/simulator/isolation tests: 14 passed.
- Full neural suite: 75 passed.
- Repository CI production suite, using the exclusions defined in
  `.github/workflows/neural-ms2lda.yml`: 87 passed, with the two existing
  empty-document NumPy warnings.
- Black check: passed on all changed Python files.
- Ruff: passed on all changed Python files.
- JSON/CSV parsing, `git diff --check`, exact-simplex/non-negativity/zero-support,
  finite-gradient, deterministic-inference, validation-isolation and staged
  binary-size checks: passed.

For completeness, an initial unfiltered `pytest -q tests` invocation ran four
legacy files that CI explicitly excludes (`test_callbacks.py`,
`test_generate_corpus.py`, `test_integration.py`, and `test_utils.py`). It
reported 134 passes and 33 failures in untouched interfaces and download tests.
Those failures are not regressions from this campaign and are not represented
as a passing suite. The download test also created two ignored 5.7 GB temporary
model directories inside the checkout; both newly created directories were
verified untracked and removed. The preserved shared Spec2Vec/MAG assets were
not changed.

## Evidence map

- `EXPERIMENT_LOG.md`: predeclarations, stopping rules, results, numerical
  failure, and decisions in chronological order.
- `synthetic_summary.csv`, `synthetic_by_seed.csv`, `high_k_stress.csv`: compact
  synthetic evidence.
- `comparison.csv`: real baseline/candidate diagnostics and gate booleans.
- `metrics.json`, `chemical_scores.csv`, `theta_support_summary.csv`,
  `duplicate_component_summary.json`, `fragment_mass_summary.json`, and
  `top_words.csv`: promoted candidate validation outputs.
- `validation_access_audit.json`: explicit train/validation-only input view and
  chemistry audit.
- `provenance.json`: local paths, byte sizes, SHA-256 values, and the preserved
  pre-correction failure.
- `configs/`: every decision-bearing synthetic config plus the real config.

No `.pt`, `.npy`, FAISS index, database, raw MSnLib asset, or candidate test
artifact is committed here.
