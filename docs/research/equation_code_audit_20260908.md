# Current-model equation and implementation audit — 8 September 2026

## Outcome and scope

This follow-up to the [scientific code-quality review](code_quality_review_20260908.md)
checks the current whole-spectrum model in both directions: each numbered or
meaningful unnumbered mathematical statement has an implementation, and each
implemented scientific operation has an explicit interpretation in the report.
The selected variant remains `ReducedContextualETM` with
`variant="reduced_document_context"`. Tomotopy remains the primary comparator;
plain ETM is the enhancement reference.

The decoder, contextual evidence, shifted Gaussian posterior, reparameterization
and entmax mapping agree with the stated model. The necessary corrections are
descriptions of numerical safeguards, completion preparation and chemical
evaluation, plus rejection of malformed inputs to synthetic matching. No model
architecture, parameter initialization, state-dictionary key, probability
kernel, saved weight or historical result was changed by this audit.

The implementation remains function-first. Small PyTorch modules register
learned state; direct tensor functions implement the mathematics. A further
class hierarchy or framework would increase the reader's burden. Historical
variant switches remain separate from the chosen scientific specification.

## Complete mathematical reading map

Paths below are relative to `benchmarks/neural_ms2lda/` unless qualified.
Equation labels identify the current manuscript, not the archived leave-one-out
description. `B` is batch size; `D`, `V`, `K`, `R` and `E` denote spectra,
vocabulary types, fitted topics, planted motifs and embedding dimensions.

| Statement or equation | Exact implementation and convention |
| --- | --- |
| Peak intensity normalization and filtering | `spectra._clean_peaks`: finite positive m/z and nonnegative intensity; divide by the full spectrum maximum; apply the protocol m/z/intensity limits, peak cap and minimum count. |
| Fragment/loss construction and pseudo-counts | `spectra._spectral_word`, `_clean_peaks`: round m/z to two decimal places, not necessarily two printed trailing digits; each applicable channel gets `rint(100 * I)` copies. `data._matrix` accumulates copies in fixed vocabulary columns. |
| `N_d = sum_w c_dw`; `x_dw = c_dw / N_d` | `topic_model_training.dense_normalized`: integer-count rows have either positive mass at least one or zero mass. Positive rows are normalized; zero rows stay zero. Raw counts, not `x`, remain likelihood targets. |
| Chemical grouping and 70/10/20 allocation | `spectra._structure_keys`, `assign_scaffold_splits`, `audit_split_disjointness`: connectivity verification, nonchiral cyclic scaffold groups, intact acyclic connectivity groups, deterministic allocation by group size and target deficit. |
| Training-only vocabulary and SGNS | `spectra.build_training_vocabulary`; `data._positive_pairs`, `_Sgns.forward`, `train_token_features`: within-spectrum count-weighted positive pairs, frequency-power negative sampling, positive/negative logistic terms. The source/context tables are averaged and unit-normalized. `etm_baselines.load_sgns_embeddings` removes the stored channel flags and re-normalizes the 48 SGNS coordinates used by ETM. |
| `eq:base-beta`: published ETM emissions | `etm_baselines.CanonicalETM.topic_word_distribution`: `alphas(rho)` is `V x K`; softmax over words followed by transpose gives `K x V` topic rows. |
| `eq:base-generator`: Gaussian-to-softmax topic mixture and additive emissions | `CanonicalETM.document_topic_mixture` and reconstruction/completion functions: one Gaussian latent row maps to a softmax mixture, and word probabilities are `theta @ beta`. No product-of-experts decoder is used. |
| `eq:base-posterior`: two-layer MLP Gaussian posterior | `CanonicalETM.posterior` and the inherited ETM layers in `ContextualSparseETM`: two affine/ReLU layers, then independent mean/log-variance heads, each `B x K`. |
| LDA Dirichlet priors and token assignments | `tomotopy.train_tomotopy` constructs the pinned Tomotopy LDA sampler with the declared initial concentrations. Its mixture-concentration vector is optimized every ten iterations; it is not fixed at its initial value. `_infer_theta` uses the learned prior when observed words are absent. |
| `eq:beta`: half-mass spectral channels | `contextual_sparse_etm.channel_balanced_topic_word_distribution`: raw word/topic inner products, separate word softmaxes in fragments and losses, each multiplied by one half. No learned channel gate exists. |
| Channel-generative interpretation and constant `-log(2)` | The preceding decoder selects the channel with fixed probability one half and the word conditionally within that channel. The half factor is constant per observed pseudo-token; raw-count weighting does not equalize channel gradient contributions. |
| Unit normalization | `contextual_sparse_etm.unit_normalize_rows`: divide by `max(norm(v), 1e-12)`. The same word/topic directions enter contextual matching; the decoder still uses raw topic vectors. |
| `eq:document-context`: full-spectrum mean and contextual words | `contextual_reductions.pooled_evidence(context="document", routing=2, score_scale=1)`: `s = x @ rho_hat`; for each observed word, `h = normalize(rho_hat_w + context_scale * s_d)`. The scored word is included in the mean. |
| `eq:document-evidence`: cosine, top-two local softmax, pooled `r` | The same function scores unit directions, selects two topics per observed word and pools by `x_dw` with `index_add`. It re-normalizes accumulated row mass; zero-input rows get uniform `r`. The local attention is not a claimed exact latent-topic posterior. |
| `eq:posterior-offset`: centered logarithmic mean correction | `contextual_sparse_etm.centered_log_evidence_offset`: centered `log(r + 1/K)`, equivalent to centered `log(1 + K*r)`. A detached roundoff correction preserves both the exact-zero uniform value and its nonzero analytic derivative. |
| Complete shifted posterior | `contextual_reductions.ReducedContextualETM.posterior`: `m = MLP_mean(x) + offset(r)`; log variance remains the MLP head. The corrected mean enters both reconstruction and KL. |
| Gaussian reparameterization | `contextual_sparse_etm.reparameterized_gaussian`: `z = m + exp(ell/2) * epsilon`, with one independent standard-Gaussian sample per spectrum per step. `ell` is log variance, not log standard deviation. |
| `eq:entmax`: sparse simplex map | `contextual_sparse_etm.entmax15_document_mixture`: the published 1.5-entmax map, `theta_k = [z_k/2 - tau(z)]_+^2`, with a row-sum correction that preserves exact zeros. No spectrum-level top-two cap is imposed. |
| `eq:elbo`: raw-count variational target | `topic_model_training.sparse_reconstruction_loss(scaling="raw_counts")` sums observed counts times negative log mixture probability within spectra, then averages spectra. `scripts/run_minimal_etm.fit` adds mean KL with coefficient one. The executable probability floor and Adam weight decay are described below. |
| `eq:kl`: Gaussian KL to a standard-normal prior | `contextual_sparse_etm.diagonal_gaussian_kl`: `0.5 * sum_k(m_k^2 + exp(ell_k) - 1 - ell_k)`. No simplex density, inverse entmax map or entmax Jacobian is asserted. |
| Deterministic inference | `model_inference.infer_document_topics`: evaluation mode, `sample=False`, giving `entmax(m)` rather than `E_q[entmax(z)]`. `reproducibility.normalize_probability_rows` performs export normalization. |
| Planted synthetic truth | `synthetic_msms._true_theta` normalizes all labeled signal counts, excluding background/noise; `_true_beta` normalizes labeled training counts within the retained vocabulary. Neither truth array enters optimization. |
| One-to-one word recovery and aligned mixture recovery | `synthetic_msms.matched_truth_metrics`: Hungarian maximization of total word cosine; average over all `R` motifs; align selected fitted columns into truth order before per-spectrum cosine. Zero/numerically zero aligned mass scores zero. Unmatched fitted mass is not directly penalized. |
| Recovered motifs | The same matching function counts matched word cosines `>= 0.50`, out of all `R` planted motifs. This is a reporting threshold, not a proof of identifiability. |
| Physical-peak completion split | `spectra.completion_document`, `renormalize_peak_groups`, `data.prepare_data` for real data; `synthetic_msms._completion_views` for synthetic data. Paired fragment/loss words stay together, but observed count preparation differs as documented below. |
| `eq:completion`: token-weighted completion NLL | `model_evaluation.completion_metrics`: sum negative log probabilities weighted by withheld counts, divided by in-vocabulary withheld count mass. The logarithm uses the floor below. Eligibility depends on withheld mass, not observed mass. OOV counts are reported separately. |
| `eq:effective`: entropy-effective mixture size | `model_evaluation.theta_support_diagnostics`: `exp(-sum(theta * log(theta)))`, with zero contributions handled explicitly and positive logarithm arguments floored. Take the median across full-spectrum validation rows within each fit before averaging fits. |
| Exact support and dominant-topic count | `theta_support_diagnostics` counts `theta > 0` with no post-hoc cutoff. `mixture_distribution_summary` / inventory diagnostics count distinct `argmax` topics; NumPy resolves exact dominant-topic ties to the lowest index. This tie rule does not imply that PyTorch top-two routing has the same tie rule. |
| MAG consensus `A_k` | `mag.topic_spectra`, `chemical._mag_matches`, `_annotate_topics`, `mag.consensus_fingerprint`: top-word pseudo-spectra, excluded-connectivity retrieval library, legacy masked-motif profile clustering, refinement, then MACCS bits with cluster frequency at least 0.8. |
| `eq:sos`: mean fingerprint containment over `C_k` | `chemical._associated_record_indices`, `_topic_scores`, `_calculate_sos`: one dominant topic per spectrum; deduplicate connectivity groups within a topic; average `intersection(annotation,molecule) / annotation_bits` across valid associated molecules. |
| Evaluable, useful and conditional mean SOS | `_topic_scores` requires an optimized motif, available consensus and nonempty valid associated molecules. An available all-zero consensus scores zero. Usefulness is SOS `>= 0.6`; the corpus mean is unweighted over evaluable motifs only. |
| Means and error bars | The report generator aggregates independent fit-level values with arithmetic mean and sample SD, not standard error, a confidence interval, or a per-spectrum SD. Within-fit medians remain medians before aggregation. |

## Numerical and preprocessing distinctions

### Probability flooring is an executable approximation

The mathematical generator defines a multinomial likelihood. Training and
completion evaluate `log(max(p, 1e-12))`. Below the floor, the reconstruction
term has zero derivative with respect to `p`; the floored scalar is not an
exact multinomial log likelihood or a guaranteed exact lower bound. The ideal
ELBO remains the modeling target. The optimizer additionally uses Adam's
coupled weight decay of `1.2e-6`, a numerical training regularizer rather than
an auxiliary motif-quality or contextual objective. Neither convention was
changed in this audit.

A read-only CUDA inference check reloaded the three selected real checkpoints,
used only validation observed/count files, and scored the saved exported
`beta` matrices with and without the floor. All runs have 95,178 scored nonzero
document-word positions and 1,277,983 withheld in-vocabulary pseudo-tokens.
The run identifiers below are for reproducibility, not separate conditions.

| Run | Minimum scored probability | Floored positions | Floored token copies | Floored NLL | Unfloored NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 3.05751e-18 | 7 | 14 | 9.4997548373 | 9.4998250114 |
| 23 | 1.76848e-16 | 14 | 141 | 9.5454196812 | 9.5456213912 |
| 42, successful retry | 7.90104e-17 | 9 | 28 | 9.5283875752 | 9.5284393096 |

The changes are approximately 0.0000702, 0.0002017 and 0.0000517 nats per
pseudo-token. This does not change the reported comparison at its displayed
precision. Saved metrics were not overwritten with these recomputations.
These are final-checkpoint completion checks; they do not establish how often
the floor was active at earlier optimization steps or for other models.

Checkpoint SHA-256 values, in the same order:

```text
96e2b9063efca8fa7891c3837056121ea5a600b93e0e122414a3c57d4207ae76
41de7fc3c88d67fbd33c403a12e2f6d23c1537970cd622acf4ace1b5997a1f8f
d561a122991c9f29e13c30b891d76f401a7dcd554f9acaeeda842d831faafe8c
```

### Empty observed spectra remain scoreable

The selected real validation view has one empty observed row (zero-based
index 3696) with eight in-vocabulary withheld tokens. It is encoded as `x=0`;
uniform contextual evidence contributes zero offset, leaving the MLP's learned
zero-input posterior. This is not necessarily the prior mixture. Tomotopy's
empty-input rule uses its learned normalized Dirichlet prior.

A different row (index 598) has zero in-vocabulary withheld mass and 84 observed
tokens; only that row is excluded from completion scoring. Hence there are
3,888 eligible completion spectra but 3,889 full-spectrum chemical associations.
Training and full validation have no zero-count rows.

### Real and synthetic observed-count preparation differ

Real completion starts from retained physical peaks, re-max-normalizes observed
intensities and then re-discretizes them. Synthetic completion splits already
generated integer counts unchanged. Both later normalize the encoder input and
keep fragment/loss pairs together. Synthetic counts can retain a rounding
dependence on the full-spectrum maximum; they should not be described as having
undergone the real-data observed-intensity reconstruction. Existing simulation
inputs and results remain intact.

### MAG's named control is not a direct spectral-cosine cutoff

The inherited `annotation_refined.agglomerative_clustering` receives a matrix
whose rows are retrieved-hit profiles against masked versions of a motif.
It applies complete-linkage agglomeration with the default Euclidean metric to
`1 - profile`, at distance threshold `1 - cosine_similarity`. Therefore the
recipe's legacy control value 0.9 means distance threshold 0.1 on those profile
rows. It does not require direct pairwise spectral cosine at least 0.9.
`hit_clustering(criterium="best")` retains the cluster containing the leading
retrieval hit. This shared inherited algorithm is unchanged for all comparators.

Available but all-zero consensus fingerprints and unavailable consensus are
also different: the former receives SOS zero, while the latter cannot define
the annotation containment calculation and makes the motif unevaluable.

## Bounded reachability proof for the earlier gradient correction

The earlier review fixed the derivative of the centered log offset at exactly
uniform evidence, without changing its forward value. For a nonempty spectrum
with `n_d` observed vocabulary types, top-two routing can give positive mass to
at most `2*n_d` topics. If `2*n_d < K`, at least one evidence entry is exactly
zero. Since a uniform probability vector has every entry positive, the old
uniform-only branch is then unreachable regardless of the learned embeddings
or scalar context strength.

| Training inputs | Empty rows | Maximum observed word types | Maximum evidence support | Fitted K |
| --- | ---: | ---: | ---: | ---: |
| Shared real training matrix | 0 | 494 | 988 | 1,000 |
| Synthetic realization 11 | 0 | 51 | 102 | 128 |
| Synthetic realization 23 | 0 | 50 | 100 | 128 |
| Synthetic realization 37 | 0 | 52 | 104 | 128 |

This proves that the uniform-only gradient correction cannot alter any of the
three selected real training trajectories or any of the three selected
synthetic `K=128` trajectories. The input hashes used in this check are:

```text
real:         b4808f341864ba1df246a5093992f52397c2f96e0b48b23245b9b17e30aa5b52
synthetic 11: eedeb9a9244b1cd05d18cd95ab70c7ca11bab84f3cfc3992b1e112e65b4b2d6b
synthetic 23: 635b34b6606cdc6b7702019533bc5f821ad89af8dc5132397f3c717b3325dec6
synthetic 37: cf6990f3457e1d9416791386fa0ab67eb7998d5d15c1e050d64ac5bfc791b8c0
```

The bound does not prove unreachability for `K=36`, does not establish all-input
equivalence, and is not an independent post-selection performance evaluation.
The three `K=36` runs retain the historical-source exact-trajectory caveat;
there is no observed output defect here requiring their replacement.

## Regression checks and preservation

`benchmarks/neural_ms2lda/tests/test_current_model_equations.py` contains 40
CPU-only research checks. The selected
model reference explicitly constructs full-spectrum evidence, both MLP layers,
the corrected Gaussian mean/log variance, analytic KL, sampled and mean-only
entmax mixtures, the balanced decoder and the raw-count reconstruction term.
An independent threshold-bisection calculation checks entmax values. A separate
test derives its smooth-support Jacobian as `diag(s) - s*s.T/sum(s)` for
`s = sqrt(theta)`, including zero derivatives on inactive components and along
the common-shift direction. It does not differentiate the bisection reference
or assert differentiability at support/routing changes.

Additional cases cover floored probability values/gradients, token weighting,
zero-observed completion eligibility, real/synthetic count preparation,
consensus availability, matching shape/finite/mass boundaries,
`K < R` rejection, and the top-two support bound. Shared helper docstrings now
distinguish current equation labels from archived leave-one-out labels.

The separate production test `tests/test_mag_profile_clustering.py` verifies
actual MAG clustering on a tiny numeric fixture. Importing MAG needs the full
production dependency stack, so that test runs in the existing production CI
job rather than adding optional skips or heavy dependencies to focused neural
checks. No retrieval library or fitted annotation model is needed by the test.

The complete matching-result dictionaries for all six selected synthetic fits
were recomputed before and after the new guards, using the original function
from commit `3fb579d` and the same saved arrays. They are exactly equal. No
renormalization, score clipping, tie-rule change or new metric was introduced
by those guards.

An executable-AST comparison against that commit, ignoring docstrings and
comments, also confirms that the four edited model/loss/evaluation/chemistry
modules are unchanged. In the synthetic module, the original executable AST
is unchanged after the new matching input guards. These checks concern this
bounded audit, not every unrelated change made elsewhere in the repository.

The 40 research checks and the standalone production MAG check all passed with
`NUMBA_DISABLE_JIT=1`; importing inherited MAG produced only the two already-known
Lark deprecation warnings. Black and research-config Ruff pass on the seven
Python files changed by this audit. No full model fitting,
reserved test-set access, result replacement or external publication was
performed by this scientific-code audit. Later comparator repetitions are
separate, explicitly recorded experiments.
