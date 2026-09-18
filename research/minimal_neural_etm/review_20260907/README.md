# Independent model-form review, 7--8 September 2026

## Final morning recommendation, 8 September

Keep the published ETM MLP Gaussian recognition backbone and use **whole-spectrum
mean context instead of leave-one-out context** (`reduced_document_context`).
Three paired seeds preserve useful-motif breadth (390.7 versus 389.7 on average),
slightly improve NLL (9.5245 versus 9.5397), and preserve compactness (3.72 versus
3.76 effective topics). No claim of chemical superiority, global minimality or
new unbiased test confirmation follows. The full report records the trade-offs.

All six detached overnight queues have stopped. Phase 3 has 87 successful
synthetic fits, 20 successful real fits with complete chemistry, and one real
numerical failure. Across all three new phases there are **166 successful fits
and one failed configuration**, in addition to 53 audited older records.
The earlier interrupted document-context execution is separately preserved,
not counted as another independent seed. Nothing further is training.

The one-layer width-800 MLP retains chemistry (400.7 useful) at 0.232 nats higher
mean NLL, or approximately 26.1% higher perplexity, while removing only 3.3% of
parameters. Width 100 removes 87.7% of parameters with 377 versus 394 useful
motifs and approximately 16.1% higher perplexity in its one real seed. It is
promising, not rejected for having 790 winners; future paired repeats are
needed before promotion. Full-document plus shallow-800 loses useful breadth
(353), so untested combinations are not assumed to work.

Entmax has direct support for compactness: restoring softmax yields 395 useful
motifs but 47.58 effective topics instead of 3.72. Context removal gives 366
useful motifs and weaker synthetic recovery; that shows benefit, not absolute
necessity. Fixed-c and hard-top-1 forms remain plausible one-seed trade-offs.
Attention-only yields 341.3 useful motifs on average with better NLL and far
fewer weights, but is not the primary candidate under the latest published-ETM
lineage preference. Its replaceable encoder shell remains a tested research
boundary for future DreaMS work, not an implemented foundation-model study.

The linear-offset real fit terminates with a non-finite training loss after
the completed epoch-80 checkpoint and before finishing 120 epochs. No final
score exists and no numerical cause beyond that error was instrumented.
It was not retried or retuned. See `evidence/within_model/failed_runs.json` for
the log/config/checkpoint hashes; absent values are not zeros. All 20 completed
real chemical evaluations have zero MAG exceptions, exclude held-out compounds
from the annotation library and use the same 3,889 spectrum associations.
Old flags are unchanged in the machine-readable audit, but do not drive the
recommendation: the document-context run with 799 winners is not a scientific
failure at 800. All inclusive SOS cutoffs remain reported.

The source report now distinguishes this recommendation from the historical
leave-one-out model's sealed test results. No production integration, new test
access, DreaMS installation, commit or push is performed. The experiment
protocol below is chronological; later decisions supersede earlier ones
without erasing them.

## Question and bounded protocol

Can a simpler neural extension of published ETM retain both chemically useful
motif breadth and compact spectrum mixtures? Parameter count alone is not our
definition of simplicity: the number of computational mechanisms, special
constants, and inference-state requirements also matters.

The existing uncommitted BatchNorm/prior study and the sealed Contextual Sparse
ETM results are inputs to this review. Their artifacts are preserved. The review
uses the canonical `ms2lda-neural` environment and native Linux RTX 5070.

**Phases 1/2 result (superseded as a final decision by the completed phase 3):**
retain Contextual Sparse ETM after the broader-family screen. No new candidate
passed the original full real-validation replacement gate. Those phases added 54 synthetic
fits and five real-validation fits (59 total), in addition to auditing the prior
53 records. The strongest simple follow-on synthetically, ProdLDA plus entmax,
does not retain useful motif breadth on real data. See the final decision and
verification record below; the staged protocol is preserved in order.
The user-requested within-model review and subsequent clarification to treat
thresholds as reference points are recorded in [the phase-3 protocol](within_model_protocol.md).
The subsequent brief accepts MLP removal and asks for a future DreaMS/direct-
spectrum encoder boundary, not a DreaMS implementation now. That boundary is
implemented and compatibility-tested in `benchmarks/neural_ms2lda/attention_etm.py`;
the full-validation audit is `evidence/within_model/attention_interface_audit_seed42.json`.
The no-MLP posterior remains a learned nonlinear attention encoder with global
diagonal Gaussian variance. The current ETM token decoder is unchanged; a
fully tokenizer-free likelihood remains a separate future scientific task.
The latest clarification revisits MLP removal: retain an MLP-based model as
the main publication candidate while completing the capacity comparisons;
attention-only remains an ablation, not a production replacement. The report
distinguishes evidence for keeping an MLP from novelty claims: the MLP is
inherited from published ETM, while even its removal does not reduce this
model to classical LDA or ordinary NMF. Both preference changes are preserved
in the phase-3 protocol rather than retrospectively rewriting the study.
The final selection priority is a conventional, published ETM backbone with
few justified adaptations. Parameter count is secondary to that lineage;
the original MLP Gaussian recognition structure is retained as the base.

Two hypotheses are fixed before the new runs:

1. **BatchNorm plus entmax, without context or channel balancing.** The prior
   study showed that BatchNorm activates useful topics but can leave very dense
   mixtures. Replace its softmax with published 1.5-entmax, retaining the
   standard-normal latent prior. Test learned and fixed normalization scales as
   two explicitly labelled recipes, not as extra architecture ingredients.
   Recompute final normalization moments on training data only, as the existing
   implementation requires for reliable frozen inference.
2. **Remove channel balancing from the current model.** Keep its contextual
   posterior and entmax, but restore ETM's single vocabulary softmax decoder.
   The published comparison establishes what balancing does in dense ETM; it
   does not establish that balancing is necessary in the complete model.

The published sources are Dieng et al. (TACL 2020,
https://aclanthology.org/2020.tacl-1.29/), Srivastava and Sutton (ICLR 2017,
https://arxiv.org/abs/1703.01488), and Peters et al. (ACL 2019,
https://aclanthology.org/P19-1146/). These candidates are explicit ETM adaptations,
not claimed reproductions of AVITM or a published BatchNorm-entmax model.

All training uses the existing matrices and train-only SGNS coordinates, 800
hidden units, raw pseudo-count reconstruction plus Gaussian KL, 120 epochs,
batch size 200, learning rate 0.005, Adam beta1 0.9, and weight decay 1.2e-6.
The current Contextual Sparse ETM is rerun through the same runner for paired
synthetic controls. No loss-weight or concentration search is performed.

The stage order is:

- Seed 11, 18 planted motifs, K=36 and K=128. A candidate must remain finite,
  avoid catastrophic topic duplication, have median effective topics <=5,
  and be within 0.05 beta recovery and 0.10 theta recovery of the paired current
  model at both K values. A failed candidate stops here.
- Confirm survivors and current controls on seeds 23 and 37 at both K values.
  Apply the same recovery/sparsity criteria to the three-seed means, and require
  each run to remain finite without catastrophic duplication. At K=128 the
  candidate's mean recovered-motif count must retain at least 90% of the
  current model's mean.
- At most two scientifically distinct survivors advance to MSnLib validation,
  K=1000, training seed 7043. Use the existing sealed validation view and the
  dominant-topic chemical association rule. Prefer the simplest candidate
  retaining at least 95% of current evaluable/useful counts, mean SOS no worse
  by more than 0.02, NLL no worse by more than 2%, median effective topics <=5,
  at least 800 validation topic winners, and no catastrophic duplication.
  These are development decision tolerances, not statistical significance tests.
- Confirm a proposed replacement on two further training seeds before
  recommending it as the primary model. Otherwise keep it as a research control.

This is validation-based model development on a dataset whose earlier test
results have already been reported. New candidates never load test matrices or
produce new test scores. Historical test results remain explicitly attributed
to the frozen historical model. Same-split seeds are not independent datasets.

All new runs use fresh output directories under
`output/benchmarks/minimal_neural_20260907`. Results, input/source hashes,
checkpoints and failures are retained. The report will document negative
results and distinguish demonstrated minimality from an unproved global claim.

## Review findings and results

### Synthetic decision, recorded before real-data training

All 24 synthetic runs completed and passed the declared mean gates. At K=128,
the current model recovered 16.67 of 18 motifs on average, compared with 15.67
for learned-scale BatchNorm-entmax, 15.33 for fixed-scale BatchNorm-entmax, and
16.33 after removing channel balancing. No run met the repository's strict
catastrophic-duplication definition (a connected component covering at least
half the topics at cosine >=0.999). This does not mean all topics are distinct.

Advance **fixed-scale BatchNorm-entmax** and **unbalanced contextual ETM**.
Fixed-scale normalization is closer to the cited published recipe, removes
learned normalization scales, and has better mean beta/theta recovery and NLL
than learned-scale normalization at both K values. The slightly lower high-K
recovered count still passes the predeclared retention threshold. The other
BatchNorm recipe is retained as synthetic evidence, not promoted to an extra
real-data trial.

The runner uses batch size 200, whereas the historical paper's real-data
reference uses 256. Before any new real result is inspected, add a **paired
current-model run at batch size 200**, same initialization and shuffle seeds,
to isolate architecture from this optimization difference. Keep the frozen
256-batch validation reference visible and require any replacement to pass
the stated retention criteria against both current-model references. This
strengthens the gate; it is not a post-result threshold relaxation.

### Reporting and validation plan

The requested existing LaTeX/PDF manuscript remains the only report surface.
Its abstract is the technical summary; Methods owns model and metric
definitions; the new validation-review Results section owns simplification
evidence; Discussion and Limitations own the recommendation, uncertainty and
next evidence. Existing historical sections and frozen values are preserved.
Use exact multi-metric tables (not an aggregate rank or underpowered scatter)
for the eight synthetic group summaries and real validation comparisons.
Tables use neutral typography, explicit split/seed/batch labels, adjacent
interpretation, and full-width layout verified in the final PDF. Evidence,
code/input hashes, and numerical QA remain linked from supporting source notes.

## User-requested expansion beyond the initial ETM ablations

After the initial synthetic screen and the first real BatchNorm-entmax result,
the user explicitly requested a broader search and suggested the official Pyro
ProdLDA tutorial. The initial two-survivor limit applies to that initial phase,
not to this newly requested phase. No initial thresholds are relaxed.

Before any expanded training, the following separate exploratory protocol is
recorded. Outputs go to `output/benchmarks/published_neural_20260907` and a
separate evidence subdirectory. This phase tests published families rather than
only changing the preceding agent's ETM model.

- **Pyro-reference ProdLDA:** port the tutorial's two 100-unit softplus layers,
  0.2 encoder/mixture dropout, affine-free Gaussian-head and decoder BatchNorm,
  standard-normal latent prior, and product-of-experts softmax decoder into
  the existing PyTorch environment. Analytic Gaussian KL plus one reconstruction
  sample is the same optimization objective as its mean-field ELBO, up to the
  count-only multinomial constant and batch reduction. This is not an executed
  Pyro SVI run or an exact reproduction of the original authors' TensorFlow
  recipe. Source: https://pyro.ai/examples/prodlda.html.
- **Direct-Dirichlet PoE:** the JMLR DVAE architecture with one 100-unit ReLU
  layer, 0.25 encoder dropout, unit-scale/learnable-bias posterior and decoder
  BatchNorm, softplus concentration floored at 1e-5, and a PoE decoder. Use
  PyTorch's implicit reparameterization and analytic Dirichlet KL, as in the
  authors' implicit-gradient comparison, not their principal RSVI estimator.
  Use per-topic prior 0.02 and 100-epoch linear KL warmup. This is an explicit
  port/protocol adaptation, not a claimed reproduction of the paper's scores.
- **Direct-Dirichlet neural LDA:** keep that neural Dirichlet inference network
  but restore the classical additive categorical-mixture decoder with a free
  row-softmax topic matrix. The generative model is LDA with point-estimated
  topics; the neural enhancement is amortized posterior inference.
- **Direct-Dirichlet ETM:** replace only the preceding free topic matrix by
  ETM's fixed training-SGNS embedding decoder. This tests whether published
  embedding sharing suffices without context, entmax, or channel constraints.

The Dirichlet source is Burkhardt and Kramer (JMLR 2019),
https://jmlr.org/papers/v20/18-569.html, especially sections 4.4--5.3, and their
`nvdm_dirichlet_implicitGradients.py` implementation. Prior 0.02 is per topic,
so its total concentration grows with K; a sparse prior is not assumed to imply
a sparse posterior mean. No extra sparsity loss is introduced.

All four use raw input/count reconstruction, Adam lr=0.001/beta1=0.9, no weight
decay, batch 200, 120 epochs, training-only EMA BatchNorm statistics frozen at
inference, and seeds 11/23/37 at K=36/128 on the existing synthetic inputs. These
are fixed published-inspired recipes, not equal-capacity replicas of the
800-unit ETM controls. No per-family hyperparameter sweep is conducted.

The same recovery/compactness/duplication gates describe potential replacements.
Because the synthetic generator is additive, its recovery criterion alone does
not fairly rank PoE likelihood families. Therefore the published ProdLDA
baseline receives one K=1000 MSnLib validation run regardless of its synthetic
gate; at most one further distinct Dirichlet candidate advances if it passes.
All model-selection and replacement gates on real validation stay unchanged,
including the requirement for two additional seeds before promotion.

For PoE models, score completion with the actual frozen decoder, never
`theta @ beta`. Export a topic as the decoder's conditional distribution at a
one-hot topic vector, including frozen BatchNorm. These are conditional
prototypes, not additive LDA emissions; their biological comparability and
possible over-sharpening are limitations to inspect, not silently ignore.
For direct-Dirichlet additive models, normalized concentrations are the exact
variational mean and yield the variational one-token predictive distribution.
No candidate-test files are accessed in either phase.

### Expanded screen outcome and one follow-on hypothesis

All 24 published-family synthetic runs completed. ProdLDA recovered all 18
planted topics at K=128 in each seed, but averaged 59.46 effective topics per
spectrum; at K=36 the respective means were 16 recovered and 15.49 effective.
The direct-Dirichlet variants were also dense (roughly 104--111 effective
topics at K=128); the additive variants recovered no topics at beta cosine
0.50 under this fixed budget. These are failures of these recipes to meet the
replacement target, not impossibility results for their model families.

Before any further candidate is trained, add exactly one natural follow-on:
**Pyro-reference ProdLDA + 1.5-entmax**, replacing only the Gaussian-latent
softmax link by entmax in training and inference. Keep the same prior, encoder,
decoder, BatchNorm, dropout, loss, and optimizer. This changes the induced
simplex prior, not the Gaussian latent ELBO. There is no contextual routing,
embedding pretraining, channel balance or auxiliary objective. The motivation
is the observed combination of strong prototype recovery and dense mixtures,
not an unreported hyperparameter sweep.

Run this one enhancement on all six synthetic seed/K combinations using the
same protocol. Apply the original synthetic gate against paired current ETM;
advance it to one real-validation run only if it passes both K levels. Stock
ProdLDA remains the unconditional real published control. No direct-Dirichlet
candidate advances. This explicitly extends the earlier four-recipe screen;
all its negative runs remain in the evidence. Any real passing candidate still
needs two further real seeds, and no candidate is evaluated on test data.

The six follow-on runs passed the recorded synthetic gates: mean beta/theta
recovery 0.804/0.933 and 17.67 recovered motifs at K=36 (effective topics 1.96);
0.834/0.923 and 18 recovered motifs at K=128 (effective topics 4.34). Advance
ProdLDA-entmax to real validation. Stock ProdLDA and this enhancement may train
concurrently on the GPU; their wall times are recorded but are not performance
comparisons. The fixed initial real comparison is also complete: the paired
current model yields 675 evaluable/394 useful motifs, whereas fixed-BN-entmax
yields 67/42 and removal of channel balancing yields 524/322. Neither initial
candidate passes the joint real gate.

## Phases 1/2 decision and publication framing

| New real-validation fit | Evaluable | Useful | Mean SOS | NLL | Median effective topics | Winners |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current model, paired batch-200 control | 675 | 394 | 0.623 | 9.531 | 3.72 | 829 |
| ETM + fixed BN + entmax | 67 | 42 | 0.627 | 9.146 | 4.23 | 80 |
| Current model without channel balance | 524 | 322 | 0.629 | 9.582 | 3.80 | 827 |
| Pyro-reference ProdLDA port | 537 | 332 | 0.636 | 9.004 | 657.38 | 846 |
| ProdLDA + entmax | 190 | 118 | 0.629 | 9.122 | 2.84 | 207 |

All new real fits use the same 27,222 training/3,889 validation rows and training
seed 7043, with the recipe differences above. The frozen current reference
(batch 256) gives 672/400 evaluable/useful motifs. No candidate meets the joint
criteria against both references, so none advances to extra real seeds or a
replacement claim. Every chemical run completes with zero MAG exceptions and
uses the held-out-compound-filtered library index.

The published-model suggestion was useful: the ProdLDA-entmax variant recovers
all 18 planted motifs in all three high-K synthetic runs, and its core is an
established architecture plus one published sparse mapping. Nevertheless it
serves only 207 real topic winners, despite 900 optimized prototypes. Reporting
those 900 without the 190 evaluable/118 useful counts would misrepresent utility.
Its actual PoE completion probability is better than the current model's, but
that is not the same target as chemical breadth plus compact explanations.
The two real PoE models each have 23,568,500 trainable parameters versus
19,278,001 for the current model. They simplify custom mechanisms, not parameter
count; their unconstrained topic matrix is larger despite the narrower encoder.

For publication, describe the current form as **ETM plus three transparent
adaptations**: channel-normalized beta, contextual variational evidence, and
the entmax latent-to-simplex link. The latent standard-normal prior and Gaussian
ELBO remain valid; the induced simplex prior changes and is not logistic-normal.
No invertible transform or entmax Jacobian is needed for an ELBO formulated in
Gaussian latent space. Evaluation uses a plug-in Gaussian-mean estimate, not an
integrated posterior predictive likelihood. One extra scalar is a parameter
count, not evidence that there is only one additional modeling assumption.

The report now scopes the old claim that context is "necessary" to its original
ablation, and instead says it **improves recovery in that ablation**. The new
ProdLDA-entmax synthetic results demonstrate why a universal necessity claim
would be false. The original context ablation removes the entire evidence
branch, not leave-one-out mixing alone. A fixed `c=0` control retaining routing
and the offset is a useful untested within-branch simplification. The necessity
of each top-2/smoothing/context choice has not been separately established.

This is a defensible retention decision for the tested recipes, not proof of
global minimality, an equal-capacity benchmark, or publication-ready external
validation. Tuning budgets, prior concentration, encoder width and BN inference
statistics can matter. The direct-Dirichlet negative screen does not rule out
that family. ECRTM, NSTM and Wasserstein-style models were considered in the
literature review but not newly trained here: their extra transport/clustering
objectives would need a separate faithful benchmark, not a renamed ETM run.

## Phase 3: surgical within-model review

The user requested a stronger audit of the individual mechanisms after the
broader comparisons. The retention recommendation above is the outcome of
phases 1/2, not a statement that all components are necessary. See the separately
recorded [within-model protocol](within_model_protocol.md) for exact algebraic
reductions, eight atomic ablations, staged combinations and confirmation rules.
Results from this phase are recorded in `evidence/within_model/`, including
the explicit failure, all seeds and all earlier negative results.

## Reproduction and verification

Use fresh output directories; the runners deliberately reject overwrites.
Example expanded synthetic run:

```bash
python -m scripts.run_published_neural \
  --synthetic-root output/benchmarks/minimal_neural_20260905 \
  --output output/benchmarks/published_replay/seed11_k128_prodlda_entmax \
  --variant prodlda_entmax --seed 11 --topics 128 --device cpu --threads 4
```

The synthetic-root argument names the **parent** of `synthetic_artifacts`.
Published-family synthetic runs use CPU; initial ETM-review and all new real
fits use CUDA.
For real validation, create a sealed view with
`scripts.prepare_msnlib_validation_view`, train with `--validation-run VIEW`
and `--output VIEW/models/minimal_etm --topics 1000 --seed 42`, then run
`python -m benchmarks.neural_ms2lda.chemical --run VIEW --data-root ASSETS
--method minimal_etm --split validation`.
The name `minimal_etm` is the isolated experiment artifact slot, not a claim
that a PoE architecture is ETM. The model result records decoder/prototype
semantics. New test evaluation is rejected for this slot.

Regenerate the inventories with `scripts.summarize_minimal_etm` using each raw
output root, then run the three review report generators:

```bash
python -S -m scripts.generate_minimal_etm_review
python -S -m scripts.generate_published_neural_review
python -S -m scripts.generate_contextual_reduction_review
```

The generators require all recorded candidates and seeds, verify recipes and
meaning, preserve negative results, and produce means, sample SDs and every
decision gate with source hashes. Earlier pre-extension code hashes remain
recorded; a repeat of the original seed-11/K=36 ProdLDA run under the final
extension-capable code reproduces beta and theta bit-for-bit. That replay is
QA, not an additional independent scientific fit.

Verification completed at the end of phases 1/2 (the phase-3 closeout follows):

- 132 focused scientific tests and 124 production tests pass (two existing
  Lark deprecation warnings). Exact workflow Black/Ruff checks pass on 58 files;
  dependency consistency and the production wheel build pass.
- Reloaded real ProdLDA and ProdLDA-entmax checkpoints reproduce all validation
  mixtures bit-for-bit, reproduce reported NLL, and leave every model buffer
  unchanged. The initial fixed-BN-entmax checkpoint was similarly checked.
- All models share the benchmark's probability floor 1e-12. For stock ProdLDA,
  the floored-token fraction is 0.0114% and unclipped NLL is 9.003943 versus
  reported 9.003884; entmax-ProdLDA has no clipped completion tokens. This cannot
  explain the breadth decision.
- The report's mathematical description, staged protocol, initial/expanded
  comparison tables, limitations and source inventory are updated. Historical
  test evidence and its seal are preserved; no model is promoted into production.
- Isolated built-wheel imports of MS2LDA and its production entry point pass;
  the wheel contains no experimental benchmark or research runner.

## Final phase-3 verification, 8 September

- 193 focused scientific tests and 124 production tests pass; the latter
  retain two existing Lark deprecation warnings. Exact workflow Black checks
  pass on 67 files and Ruff reports no issues. `pip check` passes.
- The production wheel builds and imports MS2LDA and `scripts.ms2lda_runfull`
  from a temporary extracted wheel outside the checkout. It contains no
  experimental benchmark or additional research runner. The active editable
  development installation is not replaced.
- All four report generators run without site packages (`python -S`). Two
  consecutive generations are byte-identical, every frozen historical evidence
  file is unchanged, and the explicit failed-run log/config/checkpoint hashes
  match the local sources. Phase-3 inventory validation requires 107 successful
  fits plus the separately recorded failure; failure cannot become a zero score
  or silently replace a missing completed seed.
- All three recommended full-document-context checkpoints reload on CPU using
  their unchanged, hash-verified validation inputs. Across all 3,889 rows per
  seed, maximum theta disagreement with saved GPU predictions is 7.75e-7 and
  completion-NLL disagreement is at most 1.02e-8 nats. Model state remains
  unchanged. `recommended_checkpoint_reload_audit.json` records the details.
- That audit also distinguishes float32 accumulation error from model error.
  Non-contiguous CPU beta rows can give a naive NumPy mass-sum error of 1.48e-4;
  float64 sums expose a smaller raw CPU-softmax mass error up to 1.10e-5.
  The existing runner already normalizes exported rows in high precision.
  Saved GPU channel-mass error is below 8.1e-8, and independent float64 decoder
  evaluation verifies the channel equation to 1e-12. No fitted model or
  reported metric is changed by these checks.
- The updated 29-page report compiles without undefined references, LaTeX
  warnings or overfull boxes. Every final page is rendered and visually
  inspected, including all four new tables, the numerical-failure row,
  historical figures and final recommendation. Canonical LaTeX/PDF artifacts
  are preserved; no new HTML or hosted artifact is substituted.
- No new scientific fits, test-matrix access, foundation-model download,
  production promotion, commit or push is performed during this morning
  closeout. Remaining scientific uncertainty is explicitly reported, not hidden
  by passing a numerical flag.

## Model-focused report refocus, 8 September

At the user's request, the main scientific report now follows only the chosen
whole-spectrum-context model. Its existing Abstract, Introduction, Related
work, Materials and methods, Results, Discussion, Limitations,
Data/code/reproducibility, Conclusion and References structure is retained.
Methods explain the published ETM base and three explicit enhancements. The
main comparison is Tomotopy LDA; plain ETM is a short secondary base-model
check. The architecture-search catalogue is no longer in the main manuscript.

The preceding 29-page audit is preserved here as
`model_form_review_report_20260908.tex`,
`model_form_review_report_20260908.pdf`, and
`model_form_review_report_sources_20260908.md`. Its PDF SHA-256 is
`089b63c270ffd2cd576f63e0345105367d47a3329e1f687b0e94d017c067169e`.
All full experimental inventories and interpretation above remain intact.

The new current-model fragments select exactly six synthetic and three real
validation fits for `reduced_document_context`. They cannot substitute the
historical model's test values. Tomotopy is the only comparator in the main
quantitative table, and missing LDA sparsity remains explicitly unreported.
The primary finding is greater useful-motif breadth and better completion NLL,
with lower conditional mean SOS. The validation-selection limitation remains
visible; no scientific result or fitted model changed in this editorial pass.

Verification for the focused report:

- 206 scientific tests pass, including independent whole-spectrum equations
  and topic/context gradients, exact selected-fit inventory checks,
  validation-versus-test guardrails, and preserved report section order.
  The historical equation-contract test now checks its archived manuscript;
  a separate contract checks the selected model's manuscript.
- Exact workflow Black and Ruff checks pass on the same 67-file scope.
  All four report generators execute with `python -S`; repeated generation
  is byte-identical, and all frozen evidence files are unchanged.
- The shared original-model module has a documentation-only introduction
  update identifying the selected subclass and archived equation labels.
  Its computations, checkpoints and saved source-provenance records are not
  rewritten. Production code and the Tomotopy backend are unchanged.
- The final report is eight pages, with two explanatory diagrams and four
  tables. Every final page is visually checked; the final LaTeX build has no
  undefined references, warnings, underfull or overfull boxes.
- The canonical and output copies have identical PDF SHA-256:
  `5cdadf5be958d1c0259bda2eef79450ab34892bc7a527ea7aeb645ebb250b6e6`.
  No new training, test evaluation, production promotion, commit or push is
  performed. The previous closeout's production/wheel checks are retained as
  historical checks and are not described as newly rerun here.
