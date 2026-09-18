# Phase 3: within-model minimality, requested after the broader review

Recorded before fitting any phase-3 candidate. Phases 1/2 are retained in full;
their retention decision did not establish necessity of individual mechanisms.
The original four-way synthetic ablation removes the entire evidence branch.
It cannot separate leave-one-out mixing, top-2 routing and the log offset.

## Exact derivations first

For row-centering operator C and topic evidence r, the offset is exactly
`C log(r + 1/K) = C log(1 + K r)`. There is no additional free parameter;
unit total smoothing mass remains a fixed modeling choice.
Centering cannot just be deleted: entmax and softmax are translation invariant,
but the Gaussian KL is not. The centred form removes an unnecessary common-mode
mean penalty in the evidence branch (the residual neural mean is unrestricted).
With c=0, the entire leave-one-out branch reduces to vocabulary-level attention
`A[w,:] = top2-softmax(cos(rho[w], alpha))`, followed by `r = x A`.
With a softmax link, deterministic theta is proportional to
`exp(neural_mean) * (1 + K r)`; this is an interpretable multiplicative posterior
correction within a conventional logistic-normal ETM, not the entmax prior.

## Atomic screen

Use eight explicitly named reductions, with all other components unchanged:

1. `reduced_no_loo`: c=0, remove context mixing and its scalar; compute r=xA.
2. `reduced_document_context`: full-document mean instead of leave-one-out mean.
3. `reduced_fixed_context`: retain leave-one-out mixing but fix c=1.
4. `reduced_full_routing`: all-topic softmax instead of truncating to top two.
5. `reduced_linear_evidence`: replace centred log by its first-order Taylor
   approximation at uniform evidence, `(K/2) * (r - mean(r))`; no tuned scale.
6. `reduced_softmax`: conventional softmax link, retaining the entire evidence
   branch. The published ETM simplex prior is restored.
7. `reduced_shallow`: remove the second hidden ReLU layer; retain width 800.
8. `reduced_evidence_only`: remove the residual MLP; topic-embedding attention
   supplies the Gaussian mean and a global learned diagonal variance is used.
   This remains a learned nonlinear amortized attention encoder, but is not
   represented as the published ETM's standard MLP encoder.

Train each at K=128, seeds 11/23/37, then K=36 for all passing K=128 candidates.
Use the phase-1 paired current controls and identical data/input initialization,
120 epochs, batch 200, hidden 800 unless explicitly reduced, raw counts,
Adam lr=.005, betas=(.9,.999), weight decay=1.2e-6, CUDA, 4 CPU threads.
No learning-rate or architectural search inside a candidate. All fits and
negative results are retained under `output/benchmarks/contextual_reduction_20260907`.
Reuse the recorded synthetic gates (beta within .05, theta within .10,
mean of per-seed median effective topics <=5, no catastrophic duplicate;
at K=128 retain >=90% recovered motifs). No new candidate accesses test arrays.

Advance every candidate passing both K levels to real K=1000 validation,
initial seed argument 42 (training seed 7043), same optimizer and budget.
Reuse the real gate against BOTH frozen and paired current references:
evaluable and useful motifs >=95%, SOS within .02, completion NLL within 2%,
median effective topics <=5, >=800 winners, no catastrophic duplication.
Run complete chemical evaluation even for initially weak real fits.

## Combinations and confirmation

Successful removals are not assumed additive. If no-context and shallow-encoder
reductions pass the synthetic screen, explicitly train their combination on
all six synthetic settings. If no-context passes, also test top-1 vocabulary
routing (no score softmax); if both pass, test no-context + shallow + top-1.
Top-1 attention has zero routing gradient almost everywhere; the decoder and
neural MLP still learn. It is not presented as differentiable soft attention.
Combinations must pass the same synthetic and real gates. If document-context
and shallow both pass, also test their combination on all six synthetic
settings (the no-context branch may fail independently). If fixed c=1 also
passes, test document-context + fixed-c, and add shallow if that pair passes.
These additional branches are recorded while the atomic screen is running,
before either fixed-c or shallow results have been observed. Other combinations
are not silently searched; any further extension must be recorded first.

Any candidate passing the initial real gate receives two additional real
training seeds, arguments 11 and 23. Fit matched current controls at those
seeds too. Require the same real gates per seed against the same-seed control
and frozen reference before recommending replacement. Seeds share a validation
split: they assess training stability, not external generalization. Gates are
decision tolerances, not statistical non-inferiority confidence bounds.

The result can establish a best-supported reduced model within this declared
edit set and budget, not global or absolute mathematical minimality. Failed
synthetic screens do not prove real-data impossibility. Channel balancing and
whole-evidence removal already have direct evidence in earlier phases; retain
that evidence rather than re-counting it as new independent runs.

### Numerical boundary correction during the atomic screen

The one-layer model recovers exactly 15 motifs on average against 50/3 for the
control: exactly 90%, a pass by the recorded inclusive rule. Binary floating
point evaluates `0.9*(50/3)` as 15.000000000000002. Correct the shared generator
to recognize equality within 1e-12; this fixes numerical evaluation, not the
threshold. Re-generate prior review gates and test the boundary explicitly.

### One contrast-controlled routing follow-on

After the atomic synthetic screen, before fitting this follow-on: unit-
temperature all-topic routing recovered only 3.33 motifs at K=128. This does
not isolate the necessity of hard truncation: cosine scores lie in [-1,1], so
untruncated softmax can become a weak, nearly uniform evidence correction.
Test exactly one smooth alternative, `reduced_scaled_full_routing`, retaining
all other current mechanisms and replacing top-2 with all-topic softmax of
`sqrt(L) * cosine`, where L is the fixed embedding dimension. For independent
isotropic unit vectors cosine variance is 1/L, motivating this fixed variance
normalization without choosing a temperature from results. This is a stated
adaptation, not a claim to reproduce a specific published attention model.
Run all six synthetic settings and apply the unchanged gates; advance to real
validation only if both K levels pass. No further temperature sweep is planned.

### Execution recovery, not a model change

The first document-context real fit was terminated with exit code 143 after
the epoch-80 message and before saving final weights; it reported no numerical
failure. Its incomplete directory is preserved and its interruption is listed
in `evidence/within_model/interrupted_attempts.json`. A fresh same-seed retry
uses a detached, logged local worker. This is not an additional independent
seed or a scientific negative result.

The runner now saves model, optimizer, minibatch-generator and CPU/CUDA random
states every 20 epochs and supports exact continuation in a fresh directory.
Recovery may not change the model, data hashes, seed or optimization budget.
CPU and CUDA tests verify bit-identical trajectories with/without checkpoint
I/O and across continuation. Two earlier synthetic formulations also replay
bit-for-bit after the reduction module's purely additive variant extensions.
Recovery timings are operational metadata, not fair model-speed comparisons.

### Hard-routing follow-on after the fixed-weight screen

Fixing c=1 passes both synthetic settings, even though c=0 fails. Therefore the
earlier no-context top-1 branch would leave the most obvious routing deletion
untested. Before fitting it, add exactly `reduced_fixed_context_top1`: retain
leave-one-out context with c=1 and replace the two-way local softmax by a single
hard argmax assignment. This jointly removes the learned scalar and two-way
soft weighting; the fixed-c=1 atomic control separates the first removal.
The routing gradient is zero almost everywhere, so keeping c trainable would
be misleading. The residual MLP, Gaussian variational family and embedding
decoder remain neural and trainable; no straight-through surrogate is used.
Run all six synthetic settings, apply the unchanged gates, and advance only
if both pass. This is a disclosed extension, not part of the original eight
atomic ablations, and does not open a search over arbitrary top-k values.

### Descriptive one-layer repeats after its first real result

The one-layer model gives 672 evaluable/402 useful motifs, mean SOS 0.6247,
3.60 effective topics and 830 winners. Its NLL 9.7637 misses the strict
2%-worsening limit (2.62% above the frozen reference), while chemical breadth
is preserved. This is not evidence that the second layer is essential for
motif discovery. Before the repeats, record a descriptive comparison of this
specific near-miss at seed arguments 11 and 23, each with a same-seed current
control. The user prioritizes the minimum defensible neural form; the repeats
assess stability of this trade-off rather than silently changing the gates.
The original strict rule remains unchanged, the initial failure remains
reported, and no predictive-fit advantage is claimed when it is absent.
These two current controls also serve any genuine joint-gate confirmations.

### User clarification of the decision target

After seeing the one-layer result and before its repeat-seed results, the user
explicitly accepted the approximately 2.6% predictive-fit trade-off, regarding
the difference from 2% as immaterial to this simplification goal. Keep the old
2% gates visible for auditability, but do not use that cutoff to retain a layer
whose chemical utility is unnecessary. Report each model's actual NLL change
alongside the unchanged chemistry, sparsity, duplication and inventory checks.
This accepts a modest loss of the observed scale, not an unlimited loss or a
claim that the original numerical gate was passed. No new test data are used.

### Broader clarification: reference thresholds, not decision laws

The user then explicitly asked not to be rigid about any arbitrary thresholds.
Accordingly, retain every original numerical comparison for an audit trail,
but do not translate a small cutoff miss into a claim of scientific necessity.
Evaluate the magnitude and consistency of trade-offs, chemical breadth,
within-spectrum compactness, across-corpus coverage, and what is actually
removed from the model. The 95% chemical-retention and 800-winner criteria are
reference points too, not privileged biological constants. No new threshold
is chosen to make a favored candidate pass.

Before fitting the following real variants, extend the declared validation
set with three informative alternatives that missed the synthetic screen:

- `reduced_document_context_fixed_shallow`: the three-way simplification
  recovered 14.33 rather than the threshold of 15 planted motifs at K=128;
  that small absolute difference should not prevent a real-data comparison.
- `reduced_no_loo`: removing context altogether is a major conceptual and
  computational deletion. Its synthetic degradation is larger, but does not
  establish that real chemical utility needs context.
- `reduced_softmax`: this directly restores the published logistic-normal
  ETM simplex construction and therefore has particular explanatory value.

The latter two receive their previously unscheduled K=36 three-seed fits as
well, so the additional comparisons have complete two-K synthetic summaries.
Those six small supplemental fits use two CPU threads to reduce contention;
CUDA, seeds, initialization, loss and optimizer remain unchanged. All real
fits and the original phase-3 synthetic screen use four CPU threads. Thread
counts remain recorded in each immutable configuration; wall times are not
used to compare model efficiency across concurrent workers.
A representative seed-11 no-context K=36 replay at four threads has
bit-identical beta and validation theta to the two-thread original; it is a
numerical audit, not another independent scientific fit.
All three then receive the unchanged real seed-42 fit and chemical evaluation,
regardless of their old synthetic flags. The two all-topic-routing variants
remain documented synthetic negatives: their much larger recovery losses do
not make them promising simplifications under the tested fixed recipes.

The already-planned hard-top-1 candidate also advances after its six synthetic
fits. Repeats are required for a recommended replacement, including promising
near misses; passing an old flag alone does not determine recommendation.
Specific additional repeat choices will be recorded before seeing those
repeat results. This is an explicitly disclosed adaptive development review,
not a preregistered confirmatory comparison, and not a proof of global
minimality. All negative and interrupted attempts remain discoverable.

This clarification supersedes the original rule that every strict initial
pass automatically receives repeats. Repeats now follow a recorded scientific
shortlist, not a discontinuous flag; a recommended replacement still requires
the planned same-seed comparisons. At the time of this change, the only two
completed phase-3 real fits are one-layer and document-context + one-layer,
neither of which passes every old reference flag. The already-recorded
one-layer repeats continue unchanged.

### Chemical-score threshold sensitivity

As a descriptive check after that clarification, summarize the same eligible
topic SOS scores at inclusive cutoffs 0.5, 0.6, 0.7, 0.8 and 0.9. This changes
neither annotations, training, topic associations, nor the established
definition of a useful motif (SOS >=0.6). It checks whether a comparison rests
only on that single cutoff; all five counts are retained, with no favorable
cutoff selected as a replacement endpoint. The full eligibility definition
and conditional nature of these scores still apply. This is sensitivity
analysis, not five independent confirmations or a new composite ranking.

### One bounded encoder-capacity follow-on

Before fitting it, add `reduced_shallow_narrow`: the same one-layer contextual
ETM with hidden width 100 instead of 800. The width is taken from the already
reviewed Pyro neural-topic-model example, not selected by an MSnLib width
sweep; this remains an ETM adaptation, not a ProdLDA reproduction. Deleting
the second 800-unit layer alone removes only 640,800 parameters (3.3%); the
input-to-hidden matrix dominates the remaining parameter count. This follow-on
tests whether the conventional neural encoder can be much smaller without
adding another mechanism.

Run all six synthetic settings and one real seed-42 validation fit, with
complete chemistry regardless of the old synthetic flags. Keep the remaining
recipe unchanged, including four CPU threads. The decoder topic vectors are
initialized before the encoder, so their seeded initialization is identical
across widths; differently shaped neural layers use their native initialization
and are not claimed to be bit-matched. No additional width search is planned.
Shortlisted replacement forms still require the recorded repeat comparisons.

### Document-context repeat shortlist

The completed seed-42 document-context fit yields 669 evaluable/389 useful
motifs, mean SOS 0.62768, NLL 9.52839, 3.687 effective topics and 806 winners.
The paired current control gives 675/394, SOS 0.62285, NLL 9.53057, 3.721
effective topics and 829 winners. This is a meaningful computational deletion
with small initial trade-offs, not merely a numerical reference-flag pass.
Before fitting its repeats, add document context at seed arguments 11 and 23
to the scientific shortlist, using the already-planned same-seed controls.

### Attention-only repeat shortlist

The seed-42 evidence-only form has 49,001 trainable parameters, compared with
19,278,001 for the current model. It gives 537 evaluable/326 useful motifs,
mean SOS 0.63074, NLL 9.12266, 4.371 effective topics and 631 winners. This is
17.3% fewer useful motifs but 99.75% fewer trainable weights, better completion
NLL, and five more motifs at SOS >=0.8 than the paired control. It is a serious
compactness/breadth trade-off, not ruled out merely by the old reference flags.
Before observing repeat results, add seed arguments 11 and 23 for this form.
The frozen SGNS embedding table and its pretraining cost remain necessary;
trainable-parameter reduction is not the same as total memory or runtime
reduction. All comparisons continue to report chemical counts and likelihood
separately, without a composite score chosen to favor the small model.

### MLP removal accepted; input representation remains replaceable

Before attention-only repeat results, the user explicitly accepted removing
the MLP and requested a clean future path to DreaMS or another direct-spectrum
encoder, without implementing that integration now. The preferred low-complexity
direction is therefore the attention-only model, subject to reporting its
actual repeated-seed breadth/fit trade-off. This does not make the historical
MLP-free initial result a match to the old chemical-retention flags.

Add a research-only compositional shell with an opaque-observation posterior
encoder contract returning Gaussian mean/log variance. The existing attention
encoder and a toy structured-peak test encoder exercise that boundary; no
DreaMS package, pretrained weights, new data or foundation-model evaluation
are introduced. Check imported-state predictions, gradients and frozen decoder
against the already-fitted MLP-free ablation. Removing tokenization from the
encoder is distinct from replacing the token-emission likelihood; a fully
tokenizer-free observation model and motif interpretation are deferred.

One final nested reduction is declared before its fits:
`reduced_attention_document_fixed` combines evidence-only inference with
full-document context and fixed c=1. Earlier successes for context deletions
in MLP-based models cannot establish that they survive MLP removal. This
tests the especially simple attention-only form without leave-one-out mixing
or a learned context scalar, rather than assuming independent removals add.
Run all six synthetic settings and one seed-42 real fit with complete chemistry,
irrespective of old reference flags, retaining the remaining paired recipe.
If this form is selected over the recorded attention-only shortlist, record
its two same-seed repeats before running them. This closes the candidate set;
no additional temperature, width or variance-family search is planned here.
The study still cannot prove global/absolute neural-model minimality.

### Publication framing clarification: retain an MLP candidate

Before the remaining MLP and attention-only repeat results, the user revisited
the preference to remove the MLP, expressing concern about publication novelty
and equivalence to ordinary LDA or NMF. Retain an MLP-based ETM as the main
publication candidate for now and keep the already-planned no-MLP/nested runs
as explicit ablations. This supersedes the earlier preference for an MLP-free
primary model, without erasing that earlier stage or changing any result.

The rationale for an MLP must be demonstrated capacity or chemical-inventory
benefit, not novelty of a standard neural layer: published ETM already uses
an MLP recognition network. Removing it here does not yield classical LDA or
ordinary NMF; the embedding-constrained channel-specific emissions, nonlinear
amortized attention, Gaussian-entmax construction and variational objective
remain. Attention and sparse topic modelling also have published precedents,
so the report must avoid claiming those ingredients individually as new.
The contribution is a precisely specified spectral adaptation and validated
trade-off; the MLP's minimum useful depth/width remains an empirical question.
The replaceable input boundary and deferred DreaMS direction apply with or
without an MLP. No additional candidates or new experiments are added by this
framing clarification, and all scheduled comparisons remain in the report.

### Explicit priority: a published ETM backbone

The user then clarified that the principal reason to retain the MLP is proximity
to an established published base, reducing the burden of defending a new model.
This is now the primary selection constraint: keep published ETM's conventional
MLP Gaussian recognition structure, embedding-based emission construction and
latent-space ELBO as the starting point; prefer a small number of named,
empirically justified adaptations. A smaller standard MLP is a capacity setting,
whereas replacing it entirely with attention is a more substantial recognition-
family redesign and remains an ablation, not the recommended publication base.

The literature justifies inherited ETM machinery, not arbitrary modifications.
In particular, entmax changes the induced simplex prior and channel balancing
changes the emission normalization; those departures require explicit equations,
citations, checks and ablations. Contextual evidence changes recognition rather
than the generative prior and likewise needs evidence for its benefit. No claim
that the modified model is identical to the published ETM is permitted. This
clarification adds no experiments and does not discard the completed broader
search. The final recommendation must respect published lineage before using
parameter count as a reason to favor a different encoder family.

### 8 September completion record (after all scheduled outcomes)

The closed inventory has 87 successful synthetic fits and 20 successful real
fits with full chemical evaluation, plus one failed real configuration:
`reduced_linear_evidence`, seed argument 42. It raises non-finite training loss
after the epoch-80 recovery checkpoint; no final score is imputed and no rescue
tuning is performed. A separate SIGTERM-interrupted document-context attempt
and its from-scratch retry remain recorded, not counted as independent seeds.
All six overnight queues have stopped; no new candidates are added at closeout.

The three paired repeat sets support full-document context with the standard
two-layer MLP as the simplest clearly supported change under the final
published-lineage priority. Its useful count averages 390.7 versus 389.7 and
NLL 9.5245 versus 9.5397; a single 799-winner result is not rejected at the old
800 reference. Shallow-800 remains a chemical/predictive trade-off, and the
100-unit MLP remains promising but has only one real seed. The full report
quantifies these comparisons and the other one-seed options without claiming
that all retained mechanisms are absolutely necessary. No new thresholds are
introduced, no historic flags are changed, and no new test data are accessed.
These are post-result recommendations, not retrospectively preregistered rules.
