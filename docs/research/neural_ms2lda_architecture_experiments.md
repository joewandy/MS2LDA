# Neural MS2LDA architecture experiments

This ledger records seed-42 architecture development after the published K=500
checkpoint. Test matrices and chemical outcomes were kept sealed until an
architecture satisfied every predeclared validation gate.

## Frozen development baseline

- Source commit: `9cd8354895b77bac2df611d9a7a137ecb4e72998`
- Selected checkpoint: epoch 38,
  `6b424d133e66a5b9623572074fcc244566c701cf48e04e8dca363f09301c7961`
- Mean train-co-occurrence NPMI: `-0.5698579778`
- Top-word pairs with no train co-occurrence: `0.6629333333`
- Top-word diversity: `0.7354`
- Validation median effective topics per spectrum: `15.2294549341`
- Validation completion NLL per token: `8.3394851709`

The success gates are mean NPMI at least `-0.4198579778`, undefined-pair
fraction at most `0.5629333333`, diversity at least `0.70`, median effective
topics at most `11.4220912006`, and validation NLL at most `8.5896697261`.

## Family 1: mutual positive-NPMI topic formation

Hypothesis: a global differentiable topic loss that rewards probability mass
on strong train-only word co-occurrence edges will prevent the decoder from
assembling topics from incompatible fragment/loss neighbourhoods.

The graph uses words seen in at least ten training spectra, pairs seen in at
least three, positive NPMI, and at most sixteen mutual nearest neighbours per
word. It contains 58,809 undirected edges, has maximum degree 16, and never
reads validation or test spectra. Its fixed loss weight is 1.0; no parameter
sweep is performed.

Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-graph-v1` under the external
MSnLib run root.

Outcome: partial success, retained as evidence but not as the final model. The
goal-aware selector chose epoch 24 (`4c88d7992d2ec356b4a74516facd7e9c20b7de14fb266a7a443e745018ce28eb`).
It achieved NPMI `0.0635905694`, undefined-pair fraction `0.1809333333`, and
validation NLL `8.3789200793`, passing all three corresponding gates. It failed
diversity (`0.2752`) and median effective topics (`20.7887236982`). At epoch 40,
diversity improved to `0.5538` and effective topics to `17.6548724026`, while
NPMI remained much better than baseline at `-0.0182167259`; the missing gates
still failed. No test matrix or chemical outcome was evaluated.

Decision: retain the co-occurrence graph objective and test a separate
nearest-topic separation architecture. Do not alter the completed family-1
run or interpret its coherence-only gain as sufficient.

## Family 2: co-occurrence plus nearest-topic margin

Hypothesis: replacing the normalized all-pairs ERNTM loss with a direct margin
on each prototype's nearest neighbours will preserve distinct topic identities
while the family-1 graph objective maintains coherent word assembly.

For each normalized prototype, the eight nearest other prototypes are
penalized when cosine similarity exceeds 0.3. The squared-margin loss has fixed
weight 5.0; at the frozen baseline its weighted prototype-gradient norm is
`0.0663`, comparable to the graph objective's `0.1059`. The old ERNTM loss is
disabled. No separation parameter sweep is performed.

Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-nearest-margin-v1` under the
external MSnLib run root.

Outcome: rejected early after the atomic epoch-5 checkpoint. At epoch 4,
nearest-prototype median cosine had worsened to `0.9863244891` (95th percentile
`0.9997389317`), diversity was `0.2538`, and median effective topics was
`20.4654023650`. Coherence and NLL moved in the intended direction, but the
separation mechanism did not stabilize topic identity.

Cause: the margin was evaluated only in four topic updates per epoch, whereas
every router minibatch moved the same prototypes. The high-frequency router
path therefore overwhelmed the low-frequency separation path. The run was
stopped rather than spending the full budget on a mechanistically invalid
placement. Its checkpoints and `aborted.json` remain in place; no test matrix
or chemical outcome was evaluated.

Decision: retain the same graph and margin definition, but apply separation to
every optimizer step that updates prototypes. Record that corrected placement
as a new, independently locked run.

### Corrected family-2 placement

The same eight-neighbour, cosine-0.3, weight-5 margin is applied during both
router and topic updates. On a frozen baseline router minibatch, the router's
prototype-gradient norm is `0.0700`; the weighted separation gradient is
`0.0663`, so the correction is material without dwarfing the primary router
objective. Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-nearest-margin-router-v2`.

Outcome: partial success, concluded after the atomic epoch-10 validation
checkpoint (`17d767cdab3d15dd63667d736ce150c52e8468a16e0557e081405f5587dadf53`).
Median nearest-topic cosine fell from `0.7665` at the frozen baseline to
`0.3567`. NPMI (`-0.2444199847`), undefined-pair fraction (`0.3771555556`),
diversity (`0.8064`), and NLL (`8.4949500752`) passed their gates. Median
effective topics remained `21.7368387533`, failing the `11.4220912006` gate.
The family-1 annealing control still had `17.6548724026` effective topics at
epoch 40, so continuing an otherwise unchanged aggregation mechanism was not
justified. The run's `family_result.json` records the decision and confirms
that no test matrix was loaded.

Decision: retain the graph and corrected separation placement, then isolate
the document-mixture mapping.

## Family 3: spectrum-level quadratic mixture

Hypothesis: count-normalizing routed token mass is too diffuse even when the
topics themselves are coherent and distinct. Replace only that document-level
mapping with `theta = mass^2 / sum(mass^2)` during both training and inference.
This smooth fixed transformation preserves every routed topic and remains
differentiable; it is not a gate-tuned hard topic cap.

On the frozen family-2 epoch-10 validation checkpoint, applying the mapping
without retraining reduced median effective topics from `21.7368387533` to
`7.8643187197`. NLL moved from `8.4949500752` to `8.7395792843`, showing that
end-to-end training must recover `0.1499` NLL to pass. For comparison, canonical
sparsemax and 1.5-entmax projections were rejected before training because they
collapsed mixtures to `2.6981` and `4.2348` effective topics while degrading
NLL to `9.3064` and `9.0400`, respectively.

Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-nearest-margin-quadratic-v1`.

Outcome: rejected after the atomic epoch-10 validation checkpoint
(`55b1e556fb314d9ad433ef42f87fa01c2399e8ef5b5e2acc3f6d4ec28d27c1d9`).
NPMI (`-0.2602373898`), undefined-pair fraction (`0.3873333333`), diversity
(`0.8450`), and effective topics (`8.2246973104`) passed. Validation NLL stayed
at `8.8648398924`, above its `8.5896697261` gate, despite falling training
completion loss. The fixed inference transform therefore created a structural
reconstruction penalty rather than a transient optimization delay. No test
matrix was loaded.

Decision: preserve ordinary probabilistic aggregation and learn concentrated
routing from a spectrum-level objective instead.

### Corrected family-3 mechanism: learned local concentration

Hypothesis: minimizing each document mixture's Gini impurity during router
training will make peaks in the same spectrum share a small topic set, while
the existing corpus-level Sinkhorn target prevents global topic collapse.
Inference remains a single pass with ordinary count-normalized routed mass;
there is no post-hoc truncation or sharpening.

The fixed Gini weight is 2.0. On a frozen baseline minibatch, its weighted
prototype-gradient norm was `0.0080` versus `0.0692` for the initial router
objective, and its weighted router-head norm was `0.0663` versus `1.6396`.
After convergence of the old baseline, the relative contribution becomes
material but remains below the primary router objective. This weight was set by
gradient scale, not a validation sweep.

Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-nearest-margin-gini-v1`.

Outcome: rejected after epoch 4
(`761b4a4b180ed3b91ed77e2c5db5e5e70ea319a9a4fe79d8903fd0af466f6eb5`).
The first validation already overshot to `2.0872` effective topics and only 17
corpus-active topics. By epoch 4, effective topics were `1.9398`, corpus-active
topics fell to 14, diversity fell to `0.5040`, and NLL was `8.9007`. The
existing Sinkhorn objective did not prevent the always-on local penalty from
causing corpus collapse. No test matrix was loaded.

Decision: do not tune the penalty weight. Replace it with a parameter-free
hierarchical routing architecture that shares spectrum evidence before token
assignment.

### Corrected family-3 architecture: local-document product of experts

Hypothesis: independent peak scores remain diffuse because shared spectrum
evidence only enters through a learned leave-one-out correction. For each peak,
add its local topic cosine score to a whole-spectrum topic cosine score before
the existing top-2 route. This product-of-experts score uses a count-weighted
sum of projected token features, has fixed coefficient 1.0, adds no trainable
parameter, and keeps one-pass inference.

On the frozen family-2 epoch-10 checkpoint, this exact routing rule reduced
median effective topics from `21.7368387533` to `14.4087603350`. NLL changed
from `8.4949500752` to `8.6283609837`, only `0.0387` above the gate before any
end-to-end adaptation. Topic-word metrics are unchanged in this diagnostic.

Run directory:
`neural-ms2lda-architecture-seed42/cooccurrence-nearest-margin-hierarchical-v1`.

Outcome: accepted. The declared 40-epoch run selected epoch 10 by the frozen
gate-aware rule. Its checkpoint SHA-256 is
`f24ffdd58d6bdd47a5b62a06367f828b18f93eb467eebc2530591f808462c6a4`.
Validation NPMI was `-0.3506510228` (improvement `0.2192069550`), undefined
top-word pairs were `0.4788` (reduction `0.1841333333`), diversity was `0.8732`,
median effective topics was `11.3754035943` (reduction `25.3066%`), and NLL was
`8.4337558516`. All five predeclared gates passed. The architecture and
checkpoint were locked before the test partition was accessed.

The single final test evaluation was stable. Median effective topics fell from
`14.7503768131` to `11.1317729488` (`24.5323%`), and the 95th percentile fell
from `41.9391949071` to `29.5392102226` (`29.5666%`). Test NLL moved from
`8.3563395270` to `8.4497875551`, a `1.1183%` degradation within the `3%`
tolerance. Corpus-active topics increased from 132 to 147.

Post-lock leakage-controlled chemistry also improved: annotation coverage rose
from `0.272` to `0.386`; high-confidence associated spectra rose from 106 to
379, eligible topics from 26 to 53, and mean SOS from `0.5792372947` to
`0.6264139084`. Dominant-topic mean SOS stayed essentially unchanged
(`0.6129487934` to `0.6118691618`) while eligible topics rose from 105 to 178.
No chemistry label entered training, and every held-out compound remained
excluded from the MAG reference index.

Decision: retain the co-occurrence graph, nearest-topic margin on every
prototype update, and parameter-free local-document product-of-experts router
as the verified seed-42 architecture. Do not continue validation tuning after
the one-time test confirmation.
