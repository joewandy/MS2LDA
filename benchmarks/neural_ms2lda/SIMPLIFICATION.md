# Neural MS2LDA simplification rationale

## Decision rule

Prefer simpler code when results are practically tied, while retaining every
exact measurement and mechanical gate miss in the scientific ledger. Neural
MS2LDA must also retain a nonlinear representation-learning path so that future
pretrained spectrum embeddings, set encoders, or transformer encoders can be
integrated without replacing the topic model.

Parameter count is not the only measure of simplicity. Removing feature
pipelines, extra data products, loss terms, initialization procedures, and
recovery logic reduces executable paths, runtime cost, tests, and scientific
assumptions even when most parameters reside in the retained nonlinear router
and motif prototypes.

## What the ordered ablations established

The final deep model removes seven auxiliary mechanisms:

- Jensen--Shannon agreement between partial views;
- local reconstruction;
- dead-topic recycling;
- weighted spherical k-means++ initialization;
- Fourier mass coordinates;
- paired partial views; and
- adaptive fragment/loss channel mass.

Direct deletions showed that several other components serve distinct purposes:

- Removing the document gate left 40 evaluable and 20 useful motifs.
- Removing additive whole-document evidence left 230 evaluable and 128 useful
  motifs.
- Removing leave-one-out context left 139 evaluable and 89 useful motifs.
- Top-1 routing left 105 evaluable and 64 useful motifs.
- Removing Sinkhorn left 237 evaluable and 125 useful motifs.
- Removing positive-NPMI regularization left 212 evaluable and 120 useful
  motifs.
- Removing prototype separation left 278 evaluable and 169 useful motifs.
- Removing SGNS coordinates left 251 evaluable and 148 useful motifs.
- Joint per-batch optimization was stopped after four epochs because its
  projected runtime was operationally impractical.

Combining the additive document score and detached multiplicative gate into one
conditioning term left 101 evaluable and 64 useful motifs. These two uses of
whole-spectrum evidence are therefore not interchangeable.

## Deep U1 selection

Historical U1 was the simplest measured model that retained the nonlinear
context MLP while incorporating all seven auxiliary removals. Its earlier run
reported 836 optimized, 427 evaluable, and 272 useful motifs, mean SOS
0.6533397611, and validation NLL 8.8307366205.

An initial scratch reconstruction produced 823 optimized, 423 evaluable, and
253 useful motifs, mean SOS 0.6435336069, and validation NLL 8.8390296959. It
narrowly missed the useful-motif threshold by six motifs and mean SOS by
0.003272. Those exact misses remain in the selected ledger row's note.

The final provenance-grounded U1 implementation was then locked under
deterministic PyTorch execution from a fresh seed-42 initialization. It
completed all 40 epochs in 2,149.83 seconds and remained finite. Validation
likelihood took approximately 3 seconds and MAG/SOS approximately 122 seconds.
The lock produced:

| Metric | Observed | Historical reproduction threshold | Outcome |
| --- | ---: | ---: | :---: |
| Optimized motifs | 843 | 795 | pass |
| Evaluable motifs | 429 | 406 | pass |
| Useful motifs | 268 | 259 | pass |
| Mean SOS | 0.6506700670 | 0.646806 | pass |
| Validation NLL | 8.8320026353 | 8.919044 maximum | pass |
| Finite/stable | yes | yes | pass |

The final lock passes every historical U1 reproduction threshold. The earlier
two misses remain visible rather than being erased. The model also improves
substantially over the accepted control's 663 optimized, 312 evaluable, and 185
useful motifs and mean SOS 0.6323301481. U1 removes seven independent mechanisms
and reduced observed fitting time from 78.7 to about 35.8 minutes.

The trade-off is predictive likelihood: U1 validation NLL is 8.8320026353
versus 8.5014469154 for the control. That difference is not described as parity.
The selection deliberately prioritizes useful chemical motif discovery and a
simpler deep formulation, with NLL retained as an explicit secondary cost.

## Deep-architecture boundary

U1 retained a bias-free token projection and a trainable nonlinear context
router:

`Linear(256, 256) -> GELU -> LayerNorm(256) -> Linear(256, 128)`.

U7 replaced that router with a single linear context map. The later S-series
reduced the representation further to diagonal or fixed operations. Those rows
remain valuable negative-control evidence, but they are ineligible final models
because they remove nonlinear representation learning.

A future DreaMS spectrum embedding can augment or replace the current pooled
spectrum context before the nonlinear router. This integration point is a
property of the architecture, not a dormant compatibility path: the selected
implementation has no DreaMS dependency or optional experimental branch.

## Minimal nonlinear campaign

The final bounded campaign tested whether the nonlinear boundary itself could
be expressed more simply. Selection used seed 42 and validation only. Every
candidate had to pass the existing absolute chemistry floors, retain 95% of the
immediate baseline's motif counts and 99% of mean SOS, and keep NLL within 105%
under the chemistry-first rule. One narrow chemistry miss could qualify for the
predeclared tie band; no rescue run or hyperparameter sweep was allowed.

M1 replaced U1's two-layer normalized router with one bias-free
`Linear(256, 128) -> GELU` residual map. It reduced the model from 233,600 to
167,168 parameters and produced 884 optimized, 408 evaluable, and 265 useful
validation motifs, mean SOS 0.6580793714, and validation NLL 8.9741399256. All
absolute and relative chemistry gates passed without the tie allowance. NLL
missed the historical 101% reporting reference but remained inside the
predeclared 105% ceiling. A fresh lock replay reproduced every state tensor,
saved validation array, metric, and gate decision exactly.

M2 then replaced the 128-dimensional learned geometry with 50 learned feature
scales, a 100-to-50 nonlinear residual map, and 50-dimensional prototypes. It
produced 839 optimized, 430 evaluable, and 262 useful motifs, mean SOS
0.6398936376, and validation NLL 8.9224852945. Although all absolute floors
passed, optimized motifs missed the relative threshold by one and mean SOS fell
below both the standard and tie thresholds. Two relative misses require
rejection, so the campaign stopped and the planned unit-exponent M3 was not run.

M1 is therefore the final architecture. Its one reporting-only test evaluation
was performed after selection and was not used to tune or reopen the campaign.

## Final retained mechanisms

The selected M1 model retains top-2 routing, leave-one-out context, additive
whole-document evidence, the detached document gate, Sinkhorn targets, positive
NPMI, prototype separation, SGNS coordinates, routing-temperature annealing,
and alternating router/topic optimization. Each survived because its direct
removal caused a material failure or because it defines the required nonlinear
model family.

The final code has no implementation of the seven accepted removals. The
ablation ledger is historical evidence only; it does not drive runtime switches
or loader compatibility.

## Bounded future work

The architecture is now frozen. Further work should establish robustness and
interpretability before adding capacity: repeat the fixed model across
predeclared seeds, inspect neural-only motifs with chemical experts, and then
test one frozen DreaMS adapter as a separate study. Fine-tuning or a different
set encoder should not be combined with that first DreaMS experiment.

## Evidence boundary

Tomotopy is fixed: its committed rows and diagnostics are preserved and it is
not rerun. The final neural test workflow occurs only after M1 is frozen and is
used for reporting rather than further architecture selection. Exact scientific
values live in the one ablation ledger and canonical `results.json`; no alternate
result format is retained.
