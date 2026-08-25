# Neural MS2LDA handover

## Current state

The repository supports one seed-42, K=1000, 40-epoch deep Neural MS2LDA
architecture under a six-CPU-thread allowance. It is a single-dataset,
single-seed research result, not evidence that the production Tomotopy backend
should be replaced.

The selected M1 model is the simplest tested architecture that retains a
nonlinear representation-learning path. Its deterministic lock produced 884
optimized motifs, 408 high-confidence evaluable motifs, 265 useful motifs, mean
SOS 0.6580793714, median SOS 0.6488636364, and validation NLL 8.9741399256.

An earlier scratch reconstruction narrowly missed two thresholds derived from a
historical U1 run: six useful motifs and 0.003272 mean SOS. The
provenance-grounded deterministic lock passed every historical U1 reconstruction
threshold and improves the accepted control's validation motif inventory while
removing seven auxiliary mechanisms. The project rule prefers the materially
simpler formulation when the scientific result is practically tied. M1 passed
every absolute and relative chemistry gate without the tie allowance. Its NLL
missed the historical 101% reporting reference but passed the predeclared 105%
chemistry-first ceiling. M2 then failed optimized-motif and mean-SOS retention,
so the campaign stopped without constructing M3.

Tomotopy is fixed. Its committed model evidence, rows, diagnostics, and
iteration counts are preserved without retraining or reevaluation.

## Implemented model

1. Train 48-dimensional SGNS token coordinates on training spectra only. Append
   two fragment/loss indicators and normalize the resulting 50-dimensional
   token features.
2. Apply a bias-free 50-to-128 projection. Select 1,000 distinct prototype
   starting tokens with the seed-42 uniform permutation and initialize learned
   128-dimensional prototypes from those projected tokens.
3. Concatenate each projected token with its count-weighted leave-one-out
   spectrum context. Apply one bias-free `Linear(256, 128)` followed by GELU,
   add the nonlinear correction to the token vector, and normalize.
4. Add whole-spectrum prototype evidence, retain and renormalize the top two
   token assignments, and aggregate their count-weighted mass.
5. Multiply routed topic mass by detached whole-spectrum evidence raised to
   0.75. Exact zero support is preserved and empty spectra receive a uniform
   mixture.
6. Decode with cosine logits and separate fragment/loss softmaxes. Each channel
   receives exactly half of the topic probability mass.
7. Train on full spectra with alternating router and topic blocks. Retain
   Sinkhorn balancing, positive-NPMI regularization, prototype separation,
   routing-temperature annealing, deterministic execution, finite checks, and
   gradient clipping.

The model has 167,168 learned parameters: 6,400 in the token projection, 32,768
in the nonlinear context router, and 128,000 in the topic prototypes.
The implementation is unconditional: there are no architecture flags,
experimental switches, schema variants, or loader compatibility branches.

## Removed mechanisms

The final implementation contains no Fourier mass coordinates, paired partial
views, Jensen--Shannon view consistency, local reconstruction, dead-topic
recycling, weighted k-means++ initialization, or adaptive fragment/loss channel
mass. Their exact ablation measurements remain in the single ledger; their code
paths and protocol fields do not.

Shallow U7 and S-series endpoints are excluded because they remove the
nonlinear learned representation block. A future DreaMS embedding may augment
or replace the pooled spectrum context before the nonlinear router. No DreaMS
dependency or compatibility interface is included now.

## Authoritative evidence

| Purpose | Location |
| --- | --- |
| Mathematical and scientific description | `docs/research/neural_ms2lda_report.tex` |
| Reviewed compiled report | `docs/research/neural_ms2lda_report.pdf` |
| Fixed study constants | `benchmarks/neural_ms2lda/protocol.json` |
| Model and one-pass inference | `benchmarks/neural_ms2lda/model.py` |
| Training objectives | `benchmarks/neural_ms2lda/objectives.py` |
| Alternating updates | `benchmarks/neural_ms2lda/training.py`, `optimization.py` |
| Data and train-only token features | `benchmarks/neural_ms2lda/data.py`, `spectra.py` |
| Canonical model | `benchmarks/neural_ms2lda/results/seed42/trained_model/` |
| Paper-facing evidence | `benchmarks/neural_ms2lda/results/seed42/results.json` |
| Ablation ledger | `benchmarks/neural_ms2lda/results/seed42/ablation_results.json` |
| Simplification rationale | `benchmarks/neural_ms2lda/SIMPLIFICATION.md` |

`results.json` is the sole paper-facing numerical source. The model artifact
contains exactly `weights.pt`, `model.json`, and `vocabulary.json`; tensor shapes
encode architecture dimensions and `model.json` contains only the two inference
temperatures. The loader supports only the selected M1 model.

## Reproduction

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml

conda run -n ms2lda-neural python \
  scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets

conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/new-run
```

Use `status --run /path/to/new-run` for progress. A run directory is bound to
its resolved data root. An interrupted neural fit restarts from deterministic
initialization; completed stages may be reused only with that bound input path.

## Verification

```bash
black --check benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py

ruff check --select E,F,I benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py

pytest -q benchmarks/neural_ms2lda/tests
python scripts/generate_neural_ms2lda_report.py
```

Also run the production regression suite with `NUMBA_DISABLE_JIT=1` and the four
documented frozen-upstream exclusions, build and inspect the wheel to confirm
that research code is excluded, import the installed wheel outside the
checkout, compile the LaTeX report deterministically, and visually inspect every
rendered PDF page.
