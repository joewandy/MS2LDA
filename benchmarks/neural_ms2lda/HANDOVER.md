# Neural MS2LDA handover

## Current state

The repository supports one neural MS2LDA architecture: seed 42, K=1000, 40
epochs, and six CPU threads. It is a single-dataset, single-seed research result,
not evidence that the production Tomotopy backend should be replaced.

The model was selected by an ordered validation-only simplification study. The
accepted control was loaded and reevaluated without retraining. Candidates never
received test matrices or records. An initial lock replay exposed
multithreaded-gradient nondeterminism; strict deterministic PyTorch execution
was then enabled before the corrected candidate and independent lock runs.
After the architecture was frozen, the final neural test workflow was executed
once. Tomotopy was neither retrained nor reevaluated during selection or final
confirmation.

## Implemented model

1. Train 48-dimensional SGNS token coordinates using training spectra only;
   append fragment/loss indicators and normalize the resulting 50-dimensional
   vectors.
2. Project tokens into normalized 128-dimensional space and initialize 1,000
   topic prototypes from a seed-42 uniform sample of distinct projected tokens.
3. Apply one identity-initialized, bias-free 128-by-128 linear map to each
   count-weighted leave-one-out document context and add the correction to the
   token vector.
4. Add whole-spectrum prototype evidence, retain and renormalize the top two
   token assignments, and aggregate their count-weighted mass.
5. Multiply routed topic mass by detached whole-spectrum evidence raised to
   0.75. Empty spectra receive an explicit uniform mixture.
6. Decode with cosine logits and separate fragment/loss softmax operations,
   assigning exactly half of each topic's probability mass to each channel.
7. Train on full spectra with alternating router and topic blocks. Retain
   Sinkhorn balancing, positive-NPMI regularization, prototype separation,
   routing-temperature annealing, deterministic PyTorch algorithms, finite-loss
   checks, and gradient clipping.

There are no paired views, Fourier mass coordinates, Jensen-Shannon consistency,
local reconstruction, dead-topic recycling, k-means++ initialization, adaptive
channel weighting, context MLP, joint-optimization alternative, architecture
flags, or loader compatibility paths.

## Authoritative evidence

| Purpose | Location |
| --- | --- |
| Mathematical and scientific description | `docs/research/neural_ms2lda_report.tex` |
| Reviewed compiled report | `docs/research/neural_ms2lda_report.pdf` |
| Fixed study constants | `benchmarks/neural_ms2lda/protocol.json` |
| Model and one-pass inference | `benchmarks/neural_ms2lda/model.py` |
| Training objectives | `benchmarks/neural_ms2lda/objectives.py` |
| Alternating updates | `benchmarks/neural_ms2lda/training.py`, `optimization.py` |
| Canonical model | `benchmarks/neural_ms2lda/results/seed42/trained_model/` |
| Paper-facing evidence | `benchmarks/neural_ms2lda/results/seed42/results.json` |
| Validation ablation ledger | `benchmarks/neural_ms2lda/results/seed42/ablation_results.json` |

`results.json` is the only numerical source for the paper. The trained model
contains exactly `weights.pt`, `model.json`, and `vocabulary.json`; the loader
supports only this architecture. The ledger contains validation experiment
outcomes and no test results.

## Reproduction

The public Zenodo inputs may live at any path. Create the environments, acquire
the data, and run:

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
that research code is excluded, import the installed wheel outside the checkout,
compile the LaTeX report deterministically, and visually inspect every rendered
PDF page.
