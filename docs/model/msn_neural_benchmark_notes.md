# MSn Neural Topic Model Benchmark Notes

## Goal

This branch keeps a reproducible MSn benchmark harness for testing topic models
against the motif-substructure quality metric. The useful output from any model
is the same:

- `theta.npy`: spectra by motif memberships.
- `beta.npy`: motif by fragment/loss-word distributions.
- `vocab.json`: the fragment/loss-word vocabulary matching `beta`.

Those files can be exported to `annotations.csv` and `memberships.csv`, then
scored with `evaluate_motif_substructure_quality.py`.

## Pipeline

Prepare model-independent input once:

```bash
conda run -n MS2LDA_v2 python scripts/prepare_msn_benchmark_input.py \
  --dataset datasets/Corinna_Library_filtered_positive.mgf \
  --out-dir notebooks/Paper_results/MSn_evaluation/msn_cache_full \
  --overwrite
```

Train a model:

```bash
conda run -n MS2LDA_v2 python scripts/run_msn_topic_model_experiment.py \
  --input-cache notebooks/Paper_results/MSn_evaluation/msn_cache_full \
  --model lda \
  --out-dir notebooks/Paper_results/MSn_evaluation/lda_full_1000 \
  --n-motifs 1000 \
  --overwrite
```

Export and score:

```bash
conda run -n MS2LDA_v2 python scripts/export_msn_model_outputs.py \
  --input-cache notebooks/Paper_results/MSn_evaluation/msn_cache_full \
  --model-dir notebooks/Paper_results/MSn_evaluation/lda_full_1000 \
  --max-eval-motifs 0 \
  --overwrite

conda run -n MS2LDA_v2 python scripts/evaluate_motif_substructure_quality.py \
  notebooks/Paper_results/MSn_evaluation/lda_full_1000/annotations.csv \
  notebooks/Paper_results/MSn_evaluation/lda_full_1000/memberships.csv \
  --out-dir notebooks/Paper_results/MSn_evaluation/lda_full_1000
```

Optional Stage 2 encoder experiment:

```bash
conda run -n MS2LDA_v2 python scripts/run_msn_stage2_encoder_experiment.py \
  --input-cache notebooks/Paper_results/MSn_evaluation/msn_cache_2000_rescue \
  --teacher-model-dir notebooks/Paper_results/MSn_evaluation/neural_lda_2000_s64_u20_w1_p3 \
  --out-dir notebooks/Paper_results/MSn_evaluation/stage2_encoder_2000_s64_p3 \
  --overwrite
```

## Script Reference

`scripts/msn_benchmark_pipeline.py` is the shared implementation module. Do not
run it directly. It contains the cache loader, BoW construction, LDA/NMF/neural
training functions, motif export helpers, and Spec2Vec annotation wrapper used
by the command-line scripts below.

`scripts/prepare_msn_benchmark_input.py` cleans the MSn MGF and writes the
model-independent cache:

```text
bow.npz
vocab.json
spectra_metadata.csv
documents.jsonl.gz
cache_summary.json
```

Run this once per dataset or subset. Use `--limit-spectra` for smoke tests.

`scripts/run_msn_topic_model_experiment.py` trains one model from either an MGF
or a prepared cache. Prefer `--input-cache` for reproducible runs. Supported
models are `lda`, `nmf`, `sparse-neural`, and `neural-lda`. It writes:

```text
theta.npy
beta.npy
vocab.json
train_history.json
run_summary.json
model_checkpoint.pt
```

`scripts/export_msn_model_outputs.py` converts a model directory containing
`theta.npy`, `beta.npy`, and `vocab.json` into benchmark input files:

```text
annotations.csv
memberships.csv
topic_diagnostics.csv
export_summary.json
```

Use `--max-eval-motifs 0` to export every motif with nonzero membership.

`scripts/evaluate_motif_substructure_quality.py` scores `annotations.csv` and
`memberships.csv`. It writes per-motif scores, associated molecules, and a JSON
summary containing coverage, mean SoS, and QAC.

`scripts/run_msn_stage2_encoder_experiment.py` trains an encoder to imitate an
existing Stage 1 model's `theta.npy` using the fixed Stage 1 `beta.npy`. Use it
only after a Stage 1 run exists. It writes another model directory with
predicted `theta.npy`, copied `beta.npy`, validation metrics, and checkpoint
files.

## Models Kept

`lda` is the tomotopy baseline and should remain the reference point.

`nmf` is a simple KL-NMF baseline with the same `theta`/`beta` output contract.

`sparse-neural` is the original amortized encoder attempt. It is useful as a
negative control because it tends to collapse to too few active motifs on the
full MSn data.

`neural-lda` directly optimizes trainable document-topic and topic-word
parameters:

```text
X ~= theta @ beta
```

This is closer to regularized pLSA or KL-NMF than to standard LDA. It is neural
only in the sense that the optimization is implemented in PyTorch; it does not
yet provide inference for unseen spectra.

`run_msn_stage2_encoder_experiment.py` trains an encoder after Stage 1 so that
new spectra can be mapped to the learned motif space. This is the right place
to test whether a neural method adds value through fast inference.

## Current Results

The full-data LDA baseline trained and scored successfully:

| Model | Spectra | Motifs | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LDA full | 38,888 | 1,000 | 0.832 | 0.6235 | 0.5188 |

On the 2000-spectrum subset, the best theta-sharpened `neural-lda` run roughly
tied LDA at the standard `0.5` membership threshold:

| Model | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: |
| LDA 2k | 0.592 | 0.5459 | 0.3232 |
| neural-lda 2k | 0.604 | 0.5344 | 0.3228 |

The Stage 2 encoder did not yet generalize well. A dense BoW MLP overfit the
training spectra and reached only about `0.587` validation theta cosine against
the Stage 1 teacher. The token-set encoder failed to learn useful validation
theta assignments in the small smoke run.

## Interpretation

The benchmark harness is useful. The current neural models are not yet a clear
improvement over tomotopy LDA.

The main open problem is not producing `theta`/`beta` for the training spectra.
The hard part is learning stable, reusable motif structure and then inferring
good motif memberships for unseen spectra without collapsing to a few generic
topics.
