# Real-MSnLib gated ETM experiment log

Evidence boundary: validation only. Candidate test theta, completion matrices,
chemistry, MAG/SOS, and result artifacts must not be opened, loaded, computed,
or summarized during this campaign.

## 1. Frozen balanced ETM plus detached shared-geometry gate

- **Hypothesis:** training balanced fixed-SGNS ETM with detached evidence from
  the same SGNS/topic geometry will reduce diffuse document-topic assignment,
  improve topic utilization and chemical breadth, and retain balanced ETM's
  broad MAG-optimizable beta inventory without importing another M1 mechanism.
- **Exact architectural/config change:** relative to the previously trained
  fragment/loss-balanced ETM, compute the normalized count-weighted document
  vector `u = normalize(x @ rho)`, then
  `g = softmax(2 * cosine(u, alpha) / 1.0)`, and use
  `normalize(theta_ETM * stopgrad(g) ** 1.0)` for every training and inference
  reconstruction. The ETM encoder, posterior, KL, fixed train-only 48D SGNS,
  50/50 fragment/loss beta, K=1000, seed, optimizer, batch size, and 120 epochs
  remain locked.
- **Why justified:** the synthetic three-seed screen improved held-out NLL,
  true beta recovery, topic use, and beta redundancy while preserving true
  theta recovery. The prior real balanced ETM already repaired optimized beta
  coverage (911 motifs), leaving diffuse theta and weak chemistry as the
  measured failure.
- **Training command:**
  `conda run --no-capture-output -n ms2lda-neural python -m scripts.run_msnlib_model_comparison train --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --method etm_balanced_gated --device cpu --etm-epochs 120 --etm-batch-size 256 --gate-temperature 1.0 --gate-gamma 1.0`
- **Chemical command:**
  `conda run --no-capture-output -n ms2lda-msnlib-mag python -m scripts.run_msnlib_model_comparison chemical --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --data-root /Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680 --method etm_balanced_gated_t1_g1`
- **Result:** pending the source preflight and full validation run.
- **Comparison with prior model:** pending.
- **Decision:** run this frozen candidate before any gate-strength or additional
  mechanism experiment.

## Follow-up rule

No follow-up is preselected. After experiment 1, any additional validation-only
run must record the observed failed metric and the one-dimensional intervention
it motivates before training begins. Separation remains conditional on a clear
real duplicate-component failure; NPMI remains conditional on good inventory
and assignment accompanied by weak SOS/coherence.
