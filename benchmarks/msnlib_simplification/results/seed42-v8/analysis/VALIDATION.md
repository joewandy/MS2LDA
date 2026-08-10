# Validation review

Status: **Ready to share as the model-form recommendation for this completed study.**

## Evidence checks

- The result verifier reports 226 required artifacts, zero missing, 10 arms, four budgets, two representations, and two splits.
- All four neutral report hashes match `report/complete.json`; all 53 frozen source files match the frozen code manifest.
- The analysis notebook executes top to bottom without errors and reproduces the 80 factorial metric rows, 160 chemical rows, and 480 provided scaffold-bootstrap rows.
- The recommendation is selected on validation and checked once on test.
- Exact chemical denominators are retained for both association modes; SOS is never interpreted without eligible-topic coverage.
- Topic activity is checked directly: the recommended form uses 351 active topics on validation and 352 on test versus 354 for the current form, with a median of 12 active topics per spectrum in both forms.

## Recomputed recommendation checks

The proposed `dreams_prior__topic_direct` form at one VB step passes all seven frozen preservation rules on both splits.

- Validation completion NLL: -0.780% relative to current; test: -0.816%.
- Validation fifth-percentile cosine delta: -0.0027; test: -0.0019.
- Validation dominant-topic SOS delta: +0.0001; test: -0.0014.
- Validation high-confidence SOS delta: +0.0140; test: -0.0054.
- High-confidence eligible topics: 36 versus 37 on validation and 55 versus 55 on test.

The cross-budget paired scaffold-group bootstrap also supports the comparison. Mean per-document NLL delta is -0.0705 with a 95% interval of [-0.0768, -0.0643] on validation and -0.0738 [-0.0785, -0.0690] on test. Mean cosine delta is small: -0.00142 [-0.00243, -0.00040] on validation and -0.00056 [-0.00123, +0.00018] on test.

## Remaining interpretation boundaries

- The simplification removes DreaMS from inference but not from topic discovery.
- The direct encoder is distilled from 50-step VB targets and retains one local VB correction; it is not a VB-free neural model.
- Simplification-arm latency is warm model-only latency. Count-only end-to-end timing remains a focused implementation check.

## Artifact QA

The portable report passed canonical artifact validation, self-contained packaging, desktop and narrow-width browser verification, source-dialog interaction, overflow checks, and no-network checks.
