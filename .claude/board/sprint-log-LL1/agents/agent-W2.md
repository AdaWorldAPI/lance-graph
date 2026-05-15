# Agent W2 — nars-engine-dispatch (PR-LL-1)

**File:** `crates/lance-graph-planner/src/cache/nars_engine.rs`

## (a) Mask weights assigned

- **`intervention_style()`** — weights `[0.0, 0.0, 0.15, 0.15, 0.0, 0.05, 0.50, 0.15]`. MASK_PO (index 6) = 0.50 highest, reflecting do-calculus severs the subject confounding plane.
- **`counterfactual_style()`** — weights `[0.0, 0.05, 0.05, 0.10, 0.0, 0.05, 0.25, 0.50]`. MASK_SPO (index 7) = 0.50 highest, all-planes chain required for rung-3 reasoning.
- **`inference_to_pearl_mask()`** — explicit dispatch: `Intervention → MASK_PO`, `Counterfactual → MASK_SPO`, all others → MASK_SPO (conservative default).
- **`nars_infer()`** — added Intervention (Abduction ×0.85) and Counterfactual (Deduction ×0.70) arms.
- **`to_causal_edge()`** — explicit match mapping local inference bytes 7→Intervention, 8→Counterfactual to protocol enum.

## (b) TODOs left for future tuning

- All style weights marked TUNED-LATER; replace after PR-LL-4 GRPO training data.
- `inference_to_pearl_mask` fallthrough for Deduction/Induction/Abduction defaults to SPO; per-type masks to tune in PR-LL-4.
- `nars_infer` Intervention/Counterfactual arms are Abduction/Deduction proxies; replace with dedicated do-calculus truth functions.

## (c) Cargo check result

`rustup run 1.95.0 cargo check -p lance-graph-planner --features default` — exit 0, no warnings.
