# Agent W3 — causal-edge-g (LL1)

**Status:** COMPLETE
**File:** `crates/causal-edge/src/edge.rs`
**Build:** Workspace `cargo check` exits 0

**Changes:**
- Reclaimed `Reserved5` slot → `InferenceType::Intervention` (do-calculus, Pearl rung 2)
- Reclaimed `Reserved6` slot → `InferenceType::Counterfactual` (Pearl rung 3)
- Updated `from_bits()` decoder accordingly
- Doc comments cite Pearl's do-operator + reference `CausalEdge64::counterfactual_ready` for the confidence gate

**Notes:** Agent's textual task-notification was off-task (permission-pattern audit) but its actual code modification landed correctly. Backfilled by main thread. No bit-layout change beyond renaming Reserved slots; binary compat preserved.
