# Agent W4 — arigraph-intervene (LL1)

**Status:** COMPLETE
**File:** `crates/lance-graph/src/graph/arigraph/triplet_graph.rs`
**Build:** Workspace `cargo check` exits 0

**Changes:**
- Added `pub enum ContextTag` with variants `Observation` (default) and `Intervention` (Pearl rung 2 marker)
- Added `pub struct CounterfactualSpoG { triplet, context }` — caller-owned value representing the substituted triple
- Added `TripletGraph::intervene_on(subject, predicate, new_object)` method producing a `CounterfactualSpoG` with `ContextTag::Intervention`
- Original graph NOT mutated (read-only borrow on self)
- Sentinel byte `0xFF` in `raw_g` field documented as a placeholder until contract-level G enum lands

**Notes:** Agent completed silently (no explicit task-notification arrived but file diff confirms work). Backfilled report by main thread.
