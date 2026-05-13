# HOT_PATH_CYPHER_COMPLETION.md — Completing Real Cypher in the Hot Path

**Date**: 2026-05-08  
**Context**: Builds directly on `UNIFIED_SOA_SURFACE_PLAN.md` and the canonical SoA surface (`BindSpace` + `MetaWord` + `cycle_fingerprint` + `OrchestrationBridge`).

---

## Current State (Problem)

The hot-path Cypher support currently lives in:

**`crates/cognitive-shader-driver/src/cypher_bridge.rs`**

It is explicitly a **Phase 1 stub**:
- Only does naive `starts_with("CREATE")` / `starts_with("MATCH")` classification.
- Routes via `OrchestrationBridge` for `lg.cypher` step types.
- Marked as temporary until real parser wiring.
- All real parsing logic is deferred because pulling the full `lance-graph` parser + DataFusion dependencies would bloat the crate.

This means the **hot path** (`cognitive-shader-driver` + `BindSpace` + `CausalEdge64`) currently has almost no meaningful Cypher capability, while the cold path (`datafusion_planner`) has more mature support. This is a major gap in the architecture.

---

## Target State (Vision)

**Real Cypher (and eventually other languages) should be supported in the hot path** with these properties:

- Routed through the **canonical** `OrchestrationBridge` + `UnifiedStep`.
- Parsed intent is mapped into the **canonical SoA surface** (`BindSpace` + `MetaWord` + `CausalEdge64`).
- Minimal dependency footprint on the hot path (avoid pulling full DataFusion/parser into `cognitive-shader-driver` if possible).
- Evolutionary: Existing lab consumers continue to work via adapters.
- Foundation for the **3-byte polyglot tag** vision (one compact representation that can represent Cypher, Gremlin, GQL, SQL, NARS, etc.).

---

## Proposed Approach (Debt-Free & Evolutionary)

### Phase 2 of `cypher_bridge` — Incremental Completion

**Guiding Principle**:  
Do **not** embed a full Cypher parser inside `cognitive-shader-driver`. Instead:

1. Keep the bridge lightweight.
2. Use a **pluggable parser** or **parse result DTO** from `lance-graph-contract` or a thin dependency.
3. Map the parse result into the canonical surface (`BindSpace` operations + `CausalEdge64` creation).

#### Step 2.1 — Define a Canonical Parse Result DTO (in `lance-graph-contract`)

Create a small, stable type such as:

```rust
// In lance-graph-contract (new or extended module)
pub struct CypherParseResult {
    pub operation: CypherOperation,      // Create, Match, Merge, etc.
    pub patterns: Vec<Pattern>,          // Simplified pattern representation
    pub filters: Vec<Filter>,            // WHERE clauses (simplified)
    pub projections: Vec<Projection>,    // RETURN items
    pub causal_hints: Option<CausalHints>, // Optional mapping hints to CausalMask
}
```

This DTO lives in the contract (zero heavy deps) and can be produced by different parser implementations.

#### Step 2.2 — Pluggable Parser Hook

In `CypherBridge`, add an optional parser trait:

```rust
pub trait CypherParser: Send + Sync {
    fn parse(&self, query: &str) -> Result<CypherParseResult, CypherParseError>;
}
```

Default implementation in Phase 2 can still be the stub, while a real implementation (from `lance-graph` or a dedicated small parser crate) can be injected later.

#### Step 2.3 — Map Parse Result → Canonical SoA Surface

Inside the bridge (or a helper):

```rust
fn apply_to_bindspace(
    &self,
    parse_result: &CypherParseResult,
    bindspace: &mut BindSpace,
    causal_edge: &mut CausalEdge64,
) {
    // Example mappings:
    match parse_result.operation {
        CypherOperation::Match => {
            causal_edge.set_causal_mask(CausalMask::SO); // Association level
            // Populate BindSpace rows / fingerprints based on patterns
        }
        CypherOperation::Create => {
            causal_edge.set_inference_type(InferenceType::Induction);
            // Create new entities in BindSpace
        }
        // ...
    }

    // Apply causal hints if present
    if let Some(hints) = &parse_result.causal_hints {
        causal_edge.apply_causal_hints(hints);
    }
}
```

This keeps the mapping logic clean and directly leverages the unified SoA surface.

---

## Connection to Other Threads

| Thread | How Hot-Path Cypher Helps / Depends |
|--------|-------------------------------------|
| **Unified SoA Surface** | Cypher completion is built **on top of** the canonical surface defined in `UNIFIED_SOA_SURFACE_PLAN.md`. |
| **3-byte Polyglot Tag** | Parsed Cypher can be reduced to (or enriched by) the compact tag. The tag becomes a fast-path representation of the parse result. |
| **8-Mask Superposition** | Cypher queries can carry explicit `CausalMask` preferences (e.g., interventional vs counterfactual reasoning). |
| **Thinking Styles** | Certain Cypher patterns can influence or be influenced by `AgentStyle` (e.g., exploratory MATCH vs precise CREATE). |
| **JC Mathematical Proofs** | The `jc` crate already validates Pearl 2³ behavior — hot-path Cypher should respect and leverage these guarantees. |

---

## Recommended Short Sprint Scope (Documentation + Planning First)

Since we are staying inside `.grok/` for now, the immediate deliverables are:

1. This plan document (done).
2. Draft `CypherParseResult` + `CypherParser` trait definitions (as markdown examples in this file or a follow-up).
3. Clear mapping rules from common Cypher constructs to `CausalEdge64` fields.
4. Update to `INTEGRATION_PLANS.md` and `LATEST_STATE.md` (already partially done via previous updates).

**Implementation sprint** (when ready) could be:
- Add `CypherParseResult` to `lance-graph-contract`.
- Implement a minimal real parser or adapter in `cognitive-shader-driver`.
- Wire the mapping logic into `CypherBridge::route()`.
- Add tests using the canonical surface.

---

## Open Questions (for Discussion)

- Should the real Cypher parser live in `lance-graph` and be called via a trait, or should we create a small dedicated `cypher-parser` crate?
- How much of the parse tree should be preserved vs simplified into the `CypherParseResult` DTO?
- Should `causal_hints` in the parse result be optional or always populated based on query structure?

---

**This plan is designed to be incremental, respect the canonical SoA surface, and avoid introducing new technical debt.** It positions hot-path Cypher as a first-class citizen of the resonance-based cognitive system rather than a bolted-on feature.