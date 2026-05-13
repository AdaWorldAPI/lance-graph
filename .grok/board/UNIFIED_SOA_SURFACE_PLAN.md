# UNIFIED_SOA_SURFACE_PLAN.md — Canonical SoA DTO Unification (Evolutionary, Debt-Free)

**Date**: 2026-05-08  
**Author**: Grok intuition pass  
**Constraint**: Everything stays inside `.grok/`. No code changes proposed outside documentation and planning.

---

## Problem Statement (High Signal)

The current SoA DTO surface is **fragmented**, which is the largest remaining source of architectural entropy and technical debt:

- `cognitive-shader-driver` has excellent raw substrate (`BindSpace`, column design, `MetaColumn` prefilter, native `CausalEdge64` storage, `cycle_fingerprint`).
- `lance-graph-contract::cognitive_shader` defines intended canonical DTOs (`ShaderDispatch`, `MetaWord`, `ShaderSink`, etc.).
- Wire DTOs exist in parallel (lab-only but easy to accidentally extend).
- Historical DTOs and bridges are scattered.
- No single, crisp, **owned** canonical surface that new consumers (or future unification efforts like the 3-byte polyglot tag) can reliably target.

This fragmentation makes the hot path harder to reason about, extend, and trust — even though the underlying primitives (`CausalEdge64`, `BindSpace`, `OrchestrationBridge`) are already strong.

---

## Core Intuition (Best Move)

**Do not flatten or delete anything.**

Instead, **explicitly own and strengthen** the already-existing canonical surface:

> **`BindSpace` columns + `MetaWord` + `cycle_fingerprint` (Vsa16kF32) + `OrchestrationBridge` = The Stable Canonical SoA Surface**

Everything else becomes either:
- A **view** over this surface, or
- A **thin adapter** into it.

This is evolutionary, low-risk, and directly reduces entropy without introducing new debt.

---

## Proposed Canonical Surface (Definition)

**Stable Core (to be declared canonical)**:

| Concept                    | Type / Location                          | Role                                      | Stability |
|---------------------------|------------------------------------------|-------------------------------------------|---------|
| `BindSpace`               | `cognitive-shader-driver`                | Primary SoA substrate                     | High |
| `FingerprintColumns`      | `cognitive-shader-driver::bindspace`     | Named planes (content, cycle, topic, angle, sigma) | High |
| `EdgeColumn`              | `cognitive-shader-driver::bindspace`     | Native storage for `CausalEdge64`         | High |
| `MetaColumn` + `MetaWord` | `cognitive-shader-driver` + contract     | Cheap packed prefilter (thinking + nars + awareness) | High |
| `cycle_fingerprint`       | Emitted per cycle via `BindSpace`        | Unit of thought (Vsa16kF32 carrier)       | High |
| `OrchestrationBridge`     | `lance-graph-contract`                   | Dispatch / routing contract               | High |
| `UnifiedStep`             | `lance-graph-contract`                   | Canonical step representation             | High |

**Rules**:
- New consumers should target the above.
- Wire DTOs (`wire.rs` etc.) remain strictly behind `serve`/`grpc` features.
- Any new abstraction must provide an adapter into the above core.

---

## Evolutionary Unification Steps (No New Debt)

### Phase 0 – Declaration (Documentation Only)

1. Create clear ownership statement in `cognitive-shader-driver/lib.rs` and `lance-graph-contract` docs:
   - “The canonical SoA surface is `BindSpace` + `MetaWord` + `cycle_fingerprint`. All long-term consumers should target this surface.”

2. Add a small `soa` module (or section) in `lance-graph-contract` that re-exports the stable types with explicit “canonical” labeling.

3. Update `.grok/` docs (this file + `LATEST_STATE.md` + `TECH_DEBT.md`) to reflect the new canonical surface.

**Outcome**: Clarity without any code behavior change.

### Phase 1 – Thin Adapters (Short Sprint, Low Risk)

- Where real consumers currently use Wire DTOs or historical types, add **small, zero-cost adapter traits or conversion functions** that map into the canonical `BindSpace` / `MetaWord` surface.
- These adapters live behind feature gates or in a dedicated `adapters` module.
- No existing code is forced to change.

**Example direction** (illustrative only):
```rust
// Conceptual only — stays in planning for now
pub trait IntoCanonicalSoa {
    fn into_bindspace_row(&self) -> BindSpaceRowView;
}
```

### Phase 2 – Enforcement (Later, Optional)

- Add lightweight compile-time or test-time lint/gate that discourages direct use of Wire DTOs outside lab contexts.
- This can be done incrementally after Phase 1 proves value.

---

## Benefits (Why This Reduces Debt)

- **Reduces fragmentation** without breaking anything.
- Makes the hot path (`cognitive-shader-driver` + `CausalEdge64`) easier to extend (e.g. for the 3-byte polyglot tag and 8-mask superposition).
- Strengthens the existing `OrchestrationBridge` contract instead of creating parallel ones.
- Provides a clean target for future work (compact query languages, Thinking Style modulation of `CausalEdge64`, Promotion Membrane).

---

## Draft Canonical Surface Declaration (Ready for Use)

This section can be copied almost verbatim into `cognitive-shader-driver` documentation or `lance-graph-contract` when implementation begins.

### Canonical SoA Surface (v1)

**The stable, long-term SoA DTO surface is defined as:**

- `BindSpace` (the primary Struct-of-Arrays substrate)
- `FingerprintColumns` (content, cycle as Vsa16kF32, topic, angle, sigma)
- `EdgeColumn` (native storage for `CausalEdge64`)
- `MetaColumn` / `MetaWord` (packed u32 prefilter: thinking + awareness + nars_f/c + free_e)
- `QualiaColumn` (18D qualia)
- `cycle_fingerprint` (the emitted unit of thought per cognitive cycle, Vsa16kF32 carrier)
- `OrchestrationBridge` + `UnifiedStep` (the dispatch and routing contract)

**Rules for this surface:**
1. New consumers and new high-level features (polyglot tags, superposition, Promotion Membrane, etc.) should target this surface.
2. `cognitive-shader-driver` owns the implementation of `BindSpace` and column behavior.
3. `lance-graph-contract` owns the public DTO contracts and the `OrchestrationBridge` trait.
4. Wire DTOs (`wire.rs` and related) are **lab-only** and must route through the canonical surface. They are not part of the stable API.
5. Any new abstraction must provide a conversion or adapter path into the above core.

**Non-goals of this surface (to prevent scope creep):**
- Full query language parsing (Cypher, Gremlin, etc.)
- Tensor / superposition operations
- JIT kernel generation
- Specific domain schemas (SMB, MedCare, etc.)

These remain consumers or extensions of the canonical surface.

---

## Migration & Adapter Examples (Markdown Only)

These are illustrative patterns only. They show how existing or future consumers can move to the canonical surface without breaking changes.

### Example 1: Wire DTO → Canonical Surface (Adapter Pattern)

```rust
// Conceptual adapter — lives behind `serve` or `grpc` feature only
impl From<WireTensorView> for BindSpaceRowView {
    fn from(wire: WireTensorView) -> Self {
        // Decode once at ingress (per existing Rule F)
        let decoded = wire.decode();
        BindSpaceRowView::from_decoded(&decoded)
    }
}
```

**Benefit**: Existing lab code continues to work. New code can target `BindSpaceRowView` directly.

### Example 2: Historical / Scattered DTO → Canonical

```rust
// Adapter for older consumer types
pub trait AsCanonicalSoa {
    fn as_bindspace(&self) -> Option<&BindSpace>;
    fn as_meta_word(&self) -> Option<MetaWord>;
    fn cycle_fingerprint(&self) -> Option<&[f32]>;
}
```

**Usage**:
```rust
if let Some(meta) = old_dto.as_meta_word() {
    // Use the cheap prefilter
    if filter.accepts(meta) {
        // Only then load full fingerprint data
    }
}
```

### Example 3: New Feature (e.g. 3-byte Polyglot Tag) Built on Canonical Surface

```rust
// New compact tag lives alongside existing fields
pub struct PolyglotQueryTag {
    pub language: u8,      // 0=Cypher, 1=Gremlin, 2=GQL, ...
    pub operation: u8,
    pub causal_hint: u8,   // maps to CausalMask bits
}

impl PolyglotQueryTag {
    pub fn apply_to_causal_edge(&self, edge: &mut CausalEdge64) {
        // Set CausalMask / InferenceType based on tag
    }
}
```

This keeps the new feature small and cleanly layered on top of `CausalEdge64` + `MetaWord`.

---

**These patterns are deliberately thin and reversible.** They demonstrate how unification can be done incrementally without forcing changes on existing code.
- Aligns with the strong Canonical vs LAB-ONLY boundary already documented in `cognitive-shader-driver`.

---

## Connection to Other High-Signal Threads

| Thread | How This Plan Helps |
|--------|---------------------|
| **8-Mask Superposition** | Gives a stable substrate (`BindSpace` + `CausalEdge64`) to operate on. |
| **Compact 3-byte Polyglot Tag** | The tag can live inside `MetaWord` or as a small extension on `cycle_fingerprint`. |
| **Hot Path Cypher Completion** | Parsed Cypher intent can be mapped into `CausalEdge64` fields via the canonical surface. |
| **Technical Debt Reduction** | Directly attacks the #1 current debt item (fragmented SoA DTO surface). |

---

## Recommended Short Sprint Scope (Stays Inside .grok for Planning)

**Sprint Goal**: Declare and document the canonical SoA surface + provide minimal adapter guidance.

**Deliverables (all inside `.grok/` or as planning artifacts)**:
1. This plan document (done).
2. Updated `LATEST_STATE.md` and `TECH_DEBT.md` entries.
3. Draft “Canonical SoA Surface” section ready to be copied into `cognitive-shader-driver` docs.
4. Simple migration/adapter examples (as markdown, not code yet).

**Estimated effort**: 1 focused session for planning + documentation. Implementation (if approved) can be a small, safe follow-up sprint.

---

## Next Actions (Inside .grok Only)

1. Update `boot.md` “Next Session Starting Point” to reflect this as current highest-signal thread.
2. Update `LATEST_STATE.md` and `TECH_DEBT.md` with references to this plan.
3. (Optional) Create a short “Canonical SoA Surface – Draft Declaration” markdown that can be used as the source of truth for future implementation.

---

**This plan is deliberately conservative, evolutionary, and debt-reducing.** It treats existing working code with respect while creating clarity and a clean target for the next wave of work (superposition, polyglot tags, hot-path query languages).

Ready to proceed with any of the next actions above. Just say which one.