# PR-B-1: ContextBundle as typed OGIT surface

**Sprint:** Sprint-3 (12+meta CCA2A)
**Worker:** W3
**Tech-debt ticket:** TD-CONTEXT-BUNDLE-2
**Pattern:** B — ContextBundle per G (typed surface)
**Status:** DESIGN PHASE
**Branch:** `claude/tier-1-implementation-specs`
**Foundation PR:** No upstream PR dependencies. PR-A-1, PR-C-1, PR-D-1, PR-E-1, PR-F-1, PR-J-1 all depend on this.

---

## 1. Goal

Introduce `ContextBundle` — the **typed OGIT surface** that consumers see when they look up a `G` (ontology generation/namespace). Today, ontology context is scattered across ad-hoc lookups (codebooks here, vocabularies there, thinking-style hints in HashMaps). PR-B-1 collapses all of this into a single named-slot struct, registered in `OntologyRegistry` and resolved by `G: u32`.

This is the **foundation** for Pattern B; downstream PRs (D = OWL hydrators, C = ConsumerPointer bridge, A = SPO routing) all read/write through this surface.

### Non-goals

- Hydrating slots from real OWL/SHACL sources (that's PR-D-1 OwlHydrator).
- Implementing ConsumerPointer behavior (that's PR-C-1 GenericBridge).
- Defining the SPO `g: u32` slot itself (that's PR-A-1; this PR consumes the same `u32`).
- Migrating existing call sites off legacy lookups (that's PR-J-1).

---

## 2. Files to touch

| Path | Status | Purpose |
| --- | --- | --- |
| `crates/lance-graph-ontology/src/context_bundle.rs` | NEW (~150 LOC) | `ContextBundle` struct + 12 slot type stubs |
| `crates/lance-graph-ontology/src/registry.rs` | EDIT (+~30 LOC) | Add `resolve(g)` and `register(bundle)` to `OntologyRegistry` |
| `crates/lance-graph-ontology/src/lib.rs` | EDIT (+2 LOC) | `pub mod context_bundle; pub use context_bundle::*;` |
| `crates/lance-graph-ontology/tests/context_bundle_resolve.rs` | NEW | Register + resolve smoke test |
| `crates/lance-graph-ontology/tests/context_bundle_inherits.rs` | NEW | Inheritance chain test |
| `crates/lance-graph-ontology/tests/context_bundle_thinking_styles_smallvec.rs` | NEW | SmallVec inline-storage test |

Total: ~200 LOC + 3 tests. ~1 day.

---

## 3. API sketch

### 3.1 ContextBundle struct (12 named slots)

```rust
// crates/lance-graph-ontology/src/context_bundle.rs
use std::sync::Arc;
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Typed OGIT surface keyed by ontology generation `G`.
///
/// Every consumer that needs ontology context for a given `G` resolves a
/// `&ContextBundle` from `OntologyRegistry::resolve(g)`. The bundle exposes
/// 12 named slots; most are `Option<Arc<…>>` because they are populated
/// lazily by Pattern D hydrators.
pub struct ContextBundle {
    pub g: u32,
    pub version: u32,
    pub domain_name: SmolStr,
    pub inherits_from: Option<u32>, // parent G (DOLCE = root, G=0)

    // OWL overlay (5 slots, populated by Pattern D OwlHydrator)
    pub ontology: Option<Arc<OntologySlot>>,
    pub codebook: Option<Arc<CodebookSlot>>,
    pub schema: Option<Arc<SchemaSlot>>,
    pub labels: Option<Arc<LabelsSlot>>,
    pub vocabulary: Option<Arc<VocabularySlot>>,

    // Operational behavior (Pattern C ConsumerPointer)
    pub consumer_pointer: Option<Arc<ConsumerPointer>>,

    // Per Pattern G (best-practice thinking inheritance)
    pub thinking_styles: SmallVec<[u8; 8]>,
    pub thinking_adjacency: Option<Arc<AdjacencyStore<u8>>>,
    pub qualia_codebook: Option<Arc<QualiaCodebook>>,

    // Per-G specialization
    pub mul_threshold_profile: Option<MulThresholdProfile>,
    pub trust_texture_set: SmallVec<[u8; 4]>,
    pub flow_state_set: SmallVec<[u8; 4]>,
}
```

**Slot count check:** 1=ontology, 2=codebook, 3=schema, 4=labels, 5=vocabulary, 6=consumer_pointer, 7=thinking_styles, 8=thinking_adjacency, 9=qualia_codebook, 10=mul_threshold_profile, 11=trust_texture_set, 12=flow_state_set. (`g`, `version`, `domain_name`, `inherits_from` are bundle metadata, not "slots" in the OGIT sense.)

### 3.2 Slot type stubs (one struct per slot — full impls land in subsequent PRs)

```rust
/// OWL ontology graph for this G. Hydrated by Pattern D `OwlHydrator`.
pub struct OntologySlot { /* hydrated by Pattern D OwlHydrator */ }

/// Predicate / edge-type codebook (u16 -> SmolStr). Hydrated by codebook hydrators.
pub struct CodebookSlot { /* hydrated by codebook hydrators */ }

/// MappingRow cascade-cols for this G (per-domain SPO -> column mappings).
pub struct SchemaSlot { /* MappingRow cascade-cols per G */ }

/// Display names + alt-names for predicates / classes.
pub struct LabelsSlot { /* display names + alt-names */ }

/// Tokenizer / vocabulary used for embedding lookups in this G.
pub struct VocabularySlot { /* tokenizer per G */ }

/// Pattern C: pluggable consumer (CSE, SDU, SBT…) for this G.
/// Lives in `lance-graph-contract::consumer::ConsumerPointer` (see Q1 below).
pub struct ConsumerPointer { /* Pattern C; W4's spec details this */ }

/// CSR adjacency over thinking-style atoms (graph of which styles compose).
pub struct AdjacencyStore<T> { /* CSR over thinking-style atoms */ _marker: std::marker::PhantomData<T> }

/// References `qualia.rs::FAMILY_CENTROIDS`. Per-G qualia slice.
pub struct QualiaCodebook { /* references qualia.rs FAMILY_CENTROIDS */ }

/// Multiplication threshold profile (recall-vs-precision knobs per G).
pub struct MulThresholdProfile { /* per-G recall/precision knobs */ }
```

### 3.3 OntologyRegistry extension

```rust
// crates/lance-graph-ontology/src/registry.rs (additive)
use std::collections::HashMap;
use crate::context_bundle::ContextBundle;

pub struct OntologyRegistry {
    // existing fields …
    bundles: HashMap<u32, ContextBundle>,
}

impl OntologyRegistry {
    /// Resolve the typed OGIT surface for a given `G`. Returns `None` if no
    /// bundle has been registered (caller may fall back to inherited parent
    /// or the DOLCE root at G=0).
    pub fn resolve(&self, g: u32) -> Option<&ContextBundle> {
        self.bundles.get(&g)
    }

    /// Register a bundle. Panics if `bundle.g` is already registered (use
    /// `replace` for hot-swap during hydration).
    pub fn register(&mut self, bundle: ContextBundle) {
        let g = bundle.g;
        if self.bundles.insert(g, bundle).is_some() {
            panic!("ContextBundle for G={g} already registered; use replace()");
        }
    }

    /// Hot-swap a bundle (used by Pattern D hydrators on schema reload).
    pub fn replace(&mut self, bundle: ContextBundle) -> Option<ContextBundle> {
        self.bundles.insert(bundle.g, bundle)
    }
}
```

### 3.4 Inheritance helper

```rust
impl ContextBundle {
    /// Set-union for SmallVec slots, override for scalar slots.
    /// Codifies the inheritance semantics resolved in Q4 below.
    pub fn merge_with(&mut self, parent: &ContextBundle) {
        for s in &parent.thinking_styles {
            if !self.thinking_styles.contains(s) {
                self.thinking_styles.push(*s);
            }
        }
        for t in &parent.trust_texture_set {
            if !self.trust_texture_set.contains(t) {
                self.trust_texture_set.push(*t);
            }
        }
        for f in &parent.flow_state_set {
            if !self.flow_state_set.contains(f) {
                self.flow_state_set.push(*f);
            }
        }
        // Scalar slots: child overrides parent only if child slot is None.
        if self.qualia_codebook.is_none()       { self.qualia_codebook       = parent.qualia_codebook.clone(); }
        if self.thinking_adjacency.is_none()    { self.thinking_adjacency    = parent.thinking_adjacency.clone(); }
        if self.mul_threshold_profile.is_none() { self.mul_threshold_profile = parent.mul_threshold_profile.clone(); }
        // OWL slots are NOT inherited — each G has its own ontology surface.
    }
}
```

---

## 4. Initial seed bundles

Hand-code 3 minimal bundles for tests (full set arrives via Pattern D hydrators in PR-D-1+):

| G | name | thinking_styles | inherits_from |
| --- | --- | --- | --- |
| 0 | `dolce` | `[Analytical, Deductive]` | `None` (root) |
| 2 | `healthcare` | `[Differential, EvidenceBased]` | `Some(0)` |
| 3 | `gotham` | `[LinkAnalytic, AttributionTracing]` | `Some(0)` |

```rust
// crates/lance-graph-ontology/src/context_bundle.rs (seed helpers)
pub fn seed_dolce() -> ContextBundle {
    ContextBundle {
        g: 0, version: 1, domain_name: "dolce".into(), inherits_from: None,
        ontology: None, codebook: None, schema: None, labels: None, vocabulary: None,
        consumer_pointer: None,
        thinking_styles: SmallVec::from_slice(&[ThinkingStyle::Analytical as u8,
                                                ThinkingStyle::Deductive  as u8]),
        thinking_adjacency: None, qualia_codebook: None,
        mul_threshold_profile: None,
        trust_texture_set: SmallVec::new(),
        flow_state_set:    SmallVec::new(),
    }
}
// seed_healthcare() and seed_gotham() analogous, with inherits_from = Some(0).
```

(`ThinkingStyle` enum lives in `lance-graph-ontology::thinking`; if not yet present, use raw `u8` constants `0..=255` and let PR-G land the named enum.)

---

## 5. Test plan

| Test file | What it asserts |
| --- | --- |
| `tests/context_bundle_resolve.rs` | Register `seed_dolce()`, then `registry.resolve(0)` returns `Some(&bundle)` with `bundle.domain_name == "dolce"`. `resolve(99)` returns `None`. |
| `tests/context_bundle_inherits.rs` | After registering DOLCE + Healthcare, walking `inherits_from` from G=2 reaches G=0 in one hop. `merge_with` produces the expected style union `[Differential, EvidenceBased, Analytical, Deductive]` (order: child first, then parent extras). |
| `tests/context_bundle_thinking_styles_smallvec.rs` | A bundle with 8 thinking styles stays inline (no heap alloc); pushing the 9th forces heap promotion. Assert via `bundle.thinking_styles.spilled() == false` then `== true`. |

---

## 6. Dependencies

- **No upstream PR dependencies.** This is the foundation for Pattern B.
- **Downstream consumers:** PR-A-1 (SPO `g: u32` slot uses `resolve(g)` for routing), PR-C-1 (GenericBridge populates `consumer_pointer`), PR-D-1 (OwlHydrator populates `ontology`/`codebook`/`schema`/`labels`/`vocabulary`), PR-E-1, PR-F-1, PR-J-1.
- **External crates:** `smallvec`, `smol_str` — already in workspace.

---

## 7. Acceptance criteria

- [ ] `ContextBundle` type with all 12 slots defined in `crates/lance-graph-ontology/src/context_bundle.rs`.
- [ ] `OntologyRegistry::resolve(g) -> Option<&ContextBundle>` returns the registered bundle.
- [ ] `OntologyRegistry::register(bundle)` stores by `bundle.g`; panics on duplicate.
- [ ] `OntologyRegistry::replace(bundle)` returns the previous bundle (for hydrator hot-swap).
- [ ] `ContextBundle::merge_with(parent)` implements set-union for SmallVec slots, override-if-None for scalar slots.
- [ ] 3 hand-coded seed bundles exist (`seed_dolce`, `seed_healthcare`, `seed_gotham`).
- [ ] All 3 tests green (`cargo test -p lance-graph-ontology context_bundle`).
- [ ] Backwards-compat: existing `OntologyRegistry::enumerate(ns)` still works (no signature changes).
- [ ] `pub use context_bundle::*` re-exports the type from the crate root.

---

## 8. Effort

Small. ~200 LOC + 3 tests. ~1 day end-to-end.

---

## 9. Open questions for engineer

1. **ConsumerPointer location** — contract crate (zero-deps canonical) vs ontology crate (co-located with registry)?
   **Recommend:** `lance-graph-contract::consumer::ConsumerPointer`, since every consumer pulls contract anyway and we avoid an `ontology -> contract` reverse dep.
2. **Slot `Arc<T>` vs `Box<T>`** — Arc enables sharing across threads (ractor actors); slight overhead.
   **Recommend:** Arc. The slots are rarely cloned per-message; the cost is amortized.
3. **Hydration order at startup** — declare bundles eagerly at registry init, or lazy-hydrate on first `resolve`?
   **Recommend:** eager for active G (DOLCE + per-deployment domain), lazy for inert. Add a `hydrated: bool` flag if we need to differentiate.
4. **Inheritance semantics** — set-union for SmallVec slots, override for scalar slots — codify with explicit `merge_with(parent)` method (see s3.4). Confirmed.

---

## 10. Cross-references

- `.claude/plans/ogit-g-context-bundle-v1.md` — D-OGIT-G-2 design doc.
- `.claude/board/TECH_DEBT.md` — TD-CONTEXT-BUNDLE-2 ticket.
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` — W2 sister; uses `resolve(g)` for routing.
- `.claude/specs/pr-c-1-generic-bridge.md` — W4 sister; populates `consumer_pointer` slot.
- `.claude/specs/pr-d-1-fma-owl-hydrator.md` — W9 sister; populates `OntologySlot` and friends.
- `.claude/specs/sprint-3-execution-plan.md` — W1 master plan.

---

## 11. Self-review notes (W3)

- Bundle is intentionally a plain struct (no trait object) so per-slot access is zero-cost field load.
- `Option<Arc<…>>` is uniform across hydrated slots so hydrators have a single insertion pattern.
- `SmallVec<[u8; N]>` chosen for `thinking_styles` (typical 2-6 styles per G), `trust_texture_set` and `flow_state_set` (typical 1-3 each) — keeps a bundle on the stack for the common case.
- `merge_with` deliberately does **not** inherit OWL slots (`ontology`, `codebook`, `schema`, `labels`, `vocabulary`) — those are per-G surfaces; inheritance there would silently leak DOLCE class-IDs into specialized domains.
- `register` panics on duplicate to surface accidental double-registration during boot; `replace` exists for the legitimate hydrator hot-swap path.
- No public mutator on `bundles` HashMap directly — only `register`/`replace` — to keep invariants enforceable later (e.g. when we add per-G locks).
