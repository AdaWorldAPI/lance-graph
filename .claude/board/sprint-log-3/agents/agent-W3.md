# Agent W3 — Sprint-3 (12+meta CCA2A)

**Role:** Worker Agent W3
**Branch:** `claude/tier-1-implementation-specs`
**Pattern:** B — ContextBundle per G (typed surface)
**Tech-debt:** TD-CONTEXT-BUNDLE-2
**Phase:** DESIGN

---

## Deliverables

| Path | Status | Size target |
| --- | --- | --- |
| `.claude/specs/pr-b-1-context-bundle.md` | DELIVERED | ~10 KB |
| `.claude/board/sprint-log-3/agents/agent-W3.md` (this file) | DELIVERED | n/a |

---

## Session log

### Entry 1 — spec authored
- Authored `.claude/specs/pr-b-1-context-bundle.md` covering:
  - 12 named slots on `ContextBundle` (ontology, codebook, schema, labels, vocabulary, consumer_pointer, thinking_styles, thinking_adjacency, qualia_codebook, mul_threshold_profile, trust_texture_set, flow_state_set).
  - `OntologyRegistry::resolve(g)` / `register(bundle)` / `replace(bundle)` API.
  - 12 slot type stubs, each one-liner placeholder for the eventual full impl in subsequent PRs (D, C, F…).
  - 3 hand-coded seed bundles: DOLCE (G=0, root), Healthcare (G=2), Gotham (G=3).
  - 3 tests: resolve smoke, inheritance chain, SmallVec inline-storage assertion.
  - `merge_with(parent)` helper codifying inheritance semantics: set-union for SmallVec slots, override-if-None for scalar slots, **no inheritance** for OWL slots.
- Pushed via pygithub to `claude/tier-1-implementation-specs`.

### Entry 2 — open questions resolved
1. **ConsumerPointer location:** recommend `lance-graph-contract::consumer::ConsumerPointer` (zero-deps canonical, avoids reverse dep).
2. **Arc vs Box for slots:** recommend Arc (ractor actor sharing).
3. **Hydration order:** recommend eager for active G, lazy for inert.
4. **Inheritance semantics:** explicit `merge_with` method, set-union for SmallVec, override-if-None for scalar, OWL slots never inherited.

### Entry 3 — self-review
- Confirmed 12-slot count matches W1 master plan and Pattern B canonical post-PR #359.
- Confirmed no PR dependencies — this is the foundation.
- Confirmed downstream consumers (PR-A-1, PR-C-1, PR-D-1, PR-E-1, PR-F-1, PR-J-1) all reachable via `OntologyRegistry::resolve(g)`.
- Cross-references to W2 (PR-A-1 SPO g:u32 slot), W4 (PR-C-1 GenericBridge), W9 (PR-D-1 OwlHydrator), W1 (sprint-3 master) included.
- File-touch list and acceptance criteria align with the ~200 LOC + 3 tests budget.
- No code committed yet (DESIGN PHASE) — implementation lives in subsequent worker rotation.

---

## Hand-off notes for IMPL phase

- **Engineer should resolve Q1** (ConsumerPointer crate location) before starting; impacts the `use` line in `context_bundle.rs` and whether `lance-graph-contract` needs a new `consumer` module added in the same PR.
- **Seed bundles can use raw `u8`** for thinking-style atoms if the named `ThinkingStyle` enum has not yet landed via PR-G; spec is explicit about this fallback.
- **Backwards compat checklist:** `OntologyRegistry::enumerate(ns)` must continue to work — engineer must add `bundles: HashMap<u32, ContextBundle>` as a new field, not replace existing storage.
- **Test coverage:** the SmallVec spilled-test pins the inline cap at 8; if W1 changes the cap to 4 or 16 in a later sprint, this test must be updated in lockstep.

---

## Status

DELIVERED. Awaiting impl-phase pickup.
