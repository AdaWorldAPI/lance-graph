# PR-A-1: SPO-G with u32 OGIT slot

**Tier-1 implementation spec — Pattern A canonical (post-PR #359 letter assignment).**
**Tech-debt anchor:** TD-OGIT-G-SLOT-1.
**Sprint-3 owner:** W2 (this spec) → engineer pickup.

---

## Goal

Extend `(S, P, O)` triples to `(S, P, O, G)` quads where `G` is a `u32` OGIT
(Ontology Graph Index Table) slot. This lands the canonical Pattern A shape
across the SPO store, the warm string-keyed Arigraph cache, and the
SpoBridge promotion path. Backwards-compatible: legacy `(S,P,O)` callers
continue to work with `g = 0`.

The `G` slot is the join key for the ContextBundle resolver (PR-B-1) and the
generic bridge (PR-C-1). Once it lands, every fingerprint-keyed quad carries
its ontology context, and Lance MVCC versioning gives us a temporal axis for
free (`read_as_of_version(g, v)`).

---

## Files to touch (Rust)

| File | Change |
|---|---|
| `crates/lance-graph/src/graph/spo/mod.rs` | Schema migration — add `g: u32` column to the SPO Lance schema |
| `crates/lance-graph/src/graph/spo/store.rs` | `SpoStore::insert(s, p, o)` → `insert(s, p, o, g)`; legacy shim with `g = 0` for backwards-compat callers |
| `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` | String-keyed warm cache rows also carry `g` so warm/cold agree |
| `crates/lance-graph/src/graph/arigraph/spo_bridge.rs` | Extend PR #355 D-ONTO-V5-9 writer: `SpoBridge::promote_to_spo(triple, g)` takes a required `g` parameter |
| `crates/lance-graph-contract/src/spo_g.rs` | **NEW** — `SpoQuad` type definition (lives in the contract crate so PR-B-1 / PR-C-1 can depend on it without pulling in lance-graph) |
| `crates/lance-graph/src/migrations/001_spo_g_column.rs` | **NEW** — one-shot migration script that adds the column to existing Lance datasets with default `0` |

---

## Schema migration

- Lance dataset gets a new `g: u32` column.
- Default value `0` for every pre-existing row (preserves backwards-compat for
  the 33 medcare regulatory + 13 smb-realtime fixture datasets already on
  disk in CI).
- Migration is **scripted**, not manual: `migrations/001_spo_g_column.rs`
  detects schema version `< 1`, runs `add_columns(...)`, bumps the version
  marker, and is idempotent on re-run.
- Lance MVCC versioning is already present at the dataset level; once `g`
  lands in the row schema, `read_as_of_version(g, v)` becomes the public
  temporal entry point (see API sketch).

---

## API sketch (~50 LOC)

```rust
// crates/lance-graph-contract/src/spo_g.rs
//
// Canonical Pattern A quad. Lives in the contract crate so downstream
// consumers (PR-B-1 ContextBundle resolver, PR-C-1 generic bridge) can
// depend on the type without pulling in the full lance-graph crate.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpoQuad {
    pub subject_fp:   u64,
    pub predicate_fp: u64,
    pub object_fp:    u64,
    /// OGIT slot — index into the OntologyRegistry (PR-B-1).
    /// `0` is reserved for the DOLCE root context (see open-question 1).
    pub g: u32,
    /// Lance MVCC version stamp. Populated by the store on insert.
    pub version: u32,
}

// crates/lance-graph/src/graph/spo/store.rs
impl SpoStore {
    pub fn insert(
        &mut self,
        s: u64,
        p: u64,
        o: u64,
        g: u32,
    ) -> Result<SpoQuad>;

    /// Legacy shim — calls `insert(s, p, o, 0)`.
    /// Marked `#[deprecated]` so new callers are nudged toward the
    /// g-aware entry point.
    #[deprecated(note = "pass g explicitly; use insert(s,p,o,g)")]
    pub fn insert_legacy(&mut self, s: u64, p: u64, o: u64) -> Result<SpoQuad>;

    /// Time-travel read. Returns the partial-state view of slot `g`
    /// at Lance dataset version `v`.
    pub fn read_as_of_version(
        &self,
        g: u32,
        v: u32,
    ) -> Result<impl Iterator<Item = SpoQuad>>;
}

// crates/lance-graph/src/graph/arigraph/spo_bridge.rs
impl SpoBridge {
    /// Extends the PR #355 D-ONTO-V5-9 promotion writer with `g`.
    /// `g` is REQUIRED (no default) — see open-question 2.
    pub fn promote_to_spo(
        triplet: &TripletString,
        g: u32,
    ) -> SpoQuad { /* ... */ }
}
```

---

## Test plan

| Test | Coverage |
|---|---|
| `tests/spo_g_round_trip.rs` | Write `(S, P, O, G=42)`, read back and assert exact match on all four fields. |
| `tests/spo_g_default_zero.rs` | Old-style callers via `insert_legacy` see `g == 0` on read-back. |
| `tests/spo_g_lance_mvcc.rs` | Insert at v1, mutate at v2, assert `read_as_of_version(g, 1)` returns the v1 partial-state. |
| `tests/spo_g_arigraph_bridge.rs` | String-keyed warm cache and fingerprint-keyed cold store agree on `g` for the same triple. |

**Regression gate:** all 33 medcare regulatory tests + 13 smb-realtime tests
must stay green (backwards-compat proof).

---

## Dependencies

- **PR-B-1 (W3 spec)** must land first. PR-B-1 introduces
  `OntologyRegistry::resolve(g) -> Option<&ContextBundle>`, which is the
  resolver every downstream consumer of `SpoQuad.g` calls. PR-A-1 puts the
  `g` slot on the wire; PR-B-1 makes it dereferenceable.
- No external crate dependencies. Pure-Rust schema change against existing
  `lance` and `lance-graph-contract` crates.

---

## Acceptance criteria

- [ ] `SpoQuad` type defined in `lance-graph-contract`.
- [ ] `SpoStore` + `SpoBridge` + `triplet_graph` all carry `g`.
- [ ] Schema migration scripted (`migrations/001_spo_g_column.rs`); no
      manual Lance dataset surgery anywhere in the repo.
- [ ] Backwards-compat: legacy callers see `g = 0`; `insert_legacy` marked
      `#[deprecated]`.
- [ ] Lance MVCC time-travel exposed as `read_as_of_version(g, v)`.
- [ ] All existing tests green (33 medcare + 13 smb-realtime + crate-level).
- [ ] 4 new tests landed for `g` coverage (round-trip, default-zero, MVCC,
      bridge agreement).

---

## Effort

**Medium.** ~300 LOC of Rust + the migration script + 4 new tests.
Estimate **1-2 engineer-days** end-to-end (including review cycle).

---

## Open questions for the engineer

1. **Should `g = 0` mean "DOLCE root context" or "unscoped (legacy)"?**
   Recommend **DOLCE root**. This is consistent with the PR #355
   `NamespaceRegistry::seed_defaults` convention, where slot `0` is already
   reserved for the canonical root namespace. Treating `0` as "unscoped"
   creates a footgun where legacy rows masquerade as semantically-meaningful
   data.
2. **`SpoBridge::promote_to_spo` signature — required `g: u32` parameter or
   defaulted to `0`?** Recommend **required**. The whole point of PR-A-1 is
   to force every promotion site to think about ontology context. A default
   would silently leak `g = 0` everywhere and defeat the migration.
3. **Migration order with PR-B-1.** Spec PR-B-1 should land **first** so
   `SpoQuad` can reference `ContextBundle` types if needed (e.g., a future
   `SpoQuad::resolve_bundle(&registry) -> Option<&ContextBundle>`
   convenience method). PR-A-1 lands second once the resolver exists.

---

## Cross-references

- `.claude/plans/ogit-g-context-bundle-v1.md` — D-OGIT-G-1 (this PR's
  plan-doc anchor)
- `.claude/board/TECH_DEBT.md` — TD-OGIT-G-SLOT-1
- `.claude/knowledge/tier-0-pattern-recognition.md` — Pattern A section
- `.claude/specs/pr-b-1-context-bundle.md` — W3 sister; required dependency
- `.claude/specs/pr-c-1-generic-bridge.md` — W4 sister; downstream consumer
- `.claude/specs/sprint-3-execution-plan.md` — W1 master execution plan
- PR #355 — D-ONTO-V5-9 SpoBridge writer (extended by this PR)
- PR #359 — canonical Pattern A letter assignment
