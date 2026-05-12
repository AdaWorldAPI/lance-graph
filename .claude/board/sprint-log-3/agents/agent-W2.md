# Agent W2 — Sprint-3 log

**Role:** Worker Agent W2 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Tier-1 implementation specs.
**Tech-debt anchor:** TD-OGIT-G-SLOT-1.
**Pattern letter (post-PR #359):** Pattern A — SPO-G with `u32` OGIT slot.

---

## Deliverable

`.claude/specs/pr-a-1-spo-g-u32-slot.md` — PR-ready spec for the first
concrete Tier-1 implementation. After this spec, an engineer picks up the PR
and starts coding.

## Status

**DONE — spec drafted and pushed to branch.**

## Decisions logged

1. **`SpoQuad` lives in `lance-graph-contract`, not `lance-graph`.** This
   keeps PR-B-1 (ContextBundle resolver) and PR-C-1 (generic bridge) free
   of a heavy lance-graph dependency. The contract crate is already the
   canonical home for cross-cutting types.
2. **`g = 0` is reserved for the DOLCE root context**, not "unscoped /
   legacy". Aligns with PR #355 `NamespaceRegistry::seed_defaults` slot-0
   convention. Avoids the legacy-rows-look-semantic footgun.
3. **`SpoBridge::promote_to_spo` takes `g` as a required parameter.**
   No `Option<u32>`, no default. Forces every promotion site to think about
   ontology context — that's the whole point of the migration.
4. **Schema migration is scripted** (`migrations/001_spo_g_column.rs`),
   never manual. Idempotent on re-run; bumps a schema-version marker so
   subsequent migrations chain cleanly.
5. **Lance MVCC time-travel is exposed as `read_as_of_version(g, v)`.**
   The `g` slot is what makes per-context time-travel coherent — without
   it you can only time-travel the whole dataset.
6. **Legacy `insert_legacy` shim is `#[deprecated]`.** Backwards-compat is
   required for the 33 medcare regulatory + 13 smb-realtime CI fixtures,
   but new callers get a compile-time nudge toward the g-aware entry point.

## Dependency call-out

PR-A-1 depends on PR-B-1 (W3) landing first. PR-B-1 introduces
`OntologyRegistry::resolve(g)`, the resolver every consumer of `SpoQuad.g`
calls. Without PR-B-1, the `g` slot is a dangling `u32` with no
dereference path. Open-question 3 in the spec flags this for the engineer.

## Cross-worker handover

- **W1 (master plan):** `sprint-3-execution-plan.md` — references PR-A-1
  as the first concrete Tier-1 deliverable.
- **W3 (PR-B-1, ContextBundle):** sister spec; PR-A-1 explicitly lists it
  as a required precursor.
- **W4 (PR-C-1, generic bridge):** downstream consumer of `SpoQuad`; will
  import the type from `lance-graph-contract`.

## Files written this session

- `.claude/specs/pr-a-1-spo-g-u32-slot.md` (spec, ~7.5 KB)
- `.claude/board/sprint-log-3/agents/agent-W2.md` (this log)

## Next handover

Engineer pickup. The spec's "Open questions for the engineer" section has
three items with recommendations; engineer should confirm or push back
before coding starts.
