# lite-unified-surrealql-lance-v1 — one store + one query surface, behind a feature gate

> **Status:** CONJECTURE / design. **Test via feature gate; do NOT commit the
> stack change.** Needs a convergence + cross-domain + truth-architect probe
> (mechanism-vs-rhyme + the query-shape measurement) before any promotion.
> **Date:** 2026-06-18. **Parent threads:** the DO-arm (`ExecTarget::SurrealQl`,
> `lance-graph-contract::action`), `docs/STACK_SCAFFOLD.md`, the
> "cold TS + kanban stay Lance-native" ruling.

## Epiphany (less is more)

Today there are **two query engines over the same lance storage** (lance-graph's
*datafusion* planner + surreal's *SurrealQL*) and **two storage engines**
(lance vs rocksdb). The "lite unified" bet collapses both: **ONE store (lance-KV)
+ ONE primary query surface (SurrealQL via the AR-API adapter)**, datafusion
**feature-gated**, rocksdb **dropped**. Cypher/SQL/neo4j lower to SurrealQL —
which is *natively* graph (`->edge->`), a better target than Cypher→datafusion-SQL.

## The bet, as a feature gate (default-OFF)

A `lite-unified` feature that, when ON:
1. **Storage = surreal kv-lance** (one store; drop rocksdb). *Blocked on:* surreal
   kv-lance is implemented as a module but not yet feature-wired
   (`surrealdb/core/src/kvs/lance/`, the `.claude/lance-backend` integration).
2. **Query/exec = SurrealQL** via the AR-API adapter. The polyglot parser
   (Cypher/GQL/Gremlin/SPARQL/neo4j) lowers to **SurrealQL** (or the DO-arm
   `ActionInvocation`) instead of datafusion SQL. *Missing today:* the
   polyglot→SurrealQL lowering (today it's polyglot→datafusion).
3. **datafusion = `optional`, OFF** on this path. Kept behind a separate
   `datafusion-analytical` feature for the workloads that genuinely need
   vectorized/analytical SQL (joins, aggregations) — SurrealQL's weak spot.
4. The DO-arm `ExecTarget::SurrealQl` becomes the **primary** exec path, not one
   of four.

## What stays regardless (NOT datafusion)

lance vector search, CAM-PQ / bgz17 codec stack, the cognitive substrate
(BindSpace→MailboxSoA, the write contract, the SPO/AriGraph tissue). These are
orthogonal to the query-engine choice.

## Where it's a win vs a downgrade (the honest split)

- **Win (the bulk):** graph traversal, AR CRUD, cognitive/SPO, vector search —
  SurrealQL-on-lance fits, and Cypher→SurrealQL graph is a *better* lowering.
  Footprint: drop the rocksdb C++ build outright; make datafusion (a large Rust
  dep) optional.
- **Downgrade:** heavy analytical SQL (multi-way joins, aggregations, columnar
  scan) — datafusion's strength, SurrealQL's weakness. Hence datafusion stays
  feature-gated, not deleted.

## Falsifier (truth-architect — measure before promoting)

Take lance-graph's `datafusion_planner` test queries (the Cypher→SQL cases) and
check **SurrealQL can express each**. Covered → drop datafusion for that path;
analytical gaps → keep `datafusion-analytical` for those only. Also measure the
real footprint delta (`cargo tree --no-default-features` + release `cargo bloat`)
once kv-lance is feature-wired — the proxy is lance-graph ≈ 889 crates, surreal
(all backends) ≈ 1148; the marginal SurrealQL-engine cost is ~260 crates, rocksdb
is a separate C++ build.

## Increments (all behind `lite-unified`, none committed to the default path)

1. **Probe (no code):** convergence + cross-domain (mechanism-vs-rhyme) +
   truth-architect (the datafusion_planner query-shape coverage check). Gate.
2. **Wire surreal kv-lance** as a feature (finish the `.claude/lance-backend`
   integration; add the `kv-lance` feature + lance dep + `mod lance` in `kvs/mod.rs`).
3. **Polyglot→SurrealQL lowering** — the missing front-end leg (parallel to the
   existing polyglot→datafusion).
4. **`datafusion` → `optional`** + a `datafusion-analytical` feature; default the
   common path to SurrealQL-on-lance under `lite-unified`.
5. **Measure** footprint + query-shape coverage; promote CONJECTURE→FINDING or
   correct.

## Blockers / open questions

- **OQ-LU-1:** surreal kv-lance feature-wiring (the integration TODOs).
- **OQ-LU-2:** does SurrealQL cover the lance-graph datafusion_planner query
  shapes the live workloads actually use? (the falsifier).
- **OQ-LU-3:** is the polyglot→SurrealQL lowering cleaner than polyglot→datafusion
  for the non-graph dialects (SPARQL/Gremlin)?
- Do NOT touch the default build until the probe is green.
