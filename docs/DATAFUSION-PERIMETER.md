# The DataFusion Perimeter

> **Status:** measured 2026-07-29, all numbers reproduced from the working tree.
> **Purpose:** answer "what would it cost to build lance-graph without DataFusion,
> and what would we lose" with evidence rather than estimate.
> **Motivating target (operator):** a substrate small enough for a Raspberry Pi —
> home automation, toy robots, anything wanting a breath of AGI without a
> data-warehouse build. Compiling DataFusion is outside that budget.

---

## 1. The headline: `lance` pulls DataFusion, so gating our own dep buys nothing

```
datafusion v53.1.0
├── geodatafusion v0.4.0 → lance-geo v7.0.0 → lance-datafusion v7.0.0 → lance v7.0.0 → lance-graph
├── lance v7.0.0                                                      → lance-graph
├── lance-datafusion v7.0.0
├── lance-index v7.0.0 → lance-datafusion
└── lance-graph, lance-graph-catalog                                  (our direct edges)
```

**This is the fact that governs everything else.** `lance-graph` has 5 direct
`datafusion*` entries in `[dependencies]`, but `lance = 7.0.0` reaches DataFusion
through **three independent paths** of its own. Feature-gating only our direct
edges would produce a flag that removes **zero** crates while `lance` is present —
a change that looks like a win and delivers nothing.

> **Corollary: Lance storage and a Pi-sized binary are mutually exclusive today.**
> Any configuration keeping `lance = 7.0.0` keeps the full closure, ACID/WAL
> included.

## 2. Dependency closures (measured)

`cargo tree -e normal --prefix none | sort -u | wc -l`:

| crate | transitive crates |
|---|---|
| `lance-graph` | **598** |
| `lance-graph-cognitive` | 70 |
| `lance-graph-planner` | **47** |
| `lance-graph-contract` | **1** (itself — zero deps, as advertised) |

The Pi-viable substrate is `contract` + `planner` + `cognitive` + `ndarray`:
reasoning seam, 16 strategies, MUL, thinking, SoA/kanban, temporal — with NEON
already present in `ndarray`. **None of those crates has a DataFusion edge.**

The cost of that path is Lance persistence. The standing wave itself is cheap
(32k mailboxes × 512 B = 16 MB, nothing on a Pi 4); what is lost is *versioned
durable storage*. **Open question, not yet measured:** can `lance-core` /
`lance-file` be used without `lance-datafusion`? That single answer decides
whether a Pi build gets real time-travel or memory-only state.

Compile *time* is the smaller half of the argument. Compiling DataFusion is
memory-hungry; a 4 GB Pi is more likely to thrash or OOM than to merely take
27 minutes. A 47-crate closure is a Pi Zero 2W conversation, not a Pi 5 one.

## 3. Where DataFusion actually appears in `lance-graph`

**26 of 113 source files.** 16 of those live inside `datafusion_planner/`; the
rest are the SQL/Cypher execution band: `sql_query`, `sql_catalog`,
`spark_dialect`, `table_readers`, `lance_native_planner`, `lance_vector_search`,
`query`, `nsm/nsm_word.rs`, `cam_pq/udf.rs`, plus `error.rs` and `lib.rs`.

**87 files never mention it.**

### `error.rs` does NOT cascade — this was the feasibility gate

Coupling is **3 lines**: one `GraphError` variant carrying
`datafusion_common::DataFusionError`, and one `From` impl. Both take a `#[cfg]`
directly. 30 files name `GraphError`/`Result` and **none needs touching** — the
variant is additive, and nothing matches it exhaustively (confirmed: `error.rs`
produced no errors in the probe below).

This was the difference between an afternoon and a wave. It came out on the
afternoon side.

### The substrate is untouched

`graph/spo`, `graph/blasgraph`, `graph/neighborhood` — **0 DataFusion references
each**. Parser, AST, semantic analysis, logical plan, graph router: none. The
search primitives compute without it.

## 4. Probe: gate the module, count the breakage

Method: add a non-default `sql` feature, `#[cfg(feature = "sql")]` on
`pub mod datafusion_planner;`, then `cargo check -p lance-graph`.

**Result: 5 errors — 4 real sites + 1 cascade.**

| site | what breaks |
|---|---|
| `query.rs:927` | `use crate::datafusion_planner::{DataFusionPlanner, GraphPhysicalPlanner}` |
| `lance_vector_search.rs:35` | `use crate::datafusion_planner::vector_ops` |
| `lance_native_planner.rs:15` | `use crate::datafusion_planner::GraphPhysicalPlanner` |
| `cam_pq/udf.rs:25` | `use crate::datafusion_planner::vector_ops` |
| `lib.rs` ×4 | **cascade** — `pub use query::{…}` failing because `query.rs` failed |

No cascade into the substrate. (Probe was reverted; the tree is unchanged.)

**Caveats on that number, stated so it is not over-read:**
1. It gated the **module**, not the **dependency**. Making the 5 deps `optional`
   will surface direct `datafusion::` uses in `sql_query`, `sql_catalog`,
   `spark_dialect`, `table_readers`, `nsm/nsm_word.rs`. Bounded by the 26-file
   list, but larger than 5.
2. `cargo check -p` builds neither tests nor examples. `--all-targets` finds more.
3. Per §1, it removes no crates while `lance` is present.

## 5. The public surface is three names

Types from `datafusion_planner` referenced elsewhere in the workspace:

| files naming it | type |
|---|---|
| 6 | `ScanStrategy` |
| 2 | `GraphPhysicalPlanner` |
| 1 | `DataFusionPlanner` |

A ~6K-LOC subsystem behind a three-name door.

### Per-child public surface

| child | `pub` items |
|---|---|
| `vector_ops.rs` | **15** |
| `analysis.rs`, `cost_estimation.rs`, `predicate_pushdown.rs` | 6 each |
| `mod.rs`, `test_fixtures.rs` | 5 each |
| `builder/{mod,aggregate_ops,basic_ops,expand_ops,helpers,join_builder}`, `config_helpers`, `expression`, `join_ops`, `scan_ops`, `udf` | **0** |

**Ten of seventeen children are fully internal** — the whole `builder/` tree, the
scan and join ops. That is the relational machinery, and nothing outside can
reach it.

The leak is **`vector_ops`** (15 public items, both external `use` sites). It is
CAM-PQ distance exposed as DataFusion UDFs — the least SQL-ish part of the
module, and the one piece that needs a decision rather than a `#[cfg]`.
`ScanStrategy` is the other: if it is a plain enum describing *how* to scan it
belongs in `contract`; if it names DataFusion structures it stays gated.

## 6. What is NOT implicated (checked, not assumed)

- **The batch writer.** `lance-graph-planner/src/batch_writer.rs` imports exactly
  `MailboxId` and `KanbanMove`, both from `lance-graph-contract`. No DataFusion,
  no Lance. It batches kanban moves; it is not a storage writer. **ACID/WAL is
  Lance's** — transactions, manifest, versions live in the file format. DataFusion
  is a query engine with no durability layer.
- **RaBitQ.** Zero hits in `lance-graph`; it lives in `thinking-engine`
  (`inference_backend.rs` + its own manifest). If lancedb ships RaBitQ natively,
  the dependency runs the other way — lance-graph could consume it.
- **`cam_pq` is V1.** The code is `[u8; 6]` / `6 * 256` — six subspaces, one byte
  each, matching the canon's "path = HEEL+HIP+TWIG = 6 bytes = the CAM-PQ 6×256
  code". **It is not the V3 `6×(u8:u8)` carving** (6 rails of two *separate*
  bytes, 12 B, `palette256:palette256`, ClassView-projected). Same number 6,
  different shape — do not read the coincidence as continuity. Zero mentions of
  V3/rail/palette256/facet in `cam_pq/`. *(Caveat: `[u8; 6]` was observed in a
  test fixture and `6 * 256` in a size estimate; the authoritative code width may
  live in `ndarray::hpc::cam_pq`, which lance-graph implements `CamCodecContract`
  against.)*

## 7. The fossil: `[features] datafusion = []`

`lance-graph-planner/Cargo.toml` declares a **feature** named `datafusion` with
an empty list — **not a dependency**. Zero `.rs` files in the planner reference
DataFusion.

It is a reserved slot for a dependency that was expected and never arrived.
Historically (operator) `datafusion_planner` was taken on as the **orchestrator**
— the polymath that would dispatch the 4096 `0xFFFFFF`-shaped commands — before
V3 answered that with `classid → ClassView` + kanban + `UnifiedStep`. Upstream's
own reason was different again: columns-and-rows join so Cypher could traverse.

So the dependency currently serves **neither** of its two historical jobs, and
the planner grew its replacement in parallel — 16 strategies, MUL, sigma chain —
which is exactly why the reserved slot stayed empty.

**Naming consequence:** if a feature lands, name it for its *role*
(`legacy-dispatch`) rather than its syntax (`sql`), so the manifest carries this
history instead of the next reader re-deriving it.

## 8. `LazyLock` cannot substitute for a feature

`LazyLock` is runtime laziness. If DataFusion is in `[dependencies]`, all 598
crates compile and link whether or not the lock initialises. `#[cfg]` is the only
mechanism in Rust that removes a dependency **without removing code** — the
source stays in the repo verbatim, it simply is not compiled. That *is* the
non-destructive option.

There is a real `LazyLock` target next door, on a different problem:
`datafusion_planner/udf.rs` already caches its UDFs that way, but
`sql_catalog.rs:38` constructs a `SessionContext` eagerly. Worth making lazy on
its own merits — a startup win, orthogonal to the closure. **Keep it out of any
gating work**, or it makes the blast-radius measurement unreadable.

## 9. Method note — proxies that lied

Recorded because these cost real time today and will recur:

| proxy | claimed | actual |
|---|---|---|
| single-line `grep datafusion` on `Cargo.toml` | "planner depends on DataFusion" | it is a `[features]` entry |
| `grep -c` per crate | dependency counts | cannot distinguish `[dependencies]` from `[features]`; needs section-aware parsing |
| `grep -lw <filename>` for module usage | reference counts | `mod` scored 831, `helpers` 58, `analysis` 50 — common identifiers, not references |
| compiler error count | independent failure sites | 5 errors were 4 sites + 1 cascade through `pub use` |

**Rule:** for dependency questions use `cargo tree`, not `grep`. The `lance →
lance-datafusion` edge in §1 — the single most decision-relevant fact here — is
invisible to every grep and appears only in the resolved graph.

## 10. Open questions, in decision order

1. **Can `lance-core` / `lance-file` be used without `lance-datafusion`?**
   Decides whether a Pi build gets versioned storage or memory-only state.
2. **Full dependency gate blast radius** — make the 5 deps `optional`,
   `cargo check --no-default-features --all-targets`, count. Bounded by §3's 26.
3. **`ScanStrategy`** — contract-shaped enum, or DataFusion-shaped?
4. **`vector_ops`** — does CAM-PQ-as-UDF belong on the substrate side or the
   query side? Note §6: the CAM-PQ it serves is V1.
5. **Call-site census** — is anything still using `datafusion_planner` to *reach
   behaviour it cannot address directly*? Those callers are V3 migration
   candidates; genuine relational users are not. Do **not** measure it as "who
   uses joins" — that frames it as a query engine and misses the dispatch role.

---

*All figures reproducible from the working tree at `ceca232`. Where a number is
an estimate or a single-observation inference, it says so.*
