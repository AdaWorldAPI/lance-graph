# The DataFusion Perimeter

> **Status:** measured 2026-07-29, all numbers reproduced from the working tree.
> **Purpose:** answer "what would it cost to build lance-graph without DataFusion,
> and what would we lose" with evidence rather than estimate.
> **Motivating target (operator):** a substrate small enough for a Raspberry Pi —
> home automation, toy robots, anything wanting a breath of AGI without a
> data-warehouse build. Compiling DataFusion is outside that budget.

> **⊘ CORRECTION PASS (2026-07-29, same day).** The first edition merged as
> **#870** while this pass was being written, so the uncorrected text was briefly
> on `main` and these corrections land as a **follow-up**, not as an edit to that
> PR. Four findings from the codex review were verified and are folded in below. Two changed conclusions:
> **(a)** the §1 corollary "Lance storage and a Pi-sized binary are mutually
> exclusive" was **wrong** — it holds for the `lance` façade, not for Lance
> storage; `lance-table` carries the versioning primitives at 280 crates and
> **one** datafusion crate (§2a). **(b)** `ScanStrategy` was **not** part of the
> `datafusion_planner` public surface — the 6 hits were a same-identifier grep
> collision with a different type (§5). Two were measurement hygiene: build-edge
> exclusion (§2) and an unreproducible closure recipe (§2). Superseded text is
> retained and labelled, never silently overwritten.

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

> **Corollary (as first written — ⊘ SUPERSEDED, see §2a): "Lance storage and a
> Pi-sized binary are mutually exclusive today. Any configuration keeping
> `lance = 7.0.0` keeps the full closure, ACID/WAL included."**
>
> The second sentence is true and still governs. The first **does not follow from
> it** — it silently equates "Lance storage" with the `lance` façade crate. The
> versioning primitives live one layer down in `lance-table`, which has no
> `lance-datafusion` edge. Corrected in §2a. The error is worth naming: a
> conclusion about a *capability* was drawn from a measurement of a *crate*.

## 2. Dependency closures (measured)

**Recipe (corrected).** The first edition printed
`cargo tree -e normal --prefix none | sort -u | wc -l`. That pipeline does **not**
reproduce the numbers below: `cargo tree` marks a repeated subtree with a `(*)`
suffix, so `foo v1` and `foo v1 (*)` survive `sort -u` as two lines. On
`lance-graph` the printed recipe yields **828**, not 598. Use:

```sh
cargo tree -p <crate> -e normal,build --prefix none | sed 's/ (\*)$//' | sort -u | wc -l
```

*(Codex read the printed recipe and inferred the published counts were inflated.
They were not — the counts were already deduplicated; the recipe was wrong.
Cross-checked two ways: stripping `(*)` and reducing to `name version` pairs both
give 598. A recipe that does not reproduce its own number is a defect regardless
of which end is wrong, and it is the reader who pays.)*

| crate | `-e normal` | `-e normal,build` |
|---|---|---|
| `lance-graph` | **598** | 615 |
| `lance-graph-cognitive` | 70 | 82 |
| `lance-graph-planner` | **47** | 55 |
| `lance-graph-contract` | **1** (itself) | **16** |

**Build edges belong in a build-cost question.** `-e normal` hides them, and for
`contract` that is the whole story: it has zero *normal* dependencies exactly as
advertised, and a build script that pulls **15** crates — `serde_yaml`, `serde`,
`glob`, and transitives (`syn`, `quote`, `proc-macro2`, `indexmap`, …). All pure
Rust, all compiling natively on a Pi, none of them DataFusion. "Zero deps" stays
true of the linked artifact; it was never true of the build.

The Pi-viable substrate is `contract` + `planner` + `cognitive` + `ndarray`:
reasoning seam, 16 strategies, MUL, thinking, SoA/kanban, temporal — with NEON
already present in `ndarray`. **None of those crates has a DataFusion edge.**

Compile *time* is the smaller half of the argument. Compiling DataFusion is
memory-hungry; a 4 GB Pi is more likely to thrash or OOM than to merely take
27 minutes. A 55-crate closure is a Pi Zero 2W conversation, not a Pi 5 one.

## 2a. Lance persistence WITHOUT DataFusion — measured, and the answer is yes

The first edition left this as the top open question. It is now measured, in
isolated scratch crates outside this workspace (a `cargo tree -p` inside the
workspace resolves under **feature unification** with every other member, which
inflates any crate measured there — that is why the probe had to leave the tree):

| dependency | closure (`-e normal,build`, deduped) | datafusion crates |
|---|---|---|
| `lance = 7.0.0` | **571** | **32** |
| `lance-table = 7.0.0` | **280** | **1** |

That one crate is `datafusion-common`, reached via `lance-file`, where it is a
**non-optional** dependency. `lance-core`'s datafusion deps
(`datafusion-common` + `datafusion-sql`) are behind an **optional `datafusion`
feature**; `lance-table`'s own manifest names DataFusion nowhere.

**And `lance-table` is where the versioning lives** — `Manifest`
(`format/manifest.rs`), `Transaction` (`format/transaction.rs`), `CommitHandler`
(`io/commit.rs`). So the trade is not "versioned storage vs memory-only". It is:

> **`lance-table`** — versions, manifests, commits, ACID; 280 crates; one
> DataFusion crate.
> **`lance`** — the above *plus* `Dataset`/`WriteParams`/`InsertBuilder`
> conveniences and the query surface; 571 crates; 32 DataFusion crates.

The Pi question is therefore **not** "can we have durability" but "can we afford
to re-create the `Dataset` conveniences we actually use on top of `lance-table`".
That is a bounded, countable question, and **§10 Q1 now answers it** — the whole
production surface is `Dataset::open`, an append (`WriteMode`/`WriteParams`/
`InsertBuilder`), and `dataset::Version`, plus one public signature that leaks
`&lance::Dataset` and one genuinely query-side use.

**The standing wave never needed either.** 32k mailboxes × 512 B = 16 MB; it is
an in-memory array. Persistence is what the two options above buy.

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

## 5. The public surface is three names — *(membership corrected)*

Census method (corrected): every reference to `datafusion_planner` from outside
`src/datafusion_planner/`, resolved to what it actually imports — not a grep for
type names, which cannot tell two same-named types apart.

| site | imports |
|---|---|
| `query.rs:927` | `DataFusionPlanner`, `GraphPhysicalPlanner` |
| `lance_native_planner.rs:15` | `GraphPhysicalPlanner` |
| `lance_vector_search.rs:35` | `vector_ops` |
| `cam_pq/udf.rs:25` | `vector_ops` |

**Four use sites, three names: `DataFusionPlanner`, `GraphPhysicalPlanner`,
`vector_ops`.** A ~6K-LOC subsystem behind a three-name door — the headline
survives; its membership does not.

> **⊘ CORRECTED — `ScanStrategy` was never on this list.** The first edition
> published `ScanStrategy` at 6 files, ranking it the *widest* part of the
> surface. Those 6 files are in `lance-graph-planner` and import
> `crate::ir::logical_op::ScanStrategy` — a **different type in a different
> crate** (variants `Cascade`/`Full`/`Index`/`CamPq`). The DataFusion
> `ScanStrategy` (`datafusion_planner::cost_estimation`, variants
> `Cascade`/`FullScan`) is used in exactly two files, **both inside
> `datafusion_planner/`**: its own `cost_estimation.rs` and
> `predicate_pushdown.rs`. It is fully internal.
>
> This also kills the proposal that followed from it ("if it is a plain enum
> describing *how* to scan it belongs in `contract`") — there is nothing to
> move, and moving the planner's own `ScanStrategy` into `contract` would have
> been a change made for a reason that did not exist.
>
> The failure mode is the one §9 of this very document already names: a
> same-identifier grep counted as a reference census. Writing the rule down did
> not stop me applying the anti-pattern four sections later. **A name is not an
> identity; only a resolved import is.**

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

The leak is **`vector_ops`** (15 public items, half the external use sites). It
is CAM-PQ distance exposed as DataFusion UDFs — the least SQL-ish part of the
module, and **the only piece needing a decision rather than a `#[cfg]`**. With
`ScanStrategy` withdrawn (above), the other two names are the planner entry
points themselves, which gate cleanly with the module.

## 6. What is NOT implicated (checked, not assumed)

- **The batch writer.** `lance-graph-planner/src/batch_writer.rs` imports exactly
  `MailboxId` and `KanbanMove`, both from `lance-graph-contract`. No DataFusion,
  no Lance. It batches kanban moves; it is not a storage writer. **ACID/WAL is
  Lance's — specifically `lance-table`'s** (`Transaction`, `Manifest`,
  `CommitHandler`; see §2a), not the `lance` façade's and not DataFusion's.
  DataFusion is a query engine with no durability layer, which is why gating it
  cannot cost durability.
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

### Round two — proxies that lied in the FIRST EDITION OF THIS DOCUMENT

The four above were collected while researching. These four were published:

| proxy | claimed | actual |
|---|---|---|
| `grep -c ScanStrategy` across the workspace | a 6-file public surface, the widest name in §5 | two same-named types in two crates; the DataFusion one is used in 2 files, both internal (§5) |
| the `lance` façade crate | "Lance storage" as a whole | `lance-table` gives versions/manifests/commits at 280 crates and 1 datafusion crate vs 571/32 (§2a) |
| `cargo tree -p X` inside the workspace | crate X's closure | resolves under workspace **feature unification**; an isolated scratch crate is the only honest measurement (§2a) |
| `-e normal` | the build's dependency cost | excludes build edges — `contract` goes 1 → 16 (§2) |

**The pattern across both rounds is one thing: a proxy that is cheap to compute
gets published as the quantity it resembles.** A grep resembles a census; a
façade crate resembles a capability; an in-workspace tree resembles a closure.
Each is one command away from the real measurement, and the real measurement
changed the answer every time it was run.

Worth stating plainly: §9 existed, with the same-identifier-grep failure named in
it, **before** §5 was written with a same-identifier grep. Documenting an
anti-pattern does not immunise the document against it. The defence is
mechanical — resolve imports, leave the workspace, count edges — not vigilance.

## 9a. The object-store provider is the same class of fact — and the same trap

Cross-reference, because it is the identical failure shape one crate over:
a capability that "obviously exists" is behind a **feature that is off by
default**, and the resulting error is diagnosed at the wrong layer.

Source-verified from the vendored manifests (two releases apart, both agree):
**`lancedb` ships `default = []`**, and its `aws` feature is what forwards to
`lance/aws` + `lance-io/aws` (+ `object_store/aws` directly in the newer
release, transitively via `lance-io` in the older). Meanwhile **`lance-io`
carries `aws` in its OWN defaults** — so the intuition "the Lance stack does
object storage by default" is true one layer down and false at the layer we
depend on. `lancedb` is the layer that opts out.

Consequence, and it is the §9 lesson restated: without the feature an
object-store URI fails at **provider lookup by scheme**, before any credential,
endpoint or region is consulted — so credential and endpoint debugging is
spent on code that is not in the binary. **An error naming a *scheme* is a
build problem; an error naming a *credential/host/region* is a config
problem.** Read the error's noun before touching configuration.

**Correction (review round, PR #901) — and it is the §9 lesson landing on this
section itself.** The paragraphs above are true *about `lancedb`*, and the first
draft presented them as the gate on **this repository's** object-store reads. They
are not. `crates/lance-graph/Cargo.toml` takes `lance` as a **direct,
non-optional** dependency **with default features**, and `lance`'s own `default`
includes `aws`; `lancedb` is `optional = true, default-features = false` behind a
separate feature, and no production path here opens datasets through it. So for
this crate the provider **is** compiled in, and a scheme-named error would mean
something else entirely.

The rule survives; its **first step** was missing: *resolve which crate opens the
URI, then read that crate's features — and check how this manifest takes it, since
a `default-features = false` on the dependency line overrides the upstream
default.* Diagnosing the right facts about the wrong crate is the §9 failure mode,
and this section had it.

Full treatment — the three-layer hydration model, why the object store must
not be the runtime store, the flush/rehydrate lifecycle, and the probe record
behind the correction above:
`.claude/knowledge/s3-hydration-lifecycle.md` (§3a for the crate-resolution step).

## 10. Open questions, in decision order

1. **What does this repo actually use from `lance` that `lance-table` lacks?**
   *(Replaces the first edition's "can `lance-core`/`lance-file` be used without
   `lance-datafusion`" — answered in §2a.)* The trade is known (280/1 vs 571/32),
   so the question is now a census, and the census is **done** — it is short:

   | symbol used from `lance` | production sites | shape |
   |---|---|---|
   | `dataset::Dataset` (`::open`) | `graph/versioned.rs:40`, `graph/scheduler.rs:46`, `query.rs:843`, `ontology/lance_cache.rs:27`, `callcenter/audit.rs:186,275,772` | open a versioned table |
   | `WriteMode` / `WriteParams` / `InsertBuilder` | `graph/versioned.rs:40`, `ontology/lance_cache.rs:27`, `callcenter/audit.rs:186`, `callcenter/audit_sink/lance_sink.rs:284` | append a `RecordBatch` |
   | `dataset::Version` | `graph/versioned.rs:441` | list versions |
   | `lance::Dataset` in a **public signature** | `lance_vector_search.rs:222` | `search_lance(&self, dataset: &lance::Dataset)` |
   | `lance::datafusion::LanceTableProvider` | `query.rs:789` | query-side by construction |

   *(Verified production, not test: `audit.rs:186` is inside `pub async fn flush`,
   `lance_sink.rs:284` inside `async fn write_batch_to_lance`. Test/bench-only
   sites — `graph/hydrate.rs:174`, `query.rs:2041`, `benches/graph_execution.rs`,
   `callcenter/tests/`, `bin/audit_verify.rs` — are excluded from the table
   because they do not constrain the shipped artifact.)*

   **The answer this suggests: open + append + list-versions, and one leak.**
   Everything except `LanceTableProvider` is the versioning/commit surface that
   §2a shows `lance-table` already carries — so the façade is being used as a
   convenience wrapper, not for capability. The real cost is re-creating
   `Dataset::open`/append ergonomics over `Manifest` + `CommitHandler` +
   `lance-file`, **once**, and the one genuine blocker is
   `lance_vector_search.rs:222`, which puts `&lance::Dataset` in a *public*
   signature — that is an API break, not an internal refactor, and it is the
   only site in the table that is.
   **Still unmeasured:** whether a hand-rolled open/append over `lance-table`
   reads datasets written by `lance` byte-identically. Do not schedule the
   re-creation before that round-trip is green — this whole document exists
   because a plausible inference was published ahead of the measurement.
2. **Full dependency gate blast radius** — make the 5 deps `optional`,
   `cargo check --no-default-features --all-targets`, count. Bounded by §3's 26.
3. **`vector_ops`** — does CAM-PQ-as-UDF belong on the substrate side or the
   query side? Note §6: the CAM-PQ it serves is V1. Now the *only* §5 name
   needing a decision rather than a `#[cfg]`.
4. **Call-site census** — is anything still using `datafusion_planner` to *reach
   behaviour it cannot address directly*? Those callers are V3 migration
   candidates; genuine relational users are not. Do **not** measure it as "who
   uses joins" — that frames it as a query engine and misses the dispatch role.

*(The first edition's Q3, "`ScanStrategy` — contract-shaped or DataFusion-shaped?",
is **withdrawn, not answered**. It was a question about a type that was not on the
surface it was asked about; see §5.)*

---

*All figures reproducible from the working tree at `ceca232`, with the §2 recipe
(`-e normal,build`, `(*)` stripped) — the first edition's recipe did not
reproduce its own numbers. §2a's two rows require **isolated scratch crates**
outside this workspace; measured in-tree they resolve under feature unification
and are not comparable. Where a number is an estimate or a single-observation
inference, it says so.*

*First edition `a50299c`; correction pass same day. Superseded claims are
retained in place and labelled `⊘`, per the append-only board convention — a
reader who saw the first edition can find every retraction rather than a
silently different document.*
