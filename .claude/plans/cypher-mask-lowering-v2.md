# cypher-mask-lowering-v2 — Cypher on masks, as a fork-owned replacement behind one switch

> **Status:** PROPOSAL (D-CML-0..9). Plan only; no code is authorized by this file.
> **Written against:** `main` `0d31c54f` (2026-09-30).
> **Supersedes:** `cypher-mask-lowering-v1.md` on three points, listed in §1.
> Everything else in v1 still stands: the lowering table (§3), the placement ruling
> (§5.2), the substrate laws (§5.3), the non-goals N-2..N-12 and the falsifier discipline
> (§7.5). v1 is cited here, not restated.
> **Board:** `STATUS_BOARD.md` § cypher-mask-lowering-v2 · entry
> `entries/2026-09-30-cypher-mask-v2-is-a-replacement-not-a-phase.md`.

---

## §0 — In plain words

We want Cypher queries answered by mask operations, with no DataFusion on the path.
We also must not edit the upstream files that Cypher currently lives in: they are
inherited from upstream, and editing them blocks taking upstream changes cleanly.

So v2 does not change the existing engine. It builds a **second engine beside it**, in
a new crate we own. That crate reuses the upstream parser, semantic analysis and
logical plan exactly as they are, through their public API, and lowers the resulting
plan to `lance-graph-mask-risc` programs. Our own surface chooses between the two
engines with **one switch**.

When the mask engine cannot answer a query, it **refuses** with a named reason.
It never quietly hands the query to DataFusion.

---

## §1 — What changes against v1, and why

| v1 said | v2 says | why |
|---|---|---|
| The seam is "Phase 2.5" inside `query.rs` `create_logical_plans`, with a `mask_lower` hook there | **No seam inside any upstream file.** The new crate calls the public parser / semantic / logical-plan API itself | Editing `query.rs` edits an upstream file. It is also impossible as a dependency: `lance-graph` cannot depend on a crate that depends on `lance-graph` |
| Three outcomes: `Full` / `Split` / `Grace` (a mask prefix, with DataFusion finishing the residue) | **Two outcomes: `Answer` / `Refusal(reason)`.** `Split` and `Grace` are removed | The operator does not want DataFusion on the surface. A `Split` is DataFusion on the surface under another name. v1's own OQ-11 had already doubted that `Split` pays |
| `label → classid` is answered by one new field on `GraphConfig`'s `NodeMapping` (`lance-graph-as-the-modelgraph-v1.md` §16.2) | **The binding lives in the new crate.** It is a `LabelBinding` table (label → `LabelDTO` → classid) that is passed next to the query, never inside `GraphConfig` | `config.rs` is an upstream file. `LabelDTO` (contract, `ogar_codebook.rs`) is already the route, and it still has zero consumers |

v1's §6 N-1 ("this does not deprecate DataFusion") **still holds for the upstream
engine**, and v2 honours it by never touching that engine. What v2 changes is that
**our surface** stops routing through it once the switch is flipped.

---

## §2 — The upstream / fork line (the working assumption, and how it is checked)

| side | files | v2 may |
|---|---|---|
| **upstream** (inherited) | `crates/lance-graph/src/{parser,ast,semantic,logical_plan,query,config,error}.rs`, `datafusion_planner/**` | **read and call through their public API only. Zero edits.** |
| **fork** | `crates/lance-graph/src/graph/**`, the `BlasGraph` backend, every other crate in the workspace | edit, if a wave needs it. v2 plans no edit to `lance-graph` at all |

This split is an **assumption**, drawn from which modules exist in the upstream
lance-graph project and which were added here. **D-CML-0 checks it** against the
upstream history before any code lands. If a file turns out to be fork-owned, the rule
for that file relaxes; the design does not change.

**The public API v2 consumes** (verified by reading, 2026-09-30):

- `lance_graph::parser::parse_cypher_query(&str) -> Result<CypherQuery>` (`parser.rs:23`);
- `lance_graph::semantic::SemanticAnalyzer::{new(GraphConfig), analyze(..)}` (`semantic.rs:65,74`);
- `lance_graph::logical_plan::{LogicalPlanner::{new, plan}, LogicalOperator}` (`logical_plan.rs:19,160,168`);
- the `ast` types that `LogicalOperator` carries (`BooleanExpression`, `ValueExpression`, `PropertyValue`, …).

The census example `examples/w0b_corpus_census.rs` already drives exactly this chain
from outside the planner. It is the proof that the chain is reachable without edits.

---

## §3 — The crate

**`crates/lance-graph-cypher-mask`** (fork-owned, new).

| depends on | for | never for |
|---|---|---|
| `lance-graph` | the parser, semantic and logical-plan API of §2 | `CypherQuery::execute`, `datafusion_planner`, `SqlDialect`, anything that runs a query |
| `lance-graph-mask-risc` | `Program`, `MaskOp`, `Pred`, `Terminal`, `execute`, `Planes` | — |
| `lance-graph-contract` | `LabelDTO`, `soa_view::MailboxSoaView`, `canonical_node`, `facet` | cognitive modules (same import fence as quack: named modules only) |

**Public surface: one function and two types.**

```rust
pub fn run(text: &str, bind: &LabelBinding, view: &MailboxSoaView<'_>) -> Result<Answer, Refusal>;
pub enum Answer  { Mask(/* the final population, a mask handle */), Scalar(i64), Bool(bool) }
pub enum Refusal { /* one variant per row of §5, each carrying the offending construct */ }
```

- `Answer` carries a mask, never a `Vec` of row ids.
- Turning a mask into rows or Arrow batches is **rendering**. It happens at the
  consumer, through one function whose name starts with `materialize`, and its cost
  is O(n), stated in its doc comment. This is v1 §7.1's one-materialiser rule, kept.
- **Traversal counts nothing.** Hops and fixpoints move masks. The only popcount is a
  terminal on the *final* mask (`count(DISTINCT n)`, `exists`). A count of paths or
  walks is refused (§5 row R-BAG).

**OQ-CML-1 — DataFusion is still linked.** `lance-graph` depends on `datafusion`
unconditionally (in `crates/lance-graph/Cargo.toml` the `datafusion` entry under
`[dependencies]` carries no optional flag), and `error.rs` (`GraphError`) puts `DataFusionError` inside `GraphError`. So depending on the
parser pulls DataFusion into the **build graph**, even though the new crate calls none
of it. This is "off the surface", not "out of the binary". Making DataFusion optional
upstream would be an upstream edit, and that is ruled out. The two options, to be
decided by **measurement in D-CML-0**:

- **(a)** accept it as a link-time dependency. Measure whether any DataFusion symbol
  survives into a release binary of the new crate.
- **(b)** copy `parser.rs` / `ast.rs` / `semantic.rs` / `logical_plan.rs` into the new
  crate as a pinned snapshot, with a CI diff against upstream that goes red on drift.

**(a) is the default.** (b) only if (a) measures DataFusion in the binary, or its
build cost is judged unacceptable.

---

## §4 — What lowers, and to which mask-risc shape

These are the rows of v1 §3, re-sorted by what the mask-risc IR already provides
today. Nothing here adds an op to the IR. A missing op is a STOP (v1 N-10): it lands
substrate-first, in mask-risc or `ndarray::simd`, with its own parity test.

| Cypher | lowering | IR today |
|---|---|---|
| `(n:Label)` | `Pred` equality on the classid lane, with the classid taken from `LabelBinding` | yes — `EqU32` / `MatchU64` / strided variants |
| `WHERE n.p <op> literal` | one `Pred` per leaf | yes, for i32 / u32 / u64 lanes (v1 OQ-3 covers narrower lanes) |
| `AND` / `OR` / `NOT` / `XOR`, up to 3 leaves per pass | `fuse` → one `Ternlog{imm}` | yes (D-MRX-3) |
| `(a)-[:R]->(b)` — **the mask join** | src mask → `Gather` through the row's **in-row** target lane → dst mask (§4.1) | `Gather` and `ScatterOrU32` yes. The in-row lane is §4.1's question |
| a join on equal keys across two populations | `EqU32Via` / `GroupKey::Via` through an index lane | yes |
| `count(DISTINCT n)`, `exists` | `Terminal::Count` / `Any` on the final mask | yes |
| `sum/min/max(n.p)` over the final mask | `MaskedSum/Min/MaxI32` | yes (i32) |
| `RETURN n` | `Terminal::Keep` → `Answer::Mask` | yes |
| `*1..k`, `*` | delta-frontier fixpoint over the hop (§4.2) | the loop lives in the new crate; each step is one `Program` |

### §4.1 — The hop reads relations that are stored in the row

A hop does **not** join against an external edge table. The substrate stores a row's
relations **inside the row**:

- the second facet (bytes 16..32, `EdgeBlock = FacetCascade`);
- the `CausalWitness` value tenant (`canonical_node.rs`, `ValueTenant::CausalWitness = 14`:
  24 signed i4 loci, each a context pointer within a ±8 window, all-zero = unbound);
- the episodic-basin rails (32 bytes of references).

The hop reads one of these as a **lane** (`LaneRef::Strided` over the 512-byte row)
and moves the mask from source rows to the rows they point at.

**Absolute targets** (a lane holding a row ordinal) → `ScatterOrU32`.
**Relative targets** (the witness loci: offsets of ±8) → a **shift of the mask by the
offset**, one shift per locus value, OR-ed together.

**OQ-CML-2 — which carrier holds absolute and which relative targets, per relationship
type; and whether mask-risc needs a `Shift` op.** mask-risc has no shift op today. If
the relative case is real, `Shift` is a STOP → substrate-first (mask-risc IR plus the
`ndarray::simd` primitive). It must not be written locally.

The v1 laws apply unchanged. **One population per mask** (v1 §5.3b): a hop's target
must index the same row space. **The transpose law** (§5.3c): `<-[:R]-` reads the
reverse lane if one exists. If none exists, it is refused (R-TRANSPOSE) — never
answered by reading the forward lane backwards.

### §4.2 — Variable length is reachability, and only reachability

`*1..k` lowers to: frontier = hop(frontier) AND NOT visited, accumulate, repeat until
`k` steps or `Any(frontier) == false`. The result is the **set of rows reached**.

Cypher's own semantics for a variable-length pattern bind *paths*. So any query whose
answer depends on paths is refused (R-BAG): `count(*)` over the pattern, returning the
path, or `length(p)`. #1305's measured finding says exactly this: DataFusion counts
walks (5), the trail count is 4, and the mask gives the support.

**The deliberate rule: v2 answers reachability questions and refuses path questions.**
It does not approximate one with the other.

---

## §5 — The refusal list (replaces v1 §4's `[GRACE]` list)

Each row is a `Refusal` variant. The classifier (D-CML-2) must name the variant from
the `LogicalOperator` alone, **before** any program runs.

| variant | construct | why it has no mask form |
|---|---|---|
| **R-ORDER** | `ORDER BY`, `SKIP`/`LIMIT` after order | a mask has no order |
| **R-BAG** | `count(*)` over more than one hop, `count(expr)` without `DISTINCT` over a multi-binding pattern, `collect`, returning paths, `length(p)` | a mask is support, not a bag (#1305) |
| **R-DISTINCT-VALUE** | `DISTINCT` over a projected value (not a node) | a value set is not a row set |
| **R-STRING** | string predicates beyond equality through a dictionary lane (`CONTAINS`, `STARTS WITH`, regex) | variable-width values; v1 §4.2 |
| **R-UNWIND / R-WITH-AGG** | `UNWIND`, aggregation inside `WITH` | bag re-entry |
| **R-VALUE** | vector distance / similarity, NARS truth, floats | values, not Boolean; v1 N-7. mask-risc is integer-only |
| **R-CROSS-SPACE** | a join whose two sides index different row spaces with no index lane between them | the one-population law |
| **R-TRANSPOSE** | an incoming hop with no reverse lane | the transpose law |
| **R-UNBOUND-LABEL** | a label with no `LabelBinding` entry | no classid to test; guessing one is the confident-and-wrong quadrant (v1 OQ-1) |
| **R-MUTATION** | `CREATE` / `SET` / `DELETE` / `MERGE` | v1 N-8; writes go through the commit gate |

**Refusing is the correct answer here, not a missing feature.** A refused row becomes
a lowering only through a new D-id with its own falsifier. It is never added by
silently widening the classifier.

**What the refusal list costs today, measured.** The W0-b census
(`examples/w0b_corpus_census.rs`; recorded in
`EPIPHANIES-ARCHIVE-2026-09-20.md` under
`E-THE-SPINE-IS-WHATEVER-THE-READER-ALREADY-HAS-AN-ADDRESS-FOR-1`; 303 classified
queries from a walk of the tree) found 113 **Full** (37.3 %). Everything else was
`Split` or `Grace`. Under v2 those become **refusals**, so the mask engine answers
**37.3 % of the committed corpus** on day one. The number is a property of the
corpus's test queries, most of which exercise DataFusion features on purpose. It is
not a property of the queries a consumer actually sends. D-CML-2 re-runs the census
under the v2 classifier and reports the count for each refusal variant.

---

## §6 — The switch

**One switch, on our surface, choosing a whole engine — never one query at a time.**

```rust
// in lance-graph-cypher-mask
pub enum Route { Mask, Upstream }
```

- `Route::Mask` → `run(..)`. A refusal is returned to the caller as a refusal.
- `Route::Upstream` → the caller's existing `CypherQuery::execute` call, unchanged.
  The new crate does **not** wrap it. Under `Mask`, the upstream path is not reached.
- **No mixing.** There is no mode where a refused query falls through to upstream.
  That is `Split` by another name, and §1 removed it.

**Where the switch is read.** In each fork-owned consumer's entry point, as one
`const ROUTE: Route` (or one Cargo feature that sets it). The consumers that parse
Cypher outside tests today:

- `crates/cognitive-shader-driver/src/cypher_bridge.rs`;
- `crates/lance-graph-planner/src/strategy/cypher_parse.rs`;
- `crates/lance-graph-python/src/graph.rs`.

Rerouting one of them is a one-line change at its call site, plus rendering
`Answer` → its own output type at its boundary.

**OQ-CML-3 — is `lance-graph-python` fork-owned?** If it came from upstream, it keeps
`Route::Upstream`. Our Python surface then needs its own entry point in a fork crate,
not an edit to the inherited one.

**The flip gate.** A consumer flips to `Route::Mask` only when the census of **its
own** queries (not the test corpus) shows no refusal it needs. Until then it stays on
`Upstream`. The switch exists from D-CML-1 on. Flipping it is a per-consumer decision
with a number attached.

---

## §7 — Build order (each step names what it depends on and what would stop it)

| D-id | what | depends on | STOP if |
|---|---|---|---|
| **D-CML-0** | Verify the §2 upstream/fork split against upstream history. Create the crate skeleton (workspace member, three deps, import fence test). Measure OQ-CML-1(a): DataFusion symbols in a release build of the skeleton | — | a §2 file is fork-owned (the split is redrawn, not the design) |
| **D-CML-1** | `Route` + `run` stub that refuses everything with `R-UNBOUND-LABEL`. The switch compiles and is honest from day one | 0 | — |
| **D-CML-2** | **The classifier.** Walk the public `LogicalOperator` and return `Lowerable` or the §5 variant. Reimplement #1305's carrier-kind logic (`consumer_semantics()`: TerminalSet / EarlierSet / TerminalCount / EarlierCount / Bindings) **here**, over the public enum, because #1305 put it inside `logical_plan.rs`, an upstream file. Re-run the W0-b census under v2 and report per-variant counts | 0 | — |
| **D-CML-3** | `LabelBinding` — label → `LabelDTO` → classid, built from `LabelDTO::from_canonical`. First consumer of `LabelDTO`. Settle the classid width (v1 OQ-1: `u16` in `class_view.rs`, `u32` facet prefix) by reading a real bake | 0 | the bake's labels are not in the codebook (modelgraph §16.3 already measured this caveat), in which case the binding is supplied explicitly by the consumer, never guessed |
| **D-CML-4** | Node + predicate + Boolean lowering (v1 Wave 1 scope) through `mask_risc::execute`. The class scan is a `Pred` over `MailboxSoaView`. **`mailbox_scan::match_nodes_by_class` (which returns a `Vec`) is not used and not edited** | 2, 3 | — |
| **D-CML-5** | The hop over an in-row lane (§4.1), absolute targets first | 4, OQ-CML-2 | the relationship's targets are only relative and no `Shift` exists → substrate-first PR in mask-risc/ndarray, then resume |
| **D-CML-6** | Relative-target hop (witness loci) via `Shift` | 5 + the substrate `Shift` PR | — |
| **D-CML-7** | Variable length as reachability (§4.2) | 5 | — |
| **D-CML-8** | **The differential.** Reference = the upstream DataFusion engine, as a **dev-dependency only**, compared on **support** (`DISTINCT` node sets), never on bag counts. Second reference = quack's DuckDB oracle fixtures, where a query is expressible there | 4 (grows with 5–7) | — |
| **D-CML-9** | Reroute the first fork consumer (`cognitive-shader-driver`) to `Route::Mask`, after its own query census passes the flip gate | 8 | its census needs a refused construct → it stays on `Upstream`, and the number is recorded |

Two things run in parallel: D-CML-2 and D-CML-3 both depend only on D-CML-0. D-CML-4
is the first step that executes anything.

---

## §8 — Falsifiers (v1 §7's discipline; the rows that are new or changed)

| id | assertion | its disable |
|---|---|---|
| **F-CML-UP** | no file in §2's upstream list differs from `origin/main` at the merge base, for every PR of this plan | touch one byte of `parser.rs`; the gate must go red |
| **F-CML-FENCE** | the new crate references no `lance_graph::{query::CypherQuery::execute, datafusion_planner, ExecutionStrategy}` symbol outside `#[cfg(test)]` | add one call; the gate must go red |
| **F-CML-REFUSE** (can-fire) | every §5 variant is produced by at least one committed query | delete a variant's arm; its query must now either lower (and fail the differential) or panic |
| **F-CML-QUIET** (can-stay-silent) | a lowerable query produces no refusal, on a corpus where the refusals are a minority of queries that are not trivial | force the classifier to refuse `WHERE`; the Full count must drop |
| **F-CML-SUPPORT** | for every lowerable query, the mask equals the DataFusion `DISTINCT` node set of the returned variable, as a set | the v1 wrong-immediate test (`AND2 0xC0` for `AND3 0x80`) must go red on at least one fixture |
| **F-CML-BAG** | a 2-hop `count(*)` is **refused**, not answered with a popcount (#1305's 4 vs 3) | remove R-BAG; the differential must disagree |
| **F-CML-NOMIX** | under `Route::Mask`, a refused query never reaches `CypherQuery::execute` | a counting shim on the upstream entry must read 0 |

---

## §9 — Relation to #1303 and #1305

- **#1305** (`cypher-mask-multiplicity-contract-v1`) measured the facts v2 depends on:
  support vs bag, walks vs trails, and the five carrier kinds. Its code sits in an
  upstream file, so v2 takes **the findings and the classification** and reimplements
  them in D-CML-2. It takes none of the edit.
- **#1303** is independent. The execution-socket text in its §11/§11R (external
  relation tables, counting in the socket) is **not** carried into v2. §4.1's in-row
  hop replaces it.

Neither PR is closed or changed by this plan. That decision is the operator's.

---

## §10 — Non-goals added by v2 (v1's N-1..N-12 still apply)

- **N-13 — no edit to any §2 upstream file**, not even a `pub` or a doc comment.
- **N-14 — no per-query fallback to DataFusion**, not even in a feature-gated debug mode.
- **N-15 — no external edge table.** Relations are read from the row (§4.1). If a
  relationship type has no in-row carrier, it is refused, not given a side table.
- **N-16 — no counting in traversal.** Only terminals count, and only on the final mask.

---

## §11 — Board hygiene owed when this plan lands

- STATUS_BOARD rows D-CML-0..9 (Queued).
- INTEGRATION_PLANS prepend.
- The board entry named in the header.
- `entries_index.py --write`, then `SUPERSESSION-INDEX` regenerated **last**.
