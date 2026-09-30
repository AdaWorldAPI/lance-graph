# cypher-mask-lowering-v2 — Cypher on masks, as a fork-owned replacement behind one switch

> **Status:** PROPOSAL (D-CML-0..9). Plan only; no code is authorized by this file.
> **Written against:** `main` `0d31c54f` (2026-09-30).
> **Supersedes:** `cypher-mask-lowering-v1.md` on three points, listed in §1.
> Everything else in v1 still stands: the lowering table (§3), the placement ruling
> (§5.2), the substrate laws (§5.3), the non-goals N-2..N-12 and the falsifier discipline
> (§7.5). v1 is cited here, not restated. **One caveat on that table, from #1305
> (§12 H-6):** its count, terminal and fixpoint rows hold only BEFORE the first hop.
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

**Public surface: one function, an answer, and a refusal.**

```rust
pub fn run(text: &str, bind: &LabelBinding, view: &MailboxSoaView<'_>) -> Result<Answer, Refusal>;
// `LabelBinding::graph_config()` builds the upstream `GraphConfig` the planner needs,
// through the public `GraphConfig::builder()` (`with_node_label` / `with_relationship`)
pub struct Answer { pub items: Vec<Item> }  // one per RETURN item, in RETURN order
pub enum Item    { Mask(/* a mask handle */), Scalar(i64), Bool(bool) }
pub enum Refusal { /* one variant per row of §5, each carrying the offending construct */ }
```

`Answer.items` is sized by the query's `RETURN` clause (a handful), never by rows. If
any single item cannot be lowered, the whole query is refused. There are no partial
answers.

**The pipeline inside `run`, in order.** Every failure along it is a `Refusal`, so no
failure can produce an `Answer`:
1. **Parse.** A parser error becomes R-UNPARSED, carrying the upstream `GraphError`.
2. **Label check against `LabelBinding`**, by walking the parsed AST. This happens
   **before** planning: the planner rejects an unmapped label itself, so the check has
   to run first for R-UNBOUND-LABEL to be reachable.
3. **Semantic analysis.** A non-empty `SemanticResult.errors` becomes R-UNPLANNED,
   carrying the errors.
4. **Logical planning.** A planner error becomes R-UNPLANNED.
5. **Classification** (D-CML-2) over `(LogicalOperator, &LabelBinding, the view's lane
   catalogue)`. R-TRANSPOSE and R-CROSS-SPACE need to know which lanes exist, and the
   plan alone cannot say.
6. **Lowering and execution.**

- **The upstream planner needs a `GraphConfig`.** `LogicalPlanner::plan_return_clause`
  looks up each label's `NodeMapping` and fails with "Node label … doesn't exist" when
  there is none. `LabelBinding` therefore owns a `GraphConfig`. It builds that config
  through the public builder: one `with_node_label` per bound label, and one
  `with_relationship` per bound relationship type. The mapping's `id_field` and
  `property_fields` exist only so the planner accepts the query. The mask answer never
  reads them. Building the config is a use of the upstream API, not an edit to it.
- `Answer` carries a mask, never a `Vec` of row ids.
- Turning a mask into rows or Arrow batches is **rendering**. It happens at the
  consumer, through one function whose name starts with `materialize`, and its cost
  is O(n), stated in its doc comment. This is v1 §7.1's one-materialiser rule, kept.
- **Traversal counts nothing.** Hops and fixpoints move masks. The only popcount is a
  terminal on the *final* mask (`count(DISTINCT n)`, `exists`). A count of paths or
  walks is refused (§5 row R-BAG). After a push hop there is no terminal at all, only
  the mask itself (§4.1).

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
| `(n:Label)` | `Pred` equality on the classid lane, with the classid taken from `LabelBinding` | **only for a `u32` classid:** `EqU32Strided` over the canonical rows' key bytes 0..4. `MailboxSoaView::class_id()` is `&[u16]`, and mask-risc has no `u16` lane. Reading that view needs a narrow predicate, and that is a **STOP → substrate-first** (v1 OQ-3). Widening the column into a copy is not allowed |
| `WHERE n.p <op> literal` | one `Pred` per leaf | yes, for i32 / u32 / u64 lanes (v1 OQ-3 covers narrower lanes) |
| `AND` / `OR` / `NOT` / `XOR`, up to 3 leaves per pass | `fuse` → one `Ternlog{imm}` | yes (D-MRX-3) |
| `(a)-[:R]->(b)` — **the mask join** | depends on which row holds the pointer (§4.1): a **pull** (`Gather` / `EqU32Via`) when the target row points back at its source, a **push** (`ScatterOrU32`) when the source row points at its target | pull: yes, and it can be chained. Push: yes, but **only as the final hop** (§4.1) |
| a join on equal keys across two populations | `EqU32Via` / `GroupKey::Via` through an index lane | yes |
| `count(DISTINCT n)`, `exists` | `Terminal::Count` / `Any` on the final mask | yes |
| `min/max(n.p)` over the final mask | `MaskedMin/MaxI32` | yes (i32). Correct after a hop too: repeating a value does not change a min or a max |
| `sum(n.p)` | `MaskedSumI32` | **before any hop only.** After a hop, Cypher sums once per path, so repeated values add up. That is refused (R-BAG) |
| `avg(n.p)` | — | **refused everywhere (R-VALUE).** An average can be fractional, and `Item::Scalar(i64)` cannot hold it. An exact `(sum, count)` rational item is v1 OQ-7's question and needs its own D-id |
| `RETURN n` | `Terminal::Keep` → `Answer::Mask` | yes |
| `*1..k`, `*` | delta-frontier fixpoint over the hop (§4.2) | **only over pull hops.** Each step is one `Program`, and each step's output mask is a resident plane for the next. Over push hops it is a STOP (§4.1) |

### §4.1 — The hop reads relations that are stored in the row

A hop does **not** join against an external edge table. The substrate stores a row's
relations **inside the row**:

- the second facet (bytes 16..32, `EdgeBlock = FacetCascade`);
- the `CausalWitness` value tenant (`canonical_node.rs`, `ValueTenant::CausalWitness = 14`:
  24 signed i4 loci, each a context pointer within a ±8 window, all-zero = unbound);
- the episodic-basin rails (32 bytes of references).

The hop reads one of these as a **lane** (`LaneRef::Strided` over the 512-byte row)
and moves the mask from source rows to the rows they point at.

**Absolute targets** (a lane holding a row ordinal). There are two cases, and which one
applies depends on whose row holds the pointer:

- **Pull.** The *target* row stores the ordinal of its source. The hop is
  `Gather(src_mask, lane)` (or `EqU32Via`). Its output is an ordinary mask over the
  target rows, so it can feed the next hop, a predicate, or a fixpoint step.
- **Push.** The *source* row stores the ordinal of its target. The hop is
  `Terminal::ScatterOrU32`. mask-risc's survival condition (`ir.rs`, on `ScatterOrU32`)
  allows the scattered mask **only as the externally demanded result**: *"never an
  intermediate: a follow-on program or fold that consumes it is the forbidden
  projection → population → projection shape."* A push hop is itself the program's one
  terminal. So it answers **only `RETURN b`**, the target mask itself:
  - no `count`, `exists`, `sum` or `min`/`max` over `b`;
  - no `WHERE` on `b`;
  - no further hop.

  Each of those would consume the scattered mask. The one exception is
  `count(DISTINCT b)`, which could become `Terminal::CountKeyRunsU32`, but only over a
  key-ordered lane; any other lane is refused. A chain of two push hops, a fixpoint over
  push hops, or anything computed on a push hop's result is a
  **STOP → substrate-first**. It needs either a mask-risc ruling that admits a scattered
  mask as a resident plane, with its own law and falsifier, or a pull lane on the
  target rows. Until then such a query is refused (R-CHAIN).

**Relative targets** (the witness loci: offsets within ±8). Each row stores its **own**
offset, so a row's offset is applied only to that row. The source mask is split into
one part per offset value `d`:
1. `part_d = src AND (locus == d)`, one predicate per non-zero value `d` in −8..=8;
2. each `part_d` is shifted by `d`;
3. the shifted parts are OR-ed together.

The all-zero locus means **unbound** and is excluded, never read as "offset 0". Shifting
`src` as a whole would apply one row's offset to every other row.

**OQ-CML-2 — which carrier holds absolute and which relative targets, per relationship
type; and whether mask-risc needs a `Shift` op.** mask-risc has no shift op today. If
the relative case is real, `Shift` is a STOP → substrate-first (mask-risc IR plus the
`ndarray::simd` primitive). It must not be written locally.

The v1 laws apply unchanged. **One population per mask** (v1 §5.3b): a hop's target
must index the same row space. **The transpose law** (§5.3c): `<-[:R]-` reads the
reverse lane if one exists. If none exists, it is refused (R-TRANSPOSE) — never
answered by reading the forward lane backwards.

### §4.2 — Variable length is reachability, and only reachability

`*1..k` and `*` (lower bound 1, over pull hops only) lower to: frontier = hop(frontier)
AND NOT visited; visited |= frontier; repeat until `k` steps or until
`Any(frontier) == false`. The result is `visited`: the **set of rows reachable in 1..k
steps**.

- **`visited` starts EMPTY, not seeded with the start set.** A start row that is
  reachable again through a cycle (`A→B→A`) must appear in the answer. Seeding
  `visited` would drop it.
- This is exact for a lower bound of 1. A shortest reaching walk of length ≤ k is
  always a trail, so walk-reachability and trail-reachability give the same set.
- **A lower bound of 0** adds the start set.
- **A lower bound above 1** (`*2..2`, `*m..n`, `*m..`) is **refused (R-DEPTH).** With
  `min > 1`, walk semantics and Cypher's trail semantics give different endpoint sets:
  a walk may repeat an edge, a trail may not. A frontier without the visited check
  computes walks, and the visited check computes shortest distances. Neither is the
  trail answer. A depth-aware lowering needs its own D-id and falsifier.

Cypher's own semantics for a variable-length pattern bind *paths*. So any query whose
answer depends on paths is refused (R-BAG): `count(*)` over the pattern, returning the
path, or `length(p)`. #1305's measured finding says exactly this: DataFusion counts
walks (5), the trail count is 4, and the mask gives the support.

**The deliberate rule: v2 answers reachability questions and refuses path questions.**
It does not approximate one with the other.

---

## §5 — The refusal list (replaces v1 §4's `[GRACE]` list)

Each row is a `Refusal` variant. Each is named **before** any program runs, at one of
the pipeline steps in §3: R-UNPARSED at parse, R-UNBOUND-LABEL at the label check,
R-UNPLANNED at semantic analysis and planning, and every other variant in the
classifier.

| variant | construct | why it has no mask form |
|---|---|---|
| **R-ORDER** | `ORDER BY`, and **any** `SKIP` / `LIMIT`, with or without an order (the planner emits standalone `Offset` / `Limit`) | a mask has no order and no position; truncating a mask is not defined |
| **R-BAG** | **any non-`DISTINCT` count or sum after one or more hops** (`count(*)`, `count(expr)`, `sum`), plus `collect`, returning paths, and `length(p)` | a mask is support, not a bag (#1305). Even ONE hop has bag semantics: two sources, or two parallel relationships, reaching one target give `count(*) = 2` over one mask bit |
| **R-CHAIN** | anything that consumes a push hop's result (§4.1): another hop, a fixpoint, a `WHERE` on the target, or any aggregate over it except `count(DISTINCT)` over a key-ordered lane | mask-risc forbids a scattered mask as an intermediate, and the push hop is already the program's one terminal |
| **R-SHAPE** | a pattern that is not one chain: hops that do not connect end to end, a variable bound twice (at a hop or at the pattern's start), or two disconnected patterns | a mask chain carries one frontier; these were real bugs caught on #1305 (§12 H-4) |
| **R-DEPTH** | a variable-length pattern with a lower bound above 1 | walk ≠ trail at `min > 1` (§4.2) |
| **R-DISTINCT-VALUE** | `DISTINCT` over a projected value (not a node) | a value set is not a row set |
| **R-STRING** | string predicates beyond equality through a dictionary lane (`CONTAINS`, `STARTS WITH`, regex) | variable-width values; v1 §4.2 |
| **R-UNWIND / R-WITH-AGG** | `UNWIND`, aggregation inside `WITH` | bag re-entry |
| **R-VALUE** | vector distance / similarity, NARS truth, floats, and `avg` (fractional) | values, not Boolean; v1 N-7. mask-risc is integer-only |
| **R-CROSS-SPACE** | a join whose two sides index different row spaces with no index lane between them | the one-population law |
| **R-TRANSPOSE** | an incoming hop with no reverse lane | the transpose law |
| **R-UNBOUND-LABEL** | a label with no `LabelBinding` entry | no classid to test; guessing one is the confident-and-wrong quadrant (v1 OQ-1) |
| **R-UNPLANNED** | a non-empty `SemanticResult.errors`, or an error from the upstream logical planner | the refusal carries the upstream errors; nothing is lowered from a plan that does not exist |
| **R-UNPARSED** | anything the upstream parser rejects, **including `CREATE` / `SET` / `DELETE` / `MERGE`**: the reused parser accepts only reading clauses followed by `RETURN`, so mutations never reach a `LogicalOperator` | the refusal carries the parse error. Mutations stay out of scope (v1 N-8: writes go through the commit gate) |

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
not a property of the queries a consumer actually sends. **The 37.3 % is an upper
bound (§12 H-5):** once path multiplicity is classified, #1305 reports 70 Full of 313
(≈ 22 %) — reported there, not re-run here. D-CML-2 re-runs the census
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

**Where the switch is read.** In a fork-owned consumer's entry point, as one
`const ROUTE: Route` (or one Cargo feature that sets it). **A consumer can take the
switch only if it has both halves:** an existing call that executes Cypher (for
`Upstream`), and a `MailboxSoaView` plus a `LabelBinding` (for `Mask`). The three
non-test files that mention Cypher today do not all have both:

- `crates/cognitive-shader-driver/src/cypher_bridge.rs` is a **stateless prefix
  classifier**. It has no `lance-graph` dependency, no execute call, and no view in its
  `route` method. **Neither route exists there yet.** It needs storage and config
  plumbing first, and that plumbing is D-CML-9's scope. It is not a one-line reroute.
- `crates/lance-graph-planner/src/strategy/cypher_parse.rs`: a strategy module, not an
  executor. It is read at D-CML-9.
- `crates/lance-graph-python/src/graph.rs` executes Cypher. Whether it is fork-owned is
  OQ-CML-3.

For a consumer that has both halves, rerouting is a change at its call site, plus
rendering `Answer` into its own output type at its boundary.

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
| **D-CML-2** | **The classifier.** Walk the public `LogicalOperator` and return `Lowerable { variables whose node sets are asked for }` or a §5 `Refusal`. Two answers only. **#1305's `consumer_semantics()` is NOT ported:** its count and binding kinds describe per-path state to be carried through a hop, and v2 refuses those queries instead (§12). Ported from #1305: the three pattern-shape refusals (§12 H-4) and the fixtures (§12 H-1..H-3). Re-run the W0-b census under v2 and report per-variant counts | 0 | — |
| **D-CML-3** | `LabelBinding` — label → `LabelDTO` → classid, built from `LabelDTO::from_canonical`. First consumer of `LabelDTO`. Settle the classid width (v1 OQ-1: `u16` in `class_view.rs`, `u32` facet prefix) by reading a real bake | 0 | the bake's labels are not in the codebook (modelgraph §16.3 already measured this caveat), in which case the binding is supplied explicitly by the consumer, never guessed |
| **D-CML-4** | Node + predicate + Boolean lowering (v1 Wave 1 scope) through `mask_risc::execute`. The class scan is `EqU32Strided` over the canonical rows' `u32` classid. **`mailbox_scan::match_nodes_by_class` (which returns a `Vec`) is not used and not edited** | 2, 3 | the only classid lane available is `MailboxSoaView`'s `&[u16]` → substrate-first narrow predicate (v1 OQ-3), then resume |
| **D-CML-5** | The hop over an in-row lane (§4.1): pull hops (chainable) first, then a push hop as the last hop only | 4, OQ-CML-2 | the relationship's targets are only relative and no `Shift` exists, or a query needs push hops chained → substrate-first PR in mask-risc/ndarray, then resume |
| **D-CML-6** | Relative-target hop (witness loci) via `Shift` | 5 + the substrate `Shift` PR | — |
| **D-CML-7** | Variable length as reachability, lower bound 0 or 1, over pull hops (§4.2). `visited` starts empty | 5 | — |
| **D-CML-8** | **The differential.** Reference = the upstream DataFusion engine, as a **dev-dependency only**, compared on **support** (`DISTINCT` node sets), never on bag counts. Second reference = quack's DuckDB oracle fixtures, where a query is expressible there | 4 (grows with 5–7) | — |
| **D-CML-9** | The first consumer that can take the switch. `cognitive-shader-driver`'s bridge has neither route today, so this step first gives it a `MailboxSoaView`, a `LabelBinding` and a defined legacy route. Then it flips to `Route::Mask`, after its own query census passes the flip gate | 8 | its census needs a refused construct → it stays on its legacy route, and the number is recorded |

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
| **F-CML-BAG** | a `count(*)` after ONE hop, over a fixture with two sources reaching one target, is **refused**, not answered with a popcount. So is the 2-hop case (§12 H-1: 4 paths vs 3 nodes) | remove R-BAG; the differential must disagree |
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

---

## §12 — Harvested from #1305 (not merged; evidence stays on its branch)

#1305 (`cypher-mask-multiplicity-contract-v1`) is **closed unmerged**: its code put a
per-path carrier classifier (`consumer_semantics()`) into the upstream
`logical_plan.rs`, which v2 rules out twice (upstream edit; counting in traversal).
Its **measurements** are correct and v2 depends on them. These six are the must-haves.
Everything else — the classifier's five kinds, the count lanes (D-CMM-4/5/6),
`carrier_sufficiency.py`, the edits to v1 and to the DataFusion test file — stays on
branch `ccr-2fcc2bd3-8o7m2l` at `67abd29` and is not needed by v2.

| id | fact | where v2 uses it |
|---|---|---|
| **H-1** | **A mask is support, not bag.** Cypher `count(*)` after a hop counts PATHS; a popcount counts NODES. Fixture: KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5}; 2-hop `count(*)` = **4** (paths), mask = **3** (end nodes). Same 4 vs 3 for the var-length form | §5 R-BAG; F-CML-BAG |
| **H-2** | **A forward chain gives the exact node set of the LAST variable only.** An earlier variable's set is not the forward frontier (`count(DISTINCT b)`: 3 vs 4); it needs a backward hop over the reverse lane, or R-TRANSPOSE | D-CML-4/5 lowering; the classifier's `Lowerable` must name WHICH variables are asked for |
| **H-3** | **DataFusion counts walks, and walks ≠ trails.** On {1→2, 2→1, 2→2} DataFusion returns **5** walks; trails are **4**. They diverge without a cycle too: `(a)->(b)<-(c)` on {1→2} = 1 walk, 0 trails. "Path count" is therefore not one notion | D-CML-8 compares on `DISTINCT` node sets ONLY, never counts; §4.2 refuses path questions |
| **H-4** | **Three pattern shapes are not a chain** and must be refused, not lowered as one (each was a real bug caught in review on #1305): hops that do not connect end to end; a variable bound twice (at a hop or at the pattern start); two disconnected patterns | D-CML-2 refusals |
| **H-5** | **The census drops once multiplicity is classified:** Full 117 → **70** of **313** (≈ 22 %), against the 37.3 % (113/303) quoted in §5 | §5 cost; §6 flip gate |
| **H-6** | **v1's terminal rows are pre-hop only.** T-1..T-7, T-11 and R-6..R-8 hold before the first hop; after a hop they hold only as `DISTINCT` node sets | §0 header caveat; D-CML-4..7 |

