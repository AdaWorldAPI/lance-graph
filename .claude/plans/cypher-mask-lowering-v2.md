# cypher-mask-lowering-v2 — Cypher on masks, as a fork-owned replacement behind one switch

> **Status:** PROPOSAL (D-CML-0..10, D-CML-3b, D-CML-5a). Plan only; no code is authorized by this file.
> **Council:** 5+3, ratified v3 (2026-09-30). The change ledger is §13.
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

- `lance_graph::parser::parse_cypher_query(&str) -> Result<ast::CypherQuery>` (`parser.rs:23`).
  This returns the **AST** type `ast::CypherQuery`. It is a different type from
  `query::CypherQuery` (`query.rs:121`), which owns `execute`. This plan always writes
  the full path.
- `lance_graph::semantic::SemanticAnalyzer::{new(GraphConfig), analyze(&ast, &params)}`
  (`semantic.rs:65,74`). `analyze` also takes a parameter map (`semantic.rs:77`). The
  upstream DataFusion path plans the **parameter-substituted** AST that the analyzer
  returns (`query.rs:934,945`), not the raw AST. So does `run`.
- `lance_graph::logical_plan::{LogicalPlanner::{new, plan}, LogicalOperator}` (`logical_plan.rs:19,160,168`);
- the `ast` types that `LogicalOperator` carries (`BooleanExpression`, `ValueExpression`, `PropertyValue`, …).

The census example `examples/w0b_corpus_census.rs` drives **parse → plan** from outside
the planner, which proves that part of the chain is reachable without edits. It does
**not** run `SemanticAnalyzer`, and it plans with an empty `GraphConfig::default()`
(`w0b_corpus_census.rs:428,475-480`). So every `RETURN n` query in its corpus fails to
plan, and its figures are not comparable to a run with a `LabelBinding` (§5).

---

## §3 — The crate

**`crates/lance-graph-cypher-mask`** (fork-owned, new).

| depends on | for | never for |
|---|---|---|
| `lance-graph` (**`default-features = false`**) | the parser, semantic and logical-plan API of §2 | `query::CypherQuery::execute`, `datafusion_planner`, `SqlDialect`, anything that runs a query. With default features, `lance-graph` also pulls in `planner`, `lance-graph-cognitive`, `bgz17` and `bgz-tensor` (`Cargo.toml:98-139`). Those are not needed here |
| `lance-graph-mask-risc` | `Program`, `MaskOp`, `Pred`, `Terminal`, `execute`, `Planes` | — |
| `lance-graph-contract` | `LabelDTO`, `canonical_node` (`NodeRow`), `facet` | cognitive modules (same import fence as quack: named modules only) |

**Public surface: one function, an answer, and a refusal.**

```rust
pub fn run(
    text: &str,
    params: &HashMap<String, serde_json::Value>, // the same map `analyze` takes
    bind: &LabelBinding,
    rows: &[NodeRow],
) -> Result<Answer, Refusal>;
// `rows` = the baked canonical spine (512-byte `NodeRow`s), borrowed, never copied.
// `LabelBinding::graph_config()` builds the upstream `GraphConfig` the planner needs,
// through the public `GraphConfig::builder()` (`with_node_label` / `with_relationship`)
pub struct Answer { pub items: Vec<Item> }  // one per RETURN item, in RETURN order
pub enum Item    { Mask(MaskHandle), Scalar(i64), Bool(bool) }
// MaskHandle: opaque and owner-held. It is not a `Vec<u64>` of words or of row ids.
pub enum Refusal { /* one variant per row of §5, each carrying the offending construct */ }
```

`Answer.items` is sized by the query's `RETURN` clause (a handful), never by rows. If
any single item cannot be lowered, the whole query is refused. There are no partial
answers.

**Why `&[NodeRow]` and not `MailboxSoaView`:** `MailboxSoaView` exposes `class_id()`
only as `&[u16]` (`soa_view.rs:89-102`), and it has no row-bytes accessor. The `u32`
classid at key bytes 0..4 and the in-row relation bytes can only be reached as strided
lanes over the 512-byte rows (`LaneRef::Strided`, `ir.rs:42-51`). `tests/strided.rs:58`
already exercises `RECORD_STRIDE = 512` with the classid at +0. Borrowing the rows as
bytes must be a zero-copy view. If `NodeRow` has no sound byte view, that is a STOP →
contract-first (D-CML-0 checks it).

**The pipeline inside `run`, in order.** Every failure along it is a `Refusal`, so no
failure can produce an `Answer`:
1. **Parse.** A parser error becomes RF-UNPARSED, carrying an **owned reason** (message
   and source position) that the front-end adapter extracts from the upstream
   `GraphError`. The `GraphError` itself never crosses the crate's public surface: it
   wraps `DataFusionError`, `lance::Error` and `ArrowError` (`error.rs:43-74`). The same
   conversion applies to every upstream error in steps 3 and 4.
2. **Label check against `LabelBinding`**, by walking the parsed AST, before
   planning. The upstream check is **partial**: the planner rejects an unmapped label
   only when a node variable is returned (`logical_plan.rs:517-526`), and semantic
   analysis only warns (`semantic.rs:636-642`). So `MATCH (n:X) RETURN count(n)` plans
   fine upstream, and the label check must be our own.
3. **Semantic analysis** with the query's parameters. A non-empty
   `SemanticResult.errors` becomes RF-UNPLANNED, carrying the errors.
4. **Logical planning** of the analyzer's substituted AST. A planner error becomes
   RF-UNPLANNED.
5. **Classification** (D-CML-2) over `(LogicalOperator, &LabelBinding)`. The
   relationship declarations that RF-TRANSPOSE, RF-CROSS-SPACE and RF-UNDECLARED-REL
   need come from the binding (§4.1), not from inspecting the rows.
6. **Lowering and execution.** Each `RETURN` item is its own `Program`, and each
   program re-evaluates the shared match prefix. Items never share a scattered
   intermediate. An `ExecError` raised during execution (e.g. `LaneNotOrdered`,
   `ir.rs:354-365`; `SumRowBound`, `reference.rs:483-485`) becomes RF-EXEC, carrying
   the error. It is a refusal, never a partial answer.

- **One front-end adapter.** Steps 1–4 live in one module that turns upstream types
  into the lowering's input. The rest of the crate never names an upstream type. If
  planning moves to ogar-loco / ogar-r2il (F7), or if the upstream planner changes
  shape, only that adapter is replaced. `LogicalOperator` is the chosen input for now
  because it is already normalised and the census drives it. Its known bias is that
  `VariableLengthExpand` is documented as unroll-and-union (`logical_plan.rs:70-71`),
  i.e. walk-shaped. §4.2 ignores that expansion and reads only the bounds.
- **The upstream planner needs a `GraphConfig`.** `LogicalPlanner::plan_return_clause`
  looks up each label's `NodeMapping` and fails with "Node label … doesn't exist" when
  there is none. `LabelBinding` therefore owns a `GraphConfig`. It builds that config
  through the public builder: one `with_node_label` per bound label, and one
  `with_relationship` per bound relationship type. The mapping's `id_field` and
  `property_fields` exist only so the planner accepts the query. `with_relationship`
  also requires non-empty `source_id_field` / `target_id_field` (`config.rs:165,
  222-234`). These are **placeholders**, not an edge table (F3). A test asserts that
  the lowering reads none of these fields. Building the config is a use of the
  upstream API, not an edit to it.
- `Answer` carries a mask handle, never a `Vec` of row ids and never a word vector.
- Turning a mask into rows or Arrow batches is **rendering**. It happens at the
  consumer, through one function whose name starts with `materialize`, and its cost
  is O(n), stated in its doc comment. This is v1 §7.1's one-materialiser rule, kept.
- **Traversal counts nothing.** Hops and fixpoints move masks. The only popcount is a
  terminal on the *final* mask (`count(DISTINCT n)`, `exists`). A count of paths or
  walks is refused (§5 row RF-BAG). After a push hop there is no terminal at all, only
  the mask itself (§4.1).

**OQ-CML-1 — what "DataFusion is linked" means, split into four questions.**
⊘ The earlier wording asked one question, "is DataFusion linked?", and answered it with
an `nm` scan. The premise gate (`.claude/agents/premise-auditor.md`, 2026-10-03) showed
that "linked" covered four concepts with different answers and different deciders:

| | question | status | how it is decided |
|---|---|---|---|
| **C2 build graph** | does compiling the new crate compile DataFusion? | **certain, yes**: `datafusion`, `lance`, `arrow`, `object_store` and `lance-graph-hydrate` are non-optional (the `[dependencies]` block of `crates/lance-graph/Cargo.toml`), and `pub mod datafusion_planner` is ungated (`crates/lance-graph/src/lib.rs`) | not measured. **Accepted as a stated build cost.** Removing it needs an upstream edit (ruled out) or a pinned copy of the front end (8 files, 6,845 lines including `error.rs`, `config.rs`, `case_insensitive.rs`, `parameter_substitution.rs`; and the copied `error.rs` still names `DataFusionError`) — a second AST authority, so a STOP, not a remedy |
| **C4 call path** | does `run` execute any DataFusion code? | must be **no** | the import fence F-CML-FENCE (§8), with its disable per path; not `nm` |
| **C5 public types** | does any public type of the new crate name an upstream type? | must be **no** | a fork-owned design rule, decided here: upstream errors become owned reasons in the adapter (§3 step 1). F-CML-SURFACE (§8) checks it |
| **C3 binary residue** | does a release artifact still contain `datafusion*` symbols? | an observation | the `nm` step over a `--release` `[[example]]`. It is **recorded, not a gate**: a hit does not by itself trigger a STOP, because C3 can be non-empty for reasons that are not C4 (e.g. a `Display` impl reached through a formatted error) |

---

## §4 — What lowers, and to which mask-risc shape

These are the rows of v1 §3, re-sorted by what the mask-risc IR already provides
today. Nothing here adds an op to the IR. A missing op is a STOP (v1 N-10): it lands
substrate-first, in `ndarray::simd` and then mask-risc, with its own parity test.

**The one constraint behind most rows:** `run` gets only the rows. Over row bytes, the
IR reads a strided lane in exactly four places: `EqU32Strided`, `NeU32Strided`,
`MatchFacetStrided` and `MaskedStridedGroupSum` (`ir.rs:30-34`; `reference.rs:82-84`).
Everything else (ordered compares, `MaskedSum/Min/MaxI32`, `Gather`, `ScatterOrU32`,
`CountKeyRunsU32`, `EqU32Via`) needs a contiguous lane (`reference.rs:70-74, 81,
380-381, 428-433, 456-459, 479-491, 522-534`). Copying a column out of the rows to feed
them is a population-sized copy, so those rows read **NO** until D-CML-5a lands.

| Cypher | lowering | IR today |
|---|---|---|
| `(n:Label)` | `Pred::EqU32Strided` on the `u32` classid at key bytes 0..4 of each 512-byte row, with the classid taken from `LabelBinding` | **yes.** `EqU32Strided` reads a little-endian `u32` at `first_offset + i*stride` (`ir.rs:184`; `exec.rs:667-672`), and `tests/strided.rs:58` covers stride 512 |
| `WHERE n.p = / <> literal` on a declared `u32` property field | `EqU32Strided` / `NeU32Strided` at the field's declared offset (D-CML-3) | **yes** |
| `WHERE n.p` masked-equality on a declared facet field | `MatchFacetStrided` | **yes** |
| `WHERE n.p < / <= / > / >= literal` | ordered `Pred` | **NO over in-row bytes** (contiguous `I32` only). STOP → D-CML-5a |
| `AND` / `OR` / `NOT` / `XOR`, up to 3 leaves per pass | `fuse` → one `Ternlog{imm}` | yes (D-MRX-3) |
| `(a)-[:R]->(b)` — **the mask join** | depends on the relationship's **declared** carrier and direction (§4.1): a **pull** (`Gather`) when the target row holds the pointer, a **push** (`ScatterOrU32`) when the source row does | **NO, twice.** (1) No contract reading of any in-row carrier is a row pointer today (§4.1): STOP → contract-first (D-CML-3b). (2) `Gather` / `ScatterOrU32` take only a contiguous `U32` lane: STOP → substrate-first (D-CML-5a) |
| a join on equal keys across two populations | `EqU32Via` / `GroupKey::Via` through an index lane | **NO.** `run` has one population, so a second population is RF-CROSS-SPACE. Within one population, an in-row index lane waits on D-CML-5a |
| `count(DISTINCT n)`, `exists` | `Terminal::Count` / `Any` on the final mask | yes |
| `min/max(n.p)` over the final mask | `MaskedMin/MaxI32` | **NO over in-row bytes** until D-CML-5a. Semantically correct after a hop too: repeating a value does not change a min or a max |
| `sum(n.p)` | `MaskedSumI32` | **NO over in-row bytes** until D-CML-5a, and then **before any hop only.** After a hop, Cypher sums once per path, so repeated values add up. That is refused (RF-BAG) |
| `avg(n.p)` | — | **refused everywhere (RF-VALUE).** An average can be fractional, and `Item::Scalar(i64)` cannot hold it. An exact `(sum, count)` rational item is v1 OQ-7's question and needs its own D-id |
| `RETURN n` | `Terminal::Keep` → `Item::Mask` | yes |
| `ORDER BY` / `LIMIT n≥1` when **every** `RETURN` item is a scalar aggregate (one row) | no-op, dropped | yes. A one-row result is unchanged by ordering or by `LIMIT ≥ 1`. `SKIP ≥ 1` or `LIMIT 0` empty it, so those are refused (RF-ORDER) |
| `*1..k`, `*` over a **directed** relationship | delta-frontier fixpoint over the hop (§4.2) | **only over pull hops, and only after D-CML-3b and 5a.** Each step is one `Program`; each step's output mask is re-supplied to the next as a caller-owned `Foreign` plane. Over push hops it is a STOP (§4.1). Undirected variable length is refused (RF-DEPTH) |

### §4.1 — The hop reads relations that are stored in the row

A hop does **not** join against an external edge table. The substrate stores a row's
relations **inside the row**: in the second facet (bytes 16..32; `EdgeBlock` is today a
type alias of `EdgeFacet`, `canonical_node.rs:786`) or in a value tenant that the
class's `ClassView` names. **Which carrier a relationship type uses, and in which
direction its pointer is stored, is DECLARED, never decided by the lowering and never
guessed from availability.** This follows `cypher-kanban-ast-unification-v1` Inc 0: the
representation is classid-resolved through `EdgeCodecFlavor` / `ReadMode`.

- In v2 the declaration is carried by `LabelBinding`, per relationship type:
  `(carrier, field offset, field width, direction ∈ {pull, push}, reverse lane if any)`.
- A relationship type with no declaration is refused (RF-UNDECLARED-REL).
- **A declaration must agree with the class's `ClassView`.** Before a hop lowers, the
  declared carrier is checked against the class's `edge_codec_flavor` / `ReadMode`
  (`class_view.rs:1271`). A mismatch is refused (RF-UNDECLARED-REL). The edge facet is
  16 content-blind bytes, and under `Pq32x4` or `CoarseResidue` they are not ordinals
  at all.
- **No contract reading is a row pointer today.** All three `EdgeCodecFlavor` readings of
  the second facet (`CoarseOnly`, `CoarseResidue`, `Pq32x4`; `canonical_node.rs:807-819`)
  are vector-codec indices. The episodic-basin rail holds a `subject u16` plus a version
  range into the stream (`canonical_node.rs:1060-1067`), not a row ordinal. A row
  address in a 64k table is one `u8:u8` rail (`ir.rs:210-213`), which canon forbids
  widening. So an absolute pull/push hop needs a **contract-first** pointer reading
  first (D-CML-3b): a `ClassView` / flavor variant naming a row-pointer field and its
  width. Until it exists, every relationship declaration is refused, and the refusal
  says why.
- Moving the declaration into `ClassView` is a contract change and gets its own D-id,
  not this plan (OQ-CML-2).

**Relative targets: the `CausalWitness` loci — DEFERRED, not withdrawn.** The loci are
real relational pointers: each nibble is a "context pointer" to the node at
`self_pos + offset` (`causal_witness.rs:48-53`, `resolves_to` at `:329-345`), and the
named loci are relations (`Antecedent`, `BasinAnchor`, `SupportedBy`, `Supports`,
`Kausal`; `:131-138`). But they are **relative** (signed i4 in `[-8, +7]`, `:33`,
clamp `:291`), they point into the `temporal.rs` stream window rather than at a row
ordinal, and they are EXPERIMENTAL (`:9`). No relationship type is bound to them. A
relative hop (D-CML-6) therefore needs, before it can lower:
- a declared carrier kind `Relative{locus}`, valid only when the declared row order
  equals the temporal stream order;
- a **linear mask-shift** primitive. This is the one substrate gap: the per-offset
  partition already exists as `MatchFacetStrided` with a nibble care mask over the
  witness lane (row offset 204, payload 208, `canonical_node.rs:1220-1225`), and
  `ndarray` ships only `mask_shift_morton` (`simd_masking_ops.rs:4149`).

The partition rule stays as written in v1-draft: one part per non-zero offset, each part
shifted by its own offset, then OR-ed; the all-zero locus means unbound.

**Absolute targets** (a field holding a row ordinal, once D-CML-3b defines one). There
are two cases, and which one applies depends on whose row holds the pointer:

- **Pull.** The *target* row stores the ordinal of its source. The hop is
  `Gather(src_mask, lane)`. `EqU32Via` is not a hop form: it tests a foreign value lane
  against a constant (`ir.rs:148-158`) and cannot ask "is the source in the frontier".
  `Gather` reads its source mask from `Foreign::planes` (`ir.rs:94-113, 247-259`),
  never from a scratch slot. So the source mask is a caller-owned plane, and **every
  pull hop ends its program.** Its output is an ordinary mask over the target rows,
  which the next program takes as its `Foreign` plane. A self-hop passes the same
  table's plane as "foreign"; the one-population law and `Foreign`'s
  another-table wording are in tension, and OQ-CML-4 records it for D-CML-5a.
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
  target rows. Until then such a query is refused (RF-CHAIN).

**Which variable is returned is independent of which side holds the pointer.**
`MATCH (a)-[:R]->(b) WHERE b.p = k RETURN a`, where the source row holds the pointer,
does not consume a scattered mask. It filters `b` into a plane and pulls it back through
`a`'s pointer lane with `Gather`, giving a mask over `a`. RF-CHAIN applies only when a
scattered mask would be consumed.

**OQ-CML-2 — the relationship declaration.** Per relationship type: which carrier holds
the pointer, and whether a reverse lane exists. v2 answers this per binding (above).
Where the declaration lives long-term (`ClassView`, next to `edge_codec_flavor`) is open.

**OQ-CML-4 — self-hop and `Foreign`.** `Foreign::planes` is documented as a mask over
another table's row space. A self-hop needs it over the same table. D-CML-5a decides
whether that is admitted as-is, or needs its own law.

The v1 laws apply unchanged. **One population per mask** (v1 §5.3b): a hop's target
must index the same row space. **The transpose law** (§5.3c): `<-[:R]-` reads the
reverse lane if one exists. If none exists, it is refused (RF-TRANSPOSE) — never
answered by reading the forward lane backwards.

### §4.2 — Variable length is reachability, and only reachability

`*1..k` and `*` (lower bound 1, over pull hops only) lower to: frontier = hop(frontier)
AND NOT visited; visited |= frontier; repeat until `k` steps or until
`Any(frontier) == false`. The result is `visited`: the **set of rows reachable in 1..k
steps**.

- **`visited` starts EMPTY, not seeded with the start set.** A start row that is
  reachable again through a cycle (`A→B→A`) must appear in the answer. Seeding
  `visited` would drop it.
- This is exact for a lower bound of 1 **over a directed relationship**. There, a
  shortest reaching walk of length ≤ k is always a trail, so walk-reachability and
  trail-reachability give the same set. Over an **undirected** relationship it fails:
  `a-[*1..2]-a` over one edge is a walk that uses the edge twice, not a trail. So
  undirected variable length is refused (RF-DEPTH).
- **A lower bound of 0** adds the start set.
- **A lower bound above 1** (`*2..2`, `*m..n`, `*m..`) is **refused (RF-DEPTH).** With
  `min > 1`, walk semantics and Cypher's trail semantics give different endpoint sets:
  a walk may repeat an edge, a trail may not. A frontier without the visited check
  computes walks, and the visited check computes shortest distances. Neither is the
  trail answer. A depth-aware lowering needs its own D-id and falsifier.

Cypher's own semantics for a variable-length pattern bind *paths*. So any query whose
answer depends on paths is refused (RF-BAG): `count(*)` over the pattern, returning the
path, or `length(p)`. #1305's measured finding says exactly this: DataFusion counts
walks (5), the trail count is 4, and the mask gives the support.

**The deliberate rule: v2 answers reachability questions and refuses path questions.**
It does not approximate one with the other.

---

## §5 — The refusal list (replaces v1 §4's `[GRACE]` list)

Each row is a `Refusal` variant. The `RF-` prefix avoids collision with v1's hop-row
ids R-1..R-9. Each variant is named at one of the pipeline steps in §3:
- RF-UNPARSED at parse;
- RF-UNBOUND-LABEL at the label check;
- RF-UNPLANNED at semantic analysis and planning;
- RF-EXEC at execution (step 6);
- every other variant in the classifier, before any program runs.

RF-UNWIND is a classifier refusal: `UNWIND` parses (`parser.rs:81-86`). RF-CROSS-SPACE
generalises v1 §7.2 F-H4 (`cypher-mask-lowering-v1.md:820`). The house precedent for a
typed refuse-not-fallback lowering is quack's `#[non_exhaustive] enum LowerError`
(`lance-graph-quack/src/lib.rs:894-925`). `Refusal` follows it: `#[non_exhaustive]`,
and each variant carries its construct.

| variant | construct | why it has no mask form |
|---|---|---|
| **RF-ORDER** | `ORDER BY` / `SKIP` / `LIMIT` over a result that is not a single scalar row, and `SKIP ≥ 1` / `LIMIT 0` always. The planner emits standalone `Sort` / `Offset` / `Limit` (`logical_plan.rs:110-126, 216-230`) | a mask has no order and no position; truncating a mask is not defined. Over a one-row scalar result ORDER BY and `LIMIT ≥ 1` are no-ops (§4) |
| **RF-BAG** | **any non-`DISTINCT` count or sum after one or more hops** (`count(*)`, `count(expr)`, `sum`), plus `collect`, returning paths, and `length(p)` | a mask is support, not a bag (#1305). Even ONE hop has bag semantics: two sources, or two parallel relationships, reaching one target give `count(*) = 2` over one mask bit |
| **RF-CHAIN** | anything that consumes a push hop's result (§4.1): another hop, a fixpoint, a `WHERE` on the target, or any aggregate over it except `count(DISTINCT)` over a key-ordered lane | mask-risc forbids a scattered mask as an intermediate, and the push hop is already the program's one terminal |
| **RF-SHAPE** | a pattern that is not one chain: hops that do not connect end to end, a variable bound twice (at a hop or at the pattern's start), or two disconnected patterns | a mask chain carries one frontier; these were real bugs caught on #1305 (§12 H-4) |
| **RF-DEPTH** | a variable-length pattern with a lower bound above 1, or over an undirected relationship | walk ≠ trail (§4.2) |
| **RF-DISTINCT-VALUE** | `DISTINCT` over a projected value (not a node) | a value set is not a row set |
| **RF-STRING** | string predicates beyond equality through a dictionary lane (`CONTAINS`, `STARTS WITH`, regex) | variable-width values; v1 §4.2 |
| **RF-UNWIND / RF-WITH-AGG** | `UNWIND`, aggregation inside `WITH` | bag re-entry |
| **RF-VALUE** | vector distance / similarity, NARS truth, floats, and `avg` (fractional) | values, not Boolean; v1 N-7. mask-risc is integer-only |
| **RF-CROSS-SPACE** | a join whose two sides index different row spaces with no index lane between them | the one-population law |
| **RF-TRANSPOSE** | an incoming hop with no reverse lane | the transpose law |
| **RF-UNDECLARED-REL** | a relationship type with no carrier/direction declaration in `LabelBinding` | the lowering never guesses where a relation is stored (§4.1) |
| **RF-NOT-LOWERED** | a construct that has a planned lowering whose D-id has not landed yet (the D-CML-1 stub, and every row marked NO in §4 until its D-id lands); carries the D-id | the lowering exists on paper only. This is an honest "not yet", never a false reason |
| **RF-EXEC** | an `ExecError` raised during execution (`LaneNotOrdered`, `SumRowBound`, …) | execution-time checks are part of the IR's contract; the error is carried, never swallowed |
| **RF-UNBOUND-LABEL** | a label with no `LabelBinding` entry | no classid to test; guessing one is the confident-and-wrong quadrant (v1 OQ-1) |
| **RF-UNPLANNED** | a non-empty `SemanticResult.errors`, or an error from the upstream logical planner | the refusal carries the upstream errors; nothing is lowered from a plan that does not exist |
| **RF-UNPARSED** | anything the upstream parser rejects, **including `CREATE` / `SET` / `DELETE` / `MERGE`**: the reused parser accepts only reading clauses followed by `RETURN`, so mutations never reach a `LogicalOperator` | the refusal carries the parse error. Mutations stay out of scope (v1 N-8: writes go through the commit gate) |

**Refusing is the correct answer here, not a missing feature.** A refused row becomes
a lowering only through a new D-id with its own falsifier. It is never added by
silently widening the classifier.

**What the refusal list costs today — a census fraction, not a bound on v2.** The W0-b census
(`examples/w0b_corpus_census.rs`; recorded in
`EPIPHANIES-ARCHIVE-2026-09-20.md` under
`E-THE-SPINE-IS-WHATEVER-THE-READER-ALREADY-HAS-AN-ADDRESS-FOR-1`; 303 classified
queries from a walk of the tree) found 113 **Full** (37.3 %). Everything else was
`Split` or `Grace`. Under v2 those become **refusals**. So 37.3 % is the census's
**Full fraction under an empty config**. It is not what v2 answers on its first day:
D-CML-1 refuses everything, and the corpus labels are unbound. The number is a property of the
corpus's test queries, most of which exercise DataFusion features on purpose. It is
not a property of the queries a consumer actually sends. Two further reasons it
overstates:
- The census ran with an empty `GraphConfig`, so it never exercised a label binding.
- The corpus labels (`Person`, `Company`, …) are not in the 123-entry OGAR codebook
  (modelgraph §16.3), so `LabelDTO::from_canonical` resolves none of them. A real run
  needs an explicitly supplied binding.

**It is not an upper bound either.** 39 of 342 candidate queries were never classified
(`EPIPHANIES-ARCHIVE-2026-09-20.md:2156`), among them the bare-node `RETURN n` queries
that fail to plan without a binding (`w0b_corpus_census.rs:479-491`) and that v2 could
answer once a binding exists. Two snapshots, each with its own denominator:
- W0-b: 113 Full of 303 classified (37.3 %);
- #1305, after classifying path multiplicity: 70 Full of 313 (≈ 22 %; §12 H-5),
  reported there and not re-run here.

D-CML-2 re-runs the census under the v2 classifier and reports the count for each
refusal variant.

---

## §6 — The switch

**One switch, on our surface, choosing a whole engine — never one query at a time.**

The switch is not a third engine enum. There are two enums it must not be confused
with:
- Upstream `query::ExecutionStrategy { DataFusion, LanceNative, BlasGraph }`
  (`query.rs:109-117`). `LanceNative` is the natural slot for a mask engine, but it is
  unusable: filling it would edit `query.rs` (F1) and create a dependency cycle.
- The graph-side `Backend::MailboxSoa` (`graph_router.rs:46-62`, cypher-kanban Inc 0).
  It stays a separate hit-level tag; v2 neither extends nor supersedes it.

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
`Upstream`), and the baked rows plus a `LabelBinding` (for `Mask`). Many non-test files
mention Cypher (74 under `crates/`). Three candidate consumers were inspected, and not
all have both halves:

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

Other candidates, not yet inspected: `lance-graph-cognitive/src/cypher_bridge.rs`,
`holograph/src/query/transpiler.rs`, `lance-graph-callcenter/src/unified_bridge.rs`,
`cognitive-shader-driver/src/serve.rs`. Enumerating every call site that executes
Cypher is an explicit input of D-CML-9.

**OQ-CML-3 — is `lance-graph-python` fork-owned?** If it came from upstream, it keeps
`Route::Upstream`. Our Python surface then needs its own entry point in a fork crate,
not an edit to the inherited one.

**The flip gate.** A consumer flips to `Route::Mask` only when the census of **its
own** queries (not the test corpus) shows no refusal it needs. That census must be
**non-trivial**: queries non-empty, and executed queries, not prefix classifications.
A stateless classifier's census proves nothing. Until then the consumer stays on
`Upstream`. The switch exists from D-CML-1 on. Flipping it is a per-consumer decision
with a number attached.

**Expected consequence, stated plainly:** `ORDER BY … LIMIT` and `count(*)` after a hop
are among the most common things consumers write, and both are refused. So the switch
is not expected to flip for a general-purpose consumer until bag-valued and ordered
answers get their own D-ids. The first realistic flipper is a consumer whose questions
are reachability or support questions.

---

## §7 — Build order (each step names what it depends on and what would stop it)

| D-id | what | depends on | STOP if |
|---|---|---|---|
| **D-CML-0** | Verify the §2 upstream/fork split against upstream history. Create the crate skeleton: a workspace `members` entry (not `exclude`, no own `[workspace]`), three deps with `lance-graph` at `default-features = false`, and an import-fence test. Add CI lines: a test step in `rust-test.yml`, clippy and rustfmt lines in `style.yml`, one release `nm` step over a `[[example]]` binary recording OQ-CML-1's C3 residue (an observation, not a gate), and one `cargo tree -e features -i lance-graph` check that fails if `planner` is active in the new crate's feature set (feature unification can turn it back on). Confirm that `NodeRow` has a sound zero-copy byte view. Update `lance-graph-mask-risc/src/lib.rs:38,115`, which still names v1's in-`query.rs` `mask_lower` seam. Add a `LATEST_STATE` new-member row | — | a §2 file is fork-owned (the split is redrawn, not the design); or `NodeRow` has no sound byte view (STOP → contract-first) |
| **D-CML-1** | `Route` + `run` stub that refuses everything with `RF-NOT-LOWERED`. The switch compiles, and its refusal reason is true | 0 | — |
| **D-CML-2** | **The classifier.** Walk the public `LogicalOperator` and return `Lowerable { variables whose node sets are asked for }` or a §5 `Refusal`. Two answers only. **#1305's `consumer_semantics()` is NOT ported:** its count and binding kinds describe per-path state to be carried through a hop, and v2 refuses those queries instead (§12). Ported from #1305: the three pattern-shape refusals (§12 H-4) and the fixtures (§12 H-1..H-3). Re-run the W0-b census under v2 and report per-variant counts | 0 | — |
| **D-CML-3** | `LabelBinding` — label → `LabelDTO` → `u32` classid; per property a layout declaration `(field offset, width, kind)`; per relationship type its carrier/direction declaration (§4.1), checked against the class's `ClassView`. Built from `LabelDTO::from_canonical` where the codebook covers the label, and otherwise **supplied explicitly** by the consumer, never guessed. It is the first consumer of `LabelDTO`. The corpus labels are not in the codebook (modelgraph §16.3), so the explicit path is the common one | 0 | — |
| **D-CML-3b** | **Contract-first:** a row-pointer reading of an in-row field, as a `ClassView` / `EdgeCodecFlavor` variant with a declared field width (a `u8:u8` rail stays two bytes, never widened). Its own contract PR | 0 | — (its own PR, its own gates) |
| **D-CML-4** | Node + equality/facet predicate + Boolean lowering through `mask_risc::execute` over the borrowed rows: `EqU32Strided`, `NeU32Strided`, `MatchFacetStrided` and `Ternlog`, plus `Count` / `Any` / `Keep`. Ordered compares and value aggregates return RF-NOT-LOWERED until 5a. The class scan is `EqU32Strided` on the `u32` classid. **`mailbox_scan::match_nodes_by_class` (which returns a `Vec`) is not used and not edited** | 2, 3 | — |
| **D-CML-5a** | **Substrate-first, two layers, its own D-id:** strided lanes **with a declared field width** (like `MaskedStridedGroupSum`'s `group_bytes`, `ir.rs:286-291`) for ordered `I32` compares, `MaskedSum/Min/MaxI32`, `Gather`, `ScatterOrU32`, `CountKeyRunsU32` and `EqU32Via`'s key and foreign-key lanes. First in `ndarray::simd` (only `eq_u32_strided_to_mask`, `masked_strided_group_sum` and `ternary_match_strided_to_mask` are strided today, `simd_masking_ops.rs:235, 965, 3240`), then in mask-risc, each with a differential against the contiguous form. Proposed as a scope addition to mask-risc PR5 (`mask-risc-executor-v1.md:30-31`, which scopes `hop.rs` as an edge-lane join-elision probe); that plan's owner decides. Also decides OQ-CML-4 | 0 | — (its own PRs, its own gates) |
| **D-CML-5** | The hop over a declared in-row carrier (§4.1): pull hops (chainable, one program each) first, then a push hop as the last hop only | 4, **3b, 5a** | a query needs push hops chained → a mask-risc ruling first |
| **D-CML-6** | **Deferred:** the relative hop over the witness loci (§4.1) | 5, a `Relative{locus}` carrier kind, the linear mask-shift primitive in `ndarray::simd` | the declared row order is not the stream order → refused |
| **D-CML-7** | Variable length as reachability, lower bound 0 or 1, over pull hops (§4.2). `visited` starts empty | 5 | — |
| **D-CML-8** | **The differential.** Reference = the upstream DataFusion engine, as a **dev-dependency only**, compared on **support** (`DISTINCT` node sets), never on bag counts. Second reference = quack's DuckDB oracle fixtures, where a query is expressible there | 4 (grows with 5–7) | — |
| **D-CML-9** | Enumerate every call site that executes Cypher (§6). Then the first consumer that can take the switch. `cognitive-shader-driver`'s bridge has neither route today, so this step first gives it the baked rows, a `LabelBinding` and a defined legacy route. Then it flips to `Route::Mask`, after its own **non-trivial** query census passes the flip gate. **Dependency caution:** `cognitive-shader-driver` → new crate → `lance-graph` → planner, while planner has a dev-dependency on `cognitive-shader-driver` (planner `Cargo.toml:65`). The new edge must keep `lance-graph` at `default-features = false`, and `with-planner` must stay off on it | 8 | its census needs a refused construct → it stays on its legacy route, and the number is recorded |
| **D-CML-10** | Board follow-ups from the council: a storno note on `lance-graph-as-the-modelgraph-v1.md` §16.2 (the `NodeMapping`-field route is superseded by v2's `LabelBinding`) | — | — |

Four things run in parallel: D-CML-2, D-CML-3, D-CML-3b and D-CML-5a. D-CML-4 is the
first step that executes anything. D-CML-5 waits on 3b and 5a.

---

## §8 — Falsifiers (v1 §7's discipline; the rows that are new or changed)

| id | assertion | its disable |
|---|---|---|
| **F-CML-UP** | `git diff --quiet <merge-base>..HEAD -- <§2 upstream list>` is empty, for every PR of this plan | touch one byte of `parser.rs`; the gate must go red |
| **F-CML-FENCE** | outside `#[cfg(test)]`, the new crate references no path under `lance_graph::query`, `lance_graph::datafusion_planner`, `lance_graph::sql_*`, or `datafusion*` | add one reference per path; the gate must go red for each |
| **F-CML-CARRIER** | a relationship declared on an ordinal carrier over a class whose `edge_codec_flavor` is `Pq32x4` is refused | drop the `ClassView` check; the query must now lower |
| **F-CML-UNDIRECTED** | `a-[*1..2]-a` over one undirected edge is refused (RF-DEPTH) | drop the check; the answer must now wrongly contain `a` |
| **F-CML-SURFACE** | no public item of the new crate (including every `Refusal` payload) names a `lance_graph`, `datafusion*`, `lance*` or `arrow*` type; upstream errors cross only as owned reasons (OQ-CML-1 C5) | add a `GraphError` field to one `Refusal` variant; the gate must go red |
| **F-CML-PLACEHOLDER** | the lowering reads none of the `GraphConfig` placeholder fields (`id_field`, `property_fields`, `source_id_field`, `target_id_field`) | make the lowering read one; the gate must go red |
| **F-CML-UNDECLARED** | a relationship type with no declaration is refused (RF-UNDECLARED-REL), and the same query with a declaration lowers | drop the check; the undeclared query must now lower or panic |
| **F-CML-REFUSE** (can-fire) | every §5 variant is produced by at least one committed query | delete a variant's arm; its query must now either lower (and fail the differential) or panic |
| **F-CML-QUIET** (can-stay-silent) | a lowerable query produces no refusal, on a corpus where the refusals are a minority of queries that are not trivial | force the classifier to refuse `WHERE`; the Full count must drop |
| **F-CML-SUPPORT** | for every lowerable query, the mask equals the DataFusion `DISTINCT` node set of the returned variable, as a set | the v1 wrong-immediate test (`AND2 0xC0` for `AND3 0x80`) must go red on at least one fixture |
| **F-CML-BAG** | a `count(*)` after ONE hop, over a fixture with two sources reaching one target, is **refused**, not answered with a popcount. So is the 2-hop case (§12 H-1: 4 paths vs 3 nodes) | remove RF-BAG; the answer is asserted against the hard-coded H-1 values (2 at one hop, 4 at two hops) and must disagree. The support-only differential cannot see a count, so it is not the disable. The same applies to F-CML-REFUSE's arms for RF-BAG and RF-VALUE |
| **F-CML-NOMIX** | under `Route::Mask`, a refused query never reaches `query::CypherQuery::execute` | a counting shim on the upstream entry must read 0 |

---

## §9 — Relation to #1303 and #1305

- **#1305** (`cypher-mask-multiplicity-contract-v1`) measured the facts v2 depends on:
  support vs bag, walks vs trails, and the five carrier kinds. Its code sits in an
  upstream file, so v2 takes **the findings and the classification** and reimplements
  them in D-CML-2. It takes none of the edit.
- **#1303** is independent. The execution-socket text in its §11/§11R (external
  relation tables, counting in the socket) is **not** carried into v2. §4.1's in-row
  hop replaces it.

Both PRs are closed unmerged (operator, 2026-09-30).

---

## §10 — Non-goals added by v2 (v1's N-1..N-12 still apply)

- **N-13 — no edit to any §2 upstream file**, not even a `pub` or a doc comment.
- **N-14 — no per-query fallback to DataFusion**, not even in a feature-gated debug mode.
- **N-15 — no external edge table.** Relations are read from the row, through a
  declared carrier (§4.1). A relationship type with no declared in-row carrier is
  refused (RF-UNDECLARED-REL), not given a side table. The `GraphConfig` relationship
  fields are placeholders (F-CML-PLACEHOLDER).
- **N-16 — no counting in traversal.** Only terminals count, and only on the final mask.

---

## §11 — Board hygiene owed when this plan lands

- The plan file `.claude/plans/cypher-mask-lowering-v2.md`; v1's status line reads
  SUPERSEDED-IN-PART.
- STATUS_BOARD rows D-CML-0..10, D-CML-3b and D-CML-5a (Queued; D-CML-6 Deferred).
- `AGENT_LOG.md` prepend for the council run, written by the orchestrator only.
- At D-CML-0: the `LATEST_STATE` new-member row. After merge: the `PR_ARC_INVENTORY`
  prepend.
- Merge order: #1306 and #1307 both prepend `INTEGRATION_PLANS` and both regenerate
  `SUPERSESSION-INDEX` and `entries/README.md`. Whichever merges second merges `main`,
  renumbers its heading, and **regenerates** the two generated files. It never
  hand-merges them.
- INTEGRATION_PLANS prepend.
- The board entry named in the header.
- `python3 .claude/tools/entries_index.py --write`, then `SUPERSESSION-INDEX`
  regenerated **last**, after every other board write.

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
| **H-1** | **A mask is support, not bag.** Cypher `count(*)` after a hop counts PATHS; a popcount counts NODES. Fixture: KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5}; 2-hop `count(*)` = **4** (paths), mask = **3** (end nodes). Same 4 vs 3 for the var-length form | §5 RF-BAG; F-CML-BAG |
| **H-2** | **A forward chain gives the exact node set of the LAST variable only.** An earlier variable's set is not the forward frontier (`count(DISTINCT b)`: 3 vs 4); it needs a backward hop over the reverse lane, or RF-TRANSPOSE | D-CML-4/5 lowering; the classifier's `Lowerable` must name WHICH variables are asked for |
| **H-3** | **DataFusion counts walks, and walks ≠ trails.** On {1→2, 2→1, 2→2} DataFusion returns **5** walks; trails are **4**. They diverge without a cycle too: `(a)->(b)<-(c)` on {1→2} = 1 walk, 0 trails. "Path count" is therefore not one notion | D-CML-8 compares on `DISTINCT` node sets ONLY, never counts; §4.2 refuses path questions |
| **H-4** | **Three pattern shapes are not a chain** and must be refused, not lowered as one (each was a real bug caught in review on #1305): hops that do not connect end to end; a variable bound twice (at a hop or at the pattern start); two disconnected patterns | D-CML-2 refusals |
| **H-5** | **The census drops once multiplicity is classified:** Full 117 → **70** of **313** (≈ 22 %), against the 37.3 % (113/303) quoted in §5 | §5 cost; §6 flip gate |
| **H-6** | **v1's terminal rows are pre-hop only.** T-1..T-7, T-11 and R-6..R-8 hold before the first hop; after a hop they hold only as `DISTINCT` node sets | §0 header caveat; D-CML-4..7 |


---

## §13 — 5+3 council change ledger (v1-draft → draft v2)

The 5 were prior art, iron rules, code truth, cascade impact and different views; they
filed 38 findings. The raw output is banked and is not reproduced here.

| # | change | source |
|---|---|---|
| L1 | In-row hops need strided index lanes for `Gather` / `EqU32Via` / `ScatterOrU32`. These do not exist, so they become a STOP → substrate-first **D-CML-5a** (owned by mask-risc PR5 `hop.rs`). The §4 hop row now reads NO | code truth (high) · prior art (mask-risc-executor PR5) |
| L2 | `run` takes the baked `&[NodeRow]`, not `MailboxSoaView`. The view has no row bytes and only a `u16` class id. The classid scan becomes `EqU32Strided` (yes today). D-CML-0 checks for a zero-copy byte view | code truth · iron rules (F8) |
| L3 | `CausalWitness` loci recharacterised as relative temporal-window pointers (v3 re-grades the draft's withdrawal to **Deferred**, R3) | code truth |
| L4 | API precision: `ast::CypherQuery` vs `query::CypherQuery`; `analyze(&ast, &params)`; planning uses the substituted AST | code truth |
| L5 | The census proves parse→plan only (no semantic step, empty `GraphConfig`). Its figures are not comparable (v3: not a bound either, R7) | code truth · prior art (§16.3 labels not in codebook) |
| L6 | `lance-graph` is a dependency with `default-features = false`. The D-CML-9 dev-dependency cycle risk is named | code truth · cascade |
| L7 | The switch is explicitly distinguished from `ExecutionStrategy::LanceNative` and `Backend::MailboxSoa`. `Route` stays (the one switch); it is not a third engine enum | prior art |
| L8 | `Refusal` follows quack's `LowerError` precedent (`#[non_exhaustive]`). RF-CROSS-SPACE is linked to v1 F-H4 | prior art |
| L9 | modelgraph §16.2 storno becomes D-CML-10 | prior art |
| L10 | Refusal ids renamed `R-*` → `RF-*` to avoid v1's R-1..R-9 hop rows | prior art |
| L11 | Relationship carrier + direction are **declared** (in `LabelBinding` now, `ClassView` later), never decided by the lowering. New RF-UNDECLARED-REL and F-CML-UNDECLARED | different views · prior art (cypher-kanban Inc 0 classid-resolved rule) |
| L12 | The returned variable is independent of the pointer side. RF-CHAIN only when a scattered mask would be consumed | iron rules (false refusal) |
| L13 | RF-ORDER narrowed: ORDER BY / `LIMIT ≥ 1` over a one-row scalar result is a no-op; `SKIP ≥ 1` / `LIMIT 0` stay refused | iron rules (over-refusal) |
| L14 | `Item::Mask(MaskHandle)` is an opaque owner-held handle, never a word or row-id `Vec` | iron rules |
| L15 | `GraphConfig` relationship fields are placeholders. New F-CML-PLACEHOLDER | iron rules (F3 pressure) |
| L16 | Multi-item `RETURN`: each item is its own program re-evaluating the prefix | iron rules |
| L17 | D-CML-0 file list completed: members, CI lines, `nm` step, `LATEST_STATE` row, mask-risc seam doc | cascade |
| L18 | Merge-order note for #1306/#1307 generated files | cascade |
| L19 | The one front-end adapter is named, as the exit path to loco/r2il; `LogicalOperator`'s walk-shaped bias is named | different views ×2 |
| L20 | Flip gate: the census must be non-trivial, and the no-flip consequence is stated plainly | different views ×2 |
| L21 | §9 corrected: both PRs are closed | orchestrator |

### v2 → v3 (the three reviewers: overclaim-auditor, dilution-collapse-sentinel, firewall-warden)

Verdicts: 1 BLOCK (§4, overclaim), 27 FIX, the rest PASS. All resolved here.

| # | change | source |
|---|---|---|
| R1 | **The BLOCK:** ordered compares and `MaskedSum/Min/MaxI32` need contiguous `I32` lanes, so over row bytes they read NO. D-CML-4 is narrowed to equality/facet predicates; D-CML-3 gains a property layout declaration | overclaim |
| R2 | D-CML-5a rescoped: strided lanes with a declared field width, for compares, value terminals, `Gather`, `ScatterOrU32`, `CountKeyRunsU32` and `EqU32Via`; `ndarray::simd` first, then mask-risc; its own D-id, proposed to PR5 | overclaim · dilution |
| R3 | The witness loci are real relative pointers. D-CML-6 is re-graded Withdrawn → **Deferred**, with its two preconditions. The one substrate gap is a linear mask shift | dilution (collapse) · overclaim |
| R4 | No contract reading of an in-row carrier is a row pointer. New **D-CML-3b** (contract-first pointer reading). Declarations are checked against `ClassView::edge_codec_flavor`; F-CML-CARRIER | dilution · firewall |
| R5 | Pull hops: `Gather` only (`EqU32Via` is not a hop form); the source is a `Foreign` plane; every pull hop ends its program; OQ-CML-4 | overclaim · dilution |
| R6 | Undirected variable length refused (walk ≠ trail); F-CML-UNDIRECTED | overclaim |
| R7 | Census regraded: a Full fraction under an empty config, neither a day-one figure nor an upper bound; two snapshots with their own denominators | overclaim |
| R8 | New refusals RF-EXEC (step 6) and RF-NOT-LOWERED (the stub and every NO row); the "before any program runs" absolute scoped to the classifier | overclaim |
| R9 | `run` takes `params`; DataFusion path re-cited `query.rs:934,945`; step 2's reason corrected (the upstream check is partial) | dilution · overclaim |
| R10 | Join row: NO; RF-CROSS-SPACE for a second population | firewall · dilution |
| R11 | OQ-CML-1(b) is a STOP, not a pre-authorised copy; the `nm` step measures a `[[example]]`, with a disable; a `cargo tree` features check | firewall |
| R12 | Falsifiers: F-CML-UP non-vacuous; F-CML-FENCE by module path; F-CML-BAG asserts H-1 values | firewall · overclaim |
| R13 | §6: 74 files mention Cypher; three inspected; four more named; enumeration is a D-CML-9 input | overclaim |
| R14 | §11 board list completed | firewall |

**Findings recorded but not adopted:**
- "Make `Route` a consumer-side `const` instead of a crate type" (iron rules, AP6). Not
  adopted: the operator asked for one switch, and a crate-owned enum keeps its two
  values identical across consumers.
- Moving the relationship declaration into `ClassView` now (different views). Not
  adopted in this plan: it is a contract change and gets its own D-id (OQ-CML-2).
