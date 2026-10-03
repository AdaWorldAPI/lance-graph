# frontend-parity-witness-v1 — do SQL, Cypher, Gremlin and SurrealQL lower onto one population algebra?

> **Status:** MEASURED for SQL ↔ Gremlin (code: `crates/lance-graph-quack/tests/gremlin_parity.rs`, 13 tests).
> ANALYSIS for Cypher (against the #1306 plan, which has no code yet) and SurrealQL (AST read, not wired).
> D-FPW-0..4. Ratifies nothing in mask-risc or Quack; adds no production code.
> **Board:** `STATUS_BOARD.md` § D-FPW · entry `entries/2026-10-03-frontend-parity-witness.md`.

## §0 — The question, and the short answer

Do independently designed query languages reduce to the same execution primitives?
For the shapes the substrate already supports, **yes, and the meeting point is
measurable**: a Gremlin traversal and its SQL equivalent produce the *same*
`lance_graph_quack::Query` value (`assert_eq!` on the struct), then the same
`Program`, then DuckDB's committed answer. The schema declaration and adapter are
about 450 lines of test code (the file is about 1,500 with the oracle and tests)
and add no operator.

Every traversal that does not lower is refused with a named reason. Two of the
refusals are real substrate gaps, and both were already named independently by
#1306 (Cypher) and #1308/#1310 (chained hops). No new primitive is proposed here.

> **⊘ Corrections (2026-10-03, `population-law-crosscheck-v1.md`).** Three conclusions
> below are revised, not deleted:
> - **§2 / §6 G2 / §8:** "a second fan-out cannot keep the invariant" is wrong as a rule.
>   A fan-out onto a population that holds an fk to the current anchor re-anchors again
>   (at the cost of read depth, G1). The barrier is carrying a fan-in aggregate to the
>   next population, not the hop count.
> - **§6 G4:** the witness used `Country` (`u32`), so even the local sum cannot lower.
>   It does not isolate a foreign-value sum. By source no such operator exists, but it
>   is not measured here.
> - **§3:** the adapter and the oracle both leave the cursor on the vertex after
>   `values(f)`, so `values(f)` followed by a non-reducing step returns vertices in both.
>   No test exercises that shape; adapter/oracle agreement there would prove nothing.

## §1 — What exists (A)

| layer | in code? | where |
|---|---|---|
| **mask-risc** (the one evaluator) | yes | `crates/lance-graph-mask-risc/src/ir.rs`: `Pred` (incl. `EqU32Via`, `Range`, strided), `MaskOp` (`And/Or/Xor/AndNot/Not/Ternlog/Gather`), one `Terminal` per `Program` (`Count/Any/All/MaskedSum|Min|Max/Keep/BlendI32/ScatterOrU32/ScatterCountU32(held)/CountKeyRunsU32/GroupSumI32/GroupSumViaI32/GroupReduce{Lane,Via,Pair}`) |
| **Quack** (the SQL-shaped lowering) | yes | `crates/lance-graph-quack/src/lib.rs`: `Query{Filter, Agg}`, `lower`, `lower_fused`, `lower_group_by[_auto]`, `lower_group_having`, `lower_avg`; DuckDB differential (`tests/duckdb_differential.rs`, 38 cases) |
| **DuckPG / DuckGQL** | **no** | 0 hits for `duckgql`, `duck_gql`, `duckpg`, `duck_pg` across `crates/`, `.claude/`, `docs/`. The SQL frontend in the tree is Quack, with DuckDB as oracle only |
| **Cypher #1306** | plan only | `.claude/plans/cypher-mask-lowering-v2.md` (ratified v3). Lowering table §4, refusal list §5 |
| **R2IL / FoldDialect** | test-only probe | `crates/r2il-mask-abi-probe/tests/row_bridge.rs`: loco programs over OGAR's `0xE2..` fold band that run one mask-risc `Program` per fold boundary — the existing multi-terminal witness (`SUM − SUM`) |
| **"Northstar"** | prose | an epiphany name (`E-OGAR-NORTHSTAR-1`), not a layer |
| **"V4"** | not found as a name | in `mask-risc/src` and `quack/src` (closed space) |
| **dir-sim** | yes, excluded crate | another Quack consumer: anti-join = `negate(Semijoin)` (`entries/2026-10-03-dir-sim-soa-quack.md`) |

## §2 — The anchor invariant (why one hop of anything is one population)

Gremlin counts traversers with bulk; SurrealQL flattens per-hop results into a bag
(`core/src/val/value/get.rs:499`); Cypher binds paths; SQL joins produce rows. All
four are **bag** semantics by default. The witness keeps one table as the ANCHOR:
every surviving anchor row is exactly one traverser of bulk 1.

- A **functional** hop (each anchor row names ≤ 1 target) leaves the anchor where
  it is; the target's fields are read through the fk (`EqU32Via`, `GroupKey::Via`).
  Consumption class **functional-indexed**: resident reads composed.
- A **fan-out** (reverse fk, or an edge table) moves the anchor to the table whose
  rows ARE the paths (the child, or the edge table), carrying the old predicates
  across the fk. One fan-out is therefore still one population, still one `Program`.
- A second fan-out, or a read two fks deep, cannot keep the invariant. Those are
  the two gaps (§6).

Consequence for Cypher: #1306's RF-BAG ("any non-DISTINCT count after one or more
hops") is **broader than necessary**. After a functional hop, or a single hop over
an edge population, a bag count is a count of anchor rows — exact. This is an input
for D-CML-2, not an edit to that plan.

## §3 — Gremlin step → primitive (C)

| step | lowers to | class |
|---|---|---|
| `V(T)` / `hasLabel(T)` | anchor = T, `Filter::Plane(alpha)` | pointwise |
| `has(f, eq)` on the anchor | `Filter::Cmp` (`EqU32`/`EqI32`) | pointwise |
| `has(f, gt)` on an `i32` anchor field | `Cmp::GtI32` | pointwise |
| `out(rel)`, functional | cursor moves, anchor stays; `Semijoin(fk, target alpha)` | functional-indexed |
| `has(f, eq)` after a functional `out` | `Filter::EqU32Via` | functional-indexed |
| `in(rel)`, reverse of fk | re-anchor on the child; carried predicates become `EqU32Via`/`Semijoin` | fan-out → one population |
| `out/in(rel)` over an edge table | re-anchor on the edge table; both endpoints are fk reads | fan-out → one population |
| `where(out(..).has(..))` | predicate atoms on the anchor | functional-indexed |
| `where(in(..).has(..))` | child anchor + set over the fk | fan-out, set |
| `count()` | `Agg::Count` (bag = anchor rows) | demanded output |
| `dedup().count()` | `Agg::CountDistinctOrderedU32{fk}` (physically refused on an unordered lane, never a seen-set) | demanded output |
| `dedup()` then return | `Agg::ScatterOrU32` (the set mask IS the demanded result) | demanded output |
| `values(f).sum()` on the anchor | `Agg::SumI32` | demanded output |
| `groupCount().by(f)` | `Agg::GroupReduce{Local | Via, Count}` | demanded output |

Measured parity (all against the independent row-at-a-time bulk oracle in the test):

- `where(out('billedTo').has('country',3)).values('amount').sum()` = **1237848** = DuckDB `join_sum_country`; the lowered `Query` is `==` the SQL-shaped one.
- `V(partner).has(country,3).in('billedTo').has(status,1)...sum()` (the other direction) lands on the **same anchor and the same atom multiset**, same 1237848.
- `out('billedTo').groupCount().by('country')` = DuckDB `join_group_count_country`.
- `V(doc).where(in('partOf').has(status,1)).count()`: oracle 511 = DuckDB; lowered terminal and physical refusal (`LaneNotOrdered`) are identical to Quack's SQL path.
- One M:N hop (doc → tradesWith → partner, edge table = `line`) with predicates on both endpoints: one `Program`, matches the oracle; `groupCount` by target field matches.
- Bag vs set after a fan-out stay apart: bag `Count` ≠ set `CountDistinctOrdered`; `dedup()`+return scatters a mask whose popcount = oracle's set; returning a bag without `dedup` is refused.

Adapter tax (debug build, one run, not asserted): adapter ≈ 1.9 µs, Quack lowering ≈ 1.0 µs, execution over 4,096 rows ≈ 167 µs (the timed loop also builds lanes and scratch). Adding a frontend adds lowering rules, not a data path.

## §4 — Parity matrix (B)

Classification: **E** existing primitive · **C** composition of existing primitives · **G** missing general primitive · **F** frontend-specific semantics · **O** out of scope. Consumption class in brackets: P pointwise · FI functional-indexed · NFI non-functional-indexed · SC scalar control · D demanded output.

| construct | SQL / Quack | Cypher (#1306 plan) | Gremlin (witness) | SurrealQL (AST read) |
|---|---|---|---|---|
| population selection | E `Plane` [P] | E classid `EqU32Strided` [P] | E `V(T)` [P] | E `FROM table` [P] |
| predicate, Boolean composition | E `Cmp`, `And/Or/Not`, `Ternlog` [P] | E (+ `fuse`) | E | E `Cond(Expr)` |
| projection | E `Agg::Rows` [D] | E `Keep` | E return at anchor | E `Fields` |
| property lookup through fk | E `EqU32Via` [FI] | NO over row bytes (D-CML-5a) | E | E idiom `.field` after a hop |
| functional relation | E `Via` reads [FI] | needs D-CML-3b pointer | E | E `->rel->` when 1:1 |
| many-to-many, one hop | C re-anchor on edge table [P over edges] | RF-CROSS-SPACE (one population) | C | C edge tables are records with `in`/`out` |
| forward / reverse traversal | C | pull/push + RF-TRANSPOSE | C both directions over fk or edge table | `Dir::{In,Out,Both}` |
| grouping, count, sum, min, max | E `GroupReduce`, `GroupSum*` [D] | count yes; sum NO after hop (RF-BAG) | E (count; sum on anchor) | E `GROUP BY`, `count()`, `math::sum` (named functions) |
| distinct | E ordered `CountKeyRuns`; held seen-set [D] | E `count(DISTINCT n)` | E `dedup` | F none per hop (bag) |
| semi-join | E `Semijoin` / `EqU32Via` [FI] | E `Gather` (pull) | E `where(out..)` | C subquery in `Cond` |
| anti-join | C `negate(Semijoin)` (dir-sim) | C `NOT` | C `not(where..)` (not in witness) | C `NOT` subquery |
| multi-hop, functional | **G** composed via [FI] | STOP (needs pointer + 5a) | **G** refused | **G** |
| multi-hop, fan-out after hop | **G** barrier + workspace [NFI] | RF-CHAIN | **G** refused | **G** |
| ordered compare through fk | **G** `EqU32Via` only [FI] | NO (5a) | **G** refused | **G** |
| sum of a foreign value | **G** no `Σ v[fk[i]]` [FI] | RF-BAG after hop | **G** refused | **G** |
| ordered compare on `u32` | **G** IR orders `i32` only | NO over bytes | refused | — |
| bounded variable length | O here | §4.2 frontier fixpoint (pull only) | O `repeat().times()` | O `{1..3}` `Recurse` |
| path result, path count | F | RF-BAG | F `path()` | F `RecurseInstruction::Path` |
| shortest path | O | O | O | F `Shortest` |
| ordering, limit, range | F | RF-ORDER | F | F `ORDER/LIMIT/START` |
| mutation | O (commit gate) | RF-UNPARSED | F/O `addE`, `property` | O `CREATE/RELATE/UPDATE/DELETE` |
| side effects | O | O | F/O `aggregate`, `sideEffect` | O |

## §5 — SurrealQL AST harvest (D)

Read in `/home/user/surrealdb` (fork HEAD `8f5adb2`); two claims re-read by hand
(`sql/lookup.rs:12`, `sql/dir.rs`, `val/value/get.rs:495-502`).

**Worth mirroring (structure):**
- A graph arrow is one idiom `Part::Graph(Lookup)`; `Lookup { kind: Graph(Dir) | Reference, what: Vec<LookupSubject>, cond, expr, group, order, limit, start, split, alias }`. **Each hop is a mini-SELECT** — the same shape as "re-anchor on the edge population, then filter, then fold".
- Edges are first-class record tables with `in`/`out` fields (`doc/edges.rs:57-74`): exactly the edge-as-population geometry the witness uses (S = `in`, O = `out`).
- Aggregates are named function calls resolved at plan time (`catalog/aggregation.rs:538-572`), not AST nodes — the adapter maps names to `GroupFold`, as Quack maps `Agg`.
- Recursion is a typed idiom part (`Recurse::{Fixed, Range}`, `RecurseInstruction::{Path, Collect, Shortest}`), so path/set/shortest are distinguished in the AST itself — the same split this repo needs (§4.2 of #1306).

**Not reusable as a dependency:**
- Every production AST type is `pub(crate)` (`core/src/sql/mod.rs`); `parse()` is public but its result is opaque outside the crate.
- The public arena AST (`surrealdb-ast` / `surrealdb-parser`) is `publish = false`, self-labelled unstable, and not what core runs.
- License: Business Source License 1.1 (change date 2030-01-01). Depending on it is a licensing decision for the operator, not this plan.

**So the thin adapter is real but has a seam:** a fork-side lowering hook inside
`surrealdb-core` (which already carries AdaWorldAPI features `kv-lance`,
`op-bridge`, `lance-graph`) that walks `sql::Lookup` and emits a Quack `Query` would
need no universal AST. That hook is a separate decision (D-FPW-3).

## §6 — Converged algebra, and the gaps that survive (E, F)

**The smallest set every frontend above needs** (all existing): resident plane;
pointwise predicate; Boolean composition; fk-read predicate (`EqU32Via`); semijoin
over a resident plane (`Gather`); one-terminal folds (`Count`, `Masked*`,
`GroupReduce{Lane,Via,Pair}`); ordered distinct count; scatter as a demanded result.
Plus one lowering rule, not a primitive: **re-anchor a fan-out on the population
whose rows are the paths.**

**Gaps that pass the two-independent-witness rule** (each is demanded by SQL and
Gremlin in this file, and independently by Cypher #1306 or #1308):

| gap | class | witnesses | already named as |
|---|---|---|---|
| G1 composed functional read `v[fk2[fk1[i]]]` | FI | SQL 3-table fk chain; Gremlin `out().out()`; Cypher fixed 2-hop | #1308 option (a), tile-local gather chain |
| G2 barrier + computation-private workspace (fan-out after a hop) | NFI | SQL 3-way join; Gremlin `out().in()`; Cypher RF-CHAIN; GQL | #1308 (b)/(d), #1310 "b-lite" |
| G3 ordered compare through an fk | FI | SQL `JOIN … WHERE p.x > k`; Gremlin `out().has(gt)`; Cypher §4 NO rows | D-CML-5a (ordered ops over bytes; partly) |
| G4 sum of a foreign value `Σ v[fk[i]]` | FI | SQL `SUM(p.x)` over a join; Gremlin `out().values().sum()`; Cypher `sum(b.p)` | not named before |
| G5 ordered compare on `u32` | P | SQL `doc_id` range (`fixture.rs`); Gremlin `has(gt)` on `u32` | fixture.rs note |

None is built here. G1 and G2 are the two that decide traversal; G3–G5 are small.

## §7 — Frontend-specific semantics that must not leak down (G)

- Gremlin `groupCount` omits zero keys; SQL `LEFT JOIN` keeps them. The sink has
  K slots either way; presentation is the frontend's.
- Bag vs set: every frontend defaults to bags. A mask is a set. `dedup` /
  `DISTINCT` / SurrealQL's lack of per-hop dedup are frontend semantics; the
  substrate carries bags only as anchor rows, and refuses returning a bag of far
  elements (`BagOfElementsNotAMask`).
- Paths, ordering, positions, mutation, side effects: refused by name, never
  approximated.

## §8 — Connected-computation implications (H)

- **Pointwise** and **functional-indexed** cover everything that lowers here, in
  ONE `Program`, including one M:N hop (edge-population re-anchor).
- **Non-functional-indexed** begins exactly at the second fan-out (G2). All four
  frontends hit it at the same place.
- **Multi-terminal** is independently demanded (Gremlin `group().by(..).by(sum, count)`,
  Cypher multiple RETURN aggregates, SQL `AVG = SUM/COUNT` which Quack already
  plans as two programs, FoldDialect `SUM − SUM`). It is **not** a traversal
  requirement: none of the witness traversals needed it.
- **Multi-phase** is G2. It is the only place a population-sized workspace is
  mathematically required.

## §9 — ABI implications (I)

- The adapter needed exactly four facts per relation: carrier kind (fk | edge
  table), the fk column(s), source and target table. That is #1306's
  `(carrier, field, direction, reverse lane)` declaration, plus "edge table" as a
  carrier kind #1306 does not have (it has one population).
- Coordinate identity: every foreign plane/lane carries its own row count
  (`ForeignPlane::rows`), and the anchor's `Planes::n_rows` is the only
  `n_rows`. The witness never compares equal-length populations as equal.
- ClassId: tables here are fixture tables, not classids. Binding `Table` →
  classid is #1306's `LabelBinding`; nothing in the witness depends on it.
- Version: not exercised (single version). A workspace for G2 must carry
  `(coordinate space, version)` — open.

## §10 — The PR (J)

- `crates/lance-graph-quack/tests/gremlin_parity.rs` — test-only: typed step
  vocabulary, schema declaration, adapter, refusals, bulk oracle, 13 tests.
- This plan, one board entry, `STATUS_BOARD` / `INTEGRATION_PLANS` rows.
- No change to `mask-risc`, Quack `src/`, or any upstream file.

Disable checks run (each red, then restored): dedup does not mark the set; a
re-anchor drops carried predicates; a bag emitted as a mask; an edge hop drops the
target-exists semijoin; carry keeps a stale via atom. The edge-hop disable is
caught only by the structural `Query` comparison: the fixture has no dangling fk,
so execution cannot see it.

## §11 — Falsifiers (K)

The "shared algebra" hypothesis is too broad if any of these is found:

1. A traversal the oracle answers, that lowers, and whose execution disagrees with
   the oracle or DuckDB.
2. A Gremlin traversal and its SQL equivalent that both lower but to `Query`
   values with different atom multisets.
3. A frontend construct that needs a new primitive and has no second independent
   frontend demanding the same semantics (it would belong in that frontend).
4. A case where re-anchoring on the edge population gives a different bag count
   than the path count (would break §2).
5. Cypher or SurrealQL wiring that cannot reach `Query` without a universal AST in
   between.

## D-ids

| D-id | scope | status |
|---|---|---|
| D-FPW-0 | inventory + parity matrix (this file) | Shipped (this PR) |
| D-FPW-1 | Gremlin witness `tests/gremlin_parity.rs` | Shipped (this PR) |
| D-FPW-2 | feed §2 (functional hop ⇒ exact bag count) into #1306 D-CML-2's classifier | Queued |
| D-FPW-3 | SurrealQL fork-side lowering hook (`sql::Lookup` → `Query`); licensing decision first | Queued, operator |
| D-FPW-4 | G4 (foreign-value sum) — needs a second look against `GroupReduce` before being called a primitive | Queued |
