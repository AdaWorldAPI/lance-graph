# PostgreSQL API → ClassView polyfill map (draft, 2026-10-21)

**Status:** WORKING-MODEL / DRAFT. Read-only survey of the tree at #1668 (and the D-QSP plan in
#1667). Nothing here is built, ratified or a parity claim. PostgreSQL is the semantic
oracle; DuckDB the performance comparator (D-QSP plan :7). No SQL engine, ORM, planner or interpreter
is proposed: each row says which **existing** polyfill surface expresses a PG construct, under which
declared law, or that it is ABSENT.

Stack: `ClassView (method + law) → Quack Binding (coordinates + witnesses) → optional DR (physical
mapping) → relalg/R2IL → mask-RISC`.

## Sources read first-hand (file:line), and what was NOT read

READ (full regions):
- `lance-graph-contract/src/set_algebra.rs:1-260` (module laws, `Cmp:81`, `Res:93`, `SetExpr:125`,
  `Logic:185`, `normalize:195`, `normalize_kleene:201`, `normalize_with:206`); test names
  `normalize_kleene_keeps_the_true_rows_under_null:906`, `normalize_kleene_never_grows_leaf_count:1015`.
- `lance-graph-contract/src/set_rules.rs:1-340` (`Arith:34`, `Truth:49`, `SumFamily:58`,
  `Population:67`, `Semantics:74`, `Measure:107`, `Rule:184`, `Refusal:205`, `CATALOGUE:236`); test
  names `experiment_e_*:1123-1513`.
- `lance-graph-contract/src/set_program.rs:1-120, 235-262` (`SetProgram:49`, `share:245`).
- `lance-graph-contract/src/class_view.rs:700-1060, 1380-1428` (`ComputeEdge:730`,
  `execute_compute_dag:962`, `execute_defaults:1018`, `trait ClassView:1049`, `compute_dag:1403`,
  `default_targets:1420`). Locations only for `FieldMask:70`, `WideFieldMask:243`.
- `lance-graph-quack/src/lib.rs:1-168, 210-300, 330-760, 1064-1095, 1124-1885` (`Cmp:218`,
  `Filter:340`, `in_u32:446`, `in_i32:455`, `is_not_null:468`, `is_null:475`, `sql_where:549`,
  `LowerError:1066`, `Agg:1127`, `GroupAddr:1235`, `GroupAgg:1263`, `Query:1301`, `GroupBy:1314`,
  `AvgPlan:1510`, `lower_avg:1522`, `lower_group_avg:1573`, `avg_finish:1611`, `HavingCmp:1618`,
  `GroupHaving:1654`, `lower_group_having:1688`, `GroupHavingPlan::finish:1787`,
  `normalize_group_sink:1858`).
- `lance-graph-quack/src/bind.rs:1-420` (`BoundField:77`, `validity:87`, `Binder:92`, `Op:137`,
  `BindError::UnknownValue:224`, `Draft::bind:288`).
- `lance-graph-mask-risc/src/ir.rs:286-560` (`Terminal:288`, `GroupKey:490`, `GroupFold:510`,
  `GROUP_SUM_SYM_MAX_ROWS:697`, `MASKED_SUM_I32_MAX_ROWS:705`); `value.rs` `ExecError::SumRowBound:151`,
  `LaneNotOrdered:181` (located).
- `lance-graph-relalg/src/{lib.rs, ir.rs, join_body.rs}` full; `lower.rs:1-80`; `program.rs:1-60`.
- `lance-graph-r2il/src/lib.rs:1-60` (module doc incl. implemented/refused calls) + `FoldError` variants.
- `dir-rmap-probe/src/pop.rs:1-60`, `qr.rs:1-30`, `prep.rs:95-112` (`Refusal`).
- `lance-graph-contract/examples/k3_pg_parity.rs` (full), board entry
  `2026-10-21-d-rft-0-kleene-normalization.md` (full), `2026-10-20-six-decisions.md:1-60` (ADMISSION B),
  D-QSP plan (full).

CLAIMED (not read; named from grep hits or other docs only): test bodies of
`lance-graph-quack/tests/sql_null_3vl.rs` (names only, plus lines 669-784 headers),
`tests/duckdb_differential.rs` (fn names only), `mask-risc/tests/distinct.rs` (fn names only),
`relalg/src/normalize.rs` bodies, `relalg/src/lower.rs` past line 80, all of `exec.rs`, `fuse.rs`,
`graph_hop.rs`, `gremlin_parity.rs`, `spofc_cycle.rs`, the remaining 2,000+ lines of `class_view.rs`
(FieldMask/WideFieldMask methods, rail/facet readings), and the Linear/EXO tickets.

## Legend

- **Carrier**: `B` = Boolean row mask (`u64` words); `(T,F)` = Quack's polarity pair (T/F masks, U =
  neither) — note: the code uses **(T, F)**, not (T, U), `lib.rs:533-548`; `ℕ` = count; `Σ0` = coalescing
  sum (empty → 0); `Σsym` = sentinel sum (empty → `SYM_EMPTY_I64` marker); `seed` = MIN/MAX seed
  `i64::MAX/MIN` as empty-marker; `K-mask` = K-bit group presence mask.
- **Semantics vector** = `set_rules::Semantics { universe, arith, truth, sums, population }` (`:74`).
- **Refusal type**: `Sem` = semantic refusal (the rewrite/answer would change meaning:
  `set_rules::Refusal` `:205`); `Route` = route/binding/physical refusal (shape not admitted: Quack
  `LowerError :1066`, `BindError`, mask-risc `ExecError::{LaneNotOrdered,SumRowBound}`, r2il
  `FoldError`, DR `prep::Refusal{Stale,OutOfFocus,TooLarge,Duplicate}`). **Neither name exists as a type**
  (`RouteRefusal`/`SemanticRefusal` grep: 0 hits in crates/) — see gap G6.
- Status: **VERIFIED-IN-CODE** (surface exists and its doc/test states this exact semantics),
  **PARTIAL** (gap named), **ABSENT**.

## Stratum 1 — K3 predicates (D-QSP-0)

| PG construct | PG semantics (exact, NULL) | Polyfill surface | Declared law / carrier | Status | Oracle / test | Refusal |
|---|---|---|---|---|---|---|
| `x = c`, `<>`, `<`, `<=`, `>`, `>=` (int4 col vs const) | UNKNOWN if `x` NULL | Quack `Filter::Cmp(col, Cmp::{Eq,Ne,Lt,Le,Gt,Ge}I32)` `lib.rs:218`; NULL via `sql_where` `:549` (leaf `T=valid∧leaf`, `F=valid∧¬leaf`); contract `Res::I32` `set_algebra.rs:93` | `Truth::ThreeValued`; carrier `(T,F)` → WHERE takes `T` | VERIFIED | `sql_null_3vl::where_is_sql_three_valued_on_the_required_cases:274`, `random_trees_agree_with_the_oracle_both_lowerings:429`; `k3_pg_parity` (PG 16.14, 2000 exprs) | — |
| `x = c`, `<>` on coded/text col (u32 code) | same | `Cmp::EqU32/NeU32` `lib.rs:233`; `Res::U32` `set_algebra.rs:97`; text literal → code via `Binder::code` `bind.rs:92` | ThreeValued; `(T,F)`; codebook must be frozen per version (ADMISSION B) | VERIFIED (=,<>) | same; `k3_pg_parity` covers `u0` | `BindError::UnknownValue` (Route) — see trap T9 |
| `<`,`>` on text / u32 order | collation order | none: Quack u32 has only `=,<>`, ternary match; r2il `IntLess` → `NoUnsignedLaneCompare` (`r2il lib.rs` doc) | — | ABSENT | propose PG diff with `COLLATE "C"` | Route (`FoldError::NoUnsignedLaneCompare`) |
| `x op y` (column vs column) | UNKNOWN if either NULL | none: `Cmp` takes a scalar only (`lib.rs:212-216`); r2il `IntSub` lane-lane → `LaneVersusLane` | — | ABSENT (G4) | propose `WHERE a0 < a1` diff | Route |
| `AND`, `OR`, `NOT` | Kleene | `Filter::And/Or/Not` + `sql_where` (NOT flips polarity, AND/OR De Morgan dual on F) `lib.rs:533-548`; contract `SetExpr::And/Or/Not` + `normalize_kleene` `:201` | ThreeValued; `(T,F)` | VERIFIED | `sql_null_3vl:429`; `set_algebra:906` (3000 random); `k3_pg_parity` | — |
| `IS NULL` / `IS NOT NULL` | never UNKNOWN | `Filter::is_null(valid)` = `Not(Plane)` `:475`; `is_not_null` = `Plane(valid)` `:468`. Contract `SetExpr` has NO `IsNull` atom; `x >= i32::MIN` stands in for IS NOT NULL inside `normalize_kleene` only | two-valued plane leaf `B` | VERIFIED (Quack) / PARTIAL (contract: IS NULL inexpressible in `SetExpr`) | `sql_null_3vl::null_is_not_zero:373`, `null_is_not_the_empty_string_or_false:562` | — |
| `x IS [NOT] DISTINCT FROM c` (c non-NULL) | two-valued; NULL vs c → TRUE (distinct) | composable: `Or(is_null(v), And(Plane(v), Cmp(Ne c)))` with no `sql_where` on that subtree | two-valued `B` | PARTIAL (no named builder, no test) | propose PG diff | — |
| `x IS [NOT] DISTINCT FROM y` / `FROM NULL` | NULL ≡ NULL | `FROM y`: needs col-col compare (G4). `FROM NULL` is `IS [NOT] NULL` (`is_null` / `is_not_null`); no NULL operand needed | two-valued `B` | ABSENT (`FROM y`) / composable (`FROM NULL`, no named builder) | — | Route (`FROM y`) |
| `p IS [NOT] TRUE/FALSE/UNKNOWN` | collapses UNKNOWN | no surface: `sql_where` only exposes `T(φ)`; `rewrite(_, false)` (F) is private `:556` | would need `(T,F)` exposed | ABSENT (G3) | — | — |
| `BETWEEN a AND b` | `x>=a AND x<=b`, K3 | `Filter::and([Ge a, Le b])`; contract interval fusion in `normalize_kleene` (rewrite 3) | ThreeValued | VERIFIED by composition (no named builder) | `set_algebra:906` interval cases | — |
| `NOT BETWEEN`, `BETWEEN SYMMETRIC` | K3; symmetric swaps bounds | `negate(and(...))` + `sql_where`; symmetric = bound swap at bind | ThreeValued | PARTIAL (SYMMETRIC: bind-side, no builder) | propose | — |
| `x IN (c1..cn)` | OR of `=`; K3 | `Filter::in_u32/in_i32` `:446/:455` (empty list refused → `LowerError::EmptyJunction`) | ThreeValued | VERIFIED | `duckdb_differential::in_count` (CLAIMED body) | Route (`EmptyJunction`) — PG rejects `IN ()` syntactically, consistent |
| `x NOT IN (c1..cn)`, no NULL in list | AND of `<>`; NULL x → UNKNOWN | `negate(in_*)` + `sql_where` | ThreeValued | VERIFIED by composition | covered by `sql_null_3vl` random trees (CLAIMED) | — |
| `x NOT IN (…, NULL)` | never TRUE → WHERE keeps **0 rows** | ⊘ corrected (was "ABSENT, would need a constant-UNKNOWN leaf"): Quack has no dedicated NULL-literal `Filter` variant and needs none. The NULL member is a `Cmp` on a NULL operand: a lane alias whose validity plane is all clear, listed in `sql_where`'s `nullable`, so `T = F = ∅`. `negate(or([in_i32(..), Cmp(N, ..)]))` | ThreeValued; `(T,F)` | VERIFIED (#1682, existing Quack) | `tests/sql_null_operand.rs`; `examples/qsp_null_operand_pg_parity.rs` (lab, SKIP without a server) | — |
| `x = NULL` | UNKNOWN always | ⊘ corrected (was ABSENT): `Cmp(N, ..)` on the NULL operand above; `T = F = ∅` on every row, distinct from a FALSE literal (a present all-zero `Plane`, `F` = every row) | ThreeValued; `(T,F)` | VERIFIED (#1682, existing Quack) | same as above | — |
| `COALESCE(x, c)`, `NULLIF(x, c)` | value expressions | none at query time (`lib.rs:166-167`, `sql_where:542`). `ClassView::default_targets` + `execute_defaults` (`class_view.rs:1018`) is a **write-if-blank mutation** over a presence `WideFieldMask`, not a read-time COALESCE | different law (mutation, phase "defaults before compute_dag") | ABSENT (G5) | — | — |
| `CASE WHEN p THEN a ELSE b END` (projection) | UNKNOWN p → ELSE | `Agg::BlendI32 {then, els}` `lib.rs:1147` over mask `T(p)` from `sql_where`; grouped → `LowerError::GroupedBlend` | blend mask `B`; output validity NOT carried | PARTIAL (nullable `then/else` lose NULL; grouped refused) | propose | Route (`GroupedBlend`) |
| `CASE` inside `WHERE` | `CASE WHEN p THEN q ELSE r` keeps `T(p)∧T(q) ∨ ¬T(p)∧T(r)` | needs `IS NOT TRUE` (`¬T(p)` = `F∪U`) — not expressible through `sql_where` | — | ABSENT (G3) | — | — |

## Stratum 1b — WHERE collapse vs CHECK

| PG construct | PG semantics | Surface | Law | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `WHERE φ` | keep rows where φ is TRUE (designated value TRUE) | `Filter::sql_where` `:549` returns `T(φ)`; measures count TRUE rows (`set_rules::Truth::ThreeValued` doc `:49-56`) | `(T,F)` → `T` | VERIFIED | `sql_null_3vl`, `k3_pg_parity` | — |
| `CHECK (φ)` / `FILTER`-free constraint test | passes on TRUE **or UNKNOWN** (= `¬F(φ)`) | expressible: `Filter::negate(Filter::negate(φ).sql_where(n))` = `¬T(¬φ)` = `¬F(φ)`; no named API; `normalize_kleene` may NOT be used (it preserves only the TRUE set) | `(T,F)` → `¬F` | PARTIAL (G3; no test) | propose: `SELECT count(*) FILTER (WHERE (φ) IS NOT FALSE)` diff | — |

## Stratum 2 — aggregates (D-QSP-1)

| PG construct | PG semantics | Surface | Law / carrier | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `COUNT(*)` | rows passing WHERE; empty → 0 | `Agg::Count` → `Terminal::Count` `ir.rs:290`; relalg `Measure::Count`; contract `Measure::Count` `set_rules.rs:111` | `ℕ` | VERIFIED | `duckdb_differential::sel_count_posted` etc. (CLAIMED bodies) | — |
| `COUNT(x)` | non-NULL x only | caller ANDs `Plane(valid(x))` into filter (`sql_where` doc `:543-544`; `AvgPlan` doc `:1500-1505`) | `ℕ` over `T ∧ valid` | VERIFIED (convention, not a type) | `sql_null_3vl::aggregates_keep_their_null_conventions:483` | — |
| `COUNT(DISTINCT x)` | distinct non-NULL values | `Agg::CountDistinctOrderedU32` `:1170` → `Terminal::CountKeyRunsU32` `ir.rs:401`; key lane must be KEY-ORDERED, enforced; unordered → refused (no seen-set fallback; `ScatterCountU32` held) | `ℕ`; NULLs excluded only if caller gates `valid(x)` | PARTIAL (ordered lanes only; doc at `:1159` links a non-existent `Agg::CountDistinctU32`) | `mask-risc/tests/distinct.rs::on_a_lane_in_key_order_…`, `a_lane_out_of_key_order_is_refused_…` | Route (`ExecError::LaneNotOrdered`) |
| `SUM(x)`, non-empty | Σ non-NULL; int4 → bigint | `Agg::SumI32` → `MaskedSumI32` (i64, exact up to `2^32` rows) `ir.rs:298` | `Arith::Exact`, `Σ0` | VERIFIED (with `valid(x)` gate) | `duckdb_differential::sum_posted` (CLAIMED) | Route (`SumRowBound` past `2^32`) |
| `SUM(x)` empty / all-NULL | **NULL** | scalar: `MaskedSumI32` returns 0 → **wrong** unless paired with a count (`avg_finish`-style `count==0 → NULL`); grouped: `GroupAgg::SumI32 → GroupFold::SumSymI32` `lib.rs:1290-1294`, `ir.rs:525` keeps empty distinct from 0; contract `SumFamily::Sym` `set_rules.rs:58` | scalar `Σ0` (wrong carrier); grouped `Σsym` + `K-mask` via `normalize_group_sink` `:1858` | PARTIAL (G7: no scalar NULL-preserving SUM) | Kleene entry "Open": no PG differential for empty SUM yet | Sem (`Refusal::SymFamily` blocks unsound rewrites) |
| `AVG(x)` | Σ/count non-NULL; empty → NULL | `lower_avg` `:1522` (two programs) + `avg_finish` `:1611` (`count==0 → None`) | `Σ0 / ℕ`; numeric vs f64 divergence | PARTIAL (PG `avg(int)` is `numeric`, here `f64`; D-QSP-3) | propose | — |
| `MIN(x)`/`MAX(x)` empty/all-NULL | NULL | scalar `MaskedMinI32/MaxI32` → `None` if empty `ir.rs:300`; relalg `Answer::Min(Option)` `lower.rs:31`; r2il pushes seed `i64::MAX/MIN` (r2il doc) | `seed` as NULL marker | VERIFIED (Quack/relalg) / trap in r2il (T7) | `duckdb_differential::min_posted_cc3` (CLAIMED) | — |
| `agg(x) FILTER (WHERE p)` | per-aggregate row filter, K3 on p | one `Query` per aggregate (each with its own `filter`) — the `AvgPlan` pattern; `GroupHaving` forces ONE shared filter for all aggs `:1656-1667` | `T(p)` per program | PARTIAL (scalar OK; grouped multi-agg with different FILTERs ABSENT) | `k3_pg_parity` uses `count(*) FILTER (WHERE …)` as its PG probe | — |
| `HAVING` | group predicate over aggregates; NULL agg → UNKNOWN → dropped | `GroupHaving` / `lower_group_having` `:1688` + `GroupHavingPlan::finish` `:1787` (unreached group never emitted; `HavingCmp` vs i64 const; conjunction only) | `K-mask` (`present`, `keep`) | PARTIAL: conjunction of `agg op const` only; no OR/NOT; no agg-vs-agg; correct NULL only when every agg's NULL ⇔ group unreached (T4) | propose PG diff incl. `HAVING min(x) > 0` with an all-NULL-x group | Route (`HavingAggOutOfRange`, `EmptyGroupUniverse`) |

## Stratum 3 — grouping / dedup

| PG construct | PG semantics | Surface | Law | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `GROUP BY k` (non-NULL categorical) | one group per distinct k | `lower_group_by` (K programs) `:1407`, `Agg::GroupReduce{GroupAddr::Local}` `:1201`, `GroupSumI32`; `lower_group_by_auto` `:1885` | key lane `0..K`; keys ≥ K **silently dropped** (zero-fallback) | VERIFIED for dense dictionary keys | `duckdb_differential::run_group*` (CLAIMED) | Route (`EmptyGroupUniverse`) |
| `GROUP BY k` via FK, two-column | | `GroupAddr::Via` / `GroupAddr::Pair` `:1235-1260` | fused; dangling fk dropped | VERIFIED (shape) | CLAIMED | — |
| `GROUP BY k` with NULL keys | all NULLs form **one** group | none: NULL payload is garbage; dropped if ≥K else mis-bucketed; caller must gate `valid(k)` → PG's NULL group lost | needs a reserved NULL code from the binder | ABSENT (G2) | propose PG diff with NULL keys | — |
| `SELECT DISTINCT k` | set of values, NULLs not distinct | `Agg::ScatterOrU32` into a K-domain mask (legal only as the demanded result) / ordered `CountDistinctOrderedU32` for its count | `B` over key domain | PARTIAL (no NULL member; no multi-column DISTINCT) | propose | — |
| "NULLs not distinct" (DISTINCT/GROUP BY/UNION) vs `=` | equality for grouping treats NULL=NULL; `=` does not | no carrier of the NULL group (G2) | — | ABSENT | — | — |

## Stratum 4 — set operations

| PG construct | PG semantics | Surface | Law | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `UNION` / `INTERSECT` / `EXCEPT` over **row identity of one table** | dedup; NULLs not distinct | `SetExpr::Or/And`, `Rel::diff` `relalg ir.rs:106`, `Runs::{union,intersect,difference}` `set_algebra.rs:636-675`; DR `Pop::{or,and,and_not}` `pop.rs:58-78` | `Population::Distinct`, `B` | VERIFIED only when both inputs are selections of the SAME population and the projection is the row id | relalg suites (CLAIMED) | Route (`Pop` → `Refusal::Stale` across versions) |
| `UNION` etc. over projected values | dedup on projected tuples | none (row-set ≠ value-set: two rows may project equal) | — | ABSENT | — | — |
| `UNION ALL` | bag sum | count only: `Measure::plus(Count(A), Count(B))` `set_rules.rs:137` | `ℕ`, `Population::Bag` | PARTIAL (counts/sums only; no bag carrier for rows) | propose | Sem (`Refusal::Bag` for range/complement) |
| `INTERSECT ALL` / `EXCEPT ALL` | bag min / monus | none | — | ABSENT (G8) | — | — |

## Stratum 5 — joins (D-QSP-2)

All join leaves are under **ADMISSION B**: `Filter::Semijoin`, `Pred::EqU32Via`, `MaskOp::Gather` "need
call-site classification before any removal" (`six-decisions.md:44-46`); no value-directed foreign gather
at PREPARE.

| PG construct | PG semantics | Surface | Law | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `INNER JOIN` child→parent (n:1, PK) as filter | rows of child with matching parent satisfying p | `Filter::EqU32Via{fk,key,v}` `lib.rs:362`; NULL fk → UNKNOWN (`sql_where` doc `:518-522`); dangling fk → no row | `B`, ThreeValued on fk; PK uniqueness ASSUMED, not witnessed | PARTIAL (no witness that the foreign key lane is a PK; foreign-side NULL "not modelled" `:530`) | `sql_null_3vl::semijoin_over_a_null_fk_is_false_and_eq_via_is_unknown:784` | — |
| `INNER JOIN` aggregate over parent key | `SUM(l.a) GROUP BY p.k` | `Agg::GroupSumViaI32`, `GroupAddr::Via` | fused 2 hops, zero-fallback drops | VERIFIED (shape) | CLAIMED | Route (`SumRowBound`) |
| INNER JOIN fan-out (1:n, n:m) multiplicity | output row per match | none: `ScatterOrU32` is set-valued (union across repeats) `ir.rs:325` | needs bag/ℕ-per-row carrier | ABSENT (G8) | propose | Sem (`Bag`) |
| `LEFT JOIN` | NULL-extended unmatched rows | none (no join output, no NULL extension) | — | ABSENT | — | — |
| `EXISTS` (SEMI) | known TRUE/FALSE; NULL fk → FALSE | `Filter::Semijoin{fk, foreign}` over a RESIDENT foreign plane `:383`; NULL fk gated, `T=valid∧leaf`, `F=¬T` | two-valued `B` | VERIFIED (resident-plane precondition; ADMISSION B pending) | `sql_null_3vl:784` | — |
| `NOT EXISTS` (ANTI) | TRUE for NULL fk | `negate(Semijoin)` + `sql_where` → `F(Semijoin)` = `¬(valid∧leaf)` | `B` | VERIFIED | `sql_null_3vl:784` (doc lines 779-781) | — |
| `x NOT IN (SELECT y …)` | UNKNOWN if x NULL; **no rows at all** if any `y` NULL | none; must NOT be lowered as `negate(Semijoin)` | needs `(T,F)` with "subquery contains NULL" flag | ABSENT (T5) | propose: same fixture, NOT IN vs NOT EXISTS | — |
| NULL join keys | NULL never equals | `EqU32Via` UNKNOWN; `Semijoin` FALSE | as above | VERIFIED (fk side only) | `sql_null_3vl:784` | — |
| general n-ary join bodies | | `relalg::join_body::is_alpha_acyclic` `join_body.rs:21` — a GYO **diagnosis only**, no executor | — | ABSENT (diagnosis exists) | `acyclic/cyclic_bodies_are_recognised` | — |

## Stratum 6 — recursion (D-QSP-4)

| PG construct | PG semantics | Surface | Law | Status | Oracle | Refusal |
|---|---|---|---|---|---|---|
| `WITH RECURSIVE … UNION` (reachability) | set fixpoint; terminates on cycles | none general (relalg lib doc: "recursion (μ) … not built" `lib.rs:28-29`). Lab: DR256 Warshall closure of nested groups inside one 256-slot tile `qr.rs:8-14, 351` | set semantics, bounded ≤ 256 slots | PARTIAL-lab (one fixed relation; D-RFT-3 datafrog oracle is the prerequisite) | DR probe tests (CLAIMED); plan D-RFT-3 | Route (`prep::Refusal::TooLarge` > 256 groups — refused, never truncated) |
| `WITH RECURSIVE … UNION ALL` on cycles | does not terminate (PG loops / hits limits) | none | must REFUSE, not answer | ABSENT | — | (needs a refusal) |
| negative dependency / stratification | PG forbids non-monotone recursive refs | none | — | ABSENT | — | Sem (needed) |
| evolving snapshot / support deletion | — | DR `Pop::repair` re-derives from direct groups (multi-support safe) `pop.rs:22-28`; `Refusal::{Stale,OutOfFocus}` | witnessed version | PARTIAL-lab | CLAIMED | Route (`Stale`) |

## Stratum 7 — scalars (D-QSP-3) — scope marker only

- Integer: only `i32` lanes compared to constants and `i32→i64` exact sums (`Arith::Exact`,
  `MASKED_SUM_I32_MAX_ROWS = 2^32` `ir.rs:705`, refused past it). No arithmetic expressions at all (r2il
  `IntAdd` → `Unimplemented`). PG `int4 + int4` overflow **errors**; `Arith::Checked` exists as a law
  (`set_rules.rs:43`) but no evaluator implements checked expressions. Literal out of `i32` → `BindError::OutOfRange`.
- `numeric`, `float`: ABSENT (`Arith::Ieee` exists only as a refusal guard; `avg_finish` is `f64`).
- Collation / text order: ABSENT (codebook `=` only; plan :49 requires a separate order-preservation proof for `Lt`).
- Casts, `u32`/`u64` ordered compares: ABSENT (`NoUnsignedLaneCompare`).

## Out of scope (listed, not mapped)

Transaction isolation / MVCC (snapshot witnesses ≠ MVCC parity), DDL (`create_table` / `Registrar` is a
test-only seam `bind.rs:~390`), time zones, client protocol, error message text, `ORDER BY`/`LIMIT`
(Quack doc: absent `lib.rs:165`), strings beyond codebook equality, window functions.

---

## (1) How a ClassView method declares this — minimal law per stratum

Today `trait ClassView` (`class_view.rs:1049`) has **no per-method semantic accessor**: its method-shaped
metadata is `compute_dag` (ordering only, explicitly "NOT an ActionDef-body interpreter" `:950-954`) and
`default_targets`. The D-QSP plan wants laws "per method/operation, not one Semantics(ClassId)"
(plan :22). Minimal declaration, reusing existing types wherever one fits:

| Stratum | What the method must declare | Existing type that fits | Missing |
|---|---|---|---|
| K3 predicates | per field: nullable? and its validity plane; logic of the predicate | `BoundField.validity: Option<Mask>` (`bind.rs:87`); `set_algebra::Logic::{TwoValued,Kleene}` (`:185`); `set_rules::Truth` (`:49`) | a ClassView-side link field-position → validity (FieldRef has none; Binder owns it) — G9 |
| WHERE vs CHECK | designated values: `{T}` or `{T,U}` | none | a "designation" flag; nothing models `¬F` — G3 |
| aggregates | empty-result convention per measure; arithmetic mode | `SumFamily::{Plain,Sym}`, `Arith::{Exact,Wrapping,Checked,Ieee}`, `GroupFold` seeds | scalar NULL-preserving SUM terminal — G7 |
| grouping/dedup | NULL-key equality (NULLs-not-distinct) | none | G2 |
| set ops / joins | set vs bag; key uniqueness witness | `Population::{Distinct,Bag}`; `Semantics.universe` | bag carrier (G8); PK/uniqueness witness for `EqU32Via` |
| recursion | monotone? set/bag? bound | none (only DR `TooLarge`) | stratification + termination law |
| all | which refusals are semantic vs route | `set_rules::Refusal` (semantic); scattered route errors | a common two-sided refusal shape — G6 |

So the minimal per-method declaration is a `Semantics` vector (existing) **plus** three missing
dimensions: designation (`{T}` vs `{T,U}`), NULL-key equality, and per-field validity binding. Do not
reuse `hotplug::register_law_for` (plan :22).

## (2) Top 8 gaps, ranked by constructs unblocked

1. ⊘ **G1 — NULL literal: not a gap for the tested subset (corrected 2026-10-21, #1682).** This item said G1 needs
   a constant-UNKNOWN leaf in `Filter`/`SetExpr`. Quack has no dedicated NULL-literal variant and none is needed for
   `x = NULL`, `NOT IN (…, NULL)`, `IN (…, NULL)` and the K3 truth tables: a NULL operand is a lane alias with an
   all-clear validity plane under `sql_where`. Still not shipped: `NULLIF` results and NULL propagation through CASE
   (value expressions, G5), the NOT IN subquery's NULL branch, column-to-column comparison (G4), nullable
   arithmetic. The rest of this ranking is unchanged.
2. **G3 — expose `(T,F)` / `IS [NOT] TRUE|FALSE|UNKNOWN`** (make `rewrite(_, false)` or a `known`
   mask public). Unblocks CHECK designation, CASE in WHERE, `IS NOT TRUE`, `IS UNKNOWN`, NOT IN
   subquery, outer-NOT over normalized sets (≈6).
3. **G2 — NULL group key / NULLs-not-distinct** (binder-reserved NULL code + validity). Unblocks GROUP BY
   NULL, DISTINCT, UNION/INTERSECT/EXCEPT over nullable columns, COUNT(DISTINCT) NULL handling (≈5).
4. **G8 — bag / multiplicity carrier** (ℕ per row or per key). Unblocks INNER fan-out, LEFT counts,
   UNION ALL rows, INTERSECT ALL, EXCEPT ALL (≈5).
5. **G4 — column-vs-column compare.** Unblocks `x op y`, `IS DISTINCT FROM y`, BETWEEN with column
   bounds, join residuals (≈4).
6. **G5 — read-time NULL-aware value expressions** (COALESCE, NULLIF, nullable CASE output validity). (≈3)
7. **G7 — scalar NULL-preserving SUM** (a scalar `SumSymI32`, or a declared "SUM = Σ0 + count" finish
   like `avg_finish`). Unblocks empty/all-NULL `SUM`, AVG over numeric, HAVING on scalar-like groups (≈3).
8. **G6 — typed two-sided refusal** (`RouteRefusal` vs `SemanticRefusal` as one shape over the six
   scattered error enums), plus **G9** per-field validity in ClassView. Cross-cutting: every row's
   "Refusal" column becomes checkable rather than prose.

## (3) Known traps

- **T1. `normalize_kleene` preserves only the TRUE set.** It is not closed under an outer `NOT`: negating
  a Kleene-normalized expression is not the Kleene negation of the original's meaning on UNKNOWN rows.
  Normalize at the WHERE boundary only; never feed a normalized subterm to CHECK (`{T,U}`) or under a NOT.
- **T2. Two-valued `normalize` still lives on shared paths.** `set_program::share` calls `normalize`
  (`set_program.rs:246`); relalg's `normalize` is two-valued (sound only because relalg lanes carry no
  NULL). Any nullable lane routed through `share` reintroduces the `x<5 OR x>=5 → All` bug (PG: 1 row vs 2).
- **T3. `Sym` is assumed two-valued** under `Truth::ThreeValued` (Kleene entry "Open"): binding a `Sym`
  to a nullable predicate silently re-enables `p ∨ ¬p → All`. Contract statement, not a check.
- **T4. HAVING shares one filter across aggregates.** `finish` infers "NULL ⇔ unreached group" from sink 0
  (`lib.rs:1772-1776`). With `COUNT(*)` and `MIN(x)` over a nullable `x`, a reached group whose x are all
  NULL has PG `min = NULL` (HAVING UNKNOWN → dropped) but here MIN either sees NULL payloads (no
  `valid(x)` gate) or COUNT(*) loses rows (gate added). No correct encoding exists today.
- **T5. NOT IN ≠ NOT EXISTS.** `negate(Semijoin)` is NOT EXISTS (NULL fk → TRUE). SQL `NOT IN (subquery)`
  returns no rows if the subquery yields any NULL, and UNKNOWN for a NULL outer value. Lowering NOT IN to
  the anti-semijoin is a silent wrong answer.
- **T6. GROUP BY NULLs are equal while `=` is not reflexive on NULL**, and Quack's group key zero-fallback
  silently DROPS out-of-range keys and mis-buckets NULL payloads `< K` unless the caller gates validity —
  which then loses PG's NULL group.
- **T7. Empty-marker leakage.** `MaskedSumI32` empty → `0` (PG NULL); r2il `MIN/MAX` empty push
  `i64::MAX/MIN` onto the scalar stack; `SumSymI32` reserves `i64::MIN`. Any later scalar arithmetic
  (`IntSub` in r2il) on an unchecked marker yields a plausible wrong number. Grouped sinks must pass
  `normalize_group_sink` before use.
- **T8. Bag laws.** Idempotence (`A ∪ A = A`), absorption and `Count(¬A) = N − Count(A)` hold for distinct
  rows only; `set_rules` refuses the range/complement rules under `Population::Bag`, but `SetExpr`
  normalization itself (idempotence, absorption) has no bag guard — never normalize a UNION ALL /
  fan-out description with it.
- **T9. Unknown-literal refusal on `<>`.** `Draft::bind` refuses a text literal the codebook has never
  seen (`BindError::UnknownValue`, `bind.rs:345`) for BOTH `=` and `<>`. PG answers `x <> 'never-seen'`
  as TRUE for every non-NULL row. This is a refusal, not a wrong answer, but it is a PG divergence on
  a valid query.
- **T10. Semijoin precondition.** `Filter::Semijoin`'s foreign plane must be RESIDENT, never another
  program's `Keep` output (`lib.rs:388-398`); a subquery-as-plane pipeline is the forbidden
  population-intermediate shape and an ADMISSION B violation. ADMISSION B classification of
  Semijoin/EqU32Via/Gather call sites is still open.
- **T11. `EqU32Via` assumes PK uniqueness of the foreign row and models no foreign-side NULL**
  (`lib.rs:530`); a non-unique parent turns INNER JOIN multiplicity into a set.
- **T12. P0 performance law.** Every surface above folds masks (ns per op, µs per batch). A
  tuple/row-decoding PG oracle harness is fine **as the oracle** (it lives in the example/test), never
  as an execution path; a ms number means the path left the fold (lance-graph CLAUDE.md P0).

## Proposed differential (not built)

Extend `examples/k3_pg_parity.rs`' shape (random tables with NULLs → SQL text → `psql` → compare to an
independent row oracle and to Quack/contract): one file per stratum (`k3_*`, `agg_*`, `group_*`,
`join_*`), each case asserting either PG-equal answer or a named refusal. Without a server it prints
`SKIP` (lab instrument; no CI gate yet — Kleene entry "Not done").
