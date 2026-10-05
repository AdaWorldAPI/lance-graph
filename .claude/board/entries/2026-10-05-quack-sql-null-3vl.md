# 2026-10-05 — Quack WHERE is SQL three-valued over nullable columns; SQL READ coverage re-evaluated

**Status:** TEST-PINNED (`crates/lance-graph-quack/tests/sql_null_3vl.rs`, `bind::tests::a_nullable_field_binds_three_valued`) · OPEN (coverage items below)

## DECISION — the NULL representation (operator, 2026-10-05)

A nullable column is **an ordinary value lane plus its resident validity plane** — the separation DuckDB uses. `value 0, valid 1` is a real zero; `value 0, valid 0` is NULL. NULL never becomes a value: no boxed or tagged NULL, no second value representation, no second bitmap, no expression interpreter.

## What landed

`Filter::sql_where(&self, nullable: &[(Col, Mask)]) -> Filter` (plus `Filter::is_null` / `is_not_null` over the validity plane). It rewrites a filter into an ordinary two-valued filter that keeps exactly the rows SQL 3VL makes TRUE.

- **Model:** each subformula lowers to `T(φ)` (TRUE rows) or `F(φ)` (FALSE rows); UNKNOWN = neither. Leaf on nullable `x`: `T = valid ∧ leaf`, `F = valid ∧ ¬leaf`. `NOT` swaps T/F; `AND`: `T = ⋀T`, `F = ⋁F`; `OR`: dual. `WHERE` keeps `T(root)`. This is the `(true, known)` pair in its dual-rail form: `known = T ∪ F`.
- **mask-risc unchanged.** The output uses the same `Cmp`/`Plane`/`And`/`Or`/`Not` nodes, so `lower` and `lower_fused` run it as is; `valid(x)` is an ordinary resident-plane gate the survivor skip uses.
- **Non-nullable fast path:** a filter (or subtree) reading no listed column is returned unchanged — identical program, both lowerings. Linear size.
- **Leaves:** `Cmp` nullable by column — including `Cmp::Range`, which the executor answers from the row ordinal but which stands for a comparison on its provenance lane (codex P2; the first cut exempted it and kept NULL rows inside `[lo, hi)`); `EqU32Via` by its `fk` (a NULL fk makes `f.v = c` UNKNOWN); `Plane` never (a known Boolean — which is what makes IS [NOT] NULL exact).
- **`Semijoin` is NOT three-valued.** It is `EXISTS(SELECT 1 FROM foreign f WHERE f.rid = this.fk AND …)`; with a NULL fk the inner match is empty and `EXISTS` is FALSE, a known answer. Its `Gather` still reads the foreign plane at the NULL row's stale payload, so a listed fk gates it: `T = valid ∧ leaf`, `F = ¬(valid ∧ leaf)` — `NOT semijoin` KEEPS NULL-fk rows, `NOT eq_via` does not.
- **Binder wiring.** `BoundField` gains `validity: Option<Mask>` — the binder owns the schema, so it owns nullability. `Draft::bind` collects the validity of every field a predicate reads and returns `Filter::and(parts).sql_where(&nullable)`; with no nullable field the bound query is byte-identical to before. Four construction sites, all tests (quack, report, dir-sim).

## Gates

Independent row-at-a-time Kleene oracle over the ORIGINAL filter; NULL rows carry payloads that WOULD match (`5`, `0`, `7`). The required cases (`=`, `<>`, `NOT`, `AND`, `OR`, their negations, `[NOT] BETWEEN`, `[NOT] IN`, `IS [NOT] NULL`) on all four validity combinations, plus 300 random trees of depth ≤ 3, through both `lower` and `lower_fused`, rows and `COUNT(*)`. Pins: `NULL ≠ 0` (i32), `NULL ≠ ''` / `false` (u32 ordinal 0); the aggregate conventions (`COUNT(*)` vs `COUNT(x)`, SUM/MIN/MAX/AVG over valid only, empty ≠ zero via COUNT/MIN/AVG). Falsifier: `NOT (valid(x) ∧ x = 5)` and the raw unrewritten filter are both shown wrong on the fixture. The rewrite is one pass (each node visited once; the first cut rescanned every subtree, quadratic down a `NOT` chain — codex P2). Disable runs, each red: `Range` exempted from gating; F(leaf) without validity; NOT as gate-then-negate; F(AND) without De Morgan; T(leaf) without validity; `Semijoin` treated as an UNKNOWN leaf; `Draft::bind` without `sql_where`.

Nullable fk (`fk::semijoin_over_a_null_fk_is_false_and_eq_via_is_unknown`): every third row NULL in the fk, payload `r % 4` pointing into a 4-row foreign table whose kept plane `{0,2}` and value lane `[9,1,9,1]` make NULL payloads match (≥ 10 trap rows asserted). Nine composites against an independent oracle (EXISTS = FALSE, via = UNKNOWN), both lowerings; can-fire both ways.

## SQL READ coverage after this change (re-evaluated; nothing implemented)

| item | class | priority |
|---|---|---|
| ⊘ ~~ordered `< <= > >=` on u32/u64 … **P0**~~ — downgraded, see the classification below | Quack + mask-risc | **P1 (bounded)** |
| ordered / range predicate on a joined dimension through an fk | Quack + mask-risc semantic gap; no current workload asks for it | P1 |
| i64 / float / decimal lanes | Quack + mask-risc; no current workload asks for it | P1 |
| multi-column / mixed GROUP BY, `SELECT DISTINCT x, y` | binder concern (destination binding → existing `Local`) | **P0 for the stack, not Quack** |
| NOT IN with a NULL literal, `x = NULL` | binder (constant UNKNOWN/FALSE leaf) | P2 |
| COALESCE / NULLIF / nullable arithmetic / projection arithmetic | binder (derived lanes) + result layer | P1 |
| `Agg::All` means "every one of n_rows", not "every filtered row" | Quack semantic bug (contract) | P1 |
| scalar `SUM` over no rows = 0, not NULL | result/finalization (pair with COUNT) | P1 |
| general / grouped `COUNT(DISTINCT)` | Quack + binder (presence fold over a bound destination) | P1 |
| HAVING OR / NOT, aggregate vs aggregate, AVG; per-aggregate filters | Quack semantic gap | P1 |
| foreign-side NULL / dangling fk in `EqU32Via` | Quack semantic gap (and a DuckDB case) | P1 |
| ORDER BY aggregate / TOP-k; multi-key ORDER BY | result/finalization | P1 / P2 |
| LIMIT / OFFSET over rows (bounded first-N) | result/finalization; physical binding for ordered projections | P1 |
| INNER/OUTER JOIN result rows (joined values; NULL extension) | result/egress contract; FULL OUTER | P1; FULL P2 |
| strings: `=`/`IN` bound; ranges/`LIKE 'p%'` need a sorted versioned codebook + ordered u32; `%x%`/regex bound over the dictionary | binder | P1 / P2 |
| set operations over rows (`OR`/`AND`/`AND NOT`); over values via a shared K | covered / result-layer K-combine | — / P1 |
| subqueries: EXISTS/IN via fk covered; scalar & correlated need Query→Query composition | Quack composition contract | P1 |
| window functions | Quack semantic gap (not started) | P2 |
| aliases, SQL parsing | frontend syntax | outside |
| INSERT / UPDATE / DELETE / MERGE / DDL | future DO | outside |

⊘ The line that stood here ("P0 SQL READ gaps remain: ordered comparisons beyond i32") is withdrawn: a missing primitive is not a P0 without a workload that needs it.

### Unsigned SQL-visible fields, classified (2026-10-05)

| class | meaning | ordering | current fields |
|---|---|---|---|
| **A** numeric / order-semantic (year, sequence, timestamp) | `<` means something | needed | SAP `WORK_DAY` — an **i32** day lane, ranged with `GeI32`/`LeI32` (`lance-graph-sap/src/query.rs:35-36`); IAM prefix depth — `GeI32` (`dir-sim/src/lib.rs:141`); report measures / derived buckets — i32 (`report/src/exec.rs:242-255`). **None is u32/u64.** |
| **B** identity / ordinal / codebook / fk | equality only; `<` would invent semantics | must NOT exist | SAP `EMPLOYEE` NUMC (`EqU32`); IAM key / value ids (`EqU32`); report categorical ordinals — refused as `ReportError::OrderedCompareOnOrdinal` (`report/src/selection.rs:231`); `gremlin_parity` refuses `OrderedU32`. |
| **C** ordered address / codebook domain | order exists in the address space, not the value | `OrderedLaneWitness` + `Range` / binding | report row ranges (`Cmp::Range`, `selection.rs:179`). |

**Verdict:** no current Report / IAM / SAP workload needs class-A u32/u64 ordering. Ordered unsigned comparison is **bounded P1**, opened only by a named class-A unsigned field; it does not block Destination Binding. The u64 kernel in ndarray existing is not a reason to wire it.

**P0 SQL READ gaps after #1334:** none demonstrated. The NULL Boolean algebra is solved, and so is its binder wiring for `Draft` (nullability is a `BoundField` fact). A frontend that builds `Filter` directly, bypassing `Draft::bind`, still has to call `sql_where` itself.

## OPEN

- ⊘ ~~Whether `Binder` / `FieldKind` should carry nullability~~ — CLOSED in this PR (`BoundField::validity`).
- SAP's `CatsBatch::bind` carries NULL in OPTIONAL fields as in-lane SENTINELS, not validity planes (`lance-graph-sap/src/bind.rs`: dictionary code `0` :150, optional `pernr_d` `u32::MAX` :110, timestamp `0` :129) — a second NULL representation beside the one ruled here. Its only query (`query.rs:33-37`) reads required fields (`employee_number`, `work_date_utc`), so no wrong answer exists today; `=` on a sentinel-coded field is also safe (codes start at 1). The first SAP predicate over an optional field — or any `<>` / `NOT` on one — must first move that field to a validity plane and `sql_where`, else `<>` keeps NULL rows. P1, tied to that first query.
- Report's `Selection` builds filters without `Draft`; its plane 0 is table liveness, and no report field is declared nullable today.
- Ordered-compare substrate, VERIFIED-IN-CODE, for when a class-A field appears: ndarray HAS `gt/lt/ge/le_u64_to_mask` (`simd_masking_ops.rs:3570-3658`), unwired in mask-risc `Pred` and Quack `Cmp`; ndarray has NO ordered `u32` mask primitive. P1, not started.
