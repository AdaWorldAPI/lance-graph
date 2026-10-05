# 2026-10-05 — Quack WHERE is SQL three-valued over nullable columns; SQL READ coverage re-evaluated

**Status:** TEST-PINNED (`crates/lance-graph-quack/tests/sql_null_3vl.rs`) · OPEN (coverage items below)

## DECISION — the NULL representation (operator, 2026-10-05)

A nullable column is **an ordinary value lane plus its resident validity plane** — the separation DuckDB uses. `value 0, valid 1` is a real zero; `value 0, valid 0` is NULL. NULL never becomes a value: no boxed or tagged NULL, no second value representation, no second bitmap, no expression interpreter.

## What landed

`Filter::sql_where(&self, nullable: &[(Col, Mask)]) -> Filter` (plus `Filter::is_null` / `is_not_null` over the validity plane). It rewrites a filter into an ordinary two-valued filter that keeps exactly the rows SQL 3VL makes TRUE.

- **Model:** each subformula lowers to `T(φ)` (TRUE rows) or `F(φ)` (FALSE rows); UNKNOWN = neither. Leaf on nullable `x`: `T = valid ∧ leaf`, `F = valid ∧ ¬leaf`. `NOT` swaps T/F; `AND`: `T = ⋀T`, `F = ⋁F`; `OR`: dual. `WHERE` keeps `T(root)`. This is the `(true, known)` pair in its dual-rail form: `known = T ∪ F`.
- **mask-risc unchanged.** The output uses the same `Cmp`/`Plane`/`And`/`Or`/`Not` nodes, so `lower` and `lower_fused` run it as is; `valid(x)` is an ordinary resident-plane gate the survivor skip uses.
- **Non-nullable fast path:** a filter (or subtree) reading no listed column is returned unchanged — identical program, both lowerings. Linear size.
- **Leaves:** `Cmp` nullable by column (`Cmp::Range` reads the row ordinal, never); `EqU32Via` / `Semijoin` by their `fk`; `Plane` never (a known Boolean — which is what makes IS [NOT] NULL exact).

## Gates

Independent row-at-a-time Kleene oracle over the ORIGINAL filter; NULL rows carry payloads that WOULD match (`5`, `0`, `7`). The required cases (`=`, `<>`, `NOT`, `AND`, `OR`, their negations, `[NOT] BETWEEN`, `[NOT] IN`, `IS [NOT] NULL`) on all four validity combinations, plus 300 random trees of depth ≤ 3, through both `lower` and `lower_fused`, rows and `COUNT(*)`. Pins: `NULL ≠ 0` (i32), `NULL ≠ ''` / `false` (u32 ordinal 0); the aggregate conventions (`COUNT(*)` vs `COUNT(x)`, SUM/MIN/MAX/AVG over valid only, empty ≠ zero via COUNT/MIN/AVG). Falsifier: `NOT (valid(x) ∧ x = 5)` and the raw unrewritten filter are both shown wrong on the fixture. Disable runs, each red: F(leaf) without validity; NOT as gate-then-negate; F(AND) without De Morgan; T(leaf) without validity.

## SQL READ coverage after this change (re-evaluated; nothing implemented)

| item | class | priority |
|---|---|---|
| ordered `< <= > >=` on u32/u64; i64/float/decimal/date lanes | Quack + mask-risc semantic gap (ndarray side to verify) | **P0** |
| ordered / range predicate on a joined dimension through an fk | Quack + mask-risc semantic gap | **P0** |
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

**P0 SQL READ gaps remain:** ordered comparisons beyond i32 (and ordered via-predicates). These refuse loudly rather than answer wrongly.

## OPEN

- Whether `Binder` / `FieldKind` should carry nullability so `bind::Draft` calls `sql_where` itself (today the caller supplies `nullable`).
- Ordered-compare coverage, VERIFIED-IN-CODE: ndarray HAS `gt/lt/ge/le_u64_to_mask` (`simd_masking_ops.rs:3570-3658`), unwired in mask-risc `Pred` and Quack `Cmp`; ndarray has NO ordered `u32` mask primitive (only `eq`/`ne`/ternary/`via`). So u64 ordering is a wiring PR (mask-risc + Quack); u32 ordering is substrate-first (ndarray, then mask-risc, then Quack).
