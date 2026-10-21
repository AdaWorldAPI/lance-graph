# 2026-10-21 — D-QSP-0 remainder, slice 1: scalar aggregates under SQL NULL

**Status:** MEASURED, TEST-PINNED. Plan: `d-qsp-classview-quack-postgresql-parity-v1.md` (D-QSP-0 remainder; the D-QSP-1 aggregate rows overlap). Claim: EXO-50 `session_013z2e1YXkm6o6HNrDn5BPWA/026`, track C of `session_01G22yT6htkcdyXsihxxXdrg/033`. Gap G7 and trap T7 of `entries/2026-10-21-pg-classview-polyfill-map.md`.

## The defect

Scalar `Agg::SumI32` lowers to `Terminal::MaskedSumI32`, which starts from the additive identity. A selection with no non-NULL `x` therefore answers `SumI64(0)`. PostgreSQL answers NULL, and `0` is also the answer for values that really sum to zero, so the scalar fold alone cannot tell the two apart. `tests/sql_null_3vl.rs::aggregates_keep_their_null_conventions` pinned COUNT, MIN, MAX and AVG on the empty case, but not SUM.

## The change

`crates/lance-graph-quack/src/lib.rs`, beside `AvgPlan`:

- `lower_sum(filter, val, valid: Option<Mask>) -> SumPlan` lowers the two existing terminals `MaskedSumI32` and `Count`, both over `filter ∧ valid(val)`. Taking `valid` here means the NULL gate cannot be kept on one half and forgotten on the other.
- `sum_finish(sum, count) -> Option<i64>` returns `None` (SQL NULL) when `count == 0`.
- `SumPlan` is a type alias of `AvgPlan`: the same two folds, finished differently.

No new terminal, operator, rule or registry. `set_algebra`, `set_rules`, relalg and mask-risc are unchanged.

## Supported SQL subset (exactly what is pinned)

One table, two nullable `int4` columns `x`, `y`. `WHERE` is any tree of `=, <>, <, <=, >, >=` against an integer constant, `IS NULL`, `IS NOT NULL`, `AND`, `OR`, `NOT` (depth ≤ 3), read through `Filter::sql_where`. Aggregates, ungrouped:

| SQL | Quack | empty / all-NULL `x` |
|---|---|---|
| `count(*)` | `Agg::Count` over `sql_where` | `0` |
| `count(x)` | `Agg::Count` over `sql_where ∧ valid(x)` | `0` |
| `sum(x)` | `lower_sum` + `sum_finish` | NULL |
| `min(x)`, `max(x)` | `Agg::MinI32` / `MaxI32` over `sql_where ∧ valid(x)` | NULL |
| `avg(x)` | `lower_avg` over `sql_where ∧ valid(x)` + `avg_finish` | NULL; compared to `avg(x)::float8` with relative tolerance 1e-12 |

Plus P-EMPTYSUM: an empty `WHERE` gives one NULL answer ungrouped and zero groups for `GroupReduce { SumI32 }` after `normalize_group_sink`.

## Evidence

| what | result |
|---|---|
| `tests/sql_aggregate_null.rs`, 150 tables × 8 random `WHERE` (column-lane Kleene oracle, folds over positions) | all six aggregates equal the oracle; corpus asserted to contain empty, all-NULL and real-zero-sum cases |
| same file, targeted cases (NULL payload 9) | empty → NULL; all-NULL → NULL; `0+0` and `2+(-2)` → `0`; payloads excluded; the lone scalar fold reads both cases as `SumI64(0)` |
| same file, P-EMPTYSUM | ungrouped one NULL; grouped zero groups; a non-empty selection yields 2 groups |
| `examples/qsp_agg_pg_parity.rs`, PostgreSQL 16.15, 400 tables, 3600 queries | Quack = PostgreSQL = oracle on all six aggregates, 0 mismatches; 1455 empty, 507 all-NULL, 64 real-zero sums; naive SUM differs on exactly 1962 = 1455 + 507 (asserted) |
| quack crate | every suite passes; clippy `-D warnings` clean (Rust 1.99.0) |

Disable runs, each red on its named test(s) at an assertion line:

- D1 (`sum_finish` always `Some`): both sum tests red; the PostgreSQL example reports 1962 `sum(x)` mismatches and exits 101;
- D2 (`lower_sum` ignores `valid`): both sum tests red;
- D3 (`lower_sum` gates only the sum half): both sum tests red;
- D4 (`normalize_group_sink` marks every slot present): only the P-EMPTYSUM test red.

## Reproduce

```text
cargo test -p lance-graph-quack --test sql_aggregate_null
PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
  cargo run --release -p lance-graph-quack --example qsp_agg_pg_parity
```

Without a server the example prints `SKIP` and exits 0. It is a lab instrument; no PostgreSQL arm runs in CI.

## Open

- `SUM` over a grouped key: the coalescing `Agg::GroupSumI32` still answers `0` for an unreached group (documented zero-fallback). The NULL-preserving form is `GroupReduce { GroupAgg::SumI32 }`.
- A reached group whose `x` are all NULL (PostgreSQL: the group appears with `sum = NULL`) is trap T4; it belongs to the GROUP BY slice together with NULL keys (G2).
- Not covered: `COALESCE`/`NULLIF`, nullable arithmetic, `int4` sum overflow past `MASKED_SUM_I32_MAX_ROWS`, `numeric` AVG.
- NULL literals and `NOT IN (…, NULL)` (G1) are not pinned here, but they need no new carrier. ⊘ Corrected: EXO-50 `/027` and `/028` said G1 needs a constant-UNKNOWN `Filter` leaf and a FALSE carrier; that was asserted without reading the substrate and is false. A column's NULLs are its presence plane (`lib.rs:18`), and `sql_where`'s leaf arm (`lib.rs:580-591`) rewrites a leaf on a nullable column to `T = valid ∧ leaf`, `F = valid ∧ ¬leaf`. A NULL operand is a borrowed alias of a lane with an all-clear presence plane, so both polarities are empty (UNKNOWN); a constant FALSE is an all-zero `Filter::Plane`. On `x = {1, 2, NULL, 1}` the existing API returns `0, 2, 2, 0, 0, 0` for `x NOT IN (1, NULL)`, its negation, `x IN (1, NULL)`, its negation, `x = NULL` and its negation, identical to PostgreSQL 16.15 (EXO-50 `/029`). The G1 rows of `entries/2026-10-21-pg-classview-polyfill-map.md` ("ABSENT") are wrong for the same reason. Now pinned (slice 2, see below).

## Slice 2: NULL operands (G1), over existing Quack

**Contract.** SQL NULL is absence of validity, never `0` and never FALSE. A nullable value is a value lane plus a validity plane (1 = present, 0 = NULL). For a nullable predicate, `T = valid ∧ p`, `F = valid ∧ ¬p`, UNKNOWN = `¬(T ∨ F)`. Quack's `T` is `f.sql_where(nullable)` and `F` is `Filter::Not(f).sql_where(nullable)`.
- A NULL operand (`x = NULL`, `NULL < c`, `NULL::boolean`, the NULL member of `IN`) is `Filter::Cmp(N, …)` on a column `N` whose value lane is an alias of an existing lane and whose validity plane is all clear, listed in `nullable`. Its `T` and `F` are empty on every row.
- A FALSE literal is `Filter::Plane` over an all-zero plane that is present: its `F` is every row. TRUE is `Filter::Plane` over an all-ones plane.

**Reused, nothing added.** `Filter::{Cmp, Plane, And, Or, Not}`, `Filter::sql_where`, `lower`, `lower_fused`, mask-risc `execute_into` with `Out::Mask`, and mask-risc `Xor` then `Any` for every comparison. No new `Filter` variant, no evaluator, no counting loop, no new terminal, no change to `lib.rs`.

**Evidence.**

| what | result |
|---|---|
| `tests/sql_null_operand.rs`, 11 tests, 8 rows, NULL payload = the IN value | `x = NULL` and `NOT (x = NULL)` give `T = F = ∅`; `x IN (1, NULL)` gives `T` = the matches, `F = ∅`; `x NOT IN (1, NULL)` gives `T = ∅`; full AND / OR / NOT tables over TRUE, FALSE, UNKNOWN; FALSE ≠ UNKNOWN on the `F` plane; scrambling every invalid payload and the N lane changes no plane; both lowerings agree |
| `examples/qsp_null_operand_pg_parity.rs`, PostgreSQL 16.15, 2000 random expressions × 200 rows | `T` and `F` planes equal PostgreSQL on every expression in both lowerings, and again with every invalid payload re-drawn; 165278 UNKNOWN row-verdicts, 1369 expressions with a NULL operand, 250 `NOT IN (…, NULL)` with an empty `T` |
| built-in disable arms of the example | UNKNOWN read as FALSE differs from PostgreSQL on 1050 expressions; N left out of `nullable` on 1107 (both asserted > 0) |
| quack crate | 137 tests pass; clippy `-D warnings` clean |

Disable runs, each red at an assertion line:

- D1 (test: NULL operand mapped to the FALSE plane): 9 of 11 tests red;
- D2 (test: N not listed in `nullable`): 9 of 11 red;
- D3 (test: `sql_where` skipped, two-valued): 9 of 11 red;
- D4 (production, reverted after the run: `sql_where`'s leaf arm without its validity gate): 9 of 11 tests red, and the PostgreSQL example exits 101 on the first expression that reads a NULL payload.

**Reproduce.**

```text
cargo test -p lance-graph-quack --test sql_null_operand
PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
  cargo run --release -p lance-graph-quack --example qsp_null_operand_pg_parity
```

**Boundaries.** Quack has no dedicated NULL-literal `Filter` variant, and none is added: the subset above is expressible through a borrowed lane alias and an all-clear validity plane. The PostgreSQL examples print `SKIP` and exit 0 without a server; they are lab evidence and do not run in CI, so CI proves only the committed tests. #1682 was merged while 7 of its 17 checks were still running; all 18 later finished green on `96e8010` (checked after merge).

**Still open.** `x IS NOT DISTINCT FROM y` and other column-versus-column comparisons (G4); `COALESCE`, `NULLIF` and nullable arithmetic (value expressions); the NULL branch of a `NOT IN` subquery; the stale G1 rows in `entries/2026-10-21-pg-classview-polyfill-map.md`, which belong to their author's session.
- Two folds cost two passes over the filter, as for AVG. A fused sum+count sink is a cost question, not a correctness one; no timing was taken here.
