# 2026-10-21 — D-RFT-0 / D-QSP-0: set normalization under SQL NULL (Kleene form)

**Status:** MEASURED, TEST-PINNED. Plans: D-RFT-0 and D-QSP-0 in #1667 (`d-rft-relalg-thought-git-frontier-v1.md`, `d-qsp-classview-quack-postgresql-parity-v1.md`; not yet on main). Claim: EXO-50 `session_01G22yT6htkcdyXsihxxXdrg/031`.

## The defect

`contract::set_algebra::normalize` is two-valued. Two of its rewrites assume a row is either in a set or in its complement:

- a union that covers a whole lane becomes `All`. This covers interval unions, `x≠a ∨ x≠b`, `x≠a ∨ (interval ∋ a)`, and on `U32` lanes `≠a ∨ ≠b` and `≠a ∨ =a`;
- a conjunct whose interval covers the whole lane is dropped.

On a NULL row both turn UNKNOWN into TRUE. PostgreSQL 16.14: `SELECT count(*) FROM (VALUES (1),(NULL)) v(x) WHERE x<5 OR x>=5` is 1, but `All` is 2.

Because of this, `set_rules::SetNormalize` refused every rewrite under `Truth::ThreeValued`. That refusal was correct but blanket: it also gave up De Morgan, flattening, absorption and interval fusion, all of which keep the TRUE rows.

Not affected:
- Quack `Filter::sql_where`, whose T/F polarity encoding is correct and is pinned by `tests/sql_null_3vl.rs`;
- relalg, whose lanes carry no NULL (its two-valued `normalize` is unchanged; 8 suites pass).

## The fix (precise, not blanket)

`normalize_kleene(e)` (and `normalize_with(e, Logic::{TwoValued, Kleene})`) runs the same rewrites with the lane-cover cases replaced:

- a covered `I32` lane becomes `x >= i32::MIN`, which is TRUE exactly on the non-NULL rows. No new vocabulary is needed; it is an existing `Res`;
- a covered `U32` lane keeps its atoms, because no single `U32` atom is TRUE exactly on non-NULL rows;
- the whole-lane atom is emitted only where no `!=` already names the lane. Every `!=` is FALSE on NULL rows anyway, so the atom would be redundant. This keeps the leaf count from growing.

Why the rest is sound: De Morgan and the negated comparison are Kleene equivalences. After negation normal form no negation sits above a junction, so any rewrite that keeps a subterm's TRUE rows keeps the whole expression's TRUE rows (e.g. `p ∧ ¬p → Empty`). A `Sym` is two-valued by contract, so its complement rule stays.

`SetNormalize` under `ThreeValued` now applies the Kleene form. Its catalogue entry has no refusals. The `Complement` and `RelationalDifference` guards are unchanged and still refuse under `ThreeValued`.

## Evidence

| what | result |
|---|---|
| contract `set_algebra`: Kleene evaluator, 3000 random expressions with NULL lanes | TRUE rows identical, idempotent; > half rewritten; two-valued form wrong on > 30 of the same inputs |
| contract `set_algebra`: the six lane covers one at a time, rows `[3, NULL]` | Kleene keeps 1 row in 2; two-valued keeps both |
| contract `set_rules`: 600 random measures over NULL tables, all rules, three-valued | answer = Kleene oracle; `SetNormalize` fired > 300 times; two-valued normalization alone wrong on > 5 |
| example `k3_pg_parity`, PostgreSQL 16.14, 2000 expressions × 200 rows | as written = Kleene oracle = `normalize_kleene` on 2000/2000 (1537 rewritten); two-valued `normalize` wrong on 474 |
| contract suite | 1549 + 1 passed; clippy `-D warnings` clean (Rust 1.99.0) |

Disable runs, each red on its named test:
- D1 (I32 cover → `All`): 3 red;
- D2 (no whole-lane atom): 4 red;
- D3 (U32 cover → `All`): 2 red;
- D4 (`SetNormalize` uses the two-valued form): 2 red;
- D5 (whole-lane atom also beside `!=`): green at first, a size guard only. Pinned by `normalize_kleene_never_grows_leaf_count`, then red.

Re-pinned: `experiment_e_null_refuses_set_normalization` → `experiment_e_null_normalizes_in_the_kleene_form` (rewrite applied, count 2 kept). The complement experiment now isolates `Complement` with `rewrite_with`.

## Reproduce

```text
cargo +1.99.0 test --release -p lance-graph-contract --lib set_
PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
  cargo +1.99.0 run --release -p lance-graph-contract --example k3_pg_parity
```

Without a server the example prints `SKIP` and exits 0.

## Open

- **`Sym` is assumed two-valued** under `Truth::ThreeValued`. A consumer that binds a `Sym` to a nullable predicate must not take this path. That is a contract statement, not a check.
- **Out of scope (D-QSP-0 remainder):**
  - a NULL literal (`x = NULL`, `NOT IN (…, NULL)`);
  - `COALESCE` / `NULLIF`;
  - nullable arithmetic;
  - empty / all-NULL `SUM` (the `SumFamily::Sym` path exists; no PostgreSQL differential yet);
  - `COUNT(expr)` vs `COUNT(*)`.
- **Not done:** a PostgreSQL arm in CI. The example is a lab instrument only.

## Correction: WHERE-terminal admission (post-merge review)

Written by `session_013z2e1YXkm6o6HNrDn5BPWA` on the operator's instruction (EXO-50 `/038`), from `GPT:gpt-ada-01/20261021-pr1674-1682-consolidated-review-01` item 3.

- `normalize_kleene` keeps the TRUE rows only; #1677 measured FALSE rows moving on 337/2000. It now returns `set_algebra::WhereSet` instead of a `SetExpr`. `where_true()` / `into_where_true()` name the WHERE-only restriction at every call site; `normalize_with` is private, so the Kleene result cannot leave the module as a plain `SetExpr`. `set_rules`' `SetNormalize` (a measure, terminal) and the two PostgreSQL examples take the set out explicitly. No new rule or framework; the existing `Semantics::truth` gate is unchanged.
- Test `the_kleene_form_preserves_true_rows_but_not_false_rows`: `p ∧ ¬p` with `p` UNKNOWN keeps its (empty) TRUE rows, becomes `Empty`, and `NOT` over it is TRUE where `NOT (p ∧ ¬p)` is UNKNOWN.
- GKAT (#1679): `Cond::Not` negates a projected guard. `tests/k3_negation_duality.rs` checks on all four atoms (TRUE, FALSE, NULL under both payloads) that `Where(NOT p) = NOT Check(p)`, `Check(NOT p) = NOT Where(p)`, that `NOT Where(p)` differs from SQL `WHERE NOT p` exactly on the two NULL atoms, and that no form reads a NULL payload. Two disables red (`Where` or `Check` without the validity test). The GKAT algorithm is unchanged; `flow.rs` docs name the boundary.
- Unchanged after the change: contract 1626 tests pass; `k3_pg_parity` 2000/2000; `k3_where_pg_parity` TRUE/FALSE equal 2000/2000, FALSE moved on 337, two-valued wrong on 499.
