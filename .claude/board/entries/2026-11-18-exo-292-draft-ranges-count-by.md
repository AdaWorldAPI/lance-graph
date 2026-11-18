# EXO-292 (R7): range operators and a grouped count in Quack's `Draft`

**Date:** 2026-11-18 · **Status:** MEASURED (test-pinned) · **Scope:** lance-graph-quack (`bind.rs`)

The OGIT cache needs range predicates and per-type counts through the typed front end. `Draft` admitted `=` and `<>` only, although `Cmp` and mask-risc already had the four signed comparisons and `Agg::GroupReduce` already counted per key.

Shipped, front end only (no new `Cmp`, `Agg`, lowering or terminal):

- `Op::{Lt, Le, Gt, Ge}`, `Draft::where_cmp`, `Draft::where_between` (both ends included), and the same on `FieldRef`. They bind an integer field to `Cmp::{Lt,Le,Gt,Ge}I32`, and are three-valued over a nullable field like `=`.
- `Draft::count_by(field)`: binds to `Agg::GroupReduce { key: Local(col), agg: Count }`. The caller's `Out::I64` sink is the group universe, one slot per code.
- Refusals: a range over a coded field is `BindError::Unordered`, raised before the codebook is asked (codes are identities; their order means nothing). A group key that is not a non-nullable coded field is `BindError::GroupKey` (an integer lane has no universe the binder can size; a NULL key would be counted into the group of code 0).
- `bind_typed` witnesses the group key's code space, because the counts are indexed by code: a grouped count bound before a renumbering is not admitted after it.

Evidence: seven tests in `bind.rs` against a row-by-row scalar reading of the 200-row fixture. Kept rows for `<, <=, >, >=` at 33: 52, 55, 144, 147. `between(30, 33)`: 15. Nullable range: 6 NULL rows whose payload passes are dropped. `count_by` under `age >= 40`: [40, 40, 40], sum 120 = the plain count. Disables, each red on its own test: `>` lowered as `>=` (2 tests), the `Unordered` guard off, the `GroupKey` guard off, the key left unwitnessed.

Not done: `lower_sel` still admits no `=` / `<>` over a selection (named in EXO-292 as context, not asked). No grouped sum, min or max in `Draft`. A group key reached through a foreign key is not offered: it is the read shape EXO-140 has not classified.

OPEN: none.
