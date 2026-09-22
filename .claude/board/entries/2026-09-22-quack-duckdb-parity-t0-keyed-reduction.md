# 2026-09-22 — Quack/DuckDB parity by T0 folding: the keyed-reduction family, and the primitive basis

**Status:** MEASURED (W-B and W-A shipped on this branch) · OPEN (W-C, W-D)
**Depends on:** ndarray #320 (keyed-reduction family), lance-graph #1262 (fold seam)

## What landed (W-B)

`GROUP BY` with `COUNT` / `MIN` / `MAX` now folds before dispatch, as `SUM`
already did (#1262). One program ends in `Terminal::GroupReduce { mask, key:
GroupKey::{Lane, Via}, fold: GroupFold::{Count, MinI32, MaxI32} }` and writes
a K-slot `Out::I64` sink. The previous shape was K+1 programs plus an O(N)
kept mask.

- `GroupFold::seed()` is 0 for COUNT, `i64::MAX` for MIN and `i64::MIN` for
  MAX. All three are outside the i32 value range, so a slot still holding its
  seed is exactly an empty group, i.e. SQL `NULL`. The harness and the #1262
  re-pin both assert this correspondence rather than assuming it.
- DuckDB differential: 21/21 match DuckDB 1.5.5. The 5 new cases are
  `group_min_cc`, `group_max_cc`, `group_max_cc_sparse` (3 empty groups →
  `NULL`), `join_group_count_country` and `join_group_min_country`. The
  existing 16 expected values did not move.
- `Agg::Rows` (a grouped projection) has no keyed-reduction law and stays
  forest residue.
- Disable-verified, each turning red:
  1. quack declining to fold COUNT
  2. resident MIN dispatched to MAX
  3. the harness printing the seed instead of `NULL`

## The primitive basis — reuse like the polyfill, expose like lgj-abi

In ndarray, `simd.rs` is a facade of names over backends. Parity work
reuses that same shape: **a small set of FAMILIES, each one private walker
with named public instances.** It does not add one kernel per SQL feature.
The keyed-reduction family is the worked example. There is one private
`group_walk(mask, n, GroupKeyAddr::{Resident, Via}, out, fold)`, and eight
public names are closures over it (sum/count/min/max × resident/via). A ninth
fold (e.g. `ANY`) is one closure and one name. It is not a new kernel.

| family | T0 facade (ndarray) | mask-risc | lgj-abi verb (T2) |
|---|---|---|---|
| predicate: type × cmp, `_under` gate | `{eq,ne,lt,le,gt,ge,ternary_match}_{i32,u32,u64,u8}_to_mask[_under]`, `eq_u32_via_to_mask` | `Pred::*` | `lgj_op_eq_u32`, `lgj_op_gt_i32`, `lgj_op_ternary_match` |
| mask algebra | `mask_{and,or,andnot,xor,not,ternlog}[_assign]`, `mask_set_range` | `MaskOp::*` | `lgj_mask_{and,or,andnot,ternlog}` |
| scalar reduction | `popcount_batch_u64`, `mask_{any,all}`, `masked_{sum,min,max}_i32` | `Terminal::{Count,Any,All,MaskedSum/Min/Max}` | `lgj_mask_count`, `lgj_reduce_*` |
| address movers | `mask_gather_u32`, `mask_scatter_or_u32`, `*_via` | `MaskOp::Gather`, `Terminal::ScatterOrU32` | `lgj_hop` |
| keyed reduction | `masked_group_{sum,count,min,max}_{i32,u32}[_via]` | `GroupSumI32`, `GroupSumViaI32`, `GroupReduce` | `lgj_plan_group_sum_i32` (SUM only) |
| rank / run / range | `masked_key_run_count_u32`, `mask_set_range` | `CountKeyRunsU32`, `Pred::Range` | — |

The lgj-abi column is the ABI analog of the facade column. `plan_lower.rs`
already carries `GroupKey::{Local, Via}`, the same address split as
`GroupKeyAddr`. The follow-up is to widen `lgj_plan_group_sum_i32` into a
`GroupReduce`-shaped plan verb (fold as a parameter). It should not become
three more exports. Java keeps seeing only the boring `sql()` / `sumBy()`
surface; it never names a fold.

## W-A landed — no new primitive, as predicted

Eight more cases, 29/29 against DuckDB 1.5.5, existing values unchanged:
- `SUM(CASE WHEN p THEN x ELSE 0 END)` is the plain masked SUM with `p` in
  the mask.
- `AVG` is `lower_avg` / `lower_group_avg`: SUM and COUNT over the same
  filter, finished by `avg_finish` outside the hot path. The float digits
  match DuckDB exactly (scalar, resident-keyed and via-keyed). It costs two
  passes; a fused sum+count sink would make it one.
- `NOT EXISTS` is `Not` around the factored fk predicate (`EqU32Via`), one
  program. The earlier prediction ("`mask_andnot` after a hop") was wrong in
  shape: no hop and no second mask are needed for the many-to-one direction.
  The one-to-many direction (docs with no posted line) still needs the
  ordered-key projection, same as `COUNT DISTINCT`.
- NULLs are a validity plane ANDed into the filter; `COUNT(col)`,
  `SUM(col)` and `AVG(col)` follow. The fixture gained one derived nullable
  column (no RNG draw).

Disable-verified: an off-by-one AVG denominator, a dropped `Not`, a dropped
validity plane, and NULL written as `0` each turned their cases red.

## What is still open (not done here)
- **W-C:**
  - `HAVING` over the K-sized sink. Correction to the plan above: filtering
    K result slots is finalization over an O(K) sink, like `avg_finish`, so
    it needs NO SIMD primitive at realistic K. What it DOES need is **group
    existence**, and that is OPEN:
    - A MIN/MAX slot at its seed is an empty group, so `HAVING MIN(x) > c`
      must skip it. A naive filter on `i64::MAX` would wrongly KEEP it.
    - A SUM slot cannot tell an empty group from a group summing to 0.
      SQL gives the empty group `NULL` (excluded by any `HAVING`), not 0.
    - Both are solved by carrying a count beside the value. That is the
      same fused sum+count sink AVG wants to become one pass instead of
      two, so one keyed-reduction member closes two gaps. Decide that sink's
      shape before writing `HAVING`, not after.
  - `ORDER BY rid LIMIT n`. This needs a first-n select, a rank/select member.
- **W-D:** multi-key `GROUP BY` via a fused composite address. This is a
  third `GroupKeyAddr` variant; the walker is unchanged.
- **Out of scope for T0 parity:** arbitrary-value sort, m:n hash joins,
  window functions, strings.
- **Downstream:** lance-graph CI resolves ndarray via the local path dep, so
  this branch goes green only after ndarray #320 merges.
