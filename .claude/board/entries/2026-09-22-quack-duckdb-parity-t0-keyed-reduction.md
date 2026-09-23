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

## W-C HAVING landed — the empty-group decision (2026-09-23)

**DECISION:** an empty group is marked by a SEED THE FOLD CANNOT REACH, not
by a count carried beside the value. The rule is uniform: **empty ⇔ the slot
still holds `GroupFold::seed`** (`GroupFold::is_empty_slot`). The
entry above said "decide the fused sum+count sink's shape before writing
`HAVING`"; this supersedes that, because the sentinel closes the NULL gap
with no second sink.

- Seeds: `Count` → 0 (never empty — a zero count is an answer), `MinI32` →
  `i64::MAX`, `MaxI32` → `i64::MIN`, **new `SumSymI32` → `i64::MIN`**. MIN
  cannot share SUM's seed (a min fold must start from its identity), so the
  rule is "equals its own seed", not "equals `i64::MIN`".
- **BASIS:** the same move as a 4-bit code read as −7..+7 + NaN rather than
  −8..+7: give up one representable value to get a NULL code and a
  range closed under negation. At 4/8 bit that also removes a median bias; at
  i64 the bias is negligible, and what is bought is closure + NULL.
- Cost: exactly one row of range, for this fold only.
  `GROUP_SUM_SYM_MAX_ROWS = 2^32 − 1`, because exactly 2^32 rows of
  `i32::MIN` sum to `i64::MIN`. `MASKED_SUM_I32_MAX_ROWS` stays `2^32`: the
  coalescing sums may legitimately produce `i64::MIN`.
- Kernel: ndarray #321 `masked_group_sum_sym_i32{,_via}`, first row
  REPLACES the marker, later rows `wrapping_add`. Two named closures over the
  shared `group_walk`; no new bit loop.

**HAVING is finalization.** `lower_group_having` emits one `GroupReduce` program
per aggregate over the same filter and key; `GroupHavingPlan::finish` is
the O(K) pass that yields a K-bit group mask. A group survives iff it was
REACHED (read off the first sink: count ≠ 0, or slot ≠ seed) and every
comparison holds. There is no per-predicate NULL check. Every sink shares the
filter and the key, so a reached group is non-empty in all of them, and a
guard that cannot fire would be decoration.

Five DuckDB cases, 34 total: HAVING on the selected aggregate, on a different
aggregate, fk-keyed with a conjunction, and two over a filter that empties
three groups (`HAVING COUNT(*) >= 0` and `HAVING SUM(amount) < 10000` with
the SUM sink carrying reachedness). Disable-verified red:
- reachedness off → both sparse cases;
- non-count emptiness off → the SUM-first sparse case;
- the executor routing `SumI32` through the coalescing kernel → both
  mask-risc differentials.

**Boundary refinement (same day, operator-raised):** the marker is free
inside the fold and a silent wrong answer outside it — any full-range
consumer would sum, sort or negate `i64::MIN`. Two consequences, both
landed:
- **Named, not implied.** ndarray exposes the reservation only under a
  `_sym` suffix (`masked_group_sum_sym_i32{,_via}`, `SYM_EMPTY_I64`); every
  unsuffixed reduction stays full two's-complement and never treats
  `i64::MIN` specially (pinned by a test). mask-risc names the fold
  `GroupFold::SumSymI32` with `GROUP_SUM_SYM_MAX_ROWS`. This is the 4-bit
  analogue of choosing −7..+7 + NaN by name, never silently narrowing a
  −8..+7 consumer.
- **NULL leaves as a mask.** `normalize_group_sink` is the quack boundary:
  it returns a K-bit presence mask and zeroes absent slots in place; `finish`
  runs it over every sink and returns `{present, keep}`. The raw sink is
  internal encoding. Row NULL and group NULL now have the same shape.
  Disable-verified: no zeroing → the leak assertion and unit tests fail;
  normalizing only the first sink → the multi-sink unit test fails.

REVISIT WHEN: a single-pass AVG is wanted — the fused sum+count sink is still
the route to that, now for speed only, not for NULL.

## What is still open (not done here)
- **W-C:** `ORDER BY rid LIMIT n`. This needs a first-n select, a
  rank/select member.
- **W-D:** multi-key `GROUP BY` via a fused composite address. This is a
  third `GroupKeyAddr` variant; the walker is unchanged.
- **Out of scope for T0 parity:** arbitrary-value sort, m:n hash joins,
  window functions, strings.
- **Downstream:** lance-graph CI resolves ndarray via the local path dep, so
  the HAVING branch goes green only after ndarray #321 merges.
