# 2026-09-23 — Quack HAVING: the `_sym` SUM, the presence-mask boundary, the twin check

**Status:** MEASURED (landed on lance-graph #1266) · OPEN (see end)
**Depends on:** ndarray #321 (`masked_group_sum_sym_i32{,_via}`, `SYM_EMPTY_I64`)
**Supersedes:** the W-C "group existence is OPEN" item of
`2026-09-22-quack-duckdb-parity-t0-keyed-reduction.md` (left verbatim there;
that entry is not edited). Its instruction "decide the fused sum+count sink's
shape before writing `HAVING`" is resolved differently below: a reserved code
inside the fold, a mask at the boundary.

## W-C HAVING landed — the empty-group decision

**DECISION:** an empty group is marked by a SEED THE FOLD CANNOT REACH, not
by a count carried beside the value. The rule is uniform: **empty ⇔ the slot
still holds `GroupFold::seed`** (`GroupFold::is_empty_slot`). The 2026-09-22
entry said "decide the fused sum+count sink's shape before writing
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

**Two SUM spellings are a deliberate twin, not debt** (operator, same
day). Full-range `GroupSumI32` + `COUNT` and `_sym` `SumSymI32` answer the
same question two independent ways, so either can check the other whenever
something looks off. `sym_sum_agrees_with_full_range_sum_plus_count` keeps
that check permanent: presence ⇔ count ≠ 0, equal values where present,
full-range 0 where absent, on both key addresses. It is also the only check
that can see a real sum colliding with the reserved code. Disable-verified:
routing `SumSymI32` through the full-range kernel turns it red.

REVISIT WHEN: a single-pass AVG is wanted — the fused sum+count sink is still
the route to that, now for speed only, not for NULL.

## What is still open (not done here)
- **W-C:** `ORDER BY rid LIMIT n` — a first-n select, a rank/select member.
- **W-D:** multi-key `GROUP BY` via a fused composite address.
- **Debt:** `TD-SYM-SUM-MERGE-IS-NOT-ADDITION-1` — `_sym` partial sinks must
  merge by the fold's rule, never `+`; dormant until a parallel reduction.
