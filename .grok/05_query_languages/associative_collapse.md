# Associative collapse

The rewrite that turns repeated folds of one borrowed aperture into one fold, before `lower` returns. It is the second deforestation. The first deleted the pair list and left the mask. This deletes the mask's second walk.

Invariant page: `board/LANE_FOLD.md`. Integration: `board/LANE_FOLD_INTEGRATION.md`. Rooms: `board/LANE_FOLD_SEVENTEEN_ROOMS.md`. Literature: `05_query_languages/lane_fold_deforestation.md`.

## The law

A terminal \(f\) over an aperture \(A\) collapses across repeats when

\[
f(A) = f(A_1) \oplus f(A_2)
\]

for any split of \(A\) into disjoint \(A_1, A_2\), and \(\oplus\) does not depend on order. The terminals that satisfy this on the lane:

| terminal | \(\oplus\) | identity | collapses to |
|---|---|---|---|
| `COUNT` | addition | 0 | one popcount |
| `SUM` of `i32`, widened to `i64` | addition | 0 | one masked sum |
| `MIN` | minimum | \(+\infty\) (empty seed) | one min |
| `MAX` | maximum | \(-\infty\) | one max |
| `AVG` | not itself | — | collapse the sum and the count, divide once at the end |
| `EXISTS` | or | false | one word-or of the mask, short-circuit on the first live word |

`AVG` is not associative. The pair `(sum, count)` is. Dividing per fold and averaging the averages is the wrong rewrite. A group-sum is associative per group, so K repeated group-sums of the same key collapse to one `GroupSum`, not to one scalar.

A full rail count collapses further, to the constant 65536. A shift window of width `w` on a dense rail collapses to `w`. Those are room 13, closed forms, and they are a separate pass of the same rewriter. Do not mix them into the homomorphism pass. A constant fold that is wrong is worse than a fold that walked.

## The process

Input is a plan whose terminals borrow apertures. Output is a plan with fewer terminals. Quack runs this before `lower`. It does not execute.

1. Name each aperture. A borrowed mask of 1,024 words, or a shift `d: u16` on a named lane. Two terminals share an aperture only if they borrow the same value. A fold that writes a mask starts a new aperture. The group breaks there.
2. Partition terminals by `(aperture, operation, lane)`. Sum of amount and sum of tax share the mask and do not merge. They may share a pass later. That is independent-terminal fusion, a different rewrite.
3. Inside a partition of size `n`, emit one terminal. The `n` consumers receive that value. If the request was "add this sum to itself `n` times", scale by `n` after the one reduction. Those two requests must be distinguished in the plan, or the rewrite will silently multiply.
4. Leave non-members alone. Median, first-value, string aggregate, a fold whose aperture changed, a consumer who asked for rows.
5. Record the collapse in the metric line: `collapsed_terminals`, `remaining_terminals`. A million going to one is the proof. A million going to a million is the rewrite not firing.

The executor then sees one terminal. The 999,999 copies are of the result, not of the walk. On the count probe a plane fold was 605 ns and a word-fold was 0.6 ns. A million uncollapsed word-folds are 0.6 ms. Collapsed, they are one plane pass.

## Where it is illegal

- The operation is not a homomorphism. Two medians of halves are not the median.
- The aperture is not stable. Fold 50 changes the mask. The group ends at that boundary.
- The consumer asked for rows. Collapse still answers an aggregate. It does not answer a pair list. That request is capped and labeled, or refused.
- The carry bound. `SUM` of `i32` widens to `i64` and rejects past `MASKED_SUM_I32_MAX_ROWS`. Collapsing `n` sums does not loosen that bound. Scaling by `n` after the reduction can overflow the `i64`. Check the scale, do not wrap.
- Empty `MIN` / `MAX`. The seed is the identity, not a row. A group no selected row named is still the SQL NULL. Collapse must not turn an empty group into a zero.

## Best version

The best version is a planner that emits one borrowed aperture and one terminal per homomorphism class, with the executor unchanged except for a tile-level zero skip.

A query arrives. The lowering names apertures instead of allocating them. Filters become gates on the running mask, ordered by dead-word fraction when the plane is not already resident. A shift join is a rebased borrow, not a walk. Repeated sums of that aperture become one `MaskedSum`. The differential checksum is itself a fold over the same aperture, so the proof does not build a row vector in order to hash it.

What the best version does not contain: a pair list, a hash table under the fold's name, a runtime hill-climb inside quack, a fused plan that claims the skip, a route through the cognitive layers. The residual stays visible. A real zip, a non-associative consumer, a string, a scattered extract the caller asked to see, a cyclic n-hop. Those are refusals with their own names.

Measured, the best version is allowed to claim: checksum match against DuckDB, `alloc_bytes_exec = 0`, `pair_relation_bytes = 0`, `collapsed_terminals` printed, tile ns and plane ns on separate lines. It is not allowed to claim the 0.6 ns figure for a join, or the 4–12 ns tile band until a single-tile probe exists.

## Next integration steps

These extend `LANE_FOLD_INTEGRATION.md`. Order is the order of leverage.

1. Land the collapse rewriter in quack, behind `Plan::check`, with the metric line. No kernel change. A test with 1,000 identical sums asserts one terminal in the lowered program.
2. Distinguish "repeat the value" from "scale the value" in the query type, so step 1 cannot silently multiply.
3. Tile-level zone skip in `execute`. Dead-tile fraction printed beside dead-word fraction.
4. Fuse-or-gate switch from that fraction. A fused plan with `claims_skip` does not pass `Plan::check`.
5. Range-as-span for `Pred::Range` on an ordered rail. Until then that predicate is a scan.
6. Timed column on the DuckDB differential, same cases, checksum first.
7. Move `lane_guard.rs` onto the build. A note that restates the type is already stale.

Done when a million identical sums lower to one terminal, the oracle still matches, and the three timings stay on three lines.
