# 2026-11-14 — D-ARGMINMAX-1: argmin / argmax as a fold that returns a coordinate

**Kind:** production change in `lance-graph-mask-risc` (two terminals, one
value), lab consumer in `crates/dr256-ternlog-lab`, kernel in the ndarray fork
(`masked_argmin_i32`, `masked_argmax_i32`, `masked_argminmax_i32`).
**Grade:** MEASURED for the kernel and the terminals; the DR `LocalMap`
inverse step is NOT TESTED (see "Not established").
**Depends on:** ndarray branch `ccr-1fd55114-ndx7e4` (the kernel). This
crate does not build against ndarray `master` until that lands.

## Question

Can argmin / argmax run on the existing frame-bound population with SIMD and
return a coordinate in the same address space, with no second population,
identity or executor?

## What existed (read at source)

| capability | owner | state before |
|---|---|---|
| min / max value over a mask | `ndarray::simd::masked_min_i32` / `masked_max_i32`; `Terminal::MaskedMinI32` / `MaskedMaxI32` | scalar walk over set bits, value only |
| grouped min / max | `masked_group_min_i32` family; `GroupFold::MinI32` / `MaxI32` | value only |
| unmasked argmin / argmax | `ndarray::hpc::reductions::argmax_f32` / `argmin_f32` | `f32`, index lanes in `u32`, lowest index on ties, NaN skipped |
| coordinate of an extreme under a mask | none | two programs (`MaskedMinI32`, then `Pred Eq` under the mask with `Keep`) plus a scan of the kept mask by the caller |
| fused min + max with positions | none | — |

## Decisions

- **Tie rule: lowest row.** Same as `argmax_f32` and as upstream. The row is
  the row of the `Planes` the program runs over, the coordinate every mask
  and lane of that program is already in.
- **Merge law:** better value wins; on equal value the lower row wins.
  Associative and commutative, so tiles, extents and slabs merge in any order.
- **No index vector.** The kernel keeps one scalar candidate and asks each
  16-lane group one packed compare. Positions are `usize` from the group
  base, so there is no index width to overflow. Upstream `argminmax` carries
  positions in SIMD lanes and chunks its input to keep them in range; that
  mechanism is not needed here and was not ported.
- **A new terminal, not a composition.** The existing operators give the
  value, or a mask of every row holding it; the single coordinate needs a
  second program, a kept mask as large as the population and a scan outside
  the executor. `the_coordinate_equals_what_two_existing_programs_find` shows
  the two routes agree; the benchmark shows the cost (arm E against arm A).
- **Result type:** `Value::OptRowI32(Option<(usize, i32)>)`. In the lab fold
  the answer is `(CoordSpace, row, value)`: the existing space type, no new
  identity.
- **Upstream code harvested:** none. `jvdd/argminmax` (MIT, commit
  `0910e37`) was read for its tie rule, its two NaN policies, its overflow
  chunking and its Arrow validity bitmap; the kernel here is written against
  `I32x16::gt_bitmask` and shares no code with it.

## Falsifiers

- ndarray `tests/masked_argminmax_i32.rs`: 17 tests against a scalar oracle
  that never calls the kernel (empty, one row, ties, identical values,
  `i32` extremes, sparse and dense masks, tail bits past the length,
  masked-out extremes, segment merges in three orders, a reordered lane,
  positions past 2^17, monotone lanes, short mask panics). Three rows in the
  cross-backend parity program; the native arm (AVX-512 host) passes.
- `lance-graph-mask-risc/tests/arg_extrema.rs`: 9 tests, executor at two
  tile widths against the reference evaluator; partial extents merged in
  three orders; the two-program equivalence. The four new terminals are also
  in `tests/differential.rs`.
- `dr256-ternlog-lab/tests/arg_coordinate.rs`: 7 tests on 3 slabs plus a
  partial one: scalar oracle, lowest slab then lowest row, the same row in
  two slabs is two coordinates, two tables do not share coordinates, a moved
  source frame returns a refusal and never a coordinate.
- Disable runs, each exit 101 with the anchor asserted: ndarray 5 of 5
  (tie comparison, tail lanes, mask on the max side, group base, sparse-path
  base); lance-graph 6 of 6 (tile base, later tile wins a tie, oracle tie
  order, later slab wins a tie, slab space on each side).
- One knob is NOT a correctness guard: raising `ARG_FOLD_SPARSE_WORD` from 4
  to 64 leaves all 17 tests green. Both paths give the same answer; the
  constant only chooses the cheaper one.

## Measured

Intel Xeon 2.10 GHz, 4 vCPU, AVX-512, release, `target-cpu=native`,
minimum of 5 runs, ns per element. 0 allocations in every arm. A = new
argmin, B = new fused argmin+argmax, D = existing value-only min, E =
two-pass (min, then equality mask, then first bit), G = scalar bit walk with
position, H = upstream `argminmax` (unmasked, dense rows only).

| rows | data | mask | A | B | D | E | G | H |
|---|---|---|---|---|---|---|---|---|
| 65,536 | random | dense | 0.167 | 0.196 | 0.823 | 0.994 | 1.840 | 0.084 |
| 65,536 | random | half | 0.168 | 0.208 | 0.507 | 0.666 | 0.962 | n/a |
| 65,536 | random | 1 in 64 | 0.068 | 0.093 | 0.046 | 0.242 | 0.028 | n/a |
| 65,536 | descending | dense | 1.741 | 1.930 | 0.762 | 0.907 | 1.753 | 0.083 |
| 16,777,216 | random | dense | 0.188 | 0.225 | 0.832 | 1.182 | 1.900 | 0.144 |
| 16,777,216 | random | half | 0.191 | 0.226 | 0.573 | 0.795 | 0.938 | n/a |
| 16,777,216 | random | 1 in 64 | 0.149 | 0.160 | 0.135 | 0.351 | 0.120 | n/a |
| 16,777,216 | descending | dense | 1.814 | 1.838 | 0.861 | 1.097 | 1.869 | 0.138 |

- Dense and half masks: the coordinate fold is 3 to 5 times faster than the
  existing value-only fold, and gives the position as well.
- **Loses on a lane that keeps improving in row order** (descending for the
  minimum): every group has a better lane, the fold reads those lanes one by
  one, and it is about 2 times slower than the value-only fold.
- **Loses on a very sparse mask** to a plain bit walk (0.068 against 0.028
  at 65,536 rows).
- Upstream is 1.3 to 2 times faster on a dense unmasked lane.
- Tried and removed: narrowing an improving group by repeated packed
  compares. It was slower on every shape (dense random 0.78, descending 3.1).

## Not established

- `i32` only. No `u8`, `u32`, `u64`, `f32`, bf16 or Codec9 kernel; no NaN
  policy was chosen because no float kernel was written.
- Only the AVX-512 arm was run. AVX2, NEON, wasm and scalar build from the
  same source through `I32x16`, but their parity arms were not run here.
- The DR `LocalMap` step (local position back to a global id through
  `TileMap::global`, and a changed map digest) is not exercised. The lab
  covers `CoordSpace` and the source frame, not the map table.
- The XLOG `Cube` is not touched: it holds relation lanes of local
  positions, has no value lane, and its scope is one tile and binary
  relations.
- No grouped argmin / argmax. No Quack `Agg`, no R2IL opcode: the terminals
  are reachable only by building a `Program` directly.
- JC is not changed. No statistic in `jc::stats` needs an extreme today; its
  pattern is to interpret sufficient statistics that ndarray folds
  (`*_from_power_sums`). A peak-deviation diagnostic built the same way is a
  proposal, not a result, and an extreme carries no significance claim.
- `argmin-rs/argmin` and `finitediff` were cloned (commit `b300285`,
  MIT / Apache-2.0) and not evaluated.

## Open decisions

1. Whether the monotone-lane cost needs an `I32x16` select by bitmask in the
   ndarray wrappers (all backends), which would let an improving group be
   reduced without lane reads.
2. Whether a Quack aggregate and an R2IL word should name the new terminals.
3. Which other lane types get a coordinate fold, and the NaN rule for floats.
