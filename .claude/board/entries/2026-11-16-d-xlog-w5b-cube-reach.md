# D-XLOG-W5-B: stored one-column sets and `Cube::reach` (2026-11-16)

**Status:** TEST-PINNED, lab. Decisions D1–D3 taken as (a) (operator, this
date), from the W5-A analysis (`2026-11-16-d-xlog-w5a-replay-reachable-admission.md`).

## What

`crates/lance-graph-xlog/src/cube.rs`:

- `Cube::add_set(table, ids) -> SetId` (D1). A one-column set is stored as
  one bit per local position, placed under the cube's frame like a relation's
  lanes. A tiled cube is refused.
- `SetExpr` (D2). Leaves are stored sets; the operators are and, or and
  and-not only, with no constants and no complement. It is plain data, and
  the terminal evaluates it, so no computed mask crosses a terminal.
- `Cube::reach(table, rel, seed, target)` (D3). Seeded semi-naive over
  position masks: `delta' = (OR of the delta's edge rows ∧ target) andnot
  full`. It returns `seed ∪ reach(seed)`. The frame is re-checked, and the
  round bound is the one `closure` uses (`FixpointBound`).
- `SetExpr::Reach { rel, seed, target }`: `reach` as an expression node, so a
  reachable set composes inside one terminal (`rollback = Reach(..) ∧
  blocked`), and a recursive head is never stored and handed back as a mask.
- `Cube::set_mask(table, expr)`: the read terminal for an expression. It
  re-checks the frame and returns an answer to download, never an input.
- `CubeRefusal::UnknownSet`. No crate matches exhaustively on `CubeRefusal`
  (checked in this tree and the xlog fork).

## Tests (`tests/reach.rs`, 8; the eighth is `reach_composes_as_an_expression`)

- G1c witness: `promoted {1}`, `replay_reachable {1,2,3}` in 3 rounds,
  `rollback_hit {3}`. Blocking 4 leaves rollback empty.
- Target-column fixture: every edge's source is on the frontier and 4 is not,
  so a filter applied to the source would admit 4.
- Differential over 40 random graphs (n ≤ 64): a BFS that shares no code with
  the cube, and `S ∪ row_OR(closure(E_F), S)` through the cube's own
  `closure`. Anti-vacuity: at least 20 graphs grow past their seed.
- Zero programs: `CubeStats` is unchanged by composing expressions and by
  `reach`.
- Refusals: unknown set (leaf and nested), unknown relation, a table that
  never issued the frame (`Map(Stale)`), an id outside the map. An advanced
  table gives the same answer.

## Disable runs (anchor asserted, restored from `b2d13c32d`)

| disable | red |
|---|---|
| D1 target filter dropped | 4 of 7 |
| D2 filter on the source column | 4 of 7 |
| D3 seeds not included | 4 of 7 |
| D4 `And` evaluated as `Or` | 1 of 7 (G1c) |

`cargo test -p lance-graph-xlog`: 12 suites, all green.
`cargo clippy -p lance-graph-xlog --all-targets -D warnings`: clean.

## Not done

- **W5-C, the fork's routing:** AdaWorldAPI/xlog#9 (merge after this PR).
  It routes one-column set SCCs and the one-step reach shape, records the
  path in `Report::routed`, and checks against a BFS on 30 inputs plus the
  G1c witness. The census bump that makes the coverage probe supply
  one-column bases is the step after. `support_2`/`usable` stays out of scope.
- After the `SetExpr::Reach` refactor, the target-filter disable was re-run:
  5 of 8 tests red.
- `reach` does not use `wide::run_fixpoint` (its rows are `WideRows`, and the
  filter would have to be threaded through `closure_wide` and `refresh`). It
  mirrors `closure`'s loop and bound instead. A tiled `reach` would need that
  path.
