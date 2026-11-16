# D-XLOG-W5-A: can the existing cube admit `replay_reachable`? (2026-11-16)

**Status:** VERIFIED-IN-CODE, read-only. No code changed, nothing admitted.
Plan `.claude/plans/xlog-wiring-to-hot-path-v1.md` W5/W6. Acceptance target:
the G1c witness (`crates/xlog-coverage-probe/tests/witness.rs`, `p4_replay_witness`).

## The rule and its known answer

```
promoted(P)         :- frontier_pred(P), widened_pred(P).
replay_reachable(P) :- promoted(P).
replay_reachable(Q) :- replay_reachable(P), frontier_edge(P, Q), frontier_pred(Q).
rollback_hit(P)     :- replay_reachable(P), blocked_pred(P).
```

With frontier `{1,2,3}`, widened `{1}`, edge `1->2->3->4` and blocked `{3}`,
naive = Datafrog = `promoted {1}`, `replay_reachable {1,2,3}`,
`rollback_hit {3}`. The edge `3->4` exists but 4 is not on the frontier. A
traversal that applies the frontier filter to the wrong column, or not at
all, returns 4, so the witness can fail.

In set form: `replay = S ∪ reach(S)`, where `S = frontier ∧ widened`, each
step is `next = image(delta, edge) ∧ frontier`, and the recursion is the
semi-naive `delta' = next andnot full`. Every step is a mask operation over
the tile's positions.

## Verdict: not admissible with the operations the cube admits today

Four blockers, each read from source.

| # | blocker | where |
|---|---|---|
| B1 | **No unary relation.** Cube relations are binary `(u32, u32)` lanes; `Cube::new` and `add_relation` take `&[(u32, u32)]`. Five of the seven predicates are unary. The fork refuses at the first one: `frontier_pred` "is not a binary relation in the cube". | `lance-graph-xlog/src/cube.rs:350`, `:430`; xlog `xlog-cube/src/lib.rs:109`, `:363` |
| B2 | **No set filter on a column.** `frontier_edge(P,Q), frontier_pred(Q)` restricts column 1 of a binary relation to a set. `Cube::bind` binds one value (`bind(.., col, value: u32)`), and a join side must be a scan or a bind: a join over a join is `NotASide`. So the filter cannot be composed from views. | `cube.rs:479-504`, `:531-536` |
| B3 | **The one recursive terminal is the wrong shape.** `Cube::closure` and `closure_wide` compute `edge⁺` of one *base* `RelId`, all pairs, with no seed and no filter (`closure`: ≤ 64 positions). The rule needs reachability from a seed set, with the step filtered by a target set. Feeding `closure` a filtered edge relation would first need `add_relation` over host-built rows, which is the materializing shortcut W6 is meant to remove. | `cube.rs:845-889`; `wide.rs:395-415` |
| B4 | **Recursion has no cube owner for this shape.** The fork's `recursive_scc` loops on the host over owned `BTreeSet` rows, re-adds `delta` and `full` as cube relations every round, and refuses unless each is binary (`ok_or(Refusal::NotBinary(..))`). That is the D-STAUNEN-5 shortcut named in W6, not a mask-native fixpoint. | xlog `xlog-cube/src/lib.rs:494-569` |

## The constraints any admission must keep

- **`NotASide`.** Recursion cannot be built by stacking views
  (`replay ⨝ edge` over a join is refused). The loop must be inside a terminal,
  as `closure`'s is.
- **No computed-mask reuse across programs, no Gather over a produced mask.**
  Quack allows `Gather` only over a resident plane, never over "a mask another
  `Program`'s `Keep` terminal just computed". A per-round "keep replay, then
  gather edges through it" pipeline is exactly the forbidden shape. Each
  round's delta must stay inside one terminal, as `closure`'s
  `mask_andnot` / `mask_or_assign` do (`lance-graph-quack/src/lib.rs:153`, `:377-386`).
- **Base sets are resident; derived sets are not.** `frontier_pred` and
  `widened_pred` are base data and can be loaded once, like lanes. `promoted`
  is computed. If it is handed to a second terminal as a mask, that is a
  produced mask crossing a terminal boundary.

## What already exists and fits

`wide::run_fixpoint` (private) is the seeded semi-naive kernel this rule needs:
from a seeded `full`/`delta` it runs `delta' = (delta ∘ edge) andnot full;
full |= delta'` with a round bound (`FixpointBound`). `closure_wide` seeds it
with one source's row; seeding it with `S` instead gives `S ∪ reach(S)`.
`position_mask` (private) builds the seed. The one missing operation is the
target filter, `next &= frontier`, before the `andnot` (`wide.rs:295-337`, `:532`).

## Smallest admission shape (proposal for W5-B, not built)

One terminal, for example `Cube::reach(table, rel, seed, target)`:

- unary inputs given as **expressions over resident unary sets** (here
  `seed = frontier ∧ widened`, `target = frontier`), evaluated inside the
  terminal, so no produced mask crosses a boundary;
- unary sets loaded at construction as position masks beside the binary lanes
  (that is the B1 decision: a new resident kind, not a binary encoding);
- the loop is `run_fixpoint` with the target filter; the result is a position
  mask (`rollback_hit = result ∧ blocked` stays a mask `and`);
- the frame is re-checked as every terminal does, with the existing round bound.

Rejected alternative: encode each unary set as a diagonal binary relation
`(x, x)`. That makes the two non-recursive joins expressible with existing
views, but leaves B3 and B4 untouched and doubles the lanes. A host encoding
trick, not an owner for the recursion. (Not measured.)

## Acceptance for W5-B

1. `p4_replay_witness`'s fixture through the new terminal gives exactly
   `{1,2,3}` and `{3}`. Node 4 is absent.
2. Disable: drop the target filter → 4 appears, red. Filter the source column
   instead of the target → red on a fixture where they differ.
3. Differential: `reach(S)` equals the row-OR over `S` of `closure` on the
   target-filtered edges, which keeps the existing `closure` as the oracle
   inside the cube.
4. Composing the expressions runs no program (`CubeStats`), as for views.

## Open decisions (operator)

- D1: is a unary resident set a cube kind (B1), or does W5 stay binary-only?
- D2: may a terminal take an expression over resident sets as input (the
  route that keeps "no computed-mask reuse"), or must seeds be named base
  relations only?
- D3: the recursion owner (W6): this terminal generalizes `closure` and starts
  replacing `xlog-cube`'s host loop for unary reachability. It does not
  replace it for n-ary SCCs (`support_2`/`usable` stays outside).
