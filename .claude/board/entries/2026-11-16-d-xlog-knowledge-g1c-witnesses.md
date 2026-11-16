# D-XLOG-KNOWLEDGE-G1c: positive witnesses for the 15 empty-only reference agreements (2026-11-16)

**Status:** MEASURED, lab only. Plan `.claude/plans/xlog-knowledge-spine-v1.md` §4 G1.
Follows G1b (draft PR #2109), whose census gave 15 refused rules a naive/Datafrog
`agree` over empty answers only.

## Why they were empty

The census seeds only binary `u32` base relations. The four programs behind the
15 rules each read a unary or a ternary base. Those bases were empty on every
seed, so both evaluators answered the empty set. All 15 are satisfiable.

## What

- `witness(src, rows)` in `src/lib.rs`: the naive oracle and Datafrog over
  caller-supplied base rows (any arity), with no seeds and no cube.
- `tests/witness.rs`: the four xlog@656def9 literals, embedded verbatim and
  checked byte-identical against the source. Each program has a hand-derived
  fixture and expected set. Both evaluators must return exactly that set
  (`assert_eq!` on the sorted rows, not on a count), and every expected set is
  asserted non-empty. Each program has a negative control. Each program's census
  rungs and reasons are re-checked unchanged through `reconcile`.

## The 15 rules

| program (xlog@656def9 `tests/…`, literal) | head | census reason | minimal witness | expected rows |
|---|---|---|---|---|
| `test_external_consumer_surface_preservation.rs`, 1 | `tensor_support` | `committed_input` not binary | `committed_input = {1, 2}` | `{1}, {2}` |
| same | `__xlog_query_0` | reads refused `tensor_support` | same | `{1}, {2}` |
| `test_recursive_setop_profile.rs`, 0 | `support_2` | join key outside its input | commit `{(10,1,2),(11,2,3),(11,3,4)}`, len_2 `{5,6}`, body_0 `{(5,20,10),(6,30,20)}`, body_1 `{(5,20,11),(6,30,11)}` | `(20,1,3,5,10,1,2,11,2,3)`, `(30,1,4,6,20,1,3,11,3,4)` (the second only after a recursion round) |
| same | `usable` (base rule) | join key outside its input | same | `(10,1,2),(11,2,3),(11,3,4),(20,1,3),(30,1,4)` |
| same | `usable` (recursive rule) | join key outside its input | same | same head, same set |
| same | `__xlog_query_0` | reads refused `support_2` | same | = `support_2` |
| same | `__xlog_query_1` | reads refused `usable` | same | = `usable` |
| `test_sort_labels.rs`, 0 | `support_1` | Project over Project | rule `{7}`, commit `{(1,2,3),(4,5,6)}` | 4 rows: commit × {7} × commit |
| same | `__xlog_query_0` | reads refused `support_1` | same | = `support_1` |
| same | `usable` | reads refused `support_1` | same | `(1,2,3),(4,5,6)` |
| same | `__xlog_query_1` | reads refused `usable` | same | = `usable` |
| `test_widened_frontier_replay.rs`, 0 | `promoted` | `frontier_pred` not binary | frontier `{1,2,3}`, widened `{1}`, edge `{(1,2),(2,3),(3,4)}`, blocked `{3}` | `{1}` |
| same | `replay_reachable` (base rule) | reads refused `promoted` | same | `{1},{2},{3}` (stops before 4: not a frontier predicate) |
| same | `replay_reachable` (recursive rule) | reads refused `promoted` | same | same head, same set |
| same | `rollback_hit` | reads refused `replay_reachable` | same | `{3}` |

Independent agreement: naive = Datafrog = the hand-derived set for all 15
(rules sharing a head share its answer). None stays empty.

## Negative controls (committed tests)

- P1: without `committed_input(2)`, both heads are `{1}`.
- P2: without rule 6, exactly the two round-2 rows disappear.
- P3: a second rule id doubles `support_1` (8 rows) and leaves `usable` unchanged.
- P4: blocking 4 (never reached) leaves `rollback_hit` empty.
- `compare` reports a disagreement for a witness with one extra row.

## Disable runs (anchor asserted, restored from commit `017bfa4a7`)

- D1, naive oracle stops after one round: P2 witness, P2 control and P4 red (3/10).
- D2, the Datafrog join takes the wrong right-hand column: P2 ×2, P3 and P4 red (4/10).
- D3, `witness` gives Datafrog an empty store: every witness test red (5/10).

## Not changed

The cube evaluator, the census TSV and its rungs and reasons, mask-risc,
floats, aggregates, and W5/W7.

## W5/W7 probe nominations (proposal only)

1. `replay_reachable` (`test_widened_frontier_replay.rs`) is a W5 probe: a
   recursive rule with a known three-row answer and a stop condition (edge to a
   non-frontier node) that a wrong fixpoint would cross. It is refused today
   only through a unary base.
2. `support_2` / `usable` (`test_recursive_setop_profile.rs`): a mutually
   recursive pair with a ten-column head built from repeated variables, and one
   row that exists only after the second round. It belongs to the column-
   construction refusal cluster. It is a W7 probe only if W7's executor decision
   (mask fold over a helper relation, or plain data reads) covers column
   construction; otherwise it is a second W5 probe.

## Open

44 unavailable refused rules (floats, `is`, aggregates, ILP, non-integer facts)
are unchanged.
