# D-XLOG-COV-2: census over xlog `007bb19` with one-column bases (2026-11-16)

**Status:** MEASURED, lab only. Follows D-XLOG-W5-B (lance-graph `SetExpr`,
`Cube::reach`) and D-XLOG-W5-C (AdaWorldAPI/xlog#9, merged as `007bb19`).

## What

- `xlog-coverage-probe` pins the four xlog crates to `007bb19`.
- A base used only at arity 1 whose facts fit `u32` is `BaseRows::Unary`:
  seeded per seed (stated facts, else `0..6` / `0..4` / 1..=12 random values),
  read by the naive oracle as one-column rows, handed to the cube with
  `put_set`. One-column `u32` facts of derived predicates also go to `put_set`.
- XLOG's reference evaluator still reads binary bases only; one-column bases
  are unavailable to it (`xlog_ref` unavailable for those rules).
- `RuleRecord::route` and a 15th TSV column `route` (`set`, `reach`, empty)
  from `Report::routed`; a routed head is not `host_loop`.

## Result

`census-007bb19.tsv` (replaces `census-656def9.tsv`, kept in git history):

| rung | 656def9 | 007bb19 |
|---|---|---|
| oracle-agrees | 87 | 102 |
| refused | 94 | 79 |
| refused-on-data | 49 | 49 |
| oracle-disagrees / lowered-unverified | 0 / 0 | 0 / 0 |
| native-tested | 43 | 58 |

Row-by-row against the previous census with `route` stripped: 438 rows, 15
changed, and they are exactly the 15 routed rows (13 `set`, 2 `reach`), each
refused → oracle-agrees. The other 423 rows are byte-identical. Programs:
`test_external_consumer_surface_preservation` literal 1 (2 rules),
`test_same_head_union_batching` literal 0 (9), `test_widened_frontier_replay`
literal 0 (4). Datafrog agrees on all 102; the reference column shows 0
naive/Datafrog disagreements, refused rules 26 agree / 9 empty-only /
44 unavailable.

## Falsifiers

- `put_set` call removed from `run_cube`: `p1_agrees_on_the_cube_as_a_set`
  and `p4_agrees_on_the_cube_with_reach` fail (cube answers empty, oracle
  does not). Restored from the commit.
- Unit tests pin the classification: negative constants and mixed arity are
  not `Unary`.

## Open

- W7: the remaining refusals that need column construction (`support_2`,
  `usable`).
- `reach` on a tiled cube is refused; it needs its own probe.
