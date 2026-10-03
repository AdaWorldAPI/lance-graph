# Directory desired-state simulation over SoA + Quack (2026-10-03)

**Status:** MEASURED. The `crates/lance-graph-dir-sim` crate is excluded from
the workspace because it path-depends on the OGAR sibling, the same shape as
`lance-graph-report-ogar`. It consumes the semantic vocabulary in OGAR
`ogar-dir-sim` (OGAR PR #314).

A directory version is a shared `Arc<Snapshot>` of SoA lanes plus a
delta-sized overlay. Invariants are Quack programs:

- **Edge integrity** is an anti-join: `negate(Semijoin)`, lowered to two
  `MaskOp::Gather` over the node-kind planes.
- **SMTP / UPN uniqueness** is `GroupReduce Count` keyed on a normalized-key
  dictionary id, folded over base + overlay. The overlay rows are admitted by a
  `Semijoin` against the active-user plane.

Population rules combine two folded `GROUP BY` sinks at the consumer. They
never pass one program's mask to another program's `Semijoin`.

| quantity (one membership mutation, `tests/alloc.rs`) | 1k users | 100k users |
|---|---|---|
| bytes allocated by `simulate` | 853 | 853 |
| bytes allocated by same-root `diff` | 1,208 | 1,208 |

The suite has 23 tests. Ten guards were disabled one at a time and each turned
its test red.

Open for the operator:

- **`VersionedGraph` (`u32` node ids, additions-only diff).** It cannot
  persist 128-bit directory identity. The choice is to widen it upstream or
  use a dedicated directory dataset.
- **Row-population masks.** There is still no shared row-population mask type
  in the contract; this crate uses mask-risc bitmaps directly.
