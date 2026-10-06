# 2026-10-06 — Difference and plateau as an expression (Round 5)

## MEASURED

`crates/lance-graph-quack/tests/diff_plateau.rs`, 3 tests, 5 disable runs
red.

Two resident planes `a`, `b` over one row space. The diff is `a XOR b`,
its size is `POPCOUNT`, and the plateau is `NOT ANY`. Four paths, each
checked against a bit-at-a-time row oracle on eleven cases:

- empty diff (1 row and ragged)
- one bit at row 0, at the last row of word 0 and the first of word 1
- one bit at the last row of tile 0 and the first of tile 1 (16384 rows)
- one bit at the last row of a ragged `3 × 16384 + 77`
- sparse scattered (every 997th row)
- dense

| path | ops | lowering | scratch slots | slot words written | heap | population bytes |
|---|---|---|---|---|---|---|
| A materialized `Vec<u64>` diff | — | — | — | — | — | 6160 (770 words) |
| B mask-risc `Xor` → Count / Any | 1 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |
| C quack `lower`, `(a∧¬b)∨(¬a∧b)` | 5 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |
| D quack `lower_fused`, same filter | 1 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |

The can-fire twin is the same count with a `Range` pred stacked on the
`Xor` slot. It runs `Tiled` with 2 slots and 512 words written, and still
matches the oracle, so the zeros above are measurements.

Disable runs, each red: `Xor` replaced by `And`; plateau read as `ANY`;
quack diff spelled with `AND` instead of `OR`; reference path ANDing words;
oracle limited to the first word.

## FINDING

- **Diff count and plateau are each one ternlog fold over the two resident
  planes.** No diff population is written, no scratch slot is provisioned,
  and execution allocates nothing.
- **Quack's AND/OR/NOT spelling reaches the same single table** as a direct
  `Xor`, under plain `lower` too. The program collapse sees through it.

## OPEN

- Two planes over one row space only. A diff between two Lance versions,
  or between a horizon and `delta.resulting`, would first need both states
  readable as resident planes over the same rows.
- A diff over a third plane (diff restricted to a gate) would still fold
  (≤ 3 planes). A diff that needs a `Pred` (a range, a lane compare) runs
  `Tiled` and writes slot words, per the twin.
