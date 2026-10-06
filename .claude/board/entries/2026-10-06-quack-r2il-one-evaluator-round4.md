# 2026-10-06 — Quack and R2IL on one evaluator: the DAV candidate filter (Round 4)

## MEASURED

`crates/r2il-mask-abi-probe/tests/row_bridge.rs`, § "Round 4" (4 new tests;
the census test extended). `FoldDialect` gains four lowering arms. Each maps
onto a `Pred`/`MaskOp`/`Terminal` mask-risc already had, and none adds a
kernel:

- `LOAD` space 3: a resident mask plane as an `Addr::Plane`, used as
  `Operand::Plane`
- `IntNot` (R2IL ordinal 23) → `MaskOp::Not`
- `RANGE` (fold band, arity 0, bounds as immediates) → `Pred::Range`
- `KEEP` → `Terminal::Keep` into a caller-lent sink

The DAV candidate filter `is_null(valid) AND lo <= row < hi` is spelled
three ways: `quack::lower`, R2IL bytes NOT-first, R2IL bytes RANGE-first.
The kept mask is bit-identical to a row oracle at n ∈ {21, 64, 65, 130,
256, 1000}, and so is the count. All paths execute through
`mask_risc::execute_into`.

Physical work at n = 1000 (16 words):

| path | passes | derived words | lowering |
|---|---|---|---|
| `quack::lower` | 3 (`Not`, gated `Range`, trailing `And`) | 48 | Tiled |
| R2IL NOT-first | 2 (`Not`, `Range` gated under it) | 32 | Tiled |
| R2IL RANGE-first | 3 (`Range`, `Not`, `And`) | 48 | Tiled |

The COUNT forms are all `Tiled` too. Positive-polarity control
(`plane AND range`, COUNT): quack and R2IL both lower to ONE `Range` gated
under the resident plane, which becomes `Lowering::Range`. That path writes
no derived words, and both match the oracle.

KEEP-path dialect allocation is 16,648 B at both 16,384 and 1,048,576 rows.

Disable runs, red: the plane-AND peephole replaced by a real `And`;
`IntNot` without the complement; `RANGE` losing its upper bound; `KEEP`
writing nowhere; `KEEP` pushing a status scalar; a population-sized buffer
allocated per fold.

The peephole was made commutative to fold the RANGE-first order. mask-risc
refused the result with `ScratchReadBeforeWrite`: an earlier pass would be
gated under a later slot.

## FINDING

- **One evaluator holds.** The DAV filter needs no second population
  evaluator, only lowering arms in the R2IL dialect.
- **Pass count is byte-order dependent on the R2IL side; quack's is not.**
  Folding the RANGE-first order needs the `Not` hoisted ahead of the
  `Range` (a scheduling rewrite), not a wider peephole.
- **The zero-write fold is blocked by polarity, in both frontends.**
  `fused_terminal` folds a `Range` gated under a resident plane. The DAV
  candidates need the complement of that plane, which exists only once
  written into a scratch slot. So the missing algebra is narrow: a
  `Range` gate with negative plane polarity (or a resident "unobserved"
  plane).
- **The shared middle is Address / mask expression / fold / Scalar /
  Terminal.** The dialect's stack holds only `Scalar`, `Slot` and
  `Address` (lane, `VIA`, plane).

## OPEN

- quack's trailing `And` after a gated conjunct is now pinned in two
  programs (the join and the DAV filter).
