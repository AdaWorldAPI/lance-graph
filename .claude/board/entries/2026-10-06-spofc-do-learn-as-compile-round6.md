# 2026-10-06 — SPOFC → do → learn as a compile (Round 6)

## MEASURED

`crates/lance-graph-quack/tests/spofc_cycle.rs`, 6 tests, 7 disable runs
red. No primitive, write path or learn op added; `revision.rs` untouched.

One cycle over a resident observation population (`S_s`, `O_o` one-hot,
`OLD`, `NEW` disjoint; 49 229 rows, ragged):

| step | lowers to |
|---|---|
| SPOFC | per candidate `(s,o)`: `\|X\| = COUNT(S_s ∧ W)`, `\|X∧Y\| = COUNT(S_s ∧ O_o ∧ W)` → `arm_to_truth_u8` → scalar compare → one claim bit |
| do | the owner writes the `NEW` rows into the planes before the cycle; the evaluator only reads |
| learn | `EncounterEvidence` → `GadamerRevision::revise` → verdict → `delta.resulting` |

- 66 reads per cycle (2 windows × (1 window size + 16 × 2 counts)): 64 are
  `Ternlog` folds over ≤ 3 planes, 2 are a `Count` straight over one plane.
  0 slot words written, 0 heap during execution.
- Compiled prior, encounter and revision equal a bit-at-a-time reference
  at 49 229, 16 385 and 4 001 rows.
- The fixture reaches `HorizonFusion` / `IncreaseEligible`; the
  contradicted claim stays in `unresolved_tension`.
- Replaying the same encounter from `delta.resulting` gives
  `ContradictionPreserved`, no new root; from the prior (revision bypassed)
  it gives `IncreaseEligible` again.
- Revision's kind equals a decision table over 7 `Any` folds (each over
  ≤ 2 masks) plus `closes_cycle`, on all 32 768 cases of a 2-bit universe.
- `delta.resulting` equals three `TernlogKeep` outputs (roots, inherited,
  tension) plus `proposed_claims` passed through.

Disable runs, each red: fusion condition on `withdrawn` instead of
`proposed`; `same_projection` via one-sided `AndNot`; roots kept without
`\ ancestry`; contradiction threshold off; replay from the prior; window
plane dropped from a read; `recycles` with operands swapped.

## FINDING

- **One SPOFC → do → learn cycle compiles onto existing pieces.**
  Evidence = fused count folds; truth = `arm_to_truth_u8`; claim =
  scalar compare; learning = `GadamerRevision`. No missing primitive.
- **Revision's decision is LAW over folds.** Its mask questions are fused
  `Any` folds; its write is three `Keep` masks of claim width, not row
  width. `introduced ∪ preserved = proposed`, so that test is one fold.
- **`RevisionKind::Suspended` is unreachable.** Reaching it needs no
  resistance and an empty `revised_claims`, which makes `same_projection`
  true, so `Echo` (or `ClosedCycle`) fires
  first. Unreached in the exhaustive 2-bit universe.
- **The `\ ancestry` in the roots update binds only when ancestry ≠ the
  prior's own roots.** In this cycle they are equal and the `AndNot` is
  redundant; the exhaustive fixture needed disjoint roots to make it bind.

## OPEN

- Cost: one population pass per count, 2 per candidate per window. A
  grouped `Pair` count is one pass but writes a K-slot result.
- SPOFC f/c does not reach `GadamerRevision` (mask-only); f/c survive only
  through the claim threshold. Where NARS f/c revision meets horizon
  revision is not decided here.
- `RevisionDelta` carries derived masks (preserved / introduced /
  withdrawn / revised) and a `prior` clone that no replay reads.
- 0..63 orchestration: not touched.
