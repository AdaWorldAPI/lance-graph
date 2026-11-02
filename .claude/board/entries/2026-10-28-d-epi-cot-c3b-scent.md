# D-EPI-COT-C3B: scent ranks candidate mediators by testability, not raw uncertainty

**Status:** TEST-PINNED (fixture world). Code: `crates/lance-graph-planner/src/nars/scent.rs`. Doc: `docs/research/epistemic-cot/c3b-scenting.md`.

## Decision

- `scent` ranks candidate mediators of an `A → Y` edge. Testable leads come first: one link is bound in the sealed chain and grounded in the ledger, and the other link is missing. Within a status, leads are ordered by uncertainty (binary entropy of the NARS expectation).
- It reads only. It writes no edge, truth or certification, and it stores no state.
- A settled lead reopens as contested on independent contradiction. The edge keeps its certification.

## Evidence

- 6 tests; 4 disables, each red.
- On the fixture, scent hydrates the true mediator in 1 investigation. Uncertainty-only ordering takes 4, because all four candidates have uncertainty 1.0.

## Open

- One world, one seed; extend STAUNEN-0 with this ranking.
- No budget, habituation or learning-progress terms.
- Maturation bits 50..52 are unwritten (order for new readings: `7 = Staunen … 0 = Wisdom`).
