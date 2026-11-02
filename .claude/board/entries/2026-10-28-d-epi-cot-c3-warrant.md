# D-EPI-COT-C3 (first slice): tactics run only on grounded, Epi5-eligible markers

**Status:** TEST-PINNED. Code:
- `crates/lance-graph-planner/src/nars/warrant.rs`
- `epistemic_state5::RecipeRule`
- `certification::CertificationModel::measured_only`

Plan: `docs/research/epistemic-cot/c3-plan.md` §8.

## Decision

- **SCOPE:** a planner tactic (RCR 4, TR 6, CAS 8) runs only on `Marker`s. A marker is three things together:
  - the ledger's pooled truth and roots;
  - bits 59..63 projected through `Epi5Declarations`;
  - a thought slot and generation.

  Each premise must satisfy the recipe's `RecipeRule`. The rule's requires/forbids are policy pins: Associated for 4/6/7/8, and no intermediate-known for RCR.
- **Judgments** carry the tactic's truth, the proof's roots, the thought, and `Unknown × Open`. A term-logic step earns no certification. The only promotion is `pearl::reason` + `revise` over executed trial arms (pinned in the same test).
- **`measured_only`:** an unmeasured outcome is excluded, not read as negative. That includes trial arms, which `causes` reads without the universe.

## Evidence

- 9 new tests. 9 disables, each red.
- One disable (trial arms unrestricted) was green on the first fixture: every control unit was measured. An unmeasured-control fixture was added before recording.
- Planner lib 485 pass; contract lib 1561 pass.
- Clippy `--lib --tests` reports the same 8 sites as main `694b243bd`.

## Open

- Effect size, sample size, uncertainty and multiple comparisons are still unreported by `CertificationModel`.
- Missing-at-random is assumed.
- ASC and CR are not run through `fire`.
- The lab affordance law still has its own rule type.
- C3b and C3c wait on the bits 50..52 order decision and the TFLite location.

## Addendum, same day: ledger → `pearl::hydrate`

- `nars::mediator::hydrate_from_ledger` connects the ledger's grounded deductions of `A → Y` through `B` to the unchanged `pearl::hydrate`. It returns derivations, roots, the `Hydration` result, the edge and the mediated-path sensitivity.
- The ledger never creates a chain, and a path the chain does not witness leaves the edge unchanged. A chain-only path has no ledger receipt and is refused.
- `Law::Deduction` admits `Impl` in an explicit step: never closure, and `Rel` never composes.
- 5 tests; 5 disables red. The first grounding disable did not bind (no proofs exist without evidence); it was replaced before recording.
- Maturation order for new declared readings is `7 = Staunen … 0 = Wisdom`. The TFLite kernels are in ndarray.
