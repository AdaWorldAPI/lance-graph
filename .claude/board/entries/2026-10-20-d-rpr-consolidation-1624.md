# D-RPR consolidation: #1624 folded into the merged #1622 plan (2026-10-20)

**Status:** DONE. **Basis:** VERIFIED-IN-CODE (plan diff) and MEASURED (the banked draft re-run on `main` at `4dbd085`).

Two sessions distilled the same research brief. #1622 merged first and is canonical for D-RST and D-RPR. #1624 had defined a second `recursive-perspective-rebinding-v1.md`, an entry of the same name, and status rows D-RPR-0..6 whose numbers meant different things. Those are withdrawn. Their content that #1622 lacked is now `recursive-perspective-rebinding-v1.md` §11, under the existing D-RPR ids.

## Folded in

- D-RPR-1 sub-cases 1a–1f, each with a kill condition and a disable run (§11.2). They were D-RPR-1..6 in #1624.
- Anti-vacuity rules for the chain (§11.3), including one measured failure: a `discover`-seeded first focus already held the first answer.
- Independent oracles (§11.4); the green → red → green disable protocol (§11.5); refusals by axis: version, focus, scope, ordering, thought identity, narrow frame, capacity (§11.6).
- Preparation and reuse costs already measured: break-even 76 / 11 / 1.8; CSR against tile; escape reporting cost; bytes; the zero-copy tension (§11.7).
- Two-variable and multiplicity limits (§11.8).

## Corrected

- WORLD, ANGLE and THOUGHT are metaphors, not types or registers. There are pointer notes in §1 and §3.
- Thought identity is #1605's `ThoughtCommitId` / `DurableRef`.
- #1624's claim that D-RST resolved to no plan predates #1622 and is withdrawn.
- D-MTP-1's `Semijoin` path is gather-shaped.
- The two meanings of `ThoughtMask` are recorded as OPEN.

## Preserved unchanged from #1624

The 19 artifact back-links, 11 plan forward-links and entry `2026-10-20-prs-1601-1617-plan-links.md`, rebased onto `main`.

## Draft probe

`.claude/harvest/d-rpr-1-chain-probe-draft/` (patch + README). 5 of 6 tests pass on `main`. The failure is the first-focus anti-vacuity case. It is not merged because D-RPR-1 is not authorized.
