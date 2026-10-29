# D-EPI-COT C3b — scenting candidate mediators

Status: TEST-PINNED on a fixture world. Code:
`crates/lance-graph-planner/src/nars/scent.rs`. Stacked on #1840. Independent
of #1841.

## Question

An edge at `IndirectUnknown × Causes` (raw5 22) is certified, but its
mechanism is open. Which candidate mediator `B` should be investigated next?

## Answer: `nars::scent::scent`

`scent` reads the #1840 ledger and the sealed chain and ranks the
candidates. It writes no edge, truth value or certification, and it stores
nothing between calls. Only `pearl::hydrate` moves the topology.

**Lead status.** Each candidate gets one of four statuses. A link "holds"
when it is bound in the sealed chain and its pooled frequency is at least
0.5 (policy pin).

| status | when |
|---|---|
| `Testable` | one link is missing from the chain, and the other is in the chain and grounded in the ledger. One observation decides whether `B` hydrates. |
| `Contested` | both links are bound, but one link no longer holds. |
| `Open` | neither link is anchored, or both are missing. |
| `Settled` | both links are bound and both hold. |

**Order.** Leads sort by status in the order above. Within a status, the
link to investigate with the highest uncertainty comes first. Uncertainty is
the binary entropy of the NARS expectation, or 1 for a link with no
evidence. Ties go to the lower mediator id.

**Why uncertainty alone is not enough.** Every candidate with no evidence
scores the maximum uncertainty. D-CE64-STAUNEN-0 measured the same failure
for raw surprise: 0.649 resolved, against 0.964 for round-robin.

## Falsifiers

| amendment item | test | disable that turns it red |
|---|---|---|
| S4: a productive lead beats equally uncertain noise | `s4_a_testable_lead_outranks_equally_uncertain_noise`: scent hydrates in 1 investigation, uncertainty-only in 4 | order by uncertainty only |
| no grounded mechanism, no test | `a_chain_link_without_evidence_is_not_an_anchor` | an anchor that needs only the chain binding |
| S1/S8: receptivity survives maturity | `s1_a_settled_lead_reopens_on_independent_contradiction`: the edge stays `IndirectKnown × Causes` | `Settled` ignores the evidence |
| S3: replay is not evidence | `s3_replay_does_not_change_the_scent` | — (the ledger's root dedup) |
| S7: compute does not mature anything | `s7_repeated_scenting_advances_nothing` | — (stateless by construction) |
| S2: no certification change | structural: `scent` takes the edge by value and returns only leads | — |

## What this does not show

- **The 1-vs-4 result comes from one hand-built world** with one true
  mediator and three noise candidates. It shows the ranking can separate a
  testable lead from noise. It does not measure performance across world
  shapes or seeds. The STAUNEN-0 probe, extended with this ranking and a
  second world, is the place for that measurement.
- **Maturation (bits 50..52) is not used.** The order for new declared
  readings is `7 = Staunen … 0 = Wisdom`. Until the reading, provenance,
  migration and the `learn` collision are settled, nothing writes those bits.
- **No learning.** C3c (LUT9 ranking) is separate, after the EXO-152 owner
  check.
- **No budget, habituation or learning-progress terms yet.** STAUNEN-0
  showed these matter under scarcity.
