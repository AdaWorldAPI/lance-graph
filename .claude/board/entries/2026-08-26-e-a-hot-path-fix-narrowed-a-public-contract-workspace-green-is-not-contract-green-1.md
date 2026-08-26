# E-A-HOT-PATH-FIX-NARROWED-A-PUBLIC-CONTRACT-WORKSPACE-GREEN-IS-NOT-CONTRACT-GREEN-1

**Date:** 2026-08-26
**Status:** FINDING (measured)
**Issue:** `ISS-MUL-GATE-OUTCOME-COUPLED-TO-PRODUCER-GROUND`
**Plan:** `.claude/plans/mul-gate-outcome-vs-ground-v1.md`
**Arc:** #1045 (de-stringing) → #1052 (ordinal alignment) → this

---

## The finding

**A change can be correct at the layer it was reasoned about and still narrow a
contract one layer up.** PR #1045 removed `reason: String` from
`mul::GateDecision::{Hold, Block}` and replaced it with the typed pair
`{ texture: TrustTexture, flow: FlowState }`. Judged as a hot-path fix it is
right and stays: `gate_decision_i4` had been allocating five heap strings per
call on an i4-quantised path, the pair is exactly what the evaluator already
computed, and the type became `Copy` + SIMD-packable.

Judged as a *public contract*, the same edit promoted **MUL-specific ground into
the universal gate outcome**. `GateDecision` is the return type of
`MulProvider::gate_check` — anyone's gate. After #1045 there is no way to say
`Block` without also claiming a trust texture and a flow state. A consent veto
is not a trust texture. An evidence contradiction is not a flow state.

The generalizable shape:

```text
OUTCOME  (universal, transported, ordinal)   ≠   GROUND (producer-owned)
Flow / Hold / Block                              why THIS producer said so
```

The gate transports the *what*. The *why* stays with the producer and reaches
the record through witness/alpha. A public verdict type that carries one
producer's vocabulary has made every other producer lie or fail to compile.

## The second finding: workspace-green is not contract-green

#1045 passed every gate this repo has — clippy `-D warnings`, member-tests, the
full contract suite — because every *in-workspace* caller matches with `{ .. }`
or constructs through the evaluator. The consumer that broke is
`ada-rs/src/contract_impls.rs:72`, which implements `MulProvider` and builds
`Block { reason: format!(…) }`. It binds `lance-graph-contract` by **git URL
with no branch or rev** and carries **no `Cargo.lock`**, so the break is live on
a fresh build — not deferred to some future `cargo update`.

Hence the rule the plan carries as its durable half:

> **A source-breaking change in `lance-graph-contract` is not verified until
> known unbound-git consumers BUILD against the proposed head.** Not grep. Not
> "all workspace callers use `{ .. }`."

Grep cannot see a consumer that is not in the tree, and a green workspace is
evidence about the workspace only.

## Measured detail (2026-08-26, `743ce64`)

- **The public surface is two traits, not one.** `MulProvider::gate_check`
  (`mul.rs:245`) and `PlannerContract::gate_check` (`plan.rs:160`) both return
  `mul::GateDecision` (`plan.rs:7`). An arc touching only the first leaves half
  the surface coupled.
- **`MulGateDecision` is already taken**, by
  `lance-graph-planner/src/mul/gate.rs:19` (`Proceed`/`Sandbox`/`Compass`), and
  its own doc-comment records that it was renamed from `GateDecision` in M15 *to
  escape this exact collision*. The obvious name for a new MUL-ground type would
  re-collide a name that already fled once. `.claude/v3/COMPONENT-MAP.md` lists
  GATE-1 (two `GateDecision` types) as a known key risk, and rules
  `mul::GateDecision` **REUSE — the LIVE kanban gate** consumed by
  `KanbanColumn::advance_on_gate`.
- **`TrustTexture` exists twice with different variants** — contract:
  `Calibrated/Overconfident/Underconfident/Uncertain`; planner:
  `Murky/Dissonant/Fuzzy/…`. Independent corroboration that trust texture is a
  producer's ground, not a universal field.
- **The in-tree excluded crates are protected, and are not coverage.**
  `lance-graph-ogar`, `symbiont`, `cognitive-stack` reach the contract
  transitively through OGAR but each `[patch]`es it onto the working-tree path
  copy. They cannot break on a main-branch change — and equally cannot
  demonstrate that main is safe.

## The stopgap that was NOT pushed

The obvious consumer-side patch — reconstruct `texture`/`flow` from the
assessment at ada-rs's consent veto — compiles. It was written, it built
clean, and it was reverted unpushed. It would make a consent veto *claim* a
trust texture and flow state the producer never asserted: provenance
fabricated to satisfy a type. **In this architecture a red compiler is
preferable to a green lie**, because the lie survives into the witness record
and the red build does not.

## What is NOT concluded

- #1045 is not reverted, and its typed heap-free evaluator is not in question.
- No `enum GateGround { TrustFlow, Consent, MedicalEvidence, … }`. That moves
  the coupling outward and makes `lance-graph-contract` own everybody's
  epistemology. The producer already holds its ground; it stays there.
- `collapse_gate::GateDecision` (aligned in #1052) is a second public type with
  its own consumers; whether it becomes `GateLevel` or merely documents one is
  deferred as OQ-GATE-1, not folded in.

Falsifiers F-GATE-1..5 and deliverables D-GATE-1..6 live in the plan. The
load-bearing pair is F-GATE-2/F-GATE-3: a consent veto and an evidence
contradiction must each emit `Block` without inventing MUL provenance. Both go
red under #1045's shape — that is what makes them tests rather than assertions.
