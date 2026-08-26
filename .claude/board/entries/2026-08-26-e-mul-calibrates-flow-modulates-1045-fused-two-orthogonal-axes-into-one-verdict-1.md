# E-MUL-CALIBRATES-FLOW-MODULATES-1045-FUSED-TWO-ORTHOGONAL-AXES-INTO-ONE-VERDICT-1

**Date:** 2026-08-26
**Status:** FINDING (measured) — storno-corrects the same day's
`E-A-HOT-PATH-FIX-NARROWED-A-PUBLIC-CONTRACT-…-1` thesis
**Plan:** `.claude/plans/mul-calibration-not-verdict-v1.md`
**Arc:** #1045 → #1052 → #1054 → this

---

## The finding

**MUL calibrates; flow modulates; the gate adjudicates. #1045 fused the first
two into the third's payload.**

`GateDecision::Hold { texture: TrustTexture, flow: FlowState }` asserts
structurally that a hold *has a ground consisting of trust texture plus flow
state*. Measured, those are two independent coordinates:

| axis | question | measured consumer |
|---|---|---|
| `TrustTexture` + `DkPosition` | how trustworthy is my own uncertainty estimate? | the gate |
| `FlowState` (Csikszentmihalyi) | does the current cognitive regime FIT? | **thinking-style adaptation** — `planner/thinking/style.rs:272-275`, `FlowState → StyleFamily` |
| `allostatic_load` | recovery / stability | homeostatic control |

Flow is no more the provenance of a MUL hold than temperature is the provenance
of a blood-pressure reading. The distinction that settles it:

```text
Gate              = may this action / transition proceed?
Flow/Homeostasis  = should cognition change its way of attending/thinking?
```

`Boredom / Flow / Anxiety` are not three small deciders and not a homunculus —
they are regions of a control field measuring how well the current thinking
style carries the current attention field, driving *parameter changes to the
thinking process*, never decisions about content.

## Why the payload is measurably a second projection

`MulAssessment` already carries the axes apart (`trust`, `dk_position`,
`homeostasis{flow_state, allostatic_load}`, `complexity_mapped`,
`free_will_modifier`), and the trait boundary is
`gate_check(&self, assessment: &MulAssessment) -> GateDecision` — **the caller
passes the assessment, so it already holds both coordinates.** Fusing them into
the return value is this workspace's own `zero-copy-warden` SECOND-PROJECTION
shape, one layer up: a second reading stored beside the first.

Stated fairly, the other half: `gate_decision_i4(qualia, mantissa)` takes raw i4
qualia and *derives* the coordinates, so there the payload surfaces work the
caller did not do. **#1045's payload is right for the evaluator and wrong for the
trait.** That asymmetry is the finding in one line — and it means `reason()` is a
diagnostic rendering of two coordinates, not "the reason this decision happened".

## The topology this corrects

- **Two parallel MUL implementations**, neither calling the other, with disjoint
  `TrustTexture` vocabularies (contract: Calibrated/Overconfident/…; planner:
  Murky/Dissonant/Fuzzy/…). The planner's `mul` imports only `escalation` from
  the contract.
- **The planner's evaluator is the operator diagram**, check for check: Not
  Mount Stupid / Complexity mapped / Not depleted / Trust not murky-dissonant,
  then `Proceed{free_will_modifier}` → the FREE WILL MODIFIER box,
  `Sandbox{reason}` → SANDBOX/HUMAN REQUEST, `Compass` → the COMPASS FUNCTION.
  The M15 note calling this a name collision has the causality backwards: they
  are two different concepts, one of which is not MUL.
- **`contract::mul::GateDecision` is consumed only by execution/commit gates** —
  `kanban::advance_on_gate` (phase DAG, Block→Prune Libet veto), `action.rs`
  ActionState, `sigma-tier-router` Rest dispatch, `kanban_actor::mul_target`.
  Not one routes it to a compass or a learn-first path. It is the execution gate
  wearing MUL's module name.
- **Zero in-tree implementors** of `MulProvider` or `PlannerContract`
  (`grep -rn 'impl.*MulProvider for\|impl.*PlannerContract for' crates/` → no
  matches). The canonical evaluator is a free function. Both traits exist solely
  as an external verdict surface.

## What this does to the prior thesis

`GateLevel` (D-GATE-2) and the trait-signature change (D-GATE-3) are
**WITHDRAWN**: they would have generified the *execution* gate under a cleaner
name, and would still have left the two axes fused. The corrected order is
**axis orthogonality first**, then `DOMAIN EVIDENCE → MUL CALIBRATION → PLANNER
HINT`.

`MulHint{Trusted, Explore, Sandbox, Human}` is **not** adopted: the code already
carries that shape as `MulGateDecision{Proceed, Sandbox, Compass}`, and minting a
fourth gate enum beside three existing ones would be the same mistake a third
time.

## The loop this belongs to

```text
bad fit → Flow/Homeostasis detects tension → Resonance proposes another style
       → Alpha field reshapes → same world, different reading → new ΔF
       → reinforce / revise / explore
```

Same shape already ratified for styles per stratum (**resonance selects, ΔF
qualifies**), seen from the homeostatic side.

## What is NOT concluded

- #1045 is not reverted; no strings return to the hot path. Its core — typed
  `Copy` state, SIMD≡scalar equivalence, prose not stored — stands whole.
- No universal `GateGround` sum type; no fabricated `TrustTexture`/`FlowState`
  in ada-rs or MedCare; the stopgap stays unpushed.
- **No change to CollapseGate or Kanban** until their role is measured: T3
  measures what consumes the type, not whether the phase DAG is right.
- Five `TrustTexture` definitions exist workspace-wide (two MULs, causal-edge,
  arigraph orchestrator); named so it is not rediscovered, not in scope.
