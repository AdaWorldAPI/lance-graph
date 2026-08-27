# E-THE-FUSED-PAYLOAD-IS-INERT-AT-EVERY-EXECUTION-GATE-THAT-CONSUMES-IT-1

**Date:** 2026-08-27
**Status:** FINDING (measured)
**Deliverable:** D-MCAL-1
**Measurement:** `.claude/plans/mul-consumer-census-v1.md`
**Arc:** #1045 → #1052 → #1054 → #1055 → this

---

## The finding

`GateDecision::Hold { texture, flow }` and `Block { texture, flow }` require
two coordinates that **not one execution-gate consumer reads**. All four
class-C consumers destructure the payload away:

```rust
kanban.rs:146        GateDecision::Block { .. } => …   // Libet veto
action.rs:301,373    GateDecision::Hold  { .. } => …   // ActionState
sigma-tier-router:365                    { .. }        // Rest dispatch
supervisor::kanban_actor::mul_target     { .. }        // next column
```

The plan (`mul-calibration-not-verdict-v1` T6) *argued* the payload was a
second projection of coordinates the caller already holds. The census
**counts** it, and the measured result is one step worse than the argument:
at the gates, the payload is not redundant, it is **inert**. A required field
that no consumer of that arm reads is not redundancy — it is a tax with no
payer.

## Why that matters more than it sounds

A field nobody reads is not harmless, because the type still demands it. Every
producer must supply two coordinates to construct a `Hold` or a `Block`. A
producer that has neither axis has exactly two options: acquire them, or
invent them. Measured downstream, both chose to invent:

- **ada-rs** `contract_impls.rs:72` — consent veto, no trust axis, no flow axis.
- **MedCare-rs** `patient_thought.rs:236` + `lib.rs:521` and their `Hold`
  twins — NARS expectation, evidence cardinality, Pearson r, Spearman ρ. No
  MUL coordinate exists anywhere in that data flow.

MedCare's own source states the mechanism plainly:

> `advance_on_gate` — it matches on the variant alone — so this mapping
> is descriptive, not behavior-affecting

Both halves of that sentence are true, and together they are the defect: the
values are unread, therefore unconstrained, therefore free to be invented,
therefore invented. MedCare's board independently recorded this as a
fabrication finding on the same day this census ran.

## The generalisation

> **A required field that no consumer reads does not stay empty — it fills
> with fiction.** Inertness and fabrication are the same defect seen from the
> two ends of the type. The reader's end shows a field that could be deleted
> with no behaviour change; the writer's end shows a field being filled by
> hand to satisfy a compiler. Neither end alone looks like a bug. Together
> they are one.

The corollary for review: when a payload is added to a verdict "for
diagnostics", the falsifiable question is not *is it useful?* but **which
consumer reads it, by name?** If the answer is none, the field is not
diagnostic — it is a fabrication surface with a docstring.

## Scope

- This is about the **fused payload at the trait/verdict boundary**, not about
  `gate_decision_i4`. The evaluator *derives* both coordinates from i4 qualia
  and surfaces work the caller did not do; there the payload is earned. That
  asymmetry is the whole of T6 and is unchanged.
- #1045 is **not** reverted. Its core — no strings in the hot path, typed
  `Copy` state, SIMD ≡ scalar — stands. What this finding measures is the one
  step past it.
- No code moves on this entry. D-MCAL-2/3/4/5 act; this records why.
