# Plan: separate the gate OUTCOME from producer-owned GROUND (`mul-gate-outcome-vs-ground-v1`)

> **Status:** SUPERSEDED (thesis) 2026-08-26 — measurements M1–M5 and the
> consumer-build rule STAND; the OUTCOME-vs-GROUND thesis and `GateLevel`
> (D-GATE-2/3) are WITHDRAWN. Successor:
> `.claude/plans/mul-calibration-not-verdict-v1.md`. Reason: measured — the
> type `GateLevel` would have generified is the EXECUTION gate, not MUL's
> output; MUL's actual output already exists as the planner's
> `MulGateDecision{Proceed, Sandbox, Compass}`.
> **Origin:** follow-up arc to PR #1045 (`GateDecision` de-stringing) and
> PR #1052 (`collapse_gate` ordinal alignment).
> **Issue:** `ISS-MUL-GATE-OUTCOME-COUPLED-TO-PRODUCER-GROUND`
> **Predecessor claim being finished, not reversed:** #1045's removal of
> `reason: String` from the MUL hot path was correct and stays.

---

## 0. The correction in one paragraph

#1045 correctly removed redundant prose from the MUL hot path, but in doing so
it promoted **MUL-specific ground** (`TrustTexture`, `FlowState`) into the
**public gate outcome contract**. Every implementor of `MulProvider` — and of
`PlannerContract`, which returns the same type — must now phrase its verdict in
MUL's vocabulary, whether or not MUL's vocabulary is where its reason lives. A
consent veto is not a trust texture. An evidence contradiction is not a flow
state. The fix is to separate **what the gate decided** (universal, transported,
ordinal) from **why this producer decided it** (producer-owned, never
transported through the gate type).

```text
PUBLIC / TRANSPORTED

#[repr(u8)]
GateLevel
  Flow  = 0
  Hold  = 1
  Block = 2

            │
            │ outcome only
            ▼

PRODUCER-SPECIFIC GROUND

MUL:      TrustTexture, FlowState
Ada:      ConsentVeto, DK state, AllostaticLoad
MedCare:  EvidenceContradiction, MissingEvidence

            │
            ▼

witness / alpha / producer address
```

The law at the public boundary:

```text
GateLevel        = WHAT
producer witness = WHY
```

**Explicit anti-goal.** Do NOT introduce a
`enum GateGround { TrustFlow(..), Consent(..), MedicalEvidence(..), .. }`.
That moves the coupling outward and makes `lance-graph-contract` own everybody's
epistemology. The producer already holds its ground; it stays there.

---

## 1. Measured state (2026-08-26, working tree at `743ce64`)

Four findings that shape the design. Each was measured, not assumed.

### M1 — the breaking surface is TWO traits, not one

| trait | file:line | signature |
|---|---|---|
| `MulProvider` | `mul.rs:245` | `fn gate_check(&self, assessment: &MulAssessment) -> GateDecision` |
| `PlannerContract` | `plan.rs:160` | `fn gate_check(&self, situation: &SituationInput) -> GateDecision` |

`plan.rs:7` imports `crate::mul::GateDecision`, so both traits return the same
type. Any arc that changes only `MulProvider` leaves half the surface coupled.

### M2 — `MulGateDecision` is already taken, and was renamed to escape this exact collision

`lance-graph-planner/src/mul/gate.rs:19` defines
`pub enum MulGateDecision { Proceed { free_will_modifier }, Sandbox { reason },
Compass }` — the planner's Meta-Uncertainty verdict, **not** a Flow/Hold/Block
gate. Its own doc-comment records the reason for the name:

> *Renamed from `GateDecision` (M15) — this is the planner's
> Meta-Uncertainty-Layer verdict (Proceed/Sandbox/Compass), NOT the contract's
> kanban gate.*

Naming the new internal MUL-ground type `MulGateDecision` would therefore
**re-collide a name that was already renamed once to escape this neighbourhood**.
See §2 for the recommended variant, which needs no new name at all.

### M3 — `TrustTexture` is itself producer-scoped, twice over

`contract::mul::TrustTexture` = `{Calibrated, Overconfident, Underconfident,
Uncertain}`. `lance-graph-planner::mul::trust::TrustTexture` carries
`{Murky, Dissonant, Fuzzy, …}` and is matched on in `gate::check`. Two crates,
one name, different variants — independent corroboration that trust texture is
*a producer's ground*, not a universal gate field.

### M5 — the V3 ruling already flags this neighbourhood

`.claude/v3/COMPONENT-MAP.md` lists **GATE-1 (two `GateDecision` types)** among
its key risks, and rules:

| symbol | verdict | note |
|---|---|---|
| `mul::GateDecision` {Flow/Hold/Block} | **REUSE** | *the LIVE kanban gate* — `KanbanColumn::advance_on_gate` consumes THIS one |
| `collapse_gate::GateDecision` + `MergeMode` | REPURPOSE | per-row write-merge gate |

Two consequences bind this arc. First, `mul::GateDecision` is REUSE, not
RETIRE — so the ground type is kept and given a `level()`, never removed
(§2 already does this; the ruling confirms it rather than being worked around).
Second, `advance_on_gate` consumes the **ground** type directly, not the trait
return, so changing the two `gate_check` signatures to `GateLevel` must leave
the kanban consumer untouched. D-GATE-4 verifies that rather than assuming it.

### M4 — consumer enumeration must be per-symbol, not per-crate

Confirmed unbound-git consumers of `lance-graph-contract`:

| consumer | binding | uses `mul::GateDecision`? |
|---|---|---|
| `ada-rs` | `git = ".../lance-graph"`, **no branch/rev** | **yes** — `src/contract_impls.rs:72` `impl MulProvider` |
| OGAR `ogar-class-view` | `git = ".../lance-graph#main"` (in OGAR's own repo) | unverified — its surface is `ClassView`; must be checked, not assumed |

In-tree EXCLUDED crates (`lance-graph-ogar`, `symbiont`, `cognitive-stack`)
reach the contract transitively through OGAR **but each carries
`[patch."https://github.com/AdaWorldAPI/lance-graph"] lance-graph-contract =
{ path = "../lance-graph-contract" }`**, folding it onto the working-tree copy.
They are therefore *protected* from a main-branch break — they are not
unbound consumers, and must not be counted as coverage either.

---

## 2. Target shape (recommended variant: no rename, minimal churn)

```rust
// contract::mul — PUBLIC OUTCOME
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GateLevel { Flow = 0, Hold = 1, Block = 2 }

// contract::mul — MUL-INTERNAL GROUND (today's GateDecision, unchanged shape)
pub enum GateDecision {
    Flow,
    Hold  { texture: TrustTexture, flow: FlowState },
    Block { texture: TrustTexture, flow: FlowState },
}
impl GateDecision { pub fn level(&self) -> GateLevel { … } }   // to_disc stays

// the two traits return the OUTCOME
trait MulProvider     { fn gate_check(&self, a: &MulAssessment)    -> GateLevel; }
trait PlannerContract { fn gate_check(&self, s: &SituationInput)   -> GateLevel; }
```

`gate_decision_i4`, the SIMD batch paths, `reason()`, and the full-equality
scalar-vs-batch falsifier all keep operating on `GateDecision` unchanged, so
**#1045 loses none of its value**: the MUL hot path stays heap-free and typed,
and its evaluator still proves scalar and batch produced the same ground.

Rejected alternative (record, do not delete): renaming today's `GateDecision`
to `MulGateDecision` and freeing `GateDecision` for the outcome — rejected by
M2, and it would churn every in-workspace ground call site for a name change
that buys nothing the `level()` accessor doesn't.

Open question **OQ-GATE-1**: `collapse_gate::GateDecision.gate: u8` (aligned in
#1052) is the same ordinal ladder in struct form. Whether it should *become*
`GateLevel` or merely be *documented as* carrying one is deferred — it is a
second public type with its own consumers, and folding it in would widen this
arc past what one review can hold.

---

## 3. Hard compatibility rule (the arc's real deliverable)

> **A source-breaking change in `lance-graph-contract` is not verified until
> known unbound-git consumers build against the proposed head.**

Not grep. Not "all workspace callers use `{ .. }`". **Build them.** #1045 passed
every workspace gate — clippy `-D warnings`, member-tests, the full contract
suite — and still broke a real consumer, because the consumer is not in the
workspace and pins no rev. Workspace-green is not contract-green.

At minimum, per M4: `lance-graph` itself, `ada-rs`, MedCare-rs; plus any further
consumer enumerated from GitHub *before* implementation, checked per-symbol.

---

## 4. Falsifiers (each must be able to fail)

| id | falsifier | fails when |
|---|---|---|
| F-GATE-1 | MUL internal: scalar and SIMD produce identical `GateLevel` **and** identical ground over a non-degenerate qualia corpus | a batch path diverges from scalar on either half |
| F-GATE-2 | Ada: a consent veto emits `Block` **without** constructing `TrustTexture`/`FlowState` | the only way to say Block still requires MUL provenance |
| F-GATE-3 | MedCare: an evidence contradiction emits `Block` without inventing MUL provenance | same, second producer |
| F-GATE-4 | ordinal/ABI: `Flow=0, Hold=1, Block=2` across `GateLevel`, `mul::GateDecision::to_disc`, `collapse_gate::GateDecision.gate`, and `ndarray::hpc::qualia_gate::QualiaGateLevel` | any drifts (extends #1052's cross-type falsifier to the new type) |
| F-GATE-5 | consumer build: each enumerated consumer compiles against the exact proposed contract SHA | any fails to build — the change is then unverified, not "mostly fine" |

F-GATE-2 and F-GATE-3 are the load-bearing pair: they are the tests that would
have gone red under #1045's shape, and green under this one. If a proposed
design cannot make both go red before the fix, the design has not addressed the
finding.

Anti-vacuity notes (per `CLAUDE.md` § falsifiability rule): F-GATE-1's corpus
must contain inputs that reach every arm of `gate_decision_i4` — a corpus that
only ever produces `Flow` proves nothing. F-GATE-5 must be shown red first
against today's `main` (ada-rs fails to compile) and green after.

---

## 5. Deliverables

| D-id | Deliverable | Gate |
|---|---|---|
| D-GATE-1 | enumerate real `lance-graph-contract` git consumers from GitHub, per-symbol (which import `mul::GateDecision` / `PlannerContract`) | none — pure measurement, runs first |
| D-GATE-2 | `GateLevel` (`#[repr(u8)]`, 0/1/2) in `contract::mul`; `GateDecision::level()` | F-GATE-4 |
| D-GATE-3 | `MulProvider::gate_check` + `PlannerContract::gate_check` return `GateLevel` | F-GATE-1 |
| D-GATE-4 | in-workspace call-site migration (planner, driver, kanban, deepnsm-v2) | member-tests + full suite |
| D-GATE-5 | consumer-build gate: documented procedure + one recorded red→green run per consumer | F-GATE-5 |
| D-GATE-6 | consumer-side landings (ada-rs first) once D-GATE-3 is on main | F-GATE-2 |

D-GATE-1 gates everything: implementing before the enumeration repeats #1045's
exact mistake one level up.

---

## 6. Explicitly out of scope

- `collapse_gate::GateDecision` restructuring (OQ-GATE-1 above).
- Any `GateGround` sum type (§0 anti-goal).
- The `EpistemicDeficit` work in `.claude/plans/kognitionswirtschaft-v1.md`
  (which carries a ⊘ NOT-CANONICAL header). Its D-KW-2 names
  `Hold.reason` demotion; that half is already done by #1045 and is not
  re-opened here.
- Reverting any part of #1045. The hot-path de-stringing stays.
