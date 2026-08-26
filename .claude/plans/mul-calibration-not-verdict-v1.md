# Plan: MUL calibrates, it does not adjudicate (`mul-calibration-not-verdict-v1`)

> **Status:** PROPOSAL (unbuilt) — 2026-08-26. PLAN/BOARD ONLY, no code.
> **Supersedes the thesis of:** `.claude/plans/mul-gate-outcome-vs-ground-v1.md`
> (#1054). That plan's *measurements* stand; its *thesis* (OUTCOME vs GROUND,
> introduce `GateLevel`) is withdrawn — see §3.
> **Issue:** `ISS-MUL-GATE-OUTCOME-COUPLED-TO-PRODUCER-GROUND` (storno appended)
> **Arc:** #1045 → #1052 → #1054 → this

---

## 0. The corrected thesis — two corrections, in order

**First: restore the orthogonality of the state axes that #1045 accidentally
packed into the gate verdict.** Only then talk about outcome vs ground.

```text
WRONG ABSTRACTION                 RIGHTER SHAPE

GateDecision                      MUL state ─────────┐  calibration
   └ { TrustTexture, FlowState }   Flow axis ─────────┤  attention/style adaptation
          ↑            ↑           Homeostasis ───────┼→ planner policy → hint
       MUL axis    Flow axis       Complexity ────────┤
       accidentally fused          Impact ceiling ────┘
```

`FlowState` is not MUL's provenance. MUL asks one narrow question — *how
trustworthy is my own uncertainty estimate?* (calibrated / overconfident /
under-informed). The Csikszentmihalyi axis asks a different one, and its job
here is **not** a felt-state readout: it is **modulation of the attention field
— switching the thinking style as adaptation** (see T7, measured). Homeostasis
is a third regulator. Flow is no more the provenance of a MUL hold than
temperature is the provenance of a blood-pressure reading: both can matter to a
decision; they are not the same variable. The source already says so —
`mul.rs:127` is doc-commented **"Flow state (Csikszentmihalyi)"**.

**Second: `DOMAIN EVIDENCE → MUL CALIBRATION → PLANNER HINT`**, not
`OUTCOME vs GROUND`.

MUL calibrates the calibrator. It does not receive verdicts from domains and it
does not issue domain verdicts. A consent veto and an evidence contradiction are
**domain facts**: they enter as `SituationInput` / constraints, and the domain's
own execution gate acts on them.

**#1045 is therefore more correct than #1054 assumed.** Its valuable core stands
entirely — string allocation out, typed `Copy` state kept, SIMD≡scalar
equivalence, redundant prose not stored. The error was one step of reasoning
past it:

```text
typed state exists                    typed state exists
      ↓                                     ↓
therefore stuff that state       vs.  keep it in the assessment/state field
inside GateDecision                         ↓
                                      derive planner behaviour separately
```

What broke in ada-rs was not a contract that became too narrow — it was a
consumer that had been **using MUL as a generic three-state control channel**.
The break is the diagnosis, not the defect.

**Consequence: `GateLevel` is withdrawn for now.** §2 shows the type it would
have generified is the *execution* gate, not MUL's output. Introducing it would
restore the leak under a cleaner name — and would still leave the two axes fused.

---

## 1. Measured topology (2026-08-26, `743ce64`)

### T1 — there are TWO MUL implementations, and neither calls the other

| | `lance-graph-contract::mul` | `lance-graph-planner::mul` |
|---|---|---|
| `SituationInput` | `mul.rs:12` | `mul/mod.rs:32` (own) |
| `MulAssessment` | `mul.rs:50` | `mul/mod.rs:83` (own) |
| `TrustTexture` | `mul.rs:82` — Calibrated/Overconfident/Underconfident/Uncertain | `mul/trust.rs:30` — Murky/Dissonant/Fuzzy/… (own) |
| `FlowState` | `mul.rs:127` | `mul/homeostasis.rs:15` (own) |
| evaluator | `i4_eval::gate_decision_i4(qualia, mantissa)` | `mul::gate_check(assessment)` |
| **output** | `GateDecision{Flow, Hold{..}, Block{..}}` | `MulGateDecision{Proceed{free_will_modifier}, Sandbox{reason}, Compass}` |

The planner's `mul` module imports from the contract **only** `escalation`
(`mul/escalation.rs:10`) — never the MUL types. It is a full parallel
implementation, not a consumer.

### T2 — the planner's evaluator IS the operator diagram, line for line

`lance-graph-planner/src/mul/gate.rs` checks, in order:

| gate.rs | diagram's GATE box |
|---|---|
| `// Check 1: Not Mount Stupid` | ☐ Not Mount Stupid |
| `// Check 2: Complexity mapped` | ☐ Complexity mapped |
| `// Check 3: Not depleted` | ☐ Not depleted |
| `// Check 4: Trust not murky/dissonant` | ☐ Trust not murky/dissonant |
| `// Check 5: trust fuzzy → Compass` | → COMPASS FUNCTION |

and its outputs land exactly where the diagram sends them
(`lance-graph-planner/src/lib.rs:187-230`):

- `Proceed { free_will_modifier }` → the diagram's **FREE WILL MODIFIER** box
- `Sandbox { reason }` → **SANDBOX / HUMAN REQUEST** (`PlanError::GateBlocked`)
- `Compass` → the **COMPASS FUNCTION**

**The diagram's MUL already exists in code.** It is `MulGateDecision`, and the
M15 note recording that it was renamed *away from* `GateDecision` "because the
unqualified name collided" has the causality backwards: the two types are not
two spellings of one concept competing for a name — they are **two different
concepts**, one of which is not MUL.

### T3 — `contract::mul::GateDecision` is consumed only by execution/commit gates

| consumer | what it does with it |
|---|---|
| `kanban.rs:146 advance_on_gate` | phase-DAG movement: Flow→successor, **Block→Prune** (Libet free-won't veto), Hold→stay |
| `action.rs:301,373` | `ActionState::{Committed, Pending, Cancelled}` |
| `canonical_node.rs:1787` | `gate_decision_i4` → `advance_on_gate` |
| `sigma-tier-router/src/lib.rs:365` | `Block` → `Rest { GateBlocked }` dispatch |
| `lance-graph-supervisor::kanban_actor` | `mul_target` → next column |

**Not one consumer routes it to a compass, an exploration, or a
learn-first path.** Every consumer commits, cancels, or defers *work*. Its
measured role is the **execution/commit gate**, living in `mul.rs` under MUL's
name.

### T4 — nothing in this workspace implements `MulProvider` or `PlannerContract`

```sh
grep -rn 'impl.*MulProvider for\|impl MulProvider\|impl.*PlannerContract for' \
  --include='*.rs' crates/    # → no matches
```

The canonical MUL evaluator is a **free function** (`gate_decision_i4`), not a
trait method. So `MulProvider::gate_check` is not "how MUL is computed here" —
it is a surface whose only implementors are external. That is option **(c)** in
the question *"why does `MulProvider` exist?"*: it lets arbitrary consumers
decide gates. It has zero in-tree justification.

### T5 — `FlowState` is orthogonal to the calibration axis (operator, 2026-08-26)

Csikszentmihalyi flow/anxiety/boredom is **its own channel** — the diagram's
separate *FLOW & HOMEOSTASIS — qualia awareness of cognitive state* box — that
feeds the gate beside the trust/DK calibration axis. So `Hold { texture, flow }`
is not "MUL's ground" as one thing: it pairs **two orthogonal axes** that happen
to meet at the gate. This is why the pair reads as a natural payload and is
still wrong to demand from a producer that has neither axis.

### T6 — the axes were ALREADY separated; the verdict is a second projection

`MulAssessment` (`mul.rs:50`) already carries them apart:

```rust
pub struct MulAssessment {
    pub trust: TrustQualia,        // calibration axis (value + texture)
    pub dk_position: DkPosition,   // calibration axis
    pub homeostasis: Homeostasis,  // { flow_state (Csikszentmihalyi), allostatic_load }
    pub complexity_mapped: bool,
    pub free_will_modifier: f64,
}
```

The trait boundary is `gate_check(&self, assessment: &MulAssessment)
-> GateDecision` (`mul.rs:245`) — **the caller passes the assessment, so it
already holds `assessment.trust.texture` and
`assessment.homeostasis.flow_state`.** The `Hold { texture, flow }` payload is a
**second projection** of two coordinates the caller has in hand, fused across
axes, at exactly the boundary where they were already separate. That is this
workspace's own `zero-copy-warden` SECOND-PROJECTION shape — *stores a second
reading beside the first* — one layer up, in a type rather than in bytes.

**Stated fairly, the other half:** `gate_decision_i4(qualia, mantissa)`
(`i4_eval`) does **not** take an assessment — it takes raw i4 qualia and
*derives* the two coordinates. There the payload surfaces work the caller did
not do, and is not redundant. So #1045's payload is right for the evaluator and
wrong for the trait. That asymmetry is the whole finding in one line.

**Therefore `reason()` is suspect too** — not worthless, but mislabelled. It is a
*diagnostic rendering of two coordinates*, not "the reason this decision
happened". The reason lives in whatever policy combined the axes.

### T7 — the flow axis's measured job is thinking-style adaptation, not provenance

`lance-graph-planner/src/thinking/style.rs:272-275`:

```rust
FlowState::Flow    => StyleFamily::Analytical,
FlowState::Anxiety => StyleFamily::Deliberate,
FlowState::Boredom => StyleFamily::Creative,
FlowState::Apathy  => StyleFamily::Exploratory,
```

The flow axis selects the **thinking-style family** — attention-field modulation
and style switching as adaptation. It is a **dispatch input**, consumed in the
planner's thinking layer, not a coordinate that explains why a gate said Hold.
`style.rs:9` imports it from the planner's own `mul::homeostasis`, so this
consumer is wired to the planner's copy, not the contract's.

Two axes, two consumers, two layers: calibration → the gate; flow → the style
dispatch. #1045 additionally froze the flow coordinate into a contract gate
verdict, where nothing consumes it as flow.

### T8 — the flow axis is a CONTROL SIGNAL, not a decision instance (operator, 2026-08-26)

`Boredom / Flow / Anxiety` are not three small deciders and not a homunculus.
They are three regions of a control field measuring **how well the current
thinking style carries the current attention field**:

```text
challenge << skill  → boredom  → field too narrow / under-stimulated
                               → widen search, raise exploration, maybe switch style
challenge ≈  skill  → flow     → current style/attention geometry is paying
                               → preserve / reinforce
challenge >> skill  → anxiety  → style cannot absorb the task's complexity
                               → reduce commitment, broaden or restructure attention,
                                 possibly elevate rung / change style
```

Those are **parameter changes to the thinking process**, never decisions about
content. So the distinction that settles the payload question is:

```text
Gate              = may this action / transition proceed?
Flow/Homeostasis  = should cognition change its way of attending/thinking?
```

`FlowState` is never a *reason* for `Hold` or `Block`. It is feedback on whether
the current cognitive regime fits the problem structure — adaptation and
plasticity, not adjudication.

**Role separation (the organ list):**

| organ | question it answers |
|---|---|
| MUL | how trustworthy is my epistemic self-confidence? |
| Flow / Homeostasis | does the current cognitive regime FIT? |
| Resonance | which alternative thinking style is proposed? |
| Alpha field | where does attention currently conduct? |
| ΔF | did the adaptation help? |
| Planner / Rubicon | commitment and action |

**The loop this closes:**

```text
bad fit → Flow/Homeostasis detects tension → Resonance proposes another style
       → Alpha field reshapes → same world, different reading → new ΔF
       → reinforce / revise / explore
```

This is the same shape already ratified for styles per stratum
(`.claude/board/entries/2026-08-26-e-styles-anchor-at-rung-4-…-1.md`:
**resonance selects, ΔF qualifies**) — here seen from the homeostatic side. T7
is its measured footprint in code (`style.rs:272-275`, flow → `StyleFamily`);
T8 is why that wiring is the axis's *correct* home and the gate verdict is not.

---

## 2. Classification of every `mul::GateDecision` producer/consumer

Classes: **A** real MUL (trust/DK/flow/homeostasis) · **B** domain evidence or
constraint · **C** execution/commit gate · **D** planner navigation hint.

| site | role | class | verdict |
|---|---|---|---|
| `contract::mul::i4_eval::gate_decision_i4` | qualia+mantissa → decision | **A** | the canonical evaluator; keep, unchanged |
| `contract::kanban::advance_on_gate` | phase DAG / Libet veto | **C** | legitimate gate, **wrong input type name** — historical conflation |
| `contract::action.rs:301,373` | ActionState commit/cancel | **C** | same |
| `contract::canonical_node.rs:1787` | evaluator → phase | **A→C** | the one place A legitimately feeds C |
| `sigma-tier-router:365` | Block → Rest dispatch | **C** | same |
| `lance-graph-supervisor::kanban_actor::mul_target` | → next column | **C** | same |
| `deepnsm-v2::evidence.rs:476-493` | asserts `to_disc` ordering | **B** (consumer) | reads the ordering; does not produce a verdict — fine |
| `contract::mul::MulProvider::gate_check` | trait, **0 in-tree impls** | **(c) leak** | external consumers deciding gates |
| `contract::plan::PlannerContract::gate_check` | trait, **0 in-tree impls**, returns the **C** type from a planner trait | **(c) leak** | invalid at both ends |
| **ada-rs** `contract_impls.rs:72` | consent veto → `Block` | **B** | consent is domain evidence; must not emit a MUL decision |
| MedCare (unverified, D-MCAL-1) | contradiction → `Block` | **B** presumed | same, pending measurement |
| `lance-graph-planner::mul::gate_check` | → Proceed/Sandbox/Compass | **D** | **the diagram's actual MUL output — already exists** |

Nothing in class **D** is reachable through the contract today. That is the gap.

---

## 3. Answers to the five questions

1. **Why does `MulProvider` exist?** Measured: **(c)**. Nothing in-tree
   implements it; the canonical evaluator is a free function. It was not built
   to provide input to MUL (that is `SituationInput`) nor to implement MUL
   (that is `gate_decision_i4`).
2. **Is `PlannerContract::gate_check → mul::GateDecision` still valid?** No, on
   two counts: it returns the **execution-gate** type, and the planner's own MUL
   returns `Proceed/Sandbox/Compass` — a shape the contract cannot express.
3. **Does the architecture imply one canonical evaluator?** Yes per layer, and
   **no external implementations**: `gate_decision_i4` (contract) and
   `mul::gate_check` (planner). Two is already one too many (§5 OQ-MCAL-2).
4. **Is `advance_on_gate(&mul::GateDecision)` legitimate MUL coupling?**
   The *coupling* is legitimate — A feeding C at one measured site
   (`canonical_node.rs:1787`) is exactly the Rubicon crossing. The **naming** is
   the conflation: a phase gate typed by a module called `mul`.
5. **Should #1054's thesis be rewritten?** Yes. See §0. Its measurements (M1–M5)
   and its durable compatibility rule survive verbatim.

---

## 4. Smallest corrected plan diff

Withdrawn from #1054: **D-GATE-2** (`GateLevel`) and **D-GATE-3** (change the two
trait signatures). Both presuppose the trait surface is worth keeping.

| D-id | Deliverable | Gate |
|---|---|---|
| D-MCAL-0 | **axis restoration**: coordinates stay in the assessment/state field; stop fusing them into the verdict at the trait boundary (evaluator unchanged, per T6) | F-MUL-7 |
| D-MCAL-1 | finish the per-symbol consumer enumeration from GitHub (was D-GATE-1) — classify each as A/B/C/D | measurement only |
| D-MCAL-2 | decide the fate of `MulProvider` + `PlannerContract::gate_check`: **remove, or narrow to input-supply**, given 0 in-tree impls | F-MUL-5 |
| D-MCAL-3 | name the execution gate what it is (doc-first; a rename is a separate, later PR) | F-MUL-4 |
| D-MCAL-4 | express a consent veto and an evidence contradiction as domain evidence + domain execution gate, no MUL ground | F-MUL-1, F-MUL-2 |
| D-MCAL-5 | if a public MUL output is still needed, derive it from the planner's `Proceed/Sandbox/Compass` — **never** invent a fresh enum | F-MUL-4 |
| D-MCAL-6 | consumer-build gate (unchanged from D-GATE-5): BUILD them, don't grep them | F-MUL-6 |

**`MulHint{Trusted, Explore, Sandbox, Human}` is NOT adopted.** The code already
carries that shape as `MulGateDecision{Proceed, Sandbox, Compass}`; minting a
fourth gate enum beside three existing ones would be the same mistake a third
time. If D-MCAL-5 fires, it promotes the existing type.

---

## 5. Falsifiers

| id | falsifier | fails when |
|---|---|---|
| F-MUL-1 | a consent veto is expressible without constructing any MUL ground | the only path to a veto runs through `TrustTexture`/`FlowState` |
| F-MUL-2 | an evidence contradiction likewise | same, second domain |
| F-MUL-3 | the canonical evaluator still proves scalar ≡ SIMD typed ground after #1045 | a batch path diverges (regression guard on #1045) |
| F-MUL-4 | a "need more data" state produces navigation behaviour (learn / map / recover / sandbox), **not** a domain `Hold` | it silently becomes a phase-stay with no learning path |
| F-MUL-5 | removing/reframing external `MulProvider` impls still delivers the same domain behaviour through evidence/input/constraints | behaviour is lost — the trait was load-bearing after all |
| F-MUL-7 | the two axes are independently variable at the gate boundary: fixed `TrustTexture` with varying `FlowState` (and vice versa) must be constructible and must reach their two consumers — the gate and the style dispatch — as two coordinates | one axis cannot move without the other; they are still fused |
| F-MUL-6 | every known cross-repo consumer classified per symbol, then **built** against the eventual SHA | classification skipped or replaced by grep |

Anti-vacuity: F-MUL-1/2 must be shown **red first** against today's `main`
(ada-rs cannot express a veto without MUL ground) and green after.

---

## 6. Invariants carried forward (non-negotiable)

- #1045 is **not** reverted; no strings return to the hot path.
- No universal `GateGround` sum type in the contract.
- No fabricated `TrustTexture`/`FlowState` in ada-rs or MedCare; the ada-rs
  stopgap stays unpushed.
- **No change to CollapseGate or Kanban until their role is measured** — §1 T3
  measures *what consumes* the type, not whether the phase DAG is right.
- A source-breaking contract change is not verified until known unbound-git
  consumers **build** against the proposed head.

---

## 7. Open questions where code and diagram disagree

- **OQ-MCAL-1** — the diagram shows ONE MUL. The code has two, with disjoint
  `TrustTexture` vocabularies (T1). Which is canonical? Neither can absorb the
  other without a decision.
- **OQ-MCAL-2** — the contract's `GateDecision` has no `Compass` arm and the
  planner's has no `Hold`. Under the diagram, "hold" is not a state at all —
  failure routes to learn/map/recover/sandbox. Is contract-`Hold` a phase-stay
  masquerading as a MUL verdict (F-MUL-4)?
- **OQ-MCAL-3** — the diagram's Impact Ceiling / Human Humility Factor box has
  no in-tree representation found. Unbuilt, or elsewhere under another name?
- **OQ-MCAL-4** — `free_will_modifier` exists on both `MulAssessment` (contract)
  and `MulGateDecision::Proceed` (planner). The diagram places it strictly
  **after** the gate. The contract's placement predates the gate.
- **OQ-MCAL-5** — three `TrustTexture` definitions exist beyond the two MULs
  (`causal-edge/src/layout.rs:141`, `arigraph/orchestrator.rs:114`). Five total.
  Out of scope here; named so it is not rediscovered.
