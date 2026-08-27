# Measurement: per-symbol MUL consumer census (`mul-consumer-census-v1`)

> **Status:** MEASUREMENT COMPLETE — 2026-08-27. Measurement only, no code change.
> **Discharges:** D-MCAL-1 (`.claude/plans/mul-calibration-not-verdict-v1.md` §4).
> **Gates:** F-MUL-6's first half (classification per symbol). The second half —
> **BUILD** the consumers — is D-MCAL-6 and is NOT discharged here.
> **Measured at:** lance-graph `8b99ef30`; MedCare-rs `17871e06`; ada-rs, OGAR,
> ndarray at their session checkouts; org-wide search 2026-08-27.

---

## 0. What this measures, and what it does not

D-MCAL-1 asks for the per-symbol consumer enumeration that §2 of the
calibration plan sketched from partial evidence, with every site classified:

| class | meaning |
|---|---|
| **A** | real MUL — trust / DK / flow / homeostasis is actually computed |
| **B** | domain evidence or constraint wearing MUL's type |
| **C** | execution / commit gate wearing MUL's name |
| **D** | planner navigation hint |

Two rows in the plan's §2 table were marked *presumed* or *unverified*. Both
are now measured, and **one of them was understated**: MedCare's
contradiction path is not one site but two, and the fabrication is
self-documented in the source.

**Not measured here:** whether any consumer still *builds*. That is D-MCAL-6,
deliberately separate — a census is a grep, and this plan's own F-MUL-6 says a
classification is not a build.

---

## 1. The symbol set

Nine symbols, taken from `contract::mul` and `contract::plan`:

`GateDecision` · `MulProvider` · `MulAssessment` · `SituationInput` ·
`TrustTexture` · `FlowState` · `DkPosition` · `PlannerContract` ·
`i4_eval::gate_decision_i4`

**Name-collision guard applied throughout.** Four distinct types in this
workspace answer to `GateDecision` or `TrustTexture`. A census that greps the
bare name over-counts by a wide margin (906 raw hits across 69 in-tree files).
Every row below was resolved to its *defining module* before classification:

| spelling | defining module | this census counts it? |
|---|---|---|
| `contract::mul::GateDecision` | `mul.rs:158` | **yes** — the subject |
| `contract::collapse_gate::GateDecision` | `collapse_gate.rs:59` | no — different type (ordinals aligned in #1052) |
| `planner::mul::gate::MulGateDecision` | `planner/src/mul/gate.rs` | no — class **D**, listed separately |
| `contract::membrane::MembraneGate` | RBAC trait | no — correctly-named, unrelated |

---

## 2. In-tree census

### 2a. Class A — real MUL (the evaluator and its inputs)

| site | symbol | note |
|---|---|---|
| `contract/mul.rs:158` | `GateDecision` | the type itself |
| `contract/mul.rs` `i4_eval::gate_decision_i4` | evaluator | derives both coordinates from i4 qualia + mantissa; the ONE site where the payload is earned |
| `contract/mul.rs` `i4_eval::batch`, `benches/i4_batch.rs` | evaluator (SIMD) | batch form of the same |
| `contract/canonical_node.rs:1787` | evaluator → phase | the single measured **A→C** hand-off |
| `supervisor/src/cycle_driver.rs:96` | `gate_decision_i4` | evaluator consumer |
| `supervisor/src/kanban_actor.rs:39` | `gate_decision_i4` | evaluator consumer |
| `supervisor/tests/{probe_ignition,probe_ignition_64k,d_ign_b_lenses}.rs`, `examples/measure_wal_curve.rs` | `gate_decision_i4` | test/probe consumers |
| `cognitive-shader-driver/src/driver.rs:45` | `MulAssessment`, `SituationInput`, `MulThresholdProfile` | assessment producer |

**Count: 1 canonical evaluator, 1 legitimate A→C hand-off.** Unchanged from
the plan's §1 T2/T3 reading.

### 2b. Class C — execution / commit gates typed by `mul`

| site | what it does with the decision | reads the payload? |
|---|---|---|
| `contract/kanban.rs:146` `advance_on_gate` | phase-DAG move; `Block → Prune` (Libet veto) | **no** — matches `Block { .. }` / `Hold { .. }` |
| `contract/action.rs:301,373` | `ActionState::{Committed, Pending, Cancelled}` | **no** |
| `sigma-tier-router/src/lib.rs:365` | `Block → Rest { GateBlocked }` | **no** |
| `supervisor::kanban_actor` `mul_target` | → next kanban column | **no** |

**Newly measured and load-bearing: not one class-C consumer reads
`texture` or `flow`.** Every one of them destructures with `{ .. }`. The
payload that #1045 added to the verdict is, at every execution-gate consumer
in the workspace, **inert**. This is the empirical form of the plan's T6
"second projection" argument — previously reasoned, now counted.

### 2c. Class B — domain evidence wearing the MUL type (in-tree)

| site | note |
|---|---|
| `deepnsm-v2/src/evidence.rs:476-493` | asserts `to_disc` ordering only; **reads**, never produces. Not a violation. |
| `lance-graph/examples/graph_self_reasoning.rs:47` | example constructs `Hold`/`Block` with hand-picked pairs |
| `cognitive-shader-driver/examples/probe_revision_kanban_hinge.rs:323` | same, in a probe |

The two example/probe sites construct the same fabricated pairs the external
consumers do (§3). They are *examples*, so they are documentation of the
anti-pattern rather than production behaviour — but they are also where a new
consumer learns the shape, which is how it spread.

### 2d. Class D — the planner's actual MUL output

| site | note |
|---|---|
| `planner/src/mul/gate.rs` `MulGateDecision{Proceed, Sandbox, Compass}` | the diagram's MUL, per T2 |
| `planner/src/api.rs:78` | re-exported as `Gate` |
| `planner/src/lib.rs:187-230` | routes the three arms |

**Still unreachable through the contract.** Confirmed: no contract module
names `MulGateDecision`. The §2 gap statement stands.

### 2e. The trait surfaces

| trait | in-tree impls | external impls |
|---|---|---|
| `contract::mul::MulProvider` | **0** | **1** (ada-rs) |
| `contract::plan::PlannerContract` | **0** | **0** |

`PlannerContract` measured at **zero implementors anywhere** — in-tree or
across the whole org. Its doc-comment still instructs `ladybug-rs`,
`crewai-rust` and `n8n-rs` to call it; the latter two were EVICTED
2026-06-21, and `ladybug-rs` does not implement or call it. Its
`gate_check(&self, situation: &SituationInput) -> GateDecision` signature is
**worse** than §2 recorded: it takes `SituationInput`, not `MulAssessment`, so
it is a planner trait that performs a MUL assessment *and* returns the
execution-gate type — invalid at three points, not two.

---

## 3. Cross-repo census

Dependent repos were found by one org-wide manifest search
(`lance-graph-contract` in `Cargo.toml`): **58 manifests across 17 repos**.
Of those, the MUL symbols appear in **four**.

### 3a. ada-rs — class **B**, one `MulProvider` impl

| site | symbol | class |
|---|---|---|
| `src/contract_impls.rs:22` | `impl MulProvider for AdaMulAdapter` | the org's only implementor |
| `src/contract_impls.rs:72` `gate_check` | consent veto → `Block` | **B** |
| `src/contract_impls.rs:81` | DK MountStupid → `Hold` | **A** (genuinely DK) |
| `src/contract_impls.rs:88` | allostatic load → `Hold` | **A** (genuinely homeostatic) |
| `src/world_model_feed.rs:7` | `SituationInput` | input supply, no verdict |
| `src/lance_bridge.rs:57` | `pub use ...::mul` | re-export only |

The gate_check is **mixed**: two of its three non-Flow arms are real MUL
(Dunning-Kruger, allostatic load), and exactly one — the consent veto — is
domain evidence. That is a sharper result than "ada-rs is class B": the
adapter is a legitimate class-A implementor with **one** class-B arm
smuggled in at the top, and it is that arm which #1045 broke.

### 3b. MedCare-rs — class **B**, two sites, fabrication self-documented

| site | trigger | constructed payload |
|---|---|---|
| `crates/medcare-first-thought/src/patient_thought.rs:236` | evidence contradiction (`observed > 0 && e < 0.5`) | `Block { texture: Uncertain, flow: Anxiety }` |
| `crates/medcare-first-thought/src/patient_thought.rs:245` | equivocal / unobserved | `Hold { texture: Calibrated, flow: Boredom }` |
| `crates/medcare-first-thought/src/lib.rs:521` | anti-valid jc verdict (`validity_r`/`rho`) | `Block { texture: Uncertain, flow: Anxiety }` |
| `crates/medcare-first-thought/src/lib.rs:530` | neither valid nor anti-valid | `Hold { texture: Calibrated, flow: Boredom }` |

**Nothing in MedCare measures trust texture or flow state.** The inputs are a
NARS truth expectation, an evidence-set cardinality, and two reliability
coefficients (Pearson r, Spearman ρ). Neither MUL coordinate exists anywhere
in that crate's data flow. Both coordinates are chosen by hand to satisfy the
contract's own LUT, and the source says so:

> `advance_on_gate` — it matches on the variant alone — so this mapping
> is descriptive, not behavior-affecting

That comment is accurate (§2b confirms the payload is never read) and is
precisely the problem: a required field that no consumer reads, filled with
values no producer measured. MedCare's own board recorded this as a
*GateDecision-Fabrikationsfund* on 2026-08-27, independently and on the same
day this census ran.

**This is the strongest available evidence for F-MUL-2** and it is stronger
than the plan assumed: §2 listed MedCare as "**B** presumed, pending
measurement" at one site. It is confirmed **B**, at four sites, with the
fabrication documented in-source by the author.

### 3c. OGAR — not a consumer

`crates/ogar-doc-ir/src/resolve.rs:22` names `PlannerContract` in a
doc-comment analogy about dependency inversion. No import, no call. **Mention,
not consumer.**

### 3d. bardioc — consumer of the *other* `GateDecision`

`substrate-b/src/canonical.rs` uses
`contract::collapse_gate::GateDecision::FLOW_BUNDLE`. This is the type whose
ordinals #1052 realigned, not `mul::GateDecision`. **Counted as a
non-consumer of MUL**, and recorded here because it is the collision hazard in
the wild: an external repo consuming one `GateDecision` by that bare name.

### 3e. n8n-rs / crewai-rust — EVICTED, own types

`n8n-rs` (`n8n-contract/src/{free_will,impact_gate}.rs`) and `crewai-rust`
(6 sites) each define and use their **own** `GateDecision`. Neither imports
the contract's. Consistent with the 2026-06-21 eviction; no bump obligation.

---

## 4. Result table

| class | in-tree sites | external sites | total |
|---|---|---|---|
| **A** real MUL | 9 | 2 (ada-rs DK + load arms) | 11 |
| **B** domain evidence | 3 (2 examples + 1 read-only) | 5 (ada-rs consent + MedCare ×4) | 8 |
| **C** execution gate | 4 | 0 | 4 |
| **D** planner hint | 3 | 0 | 3 |
| trait leak | 2 traits, 0 in-tree impls | 1 impl (ada-rs) | — |

---

## 5. What the census changes in the plan

Three corrections to `mul-calibration-not-verdict-v1.md`, filed as an addendum
rather than an edit (append-only):

- **C1 — the payload is inert at every class-C consumer.** T6 argued the
  `Hold { texture, flow }` payload is a *second projection* of coordinates the
  caller already holds. Measured: it is worse than redundant at the execution
  gates — all four destructure it away. The type demands data that no consumer
  of that arm reads.
- **C2 — `PlannerContract` has zero implementors org-wide**, and its
  `gate_check` takes `SituationInput` (not `MulAssessment`), making it invalid
  at three points. §2 recorded two.
- **C3 — MedCare is confirmed, not presumed, and is four sites not one.** The
  fabrication is documented in-source by its own author.

One item for OQ-MCAL-5, which counted five `TrustTexture` definitions:

- **A sixth exists.** `ada-rs/src/memory/trust.rs:22` defines
  `TrustTexture{Crystalline, Solid, Fuzzy, Murky, Dissonant}` — the
  *planner's* vocabulary, in a consumer repo, beside that same repo's import
  of the *contract's* four-variant `TrustTexture` in `contract_impls.rs`. One
  crate, two incompatible types of the same name. Recorded, out of scope.

---

## 6. Falsifier status after this census

| id | status |
|---|---|
| F-MUL-1 (consent veto without MUL ground) | **RED on main, confirmed** — `contract_impls.rs:72` has no path to a veto that does not construct a `GateDecision` |
| F-MUL-2 (evidence contradiction likewise) | **RED on main, confirmed at four sites** — MedCare, with in-source admission |
| F-MUL-5 (trait removal preserves behaviour) | measurable now: 1 impl to migrate (ada-rs), 0 for `PlannerContract` |
| F-MUL-6 (per-symbol classification, then BUILD) | **first half discharged here**; build half is D-MCAL-6 |

The anti-vacuity requirement in §5 of the calibration plan — "F-MUL-1/2 must
be shown red first against today's `main`" — is satisfied by §3a and §3b with
file:line evidence, before any code moves.
