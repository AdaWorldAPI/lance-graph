# Neurofunctional crosswalk: OpenPsi/Rubikon control and Moore-local synaptic plasticity

**Date:** 2026-10-17  
**Status:** ARCHITECTURE / COMPARATIVE RESEARCH. No runtime code, contract, ABI, or authority change.

**Operator thesis:** Moore tenant + NARS16×8 is the intended **local synaptic-plasticity substrate**. OpenPsi-style demands and the Heckhausen/Gollwitzer + O'Reilly/Axon Rubikon lineages supply **motivation, goal commitment and control**. Merged lance-graph [#1571](https://github.com/AdaWorldAPI/lance-graph/pull/1571) supplies a **named-state calibration profile**, not synaptic truth, biological feelings or permission to act.

This is a mapping of *computational functions*, not a claim that a Rust module is anatomically equivalent to a brain region. The relevant sources, actual code, open migrations and independent falsifiers are kept apart.

## 1. Three interacting timescales

```text
FAST / LOCAL: 8-neighbour Cartesian membrane
  pre/post activity at (i, neighbour-slot)   [NOT plain presence]
        | 
  MoorePalettePairs:  P/O semantic palette pair  (8 × 16 bits)
  Nars16x8:          F/C evidence reading        (8 × 16 bits, candidate)
  MooreNars16:       Pearl/Energy/Plasticity/Polarity/Epi5
                                               (8 × 16 bits)
        |
  reciprocal membrane agreement; observed independent evidence
        |
  local eligibility + inhibited / excitatory revision proposals
        ^                                  |
        | (modulation)                     | measured outcome
        |                                  v
MID / CONTROL: motivation + volition
  ada-rs GoalGenerator / OpenPsi demand and modulators
       -> desirability / feasibility / cost
       -> Rubikon commitment gate
       -> maintained goal / if-then cue policy
       -> authorized action (separate permission)
       -> outcome / temporal credit / disengagement
        |                                  |
        +------------ modulation ---------+
                                           |
SLOW / EPISODIC: thought-Git
  0 sealed givens; 1..63 hypothesis branches
       -> rebase -> independent evaluation -> squash / land
       -> replay / metacognitive revision / reusable method
```

The key separation is **what is learned locally**, **what goal is selected**, and **what is justified by evidence**. These must remain different state and authority domains even when compiled to the same R2IL/mask-RISC algebra.

## 2. Source-backed inventory and status

| Source | Actual surface | Proposed role | Status |
|---|---|---|---|
| [ada-rs goal_generator.rs](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/cognition/goal_generator.rs) | `AutonomousGoal`, `GoalOrigin` (diagonal lack, counterfactual regret, curiosity, dream, novelty, external), `NeedLevel`, urgency/satisfaction | demand / goal candidate production | Rust implementation exists; heuristics not evidence of human motivation |
| [ada-rs will_engine.rs](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/cognition/will_engine.rs) | `Volition`, `WillConstraint`, `WillState`, `synthesize`, blocked-goal checks | volition ranking / constraints | exists; not integrated with canonical Rubikon commit gate |
| [ada-rs qualia_17d.rs](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/feel/qualia_17d.rs) | 17 named numerical axes | state telemetry input | coordinate contract, not evidence of experience |
| [LG PR #1571](https://github.com/AdaWorldAPI/lance-graph/pull/1571) | `qualia_profile`: 219 named patches, PCS 17D, palette, K residual, mix, device/PCS/observer separation | device→state vocabulary / policy appraisal *after calibration* | MERGED; five shared axis names only, twelve unmeasured transforms, even shared semantics are conjectural |
| [hyperon-openpsi](https://github.com/iCog-Labs-Dev/hyperon-openpsi) | experimental MeTTa demand, modulator, feeling and planner modules | motivation and competing-demand reference | experimental, not a complete available LG cognitive runtime |
| [hyperon-openpsi PR #117](https://github.com/iCog-Labs-Dev/hyperon-openpsi/pull/117) | tie-breaking and demand→distinct-goal repair | negative control: priority must deterministically affect actual choice | OPEN when checked 2026-10-17 |
| [LG Rubicon plan](../plans/rubicon-volition-v1.md) | goal candidate→commitment→cue policy→action→evaluation/disengagement | Heckhausen/Gollwitzer phases | PROPOSED |
| [Axon Rubicon](https://github.com/emer/axon/blob/main/Rubicon.md) | cost/benefit selection, BG-like gating, maintained goal, temporal credit | computational reward / maintenance analogy | external hypothesis and implementation, not shipped LG |
| [FLOW-COG Rubicon convergence](../../docs/architecture/FLOW_COG_RUBICON_ONA_CONVERGENCE.md) | two Rubicon lineages, ONA evidence and OpenPsi ranking separated | source-of-truth ownership | architectural document, not a new runtime |
| [Moore tenant](../../crates/lance-graph-contract/src/moore_tenant.rs) | three eight-lane value tenants, reciprocal slots, `MooreNars16`, CE64 lift/refusals | **local synaptic carrier** | physical layout exists; learned update and witness migration open |
| [Moore one-hop probe](../../crates/cognitive-shader-driver/examples/moore_plasticity_probe.rs) | fixed eight-neighbour schedule, synchronous snapshot update, mask freeze | local perturbation and gated update | probe, not biological STDP |
| [MooreNars16 parity test](../../crates/lance-graph-planner/tests/moore_tenant_isa_equivalence.rs) | CE64 reconstruction observed equivalent for tested shipped operations | encoding equivalence gate | tested in declared homogeneous/core scope |
| [Moore Gomoku](../../crates/cognitive-shader-driver/examples/moore_nars_gomoku_probe.rs) | delayed outcome credited to relative `signature × recipe`, not positional moves | learning benchmark | task probe; not complete synaptic-to-Rubikon loop |

OpenCog/Hyperon contributes typed compositional symbolic methods and multiple possible bindings. ONA/NAL contributes truth revision and evidence arithmetic. OpenPsi-inspired ranking contributes *motivation*. They should **compile through existing methods**, not become rival persistent graph stores, trust authorities, or scheduler engines.

## 3. Local plasticity is Moore + NARS16×8, not a CE64 rewrite

At each deterministic Cartesian cell coordinate, the eight neighbour slots are NW, N, NE, W, E, SW, S, SE. Proposed shared membranes match opposite slots (E↔W, NE↔SW) at the neighbour's own coordinate: **no foreign-row gather**. Reciprocal consistency and version equality are explicit tests.

Each of the three tenants has eight `u16` lanes, together **48 bytes per cell** for the represented neighbour readings:

- `MoorePalettePairs`: semantic P/O as `(u8,u8)` per neighbour.
- `Nars16x8` (candidate reading): F/C as `(u8,u8)` per neighbour. NARS F/C denotes evidence support, not an arbitrary synaptic reward score.
- `MooreNars16`: **Pearl3 (bits 0..2), Energy4 (3..6), Plasticity3 (7..9), Polarity1 (10), Epi5 (11..15)**. The exact bit allocation lives in `moore_tenant.rs`. These are Moore-local readings of the canonical CE64 ISA contract, not a new 128-bit causal edge.

Local context answers address/neighbor direction; use existing canonical fields rather than storing pointers. An explicit observation or activation event is required to interpret a synapse as firing.

**Two still-open migrations are important:**

1. The 2026-10-17 operator decision identifies the **Cartesian reciprocal membrane** as the local witness. Current `MooreTenantMut::lift_ce64` instead expects homogeneous W across all eight lanes and refuses `MixedWitness`; W also has a competing **thought-layer breadcrumb** reading. Do not claim the reciprocal membrane law already powers the reader.
2. The operator decision treats CE64 46..49 as signed **excitation/inhibition** and 50..52 as a local maturation/plasticity gear; existing ISA code reads 46..49 as an inference opcode, and 50..52 has an unresolved novelty-vs-plasticity conflict. Only explicit versioned migration can resolve this. Never consume reserved bits silently or create CE128 to escape it.

**The presence bit is not a spike.** It means a thought/value is represented at an address, not that any firing event took place. Without a separate timestamped activation trace, “co-present means co-active” would fabricate plasticity evidence.

## 4. OpenPsi + Heckhausen Rubikon: control the goal, not the synapse

Psychological Heckhausen/Gollwitzer Rubikon:

```text
PREDECISIONAL: consider goals (desirability, feasibility, uncertainty)
  -> witnessed and revocable GOAL COMMITMENT  [Rubikon]
PREACTIONAL: form if (cue) then (proposed action) plan
ACTIONAL: action only if policy/capability/role still authorizes it
POSTACTIONAL: observe outcome, evaluate, disengage or re-plan
```

Computational O'Reilly/Axon Rubicon contributes BG-inspired goal selection, maintained engaged goal, learned cost/benefit, and *temporal credit assignment* when reward or omission is observed later. Neither authority nor psychological truth follows merely from having that state machine.

The [OpenPsi source](https://github.com/iCog-Labs-Dev/hyperon-openpsi/blob/main/main/feeling-updaters.metta) distinguishes **Activation, Resolution, SecuringThreshold, SelectionThreshold, Pleasure** as modulators; [modulator-updaters](https://github.com/iCog-Labs-Dev/hyperon-openpsi/blob/main/main/modulator-updaters.metta) contains numeric helper functions. OpenPsi's priority score must not be conflated with NARS confidence or certified CE64 causes. PR #117 highlights a concrete risk: two demands can tie and whichever input happens to be first wins; a broken goal mapper can discard demand identity entirely. Use deterministic tie and downstream-effect falsifiers.

In `ada-rs`, the `Volition.confidence` float comes from the **will-engine heuristic**; it is not a readout of NARS `C`. #1571's 219 labels are not a substitute for a validated control signal; the device→PCS transform is presently incomplete. No emotion label or utility score authorizes a side effect.

The invariant remains:

```text
demand → ranked goal candidate → independent commitment gate
       → cue-bound proposal → current action permission
       → executed outcome receipt → revised belief / choice
```

## 5. Three-factor synaptic learning: the testable new seam

A candidate three-factor update separates **local activation eligibility**, a **later global/modulatory learning signal**, and **evidence-correct revision**.

Illustrative mathematical policy, **not shipped code or a new register field**:

```text
eligibility_ij(t+1) = λ · eligibility_ij(t)
                      + coactivity(pre_i(t), post_j(t+Δ))

Δeffective_gain_ij = η · outcome_modulator(t) · eligibility_ij(t)
```

- `eligibility_ij` is a bounded versioned temporal trace, not persistent NARS confidence, nor inferred from a set presence bit.
- `outcome_modulator` comes from an independently observed result relative to a maintained goal, or an explicitly ablatable exploratory policy; it is **not** the raw word "joy" or a high `EpistemicState5` code.
- A gain adjustment governs *future local signal propagation or recipe selection*. `F/C` is revised only when new, independent evidence warrants it. One source copied into two thought branches is still one source.
- Correlated pre/post activity proposes association. It does not establish a Pearl causal relationship or warrant CE64 certification `Causes`.
- Inhibition, polarity, saturation, temporal delay, source-root overlap and Simpson-stratum preservation require their own oracles.
- Do not allocate new byte lanes or an entirely separate plasticity engine before testing what the existing tenants, temporal envelopes and pure R2IL folds can express.

Thus the stronger replacement for naïve Hebb is: **coactivity makes a local membrane eligible; a witnessed outcome selectively gates the update; independent evidence constrains what it is legal to believe about that membrane**.

## 6. End-to-end falsifier scenario

A small sealed environment has **two competing goals**, **delayed feedback**, and observable **local neighbour interactions**. Hold evidence, budget, policies and clocks constant across arms.

1. Hydrate given observations with distinct source roots and explicit observation coverage.
2. Create **activity events** separate from the Moore presence mask; check reciprocal neighbour identity.
3. Derive candidate goals via `ada-rs::GoalGenerator`; optional OpenPsi-style demand and modulator policy changes *ranking*, not truth.
4. Deliberate and cross the Rubikon only through an independent current commitment gate.
5. Hold the chosen goal across a delay; emit an if-then action only with current permission.
6. Observe completion **or demonstrably covered omission**. Preserve time/source/provenance.
7. Use the observed outcome to update eligible local membranes while preserving NAL source-root accounting.
8. Rebase the thought-Git branch against the new sealed reference; retract changed-premise consequences before replay.
9. Measure held-out predictive performance, correct goal selection, appropriate disengagement, false causal certification, memory, and CPU.

**Controls:** no plasticity, Hebbian-only, outcome-gated, shuffled/late reward, permuted demand ordering, no Rubikon commitment, no cue plan, and incorrect reciprocal-slot implementation.

### Kill conditions

| Gate | What must be falsified |
|---|---|
| NF-0 | Presence accidentally treated as firing |
| NF-1 | Moore E↔W / NE↔SW reciprocity fails on a sealed version |
| NF-2 | Reconstituted Moore three-tenant state diverges from canonical CE64 for declared supported operations |
| NF-3 | One physical observation copied into multiple thought worlds increases confidence twice |
| NF-4 | Shuffled delayed reward strengthens the same “correct” synapses as real credit |
| NF-5 | OpenPsi tie/goal mapping depends on iteration order or discards demand identity |
| NF-6 | Goal cue dispatches after authorization/revocation changes |
| NF-7 | Unmeasured PCS axes silently become known affect coordinates |
| NF-8 | CE64 legacy W, opcode nibble, or reserved Epi5 passes as current canonical reading |
| NF-9 | Simpson pooling reverses an association and still upgrades causal certification |
| NF-10 | Gated plasticity fails to outperform matched-cost baselines on previously unseen cases |

All scenarios need a from-scratch independent oracle, a deliberately broken alternative that fails, measured wall/CPU/RSS, and replayable versioned receipts.

## 7. Architecture ownership and next research decision

- **Synapse:** Moore-local tenants and reciprocal membranes. This is the **intended plasticity substrate**.
- **Belief revision:** ONA/NAL/CE64 with *witnessed independent source roots*, not reward arithmetic.
- **Demand/appraisal:** `ada-rs` goal/will, experimental OpenPsi scoring, optionally #1571 state-profile readings where actually calibrated.
- **Rubikon:** distinct goal selection, maintained commitment, implementation intention and disengagement, with independent authority gates.
- **Method composition:** Hyperon/OpenCog expression semantics through existing typed classes and the R2IL/mask-RISC execution seam.
- **Durability/replay:** thought-Git's sealed versions, not reusable thought-slot numbers.
- **Self-observation:** alpha/holographic saccades observe where attention moved, not what was true.

**First high-leverage milestone:** Under a fixed evidence and time budget, demonstrate that witnessed delayed goal outcomes improve a local Moore-plasticity learner **without altering NARS evidence math, conflating presence with activity, double counting a witness, or crossing an action gate autonomously**.

If the learner cannot beat a Hebbian-only and a shuffled-modulator control on unseen cases, the proposed cross-layer cognitive benefit is **refuted for that fixture**, however compelling the anatomical analogy may appear.

## Source reading order

1. [Moore synapse / membrane decisions](../board/entries/2026-10-17-ce64-one-layout-moore-synapse-membrane.md), [Moore tenant](../../crates/lance-graph-contract/src/moore_tenant.rs), [Moore parity tests](../../crates/lance-graph-planner/tests/moore_tenant_isa_equivalence.rs), [one-hop plasticity probe](../../crates/cognitive-shader-driver/examples/moore_plasticity_probe.rs).
2. [Ada-rs GoalGenerator](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/cognition/goal_generator.rs), [WillEngine](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/cognition/will_engine.rs) and [Qualia17D](https://github.com/AdaWorldAPI/ada-rs/blob/master/src/feel/qualia_17d.rs).
3. [LG #1571](https://github.com/AdaWorldAPI/lance-graph/pull/1571), [QualiaProfile](../../crates/lance-graph-contract/src/qualia_profile.rs).
4. [OpenPsi experimental MeTTa](https://github.com/iCog-Labs-Dev/hyperon-openpsi), [PR #117](https://github.com/iCog-Labs-Dev/hyperon-openpsi/pull/117), [Axon Rubicon model](https://github.com/emer/axon/blob/main/Rubicon.md).
5. [Heckhausen/Gollwitzer sources](../../.arxiv/2026-10_rubicon_gollwitzer_volition_source_excerpts.md), [Rubicon plan](../plans/rubicon-volition-v1.md), [FLOW-COG](../../docs/architecture/FLOW_COG_RUBICON_ONA_CONVERGENCE.md).
6. [Cartesian epistemic self-model](cartesian-epistemic-self-model.md), [thought-Git](../plans/thought-git-api-v1.md) and [CE64 reading](../../crates/lance-graph-contract/src/epistemic_state5.rs).

**Status of each claim:** `CURRENT-CONTRACT` means source code, `MEASURED` means a documented probe, `DECISION` means a recorded architectural preference, and `PROPOSED` means a testable transfer. No architecture diagram is implementation proof.

