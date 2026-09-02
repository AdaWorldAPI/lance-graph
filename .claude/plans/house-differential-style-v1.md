# House differential style v1 — a differential-diagnosis style program over shipped rung-3 tactics

> **Status:** PROPOSAL, plan-only (no code, no opcode, no tenant, no ClassView,
> no axis vocabulary, no new struct). **Baseline:** lance-graph `94543a5`
> (the #1136 merge). **Date:** 2026-09-02. **Working label:** "House" is a
> working name for the differential-diagnosis style pattern (rank hypotheses,
> attack the leader, keep the minority alive, order the test nobody ordered,
> eliminate only on independent evidence). It names a composition; it never
> becomes an identifier in code, a classid, or a vocabulary row.
> **Predecessors:** `post-teardown-buildup-survey-v1` (atoms / recipes /
> styles framing; PROBE-POP-READOUT-1 KILL), `persona-vs-rung-ladder.md`
> (rung-content ladder; styles are programs applicable at any rung),
> `rung-persona-orchestration-v1` (D-PERSONA-1 shipped the council +
> epiphany loop this plan composes). **Sibling:**
> `thinking-engine-harvest-closure-v1` (the ghost-prior harvest this plan's
> D-HOUSE-4 waits on). **Domain fence:** the medical meaning of a differential
> lives in MedCare-rs (private). This plan carries only the reasoning
> mechanics; no clinical vocabulary, code, or data enters lance-graph.

## 0. Why a plan and not a struct

A grep for the pattern's name over `.claude/`, `crates/thinking-engine`,
`crates/lance-graph-contract` and `crates/lance-graph-planner` returns
nothing. The mechanics are nevertheless shipped, in pieces, across four
surfaces. Naming the composition is therefore the whole deliverable; the
first code this plan licenses is a falsifier, not a feature.

```
families = state / evidence carriers        (unchanged; six, see survey §1)
atoms    = readouts over the BeliefArena     (expectation, contradiction, gaps)
recipes  = the rung-3 tactics RCR / ASC / CR (+ TR, CAS as needed)
style    = the House program: WHICH recipes fire, in WHAT order, with WHICH
           periphery budget — a policy, applicable at any rung
```

## 1. What is already shipped (the parts list, file-cited)

| House move | Shipped surface | Where |
|---|---|---|
| The whiteboard: ranked hypotheses from an observation | RCR abduction over a shared predicate → `Frontier { candidates, gaps }`, `Candidate { stmt, truth, premises, rung, tactic }`, `ReasoningGap` | planner `nars/tactics.rs:177 rcr_abduce`, `:71 Candidate`, `:88 ReasoningGap` |
| Attack the leading hypothesis | ASC: `challenge_target = ⟨1−f, c⟩`; counter-evidence admitted ONLY with a disjoint stamp; self-refutation `BlockedSelfReference` | `tactics.rs:457 challenge_target`, `:479 asc_challenge`, `:463 AscOutcome` |
| Thesis vs antithesis on one claim | CR: NARS revision, `|f₁−f₂|` contradiction depth preserved, never averaged | `tactics.rs:503 cr_synthesize` |
| The team argues; the minority view stays on the board | `InnerCouncil::deliberate`, `is_split(0.7, 0.5)` amplified ×1.2; a split quorum is a Contradiction, never averaged; the minority pole is forked as a −6 counterfactual nibble | contract `escalation.rs:137 InnerCouncil`, `:116 is_split`; `quorum.rs:215 quorum_project`; `counterfactual.rs:140 deposit_counterfactual` |
| The test nobody ordered (the contrarian) | excluded tactics run as watchers; a watcher that moves the score may force rung elevation; it never decides | contract `recipes.rs:557 peripheral_recipes`, `:573 peripheral_sample`, `:601 peripheral_sample_rotating`, `:641 peripheral_sample_where`; planner `strategy/style_strategy.rs:121 peripheral_dissent`, `:184 cross_family_dissent` |
| Cheap passes first, the counterfactual leap last | `Recipe::min_rung` / `admissible_at`, `RungLevel::for_pass`; admissible set 4 → 11 → 24 → 34; `ExtremelyHard` gated to `Counterfactual` | `recipes.rs:505–545` |
| Pattern memory and its failure mode (anchoring) | `GhostField::{imprint, bias, prediction, free_energy}` (excluded crate); contract twin `GhostEcho` + `WisdomMarker` (floor 0.1, decay 0.85) | `thinking-engine/src/ghosts.rs:75–200`; contract `escalation.rs:289 GhostEcho`, `:312 WisdomMarker` |
| Saturation: when everything is admissible, stop picking a winner | passive `quorum_mantissa` + `meta_basin` outlier SUGGESTIONS | `E-SATURATION-SWITCHES-TO-PASSIVE-QUORUM-1` |
| The one selection edge that exists today | `tactic_for_bias(GraphBias) -> TacticChoice` — a pure LUT driven by graph health, not by style | planner `nars/tactic_select.rs:77` |
| The consumer shape | `differential() -> Vec<DiffRow { disease, expectation, frequency, confidence, contradiction }>`, `Frontier::gaps` | MedCare-rs `medcare-cohorts/src/reasoning.rs:363`, `:499` (private repo; cited by path only) |

Everything in the table is consumed as-is. Nothing in it is modified by
this plan.

## 2. The composition (the style program, recipe level)

One House cycle over a `BeliefArena`, written as the order in which shipped
tactics fire. Each step names its output atom so the falsifier in §4 can
measure it.

1. **Board.** `rcr_abduce(arena, throttle)` from the observation's
   predicate. Output: `Frontier.candidates` ranked by `truth.expectation()`
   (the ranking key MedCare's `DiffRow` already uses). `Frontier.gaps`
   records why candidates were NOT formed (`NoSharedMiddle`, `HubExcluded`,
   `BudgetExhausted`, …) — this is the raw material for step 5.
2. **Periphery.** `RungLevel::peripheral_sample_where(k, eligible)` picks the
   watchers the current rung excludes; `StyleStrategy::peripheral_dissent`
   runs them as observers. If one moves the score beyond `tol`, the rung
   elevates and step 1 re-runs with the wider admissible set. The periphery
   never votes (E-PERIPHERAL-DISSENT-GUARDS-THE-STRATIFICATION-1).
3. **Challenge the leader.** For the top candidate, `asc_challenge` with the
   best available counter-evidence. Only a DISJOINT stamp is admitted; a
   `BlockedSelfReference` outcome is recorded, not retried with the same
   sources. For a contested claim carried by two candidates,
   `cr_synthesize` keeps both frequencies' distance as contradiction depth.
4. **Council.** `InnerCouncil::from_signals(trust, humility, flow, load)`
   (planner `mul/escalation.rs::verdict_from` already derives these from a
   `MulAssessment`). A split verdict routes through `quorum_project` →
   contested → `deposit_counterfactual(minority pole)`. The minority
   diagnosis stays addressable under the −6 lane and can win later via
   `revise_if_minority_wins` when the mailbox arm exists.
5. **Discriminating evidence (OPEN — the first readout this plan asks for).**
   For the top two candidates, name the single observation whose presence
   would swap their order. Definition (a READOUT, no state): clone the
   arena, `observe` a hypothetical premise with the candidate's own truth,
   re-rank, report the premise with the largest rank displacement.
   `Frontier.gaps` gives the candidate premises; the clone-and-observe is
   what `dismech_counterfactual` already does for a cut edge, run here for
   an added one. Not built; D-HOUSE-2.
6. **Elimination (OPEN — the second readout).** "Ruled out" is NOT "ranked
   low" (MedCare's own plan says so). Definition: a candidate is eliminated
   when an ASC challenge with DISJOINT counter-evidence drove its
   `expectation()` below a pre-registered floor AND the contradiction depth
   is recorded on the belief. Both facts are already on the arena entry
   (`truth`, `contradiction`, `stamp`); elimination is a predicate over
   them, never a new enum. Falsifier: an eliminated candidate must not
   re-enter the top-k without a NEW disjoint stamp. D-HOUSE-3.
7. **Memory and the anchoring alarm (gated on the sibling plan).** On a
   green-flip (`CollapseHint::Flow` plus an `Epiphany`) a `WisdomMarker`
   is laid; on the next case the marker biases the initial ranking. The
   alarm is `free_energy`: prior says X, evidence says Y, surprise rises →
   the prior is demoted, not the evidence. The contract has the marker but
   no field (no per-atom `bias`/`prediction`/`free_energy`); the excluded
   crate has the field but no consumer. D-HOUSE-4 waits for
   `thinking-engine-harvest-closure-v1` D-TEH-2 (ghost prior harvested with
   a falsifier) and is NOT built by porting `GhostField` as a singleton.

What the style ADDS beyond the parts: the order (1→7), the periphery budget
`k`, the elimination floor, and the rule that step 3 runs before any
commit. It adds no arithmetic of its own.

## 3. Persona, rung, style — where House sits on the ladder

- **Rung content:** House is a rung-3 composition (the 34 recipes ARE the
  runbooks: RCR #4, TR #6, ASC #7, CAS #8, CR #11). Its ADMISSION follows
  `Recipe::min_rung`: on a shallow pass only the gate/datapath tactics fire,
  and the House cycle degenerates to "board + periphery"; ASC/CR (Control
  bucket) open at `Analogical`. That degeneration is the pyramid working as
  designed, not a defect.
- **Style application is not confined to rung 4** (scope correction of
  2026-08-30 in `persona-vs-rung-ladder.md`). House is a program over the
  atom/mask substrate; the landing zone named for such programs is
  `.claude/v3/FUTURE-DESIGN.md` E-THINKING-STYLES-ARE-CLASSES-1 (style = StepMask ×
  WideFieldMask + rung set + KausalSpec). House would be ONE such class.
  This plan does not mint it (D-TSC-2/3 are the mint vehicle, still
  queued).
- **Persona is out of scope by ruling O3.** `PersonaProfile` /
  `CognitiveBaseline` (12 dims incl. `precision_drive`,
  `epistemic_humility`) and the adjective-36 are the unwired
  persona-modelling storyline. A "House persona" would be a data card there;
  nothing in §2 needs it. `StyleFamily` (12) is enough to address the
  program: the natural default is `Analytical` with a raised periphery
  budget; measured, not assumed (§4).
- **The missing edge is O2**, unchanged: nothing selects recipes FROM a
  style. `tactic_for_bias` selects from graph health. A House program needs
  the style→recipe edge; §4's probe drives the cycle by hand so the edge's
  absence does not block the measurement, and the probe's result is the
  evidence for building the edge.

## 4. Falsifier first — PROBE-HOUSE-DIFFERENTIAL-1 (D-HOUSE-1)

**Question.** Does the House cycle (§2 steps 1–4, run by hand) recover a
planted cause more often than RCR alone, above a size-preserving shuffle
null, on synthetic arenas whose distractors share predicates with the
planted cause?

**Why synthetic.** No clinical data may enter lance-graph (MedCare-rs
commitment #9; the private repo is the only place that corpus exists), and
the KJV SPO stream has no observation→cause structure. Planted-cause
recovery is the standard falsifier for an abductive ranker and needs no
domain vocabulary.

**Fixture (pre-registered).**
- N = 200 arenas, SplitMix64 seed `0x9E3779B97F4A7C15 ^ i`.
- One planted cause `C*` with 4–6 observations `O_j`, each an
  `O_j → C*` belief (`Copula::Inh`, frequency 0.9, confidence 0.6,
  disjoint stamps).
- 5 distractor causes sharing 1–3 observations each with `C*` (the
  shared-predicate trap RCR abduces over).
- One disjoint counter-evidence stamp per distractor (frequency ≤ 0.3) that
  is ONLY visible to step 3.
- Half the arenas additionally carry a "far" fact reachable only by a tactic
  the starting rung excludes (the periphery's job).

**Arms.**
- A0: RCR alone (`rcr_abduce`, rank by expectation).
- A1: RCR + ASC on the leader (steps 1, 3).
- A2: full cycle (steps 1–4; `k = 4` periphery, `tol = 0.02`).
- AN: A2 on 25 permutations of the observation→cause links per arena
  (size-preserving null).

Three additional arms (added 2026-09-02 on operator direction; same
harness, same null, same arenas — each is a READING over shipped state, none
adds a register):

- **A_res — gestalt resonance first (rung 0–1 prior).** Rank candidates by
  resemblance BEFORE any abduction — the perspectival
  `awareness_dto::ResonanceDto` / `HdrResonance` shape (x,y,z per archetype,
  `is_epiphany`, `is_unanimous`), the pyramid's base per the triangle plan and
  the D-TRI-3 nail→hammer direction — then run A2 seeded by that order.
  Measures whether a gestalt prior helps the board or anchors it.
- **A_dye — signed qualia as Kontrastmittel.** Per candidate, ONE signed i4
  derived from the ASC outcome (+ `Revised`, − `BlockedSelfReference`,
  0 `NoTarget`), read into the existing `QualiaI4_16D` lanes (tenant 1) and
  consulted by the ranking. A contrast agent, not data: it exists only to
  make contested structure visible. The lane it uses is chosen by the arm,
  not minted for it; if no lane fits, the arm reports that and stops
  (OQ-CSV-1, the 17D-vs-16D reconciliation, is answered by this arm's need,
  never pre-empted).
- **A_ap — humor as aperture.** The cycle has one knob, `k` in
  `peripheral_sample_where(k)`. Fixed `k = 4` in A2; here
  `k = f(incongruity)`, with incongruity read from what is already emitted
  (CR contradiction depth `|f₁−f₂|`, or the arena's free energy). Controlled
  incongruity opens the periphery wider exactly when the board looks too
  tidy; a tidy board with a hidden far fact is the case it should catch.

**Metrics.** p@1 and p@3 for `C*`; elimination false-positive rate (the
planted cause must never be eliminated); periphery fire rate.

**PASS iff** (a) A2 p@1 − A0 p@1 ≥ 0.05 AND A2 p@1 > AN p95, (b) planted
cause eliminated in 0 arenas, (c) periphery fires on ≥ 10 % and ≤ 90 % of
arenas (can-fire AND can-stay-silent). **KILL** otherwise, and the plan
stops at §1 as a parts list. The three additional arms are graded
SEPARATELY against A2 (not against A0): an arm PASSES its own gate iff it
clears the rank-inertness guard AND adds ≥ 0.03 p@1 over A2 above the same
null; each arm's verdict is reported on its own line, so a KILL on one never
hides a PASS on another.

**Anti-vacuity guards (each disable-verified before the run is trusted).**
- Removing the disjoint-stamp gate in the fixture must make A1 worse than
  A0 (otherwise the guard was never binding).
- Zeroing the far facts must drop the periphery fire rate to ~0.
- A2 with `k = 0` must equal A1 to the bit.
- **Per-arm rank-inertness guard (the 2026-09-02 lesson: `curiosity_gestalt`
  was rank-identical to `curiosity` at ρ = 1.000000).** Each of A_res / A_dye
  / A_ap must produce a ranking that DIFFERS from A2 (Spearman ρ < 1 over
  the candidate set) before its p@1 is read; an arm with ρ = 1 is reported
  as decoration, not as a null result.
- A_res must NOT beat the planted cause's rank on no-distractor arenas (a
  prior that overrides clean evidence is anchoring, measured).
- A_dye with all signs forced to 0 must equal A2 to the bit.
- A_ap inertness: raising `f` must admit a watcher on some arena, lowering it
  must silence one; if neither moves, the knob is theatre and the arm is
  reported KILL on its own.

**Deliverable shape.** One example in `lance-graph-planner/examples/`
(no library surface), plan §4a RESULT, one board entry, STATUS row.
Nothing else lands from a PASS except a licence to design the O2 edge.

## 5. Deliverables

| D-id | title | scope | status |
|---|---|---|---|
| D-HOUSE-0 | this plan: parts list, composition, ladder placement, falsifier | plan-only | Shipped (this PR) |
| D-HOUSE-1 | PROBE-HOUSE-DIFFERENTIAL-1 (§4) | planner example | Queued — next step |
| D-HOUSE-2 | discriminating-evidence readout (§2 step 5) | planner `nars` readout, no state | Gated on D-HOUSE-1 PASS |
| D-HOUSE-3 | elimination predicate (§2 step 6) + never-re-enter falsifier | planner `nars` readout | Gated on D-HOUSE-1 PASS |
| D-HOUSE-4 | anchoring alarm over a harvested ghost prior (§2 step 7) | planner, after D-TEH-2 | Gated on `thinking-engine-harvest-closure-v1` D-TEH-2 |
| D-HOUSE-5 | the style→recipe edge (O2) with House as its first program — two faces after PASS: (a) persona modelling as a LENS (a readout policy over the same arena: admitted recipes, periphery budget, elimination floor, council weights; philosophical labels are further lenses, and two lenses with identical rankings are one lens), (b) the program materialised over the loco palette (`ogar-loco` / `ogar-r2il`, the ruled convergence for style application; no new atom — the 226-atom palette's 30 reserved slots stay untouched; any opcode rides the batched OGAR mint, never solo) | contract + planner (+ OGAR mint) | Gated on D-HOUSE-1 PASS and the D-TSC-2/3 mint |
| D-HOUSE-6 | consumer pointer: `DiffRow` gains READ fields `eliminated` / `discriminator` | MedCare-rs (out of scope here) | pointer only |

## 6. What this plan does NOT do

- No opcode, tenant, ClassView, axis set, or lane; no `ENVELOPE_LAYOUT_VERSION`
  change.
- No new struct for the differential: `Frontier` + `BeliefArena` entries are
  the state; §2 steps 5–6 are readouts.
- No port of `GhostField` into the contract or planner as a singleton; the
  ghost prior arrives per-mailbox through the sibling plan or not at all.
- No persona modelling, no adjective-36 wiring (O3).
- No clinical vocabulary, fixtures or data in lance-graph.
- No name "House" in any identifier, test name, or vocabulary row.

## 7. Provenance

Read-only survey on 2026-09-02 over `94543a5`: `persona-vs-rung-ladder.md`
(full), `recipes.rs` (rung admission + periphery), `escalation.rs`,
`quorum.rs`, `counterfactual.rs`, planner `nars/{tactics,tactic_select,
stance}.rs`, `mul/escalation.rs`, `thinking-engine/src/{ghosts,persona,
meaning_axes}.rs`, board entries `E-PERIPHERAL-DISSENT-GUARDS-THE-
STRATIFICATION-1`, `E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1`,
`E-SATURATION-SWITCHES-TO-PASSIVE-QUORUM-1`, `E-DIALECTIC-V1-TACTICS-IN-
PLANNER-1`; MedCare-rs `docs/REASONING_CHAIN_INTEGRATION_PLAN.md` §A rows
(private; cited by path).
