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

```text
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
| Thesis vs antithesis on one claim | CR: NARS revision, the absolute frequency difference kept as contradiction depth, never averaged | `tactics.rs:503 cr_synthesize` |
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

1. **Board.** `rcr_abduce(arena, throttle)`. Abduction direction matters
   (Codex P1 on #1137, verified in `tactics.rs:177`): RCR pairs two `Inh`
   beliefs sharing a PREDICATE, `{P→M (rule), S→M (obs)} ⊢ S→P`. So the
   arena must carry cause knowledge as `cause→feature` rules and the case as
   `case→feature` observations; the shared predicate is the FEATURE, and the
   candidates come out as `case→cause`. Beliefs written the other way round
   (`feature→cause`) make RCR relate features to each other and the cause
   never appears as a candidate. The board = the candidates with
   `stmt.s == case`, ranked by `truth.expectation()` (the ranking key
   MedCare's `DiffRow` already uses); the throttle's `hub_indegree` must
   exceed the number of causes sharing a feature, or shared features are
   barred as hubs. `Frontier.gaps` records why candidates were NOT formed
   (`NoSharedMiddle`, `HubExcluded`, `BudgetExhausted`, …) — the raw
   material for step 5.
1b. **Admit.** `rcr_abduce` does not mutate the arena; a `Candidate` is a
   proposal. Before any tactic can act on the leader it must be admitted:
   `BeliefArena::admit_derived` (`belief.rs:226`) for the top-k. Without
   this step `asc_challenge` returns `AscOutcome::NoTarget` for every
   leader and the challenge arm is vacuous (Codex P1, verified).
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
   `MulAssessment`) → `CouncilVerdict`; on `verdict.split`,
   `deposit_counterfactual(&verdict, &mut edge)` tags the minority
   diagnosis's edge with the −6 mantissa (the planner already has
   `impl EpisodicEdge for CausalEdge64`, D-DCR-3). **Not on the path:**
   `quorum.rs::quorum_project` and `quorum_project_blackboard` are
   unconditional `todo!` scaffolds blocked on D-ATOM-1/3 (Codex P1,
   verified at `quorum.rs:216, 253`), and `CounterfactualMailbox` /
   `revise_if_minority_wins` are the v3 `todo!` arm. The probe uses the
   council verdict and the deposit only; the minority stays addressable
   under the −6 lane and may win later when the mailbox arm exists.
5. **Discriminating evidence (OPEN — the first readout this plan asks for).**
   For the top two candidates, name the single observation whose presence
   would swap their order. Definition (a READOUT, no state): clone the
   arena; for each candidate premise `observe` it with the candidate's own
   truth and re-rank; keep ONLY premises that actually swap the top two;
   among those report the one that leaves the largest post-swap margin,
   ties broken by ascending premise id. If no single premise swaps the
   order the readout returns an explicit `None` carrying the current
   top-two margin ("no single observation discriminates") — never the
   largest non-swapping displacement (CodeRabbit on #1137: a largest
   displacement can preserve the order, and the no-swap case must be
   defined).
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

The list above reads linearly; the execution is a LOOP. On a shallow pass
steps 3–4 do not exist yet (Control-bucket tactics open at `Analogical`), so
the real shape is: board → periphery → (elevate) → board again → challenge →
council, and only after the loop settles do the two readouts (5–6) and the
memory step (7) run. §4b's aperture is definable only because of this loop.

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
- One planted cause `C*` with 4–6 features `O_j`, written as RULES
  `C* → O_j` (`Copula::Inh`, frequency 0.9, confidence 0.6, disjoint
  stamps), and one `case` with observations `case → O_j` for every planted
  feature (the direction RCR needs; see §2 step 1).
- 5 distractor causes `D_i`, each with rules `D_i → O_j` sharing 1–3
  features with `C*` (the shared-predicate trap RCR abduces over) plus 2–3
  private features the case does not show. `Throttle::hub_indegree` is set
  above 7 so a feature shared by all six causes and the case is not barred
  as a hub.
- One disjoint counter-evidence stamp per distractor (frequency ≤ 0.3) that
  is ONLY visible to step 3.
- Half the arenas additionally carry a "far" fact reachable only by a tactic
  the starting rung excludes (the periphery's job).

**Arms.**
- A0: RCR alone (`rcr_abduce`, rank by expectation).
- A1: RCR + admit + ASC on the leader (steps 1, 1b, 3).
- A1c: A1 + council (steps 1, 1b, 3, 4; no periphery) — the control that
  isolates the periphery's contribution.
- A2: full cycle (steps 1, 1b, 2, 3, 4; `k = 4` periphery, `tol = 0.02`).
- AN: A2 on 25 permutations of the observation→cause links per arena
  (size-preserving null).

**Metrics.** p@1 and p@3 for `C*`; elimination false-positive rate (the
planted cause must never be eliminated); periphery fire rate.

**PASS iff** (a) A2 p@1 − A0 p@1 ≥ 0.05 AND A2 p@1 > AN p95, (b) planted
cause eliminated in 0 arenas, (c) periphery fires on ≥ 10 % and ≤ 90 % of
arenas (can-fire AND can-stay-silent). **KILL** otherwise, and the plan
stops at §1 as a parts list.

**Anti-vacuity guards (each disable-verified before the run is trusted).**
- Removing the disjoint-stamp gate in the fixture must make A1 worse than
  A0 (otherwise the guard was never binding).
- Zeroing the far facts must drop the periphery fire rate to ~0.
- A2 with `k = 0` must equal A1c to the bit (A1 lacks the council, so
  equality with A1 was the wrong control — CodeRabbit on #1137).
**Deliverable shape.** One example in `lance-graph-planner/examples/`
(no library surface), plan §4 RESULT (added as a dated subsection when the
run reports), one board entry, STATUS row.
Nothing else lands from a PASS except a licence to design the O2 edge.

## 4b. Three follow-on probes — kept OUT of D-HOUSE-1 (external review, 2026-09-02, adopted)

An earlier revision of this plan folded three extra arms into D-HOUSE-1. An
external review (GPT-class, adjudicated against the code per
`E-EXTERNAL-REVIEW-ADJUDICATED-1`) objected, and the objection holds: D-HOUSE-1
asks ONE causal question — does attacking the leader and preserving dissent
improve planted-cause recovery — and five hypotheses in one harness destroy
attribution. #1136's value came from isolating that one arm was rank-inert.
So the three become separate, gated probes, each run on the SAME arenas and
null AFTER D-HOUSE-1 has reported, each graded against the D-HOUSE-1 A2
ordering, each with its own rank-inertness guard (Spearman ρ < 1 against
A2 before any p@1 is read) and its own inertness falsifier. Re-checking the
processing order also found three defects in the folded version; they are
fixed here.

- **D-HOUSE-1b — resonance prior: anchoring vs recognition.** Ingredient:
  the perspectival `awareness_dto::ResonanceDto` / `HdrResonance` shape, the
  pyramid base. TWO sub-arms, because the pyramid names a different shape
  than "seed the board": `res-seed` (resonance orders the board, RCR runs
  regardless) and `res-gate` (a unanimous, epiphany-grade resonance commits
  WITHOUT RCR; otherwise RCR runs — the faithful base-first form, and the
  more dangerous one). Falsifiers: neither sub-arm may beat the planted
  cause's rank on no-distractor arenas (a prior overriding clean evidence
  is anchoring, measured); `res-gate` must abstain on ≥ 10 % of arenas
  (a gate that always commits is not a gate).
- **D-HOUSE-1c — signed qualia as contrast agent.** Per candidate ONE signed
  i4 from the ASC outcome (+ `Revised`, − `BlockedSelfReference`,
  0 `NoTarget`), read into the existing `QualiaI4_16D` lanes (tenant 1) and
  consulted by the POST-challenge re-rank only — the dye cannot inform the
  initial board because ASC has not run yet. Order fix: ASC runs on the
  top-k (k = 3), not the leader alone; a single dyed candidate is trivially
  inert or trivially decisive and would trip the ρ guard for the wrong
  reason. No lane is minted: if no existing lane fits, the probe reports
  that and stops (OQ-CSV-1 is answered by the probe's need, never
  pre-empted). Falsifier: all signs forced to 0 must equal A2 to the bit.
- **D-HOUSE-1d — aperture (the "humor" hypothesis, held as a STYLE
  hypothesis).** The cycle has one knob, `k` in `peripheral_sample_where(k)`.
  The candidate operator is incongruity → aperture, `k(pass n) =
  f(incongruity at pass n−1)` with fixed `k` on pass 1 — the order fix: `k`
  is spent at step 2 but CR contradiction depth exists only after steps 3–4,
  so the knob can only read the PREVIOUS pass of the fixpoint loop. The word
  "humor" does not substitute for that operator; until the operator is
  defined and measured, this row is a hypothesis, not a scheduled probe.
  Inertness falsifier when it runs: raising `f` must admit a watcher on some
  arena, lowering it must silence one.

Ghost-family fence for all three (banked as
`E-A-GHOST-TRACE-IS-NOT-THE-COUNTERFACTUAL-LANE-1`): `GhostType` /
`GhostField` / `GhostEcho` / `WisdomMarker` are LINGERING TRACES (Staunen =
persistent wonder, Wisdom = harvested knowing); the −6 counterfactual lane
(`deposit_counterfactual`, `CounterfactualMailbox`) is a NON-AUTHORITATIVE
RUNG. The second may consume the first as a starting prior; it is never one
of them, and none of these probes reads `WisdomMarker` as counterfactual
state.

## 5. Deliverables

| D-id | title | scope | status |
|---|---|---|---|
| D-HOUSE-0 | this plan: parts list, composition, ladder placement, falsifier | plan-only | Shipped (this PR) |
| D-HOUSE-1 | PROBE-HOUSE-DIFFERENTIAL-1 (§4) | planner example | Queued — next step |
| D-HOUSE-1b | resonance prior probe, `res-seed` + `res-gate`, anchoring vs recognition (§4b) | planner example, same arenas | Gated on D-HOUSE-1 reported |
| D-HOUSE-1c | signed-qualia contrast probe over top-k ASC, post-challenge re-rank only (§4b) | planner example, same arenas | Gated on D-HOUSE-1 reported |
| D-HOUSE-1d | aperture hypothesis: incongruity → `k` across passes; held until the operator is defined (§4b) | hypothesis | Not scheduled |
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
