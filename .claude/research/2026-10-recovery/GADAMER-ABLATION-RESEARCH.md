# CARE × CE64 × Pearl × Ablation → Gadamer revision (read-only report)

lance-graph `origin/main`, read through a detached worktree. No code changed,
nothing built or run. Grades: VERIFIED = read in code and pinned by a test;
PARTIAL = exists but narrower than the seam needs; CANDIDATE = exists only in
an example/probe; MISSING = closed search found nothing.

## A. Capability map

| seam | grade | evidence |
|---|---|---|
| EpistemicState5 layout, 24 states, 8 reserved refused | VERIFIED | `epistemic_state5.rs:13-16, :68-71, :340-352`; tests `:542, :652` |
| State changes are guarded | PARTIAL | `with_topology`/`with_certification` (`:399-412`) have no guard; the doc defers it to "the evidence path" |
| Observation can never reach `Causes` | VERIFIED | `certification.rs:306-327` (Causes needs ≥2 intervention-backed sources and a randomized arm); planner test F1 `pearl.rs:690` |
| Observed vs asserted is provenance, not a state coordinate | VERIFIED as doctrine | `epistemic_state5.rs:36-39`; lives in `SupportReceipt.basis` (`causal_audit.rs`, not read) |
| Epistemic population masks | VERIFIED | `facts_population`, `COMPILED_FACTS_V1`; tests `:706, :746, :774` |
| Recipe eligibility from those masks | CANDIDATE | the law `CausalEdge64 × RecipeLaw → EligibleRecipes: u64` exists only as a doc comment (`:31-33`) and in the D-CTX-7 probe's `affordance_law.rs`; shipped dispatch never reads EpistemicState5 |
| Recipe dependency ordering from `requires()`/`writes()` | VERIFIED but vacuous | `rung_schedule.rs:188-227`; measured depth is always 1 (`:321`) |
| Hydration of an unknown intermediate | VERIFIED, narrow | `planner::pearl::hydrate` (`pearl.rs:402-425`): both bindings in the sealed chain, promotes only the path's own edge, IndirectUnknown → IndirectKnown only, certification untouched; tests F5 `:776`, `:1131` |
| Deductive closure over hydrated relations | MISSING | no transitive operation anywhere in `pearl.rs`; `pearl_junction::EdgePair::classify` only labels Chain → Deduction (`:94, :236`) |
| Deduction does not mint support | **VIOLATED** in the shipped replay | `chain_replay::replay_step` revises truth at every hop with no evidential-base check (`:189-209`); measured earlier: confidence 200 → 237 over 5 deduction hops, and revising an edge with itself raises it |
| Pearl association | VERIFIED | `pearl.rs:247-259`, earns at most CausalCandidate |
| Pearl intervention | PARTIAL | `pearl.rs:260`: a randomized trial-arm comparison, not do() on a graph |
| Pearl counterfactual | PARTIAL | `pearl.rs:265-291`: replay with one step cut; earns nothing (test F3 `:731`) |
| Confounder diagnostic | VERIFIED | `pearl.rs:292-310` Simpson's direction check; never demotes (F6 `:804`) |
| CE64 ISA intervention / counterfactual | REFUSE (by design) | `isa.rs:150` accepts 5 opcodes; `edge.rs:751-765` always `Err`; test `isa.rs:483` |
| A real SCM do() and counterfactual | exists, separate | `crates/learning/src/scm.rs`: `mutilate` (`:615`), abduction–action–prediction (`:533`), f32 CPTs; not connected to CE64 |
| Single-step ablation, detached | VERIFIED | `chain_counterfactual.rs:220-269`: cut copy, own `cast_seq` range, tagged −6, never written back (F4 `:751`) |
| First divergence of two traces | VERIFIED | `chain_replay::first_divergence` (`:343`), tests `:454, :549`; unused by the counterfactual |
| Downstream dependency cone | MISSING | no probe or module computes it |
| Four-way ablation readout | CANDIDATE | `crossword_mask_propagation_probe.rs:345-430`: Redundant / ShortcutOnly / **SilentlyNecessary** / Necessary; test `:1220` |
| Overdetermination | VERIFIED for one case | planner F7 (`pearl.rs:844`): OR gate, cut gives TruthOnly, certification stays Causes |
| Joint cuts / 64 batched scenarios | MISSING | `Edit::CutStep(usize)` is the only edit; the u64 OR gate is a test fixture (`pearl.rs:608-618`) |
| Gadamer revision of claims | VERIFIED | `revision.rs:326-450`; Echo / ClosedCycle mint nothing |
| Gadamer gates numerical revision | VERIFIED | `planner/tests/gadamer_isa_revision.rs`: Echo ×10 and ClosedCycle ×10 keep confidence at 100; new roots raise it above 200 |
| Gadamer as writer of EpistemicState5 | CANDIDATE | D-CTX-7 `revision_epistemic_writer_probe.rs:258-287`: Open → Associated on enough earned support, back on ContradictionPreserved |
| `CounterfactualVerdict` computed by anything | MISSING | only hand-built in `kanban.rs:499-522` tests; every probe uses `NotRun`; so `is_acceptable()` is unreachable on any real path |
| Horizon identity revision (question, language, grammar, codebook, lens) | MISSING by construction | `revision.rs` copies all five unchanged into `resulting`; no probe changes them |
| Persistence of a revised horizon | MISSING | the module "has no production write capability" (`revision.rs:6-7`); horizons live only in probe replays |
| Teleology / purpose attribution | MISSING, and one defect | only labels exist (`ThematicRole::Purpose`, `DependencyType::Intentional`, `WorldDomain::Intentional`); `learning/causal_bridge.rs:190` maps Intentional to (Do, Causes): purpose collapses into causation |
| Commit gate / Rubicon | PARTIAL | `rubicon_witness.rs` reads attention breadth; the commit-gate idea is `.arxiv/2606.14581` (the CARE program-synthesis paper: replay, compare, decision record, then commit), a separate source from the CARE certification vocabulary |

The CARE relational vocabulary (Associated / Related / Contributes / CausalCandidate /
Causes) is defined in board entry `2026-10-06-reasoning-band-as-relational-certification.md`
(D-GSO-7a) without the name CARE. "CARE" on the board is only `D-SCF-CARE-PAIR-0`.

## B. Two kinds of revision

| | numerical NARS revision | Gadamer revision |
|---|---|---|
| operates on | one statement's (f, c) | an `InterpretiveHorizon`: projected claims, independent roots, inherited roots, unresolved tension |
| identity | statement unchanged | horizon id and revision index advance |
| code | `isa.rs` revision; `chain_replay::replay_step`; `CausalEdge64::learn` | `GadamerRevision::revise` |
| persistence | CE64 F/C lane; per #1466 a sparse SET record | none yet |
| guard against self-support | none in `replay_step` | `has_new_root`; Echo, ClosedCycle; inherited roots never count |
| what it can change | how much is believed | which claims are the working whole, and which tension is held |
| what it cannot change today | — | the question, language, grammar, codebook or lens |

So the current policy is a revision of claims inside a fixed horizon, not a full
horizon revision. The only verified link between the two is the
`gadamer_isa_revision` test: numerical revision runs only on `IncreaseEligible`.
Note one wording risk: the doc on `HorizonFusion` calls it "thesis × antithesis ×
synthesis". The behaviour is Gadamerian (tension accumulates by union, never
cleared), but the vocabulary invites the Hegelian misreading.

## C. Teleology

Nothing distinguishes intentional goals, functions, means–end inference,
retrospective rationalisation or unsupported purpose attribution. What is
representable today:

- a purpose as a **claim bit** inside a horizon, with its roots, so it can be
  withdrawn, exposed or preserved in tension by `GadamerRevision`;
- a cause, earned by the certification ladder.

What is not: any evidence class that licenses a purpose (intent evidence,
selection history for a function), so a purpose claim can only be supported by
the same roots that support a cause. The live defect is
`causal_bridge.rs:190`, which turns "A intends B" into a causal certification.
`.arxiv/2607.21273` is the only doctrine: a predictor may update belief, never
purpose; its falsifiers are not implemented.

The ablation scenario ("X in order to Y; remove X; Y survives via Z") maps
cleanly onto existing readouts: chain cut gives TruthOnly or Inert at the
terminal, the crossword readout separates Redundant from SilentlyNecessary,
and F7 already shows overdetermination keeping `Causes`. What is missing is the
step that turns that result into resistance against the purpose claim: nothing
connects an ablation result to `EncounterEvidence.resistance`.

## D. Ablation as introspection

Composable from existing parts, measured separately:

1. **Reachability / closure:** run the factual and cut traces through
   `replay_chain`; `first_divergence` locates the first changed step.
2. **Cone:** the steps whose `(predicate, edge)` differ after that index; a
   per-step XOR of the two traces. Not implemented.
3. **Evidential support:** per-step (f, c) difference, already in the trace.
4. **Certification:** `pearl::reason` with `Edit::CutStep` gives LoadBearing /
   TruthOnly / Inert; F3 pins that a load-bearing cut is not `Causes`.
5. **Terminal verdict:** `chain_counterfactual::EdgeRole::is_load_bearing`.
6. **Uniqueness / redundancy:** the crossword four-way readout; its
   SilentlyNecessary case is exactly "the answer survives but now rests on
   something else".

Overdetermination needs joint cuts. The edit type supports one cut. A u64 mask
can carry 64 cut scenarios for Boolean reachability over a step mask, but that
is reachability only; each scenario's CE64 truth replay is a separate run and
must be counted as such.

`.arxiv/2607.21573` (necessity by disruption), `2202.08816` (necessity and
sufficiency) and `2603.04209` (search must exceed edge deletion) are the
literature this lines up with; the last one says single-edge deletion alone is
not a complete counterfactual search, which is the current limit.

## E. 64 logical lanes

Do not map 64 lanes onto the AlphaTunnel. Its 10 lanes are the recipe rungs
(`rung_schedule::LEVELS = 10`, `MAX_RUNG = 9`); widening it would redefine rungs.
The five axes stay separate: rung (≤ 10), CE64 bit position, SIMD lane, cut
scenario ordinal (could be 64, one bit each), Alpha overlay owner.

The 34 recipes do not cover the sequence. Their `requires()`/`writes()` range
over eight scalar thought fields (Sd, FreeEnergy, Dissonance, Temperature,
Confidence, Rung, Candidates, Beliefs); none reads or writes EpistemicState5, a
chain, a Pearl measurement or a horizon. The stages map onto operations, not
recipes:

| stage | existing operation | status |
|---|---|---|
| prior horizon | `InterpretiveHorizon` | VERIFIED type, no persistence |
| relational classification | `certification.rs` ladder | VERIFIED |
| epistemic projection | `Epi5Reading::project` | VERIFIED |
| eligibility | `facts_population` + affordance law | CANDIDATE |
| hydration | `pearl::hydrate` | VERIFIED, narrow |
| deduction / induction / abduction | ISA opcodes, `pearl_junction` label | VERIFIED op; closure MISSING |
| Pearl | `pearl::reason` | PARTIAL (no graph do()) |
| ablation | `chain_counterfactual` + `first_divergence` | PARTIAL (no cone, single cut) |
| teleological test | — | MISSING |
| Gadamer revision | `GadamerRevision::revise` | VERIFIED for claims |
| admission | `RevisionVerdict::is_acceptable` | unreachable: verdict never computed |

Ordering would have to come from these operations' real inputs and outputs, not
from the recipe schedule, which currently has depth 1.

## F. Falsification plan

| test | already covered by | still needed |
|---|---|---|
| T1 numerical ≠ hermeneutic | inverse only (`gadamer_isa_revision`) | revise (f, c) with a fixed encounter; assert `RevisionKind` is not Fusion/Expansion |
| T2 ablation exposes an assumption | — | chain whose claim rests on one step; cut; feed the lost claim as resistance; expect AssumptionExposed |
| T3 false purpose attribution | — | after fixing or bypassing `causal_bridge.rs:190`: an Intentional claim with only associational roots must not certify |
| T4 echo / closed cycle | VERIFIED (`gadamer_isa_revision`, `recipe_quartet` replay → Echo) | — |
| T5 genuine expansion | VERIFIED (`observation_revision_probe`) | — |
| T6 fusion keeps tension | contract only (`revision.rs` union) | assert `unresolved_tension ⊇ prior ∪ contradictions` after a Fusion |
| T7 redundant causation | VERIFIED (F7, crossword test) | joint cut |
| T8 question / lens revision | cannot be expressed | the falsifier is the type itself: `resulting.question == prior.question` always; a test case where the claims cannot be stated under the old codebook would show the gap |
| T9 replay invariance | VERIFIED (`chain_replay` tests, F9, alpha byte-identity) | — |
| T10 cross-layer | PARTIAL (`quack/tests/spofc_cycle.rs`: revision lowered to mask-risc equals the reference) | add the ablation path |

Failure criteria: any IncreaseEligible without a new root; any certification above
CausalCandidate from a cut or an association; any change to recorded observations
from a counterfactual arm; any T8 case resolved by silently reusing the old lens.

## G. Smallest missing seam

**Nothing computes `CounterfactualVerdict`.** `pearl::reason` produces
`Reaction::{LoadBearing, TruthOnly, Inert}` and `chain_counterfactual` produces
`EdgeRole::is_load_bearing`, but no code turns either into
`RevisionVerdict.counterfactual`. Every real path therefore ends at `NotRun`,
and `is_acceptable()` (the admission decision) is unreachable end to end.

The minimal experiment, one planner test, no production change:

```
CE64 edge (IndirectUnknown × Related)
  → hydrate (IndirectKnown)
  → pearl::reason(SPO, Edit::CutStep(k))  → Reaction
  → Reaction → CounterfactualVerdict
  → EncounterEvidence from the measurement
  → GadamerRevision::revise → RevisionVerdict { effect, counterfactual }
  → is_acceptable()
```

Two decisions block it, both semantic:

1. **TruthOnly.** LoadBearing → Necessary and Inert → Dispensable are clear.
   TruthOnly (the truth moved, the verdict did not) is neither; mapping it to
   Dispensable hides exactly the overdetermination case, mapping it to Necessary
   overclaims. It may need to stay NotRun, or the verdict needs a fourth reading
   (the crossword probe already distinguishes four).
2. **Claim ↔ step linkage.** `recipe_quartet_probe:506-540` shows that ablating
   a horizon's roots is degenerate (it marks every eligible encounter Necessary).
   The attack must cut the chain step that supports a claim bit, so the test
   needs a declared map from claim bits to chain steps. That map does not exist
   (the quartet probe names it as P7).

Out of scope until those two are decided: the cone, joint cuts, teleology,
horizon-identity revision, and the deduction-mints-support defect in
`replay_step`, which should be its own fix.

## Central question

The substrate can revise which claims form the working interpretation, preserve
tension, and refuse echoes and self-support. It cannot today revise the
question, language, grammar, codebook or lens, cannot attribute purpose
separately from cause, and cannot carry an ablation result into the revision.
The decisive missing link is not a new engine: it is the mapping from an
existing ablation reaction to the existing counterfactual verdict, plus the
claim-to-step map that makes that ablation a real attack.
