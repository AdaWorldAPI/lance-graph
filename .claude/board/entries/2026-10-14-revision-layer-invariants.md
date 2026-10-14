# 2026-10-14 — Revision layers: Gadamer gate invariants, Rubikon goal revision

**Status:** DECISION (approved 2026-10-14, points 1–4). The invariants are
TEST-PINNED; points 2–4 are specifications whose tests are OPEN.

## 1. Gadamer revision: invariants that must never be touched

`lance_graph_contract::revision::GadamerRevision` decides **whether** and
**in which sense** an encounter revises an interpretive horizon. It is set
algebra over masks (union, intersection, difference, subset over claims,
roots, resistance and contradictions); it never counts.

| # | invariant | pinned by |
|---|---|---|
| G1 | An echo or a closed cycle earns zero evidential weight (`Echo`, `ClosedCycle` → `NoIncrease`). | `revision.rs`: `repeated_derived_claim_is_an_echo_and_adds_no_weight`, `self_supporting_ancestry_is_a_closed_cycle`; `planner/tests/gadamer_isa_revision.rs`: `echo_cannot_mint_confidence`, `a_closed_cycle_cannot_mint_confidence` |
| G2 | A synthesis must be earned: `HorizonFusion` needs a new independent root; the same contradiction without one is `ContradictionPreserved` → `Suspend`. No `IncreaseEligible` kind is reachable without a new root. | `contradiction_without_a_new_root_suspends_instead_of_synthesising`, `no_increase_eligible_outcome_is_reachable_without_a_new_root`, the `debug_assert` in `revise` |
| G3 | Gadamer, not Hegel: `unresolved_tension` grows by union and is never cleared. | `fusion_accumulates_tension_and_never_clears_prior_tension` |
| G4 | The evidential effect is categorical (`IncreaseEligible` / `NoIncrease` / `Suspend`), deliberately coarser than a confidence. It is never replaced by a number. | the type `EvidentialEffect` |
| G5 | The gate sits above the register: the register never sees a horizon, the gate never computes a truth value. The register alone would inflate an echo. | `gadamer_isa_revision.rs`: `the_register_alone_would_inflate_an_echo`, `new_independent_roots_are_revised` |
| G6 | Eligibility is not acceptance: commit needs `IncreaseEligible` AND a counterfactual verdict `Necessary`; revision outputs stop before any world mutation. | `RevisionVerdict::is_acceptable`, `revision_verdict_alone_cannot_reach_commit` |
| G7 | Revision is relative to a question: the horizon carries `question`, `language`, `grammar`, `codebook`, `lens` and `projected_claims` (the anticipated whole the parts are read against). | the type `InterpretiveHorizon` |

## 2. The layers, kept apart

| layer | question | operands | algebra | output |
|---|---|---|---|---|
| Gadamer gate | has understanding moved, and was it earned? | claim / root / resistance / contradiction masks | set algebra | `RevisionKind` + effect |
| CE64 register (`isa::truth`) | how much evidence is pooled? | `(f, c)` codes | arithmetic (exact codes for deduction / induction / abduction, #1548) | a number |
| folded evidence | how often is it observed? | extension masks | AND + popcount at the terminal | a number from counts |
| Rubikon (kanban phases) | committed, and did the action meet the intention? | the intention's phase | phase transitions on a gate | a phase |

Counting layers answer "how much"; the Gadamer gate answers "whether". A
counting layer runs only after the gate admits the encounter (G5).

## 3. Specifications (approved, tests OPEN)

- **R1 — goal revision only outside `CognitiveWork`.** A goal's horizon is
  revised in `Planning` (pre-Rubicon, including the veto) and in
  `Evaluation` (postactional); never in `CognitiveWork`, where the
  intention is shielded (Heckhausen). Matches `alpha-channel-rung-overlay-v1`
  §3f ("revision runs in the pre-act window"). OPEN: a test that a goal
  revision attempted in `CognitiveWork` is refused, and that
  `Evaluation → Plan` reopens the goal.
- **R2 — agreement on a goal is a `HorizonFusion`** between two horizons,
  where the other party's statement supplies the new independent root.
  Without it: `ContradictionPreserved` → `Suspend`, the disagreement kept.
  No new type. OPEN: what counts as an independent root for a person (an
  utterance, a signed decision) — a modelling decision.
- **R3 — a goal is never revised by counting.** Evidence may inform
  feasibility before the Rubicon; it never decides what the goal is.
- **R4 — evidence-layer folding** (popcount of intersections) runs only under
  the Gadamer gate and only for observational statements, never for goals.

## OPEN (found while checking, pre-existing)

`execute` writes witness (53..58) and epistemic state (59..63) as 0, and
`syllogize` writes plasticity (50..52) as `0b111` and epistemic state as 0
(`isa::contracts::{FORWARD, SYLLOGIZE}`). A derived conclusion therefore does
not carry the hydration-set direct/indirect, known/unknown or CARE bits of its
premises. Whether 0 is the right state for a derived edge is not decided.
