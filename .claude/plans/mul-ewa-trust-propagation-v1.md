# mul-ewa-trust-propagation-v1 — trust is point-wise today; the sandwich is a CANDIDATE propagation operator

> **Status: PROPOSED — PLAN/BOARD ONLY. Measure-before-carve.** No contract
> change, no wiring, until W1's numbers land (the STOP rule in §4). Same
> ratified shape as `token-value-tenant-v1` (#1072).
> **Operator directive (2026-08-28):** *"cognitive speedbumps" > check
> epistemic potholes > revision in kanban y rubicon model* + *"check for
> synergies MUL <> EWA"* + *"please explore for possible integration plan."*
> **Arc:** D-MCAL (#1065–#1070) → `mul-calibration-not-verdict-v1` (thesis
> untouched by this plan; this plan produces CALIBRATION DATA for it) → this.

## §0 The thesis, one paragraph

MUL answers "how much do I trust HERE": `MulAssessment` is scalar-only —
`TrustQualia { value: f64, texture }`, `DkPosition`, `Homeostasis`,
`free_will_modifier` (`contract/src/mul.rs:50-61`, verified at HEAD). Nothing
in the workspace answers "what does that trust become N hops away." jc's EWA
sandwich — `Σ_path = M_n·…·M_1·Σ_0·M_1ᵀ·…·M_nᵀ`, Pillar 6 (2×2, tightness
1.467× ≤ 1.75) / Pillar 7 (3×3, PSD ≥ 0.999) — is the **candidate** for that
operator: *filling indirect unknowns*, applied to trust itself.
**Wording discipline, held throughout** (CodeRabbit, #1074): "certified"
applies ONLY to jc's *numerical* properties (PSD-preservation, tightness),
never to the *semantic* claim that trust composes this way — that claim is
exactly what EPIPHANIES:12867 defers to jc, and what W1 measures. And the
kanban×Rubicon model already has the revision exit this would calibrate:
`KanbanColumn::Plan = 4` — "re-enter Planning **carrying the witness**" — the
epistemic-pothole handler. The question this plan measures before building
anything: **does sandwich-propagated uncertainty rank multi-hop derived
beliefs differently — and better — than the naive scalar decay every stack
defaults to?** If not, NO-BUY, numbers banked.

## §1 What is established (verified at HEAD this session, file:line)

- **Zero MUL↔EWA wiring exists.** No `jc`/`ewa`/`sandwich` reference in
  `contract/src/mul.rs` or `planner/src/mul/*`; no MUL reference in `jc`
  (grep hits are the substring "mul" in multiplication contexts). No
  variance/covariance/propagation surface anywhere in MUL.
- **jc is zero-dep BY CONSTITUTION** (`crates/jc/Cargo.toml`: "Default build
  is zero-dep — honors the standalone constitution"; `sigker` is the one
  optional feature). A probe consuming real chains therefore CANNOT live in
  jc — see §3 placement.
- **The sandwich is certified, twice**: `jc::ewa_sandwich` (Pillar 6, 2×2 —
  "certified by `cargo run --release --example prove_it` at tightness
  1.467× ≤ 1.75", cited in `jc/examples/splat_to_ewa_bridge.rs`) and
  `jc::ewa_sandwich_3d` (Pillar 7: PSD-preservation ≥ 0.999, Smith-1961
  closed-form eigendecomposition, CV formula `√(2/n)·√(1+3σ²n)`).
- **The revision exit exists and is forward-only-preserving**:
  `KanbanColumn::{Planning=0, CognitiveWork=1, Evaluation=2}` +
  3-way terminals `{Commit=3 (DECLARED-UNWIRED calcify), Plan=4 (re-enter
  Planning carrying the witness), Prune=5 (Libet free-won't)}`;
  `advance_on_gate(GateDecision)` gates on LOCAL axes only
  (`GateDecision::from_axes(TrustTexture, FlowState)`, #1068).
- **Lawful-composition tools ship one per layer, unconnected**: the S4
  disjointness guard (`deepnsm-v2::belief::BeliefArena::revise_at` —
  DISJOINT stamps → `Revised{synthesis_c, depth=|f₁−f₂|}`, OVERLAPPING →
  Choice); the EWA sandwich (jc); Fisher-z averaging
  (`contract::distance::similarity_z`). Same theme three times: uncertainty
  composes lawfully or it inflates.
- **The composition-legality question is ALREADY assigned to this pairing**:
  EPIPHANIES:12867 — where roll-up applies "is a MATH question OWNED BY THE
  JC CRATE (jirak/pearl/ewa_sandwich), **deferred**." This plan produces
  measured INPUT to that question; it does not answer it.
- **Real multi-hop data exists, with a poisoned-well caveat**: the KJV
  arena carries real derivation chains (92,464 derived statements, F1
  EPIPHANIES:10383; 35,613 pronoun bindings with per-hop SelectionalFit
  margins). BUT `TD-NARS-REVISION-UNGUARDED` rules the planner's
  revise-all-history confidences "suspect upward" — so ground truth in W1
  comes ONLY from the S4-guarded arena, never the unguarded paths.

## §2 Delineation — what this plan deliberately does NOT collide with

| neighboring artifact | relationship |
|---|---|
| `mul-calibration-not-verdict-v1` (PROPOSAL, live) | **Feeds it, never redefines it.** That plan owns MUL's output identity ("calibrates, does not adjudicate"). Path-propagated Σ is calibration DATA for that thesis. No output vocabulary is touched here. |
| `ISS-MUL-GATE-NAMED-FOR-THE-WRONG-LAYER` + F-MUL-6 | **Untouched.** The rename stays blocked on F-MUL-6; this plan renames nothing and adds no gate variant. |
| `dialectic-engine-v1` (ACTIVE) | The S4-guarded `BeliefArena` is this plan's INSTRUMENT (ground-truth source), never modified. |
| `TD-NARS-REVISION-UNGUARDED` | **Not bundled.** Its payment path is already prescribed (thread the stamp guard through ndarray's callers). This plan only inherits its LESSON: unguarded confidences are not ground truth. |
| tarski-markov-hhtl-seam register (HELD) | Untouched. No fold, no accumulation mechanism, no rung/stamp delegation proposed. |
| `token-value-tenant-v1` (#1072) | Sibling shape, disjoint domain. Shared discipline only. |

## §3 The carrier decision (deferred to W1, both candidates named)

Minimal carrier for a trust second-moment, BOTH additive-DTO-only — **no
`ValueTenant` carve, no layout change, no `ENVELOPE_LAYOUT_VERSION` bump is
proposed anywhere in this plan**:

- **K1 — `TrustSigma { s11, s12, s22 }: Option<_>` on `TrustQualia`** — a 2×2
  SPD over (value, calibration). Pillar 6 is literally the certified 2×2
  case. `None` ⇒ today's behavior byte-identical (zero-fallback).
- **K2 — per-hop Σ derived, never stored** — the probe computes Σ from
  existing per-hop quantities and only the READ is new. If W1 shows K2
  suffices, K1 is never minted at all.

**K2 is UNDERSPECIFIED as stated, and defining it is D-MEP-1's first
deliverable, not an assumed input** (codex P1, #1074): a symmetric 2×2 Σ has
**three** free values (`s11, s12, s22`) and sandwich propagation additionally
needs a **hop transform `M_k`** — while `NarsTruth` supplies only TWO scalars
(`frequency`, `confidence`) and the alternative source (SelectionalFit margin)
is a single scalar. So W1 cannot begin by "computing Σ"; it must first fix,
and write down, a construction:

- **Σ₀ (the seed):** which two quantities are the axes, and what the
  off-diagonal `s12` means. A diagonal seed (`s12 = 0`) is the honest default
  — it asserts no measured correlation — and must be declared as such rather
  than smuggled in.
  **Positive-trace precondition (CodeRabbit, #1074):** `s12 = 0` alone does
  NOT make the readout well-defined — a seed with `s11 = s22 = 0` has
  `trace(Σ₀) = 0` and the normalized readout below divides by zero. The seed
  construction MUST therefore also satisfy `trace(Σ₀) > 0` (equivalently, at
  least one variance axis is strictly positive). A chain whose seed fails
  this is **EXCLUDED wholesale and its count REPORTED**, exactly like a chain
  with missing hop data — never silently mapped to 0, 1, or NaN, each of
  which would masquerade as a measured propagation result. Note the exclusion
  is not cosmetic: a zero-variance seed asserts perfect certainty about both
  axes, so there is no uncertainty for the operator under test to propagate,
  and including it would score arms (a) and (b) as tied for a reason that has
  nothing to do with propagation.
- **`M_k` (the hop transform) — PREDECLARED:** `M_k = √(per-hop trust)·I`
  (isotropic) is the CONTROL form; the treatment is a margin-scaled
  anisotropic form, declared in D-MEP-1 before any run.
- **The scalar baseline is PREDECLARED as the PRODUCT of per-hop trust, and
  `min` is explicitly rejected** (CodeRabbit, #1074). This is forced, not
  chosen: under `M_k = √(t_k)·I` the sandwich gives
  `Σ_n = (∏ t_k)·Σ₀`, so trace scales by **∏ t_k** — the product, never the
  minimum. Baselining against `min` would make arm (b) differ from arm (a)
  for a reason that has nothing to do with propagation, manufacturing a
  spurious BUY. Readout normalization is fixed with it: both arms report
  `trace(Σ_n)/trace(Σ₀)`, a unitless ratio, so the two are on one scale.
- **F-MEP-1b (construction-honesty gate):** if arm (b) with an isotropic
  `M_k` and a diagonal Σ₀ is algebraically equivalent to arm (a), the probe
  MUST report that equivalence rather than a spurious difference — and any
  measured divergence then comes only from a NON-isotropic `M_k`, which is
  the claim actually under test.

**Probe placement**: `deepnsm-v2/examples/` (sole dep = contract; the real
chains + S4 arena live there). The 2×2 sandwich math is INLINED in the
probe (~15 lines) — jc's own precedent (`ewa_sandwich_3d` is "self-contained
in f64 … without any dependency on the graphics crate") — and W0 gates the
inlined math against jc's certified output so the math is never silently
forked.

## §4 Waves — Opus filigree / Sonnet grind / Haiku contract-gated churn

**W0 — parity anchors (Sonnet).**
1. Run jc's Pillar-6/7 provers green in THIS checkout (`prove_it`); bank the
   tightness/PSD numbers.
2. Inline-sandwich parity gate: the probe's 2×2 math must reproduce jc's
   certified propagation on identical seeded inputs.

   **The executable mechanism is a CHECKED FIXTURE, because the probe cannot
   call `jc` and must not re-type its math** (CodeRabbit, #1074 — this was
   raised, the thread was resolved, and the mechanism never actually landed;
   the requirement was stated without a way to run it). The dependency facts
   that force this shape, verified in-tree rather than assumed:
   - `crates/jc` is a workspace MEMBER, zero-dep by default; its production
     constitution is standalone.
   - `crates/deepnsm-v2` is workspace-EXCLUDED and carries its own
     `[workspace]` table, with `lance-graph-contract` as its SOLE dependency
     — so the probe cannot reach `jc` at all, and adding that edge would
     break the very property the crate's own manifest comment defends.

   So the comparison is staged, not linked:
   - **Generator (jc side):** a `jc` example calls the REAL
     `jc::ewa_sandwich` over a fixed seeded input set and emits a small
     committed table of `(Σ₀, M_k…, Σ_n)` triples in f64 hex (bit-exact, no
     decimal round-trip), stamped with the **jc source commit** it was
     generated from.
   - **Assertion (probe side):** the deepnsm-v2 probe's inlined 15 lines run
     the same inputs and must reproduce the committed outputs bit-for-bit.

   **F-MEP-0 stays genuinely falsifiable under this shape**, which is the
   reason a jc-side harness that *re-types* the probe's math was rejected:
   with two independent copies, perturbing the probe's copy would leave the
   harness's copy — and therefore the comparison — untouched, so the gate
   could not fail and would be vacuous by construction. Against a committed
   fixture, perturbing one matrix entry in the inlined sandwich turns the
   assertion red, because the fixture is not derived from the code under
   test.
   - F-MEP-0 (disable-verified): perturb one matrix entry in the inlined
     sandwich → parity goes red. If this cannot be made to fail, the gate
     is vacuous and W0 is not done.

   **Staleness guard — the one real cost of a fixture over a live call.** A
   committed table can silently drift from a `jc` that has since changed.
   The stamp is what makes drift detectable rather than invisible: W0 is NOT
   done unless it matches the `jc` source in the checkout, and a mismatch
   REGENERATES the fixture rather than being waived. (Step 1 already runs
   jc's own Pillar-6/7 provers green in this checkout, so a jc that has
   genuinely regressed is caught there, not here.)

   **A commit hash alone is NOT a sufficient stamp** (CodeRabbit, #1074). A
   commit id describes what was *committed*, not what is on disk: a fixture
   generated from a dirty working tree — uncommitted edits under
   `crates/jc` — carries a stamp that matches perfectly while the bytes it
   was produced from exist nowhere in history. The guard would report
   "current" for a fixture nobody can reproduce, which is worse than no
   stamp, because it manufactures confidence. Both halves are therefore
   required:
   - **Content hash:** the stamp includes a hash of the actual
     `jc::ewa_sandwich` source text the generator ran against, and W0
     verifies it against the checkout's current bytes — so ANY edit,
     committed or not, invalidates the fixture.
   - **Clean-tree requirement:** the generator REFUSES to emit a fixture
     while `crates/jc` has uncommitted changes, so a stamped artifact always
     corresponds to a reachable commit as well as to specific bytes.

   The two are not redundant: the content hash catches a stale fixture at
   *verification* time, the clean-tree check stops an irreproducible one from
   being *created*.

**W1 — the information probe (STOP GATE for everything below; Sonnet arms,
Opus adjudication).** Over real multi-hop chains (KJV derivation chains
and/or anaphora chains with per-hop margins), rank derived beliefs by
suspicion under two arms:
- (a) **scalar baseline** — naive decay: the **product** of per-hop trust
  (predeclared above; `min` rejected), the default every stack uses;
- (b) **EWA arm** — per-hop 2×2 Σ (K2 derivation), sandwich-propagated,
  read out as the ONE scalar the protocol table fixes:
  `trace(Σ_n)/trace(Σ₀)`. (An earlier draft here read "largest eigenvalue
  or trace" — that left the readout unfixed at the arm definition while the
  table below fixed it, i.e. exactly the post-hoc choice the
  pre-registration exists to close. Trace is the readout, normalized,
  everywhere in this plan.)

**The observation unit is ONE per chain** (CodeRabbit, #1074). Each
qualifying chain contributes **exactly one** suspicion score per arm and
**exactly one** binary S4 label; the label is `1` if the chain carries **≥ 1**
S4 error event and `0` otherwise. Without this, a chain with many S4 events
would enter the AUC repeatedly and silently weight the result by event
count — and the per-half floors would be counting two different things in
the two halves. Event COUNTS are still reported (they are the input to the
F-MEP-5 concentration statistic below), but they never become weights, never
become multiple observations, and never enter the AUC.

- F-MEP-1 (anti-vacuity): the two rankings must DIVERGE non-trivially
  (Spearman ρ < 0.95 over the suspicion ordering). Identical rankings ⇒
  the seam buys nothing ⇒ **NO-BUY immediately**, numbers banked.
  **Zero-variance guard (CodeRabbit, #1074):** Spearman ρ is UNDEFINED when
  either arm's suspicion scores are all tied — the rank-variance denominator
  is zero and the result is NaN, which must never reach a comparison against
  `0.95` (in IEEE 754 every such comparison is false, so an unguarded NaN
  would silently read as "diverged" and wave the run through the gate whose
  entire job is to catch a non-divergent pair). Before computing ρ the probe
  REPORTS each arm's **distinct-score count**; if either arm has `< 2`
  distinct scores the run stops as **UNDERPOWERED** — not NO-BUY, since an
  all-tied arm is a statement about the construction or the cohort, not a
  measured verdict on the operator.
- F-MEP-2 (the buy signal): on the divergent subset, arm (b) must predict
  an INDEPENDENT error signal better than (a). Error signal = S4-guarded
  arena events only (Choice-on-overlap hits, revision conflicts,
  `depth=|f₁−f₂|` spikes) — never unguarded-path confidences
  (TD-NARS-REVISION-UNGUARDED's lesson as a methodological fence).
- F-MEP-3 (null control, `shuffle_beliefs_null` precedent): arm (b)'s
  advantage must beat a stamp-shuffled null on the same chains.
  **Fully specified, because "beat a null by 2σ" names neither a
  distribution nor a side** (CodeRabbit, #1074):
  - **Redeals: `N_null = 1000`**, seeds `base_seed + i` for `i` in `0..1000`,
    so the null set is reproducible. The SplitMix64 Fisher-Yates algorithm
    and seed formula are `shuffle_beliefs_null`'s, unchanged.
  - **The permuted unit is the CHAIN-LEVEL BINARY LABEL, not the individual
    `(p, o, n)` record** (CodeRabbit, #1074). `shuffle_beliefs_null` permutes
    records, but W1's observation unit is one label per chain (fixed in the
    previous round), and permuting records does NOT preserve the chain-level
    class counts — a redeal could concentrate several events onto one chain
    and empty another, so the null distribution would be over a *different*
    quantity than the observed statistic and could not bound it. Each redeal
    therefore permutes the half-1 label vector, which preserves the positive
    and negative CHAIN counts exactly in every draw. That is what makes it a
    genuine permutation null: only the *pairing* between suspicion score and
    label is destroyed, never the class balance.
    (Noted as an interaction, not a defect in `shuffle_beliefs_null` — the
    function is right for the corpus-scale belief-structure question it was
    written for; it is the unit that had to follow W1's observation unit.)
  - **Degenerate redeal behaviour.** Because the permutation preserves class
    counts, a redeal can only be single-class if the observed half-1 data
    already was — which the per-half floors (`≥ 10` of each class) and the
    single-class guard both reject before the null is ever built. So no
    redeal can produce an undefined AUC that the observed run did not
    already fail on. If that invariant is ever violated at runtime, the probe
    stops as **UNDERPOWERED** and reports the offending draw rather than
    substituting a value for its ΔAUC — an imputed `0.0` would silently drag
    the null mean toward zero and make the `+2σ` bar easier to clear.
  - **Statistic: `ΔAUC` itself** — recomputed on half 1 under each redeal,
    giving a null distribution of the SAME quantity the BUY rule reads. A
    null over some other statistic would not bound the decision being made.
  - **One-sided.** The claim is directional (EWA predicts error BETTER),
    so the criterion is `ΔAUC_observed ≥ mean(null) + 2·sd(null)`. A
    two-sided reading would credit a significant result in the wrong
    direction.
  - **Zero null variance ⇒ UNDERPOWERED, never an automatic pass.** If
    `sd(null) == 0` the `+2σ` bar collapses onto the mean and ANY positive
    observed ΔAUC would clear it — a gate that cannot fail. The probe
    reports the degenerate null and stops.

**The statistical protocol is PREDECLARED — every value below is fixed
BEFORE the probe runs** (CodeRabbit, #1074: unspecified choices "can be made
after results and can produce a spurious BUY"). This block is the
pre-registration; changing any of it after seeing numbers invalidates the
run and requires a re-pin with the change stated:

| knob | predeclared value |
|---|---|
| cohort | every S4-guarded arena chain of hop-length **≥ 2** (single-hop chains cannot distinguish propagation from its seed) |
| readout | **`trace(Σ_n)/trace(Σ₀)`** — ONE scalar, fixed, and **normalized**; the seed's `trace(Σ₀) > 0` precondition above is what makes it well-defined. Raw `trace(Σ_n)` is NOT the readout: it is not comparable across chains with different seed magnitudes, so ranking on it would sort partly by how uncertain a chain STARTED rather than by what propagation did to it (CodeRabbit, #1074 — caught where W3 had inherited the raw form). Largest eigenvalue is NOT evaluated; picking between readouts post-hoc is the defect this row exists to prevent. **Every wave uses this one scalar** — W1's ranking and W3's `p50`/`p90` cut points alike. |
| ties | equal readout ⇒ equal rank (average-rank convention, standard for Spearman) |
| missing hop data | chain EXCLUDED wholesale, never imputed; the excluded count is REPORTED |
| minimum sample | **n ≥ 200** qualifying chains **in total, AND per-half floors that the total does not imply** (CodeRabbit, #1074): each half independently needs **≥ 50 chains, ≥ 10 positive and ≥ 10 negative S4 events**. A cohort-level `n` says nothing about how it landed either side of a hash split, so the total is a necessary and NOT a sufficient condition. Below ANY of these the probe reports UNDERPOWERED with the failing count named, and stops — a valid, honest exit. The floors are declared here, before any run, precisely so they cannot be relaxed after seeing which one bites. |
| split | **deterministic, chain-level, leakage-safe**: partition key = the chain's ROOT SUBJECT id (never the chain id — two chains sharing a root would otherwise straddle the split and leak); half = `blake3(root_subject_id ‖ "mep-w1-v1")[0] & 1`. Half **0** FITS — every free choice (which suspicion ranking, any construction detail left open by D-MEP-1) is fixed here, and its ΔAUC is DIAGNOSTIC ONLY, never the number the BUY rule reads. Half **1** EVALUATES: **the ΔAUC the BUY threshold is applied to is computed on half 1 ALONE, exactly once, with no re-fitting** — reporting a half-0 ΔAUC or a pooled ΔAUC as the decision number is the defect this row exists to prevent. The literal salt is part of the pre-registration, so a re-run reproduces the identical partition. |
| comparison metric | AUC of suspicion-rank vs the binary S4 error signal |
| **degenerate AUC** | AUC is UNDEFINED when a half carries no positive or no negative S4 event, and `n ≥ 200` does **not** prevent that (CodeRabbit, #1074). Both halves' **class counts are REPORTED unconditionally**; if either half is single-class the probe stops as **UNDERPOWERED** — never NO-BUY, since a degenerate split is a statement about the cohort, not about the operator under test, and never a computed AUC on a one-class half. This is the ZERO-count guard only; the ≥ 10-per-class floors in the minimum-sample row are what stop a *technically* two-class half from producing an AUC too unstable to decide on. |
| BUY threshold | **All three, on half 1:** (i) `AUC(b) > 0.5` — the EWA arm must be predictive AT ALL, not merely less anti-predictive than the baseline. Without this, `AUC(b) = 0.20` over `AUC(a) = 0.10` clears a ΔAUC bar while both arms rank *backwards*, and the "win" is a bigger error (CodeRabbit, #1074). (ii) ΔAUC **≥ 0.05** over arm (a). (iii) clearing the F-MEP-3 null by ≥ 2σ of the shuffle distribution. An arm that is anti-predictive (`AUC ≤ 0.5`) is a NO-BUY however large its ΔAUC — and if BOTH arms land below 0.5 the probe reports that inversion explicitly, since a systematically backwards ranking is a finding about the suspicion construction, not a quiet NO-BUY. |

Anything short of ALL THREE conditions is NO-BUY. "Better" has no meaning in
this plan outside this table.

**W2 — the carrier (GATED on W1 BUY; Opus review, Sonnet transcription).**
Mint K1 ONLY if W1 showed derived-per-read Σ (K2) insufficient (e.g. a
consumer needs Σ across a boundary where the hop quantities are gone).
`Option<TrustSigma>` on the contract DTO; SPD-validity guard (refuse, never
clamp, a non-PSD write — jc Pillar 6's own invariant applied at the type
boundary).

> **⚠ `Option` DOES NOT make this additive for SOURCE compatibility** (codex
> P1, #1074 — verified at HEAD). `contract::mul::TrustQualia` is a `pub
> struct` with `pub` fields and **no `#[non_exhaustive]`**, and it is
> constructed by struct literal in-tree (`contract/src/exploration.rs:907`,
> `contract/src/mul.rs:603`, `:887`) as well as by external consumers. Adding
> ANY field — optional or not — breaks every such literal with `E0063:
> missing field`. `None` gives *behavioural* compatibility, never *source*
> compatibility. Note also there are **two** `TrustQualia` types (the contract
> one and `planner/src/mul/trust.rs:15`); W2 must state which it carves.
>
> So W2's carve is not "add a field" but a construction-path decision, and it
> is part of the deliverable:
> (a) `#[non_exhaustive]` + a constructor (itself breaking for existing
> external literals — it buys future additivity, not this one);
> (b) a side table keyed by assessment identity (zero DTO change);
> (c) **K2-only — no carrier at all**, which W1 may well render sufficient.
> **(c) is the default; (a)/(b) require W1 to have shown a boundary where the
> hop quantities are genuinely gone.**

- F-MEP-4: consumer-build gate (the F-MUL-6 method — a real compile of every
  live consumer, never a grep). Two-sided: the chosen path must compile every
  existing consumer UNCHANGED, and the disable — adding the field bare to
  `TrustQualia` — must FAIL that build with `E0063`, proving the gate can
  see the breakage it exists to catch.

**W3 — path-aware gating probe (GATED on W2 or K2-suffices; Sonnet).**

> **⚠ CORRECTED BEFORE ANY WORK STARTED (codex P1, #1074 — confirmed by
> reading the source).** This wave originally named "**the Commit→Plan flip
> rate**" as its headline metric. That metric is **unmeasurable through
> `advance_on_gate`**, because `Plan` is structurally unreachable from it:
> `advance()` is *"the first non-`Prune` successor"* and `Evaluation`'s
> `next_phases()` is `[Commit, Plan, Prune]`, so `advance()` returns
> **`Commit`, always**; `veto()` returns `Prune`; `Hold` returns `None`.
> **Precondition, stated because it is load-bearing** (CodeRabbit, #1074):
> the reachable set `{Commit, Prune, None}` holds **starting from
> `Evaluation`** — which is where W3 measures, since that is the only phase
> whose successors include `Plan` at all. From other phases `Flow` yields
> that phase's own first non-`Prune` successor (`Planning → CognitiveWork`,
> `CognitiveWork → Evaluation`, `Plan → Planning`); the claim is about
> `Evaluation`'s 3-way terminal, not about the DAG as a whole.
>
> **The finding this surfaces is worth more than the metric it cost**, and it
> is recorded here rather than papered over: **`Plan = 4` — the revision exit,
> "re-enter Planning carrying the witness", the very transition this whole
> plan is motivated by — is legal in the DAG but reachable by NO named routing
> primitive.** `advance`/`veto`/`advance_on_gate` cannot emit it; only a
> caller hand-walking `next_phases()` can. Whether that gap is intentional
> (Rubicon-forward discipline: revision must be a deliberate act, never a
> gate's automatic output) or an omission is **an open question for the
> operator — not something this plan resolves or builds.** Logged as
> `ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE`.

**The Σ → `TrustTexture` mapping is PREDECLARED, deterministic, and single**
(CodeRabbit, #1074 — "different mappings produce different flip rates"):
using the **NORMALIZED** readout W1 fixed — `trace(Σ_n)/trace(Σ₀)`, not raw
`trace(Σ)`. This correction matters and is not cosmetic (CodeRabbit, #1074):
raw traces are not comparable across chains with different seed magnitudes,
so percentile cut points over raw traces would sort chains largely by **how
uncertain they started**, handing `Overconfident` to any chain with a big
seed regardless of what propagation did to it. The normalized ratio is the
quantity that actually measures propagated change, and it is the same scalar
W1 ranks on — so the two waves cannot disagree about what was measured.

**W3 is measured on half 1 ONLY** (CodeRabbit, #1074). The two halves keep
exactly the roles the W1 split row gives them: **half 0 FITS** the cut points
(it is the clean-chain sample the percentiles are computed from) and **half 1
EVALUATES** the flip rate. Measuring the flip rate on half 0 would evaluate
outcome-derived cut points on the very data they were derived from, which
inflates the apparent effect by construction; W4 therefore consumes the
half-1 number ONLY, and any half-0 flip rate is exploratory and explicitly
not eligible for the verdict.

**"Clean chain" is defined, not assumed**: a chain **in half 0** carrying
**zero** S4 error events — the same binary signal W1's AUC uses, so the two
waves cannot drift apart on what "clean" means. Cut points are the **50th and
90th percentiles by the nearest-rank method** (`ceil(p/100 · N)`-th value of
the ascending normalized-readout list — no interpolation, so the result is
exact and reproducible across implementations). If the clean set is empty or
`N < 50`, W3 stops as **UNDERPOWERED** and reports the count, rather than
deriving cut points from a sample too small to place a 90th percentile. `TrustTexture` is then read
off **explicit, non-overlapping, exhaustive intervals**, so a value landing
exactly ON a cut point has exactly one outcome (CodeRabbit, #1074 — the
earlier wording paired "at or above the 90th ⇒ `Overconfident`" with a
ties-to-lower-suspicion rule, which assigned a value equal to `p90` two
different textures):

| condition | texture | resulting `GateDecision` † | strength |
|---|---|---|---|
| `trace ≤ p50` | `Calibrated` | `Flow` | proceed |
| `p50 < trace ≤ p90` | `Overconfident` | `Hold` | pause |
| `p90 < trace` | `Uncertain` | `Block` | veto |

† **The decision column holds for `FlowState ∈ {Flow, Transition}`, and that
is GUARANTEED for every chain this metric measures — it is not an added
restriction** (CodeRabbit, #1074, raising the unconditional reading; the
variant set below is a correction to the finding as stated).

`from_axes` is texture-AND-flow, so the column is genuinely conditional:
`(Calibrated, Anxiety)` → `Hold` via the `(_, Anxiety)` arm, and
`(Calibrated, Boredom)` → `Hold` via the `_` fallthrough. **`FlowState` has
FOUR variants** (`Flow`, `Boredom`, `Transition`, `Anxiety`), so the
condition is NOT "non-`Anxiety`" — `Boredom` masks identically. Under either
of those two states `Calibrated` and `Overconfident` BOTH yield `Hold`, so
the `p50` cut point would be structurally **inert** and only the `p90`
boundary could still produce a flip.

That case cannot arise in the measured population, by construction rather
than by stipulation. The flip-rate denominator is chains whose LOCAL arm
reaches `Commit`; `advance_on_gate` reaches `advance()` only on
`GateDecision::Flow`; and `Flow` is emitted by exactly ONE arm of
`from_axes` — `(Calibrated | Underconfident, Flow | Transition)`. So a chain
in the denominator necessarily had `FlowState ∈ {Flow, Transition}`, and
since this plan holds `FlowState` FIXED across both arms, the propagated arm
reads the same state. Every chain the metric scores therefore sits in the
regime where the column above is exact.

Stated because a reader applying this table OUTSIDE the denominator — to the
whole cohort, or to a chain that never gated `Commit` — would be misled, and
because a future change to the denominator would silently break the
guarantee rather than the table.

**The bucket order is DERIVED from `GateDecision::from_axes`, not from the
variant names' English connotations** — and getting this backwards was a real
defect in an earlier draft of this table (found 2026-08-29 while verifying a
CodeRabbit finding that was itself wrong about which enum applies; see the
note below). The verified call chain is:

`trace ratio` → `TrustTexture` (this table) → **`GateDecision::from_axes(texture, flow)`**
→ `KanbanColumn::advance_on_gate(&GateDecision)`

`advance_on_gate` takes a **`&GateDecision`** — it never sees a `TrustTexture`
directly, so this table only matters through `from_axes`, which is documented
in-source as "the ONE place this mapping lives". Reading `from_axes`
(`contract/src/mul.rs`) with `FlowState` held at its locally-assessed value
(this plan never varies it): `Uncertain` → **`Block`** under EVERY flow state;
`Overconfident` → **`Hold`**; `Calibrated | Underconfident` in
`Flow`/`Transition` → **`Flow`**.

So gate strength runs `Flow < Hold < Block`, and the mapping must be MONOTONE
in suspicion against THAT ordering. The earlier draft put `Uncertain` in the
middle bucket and `Overconfident` at the top, which meant the **most**
suspicious chains produced the **milder** intervention (`Hold`) while
moderately suspicious ones produced the strongest (`Block`) — inverting the
very quantity W3 measures, since the metric is a `Commit→{Hold,Prune}` flip
rate. The corrected order above is monotone: more propagated uncertainty
never yields a weaker gate outcome.

The naming reads oddly for one bucket and is nonetheless correct:
`Uncertain` is documented as "not enough data to assess", which is exactly a
chain whose propagated covariance has blown up — and the contract routes that
to the strongest intervention. `Overconfident` ("felt >> demonstrated") sits
below it at `Hold`.

> **⊘ On the enum this table names.** A review pass asked for these buckets to
> be mapped onto `Crystalline / Solid / Fuzzy / Murky`. That is a DIFFERENT
> `TrustTexture` — four distinct types share the name (`contract::mul`,
> `causal-edge::layout`, `lance-graph-planner::mul::trust` with a fifth
> `Dissonant` variant, and `arigraph::orchestrator`). The one on this path is
> **`contract::mul::TrustTexture`**, because that is what `from_axes` takes.
> `causal-edge/src/layout.rs` carries an explicit in-source ruling against
> building the requested cast — *"Canonical: NONE — both are domain-correct
> and should keep distinct names. Do not build a cast on the old claim."* —
> so the remap is not merely unnecessary here, it is forbidden. Recorded so a
> later session does not re-open it.

Both boundaries close DOWNWARD, which is the ties-to-lower-suspicion rule
stated as arithmetic rather than as a separate sentence that can contradict
the intervals: a chain sitting exactly on a cut point keeps the calmer
texture, so a tie can never manufacture a flip.
`Underconfident` is never produced — nothing in a propagated covariance
distinguishes it from `Calibrated`, and inventing that distinction is exactly
the coordinate-fabrication the census measured. `FlowState` is held FIXED at
its locally-assessed value in both arms, so the only varying input is the one
under test. No other mapping is evaluated.

Re-scoped metric, measurable against the surface that actually exists:
`advance_on_gate` arm comparison on the SAME chains — local axes vs
propagated Σ folded into `TrustTexture` per the mapping above — measuring the
**Commit→{Hold,Prune} flip rate**: how many multi-hop beliefs that gate `Commit` under local
trust instead hold or veto under propagated uncertainty. That is the
epistemic-pothole detector expressed in the vocabulary the routing primitives
can actually speak. Reaching `Plan` would require the open question above to
be answered first.

**The denominator is PREDECLARED** (CodeRabbit, #1074 — without it the rate is
not reproducible): it is **only those qualifying HALF-1 chains whose LOCAL
arm reaches `Commit`** (half 1 per the evaluation-population rule above),
never all qualifying chains. A chain that never gated
`Commit` locally cannot flip *from* `Commit`, so including it would dilute
the rate with cases the metric is not about — and would let the number move
purely by cohort composition. Numerator and denominator therefore share one
population. **If that denominator is zero, the rate is reported `N/A` with
the count**, never `0.0` — zero flips out of zero opportunities is not a
measurement of anything, and printing `0%` would read as "the propagation
changed nothing" when the truth is "the arm never ran."
- F-MEP-5 (two-sided) — **QUANTIFIED, because "concentrate" and "must not
  flip" name no statistic and no threshold** (CodeRabbit, #1074). Both
  populations are drawn from **half 1** and BOTH are restricted to chains
  that reach local `Commit`, matching the flip-rate denominator above:
  - **Flagged** `F` = half-1 chains reaching local `Commit` with **≥ 1** S4
    error event. **Silent** `S` = half-1 chains reaching local `Commit` with
    **zero** S4 error events.
  - **Minimum opportunities: `|F| ≥ 20` and `|S| ≥ 20`, each reported.**
    This is the load-bearing addition. Without it the can-stay-silent half
    passes AUTOMATICALLY whenever no clean chain happens to reach local
    `Commit` — `|S| = 0` gives zero flips out of zero opportunities, which
    the gate would read as "correctly silent" while having observed nothing
    at all. That is precisely the vacuous-guard defect the repo's own
    falsifiability rule exists to catch, reproduced inside the test written
    to catch it. Below either floor: **UNDERPOWERED**, not PASS.
  - **Can-fire (concentration): `flip_rate(F) ≥ 2 × flip_rate(S)` AND
    `flip_rate(F) − flip_rate(S) ≥ 0.10`.** Both, because a ratio alone is
    satisfiable at trivial magnitudes (2 % vs 1 % is a 2× "concentration"
    carrying no signal), and a difference alone would pass a gate that fires
    on nearly everything.
  - **Can-stay-silent: `flip_rate(S) ≤ 0.20`.** A gate that also flips a
    fifth of the clean chains is the 150/150 defect regardless of how well
    it concentrates.
  - These four numbers are **POLICY PINS, not measurements** — nothing has
    been measured on this cohort yet, and they are written down here only so
    they are fixed before the run rather than chosen to fit it. If the probe
    runs and the pins prove badly placed, they are RE-PINNED with the
    measurement and the reason stated, never quietly relaxed to convert a
    failing gate into a passing one.
  - **A re-pin VOIDS the run it came from** (CodeRabbit, #1074). "Re-pin
    honestly and state the reason" is not enough on its own: if the current
    run's verdict survives a threshold changed *after* seeing that run's
    numbers, the pins were effectively chosen to fit the data and the
    pre-registration bought nothing — the disclosure makes it visible, not
    valid. So the procedure is: the run that motivated the change is marked
    **VOID** and its verdict discarded (its numbers are still banked, as
    motivation for the new pins), the revised pins are declared, and only a
    FRESH run under those pins may be evaluated pass/fail. There is no path
    by which the same execution both justifies a threshold and is judged by
    it.
- **No wiring lands in W3.** It is a probe against the existing
  `advance_on_gate`; changing the gate's signature or default is explicitly
  out of scope for v1.

**W4 — verdict (Opus).** Explicit **BUY / NO-BUY** against W1-W3's numbers.
NO-BUY is a valid exit at every gate; numbers are banked either way. BUY's
consequence is also bounded: a follow-up plan for the actual gate wiring —
never wiring-by-momentum inside this one.

## §5 Fences

1. jc stays zero-dep (probe lives in deepnsm-v2; math inlined + W0-gated).
2. No `ValueTenant` carve, no layout change, no version bump — DTO-only, and
   only if W1 forces it.
3. The gate RENAME stays blocked on F-MUL-6 — untouched.
4. `TD-NARS-REVISION-UNGUARDED` is paid on its own prescribed path — not
   here; here it is only a fence on ground-truth selection.
5. The tarski register stays HELD — no fold/accumulation mechanism.
6. `KanbanColumn::Commit`'s unwired calcify is not this plan's to wire.
7. `mul-calibration-not-verdict-v1`'s thesis is consumed, not amended.
8. No GPU/wgpu; the sandwich is ~15 scalar lines.

## §6 Board hygiene (same commit as this file)

`INTEGRATION_PLANS.md` PREPEND; `STATUS_BOARD.md` D-MEP-0..4 rows (Queued);
`SUPERSESSION-INDEX.md` regenerated. Cross-refs: EPIPHANIES:12867 (the
deferred jc ownership this plan feeds), E-3DGS-MU-HYDRATION-1 (the EWA-
semiring DROP this plan does not resurrect), `token-value-tenant-v1`
(the ratified measure-before-carve shape).
