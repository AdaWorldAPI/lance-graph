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
   - F-MEP-0 (disable-verified): perturb one matrix entry in the inlined
     sandwich → parity goes red. If this cannot be made to fail, the gate
     is vacuous and W0 is not done.

**W1 — the information probe (STOP GATE for everything below; Sonnet arms,
Opus adjudication).** Over real multi-hop chains (KJV derivation chains
and/or anaphora chains with per-hop margins), rank derived beliefs by
suspicion under two arms:
- (a) **scalar baseline** — naive decay: the **product** of per-hop trust
  (predeclared above; `min` rejected), the default every stack uses;
- (b) **EWA arm** — per-hop 2×2 Σ (K2 derivation), sandwich-propagated,
  read out as a scalar (largest eigenvalue or trace).
- F-MEP-1 (anti-vacuity): the two rankings must DIVERGE non-trivially
  (Spearman ρ < 0.95 over the suspicion ordering). Identical rankings ⇒
  the seam buys nothing ⇒ **NO-BUY immediately**, numbers banked.
- F-MEP-2 (the buy signal): on the divergent subset, arm (b) must predict
  an INDEPENDENT error signal better than (a). Error signal = S4-guarded
  arena events only (Choice-on-overlap hits, revision conflicts,
  `depth=|f₁−f₂|` spikes) — never unguarded-path confidences
  (TD-NARS-REVISION-UNGUARDED's lesson as a methodological fence).
- F-MEP-3 (null control, `shuffle_beliefs_null` precedent): arm (b)'s
  advantage must beat a stamp-shuffled null on the same chains.

**The statistical protocol is PREDECLARED — every value below is fixed
BEFORE the probe runs** (CodeRabbit, #1074: unspecified choices "can be made
after results and can produce a spurious BUY"). This block is the
pre-registration; changing any of it after seeing numbers invalidates the
run and requires a re-pin with the change stated:

| knob | predeclared value |
|---|---|
| cohort | every S4-guarded arena chain of hop-length **≥ 2** (single-hop chains cannot distinguish propagation from its seed) |
| readout | **trace(Σ)** — ONE scalar, fixed. (Largest eigenvalue is NOT evaluated; picking between them post-hoc is the defect this row exists to prevent.) |
| ties | equal readout ⇒ equal rank (average-rank convention, standard for Spearman) |
| missing hop data | chain EXCLUDED wholesale, never imputed; the excluded count is REPORTED |
| minimum sample | **n ≥ 200** qualifying chains; below that the probe reports UNDERPOWERED and stops — that is a valid, honest exit |
| split | metric fixed on a held-out half; the F-MEP-2 threshold is evaluated ONCE on the other half, no re-fitting |
| comparison metric | AUC of suspicion-rank vs the binary S4 error signal |
| BUY threshold | ΔAUC **≥ 0.05** over arm (a) AND clearing the F-MEP-3 null by ≥ 2σ of the shuffle distribution |

Anything short of BOTH thresholds is NO-BUY. "Better" has no meaning in this
plan outside this table.

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
using the same `trace(Σ)` readout W1 fixed, and the W1 held-out half to set
its cut points, `TrustTexture` is `Calibrated` below the 50th percentile of
the clean-chain trace distribution, `Uncertain` between the 50th and 90th,
and `Overconfident` at or above the 90th. Ties resolve to the LOWER-suspicion
texture (the conservative direction: a tie must not manufacture a flip).
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
- F-MEP-5 (two-sided): flips must CONCENTRATE on chains W1's error signal
  flagged (not uniform noise), AND a can-stay-silent half — short/clean
  chains must not flip (a gate that flips everything is the 150/150
  defect).
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
