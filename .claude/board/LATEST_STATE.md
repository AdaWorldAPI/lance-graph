## 2026-08-31 — #1120 (open): D-DCR-1 replay core + the palette membrane — CONTRACT INVENTORY DELTA

**Contract inventory: +2 consts, +2 `const fn`, no new type, no layout change.**
All four in the existing `dismech_evidence` module — zero-dep, no new dependency.

- **`dismech_evidence::DISMECH_PREDICATE_FLOOR: u8`** = `0x90`, the band floor
  (`ogar_dismech::CAUSES`).
- **`dismech_evidence::DISMECH_PREDICATES: &[(u8, &str, &str)]`** — the 19
  DisMech causal predicates in slot order, `(ordinal, name, curie)`. A
  **MIRROR** in exactly the `ogar_codebook` sense: the authority is
  `ogar_dismech::RELATIONS`, this crate is zero-dep and cannot reach it, so the
  copy lives here and the drift fuse lives in the armed tier.
- **`dismech_evidence::{dismech_predicate, is_dismech_predicate}`** — position
  lookup over the contiguous band (`ordinal - FLOOR`, mirroring
  `ogar_dismech::by_index`), fail-closed outside it.

**Why the contract and not the planner:** a recorded causal chain addresses each
step by a `u8` predicate ordinal, and the replay arithmetic never reads it — it
is the step's ADDRESS, not an operand. So nothing on the hot path would notice a
corrupt one, and a chain carrying `0xA3` (the palette's SEARCH band, not a
relation) would trace to a byte-identical, meaningless result. The mirror gives
the byte a checkable domain without the planner depending on OGAR.

**Armed tier (`lance-graph-ogar`, workspace-EXCLUDED):**
`parity::assert_dismech_palette_parity()` — the drift fuse, both directions plus
floor and length, against the real palette; `ogar-dismech` added as a git dep
alongside the existing `ogar-loco`. Forward catches a stale mirror row; reverse
catches a newly minted predicate the mirror never learned (the silent one — a
replay would refuse a legitimate ordinal and nothing in the planner could tell
that from a corrupt chain).

**Planner:** `dismech_replay` (new module, D-DCR-1) + `chain_step_predicate`.

Gates: contract lib **1282** green, clippy `--all-targets -D warnings` clean;
planner lib **375** green, lib clippy `-D warnings` clean; armed tier **75**
green; fmt clean on all three. Six disable-runs red-then-green (3 on the replay
core, 3 on the mirror/fuse: a corrupted name, a swallowed search op, a dropped
predicate).

**Pre-existing, NOT introduced here:** `lance-graph-ogar` fails
`clippy --all-targets -D warnings` on `main` too (7 errors — unused `NAMESPACE`
consts in the bridge scope-lock tests, doc-list indentation in `bridges/mod.rs`
and `rbac_impl.rs`). Confirmed by stashing this diff. Its CI line is
`cargo test`, not clippy.

---
## 2026-08-31 — #1099 MERGED (ae24f6e5): D-MAR-1 mask algebra — CONTRACT INVENTORY DELTA

**Contract inventory: +4 methods, no new type, no layout change.**

- **`class_view::FieldMask::{difference, is_subset_of}`** — `#[inline] pub const fn`, mirroring the existing `intersect`/`union`/`is_disjoint` shape exactly; `Copy` preserved, no existing signature touched. `a.difference(b)` = `a & !b`; `a.is_subset_of(b)` = `a & !b == 0`.
- **`class_view::WideFieldMask::{difference, is_subset_of}`** — same pair, `(Small, Small)` fast path plus the tier-agnostic fold. Argument order deliberately **differs from the raw Intel intrinsic**: `_mm*_andnot_si*(a, b)` is `!a & b`, ours is `self & !other` ("self minus other"), documented at every definition.
- **`chunks_view` REMOVED** (private, 2 call sites, both in `zip_fold`). It cloned *both* operands into fresh `Vec`s per wide op; `zip_fold` now reads in place via `chunk_at` and allocates only the owned result. `intersect`/`union` inherit the fix. Its materialization role is superseded by the stencil arena (operator-ruled 2026-08-31: **arena separate, masks stay values** — so `WideFieldMask`'s public surface is unchanged and the 11 files across 4 crates that consume it, RBAC's `field_mask()` included, are untouched).

Why these are substrate-tier and not consumer-side, per the plan's F4 STOP rule: the rejected external draft hand-rolled an entire `EvidenceMask` trait *because* difference and containment were absent. The fix is the two methods.

**Not shipped:** D-MAR-2 (`RevisionKind`) — gated on the plan's §5 Q1 module-home ruling per its own ordering rule F6.

Gate: contract lib **1251** green; clippy `--all-targets -D warnings` clean; fmt clean; four disable-runs red-then-green. Re-verified after rebase onto `c6647fab` — the introduced patch is byte-identical pre/post rebase.

**Below the contract:** `AdaWorldAPI/ndarray` (branch `claude/medcare-rs-continue-ufsazd`, `fd5c66f` + `fbe2d36`, **not yet PR'd**) carries the masking primitives these compose with — `U64x8`/`U32x16` `andnot` + `ternlog`, lowering to a single `vpternlogq` on AVX-512 and aligned `ymm` work on v3. Nothing in the contract depends on that landing.

---
## 2026-08-31 — #1101 MERGED (c6647fab): fmt baseline + gate for three excluded crates

`jc`, `sigker`, `thinking-engine` are workspace-**excluded**, so `cargo fmt --all` never reached them and nothing held them formatted — 214 rustfmt diffs accumulated. All three formatted and given explicit `--manifest-path` CI steps in `style.yml`, matching the pattern already used for `causal-edge`/`deepnsm`/`deepnsm-v2`. The `format` job passed on first run, so the new gates are proven, not assumed.

**Measured, against the obvious hypothesis:** the drift was NOT a toolchain artefact. rustfmt is `1.9.0` under *both* the pre-bump 1.95.0 and the pinned 1.97.1 (only the build hash differs), and both report the identical 214 on identical code — no formatting-rule change across `b2b08b07` to blame. The code was simply never formatted.

Also removed one dead declaration that blocked the sweep: `container_bs/mod.rs` declared `#[cfg(test)] pub mod tests;` for a `tests.rs` that **never existed in git** (declared in the ladybug-rs import `582a6e7b`). It compiles only because `container_bs` sits behind `#[cfg(feature = "wip")]`; rustfmt walks files ignoring `cfg`, so it aborted the whole run.

**Open, found here and NOT fixed:** `sigker`'s test build has never compiled — it sorts a `(Vec<usize>, f64)` and `f64` is not `Ord` (E0277). Verified against unmodified `sigker`. Workspace exclusion meant CI never built it. The gate added is formatting-only; the tests remain dead. See `ISSUES.md`.

(#1100 opened for the one-file subset of this and closed unmerged, superseded.)

---
## 2026-08-30 — #1085 MERGED (4cc7d71c): §11.3 method transfer — DOC/BOARD ONLY

| PR | merge | content |
|---|---|---|
| **#1085** | `4cc7d71c` | D-ARW-0 §11.3: the stockfish-rs oracle was cited for its ROLES and never for a probe (0 hits for `morton_ka`/`rung_hindsight`/`temporal_replay` across 203 plan files, as of the pre-§11.3 commit). Imports the method — recall@k vs chance with an EXTERNAL true-2D baseline, the pre-registered DROP, the shared `FacetTier::morton`, and `hindsight_stream`'s independence lesson. §16.A wired to the template. Plus a `CLAUDE.md` correction: the supersession-index input list now includes the board, with the ordering rule. |

**Contract inventory net delta: none** (DOC/BOARD ONLY; no code, no probe
run, no address arm promoted). The §16 gate is unchanged — §11.3 supplies
the instrument that could separate A2 from A3, never a verdict between them.
Review correction worth carrying: the recorded `morton_ka` run scores TWO
arms (Morton vs Chebyshev board), so it shows locality preservation, not
codebook-to-cell identity — A2 (row-major), inverse-Morton and A4
(permutation) must still be written in §16.

## 2026-08-30 — #1077 + #1078 BOTH MERGED (plans only): the convergence pair is on main

| PR | merge | content |
|---|---|---|
| **#1078** | `da3b5c76` | alpha-reason-witness W0–W8 + three D-ARW-0 archaeology passes (the BROKEN WIRE: top_k → min..max window before the P64 mask ALU) |
| **#1077** | `6bad99c8` | rubicon-loco-rung audit: eleven STOP gates F-RLR-1..11, rung mapping pinned (discriminant 0..=9), NodeGuid copy-never-re-mint, rung-4 fossil verdict on review-corrected census |

**Contract inventory net delta: none** (both PLAN/BOARD ONLY). Queued: the
wave ladders in `STATUS_BOARD.md` (D-RLR-0..6, D-ARW-0..8); the execution
loop is MedCare-rs #597 (six live waves, five HELD each behind one operator
word). Full detail: the batched arc entry.

## 2026-08-30 — #1079 MERGED (db85d6fa): the cognitive-fabric census — DOC/BOARD ONLY

### Current Contract Inventory — net delta

**None.** No type, trait, or tenant. The census
(`docs/architecture/COGNITIVE-FABRIC-CENSUS-2026-08-30.md`) is the SOURCE-FACT
feed for #1078 W0 and the open-arc CONTINUATION ledger (§8) — reading order,
fifteen arcs with gates/owners, nine traps. Board deltas: two EPIPHANIES
entries, the `persona-vs-rung-ladder.md` scope correction. The private
consumer's mirror is MedCare-rs #596 (merged same day; its #595 debugger cut
merged too).

### Queued work this unlocks

Census §8.2 rows — headline: the dedup payment (arc 1, operator-gated), the
first executable BUY (arc 4, specified/unrun), the OGAR execution semantics
decision (arc 5, operator), the rung-dispatch composition (arc 8). Arc 16
(top-level reachability) is CLOSED by this very entry.

## 2026-08-28 — #1072 MERGED: token-value-tenant-v1 PROPOSED (plan/board only)

### Current Contract Inventory — net delta

**None.** No type, trait, or tenant minted; `ValueTenant::Token = 16` is a
PROPOSAL inside `.claude/plans/token-value-tenant-v1.md`, gated on D-TVT-1's
whole-KJV measurement. The `BoardAggregates` discriminant-16 reservation is
untouched until that carve actually lands (it re-bases to 17 then).

### Queued work this unlocks

`STATUS_BOARD.md` § token-value-tenant-v1: D-TVT-0 (KJV into Tigris +
`lance-graph-hydrate` hydration) → D-TVT-1 (the STOP-gate scale re-measure,
`TokenId` vs `WordId` control) → D-TVT-2 (the carve, A-or-B decided by
D-TVT-1) → D-TVT-3 (lens write onto SPO-stream rows) → D-TVT-4 (BUY / NO-BUY).

## 2026-08-29 — #1074 + #1075 BOTH MERGED (plan, then the code split out of it)

| PR | merge | content |
|---|---|---|
| **#1074** | `b871d4ea` | `mul-ewa-trust-propagation-v1` — PLAN/BOARD ONLY as merged (recut before merge) |
| **#1075** | `69016d99` | the epistemic triptych — `revision.rs`, `fusion.rs`, `KanbanColumn::{revise, advance_on_revision}` |

**Contract inventory net delta (now on main):** `revision` and `fusion`
modules + two `KanbanColumn` primitives. Full type list in the #1075 arc entry.

**`ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE` — CLOSED** by
`KanbanColumn::revise`. The `Plan` exit was declared with semantics and
reachable by no named primitive.

**Open on main, deliberately flagged rather than silently fixed:**

1. `F-MEP-0b` still reads "covariance *or* precision". `jc::ewa_sandwich_3d` is
   explicit — *"world-space covariance matrices Σ ∈ ℝ³ˣ³ … pushed forward to
   image-space"*, `M_k = sqrt(Σ_k)`. Precision would be a NEW quantity
   borrowing the algebra, not reuse of the EWA operator. The open question is
   the **bounded monotone trust→step-transform mapping**, not the kind.
2. The Σ→`TrustTexture`→`GateDecision` production path in §0b/§4 conflicts with
   EWA scoped as attention/tension geometry.

Both misdirect W1 before it runs; neither is a defect in the merged text's own
terms. **Follow-up PR, not a session note.**

**Standing corrections carried from #1075 (both against earlier claims of
mine):** `Stamp::disjoint` is canonical for S4 pooling under KNOWN overlap, and
the **PR #854 ruling** is canonical for evidential independence NOT being
established (`event identity ≠ evidential-base membership ≠ source
dependence`). And `stance_panel` supplies four non-destructive READINGS, not
the H₃ synthesis — the Grail is unbuilt.

**Next, per the operator's audit:** `perspective-parallax-v1` (bounded Shannon
control → `PerspectiveResidual` sibling differential → Three-Mountains rotation
→ Perspective/Meta band receipts; H₃ deferred — parallax earns the FIRST buyer,
it does not define H₃) and `mul-proprioception-v1` (MUL as a proprioceptive
revision loop mapping to existing `FieldModulation`; ambiguity widens cognition,
never Blocks; the Rubicon gate stays separate).


## 2026-08-29 — the epistemic triptych lands in the contract (BRANCH, not yet merged)

> **✅ SCOPE RESOLVED (operator-instructed, 2026-08-29).** These commits were
> first landed onto `claude/happy-hamilton-0azlw4` — the branch of PR #1074,
> whose status line reads *"PLAN/BOARD ONLY. Measure-before-carve. **No
> contract change, no wiring**, until W1's numbers land."* That was a scope
> violation I flagged against myself: 1138 lines of contract code on a
> plan-only PR. #1074's STOP rule targets the EWA/Σ carrier (K1 `TrustSigma`
> on `TrustQualia`) specifically, and **none of this is that carrier**, so the
> rule's intent was intact — but its letter was not, and a reviewer of a plan
> faced Rust.
>
> **Split executed on operator instruction:** the five commits now live on
> `claude/epistemic-triptych-contract`, cut from `main`, and #1074 is recut to
> EWA measurement / attention geometry only. Each PR can now be reviewed on the
> questions it actually raises.

### Current Contract Inventory — net delta: THREE modules, one primitive

| added | where | what it is |
|---|---|---|
| `revision` (module) | `contract/src/revision.rs` | `EvidenceMask`, `InterpretiveHorizon`, `EncounterEvidence`, `BasisView`, `RevisionKind` (9), `EvidentialEffect` (3), `RevisionDelta`, `RevisionPolicy`, `GadamerRevision`, `RevisionOutcome`, `HypothesisReport`, `GroundingRequest` |
| `fusion` (module) | `contract/src/fusion.rs` | `FusionOutcome` (7), `SynthesizedClaim`, `FusionReceipt`, `fuse()` |
| `KanbanColumn::revise` | `contract/src/kanban.rs` | third routing primitive beside `advance`/`veto`; reaches `Plan` |
| `KanbanColumn::advance_on_revision` | `contract/src/kanban.rs` | `EvidentialEffect` → route, the `TEST→ACCEPT` step of #1057 |

**Why revision.rs at all:** PR #1057 (MERGED) names revision *"the only
write-back"* and *"the court of appeal"* at its ACCEPT step, while
`counterfactual.rs` had shipped twice and `revision.rs` never landed. A merged
plan's ACCEPT step was governed by a module that did not exist.

**`ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE` is CLOSED.** `Evaluation`'s
successors are `[Commit, Plan, Prune]`; `advance()` takes the first non-`Prune`
(always `Commit`), `veto()` takes `Prune`, and nothing produced `Plan`. The
documented revision exit had legal-edge status and no route. `revise()` is the
route; no edge added, no `next_phases` change.

### ⚠ PRIOR ART — three loci that already held this, uncited

Recorded because this session re-derived thinner versions of all three before
finding them, twice AFTER diagnosing the pattern:

- **`planner::nars::belief::BeliefArena::revise_at`** — canonical for **S4
  pooling / known-overlap rejection**, via `b.stamp.disjoint(stamp)`.
  > **⊘ CORRECTED same-day (operator).** This entry first said disjointness
  > "IS the independence test" and that `revise_at` is canonical for
  > independence. **Wrong** — PR #854 already ruled it:
  > `event identity ≠ evidential-base membership ≠ source dependence`
  > (line 1567 of this file). `Stamp` models source MEMBERSHIP and is lossy
  > (`1 << (id % 64)`; ids 0 and 64 alias); `causal_audit.rs:346` leaves
  > `independent_strength` `None` because there is **no dependence model**.
  > `disjoint` is SOUND but NOT COMPLETE — aliasing yields only false overlap,
  > so revision under-pools rather than double-counts.
  > **Canonical register, corrected:** `revise_at` for S4 pooling under
  > known overlap; **the #854 ruling** for the fact that true evidential
  > independence is NOT established, its remedy (`EvidenceEventId`,
  > `EvidentialBase<K>`, tri-state `Independence`) designed and unbuilt.
  > `FusionReceipt::shared_roots` is therefore CONJECTURE expressing a
  > contract the substrate cannot yet attest — not duplication.
- **`planner::nars::stance::stance_panel`** — four philosophical late-bound,
  non-destructive reads over one contradiction set. Its own doc: *"The three
  meanings of aufheben ARE `revise_at`'s three fields: cancelled = pooled
  truth, preserved = the `contradiction` field, lifted = the rung."* Plus
  Nietzsche (genealogy by flip direction), Kant (ablate modal grading; the
  delta is the reader's a-priori contribution, doubling as an inertness test),
  Wittgenstein (distinct language-games).
  > **⊘ CORRECTED same-day (operator): `stance_panel` is NOT H₃.** This entry
  > first concluded "Hegelian synthesis was already in the substrate" and that
  > the new horizon "is already four horizons". **Overcorrected.** Those are
  > four READINGS of an existing state — blend modes over the same pixels.
  > The Grail question is what new explanatory structure makes H₁ and H₂
  > intelligible as partial horizons of a larger whole; `revise_at` pools the
  > truth of the SAME `CStmt`, which is an Aufhebung metaphor, not the
  > inference of a latent mediator. **The Grail survives, and is unbuilt.**
- **`.claude/plans/epistemic-quadrant-materialization-v1.md`** + its 1859-line
  `probe_sudoku_teacher.rs` — G3 is the membrane: *"bifurcation clones the slab
  as a counterfactual world, propagates to contradiction, and **ONLY the
  elimination returns**"*. G4 measures the cost of refusing to fork.

**The finding that outranks any single defect: this architecture lives in FIVE
loci that do not cross-reference** — `revise_at`, `stance_panel`, PR #1057,
`epistemic-quadrant-materialization-v1`, and now the contract layer. Nothing is
missing; it is scattered, and each session re-derives a thinner version of a
neighbour it never read. A cross-reference pass is worth more than more code.


## 2026-08-27 — the D-MCAL arc, #1065–#1070 ALL MERGED (six of six deliverables)

### Current Contract Inventory — net delta of the arc

| symbol | change |
|---|---|
| `contract::plan::PlannerContract::gate_check` | **REMOVED** (#1066) — 0 implementors org-wide, 0 callers |
| `contract::mul::MulProvider::gate_check` | **DEPRECATED** (#1066), then given a **DEFAULT** (#1068) — deprecation alone left it required, so the documented migration produced `E0046` until the default landed |
| `contract::mul::GateDecision::from_axes` | **NEW** (#1068) — the canonical `(TrustTexture, FlowState) -> GateDecision` rule, now ONE definition shared by the i4 evaluator and the trait default |
| `contract::kanban::KanbanColumn::advance` | **NEW** (#1068) — forward successor |
| `contract::kanban::KanbanColumn::veto` | **NEW** (#1068) — `Prune` iff legal (Libet free-won't). An ergonomic named wrapper over the already-public `next_phases()` walk, **not a new capability** |
| `contract::kanban::KanbanColumn::advance_on_gate` | behaviour unchanged; now DELEGATES to the two above |
| everything else | unchanged — **no enum minted anywhere in the arc** |

### What the arc established

The `Hold/Block { texture, flow }` payload is **inert** at all four
execution-gate consumers, which is why both external producers invented
coordinates they never measured. `mul::GateDecision` is the **execution / commit
gate**, not MUL's output (rename deferred, `ISS-MUL-GATE-NAMED-FOR-THE-WRONG-LAYER`).
A public MUL output is **not** derivable from the contract, and the blocker is
**OQ-MCAL-1** — a vocabulary question, not a missing enum.

### Open, and deliberately so

- **F-MUL-6 is OPEN** — ada-rs was compiled against the arc head (3 errors, all
  pre-existing #1045, zero from the arc); MedCare-rs was **not** built.
  `ISS-F-MUL-6-HALF-BUILT`. The rename is blocked on this.
- **`ISS-PLANNER-SANDBOX-STILL-CARRIES-FREE-TEXT`** — the planner's
  `Sandbox { reason: String }`, unfixable until the counterfactual scaffold's
  four blockers clear.
- **The ada-rs stopgap stays unpushed.** A red compiler is preferable to a
  green lie; the honest route now exists.

### Two claims withdrawn during the arc, recorded because both were mine

1. D-MCAL-4's "red-state proved mechanically" — a naming artifact.
   `next_phases()` already exposed `Prune`.
2. D-MCAL-5's "no promotion needed either" — conflated arm payload with arm
   selection.

Both were caught by review after landing in several places at once. The cost of
a wrong claim is proportional to how many surfaces repeat it.

## 2026-08-27 — D-MCAL-2 (IN PR) — the two gate-returning trait methods

### Current Contract Inventory — CHANGED (one method removed, one deprecated)

| symbol | before | after |
|---|---|---|
| `contract::plan::PlannerContract::gate_check` | `fn gate_check(&self, situation: &SituationInput) -> GateDecision` | **REMOVED** |
| `contract::mul::MulProvider::gate_check` | required method, undocumented direction | **`#[deprecated]`** with migration pointer; still required, removal scheduled after D-MCAL-4/D-MCAL-6 |
| `contract::mul::MulProvider::{assess, compass}` | unchanged | unchanged — the kept, legitimate direction |
| `contract::plan::PlannerContract::{plan_full, plan_auto, set_selector, orchestrate}` | unchanged | unchanged |

**Breakage surface, measured before the cut (D-MCAL-1):** `PlannerContract` has
**zero implementors org-wide** and zero callers. The two in-tree `.gate_check(`
call sites — `lance-graph-planner/src/api.rs:637` and
`lance-graph/src/lance_native_planner.rs:61` — bind the planner's **inherent**
method returning `Gate` (`MulGateDecision{Proceed, Sandbox, Compass}`), not the
trait, and are untouched. `MulProvider` has exactly one implementor anywhere
(`ada-rs::contract_impls::AdaMulAdapter`), which is why it is deprecated rather
than cut: the invariant is that a source-breaking contract change is not
verified until the known consumer builds (D-MCAL-6).

**Tests:** +3 falsifiers in `mul.rs` discharging F-MUL-5's MUL half — both
genuine arms (Dunning-Kruger, allostatic depletion) readable off
`MulAssessment` with no verdict constructed; the mandatory can-stay-silent
twin on a non-degenerate input; axis independence (F-MUL-7's premise).
1224 lib tests green, clippy `--all-targets` clean.
## 2026-08-27 — D-MCAL-4 (IN PR) — a domain route into the phase DAG

### Current Contract Inventory — CHANGED (two methods added, no type minted)

| symbol | status |
|---|---|
| `contract::kanban::KanbanColumn::advance()` | **NEW** — forward successor (first non-`Prune`); `None` at absorbing columns |
| `contract::kanban::KanbanColumn::veto()` | **NEW** — `Prune` iff a legal successor (the Libet free-won't edge); `None` mid-`CognitiveWork` |
| `contract::kanban::KanbanColumn::advance_on_gate()` | unchanged behaviour; now DELEGATES to the two above, so there is one copy of the DAG routing rule |

**No enum minted.** D-MCAL-5's prohibition (never a fourth gate enum beside the
three that exist) is honoured: `advance` / `veto` are transitions named as
transitions, not a new verdict vocabulary. "Stay put" needs no symbol — it is
`None`.

**Why the gap existed:** before this, the only route into the phase DAG was
`advance_on_gate(&GateDecision)`, whose `Block`/`Hold` variants demand a
`TrustTexture` AND a `FlowState`. A domain measuring neither had to invent both.
Since the routing never read those coordinates (pinned by
`f_mul_4_routing_ignores_the_calibration_payload`, D-MCAL-3), naming the
transition directly loses nothing and fabricates nothing.

**Tests:** +9 in `tests/d_mcal_4_domain_evidence.rs`. F-MUL-1 (consent veto,
ada-rs's shape) and F-MUL-2 (evidence contradiction, MedCare's shape) each
assert the route is correct, that no `GateDecision` is constructed in the
domain path, and that the route is IDENTICAL to the fabricating path it
replaces. Red-first verified mechanically: the same file against `main` fails
to compile (`no method named veto`), i.e. on `main` there is no route into the
DAG that does not construct MUL ground.
replaces. **Red-first claim CORRECTED (codex review, same day):** the earlier text said
the red state was proved mechanically because the file failed to compile against
`main`. That was wrong. `next_phases()` is public on `main` and already returns
`Prune` for `Planning`/`Evaluation`, so a domain could always have routed a veto
by hand without touching `GateDecision`; the compile failure proved only that two
convenience method NAMES were absent. What is true: the honest route existed but
was unnamed, so the obvious path (`advance_on_gate`) demanded two calibration
coordinates and both measured producers invented them. This is an
ergonomics-and-naming fix with a measured behavioural consequence, not a new
capability. `veto_agrees_with_the_pre_existing_next_phases_route` pins the
equivalence.
## 2026-08-26 — #1059 MERGED (e5f750e) — the Octopus causal-CoT audit (measurement only; NO contract change)

### Current Contract Inventory — UNCHANGED

This PR added **no type, no module, no bit, no ABI symbol**. It is a measurement
report plus board entries. The inventory below is therefore stated as *what the
audit measured about existing types*, not as an addition.

| type (existing) | measured state, 2026-08-26 |
|---|---|
| `AttentionMaskSoA` (`cognitive-shader-driver`) | **no production consumer**; not read by `elect_mode` |
| `elect_mode` / `DispatchMode` (`contract::dispatch_mode`) | the actual transition fn; reads gate+surprise+contradiction, never alpha |
| `SettlementCell` (`contract::settlement`) | shipped; `cell()` never reads `CausalTopology` |
| `CausalTopology` (CE64 59..60) | shipped; **no producer stamps it** in the three repos measured |
| `ReasoningBand` (CE64 61..63) | shipped as a carrier; **gates nothing**; OGAR mints no `band_reading` |
| `RungLevel` (`contract::cognitive_shader`) | shipped, 10 levels — collides in name with `ReasoningBand`'s 8 |
| `GapKind`/`Frontier`/`Throttle` (`planner::nars::tactics`) | shipped; GATE and removal-TEST stages absent |
| `CausalEdge64` | carries **no Inc/Dec polarity** — the state axis has no carrier |
| `Sandbox` | **no symbol exists** — correct; it is a definition, not a type |

### Open follow-ups this PR created (none blocking)

`ISS-ALPHA-NOT-LOAD-BEARING`, `ISS-REASONING-BAND-GATES-NOTHING`,
`ISS-BAND-READING-UNMINTED-IN-OGAR`, `ISS-DOMAIN-LENS-BY-CONVENTION-ONLY`,
`ISS-NO-CAUSAL-SIGN-ON-EDGES`, `ISS-D-ECG-6-BUDGET-WITHOUT-ADMISSION`,
`ISS-RUNG-VS-BAND-CARDINALITY-COLLISION`. D-OCT-1..11 queued on STATUS_BOARD.

**Naming:** "Octopus" stays internal to the audit track. Nothing renamed.
Publish nothing until F-OCT-1, F-OCT-3 and F-OCT-7 have run.

## 2026-08-25 — `contract::ogar_codebook` gains `typed_field` (0x080A) — the OGAR W4 doc-layer council's cross-repo mirror obligation

### Current Contract Inventory — `ogar_codebook::CODEBOOK` +1 row

- **`("typed_field", 0x080A)`** added to `lance-graph-contract::ogar_codebook::CODEBOOK`,
  immediately before the existing `("document", 0x080B)` row it was previously
  reserved-but-unminted alongside. Wire-compatible mirror of OGAR
  `ogar_vocab::class_ids::TYPED_FIELD`, minted by OGAR's own W4 5+3 council
  (`.claude/agents/5plus3-council.md` pattern; spec v1→v2→v3 =
  `OGAR-DOC-W4-BUILD-SPEC.md`; OGAR `docs/DISCOVERY-MAP.md` `D-OGAR-DOC-LAYER`
  Status 2026-08-25, OGAR commit `3fbd9f36055299003e3c820156ad86c9518d8e78` on
  branch `claude/bpe-tokenization-architecture-3xd4eh`). No count-fuse in this
  crate needed bumping (`ogar_codebook.rs` carries no hardcoded `CODEBOOK.len()`
  literal, unlike OGAR's own three internal fuses) — the row addition alone
  keeps the mirror in sync. This is the completeness item OGAR's own
  `COUNT_FUSE` removal (2026-08-14) can no longer catch automatically: a
  mint landing on the OGAR side with no corresponding mirror row here would
  now go silently stale. `cargo test -p lance-graph-contract` (1220+7+8+7+4+21
  across the crate's test binaries) green, `clippy -p lance-graph-contract
  --all-targets -- -D warnings` clean, `cargo fmt -p lance-graph-contract
  -- --check` clean.
- **Downstream:** `paperless-rs`'s `paperless-kv::HOT_PLUG` activation (against
  OGAR's new `ogar_vocab::document_actions` table) is the consumer-side
  follow-through, tracked in that repo — not this one.

## 2026-08-23 — #1017 OPEN — the token seam: #1012's integration half answered, and the 8-bit lane's ceiling found

- **What exists now:** `PROBE-TOKEN-SEAM-1` (37 gates, 13 disable-runs) in
  `AdaWorldAPI/paperless-rs crates/paperless-token`, with
  `docs/TOKEN-SEAM-ARCHITECTURE.md` as its bounded architecture. It answers the
  question #1012 left open: ONE versioned BPE tokenization of a span drives
  Tantivy, DeepNSM-v2 and a forward-prediction input surface simultaneously,
  each consumer BORROWING, none re-tokenizing. Neither Tantivy nor DeepNSM-v2
  needed a line changed.
- **Standing laws banked:** ONE SOURCE SPAN → ONE TOKENIZATION RECEIPT;
  TOKENIZE ONCE, PROJECT MANY TIMES; AN INDEX MAY ACCELERATE THE ABI, IT MUST
  NEVER BECOME THE ABI; BPE SEQUENCE IDENTITY IS NOT A SEMANTIC WORD
  COORDINATE; and (method) A KNOB THAT DOES NOT BIND IS NOT A DISABLE — a
  fixture's SHAPE is part of a test's coverage.
- **The number that bounds prior work:** the 8-bit id lane saturates at 75 KB
  (247/255 ids, compression 3.18× → 2.03×). #1012's 3.35× is a 1 KB figure and
  must not be read as corpus-independent. The hi-byte PAGE lane is the next
  probe and no scale claim survives without it.
- **Open, in order:** (1) the paged vocabulary; (2) a real retina — this probe
  builds its `DocIr` from text, so the next one should take
  `ogar-from-docv1` on an actual scan and a `spider_doc_ir` crawl of the same
  content and check both present the same span-population shape;
  (3) a lawful resident lane (`SoaEnvelope` or a new `ValueTenant`, designed
  against the measured 30–54 % framing overhead); (4) a real forward arm —
  the seam supplies the input, but which representation a trained model prefers
  is untested; (5) a structured-evidence corpus to exercise the parallel typed
  path (`arm-discovery`'s `FeatureSpec` + category-index rows) and the shared
  span identity where the two meet.

## 2026-08-23 — #1006..#1014 MERGED — the belief-ABI arc: Step 1 audit, root-law probes, Step 2 ruling request, frontier Phases 1+2, the real-episode measurement

Nine PRs (#1010/#1011 landing stacked via #1009's merge) closed the arc:

- **What exists now:** Step-1 delegation audit (#1006, pure negative);
  `probe_tarski_signed_witness` + `probe_four_plane_causal_medium` (#1007);
  the stream-order epiphany + seam plan (#1008); three root-law probes
  (#1009 — evidence generalization / falsification asymmetry / epistemic
  fabric); the Step-2 ruling request + copula addendum + five measurement
  probes (#1012); the frontier loop Phase 1 (styles = microcode; learner =
  shipped revise + Stamp + CHOICE) and Phase 2 (R2IL typed receipts, the
  clobber result) (#1013); the REAL FunctionBehavior episode measurement
  (#1014 — 99.7% opcode over-admission vs 31.5% base rate; trigram
  type-collapse 97 vs 264; Stamp mod-64 ceiling binds at 143 episodes).
- **Standing laws banked:** CONTENT NEVER TRAVELS IN CLASSID; HIERARCHY IS
  THE ADDRESS SPACE NOT THE ONTOLOGY; GROUP MEMBERSHIP IS RELATION TOPOLOGY
  NOT HHTL ANCESTRY; MEASURE THE DISTRIBUTION BEFORE BUYING THE
  REPRESENTATION; sequential adjacency is not composition — the macro
  carrier is the def-use chain.
- **PENDING (operator's, not the agent's):** the Step-2 ruling on the six
  residue items (the stamp item now carries #1014's measured mod-64
  capacity number); the three-IF R2IL x BPE / OGAR-loco / V4 synthesis
  remains hypothesis — nothing minted, reserved, or decided.
- **Arc rows:** `PR_ARC_INVENTORY.md` (this hygiene pass); NOTE the recorded
  GAP MARKER — #976..#1005 (other sessions' arcs) still lack arc rows.

## 2026-08-21 — D-ACR-7 IMPLEMENTED — `contract::band_reading` (the 59..63 reading contract)

### Current Contract Inventory — `VersionedGraph::hydrate_from`, and the CI gate that should have caught all of this

- **`lance_graph::graph::versioned::VersionedGraph::hydrate_from`** (new) —
  the shape `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE` names.
  `local`/`s3`/`azure`/`gcs` address a store WHERE IT SITS; for the three
  remote ones that makes the object store the store, which the doctrine
  explicitly does not. They are KEPT (addressing a remote store is legitimate
  when a caller means it, and better at >1 replica); this adds the doctrine's
  own shape beside them.
  - **Ensure-hydrated, not hydrate-or-fail**: an existing destination is the
    warm path, returned as `Hydration::AlreadyLocal` rather than an error.
  - **`Hydration { Fresh(ArchiveReport), AlreadyLocal }`** (new) — the
    distinction is returned rather than folded away: a boot that fetched and a
    boot that found it are different events.
  - **`GraphError::Hydration { source, location }`** (new variant) — its own
    variant, not a flattened message, so a caller can tell a checksum mismatch
    (never retry) from a transport error (retry).
- **CI is no longer this branch's concern — PR #984 carries it.** An earlier
  draft of this entry said `lance-graph-hydrate` is "gated for the first time"
  here and that "eight more members are still ungated". Both were true when
  written and are false now: #984 gated EVERY member (each measured locally
  first), added `cargo build --workspace` as the structural net so future
  members need no line, and pinned the toolchain in all seven workflows. The
  count was also wrong — eleven, not nine, because the member check behind it
  could not see names with an underscore. See `ISSUES.md`
  `ISS-CI-GATE-IS-AN-ALLOWLIST-NINE-MEMBERS-UNGATED` (now RESOLVED, with the
  outcome recorded above its original text) and `EPIPHANIES.md`
  `E-THE-GATE-IS-A-HAND-MAINTAINED-ALLOWLIST-NOT-THE-WORKSPACE-1`.
- Tests: `lance-graph-hydrate` 39 green; `lance-graph` `--lib` green incl. two
  new `hydrate_from` falsifiers scoped to what that function adds (the warm/
  error mapping — the archive mechanics are falsified one crate down).
  `cargo clippy -p lance-graph --lib --tests -- -D warnings` clean.

### Current Contract Inventory — 1 new module in `lance-graph-hydrate`, and that crate now COMPILES

- **`lance_graph_hydrate::archive`** (new) — the `absent -> hydrated` edge for a
  dataset shipped as ONE checksum-pinned **zip** object, alongside the existing
  `copy::hydrate_dir` (a tree of objects) and `file::hydrate_file` (one plain
  object).
  - `hydrate_archive(store, remote_object, publish_dir, expected_sha256_hex,
    root) -> ArchiveReport` — composes the two mechanisms the crate already
    owns (`hydrate_file` for the pinned fetch, `publish::publish_by_rename` +
    `StagingKind::Dir` for the atomic publish) and adds only the middle:
    expanding one verified container into staging under a containment rule.
  - `ArchiveReport { files, bytes }`; `HydrateArchiveError` adds
    `EscapingEntry { entry, root }` (Zip-Slip refusal, whole archive rejected
    off the CENTRAL DIRECTORY before any byte is written) and
    `NoFiles { root }` (an all-directory tree would publish an unopenable
    dataset).
  - **Zip and not tar, by operator ruling 2026-08-22** (*"bitte als zip, nicht
    dass wir ein Verzeichnis mit einzelnen Dateien shippen"*): a zip carries a
    central directory, so entries can be enumerated and sought without a
    sequential scan — which is also what makes the whole-index validation above
    possible before extraction starts.
- **`lance-graph-hydrate` builds again.** It did not compile at `origin/main`
  (#981): `object_store 0.13.2` moved `get`/`put` onto `ObjectStoreExt`. Two
  `use` lines; four files also picked up `cargo fmt`. Why a merged crate could
  be broken at HEAD — and what that says about minting without a consumer — is
  `EPIPHANIES.md`
  `E-A-CRATE-WITH-ZERO-CONSUMERS-IS-BUILT-BY-NOTHING-AND-CAN-BE-MERGED-BROKEN-1`;
  the durable gap is `ISSUES.md` `ISS-HYDRATE-CRATE-HAS-NO-BUILD-GATE`.
- Tests: 39 green in the crate (33 pre-existing, 6 new). Both new guards are
  mutation-checked — disabling the containment check and the file counter turns
  exactly their two tests red and leaves the other four green.

### Current Contract Inventory — 1 new zero-dep module + 1 ClassView provided method + 1 gate test in `causal-edge`

- **`lance_graph_contract::band_reading`** (new, zero new bytes) — implements
  the RATIFIED spec `.claude/plans/dacr7-band-reading-contract-v1.md`. **One
  reading contract, TWO carriers**: `CausalEdge64` bits 59-60 (truth) / 61-63
  (band) — the muscle memory — and `CausalEdgeV3` byte [8] hi-2 / byte [9] lo-3
  via `truth_raw()` / `spare_raw()` — the granularity. The reading NEVER changes
  stored bytes; it declares how a consumer projects them.
  - `TruthLens { Trust, Topology }` — which vocabulary the 2-bit register means
    for this `(class, rail)`. **Doc-comment pointers only**, never imports:
    both crates are zero-dep and neither may depend on the other
    (`lance-graph-contract/Cargo.toml:10-17`, `causal-edge/Cargo.toml:20-23`).
  - `BandPresence { Absent, Present }` + `WitnessKind { None, Table,
    CausalFacet, EpisodicBasin }` — the F5 evidence-kind discriminator.
  - `EdgeProvenance { V2Stamped, V3Register, V1Legacy, Unknown }` with
    `trusted()`. **`V3Register` is a CALLER ASSERTION ("minted clean"), never an
    inference** — council BLOCK-1: `CausalEdgeV3::from_v1` (`edge_v3.rs:117`)
    has no provenance parameter and raw-copies the tail (`:138-139`), so the v1
    temporal trap (`temporal >= 512` aliases a non-zero band) reaches V3
    **transitively**. Unstated ⇒ `Unknown` ⇒ refuse.
  - `BandReading { truth_lens, band, witness }` + `ZERO_FALLBACK`
    (`Trust`/`Absent`/`None`, `== Default`), `admits`, `admits_band`,
    `project_truth`, `project_band`. **Projection order is provenance FIRST,
    then lens** — an untrusted edge is refused before its lens is even asked.
  - `BandDeclarations` — the registry. **L1 split (the council's Phase-4 fix):**
    declaration lookup is TOTAL (`reading_or_default`, zero-fallback, sibling-
    consistent with `edge_codec_flavor` / `rail_carving` / `value_schema`),
    while raw-bit projection is FALLIBLE (`Result`, `UndeclaredClass`) — it must
    FAIL, never return a plausible value.
  - **L3 audit distinction preserved:** `get()` returns `Option<BandReading>` —
    `None` = never declared vs `Some(Absent)` = explicit opt-out.
    `declare()` returns `true` on replace: a redeclaration is visible, never silent.
  - `sampling_admits(&dyn Tactic) -> bool` = `moves_confidence()` — the
    **14/34** capability filter, never `maturity().is_production()` (**31/34**).
    Asserted against the real `all_kernels()`, not a fixture.
- **`ClassView::band_reading(class, rail)`** — provided method, defaults to
  `BandReading::ZERO_FALLBACK`. Same registry-resolution pattern as its two
  siblings; selection only, never a stride change.
- **`causal-edge` `g10b_lift_preserves_truth_and_spare_ordinals_under_both_lenses`**
  — hosted THERE because both crates refuse each other. Closes a measured
  missing test: the module doc claimed the truth/spare round trip is byte-exact
  and nothing asserted it through the accessors. Fires (nonzero `Unknown` /
  `Transcendent` ordinals survive `from_v1` → `rehydrate`) **and** stays silent
  (zero ordinals are not upgraded).
- **Mints nothing**: no new tenant, no bit, no `ENVELOPE_LAYOUT_VERSION` bump,
  no `cfg` feature that re-means a stored bit (G9).
- Gates: G1 (1207 contract tests, +13), G2 clippy clean on the new module,
  G3′/G4′/G5a/G5b/G6/G7′/G8/G9/G10b all green — pre-registered before any agent ran.

## 2026-08-21 — #975 MERGED (4a43698) — two measured corrections + a retracted absence

- **`DismechTopology` doc + this file corrected**: the label-KNOWN 3,978 are
  NOT the oracle population. **2,512 over 549 diseases** carry a mediator;
  1,466 (36.9%) name nothing; **92** `INDIRECT_UNKNOWN_INTERMEDIATES` name
  mediators despite their label. 2,512 is the population REQUIRING GROUNDING —
  45 of 3,095 distinct mediator strings are exact node references (1.5%).
- **`deepnsm-v2` academic carve under-fills**: 20,845 rows are 18,559 distinct
  surface forms, carve 90.6% full, basins 73..79 empty. Falsifier added,
  disable-verified.
- **AriGraph retraction**: `crates/lance-graph/src/graph/arigraph/` is **15
  modules, ~327 KB** (`ppr`/`bm25`/`rrf`/`community`/`markov_soa`/`episodic`/
  `witness_corpus`/`retrieval`/…). It is IMPLEMENTED and UNWIRED — the state
  `E-ARIGRAPH-IS-AN-ISLAND` already recorded. Absent ⇒ build the organs;
  unwired ⇒ close the seam. Do not rebuild these under a new name.
- **Open, blocking any gold set (operator decision):** the third bucket for
  the 1,466 label-only edges and the 92 contradictory ones.

## 2026-08-20 — #974 MERGED (8a93423) — DisMech compact evidence vocabulary + citation sidecar

### Current Contract Inventory — 1 new zero-dep module (D-ACR-1)

- **`lance_graph_contract::attention_facet`** (new, zero new bytes):
  - `AttentionFocusFacet` — the **Attention reading** of the shipped
    `FacetCascade` (`classid(4) | 6×(8:8)`) under `CascadeShape::G6D2`, plus an
    explicit `depth: 0..=12` held OUTSIDE the 12 bytes (the `NiblePath`
    precedent — inferring a wildcard from zero bytes would collide with the
    zero-fallback ladder, where `0` is a dormant tier, not a terminator).
    `exact` / `prefix` (loud refusal past 12) / `whole_class` / `coarse` /
    `fine` / `axis` (all `None` past `depth` — never `0`, which is centroid
    zero's value) / `covers` / `common_prefix`.
  - `FocusAxis` — `Axis0..Axis5`, a **position, not a meaning**. The module
    names no axis semantics: the cascade reading (HEEL·HIP·…) and a candidate
    ontology-scope reading (disease/anatomy/process/substance/evidence/context)
    are both ClassView-resolved projections, demonstrated over identical bytes
    in `the_same_atom_reads_as_cascade_and_as_ontology_scope_without_changing_a_byte`.
  - `RowFocusMask` — the sparse container. Membership is `covers`, set ops are
    containment-shaped (`union` absorbs into a minimal antichain; `intersect`
    yields the deeper of a covering pair; `difference` is deliberately
    conservative). **Never a bit-OR**, and no `FieldMask`/`WideFieldMask`
    cardinality is inherited — it does not index rows at all.
  - `FOCUS_AXES = CASCADE_UNITS / 2` — derived, never a second literal.
  - **Mints nothing**: no new tenant, no bit, no `ENVELOPE_LAYOUT_VERSION` bump.
    Does not, and structurally cannot, reference `cognitive-shader-driver`'s
    `attention_mask*` (the dependency edge runs the other way).

### Current Contract Inventory — 1 new zero-dep module

- **`lance_graph_contract::dismech_evidence`** (new):
  - `DismechTopology` — the four measured `causal_link_type` states, 2 bits,
    `from_source`/`as_source`/`to_bits_2`/`from_bits_2`, plus
    `source_knows_intermediates()` / `mediator_unresolved()` which separate the
    label-KNOWN 3,978 edges from the 4,539-edge RESTRAINT CONTROL. ⚠ The
    label-KNOWN set is NOT the oracle population: 1,466 of the 3,978 (36.9%)
    name no mediator at all. The remaining **2,512** (549 diseases) is the
    population REQUIRING GROUNDING, not a usable oracle — only 45 of 3,095
    distinct mediator strings are exact node references (1.5%); 40.7% stay
    ungrounded prose after label matching. A further **92**
    `INDIRECT_UNKNOWN_INTERMEDIATES` edges DO name mediators and belong to
    neither population. Measured by `dismech_oracle_census`;
    `E-DISMECH-KNOWN-INTERMEDIATES-ARE-PROSE-NOT-IDENTITIES-1` (2026-08-21).
  - `Supports` (4, 2 b), `EvidenceSource` (5, 3 b) — round-tripped exhaustively.
  - `CitationNamespace` (PMID/DOI/ORPHA/NCT/CGGV/URL), `CitationKey`
    (`Identified{namespace,id}` | `ContentAddressed(ContentId)`),
    `BibliographyRecord{key, title: ContentId}`.
- **SOURCE-SIDE ONLY — deliberately does NOT reference `CausalEdge64`.** The
  durable causal overlay must not become a pile of hot reasoning registers
  (operator ruling); `DismechTopology -> CE64 bits 59..60` happens at
  HYDRATION, in the consumer, as a 1:1 read of `to_bits_2()`.
- **Every parse FAILS CLOSED.** `UNKNOWN` is a value the corpus asserts 408
  times, so minting it from a parse failure would forge an assertion the source
  never made — pinned by
  `unrecognised_topology_fails_closed_and_never_becomes_unknown`.
- **Citation identity never derives from the title** — pinned by
  `reference_identity_survives_a_title_rewording`, the falsifier that matters
  for an LLM-generated corpus. Where no stable identifier exists,
  `CitationKey::ContentAddressed` says so explicitly rather than synthesising a
  bibliographic id.

### Gates

`lance-graph-contract` **1178/1178** (7 new); `cargo fmt` clean; `cargo clippy
--all-targets --no-deps -D warnings` clean.

### NOT frozen by this PR (measurement says do not)

The endpoint codebook and the DisMech-local identity scheme stay OPEN until the
unprefixed population is probed — 63.6% of unprefixed endpoints are still
unresolved and the tail contains three different KINDS (provenance leakage,
mechanism propositions, lexical variants). Full numbers:
`E-DISMECH-CORPUS-CENSUS-1`. Also open: phenotype resolution must be scoped to the EXISTING `Domain::Phenomenology` (HP is its populated vocabulary, not a new domain — see the ⊘ correction on that entry)
(23.7% of resolved phenotype labels are HP/MONDO-ambiguous), and
`phenotypes[].category` at 261 distinct exceeds the 255 `Codebook` cap.

---
## 2026-08-20 — branch `claude/lance-graph-stage-3-recovery-2wrdbd` — S3.0 CLOSED AS NOT-NEEDED (no new type); #973 retraction + two overclaim corrections

### Contract Inventory — NO CHANGE

**This PR adds no type, no module, no bit, no tenant, no layout version.** It
is a retraction + audit. `crates/lance-graph-contract/src/lib.rs` is byte-clean
against `main`.

### What was withdrawn, and why

A first draft of this recovery PR reintroduced #973's `CausalLiteral`
(`4 × u16`, 8 bytes) with corrected prose. **The operator stopped it before
merge, and was right on four counts** — all now recorded as
`E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1` instance 2:

1. **Wrong universality.** Its own test asserted `TREATED_WITH` — the structure
   is GENERIC. Causality is a predicate family / qualification, never universal
   identity.
2. **`4 × u16` is not absolute.** MedCare-rs `ONTOLOGY_BAKE_STATE.md`:182:
   *"real OBO ids run past `u16` (MONDO:0700092 = 700,092)."* The V3 rail
   already solves this.
3. **A WordNet overclaim.** #875 was cited as an "EXACT structural encoding"; its
   own W5 gate reports **256 cells, occupancy median 255, over 65,292 leaves** —
   a locality/search prior, never an identity encoding.
4. **`routing_prefix()` was unearned.** Lexicographic prefix over concatenated
   ordinals, labelled an "HHTL locality projection", with no consumer and no
   measurement.

### The question that closed the slot

> **WHAT EXACT INFORMATION CANNOT BE EXPRESSED BY THE ADDRESSING THAT ALREADY
> EXISTS?**

**Nothing demonstrable.** `identity_quad::IdentityQuad` (operator-RATIFIED
2026-08-17) already carries **four exact external identities as `4 × u24` in one
96-bit V3 facet** behind a `classid(4)`, refuse-don't-truncate, bake-time
crosswalk resolution. It strictly dominates the withdrawn type. Siblings:
`ogar_elk::ClassAddr` (`u32 + u32`, a pre-bake join key by its own doc),
`canonical_node::NodeGuid` + HHTL, the V3 OBO rail.

Per the operator's rule — no concrete falsifier ⇒ **no new absolute-address
type.** S3.0 is closed as NOT-NEEDED, not filled because a plan had a slot.

### The genuinely open addressing gap (different, not fixed here)

`ClassId = u16` (`class_view.rs:54`) is near-exhausted for **relations** —
MedCare-rs `CLAUDE.md` #10: *"cannot address a relation — 11 prefixes, 8 of 280
ids over the ceiling."* A classid-mint capacity question for OGAR/lance-graph;
to be raised with the operator in session, not patched from a consumer.

### Where the work actually is

| | ADDRESS | HYDRATED SoA | TRAVERSAL |
|---|---|---|---|
| Bible / Rosetta | yes | **NO** | partial / context |
| OSM | yes | yes | overlay / junction |
| MedCare ontology | yes | yes | **YES, Stage 1** |
| DisMech oracle | source | structured | causal oracle |

The empty column is not ADDRESS. Next: **hydrate epistemic / causal nodes →
reason over them → think about the reasoning**, with the DisMech oracle
experiment as the gate (hide mechanism intermediates → hydrate the addressed
neighbourhood → let NARS/recipes recover candidates → compare against DisMech
truth).

Full audit: `.claude/handovers/2026-08-20-s3-0-cold-start-recovery-audit.md`.
Stage-2/2.5/2.6 (#971) untouched; #970's CE64 layout untouched.

---
## 2026-08-20 — lance-graph #971 (MERGED, `2cbe62d`, head `d627f5c`) — Stage 2 carves + Stage 2.5 census + Stage 2.6a V3 invariance + CE64 ⇄ V3 losslessness

Six commits, four stages. The per-stage Contract Inventory and results are in
the five dated sections below (written as the stages landed) — this entry is the
merged-PR record and the one-screen index over them; full Added/Locked/Deferred
is the `PR_ARC_INVENTORY` entry.

| stage | commits | what changed |
|---|---|---|
| Stage 2 | `ed2fe8b` | `MaturityPolicy` / `SkipReason` / `run_with`; **CAS·ETD·SDD·ICR** carved to production. `run()` behaviourally unchanged. |
| Stage 2.5 | `99794f2` `a765ef3` | `dissent_over` extraction + `Tactic::moves_confidence()`; the 5,760-cell paired census. Headline **corrected**: `0/5,760` → **1,098** same-family (0 lost) and **384** cross-family. |
| Stage 2.6a | `44bc5ab` `04cb2e1` | `cache::stage26_v3_parity` — V3 invariance on the planner's real CE64 leg. **Discordance = 0.** Brief's premise falsified and re-scoped (operator-ratified). |
| losslessness | `d627f5c` | `CausalEdge64 ⇄ CausalEdgeV3` is **bit-identical** modulo the deduplicated SPO and the deprecated v2 `temporal`. |

**The one thing a future session should not have to rediscover:**
`InferenceType` is a LOSSY compatibility projection of the 4-bit signed
mantissa — `to_mantissa(from_mantissa(m)) != m` for **8 of 16** states, the
`pack_v2` default `0 → +1` among them. Never route the mantissa through it on a
conversion path; carry the raw nibble. The three CE64-v2 tail fields
(`w_slot`, truth/topology, spare/band) cross as **RAW ORDINALS** — preservation
is not a provenance upgrade.

**Still open after this PR** (all Stage-3 inputs, none of them defects to patch):
`TD-THOUGHTCTX-IS-A-LOSSY-PROJECTION` (the 17 confidence-mute kernels + the
capability-vs-reachability question, review thread deliberately left open),
`TD-KERNEL-IDENTITY-FINGERPRINT-RAIL` (ARE/ZCF/HKF),
`ISS-PEARL-VOCABULARY-WITHOUT-PEARL-MECHANICS` (ICR),
`TD-CAUSAL-EDGE-IS-EXCLUDED-SO-CI-NEVER-LINTS-IT`.

---

## 2026-08-20 — branch `claude/carve-nars-kernels` — CE64 ⇄ V3 conversion losslessness (Stage-3 handoff gate)

### Current Contract Inventory — 8 new read accessors on `CausalEdgeV3`, no layout change

- **`CausalEdgeV3`** (`crates/causal-edge/src/edge_v3.rs`) — still exactly 12
  bytes (`const _` size assert unchanged); **two previously-dormant reserved
  bytes are now allocated**:
  - `[8]` = `w_slot(6 low) | truth/topology RAW(2 high)`
  - `[9]` = `spare/ReasoningBand RAW(3 low) | reserved(5 high)`
  - `[10..12]` still reserved (pinned by a test that asserts they stay zero).
- **New read accessors** (read-only; no new setters, per the scope fence):
  `frequency` · `confidence` · `causal_mask` · `direction` ·
  `inference_mantissa` · `plasticity` · `w_slot` · `truth_raw` · `spare_raw`.
- **`rehydrate` carries the RAW mantissa.** It no longer routes through
  `InferenceType::from_mantissa → pack → to_mantissa` (a lossy compatibility
  projection that rewrote 8 of 16 states, `0 → +1` among them). The
  `InferenceType` argument to `pack` is now an explicitly-labelled throwaway
  placeholder, overwritten by `set_inference_mantissa`.
- **No CE64 layout touch, no `ENVELOPE_LAYOUT_VERSION` bump, no new type, no
  `ThoughtCtx` wiring, no Stage-3 semantics.**

### The conversion contract, now pinned

`CausalEdge64 → from_v1 → rehydrate(same resolved SPO)` is **bit-identical**,
asserted as whole-register equality over 6 varied non-zero edges (all 4 truth
ordinals, `w_slot` at both ends of its 6 bits, spare across its 3, mantissa on
both signs). Under the v2 layout the 64 bits are fully partitioned, so field
parity *is* bit parity — and the whole-register assertion is what would catch a
field a future session forgets to enumerate.

Two exclusions, both principled:

| excluded | why |
|---|---|
| the 24-bit in-edge SPO | intentionally deduplicated into the target node's CAM-PQ facet; resupply it and the round trip is exact |
| the deprecated v2 `temporal` | not valid CE64-v2 state (bits 52..63 are the reclaim zone). NOT mapped into V3 TE — TE stays an independent producer-set signed chain offset |

`w_slot` / truth / spare are preserved as **RAW ORDINALS**: ordinal `01`
crossing means "ordinal 01 preserved", never "`IndirectKnown` is now
source-authoritative".

### Gates

- `causal-edge`: **72/72** under the default v2 layout, **38/38** under
  `--no-default-features` (v1). 11 tests in `edge_v3`, of which 7 are new.
- **5 disable-runs, each verified red-then-green** — the old lossy mantissa
  path (3 tests red), and each of the `w_slot` / truth / spare / `from_v1`
  tail carries individually.
- `cargo fmt --check` and `cargo clippy -D warnings`: **zero hits in
  `edge_v3.rs`** in both feature states. (The crate is workspace-EXCLUDED, so
  CI never lints it; 7 pre-existing clippy errors in `edge.rs`/`tables.rs` are
  untouched and now recorded in `TECH_DEBT.md`.)
- **Requirement 4 — the Stage-2.6 planner parity harness is untouched and
  green** (`cache::stage26_v3_parity`, 4 passed / 1 `#[ignore]`d generator).
  Both harnesses are needed and neither subsumes the other: the planner leg
  only ever carries `InferenceType::Deduction` (mantissa `+1` — a *surviving*
  state), so it was structurally blind to the mantissa defect. See
  `E-THE-COMPAT-ENUM-WAS-EATING-HALF-THE-REGISTER-1`.
- Downstream consumer `cognitive-shader-driver::edge_v3_compare`: **3/3**.

---

## 2026-08-20 — branch `claude/carve-nars-kernels` — Stage 2.6a: V3 representation invariance on the planner's REAL CE64 leg (measurement only)

### Current Contract Inventory — no new types; one `#[cfg(test)]` census

- **`cache::stage26_v3_parity`** — `#[cfg(test)]` sibling under `cache/`,
  compiled out of every non-test build. Needs no visibility widening: every
  symbol it uses (`NarsEngine`, `SpoHead`, `SpoDistances`, `CausalEdge64`,
  `CausalEdgeV3`) is already public. 4 gates + 1 `#[ignore]`d artifact
  generator.
- **No production code changed.** No second reasoning engine, no V3-native
  NARS, no CE64 layout touch, no `ThoughtCtx` wiring.

### Scope correction, operator-ratified

The original Stage-2.6 brief assumed `CE64 → recipe/runbook/planner`. **That
arrow does not exist** — see `E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1` for the
three checks. Building a V3 entrance there would have produced
`discordance = 0` for a trivial reason. The real uncovered leg is the planner's
`cache/nars_engine.rs`, and that is what this covers.

| surface | V3 invariance |
|---|---|
| `causal-edge` own `syllogize` | ✅ pre-existing |
| `cognitive-shader-driver` emission path | ✅ `edge_v3_compare` |
| **planner `cache/nars_engine.rs`** | ✅ **this** |

### Result

**Planner V3 representation discordance = 0**, by exact equality over 13
invariants per leg (rehydrated CE64 · SPO after resolution · NARS
frequency/confidence · causal mask · inference class · `SpoHead` round-trip ·
`forward_edge` conclusion · that conclusion's `SpoHead` · `syllogize`
conclusion edge · truth frequency/confidence · expectation). Representation-
specific fields (V3 Lokal target, TE, payload width) deliberately excluded.

The sweep spans every `inference` discriminant `to_causal_edge` maps — including
the two Pearl-rung translations (local `7`→`Intervention`, `8`→`Counterfactual`)
and the lossy `5 | 6`→`Synthesis` fold, which is exactly where a round-trip
could diverge — × every 3-bit pearl mask × truth rails and midpoint × both
palette rails. `temporal` is swept NON-zero on purpose: `to_causal_edge` passes
it to `pack` where the v2 layout makes the write a no-op, so if that ever stops
being a no-op the V3 arm (whose `rehydrate` packs `0`) diverges and this harness
says so instead of the change landing silently.

### Falsifiers — three disable-runs, all red-then-green

| assertion | disable | observed |
|---|---|---|
| resolution-conditional equivalence | corruption made a no-op | falsifier FAILS |
| the V3 arm actually goes through V3 | bypass `rehydrate`, use the direct edge | primary stays green, **falsifier FAILS** — which is exactly the vacuity it exists to catch |
| the compose tables are not inert | make every table the identity | degeneracy guard FAILS *(only after strengthening — see below)* |

**A disable-run corrected one of my own claims.** The degeneracy guard first
asserted "`forward_edge` changed the edge on some leg" and documented that as
proving the tables live. Measured: identity tables left it **green**, because
`forward` also composes the NARS truth. The discriminating form is SPO-specific
(`spo_of(fwd) != spo_of(input)`); both are kept, with the measurement written
next to them.

### JC's role, and where it stops

Every quantity at this seam is exact (`u8` indices, `u8` truth bytes, a `u64`
register), so **no naturally continuous quantity exists here for a correlation
to characterise and none was manufactured**. `binary_association` summarises the
syllogism-presence cross-tab — both categories occur, κ **defined** and 1.0.
Exact discordance is the contract.

### Artifact

`docs/probes/stage26-v3-planner-parity-discordance.csv` — one row per
discordant invariant. **Header-only is the result**, and it is the shape that
stays useful the day something diverges.

### Recorded, not patched (Stage 2.6b)

The recipe/runbook surface is causally blind; the 17 Operational-but-mute
kernels are the output-side twin of the same projection gap. Both are Stage-3
wiring questions — `E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1` and
`TD-THOUGHTCTX-IS-A-LOSSY-PROJECTION`. The 3 Demonstrations (ARE/ZCF/HKF) stay
honest Demonstrations; their blocker is a genuine substrate deliverable.

### Gates

`lance-graph-contract` 1171/1171 · `lance-graph-planner` 368/368 + 2 ignored ·
fmt clean · clippy `-D warnings` clean on both.

## 2026-08-20 — branch `claude/carve-nars-kernels` — codex + CodeRabbit review corrections: the consumer filter was filtering on the wrong predicate

### Current Contract Inventory — 1 new trait method (`crates/lance-graph-contract/src/recipe_kernels.rs`)

- **`Tactic::moves_confidence(&self) -> bool`** — non-defaulted, like
  `requires` and `maturity`; 34 per-kernel declarations. Answers the question
  the dissent channels actually ask, which **nothing on the trait previously
  exposed**: `writes()` is the census of `&mut ThoughtCtx` mutations, while
  `delta_conf` is applied by `run()` afterwards and is deliberately a separate
  effect. Measured: **31 kernels are `Operational`, only 14 can move
  `delta_conf`, and 0 declare `ThoughtField::Confidence` in `writes()`.**
- Pinned two-sided by `moves_confidence_matches_observation` (over- and
  under-declaring both fail against the probe matrix) and by
  `moves_confidence_is_strictly_stronger_than_production` (the implication is
  checked, and so is the strictness — 14 < 31 — so a future reader cannot
  conclude the two predicates are interchangeable).
- `maturity_operational_implies_an_effect` now reads the DECLARATION rather
  than re-deriving it; re-deriving would let the test pass against a lie.

### The consumer predicate is corrected — and the Stage-2.5 headline inverts

`StyleStrategy::watcher_can_dissent` filtered on `maturity().is_production()`.
`Operational` is a disjunction (*mutates a field* OR *moves confidence*) while
both channels compare only `tc.confidence`, so the filter **removed 3 mute
watchers and admitted 17 more** — preserving the exact budget loss it was
introduced to remove. `Cas` and `Etd`, both carved to production in this same
arc, are the sharp cases: both rewrite `candidates`, both return `0.0` forever.

| channel | sample changed | verdict change | direction | κ (fired) |
|---|---|---|---|---|
| same-family | 4080/5760 (70.8 %) | **1098**/5760 | 1098 gained, **0 lost** | 0.6307 |
| cross-family | 4224/5760 (73.3 %) | **384**/5760 | 366 gained, 18 lost | 0.8098 |

`n10 = 0` on same-family is the coverage argument as a measurement: removing
only mute watchers cannot remove an objection. The 18 on cross-family are the
strided sampler, not the filter. Full record:
`E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1`.

### Review findings addressed (7 total, all valid)

**codex (2 × P2)** — the predicate above; and the cross-family verdict label
reduced `(RungLevel, Mechanism)` to the rung, so a mechanism swap at the same
rung would have counted as agreement while the report claimed the mechanism
agreed. Both components now encode into one nominal label (`cross_label`).

**CodeRabbit (5 × minor)** — α and the per-rung independent unit now stated on
the Clopper-Pearson ladder; the "byte-identical" wording for `run()` replaced
with behavioural compatibility; the census chunk guard now asserts `rung`
alongside `style_idx` and `k` (the push order is style → rung → k → tol, so
two of three checks could pass on a broken chunking assumption); the report
header's blockquote continuations fixed (source indentation after `\n` turned
them into an indented code block); the `mean |Δcount|` header cell's pipes
escaped (7 cells against a 6-cell rule).

### Found while fixing the above, and worth its own line

`render_report` wrote **`Verdict change: 0/{n}` as a hardcoded literal** — it
would have printed zero regardless of the measurement, and did for one revision
after the predicate was corrected. Now computed from `binary_association`; the
re-pinned headline test is what caught it.

### Artifacts

`stage25-consumer-filter-verdict-discordance.csv` was header-only and is now
**1,482 rows / 57 KB** — the file whose emptiness was the result now carries
the flips, which is exactly the shape it was built for.

### Gates

`lance-graph-contract` 1171/1171 · `lance-graph-planner` 364/364 + 1 ignored ·
fmt clean · clippy `-D warnings` clean on both.

## 2026-08-20 — branch `claude/carve-nars-kernels` — Stage 2.5: the consumer-filter census (measurement only, no semantic change)

### Current Contract Inventory — no new types; ONE behaviour-preserving extraction + a `#[cfg(test)]` instrument

- **`StyleStrategy::dissent_over(style, ctx, rung, tol, admitted, watchers)`**
  (private) — the shared body of both dissent channels, extracted verbatim.
  Takes the ALREADY-SAMPLED watchers rather than the predicate, and returns the
  objecting `&'static Recipe` rather than either channel's return shape, so a
  caller holding a differently-filtered sample exercises the **shipped** verdict
  body instead of a reimplementation — **without production growing a probe
  knob**. `peripheral_dissent` / `cross_family_dissent` are now two `.map()`s
  over it.
  **Proven behaviour-preserving, not asserted:** the 22 pre-existing
  `style_strategy` tests pass unchanged, and the full 5,760-cell verdict
  signature is **byte-identical** pre- and post-extraction (0/5760 differences).
- **`strategy::stage25_census`** — `#[cfg(test)]` child module of
  `style_strategy` (a CHILD, so it reaches `dissent_over` / `watcher_can_dissent`
  without widening either to `pub(crate)` for an instrument). Compiled out of
  every non-test build. 5 gates + 1 `#[ignore]`d artifact generator.
- **`jc` dev-dependency: already present**, added for D-BLW-3. The census is its
  SECOND consumer under the identical standing constraint — dev-only, one
  direction, statistics consume observations and nothing feeds back. The comment
  in `Cargo.toml` now names both consumers; no new edge was created.

### Artifacts (`docs/probes/`)

- `stage25-consumer-filter-census.md` — the human-readable report (A watcher
  effect · B verdict effect · C pre-verdict numeric · D stratification ·
  headline, graded per channel against a stated rule).
- `stage25-consumer-filter-census.csv` — 1,440 rows, one per
  style × rung × k × channel. **Collapsed over `tol` losslessly**, licensed by
  the pinned invariant that the watcher sample cannot depend on `tol`
  (`stage25_tolerance_cannot_move_the_watcher_sample`). The un-collapsed form
  was 11,520 rows / ~1 MB with 7 of every 8 rows byte-identical — not "compact"
  in any sense the brief meant.
- `stage25-consumer-filter-verdict-discordance.csv` — one row per verdict flip.
  **Header-only IS the result**, and it is the shape that stays useful if a
  future run flips something, where an all-concordant dump would bury the one
  row that mattered.

### Headline

Same-family **weakly** (25.0% of configurations, mean Jaccard distance 0.1265,
retention 0.9028) and cross-family **materially** (41.7%, 0.2599, 0.7993) change
watcher sampling; **0/5,760 paired configurations changed verdict** on either
channel, on the fine elevation-target label, κ = 1.000000 and defined. Full
numbers + the clustering ladder for the zero-event bound:
`E-THE-COVERAGE-FIX-IS-REAL-AND-ASYMMETRIC-1`.

### Guardrails held (each checkable)

Stage 2.5 changed **no** Stage-2 verdict (0/5760, measured, not argued) · no
watcher-selection change · statistics consume observations and never feed back ·
`jc` is instrumentation, not a semantic dependency (dev-only, and the edge
pre-existed) · **no pre-verdict score was added** to obtain part C — the
`|Δconfidence|` margin stays local to `dissent_over`, and what was measured
instead is the finer outcome the harness already exposes · no inferential test
manufactured over an exhaustive deterministic census · style preserved as a
stratum despite being verdict-inert — and it proved to be the largest factor on
one channel.

## 2026-08-20 — branch `claude/carve-nars-kernels` — the NARS recipe-kernel carve: a maturity gate + four kernels moved from placeholder to production

### Current Contract Inventory — 2 new types + 1 new field + 3 new policy pins (`crates/lance-graph-contract/src/recipe_kernels.rs`)

- **`MaturityPolicy { Any, ProductionOnly }`** (`Default = Any`) — which
  [`KernelMaturity`] levels a dispatch will let RUN. `ProductionOnly` is the
  policy a dispatch that spends a BUDGET wants: a `Demonstration` lands no
  effect by construction, so a channel sampling `k` kernels and asking whether
  any moved the answer must not spend slots on kernels that structurally
  cannot.
- **`SkipReason { GatedOff, NonProduction(KernelMaturity) }`** — carried on the
  new **`Outcome::skip: Option<SkipReason>`** field (always `None` when
  `fired`). `fired: bool` alone conflated "the gate said no on this context"
  with "the dispatch refuses non-production kernels"; a caller that cannot
  tell them apart cannot report either honestly.
- **`Tactic::run_with(&mut ThoughtCtx, MaturityPolicy) -> Outcome`** — the
  policy-carrying sibling. **`run()` is behaviourally unchanged** — it now
  delegates with `MaturityPolicy::Any`, so its implementation differs while
  every existing caller's observable result does not (pinned by
  `run_still_means_any`). The earlier wording here said "byte-identical", which
  was wrong about the implementation and right only about the behaviour
  (CodeRabbit, PR #971). The policy is checked BEFORE `gate()` on purpose: a
  `Gate`-bucket `Demonstration` sitting in `GateState::Flow` would otherwise
  report `GatedOff` and the refusal would be invisible. A refused kernel never
  sees `ctx` — not `gate`, not `apply`.
- **Policy pins (named, documented as pins, not measurements):**
  `NEUTRAL_SCORE = 0.5` (was an unnamed literal inside `Sdd`),
  `DISTORTION_WEIGHT = 0.2`, `POLE_SENSITIVITY_WEIGHT = 0.15`.

### Four kernels carved: 27 Operational → 31; 6 Demonstration → 3; 1 Stub → 0

| id | code | was | now | what it does now |
|---|---|---|---|---|
| 8 | CAS | Demonstration | Operational | quantizes `candidates` onto the rung's HDR grid (`hdr_level`) — it computed the level and dropped it |
| 22 | ETD | Demonstration | Operational | splits at the widest adjacent gap, keeps the upper cluster; declines when no gap exceeds `NOISE_FLOOR` — it sorted a CLONE and discarded it |
| 31 | ICR | **Stub** | Operational | split-pole SENSITIVITY: `\|1 − 2·free_energy\| · confidence`, charged. **Explicitly still NOT a Pearl `do()`** — the doc rewrite is in the same commit and `ISS-PEARL-VOCABULARY-WITHOUT-PEARL-MECHANICS` stands |
| 32 | SDD | Demonstration | Operational | charges the distortion it already detected, PROPORTIONAL to the deviation, with an empty-field guard — it detected and returned a hardcoded `0.0` |

`ARE(19)` / `ZCF(24)` / `HKF(34)` remain Demonstrations, blocked on ONE shared
substrate deliverable (see `TECH_DEBT.md`
`TD-KERNEL-IDENTITY-FINGERPRINT-RAIL`). They are the three the
`ProductionOnly` policy refuses today, pinned as such.

### Consumer (`lance-graph-planner`)

- **`StyleStrategy::watcher_can_dissent(id)` + `watcher_is_eligible(r, want,
  same_family)`** — one named predicate both dissent channels sample against,
  so the falsifier proves the property the channels actually run. Coverage
  fix; **measured to change no verdict** across a 5,760-cell sweep, and
  documented at the call site as exactly that (`E-A-WATCHER-THAT-CANNOT-
  DISSENT-IS-NOT-A-WATCHER-1`).

### Gates

`lance-graph-contract` 1169/1169 + all 4 examples green (`recipe_claim_audit`
G1–G4 ALL GREEN after re-pinning; `sound` 30 → 31);
`lance-graph-planner` 359/359 + 22/22 `style_strategy`; `cargo fmt --check`
clean; `cargo clippy --all-targets --no-deps -D warnings` clean on both.
**Nine disable-runs, every one red-then-green** — policy ignored (3 tests),
policy checked after the gate, CAS rung ignored, ETD uniform guard removed,
SDD fixed-cliff, SDD empty guard removed, ICR constant charge, plus both
halves of the consumer falsifier.

### Re-pinned, not silently widened (each with the reason in the test)

`context_blind_kernels_are_input_invariant` (4 → 3, ICR is no longer blind);
`maturity_discriminates_and_is_not_all_one_label` (`stub == 1` → `== 0`, plus
a new `demonstration == 3`); `requires_matches_apply_reads` +
`requires_masks_are_varied_not_a_constant_stub` (`empty == 4` → `== 3`);
`icr_builds_counterfactual_via_xor_self_inverse` → `icr_charges_pole_
dependence_and_is_silent_at_the_midpoint`; the `recipe_claim_audit` arms for
8 / 22 / 31 and its `G2`.

## 2026-08-20 — lance-graph #970 (MERGED, `781c3b9`) — board hygiene owed and now paid

### Current Contract Inventory — 2 additive lenses over CE64 bits 59..63 (`crates/causal-edge/src/{layout,edge}.rs`)

- **`CausalTopology { Direct, IndirectKnownIntermediates, IndirectUnknownIntermediates, Unknown }`**
  over bits **59..60** — ordinal-compatible with the existing `TrustTexture`
  band that already occupies those bits (`TRUTH_SHIFT`), so it is a second
  READING of stored bits, never a relocation. No layout version, no CE64 v3.
- **`ReasoningBand`** (8 levels, `Surface … Transcendent`) over bits
  **61..63** — the previously-`SPARE_SHIFT` lens.
- **`CausalEdge64::{topology, reasoning_band, with_topology, with_reasoning_band}`**
  — consuming builders matching the crate's existing `with_truth` / `with_spare`
  convention (the brief's `set_*` phrasing was bent to current main, not the
  reverse).
- **Wire-format impact: none.** `_LAYOUT_COVERAGE` unchanged; every existing
  accessor reads the same bits it did before.

## 2026-08-19 — lance-graph #969 (MERGED `b67f195`) + dismech-rs #7 (MERGED `36f7466`) — the ownership + addressing reassessment lands; HHTL is canonical AND dormant

**The state this establishes, for any session that touches addressing:**

- **The canonical row is `key(16) | edges(16) | value(480)`** and **HHTL is
  its FIRST tenant** (`canonical_node.rs:706-730`). It is not missing, not
  elsewhere, and not owed a new carrier. The `32 × (4+12)` shape cited by an
  earlier plan came from a Java **fixture** that says so itself
  (`lance-graph-java/native/lgj-abi/src/rowstore.rs:5-8,33-39`) — a fixture
  conforms to canon, it never defines it.
- ~~**HHTL is ZERO on every baked row in both production bakes**~~
  **⊘ WITHDRAWN 2026-08-19 (same day) — inert-artifact false positive.**
  True of `obo-core.soa` (0/96 sampled; cause identified —
  `OGAR/crates/ogar-obo/src/lib.rs:349-353` zero-inits and skips key bytes
  4..12 by design). **FALSE of `all-lanes.soa`**, the current production
  golden image (394 MB, 2026-08-15, 770,360 × 512 B): **352/2,048 sampled
  records (17.2%) carry non-zero key bytes 4..10**, and the key's 6 bytes
  are a **verbatim prefix of a 24-byte positional trie path in the value
  region** (`is_a` at `value[44..56] ++ [68..80]`, `part_of` at
  `[56..68] ++ [80..92]`) with key-prefix ↔ rail-head agreement **314/314,
  zero disagreement**; deepest live path 13 levels, so the series
  continuation is in use. The all-zero reading came from a container whose
  `.data/` is gitignored and did not survive the reset — the read path was
  observing a MISSING substrate, not a dormant one. `atlas.rs:898-910` warns
  about this exact failure mode in its own doc comment.
  **The residual true statement:** the mint exists and is sound; what is
  wrong is the DOCUMENTATION — five artifacts contradict the bytes they
  describe (a stale sibling manifest; no `SHA256SUMS` for the tag; a
  published schema still calling HHTL *"dormant, zero on all shipped rows"*;
  a bake-state doc describing only the superseded V1 reading; and this
  entry, until now).
- **Five hand-rolled ancestor mechanisms** exist in one repository; two of
  them agree only **58.2%** (`atlas.rs:465`) because spanning-tree depth and
  DAG longest path are different quantities. So "use the rails" is a
  SEMANTIC change owing a written ruling, not merely a faster one.
- **`obo_store::compute_cascade` already derives HEEL/HIP/TWIG in RAM at
  load** — it is the mint, misplaced.
- **ONE HHTL fabric, several ClassView-resolved READINGS** (operator,
  2026-08-19; this is what replaced the withdrawn `AddressingMode` enum):
  DN/Zipper for tree-ish lineage (parent = truncate, ancestor = prefix),
  paired rails `6×(u8:u8)` for coupled axes, and **basin locality** where a
  relation is genuinely many-to-many (HHTL names the neighborhood, explicit
  edges carry membership). **Basin is NOT overflow** — it is a positive
  locality design, never a spill zone. Which reading a class admits is
  decided by measured topology, never by taste.
- **CE64 / EW64 are codecs and projections**, never the canonical semantic
  container. Do not mint a tenant because a packed format would be
  convenient.
- **Morton is non-canonical** — the separate `4⁴` construct, shipped and
  unconsumed as research. HHTL ancestry is never derived from it. **HHTL is
  never "V1"**; the retired V1 shape is the flat u24 *tail*.
- **Ownership:** upstream DisMech → **dismech-rs = ORACLE** (corpus fidelity
  + falsifiers only) → ogar-dismech / ogar-from-dismech = interpretation and
  mint bridge → **lance-graph = canonical substrate** → lance-graph-java =
  mechanical ABI mirror. No layer pulls another's semantic responsibility
  downward.

**Open, and gating:** the **canonical-depth ruling** — spanning-tree depth
vs DAG longest path. It is an operator ruling, not a finding, and it gates
D-HTT-9 and the first bake mint.

**In flight (not in this PR):** a pre-Phase-1 archaeology of the historical
DN/"Zipper" deep-hierarchy precedent (MedCare `RAIL_OFFENE_POSTEN.md`
Posten 21 key-facade `4..16` + edge-tenant `16..24`, and Posten 31's
2026-08-17 refinement to one register grammar), to be reused or explicitly
falsified before any HHTL mint is written.

## 2026-08-19 — ARCHITECTURE RESET (operator): DUMB STORAGE × JAVA MECHANICAL API × HHTL EPISTEMIC SPINE — freeze/seal implementation STOPPED

- **PR #968 MERGED** (merge commit 66fec27; operator-merged 14:05Z) — the
  seal STORNO + finalization map + register-grid correction + 5+3-ratified
  spec. Docs/board/probe only; zero substrate code. See PR_ARC entry.
- **Minutes later, the operator issued the ARCHITECTURE RESET** (verbatim
  charter: `docs/architecture/DUMB-STORAGE-RESET-CHARTER.md`): STOP the
  freeze/seal-centered implementation (task #25 W1–W4 do NOT launch);
  preserve the research history + falsifiers by exact reference (they are,
  in the merged #968 paths above). The new central model: an intentionally
  STUPID substrate (references, HHTL hierarchy — trie AND explicit
  reference nodes, ClassView, WideFieldMask-as-fovea, DatasetVersion,
  temporal coordinates, zero-copy) with ALL semantics (ontology, causality,
  rung, known-unknowns, awareness, nudges) as interpretations ABOVE it.
  15 falsifiers pre-registered in the charter §18. Arc order A→E
  (charter §19); ARC A = source archaeology of lance-graph +
  lance-graph-java, NO CODE until the §20 ten-item map is returned.
- The prior F-ORD/seal narration in the entries below stays accurate as
  HISTORY: `content_hash` (FNV) still exists in source — its deletion was
  part of the stopped implementation, not of the merged docs.

## 2026-08-18 — lance-graph #961 (MERGED 0cc171f) — LOTUS research arc opens: frontier audit + F-ORD-REAL falsifier

> **2026-08-19:** PR #964 merged (a73cddf) — RP-SEAL pass-1 consolidation
> (`docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` + Appendix H) with §0
> operator STORNO: canonical replay coordinates are core; compaction =
> optional storage economics; retention/tombstone + QueryReference
> projection are the temporal core. Tier 0 probes are the next arc.

### Current Contract Inventory — no new types (docs + tests only)

- `docs/lotus/LOTUS-FRONTIER-AUDIT.md` — Phase 0 archaeology; the verified
  F-ORD defect chain (`content_hash` folds arrival-minted `stream_position`,
  `persist_sink.rs:414`, contradicting the struct's own doc); seal = 3×
  O(bytes) passes with the batch resident up to 3× (descriptor doctrine
  unrealized: `BatchWriter<Vec<u8>>`); lance crate source ABSENT from sandbox
  (Phase 6 BLOCKED); §6: cycles become THIN not permeable (Regime A
  pipelining; rung-qualified frontier; linear-text vs tile split per class).
- `docs/lotus/F-ORD-REAL-FALSIFIER.md` + `cycle_driver.rs` tests: GREEN
  defect pin `f_ord_real_defect_pin_arrival_order_leaks_into_batch_hash`
  (fails loudly when a fix lands → deliberate re-pin) + `#[ignore]`d RED
  `f_ord_real_publication_identity_is_arrival_order_independent`.
- Gates: supervisor suite 28 passed / 1 ignored (`--features cycle-driver`);
  RED verified red under `--ignored`; fmt clean; zero new clippy warnings.

## 2026-08-17 — branch `claude/hydrate-staging-merge` — `lance-graph-hydrate`: the deferred staging/publish merge lands (closes `ISS-HYDRATE-DIR-AND-FILE-DUPLICATE-THEIR-STAGING-BODIES`)

### Current Contract Inventory — new private module (`crates/lance-graph-hydrate/src/publish.rs`)

- **`publish::publish_by_rename(staging, publish_path, StagingKind) -> Result<(), PublishError>`** — the PUBLISH half of `copy::hydrate_dir` and `file::hydrate_file`'s near-identical bodies, extracted per the `crate/lance-graph-hydrate`-hardening council's own named follow-up (`.claude/board/ISSUES.md` `ISS-HYDRATE-DIR-AND-FILE-DUPLICATE-THEIR-STAGING-BODIES`, filed 2026-08-17, resolved same day once the zero-consumer window it was priced against was confirmed still open). The FETCH half (list+get many objects vs stream+hash one object) stayed per-caller — genuinely different in shape, merging it would obscure more than clarify.
- **A real improvement, not just deduplication**: `publish_by_rename` runs BOTH a pre-rename re-check AND a post-rename ENOTEMPTY remap for BOTH callers. Before this: `hydrate_dir` had only the post-rename remap (no pre-check — its window was wider than it needed to be); `hydrate_file` had only the pre-check (its real defense, since POSIX file-onto-file rename SILENTLY CLOBBERS rather than erroring, so a post-rename remap can't see the danger case). Unifying to BOTH strictly narrows each window; it weakens neither.
- **`publish::remove_staging(staging, StagingKind) -> io::Result<()>`** — also absorbed the fetch-error and empty/checksum-reject cleanup call sites in both callers (not only the publish tail), so the ISS's "own cleanup ladder" half is closed as completely as the "own rename-race remap" half. Callers choose swallow (`let _ = ...`) vs propagate (`...?`) per call site — the Ok-path-must-not-lie behavior from the prior council pass (C7) is preserved exactly, not weakened by the extraction.
- **Gates:** 4 new falsifiers in `publish.rs` (publish-a-dir, publish-a-file, and — the two that could not exist before this seam — a directory-race and a file-race test, each constructing a competing publisher already at the destination and asserting BOTH `AlreadyPublished` (not a raw I/O error) AND that the winner's content survives untouched). All 4 pre-existing `hydrate_dir`/`hydrate_file` tests pass unchanged — the merge is behavior-preserving at every previously-tested path.
- **Verification status — same honest caveat as the prior two PRs on this crate**: not locally compiled in this session's container (disk-constrained; see the 2026-08-17 entries below for the full history). Real verification is this repo's CI (`rust-test.yml` et al.), watched per the standing PR-ownership protocol.

## 2026-08-17 — branch `claude/lance-graph-hydrate-hardening-council` — 5+3 council hardens `lance-graph-hydrate` (merged PR #957); corrects 3 claims made in the entry below

**Council record:** `.claude/plans/hydrate-crate-hardening-council-v1.md` —
5 savants (26 distinct findings) → consolidated draft v2 (20 decisions) → 3
reviewers (zero BLOCK, 6 FIX) → this follow-up PR. Run AFTER #957 merged
(the merge landed mid-council, before Phase 4); this is therefore a
fast-follow, not an amendment to the merged PR.

**Three claims in the entry below were WRONG and are corrected here (the
entry itself is left unedited per the append-only rule — this is the
correction, not an edit):**

1. `LifecycleState::can_flush` was claimed "encoded as a transition guard,
   not caller discipline." **False** — no function anywhere in the crate
   took or returned a `LifecycleState`; it was a checkable predicate a
   caller could ignore entirely. Now: `dirty::lifecycle_of()` exists and
   actually produces a `LifecycleState`, closing the gap the original
   claim asserted was already closed; the doc language is corrected to
   match what's true.
2. `dirty::is_dirty` was claimed to compare against "the version recorded
   at hydration time." **False** — nothing in the crate ever recorded a
   hydration version; `hydrated_at_version` was, and remains, entirely
   caller-supplied. Doc corrected to say so plainly.
3. The idempotency boundary was described as having "the" (singular)
   condition, collapsing the doctrine's two conditions into one. Corrected
   to name both: (b) empty/uncontested destination is enforced by this
   crate; (a) pinned source version is structurally the CALLER's
   responsibility (or, for the single-file case, IS the SHA-256 pin).

**Real defects found and fixed, not merely doc corrections:** a
within-process staging-nonce collision (`pid+nanos` alone can repeat within
one clock tick — `staging.rs`, new shared `staging_suffix()`); every
fetch-phase I/O error leaked its staging directory/file (now cleaned up on
every path); an `Ok`-path cleanup failure could silently falsify the
"leaves nothing" postcondition (now propagates); `hydrate_file`'s
entry-only existence check left the actual danger case — a file-onto-file
rename SILENTLY CLOBBERING an existing destination — completely unguarded
(now re-checked immediately before the rename, not only handled on the
rename's `Err` branch); `release_dir` could structurally never return `Err`,
making its own missing-directory test vacuous by this repo's own
falsifiability rule (now returns `Err` on a genuinely unreadable root,
`Ok(0)` only for `NotFound`); the warm-marker's 2-bare-integer format had no
version tag and permissive arity (now `v1 <mtime> <len>`, exact 3-token,
both a legacy-shaped and a future-wider line refused).

**Also corrected:** `copy.rs`'s module doc overclaimed that the
hydrate-aside/publish-by-rename mechanism was "generalized from
`hydration_probe.rs`'s proven mechanism" — that probe has no staging step
at all; only the raw byte-copy property is inherited from it, and the doc
now says so.

Full per-decision ledger (C1-C20) and per-finding evidence (F1-F26): see
the council plan file. No code outside `crates/lance-graph-hydrate/` was
touched.

## 2026-08-17 — branch `claude/q2-osm-map-reencoding-56p5e2` — `lance-graph-hydrate`: the generic SoA→S3→volume→Lance hydration crate (closes the named gap)

> **Minted here, not in OGAR** (operator directive): "OGAR is only the intermediary who should help to inherit the pattern as a plug and play pattern; however the pattern itself should be minted in lance-graph already." OGAR (and q2, and any future consumer) is meant to depend on this crate as a path/git dependency, never re-implement it.

### Current Contract Inventory — new crate (`crates/lance-graph-hydrate`, new workspace member)

- **Closes `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE`** (`.claude/board/ISSUES.md`, filed on the 2026-08-06 s3-hydration-lifecycle PR #901 review round): *"Something of the shape `hydrate_from(remote) -> VersionedGraph` ... is the missing piece."* This crate is that missing piece, generalized past `VersionedGraph` to any consumer — extracted from q2's own repo-local, already-working implementation (`cockpit-server/src/osm_slab_hydrate.rs` + `osm_lance.rs`) and this repo's own doctrine + measured probe (`crates/lance-graph/examples/hydration_probe.rs`, `dev_s3_env.rs`).
- **`env::HydrationSource`** — the ONE shared reading of the `AWS_*` env vars (same names `dev_s3_env.rs` already reads, so a deployment already configured for `lance-graph` needs zero new configuration to also use this crate).
- **`lifecycle::LifecycleState`** — `Absent -> Hydrated -> {Dirty | Flushed}` with the doctrine's one hard rule encoded as a transition guard, not caller discipline: `can_flush()` is `true` ONLY for `Hydrated` (never `Dirty`); `can_hydrate()` only for `Absent`; `can_release()` for `Hydrated`/`Flushed`.
- **`copy::hydrate_dir`** — the `Absent -> Hydrated` edge: a raw byte-for-byte object copy (`store.list` + `store.get`), generalized from `hydration_probe.rs`'s proven T10 mechanism (never a `Dataset` scan-and-rewrite, which silently drops deletion vectors/indexes/version history). Hydrate-aside/publish-by-rename: objects land in a private staging directory first, ONE atomic rename publishes — a caller observing the publish path sees nothing or the complete artifact, never a partial one. Refuses (`HydrateError::AlreadyPublished`) rather than overwriting an existing destination — the idempotency-boundary condition from the doctrine.
- **`file::hydrate_file`** — the single-artifact sibling (a `SHA256SUMS`-pinned sidecar, not a whole Lance directory), generalized from q2's `download_verified` (`.part` + atomic rename + checksum verification before publish).
- **`marker::WarmMarker`** — the mtime+len skip-rehash trust marker, generalized from q2's own invention (`osm_slab_hydrate.rs::trusted_via_marker`) — **not present anywhere in lance-graph's doctrine before this crate.**
- **`release::release_dir`** — idle page-cache release via `posix_fadvise(POSIX_FADV_DONTNEED)`, generalized from q2's own invention (`osm_slab_hydrate.rs::advise_dontneed`, `#[cfg(unix)]`/`#[cfg(not(unix))]` no-op pair) — also not previously in lance-graph's doctrine.
- **`dirty::is_dirty`** — the `.claude/plans/idle-flush-dataset-eviction-v1.md` §4 dirty-detector this repo's own `hydration_probe.rs` measured and gated: compares the CURRENT local `Dataset::version_id()` against the version recorded at hydration time — never a directory content hash.
- **Deliberately NOT built**: the automatic age+footprint-driven eviction SWEEP policy from the idle-flush plan (still a PROPOSAL) — this crate ships the mechanisms the policy would call (hydrate / dirty-check / flush-gate / release), not the scheduler.
- **Deps**: `lance = "=9.0.0"` + `object_store = "0.13"` (this repo's upstream-authoritative lance-family pins, unchanged — no new version anywhere in the graph, confirmed via a clean 15-line `Cargo.lock` diff with zero new package versions).
- **Verification status — HONEST, not claimed green**: the pure-Rust modules with no external deps (`lifecycle.rs`, `marker.rs`'s non-test code) were syntax/type-checked directly via `rustc --emit=metadata` (clean, no errors) in this session's container. The `lance`/`object_store`-touching modules (`copy.rs`, `file.rs`, `dirty.rs`, `env.rs`) were NOT locally compiled — this session's sandboxed container hit repeated `ENOSPC` (disk-quota) trying to build the `lance`/`arrow`/`aws-lc-rs` sub-tree that this crate's addition activates for the first time in this workspace's build graph (a previously-dormant, unbuilt transitive edge — `Cargo.lock` gained zero new package versions, only the new `lance-graph-hydrate` node itself). Real verification is deferred to this repo's own CI (`rust-test.yml` et al.), which runs on a machine with proper disk headroom — the PR is watched and any CI-reported compile/test failure will be fixed and re-pushed per this session's standing PR-ownership obligation.
- **Gates:** 8 modules × 2-4 tests each (lifecycle guards two-sided per state; marker mtime+len identity + trust/distrust; copy hydrate-aside/publish-by-rename + idempotency-boundary + empty-prefix no-publish, all against a real `object_store::local::LocalFileSystem` — not a mock; file checksum-pin accept/reject + idempotency-boundary; dirty-check against a real written-then-appended `Dataset`; release_dir file-count + missing-dir-is-zero-not-error) — written TDD-first, execution pending CI per the honest status above.

## 2026-08-13 — rail-trie geometry registered: the address places the node

### Current Contract Inventory — 1 new module, 1 new ClassView resolver

- **`contract::rail_geometry`** — deterministic node placement from the rail
  registers: `RailAxis` (Taxonomy/Mereology), `RailCarving`
  (`InterleavedPairs` 6×stride-2 in the key facet | `AxisSlab` 12(+12
  discontiguous cont)×stride-1), `RailPath` (hole rule: a value after a zero
  is not ancestry; empty path = the lane's dominant root), `TriePlacement`
  (`ring` = depth, `arc` ∈ [0,1) = radix fraction of the slots), and
  `dual_rail_placement` (primary axis PLACES, secondary OVERLAYS — two
  hierarchies on one canvas, only one of them places). The neo4j-shaped
  invariant is proven, not styled: a child's arc lands inside its parent's
  half-open interval `[arc, arc + 256^-depth)`, siblings order by slot,
  and the placement is a pure function of the row — two loads render
  identically, no solver, no scene model. The f64 boundary is pinned as a
  passing test (exact through level 6; order-preserving beyond; a glove
  needing deeper discrimination reads `slots()` directly).
- **`ClassView::rail_carving(class, axis)`** — the reading, registry-resolved
  per class like `edge_codec_flavor`: default = canon zero-fallback (key
  facet pairs at `4..16`); a bake that measured its way to a different
  carving overrides. The slab variant exists because a consumer bake
  MEASURED the pair reading and rejected it for its hierarchy (44.25 % of
  paths fit vs 99.62 % in twelve per-axis levels). Selection only — no
  carving changes `NODE_ROW_STRIDE`.
- **Boundary kept:** not a renderer (every glove reads ONE resolved
  placement; none re-derives), not a distance (the CLAM-side geodesic lives
  with the compute crate). Compute-side counterpart: ndarray `clam_v3`
  (`RailSpec` mirrors the two carvings; merged there first).

## 2026-08-12 — open-review sweep of #920–#929 — three ledger figures were wrong, and one audit method cannot do what it claims

### Current Contract Inventory — no new types (corrections to merged #927/#928 entries; append-only, corrected here not in place)

Operator: *"930 has comments … check also previous 5 … if you want go back another 5."* All 68 review comments on #920–#930 enumerated and checked against the tree rather than against the merge — **with non-uniform depth, stated rather than glossed:** #922–#930 finding-by-finding; **#920 (27) and #921 (5) spot-checked on their P1/governance items only** (both clean — `p1_noise_floor.py` is explicitly `SUPERSEDED by p1_ci_vs_floor.py`; the merged #917 entry has exactly one commit in its history, so it was never mutated), **leaving ~30 older findings there UNVERIFIED**. Within the verified range: **#922, #924, #925, #929: zero review comments. #926's 27 findings: all verified fixed in the tree** (equal-budget `grid_pts`, seam-wrapping `subgrid_min`, `find_center → None`, F7d 35→40, CT-F12 `NO-VERDICT-INSUFFICIENT-N`, persisted storm metadata, `np.roll` longitude, `__file__`-relative JSON write, net-decay E6, the 93–97 %→90.9–94.3 % headline). **#923's plan P2 is resolved** (`Status: ACTIVE (audited 2026-08-11)`). Three findings on **#927/#928 were still open**, all of them numbers frozen in append-only ledgers:

- **⊘ CORRECTION to #926's entry (line 29 below), codex P2 on #927 — "a +92.76 Pa offset moves R² in the 5th decimal" is FALSE, and the report's own carve table refutes it.** Measured: **+92.76 Pa (carve A) moved R² 0.9212 → 0.9129 = 0.0083, the THIRD decimal**; **+1.59 Pa (carve D) moved it 0.943406 → 0.943403 = 2.4e-06, the SIXTH.** The sentence fused carve A's magnitude with carve D's insensitivity. **The true and sharper statement: the `var()` BUG was blind at EVERY magnitude — that is what hid +92.76 Pa — while the STATISTIC is near-blind only in the single-digit-Pa regime, which is exactly where "lossless" was claimed on four-decimal agreement.** The banked doctrine (RMSE + mean bias in Pa beside every R²) is unchanged; only its illustration was mispaired. **Third instance this week of the same shape as #930's relation error** — two individually-true numbers asserted of one pair when each belongs to a different one.
- **⊘ CORRECTION to #927's `Added` line, codex P2 — "10 probe scripts with committed JSON results" undercounts.** Measured on the #926 first-parent diff: **15 `.py` added, 11 of them with a committed `.json`** (the four `ev*` scripts have none). The immutable scope summary omitted delivered work.
- **⊘ CORRECTION to #928's audit figures (line 17 below), codex P2 — "+13/−0, +10/−0, +0/−0" is wrong in two places.** Measured `a4e264c5..3dae97be`: LATEST_STATE **+13/−0** ✓, PR_ARC_INVENTORY **+17/−0** (never +10 at any commit in the range), and EPIPHANIES **does not appear in the net diff at all** — it was **+1/−1 at `0f9e6bcd`**, the in-place edit of a merged entry, which the revert zeroed. So the "+0/−0" was the *product of a revert*, not evidence of purity — and read correctly it is the stronger result: **the audit WOULD have caught the violation mid-PR.**
- **⚠ AND THE AUDIT METHOD ITSELF IS WEAKER THAN CLAIMED (codex P2 on #928).** *Zero removed lines proves ADDITIVE, not PREPEND* — an insert in the middle of a ledger, or an append at the end, also deletes nothing. Since #928 banked it as "worth reusing", a future session would inherit a test that cannot detect a mid-file insertion. **The correct one-command pure-prepend test is the SUFFIX check** (CodeRabbit's own analysis chain used it): `git show origin/main:F` must be a suffix of `git show HEAD:F` — i.e. `new.endswith(old)`. Zero-deletions stays useful as the cheap first screen; the suffix check is the one that proves the property.
  - **Both halves measured before banking this, per the falsifiability rule.** Constructed falsifier: a true prepend gives `zero-del=True, suffix=True`; an **append at the END** gives `zero-del=True, suffix=False` — caught only by the suffix check, which is the discrimination the old method lacks.
  - **And it fires on THIS PR, correctly.** `origin/main..HEAD` for #930: `PR_ARC_INVENTORY` / `STATUS_BOARD` / `INTEGRATION_PLANS` all `suffix=True`; **`LATEST_STATE` `suffix=False` at `+15/−0`** — the old screen passed it, the new one flags it. The flag is right and the change is still legitimate: #930 inserts shipped-PR table rows **mid-file** (§Recently Shipped PRs) and composes an **unmerged** entry in place (the PR-878 allowance). **A `False` is not automatically a violation — it is a demand for the justification to be stated.** That is the property worth having; the old test could not even ask.

## 2026-08-12 — lance-graph #929 (MERGED) — steering rescue dead; sign test retired as primary instrument

### Current Contract Inventory — no new types (2 probes + report §5.12/§5.13 + 1 epiphany)

- **CT-F16 `[G]`:** the steering-level moderator FAILED both pre-registered bars on CT-F14's own 19 storms (0.579 vs 0.70; residual +28.4 % wider; level sweep monotone toward the surface, best 850 hPa). §9.2's "single most promising fix" is superseded in place. The height ladder survives as a measurement; its operational reading died.
- **The instrument calibration `[G]`:** a 90°-rotated control scored **0.684 = CT-F14's headline**. At n=19 the sign test separates from a wrong answer only at 14/19. **Rule: attach a deliberately-wrong referent to any rate-headline; its score is the floor.**
- **The replacement instrument:** circular resultant (R̄, μ, Rayleigh) resolves the same rows at **p=0.0050** (sign test: 0.0835), estimates the offset (**−30.2°±36.5°**) instead of penalizing it, and separates the rotated control in **BOTH channels** — vs surface (the pair the sign test conflated at 0.684 = 0.684): R̄ 0.343 vs 0.516 AND μ −130.5° vs −30.2° (100.3° apart); the steering↔rotated pair confirms rotation-invariance by construction (identical R̄ 0.343, μ shifted exactly 90°). Post-hoc — instrument demo, NOT a promotion. **Sign-consistency numbers are no longer verdict-grade anywhere in this arc.**
- **Faltung frame:** resultant = first circular Fourier coefficient; CT-W6 = deconvolution (components ⊛ apparatus noise); `Z_256` circular Faltung is `DistanceLut::circular()`-native.
- **Queued, operator go-ahead pending:** **CT-W6** (dipole = neighbor far-field + bow-wave vector sum, global fit, 38 obs — the mechanistic candidate for the −30° offset and the 0.684 plateau) · CT-W2s/W5/W7 (sunflower collision facets, spiral-ADI via Fibonacci-stride pairs, Gegendruck) · CT-F17 (fresh-sample verdict) · C-register climatological calibration.

## 2026-08-12 — lance-graph #927 (MERGED) — board hygiene for #926 + the living-vs-ledger correction rule

### Current Contract Inventory — no new types (board entries + one report correction)

- **Fisher-z ring-mean ratio is 4.7×, not 5×** (18.07 / 3.84). Caught by verifying the new board entry's own figures against the committed JSONs before landing it: **9 of 10 exact, this was the tenth** — a 6 % overstatement in the favourable direction.
- **⚠ LIVING DOCUMENTS vs APPEND-ONLY LEDGERS take OPPOSITE correction discipline.** A **living document** (report, code, JSON, PR description) is landed on directly → **fix every copy**. An **append-only ledger** is read newest-first and its value is the audit trail → **freeze the merged entry, correct in a NEW one**. I edited a merged `EPIPHANIES` entry in place and unmarked while citing the *grep-for-its-twins* rule; reverted same-PR. **A correct rule generalized past its domain** — third instance of that shape in this repo (#921's doctrine-vs-domain, this arc's Fisher-z rank/tail-vs-level).
- **One-command append-only audit:** `git diff origin/main..HEAD -- .claude/board/` — **zero removed lines proves a pure prepend**, since a prepend cannot delete. Measured +13/−0, +10/−0, +0/−0 across the three board files.
- **Note on assurance:** #927 was a draft through four review cycles, so **no automated reviewer read these entries** — CodeRabbit skips drafts and the explicit trigger hit the rate limit. The figures rest on self-verification against the committed JSONs.

## 2026-08-12 — lance-graph #926 (MERGED) — the storm spine, the 12-byte L4 carrier, and the propagation failure mode

### Current Contract Inventory — no new types (probes + report + board; zero Rust, zero product code)

- **The spine `[G]`:** a surface low = **center address + 14 logical fit values** (~12 ring means + a 2-value wn-1 dipole) = **90.9–94.3 %** of in-disk MSLP variance, replicated across three independent blind samples (1980–2021, 41+ storms, four seasons). Not 93–97 % — that figure was a **36-parameter per-ring fit**, not the 14-value model the storage claim describes.
- **It fits the REAL carrier `[H]`:** le-contract §3 **L4 is a PAIR** — `6 × (8:8)`, `palette256²`. The single byte is the *selector*; the pair is a centroid-tile cell. A **12-byte facet** recovers the f64 spine to **0.07 Pa RMSE** with a **+1.59 Pa bias**. **14 logical values ≠ 14 bytes** — keep model size and carrier budget apart.
- **Fisher-z is per-read, not universal `[H]`:** **8.3× tighter** than plain rank in the storm tail on the raw field; **4.7× worse** than uniform on ring means. **Wins a RANK/TAIL read, loses an INTERPOLATE/LEVEL read** — which is why an L4 ClassView **MAY** (not must) declare an analytic codebook.
- **The directional claim is `[S]`, measured-unsupported:** CT-F14 (the properly-powered pre-registered test) = **0.684, p=0.0835 → NO-VERDICT**. The pooled figure crosses p<0.05 but is **gated** — the pre-registration had no contingency for its largest component failing its own n≥20 floor.
- **⚠ Two methodological rules this PR banks, both worth carrying forward:**
  - **R² is structurally near-blind to encoder bias.** `var(y−ŷ)` (used at 11 sites) drops the squared mean residual; in-disk variance ~1e5 Pa² means a **+92.76 Pa** offset moves R² in the 5th decimal. "Lossless" was inferred from the one statistic that could not detect the loss. **Report RMSE and mean bias in the physical unit beside every R².**
  - **The dominant defect class is PROPAGATION, not judgment.** All five documentation defects were *a claim corrected in one home and left standing in another* — prose vs artifact, body vs heading, code vs JSON, report vs PR description. When correcting a claim, **grep for its twins first.**
- **Still open, operator calls:** CT-F16 (score the dipole against **steering-level** motion — the one measured, unwired moderator) · the moist budget (θe/precip are **proxies**, not a closed entropy budget) before CT-M1..M3 · C-register climatological calibration (single-timestep today) · `domino.rs` is a *fixed tridiagonal kernel*, so the moderator model is unbuilt, not unwired.

## 2026-08-11 — lance-graph #924 (MERGED) — evaluation plan ACTIVE after a 0-of-11 spec audit

### Current Contract Inventory — no new types (plan + audit + corrections)

- **`.claude/plans/weather-substrate-evaluation-v1.md` is now ACTIVE** (audited 2026-08-11, §8). §3 carries **v2, post-audit** specs — every v1 spec had failed. Read §8 before citing any EV.
- **The audit's shape is the headline:** **22 of 24** KNOWN claims CONFIRMED with `file:line`, **0 of 11** EV specs SOUND. Reliable about *what is*, unreliable about *what would falsify* — banked as `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1`. **Pre-registration review by independent adversarial readers is load-bearing, not ceremony.**
- **Two `[G]` ledger corrections:** the bitboard primitives are at **`ndarray/src/bitwise.rs`**, not `src/hpc/`; and **a SPRITE moves for 2 bytes, not the field** — `morton_shift_motion_probe` legA rigid-translates a 24×24 sprite within a 256×256 field, toroidally (§12.18 corrects §12.16; whole-field advection is one `(dx,dy)` **per tile**, which is what EV-1 actually tests).
- **Status of the queue:** EV-1..EV-10 all **Queued, none RUN** — a v2 spec is *audited*, not *validated*. **EV-9 (Wave 0) needs no data** and closes the only two `[H]` rows (K-12/K-13, whose scratch reproducers were deleted). Everything else in Wave 1 waits on one `fetch.py` re-fetch; Wave 2 is the 16k×16k scale run.
- **Still open, operator calls:** D-1 noise floor · D-2 saturation window · D-3 `from_bearing` API · D-4 dormant-lane fix shape · D-5 helix CI wiring · D-6 harness-of-record.

## 2026-08-11 — lance-graph #923 (MERGED) — the evaluation plan + `DistanceLut::circular()`

### Current Contract Inventory — one new public constructor; one new plan

- **`helix::DistanceLut::circular()`** — `min(|a−b|, 256−|a−b|)`, the cycle-graph geodesic on `Z_256`. **A metric, proven EXHAUSTIVELY: 0 violations / 16 777 216 triples.** Use for any quantity whose range **wraps** (bearing, phase, angle); `linear()` is simply the wrong table there — it puts index 255 and 0 at *maximum* distance when they are adjacent. Same `[a,b]` amortization as every other constructor.
- **`.claude/plans/weather-substrate-evaluation-v1.md`** — the arc's known-vs-test ledger. **§1** 25 KNOWN claims with `file:line` + per-row grades; **§2** the honesty split (only K-12/K-13 lack a committed reproducer); **§3** EV-1..EV-10 in three waves (0 = no data · 1 = one fixture re-fetch · 2 = scale); **§5** the D-1..D-6 operator decision register. **Ships DRAFT-pending-audit** — §8 folds a 13-agent verify/attack pass and flips it ACTIVE.
- **The `[a,b]` amortization, stated `[G]`:** `quantize()` normalizes once per element at ingest (`quantize.rs:99`); `from_floor()` folds the SAME normalization into the table (`distance.rs:39`). Afterwards there is no `lo`, no `hi`, no division — a pure index lookup in **unit-free** units. That is what licenses cross-variable comparison, and it is O(256²) once rather than O(N²).
- **Field-vs-element, locked:** judge a normalized representation by what its **field** does. nearest-`n` = ONE palette index → 2 × LUT u8 lookups, L1 metric, CAKES-safe, `U8x64`, feeds `int8_gemm_amx_tiled(a_u8, b_i8, …)` directly. u8-palette azimuth under `circular()` = **0.352° mean**, beating nearest-`n`'s 0.972° while keeping that shape. Raw u16 azimuth is circular and tileless.
- **The stack is BUILT, not proposed** (§12.17): `perturbation-sim` (`rolling_floor` = the CI-threshold frame, Jirak-cited · `splat`/`sketch` = the magnitude/sign two-algebra rule · `cascade_key::morton48` · `hhtl` by Cheeger bisection), `ndarray::hpc::splat3d` (3DGS, `TILE_SIZE=16`, depth cascade already HHTL), `symbiont/domino.rs` (16 SoA boards per AMX 16×16 tile GEMM, real `TDPBF16PS`). **The `mu+kσ` frame ships 4× — new probes must be its 5th instance, not its 5th implementation.**
- **Still open, operator calls:** D-1 noise floor · D-2 saturation window · D-3 `from_bearing` API · D-4 dormant-lane fix shape · D-5 helix CI wiring (`[G-absence]`: excluded from the root workspace, in no workflow) · D-6 harness-of-record.

## 2026-08-11 — lance-graph #921 (MERGED) — the wind reuse, measured: the prescribed bearing-encode is 10× worse than the direct one

### Current Contract Inventory — no new types (one committed measurement, one doc section, board hygiene)

- **`crates/helix/tests/bearing_encode_paths.rs`** — the two candidate `bearing → Signed360` encodes, measured at N=65536. Runs **only by hand** (`helix` is root-workspace-excluded and in no CI workflow `[G-absence]`).
- **The invention/reuse line `[G]`:** *invention* = asserting structure the code already answers; *reuse* = applying the shipped codec to a new domain, which is what the normalized substrate exists for. **A missing entry point for a designed reuse is a plumbing gap, not a design refusal.** This corrects #920 §12.12, which declined to build the bearing-encode on the wrong grounds.
- **The finding the reuse surfaced `[G]`:** `helix-cartesian-vs-fisher2z.md` prescribes *nearest spherical-Fibonacci `(n, sign)`* for encoding a direction — **and it does not fit weather.** Horizontal bearings measure **1.9–2.7°** via nearest-`n` vs **0.000°** via a direct `(polar, azimuth)` write (mean over 24 cases: **0.972° vs 0.097°, 10×**). The golden spiral couples latitude and azimuth through ONE index, so a bearing at the horizon cannot be chosen independently; and the lattice is equal-area on the **disk**, giving latitude density ∝ `sin(2·lat)` — sparsest exactly at the equator. Normals spread over the sphere and never hit this; wind always does.
- **Rule locked:** **a doctrine written for one domain is not automatically right for the next one that reuses it.** The doc's prescription is correct *for normals* and should carry its case rather than be read as universal.
- **`[S]` — no public `from_bearing` minted.** The measurement settles which path; the API shape is an operator call and must not be built on before it is made.

## 2026-08-11 — lance-graph #920 (MERGED) — the probes ran; the doc that queued them was falsified in four places

> **⊘ This entry CORRECTS two claims in the #917 entry below it.** (1) **"helix360" is not a symbol** — the type is **`Signed360`**; pickaxe over full history returns 12 blobs, all authored by that session, zero deletions `[G-absence]`. (2) **The "2 × 24-bit hemispheres = wind in/out" reading of the `HelixResidue` lane is WRONG.** Per `.claude/knowledge/helix-cartesian-vs-fisher2z.md` — the doctrine doc with a `READ BY:` header naming exactly that kind of session, which the arc never opened — **one 6-byte `Signed360` is a complete full-sphere direction**: the `polar` sign-partition completes the sphere, so there is no second hemisphere to pair. `ResidueEdge`/`rim` is the **METRIC** carrier (`DistanceLut`, L1); `(polar, azimuth)` is the **render** carrier. The `Pair48` mint is **WITHDRAWN, not deferred**. Nothing below is deleted; it is read through §12.12 of the weather knowledge doc.

### Current Contract Inventory — no new contract types (probes, one example, three tests, one doc section)

- **`probes/weather-p1/`** — 8 re-runnable Python probes over real ARCO-ERA5 + README + 4 result JSONs. The fixture is **fetched, not committed** (and was cleared by a container reset; results re-verified for internal consistency without it). `p1_noise_floor.py` is retained **superseded, for provenance** — its `dev` is a decoded reconstruction error, the metric §12.10 rules out.
- **`crates/jc/examples/weather_substrate_reliability.rs`** — Pearson / Spearman / Cronbach α / ICC over the probe export, header-validated, with a `--shuffle` negative control.
- **`crates/helix/tests/signed360_claims.rs`** — 3 tests, each **disable-verified red-then-green**: azimuth spans the full circle (min 0 / max 65535 / 256-of-256 arcs), the polar partitions fill `[128,255]` and `[0,127]` exactly, and — pinning a **defect, not a virtue** — an all-zero lane decodes as a definite `Sign::Neg`. **Scope:** `helix` is excluded from the root workspace and named in no CI workflow `[G-absence]`, so these run **only by hand**.
- **The evaluation frame, locked:** compare each point's **bucket confidence interval** against the noise floor — **never a round-trip / decoded reconstruction error**. A one-way address over a retained original has no round-trip to score.
- **Measured `[H]`** (real ERA5, ONE timestep): Fisher-Z on weather anomalies is an **address-economy** failure, not a validity failure, at a 0.5–1 K floor (0.848 % / 0.820 % saturated, the two paths otherwise indistinguishable); at a **0.25 K** floor it becomes a validity failure too (+95.65 % interior-CI exceedance). **Standardization, not Fisher-Z, is what licenses cross-variable comparison** — 0.9997 on a shared palette vs 0.857–0.875 raw cross-unit.
- **Angular error of `Signed360` by latitude (measured this session, N=65536):** equator (|lat| 0–5°) **0.112° mean / 0.226° max**; pole (85–90°) **3.332° / 4.998°** — a **~30× spread**, best at the rim. **No resolution gain from the sign split at equal budget**: 7-bit `|y|` + sign vs 8-bit over `[-1,1]` measures **0.99–1.02× in every band** (step 1/127 vs 2/255 = 0.996×). The 256-sample *codebook* figures that suggest a √2 gain (full-sphere cap 7.17° vs hemisphere 5.07°) spend an **extra** bit; at equal budget 128-on-hemisphere = 7.17°, identical. **What the partition buys at the rim is sign EXACTNESS, not precision** — and a wind bearing is near-horizontal (`y ≈ 0`), i.e. exactly where a centred-at-128 round loses the hemisphere (#498). Caveat: the encoder's n-lattice is equal-area on the **disk**, so latitude density ∝ `sin(2·lat)` — peaked at 45°, thinning at both equator and pole; harmless for decode coverage (the `(polar, azimuth)` product grid is dense), but it is why there is still no arbitrary-bearing → code path (`from_normal`).
- **Still open, all operator calls:** the saturation-window widening; pinning a citable per-variable noise floor; the U-shaped-variable falsifier (`total_cloud_cover` / `sea_ice_cover`) that would promote the distribution-shape rule from `[S]`. **Filed, not fixed:** the `Signed360::sign()` dormant-lane defect.

## 2026-08-11 — lance-graph #917 (MERGED) — the normalized-substrate reference: palette256 × helix360

### Current Contract Inventory — no new types (docs-only; the primitives were already shipped)

- **`.claude/knowledge/weather-normalized-substrate.md`** — the product/engineering reference for weather AI on the substrate. Every code claim carries `file:line` + a `[G]`/`[G-absence]`/`[H]`/`[S]` grade; measured numbers carry their conditions AND a *not-yet-re-runnable* flag until probe P1 commits them.
- **What it establishes about already-shipped code (no new surface):** `helix::Similarity` is ONE `arctanh` core with two readings — `fisher_z` (variance-stabilizing statistic, `fisher_z.rs:14,55-58`) and `hyperbolic_depth` (`2·arctanh`, the Poincaré-disk arc length, `:60-78`). The encoder stores the **z-form** and never materializes geometry (`residue.rs:150,159`); `hyperbolic_depth` is called from **no encode path** (`[G-absence]`, grep 2026-08-11) — that is the DESIGN, not an unwired seam. Normalization is the load-bearing property: it makes 8 bits sufficient per scalar AND puts unlike variables on one scale, which is what licenses cross-variable LUT correlation.
- **helix360 = the existing `ValueTenant::HelixResidue` lane**, `ColumnKind::U8 × 6 @ row_offset 112` (`canonical_node.rs:837-840, 960-967`), documented there as *"2× the 24-bit equal-area hemisphere"* — i.e. inbound AND outbound bearing, in the width already reserved. `Signed360::{to,from}_bytes` (`residue.rs:92-105`) is ONE sanctioned reading of that content-blind register; 2×24 in/out is another (same doctrine as the V3 12-byte facet + `EdgeBlock` flavor rule).
- **`[G-absence]` no in/out PAIR-writer exists** — `encode_signed` emits one signed point (`residue.rs:182-204`); nothing in `crates/helix/src` or the contract composes the pair into the lane. Queued as probe P4; deliberately not invented inline.
- **Floor scoping RESOLVED as a policy, not a crate property** (read from `quantize.rs`/`distance.rs`): `quantize` is **linear** over a per-instance `[lo,hi]` window (`quantize.rs:99-108`) — all non-linearity is the upstream Fisher-Z — and `bucket_center` (`:248-250`) makes index meaning floor-dependent, so cross-variable code comparability is an **ingest calibration decision** (shared canonical z-floor / per-variable / hybrid), stamped by `floor_version` under the crate's own versioning contract (`quantize.rs:20-26`). Proposal `[S]`: freeze per epoch, align re-rolls with Lance dataset versions so `at(v)` rehydrates with the floor that produced the codes. Gated on probes P2/P3/P6.
- **Probe queue P1–P6** (P1 re-runnability · P2 shared-floor cross-variable · P3 per-variable resolution cost · P4 pair-writer · P5 `drift_sigma` under spatial autocorrelation · P6 Lance↔floor version alignment) — **all NOT RUN**. Per the process rule, the next deliverable is a probe, not more synthesis.

**Companion corrections in the same PR:** `weather-substrate-poc-v2.md` §0 gains append-only **⊘ C3** — the plan's Zarr object name `era5/1959-2023_01_10-full_37-1h-1440x721.zarr` does NOT exist; the session-verified object is `1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr` under `gcp-public-data-arco-era5` (`ar/` prefix). Grid math unchanged; only the name was invented.

## 2026-08-09 — branch `claude/phase-a-owned-writer` — Phase A: the artifact-backed commit contract + the SOLE owned Lance writer (`LanceCycleWriter`)

> **⊘ This entry SUPERSEDES the #911 entry below it** (operator ruling 2026-08-09). The §I.6 "every cycle publishes exactly one `DatasetVersion`" contract, the compensating delete, and the per-operation-reopen sink are all REMOVED — not repaired. Canonical record: `.claude/plans/persistence-artifact-backed-commit-v1.md`. Nothing below is deleted; it is read through that document.

### Current Contract Inventory — reshaped (lance-graph-planner) + replaced (lance-graph core, `planner` feature)

- **The governing storage rule** (`persist_sink`): **no artifact-backed semantic change → no write → no new `DatasetVersion`.** Thinking is cheaper than persisting its intermediate control flow: a thought runs its whole Rubicon ladder transiently and only an artifact-backed delta becomes durable. Mechanically the gate is the cast payload — NON-EMPTY = artifact cast (persisted), EMPTY = intent-only cast (held-intent re-stage / pure kanban step) which `persist_cycle` partitions out as **ephemeral**. Zero artifact casts ⇒ `CommitOutcome::NoChange { head }` with the sink NEVER called (zero store ops, zero rows, unchanged version). `restage_held`'s empty payload is thereby exactly what makes a re-staged intent ephemeral — no ABI gate can trip on it, because such a cast never reaches the writer. #911's deliberate empty-cycle versioning is REMOVED and its falsifier inverted.
- **`WalSink` reshaped**: `commit_cycle(&mut self, batch) -> Result<CommitOutcome, CommitError>` (the `base` param is gone — it lives in `batch.frame`), `scan_sealed(after_cycle: Option<CycleId>)` (cycle-bounded tail, pushed into the scan), `timeline() -> Vec<FrameMeta>` (replaces `versions()`; frame metadata only, never a payload). `LandedSlot` is now `{ cycle, slot }` — the derived `version` field is GONE (a physical publication position is not a per-row semantic identity). `DetachedCycleBatch` gained `batch_hash` (FNV-1a 64 over the CANONICAL content, so randomized completion order yields an identical hash).
- **Honest commit states**: `NoChange { head }` / `Committed { version, cycle, batch_hash }` / `Reconciled { version, cycle, batch_hash }`; errors `Fenced { current_head }` / `HashConflict { cycle, stored_hash, offered_hash }` / `Io(WriteFailed)` / `Ambiguous { cycle, batch_hash, cause }`. **No error may promise "nothing landed" when failure could have occurred after manifest publication** — that case reconciles or surfaces as `Ambiguous`.
- **`LanceCycleWriter`** (replaces `LanceCycleSink`) — the SOLE application writer: **non-`Clone`**, owns a **long-lived `Dataset` handle + in-memory head**, commits through **`&mut self`**. The capability split is deliberate (operator correction): `Clone + &self` belongs to producer submission and read-only projections; fire-and-forget means producers get no acknowledgement, NOT that the writer ignores the result. **No rollback, no compensating delete** — a published manifest is history; `Dataset::delete` mints another version and is not rollback (and #911's `(cycle, base_version)` predicate could destroy a concurrent same-cycle winner). Idempotency is durable and in-band: `(cycle, batch_hash)` committed with the rows, reconciled FIRST, so re-submitting the same frozen batch after a lost acknowledgement returns `Reconciled` instead of double-appending. Lance 9 has NO atomic expected-version fence for Append (rebase runs even single-attempt; strict mode is Overwrite-only — measured in `lance-9.0.0/src/io/commit.rs:914-950`); that is stated, not papered over.
- **Zero reload on the normal path, instrumented**: `LanceCycleWriter::opens()` counts every `Dataset::open` ever performed (startup + ambiguity resolution only). Layout is three row kinds — frame (1/cycle), landing metadata (1/artifact cast, payload NULL), coalesced image (1 per DIRTY ROW, the final 512-byte payload). Payload is physically `FixedSizeBinary(512)`; reads are bounded + projected (`timeline` never scans the payload column; `scan_sealed` returns transition metadata only; `scan_image` projects payload on request).
- **Gates:** 11 reopened-store falsifiers in `cycle_sink.rs` + 5 contract falsifiers in `persist_sink.rs`, including the **measured bytes-written** one — 64 transient breaths on one row cost **512 durable bytes, not 64 × 512**. Honestly deferred (named, not skipped): the real object-store RUN — S3 needs lance's `aws` feature, which our `lance = "=9.0.0"` default-features pin already enables (verified: `aws-config`/`aws-credential-types` in the graph), so an `s3://` store compiles and routes today; what is unmeasured is the credentialed commit/reconciliation/tail-read, and **no object-store durability claim is made** until it runs. Also deferred: `Continue`-from-pulse (Phase C), 64k-scale measurement.

## 2026-08-09 — branch `claude/medcare-rs-continue-ufsazd` — `lance_graph::graph::cycle_sink`: the CONCRETE cognitive-cycle Lance sink (the storage-proven `WalSink`)

### Current Contract Inventory — new module (lance-graph core, `planner` feature)

- `lance_graph::graph::cycle_sink` — **the gap `persist_sink` deliberately left is now closed**: a concrete implementation of `lance_graph_planner::persist_sink::WalSink` over the official Lance 9 insert path (`Dataset::write` / `Dataset::append` — the same `InsertBuilder` transaction machinery every Lance writer uses). No bespoke ledger, no acknowledgement protocol, no parallel replay system: Lance's own manifest/version chain IS the WAL. Gated on the default-on `planner` feature (the trait lives in the optional planner dep).
  - `LanceCycleSink::new(path)` — holds only the dataset path; opens the dataset per operation, so restart-survival is exercised on EVERY call, not just in tests.
  - `cycle_store_schema()` — one dataset, two row kinds: a per-cycle **frame row** (`kind=0`, the cycle ↔ version mapping sealed INSIDE the same atomic commit — zero sidecar state) and **landing rows** (`kind=1`: `stream_position`, `owner`, `row`, nullable `move_*` Rubicon-edge columns, `payload` witness-node bytes).
  - **The §I.6 fence, both halves.** Pre-commit: the store's current version must equal the cycle's sealed predecessor `base` (empty store ⇒ head `DatasetVersion(0)`), else refused with NOTHING written. Post-commit: the published version must be exactly `base + 1` (`sealed_version = base_version + 1` is an identity the commit path VERIFIES, never assumes). Lance has NO expected-version conditional append (the rebase runs even on a single-attempt Append commit; strict mode exists only for Overwrite — measured in `lance-9.0.0/src/io/commit.rs`), so a foreign interleaved writer can land the batch at `base + 2`; the review round (#911 Codex P1 + CodeRabbit) made the fence EFFECTIVE retroactively there: the just-published cycle rows are removed again with an official `Dataset::delete` scoped to exactly this cycle (the cycle id is an unsealed identity — the predicate is exact), and only then is the retryable `WriteFailed` returned, so "write failed" is TRUE at the visible head and the driver's regenerate-from-`Vn` contract stays sound. The one manual-reconciliation corner (compensating delete itself fails) names the orphaned version explicitly — reachable only when the one-writer doctrine was already violated by a foreign writer.
  - **The coalesced image is DURABLE** (review round): `kind = 2` image rows (`row → final 512-byte payload` after the per-row fold) persist in the same atomic commit as the landings; `LanceCycleSink::scan_image(cycle)` reads a sealed cycle's coherent end-state (projected `row`+`payload` under the kind+cycle predicate) while the per-cast history stays intact via `scan_sealed`. Every landing/image payload is gated to exactly `EPISODIC_WITNESS_BYTES = 512` (the canonical `key(16)|edges(16)|value(480)` node stride) BEFORE anything durable happens; `versions()` projects only `cycle`+`base_version` under `kind = 0` so the coarse-timeline lookup never materializes a witness payload.
  - **Order is a write-side property**: landings are stored in the already-deinterlaced `DetachedCycleBatch` order and scanned back with `scan_in_order(true)`; the sink never sorts on read.
  - **Domain-0x09 witness contract (module doc, operator-ruled):** the patient SoA at classid domain `0x09` is the ONLY place patient reasoning is written to Lance, so the store is witness-focused and maximally rich — `payload` carries the 512-byte canonical EpisodicWitness node (visited ontology addresses, executed crosswalk mappings, exact RO/ontology edge ids, supporting/contradicting/missing observations, NARS truth+confidence, differential branches), whose edges point INTO the immutable domain-0x03 ontology address space. The cycle takes ontology immutability for granted for its representation window (`base_version` names the sealed predecessor it read) and therefore never restates ontology content — the sealed versioning is a reflection of the thinking; downstream (the Gotham display, differential views) reads the sealed version, never a live recomputation.
- **Gates:** 6 tokio tests, every guarantee proven against a REOPENED dataset (fresh sink instance + fresh `Dataset::open`, never an in-memory echo): seal survives restart; stale base fenced with nothing written (store version chain, landings, timeline all untouched — checked, not assumed); sequential cycles chain V1→V2→V3 with strictly-after filtering; a zero-landing cycle advances the timeline only; an empty store reads empty (a state, not an error — `DatasetNotFound` distinguished from real I/O failure); move-nullability + 512-byte payload byte-exact round-trip. Real Lance version chain cross-checked against the returned `DatasetVersion` in-test.

## 2026-08-06 — branch `claude/vocab-tenant-bake` — `lance_graph_contract::identity_quad`: four external identifier spaces joined at BAKE time into one 96-bit facet payload

### Current Contract Inventory — new module (lance-graph-contract)

- `lance_graph_contract::ontology_warrant` — **grading a factfinder's ungraded
  verdict, without letting the grade leak back into the fact.** A factfinder
  (OGAR `ogar-elk` + siblings) answers exactly: entailed or not. That is rung 1
  and must stay exact. What IS graded is a different question — how well
  warranted a claim is given how many independent sources speak to it — and it
  is computed OVER facts, never replacing them. The two rungs are kept apart
  structurally: there is deliberately **no** method turning a `NarsTruth` back
  into an entailment.
  - `Quorum { corroborating, silent, conflicting }` — counts, not sources. The
    type cannot name who said what; provenance stays with the factfinder.
  - `Quorum::warrant() -> NarsTruth` — frequency is the share of *speaking*
    sources that corroborate; confidence grows with how many spoke.
  - `SourceVerdict { Corroborates, Silent, Conflicts }` — three variants, no
    `Unknown`: a source not consulted is not a source.
  - **The load-bearing rule: silence is abstention, not dissent.** A source with
    no path between two classes has not denied the relation, it has said
    nothing. Counting silence as dissent turns "the other ontology is sparser"
    into "the other ontology disagrees" — the opposite finding from the same
    data. Measured, not preferred: on a real cross-ontology comparison the
    sources that both spoke agreed 1,730 : 3 (99.8 %) while 1,693 were silent;
    folding silence into dissent reports ~51 %. Both numbers are computed in
    `the_measured_cross_ontology_case_reads_as_agreement` so the difference is
    visible rather than asserted.
  - Zero-dep and factfinder-agnostic: names no ontology, no vocabulary and no
    producer crate. Takes three counts, so any factfinder that can bucket its
    comparisons can feed it and the contract crate stays dependency-free.

- `lance_graph_contract::identity_quad` — the **4 x 24-bit identity tenant**. A row whose identity is asserted independently by four external identifier spaces (each in the 10^5-10^7 range) carries all four in ONE V3 facet payload, resolved once at bake time. Afterwards a read is a fixed-offset register read: no join, no crosswalk table consulted, no walk. The saving is not space, it is the disappearance of the read-time join.
  - `IdentityQuad` — `4 x u24` over the 12-byte payload, read/written **through `legacy_outliers::LegacyOutlier::WideTriple` (G2)**, not a parallel bit-math implementation. `from_slots` / `slots` / `slot` / `try_slot` / `with_slot` / `filled` / `into_facet` / `from_facet` (rides a real `FacetCascade`: classid in `0..4`, payload in `4..16`).
  - `IdentityCodebook` + `check_capacity` / `MAX_ENTRIES` — the bijective `key <-> ordinal` book. `try_new` **rejects** a non-injective key list (`CodebookError::DuplicateKey`) at construction, so a many-to-one mapping cannot exist to be discovered later; `verify_bijective()` is the explicit whole-book witness a bake runs. Overflow **refuses** rather than truncating, the same refuse-don't-widen discipline `codebook::Codebook` uses at its own 256-entry scale — and it is a **sibling of** that type, never a widening of it (a 10^6-entry space cannot be reached by splitting into 256-entry families without the split becoming the address).
  - `QuadJoin` — the bake-time join: four codebooks, one per slot, `resolve(keys) -> IdentityQuad` and the pull-back `explain(quad) -> keys`. An unknown key resolves to **absent**, never to a fabricated ordinal.
  - `IdentityCodebook::digest()` — **added in the review round**: a deterministic, length-delimited FNV-1a over the *ordered* key list. Ordinals come from sorted position, so an ordinal is a property of the whole key set; a book grown with an early-sorting key renumbers existing entries and a persisted facet then explains as a neighbouring key. `verify_bijective()` cannot see that (it is a within-book property; both revisions pass). The digest is what a bake records beside its rows so a later read can **refuse a shifted book instead of resolving a wrong key**. A **witness, not an enforcement**; append-preserving growth is deliberately absent (it would break the `binary_search` invariant), so growing a book is a **rebake**. Open residue: `ISSUES.md` `ISS-IDENTITY-CODEBOOK-ORDINAL-STABILITY`.
- **Encoding decision, load-bearing:** slots store `ordinal + 1`, so raw `0` means *absent* per the CANON zero-fallback ladder. Without the offset the zeroth entry of every codebook would be indistinguishable from a never-filled slot, and a partially-joined row would silently read as fully joined. The largest **slot value** is `MAX_ORDINAL = 2^24 - 2`; the largest **entry count** `check_capacity` admits is `MAX_ENTRIES = 2^24 - 1` (ordinals start at zero) — two separate constants, named separately because conflating them is what review caught.
- **Carving choice:** contiguous G2 `4 x u24`, deliberately NOT the axis-grouped `4x(8:8:8)` (`CascadeShape::G4D3`, `le-contract.md` §3 L5). That shape is three independently-meaningful bytes per slot — a rail reading. An exact identity split across three independently-read bytes is no longer a single invertible value, and invertibility is this tenant's acceptance criterion. **This puts the module in `legacy_outliers` territory, which `le-contract.md` §3a strongly discourages — surfaced, not resolved:** see `ISSUES.md` `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`.
- **Gates:** 14 new lib tests (12 + 2 from the review round), all falsifiable, each carrying its falsifier in the doc comment. Field-isolation matrix over all four slots (`I-LEGACY-API-FEATURE-GATED`); whole-book bijectivity sweep (not a spot check); the non-injective **negative** case; capacity gate asserted at both sides of its boundary; join anti-vacuity (genuinely multi-slot AND genuinely partial rows) plus can-stay-silent (unknown key stays absent). **Two guards were mutation-tested in-session** to confirm they bite: removing the duplicate check fails the injectivity test; dropping the absent sentinel fails four tests. `cargo fmt` + `cargo clippy -p lance-graph-contract --all-targets -- -D warnings` clean. Fixtures are synthetic; the module names no external identifier space.

## 2026-08-06 — branch `claude/s3-hydration-lifecycle` — object-store hydration doctrine + idle-flush plan v1 (docs only, no Rust)

**Documentation-only.** No crate, type, feature or test changed; nothing in `Cargo.toml` touched.

- **NEW knowledge doc `.claude/knowledge/s3-hydration-lifecycle.md`** (`READ BY:` header + per-claim evidence table, every row graded). The three-layer split — **object store hydrates / local directory IS the store / persistent volume only decides whether hydration repeats**. Lance opens a network-scheme URI natively and *that is the trap*: the wrong architecture runs and only degrades, while **deleting the mmap layer** (a remote read lands in a fresh buffer — one copy per read, no page cache), so every zero-copy guarantee under it becomes a claim about copied bytes (`zero-copy-lens-law.md`, one layer down). **Any** local directory satisfies zero-copy — a volume is an optimization on hydration *frequency*, never a correctness requirement. Carries the feature-gate diagnosis (**manifest-verified in session**: `lancedb` `default = []`; `aws` forwards to `lance/aws` + `lance-io/aws` (+ `object_store/aws`); `lance-io` carries `aws` in its OWN defaults, so the layer that opts out is `lancedb` — the reason the diagnosis goes wrong is that the mental model is correct about the wrong crate), the mechanical rule (**scheme-named error = BUILD problem; credential/host/region-named error = CONFIG problem**), the four-state lifecycle (absent/hydrated/dirty/flushed) with **flush legal only from `hydrated`** (the `dirty → flushed` edge is data loss with no error), and one reported single-observation endpoint measurement set graded as ratios-generalize / absolutes-do-not: **NOT viable as swap or as a page-fault backing store; VIABLE for hydration and build caches.**
- **NEW plan `.claude/plans/idle-flush-dataset-eviction-v1.md`** — **PROPOSAL, nothing implemented, nothing measured.** Feature-gated (off by default) idle-flush eviction: a dataset idle past a floor has its local copy dropped — **and is SKIPPED, never pushed back, if dirty** (plan §9a: the sweep does clean eviction only; push-back is a separately-triggered operation, because a background sweep that pushed first would be manufacturing the precondition for its own destructive step, unattended). The first draft of this line said "pushed back first if dirty" and contradicted the plan; corrected in the PR #901 review round. Rehydrates on next access. **Purpose is COST SMOOTHING, not capacity** (operator framing — the win is the shape of the bill; local disk bills continuously for capacity provisioned, object storage for what is kept). **Operator-set default policy:** age **> 3 days** AND footprint **> ~300 MB**, pressure-driven and age-ordered — under budget nothing is ever evicted however stale; over budget the stalest go first. **~300 MB is a SOFT spot**: no operation may ever fail to hold the number, an in-use dataset larger than the whole budget stays resident (**correctness beats the watermark**), and a sweep that reaches no target is a legitimate steady state — which forces the observability requirement that *"no candidate old enough"*, *"every candidate in use"* and *"every candidate dirty"* be distinguishable (the third reason added with §9a). Dirty detection = the **Lance dataset version**, never a hash, with an explicit **unclosed verification gate** (a cheap local version read is *assumed*, not checked — a BLOCKER if it fails). **A lease/refcount/guard protocol was CONSIDERED AND REJECTED** as disproportionate at a 3-day floor (operator scope correction): cheap check-then-act, and the bar is **"does not corrupt"** (worst case a wasted rehydration) rather than **"cannot occur"** — recorded rather than left silent so it is not re-added, with the revisit condition named (threshold dropping from days to hours). Acceptance criteria written as **fire/silence pairs** per the P0 rule, including the conjunction-splitting silence tests (under-budget-but-stale, over-budget-but-fresh) that a staleness-only policy would fail.
- **`docs/DATAFUSION-PERIMETER.md` §9a (NEW section)** — cross-reference: the object-store provider is the *same class of fact* that document already catalogues, one crate over (a capability behind a default-off feature, diagnosed at the wrong layer).
- **Review round (PR #901), all corrections additive.** 19 review comments across this PR and its sibling; several were one finding reached from different angles. **Accepted + fixed:** (1) *"safe to repeat, it is idempotent"* did not support the safety claim resting on it — a Lance dataset is a multi-file **directory**, so the fix is *hydrate aside / publish by rename / retire by rename* (knowledge §4a, plan §5a), a **filesystem-atomicity boundary that costs the reader nothing** and therefore leaves the operator's rejection of a lease protocol intact; (2) the doc fell into its own §3 trap — this repo opens datasets through **`lance`** (direct, non-optional, default features, `aws` ON), not `lancedb` (optional, `default-features = false`), corrected as §3a with a probe record; (3) §6's categorical rule silently condemned shipped `VersionedGraph::{s3,azure,gcs}` — scoped in §6a to the hot zero-copy substrate, gap recorded as `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE`; (4) *"any local path"* → **mmap-capable local filesystem**; (5) boot-viability is a **size** claim (~1 GiB ≈ 49 s at the observed rate), scoped to the measured tens-of-MB case; (6) the cost model priced only retained bytes — request/retrieval/transfer/storage-management named, with the storage-class assumption stated; (7) the thrash metric was unusable in three ways (unattributed numerator, unbounded window, threshold that could not fire at single-dataset granularity) — redefined as `eviction_caused_rehydrations` over an age-floor-bounded window at `> 0`, plus **T11b** asserting the attribution itself; (8) the plan and its board summaries disagreed on dirty candidates — **decided: clean eviction only, the sweep never initiates push-back** (§9a), with *"every candidate dirty"* as a third distinct stop reason and **T6b** asserting both halves. New acceptance tests: T6b, T9b, T11b; T9 sharpened to three enumerated interleavings. **Not accepted:** the sibling PR's carving-sanction request — an open operator question already recorded, not an oversight.
- Board: `.claude/board/EPIPHANIES.md` PREPEND ×2 (`E-OBJECT-STORE-HYDRATES-IT-DOES-NOT-STORE-1`, `E-IDLE-FLUSH-IS-COST-SMOOTHING-NOT-CAPACITY-AND-THE-3-DAY-FLOOR-PRICES-OUT-A-LEASE-PROTOCOL-1`); `.claude/board/INTEGRATION_PLANS.md` PREPEND ×1.

## 2026-08-05 — lance 9 / lancedb 0.33 / DataFusion 54 / Rust 1.97.1 — the ecosystem bump, MEASURED then LANDED across 9 repos

**Current pins:** `lance =9.0.0`, `lancedb =0.33.0`, `datafusion 54`, `arrow 58` (unmoved), `object_store 0.13.2` (unmoved), toolchain **1.97.1**. Plan: `.claude/plans/lance9-datafusion54-upgrade-probe-v1.md`.

Started as a what-breaks probe, became the bump on the operator's "bump all now, fix after". Method that made it legible: the toolchain was tested ALONE on the OLD pins first, so a toolchain failure could never be confused with a dependency failure — it came back clean, making every later red attributable to the deps. **The entire break surface was one uniform DataFusion change** — `Any` moved from a method to a supertrait on seven traits — costing 12 `as_any` deletions and no call-site changes.

**Two operator rulings landed:** `E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1` (lance family from crates.io upstream, NEVER a fork; the `AdaWorldAPI/lance` + `/lancedb` repos exist but are deliberately not depended on — CLAUDE.md's P0 carve-out now says so, since it had named them as must-fork) and, from the Stage-0 funnel probe, `E-THE-LEGEND-IS-NOT-THE-GRAMMAR-1`.

**Contract-inventory deltas:** none — the bump is pins + lint fixes. `lance-graph-ontology`'s 12 long-standing lints are cleared, which is what made `rust-toolchain.toml`'s own "bump when clippy is clean" precondition satisfiable.

**Standing gates (unchanged):** D-BLW-5; D-HWV-1/EXP-HOT-WINDOW; PROBE-ORACLE-FUNNEL Stage 1 (rig LLM arm) and Stage 2 (NARS-34 Gadamer bag); `--features cycle-driver` CI arming. **Owed:** OGAR's 22 unswept crates; `blockly-rs`/`rig` still 1.95.0.

## 2026-08-05 — PR #894 (MERGED `d7a6efc`) + OGAR #243/#244/#245 — the wishlist round-trip closes and the funnel is MEASURED

One same-day arc across the repo boundary, crate-dependency-free by ruling: **#243** (OGAR) delivered the consumer wishlist handover (F-1 double-sampling finding, W-1..W-5); **#244** (OGAR, loco session) shipped W-1 (compose-then-check), W-2 (`FnSpec.name` — OQ-1 answered "in the spec"), W-4 (`telemetry::FunnelTally`), W-5 (split-contract doc), leaving only the W-3 NARS-34 mint operator-gated; **#245** (OGAR) + **#894** (here) ran PROBE-ORACLE-FUNNEL Stage 0 over that delivery — pre-registered E1–E4 all met (floor 0.5% / legend-constrained 2.5% / stack-aware 100%; E4 discrimination KILL passes at 99.5 points vs the 50 bar). Headline finding: `E-THE-LEGEND-IS-NOT-THE-GRAMMAR-1` — the legend-knowing arm landed at the FLOOR; ~all funnel selectivity is the stack discipline, so the gated Stage-1 LLM arm's prompt must teach the discipline, not just serialize the ~382-token legend. Plan: `.claude/plans/oracle-funnel-probe-v1.md`.

**No contract-inventory delta** (docs/board here; the harness is an OGAR-side example). **Standing gates after this arc:** PROBE-ORACLE-FUNNEL Stage 1 (operator word + API), Stage 2/PROBE-GADAMER-BAG (W-3 mint), plus the unchanged D-BLW-5, D-HWV-1, and cycle-driver CI gates below.

## 2026-08-05 — branch `claude/x265-x266-plans-review-h9osnl` (PRs #892 MERGED `e9a2faf` + #891 MERGED `0c138b7`) — the four-arc session lands: whole-book fix, ignition probes, five-axis measurement, no-pump + ack-theater deletion

The oversized branch (68 commits, +24,956/−2,305) was split for reviewability at the operator's direction: **#892** = arc 1 (bible_wave Malachi-truncation fix + `deepnsm_v2::corpus` + the stance lift into `nars::stance` + D-BLW-1..4 with the tenant-not-shard retractions); **#891** (rebased onto main post-#892, pure replay, 29 commits) = arcs 2–4 (PROBE-IGNITION + D-IGN-B + PROBE-IGNITION-64K green; measure-64k-axes v1–v3 with Stage A0 + M-arm/O-arm both measured NEGATIVE as pre-registered findings; seal-vs-temporal dual-authority doctrine; hot-version-window v4 design banked NOT built; `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` + the ack theater deleted with `PhaseCensus` as the message-free visibility surface).

**Contract-inventory deltas:** `lance_graph_supervisor::kanban_actor` is now visibility-only — `PhaseCensus` (new), `mul_target`, `parse_kanban_step`; `KanbanActor`/`KanbanMsg`/`KanbanRouteError` + 5 RPC drivers DELETED. `lance_graph_planner::nars::stance` (new module, lifted from the probe example). `deepnsm_v2::corpus` (new module). onebrc-probe `lane-e` feature no longer depends on supervisor/ractor.

**Standing gates:** D-BLW-5 build operator-paused; D-HWV-1 hot-window build gated on operator word; `--features cycle-driver` CI step still unarmed (operator-approved changes only); `ISS-MARM-T1-4X-A0-GAP` open.

## 2026-08-04 — branch `claude/x265-x266-plans-review-h9osnl` (PR #887) — D-KIA-C1b SHIPPED: the additive `jc` statistics battery

`crates/jc/src/stats.rs` — **κ** (`cohen_kappa`, the gap that blocked **D3's fusion falsifier**, now closed), **ω** (`omega_total`), **φ** (delegating to `pearson`, `&[bool]` input), **R / R²** (`multiple_r_squared`, OLS with intercept), **η²** (`eta_squared`), and the significance companions (`t_test_one_sample` / `_paired` / `_welch` / `_student`, `anova_one_way`) over one shared regularised-incomplete-beta core. **107 lib + 11 doctests green, clippy-clean.**

**The effect-size family is r** (φ, R, R², η²); **Cohen's d is out by construction** — the t-tests report t/df/p as η²/R²'s significance companion, not as a d-family route.

**Additive constraint held tighter than permitted:** the entire existing-file diff is `fn` → `pub(crate) fn` on `mean` and `all_finite`, plus doc comments and one `pub mod` line. `average_ranks` / `pop_var` were NOT widened (permitted but unused); the new `sample_var` / `sample_cov` use the unbiased `n−1` divisor — a different estimator from `pop_var`'s `n`, not a duplicate.

**Validation is by cross-identity, not self-assertion:** φ vs `pearson`; R² vs `pearson²`; η² vs R² on a dummy; η² vs `t²/(t²+df)` and `F = t²`; ω vs α (equal under tau-equivalence, ω = 0.9473684 > α = 0.8684211 on the hand-computed congeneric fixture).

**Two defects, both surfaced by doc examples** (`E-EXACT-FIT-IS-WHERE-ABSOLUTE-ZERO-GUARDS-BREAK-1`): a Heywood guard written `psi < 0.0` rejected the *perfect* zero-residual model (fixed with a variance-relative tolerance + a can-it-fire test); and a doc example whose two predictors were collinear with the intercept (the example was wrong, the code right).

## 2026-08-04 — branch `claude/x265-x266-plans-review-h9osnl` (PR #884 MERGED `1e90cef`) — D-KIA-C1b scoped: the r-family, additive-only

Board/plan prose only; **no code, no runtime behaviour**. Carried the post-merge arc entries for #881/#882/#883 (see the entry below) and re-scoped the statistics work removed from #883 as its own deliverable, **D-KIA-C1b** (`Queued` — scope, not code).

**C1 audit (read-only).** `jc` is in-tree at `crates/jc/`; `reliability.rs` ships `pearson` / `spearman` / `cronbach_alpha` / `icc(ratings, IccForm)` (`Icc2_1`, `Icc3_1`), and `jirak.rs` gives C4's noise floors a local implementation to cite. Two of C2's three renames are the **same computation**, one is a real gap:

- **φ = Pearson r on two binary variables** → `jc::reliability::pearson` already computes it. Only a *named wrapper* + the marginal-capped-ceiling caveat are new; the arithmetic is not re-implemented.
- **KR-20 = Cronbach's α on dichotomous items** → `cronbach_alpha` is the right function; naming + caveat only.
- **κ is absent entirely and is NOT a renamed ICC** — a different estimator. **This is the gap, and it blocks D3's fusion falsifier.**

**Effect size means the r-family (operator ruling):** R, R², **η²** (*erklärte Varianz*), φ. **Cohen's d is explicitly OUT** — calculated separately if a mean-difference contrast is ever wanted. The **t-test** (t/df/p) is in scope as the *significance* companion to η², not a d-family route: effect size is read off η²/R². This supersedes the vague "Effektstärke / effect size" wording the PR opened with.

**ADDITIVE ONLY, with exactly one carve-out.** `pearson` / `spearman` / `cronbach_alpha` / `icc` keep their **arithmetic, signature and semantics**; new estimators land in a new module beside them, and any diff changing an existing `jc` statistic is an automatic reject. **The one sanctioned edit to an existing file is visibility only:** widening `reliability.rs`'s private helpers (`mean` / `all_finite` / `average_ranks` / `pop_var`) to `pub(crate)` plus the `pub mod` line, so the new module reuses them rather than growing a second source of truth. No body change, no signature change, no incidental cleanup.

**Process note.** #884 existed because #881/#882/#883 merged without arc entries — then merged without its own. Writing the hygiene PR does not discharge the rule for the hygiene PR itself; its entry is now in the arc.

## 2026-08-04 — branch `claude/x265-x266-plans-review-h9osnl` (PRs #881, #882, #883 MERGED) — arc hygiene restored + the legacy actor surface quarantined

**#881** post-merge arc/state for #880 and recorded that the arc had gone stale for four PRs. **#882** backfilled those four (#862/#875/#876/#879) as marked RECONSTRUCTIONS, plus the cross-PR finding that every review-found defect in three consecutive probe PRs sat in a falsifier or a label, never in a measurement.

**#883 — the operator ruling, canonical text:** *"#879 is the complete and independent production phase-progression path. KanbanActor has no assigned architectural responsibility. It is legacy experimental compatibility code retained only because existing probes or consumers still reference it. No new production architecture may depend on it. Its presence does not designate it as the future home of an ownership, planning-initiation, concurrency, cognition, reasoning, or lifecycle mechanism."*

Production path: `plan evaluation → KanbanMove intent → BatchWriter → sparse seal → one WAL/version → inline apply of the sealed transitions`. No actor bridge, fleet, owned driver, custody model, or message path required.

**Transport ≠ engine (the separation that must not blur):** `KanbanMsg::MulAdvance` and `drive_mul_advance` are legacy actor-message **wrappers only** — NOT the canonical MUL reasoning engine. `lance_graph_contract::mul::i4_eval::gate_decision_i4` is independent, consumed directly by the #879 path through `cycle_driver::shade_owner` and `run_cognitive_work_gated[_over]`, and is **not deprecated**. The NARS tactic recipes and the awareness rung ladder are **separate and untouched**.

`cycle_driver.rs` is canonical #879 code and is NOT stale — only three inherited comments were, now corrected; `run_cognitive_work` is documented as a sequential contract-probe adapter that does not define the production execution model (production cognition may run independently and concurrently over the sealed `Vn`, converging only at the deterministic ordering/coalescing/seal boundary). The obsolete ractor-drives framing is also corrected in `supervisor/lib.rs`, `contract::kanban`, `contract::soa_view`, `contract::orchestration`.

**Withdrawn across the arc, nothing substituted:** the A1 two-seam design gate (both seams), the actor-owned `emit_bootstrap_intent` milestone, the planning-initiation adapter, the future actor/nudge slice, the ownership-injection/guarantee-dummy framing.

**W1 ledger:** SHIPPED — held-owner reschedule/wake. OPEN — `run_cycle` drained-writer retry guard; missing-owner counter in `cognitive_pass`.

## 2026-08-03 — branch `claude/x265-x266-plans-review-h9osnl` (PR #880, MERGED `6bc9115`) — the kanban-64k-inverted-awareness plan + four honest module headers

**Plan landed.** `.claude/plans/kanban-64k-inverted-awareness-v1.md` (W0–W6) over the two operator anchors: (a) real thinking at 64k via kanban orchestration **in parallel**, (b) inverted awareness — ontologies as the frozen cathedral an observation layer reads into, with a statistical witness. Board rows in the same commit (INTEGRATION_PLANS prepend, STATUS_BOARD D-KIA ×7, `write-on-behalf.md` caller-status supersession). **No code paths changed.**

**What is now pinned, before any measurement:**
- **W2's falsifier thresholds are pre-registered** and not adjustable after the run: median of ≥5 runs after one discarded warm-up; can-fire = ≥2× speedup at ≥4,096 owners with ≥100 µs per-thought busy-work; stay-silent = trivial (<1 µs) bodies within ±10 %. Kill condition: failure regrades claim (a) to "64k-scale **sequential** sparse cycles" — still true, different claim.
- **Dichotomous-statistics discipline:** φ not "Pearson", KR-20 not "α", κ-family not "ICC"; ICC only on a non-binary escalation, named at its site. Reliability ≠ validity — the witness is capped at reliability until an external criterion exists.
- **`MailboxFleet`-over-the-registry is WITHDRAWN** as the W1 shape (structurally impossible: the trait's synchronous `owner()`/`owner_mut()` borrows can't reach actor-private state behind `where_is`, `cycle_driver.rs:183-190`). A1 is now a two-seam **design gate** — guarantee-dummy single owner vs per-mailbox `KanbanMsg::Advance` apply; both keep one-writer-per-mailbox with no ack state. W1 chooses.
- **The HashMap fleet is deliberate** (operator ruling this session): an order-free keyed store; ordering is recovered by `temporal.rs` HLC deinterlace at READ time. Parallelism lives in the thought phase; apply stays order-free keyed writes.

**Four module headers now report state, not intent** (a `head -5`/grep tells the truth): `actors/medcare_actor.rs` = UNWIRED STUB, never spawned (`supervisor::StubConsumerActor` is what the tree spawns, supervisor.rs:368), owns no bridge, emits no audit — and is publicly re-exported, so removal is breaking; `actors/mod.rs` = no concrete actor ships there; `soa_bake/mod.rs` = only the label-codebook half is implemented, the rest is type scaffolding; `bridges/medcare_bridge.rs` = the `#[deprecated]` migration pointer is now above the fold.

**Open, named:** the `ConsumerActor<P: PortSpec>` generalization (would re-derive the stub actor as a one-line alias — the actor-half of the bridge collapse onto `UnifiedBridge<P>`; #570 did that for OpenProject/Redmine and explicitly deferred Healthcare until the codebook promotion, which landed in a later, unnamed PR); the ownership question for `soa_bake`'s scaffolded half now that OGAR ships complete bakes emitting `NodeRow` bytes; A3 `LanceShardSink` still deferred behind its own crash falsifiers.

**Process, recorded not buried:** an operator-ruled public/private separation-of-concerns violation occurred mid-arc and was remediated the same session (files scrubbed, three unmerged commits rewritten into one and force-pushed with lease — PR branch only, `main` untouched — PR body rewritten, bot comments patched). Residue stated honestly at the time: orphaned SHAs may persist until GC, edit histories remain in the UI, and pre-existing occurrences on merged `main` were surfaced for an operator decision rather than rewritten unilaterally. **Also found:** the PR arc's memory practice has broken again — **no `PR_ARC_INVENTORY` entry exists for #862, #875, #876, or #879**. The gap is now recorded at the top of that file; reconstruction is queued, not done.
## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` (restarted on main post-#879-merge) — pre-commit failure contract corrected to deterministic regeneration + proof re-homed to the live consumer

- **#879 MERGED** (2026-08-02T12:25Z); branch restarted from `origin/main` per protocol, unmerged docs commit rebased on top. Two operator-ruled realignments applied (grain-of-salt round 2):
- **1) Pre-commit failure contract corrected** in `cycle_driver.rs` docs + tests. The authoritative rule: **sealed `Vn` + unchanged Kanban task + deterministic computation = the same provisional intent on the next sweep** — commit fails before `Vn+1` → publish nothing · mutate no owner · advance no watermark · discard provisional slots/held moves/planning results · rerun from `Vn`. `SealFailure{casts}` RECLASSIFIED from "the retry-safety mechanism" to **optional retry cache / implementation convenience** (never a provisional-planning ledger; dropping it is always sound). `recover_fleet` doc-pinned as **committed-history recovery ONLY** (`Vn+1` exists, application interrupted) — the two mechanisms share no state. `HeldIntent` doc-pinned as within-success scheduling convenience, discarded on failure. NEW authoritative falsifier `pre_commit_failure_discards_everything_and_regenerates_from_vn` (derives cycle C from Vn → injects failure → DROPS the SealFailure → asserts no version/phase/watermark change → reruns the unchanged deterministic task → asserts the SAME semantic sparse cycle regenerates → one `Vn+1`, owners advance once; object identity deliberately NOT asserted). The old byte-identical-retry test demoted to "optional-cache probe". **20 cycle_driver tests green**, clippy+fmt clean. Latency figures in review prose = operator-provided measurements, not workspace-reproduced.
- **2) Proof re-homed:** `medcare-consumer-pull-thinking-proof-v1.md` §4 now pins the proof's primary home to **MedCare-rs** (the live composition root) — lance-graph contributes only genuinely-missing GENERIC seams, never a MedCare-shaped host adapter (would reconstruct the dead lineage in miniature). Hard driver requirement added: the proof must invoke the existing **cognitive-shader-driver + MailboxSoA operational unit** — SoA-read qualia fed straight into `shade_owner` proves only the (already-proven) MUL gate; `shade_owner` participates only as the driver's existing downstream gate. Trace-and-report obligation added (every arrow file:symbol-named). F1 strengthened to require a discriminating DRIVER outcome, not gate-only.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — MedCare two-lineage investigation CONSOLIDATED → one active plan (consumer-pull thinking proof)

- Six-agent read-only investigation (code inventory · callcenter/policy · manifest→runtime chain · plans archaeology · git history · tests+build) + main-thread cross-repo verification (OGAR + MedCare-rs siblings) + `git fetch --unshallow` (5 grafted roots → 4162 commits). Full verdict in `EPIPHANIES.md` `E-TWO-MEDCARE-LINEAGES-THE-LIVE-ONE-PULLS-THE-DEAD-ONE-HOSTS-1`; classification + proof target in **`.claude/plans/medcare-consumer-pull-thinking-proof-v1.md`** (ACTIVE — the single active MedCare plan; INTEGRATION_PLANS prepended).
- Headline: **bridge migration COMPLETED** (`ddb6c840` 2026-06-21, `MedcareBridge` = deprecated alias over `UnifiedBridge<HealthcarePort>`; codebook mirror verified in sync with OGAR); **host-side manifest/supervisor/actor lineage DORMANT since birth 2026-05-13** (manifest runtime-orphaned; `MedcareConsumerActor` never constructed; `StubConsumerActor` hard-coded; Dispatch rejected pre-child; `MedCareActor`/`MedCareMessage` in neither repo; `medcare_policy` nonexistent). Two same-named, uncomposable `UnifiedBridge` types = later hardening. **Active target: one real medical thought over the live consumer-pull path into the #879 cycle loop** (F1–F4 falsifiers, incl. the ABSENT Healthcare fail-closed unknown-actor test). P0 candidates surfaced: NoopAuditSink-by-default for a HIPAA-regime domain; fail-open discarded `MEDCARE_AUDIT_SALT`. Open decisions carried (§6): OQ-2 retention 2190/3650, `Ueberweisung`/`Anamnese` canon gap, dead-lineage retire/revive, `.grok/` lineage.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — PR #879 review round: recovery/data-integrity holes fixed + scope honesty (grain-of-salt audit)

Operator-forwarded review (grain of salt); each finding verified against code before acting. **Accepted + fixed (all real):**

1. **Retry-safe seal.** `collect_casts` drained the writer, `seal_cycle` consumed the casts — a WAL failure LOST the cycle (retry saw an empty writer). Now `seal_cycle → Result<SealedCycle, Box<SealFailure>>` where `SealFailure{frame, casts, cause}` carries the complete frozen input back byte-identical; falsifier proves failed-commit → zero owner mutation → same-cycle retry → exactly one version, no cast lost/duplicated.
2. **Restart-stable stream positions.** `stream_position = cast.0` walked into the P3d-documented trap ("cast_id is provenance only" — a reconstructed `BatchWriter` restarts at 0, and `recover_and_apply` skips positions ≤ watermark → later cycles silently unrecoverable). Now `collect_casts(writer, cycle, position_base, row_of)` — `position_base` is the caller's DURABLE cursor; `SealedCycle.next_position_base` (computed over ALL slots incl. no-move landings) carries it forward. Restart falsifier pins the exact failure the raw-CastId scheme would have caused.
3. **Normal apply advances the recovery watermark.** `apply_sealed_transitions(fleet, sealed, &mut watermarks)` now moves phase + per-owner watermark TOGETHER (one owner state, the same rule recovery uses). Falsifier: normal apply → crash → `recover_fleet` with the same map replays NOTHING (previously: replay → permanent `StalePhase` stall; our own P4e negative control had proven the stall and we shipped the gap anyway).
4. **≤1-move/owner enforced PRE-seal.** The old `deferred += 1` sealed a durable move and then never applied it (and recovery WOULD apply it — divergent semantics). Now `collect_casts` partitions: first move per owner seals; every extra move (same cast or later cast — also killing the silent `moves.first()` truncation) returns as `HeldIntent`, re-staged via `restage_held` into a future cycle. Sealed set == applied set; recovery-agrees falsifier. `AppliedCycle.deferred` demoted to defence-in-depth for foreign sealed inputs.
5. **Mid-apply prefix preserved.** `apply_sealed_transitions` → `Err((partial, cause))` (mirrors `recover_and_apply`): the applied prefix + its watermarks survive a guard trip.
6. **Hold = reschedule, never strand.** `CognitiveWorkOutcome{cast, held_owners}`; held owners re-polled via `run_cognitive_work[_gated]_over`. Falsifier: a Held owner is woken on a later re-poll and advances.
7. **`recover_fleet` partitions history once** — O(history + Σtails), not O(fleet×history).
8. **P4d honesty**: wait-free at the cast/cycle boundary (sequential pass); strengthened falsifier: TWO represented owners, A unfinished, B casts + advances without waiting. Concurrent per-owner execution = the actor leg, explicitly out of driver scope.

**Declined (with reasons):** routing P4b through `KanbanActor` mailboxes — contradicts the operator-ratified writer-fires-inline sparse ruling (no message bus for P4a/P4b; the plan's explicit shape). The honesty half IS taken: module docs now state `MailboxFleet`+HashMap is the probe/registry fleet, NOT production supervisor ownership; actor-state bridging is open. **Scope honesty ledger (module doc + STATUS_BOARD):** control-loop contract PROVEN · actor-owned production wiring NOT proven · cognitive-shader-driver/MailboxSoA thought NOT proven (the MUL gate is real; its qualia inputs are extractor-fed — "shader plug" wording retired in favor of "MUL-gate plug") · durability FAKE. **19 lib tests green** (was 14; +5 net new falsifiers), clippy+fmt clean.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P4c shader plug: the REAL MUL cognitive gate wired into the CognitiveWork seam

- `lance-graph-supervisor::cycle_driver` gained the **shader plug** — the P4c thought body is no longer just a pluggable seam, it is the REAL MUL cognitive gate (mints NO decision logic; it composes `kanban_actor::mul_target` for the driver from two already-shipped contract primitives):
  - **`shade_owner(owner, qualia, mantissa, reliability) -> Option<StrategyOutcome>`** — reads the owner's current phase → `contract::mul::i4_eval::gate_decision_i4(qualia, mantissa)` (the i4 TrustTexture × FlowState gate: Flow / Hold / Block) → lowers via `KanbanColumn::advance_on_gate` (Flow → forward, Block → Prune-where-legal, Hold → rest / `None`). Packages the result as a **bootstrap-sentinel** `StrategyOutcome` (`mailbox 0`, `witness_chain_position 0`) so `owner_adapter::emit_bootstrap_intent` rebinds it to the live owner and casts write-on-behalf — **no mailbox mutated** (the step is P4b, next cycle).
  - **`run_cognitive_work_gated(fleet, applied, writer, read_gate)`** — the shader-wired form of `run_cognitive_work`: for each owner entering `CognitiveWork`, `read_gate(&Owner) -> Option<(QualiaI4_16D, i8, f32, payload)>` supplies the gate inputs and the gate decides the next move. Delegates to `run_cognitive_work` (single routing path).
  - **The qualia seam (no MailboxSoa redesign):** `MailboxSoaView` does NOT yet expose `qualia()` (deferred — `soa_view.rs` "add `fn qualia` when the first consumer arrives"). P4c is that first consumer; the caller-supplied extractor bridges the seam so the **MailboxSoa contract stays UNCHANGED** (operator constraint "do not redesign the MailboxSoa" respected — the trait method lands later without touching this code).
- **14 cycle_driver lib tests green** (was 11; +3): `shade_owner_flow_advances_forward_block_prunes_hold_rests` (three distinct outputs Flow→Evaluation / Block→Prune / Hold→None — the gate discriminates, per the anti-eigenvalue falsifiability rule), `shade_owner_at_absorbing_column_yields_nothing` (Flow/Block at Commit → `None`, DAG respected), and `run_cognitive_work_gated_flow_casts_next_intent_hold_casts_nothing` (a Flow-qualia owner casts + round-trips to Evaluation in the next cycle while a Hold-qualia owner rests at CognitiveWork — one WAL write per cycle, sparse). clippy (feature) clean on `cycle_driver.rs`, fmt clean; default build unchanged.
- **The MedCare first-thought loop now runs the real gate on the control side:** seal→step→**think (real MUL gate)**→cast→recover, all against the contract-probe sink. The ONE remaining fake is the durability leg (`FakeWalSink`); the concrete `LanceShardSink` is still deferred (gated on crash falsifiers). STATUS_BOARD D-MBX-A6-P4 updated in place. Plan `.claude/plans/cycle-loop-closure-driver-v1.md`.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P4c..P4f: cycle loop-closure driver COMPLETE (slice) — the full seal→step→think→cast→recover loop

- `lance-graph-supervisor::cycle_driver` extended P4a/P4b → **P4a–P4f** (feature `cycle-driver`; still mints NO domain types — reuses `StrategyOutcome`, `owner_adapter::emit_bootstrap_intent`, `recover_and_apply`, `LandedSlot`, `MailboxSoaOwner`):
  - **P4c** — `run_cognitive_work(fleet, applied, writer, think)`: for each owner that just entered `CognitiveWork` (an applied move with `to == CognitiveWork`), run a **pluggable thought seam** `think(&Owner) -> Option<(StrategyOutcome, payload)>` (NOT the shader — designed elsewhere) and route its Outcome into the NEXT cycle's casts via `owner_adapter::emit_bootstrap_intent` (bootstrap-sentinel rebind → write-on-behalf cast). No mailbox mutation (the step is P4b, post-seal). `MailboxFleet` gained a read accessor `owner()`.
  - **P4d** — wait-free: a completed owner casts + advances without any synchronous neighbour wait; an incomplete ("mid-thought") owner never blocks a completed one. Structural (fire-and-forget cast, no per-owner barrier — the cycle boundary is the WAL-amortization barrier, not a neighbour wait); proven by falsifier.
  - **P4e** — `recover_fleet(sink, fleet, ids, watermarks)`: composes `persist_sink::recover_and_apply` per owner over `scan_sealed`, replays only the pending tail above each owner's durable **watermark** (idempotent); keeps the earned watermark on a mid-owner error (`Err((partial, cause))` contract). `FleetRecovery{total_applied, owners_recovered}`.
  - **P4f** — sparse-routing scale probe: `CountingFleet` proves apply cost is **O(dirty), not O(fleet)** — 640 owner-resolutions over a 65 536-owner fleet (1% dirty), the sparse-cycle guarantee made measurable (`perf.p4f` log line).
- **11 cycle_driver lib tests green** (`cargo test -p lance-graph-supervisor --features cycle-driver`): the 64k/17 headline + P4c round-trip (a CognitiveWork Outcome cast in cycle N advances the owner one further legal step in N+1) + P4d wait-free + P4e idempotence with a **load-bearing-watermark negative control** (watermark lost → acyclic re-drive StalePhase-stalls) + P4f O(dirty). clippy (feature) clean on `cycle_driver.rs`, fmt clean; default (no-feature) supervisor build unchanged. Durability leg still the contract-probe fake (`FakeWalSink`, now storing landings for P4e) — control loop closed, storage NOT proven; concrete `LanceShardSink` deferred.
- **The MedCare first-thought loop is now code-complete on the control side:** seal→step→think→cast→recover all wired against the contract-probe sink. The only remaining pieces before a real first thought are (a) a concrete `LanceShardSink` (durability, gated on crash falsifiers) and (b) a real CognitiveWork thought body plugged into the P4c seam (the shader/StyleStrategy — exists; wiring is a consumer concern). STATUS_BOARD D-MBX-A6-P4 → P4a–P4f Shipped (slice). Plan `.claude/plans/cycle-loop-closure-driver-v1.md`.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P4a+P4b: cycle loop-closure driver (persist_sink's first production caller)

- `lance-graph-supervisor::cycle_driver` (NEW module, behind feature **`cycle-driver`** = optional one-way `lance-graph-planner` dep; default supervisor build stays light, no planner/ractor) — **the seam that makes the merged #878 `persist_sink` load-bearing** (it was the ZERO-caller gap). Supervisor is the runtime fleet owner; planner decides; the dep is one-way (planner never deps supervisor — verified acyclic). Mints **NO** domain types — composes `SweepSlot`/`CycleFrame`/`CycleId`/`persist_cycle`/`WalSink`/`PersistError` (planner) + `BatchWriter` (planner) + `MailboxSoaOwner`/`KanbanMove`/`DatasetVersion`/`MailboxId` (contract).
  - **P4a** — `collect_casts(writer, cycle, row_of) -> Vec<SweepSlot>` drains a `BatchWriter<Vec<u8>>`'s staged casts (one slot/cast; `stream_position`=`CastId`; `paired_move`=first intended move); `seal_cycle(sink, frame, casts) -> SealedCycle{version, transitions}` reads out the **sparse** `SealedTransition` set (only slots carrying a move, stream-ordered) then `persist_cycle` → **exactly one WAL write, one `DatasetVersion`**.
  - **P4b** — `apply_sealed_transitions(fleet, &SealedCycle) -> AppliedCycle{version, applied, deferred, missing}`: iterate **ONLY the sealed sparse transitions**, resolve each owner via the `MailboxFleet` trait (blanket-impl'd for `HashMap<MailboxId, O>`), apply **one** legal `try_advance_phase`; **every unrepresented owner is byte-identical** (never resolved). Interim **≤1 durable transition/owner/cycle** (second same-owner move → `deferred`); unknown owner → `missing` (counted, not a crash); `StalePhase`/`OwnerMismatch` corruption guards; **reads NO dataset** (version already sealed by P4a — no `scan_sealed`/`versions`/`drive_once`). `run_cycle(...)` = P4a→P4b convenience.
- **The load-bearing rule enforced in code:** a `DatasetVersion` is global knowledge, NOT permission to advance every mailbox — only the sealed sparse set advances (`E-D-MBX-SPINE-...-1` + `E-COMPLETE-CYCLE-IS-PHYSICALLY-SPARSE-...-1`).
- **7 lib tests green** (`cargo test -p lance-graph-supervisor --features cycle-driver`), headline = **64k/17 falsifier**: 65 536 mailboxes, 17 sealed transitions → exactly 17 advance (→CognitiveWork, cycle bumped), the other 65 519 byte-identical, one WAL write, **zero dataset reads**. Plus: one-WAL-write amortization, empty-sparse-set advances nobody, interim-defer, StalePhase corruption, missing-owner-counted, run_cycle round-trip. clippy (feature) exit 0, fmt clean; default build unchanged. **Durability leg still the contract-probe fake** (`FakeWalSink`) — control loop closed, storage NOT proven (Ladybug rule); concrete `LanceShardSink` still deferred. Plan `.claude/plans/cycle-loop-closure-driver-v1.md`; STATUS_BOARD D-MBX-A6-P4 → P4a+P4b Shipped(slice). Remaining: P4c (CognitiveWork thought body + cast round-trip through owner_adapter), P4d/P4e/P4f.

## 2026-08-02 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P3e persistence-sink reshaped to WAL-amortized cycle (one write per sweep)

- `lance_graph_planner::persist_sink` — **reshaped in place** (PR #878) from the per-cast durable-witness model to the **cycle/WAL** model (operator ruling on the seam-shape fork surfaced 2026-08-01). The durable unit is the **cycle/sweep**, NOT the cast: 64k concurrent thoughts stage into an owned `Vec<SweepSlot>`, `persist_cycle(sink, frame, casts)` folds+freezes them, and `WalSink::commit_cycle(base, DetachedCycleBatch)` does **exactly one atomic append → one `DatasetVersion`** (WAL amortization — 64k thoughts, one write). Types: `CycleId`, `CycleFrame{cycle, base_version}` (storage-only — NO rung/branch/semantic tags), `SweepSlot{cycle, stream_position, owner, row, paired_move, payload}` (boring landing, no `basis`), `LandedSlot{version, slot}`, `DetachedCycleBatch{frame, landings, image}` (frozen, deinterlaced, coalesced). **Write-side ordering** = `order_cycle_stably<T, K: Ord>(rows, key)` (generic over the caller's canonical key; stable-orders casts by the EXISTING `stream_position` in `freeze` BEFORE the append) — completion order never becomes storage order (physical race); `scan_sealed` returns stored order and NEVER sorts. **Epistemic horizon** = sealed read (`read Vn / write Vn+1`); an uncommitted cycle is invisible to `scan_sealed`. **Coalescing** = real per-row fold (`row -> last payload in stream order`), not last-chunk-wins. `recover_and_apply(owner, sealed, applied_through)` + the durable **watermark** idempotence + `StalePhase`/`OwnerMismatch` guards survive unchanged. `versions()` = the cheap coarse timeline a downstream time-series consumer (stockfish-rs, another session) looks up, no landing replay.
- **Write-side ordering lives HERE, not in `temporal.rs`** — that module owns query-time reader-horizon/version reading; only the write-side `deinterlace_write` additions made there earlier this session were **reverted**. `temporal.rs`'s existing query-time causal helpers (`LocalCausalRow`, `local_trajectories`/`local_trajectory_of`, layer-1 causal deinterlace) landed in the PR's earlier commits and **remain** — the reshape commit itself leaves `temporal.rs` untouched (this commit's `git diff` on it is empty), NOT the whole PR. The rung a cycle aligns at is decided by whoever schedules it, never modelled in `CycleFrame`.
- **Retired:** `DurableWitness`, `DurableReceipt`, `DurableCoordinate`, `DurableWrite`, `persist_cast`, `apply_durable_step`, `scan_witnesses`, `LandedWitness` (the per-cast surface). Vocabulary reused, not re-minted: `DatasetVersion`, `KanbanMove`/`KanbanColumn`, `MailboxId`/`MailboxSoaOwner`, `stream_position`. **No concrete Lance sink built** (deferred per operator, gated on crash falsifiers). Tests are storage/race CONTRACT probes over an in-process `FakeWalSink`, honestly labelled — real MemWAL/restart/atomic-append durability UNPROVEN. See `EPIPHANIES.md` E-THE-DURABLE-UNIT-IS-THE-CYCLE-NOT-THE-CAST-ONE-WAL-WRITE-PER-SWEEP-1.

## 2026-08-01 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P3d persistence-sink durable-witness reshape + temporal layer-1

### Current Contract Inventory — new/changed modules (lance-graph-planner)
- `lance_graph_planner::persist_sink` — the POST-write half, two clock domains, **crash-durable**. `DurableWitness{owner, cast_id, cycle, paired_move}` is CO-LOCATED with the SoA payload in one persistence generation via `DurableWrite::append(&witness, &payload)`; the in-memory `DurableReceipt` merely REFERENCES it (via `DurableCoordinate`), never the only copy of the move. **Replay order = the durable `DurableCoordinate::log_order` (WAL position, monotonic across restarts), NOT `cast_id`** (`BatchWriter`'s counter resets on restart → cross-lifetime collisions; `cast_id` is provenance only). `DurableWrite::scan_witnesses(from)` returns `LandedWitness{coordinate, witness}` (bounded tail read; `LandedWitness` is the `LocalCausalRow` implementor). `recover_and_apply(owner, landed, applied_through)` = crash recovery: temporal layer-1 → per-owner tail in durable order; **idempotence is a durable WATERMARK** (`applied_through`), NOT phase equality (unsound on the cyclic lifecycle — a completed lap returns to `Planning`); above the watermark a non-matching `from` is corruption (`StalePhase`); returns `Recovered{applied, watermark}` to persist with the SoA phase. `persist_cast` validates `paired_move.mailbox == owner` (no cross-owner move becomes durable); `apply_durable_step` keeps the sync `from==phase` guard (a stale drop is safe — the durable witness replays). `WriteFailed`/`PersistError` impl `Display`+`Error`. Async `persist_cast` (no owner borrow) / sync `apply_durable_step` (no await) split preserved. Durability proof = `DurableCoordinate`, NEVER `LanceVersion`. **No concrete `LanceShardSink` built** (deferred per operator, gated on the crash falsifiers).
- `lance_graph_planner::temporal` — **layer-1 CAUSAL deinterlacing added** (the missing half): `LocalCausalRow{owner, cast_seq}` + `local_trajectories`/`local_trajectory_of` split a globally-interleaved durable log into per-owner LOCAL chains ordered by `cast_seq` (`A@s0,C@s0,B@s0,A@s1` → A's `[A@s0,A@s1]`; interleaved owners removed). `cast_seq` doc states the monotonic-across-restarts precondition (a resettable counter is NOT valid; use a durable-log position). Composes with the pre-existing layer-2 epistemic projection (`classify`/`deinterlace`). `LandedWitness` implements `LocalCausalRow`.
- Round-2 review (ChatGPT, grain of salt): `apply_durable_step` now **borrows** `&DurableReceipt` so a reordered/stale receipt stays RETRYABLE on the happy path (not dropped-until-crash); `DurableCoordinate.wal_entry_position` renamed to opaque `seq` (API-honest — `ShardWriter::put` returns a batch position, not a WAL offset); the `FakeSink` tests are labelled **CONTRACT probes** (the fake now stores + asserts the payload) — real MemWAL/restart/atomic-co-location durability is UNPROVEN until the concrete sink (`compile+test green ≠ storage proven`).
- **Honest status:** the ordering/recovery CONTRACT is green (349 planner lib tests, clippy + fmt clean); crash-durability is contract-probed over an in-process fake, NOT storage-proven. Deferred per operator: the concrete `LanceShardSink`; and the per-cast-vs-**generation/batch** persistence seam shape (finding 5 — surfaced for operator decision, not a correctness bug). `E-THE-PAIRED-MOVE-MUST-BE-DURABLE-CO-LOCATED-NOT-IN-MEMORY-ONLY-1`.

## 2026-08-01 — branch `claude/medcare-rs-continue-ufsazd` — D-MBX-A6-P3c owner-consume adapter (rebased onto main dcd9cc9)

### Current Contract Inventory — new module (lance-graph-planner)
- `lance_graph_planner::owner_adapter` — the D-MBX-A6 `Outcome → KanbanMove` **owner-consume** adapter; completes the `D-MBX-A6-P3b` deferral (`owner-consume`). Two functions, lance-free:
  - `rebind_bootstrap(mv, owner, owner_cycle) -> Option<KanbanMove>` — rebinds the bootstrap sentinel (`mailbox 0`, `witness_chain_position 0`) to the live owner; returns `None` for a move that already names a live owner (**no ownership theft** — write-on-behalf iron rule) and for a partial sentinel.
  - `emit_bootstrap_intent(outcome, owner, owner_cycle, writer, payload) -> Option<CastId>` — rebinds `StrategyOutcome::intended_move` and `BatchWriter::cast(on_behalf = owner, …)`. **Fire-and-forget** (returns immediately; no ack/ledger/WAL/arbitration/callback); the move is the pre-write "parcel address before dispatch", the lifecycle STEP stays post-write.
- **Causal model pinned** (`E-KANBANMOVE-IS-THE-PARCEL-ADDRESS-STEP-IS-THE-DELIVERY-SCAN-1`): the `KanbanMove` is cast ahead of the write; the KanbanStep (`try_advance_phase`) is applied post-persistence on the successful `LanceVersion` (no successful write ⇒ no step). The version-completion path must apply the **paired** move, never a generic `next_phases().first()`.
- **Persistence sink = verified-but-gated.** The drain→Lance sink wires Lance 7's shipped MemWAL surface (`dataset::mem_wal::WalAppender::append(Vec<RecordBatch>)`, `memtable::BatchStore::append`, `wal::flush`, `merge_insert`) — invents nothing. NOT buildable in the private medcare-session container (`protoc` missing; lance+datafusion+arrow would exhaust disk). Offline/next-env slice; `lance-graph-planner` must add the mandatory stack (lance 7 / lancedb 0.30 / arrow 58 / datafusion 53 + protoc) there. Full handoff: `.claude/v3/knowledge/d-mbx-a6-owner-consume-and-persistence.md`.
- Gates: 5/5 `owner_adapter` probes green, 324 existing planner tests intact, `cargo fmt -p` + `cargo clippy -p lance-graph-planner` clean. `KanbanMove` uses the current 5-field main shape (post `libet_offset_us` retirement).

## 2026-07-29 — branch `claude/x265-x266-plans-review-h9osnl` — `lance_graph::reasoning`, the concept-blind consumer seam

### Current Contract Inventory — new module (lance-graph core, `planner` feature)
- `lance_graph::reasoning` — the curated consumer reasoning facade, **concept-blind by construction** (no domain vocabulary in the public crate, including doc-comments and tests) (`E-MAKE-THE-TRAP-UNREACHABLE-NOT-DOCUMENTED-1`). Re-exports the supported reasoning entry points only: `TruthValue` (all five NAL operators), `BeliefArena`/`Belief`/`CStmt`/`Copula`/`Stamp`/`ReviseOutcome`, the five tactics + `Candidate`/`Frontier`/`ReasoningGap`/`GapKind`/`Throttle`, and `counterfactual::{substitute_binding, multi_substitute_binding, worlds_differ, …}`.
- **New in the facade:** `Axis` (+ `MAX_AXES`) — one independent evidence source; takes an axis index, NOT a `Stamp`, so distinct axes yield disjoint evidence bits **by construction** and the silent stamp-collision failure (pooling degrades to CHOICE, confidence stops rising, nothing logs) cannot be expressed by a consumer. `Axis::new` refuses `index >= 64` rather than letting `Stamp::source`'s `% 64` alias axis 64 onto axis 0.
- `PremiseBundle` (owns stamp assignment) · `Resolution { stmt, truth, contradiction, axes }` · `resolve` · `differential` (returns `Frontier` so `ReasoningGap` — "what premise is MISSING to separate these" — is surfaced, not discarded).
- **`GuardRule`/`GuardViolation`/`detect_violations` — deliberately NOT inference.** No `TruthValue`, no `Belief`: a stored-value contradiction routed through the arena would become revisable and could be *softened* by later evidence. Asserted, not just documented.
- **Wiring:** `planner = ["dep:lance-graph-planner", "dep:lance-graph-cognitive"]` — one feature, one seam (operator ruling: a consumer adds ONE dep, not two). Previously `lance-graph` pulled the planner behind this feature but never `pub use`d it, so `features = ["planner"]` exposed nothing.
- **Consumer contract:** concept ids are opaque `u16` — meaning stays private to the consumer (medcare commitment #9). No `ConceptId` newtype: `CStmt.s`/`.p` are already bare `u16`, so one would add friction without adding blindness.
- Gates: 6/6 facade tests (incl. both halves of the pooling falsifier), clippy clean on default features, fmt clean. `--all-features` is broken pre-existing — `TD-LANCE-GRAPH-ALL-FEATURES-DELTA-BREAK`.

## 2026-07-29 — branch `claude/x265-x266-plans-review-h9osnl` — PROBE-SUDOKU-TEACHER G7 + the fork-closure null result

### Current Contract Inventory — probe surface only (no contract types added)
- `lance_graph_planner::examples::probe_sudoku_teacher` — now **G1–G7, ALL GATES GREEN**. New: `count_completions` (bounded solution enumerator — the fixture VALIDATOR, never called by the reasoner), `try_bifurcate_or_flag` / `ForkOutcome` (the fork's explicit third arm: neither branch contradicting ⇒ `Underdetermined`), `Verdict` + `solve_with_ambiguity_gate`, `find_ambiguous_fixture` + `find_fork_required_fixture` (both DISCOVER-and-verify, never hand-derive), and `Policy::ElectionsFirstWithHidden`.
- **G7 (new)** — the gate a search solver structurally fails: commit on a verified-unique puzzle, REFUSE on a verified-ambiguous one (2 completions, 6 differing cells, none written). Falsifier proven: flipping the refuse arm to "commit the first candidate" fails both refuse assertions with `can_commit` still true.
- **G4 (amended)** — the bifurcate-vs-refuse contrast is explicitly **NOT asserted**, with the reason in the gate's own detail line: it does not exist on this puzzle family. Scan: 26858 unique / 388 singles-stall / **0 fork-closable** / best residual 16. Debt: `TD-FORK-CANNOT-CLOSE-WHAT-SINGLES-CANNOT` (close condition: `fork_closes > 0`). G4's third half instead asserts hidden-singles soundness (Hamming 0) and non-regression (census ≥ baseline).
- **Finding worth carrying:** `base_solution_boxmajor` is the cyclic grid `(f(r)+c) mod 9` and is **provably free of 4-cell unavoidable sets** — a corner swap forces `2(c₁−c₂) ≡ 0 (mod 9)`, and `gcd(2,9)=1` ⟹ `c₁=c₂`. A fixture family can be structurally incapable of exhibiting the property under test. Detail: `EPIPHANIES` `E-A-NULL-RESULT-IS-THE-DELIVERABLE-1`; `AGENT_LOG` 2026-07-29.

## 2026-07-29 — branch `claude/happy-hamilton-0azlw4` — `invoke_recoder`: the SECOND keystone, proving `classid → ClassView → content` dispatch is class-AGNOSTIC (not fitted to one call shape)

### Current Contract Inventory — new entry
- `lance_graph_contract::recoder_adapter::{RecoderStore, RecoderCall, RecoderOut, invoke_recoder}` — the `E-CPP-KEYSTONE-1` analog for the recoder (`UnicharCompress`), same three-step dispatch as `invoke_unicharset`: ClassView composition gate (`methods_for` must list the called method) → content-store tier (state lives in the store, NEVER on the adapter) → the byte-parity-proven adapter leaf. Scope = the load-side runtime surface only: `EncodeUnichar` / `DecodeUnichar` / `code_range`; `ComputeEncoding` (training-side) and the beam-trie accessors (`IsValidFirstCode`/`GetFinalCodes`/`GetNextCodes`) stay OUT — those are Core content the recognizer's beam consumes directly, a compute-tier surface, not a step this keystone routes.
- **Two codex P2s fixed before merge** (`E-TRUNCATING-CONSTRUCTOR-IS-AN-ALIASING-HAZARD-1`): (1) `DecodeUnichar` now rejects `codes.len() > RecodedCharId::MAX_CODE_LEN` as `Ok(None)` BEFORE building the key — `from_codes` truncates, so an overlong sequence was aliasing to its own 9-code prefix and decoding to that prefix's valid id; `MAX_CODE_LEN` is now a public associated const because the boundary guard was otherwise *unwriteable from outside the module*. (2) `DispatchError::NoContentStore` no longer names "UniCharSet" in **either** its rendered message or its doc comment — it is shared by every keystone, so naming one class misreported the missing object for all others. The overlong falsifier was PROVEN to fire on revert (`Ok(Id(Some(0)))` vs `Ok(Id(None))`), and its fixture is a full-length 9-code entry precisely so it *can* fire.
- **`DispatchError` variant set still unchanged** — no new error type, no new variant (only the shared message and doc were neutralized). An out-of-range encode id or an unknown decode code are valid `Ok(..None)` results, not dispatch failures, exactly mirroring how `invoke_unicharset` treats an out-of-range id. The two existing variants cover every recoder failure mode.
- **Classid is derived from the LIVE codebook, not hardcoded** — `render_classid(0x0000, 0x0802)` where `0x0802 = "recoder"` comes from `ogar_codebook.rs:494` under the `0x08XX` OCR domain (concept in the CANON/high half per the active `ClassidOrder::CanonHigh`; Core app-prefix `0x0000` low). A drift-guard test asserts `canonical_concept_id("recoder") == Some(0x0802)`, so a future codebook renumbering fails loudly instead of silently drifting — an improvement on `unicharset_adapter`'s arbitrary `0x0001_0001` test placeholder.
- 9 new tests (1024 lib total), clippy `-D warnings` clean, fmt clean. Detail: `EPIPHANIES` `E-KEYSTONE-CLASS-AGNOSTIC-1`; `AGENT_LOG` 2026-07-29.

## 2026-07-29 — branch `claude/x265-x266-plans-review-h9osnl` (PR #863) — the A9 LE contract + the zero-copy law

### Current Contract Inventory — CausalWitness tenant, WitnessLens, and the warden pair
- `lance_graph_contract::canonical_node::ValueTenant::CausalWitness = 14` (`E-A-FIX-CAN-BE-UNFALSIFIABLE-TOO-1`) — the A9 lane, 16 B content-blind V3 4+12 facet at `row_offset 204` (`[204,220)` row-relative = value-slab `[172,188)`; the register itself at `[176,188)` after the 4 B classid). Read as **G24N4** — 24 signed i4 loci, a LANE shape name, never a `CascadeShape` variant. Slots 16..24 reserved-zero. **Adding it cost the documented two places to AUTHOR (enum variant + `VALUE_TENANTS` row) and three the compiler FORCED** (`ValueSchema::Full::field_mask()` + three carve-total test literals, 172 → 188) — the honest rule is "two to decide, three the compiler enforces". Status: EXPERIMENTAL reading, not in the operator-locked §3 catalogue; sub-byte's sanctioned home is a lane, so it needs no catalogue petition (that inference is CONJECTURE-in-use, labelled in `le-contract-is-the-tenant.md`).
- `lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, from_register_ref, project, elected}` — `#[repr(transparent)]` with const size/align asserts; `from_register_ref(&[u8;12]) -> &Self` is the free cast. Election is `WideFieldMask ∩ register`, **fail-closed** (`EMPTY` is the absent-mask fallback; `full_for()` is a render convenience, NEVER an election fallback). Field-isolation matrix over all 24 slots. `Locus::Antecedent = 7` is the relative-pronoun locus W6 binds.
- `lance_graph_contract::witness_fabric::WitnessLens<'a>` — the borrowed view over `&[NodeRow]`; `at(pos) -> Option<&'a CausalWitnessFacet>` is a cast, offsets DERIVED from `ValueTenant::CausalWitness.value_offset()` and `const _`-pinned at 176. Plus `write_register` (the producer) and five lens twins (`elect_peers_lens`, `quorum_mantissa_lens`, `trajectory_of_lens`, `standing_wave_stratified_lens`, `standing_wave_diagnosed_lens`), each equivalence-tested against its gathered original with the pre-migration body retained verbatim as a `#[cfg(test)]` oracle.
- `lance_graph_contract::dispatch_guard::guard` — **signature changed**: the gathered `window: &[(usize, CausalWitnessFacet)]` no longer exists; takes `focal_pos + &WitnessLens + visible: impl Fn(usize) -> bool`. `TemporalStream::{window_at, window_range}` likewise return borrowing iterators; no `Vec`-returning window accessor remains.
- `.claude/agents/{zero-copy-warden, lens-migration-engineer}.md` — the detection/repair pair. Warden verdicts: LENS-CLEAN / MATERIALIZES / SECOND-PROJECTION / **ELEVATED** (the one carve-out: a strictly-higher-rung derivation is legitimately stored; `Locus::Quorum` is the precedent). Carve-out is stated in BOTH the body and the frontmatter `description:` — the description is what the agent-selection layer reads, and having it disagree with the body was itself a shipped defect this arc.
- **Probes:** `lance_graph_planner::examples::{probe_sudoku_teacher, probe_antecedent_binder}` — G1–G6 and A1–A5 respectively, both ALL GATES GREEN. Read the plan's §4d-RESULTS before citing them: the Sudoku probe is a **mechanism demonstrator on engineered fixtures, not a solver** (hard fixtures go 81 → 80 Hamming), G4 under-tests its own bifurcate-vs-refuse claim, and hidden singles are proven in isolation but NOT threaded into `run_policy`.
- **Still open (9 of 11 warden findings):** `WitnessStream` (7 test-only constructors), `WitnessWindow` (BLOCKED — a stored `Vec` on `PlanContext` needs a lifetime on a shipped type), `style_lane_at` (23 sites), `SpoFacet::from_register`, `NodeGuid::facet`, and the entirely unexamined **18-parameter `revisions:` version-axis family** — which `WitnessLens` does NOT generalize to (different axis: same row at successive Lance versions, so the source is a version-range read, not a row slice).

## 2026-07-28 — branch `claude/x265-x266-plans-review-h9osnl` — rails-shaped rung lift (144-cell reasoning + Morton cascade + qualia)

### Current Contract Inventory — verb matrix cascade + epistemic reading
- `lance_graph_contract::grammar::verb_table::{FamilyQuadrant, TenseQuadrant, morton_cell, same_quadrant, quadrant_prior}` (`E-RUNG-LIFT-RAILS-SHAPED-144-QUALIA-1`) — the 4×4 Morton cascade addressing of the 144: one byte per cell `[fq:2|tq:2|fm:2|tm:2]`, high nibble = coarse quadrant (nibble-ancestry, D-TILE256-shaped), 12×12 occupied in the 16×16 palette256 page, reserves RESERVE-DON'T-RECLAIM. Inverse-pyramid residual probe MEASURED + pinned: mean 0.0774, max 0.500 = Grounds.lokal (outlier catalogue named; lokal = the axis the carve compresses worst). Deterministic throughout.
- `lance_graph_contract::grammar::verb_lexicon::{IRREGULAR_PASTS, EpistemicReading, epistemic_reading}` — the rails-shaped rung-lift condition: cue gate (that-complement licensing) + 144-cell read (tense-modulated modal = epistemic force + Morton address). Epistemic lemmas minted: SEE-class → Mirrors (0.70), KNOW-class → Abstracts (0.85) — WordNet's verb.perception/verb.cognition supersense split independently confirms the cut. Catalogue↔matrix WELD test forbids drift. Irregular pasts (knew/saw/understood/…) classify with correct tense.
- `probe_eyes_opened` B5 — awareness quale = blind × context (cell modal × Staunen-at-lift), context snapshotted BEFORE the lift's own output (codex P1 fix — the first cut leaked modal into both factors and the fix FLIPPED the corpus verdict): scene-scale asserted (3:7 = 0.85×0.133 = 0.113 > 3:6 = 0.70×0.053 = 0.037, both factors independently); corpus-scale the BLIND ranking over ten lifts crowns 3:7 (0.047; 1:4's pristine arena reads exactly 0.000 context). Codex P2 also fixed: homograph irregulars (saw/bore) stay sparse in the public classifier, resolving only under the cue gate. PROBE-QUALE-LOCAL demoted to optional sharpening. Also queued: PROBE-WORDNET-QUADRANTS, PROBE-COMMA-144 ([S] until a phase-reader is named). O7 fence held. Detail: EPIPHANIES + AGENT_LOG 2026-07-28.

## 2026-07-28 — branch `claude/x265-x266-plans-review-h9osnl` — PROBE-EYES-OPENED (the Adam awareness printed blind from the KJV bake)

### Current Contract Inventory — new clause_cues catalogues + probe example
- `lance_graph_contract::grammar::clause_cues::is_negation` (`E-EYES-OPENED-PRINTS-BLIND-1`) — predicate-polarity negation cues (`not/no/never/neither/nor/cannot`, exact catalogue; excludes `without`/`nothing`/`none` with rationale). The load-bearing capability: an extractor observing `f≈0.05` under a negation lets the arena HOLD an invalidation, so a later positive observation produces a genuine NARS revision with preserved contradiction depth. Fire+silent tests.
- `lance_graph_contract::grammar::clause_cues::is_perception_verb` — the rung-lift operators (`knew/saw/perceived/understood/realized/...` incl. KJV forms), deliberately NOT in `verb_lexicon::FAMILY_LEXICON` (epistemic level ≠ TEKAMOLO slot — conflating them flattens the Tarski ladder into the relation plane). Fire+silent tests.
- `lance_graph_planner::examples::probe_eyes_opened` — the four-blade probe (reversal / reflexive rung lift / causal chain / Hermeneutik pass-2) with a self-asserting 13-verse KJV fixture and a local-only real-corpus mode. Measured headline: on real Genesis 1–4 (106 verses) the ONLY self-referential rung lift of ten is 3:7 `they —knew→ naked`, found blind; blind contradiction ranking = {eat, die, god→good, they→respect} all genuine; pass-2 re-read converges (admitted=0, revised=0, fixed point) — NARS's S4 overlap guard is the hermeneutic circle's termination proof. Detail: `EPIPHANIES` E-EYES-OPENED-PRINTS-BLIND-1; `AGENT_LOG` 2026-07-28.

## 2026-07-27 (later) — post-#854 target shapes WITHDRAWN; zero-copy discriminator ruled; next task is the §12 substrate trace

### ⊘⊘ The redo-sequence architecture proposed after #854's merge is WITHDRAWN as unauthorized inference
Belief rows with belief GUIDs · evidence-event rows with event GUIDs · evidence edges · a "concept-belief SoA" · CSR as a "compute projection" · the opaque `BeliefHandle` carrier · the named future query API · the PR B–E sequence — **none code-proven, none owner-specified**. All marked ⊘⊘ WITHDRAWN in place in `identity-temporal-evidence-primer.md` (§5.8/§10/§11); the rule that prevents recurrence: *an unfilled semantic slot means trace the substrate, not invent a carrier* (`E-AN-UNFILLED-SEMANTIC-SLOT-IS-NOT-A-DESIGN-INVITATION-1`).

### Operator rulings now in the primer (§11)
1. **Never serialization / materialization / reconstruction / copied intermediate state / detached canonical state / sidecar. "Everything is zerocopy period."** A projection is valid only as a borrowed interpretation of resident bytes.
2. **The discriminator:** the sole permitted accumulation is entropy-reducing reasoning committed as new semantic state (NARS truth, `CausalEdge64`, qualia, meta, rung, contradiction) into the owning SoA via its Kanban. Copy/reorganize of existing state = forbidden; kernel-local arithmetic = permitted.
3. **The L3 argument:** 65,536 × 512 B = 32 MiB, fully L3-resident — repackaging is *slower* than reasoning directly over the resident SoA. The SoA is already the compute structure.

### What survives (descriptive)
Consumer census 10/10 (incl. the 3 order-dependent budget-cap sites in `tactics.rs` and positional `premises`); census caveats (f32 accumulation order, tie-breaker totality — two new census columns); A0 census-completion agents dispatched (float/tie/mutation/cardinality columns + deepnsm-v2 drift ledger). Facts: `BeliefArena` owns detached heap state (cannot survive as owner); `AdjacencyStore::from_edges` allocates a CSR (cannot be a V3 stage as written).

### Next task (blocked on nothing): primer §12
Blast-radius classification (`CODE-PROVEN` / `OWNER-SPECIFIED` / `UNAUTHORIZED INFERENCE`) + substrate trace per reasoning path: exact resident bytes, SoA vs detached, adjacency input chain, allocation inventory against the discriminator, owner+Kanban per mutation, output destination, MISSING paths reported not bridged. **No target design.**

## 2026-07-27 — branch `claude/medcare-rs-transcode-ruff-3y2olh` — **C1 WITHDRAWN (falsified design)** + D1 shipped: the four-signal settlement field

### ⊘ `source_registry` was ATTEMPTED and WITHDRAWN — it is NOT a shipped contract
`lance_graph_contract::source_registry` and both `BeliefArena` migrations were **reverted from PR #854 before merge**. They are recorded here as a **falsified design**, not an inventory entry — nothing in the tree provides them.

**Why it was withdrawn (operator-ruled).** An evidential base needs **evidence-EVENT identity**; `SourceId` modelled **source membership**, a different object. Four facts converged:
1. `SourceId` ≠ `EvidenceEventId`.
2. **No canonical evidence-event identity exists in this substrate** — verified, not assumed. `ClassId:AppId:ClassView + LanceVersion` cannot name one: `ClassId` is a class (`= u16`; the GUID's is a `u32` composite), `ClassView` is a late-bound projection *trait*, `AppPrefix::Core` is literally "no render lens", and `LanceVersion` is a **dataset snapshot**. Two rows of one class in one commit are indistinguishable; a re-observation of an *unchanged* value produces no mutation at all; one observation can span many rows. Identity belongs to an **immutable receipt**, which does not yet exist.
3. **A fixed-width digest is safe but useless for this query.** Measured (20 000 trials/cell, genuinely disjoint bases, `digest_a & digest_b == 0`): P(false overlap) ≈ **n²/m** at k=1 — 63.4 % at n=8/m=64, 22.2 % even at m=256; **k > 1 is catastrophic** (98 % at n=8/m=64/k=2) even though it improves membership FPR. A digest that reports overlap on two thirds of disjoint pairs disables evidence accumulation.
4. **`bool disjoint()` cannot express the needed distinction.** "Not known to overlap" ≠ "known disjoint". The API must be tri-state.

**The operational failures both reviewers found were smoke from this, not isolated bugs:** four examples panicking past 64 sources; `reach_out_integrate` swallowing `CapacityExceeded` into `DullShadow`; `asc_challenge` reporting capacity exhaustion as `BlockedSelfReference`; cross-registry stamp comparison. **CodeRabbit's prescribed fix — "reuse a bounded `SourceId`" — was NOT taken:** minting one identity per distinct observation is correct; the 64-ceiling is the defect.

**Rollback ≠ endorsement.** The restored local `Stamp` still models source membership, and it is still **lossy**: `Stamp::source(id) = 1 << (id % 64)` gives a bounded 64-slot horizon in which ids `0` and `64` alias. What the rollback removes is the **runtime capacity failure** (`CapacityExceeded` on the pre-cast path) — NOT the bounded membership semantics, which are unchanged and remain conservative (aliasing can only manufacture false overlap, never false disjointness, so revision under-pools rather than double-counts). It is the pre-PR baseline, kept only because it introduces no breaking API and no synchronous refusal while the correct model is designed; exact evidence-event identity remains the standing remedy for provenance collisions (codex/CodeRabbit, PR #854).

**The architectural gain, which outlives the code:** `event identity ≠ evidential-base membership ≠ source dependence ≠ object/view identity ≠ dataset version`. The registry was the sacrificial scaffold that separated them. Next shape: `EvidenceEventId` (canonical immutable receipt) + `EvidentialBase<K>` (exact inline, `overflow` → `Unknown`, ledger fallback — **no eviction**) + `OverlapKnowledge` / `Independence`, both tri-state. Open identity question: what guarantees two independently-minted events cannot collide — to be settled from the mailbox/ingestion/persistence ownership model, not from numeric capacity.

### Current Contract Inventory — one new module (D1)
- `lance_graph_contract::settlement::{SettlementSignals, SettlementCell, SettlementScope}` — settlement as a FOUR-signal field. Discriminator is **closure × competence**, NOT entropy: Crystal / **Glass** (dense closure on thin evidence — the dangerous cell a scalar hides, since it looks like Crystal from one side and Fog from the other) / GroundedUnresolved / Fog. `field_entropy` + `eigenvalue_concentration` REFINE a cell and are pinned by test never to move it — the earlier "crystal = low entropy + high closure, glass = low entropy + low closure" formulation had entropy on both axes and silently deleted competence. `SettlementScope` (arena/basin/version/branch/witness-horizon) is carried WITH the signals and `comparable_to` refuses mismatched pairs — the alignment precondition that made `wisdom − competence` meaningless, made structural. **No `glass_gap()` scalar is provided**, deliberately: the subtraction is how four signals become one again, and neither axis is calibrated yet. 7 tests incl. an orthogonality receipt + threshold-inertness.

Gate: contract 1093 green, planner 319, deepnsm-v2 96, fmt clean, no new warnings. **(deepnsm-v2 is 114 as of 2026-08-22** — PR #987 removed six modules added earlier that duplicated `insight_coca_read.rs` / `probe_antecedent_binder.rs` / the planner's TEKAMOLO write; the crate is back to its pre-session surface.)

## 2026-07-27 — branch `claude/medcare-rs-transcode-ruff-3y2olh` — causality-audit fixes A1/A2/A4/A5 + B1: typed causal edges, declared kernel effects, derived Libet window

### Current Contract Inventory — new module + two trait methods + one field REMOVED
- `lance_graph_contract::causal_audit::{AuditedRelation, RelationClassification, CausalLocus, WorldDomain, CausalScope, NonCausalKind, SupportLedger, SupportReceipt, SupportProfile, SupportBasis, SourceId, RelationId}` — the typed causal edge the `E-WE-HAVE-PEARL-VOCABULARY…-1` ruling ("AUDIT BEFORE BUILD") required. **Four orthogonal axes, never merged:** kind (sum type — a non-causal relation *cannot* carry a locus; `Unclassified` is an honest resting place), **locus** (World/Interpretive/Derivational/Experiential — *where in the architecture*, NOT subject matter), domain, scope (Type/Token — NOT grammatical voice). **Support is many-of:** a receipt ledger, not a single enum, so text-attested + derivational + cross-environment coexist; `SupportProfile` is a DERIVED projection that keeps `receipt_counts` and `distinct_sources` separate (3 independent attestations ≠ 1 attestation ×3). `is_intervention_established()` requires a causal classification AND an `InterventionBacked` receipt — a corpus edge can never reach it. Classification and support are separately addressable (support accumulates while unclassified; reclassification never rewrites receipts). 8 tests incl. two orthogonality receipts. Known gap, labelled in source: `SupportReceipt::at` is a `DatasetVersion` (storage revision), NOT an epistemic view — the `QueryReference` upgrade is owed.
- `lance_graph_contract::recipe_kernels::{KernelMaturity, Tactic::writes, Tactic::maturity}` — the effect census made executable. `writes()` = POSSIBLE writes (mirror of `requires()`'s may-read); `maturity()` = Operational/Demonstration/Stub, on the **impl** not the `Recipe` catalogue entry. Census: 27/6/1. 7 falsifier tests (`effect_census`). Fixes `Lsi`, which declared `Sd` as a required *input* it only ever wrote. Detail: `EPIPHANIES` `E-ZERO-DELTA-DOES-NOT-MEAN-NO-EFFECT-1`.
- `lance_graph_contract::kanban::{LIBET_COMMIT_WINDOW_US, KanbanMove::libet_window_us}` — **REMOVES the `KanbanMove::libet_offset_us` field** (breaking for external constructors, deliberately). The window is a projection of `(from, to)`, so a stored field could only ever disagree with the transition it describes. Orthogonality audit: `(from,to)` varies while the offset holds, but nothing makes the offset vary while `(from,to)` holds ⇒ one-directional ⇒ derived. The literal `-550_000` had THREE definitions (scheduler / soa_view test double / planner constant) with TWO different stamping conditions; now one. `size_of` assert stays an upper BOUND — `KanbanMove` is a Rust-repr microcopy, not an ABI. Migrated across 9 crates.
- `lance_graph_cognitive::world::{substitute_binding, multi_substitute_binding, BindingSubstitution, SubstitutedWorld}` (was `intervene` / `Intervention` / `CounterfactualWorld`) — **renamed away from do-calculus, algebra kept.** It severs no mechanism, recomputes no descendants, holds no exogenous background fixed; it IS an exact reversible XOR substitution primitive, which is genuinely useful and now says so. Pearl citations stripped from the module docs.

Gate: contract 1086 tests green (fmt clean, no new warnings); planner 317 + supervisor/shader-driver suites green. `lance-graph-cognitive::grammar::qualia::test_depth_detection` fails — verified PRE-EXISTING at clean HEAD via stash, untouched by this work. Board: `EPIPHANIES` `E-THE-UNCONTESTED-AXIS-IS-THE-ONE-THAT-MERGES-1` + `E-ZERO-DELTA-DOES-NOT-MEAN-NO-EFFECT-1`.

## 2026-07-26 — branch `claude/lance-graph-last-10-pr-z30uij` — D-SCI-1 Phase 2: witness-gated construction licenses (PROIEL Greek NT)

### Current Contract Inventory — new grammar witness module + planner example
- `lance_graph_contract::grammar::witness` (`E-SCI-1-WITNESS-CONSTRUCTION-LICENSE-1`) — typed construction licenses from source-language treebanks (treebank-agnostic): `VoiceClass` (Active/Passive/Middle/±Deponent(reserved)/Ambiguous; `from_proiel_code`, `licenses_agentive` — Middle licenses, Passive does not), `WitnessDisposition` (Confirmed/Compatible/Contradicted(reserved — ordering never justifies it)/TextAbsent/AlignmentUnknown — the latter two NEVER block), `ClauseSignature` (dependency-first: `has_fronted_argument` counts `obj`+`obl` because government verbs like ἀκούω+genitive make case lie; `licenses_fronted_object_active`; edition/tradition in the evidence address). 4 unit tests; contract 1037 green.
- `lance_graph_planner::examples::insight_witness_gated_read` — PROIEL greek-nt.xml → per-citation `ClauseSignature`s; the generation/elimination law (generation = English OR witness evidence; elimination = all typed constraints); fronted-NP-object candidate commits ONLY under a license, pronoun-case commitments stay witness-independent with receipts. Falsifier: Acts 3:22 Confirmed (fronted `obl` + Middle — the government-verb path), Acts 7:37 TR clause TextAbsent (surfaced, not vetoed), Deut 6:13 TextAbsent (out-of-corpus), `a prophet shall the lord your god raise up` Phase-1-miss → licensed `god—raise→prophet`, negative control held. PROIEL data gitignored (`examples/data/proiel/`, CC BY-NC-SA).

## 2026-07-26 — branch `claude/lance-graph-last-10-pr-z30uij` — D-SCI-1 Phase 1: right-corner delayed clause commitment (KJV OSV)

### Current Contract Inventory — new grammar cue catalogue + planner example
- `lance_graph_contract::grammar::clause_cues` (`E-SCI-1-RIGHT-CORNER-DELAYED-COMMITMENT-1`) — delayed-clause-commitment cue catalogues: `pronoun_case` (Nominative `I/he/she/we/they/thou/ye` · Accusative `me/him/us/them/thee` · **Ambiguous** `you/it/her` — case-eroded, never decisive) + the modal spine `is_modal_aux`/`modal_tense` (incl. KJV `shalt`/`wilt`; shall→Future, might→Potential). Zero-dep catalogues, not algorithms. 4 unit + 2 doctests.
- `lance_graph_planner::examples::insight_right_corner_read` — the `AwaitingClause` scan: `O AUX S V` → canonical ACTIVE `S —V→ O`, tense off the modal, Ambiguous-fronted never commits (honest incompleteness beats a wrong parse); coexists with the left-corner control. Live whole-KJV run: 21 commitments, 21/21 defensible (`ye—hear→him`, `thou—serve→him`, `i—pray→thee`, …; was 6 before the codex/CR #849 round: split `:?!,` + connective/recipient-PP skip, `unto`/`to` only). Phases 2-4 queued in the epiphany.

## 2026-07-23 — branch `claude/lance-graph-last-10-pr-z30uij` — D-SCI-1 TEKAMOLO value tenant + the new reasoning wired via SPO-G + all tenants

### Current Contract Inventory — new value tenant + wired-reasoning example
- `lance_graph_contract::canonical_node::ValueTenant::Tekamolo` (`E-SCI-1-TEKAMOLO-TENANT-WIRED-VIA-SPOG-1`) — the 14th value tenant (discriminant **13**), first content-blind V3 4+12 facet to become a lane: `VALUE_TENANTS` descriptor `[188,204)` U8×16 = the `TekamoloFacet` (`classid(4)+6×(u8:u8)`), read G4D3 as the `temporal · kausal · modal · lokal` when/why/how/where address (each a `256:256:256` cascade). Added to `ValueSchema::Full` (Cognitive/Compressed unchanged); Full carve 156→172 B, layout-preserving (≤480, `NODE_ROW_STRIDE` unchanged, no `ENVELOPE_LAYOUT_VERSION` bump). Field-isolation + contiguity asserts updated. 1029 contract lib tests green.
- `lance_graph_planner::examples::insight_reason_wired` — the new reasoning wired end-to-end: one clause → `Triple` SPO-G quads (a `Graph` `G` slot = `Utterance` relation beside `WordNet` `is_a`/`instance_of` rails, the two-basin symbolic-vs-field split) + the canonical value tenants Qualia(#1)/EntityType(#8)/Meta(#0)/**Tekamolo(#13)**, with a slab round-trip proof that the facet lands byte-for-byte in the `ValueTenant::Tekamolo` carve. Additive — the existing 3×SPO + 3×AriGraph grouping is untouched. Loads the two-basin store (COCA lexicon + WordNet rails) from Release assets (gitignored, skips cleanly if absent). PoC↔canonical switch points noted (`type_code()`→`ogar_codebook::canonical_concept_id`; is-a rails→`lance-graph-ontology`).

## 2026-07-23 — branch `claude/x265-x266-plans-review-h9osnl` — D-SCI-1 archetype consumer + FSM movement feeder (on top of #841)

### Current Contract Inventory — new grammar module + planner example + fsm tag
- `lance_graph_contract::grammar::verb_lexicon` (`E-SCI-1-VERB-TABLE-ARCHETYPE-CONSUMER-AND-FSM-FEEDER-1`) — the CONSUMER of the 144-cell `verb_table` archetypes: `classify_verb(word) -> Option<(VerbFamily,Tense)>` (fixed lemma→family table + regular `-ed`/`-ing`/`-es`/`-s` morphology with `-e`/doubled-consonant restoration), `slot_for(family,tense) -> TekamoloSlot` (reads `base_prior.combine(tense_modifier)` → `dominant_slot` argmax), `read_verb` (the one-shot family+tense+slot), `is_copula`/`is_causal_cue` (route `Inh`/`Impl`). Zero-dep, no model. 8 unit + 2 doctests. Starter map (corpus-tune logged).
- `lance_graph_planner::examples::insight_archetype_read` — the archetype-typed relation extractor: types each verb-mediated edge via `read_verb`; discriminative falsifier (causal corpus → plurality Kausal, grounding → Lokal) asserts the extractor consumes `verb_table`, not a flat list.
- `deepnsm_v2::fsm::Pos::Rel` — the movement feeder-tag: relativizer/complementizer promoted out of `Pos::Other`; `parse_to_spo` gains a single-level relative-clause sub-machine that preserves the matrix subject (object- + subject-relative), feeding the ±8 `antecedent` pointer (`wave.rs`). 4 new property tests. Detail: `EPIPHANIES` E-SCI-1-VERB-TABLE-ARCHETYPE-CONSUMER-AND-FSM-FEEDER-1; `AGENT_LOG` 2026-07-23.

## 2026-07-23 — branch `claude/x265-x266-plans-review-h9osnl` (PRs #817/#818/#819/#820/#822/#823/#824, MERGED) — the dialectic engine's LOOP + V4 foveated field-search (D-DIA-V2 → V4)

### Current Contract Inventory — new field-search + fold modules
- `bgz17::palette::{build_hierarchical, HierarchicalPalette}` (#823) — divisive 16-coarse×16-fine k-means codebook where `code>>4==coarse` IS centroid ancestry (the `D-TILE256` rigor condition, reusing `hhtl.rs::NiblePath`). Flat `build` untouched. `PROBE-CODEBOOK-44` retires Probe M1.
- `bgz17::palette_semiring::premultiplied_over` (#824, rung 3) — the composite floor `Σ weightᵢ·value(codeᵢ)` over `[i64;17]` signed Base17 dims (commutativity gate PASS); the alpha-over composite lives in bgz17's palette world, NOT blasgraph's HDR semiring.
- `bgz17::examples::probe_foveated_descent::foveated_descend` (#824, rung 2) — foveated morton-comma descent: materialize only the `fovea_k` nearest coarse clusters' leaves, prune periphery. fovea_k=2 → 8× prune + full recall.
- `lance_graph_planner::nars::facet_fold::{to_spo_facet, cstmt_from_spo_facet}` (#824, M26) — lossless content-blind `Belief⟷SpoFacet` byte relabel (rails 0-3 carry `CStmt` exactly incl. `Rel(v)>255` two-rail split); a `Belief` IS a reading of the M20 `awareness_facet::SpoFacet` register, never a new store.
- `lance_graph_planner::nars::insight` (#819) — the S10 insight-vs-mush detector over a before→after `BeliefArena` snapshot (reuses `GraphSignals`/`FlowState`): `insight = clamp(Δcoh + Δwonder, 0, 1)·[yield>θ]`, coherence = closure density (size-invariant).
- **Ruling (`E-FOVEATED-HHTL-TRIE-FIELD-SEARCH-1`, #820 + #822 fold):** field search = the guaranteed-terminating FLOOR of an addressing-first ladder → the whole escalation is a TOTAL function; foveation IS the pruning; deferring materialization IS Kuzu factorized processing. Folds onto shipped survivors (`SpoFacet` field element, `NiblePath` ancestry, palette-table composite) — the plan EXPANDS the ENTROPY ledger, never re-describes it.
- **Open (operator-gated):** anchor-level real-data ρ is blocked by the Base17 17-dim fold ceiling (ρ=0.2599, `TD-BASE17-FOLD-CEILING-SINGLE-WORD`), NOT the codebook (hierarchy is fidelity-neutral on real Jina data — structure-is-free confirmed); the close needs higher-dim/structured Base17 input (a design steer, not grindwork). 134 bgz17 + 24 nars tests, clippy `-D warnings` clean. Detail: `PR_ARC_INVENTORY` #817..#824; `AGENT_LOG` 2026-07-23.

## 2026-07-23 — branch `claude/x265-x266-plans-review-h9osnl` (PRs #814/#815/#816, MERGED) — the dialectic engine's REASONING LAYER: the five NARS tactics in `lance-graph-planner/src/nars` over the one engine

### Current Contract Inventory — new planner reasoning modules
- `lance_graph_planner::nars::belief::{BeliefArena, Belief, CStmt, Copula, Stamp, ReviseOutcome}` — the statement-keyed dialectic Belief arena over `TruthValue` (S2 triple-keyed dedup + CHOICE-on-`expectation()`; S3 copula-gated `close_transitive` (only Inh/Sim transit); S4 stamped `observe`/`revise_at` — disjoint→revision, overlap OR empty-stamp→CHOICE; `admit_derived` the shared throttled-frontier path, observation-ground guard keys on the STAMP).
- `lance_graph_planner::nars::tactics::{rcr_abduce, tr_diverge, cas_abstract, asc_challenge, cr_synthesize, Candidate, ReasoningGap, GapKind, Throttle, Frontier, Tactic, AscOutcome, challenge_target}` — the FIVE tactics as term logic over the one engine (RCR=abduction, TR=analogy, CAS up=induction/down=deduction, ASC=disjoint-stamp self-critique, CR=dialectic revision); S5 throttle (c_min/budget/hub-exclusion, sorted-deterministic); pinned to `contract::recipe_dispatch` (RCR=4/TR=6/ASC=7/CAS=8/CR=11).
- `lance_graph_planner::nars::truth::TruthValue::analogy` — the missing NAL analogy truth (`f=f·f_sim, c=c·c_sim·f_sim`), added by extending the one engine (never a local reimpl).
- **Ruling (`E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`):** the dialectic reasoning lives HERE (lance-graph reasoning layer); `deepnsm-v2` is the INBOUND leg (forward encode → belief stream), never a reasoning home. The V0 `deepnsm-v2/belief.rs` arena is superseded — dedup owed (`TD-DEEPNSM-V2-BELIEF-DUP`). 17 nars + 233 planner tests, clippy `-D warnings` clean. Detail: `AGENT_LOG` 2026-07-23; `PR_ARC_INVENTORY` #814/#815/#816.

## 2026-07-22 — branch `claude/x265-x266-plans-review-h9osnl` — trained Cam96 codebook SHIPPED (`deepnsm-v2/data/` + `codebook` loader) + the whole-book Bible falsifier runs ALL-GATES-GREEN (63.3% of context beyond ±5)

### Current Contract Inventory — new entries
- `deepnsm_v2::codebook::{load_cam96_space, load_cam96_codes, CodebookError}` — LE loader for the TRAINED codebook artifacts (`data/cam96_codebook.bin` 96 KB `CAM96CB1`, `cam96_codes.bin` `CAM96WD1`, `bible_vocab.txt`), produced by `probes/` from real Jina-v3 96-d embeddings of the 12,543-word KJV vocab. Held-out ρ 0.774 (96-bit) vs 0.617 (48-bit); equal-budget RQ point control 0.786 → the distribution's advantage is ALGEBRAIC (addressable bytes/rails), not raw fidelity — recorded honestly. Retires `demo()` for real use; pays the producer third of `TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED`. 2 loader tests (35 total green).
- `crates/deepnsm-v2/examples/bible_wave.rs` — the WHOLE-BOOK falsifier: KJV (PD, not committed; path arg) → verses → COCA-lemma PoS + archaic fallback → FSM → SPO stream (verse = version) → `TemporalStream` + trained codes. 4 in-code gates ALL PASS: 23,145 verses = one 64k tile; 31,327 triples / 606 subjects; sim(god,lord) 0.625 > sim(god,fish) 0.265; **63.3% of 27,086 same-subject links beyond ±5** (v1-ring forfeit), 55.7% beyond ±8 (Escalate share). `E-WHOLE-BOOK-WAVE-1`.

## 2026-07-22 — branch `claude/x265-x266-plans-review-h9osnl` — `deepnsm-v2::space::{Cam96, Cam96Space}` + `Nsm` routing/meaning split, Jina-grounded (ρ 0.828 vs 0.711; used_for reasoning 0.667 vs 0.500)

### deepnsm-v2 — CAM-PQ 96 DISTRIBUTION meaning code (operator-pinned point→distribution ladder, now measured)
- `deepnsm_v2::space::{Cam96, Cam96Space}` — the 96-bit `6×(u8:u8)` = `6×palette256:palette256` = `6×cosine²` DISTRIBUTION word code (12 axis codebooks; `encode` per-axis nearest-centroid; `distance` = Σ 12-axis squared-L2, additive-exact, absent→`+∞`; `rails()` = the 6-rail view; normalized `[x;y]` metric — NO cosine call). The granular upgrade of the 48-bit `6×cosine` POINT (`AdcSpace`, kept as the v1 reference shape). `Nsm` reshaped: **routing** (frequency-ranked vocab id, measured ⟂ meaning ρ≈−0.07) and **meaning** (`Cam96` per-word code via `with_codes`, ρ 0.828 vs Jina) are now separate axes — `word_similarity`/`triple_similarity` read the meaning code, fixing the shipped defect that read ONE `256:256` rail as the semantic code. `probes/` (3 scripts + README, `JINA_API_KEY` from env only): frequency⟂meaning; count-tier coarse (curated ρ .762 live-reproduced, random-pair AUC .567); 48 vs 96 fidelity 0.711→0.828 (+16.5%, MSE −41%); used_for SPO 2³ analogical purity 0.500→0.667 (predicate arithmetic FAILS → relations are STORED edges). 32 tests, clippy `-D warnings` + fmt clean. E-CAM96-DISTRIBUTION-MEASURED-1.

## 2026-07-22 — branch `claude/review-claude-board-files-nhqgx1` — `deepnsm-v2::wave::WitnessStream`: the standing-wave RESOLUTION complement over deepnsm-v2's version-range window (`ISS-BUNDLE-RULING-SCOPE` ruled (b))

### deepnsm-v2 — new module (existing modules untouched)
- `crates/deepnsm-v2/src/wave.rs` — `WitnessStream`: version-stamped, single-owner `CausalWitnessFacet` loci events, resolved `Causal`/`Escalate` by the SHIPPED `witness_fabric::standing_wave_grounded`/`resolve_chain` over `TemporalStream`'s version-range window. `window_at`/`window_range` mirror `TemporalStream`; absolute stream positions carry the ±8 reference horizon; a target at a not-yet-visible version → `Escalate` (widen the version read — the horizon meets the version read, `E-HORIZON-NOT-BOUND-1`). Complements `TemporalStream` (the WINDOW) with the multipass wave RESOLUTION it left open — no bundle, no shared register (`E-NO-BUNDLE-STANDING-WAVE-1`). `+pub mod wave;` + one re-export in `lib.rs`; fsm/space/spo/vocab/`TemporalStream` UNTOUCHED. 10 new tests (28 total), clippy `-D warnings` + fmt clean. Operator ruling `ISS-BUNDLE-RULING-SCOPE` path (b): keep the MarkovBundler; NO old-`deepnsm` duplication.

## 2026-07-22 — branch `claude/x265-x266-plans-review-h9osnl` — `deepnsm-v2`: DeepNSM rebuilt on the V3 palette256² architecture as the FIRST real consumer of the certified distance table (`PairPalette`/`ScalarAdc`); the existing `deepnsm` crate is untouched

### Current Contract Inventory — new consumer crate
- `crates/deepnsm-v2/` — a parallel, updated DeepNSM (workspace-`exclude`d standalone crate; single dep `lance-graph-contract` via path; own tracked `Cargo.lock` per the bgz17/deepnsm convention). Keeps the DeepNSM *signature* (frequency-ranked vocabulary + PoS FSM → SPO) but rebuilds the substrate on V3, **consuming the contract primitives rather than reimplementing them**. v1→v2 mapping: 4,096-word COCA table / 12-bit ids → `256×256` palette tile / 16-bit ids (`vocab::PaletteVocab`, `split=(id>>8, id&0xFF)`, `vocabulary = frequency × distance`); the stored `4096²` u8 distance matrix → the certified palette256² distance (`space::SemanticSpace` wrapping `recipe_substrate::PairPalette`); whole-work-is-one-tile `6×256` CAM (`space::AdcSpace` wrapping `cam::ScalarAdc`, exercising the `Σ_s‖q_s−c_s‖²=‖q−c‖²` exactness through the space); ±5 sentence ring → version-range read (`TemporalStream` over `temporal_pov::TemporalPov`/`VersionRange`); 512-bit VSA XOR → palette `(basin, identity)` addressing (`spo::Spo`); 6-state PoS FSM → SPO preserved (`fsm::parse_to_spo`). **Honest scope:** `space` ships DETERMINISTIC `demo()` codebooks (placeholder distances, no rng/clock) plus `from_axis_codebooks`/`from_codebook` for real trained ones — the crate wires the architecture and is test-proven end-to-end on the demo codebook; it reads NO real corpus (still `ISS-DCSW-REAL-CORPUS-BLOCKED`) and real semantics still need the trained-codebook producer (`TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED`). 18 tests green; clippy `-D warnings` clean; fmt clean. E-DEEPNSM-V2-PALETTE-ARCHITECTURE-1.

## 2026-07-21 — branch `claude/review-claude-board-files-nhqgx1` (PR #793, MERGED) — anti-pattern-matching PreToolUse guard; a MarkovBundler deletion mistake caught + fully reverted

**Tooling, not a contract type.** `.claude/hooks/anti-pattern-matching.sh` + `settings.json` `PreToolUse(Grep|Bash)`: fires on `Grep`/`grep`/`rg`/`sed`/`tail`/`head` and injects the rule that these are discovery-search only, never a comprehension substitute — do not act on a match before a full `Read` (operator directive). **No source change** this session: an earlier deletion of the `deepnsm` MarkovBundler cluster (only pattern-matched, never read; no tested replacement) was caught by the operator and FULLY REVERTED (`git reset --hard origin/main`, nothing lost); PR #790 + #792 CLOSED. The no-bundle ruling `E-NO-BUNDLE-STANDING-WAVE-1` was issued but its EPIPHANIES record was reverted with the over-deletion — re-recording it is an OPEN board item. Detail: `AGENT_LOG` 2026-07-21.

## 2026-07-21 — branch `claude/x265-x266-plans-review-h9osnl` — D-CSW-1 leg-2 registered INFRA-BLOCKED (no `protoc`, `lance-graph-planner` unbuildable here); D-CSW-2 CONTRACT-LEVEL mechanism probe PASSES (joint precision@25 1.000 vs 0.520/0.520 ablations, margin +0.480 each) — both registered before code, per plan discipline

### Current Contract Inventory — new entry
- `crates/lance-graph-contract/examples/probe_dcsw2_basin_rung.rs` — the D-CSW-2 contract-level scoping probe. Zero-dep, deterministic (index-derived, no rng/clock). Consumes `recipe_substrate::PairPalette` (basin co-occupancy, PR #787's certified distance table) + `witness_fabric::standing_wave_grounded` (rung survival, mirrors the exact `dispatch_guard.rs` test fixtures verbatim) on a synthetic AND-gate 4-group fixture. Result: joint precision@25 = 1.000 vs basin-only/rung-only 0.520 each (margin +0.480, registered pass ≥0.15). Promotes the JOINT-SIGNAL MECHANISM to a scoped FINDING — explicitly NOT the real-corpus D-CSW-2 claim (needs real basins from real data). `cargo test -p lance-graph-contract --lib`: 1008/1008 green; clippy + fmt clean. E-DCSW2-CONTRACT-MECHANISM-GREEN-1; plan §6.3.
- Plan `.claude/plans/causal-rung-standing-wave-v1.md` §6.2 — D-CSW-1 leg 2 (real `temporal.rs`/Lance versions, wild corpora) registered INFRA-BLOCKED in this sandbox: `lance-graph-planner` needs `protoc` (verified absent) and its dependency fetch timed out at 4.5 GB free disk; no labeled corpus sourced. Reported honestly as a registered kill of *this attempt*, not the underlying claim (leg 1's v5 core standing-wave result stays VALIDATED-IN-SCOPE).

## 2026-07-21 — branch `claude/x265-x266-plans-review-h9osnl` — `cam::ScalarAdc` + `recipe_substrate::PairPalette`: the REAL 6×256 / palette256² distance-table math wired as the contract's scalar reference (the `DistanceTableProvider` trait had none) — provably EXACT, not the byte-L1 stand-in

### Current Contract Inventory — new entry
- `lance_graph_contract::cam::{ScalarAdc, AdcMetric}` — the zero-dep scalar reference impl of `DistanceTableProvider` (the trait shipped with NO contract impl → the byte-L1 stand-ins existed by default). `precompute(query, codebook) → [[f32;256];6]` = real per-subspace query→centroid distance over the 256 trained centroids; `distance(tables, cam)` = sum of 6 lookups. `AdcMetric::{SquaredL2, Cosine}` (Cosine read through `Distance::similarity_z` FisherZ = cosine-replacement). PROVEN exact by `adc_ssd_is_exact_not_l1` (`Σ_s ‖q_s−c_s‖² = ‖q−c‖²`, additive decomposition — the property that makes it a distance table, not an approximation). The reference ndarray shadows with AVX-512 (`distance.rs:72-78` pattern); codebook passed in. 4 tests + doctest. E-ADC-SCALAR-REFERENCE-WIRED-1.
- `lance_graph_contract::recipe_substrate::PairPalette` — the palette256² special case for the `(u8,u8)` two-axis byte pair `pair_similarity` takes. Two axis codebooks; `(b0,b1)` reconstructs to `basin[b0] ++ identity[b1]`; `similarity`/`distance` = the REAL centroid distance (additive over axes, same exactness). `pair_similarity` stays the documented **no-codebook L1 default**, now pointing at `PairPalette` as the exact upgrade behind the same call shape. `real_palette_diverges_from_l1_grid` proves L1 ≠ the palette metric. 3 tests. E-ADC-SCALAR-REFERENCE-WIRED-1.

## 2026-07-21 — branch `claude/x265-x266-plans-review-h9osnl` — `dispatch_guard` + `witness_fabric::standing_wave_grounded`: the recipe grounding gate is the MULTIPASS MARKOV STANDING WAVE (operator ruling), not a coarse scalar prefilter (additive, zero-dep, edits neither peer module)

### Current Contract Inventory — new entry
- `lance_graph_contract::witness_fabric::{standing_wave_grounded, WaveGrounding}` — the multipass Markov STANDING WAVE grounding test (E-MARKOV-STANDING-WAVE-GATE-1 + E-HORIZON-NOT-BOUND-1). Runs `resolve_chain` at hop budgets `1..=passes` for one `Locus` over a `(stream_position, CausalWitnessFacet)` window; **settles inside the ±8 reference horizon** (two budgets agree, not escalated) → `WaveGrounding::Causal`; **chain leaves the horizon** → `Escalate` (the signal to search causality over time — a `temporal.rs` version-range read / the absolute AriGraph SPO+Leiden basin; NOT coincidental — a distant cause is still a cause); unbound → `Unbound`. The ±8 is the reference horizon, not a bound on awareness. Additive to the shipped `witness_fabric` (#783). +tests.
- `lance_graph_contract::dispatch_guard::{guard, GuardVerdict, GateOutcome}` — the combined recipe grounding gate composing the TWO independent organ resolutions: single-pass structural BINDING (`recipe_loci::loci_disqualifier`) ∧ multipass Markov STANDING WAVE (`witness_fabric::standing_wave_grounded`). `GateOutcome::{Fires, Escalate, Unbound}` — `Escalate` = bound but the causal chain leaves the ±8 reference horizon (a NON-LOCAL cause), the wave's INDEPENDENT catch a single-pass gate would have fired blind → the recipe escalates over time / into the absolute basin (E-HORIZON-NOT-BOUND-1). Operator ruling: the scalar `nan_disqualifier` is DROPPED as a grounding gate (a tautological subset of the organ, "a coarse filter is an insult") — retained only as `GuardVerdict::scalar_flag`, a degenerate optional sanity flag for a non-witness ctx source. Reads only PUBLIC APIs of the peer's `recipe_loci` + my `witness_fabric` — edits neither peer module. 4 tests + `examples/dispatch_guard_redundancy` (measured: single-pass fires 34/34 identically on local + beyond windows = BLIND; the wave flips 34/34 Fires→Escalate = discriminates; 4 gates green). Companion jc battery `jc/examples/rung_divergence_reliability` (Pearson/Spearman/ICC/Cronbach α on the two RUNG scales → α 0.504 DISTINCT FACETS). Zero deps. E-MARKOV-STANDING-WAVE-GATE-1 / E-SUDOKU-TISSUE-WEAVE-1 / E-HORIZON-NOT-BOUND-1.

## 2026-07-21 — branch `claude/review-claude-board-files-nhqgx1` — `recipe_loci` (Door C): the recipe dispatch gate keyed to the real 24-loci causal-witness organ + Maslow carry/prune (additive, zero-dep, nothing existing touched)

### Current Contract Inventory — new entry
- `lance_graph_contract::recipe_loci::{required_loci, loci_disqualifier, is_grounded, locus_rung, loci_rung, loci_dispatch_order, LociStep, loci_ladder, reachable, rung_level, carried_awareness, active_after_prune}` — **Door C, the 24-dimension organ gate** on recipe dispatch (closes #780 Axis B on the dispatch path). Each recipe declares which of the 16 named `causal_witness::Locus` dimensions it consumes (`required_loci`, grounded in `recipes.rs::substrate`); the rung-level walk (`loci_ladder`, in the **organ-derived** `loci_dispatch_order`) fires a recipe only when every required locus is BOUND (`loci_disqualifier` — per-specific-dimension, the 24-dim analog of `nan_disqualifier`, NOT a popcount). **Order is 24-based too:** `loci_rung(id) = max(locus_rung(l))` over the required dimensions — a recipe is as deep as its deepest organ (ICR #31 → Counterfactual apex because `Kausal`/`Contradiction` are apex dimensions), replacing the static `recipe_dispatch::rung` Tier table (kept as a documented cross-check). `rung_level` names it in the shipped `RungLevel` Maslow-pyramid vocabulary. `carried_awareness` = lower-rung groundings read UP the climb (monotone, anti-rediscovery, #777); `active_after_prune` = higher thinking subsumes lower-related (strict dimension-superset). Complements the two shipped reach-doors (A style-fan / B `select_tactic`) — neither read the organ. 11 tests + `examples/recipe_loci_walk.rs` (4 measured gates: selector 7/34, organ 34/34 when grounded, carry monotone, prune fires + apex survives). Zero deps.

## 2026-07-21 — branch `claude/x265-x266-plans-review-h9osnl` — recipes wired to real tenants: `causal_witness` (A9 24 loci) + `recipe_substrate` + `recipe_dispatch` (additive, zero-dep, nothing existing touched)

### Current Contract Inventory — new entry
- `lance_graph_contract::witness_fabric::{absolute_agreement, elect_peers, PeerElection, resolve_chain, ChainResolution, opinion_strength, is_opinion}` — the A9 witness tenant made SELF-COMPUTING (E-WITNESS-FABRIC-1, Tier-3). Functions over a slice of `(stream_position, CausalWitnessFacet)` rows (never a materialized struct): **`elect_peers`** computes the quorum/contradiction loci from the window fabric (agreement = converge on the SAME absolute event `pos_a+off_a==pos_b+off_b`, not the same offset — corrects `agrees_at`'s co-located assumption); **`resolve_chain`** follows a locus chain with a hop budget, `escalated=true` = the signal a `temporal.rs` version-range read is needed (the i4 never widens; also the seam for D-CSW-1 leg-2); **`is_opinion`** = a Contradiction locus bound across every revision (persisted-contradiction stance). +7 tests. Algebra FINDING; the real-corpus (Aesop) claim is registered CONJECTURE (no planted integers). Zero deps.
- `lance_graph_contract::dispatch_mode::{Domain, DispatchMode, Route, classify, elect_mode, route, is_ungrounded}` — the **pre-dispatch mode router** (E-DISORDER-GATE-1, Tier-1 of the expansion queue). Cynefin's mechanical core: classify the awareness state's cause↔effect relationship (Clear/Complicated/Complex/Chaotic/**Confused**) → elect a dispatch mode (Saccade=`select_tactic` / Sweep=`recipe_dispatch::ladder` / FieldGather / Stabilize). Fixes a CODE-VERIFIED DEFECT: `select_tactic` reads a NaN/ungrounded `free_energy` as the routine band (`NaN >= x` is false → the `else`); the router runs first and routes ungrounded→FieldGather. The `DkPosition::MountStupid` MUL veto downgrades Clear→Complicated (circle-of-competence at the mode level — reads MUL, doesn't rebuild it). Computed per dispatch, never stored; reads only logical markers (never qualia). +6 tests + `disorder_gate_probe` (5 gates green). Zero deps. D-DISORDER-GATE-1 / E-DISORDER-GATE-1.
- `lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, WITNESS_LOCI(24), NAMED_LOCI(16), LOCUS_LABELS, WITNESS_REGISTER_BYTES(12)}` — the **A9 CausalWitnessFacet**: the L9 `TekamoloWindowBinding` reading of the 12-byte content-blind register as **24 signed `i4` loci** (`G24N4`), each a context POINTER (`∈[−8,+7]`, `0`=unbound, sign=orientation) into the ±8 `temporal.rs` Markov window. The **THIRD ClassView reading** of the same register #729 shipped two of (FROZEN `12×u8` + Orchestration `6×(8:8)`) — a reading, NOT a layout (no stored bytes, mirrors `awareness_facet::SpoFacet`). 16 named loci (§2.9) + 8 reserved-empty (RESERVE-DON'T-RECLAIM). `get`/`set`/`with`/`at`/`is_bound`/`bound_count`/`resolves_to`/`agrees_at`/`agreement_count`/`quorum`/`contradiction`/`cause`/`antecedent`. 8 tests + doctest. Zero deps.
- `lance_graph_contract::recipe_substrate::{SubstrateView, affective_temperature, pair_similarity}` — wires the 3 real tenants (SPO `SpoFacet` + witness `CausalWitnessFacet` + qualia `QualiaI4_16D`) into the `recipe_kernels::ThoughtCtx` input via `project()`. **Qualia is ADDITIVE + STAKES-only** (temperature only, never a logical marker); **logic + causality ← SPO + 24 witness edges** (`logical_confidence`/`logical_surprise`/`logical_dissonance`/`logical_candidates`/`logical_rung`/`logical_beliefs`). A marker a MISSING tenant can't ground → **NaN/empty** (the NaN-disqualifier signal). 4 tests.
- `lance_graph_contract::recipe_dispatch::{RecipeInference, inference, rung, dispatch_order, nan_disqualifier, RecipeStep, ladder}` — the 34 recipes as a **rung-ordered, NaN-gated causal ladder** keyed by NARS inference (deduction/induction=fanout/abduction/revision/counterfactual; bridged to `nars::InferenceType` by `to_mantissa` value). `rung` = Tier base + inference escalation delta; `nan_disqualifier` = the runtime `requires()`-coverage gate (a NaN required input disqualifies); `ladder` records each step's triggering cause (Versuchsleitereffekt audit). 6 tests. Deliverable D-REC-WIRE-1; finding E-RECIPE-SUBSTRATE-WIRING-1.

## 2026-07-20 — branch `claude/ogar-docir-architecture-jjzlig` — GraphQL ergonomics as mask algebra: `selection` + `standing_mask` land in the contract (additive, zero-dep, nothing existing touched)

### Current Contract Inventory — new entry
- `lance_graph_contract::selection::{ViewId, NamedView, ViewRegistry, RailGraph, FieldVisit, walk_rails}` — nested selection WITHOUT a query document: a named view = `(ClassId, WideFieldMask, DisplayTemplate)` (a GraphQL fragment + persisted query as a mask constant; spread = `union`); `walk_rails` ANDs `view.mask ∩ present_mask` per node, emits `(key, position)` in bit order for the EXISTING `render_rows`/`facet_rows` leaf path, follows set rail-bearing bits via the dependency-inverted `RailGraph::rail_target` (rail-ness is a ClassView lens, NOT byte-derivable — verified against `CascadeShape`/`ReadMode`/le-contract §2), cycle-guarded + depth-capped. Metric-agnostic: centroid-ranked hop order composes over the facet's own distance surface. 9 tests.
- `lance_graph_contract::standing_mask::{SubscriberId, StandingInterest, fires, SubscriptionTable}` — subscriptions as standing interest masks: fires iff `dirty ∩ interest ≠ ∅` (ONE intersection per write, no re-query); generic over key; Vec+linear scan by design (consumers shard per mailbox/tenant); wide-tier (≥64) proven. 10 tests.

A third item ("prefix_select" scan pushdown) was built and then DROPPED as out-of-scope on operator correction — the address side is already the `facet::FacetCascade` algebra (`shared_prefix_tiles`/`hi_chain`/`lo_chain`); see the E-MASK-SELECTION-ALGEBRA-1 "Dropped as out-of-scope" section so it is not re-derived. Board: EPIPHANIES `E-MASK-SELECTION-ALGEBRA-1`. Integrated gate: contract lib green, clippy `-D warnings` clean, fmt clean.

## 2026-07-18 — branch `claude/happy-hamilton-0azlw4` — S07 vertical slice RUNS: `text_stream_to_soa` example (text → KG + SoA, no LLM)

Shipped `crates/lance-graph/examples/text_stream_to_soa.rs` — the end-to-end thesis as a runnable binary (COCA FSM → SPO → TripletGraph → ±5 Markov → NARS → SpoFacet 6×(8:8) → 512-B NodeRow size), one `deepnsm` dev-dep, no new primitives (3 adapter shims). Measured on Animal Farm (30k tokens, text not committed): 5,899 triples → 4,012 nodes → **1.96 MiB cold KG**, 95,410 NARS deductions + 20,131 contradictions, zero LLM. Board: EPIPHANIES `E-S07-TEXT-STREAM-NO-LLM-1`. Sample: bundled PD Aesop fables.

## 2026-07-18 — branch `claude/x265-x266-plans-review-h9osnl` (post-#733) — x265 probe wave-2 landed + bgz-tensor lane review paid

#733 MERGED: the x265 wave-2 probes — PROBE-SPRITE-REPLAY (PASS-AT-SIGNED360
scoped; ResidueEdge-24bit insufficient → Signed360-only motion primitive;
arbitrary-motion [H]) + PROBE-WH-MAG-2 (upgrades the WH-magnitude leg to
transfers-with-escape+centroid on structured tiles). EPIPHANIES
`E-X265-PROBE-WAVE-2-RESULTS`; ndarray #248 companion merged. This follow-up:
TD-BGZ-TENSOR-PRE-LANE-REVIEW **paid in part** — Opus review ruled all four
axes DEMARCATE/KEEP (nothing retired; bgz-tensor's codec occupies a lane
turbovec/PolarQuant/helix don't cover), `E-BGZ-TENSOR-LANE-REVIEW-1` + the
TD payment addendum, 4 non-gating work-items WI-1..WI-4. Remaining x265
queue: the arbitrary-motion sprite follow-up probe, PROBE-GPU-LUT (wgpu
harness). This entry + the #733 PR_ARC entry = post-merge hygiene.

## 2026-07-18 — branch `claude/happy-hamilton-0azlw4` — D-AW-2 (start): `awareness_facet::SpoFacet` reading primitive (M20 reading A1)

### Current Contract Inventory — new entry
- `lance_graph_contract::awareness_facet::{SpoFacet, Palette256Pair}` — the **A1 SpoFacet reading** of a `6×(8:8)` content-blind register (M20 assembly, `.claude/plans/soa-32-tenant-awareness-redundancy-v1.md`): six palette256² `(basin, identity)` pairs = 3 semantic-SPO + 3 episodic-witness (the operator's base design). A **reading, NOT a layout** — `from_rails`/`from_register` ↔ `to_rails`/`to_register` (loss-free), `spo()`/`witness()` splits. Reuses the shipped `soa_view::style_rails_at` convention verbatim (`rail k = (b[2k], b[2k+1])`) so it agrees pair-for-pair; touches zero bytes of the value slab. WHICH class reads its register this way is an OGAR mint (Place 2), never a base variant (operator "without hardcoding"). 6 tests + doctest. Zero in-crate deps.

## 2026-07-18 — branch `claude/review-claude-board-files-nhqgx1` (PR #729) — P4 ancestry pipeline on the SoA/ractor carrier: triangle read seam + owned columns + the 226-atom FROZEN palette256 codebook

### Current Contract Inventory — new entry
- `lance_graph_contract::soa_view::StyleLane {Frozen, Learned, Explore}` + `MailboxSoaView::{style_lane_at(row, lane) -> Option<[u8;12]>, triangle_at(row, family) -> Option<(u8,u8,u8)>}` — the deferred-binding triangle read seam (default `None`, owner overrides; `family >= 12` guard). The SoA-native replacement for the symbiont-`Vec<NodeRow>` path (symbiont deprecated; the triangle is owned by the ractor `KanbanActor`, compile-time sole-mutator E-CE64-MB-4). (P4 Brick 1.)
- `lance_graph_contract::cognitive_palette::{AtomId, AtomCatalogue, VERB/RECIPE/PERSONA/FAMILY_{COUNT,BASE}, RESERVED_BASE, ATOM_COUNT}` — the **226-atom palette256 FROZEN value codebook** (operator ruling 2026-07-18 "226 ARE the frozen"). A zero-dep ADDRESSING table (I-VSA-IDENTITIES: address only, content in registries): `0` null, `1..=144` Verb (dntree), `145..=178` Recipe, `179..=214` Persona (ThinkingStyle), `215..=226` Family (StyleFamily; local == ordinal), `227..=255` reserved (RESERVE-DON'T-RECLAIM). `resolve()` total over 256; const-asserted layout. The `12×u8` FROZEN reading of the content-blind triangle register; the LEARNED/EXPLORE orchestration reading is the `6×(8:8)` le-contract §3 register (replayable). 6 tests.

lance-graph core (non-contract, same PR — cognitive-shader-driver): three `[[u8;12];N]` triangle columns on `MailboxSoA<N>` + `style_lane_at` override + owned write ops (`set_style_lane`/`set_style_atom`/`promote_family`, `&mut self`) + `reset_row` clears them (codex #729 P2). (P4 Brick 2.)

## 2026-07-17 — branch `claude/happy-hamilton-0azlw4` (post-#714) — D-GR-1 `contract::doc_graph::{DocGraphQuery, ScoredId}` + D-GR-3b AriGraph capabilities (PPR, Leiden refinement, BM25) + G0 harness

### Current Contract Inventory — new entry
- `lance_graph_contract::doc_graph::{DocGraphQuery, ScoredId}` — zero-dep rung-aware document-graph read surface (D-GR-1). `DocGraphQuery`: `community_of` / `community_ids` / `community_members` / `neighbours` / `similar_by_ranking`, plus a provided `retrieve(seeds, RungLevel, top_k)` default carrying the rung→walk dispatch (0–1 ranking / 2 SPO-G hop / 3+ wider community-scoped walk). `ScoredId {id, score, depth}`. Impl target = AriGraph `OsintRetriever` (D-GR-2, gated on G0). 9 tests. Only in-crate dep is `cognitive_shader::RungLevel`.

lance-graph core (non-contract, same PR): `graph::arigraph::{ppr::{PersonalizedPageRank, personalized_pagerank}, bm25::Bm25Index}`; `community::refine_connected` (Leiden connectivity); `examples/g0_graph_loadbearing.rs` (P-GRAPH-LOADBEARING scaffold).

# LATEST_STATE — What Just Shipped (read this FIRST)

## 2026-07-17 — branch `claude/review-claude-board-files-nhqgx1` — D-TRI-1 value-tenant half: the autopoiesis triangle lands as 3 SoA lanes

**MERGED #717 (2026-07-17, main `74d16f92`).** The value-tenant half of the
D-TRI-1 batched mint (`triangle-tenants-gestalt-separation-v1.md` §1). Three NEW
append-only `ValueTenant`s — `FrozenStyle = 10`, `LearnedStyle = 11`, `ExploreStyle = 12`
— each `ColumnKind::U8 × 12` (12 palette256 atoms), contiguous at `row_offset`
152 / 164 / 176 (value-slab `[120,156)`), appended **after Kanban** per the
2026-07-17 operator ruling ("triangle right after the kanban board").
**Additive, reserve-don't-reclaim, layout-preserving** (Full carve 120→156 B,
`NODE_ROW_STRIDE` 512 untouched, no `ENVELOPE_LAYOUT_VERSION` bump). Added to
`ValueSchema::Full` **only** — deliberately NOT `Cognitive`, so entity classes
(OSINT/PROJECT/ERP/Commerce, which resolve to Cognitive) do **not** inherit a
thinking-style triangle; the thinking-row schema decision is deferred to P4
(ancestry pipeline). Slot `f` = `StyleFamily` ordinal 0..11 **or** compiled-
template step 0..11 (one content-blind register, ClassView-selected reading,
plan §4). **Atom 0 = null default** (zero-fallback: an un-populated lane reads
all-null, never a wrong policy). Accessors `NodeRow::{style_lane, set_style_lane,
triangle_for}` (`triangle_for(f) -> (frozen[f], learned[f], explore[f])`, the
one-glance per-family read the dispatch/perturbation/learning ancestry pipeline
resolves against; `set_style_lane` is the owner-`&mut` write that resolves the
W4b orphan-write flag). **The I-LEGACY field-isolation matrix**
(`thinking_style_triangle_tenant_carve_field_isolation_matrix`) certifies each
lane's 12 bytes flip in isolation, key/edges/other-tenant untouched.
**906 contract lib tests green; clippy `-D warnings` + fmt clean.**
**Still queued (the classid half — cross-repo OGAR mint):** chess domain 0x06
concepts, the Tasks-SoA task-row classid (cognitive-task concepts), and
BoardAggregates @ row_offset 188 — one batched OGAR-originated mint (never solo).
Refs: STATUS_BOARD D-TRI-1, plan §1/§5/§6, v3-envelope-auditor gate.

## 2026-07-16 — branch `claude/x265-x266-plans-review-h9osnl` (v5, post-#702) — probe-wave verdicts on main

#702 MERGED (autonomous merge under the operator-granted authority):
the h268-probe-wave — PROBE-WH-MAG (NEUTRAL, bare-tile WH leg closed
not-transferring), PROBE-SIG-CHECKSUM (PASS, depth-2 parallel-chord
blind-spot bound), PROBE-WALK-SPECTRUM (KILL of §10(g)'s "decorrelated
by construction"; period-17 structure confirmed; D-QUANTGATE
unaffected). Probes live in bgz-tensor/jc/helix; verdicts canonical in
EPIPHANIES `E-H268-PROBE-WAVE-1-RESULTS` + plan `h268-probe-wave-v1.md`.
Companion ndarray #246 MERGED same day. This entry + the #702 PR_ARC
entry are the post-merge hygiene. Next plateau (same goal):
TD-BGZ-TENSOR-PRE-LANE-REVIEW.

## 2026-07-16 — branch `claude/x265-x266-plans-review-h9osnl` (v4, post-#699) — Hadamard-residual-ladder honourable mention on the V3 substrate

Doc-only. #698 and #699 MERGED same day; this follow-up adds: the
le-contract §3 "Honourable mention" subsection (bgz-tensor's index +
Hadamard-residual ladder as the fourth operational mode next to the three
flavours of 256, out-of-row, with the Hambly–Lyons 2010 signature theorem
as the formal anchor — already in-workspace as jc Pillar 11 +
ndarray hpc/pillar/signature.rs B7; only the ladder→signature mapping
stays [S]), EPIPHANIES `E-PALETTE-RESIDUAL-LADDER-1`, and
the CLAUDE.md stale "bgz-tensor 0 deps" row correction (it path-deps
ndarray + holograph). No layout added to the §3 catalogue; no contract
types touched.

## 2026-07-16 — branch `claude/x265-x266-plans-review-h9osnl` (v3, post-#697) — comma-closure/96-bit facet/replayable-tile addendum

Doc-only. #697 (PROBE-GPU-LUT oracle spec pinned) MERGED same day; this
follow-up branch adds: EPIPHANIES `E-H268-REPLAYABLE-TILE-1` (prepended —
the replayable Morton 2bit×2bit 4×4 tile serving both H.268 mode-decision
and cognitive-shader dispatch, plus the D-QUANTGATE rationale restatement),
the capstone addendum pointer (§7-§10 of the ndarray matrix doc), and the
#697 post-merge PR_ARC_INVENTORY entry. No contract types touched.
Companion ndarray branch carries the matrix-doc §7-§10 addendum itself.

## 2026-07-16 — branch `claude/x265-x266-plans-review-h9osnl` (v2, post-#695) — H.268 codename + graded Morton/wgpu synergy matrix pointer

Doc-only. #695 (standards-watch + E-PRX12-STANDARDS-GROUNDING-1) MERGED same day; this follow-up branch (restarted from main per merged-branch rule) adds: the **H.268 internal codename** ruling for the ex-"x266" PR-X12 3DGS scene codec (INTERNAL ONLY, never an ITU designation), the capstone pointer to ndarray's adversarially-graded feasibility matrix (`pr-x12-h268-morton-wgpu-synergies.md` — 1× FEASIBLE-NOW / 2× NEEDS-PROBE / 7× OVERCLAIM-CORRECTED, incl. bgz17's 256×256 distance table being texture-isomorphic today with PROBE-GPU-LUT as the named gate), EPIPHANIES `E-H268-GRADED-SYNERGY-1`, and the #695 post-merge PR_ARC entry. No contract types touched.

## 2026-07-14 — branch `claude/review-medcare-rust-dt7MS` — `class_view::execute_defaults` + `ClassView::default_targets` — the Default-recipe half of the ActionDef value executor (lane-3a inc 2)

### Current Contract Inventory — new entry

- **`class_view::execute_defaults(targets, present, store, apply_default)`** (NEW free fn next to `execute_compute_dag`) + **`ClassView::default_targets(class) -> &[u8]`** (NEW default trait method, zero-fallback `&[]`, next to `compute_dag`). The **Default-recipe** (write-if-blank, `RecipeCentroid::Default` — the C# `if (field == null) field = new …` lazy-init idiom, 56 methods in the MedCare corpus) execution primitive: fires `apply_default(store, t)` for each target whose presence bit is CLEAR, in slice order; skips present AND already-fired positions (duplicate-safe — after a default fires the field IS populated; pins the C# `GetOrCreateChartPanel` init-only-on-create quirk, devcomponent_chart.cs:161-180); abort-at-target reusing `ExecuteComputeError::Compute` (`Cyclic` unreachable — defaults have no dependency order; a default reading a computed field is a Compute recipe). **Presence gate = `WideFieldMask`** (not `FieldMask`): every u8 position addressable, no 64-field ceiling — the wide-mask lesson the a2ui screen-addressing #205 correction paid for, applied at birth. Phase rule documented (not interleaved): defaults run BEFORE `execute_compute_dag`; the caller folds the fired list into its presence mask. One more brick, not a parallel path (the `screens_reachable_from` precedent): existing mask, existing error, same abort semantics as the Compute half. +7 unit tests (slice-order fire, present-skip, abort-keeps-earlier, empty-noop, wide-fire-past-64, duplicate-fires-once, hook-default-empty); 898 contract tests green; fmt + clippy clean (also carries the pending rustfmt reflow on `grammar/thinking_styles.rs` + `style_family.rs` — main was fmt-dirty on those two test files). Consumer witness lands in MedCare-rs (`chart_default_parity`, same arc).

## 2026-07-14 — MERGED #689 (merge `a260ff98`) — rung-content ladder ruled + persona demarcated + `contract::legacy_outliers` + `NodeGuid↔FacetCascade` bridge + D-TSC-1 alias fix

- **Ruling doc (MANDATORY read, linked from CLAUDE.md):** `.claude/v3/knowledge/persona-vs-rung-ladder.md` — rung 0–1 observation / 2 = 144 verb atoms / 3 = the 34 tactic recipes (= THE runbooks: `contract::recipes` + ndarray `hpc/styles`) / 4 = StyleFamily macros (frozen×learned×exploration). The adjective-36 = separate unwired persona storyline, never rung 3. Open items O1–O6 live there.
- **Contract inventory adds:** `legacy_outliers::LegacyOutlier` (G1/G2/G3 grace carvings, le-contract §3a); `From<FacetCascade> for NodeGuid` + inverse + `NodeGuid::facet()` (byte-identical, no registry detour).
- **Planner:** `thinking::ThinkingStyle` deprecated alias + `PlannerStyleExt` re-exported again (old import path compiles, deprecation warning steers).
- **Probes flipped:** TD-STYLE-TABLE-RESIDUE item 3 → DORMANT; UNIFIED_STYLES tethered-not-collapsed.
- **Companion:** OGAR #201 (D‑V1‑GRACE‑CARVINGS canon mirror, merged).

## 2026-07-12 — branch `claude/review-claude-board-files-nhqgx1` — `contract::temporal_pov::{VersionRange, TemporalPov}` — zero-dep temporal POV range filter (operator-directed)

### Current Contract Inventory — new entry

- **`temporal_pov::{VersionRange, TemporalPov}`** (NEW; one zero-dep module, additive, re-exported nowhere new — accessed via `lance_graph_contract::temporal_pov::*`). Operator directive (2026-07-12): "add a time range filter to lance-graph-contract for temporal POV using our temporal.rs research." **`VersionRange { from, to }`** is a plain half-open `[from, to)` interval over `LanceVersion` (`u64`, zero-dep mirror of `lance_graph_planner::temporal::LanceVersion`) with `contains`/`intersect`/`is_empty`/`len`/`full()` (the `[0, u64::MAX)` "latest" window). **`TemporalPov { range, rung }`** pairs the range with the reader's `rung: u8` (mirrors `QueryReference::rung`, `crates/lance-graph-planner/src/temporal.rs:122-124`) and provides `TemporalPov::at(ref_version, rung)` (mirrors `QueryReference::at`, temporal.rs:139-151 — pins the contemporary window `row_version <= ref_version` as `[0, ref_version+1)`) + `admits(version)` (the version-range half of admission only, quoting `EpistemicMode::Strict`'s doc temporal.rs:53-55). **Deliberately does NOT reimplement** `EpistemicMode`/`TemporalStatus`/`classify`/`deinterlace` — those stay in `lance-graph-planner` (downstream of this crate; re-deriving them here would be exactly the duplication this file's Type Duplication table warns against). Instantiates `E-MARKOV-TEMPORAL-STREAM-1`'s "version-range read generalizes the VSA ±5 braid" ruling and its measured worked example `D-SF-EPISODIC-1` (`stockfish-rs` — position-at-ply-*v* as a zero-copy `QueryReference::at`+deinterlace projection, 34/34 + 11/11 GREEN). +11 unit tests (range contains/empty/full/intersect incl. touching-ranges-empty; POV admits/at/rung-boundary-roundtrip/latest-sentinel edge case) + 1 doctest; `cargo test -p lance-graph-contract temporal_pov` and `cargo clippy -p lance-graph-contract -- -D warnings` both clean (scoped, no `--workspace`).

## 2026-07-11 — branch `claude/medcare-ruff-codebook-handover-5ulx0i` — `ClassView::menu_address` — the runtime Klickwege-menu radix projection

### Current Contract Inventory — new entry

- **`class_view::ClassView::menu_address(class) -> Vec<ClassId>`** (NEW default trait method). The RUNTIME projection of the harvest-side ruff `nav_digest` `[menu-quad]` `loc=` field: walks [`is_a_parent`] root-first to lower a class's menu LOCATION into the existing classid ontology as a radix-trie path `[root, …, parent, class]` — the concept ontology **is** the radix trie; the menu address is a path through it, never a stored ordinal (V3 LE-contract §3). A renderer lays out the menu by prefix from the path alone, zero value decode. Same 16-hop cap + on-stack visited cycle guard as the sibling `resolve_render_class` (never loops/panics; only alloc is the returned path). Default method → every existing `ClassView` impl gains it for free. Tests: `menu_address_walks_is_a_root_first` (30 is_a 20 is_a 7 → `[7,20,30]`; root → `[7]`), `menu_address_cycle_terminates` (2-cycle + self-loop bounded). Completes the "digest-now-ClassView-after" plan (ruff #82 = digest lowering; this = runtime projection). No new type/module; pure additive trait surface.

## 2026-07-10 — branch `claude/review-claude-board-files-nhqgx1` — `contract::style_family::StyleFamily` — M9 ThinkingStyle dedup shipped (D-TSC-1, first 5+3 council run)

### Current Contract Inventory — new entry

- **`style_family::StyleFamily`** (NEW; zero-dep `#[repr(u8)]` 12-variant enum, re-exported from `lib.rs`). The 12 abstract orchestration FAMILIES per `E-STYLE-FAMILY-VS-RUNBOOK-1` (12 = families; 36 `thinking::ThinkingStyle` = literal NARS runbooks → rung ladder / rs-graph-llm replayable chaining unit). Ordinals FROZEN to the driver `UNIFIED_STYLES` order (Deliberate=0…Metacognitive=11, discriminant-pinned). Surface: `ALL`, `name()` (= deepnsm YAML card names), `from_name`, `from_ordinal`, `default_runbook() -> ThinkingStyle` and `ThinkingStyle::family()` (total; round-trip `f.default_runbook().family()==f` pinned), `Display`. **Replaces FIVE divergent hand-rolled style tables**: planner `planner_style_to_contract` (drifted at cells 9/10/11), driver `ord_to_thinking_style` (8/9/10), contract `parse_style_name` (8/9/10 — caught by the council's overclaim reviewer as a Phase-3 BLOCK-P0), the `THINKING_RECONCILIATION.md` exemplars, and thinking-engine `contract_style_to_engine`'s 36→12 ordinal ranges. Consumers migrated same-commit: planner `thinking/style.rs` = re-export + deprecated alias + `PlannerStyleExt` (cluster/τ/modulation); thinking-engine `cognitive_stack.rs` = re-export + deprecated alias + `EngineStyleExt` (params/butterfly/all) + NEW `lance-graph-contract` path dep; `superposition.rs` enum renamed `DetectedStyle` (detection RESULT, not a card); driver keyed by `StyleFamily::from_ordinal` + G3 parity test; `nars_engine::style_vector_for` runbook-keyed accessor. Gates: G1 grep = 1 enum + 3 deprecated aliases; tests 874+212+362+101 = **1549 green**; no new clippy warnings; fmt clean. Behavior changes (documented, G7-pinned): planner arms 9/10/11, driver arms 8/9/10 (awareness bootstrap), parse arms diffuse/peripheral/intuitive, engine 36→12 ranges → canonical `family()`. Spec: `.claude/plans/dtsc1-thinkingstyle-dedup-spec-v1.md` (v3 ratified; 5+3 council per `.claude/agents/5plus3-council.md`).

> **2026-07-07 — NO-PIN + plug-and-play OGAR arming:** all OGAR deps
> (symbiont / lance-graph-ogar / cognitive-stack) switched from
> `git+branch=main` (which always pins a rev in Cargo.lock) to PATH deps on
> the local sibling `/home/user/OGAR` — operator policy: "wenn's knallt,
> dann einmal — nicht 200 Pins monitoren". `lance-graph-contract` gained the
> optional `ogar` feature (`ogar_codebook::armed`): arming pulls `ogar-vocab`
> transitively and activates a contract-side compile-time `COUNT_FUSE`
> (+ per-entry parity test). lance-graph-ogar arms it; the contract stays
> zero-dep by default. Fuse verified green vs OGAR main `68d85f02` (84==84,
> entry-for-entry).

> **Auto-injected at session start via SessionStart hook.**
> Updated after every merged PR.
> **Last updated:** 2026-05-14 (PR #372 merged: sprint-10 spec sprint, 12-worker CCA2A fleet + Opus meta-review + 8 knowledge docs, governance-only (zero .rs changes), mirrors PR #365 pattern. Sprint-11 implementation wave gated on 5 spec patches + 4 user ratifications: CSI-1 CausalEdge64 bit-reclaim Option, OQ-1 Σ4-Σ5 banding, OQ-3 plasticity granularity, OQ-5 rayon vendor. **Major findings:** (1) dual `CausalEdge64` types in workspace — `causal_edge::CausalEdge64` SPO-palette layout ≠ `thinking_engine::layered::CausalEdge64` 8-channel cascade, same name different semantics, surfaced as duplication entry #13 in TYPE_DUPLICATION_MAP and E-META-7 in EPIPHANIES; (2) p64 drift origin pinpointed at `crates/lance-graph-planner/src/cache/convergence.rs:18-22 #[allow(unused_imports)]` annotation — wiring intended for hot-path convergence never finished; (3) three-zone hot-path mental model corrects prior framing — Zone-1 thinking-engine MatVec 200-500ns + AriGraph entity_index O(1) ~20-200ns is the actual cycle-speed path, not DataFusion. Prior: 2026-05-13 (PR #366 merged: sprint-7 7-worker implementation wave for the sprint-5/6 specs + AuditSink trait unification, ~5 KLOC across 5 crates +2 new (`lance-graph-supervisor`, `lance-graph-consumer-conformance`), ~70 new tests, workspace clippy --tests --no-deps -D warnings exits 0; Opus meta verdict 4A/2B/1B-minus; OQ-7-1/2/3 all locked pre-merge; `UnifiedAuditSink` D-SDR-4 placeholder dropped, all sinks unified on `AuditSink` trait; `UnifiedBridge::with_jsonl_audit()` ergonomic constructor added for MedCare-rs sprint-2 item 5. **Adjacent landings (same day):** MedCare-rs sprint-1 10-PR sweep (#113-#122) including E1-1 OQ-3 direct migration (6 RoleGroups) consuming our `0d725d4` decision. MedCare-rs sprint-2 (5 PRs) is queued on user "go" — item 5 consumes this PR's new constructor. Prior same-day: PR #365 (13 sprint-5/6 specs + meta). Prior: PR #364 (D-SDR-3/4/5 + sprint-log-4 governance + sprint-5-9 roadmap + codex P1/P2 fixes). lance-graph #364 ships D-SDR-3/4/5 + sprint-log-4 governance + sprint-5-9 roadmap + codex P1/P2 surgical fixes (OwlIdentity 3-byte canonical, UnifiedAuditEvent 26 bytes, OgitFamilyTable sparse `HashMap<u16, FamilyEntry>`, audit super_domain via AuditChain). MedCare-rs#112 (PR-B) wires `UnifiedBridge<MedcareBridge>` + medcare-rbac + medcare-realtime substrate (+2963 LOC, 17 files, §73 SGB V + BMV-Ä §57 + BtM regulatory tests). smb-office-rs#31 (PR-C) wires `UnifiedBridge<OgitBridge>` (+111 LOC). ndarray#142 ships VBMI gate for `permute_bytes` (P0 SIGILL fix on Skylake-X / Cascade Lake / Ice Lake-SP) + Inf clamp for `simd_exp_f32`. D-SDR-5 `UnifiedBridge` surface is now consumed end-to-end across MedCare + smb-office. Prior: 2026-05-07 (PR #354). Prior: 2026-05-07 (PR #353). Prior: 2026-05-07 (PR #352). Prior: 2026-05-06 (splat-osint-ingestion-v1 PR 1+2 of 6 in flight). Prior: 2026-04-21 post PR #243.
>
> Purpose: prevent new sessions from hallucinating structure that
> already exists or proposing features already shipped. Read this
> BEFORE proposing any grammar/crystal/contract changes.

---

## 2026-07-10 — branch `claude/medcare-ruff-codebook-handover-5ulx0i` — `contract::ogar_codebook` +`external_practice` (`0x090C`, round-4b Health mint)

Paired mirror of OGAR `13e1b0f` (COUNT_FUSE 88->89, `lance-graph-ogar` compile-time equality green). **NEW row:** `external_practice 0x090C` — a referral-partner organization (FHIR `Organization`), the round-4b furnace mint. Hardened by a staged 5+3 council **and** an operator-directed three-axis grounding gate (method + storage + navigation-structure witnesses all present -> grounded [G]); the round's second candidate was refused a mint by the same gate for lacking a navigational home. No OGIT entity -> no port alias. **Verified:** `cargo test --manifest-path crates/lance-graph-ogar/Cargo.toml` (fuse green, 62+ tests), `cargo test -p lance-graph-contract` (840 lib tests). Consumer side: medcare-analytics extends `MINTED_UNSERVED_HEALTH_CONCEPTS` with `0x090C`. Merge train: OGAR -> this mirror -> medcare gate, back-to-back (W1 fuse window universal, W2 medcare-gate window present).

## 2026-07-10 — branch `claude/medcare-ruff-codebook-handover-5ulx0i` — `contract::ogar_codebook` synced to the OGAR round-2 Health mints (`0x0908..0x090B`)

Paired mirror of OGAR `2c8836f` (two-sided COUNT_FUSE: `lance-graph-ogar` compile-time assert now 88 == 88). **NEW rows:** `anamnesis 0x0908` / `investigation 0x0909` / `examination 0x090A` / `practitioner 0x090B` — harvest-derived mints surfaced by the MedCare transcode furnace exam's slag ledger (council-hardened 5+3, spec in the consumer repo), NO OGIT entity → no port alias; the 0x09 section comment now names the two provenance classes (7 OGIT-promoted + 4 harvest-derived). **Verified:** `cargo test --manifest-path crates/lance-graph-ogar/Cargo.toml` (COUNT_FUSE green, 62+ tests), `cargo test -p lance-graph-contract` (840 lib tests green). Consumer side: medcare-analytics lands the `MINTED_UNSERVED_HEALTH_CONCEPTS` fail-closed exemption ledger in the same window (its RLS boot gate derives coverage from `concepts_in_domain(Health)`). Merge train: OGAR first, this mirror immediately after, medcare gate immediately after that.
## 2026-07-10 — branch `claude/review-claude-board-files-nhqgx1` — W2d `elevation::cycle::CycleBudget` + W4a `MailboxSoA::cast_on_behalf` (the M12 allocator + the write-on-behalf pairing at the carrier)

- **W2d / M12** — `lance-graph-planner::elevation::cycle::CycleBudget`: the ONE per-cycle budget allocator. Reads the Libet anchor from the stamped Σ-commit `KanbanMove` (`from_move`; parity test pins `LIBET_CYCLE_BUDGET_US = 550_000` against the REAL `NextPhaseScheduler` stamp), carves every per-strategy `PatienceBudget` from the cycle remainder (`slice_for` — extend-don't-shadow), advisory `admits` (updates reprioritize, never gate), measured consts with provenance (66 µs/card lane-E t2; ~0.5 µs/step Addendum-5). Doc cross-refs both ways (budget.rs ↔ cycle.rs ↔ contract scheduler.rs). Planner lib 209 green. M12 → IN-FLIGHT.
- **W4a** — `cognitive-shader-driver::MailboxSoA::cast_on_behalf<P>` (feature `with-planner`): the write-on-behalf cast pairing ON the carrier — `on_behalf` is read from `self.mailbox_id()`, so a call site cannot mispair owner and payload; payload-generic per the writer's DTO purity, `BusDto` is the canonical rung-B payload (verified ownership-field-free, warden check 2). + `BatchWriter::on_behalf_of(cast)` delegation-audit getter (planner). 3 tests green incl. the literal `BusDto` arm (`with-planner,with-engine`) + stacked-casts never-refused. **Fixed pre-existing:** standalone `with-planner` failed E0432 (`planner_bridge` imports lab-only `wire`); the module is now gated `all(with-planner, any(serve, grpc))` matching its LAB-ONLY role — `with-planner` alone now means "planner dep available". ractor note per operator: ownership is a compile-time delegation declaration, never a message path — the cast is a WAL report from within the owner's context.

## 2026-07-10 — branch `claude/review-claude-board-files-nhqgx1` — `contract::step_mask::StepMask` — the compiled-template live-step selector (D-V3-W3a)

### Current Contract Inventory — new entry

- **`step_mask::StepMask`** (NEW; one zero-dep `u64` newtype module, additive, re-exported from `lib.rs`). The thinking sibling of `class_view::FieldMask` per the V3 compiled-templates ruling (`E-COMPILED-THINKING-TEMPLATES`): `askama ↔ ClassView × FieldMask :: elixir DSL ↔ Template × StepMask`. Bit `N` = the `N`-th step of a compiled template's ordered step list is LIVE for the current style/dispatch; positions are stable + append-only (the N3 rule — retire by template `version` bump, never by bit reuse). **Selection, NEVER control flow** — the module doc pins the 2026-07-02 ground-truth correction (`Step ↔ graph_flow::Task`, `ogar_name() ↔ Task::id()`; GoTo/End/WaitForInput belong to a future `ControlSignal` surface, not to mask bits). API mirrors `FieldMask` verbatim (`EMPTY`/`FULL`/`MAX_STEPS=64`/`from_positions` with the ignore-never-fold ≥64 rule/`with`/`is_live`/`count`/`intersect`/`union`/`is_disjoint`) plus three template-shaped additions: `without` (mask a step off — skipped, not awaited, per the standing-async-plan ruling: a kanban update reprioritizes the live set, it never gates the cycle), `full_for(step_count)` (live-set default, saturating at 64 — a >64-step template is a split signal, mirroring `WideFieldMask::full_for`), and `next_live(from)` (the executor's O(1) trailing-zeros ordered-walk primitive). +5 tests incl. the style-lens ∩ board-admission compose and the ordered-walk skip; contract lib **866** green (861+5); clippy `--all-targets -D warnings` clean; fmt clean. Consumer: D-V3-W3b's ElixirTemplate → graph-flow adapter judges its live set with this mask; W3d catalogue keying stays internal (P4-gated). STATUS_BOARD D-V3-W3a flipped same-commit.

## 2026-07-08 — branch `claude/medcare-rs-transcode-ruff-3y2olh` — `contract::class_view`: the JUMP connector — `screens_reachable_from` + `nav_is_fully_connected` (Klickweg reachability, cycles allowed)

### Current Contract Inventory — new entry

- **`class_view::{execute_compute_dag, ExecuteComputeError}`** (NEW; one free fn + its error enum, additive, zero new deps). The ORDER half of the ActionDef value executor (medcare transpile lane-3a increment 1): runs a class's recompute DAG over a consumer-owned store in `compute_dag_topo_order` sequence — Cyclic refused before any target runs (store untouched), consumer compute errors abort at the failing target (earlier targets stay written, later never run — matches cycle-aware `write_row` gating). Value semantics stay consumer-side as the `compute_target` closure (commitment: thinking upstream, values with the row owner; medcare's hand-ported sono `Recal_*` family is the designated first parity witness). Builds directly on #539's `ComputeEdge`/`compute_dag_topo_order`. +4 class_view tests (in-order chain w/ precedent visibility; cycle-refusal-untouched-store; abort-at-failing-target; empty-noop); class_view 41/41.
- **`class_view::{screens_reachable_from, nav_is_fully_connected}`** (NEW; two free fns, additive, zero new deps, zero signature changes). The **jump** half of the stackable-topology Lego kit — the `navigates_to` navigation graph (screens = `ClassView` nodes, clicks = edges). Reuses `ComputeEdge` as the edge *representation* (`target` = destination screen, `inputs` = source screens) and returns a plain `WideFieldMask` of reached positions — **one more brick, not a parallel path**. The invariant deliberately differs from the compute DAG: **stack** composition (a screen composes sub-views, in-family) is acyclic and reuses `compute_dag_is_acyclic` (#539) verbatim, with the stacked-fields mask minted via the sanctioned `WideFieldMask::from_universe_present(basis, skin)` brick (#669) so it is interchangeable across consumers + carries the 256-SoC guard; **jump** navigation (out-of-family) is a **reachability closure with cycles ALLOWED** (`A→B→A` = ordinary back-navigation), so `screens_reachable_from` is a forward `WideFieldMask` fixpoint (cycle-safe: the reached set only grows) and `nav_is_fully_connected(root, edges, screens)` = `screens ⊆ reached` (no orphan lane — the level-editor/Mario-editor validator at the Core layer; `screens` itself minted via `from_universe_present`). Paired with the ruff harvest half (`ruff_spo_triplet::Predicate::NavigatesTo` + `ruff_csharp_spo::EmitNavArm`, verified end-to-end on a synthetic WinForms fixture). +3 class_view tests (forward closure; cycle-does-not-reject-connectivity; orphan-fails-connectivity); class_view 33/33, contract lib green; clippy `-D warnings` clean. Consumer: medcare-rs Klickweg 1:1 parity check (`screens_reachable_from` over the harvested `navigates_to` edges vs the rendered nav graph). See EPIPHANIES `E-KLICKWEG-1`.

## 2026-07-06 — MERGED #651 (merge `16c9e0c`) — `contract::class_view::WideFieldMask`: backward-compatible >64-field masks (canonical form, repr-independent Eq/Hash)

(Post-merge inventory entry, L-1 of the criticals wave — operator ruling (c), 2026-07-06. Also records: the `claude/classview-unified-render` work in the entry below MERGED as **PR #650**, merge `598f872` — its "not yet a PR" line is superseded.)

### Current Contract Inventory — new entry

- **`class_view::WideFieldMask`** (NEW, additive sibling type; **`FieldMask` is byte-for-byte untouched** — `Copy`/const-fn/u64 semantics intact; the enum-repr-inside-`FieldMask` alternative was evaluated and REJECTED because `ClassProjection::next`/`from_positions` rely on `FieldMask: Copy`). Widens the field-mask ceiling past 64 **without touching the existing type**: `Repr::Small(u64)` (bit-identical to `FieldMask`, allocation-free) promotes once to `Repr::Wide(Box<[u64]>)` at position ≥ 64; bit N = the same logical N3 field across both reprs; lossless `From<FieldMask>`, deliberately NO lossy reverse (a fallible `TryFrom` for the ≤64 case is the named follow-up for RBAC `PermissionSpec::projection`). **Canonical form (V-L P0, found + fixed pre-merge):** `intersect`/`union` trim trailing zero chunks and demote to `Small` when they fit; `PartialEq`/`Eq`/`Hash` are hand-written over a trimmed chunk view, so **semantically equal masks are equal and hash identically regardless of representation** (the adversarial review reproduced `a.intersect(&b) != from_positions(same set)` before merge; regression test pins it). No version split needed (`0x1000→0x1001` reserved as last resort, unused); zero-dep promise held (std only). **Unblocks X7/F14** — `account.move` has 109 declared fields; everything past bit 63 previously dropped silently. Doctrine guard: `FIELD_MASK_CAP = MAX_SIBLINGS_PER_TIER` (256) still caps meaningful masks — a ≥256-field class is an `OGAR-SOC` split signal, NOT a mask-widening use case; `WideFieldMask::full_for(field_count)` is the class-conditioned shape the OGAR bitmask doc always named as the eventual expansion. Verification: contract 829 green (+7 tests: u64-pin, >64 representable, no-alloc small path, cross-tier intersect/union, full_for, lossless promote, canonical-form regression); clippy `-D warnings` clean; consumers (`lance-graph-rbac`, `lance-graph-ontology`, arm-discovery) green with ZERO source changes. Cross-repo: OGAR interim loud-fail guard in `ogar-render-askama` (>64 fields + partial `FieldMask` → `Err` instead of silent drop) ships separately in the criticals wave; OGAR `WideFieldMask` adoption follows as O-2-adjacent work.

## 2026-07-06 — branch `claude/classview-unified-render` — `contract::class_view` unified ClassView render: facet value rows + is_a-walk resolution

(Per APPEND-ONLY rule: new top-of-inventory entry. Branch work, not yet a PR — records the contract types so a new session does not re-derive them. Operator-ratified this session: any app calls the SAME unified projection — resolve (is_a rail) → carve (FieldMask) → emit — with routes as thin per-app skins.)

### Current Contract Inventory — new entry

- **`class_view::ValueRow<'a>` + `ClassView::facet_rows` + `ClassView::is_a_parent` + `ClassView::resolve_render_class`** (NEW; `ValueRow` re-exported from `lib.rs`; the three methods are DEFAULTED/provided → additive, zero signature changes, zero new deps). Closes the two gaps the q2 cockpit-server OSINT card surfaced in the contract: (a) a **VALUE column** and (b) **is_a-walk fallback** so every classid renders. **`ValueRow` = the value-projected sibling of `RenderRow`** — `{ label, predicate, position: u8, value: u8 }`: one populated field paired with the byte it reads from the node's V3 content-blind 12-byte facet payload. **Position `i` binds to facet byte `i`** per `.claude/v3/soa_layout/le-contract.md` §3 (the 12 B is a *dumb byte register the ClassView projects*; the class picks the reading — `6×(u8:u8)` rails / `4×(u8:u8:u8)` SPO / `3×(u8:u8:u8:u8)` quads all read off the SAME positional bytes). **`facet_rows(class, mask, &[u8;12]) -> Vec<ValueRow>`** iterates fields in class order, emitting a row for each position that is BOTH mask-present (C2 presence) AND `< 12` (facet-backed); **positions `>= 12` are value-slab fields, out of facet scope — skipped, never folded** onto byte `i & 11` (mirrors the `FieldMask::MAX_FIELDS` 64-guard). **`resolve_render_class`** walks `is_a_parent` (default `None` = no taxonomy) while `fields(current).is_empty()`, returning the first ancestor with a non-empty field set, else the ORIGINAL class — the **monotonic zero-fallback ladder: bespoke card → nearest ancestor's card → caller renders the generic facet dump**. Cycle- and depth-safe, zero-dep: a hard 16-hop cap + an on-stack `[ClassId;16]` visited array reject a `subClassOf` cycle (`A is_a B is_a A`) with no `HashSet`; either guard → return original (never loops, never panics). Presence-only C2 preserved throughout: the mask gates which rows exist, never what a byte means. +7 tests (facet position/byte binding + off-bit + `>=12` skip; FULL-mask 3-field; is_a→ancestor; orphan→self; 2-cycle + self-loop terminate; 20-parent depth-cap; render_rows regression canary) — class_view 21/21, contract lib 822 green; `cargo fmt`/`clippy -p lance-graph-contract --all-targets -D warnings` clean. Consumer note (Wave 2, q2 cockpit): route `resolve_render_class(classid)` first, then `facet_rows` on the resolved class for the value column; an empty `fields()` on the resolved class is the signal to render the raw facet dump.

## 2026-07-04 — branch `claude/happy-hamilton-0azlw4` — `contract::network` — the Tesseract `Network` layer graph sunk onto V3 SoA via ruff→OGAR (byte-parity vs libtesseract)

**NEW** `lance_graph_contract::network`: `NetworkType` (27 layer types, ordinal == on-wire `kTypeNames` discriminant) + `NetworkHeader` (`from_le_bytes` = the base header `Network::CreateFromFile` reads before subclass dispatch: `i8 tag | u32+str type_name | i8 training | i8 needs_backprop | i32 flags | i32 ni | i32 no | i32 num_weights | u32+str name`) + `to_facet()` (the V3 SoA sink) + `NetworkType::classid()` (the `invoke_network` dispatch seed). Executes the operator directive *"6x8:8, 16 B tenant = classid + 12 B, ruff>OGAR sink-in"*: (1) the `ruff_cpp_spo` `harvest_network` example (committed to ruff) walks the 11 network headers via libclang → the `has_function`/`virtually_overrides` SPO manifest (62 classes, 5060 triples) = the `classid → ClassView` method-resolution table, NOT a hand-rolled enum; (2) each node sinks onto `crate::facet::FacetCascade` (16 B = `classid(4) | 6×(8:8)`, read `CascadeShape::G6D2`): tier0=ni, tier1=no, tier2=flags, tiers3-4=num_weights u32, tier5=lifecycle; `facet_classid = compose_classid(network_layer=0x0804, ntype)` canon-high. Byte-parity **GREEN** on real `/tmp/eng.lstm`: Rust parse == libtesseract `Network::CreateFromFile` — `Series ni=36 no=111 num_weights=385807 name=Series` — oracle `spec()` == the model spec string (known-answer self-check, 5.5.0-hdr/5.3.4-lib ABI skew guarded). Example `network_dump.rs`; +5 contract tests; clippy `-D warnings` + fmt clean (scoped `-p lance-graph-contract`). ONE `network_layer`=0x0804 OCR-domain mint added (subclasses in classid custom-low, not 27 slots). Deferred: per-subclass payload + tree recursion, the `invoke_network` keystone, the recognizer COMPUTE leaves. Refs: EPIPHANIES `E-OCR-NETWORK-SINK-1`; plan `tesseract-rs/.claude/plans/network-ruff-ogar-sink-v1.md`. Not yet a PR.

## 2026-07-04 — branch `claude/happy-hamilton-0azlw4` — `contract::unicharcompress` — the Tesseract recoder load side (byte-parity vs libtesseract)

**NEW** `lance_graph_contract::unicharcompress`: `UnicharCompress` (the LSTM recoder's code↔id table) + `RecodedCharId` + `RecoderError`, load side only (`from_le_bytes` / `load_from_file` = C++ `DeSerialize`; `encode` / `decode` / `code_range`; `dump_encode` / `dump_decode` parity surfaces). The FIRST binary-format leaf (`TFile` little-endian: `u32 count` + per-entry `[i8 self_normalized][i32 length][i32×length code]`). Byte-parity **GREEN** on real `/tmp/eng.lstm-recoder` — encode 112/112 + decode 112/112 + code_range=111 — via the committed `examples/recoder_dump.rs`, diffed vs a libtesseract 5.3.4 oracle (the 5.5.0-header ABI skew self-validated by the `Encode∘Decode` round-trip + `enc_size=112`). +10 contract tests; `-p lance-graph-contract` clippy `-D warnings` + fmt clean. Consumed by `tesseract-core::{Recoder, recoded_to_text}` (codes→decode→ids→`ids_to_text`; +1 boundary test, 8/8). Resolves the `recoder`=0x0802 concept (OGAR #148 mint, mirrored in the "0x08XX OCR rows" line below) to its content-store module. The recoder keystone (`invoke_recoder`) is UNBLOCKED but deferred (dispatch already proven generically by E-CPP-KEYSTONE-1). Refs: EPIPHANIES `E-CPP-PARITY-7`. Not yet a PR.

---

## 2026-06-23 — IN PR (`claude/medcare-bridge-lance-graph-wmx76z`) — ActionHandler⟷RBAC⟷orchestration spine

`contract::rbac`: `ScopeSpec` (axis-3 Copy token) + `ClassRbac` §4 default methods (`roles_reaching`/`row_scope`/`field_mask`; backward-compat, probe green). `contract::class_view::FieldMask::union`. `contract::action::ActionInvocation::commit_via<R: ClassRbac>` (no-admin-bypass convergence of the inline gate). `lance-graph-rbac::{authorize_scoped, ScopedDecision}` (§5 two-stage). `lance-graph-ogar::{OgarRbac<S: GrantSource>, GrantSource}` (Q5 local newtype, §6 evaporation seam). rs-graph-llm: `graph-flow-kanban::{run_cycle, CycleOutcome}` + `graph-flow-action::dispatch_via`. Plan: integration-actionhandler-rbac-orchestration-v1.

## 2026-06-23 — IN PR (`claude/medcare-bridge-lance-graph-wmx76z`) — `contract::rbac` — `ClassRbac` trait + `Operation` promoted to contract (keystone §11 trait-placement)

The `ClassRbac` grant-resolution trait (§4) + the `Operation` it ranges over were promoted from `lance-graph-rbac` into the zero-dep contract so `lance-graph-ogar`'s `OgarClassView` (deps contract, NOT rbac) can implement the keystone's `impl ClassRbac for OgarClassView` (Q5) — the missing wire in the `contract ↔ rbac ↔ ogar ↔ callcenter` chain. **NEW** `lance_graph_contract::rbac`: `ClassId` / `ActorId` / `RoleId` / `Operation<'a>` (reads `contract::property::PrefetchDepth`, no rbac dep) / `trait ClassRbac { actor_roles, grant_permits }`. `lance-graph-rbac` **re-exports** them (`policy::Operation`, `authorize::{ClassRbac, ClassId, ActorId, RoleId}` unchanged) — `authorize()` + `ClassGrants` + `Policy` + `AccessDecision` + the `0x0B` auth membrane stay in rbac. Zero breakage: `lance-graph-callcenter` builds against the re-exports (38s); the sibling `smb-realtime` / `medcare-realtime` gates consume `AccessDecision` (unmoved) untouched. **Verified:** contract::rbac 2 tests (incl. a contract-only `impl ClassRbac` proving ogar can satisfy it) + 723 contract tests; rbac 21 tests; callcenter builds; clippy `-D warnings` + fmt clean. Follow-on (not forced here): converge `rbac::auth::ResolvedIdentity` onto the existing `contract::auth::ActorContext`; the `OgarClassView` impl needs the §6 `project_role.granted` tenant. Refs: EPIPHANIES `E-CLASSRBAC-PROMOTED-TO-CONTRACT`, OGAR `CLASSID-RBAC-KEYSTONE-SPEC.md` §11/Q5.

## 2026-06-23 — IN PR (`claude/sync-ogar-codebook-auth-domain`) — `contract::ogar_codebook` synced to OGAR #110 (Auth domain `0x0B`) — fixes the codebook parity drift #42's ogar-vocab bump surfaced

OGAR #110 minted the `0x0B` **AuthStore** class family; the contract's zero-dep mirror lagged (39 vs 43), so `lance-graph-ogar`'s compile-time `COUNT_FUSE` + runtime `assert_codebook_parity()` fired and **broke the q2 Railway build** (`cockpit-server` → `lance-graph-ogar`). Synced the mirror: **NEW** `ConceptDomain::Auth` (`0x0BXX`) + `0x0B => Auth` routing + 4 `CODEBOOK` entries (`auth_store` `0x0B01` / `auth_zitadel` `0x0B02` / `auth_zanzibar` `0x0B03` / `auth_ory_keto` `0x0B04`), and the `lance-graph-ogar::parity::domains_agree` `(O::Auth, C::Auth)` arm. Mirror is now **43** = `ogar_vocab::class_ids::ALL`. **Verified:** `cargo build --manifest-path crates/lance-graph-ogar` (COUNT_FUSE green, 36s); `cargo test --manifest-path crates/lance-graph-ogar` (`mirror_is_a_faithful_copy_of_ogar_codebook` + 53 lib tests green); `cargo test -p lance-graph-contract` (8 ogar_codebook tests green); contract clippy `-D warnings` + fmt clean. The parity guard worked as designed — the `#[non_exhaustive]`-total `domains_agree` match tripped on the new OGAR domain. Refs: q2 #41 (root `/Dockerfile`) + #42 (ogar-vocab lock bump → `302c284`); this is the contract-side completion that unblocks the live Rust deploy.

---

## 2026-06-22 — MERGED #592 (merge `48794eaf`, `claude/contract-app-prefix-mirror`) — `contract::ogar_codebook` APP-prefix (hi-u16) mirror — closes `ISS-CONTRACT-APP-PREFIX-MIRROR`

Membrane consumers can now pull BOTH halves of a render `classid` BBB-safely from `lance_graph_contract::ogar_codebook` — no hand-stamped `0x000N`. **NEW:** `AppPrefix` enum (the OGAR#95 §2 allocation table as typed data — `Core 0x0000` / OpenProject `0x0001` / Odoo `0x0002` / WoA `0x0003` / SMB `0x0004` / Healthcare `0x0005` / Redmine `0x0007`) with `prefix()` / `from_prefix()` / `render(concept)`; free fns `render_classid(prefix, concept)`, `render_classid_for_concept(AppPrefix, &str)`, `classid_app_prefix(classid)`, `classid_concept(classid)` — the wire-compat mirror of OGAR#97 `ogar_vocab::app` (`render_classid_for::<P>` / `app_of` / `concept_of`), **no `ogar-vocab` dependency**. Two parity tests: `app_prefixes_match_ogar_allocation_table` (pins the 6 prefixes vs OGAR `PortSpec::APP_PREFIX`) + `render_classid_composes_decomposes_and_preserves_the_concept_half` (pins the `0x0005_0901` MedCare-patient worked example, and that the render lens never perturbs the lo-u16 concept RBAC keys on). Follows the OGAR#98 `canonical_concept_name` mirror precedent. Closes the gap the #591 consumer spellbook surfaced. Contract lib **+2 tests** / +1 doctest; `cargo fmt -p lance-graph-contract --check` clean; `clippy -p lance-graph-contract --all-targets -D warnings` clean (also `--features guid-v2-tail`). (Incidental: the crate-wide `cargo fmt` pass also corrected pre-existing struct-literal/line-width drift in `content_store.rs` — same crate, no behavior change.) Refs: PR #592 (merged `48794eaf`), ISSUES `ISS-CONTRACT-APP-PREFIX-MIRROR` (RESOLVED), `.claude/knowledge/ogar-consumer-preflight.md` § Core-gap (CLOSED), OGAR#97/#98.

---

## 2026-06-20 — golden-image (symbiont) harness shipped to `main`; lance-7 lockstep unified end-to-end

`crates/symbiont/` (workspace-`exclude`d) compiles+links the FULL stack into ONE binary — lance-graph + lance7/lancedb0.30 + ndarray + ractor + surrealdb(kv-lance) + OGAR. **Verified green** (real git-deps build, `CARGO_EXIT=0`, 4.3 MB binary runs): unified `lance 7.0.0 / lance-index 7.0.0 / lancedb 0.30.0 / datafusion 53.1.0 / arrow 58` — no lance-6/7 split. It is a **living integration harness** (`Dockerfile` + portable git-deps `Cargo.toml`) that tracks each fork's canonical branch (`master`/`main`), **NOT** a frozen snapshot; every per-session `jirak` branch is stale (HEAD ⊂ main/master, 0 unique commits). **`TD-SURREALDB-KVLANCE-LANCE7` PAID** — surrealdb `main` carries the lance-7 bump. PR #555 adds the 5+3 council `INTEGRATION_PLAN.md` (loose-end ledger → the Spain-grid acceptance gate). **Honest state:** linked into one binary; the *runtime edges* between the five crates are still pending integration (Grid→NodeRow bridge, kanban loop). Battle-test plan (probes A1–E3) queued behind the singleton-BindSpace → SoA switch. Refs: PR_ARC #555, EPIPHANIES `E-GOLDEN-IMAGE-IS-A-LIVING-HARNESS`, AGENT_LOG 2026-06-20.

---

> **2026-06-20 — branch work (`claude/happy-hamilton-0azlw4`)** — **UNICHARSET `direction` + `mirror` transcoded + byte-parity proven (E-CPP-PARITY-6), the sixth leaf — first to read PAST the bbox CSV.** `UniCharSet` now parses the two columns after `other_case` into `directions: Vec<i32>` + `mirrors: Vec<i32>` by continuing the per-line token walk (the bbox+stats group is one whitespace token, so columns land at fixed offsets regardless of the 5-tier fallback — no bespoke tier detector). `get_direction` (`unicharset.h:712`, load default `U_LEFT_TO_RIGHT` 0, out-of-range → `U_OTHER_NEUTRAL` 10) + `get_mirror` (`unicharset.h:721`, clamped like other_case, out-of-range → -1) + `dump_direction`/`dump_mirror`. **Byte-identical 112/112 each** on real `eng.lstm-unicharset` (self-validating oracle; direction varied: 55× LTR / 33× OTHER_NEUTRAL / 2·3·4·6 for digit chars; mirror has 10 bracket/paren pairs). Additive, zero-dep; +3 contract tests (26 unicharset total), my files clippy + fmt clean; reproducible via `examples/unicharset_dump.rs {direction,mirror}`. Consumed by `tesseract-core::CharSet::{get_direction,get_mirror}`. No Core gap. Remaining UNICHARSET sub-leaf: the float stats (bbox ints + width/bearing/advance) inside the CSV. EPIPHANIES `E-CPP-PARITY-6`; TECH_DEBT (contract crate not fmt-gated in CI).
>
> **2026-06-20 — IN PR (`claude/jirak-math-theorems-harvest-rfii13`)** — **kanban×Rubicon SoA value tenant + per-tenant counters (capstone S1 green).** NEW `ValueTenant::Kanban = 9` at value-slab `[112,120)` (8 B: `phase|exec|reserved|cycle`), added to `ValueSchema::{Cognitive,Full}` — reserve-don't-reclaim, **layout-preserving** (Full 112→120 B, stride 512 untouched, no version bump). `KanbanTenant` Copy view + `NodeRow::{kanban,set_kanban}` (owner-gated write / surreal read-only / Rubicon); `KanbanColumn`/`ExecTarget` `from_u8`. **Subsumes the envelope-pointer G1** — the node carries its own phase+cycle, pinning SoA↔kanban in the LE blob (a `FixedSizeBinary(512)` store reads kanban zero-copy at any version). NEW `tenant_counter` module + feature `tenant-counters` (default OFF, zero-cost no-op; one relaxed atomic/tenant-write when on) — the capstone NaN-census instrument; `set_kanban` is the first wired cascade point. Decisions kept (I-VSA-IDENTITIES + AGI-glove): thinking-style is ClassView+`Meta`, NOT a 128-bit tenant; plan-shape ClassView-derived; MUL flow-trigger is a function, not a tenant. Contract lib **714**/715(tenant-counters)/720(guid-v2-tail), clippy `-D warnings` + fmt clean all three. Refs: AGENT_LOG (cont.¹⁷), EPIPHANIES `E-KANBAN-IS-A-VALUE-TENANT-SUBSUMES-G1`, plan `capstone-cognitive-loop-wiring-nan-census-v1` (S1 green).
>
> **2026-06-20 — IN PR (`claude/jirak-math-theorems-harvest-rfii13`)** — **Zero-copy SoA read contract: `node_rows_from_le_bytes` (the surrealdb "second brain" primitive).** The inverse of `NodeRowPacket::as_le_bytes` (WRITE) — `canonical_node::node_rows_from_le_bytes(&[u8]) -> Option<&[NodeRow]>`, a CHECKED zero-copy cast (`len % 512 == 0` AND `ptr % 64 == 0`, else `None` → caller copies, no UB; empty→Some(empty)). This IS the LE contract a backing store satisfies so its bytes ARE the SoA the cognitive shader reads in place. **Brutal verdict:** lance-graph side now zero-copy-ready end-to-end; surrealdb's kv-lance does NOT qualify as scaffolded (`val: DataType::Binary` variable-length → needs `FixedSizeBinary(512)`), and value zero-copy holds only if stored UNcompressed (key/address always zero-copy). 712 contract lib green, clippy `-D warnings` both configs + fmt clean. Refs: AGENT_LOG 2026-06-20 (cont.¹⁴), EPIPHANIES `E-SURREALDB-SECOND-BRAIN-IS-ZERO-COPY-IFF-FIXEDSIZEBINARY`.
>
> **2026-06-20 — IN PR (`claude/jirak-math-theorems-harvest-rfii13`)** — **Clean separation: NEW `lance-graph-ogar` activation crate (OGAR Active-Record surface).** The OGAR half of `ontology=OGIT / ogar=OGAR`. OGAR is the AR Core and ALREADY `impl`s the contract: `ogar-class-view::OgarClassView impl lance_graph_contract::ClassView` (32 concepts), `ogar-vocab::Class` = AR shape, `canonical_concept_id == ClassId`. NEW `crates/lance-graph-ogar` (EXCLUDED, own `[workspace]`, git-deps OGAR@main + lance-graph-contract@main = ONE source, no `[patch]`) re-exports the full AR surface (ogar-vocab + ogar-class-view + ogar-ontology + ogar-adapter-surrealql) + a **parity-guard** (`assert_codebook_parity`: bijective `ogar_codebook::CODEBOOK ⇄ ogar_vocab::class_ids::ALL` + domain agreement, FAILS build on drift). Features: `default` (light, emit-only), `surrealql-parser` (parser half), `serde`. **Auto-activation = Cargo presence**: pull the crate → real OGAR AR + drift fuse; don't → contract's zero-dep mirror + bare ClassView trait (OGAR stays headless). `cargo test --manifest-path crates/lance-graph-ogar/Cargo.toml` **3/3** green, clippy + fmt clean, contract = ONE source (git main #ff1a3452). Refs: AGENT_LOG 2026-06-20 (cont.¹³), EPIPHANIES `E-OGAR-IS-AR-CORE-AUTOACTIVATED-BY-CARGO-PRESENCE`, plan D-OVC-5. **(#563 D-OVC contract realign now MERGED to main.)**
>
> **2026-06-20 — MERGED #563 (`claude/jirak-math-theorems-harvest-rfii13`)** — **D-OVC: contract classids realigned to OGAR `0xDDCC` + `contract::ogar_codebook` wire-compat mirror.** Resolved ISS-CLASSID-OGAR-DRIFT (operator-signed). **Realigned (layout-preserving const values, no `ENVELOPE_LAYOUT_VERSION` bump):** `CLASSID_OSINT 0x0007 → 0x0700` (OSINT domain root, `>>8 == 0x07`), `CLASSID_FMA 0x0008 → 0x0901` (anatomy concept in Health domain, `0x0900` = root). **Minted:** `CLASSID_PROJECT = 0x0100` + `CLASSID_ERP = 0x0200` with `ReadMode::{PROJECT, ERP}` (Cognitive/CoarseOnly) registered in `BUILTIN_READ_MODES`; `soa_graph::{PROJECT, ERP}` DomainSpecs. **NEW `contract::ogar_codebook`** (zero-dep, **wire-compat — NO OGAR↔contract dependency**): `ConceptDomain` (7 domains, `id>>8` route), `canonical_concept_domain`, `classid_concept_domain` (D-OVC-4 classid→domain), `source_domain_concept`, `CODEBOOK` (26 project `0x01XX` + 6 commerce `0x02XX`, mirrored from OGAR `ogar-vocab` `lib.rs:1073`), `canonical_concept_id`, `LabelDTO::from_canonical` + `id_le`. Drift-guard test pins the shared `0xDDCC` ids. Contract **710** lib (default) / **716** (`guid-v2-tail`), callcenter `--features query` **211** green; clippy `-D warnings` + fmt clean both configs. Refs: AGENT_LOG 2026-06-20 (cont.¹²), plan `ogar-vocab-contract-codebook-migration-v1.md` (D-OVC-1/2/4 SHIPPED, D-OVC-3 PARTIAL), ISSUES `ISS-CLASSID-OGAR-DRIFT` (RESOLVING).
>
> **2026-06-20 — IN PR (`claude/jirak-math-theorems-harvest-rfii13`)** — **codex roll-up + 16-family-adapter edges + Callcenter DataFusion/Gremlin + aiwar POC.** Follow-up to merged #557. (1) Both codex P1 fixes rolled in: classid filter (`project_snapshot`/`nearest_anchor` only project `classid == domain.classid` rows) + the operator's **16×8-bit family-node adapter** edge model — the `EdgeBlock` reads as 16 family adapters (each byte → a FAMILY by `family & 0xFF`, collision-aware skip), dissolving the >255-member aliasing; member-by-identity resolution removed (`E-FAMILY-ADAPTER-EDGES-ARE-RENDER-STABLE`). (2) `lance-graph-callcenter`: NEW `graph_table` (`query-lite`, `GraphSnapshot` → `nodes`/`edges` arrow MemTable `TableProvider`s + `register_graph(SessionContext)`) + NEW `graph_gremlin` (always-on Gremlin/SurrealQL traversal kernel). (3) `contract::aiwar` + example: `AiwarClassView` (category ⇒ family) + `aiwar_node_rows` ingest the real `aiwar-neo4j-harvest/data/aiwar_graph.json` (221 entities → 281 nodes / 60 family hubs / 481 edges). Contract 703 lib + callcenter 10 graph tests green; contract clippy `--all-targets -D warnings` clean. q2 wires the GraphSnapshot → Quadro-2 visual. Refs: AGENT_LOG 2026-06-20 (cont.⁷), EPIPHANIES `E-FAMILY-ADAPTER-EDGES-ARE-RENDER-STABLE`, TECH_DEBT `TD-CALLCENTER-QUERY-CLIPPY`.
>
> **2026-06-20 — branch work (`claude/jirak-math-theorems-harvest-rfii13`)** — **SoA-as-graph domain foundation for the OSINT/Gotham + FMA consumers (q2 renders the pixels).** New zero-dep `contract::soa_graph`: `project_snapshot(&[NodeRow], &DomainSpec) -> graph_render::GraphSnapshot` projects the canonical 32-byte head (NodeGuid + EdgeBlock) into the EXISTING Gotham/neo4j surface (`graph_render` — reused, not duplicated) — family nodes (by u24 `family`), member/in-family/out-of-family edges, all **zero value decode**. `nearest_anchor` ranks nodes to their nearest stability-anchor family by the new `NiblePath::family_hop_count` (CLAM tree distance). Two domains registered: `OSINT_GOTHAM` (classid **`0x0007`**) + `FMA_ANATOMY` (**`0x0008`**, bones = anchor families) in `BUILTIN_READ_MODES` (`ReadMode::OSINT` Cognitive/CoarseOnly hot; `ReadMode::FMA` Compressed/CoarseOnly cold). Anchor-ness is a HEAD field (`family`), never a value type — so "FMA bones as stability anchor" stays head-only (`E-ANCHOR-IS-A-HEAD-FIELD-NOT-A-VALUE-TYPE`). De-duped the GUID→NiblePath lowering: symbiont's `hhtl_path_of` now delegates to canonical `from_guid_prefix` (third copy collapsed). 698 contract + 12 symbiont tests green, clippy clean. **Deferred (named):** q2 rendering (q2 session), Callcenter DataFusion/gremlin POC, OntologyRegistry ClassView labels. Refs: AGENT_LOG 2026-06-20 (cont.⁶), EPIPHANIES `E-ANCHOR-IS-A-HEAD-FIELD-NOT-A-VALUE-TYPE`.
>
> **2026-06-20 — branch work (`claude/happy-hamilton-0azlw4`)** — **UNICHARSET `other_case` transcoded + byte-parity proven (E-CPP-PARITY-5), the fifth leaf.** `UniCharSet` now parses the case-pair id (the token right after the script) into `other_cases: Vec<i32>`, applying the load-time clamp (`unicharset.cpp:901`: a value `>= size`, incl. the absent default, folds to the id itself). Exposes `get_other_case` + `dump_other_case`, mirroring `unicharset.h:703` (out-of-range id → `INVALID_UNICHAR_ID` -1). **Byte-identical 112/112** on real `eng.lstm-unicharset` vs tesseract's own `get_other_case` (self-validating oracle, `other_case` mode; 60/112 self, 52 real pairs, e.g. `C`→`c`). Last field cleanly reachable by token-offset; direction/mirror/bbox need the multi-tier parser (next, larger leaf). Additive, zero-dep; +4 contract tests (23 unicharset total), clippy `-D warnings` + fmt clean; reproducible via `examples/unicharset_dump.rs other_case`. Consumed by `tesseract-core::CharSet::get_other_case` (+1 boundary test, 6/6). No Core gap. EPIPHANIES `E-CPP-PARITY-5`.
>
> **2026-06-20 — branch work (`claude/happy-hamilton-0azlw4`)** — **UNICHARSET script table transcoded + byte-parity proven (E-CPP-PARITY-4), the fourth leaf — first to transcode an INTERNING side-table.** `UniCharSet` now parses the per-line script name (the token after the optional bbox/stats CSV), interns it via an `add_script`-equivalent (`unicharset.cpp:1063`, insertion-order dedup) into `scripts: Vec<String>` with `null_script` ("NULL") seeded at sid 0 (the `unichar_insert` set_script, `unicharset.cpp:680`; so `null_sid_ == 0` always), and stores `script_ids: Vec<i32>`. Exposes `get_script` / `get_script_table_size` / `script_from_script_id` / `script_of` / `dump_script`, mirroring `unicharset.h:681` (out-of-range → `null_sid_` 0). **Byte-identical 112/112** on real `eng.lstm-unicharset` vs tesseract's own `get_script` (same self-validating oracle, `script` mode; oracle table = `["NULL","Common","Latin"]` confirmed empirically before writing the Rust). Mixed-tier safe (eng id 0 is tier-5 no-CSV, others tier-1 CSV). Additive, zero-dep; +4 contract tests (19 unicharset total), clippy `-D warnings` + fmt clean; reproducible via `examples/unicharset_dump.rs script`. Consumed by `tesseract-core::CharSet::{get_script,script_of}` (+1 boundary test, 5/5). No Core gap. EPIPHANIES `E-CPP-PARITY-4`. Next leaf: the full column tier-parser (unlocks other_case/mirror/direction/bbox).
>
> **2026-06-20 — branch work (`claude/happy-hamilton-0azlw4`)** — **UNICHARSET property accessors transcoded + byte-parity proven (E-CPP-PARITY-3), the third leaf through PROBE-OGAR-ADAPTER-UNICHARSET.** `lance_graph_contract::unicharset::UniCharSet` now parses the per-line hex property bitmask (`unicharset.cpp:824`) into a `props: Vec<u8>` and exposes `get_is{alpha,lower,upper,digit,punctuation}` + `get_isngram` + `dump_properties()`, mirroring the C++ inline accessors (`unicharset.h:497+`; out-of-range id → `false`, `INVALID_UNICHAR_ID` semantics). **Byte-identical 112/112** on real `eng.lstm-unicharset` vs tesseract's own `get_is*` via a **self-validating** oracle: the same harness dumps the id↔unichar bijection (proven 112/112 reference, E-CPP-PARITY-1) AND the properties — the bijection half diffing 0 proves the 5.5.0-header/5.3.4-lib layout is sound, making the property diff (also 0) trustworthy despite the version skew. Additive, zero-dep; +5 contract tests (15 unicharset total), clippy `-D warnings` + fmt clean. Consumed by `tesseract-core` as `CharSet::get_is*` (+1 consumer-boundary test, 4/4 green). Incidental: rustfmt-1.9.0 normalized two pre-existing test-assert wraps in `class_view.rs` (whitespace-only). No Core gap, no adapter state (per `E-CPP-KEYSTONE-1` "repetition of a validated pattern"). EPIPHANIES `E-CPP-PARITY-3`.
>
> **2026-06-19 — IN PR (branch `claude/edge-distance-basin-node-epiphany`)** — **basin-IS-a-node: the substrate is a virtual tree of MailboxSoAs, navigated by pure key arithmetic.** New `graph::mailbox_scan::{members, memberof, BasinOf}` — one-to-many (`members` = direct children one HHTL tier down) / many-to-one (`memberof` = parent via `NiblePath::parent`, returns `BasinOf::Local(row)` or `BasinOf::Route(NiblePath)` when the parent lives in another shard — the HHTL prefix IS the route key, **no coarse-fingerprint table**; `None` only at the top tier). Realizes `E-BASIN-IS-A-NODE` with **no ownership restructure** — the tree is the radix trie of the keys, the SoA stays flat, the zero-copy/Lance-tombstone invariant is untouched; all navigation is **zero value decode** (F2-guarded). 16/16 mailbox_scan tests, clippy clean. **Probe (perturbation-sim `basin_placement_learning.rs`): field-perturbation placement learns the basin tree — green, mean tree-hop 1.00 vs 4.13 random (75.8 % tighter)**, promoting the one CONJECTURE in `E-BASIN-IS-A-NODE` to measured FINDING [G]. **Three epiphanies this arc:** `E-BASIN-IS-A-NODE` (basin=node; distance=hop=`node_distance(PrefixDepth)`; 4-ary fan-out = Morton tile pyramid = perturbation-learnable field), `E-FAMILY-NODE-IS-META-AWARENESS` (the parent node IS the coarse Walsh band of its subtree — meta-awareness is structural, not a column), `E-GUID-SELF-ROUTES-THE-BASIN-TREE` (HHTL-tier truncation of the GUID = every ancestor's route key; the GUID self-routes). **Capstone:** one 512 B key, read five ways — representation / ontology / compute (Morton pyramid) / learning / meta-awareness — four of the five are key-resident zero-decode. Builds on #544/#545/#548 (mailbox_scan facets) + `E-COARSE-QUANTIZER-IS-SCALE-FREE-ROUTER`.
>
> **2026-06-18 — branch work** — **OGAR → lance-graph-ontology wiring closed.** `OntologyRegistry::class_id_for_guid(&NodeGuid) -> Option<ClassId>` composes the canon GUID→NiblePath fold (`contract::hhtl::NiblePath::from_guid_prefix`) with the registry's `NiblePath ↔ entity_type` bijection — the single missing join an audit this session surfaced (both halves were built with **ZERO callers**). A node carrying a classid now resolves its ontology class → `RegistryClassView` (fields/labels/template/DOLCE). Round-trip test pins the `classid_lo ↔ entity_type` consistency the audit flagged; zero-fallback (unbound → None) + lossy-fold refusal (high classid u16 → None). Completes the third "classid → X" axis reachable from a GUID (read-mode ✅ ocr.rs, methods ✅ unicharset keystone, ontology-shape ✅ now); aligns with `E-ODOO-CORE-FIRST-STRUCTURAL` (Core-side resolution, no new predicate/type). 16 ontology tests green; `registry.rs` clippy-clean + fmt clean. EPIPHANIES `E-OGAR-ONTOLOGY-WIRED-1`. Pre-existing `lance-graph-ontology` clippy debt noted (`TD-ONTOLOGY-LINT`).
>
> **2026-06-17 — IN PR (branch `claude/odoo-spo-fk-target-deep-reads`)** — Odoo SPO corpus enrichment (odoo-rs `UPSTREAM_WISHLIST` P1 + coupled P0). The corpus `crates/lance-graph/src/graph/spo/odoo_ontology.spo.ndjson` now carries **two new predicate families** (was 7 predicates: `depends_on / emitted_by / has_function / raises / rdf:type / reads_field / traverses_relation`): **`target`** (618) + **`inverse_name`** (102) — the relational comodel/inverse keyed by the relation IRI, ruff#18 sibling-triple shape `(odoo:account_move.line_ids, target, "account.move.line")`; and **+736 deep `reads_field`** (so `reads_field` 2 095 → 2 831) — each `@api.depends('rel.leaf', …)` resolved through the new target map and lifted onto the field's emitting method as a transitive read. Corpus 22 245 → **23 701** triples. New stdlib-only generator `tools/odoo-blueprint-extractor/odoo_blueprint_extractor/spo_enrich.py` (+14 unit tests) reads `/home/user/odoo/addons` (the same source the ORM extractor parses) to build the `(model, field) → (comodel, inverse)` map; additive, deterministic, idempotent. `odoo_ontology.rs` doc + tests updated (count 23 701, histogram incl. new predicates, 2 new enrichment tests); `action_emitter`/`spo` unaffected (function count 3 328 unchanged). **Cross-repo finding (verified, not faked):** the deep reads make the cross-model recompute-ordering edge `account_move_line._compute_amount_residual → account_move._compute_amount` *visible* to `od_ontology::RecomputeDag` (baseline: 0 cross-model compute edges → enriched: 27), delivering the wishlist's P0 ask — but the audit's MISSED-1 is a unidirectional *ordering edge*, NOT a cycle, so odoo-rs's `slice_2_compute_subset_no_cross_model_cycle` no-cycle assertion legitimately still holds (the "circularity" is semantic, not a `reads_field`↔`emitted_by` back-edge). The corpus's original generator (`emit_ontology2.py`/`methods.parquet`) is absent from the tree — only its output is committed; enrichment runs at the correct additive stage over the shipped corpus + present source. See `EPIPHANIES.md` E-ODOO-FK-DEEP-READS.
>
> **2026-06-17 — MERGED #521** (lance-graph-contract: C++ codegen target `MethodSig` + `UniCharSet` content store): **+940/-4 across 8 files, additive to `lance-graph-contract` only** (zero `NodeRow`/`ValueTenant`/`ValueSchema`/stride/`ENVELOPE_LAYOUT_VERSION` impact). The Core-side of the Tesseract C++→Rust transcode. **`codegen_manifest`** — `MethodSig` (the `&'static`-backed, `const`-constructible method-signature type the generated Rust names; the method-axis sibling of `ClassView`'s field projection, distinct because `ClassView::FieldRef` is `String`-backed and can't be `const`) + `ClassMethods` + `methods_for`. **`unicharset`** — `UniCharSet` (deepnsm::Vocabulary-shaped id↔unichar bijection), `.unicharset` parser, `id_to_unichar`/`unichar_to_id`/`dump`, **zero leptonica** (pure text, never touches `Pix`). **PROBE-OGAR-ADAPTER-UNICHARSET → FINDING:** the pipeline (ruff `ruff_cpp_spo` harvest → `reassemble` → `ruff_cpp_codegen` → these types) produces a `UniCharSet` **byte-identical 112/112** to the C++ libtesseract oracle on real `eng` data — the core-first transcode doctrine is now empirically proven end-to-end (doctrine flipped CONJECTURE→FINDING). The sole id-0 diff was the `NULL`→space convention (`unicharset.cpp:882`), fixed + locked by `null_token_maps_to_space` (codex P1 flagged it independently; resolved + thread closed). Pairs with **ruff #20** (harvester + codegen side, merged same day). 644 contract lib green; clippy `-D warnings` + fmt clean. Branch `claude/happy-hamilton-0azlw4`, merge `620bd8e`. **Next honest increment:** wire `UniCharSet` lookups through `classid → ClassView → UnifiedStep` (the `classid → &UniCharSet` resolver), per-leaf one-`diff` parity.
>
> **2026-06-17 — IN PR (branch `claude/bindspace-mailbox-soa-w3-w4a`)** — W3+W4a atomic read/write shim, the first behaviour-touching step of the BindSpace→MailboxSoA migration. New `cognitive-shader-driver::backing` module (`pub(crate)`): `BackingStore<'a>` (read) + `BackingStoreWrite<'a>` (write) — an enum with a `Singleton(&BindSpace)` arm (live default) and a `#[cfg(feature = "mailbox-thoughtspace")] Mailbox(&MailboxSoA<1024>)` arm. New Cargo feature **`mailbox-thoughtspace`** — **default-OFF, NOT in `lab`**; production stays singleton-read+write until W7. `driver.run()` keeps ONE body: all six dispatch reads (meta_prefilter / qualia17d / content_row / edge / entity_type / len) re-pointed through a `self.backing()` selector (`const DEFAULT_MAILBOX: MailboxId = 0`, `debug_assert!(mailboxes.len() <= 1)`, singleton fallback when no mailbox registered); `ontology()` stays on the singleton (re-home is W4b). Gates: **W2 differential** (`tests/w2_differential.rs`, 4 tests) asserts the WHOLE `ShaderCrystal` bit-identical (`f32::to_bits()`) across both arms incl. a non-zero-window case + non-vacuity; firewall CI lint (`tests/firewall.rs`) bars the two `CausalEdge64` twins (`ndarray::hpc::causal_diff` / `thinking_engine::layered`) from `src/`; field-isolation matrix + cycle-drop footprint (~6 KB/row vs ~71.6 KB) in `mailbox_soa.rs`. `unbind_busdto` C5 downgrade: cycle-plane index recovery feature-gated OUT under `mailbox-thoughtspace` (cycle plane never migrated — D-DIST-5), headline survives via `qualia[9]`; singleton build keeps bit-exact recovery. Tests: default **97 lib + 2 firewall + 2 e2e**; feature-on **98 lib + 2 firewall + 2 e2e + 4 w2**; clippy `--all-targets` (both cfgs) + fmt clean on touched files. **Pre-existing P0 surfaced (NOT introduced, NOT fixed here):** the `with-engine` build does not compile on `main`/HEAD (`engine_bridge.rs:259` uses `QUALIA_DIMS` without importing it); the busdto round-trip tests have never run, and (separately) the D-CSV-5b i4-qualia cutover breaks the `codebook_index` round-trip (stored in i4 `qualia[9]`, ±7 range, cannot hold a u16). Left untouched to keep scope to W3+W4a — flagged for operator. Plan: `.claude/plans/bindspace-mailbox-soa-w3-w4a-impl-v1.md`.
>
> **2026-06-16 — MERGED #512** (perturbation-sim review fixes + **core-first transcode doctrine**): +591/-5 across 11 files. **Code fixes (review of #511):** `examples/calibrate.rs` divide-by-zero guard on degenerate grid; `src/hhtl.rs::basin_lambda2` `assert_eq!(keys.len(), grid.n, …)` precondition (silent corruption→loud panic); `TECH_DEBT.md` MD018 reflow. **Doctrine (the structural delivery):** new mandatory-read `core-first-transcode-doctrine.md` (218 LOC) + 3 new agent cards (`core-first-architect`, `core-gap-auditor`, `adapter-shaper`) + `BOOT.md`/`README.md` wires + EPIPHANIES entry + CLAUDE.md (+21 LOC, doctrine wire-up — NEW content unread by this session). Likely directly aligned with the ontology-first stance the operator locked on odoo-rs. Branch `claude/happy-hamilton-0azlw4`, merge `1e23c410`. 75 lib tests + clippy + fmt clean.
>
> **2026-06-16 — MERGED #513** (perturbation-sim: inertia §0 promotion gate + CAKES/CHAODA + witness standing-wave + H ingest): +1009/-2 across 10 files. Disjoint from #512 by design. **(1) §0 gate** — `GuardrailVerdict::RatifiedReuse`: `inertia_buffer` takes `ResidueEdge` `INERTIA_SLOT = 5`, reuses an existing tenant, invents no new axis → passes §0 by **reuse, not waiver**. Topology stays HHTL-OGAR GUID key; the buffer is one more value, orthogonal by key/value split. **(2) Probe 1 CAKES + CHAODA-lite** over HHTL basins: per-basin `[λ₂, size, inertia]` features; `CHAODA_FLAG=0.75` mirrors ndarray::clam's flag; example `chaoda` flags planted brittle block (basin 1.1.0, score 1.000). Full `ClamTree` ensemble path gated on local ndarray sibling. **(3) Probe 2 witness arc as standing wave** (METHODS §11): `particle == wave` via Parseval (`Hᵀ·H = N·I`), agreement to **0.00e0**; `witness_from_spectrum` is the O(N)-per-arc read-many amortization win. **(4) Probe 3** per-bus inertia (H) ingest path opened. Branch `claude/perturbation-sim-inertia-clam`, merge `8a3e335b`. Does NOT touch `canonical_node`.
>
> **2026-06-16 — MERGED #511** (perturbation-sim: substrate calibration + calibrated SoA member spec): **+886/-0 additive**. New `examples/calibrate.rs` (318 LOC) runs the ICC(2,1) + Spearman + Pearson + Cronbach α battery against perturbation-sim as ground truth; new `src/columns.rs` (177 LOC, spec only) names the calibrated `SoaMemberSpec` set; new `src/hhtl.rs` + `examples/hhtl_grid.rs` + `CLAM_CHAODA_FRAMING.md`. **Findings:** all 5 contingency factors certify by VALUE at 2-bit linear (ICC ≥ 0.96) — existing palette/turbovec tenants already suffice, §10 "statistics survive the encoding" CONFIRMED; α preserved within Δ ≤ 0.02 at ≥4-bit; cross-axis ρ wobbles ±0.15 at N=24 → read ≥6-bit. **Self-correction:** `d_lambda2`'s ICC=0 was variance-guard underflow at ~1e-7 magnitude (not heavy-tail / not near-constant — both prior guesses retracted); normalized storage fixes to 1.00 at 2-bit. **Locked:** 5 factors → existing tenants (no new columns for contingency axes); the **one genuinely additive member is `inertia_buffer`** — orthogonal to topology (`Spearman(λ₂, buffer) ≈ 0` per PR #509), spec only, promotion gated by §0 anti-invention guardrail. Significance per Jirak n^(p/2−1) (I-NOISE-FLOOR-JIRAK). Does NOT touch the operator-locked `canonical_node`. Branch `claude/perturbation-sim-calibrate-soa`, merge commit `c3dddfc9`. 71 lib tests + clippy + fmt clean.
>
> **2026-06-16 — MERGED #510** (surreal_container seam falsifier — IN-direction): `crates/surreal_container/tests/scheduler_seam.rs` — **+125/-0, one new file, zero source change**; first integration-test file in `surreal_container`. Five kill-condition-first tests pin the `SurrealMailboxView → NextPhaseScheduler::on_version → KanbanMove` contract end-to-end: legal-successor walk over the full Rubicon arc, absorbing-column guard (Commit/Prune schedule no advance), `-550_000µs` Libet-anchor pinned to the Planning→CognitiveWork Σ-commit crossing, lowering-determinism, ExecTarget-rides-onto-move. Branch `claude/sleepy-cori-aRK2x`, merge commit `0e6452c8`. **Out of scope (explicit, deferred):** the OUT-direction = planner-emit `KanbanMove` (`CognitiveCycle` sequencer + §9 LOCKED from #496) — `D-MBX-A6-P3` remains the next unblock. This PR proves the downstream half; the upstream half is still hand-rolled. Test bench is also the template the planner-emit half will be verified against once it lands. `surreal_container` is excluded from CI, so the suite runs locally/on-demand (standing CI-coverage-gap follow-up — CI tests 4 of ~30 crates).
>
> **2026-06-15 — REVERTED (operator)** — the tesseract-rs `soa` wiring below was **deleted** (branch reset to master `420de08`). Operator: *"we don't want to use original Tesseract, we want to transcode it into Rust — delete everything you copied from original Tesseract into tesseract-rs."* Wrapping the original Tesseract C engine + parsing its TSV is the wrong direction; the real goal is a **pure-Rust OCR**. The contract-side transcode (`LayoutBlock::to_node_row`) + keystone STAY — they are OCR-engine-agnostic (a pure-Rust OCR feeds the same `LayoutBlock` → `NodeRow`); only the original-Tesseract coupling was removed. The strike-through entry below is retained per APPEND-ONLY.
>
> ~~**2026-06-15 — cross-repo landed** — **tesseract-rs fork wired to the transcode.**~~ *(REVERTED — see above)* `AdaWorldAPI/tesseract-rs` branch `claude/wonderful-hawking-lodtql` commit `1687c718`: opt-in `soa` feature (default-OFF — standalone OCR build untouched) + `src/soa.rs::tsv_to_nodes(tsv, classid, min_conf) -> Vec<NodeRow>` parsing tesseract `get_tsv_text` word rows → `contract::ocr::LayoutBlock` → `to_node_row`. Contract dep is a path dep mirroring smb-office-rs (sibling checkout). **Edition-2015 compatible** (the fork has no `edition` field → 2015: root `extern crate` + submodule root-relative `use` + explicit `TryInto` — all caught + fixed by verifying in a 2015 scratch crate against the real contract before pushing, 2 tests green). Pushed via `GH_TOKEN`+pygithub (out-of-MCP-scope fork). Could NOT compile the full crate here (no tesseract C-lib) — the transcode LOGIC is what's verified; the fork's own CI needs a co-located lance-graph for `--features soa`.
>
> **2026-06-15 — branch work (post-#496)** — **tesseract OCR → NodeRow transcode POC (keystone payoff).** `lance_graph_contract::ocr::LayoutBlock::to_node_row(classid, identity) -> NodeRow` — the reference transcode any `OcrProvider` (tesseract-rs + others) reuses, the keystone end-to-end: `classid → classid_read_mode → ValueSchema` gates WHICH tenants land; `BlockKind::entity_type() -> u16` → `ValueTenant::EntityType`, `confidence: f32` → `ValueTenant::Energy`, each written at its canon slab offset via the new `ValueTenant::{value_offset(), byte_len()}` (derived accessors over the locked carve — not new properties). **`text`/`bbox` are NOT bundled** (`I-VSA-IDENTITIES`: node = identity + typed scalars; the string + pixel geometry live in an external content store keyed by `identity`). Schema-gated (`schema.has(t)` before each write) so a Bootstrap-resolving class writes an empty slab; transcoded rows ride the `SoaEnvelope` zero-copy (verified). §0 anti-invention: reuses the existing EntityType/Energy tenants, no "ocr_kind" field. +4 tests; **623 contract lib green; clippy `-D warnings` + fmt clean.** Lives in the contract (next to the `ocr` types it uses, zero-dep, testable here — no OCR C-lib, no fork); tesseract-rs just adds the contract dep + calls it (integration step). Branch, not yet a PR.
>
> **2026-06-15 — branch work (post-#496)** — **keystone (contract half): GUID decode + classid→read-mode `LazyLock`.** `lance_graph_contract::canonical_node::{GuidParts, ReadMode, classid_read_mode}` + `NodeGuid::{heel(), hip(), twig(), decode() -> GuidParts, read_mode() -> ReadMode}` (re-exported from `lib.rs`). **The "read the GUID as a GUID" surface** the operator spec'd: `decode()` returns all six canon groups (classid + HHT·HEEL/HIP/TWIG + family·"Leaf" + identity) in one read; `ReadMode` bundles the two *already-existing* read-mode axes (`ValueSchema` + `EdgeCodecFlavor`) — **NOT a new node property, NOT a SoA column** (§0 anti-invention; it's the resolution lens, nothing stored on the row); `classid_read_mode(u32)` is the **single source both the consumer and OGAR inherit** — a `LazyLock<HashMap<u32,ReadMode>>` builtin registry (same immutable-after-init pattern `lance-graph-ontology` uses for its seed namespace registry), zero-fallback to `ReadMode::DEFAULT` for any unconfigured classid. `ReadMode::DEFAULT = {Full, CoarseOnly}` mirrors the `ClassView::value_schema` POC default (paired revert; `read_mode_default_is_full_poc` guards it). `Display` deduped onto the new HHT accessors. +6 tests (decode round-trip, HHT↔Display, read-mode single-source, carrier delegation, full-slab connect); **619 contract lib green; clippy `-D warnings` + fmt clean.** Delivers the contract-side half of the #496 keystone; the ontology-side `NiblePath::from_guid_prefix` (20→≤16-nibble subset) meets it at the classid (follow-up). Branch, not yet a PR.
>
> **2026-06-15 — branch work (post-#496)** — **helix `Signed360` codec + `HelixResidue` right-sized 48 B → 6 B.** Operator caught a slab over-allocation: `HelixResidue` reserved **48 *bytes*** but the intent was a 24-bit equal-area hemisphere **doubled = 48 *bit* = 6 B** (a bits→bytes slip; 42 dead bytes), and the tenant used **none** of the `helix` crate (zero-dep contract — only a doc string). Fixed: **(1) `helix::Signed360`** — the signed full-sphere codec: `HemispherePoint::signed_lift(n,N,sign)` (`y = sign·√(1−u)` → full sphere, `r²+y²=1`), `Sign{Pos,Neg}`, and `Signed360 {rim: ResidueEdge, polar: signed-lift centred@128 (sign recoverable), azimuth: u16 over 360°}` + `ResidueEncoder::encode_signed`. +9 tests; **helix 72 lib + 7 doctests green; lib clippy `-D warnings` + fmt clean.** **(2) contract** `HelixResidue.elems_per_row` 48→6, downstream tenants shifted (Turbovec 118 / Energy 134 / Plasticity 138 / EntityType 142), budgets re-locked (**Full 154→112, Compressed 98→56**); **613 contract green.** **NO `HelixFlavour` enum** — one canonical encoding, one tenant size (a fixed-offset SoA can't vary width per-class; Hemisphere = degenerate `sign=+`); the contract stays zero-dep, the producer writes `Signed360::to_bytes` into the 6 B. Cheap NOW (POC FULL default, no persisted real instances); after instances persist it's a version bump. Branch, not yet a PR. New: `TD-HELIX-PROBE-CLIPPY` (pre-existing `probe_mantissa_fill` clippy/fmt drift, NOT introduced here — helix is excluded so CI-invisible, same class as the standing `causal-edge` 47/1 red).
>
> **2026-06-15 — MERGED #496** (integrated-cognitive-planner reference map + ValueSchema + FULL POC default): `lance_graph_contract::canonical_node::{ValueSchema, ValueTenant, VALUE_TENANTS}` — the value-side `EdgeCodecFlavor` analog (9 append-only tenants carving `[32,186)`; presets Bootstrap/Cognitive/Compressed/Full). `ClassView::value_schema()` default flipped **Bootstrap→Full (TEMPORARY POC** — every unconfigured class materialises the full slab so consumers transcode against it; `TD-VALUESCHEMA-FULL-POC-DEFAULT` revert-when-POC-concludes; type-level `ValueSchema::default()` stays Bootstrap, only class→schema *resolution* flips). New reference plan `.claude/plans/integrated-cognitive-planner-v1.md` — **§0 ANTI-INVENTION GUARDRAIL (READ FIRST)**, §1–§7 grounded file:line map, §8 7-item additive ledger, §9 3-hardener verdicts; the SPEC for the integrated-planner refactor (~90% exists; remaining = the keystone + 6 seams, NOT a new build). CI 5/5 green; contract 613 lib tests; merge `2e58e034`. **The keystone = `NiblePath::from_guid_prefix` (the 20→≤16-nibble subset) + classid→ClassView read-mode on `lance-graph-ontology::registry` (already an immutable conflict-refusing `entity_type↔NiblePath` bijection)** — the single next unblock that converges the refactor, the tesseract-rs OCR transcode (`contract::ocr` → NodeRow), AND the OGAR-identity migration (`soa-migration-diff-resolution-2026-06-13.md`). HEEL=cache `dolce_id` / HIP·TWIG=deterministic subClassOf descent / registry=recorder-not-minter (verified `registry.rs`+`wikidata_hhtl.rs`). New: `TD-COARSERESIDUE-NO-VALUE-TENANT`, `TD-LAZY-IMPORT-VERSION-PIN`; IDEAS CLAM-residue-ladder TODO.
>
> **2026-06-13 — shipped (autoattended, cross-repo)** (turbovec ⇄ ndarray): new excluded standalone crate **`crates/lance-graph-turbovec`** — Google TurboQuant (arXiv 2504.19874, the AdaWorldAPI `turbovec` fork) bridged onto the spine. `TurboVec` wraps `turbovec::TurboQuantIndex` with a `Kernel::{NativeLut, PolyfillGemm}` A/B switch. **Cross-repo (branch `claude/wonderful-hawking-lodtql` in turbovec + ndarray + lance-graph):** turbovec re-pointed from crates.io `ndarray 0.17` → the AdaWorldAPI fork (path, P0 forks-only; `blas` opt-in so default builds BLAS-free; `rust-toolchain.toml` = 1.95.0); new `turbovec::search_polyfill` (feature `ndarray-simd`) expresses scoring as a batched int8 GEMM via **`ndarray::simd::matmul_i8_to_i32`** (re-exported through `simd.rs` — AMX `TDPBUSD` tile → AVX-512 VPDPBUSD → AVX-VNNI → scalar, dispatched inside ndarray, zero intrinsics in turbovec). **Measured finding (E-TURBOVEC-AMX-WRONG-TOOL-1):** the polyfill GEMM is 11.4× SLOWER than the native nibble-LUT (TurboQuant trades the matmul away → AMX accelerates the op it removed); native LUT stays production, polyfill is the AMX-ready baseline. Placement: index → spine, kernel-math → ndarray (already owns clam/cam_pq/cascade/amx_matmul). Synergy map (HDR popcount stacking early-exit, Belichtungsmesser σ thresholds, preheating vs palette256) in `crates/lance-graph-turbovec/KNOWLEDGE.md`. Tests green in all three repos; benchmark via `examples/kernel_speed.rs`. NOT a merged PR yet (branch work).
>
> **2026-06-03 — hardened (follow-up after #460)** (D-HELIX-1 wiring): `crates/helix` now takes **ndarray as a MANDATORY, non-optional git dependency** (`git = AdaWorldAPI/ndarray @ master`), replacing the optional `path` dep + `ndarray-hpc` feature. Why: (1) codex P2 — an optional *path* dep still forces Cargo to read the local sibling manifest at resolution, so a clean checkout failed before feature selection; (2) directive "ndarray is mandatory for lance-graph". `simd.rs` always uses `ndarray::simd` (no scalar fallback); the self-contained fork → no import cycle. 63 unit + 6 doctests green; clippy/fmt clean. See E-HELIX-NDARRAY-MANDATORY.
>
> **2026-06-03 — shipped (autoattended)** (D-HELIX-1): new standalone crate `crates/helix` — the golden-spiral **Place/Residue** codec from the user's `KNOWLEDGE.md`. HHTL = deterministic PLACE; helix = orthogonal RESIDUE. Pipeline: equal-area `√u` hemisphere placement (`HemispherePoint`) → stride-4-over-17 `CurveRuler` coupling → Fisher-Z/arctanh `Similarity` alignment → EULER_GAMMA hand-off → 256-palette `RollingFloor` quantise (occupancy-drift + version stamp) → 3-byte `ResidueEdge` endpoint pair; metric-safe L1 via 256×256 `DistanceLut` (`distance_adaptive`) + non-metric byte-Hamming `distance_heuristic`. `prove()` closes the 2-D discrepancy Open Item (companion to `jc::weyl`). Zero-dep default (`edition 2021`, empty `[workspace]`, root `exclude`); optional `ndarray-hpc` feature routes batch Fisher-Z through `ndarray::simd::simd_ln_f32`. **61 unit + 6 doctests green** on BOTH feature configs; clippy -D warnings + fmt clean. ~80% overlaps existing CERTIFIED primitives by design (clean-room, user-directed) — see `crates/helix/KNOWLEDGE.md` § Overlap & Consolidation + E-HELIX-OVERLAP + TD-HELIX-OVERLAP-1. Branch claude/gallant-rubin-Y9pQd.
>
> **2026-06-01 — shipped (autoattended)** (D-A3): `lance_graph_contract::atoms` — `I4x32::pack`/`unpack` implemented (the 2 `todo!()`s gone) + new `I4x64` (256-bit / 64 signed-i4 dims, `repr(C, align(16))`, 32 B) + private `sext4`. Two's-complement signed-i4 nibble codec (byte-compatible with `QualiaI4_16D` + the `CausalEdge64` mantissa), sign-agnostic (caller pre-scales). The carrier is a deterministic **CAM address** + sparse-intensity "smell" — NO vector search, no float; the `{instance,reference}` dual is rejected ("64" = 64 poles). Contract lib **562 green** (+9), offline, zero new deps. The bipolar `−introspection..+exploration` pole semantics + asymmetric scaling ride the caller's pre-scale (A4). Plan `.claude/plans/a3-carrier-v1.md`; doctrine `.claude/knowledge/ephemeral-warm-cold-lifecycle.md`.
>
> **2026-06-01 — PR-in-flight (autoattended)** (D-EW64-2): `lance_graph_contract::episodic_edges::EpisodicEdges64::{promote, strongest}` — MRU "promote" strengthens an edge to slot 0 (the hot / most-immediate position); fire→front, un-refired ages toward slot 3 and evicts to the cold connectome; **slot order IS the strength ranking** (no per-edge weight stored — the co-addressed `CausalEdge64` plasticity carries the Hebbian weight, recency is the slot index). Realizes `E-EW64-STRENGTH-IS-CE64-PLASTICITY` (the user's "stronger immediate edges"). Zero-dep; contract lib 533 green (+5), default clippy clean, episodic_edges.rs pedantic+nursery clean. The surreal-LIVE "wingman" that drives `promote` stays GATED on OQ-11.6 (LanceDB-LIVE fallback exists) — this is the substrate-agnostic hot-tier mechanism it calls.

---

## Recently Shipped PRs (reverse chronological)

| PR | Merged | Title | What it added |
|---|---|---|---|
| *gap note* | — | **#781–#925 are NOT in this table** — carried by the dated sections above + `PR_ARC_INVENTORY.md`. Recorded 2026-08-12 (codex P2 on #930) rather than silently reconstructed; the table had stalled at #780. | — |
| **#946** | 2026-08-12 | D-CZ-1 PASSES; D-CZ-0 had NO artifact behind it; + the arc's rated formula matrix | MIXED (probe code + measurements + 2 epiphanies + a synthesis doc). **D-CZ-0 was marked DONE with no script and no JSON** — quoted in the plan, the arc, the LATEST_STATE row and 3 PR bodies; #945's self-audit called it "verified" while only comparing arc entry to PLAN (prose vs prose). Reproduced 4/9 rows (1.004/1.022/0.994/0.931); the 5 excluded land candidates are **unreproducible** (centres never recorded; no coordinates invented). `|∇p|` definition identified FROM DATA as **Pa/cell without cos(lat)** (max dev 0.069 vs next-best 0.398) → R3 ~40 % low; metric-corrected ladder 10.3/15.5/61.2/100.9, **ORDER survives**, range 9.3× → **9.8×**. **D-CZ-1 PASS**: both controls lose on both metrics in all 4 regimes and in **19/19 storms**; `GEO-DEGENERATE` saturates 72–97 %; **C1b separation 6.28** vs ≥3. **The run AMENDED C4**: ρ saturated on the diagonal (real-arm spread 3e-6…4.7e-5) so C4 could not have fired — `L` keeps ρ off-diagonal, **C4 moves to RMSE in Pa**; amendment carries its trigger and was propagated to header + bar. Committed **`--selftest`**, each assertion disable-verified (dropping tie-averaging **flips the sign** of heavy-ties ρ). Exploratory within-R4 ρ=+0.444, n=19, **p=0.0578 (not significant)** — and the tautology check does NOT dismiss it (confounds −0.035/+0.253/−0.081); committed WITH a not-a-result status string. **Plus `SUBSTRATE_FORMULA_MATRIX.md`**: 46 rated primitives (15 A / 4 B / 4 C / **20 D / 3 V** — half the inventory is negative), 14 known-vs-discovered pairs, 9 apparatus lessons, 13 gaps; built by re-extracting from committed artifacts (131 primitives, 2.35 M subagent tokens), **28 figures re-verified, 0 mismatches, 1 rounding fixed**. Every off-diagonal cross-swap cell remains unmeasured. |
| **#944** | 2026-08-12 | `substrate-comfort-zones-v1` §2 rebuilt: horse race → CROSS-SWAP transfer matrix | MIXED (plan revision + epiphany + a review-caught correction). Operator ruling: the method is cross-swap under the premise the model CAPTURES the phenomenon but is NOT calibrated — so miscalibration is the measurement CONDITION, not an arm, and RMSE under a deliberately wrong calibration is bad BY DEFINITION. §2 is now a **4×4 donor × target matrix** `M[D][T]`; primary metric Spearman `ρ`; the hypothesis's quantity is the derived **transfer loss** `L[D][T] = ρ[T][T] − ρ[D][T]`. `CAL-ABS-OWN`/`CAL-ABS-FOREIGN` collapse into ONE arm's diagonal/off-diagonal. The dynamic arms have **no donor**, so `L ≡ 0` by construction — the property under test, and **C2 requires BOTH** that it hold exactly AND that `CAL-ABS` be proven non-degenerate through the same code path (the can-it-DIFFER gate W5 paid for). Two NEW bars for the operator's constancy-is-relative point: **C1b** MEASURES constancy (`separation ≥ 3`) instead of claiming it; **C1c** gives the suitability ASSUMPTION a falsifier — regimes must differ in autocorrelation decay + rank-distribution shape, not merely in `\|∇p\|`, and a null there VOIDS the reading. `L < 0` pre-registered as informative (a donor's wider range can legitimately beat an outlier-set diagonal). **CodeRabbit caught the sticky half**: §2.3 still named the FOREIGN comparison as the hypothesis while C4/board scored against the diagonal — two incompatible verdict criteria, the weaker holding the headline; fixed `1756c518`. Fifth instance this arc of a claim consistent with its own operands but inconsistent with a SIBLING claim. Unchanged: the 4-tier regime ladder + its three preflight corrections, equal budget, geometry axis, C0 gate, output contract. **D-CZ-1 (control losability) still Queued and still gates every later cell.** Zero code, zero probe runs, zero fetches. |
| **#942** | 2026-08-12 | "mean" qualifier fix + `substrate-comfort-zones-v1` (calibration × regime, made falsifiable) | MIXED. Plan states the operator's hypothesis as an INTERACTION (a poorly-calibrated adaptive substrate does relatively better in strong/turbulent storms than in calm regimes) across **4** held-constant regimes × formula × calibration quality — the preflight SPLIT "flatland" into calm and active tiers: R1 CALM Amazon (`\|∇p\|` 10.2) → R2 OCEAN S-Pacific (14.9) → R3 ACTIVE W-Siberia (43.8) → R4 STORM (95.6), a 9.3× range. "Worse everywhere but least-worse in storms" ≠ "better only in storms", and the plan distinguishes them. D-CZ-0 preflight DONE (sample-composition arithmetic run before the first fetch — it is what forced the 4-way split; units annotated at coefficient definition). **D-CZ-1, the control-losability smoke test, is PRESCRIBED BY the plan but Queued — not yet run**, and gates every later cell. D-CZ-2..6 Queued. Also fixed CodeRabbit's Minor on #941 — #940's 25 %/9 % figures are MEANS over 19 storms, qualifier restored (4th instance this week of scope-lost-in-summary). |
| **#940** | 2026-08-12 | W6 lands — vector-sum dipole model VOID by its own anti-vacuity control; stranded stratum empty by CT-F14's filter arithmetic; same-PR sign/units correction round | B0 VOID: single-geo R²=−0.104 (worse than the mean); both controls (permuted P_bow, P_bow rotated 90°) clear the `≤single-geo+0.03` ceiling. B3 stranded (`\|v_storm\|<8 m/s`): n=0 — `displacement_km≥250`/6h implies `\|v_storm\|≥11.57 m/s` for every admitted storm, `E-THE-DISPLACEMENT-FILTER-ATE-THE-STRANDED-STRATUM-1`. Same-PR: codex+CodeRabbit caught a sign-convention bug (`D=-spine(...)`, matching `low_pole_bearing()`'s own flip; verified offline that this leaves R²/B0/B1/B3 unchanged, confirmed on the actual re-run) and a units error (`c_bow` is km⁻¹ not dimensionless; replaced with the dimensionally valid `\|c_bow·P_bow\|` vs `\|D\|` metric — mean-over-19-storms: geo≈25%, bow≈9% of mean `\|D\|`). CT-F17's gate now moot for this model form. |
| **#938** | 2026-08-12 | W5 v2 RUN lands — B2 REVERSES to genuine FAIL, B3's VOID CONFIRMED at full headline scale, both link families now verified | B2: real diffusion resolved (raw rel-L2 vs unsmoothed input = 0.190), operator's own anisotropy = **1.5251 vs the 1.25 bar** (baseline through the clean 3.35σ mask = 1.0046 — the operator alone contributes ~0.52). `domino.rs` gather-design claim REFUTED at this test point. B3: 99.68 % (family A) / 99.56 % (family B) of the QUALIFYING population's control links land on a pure Fibonacci offset (n_qualifying=4 782 017, out of the headline lattice N=7 651 227 — not a 62k sub-sample) — dominated by the two discovered strides 2584=F(18) / 4181=F(19) respectively. Same-PR fixes for 3 more codex findings: family-B histogram was previously uncomputed (now measured + JSON patched); "four orders of magnitude" corrected to ~1.9 (76.9×); B4 downgraded from verdict to explicit descriptive reading (n=19 was dropped without pre-authorization — B4 stays open). |
| **#936** | 2026-08-12 | W5 v1 RUN (B2 PASS / B3 VOID / B4 smooth) — ⊘ SUPERSEDED same day, all three verdicts compromised; v2 CODE fix merged in this PR, v2 RUN/results in flight | Codex found 4 real defects: B3 control subsampled at 250k/band (~74 % self-linked at headline, ratio 0.9996 uninformative); B2's fit floor = input σ (8 iters ≈ inert, "resolved" indistinguishable from "untouched"); bump only 1.72σ from the mask edge (analytic truncation ratio 1.2082 ≈ the measured "1.213 asymptote" — likely the MASK, not the operator); 99.38 % Fibonacci-membership figure was chat-only. Fixed same day, code merged HERE (`106ca605`, verified an ancestor of this PR's merge commit — corrects two overclaims caught by codex P2 on PR #937: the fix was described as "landing in a follow-up PR" when it had already landed, and the histogram artifact as "committed" when only the SCRIPT that produces it had landed — the tracked JSON is still v1's until the v2 run completes): full-band control, V-matched iteration scaling, bump moved to 3.35σ clearance + baseline-through-mask computed, offset histogram now computed by the script. The epiphany's MECHANISM survives (sub-cap 99.38 % measurement was never subsampled) but its "does not exist" phrasing was also softened (codex P2 on #937: 99.38 % ≠ 100 %, one N/one construction ≠ universal proof) — only the headline evidentiary number is retracted; the claim is restated at its actual evidentiary scope, not deleted. |
| **#935** | 2026-08-12 | The validation wave RUNS: T1–T4 + W2s-a executed; `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1` | T2/T3/T4 PASS (68–106× at 200q; tempered 140/140 exact vs golden 124–127/140; 39.0 % naive-rounding collapse). T1 twice-corrected under codex review of its own run: verified-permanent m* = **1.9–2.7× q** (first-crossing 1.0–1.4× was not permanent; q=17: D*(22) back above the ceiling). W2s-a: G1 VOID, G2/G4 FAIL — the CONTROL is degenerate (two translated identical grids are symmetry-uniform, CV ~1e-12; cannot lose any evenness comparison). W5 pre-registered, run in flight at merge. |
| **#934** | 2026-08-12 | Board hygiene #932/#933 + storm-geography refinement | MIXED. Refinement, each claim measurement-grounded: centering = tempered territory (spiral center structurally sub-floor at any N; shipped `find_center` already exact-register); collision annulus = golden (Go/territory framing); overlay = controlled chaos at 12 B/node; self-description asymmetry (tempered up to q, golden unbounded). Task causes the regime, geography sorts the tasks. Arc entry written one PR late, caught in #935's hygiene pass. |
| **#933** | 2026-08-12 | 4 codex fixes on merged #932 + `golden-vs-tempered-stride-v1` (head-vs-gut plan) | MIXED. Fixes: W5 bump sub-floor despite "fixed" N (local index = √(r²N), not √N → N=3·F(17)², bump r=0.75, bands 1–2 excluded); tie test redefined per-source (d1/d2 ratio); 17-TET sign convention unified; B3 control distance-matched (residue arithmetic exposed 1500/2600 as wrong-scale in disguise). Plan: T1 crossover at m≈q across 8 q; T2 golden 68–106× ahead at m=200q; T3 tempered fills q/q by proof vs golden 124/140 at q=140; T4 naive φ-rounding collapses at 39 % of q∈[8,300). "Best stride" declared metric-dependent (3 metrics → 3 different winners at q=17). |
| **#932** | 2026-08-12 | Golden-ratio index floor (≥17/21) + temperament mechanism; W5/W2s-a re-specced before any worker ran | Merged BEFORE its own 4 review findings could be addressed (fixed in #933). Floor: F-convergents behave like φ only from n≈17–21 (err 1.5e-4 → 1.8e-7 → 3.7e-9); emergent parastichy pair ≈ √N ⇒ N ≳ F(17)². Mechanism: temperament — coprime closure + distributed comma (12 fifths miss by +23.46 ct; 17-TET fifth exact by construction, +3.93 ct/fifth spread) = D-QUANTGATE's anti-moiré dither. `E-THE-GOLDEN-STEP-IS-THE-WRONG-STEP-AT-SMALL-Q-1` + `ISS-HELIX-GOLDEN-STEP-LABEL`. Chat-register quotes paraphrased out of all committed artifacts (12 sites). |
| **#930** | 2026-08-12 | #929 hygiene → grew into an open-review sweep of #920–#930 + report §10 (product-lead program) + `weather-w-probes-v1` (Sonnet worker briefs) | MIXED (started as pure hygiene, corrected its own description twice per the #927 lesson). Sweep found 3 ledger figures wrong (R² "5th decimal" claim, "10 probes" undercount, "+13/−0,+10/−0,+0/−0" audit figures) and replaced the append-only audit itself (zero-deletions → suffix check, both halves measured, the new check correctly flags this very PR). §10 = measurement standard + dipole vector-sum model incl. stranded-storm regime + corridor α + queue-and-bow + sunflower/spiral-ADI substrate. Worker briefs W5/W2s-a/W6 ready; CT-F17 gated on W6 + independent adversarial audit. See sweep entry, `.claude/board/PR_ARC_INVENTORY.md` for the correction table — the figures in #928's/#929's rows below are SUPERSEDED by that entry, not edited in place. |
| **#929** | 2026-08-12 | CT-F16 kills the steering rescue; the control scores the headline; circular resultant replaces the sign test | 2 probes (`comet_tail_f16`, `comet_tail_resultant_instrument`) + report §5.12/§5.13 + `E-THE-CONTROL-SCORED-THE-HEADLINE-1`. Steering FAILED both bars (0.579 vs 0.70; residual +28.4 % wider; sweep monotone to surface, best 850 hPa); rotated control scored 0.684 = CT-F14's headline; resultant resolves the same rows at p=0.0050, offset estimated −30.2°. |
| **#928** | 2026-08-12 | Board hygiene for #927 — chain terminator | Pure prepend (+13/−0, +10/−0). Living-documents vs append-only-ledgers correction discipline banked; zero-removed-lines diff = structural append-only audit. |
| **#927** | 2026-08-12 | Board hygiene for #926 + 4.7×-not-5× correction + append-only revert | Mixed. 9/10 figure self-verification caught the 6 % favourable rounding; an in-place edit of a MERGED EPIPHANIES entry reverted same-PR (the rule being cited was the rule broken). |
| **#926** | 2026-08-12 | weather-p1: the storm spine (90.9–94.3 %), the 12-byte L4 carrier, moderators identified but unwired | 9 919 insertions, zero Rust/product code. Spine `[G]` (3 blind samples, 41+ storms); 12-byte `6×(8:8)` facet at 0.07 Pa RMSE / +1.59 Pa bias; Fisher-z rank/tail-vs-level demarcation; CT-F14 NO-VERDICT held against a pooling rule that said ESTABLISHED; `var()`→MSE R² fix at 11 sites (hid +92.76 Pa bias). |
| **#780** | 2026-07-21 | Recipes audited → 3 real tenants wired (A9 24-loci CausalWitnessFacet + SPO + qualia) → rung-ordered NaN-gated causal ladder | 4 commits, merge `8a00988`. `recipe_claim_audit` proves the 34 kernels sound-on-proxy (28/34; 2 INERT CAS/ETD + 4 CONSTANT ARE/ZCF/ICR/HKF) but 0/34 read a real organ → wiring gap, not composition. New contract modules: `causal_witness` (A9 = 24 signed-i4 loci, THIRD ClassView reading of the #729 12-byte register, no stored bytes), `recipe_substrate` (SubstrateView projects SPO+witness+qualia; **qualia additive + stakes-only**, causality = KAUSAL witness edge; missing tenant → NaN), `recipe_dispatch` (rung-ordered NaN-gated ladder keyed by NARS inference; ICR #31 unshadowed; the systematic-sweep mode COMPLEMENTING `select_tactic`'s 8/34 saccade — E-LADDER-UNSHADOWS-SELECTOR-1). +18 tests → 970 green, clippy clean; 3 probes all-gates-green. Theory anchors: PCRLLM 2511.08392 (proof-carrying single-step, Pei Wang), Causal-ML survey 2206.15475 (SCM ladder), MME-Reasoning 2505.21327 + NARS-Swift flagged; cc-thinking-skills mapped as rung-4 macro catalogue (E-THINKING-SKILLS-ARE-RUNG-4-MACROS-1). D-REC-WIRE-1. |
| **#676** | 2026-07-10 | Post-#674 doc arc: E-NOBODY-WAITS-1 + VISION.md (graded AGI canon) + ancestry census + D-MTS-1 design inputs | Doc-only, 4 commits. E-NOBODY-WAITS-1 banked (no messages, no actors; ractor = compile-time ownership only; `&mut` IS the serialization; prime invariant "nobody waits for anything or any scheduling"; the ack-gated advance is canonical for the **ticket tier only** — `kanbanstep` stays canonical for stream reasoning, and the ack (the **SLA gate**) is reserved for the OGAR ticket tier, repurposed consumer-side as the **actionhandler queue** (see E-ACK-HARD-GATE-VS-KANBANSTEP-STREAM-1 + E-ACK-SLA-GATE-ACTIONHANDLER-QUEUE-1); supervisor KanbanMsg drivers = TD-MESSAGE-RESIDUE, leave-as-is per operator). `.claude/v3/VISION.md` = the AGI-aspiring canon, every claim graded [G]/[G-on-proxies]/[RULING]/[ASPIRATION], 2-Sonnet-preflight + 2-Opus-filigree provenance, all 9 review fixes applied. MODULE-TABLE ancestry census: thinking-engine (51) + p64-bridge (1) + cognitive-shader-driver (22 src + build.rs) with gem-status column. Plan addendum: D-MTS-1 frozen comparison — AriGraph context V3-TENANT-SHAPED; arm-discovery + DeepNSM ingest legs; CAM-PQ 6×8=48-bit path codes (address side) vs 6×palette256² = 12 B = one V3 tenant (value side). coderabbit 5/5 findings fixed in `093489c`. Merge `001839e`. |
| **#674** | 2026-07-10 | V3 W2–W6 continuation + comma quorum/awareness measured + 5+3 council + StyleFamily dedup (D-TSC-1) | 16 commits. `contract::style_family::StyleFamily` (12 orchestration families, frozen ordinals Deliberate=0..Metacognitive=11) + `ThinkingStyle::family()` (36→12 total) + `default_runbook()` — five mutually divergent hand-rolled 12-style tables replaced (E-FIVE-STYLE-TABLES-1); first live 5+3 council run (spec v1→v2→v3). D-MTS-5 comma quorum MEASURED GREEN (N_eff 11.00/12; boundary = spectral participation) + D-MTS-6 comma awareness MEASURED GREEN (k*=1: one stored truth bit/comma level ≈ aligned k=4; lattice buys ≈log₂12 effective bits; D-MTS-6b gates real CE64 shrink). `BatchWriter::ack_and_propose` first-ack-wins dedup (codex P2). 1549 tests green. Merge `cd5178e`. |
| **#632** | 2026-07-02 | Cross-session intake: RouteBucketTyped (C6) + emission_scan + OCR codebook mirror + GraphRAG-rs inventory + operator rulings | Three sibling wishlists dispositioned; C6 merged verbatim (nexgen retires vendor diff); emission_scan = 2nd scan-family instance (pattern NAMED); OCR 0x08XX mirror of OGAR #148 (fuses arc: flip fuse + two-sided COUNT_FUSE — fuse FIRED 65v68 in the merge window, cleared by lock bump, 68==68); rulings: ownership+tripwires, R-1 naming phantom (domain:appid:classview), R-2 closed (512B row frozen, strided edges), L3 schema design killed; codex P2 x2 resolved (precedence global; collision documented). Contract 792/792. Merge `df367471`. |
| **#631** | 2026-07-02 | W1b LIVE: WAL batch writer (4 probes green) + M15 rename + temporal synthesis + live oracle numbers | batch_writer implemented: BTreeMap WAL board, ack(cast, LanceVersion) join, delegation cache, never-refuses stacking (probe 4); M15 MulGateDecision rename (W2 unblocked; collapse_gate confirmed 3rd distinct type); operator rulings pinned (zero-copy descriptor casts + eager drain + mutual masking; melden macht frei — freeze retracted; temporal.rs = the read side, replay = QueryReference::at + deinterlace, M24=M25=time-travel ONE mechanism). Measured live: W3c oracle 1-2 ms framework overhead vs 8.4-8.7 s LLM round trip (rig->xAI grok-4 via FlowRunner); JITSON serve.rs = local CI oracle delta. Planner lib 204 + probes 4/4. Merge `c7149eab`. |
| **#630** | 2026-07-02 | V3 W1 START: preflight deltas + WAL writer probes + adoption scan + D-PERT-1 + temporal synthesis | Fable-5 ten-point preflight (M24 board=WAL, W6a baseline inversion, W3 oracle ratchet, W2 probe-first reorder) + operator rulings folded live: zero-copy sink (cast = descriptor never bytes, flush via NodeRowPacket::as_le_bytes), "melden macht frei" (stacked casts never refused — 4 ignored probes define W1b green), temporal.rs deinterlace = the READ side (replay = QueryReference::at + deinterlace; M24/M25/time-travel are ONE mechanism; ack carries LanceVersion). Landed code: batch_writer skeleton + 4 probes; contract::classid_scan (771 green); D-PERT-1 rename (462 green). Audits: planner-SoA type-real/wiring-dormant (M15 GateDecision rename BLOCKING before W2); M7 corrected (NodeRowPacket IS production SoaEnvelope, codex P2); graph-flow benched ~0.4-0.5us/step (two-speed confirmed); M25 KanbanSessionStorage design (graph-flow-kanban envelope exists — wire don't invent). Merge `9a6df2a1`. |
| **#629** | 2026-07-02 | V3 SUBSTRATE consolidated entry point (`.claude/v3/`) + ractor ownership attestation | `.claude/v3/` tree shipped: README (orientation), INTEGRATION-PLAN (W0–W6), COMPONENT-MAP (reuse/repurpose/retire), ENTROPY-MILESTONES (N→1 ledger), MODULE-TABLE (per-file census core/contract/planner), soa_layout/ (LE contract, tenant lanes, consumer map, routing), knowledge/ (substrate primer, mailbox-kanban model, sonnet-worker-guardrails), agents/BOOT.md (4 V3 cards); `/v3` skill + `/v3-audit` command; CLAUDE.md/BOOT.md ★ entrypoint. Review sharpenings folded: LE byte-order range-scan caveat, 3-shape legacy corpus scanner (incl. `0xAAAA_DDCC`), ractor helper-scope ruling (NOT messaging — slow; helper only: spawn/supervision/occasional control RPC). Ownership compile attestation: `KanbanActor<O: MailboxSoaOwner>` `type State = O`, owner MOVES in at pre_start; 22 supervisor tests green on the AdaWorldAPI ractor fork. Merge `28f17cd7`. |
| **#628** | 2026-07-02 | classid canon:custom half-order flip EXECUTED (P0+P1+P2) | `CLASSID_ORDER = CanonHigh` live: canon `domain:appid` HIGH / custom LOW (`0x0701_1000` = `0x07:01::1000`); ONE flippable composition + `classid_canon_compat` (mint-forward both-forms reader — RBAC authorizes pre-flip rows, no re-bake); new-form mint constants + `CLASSID_*_LEGACY` aliases; hhtl dual-form fold; OGAR#95 reconciled (prefix = custom half, values unchanged); ogar pin → `19373a2` (OGAR #147 lockstep). Fleet: OGAR #147 + MedCare #180 + woa-rs #177 merged; q2 #71 + op-nexgen #68 open. Merge `6858118b`. |
| **#625** | 2026-07-02 | brick-3 RAN: truncation-disallowed / overflow-as-SoC-reroute + DO-arm 3-bucket triage | Knowledge tier: `ast-as-partof-isa-address.md` Status → PARTIALLY MEASURED (rank-minter brick-3 RAN via `ruff_csharp_spo` → `ruff_spo_address::{mint, mint_factored}` → `medcare_probe`; naive 6-tier mint FALSIFIED; truncation DISALLOWED — overflow = SoC-reroute trigger per OGAR `256-cap-is-a-lint` #130/#140; overflow *classification* shipped upstream as `ruff_spo_address::soc`, class-conditioned shapes 6×2/4×3/3×4). New `do-arm-triage-3-bucket.md` (fuzzy→canonicalize / standard-DO→one DTO adapter + codebook swiss-knife / random→hand-port-and-graduate; C#/C++ DO-extractor gap named). Proprietary numbers stay in private MedCare-rs archive. Lock: ogar → `a0c7936` (post-#146, fuse 65==65). Mid-arc OSINT mirror commits DROPPED per OGAR #146 ruling. Branch `claude/medcare-bridge-lance-graph-wmx76z`, merge `5561908`. |
| **#627** | 2026-07-02 | classid canon:custom flip TRIGGERED (doc-only) | Operator ruling recorded + `classid-canon-custom-flip-v1.md` ACTIVE: canon `domain:appid` → hi u16, custom (`0x1000` temporary marker) → lo; `0x0701_1000` / `0x07:01::1000`; OSINT low byte = appid space (zero vocab rows, OGAR #146 67→65 fuse balanced); q2 gate WAIVED; ISSUES ×4 resolved/ruled; codex P2 guards locked (class_id via `classid_canon(id)` never `as u16`; legacy keys demote not retire). Merge `c8e1ec4`. |
| **#626** | 2026-07-02 | V3 convergence wiring: tenant-carve certification, RungElevator, P6 wave probe, seam-list plan | "Wire, don't invent": `RungLevel::{from_u8,elevate,de_elevate,pearl_level,causal_mask_bits}` + `RungElevator` (sustained-BLOCK policy over P2/P3-certified masks; converged with `escalation::rung_delta` via `apply_delta` — one ladder, two signal sources) wired through the driver (persistent elevator, `ctx.rung=1` proxy retired, grpc rung saturates-never-wraps per codex P2); BOTH V3 tenant carves matrix-certified (Cognitive + Compressed); P6 probe (wave dist == certified palette read, markov_soa verified); `[patch.crates-io] ndarray` → local sibling path (fetch deadlock gone; first in-sandbox core build, 925/925). Plan `v3-convergence-wiring-v1.md`; worker Rule 7. Branch `claude/v3-substrate-migration-review-o0yoxv`, merge `5aaee33`. |
| **#542** | 2026-06-18 | E-OGAR-IS-FOUNDRY capstone + 5+3 council + the key→row baton | Foundry/Gotham = "write the OGAR class schema + inheritance"; everything else is generic machinery over it (ontology=`classid→ClassView`+inheritance, AR=DO/THINK, pipelines=`compute_dag`, apps=Jinja-over-classes, query=Cypher⇄SurrealQL one IR). Added `MailboxSoaView::row_for_local_key -> Option<usize>` (default `None`, deferred-binding — the key→row baton for a future `Backend::MailboxSoa` router). Epiphanies `E-OGAR-IS-FOUNDRY`/`E-CYPHER-IS-THE-KANBAN-AST`/`E-GUID-IS-THE-GRAPH`; plan `cypher-kanban-ast-unification-v1`. Council corrections: `from_guid_prefix` is on `NiblePath` not `NodeGuid`; "odoo proof" = CONJECTURE; `ogar-adapter-surrealql` not a crate. Branch `claude/q2-substrate-grounding`, merge `faca377f`. |
| **#540** | 2026-06-18 | `lite-unified` additive default-OFF coexistence feature gate | **+35/-5, 2 files.** `lite-unified = []` in `crates/lance-graph/Cargo.toml` (empty until SurrealQL-on-lance lowering lands). **datafusion stays DEFAULT — NOT deprecated, NOT made optional.** Process, not switch; promoted per query-shape once OQ-LU-2a is green. Zero behavior change at default features. Branch `claude/lite-unified-gate`, merge `ef7e97ef`. |
| **#539** | 2026-06-18 | particle/wave click → `ClassView::compute_dag` (the one Core gap) + electricity-cascade join | **+570, 6 files, additive to `lance-graph-contract` only.** `class_view::{ComputeEdge, compute_dag_is_acyclic, compute_dag_topo_order}` + `ClassView::compute_dag` default (zero-fallback). `compute_dag_topo_order -> Option<Vec<u8>>` = the recompute ORDER (Kahn; `None` on cycle; leaves excluded). 4+1 epiphanies (`E-OGAR-ROUTER-ENCODER` = router+encoder, physics-duality stripped; `E-EXCEL`, `E-CHESS` = NNUE-proven shape, `E-PERTURBATION` = the cascade IS compute_dag, Weyl bound certifies incrementality; folded `E-AR-DO-WIRING`). Doc-join `crates/perturbation-sim/COMPUTE_DAG_MAPPING.md` (perturbation-sim stays zero-dep). **⚠ tracked: `ClassView::value_schema` default = `ValueSchema::Full` is a TEMPORARY POC — revert to `Bootstrap` + its test when the consumer-transcode phase ends (type-level default stays `Bootstrap`).** 13/13 class_view, clippy/fmt clean. Branch `claude/particle-wave-click-epiphany`, merge `b0255499`. |
| **#538** | 2026-06-18 | cycle-aware write contract (S2.5) + OGAR DO arm (`action.rs`) | Cycle-aware mailbox write + the Perdurant DO arm. `mailbox_soa`: `last_write_cycle`/`stale_write_count`/`WriteOutcome`/`WriteCell`/wrap-aware `write_row`. `action::{ActionDef, ClassActions, actions_for, effective_actions, ActionInvocation}`; `commit` gate = def-match → RBAC → state-guard → MUL → `ExecTarget::SurrealQl`. `object_instance` is a full `NodeGuid` (5+3 CATCH-CRITICAL). `substrate_sanity.rs` NaN/tautology harness (8 tests). `docs/OGAR_CONSUMER_API.md`. Branch `claude/soa-write-deinterlace-inc2`. |
| **#537** | 2026-06-18 | docs: STACK_SCAFFOLD + OGAR consumer-API groundwork | `docs/STACK_SCAFFOLD.md` (surreal+ractor+ndarray fork-wired reference; surreal Lance-KV status corrected to module-implemented-not-yet-feature-wired). Docs tier. |
| **#521** | 2026-06-17 | lance-graph-contract: C++ codegen target (`MethodSig`) + `UniCharSet` content store | **+940/-4 across 8 files, additive to `lance-graph-contract` only.** The Core-side of the Tesseract C++→Rust transcode. **`codegen_manifest`** `MethodSig` (`&'static`/`const`-constructible method-signature type the generated Rust names; method-axis sibling of `ClassView`'s field projection) + `ClassMethods` + `methods_for`. **`unicharset`** `UniCharSet` (deepnsm::Vocabulary-shaped id↔unichar bijection) + `.unicharset` parser + `dump()` — **zero leptonica** (pure text). **PROBE-OGAR-ADAPTER-UNICHARSET → FINDING:** the full pipeline produces a `UniCharSet` **byte-identical 112/112** to the libtesseract oracle on real `eng` data; core-first doctrine proven end-to-end. `NULL`→space (`unicharset.cpp:882`) was the sole id-0 diff, locked by `null_token_maps_to_space` (codex P1 independently flagged + resolved). Pairs with **ruff #20**. 644 contract lib green; clippy `-D warnings` + fmt clean. Branch `claude/happy-hamilton-0azlw4`, merge `620bd8e`. |
| **#512** | 2026-06-16 | perturbation-sim review fixes + **core-first transcode doctrine** + 3 new agent cards | **+591/-5 across 11 files**. Fixes from #511 review: `examples/calibrate.rs` divide-by-zero guard on degenerate grid; `src/hhtl.rs::basin_lambda2` `assert_eq!(keys.len(), grid.n, …)` precondition (silent-corruption → loud panic); `TECH_DEBT.md` MD018 reflow. **The structural delivery:** new mandatory-read `core-first-transcode-doctrine.md` (218 LOC) + 3 specialist agents (`core-first-architect`, `core-gap-auditor`, `adapter-shaper`) + EPIPHANIES entry + CLAUDE.md (+21 LOC) wire-up. Likely aligned with the ontology-first / codegen-as-cut-tail doctrine the operator just locked on odoo-rs. 75 lib tests + clippy `-D warnings` + fmt clean. Branch `claude/happy-hamilton-0azlw4`, merge `1e23c410`. |
| **#513** | 2026-06-16 | perturbation-sim: inertia §0 promotion gate + CAKES/CHAODA + witness standing-wave + H ingest | **+1009/-2 across 10 files**, disjoint from #512 by design. **§0 promotion gate** for `inertia_buffer`: `GuardrailVerdict::RatifiedReuse` — takes `ResidueEdge INERTIA_SLOT = 5`, reuses existing tenant, invents no axis → **passes by reuse, not waiver**. **Probe 1 CAKES + CHAODA-lite** over HHTL basins (`CHAODA_FLAG=0.75` mirrors ndarray::clam; example flags brittle block 1.1.0 score 1.000). **Probe 2 witness arc as standing wave** — Parseval proves `particle == wave` to 0.00e0; `witness_from_spectrum` is the O(N)-per-arc read-many amortization. **Probe 3** per-bus inertia (H) ingest path. Does NOT touch `canonical_node`. Branch `claude/perturbation-sim-inertia-clam`, merge `8a3e335b`. |
| **#511** | 2026-06-16 | perturbation-sim: substrate calibration (study as ground truth) + calibrated SoA member spec | **+886/-0 additive across 9 files** — `examples/calibrate.rs` (new, 318 LOC, ICC/Spearman/Pearson/Cronbach battery), `src/columns.rs` (new, 177 LOC, **spec only**), `src/hhtl.rs` (new, 175 LOC), `examples/hhtl_grid.rs` (new, 81 LOC), `CLAM_CHAODA_FRAMING.md` (new, 75 LOC). Calibrates the SoA value tenants against perturbation-sim's deterministic study as ground truth: **all 5 contingency factors certify by VALUE at 2-bit linear** (ICC ≥ 0.96), the §10 "statistics survive the encoding" claim **CONFIRMED**; α preserved within Δ ≤ 0.02 at ≥4-bit; read ≥6-bit for cross-axis orthogonality. The **one additive member** named: **`inertia_buffer`** — orthogonal to topology per PR #509's `Spearman(λ₂, buffer) ≈ 0`, spec only, promotion gated by §0 guardrail. Self-correction: two prior guesses on `d_lambda2`'s ICC=0 (heavy-tail / near-constant) **retracted** — it was a variance-guard underflow at ~1e-7. Significance per Jirak `n^(p/2−1)`. Does **NOT** touch `canonical_node` (operator-locked). Branch `claude/perturbation-sim-calibrate-soa`, merge commit `c3dddfc9`. |
| **#510** | 2026-06-16 | test(surreal_container): additive seam falsifier — SurrealMailboxView → VersionScheduler → KanbanMove | **`crates/surreal_container/tests/scheduler_seam.rs`** — additive: **+125/-0, one new file, zero source change**; first integration-test file in the crate. Five kill-condition-first tests pin the **IN-direction** of the surreal↔kanban↔scheduler wiring against the real `SurrealMailboxView` (not the in-crate `FakeView`): full Rubicon arc → legal successors, absorbing-column no-advance guard, Libet `-550_000µs` anchor at the Planning→CognitiveWork Σ-commit crossing, lowering-determinism, ExecTarget-survives-lowering. **Out of scope (deferred, explicit):** OUT-direction = planner-emit `KanbanMove` (`CognitiveCycle` sequencer + §9 LOCKED from #496) — `D-MBX-A6-P3` remains the next unblock. Branch `claude/sleepy-cori-aRK2x`, merge commit `0e6452c8`. CI-invisible (crate excluded). |
| **#459** | 2026-06-03 | feat(helix): golden-spiral Place/Residue codec (zero-dep + optional ndarray-hpc) | **`crates/helix`** — new standalone codec realising the Place/Residue `KNOWLEDGE.md` (HHTL = PLACE, helix = orthogonal RESIDUE). `HemispherePoint` (√u equal-area placement) → `CurveRuler` (stride-4-over-17) → `Similarity` (Fisher-Z/arctanh) → `RollingFloor` (256-palette; occupancy-drift + version stamp) → 3-byte `ResidueEdge` + `DistanceLut` (metric-safe 256×256 L1) + `prove()` (2-D discrepancy companion to `jc::weyl`). Zero-dep default (empty `[workspace]`, root `exclude`); optional `ndarray-hpc` = batch Fisher-Z via `simd_ln_f32`. 63 unit + 6 doctests green both configs; clippy/fmt clean. ~80% clean-room overlap with CERTIFIED primitives (E-HELIX-OVERLAP / TD-HELIX-OVERLAP-1). Merge commit `ef35ff1`. Branch `claude/gallant-rubin-Y9pQd`. |
| **#450** | 2026-06-01 | NAL syllogism capstone + atoms/styles/NAL → planner-DTO unification (A1/A2) | **`causal-edge::syllogism`** — hardwired NAL **figure** resolver (`Figure{Chain,ChainRev,SharedSubject,SharedObject}` + `figure()`/`syllogize()`; SPO term-sharing → Deduction/Induction/Abduction + signed mantissa; the reasoning kernel, Pearl-2³-analogue). **A1** `contract::nars::InferenceType::{to,from}_mantissa` (zero-dep cross-crate rule bridge) + `From<grammar::NarsInference>`. **A2** `rung: RungLevel` on both `ThinkingContext` structs (the meta-aware handle). Unification spec §0–14 (`.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md`) + **vart vendored** (`/home/user/vart`). Branch `claude/jolly-cori-clnf9`. _(PR_ARC #450 entry owed.)_ |
| **#411** | 2026-05-27 | Cognitive substrate: locked 33-TSV atom layer + 34-tactic recipes + escalation loop | **D-PERSONA-1** `contract::escalation` + `planner::mul::escalation` (CollapseHint/InnerCouncil/EpiphanyDetector/GhostEcho/WisdomMarker/Checklist, 13 tests). **`contract::atoms`** — LOCKED 33-dim TSV `CANONICAL_ATOMS` (3 Pearl + 9 Rung + 5 Σ + 8 Ops + 4 Presence + 4 Meta) + `I4x32` carrier. **`contract::recipes`** — 34-tactic metadata catalogue. **`contract::recipe_kernels`** — the 34 tactics as 34 `Tactic` impls + registry over a shared `ThoughtCtx`. Charter D0: **ladybug-rs has no relation, rewrite-not-port**; lattice is **SPOQ** (SPO 2³ causal + Q qualia overlay); business = OGIT sidecar; markers gate the datapath/control/gate partition. Green: escalation 13 / atoms 3 / recipes 4 / recipe_kernels 5 + 446 prior, no warnings. Branch `claude/splat3d-cpu-simd-renderer-MAOO0` (39 commits). See PR_ARC #411. |
| **#389** | 2026-05-16 | fix(sprint-12/wave-F): codex P2 — AttentionMaskBackend impl for AttentionMaskSoA + canonical MailboxId import | Codex P2 follow-on to PR #388. Adds `AttentionMaskBackend` trait impl for `AttentionMaskSoA` (Wave-F surface coherence) and converges duplicate `MailboxId` imports onto the canonical contract definition. Merge commit `b526485`. |
| **#388** | 2026-05-16 | impl(sprint-12/wave-F partial): D-CSV-10 sigma-tier-router + AttentionMask + splat ops + governance (6 of 12 workers landed) | Sprint-12 Wave F fleet partial landing. **D-CSV-10** `SigmaTierRouter` crate (Rubicon-resonance ΔF + threshold → Σ10 commit, hand-tuned threshold per OQ-CSV-6, tracked as TD-SIGMA-TIER-THRESHOLDS-1); **D-CSV-12** scalar splat op fleet on i4 (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`); **AttentionMask** SoA + actor + backend surface; W-F8 TYPE_DUPLICATION_MAP refresh (records two-`TrustTexture` coexistence as TD-TRUST-TEXTURE-DUPE-1); W-F10 sprint-11 Opus meta-review; W-F11 i4-substrate-decisions knowledge doc; W-F12 cognitive-substrate-convergence-v2 plan draft (608 lines). Merge commit `77f2d26`. |
| **#387** | 2026-05-16 | impl(sprint-11/wave-E): D-CSV-8 MUL i4 SIMD evaluation + D-CSV-9 8ch↔SPO transcoder | **D-CSV-8** integer MUL evaluation on `QualiaI4_16D` + signed mantissa (scalar i4 path; AVX-512/NEON deferred → D-CSV-13 sprint-12). **D-CSV-9** 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary; 16-mapping bidirectional round-trip; renames `set_channel` → `set_channel_u8` to widen equivalence class. Merge commit `e042c70`. |
| **#386** | 2026-05-16 | impl(sprint-11/wave-D): D-CSV-7 MailboxSoA + D-CSV-6a WitnessCorpus core (parallel workers) | **D-CSV-7** `MailboxSoA<N>` integration: W-slot referencing + per-row plasticity accumulator + `apply_edges` for baton receipt; `last_emission_cycle` u32::MAX sentinel + lib re-export + ndarray hpc-extras feature. **D-CSV-6a** `WitnessCorpus` partial (W-slot anchor + chain invariant; sorted by emission cycle, drop-oldest truncation). Full CAM-PQ-indexed corpus (D-CSV-6b) sprint-12. Merge commit `33110c8`. |
| **#385** | 2026-05-16 | impl(sprint-11/wave-C): D-CSV-5a sibling QualiaI4Column add (double-write, no read-side change) | **D-CSV-5a** sibling `QualiaI4Column` add to `cognitive-shader-driver::FingerprintColumns` per OQ-CSV-4 ratification (sibling-then-cutover). Double-writes f32 + i4 during sprint-11/12; cutover (D-CSV-5b) drops f32 column once consumers migrated. Worker recovery from stash + `[..17]` slicing + hpc-extras feature gate. Merge commit `6f58418`. |
| **#384** | 2026-05-16 | impl(sprint-11/wave-B): D-CSV-2 QualiaI4_16D type + OQ-CSV-1 ratification (Option α) | **D-CSV-2** `QualiaI4_16D` 16-dim signed-i4 type in `lance-graph-contract::qualia` + f32↔i4 migration helpers (`to_f32_17d`). **OQ-CSV-1 ratified to Option α** — canonical convergence-observable vocab (arousal/valence/tension/curiosity/…); drop dim 16 "integration" placeholder. 14 unit tests pass; codex P1 + CI gate fmt fix. Merge commit `0751a8b`. |
| **#383** | 2026-05-16 | impl(sprint-11/wave-A): D-CSV-1/3/4 — causal-edge v2 layout + InferenceType signed mantissa + CollapseGateEmission | Sprint-11 Wave A landing. **D-CSV-1** `causal-edge` crate v2 layout (signed mantissa, W-slot 6 bits per OQ-CSV-2, lens, drop temporal); feature-gated via `causal-edge-v2-layout`; crate bumped 0.1.0 → 0.2.0. **D-CSV-3** `InferenceType` signed-mantissa expansion absorbing PR-LL-1 Intervention/Counterfactual into Reserved5/6 of the canonical edge enum. **D-CSV-4** `CollapseGateEmission` wire format in contract crate (Vec instead of SmallVec to preserve zero-dep — TD-COLLAPSE-GATE-SMALLVEC-1 tracks the optimization). Merge commit `03bd175`. |
| **#372** | 2026-05-14 | specs(sprint-10): 12-worker CCA2A fleet + meta-review (governance only) | Sprint-10 spec sprint mirroring PR #365 pattern (specs precede a separate implementation wave). **38 .md files / ~580 KB / zero .rs changes.** 11 PR-ready worker specs (~370 KB) covering par-tile crate apex, CausalEdge64 v2 layout, BindSpace E/F/G/H columns, AriGraph SPO-G + ghost edges + SpoWitnessChain, MailboxSoA + AttentionMaskActor, SigmaTierRouter + banding + plasticity + KernelHandle cache, bevy cull plugin, ndarray Miri completion, sprint-10 execution plan, PR dep graph, unified test plan. Opus meta-review (~28 KB) with sprint grade B+, 6 cross-spec inconsistencies (CSI-1..6), 5 cross-cutting epiphanies (E-META-1..5), sprint-11 spawn decision = NO until 5 spec patches + 4 user ratifications. 8 knowledge docs (~123 KB) documenting: **dual `CausalEdge64` finding** (SPO-palette variant in `causal-edge` crate ≠ 8-channel cascade variant in `thinking-engine` crate, same name different bit semantics); **p64 drift origin** pinpointed at `crates/lance-graph-planner/src/cache/convergence.rs:18-22 #[allow(unused_imports)]`; **three-zone hot-path model** (Zone-1 thinking-engine MatVec 200-500ns + AriGraph entity_index O(1), Zone-2 blasgraph+neighborhood cascade 20-1200µs, Zone-3 DataFusion >1ms); **SPOW tetrahedron + ontology-aware splat vision**; **5-sprint reunification arc** to unify thinking-engine + cognitive-shader-driver SoA. **Deferred:** sprint-11 implementation wave, `Think` carrier struct unification (sprint-12+), splat shader op fleet (sprint-13+), OWL DOLCE / OntologyFilter wiring (sprint-12+), PR-J1-INT4-32D-ATOMS as Wave 0.5 prerequisite. |
| **#366** | 2026-05-13 | impl(sprint-7): 7-worker implementation wave + AuditSink trait unification | Sprint-7 CCA2A 6-parallel + 1-sequenced + 1-Opus-meta. **~5 KLOC across 5 crates + 2 new** (`lance-graph-supervisor`, `lance-graph-consumer-conformance`). Workers: **S7-W1** `parse_family_registry()` + Healthcare basins `0x10..=0x19` (unblocks MedCare-rs E1-2/E1-3/E1-4 cascade); **S7-W2** `lance-graph-contract/build.rs` codegen (zero-dep preserved; sorted-slice + binary_search, no phf — OQ-2); **S7-W3** ractor supervisor with separate 18-byte `LifecycleAuditEvent` (CC-2) + `SuperDomain::System` exempt (CC-3); **S7-W4** `assert_consumer_conformance` harness (A1-A10); **S7-W5** `CognitiveBridgeGate` trait + `UnifiedBridgeGate<B>` impl; **S7-W6** new `audit_sink/` module (`AuditSink` trait + `JsonlAuditSink` + `LanceAuditSink` + `CompositeSink`) + `audit_verify` CLI + `prev_merkle` field on UnifiedAuditEvent (canonical_bytes still 26 B); **S7-W7** SMB Foundry `0x80..=0x82` vs BSON `0xA0..=0xAD` disjoint slots (OQ-4). **Post-meta AuditSink trait unification** (`bc530a4`): dropped legacy `UnifiedAuditSink` D-SDR-4 placeholder, `UnifiedBridge::audit_sink: Arc<dyn AuditSink>`, added `with_jsonl_audit()` ergonomic constructor (OQ-7-2 + OQ-7-3 locked). **Pre-existing workspace lint debt** cleaned by Sonnet janitor across ~30 files in `lance-graph` core / `bgz-tensor` / planner / nsm (sprint-7 outputs guardrailed). **Opus meta verdict** at `.claude/board/sprint-log-7/meta-review.md`: 4A/2B/1B-minus/0 C/D/F. **Adjacent landings:** MedCare-rs sprint-1 10-PR sweep #113-#122 (E1-1 OQ-3 consumed our `0d725d4` decision; sprint-2 5 PRs queued). |
| **#365** | 2026-05-13 | specs(sprint-5-6): 13-worker parallel batch + Opus meta review | Governance-only PR. **13 PR-ready specs at `.claude/specs/`** (~300 KB) from a 12-Sonnet-worker + 1-post-meta-Sonnet-worker + 1-Opus-meta-agent parallel batch. Spec grades: 3 A (W2 d3b-jsonl, W5 pr-graph, W12 conformance), 7 B, 2 C (W10 manifest-modules needs §4.3 sorted-slice rewrite; W11 ractor-supervisor needs LifecycleAuditEvent split). 24 KB Opus meta cross-spec review at `.claude/board/sprint-log-5-6/meta-review.md`. 4 blocking OQs (W3 parser entry, W10 phf vs sorted-slice, W6 Role migration, W13 BSON namespace). CCA2A 12+1+1 pattern validated at scale: ~300 KB of PR-ready output in under an hour wall-clock; 3 workers required respawns for permission denials (settings.json patched for `.claude/board/sprint-log-5-6/**`). |
| **#364** | 2026-05-13 | D-SDR-3/4/5 + sprint-log-4 governance + sprint 5-9 roadmap + codex P1/P2 | Tier-A substrate close: **D-SDR-3** OgitFamilyTable + FamilyEntry codebook (~300 LOC), **D-SDR-4** merkle-chained UnifiedAuditEvent (~460 LOC, AuditMerkleRoot = u64 FNV-1a), **D-SDR-5** authorize_* through Policy::evaluate with audit emission (~300 LOC). **Codex P1 fix** (`3208743`): OwlIdentity widened u8→u16 slot → 3-byte canonical `[family, slot_lo, slot_hi]`; OgitFamilyTable → sparse `HashMap<u16, FamilyEntry>`; UnifiedAuditEvent canonical_bytes 25→26. **Codex P2 fix** (`e23ce89`): emit_audit uses AuditChain.super_domain() instead of static FAMILY_TO_SUPER_DOMAIN. **CI fix** (`a3c753f`): ndarray/hpc-extras opt-in for blake3. Sprint-log-4 governance corpus (12 worker specs + 2 meta reviews) + sprint-5-through-9 roadmap (70 agents = 60W + 10M across 5 sprints, mandatory 12-step plan-read-order in worker prompts). 97/97 callcenter lib tests pass. All 5 CI checks green on `c8176cb`. Adjacent: ndarray#142 (VBMI gate + Inf clamp) merged same day. |
| **#354** | 2026-05-07 | gov: #353 post-merge + cross-repo adjacent-landings | Pure governance close-out. PR_ARC entry for #353 + LATEST_STATE row. Documents the 5-PR coordinated landing across 4 repos: lance-graph #352/#353/#354 + OGIT #2 (woa+medcare bridges unblocked for OGIT-O(1)) + woa-rs #2 (cross-repo `--features ontology` integration) + MedCare-rs #109 (`?source=lance` exercising Zone 2 → Zone 3 rewriter chain). Locks: append-only board hygiene durability across 4 sequential prepends; cross-repo coordinated-landing recipe. |
| **#353** | 2026-05-07 | plan: palantir-parity-cascade v2 + SoA DTO entropy ledger + #352 post-merge governance | Three artifacts. **v2 capstone** (262 lines): integrates 4 prior Foundry parity docs. Pillar 0 carry-forward: Foundry parity IS SoA-as-canon parity. Column H (PR #272 SHIPPED) is already the Foundry Object Type bridge. 15 D-PARITY-V2 deliverables. **SoA DTO entropy ledger** (210 lines, append-only knowledge): 22 DTOs classified across 4 tiers (sensor → engine → contract → callcenter). Buckets: 9 bare-metal / 7 SoA-glue / 6 bridge-projection (3 OPEN). `ResonanceDto` IS the SoA. Codec cascade columns all OPEN today. **#352 post-merge governance**: PR_ARC + LATEST_STATE updates. |
| **#352** | 2026-05-07 | plan: lance-graph-ontology v5 + ogit-cascade v1 | Two-plan PR. **v5** (177 lines): 15 deliverables for ontology crate post-merge follow-on (D-1 dcterms:source, D-2 SpoBridge::promote_to_spo, D-9 ontology-aware MUL thresholds). 4 ratifications (smb-ontology export-only, D-9 above D-2, MulThresholdProfile in lance-graph-contract, OGIT-fork upstream non-PR). **v1 cascade** (209 lines): 15 D-CASCADE deliverables for SoA-as-canon + Zone 1/2/3 + BioPortal arsenal + bridge collapse. **Pillar 0**: OntologyRegistry IS the SoA, schema IS the DTO + name→row index. **Codec cascade per row** (target state, NOT YET WIRED — D-CASCADE-V1-7): identity Vsa16kF32 → CAM-PQ 6 B → Base17 34 B → palette key 4 B → Scent 1 B + qualia 18×f32 + meta 8 B + edge 8 B, every step O(1). |
| **#243** | *(open)* | D5+D7 categorical-algebraic inference | `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery (295 LOC, 14 tests), `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). Plans: `categorical-algebraic-inference-v1.md` (496 lines). Knowledge: `paper-landscape-grammar-parsing.md`, `session-2026-04-21-categorical-click.md`. CLAUDE.md § The Click (P-1). 12 epiphanies. |
| **#225** | *(open)* | Codec-sweep plan + D0.6/D0.7 CodecParams | 9-commit plan (`codec-sweep-via-lab-infra-v1.md`, Rules A-F, 9 starter YAMLs, CODING_PRACTICES audit) + `lance-graph-contract::cam` CodecParams/Builder/precision-ladder validation (14 tests). 147/147 contract suite |
| **#224** | 2026-04-20 | lab = API+Planner+JIT, thinking harvest, I11 measurability | `lab-vs-canonical-surface.md` extended: three-part lab stack (API + Planner + JIT), thinking-harvest subsection (REST/Cypher → `{rows, thinking_trace}` = the AGI magic bullet), I11 invariant (every layer L0→L4 emits harvest-ready trace; no black-box short-circuits) |
| **#223** | 2026-04-20 | LAB-ONLY firewall + AGI-as-SoA + I1-I10 | `lab-vs-canonical-surface.md` initial doc: canonical consumer = `UnifiedStep`/`OrchestrationBridge`, Wire DTOs are lab quarantine. AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming cognitive-shader-driver. 10 cross-cutting invariants I1-I10 (BindSpace read-only, canonical `simd::*` import, temporal budgets, temperature hierarchy, thinking IS AdjacencyStore, weights are seeds, per-cycle cascade, 4096 surface, three DTO families, HEEL/HIP/BRANCH/TWIG/LEAF) |
| **#210** | 2026-04-19 | Phase 1 grammar + knowledge docs | ContextChain reasoning ops, role_keys slice catalogue, 3 knowledge docs (grammar-landscape, linguistic-epiphanies E13-E27, fractal-codec) |
| **#209** | 2026-04-19 | sandwich layout + bipolar cells | Crystal fingerprint sandwich, VSA_permute reference, lossless bundling corrections |
| **#208** | 2026-04-19 | grammar + crystal + AriGraph unbundle | Contract grammar/ + crystal/ modules, AriGraph episodic unbundle hooks with SIMD dispatch |
| **#206** | 2026-04-18 | state classification pillars | qualia.rs (17D), proprioception.rs (7 anchors), world_map.rs, sigma_rosetta 64 glyphs + 144 verbs, Pumpkin NPC example |
| **#205** | 2026-04-18 | engine bridge + CMYK/RGB qualia | engine_bridge.rs, 12-style unified mapping, 17D vs 18D qualia distinction |
| **#204** | 2026-04-18 | cognitive-shader-driver | New crate, ShaderDispatch/Resonance/Bus/Crystal DTOs, BindSpace struct-of-arrays, full ladybug-rs import |

## Current Contract Inventory (lance-graph-contract)

> **2026-06-26 — ADDED (Phase 1 COMPLETE — FMA-V3 + CPIC-V3 mints + Genetics domain)**: the two remaining V3 identity classes that close Phase 1. `NodeGuid::{CLASSID_FMA_V3 = 0x1000_0A01, CLASSID_CPIC_V3 = 0x1000_0E00}` + `ReadMode::{FMA_V3, CPIC_V3}` (both `{V3, Compressed, CoarseOnly}`) + `BUILTIN_READ_MODES` entries, all gated `guid-v3-tail` — mirroring the OSINT-V3 (#613) pattern: the `0x1000` gen-marker in the HIGH u16, canon domain preserved in the LOW u16 so `classid_concept_domain` still routes (`0x0A01 → Anatomy`, `0x0E00 → Genetics`). **NEW Genetics domain `0x0E`** in `ogar_codebook::ConceptDomain` (+ `0x0E => Genetics` route, parity test pins `0x0E00 → Genetics`) — **operator-allocated 2026-06-26** (`0x0D` was already HR); mirror target `ogar_vocab::ConceptDomain::Genetics` (OGAR catches up under the drift guard). **Genetics framing (operator directive, `I-VSA-IDENTITIES` Test-0):** the 6 V3 basins are genomic **mereology, not labels** — a gene's identity is its *position* in the part-of hierarchy (genome → chromosome → region → locus → gene), readable as HHTL `(X;Y)` coordinates per `(part_of:is_a)` tile; the human genome is the **fixed schema view** (hence `Compressed`, a fixed reference frame), and Phase 2 shapes gene **expression as the coordinate value**. The Phase-1 V3 set (OSINT + FMA + CPIC) is now **complete → unblocks the atomic Canon:Custom flip + Phase 2** (plan §2.3 sequencing). Confirm test `read_mode_fma_v3_and_cpic_v3_route_their_domains` (gated `guid-v3-tail`): both route their domain, resolve `tail_variant == V3`, distinct classids. Additive, layout-preserving, default-V1; **739** lib green default / **750** `guid-v3-tail`, clippy `--all-targets -D warnings` + fmt clean. Plan `soa-value-tenant-migration-v2.md` §2.2 (FMA/CPIC rows wired; Genetics `0x0E`). Branch `claude/fma-cpic-v3-mint`.
>
> **2026-06-25 — ADDED (Phase 1 identity→V3, the `mint_for` tail-variant carrier)**: `lance_graph_contract::canonical_node::NodeGuid::mint_for(tail_variant, classid, heel, hip, twig, leaf, family, identity)` (`const`, feature `guid-v2-tail`) — the **key-side symmetric spine** of `soa-value-tenant-migration-v2.md` §2.1: a consumer mints its identity BY ITS CLASSID's tail (`mint_for(classid_read_mode(c).tail_variant, …)`), never hardcoding `new` vs `new_v2` — the exact analog of the Phase-2 value-side `to_node_row(classid_read_mode(c).value_schema, …)`, same `classid_read_mode(c)` lookup, sibling field. Migrating a class's identity to V3 becomes a one-line `tail_variant` flip in the registry, zero consumer rewrite ("extend the one `ReadMode`, never a public `new_v3`"). Dispatch: `V1 → new` (u24·u24 tail; `leaf` ignored — V1 has no LEAF tier), `V2 | V3 → new_v2` (the shared `leaf·family·identity` 3×u16 tail — V3 differs only in how the bytes are *read*, the `(part_of:is_a)` tile, not how they are *stored*, so it mints through the same constructor). **No silent truncation** (the footgun v2 removes): the V2/V3 arm `assert!`s `family`/`identity` fit `u16`, mirroring `new`'s own 24-bit guard. **`Cargo.toml`: `guid-v3-tail = ["guid-v2-tail"]`** — V3's mint path dispatches to `new_v2`, so the tail constructor must exist whenever a V3 classid can be minted (honest gating per `I-LEGACY-API-FEATURE-GATED`). **End-to-end confirm** (`mint_for_osint_v3_is_end_to_end_routable`, gated `guid-v3-tail`): mint OSINT-V3 via the carrier → `read_mode().tail_variant == V3` → `from_guid_prefix_v3` routes non-empty at depth 16 (the full HEEL·HIP·TWIG·LEAF cascade) **while** the v1 `from_guid_prefix` still returns `None` (the Codex-P2 EMPTY-fold is gone, both directions proven) → `decode_v2` reads the tiers back; plus `mint_for_dispatches_to_the_right_constructor_per_tail` (gated `guid-v2-tail`: V1==`new`, V2==V3==`new_v2`). Additive, zero-dep, latent-default-V1 (zero re-mint of the V1/V2 corpus, RESERVE-DON'T-RECLAIM); 737 lib green default / 744 `guid-v2-tail` / 747 `guid-v3-tail`, clippy `--all-targets -D warnings` + fmt clean. Plan: `soa-value-tenant-migration-v2.md` §2 (Phase 1). Branch `claude/identity-v3-mint`.
>
> **2026-06-25 — MODULARIZED (follow-up to #613) — `lance_graph_contract::facet`**: extracted `FacetTier` / `FacetCascade` from `canonical_node` into a dedicated, reusable `facet` module (a *reading*, NOT part of the locked node layout — the cleaner factoring; `canonical_node` re-exports both for the historical path). **Reusable lane API rounded out:** `as_u128`/`from_u128` (single-register view), `rows()` (the 4 dword rows `{domain}{schema}` / `HEEL:HIP` / `TWIG:LEAF` / `family:identity`), `prefix_distance`/`shared_prefix_tiles` (the **granularity-free LCP redout** — `vpxor`+`tzcnt`; 8:8 vs nibble is a free `>>` on the count, measured), `row_match_mask` (`vpcmpeqd`-lane), plus `as_bytes`/`ref_from_bytes` — a **zero-cost reinterpret** (`#[repr(C, align(16))]`; `as_bytes` measured to lower to `mov rax,rdi`, a literal no-op; fields read straight through as single loads). One register → row(`u32`)/tile(`u16`)/prefix(bit)/nibble(Morton) lenses, each one SIMD op (module docs). Lab-test write-up deferred. Additive, zero-dep; 741 lib green (default + `guid-v3-tail`), clippy `-D warnings` + fmt clean. EPIPHANIES `E-FACET-8-8-ALWAYS`. Branch `claude/facet-module`.
>
> **2026-06-25 — ADDED (#613, the 6-tier 8:8 homogeneous facet + V3 routing fold)**: `lance_graph_contract::canonical_node::{FacetTier, FacetCascade}` — the **ALWAYS-8:8** content-blind facet substrate. `FacetTier{lo, hi}` (2 B, `const`; `as_u16` concatenated + `morton` 2bit×2bit Morton-tile projections); `FacetCascade{facet_classid: u32, tiers: [FacetTier; 6]}` (16 B = `facet_classid(4) | 6×(8:8)=12`, harvest §5.1) — a *reading* over a borrowed `[u8;16]` with `from_bytes`/`to_bytes`/`hi_chain`/`lo_chain`/`hi_distance`/`lo_distance`. **Carries NO value-slab offset** → does NOT touch the operator-LOCKED 480 B layout (the `classid→ClassView` byte-pick is the separate, panel-gated step); content-blind — only the consumer projects meaning (`part_of:is_a` / 256:256 palette centroid / `group:member` / `column:row` / concatenated u16 …), every reading amortizing to one 2bit×2bit Morton tile cascade. **Key-side V3 routing:** `hhtl::NiblePath::from_guid_prefix_v3` (feature `guid-v3-tail`) folds the 4 HHTL tiers `HEEL·HIP·TWIG·LEAF` in FULL (both bytes, depth 16) — the facet's routing prefix; `family`/`identity` stay the basin tail. `classid` NOT folded, so `soa_graph::hhtl_path` (schema-driven by `tail_variant`) routes OSINT-V3 `0x1000_0700` non-empty — fixes the Codex-P2 latent EMPTY-fold. `from_guid_prefix`'s "reserved-zero" doc/guard scoped to **v1-fold** (NOT a global classid law). Additive, zero-dep; 739 lib green (default + `guid-v3-tail`), clippy `-D warnings` + fmt clean. EPIPHANIES `E-FACET-8-8-ALWAYS`. Branch `claude/p-a-readmode-tail-variant`.

> **2026-06-21 — ADDED (content-store for the AriGraph/OSINT episodic arc)**: `lance_graph_contract::content_store::{ContentId, SourceSpan, ContentError, ContentStore, ContentSink}` — the content-addressed **cold text/blob store** contract. `ContentId(u64)` = `hash::fnv1a` of the bytes (stable across versions — the correct content address; `DefaultHasher` must never key one; `0` = sentinel). `SourceSpan{ContentId,u32,u32}` = the fixed-size, `Copy` typed form of `template-equivalence`'s `(source_id,start,end)` provenance; `is_cited()` = "no source span → no claim" (non-sentinel content + non-empty span). `ContentStore` (cold read: `resolve(id) -> Option<&[u8]>` zero-copy slice into the mmap/backing store; `resolve_span`/`contains` defaulted) + `ContentSink` (idempotent `put -> ContentId`, dedup by content-address: many episodes → one source row). **Hot/cold firewall (ADR-022)**: the hot path (SIMD sweep, AriGraph edge traversal) touches only the fixed-size `ContentId`/`SourceSpan`; bytes hydrate cold at the membrane (the fingerprint is the hot-path stand-in for text). Nothing variable-length enters the 512 B node. Additive, zero-dep; +6 tests (stable/dedup, idempotent put, resolve_span slice, OOB/missing errors, uncited-rejected); clippy clean. Consumers: `rs-graph-llm/episodic-arc-task` (replaces its local fnv1a), `template-equivalence` (typed provenance). Plan: `.claude/plans/arigraph-osint-episodic-v1.md` (D-CC-ARI-3). Branch `claude/content-store-contract-draft`.

> **2026-06-18 — ADDED (probe-excel-compute-dag-v1 Inc 0, the `compute_dag` Core gap)**: `lance_graph_contract::class_view::{ComputeEdge, compute_dag_is_acyclic}` + `ClassView::compute_dag(class) -> &[ComputeEdge]` (default `&[]`, zero-fallback). `ComputeEdge {target: u8, inputs: &'static [u8]}` is the harvest-sourced recompute edge (`emitted_by` target ← `depends_on` inputs; field positions index the class `FieldMask`), `const`-constructible like `MethodSig`/`ActionDef` (the harvest IS the manifest). `compute_dag_is_acyclic` is the **registry-build gate** — a cyclic recompute DAG (formula loop / `@api.depends` cycle / self-loop) is rejected at build (Kahn over ≤64 positions, allocation-free; out-of-range positions ignored, no panic, mirrors `FieldMask::from_positions`). This is the Core home for computed-field recompute *dispatch* that EVERY computed-field AR consumer needs (Odoo `@api.depends`, Excel formulas, medcare lab-trends, woa calc, q2 cells — they reduce to a sheet; `E-EXCEL-SHADER-PROJECTION`) and the NNUE-incremental existence-proof shape (`E-CHESS-TENSOR-PROVEN`). **Layout-preserving**: a default trait method + a free fn, resolution metadata ABOVE the SoA, stores nothing on the row, zero `NODE_ROW_STRIDE`/`ENVELOPE_LAYOUT_VERSION` impact (core-gap-auditor's EXTEND-CORE, never an adapter-state hack). The instance recompute that consumes it is gated per-cell by the cycle-aware `write_row` (`E-SOA-CYCLE-OWNERSHIP`). Additive, zero-dep; +4 tests (default-empty, acyclic-chain, cycle/self-loop/3-cycle rejected, out-of-range ignored); 10/10 class_view, clippy/fmt clean. Sibling `ClassView::constraints` (`validation_kind`-sourced) deferred to Inc-follow-up. Plan: `.claude/plans/probe-excel-compute-dag-v1.md`. Branch `claude/particle-wave-click-epiphany`.

> **2026-06-18 — ADDED (D-DO-ARM-1, the OGAR DO arm)**: `lance_graph_contract::action::{ActionState, StateGuard, ActionDef, ClassActions, actions_for, effective_actions, ActionInvocation}` — the Perdurant DO arm completing the OGAR IR (the action-axis sibling of `codegen_manifest`'s `MethodSig`/THINK). Both the 4-agent `sale_order` AR→DO probe (runtime-archaeologist) AND the merged cross-repo PR survey (ruff/OGAR/lance-graph/openproject/tesseract) agreed this was the ONE missing wire: the THINK arm (`classid → ClassView`, `has_function → MethodSig`) is converged + merged; the DO-arm `ActionInvocation`/`ActionDef` type was ABSENT. **`ActionDef`** (static, `const`-constructible, all `&'static`/`Copy`): `predicate` (= harvested `has_function` method), `object_class` (classid), `exec` (`ExecTarget` incl `SurrealQl`), `guard` (`StateGuard` = KausalSpec field==value), `required_role` (RBAC), `overrides` (OGAR `classid→ClassView` inheritance). **`ClassActions`+`actions_for`** (zero-fallback) mirror `ClassMethods`/`methods_for`. **`effective_actions(parent, child)`** = OGAR inheritance on the action axis (child overrides parent by predicate). **`ActionInvocation`** (dynamic, `Copy`): lifecycle `ActionState{Pending→Committed|Failed|Cancelled}` (sticky terminals), S2.5 `cycle` stamp, idempotency/trace keys, HLC `emitted_at_millis`. **`ActionInvocation::commit(def, actor, impact, now)`** is the gated egress — RBAC FIRST (`auth::ActorContext` must hold `required_role` or be admin → else `Failed`), THEN MUL impact (`mul::GateDecision`: `Flow→Committed`+stamped, `Hold→`Pending/escalate, `Block→Cancelled`). This IS "commit to the external consumer (odoo/openproject/woa/tesseract) after the cycle decides sound." Dispatched via `UnifiedStep`/`ExecTarget`, NOT a per-crate endpoint. Additive, zero-dep. +5 tests green. Consumer reference: `docs/OGAR_CONSUMER_API.md`. Branch `claude/soa-write-deinterlace-inc2`.

> **2026-06-20 — ADDED (D-UNICHARSET-DIR-MIRROR, the bidi-direction + mirror leaf)**: `lance_graph_contract::unicharset::UniCharSet` gained `get_direction(id) -> i32` + `get_mirror(id) -> i32` + `dump_direction()` + `dump_mirror()`, backed by `directions: Vec<i32>` + `mirrors: Vec<i32>`. The two columns after `other_case`, read by continuing the per-line token walk (the bbox+stats CSV is one whitespace token → fixed offsets across all 5 column tiers; no bespoke tier detector). `direction` = ICU `UCharDirection` code, load default `U_LEFT_TO_RIGHT` 0, out-of-range → `U_OTHER_NEUTRAL` 10 (`unicharset.h:712`). `mirror` clamped like other_case, out-of-range → -1 (`unicharset.h:721`). **Byte-identical 112/112 each** vs tesseract's own `get_direction`/`get_mirror` on real `eng.lstm-unicharset` (self-validating oracle; direction 6 distinct values, mirror 10 bracket pairs). Additive, zero-dep. +3 tests (26 unicharset total). Consumed by `tesseract-core::CharSet::{get_direction,get_mirror}`. EPIPHANIES `E-CPP-PARITY-6`; sixth leaf of `PROBE-OGAR-ADAPTER-UNICHARSET`; first to read past the bbox CSV. Remaining sub-leaf: the float stats inside the CSV. Branch `claude/happy-hamilton-0azlw4`.

> **2026-06-20 — ADDED (D-UNICHARSET-OTHERCASE, the case-pair leaf)**: `lance_graph_contract::unicharset::UniCharSet` gained `get_other_case(id) -> i32` + `dump_other_case()`, backed by `other_cases: Vec<i32>`. The case-paired unichar id (`'C'`→`'c'`), parsed as the token after the script and clamped at load (`unicharset.cpp:901`: a value `>= size`, and the absent default = size, fold to the id itself). Out-of-range id → `INVALID_UNICHAR_ID` -1 (`unicharset.h:703`). **Byte-identical 112/112** vs tesseract's own `get_other_case` on real `eng.lstm-unicharset` (self-validating oracle `other_case` mode; 60 self / 52 pairs). Additive, zero-dep. +4 tests (23 unicharset total). Consumed by `tesseract-core::CharSet::get_other_case`. EPIPHANIES `E-CPP-PARITY-5`; fifth leaf of `PROBE-OGAR-ADAPTER-UNICHARSET`; the last field reachable by token-offset (direction/mirror/bbox need the multi-tier parser). Branch `claude/happy-hamilton-0azlw4`.

> **2026-06-20 — ADDED (D-UNICHARSET-SCRIPT, the script-table leaf)**: `lance_graph_contract::unicharset::UniCharSet` gained `get_script(id) -> i32` / `get_script_table_size()` / `script_from_script_id(sid) -> Option<&str>` / `script_of(id) -> Option<&str>` / `dump_script()`, backed by new `script_ids: Vec<i32>` + an interned `scripts: Vec<String>`. The first leaf to transcode an **interning side-table** (`add_script`, `unicharset.cpp:1063`): `null_script` "NULL" seeded at sid 0 (the `unichar_insert` set_script, `unicharset.cpp:680` → `null_sid_ == 0`), real scripts intern from 1 in id order. Script name = token after the optional bbox/stats CSV (mixed-tier safe). Out-of-range → `null_sid_` 0 (`unicharset.h:681`). **Byte-identical 112/112** vs tesseract's own `get_script` on real `eng.lstm-unicharset` (self-validating oracle `script` mode; table `["NULL","Common","Latin"]`). Additive, zero-dep, behaviour-preserving on the bijection. +4 tests (19 unicharset total). Consumed by `tesseract-core::CharSet::{get_script,script_of}`. EPIPHANIES `E-CPP-PARITY-4`; fourth leaf of `PROBE-OGAR-ADAPTER-UNICHARSET`. Branch `claude/happy-hamilton-0azlw4`.

> **2026-06-20 — ADDED (D-UNICHARSET-PROPS, the property-accessor leaf)**: `lance_graph_contract::unicharset::UniCharSet` gained the character-category surface `get_isalpha` / `get_islower` / `get_isupper` / `get_isdigit` / `get_ispunctuation` / `get_isngram` + `dump_properties()`, backed by a new `props: Vec<u8>` parsed from the per-line hex bitmask (`unicharset.cpp:824`; masked to `ISALPHA=0x1 ISLOWER=0x2 ISUPPER=0x4 ISDIGIT=0x8 ISPUNCTUATION=0x10`). Accessors mirror the C++ inline guard (`unicharset.h:497+`): out-of-range id → `false` (`INVALID_UNICHAR_ID`); `get_isngram` is always-false on the plain-table load path (`unicharset.cpp:893`). **Byte-identical 112/112** vs tesseract's own `get_is*` on real `eng.lstm-unicharset` (self-validating oracle: bijection half cross-checks the 5.5.0-header/5.3.4-lib layout, then the property half diffs 0). Additive, zero-dep, behaviour-preserving on the existing id↔unichar bijection (lenient default-0 for a missing/!hex token). +5 tests (15 unicharset total). Consumed by `tesseract-core::CharSet::get_is*`. EPIPHANIES `E-CPP-PARITY-3`; the third leaf of `PROBE-OGAR-ADAPTER-UNICHARSET` (after D-UNICHARSET-1 + D-UNICHAR-1). Branch `claude/happy-hamilton-0azlw4`.

> **2026-06-18 — ADDED (D-UNICHARSET-KEYSTONE, classid → ClassView → adapter wiring)**: `lance_graph_contract::unicharset_adapter::{UniCharSetStore, UniCharCall, UniCharOut, DispatchError, invoke_unicharset}` — steps 2–3 of `PROBE-OGAR-ADAPTER-UNICHARSET`, the keystone composing the proven `UniCharSet` adapter through the OGAR Core's three movable parts. `invoke_unicharset(registry, store, classid, call)`: (1) **ClassView composition gate** — `codegen_manifest::methods_for(registry, classid)` must list the call's method (the harvested `has_function` manifest), else `MethodNotComposed` (zero-fallback: an unconfigured classid composes nothing); (2) **content-store tier** — `UniCharSetStore::unicharset(classid)`, a consumer-provided trait (dependency-inverted like `ClassView`/`PlannerContract`; the adapter holds NO state — `I-VSA-IDENTITIES`); (3) **adapter leaf** — routes to `UniCharSet::{id_to_unichar, unichar_to_id}`. DO-in (`UniCharCall`) / DO-out (`UniCharOut`, zero-copy borrow). **Byte-parity inherited** from `UniCharSet` (112/112); the keystone proves the dispatch path is faithful (the `NULL`→space edge survives it), the gate works, and there is **no Core gap** (the doctrine's iron guard holds — the variable-length bijection rides the content tier cleanly). NOT routed through the heavy `OrchestrationBridge` (cross-subsystem router); this is the adapter-invocation primitive a `UnifiedStep` calls. Additive, zero-dep. +5 tests; clippy `--all-targets -D warnings` + fmt clean. Completes the core-first doctrine END-TO-END for the unicharset leaf (`E-CPP-KEYSTONE-1`).

> **2026-06-17 — ADDED (D-UNICHAR-1, SECOND byte-parity adapter)**: `lance_graph_contract::unichar::{utf8_step, utf8_to_utf32}` — the Tesseract `UNICHAR` UTF-8 codec that `UNICHARSET` sits on top of (`ccutil/unichar.cpp`). `utf8_step(lead) -> u8` is a `const fn` transcription of Tesseract's 256-entry lead-byte table (1/2/3/4 for legal leads, 0 for continuation bytes `0x80..=0xBF` + `0xF8..`); `utf8_to_utf32(bytes) -> Option<Vec<i32>>` mirrors `UNICHAR::UTF8ToUTF32` (lead-byte validation only, `None` on an illegal lead). **The second adapter through the transcode pipeline, byte-parity proven**: `examples/unichar_dump.rs` vs a libtesseract `UNICHAR` oracle is **268/268 identical** (256 EXHAUSTIVE `utf8_step` lead-byte values + 12 `utf8_to_utf32` corpus rows). Faithful-transcode note (the point of the exercise): Tesseract maps `0xC0`/`0xC1` to step 2 and decodes the overlong NUL `C0 80` to `[0]`; `core::str::from_utf8` REJECTS both, so a native-UTF-8 shortcut would silently diverge — mirroring the exact table is mandatory (`from_utf8_rejects_what_tesseract_accepts` test pins it). Additive, zero-dep, pure text (no leptonica). +8 tests + the `unichar_dump` example; 653 contract lib green; clippy `--all-targets -D warnings` clean. Sibling of D-UNICHARSET-1, same `PROBE-OGAR-ADAPTER-UNICHARSET` falsifier family (E-CPP-PARITY-2).

> **2026-06-17 — ADDED (D-UNICHARSET-1, byte-parity probe Rust side)**: `lance_graph_contract::unicharset::{UniCharSet, UniCharSetError}` — the Tesseract `UNICHARSET` content-store tier (the Core-First doctrine's variable-length classid-keyed registry, `deepnsm::Vocabulary`-shaped: `reverse: Vec<String>` id→unichar + `lookup: HashMap<String,u32>` unichar→id). `load_from_str`/`load_from_file` parse the `.unicharset` text format (line 1 = count, then the first whitespace token per line = unichar, id = position; property columns ignored — the `old_style_included_` plain-table scope); `id_to_unichar`/`unichar_to_id` are the two adapter leaves; `dump()` renders the `<id>\t<unichar>` table matching the C++ oracle. **The Rust side of `PROBE-OGAR-ADAPTER-UNICHARSET`** — pure text parsing, ZERO leptonica (the unicharset path never touches `Pix`), so it builds + unit-tests in-env; byte-parity is one `diff` against a libtesseract oracle harness on a leptonica-installed box (steps in `examples/unicharset_dump.rs`). Additive (a sibling content-store module, zero `NodeRow`/tenant impact). +4 tests (format parse, bijection round-trip, oracle-shape dump, typed errors) + the `unicharset_dump` example; 644 contract lib green; clippy `-D warnings` clean. Plan: `transcode-extend-core-probe-v1.md` (the deferred Option A content-store tier, now built for the probe). The classid→`&UniCharSet` `LazyLock` resolver remains the wiring follow-up.

> **2026-06-17 — ADDED (D-CPP-CODEGEN-1, C-FIRST step 2 compile target)**: `lance_graph_contract::codegen_manifest::{MethodSig, ClassMethods, methods_for}` — the Core-side target of the C++ method-resolution manifest emitted by `ruff_cpp_codegen` (the Tesseract AST-DLL pipeline's stage 2). `MethodSig` is the dispatch-relevant signature in a **`const`-constructible** shape (all fields `&'static`: `name`, `params: &'static [&'static str]`, `ret`, `is_const`, `is_static`, `overrides`) — the method-axis sibling of `class_view::ClassView`'s field projection, deliberately NOT `String`-backed (a generated `const X: &[MethodSig] = &[MethodSig { .. }]` must compile; `FieldRef` is `String`-backed and cannot). `ClassMethods{classid, methods}` is the registry ENTRY the generated code emits (classid bound OGAR-side, never minted here); `methods_for(registry, classid) -> &'static [MethodSig]` is the pure lookup with zero-fallback (unregistered classid → empty slice). **Additive** (container-architect ADDITIVE-CONFIRMED): a sibling module, zero `NodeRow`/`ValueTenant`/`ValueSchema`/stride/`ENVELOPE_LAYOUT_VERSION` impact; the runtime `classid→methods` registry DATA lives downstream (generated in the consumer repo), not here. Body-shaping flags (pure-virtual/constexpr/noexcept/operator/requires) are out of scope (they drive body generation, not the signature manifest). The 8-agent step-2 council's deferred-runtime-registry resolution. +2 tests (const-constructibility proof + zero-fallback lookup); 640 contract lib green; clippy `-D warnings` clean. Plan: `.claude/plans/transcode-extend-core-probe-v1.md` (C step 2). Consumer: `ruff_cpp_codegen::render` (AdaWorldAPI/ruff) names this type in emit-text-only output.

> **2026-06-16 — ADDED (4-task unblock-cascade)**: `lance_graph_contract::hhtl::NiblePath::{from_guid_prefix(&NodeGuid) -> Option<NiblePath>, prefix(depth: u8) -> Option<NiblePath>}` — the ontology-side keystone follow-up of #498's `classid → ReadMode` LE contract. The 20-nibble `classid · HEEL · HIP · TWIG` prefix is deterministically folded to 16 (the canon-reserved high `u16` of classid drops); returns `None` when the fold would be lossy (callers don't get silent collisions). `prefix(d)` is the O(1) single-shot ancestor view that satisfies `prefix(d).is_ancestor_of(self)` for every `d ≤ self.depth` — the routing-cache view of a deeper class path. **One layer up** in `cognitive-shader-driver::MailboxSoA<N>`: `impl MailboxSoaView + MailboxSoaOwner` (cherry-pick of `jolly-cori-clnf9::463d71b`) + the `pub phase: KanbanColumn` field — the in-RAM Rubicon owner the contract's `MailboxSoaOwner` had no real implementor for (integrated-cognitive-planner-v1 §2 Seam #3 closed). In `lance_graph::graph::scheduler`: `LanceVersionScheduler<S = NextPhaseScheduler>` — D-MBX-9-IN core impl over `VersionedGraph::versions()`, generic over the inner `VersionScheduler` policy (closes `E-SUBSTRATE-IS-THE-SCHEDULER`'s OUT-direction). In `surreal_container::view`: `SurrealMailboxView<'a>` + `read_via_kv_lance()` (D-PG-6 contract slice) — the SurrealQL read-glove the integrator wires once the cold-build of the surrealdb fork is taken; the contract surface is available today. Plus `SurrealContainerError::BlockedColdBuild` — typed signal for callers to pattern-match the cold-build gate (distinct from the pre-existing `Blocked` variant which signals coordinate/API gaps). Zero-dep contract additions (+7 hhtl tests, 632 lib green); cognitive-shader-driver +1 driving-loop test (86 lib green); lance-graph::scheduler new module (+5 tests, real tempdir Lance); surreal_container::view new module (+4 tests). All four green; clippy `-D warnings` clean on the new files. EPIPHANIES `E-UNBLOCK-CASCADE-1` records the convergence of three independent landings onto the single `MailboxSoaView` trait surface.

> **2026-06-09 — ADDED (D-IDENTITY-1, Phase A of identity-architecture)**: `lance_graph_contract::identity::{NodeGuid([u8;16]), IDENTITY_LAYOUT_VERSION}` — the workspace's first **stable binary instance identity**: a structured 128-bit UUIDv8 (RFC 9562) = the HHTL nibble-address **formalized + namespaced**. **Composed from existing committed scalars, never re-invented** (Agent A sweep confirmed the 128-bit id space was empty): octets carry `namespace:u8 | entity_type:u16 | kind:u8` (the `SchemaPtr.packed` convention) ⊕ a truncated `NiblePath` routing prefix (`PREFIX_NIBBLES=4`) ⊕ a 22-bit `shape_hash` (truncated `StructuralSignature`) ⊕ a 24-bit `local`, with UUIDv8 version(=8)/variant(=0b10) at their RFC-fixed positions + an `IDENTITY_LAYOUT_VERSION` stamp. **Eineindeutigkeit**: `entity_type` is the canonical exact class identity; the `NiblePath` prefix is the bijective DERIVED view (a *truncated* prefix can't be the identity — deep classes collide past it; the prefix `is_ancestor_of` the full path). Five readings: resolve (`entity_type`) / route (`niblepath`) / witness (frozen bytes + merkle) / ground-truth (`shape_hash` drift) / dispatch-to-store (`as_bytes` → `EntityKey`). Also added `hhtl::NiblePath::from_packed` (inverse of `packed`). Zero-dep; 599 contract lib tests (+15: field-isolation matrix, UUIDv8 gates, ancestor-prefix invariant, Display=canonical-UUID); clippy `-D warnings` clean; fmt clean. Plans: `identity-architecture-exists-vs-needs-v1.md` (exists-vs-needs map + phases A→H), `cognitive-write-roundtrip-substrate-v1.md`. Epiphany: E-IDENTITY-WHITEBOX-1.

> **2026-05-31 — ADDED (D-EW64-1 + D-VIEW-1, episodic-RISC-spine)**: `episodic_edges::{EpisodicEdges64(u64), EdgeRef{family:u8,local:u16}}` — AriGraph episodic edges, 4x[4-bit family | 12-bit local]: family 0 = intra-basin (inherited, ~98.6% per #444), 1..=15 = cross-family index into the OGIT-class-inherited palette (~1.4%; identities inherited, never on the edge — I-VSA-IDENTITIES). Plus `view_angle::ViewAngle` (4-bit view-schema selector; presence bitmask doubles as attention mask, inherited). Zero-dep; 527 contract lib tests; clippy pedantic+nursery clean. Plan: episodic-risc-spine-v1.md.

> **2026-05-31 — ADDED (D-H2H-1, head2head superposition winner-select)**: `lance_graph_contract::head2head::{Head2Head (judge), WinnerCriterion (DissonanceMin≈infight / SupportSpread≈Raumgewinn / ConfidenceMax / Tempered=default), CompetitionOutcome}`. `Head2Head::select(&Blackboard) -> Option<CompetitionOutcome>` picks the winning competing-expert bid over the existing `a2a_blackboard` (confidence/dissonance/support) — pure read + arg-extremum, **no new identity, copies nothing** (select-don't-duplicate, `I-VSA-IDENTITIES`); `margin` = the dark-horse signal. The *selection* half of head2head superposition; parallel-mailbox *execution* is the CI-gated consumer side. Zero-dep; 516 contract lib tests (+7); clippy pedantic+nursery clean.

> **2026-05-31 — ADDED (D-MBX-9-IN, VersionScheduler contract slice, on `b6e3cc6`/lance7)**: `lance_graph_contract::scheduler::{DatasetVersion(u64), VersionScheduler (trait), NextPhaseScheduler (reference impl)}`. The IN-direction dual of `MailboxSoaOwner` (`E-SUBSTRATE-IS-THE-SCHEDULER`): `on_version<V: MailboxSoaView>(&V, DatasetVersion, ExecTarget) -> Option<KanbanMove>` lowers a Lance `versions()` tick to the next legal Rubicon `KanbanMove`; `NextPhaseScheduler` is the forward-arc reference (Libet `-550ms` anchor on Planning→CognitiveWork, `None` on absorbing). Read-only over the view (**propose-not-dispose**, R1); composes only existing contract types; zero-dep. 509 contract lib tests (+6); clippy pedantic-clean. CI-gated twin = `LanceVersionScheduler` over `VersionedGraph::versions()` via callcenter `LanceVersionWatcher`. Closes D-MBX-9 IN-direction at the type level (OUT twin + core impl remain CI-gated).

> **2026-05-31 — MERGED (#441, D-CLS arc, merge `a77e119`)**: `lance_graph_contract::class_view::{FieldMask (u64 presence bitmask), ClassView (resolver trait), ClassProjection, RenderRow}` + `ClassView::render_rows` (off-bits-skipped). `ClassId = u16` (reuses `soa_view::class_id`). The class meta-DTO **flies ABOVE the agnostic SoA** — labels/shape/DOLCE resolve LATE from the OGIT cache, nothing semantic in the row (C2 presence≠semantics; N3 stable positions; out-of-range mask bits IGNORED not folded — Codex P2). Ontology side: `class_resolver::RegistryClassView` (impls `ClassView` over the live `OntologyRegistry`, DOLCE via `classify_odoo`) + `odoo_blueprint::class_signature::{StructuralSignature, OdooEntity::signature()/object_view() carrier methods, audit, shape_families, curated_entities, corpus_summary}` (deterministic FNV-1a structural-hash group-by, NOT Aerial-cluster). Zero-dep preserved; extends `ontology::{ObjectView,FieldRef,DisplayTemplate}`, reuses `class_id` (no new newtype). 497 contract + 240 ontology lib tests. D-CLS-{FM,RES,SIG,AUDIT,RENDER} all Shipped.
>
> **2026-05-30 — PR-in-flight addition** (D-MBX-A6 Phase 2 — Rubicon lifecycle + ExecTarget): `lance_graph_contract::kanban::{ExecTarget (Native/Jit/SurrealQl/Elixir), RubiconTransitionError}` + `KanbanColumn::{next_phases, can_transition_to}` (the Rubicon lifecycle DAG) + `KanbanMove.exec: ExecTarget` field + `MailboxSoaOwner::try_advance_phase()` (checked lifecycle enforcement → `Result<KanbanMove, RubiconTransitionError>`). Zero-dep; `KanbanMove` still ≤16 B; 489 contract lib tests (+4); downstream cargo-check clean. Lifecycle enforcement + planner exec-target are now contract-level invariants. Resolves the #437 deferred exec-target NOTE. Cross-ref D-MBX-A6 / #437.
>
> **2026-05-30 — PR-in-flight addition** (D-MBX-A6 Phase 1 — planner⟷ractor⟷surreal meta-DTO): `lance_graph_contract::kanban::{KanbanColumn (6: Planning/CognitiveWork/Evaluation/Commit/Plan/Prune), KanbanMove}` — the 4-phase Rubicon kanban transition; `KanbanMove` is `Copy`, ≤16 B, carries `MailboxId` + `witness_chain_position` (R4 pointer) + `libet_offset_us` (−550 ms anchor, D-MBX-8). `lance_graph_contract::soa_view::{MailboxSoaView, MailboxSoaOwner}` — zero-dep **borrow trait** for the transparent zero-copy SoA view (R1 "one SoA never transformed"); `&[T]` column accessors (energy/edges_raw/meta_raw/entity_type) mirror `MailboxSoA::*_at`; the read/owner split makes "view is read-only" structural (surreal implements only the read half). `orchestration::StepDomain::Kanban` variant + `"kanban."` prefix. Consumed via the EXISTING `OrchestrationBridge` (planner emits, ractor owns/drives via `MailboxSoaOwner`, surreal_container projects via read-only `MailboxSoaView`) — NO parallel DTO family (lab-vs-canonical ruling). Contract `[dependencies]` still empty. 485 contract lib tests green (+6 new); `cargo check` clean on planner/cognitive-shader-driver/supervisor (StepDomain variant additive-safe). Consumer impls deferred. See E-SOA-VIEW-IS-A-BORROW; `unified-soa-convergence-v1.md §5+§8.4`.
>
> **2026-05-28 — PR-in-flight addition** (bindspace→mailbox migration wave A1-A4): `lance_graph_contract::witness_table::{WitnessEntry, WitnessTable<N=64>}` — column-type primitive resolving the 6-bit W-slot in `CausalEdge64 v2` into a per-cohort `(mailbox_ref: u32, spo_fact_ref: Option<u64>)` table (`mailbox_ref` carries the full canonical `MailboxId`, NOT a truncated cohort-local index — see PR #427 Codex P2 fix). Zero-dep, 3 unit tests, `WitnessTable::{new, get, set, default}`. Cross-ref: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` §10 (architectural refinements landed in same wave). Also in same wave: `cognitive-shader-driver::MailboxSoA<N>` gains four thoughtspace columns (`edges: [CausalEdge64; N]`, `qualia: [QualiaI4_16D; N]`, `meta: [MetaWord; N]`, `entity_type: [u16; N]`) + 8 row accessors; `ShaderDriver` gains transitional `mailboxes: HashMap<MailboxId, MailboxSoA<1024>>` + `with_mailbox()` builder + `mailbox()` read accessor (sibling-shape, additive — singleton untouched). 457 contract+driver tests pass.

Types that EXIST — do NOT re-propose them:

**`grammar/`**: `FailureTicket`, `PartialParse`, `CausalAmbiguity`, `TekamoloSlots`, `TekamoloSlot`, `WechselAmbiguity`, `WechselRole`, `FinnishCase`, `finnish_case_for_suffix`, `NarsInference`, `inference_to_style_cluster`, `ContextChain` (with coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel), `RoleKey` + 47 `LazyLock<RoleKey>` instances + `Tense` enum + `finnish_case_key / tense_key / nars_inference_key` lookups, **`RoleKey::bind/unbind/recovery_margin`** (slice-masked XOR), **`Vsa10k`** + `VSA_ZERO` + `vsa_xor` + `vsa_similarity`, **`GrammarStyleConfig`** + **`GrammarStyleAwareness`** + `revise_truth` + `ParseOutcome` + `divergence_from`, **`FreeEnergy`** + **`Hypothesis`** + **`Resolution`** (Commit / Epiphany / FailureTicket) + `from_ranked` + thresholds.

**`crystal/`**: `Crystal` trait, `CrystalKind`, `TruthValue`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`, `CrystalFingerprint` (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32 / **Vsa16kF32**), `Structured5x5`, `Quorum5D`, `SentenceCrystal`, `ContextCrystal`, `DocumentCrystal`, `CycleCrystal`, `SessionCrystal`, sandwich layout constants, **`vsa16k_zero` / `binary16k_to_vsa16k_bipolar` / `vsa16k_to_binary16k_threshold` / `vsa16k_bind` / `vsa16k_bundle` / `vsa16k_cosine`** (Click switchboard carrier + algebra, 64 KB, inside-BBB only).

**`cognitive_shader`**: `ShaderDispatch`, `ShaderResonance`, `ShaderBus`, `ShaderCrystal`, `MetaWord` (u32 packed), `MetaFilter`, `ColumnWindow`, `StyleSelector`, `RungLevel`, `EmitMode`, `ShaderSink` trait, `CognitiveShaderDriver` trait.

**`cognitive-shader-driver` BindSpace substrate (2026-04-24)**: `FingerprintColumns.cycle` is now `Box<[f32]>` (Vsa16kF32 carrier, 16_384 f32 per row = 64 KB) — migrated from `Box<[u64]>` (Binary16K). New constant `FLOATS_PER_VSA = 16_384`. New methods: `set_cycle(&[f32])`, `set_cycle_from_bits(&[u64; 256])` (adapter with `binary16k_to_vsa16k_bipolar` projection), `cycle_row() -> &[f32]`. `write_cycle_fingerprint()` API unchanged (takes `&[u64; 256]`), converts internally. `byte_footprint()` for 1 row = 71_774 bytes. Other three planes (content/topic/angle) remain `Box<[u64]>`.

**CausalEdge64 — TWO distinct types in workspace (2026-05-14, PR #372 finding)**: same name, different bit semantics, different consumers. Always qualify by crate when referring to either:
- `causal_edge::CausalEdge64` (`crates/causal-edge/src/edge.rs:60`) — SPO-palette layout: S/P/O palette indices + NARS f/c + Pearl 2³ mask + direction triad + inference type + plasticity flags + temporal index. Consumed by `lance-graph-planner::cache::nars_engine` (NarsTables), `cognitive-shader-driver::BindSpace::EdgeColumn`, AriGraph SPO commit path. The unit of NARS reasoning at cycle-speed.
- `thinking_engine::layered::CausalEdge64` (`crates/thinking-engine/src/layered.rs:45`) — 8-channel cascade: BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS (each 1 byte). Source/target NOT in u64 (carried as tuple key `(target: u16, edge: CausalEdge64)`). Emitted by `TierEngine::emit_causal_edges` after MatVec; consumed by downstream tiers via `apply_edges`. The unit of cognitive-cascade dispatch in the L1 → L2 → L3 thinking pipeline.

Full reference: `.claude/knowledge/causal-edge-64-spo-variant.md` + `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` + `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md`. Reunification path (Option R-3): transcode 8-channel → SPO at thinking-engine L3 commit boundary; see `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md`.

**`escalation`** (D-PERSONA-1, 2026-05-26, branch `claude/splat3d-cpu-simd-renderer-MAOO0`): the escalation+epiphany loop = the boot checklist (a *restore* of ladybug's qualia loop on our SoA — NOT a bespoke verifier). `CollapseHint` {Flow, Fanout, RungElevate} + `fanout_width` / `noise_tolerance` / `rung_delta` (ladybug `detector.rs` formulas); `Archetype` {Guardian, Catalyst, Balanced} + `InnerCouncil::{deliberate, from_signals}` + `is_split(0.7,0.5)` ×1.2 split-amplify → `CouncilVerdict`; `EpiphanyDetector::observe` (sim > baseline×1.5 ∧ window ≥ 4) → `Epiphany`; `GhostEcho` (8 named: Affinity/Epiphany/Somatic/Staunen/Wisdom/Thought/Grief/Boundary — canonical zero-dep home, mirrors `thinking_engine::ghosts::GhostType`, see TD-GHOST-ECHO-DUP-1) + `WisdomMarker` (asymptotic decay → 0.1 floor, never zero); `GateKind` {Hard, Soft} + `ChecklistItem` + `Checklist::{step, mark_red, boot_ready, all_flow, degraded}` (green-flip = Flow + epiphany; let-it-crash = `mark_red` re-escalate). Planner wiring at `lance_graph_planner::mul::escalation::{boot_checklist, verdict_from}` (§2: 6 HARD / 3 SOFT items + a `MulAssessment` → `CouncilVerdict` adapter). 13 tests (10 contract + 3 planner).

## cognitive-shader-driver Wire Surface (lab-only, post D0.1)

Types live in `crates/cognitive-shader-driver/src/wire.rs` behind `--features serve`:

- **`WireCodecParams`** + `WireLaneWidth {F32x16, U8x64, F64x8, BF16x32}` + `WireDistance {AdcU8, AdcI8}` + `WireRotation {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}}` + `WireResidualSpec {depth, centroids}` — serde mirrors of the `contract::cam::*` types from PR #225. `TryFrom<WireCodecParams> for CodecParams` runs the precision-ladder validation (OPQ↔BF16x32, overfit guard, pow2 Hadamard) at ingress BEFORE any JIT compile.
- **`WireTensorView {shape, lane_width, bytes_base64}`** + methods `row(&AlignedBytes, usize)` / `subspace(&AlignedBytes, row, k, sub_bytes)` / `row_count()` / `col_count()` / `row_bytes()` / `element_bytes()` / `decode() -> AlignedBytes`. Per Rule E (Wire surface IS the SIMD surface, object-oriented) + Rule A (stdlib `slice::array_windows::<N>` + `ndarray::simd::*` loaders).
- **`AlignedBytes`** — heap-allocated, 64-byte-aligned owned buffer produced once by `WireTensorView::decode` per Rule F (decode at REST ingress, never inside). Safe Send/Sync; `Drop` dealloc with matching layout.
- **`WireCalibrateRequest`** extended with optional `params: Option<WireCodecParams>` + `tensor_view: Option<WireTensorView>` (new path) alongside legacy fields (`num_subspaces` / `num_centroids` / `kmeans_iterations` / `max_rows`) for back-compat.
- **`WireCalibrateResponse`** extended with `kernel_hash: u64` (= `CodecParams::kernel_signature()` of the executed kernel) + `compile_time_us: u64` + `backend: String` ("amx" | "vnni" | "avx512" | "avx2" | "legacy"; **never "scalar"** — iron rule).
- **`WireTensorViewError {Base64, SizeMismatch, ZeroShape}`** — typed decode errors.

**`proprioception`**: 7 `StateAnchor` (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D `ProprioceptionAxes`, `StateClassifier` trait, `DefaultClassifier`, `hydrate()` softmax-weighted blend.

**`qualia`**: 17-D `QualiaVector`, `qualia_to_state` projection (17→11).

**`world_map`**: `WorldMapDto`, `WorldMapRenderer` trait, `DefaultRenderer`.

**`world_model`**: `WorldModelDto` with `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()` / `is_liminal()`.

**`container`**: `Container = [u64; 256]` (16Kbit = 2KB), `CogRecord`.

**`property`** (new, SMB domain): `PropertyKind` (Required / Optional / Free), `PropertySpec` (predicate + kind + `CodecRoute` + NARS floor), `PropertySchema` (`&'static`-based, const schemas), `Schema` + `SchemaBuilder` (runtime builder: `.required()` / `.optional()` / `.searchable()` / `.free()` / `.validate()`), `CUSTOMER_SCHEMA`, `INVOICE_SCHEMA`. Maps bardioc Required/Optional/Free to I1 Codec Regime Split (ADR-0002).

**`repository`** (new, SMB domain): `EntityStore` + `EntityWriter` + `Batch` + `EntityKey` — Arrow-agnostic row store contract.

**`mail`** (new, SMB domain): `MailParser` + `ThreadLinker` + `ParseHints` + `AttachmentRef` + `PartRef`.

**`ocr`** (new, SMB domain): `OcrProvider` + `PageImage` + `OcrOpts` + `Bbox` + `BlockKind` + `LayoutBlock`.

**`splat`** (new, 2026-05-06): `SplatChannel` (6 variants: Support / Contradiction / Forecast / Counterfactual / Style / Source), `CamPlaneSplat` (q8 amplitude / width / theta_accept + 16-byte witness identity + 8-byte `replay_ref`), `AwarenessPlane16K` (256 × u64 = 2 KB pressure tile), `SplatPlaneSet` (6 channel planes = 12 KB), `CamSplatCertificate` (q8 pressure measurements + replay decision), `SplatDecision` (Proceed / RequireExactReplay / PrefetchOnly / ScenarioOnly / Drop), `TriadicProjection`, `ReasoningWitness64`. Resolves SPLAT-1 row in entropy ledger (Aspirational → Wired stage 1, entropy 4 → 2). Per `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` PR 1. Plan: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

**`tax`** (new, SMB domain): `TaxEngine` + `TaxPeriod` + `PeriodKind` + `Jurisdiction` + `PostingBatchRef` + `RuleBundle`.

**`reasoning`** (new, SMB domain): `Reasoner` + `ReasoningKind` + `ReasoningContext` + `EvidenceRef` + `Budget`.

**`cam`** (extended by PR #225): `CodecRoute` + `route_tensor` (existing), `CamByte`, `CamStrategy`, `DistanceTableProvider` trait, `CamCodecContract` trait, `IvfContract` trait, plus codec-sweep parameter shape — `LaneWidth` (F32x16 / U8x64 / F64x8 / BF16x32), `Distance` (AdcU8 / AdcI8), `Rotation` (Identity / Hadamard{dim} / Opq{matrix_blob_id, dim}), `ResidualSpec {depth, centroids}`, `CodecParams {subspaces, centroids, residual, lane_width, pre_rotation, distance, calibration_rows, measurement_rows, seed}` with `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`, `CodecParamsBuilder` fluent API, `CodecParamsError {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}` — **precision-ladder validation fires at `.build()` BEFORE any JIT compile**.

**`graph_render`** (new, q2 cockpit): `RenderNode`, `RenderEdge`, `InferredConnection`, `Contradiction`, `GraphSnapshot`, `GraphHealth`, `CypherResult`, `CypherValue`, `CypherError`, `EpisodicTrace`, `ShaderEvent`, traits `GraphSnapshotProvider`, `GraphInferenceProvider`, `CypherExecutor`, `EpisodicTraceProvider`, `ShaderEventStream`. Visual render surface for Neo4j/Palantir Gotham cockpit — q2 consumes, lance-graph arigraph produces.

**`a2a_blackboard`**, **`collapse_gate`**, **`exploration`**, **`literal_graph`**, **`orchestration_mode`**, **`jit`**, **`nars`**, **`plan`**, **`orchestration`**, **`thinking`** (36 styles, 6 clusters), **`mul`**, **`sensorium`**, **`high_heel`**.

**Sprint-11/12 D-CSV substrate types (2026-05-16, PRs #383-#389)**:
- `lance-graph-contract::qualia`: `QualiaI4_16D` (16-dim signed-i4, 9× compression vs `[f32; 18]`), `QualiaI4Column` (sibling SoA column in cognitive-shader-driver).
- `lance-graph-contract::collapse_gate`: ~~`CollapseGateEmission`~~ **REMOVED 2026-06-11** (PR #477 three-tier model tombstone commit — zero-copy SoA, no inter-mailbox handoff type; TD-COLLAPSE-GATE-SMALLVEC-1 closed as moot). `MailboxId` / `MergeMode` / `GateDecision` remain.
- `lance-graph-contract::mailbox` / `attention_mask`: `MailboxId` (canonical id type), `MailboxSoA<N>` (SoA mailbox with W-slot + plasticity accumulator + `apply_edges`), `AttentionMaskSoA`, `AttentionMaskActor`, `AttentionMaskBackend` trait.
- `lance-graph-contract::sigma_tier`: `SigmaTierBands`, `SigmaTierRouter` (Rubicon-resonance ΔF + threshold dispatch), `DispatchOutcome`, `RestReason` (Σ-tier crate surface).
- `lance-graph-contract::witness`: `WitnessCorpus` (CAM-PQ-indexed; D-CSV-6a partial in PR #386, full 6b sprint-12), `WitnessEntry`, `WitnessId`, `WitnessIndexHashMap` (anchor + chain invariant).
- `cognitive-shader-driver::bindspace`: `QualiaI4Column` (sibling SoA column, D-CSV-5a).
- `thinking-engine` + `ndarray`: `SplatField` (×2 — one in thinking-engine for Think carrier scalar ops, one in ndarray for vertical streaming).
- `ndarray::hpc::stream` (vertical streaming structs, D-CSV-11 Wave F W-F4/5/6, productization sprint-12): `QualiaI4Row`, `QualiaStream`, `InferenceRow`, `InferenceStream`, `SplatFieldStream` (+ planned `par_*` rayon variants gated behind ndarray `parallel` feature — deferred to sprint-14+).

## Current AriGraph Inventory (lance-graph/src/graph/arigraph/)

4696 LOC shipped, 7 modules:
- `episodic.rs` (210 LOC + unbundle hooks from #208) — `Episode`, `EpisodicMemory`, `unbundle_hardened`, `unbundle_targeted`, `rebundle_cold`, `UnbundleReport`, `RebundleReport`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`
- `triplet_graph.rs` (1064 LOC) — SPO graph, NARS truth, BFS, spatial paths
- `retrieval.rs` (447 LOC) — fingerprint retrieval policies
- `sensorium.rs` (539 LOC) — observation → triplets
- `orchestrator.rs` (1562 LOC) — AriGraph coordinator
- `xai_client.rs` (521 LOC) — xAI enrichment
- `language.rs` (339 LOC) — LM bridge

## Workspace Conventions (locked in CLAUDE.md)

1. **Model policy:** main thread Opus + deep thinking; subagent grindwork → Sonnet; accumulation → Opus; NEVER Haiku.
2. **GitHub reads:** zipball to `/tmp/sources/` + local grep for 3+ reads per repo. MCP only for writes (PR, comments) and single-path reads.
3. **Contract zero-dep invariant:** `lance-graph-contract` has no external crate deps. Do not add any.
4. **Read before Write:** always Read a file before overwriting. Write-over-self without Read is the documented failure mode.
5. **No JSON serialization in types.** Serde stays out of types (debug-only). Wire formats are explicit.
6. **Pumpkin framing** for externally-visible examples (clinical / game-AI disguise for the AGI primitives).

## Active Branches (local at /home/user/lance-graph)

**Sprint-12/13 open work (2026-05-16):**

- `claude/sprint-12-wave-g-fleet` — **PR #390 OPEN** — sprint-12 Wave G follow-on. Lands the remaining D-CSV deliverables not in Wave F: **W-G1** D-CSV-5b QualiaColumn cutover (drop `[f32; 18]`, flip readers to i4); **W-G2** D-CSV-6b full CAM-PQ-indexed `WitnessCorpus` (unbounded, salience decay); **W-G3** batch i4 scalar MUL (paired w/ #388 Wave F); **W-G4** Σ10 Jirak-threshold derivation (D-CSV-15 NEW v2 entry; partial — VAMPE coupled-revival still sprint-13+).
- `claude/sprint-13-preflight-planning` — **work in flight on this branch.** Sprint-13 preflight planning fleet (PP-3/4/5/6 spec drafts for D-CSV-13b/14/16/17). Governance + spec corpus only; no .rs changes on this branch.
- Sibling repo `AdaWorldAPI/ndarray`: **PR #147 merged** (`d867b1c`) — vertical streaming substrate (`QualiaI4Row`, `QualiaStream`, `InferenceRow`, `InferenceStream`, `SplatFieldStream`) ships D-CSV-11. `par_*` rayon variants deferred behind `parallel` feature (sprint-14+).

**Historical (post-#225 era, retained for arc reference):**

- `claude/teleport-session-setup-wMZfb` — shipped PRs #223 / #224 / #225 (LAB-ONLY + AGI-as-SoA + I1-I11 + codec-sweep D0.6/D0.7).
- `claude/deepnsm-grammar-phase1` — Phase 1 PR #210, merged into main.
- `main` — up-to-date post #389 (sprint-12 Wave F + codex P2 follow-on).

## Active Integration Plans

- **`elegant-herding-rocket-v1`** — grammar / NARS / crystal / AriGraph (Phase 1 shipped in #210; Phase 2 queued).
- **`codec-sweep-via-lab-infra-v1`** (NEW 2026-04-20) — JIT-first codec sweep through lab endpoint; 1 upfront rebuild, unlimited candidates afterwards. D0.6 + D0.7 shipped in #225.

## Immediate Next Work

**Queued Work — sprint-13 (specs being drafted in the sprint-13-preflight fleet on this branch):**

- **D-CSV-13b** — SIMD vectorization of D-CSV-8 i4 MUL evaluation. **IN PR (sprint-13/W-I1 salvage)** on branch `claude/sprint-13-w-i1-salvage`. AVX-512F+BW path runtime-dispatched via cached `simd_caps()` (zero ndarray dep); NEON path correctness-only per spec §7; scalar fallback. Bench on Skylake-AVX512 host: 8.7× dk / 7.4× trust / 5.2× flow / 10.2× gate_disc / 3.1× mul_assess at batch 1024 — all SHIP gates met. `#[repr(u8)]` discriminants locked on `DkPosition`/`TrustTexture`/`FlowState` per spec §5 (I-LEGACY-API-FEATURE-GATED). 449 lance-graph-contract tests green including 5 new SIMD-vs-scalar parity tests over 10 sizes.
- **D-CSV-14** — on-Think method migration for D-CSV-12 splat ops (struct-method surface per L-20 lock; depends on D-CSV-11 ndarray streaming PR #147). Spec being drafted by PP-4.
- **D-CSV-16** — NEW sprint-13 entry. Spec being drafted by PP-5.
- **D-CSV-17** — NEW sprint-13 entry. Spec being drafted by PP-3.

**Sprint-14+ Phase F items (Backlog):**

- ndarray `parallel`-feature `par_*` rayon variants for `QualiaStream` / `InferenceStream` / `SplatFieldStream` (work-stealing).
- D-REUNIFY-4/5/6 carryover from causaledge64-mailbox-rename-soa-v1 (splat op fleet on `Think`, par_* method variants, OWL DOLCE / OntologyFilter wiring).

**`codec-sweep-via-lab-infra-v1` Phase 0 remainder (carry-over):**

- **D0.1** `WireCalibrate` + `WireTensorView` (object-oriented, 64-byte-aligned decode) (~180 LOC).
- **D0.2** `WireTokenAgreement` endpoint stub — I11 cert gate (~160 LOC).
- **D0.3** `WireSweep` streaming endpoint + Lance append (~200 LOC).
- **D0.5** `auto_detect.rs` reading `config.json` for `ModelFingerprint` (~140 LOC).
- Four test gates: `kernel_contract_test`, `amx_dispatch_test` (x86_64), `wire_object_surface_test`, `no_internal_serialisation_test`.

**`elegant-herding-rocket-v1` Phase 2 (still queued):**

- **D2** DeepNSM emits `FailureTicket` on low coverage (~150 LOC).
- **D3** Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` (~220 LOC).
- **D5** Markov ±5 bundler with role-indexed VSA (~300 LOC).
- **D7** NARS-tested grammar thinking styles as meta-inference policies (~260 LOC).

## Deferred (do NOT propose these — they're explicitly parked)

**Sprint-12/13 explicit deferrals (2026-05-16):**

- **TD-COLLAPSE-GATE-SMALLVEC-1** — CLOSED 2026-06-11 as moot: `CollapseGateEmission` removed entirely (PR #477 tombstone commit), nothing left to optimize.
- **TD-SIGMA-TIER-THRESHOLDS-1** — Σ10 VAMPE-coupled Jirak-derived threshold refinement (D-CSV-15). Hand-tuned acceptable through sprint-12 per `I-NOISE-FLOOR-JIRAK`; principled Jirak 2016 derivation forwarded to sprint-13+ VAMPE coupled-revival track.
- **ndarray `parallel`-feature `par_*` rayon variants** — productized substrate ships sequentially in PR #147; rayon work-stealing wraps deferred to sprint-14+ behind an opt-in feature gate.

**Long-running parks (pre-existing):**

- CausalityFlow TEKAMOLO extension (modal/local/instrument + beneficiary/goal/source, 9 total) — struct change deferred until after Phase 2.
- D8 story-context bridge, D9 ONNX arc export, D10 Animal Farm validation, D11 bundle-perturb emergence — Phase 3/4.
- Named Entity pre-pass (NER) — biggest OSINT blocker, separate PR.
- FP_WORDS = 160 migration (currently 157) — coordinated ndarray change.
- Crystal4K 41:1 persistence compression.
- 200-500 YAML TEKAMOLO templates per language — future training pipeline.
- Python/TypeScript grammar-stack convergence.

## If You're Tempted to Propose Something

Check this file first. Then check the KNOWLEDGE_INDEX.md for which
docs cover your domain. Then load only those docs. If you're still
uncertain whether something exists, grep the actual source before
proposing a new type.

The fastest way to waste 30 turns is to re-invent what's already in
the contract. This file exists to prevent that.

---

## 2026-05-05 — Recently Shipped backfill (PRs #244–#335)

> The "Recently Shipped PRs" table above stops at #243 (last refreshed 2026-04-21). Roughly 50 PRs have merged since. This section retrofits them.

| PR | Merged | Title | What it added (one-line) |
|---|---|---|---|
| **#335** | 2026-05-05 | Claude/thought cycle soa integration plan | Two new knowledge docs: gaussian-splat-cam-plane-workaround + entropy-budget-codebook-superposition |
| **#330** | 2026-05-01 | docs: add Cursor Cloud specific instructions to AGENTS.md | AGENTS.md section: ndarray path, CI commands, fmt-drift inventory, bgz-tensor known failures |
| **#329** | 2026-05-01 | style: apply rustfmt to contract lib.rs + python bindings | Tier-A rustfmt drift in contract lib.rs + python bindings (no semantic change) |
| **#328** | 2026-05-01 | ci(test): add lance-graph-contract unit tests to test gate | `cargo test -p lance-graph-contract --lib` added to CI rust-test.yml |
| **#327** | 2026-05-01 | style(shader-driver): drop double-space alignment in bindspace.rs | Two-line rustfmt drift fix in bindspace.rs introduced by #323 |
| **#326** | 2026-05-01 | fix(sigma-propagation): correct log_norm_growth_negative test seed | Fix broken test from #322: seed at 4·I not I so attenuation reduces log-norm |
| **#325** | 2026-04-30 | chore(toolchain): bump pin 1.94.0 → 1.94.1 | rust-toolchain.toml bumped to 1.94.1 to match sibling repos |
| **#324** | 2026-04-30 | feat(shader-driver): Pillar-7 α-front-to-back-merge sink mode (B5) | AlphaFrontToBack MergeMode + EWA Kerbl-2023 compositing in stage [7] |
| **#323** | 2026-04-30 | feat(cognitive-shader-driver): add Σ-codebook-index column to FingerprintColumns (B2) | FingerprintColumns.sigma u8 column (+1 byte/row, 0.02% overhead) |
| **#322** | 2026-04-30 | feat(contract): promote EWA-Sandwich Σ-propagation kernel to contract (B1) | sigma_propagation.rs: Spd2, ewa_sandwich, log_norm_growth, pillar_5plus_bound |
| **#321** | 2026-04-30 | fix: 10 pre-existing test failures (cosine_distance, arigraph, parse_triplets) | Fixed cosine inversion, Stagnant ordering, quality_window clear, SPO arg order |
| **#320** | 2026-04-30 | ci: declare rustfmt + clippy as pinned-toolchain components | rust-toolchain.toml gets components=[rustfmt,clippy]; fixes CI fmt failure |
| **#319** | 2026-04-30 | fix(transcode): per-month day-validity in parse_iso_date_to_days | Gregorian per-month + leap-year gate before civil_to_days |
| **#316** | 2026-04-30 | feat(transcode): round-3 typed-value resolver for triples_to_batch | triples_to_batch_with_resolver: Currency→f32, Date→Date32, Id→u64 |
| **#315** | 2026-04-30 | ci: revert ndarray-branch pin — PR #115 landed on master | Remove temp ndarray branch pin from rust-test.yml + style.yml |
| **#314** | 2026-04-30 | docs(vision): clear post-F1 staleness items in medcare-foundry-vision.md | §1–§4 DRAFT/forward-tense/PR-N placeholders replaced with real anchors |
| **#313** | 2026-04-30 | feat(transcode): Phase-2-B triples_to_batch (ExpandedTriple → RecordBatch) | ExpandedTriple stream → N-row RecordBatch, lenient-Utf8, 19 tests |
| **#312** | 2026-04-30 | feat(transcode): Phase-2-A pushdown classification (Inexact for recognised filters) | OntologyTableProvider classifies entity_type/predicate/nars filters as Inexact |
| **#311** | 2026-04-30 | docs(vision): mark F1 shipped, restate next deliverable as F2 | medcare-foundry-vision.md §7: F1 parity shipped; F2 RBAC is next posture |
| **#310** | 2026-04-30 | feat(transcode): r2 fixes — typed Arrow + codec_route + partial writes + CachedOntology | Currency/Date/Id→typed Arrow; CachedOntology; validate_route; from_columns_partial |
| **#309** | 2026-04-30 | feat(callcenter::transcode): outer ↔ inner ontology mapper + parallelbetrieb | transcode submodule: zerocopy, cam_pq_decode, spo_filter, ontology_table, parallelbetrieb |
| **#308** | 2026-04-30 | feat: bilingual ontology DTO surface + bgz-tensor workspace inclusion | OntologyDto locale projection; smb_ontology + medcare_ontology; bgz-tensor in workspace |
| **#307** | 2026-04-30 | refactor: dedup FNV-1a — one canonical hash::fnv1a in contract | contract::hash::fnv1a const fn; 8 call sites unified |
| **#306** | 2026-04-30 | feat(G4): verb_table tense modulation (Quirk CGEL grounded) | 12 VerbFamily priors + tense_modifier → 144 unique cell values |
| **#305** | 2026-04-30 | feat(G3): DisambiguateOpts builder + deepnsm caller wiring real fingerprint | DisambiguateOpts builder; sign_binarize_to_binary16k; disambiguator_glue.rs |
| **#304** | 2026-04-30 | feat(G1): Pearl 2³ causality footprint with PAD-model qualia mapping | compute_pearl_mask() 3-bit SPO→CausalMask; PAD qualia footprint replaces 0.5 |
| **#303** | 2026-04-30 | feat(F6): FNV-1a scent with scent_u64 accessor + birthday collision tests | scent() FNV-1a fold-to-u8; scent_u64() full 64-bit digest; 10 tests |
| **#302** | 2026-04-30 | feat(F3): LanceAuditSink with temporal timestamps + full schema round-trip | LanceAuditSink → Lance dataset append; temporal timestamp; O(1) scan_back |
| **#301** | 2026-04-30 | feat(F1): ColumnMaskRewriter full-tree expression walk + Hash UDF hard-fail | Full-tree OptimizerRule covering Filter/Aggregate/Join; NotYetWiredHashUdf |
| **#300** | 2026-04-30 | feat(LF-12): Pipeline DAG with StepId derivation + OrchestrationBridge adapter | PipelineDag Kahn's algorithm; FNV-1a StepId; execute_via_bridge; cycle detection |
| **#299** | 2026-04-29 | revert #294/#295/#296 + clean on top | Reverts #294–#296 confabulation; corrects probe routing (M1/P2-P4 → shader-lab) |
| **#296** | 2026-04-29 | ideas: COCA-Bundle vs Jina-CLAM bucket comparison (**REVERTED by #299**) | IDEAS.md Open entry for COCA/Jina probe (premise flawed; reverted) |
| **#295** | 2026-04-29 | docs: probe-queue data-available followup (**REVERTED by #299**) | bf16-hhtl-terrain.md data-available update (inherited bad routing; reverted) |
| **#294** | 2026-04-29 | docs(probe-queue): honest "needs production data" assessment (**REVERTED by #299**) | bf16-hhtl-terrain.md probe routing table (wrong routing; reverted) |
| **#293** | 2026-04-29 | jc: drain Probe P1 (γ-phase-offset ranking discrimination) → PASS | probe_p1_gamma_phase.rs; P1 PASS: min Spearman ρ=-0.963 (Dupain-Sós) |
| **#292** | 2026-04-29 | docs(board): posthoc-correct PRs #290 #291 via canonical board mechanism | CONJECTURE banners; 5 Open IDEAS.md entries; 2 EPIPHANIES.md entries |
| **#291** | 2026-04-29 | docs: idea journal — proposed application pillars 7/8/9 captured | IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md with Pillars 7/8/9 + PASS criteria |
| **#290** | 2026-04-29 | docs: idea journal — streaming-hydration + fractal-codec captured | IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md separating two ideas |
| **#289** | 2026-04-29 | jc: Pillar 6 — EWA-Sandwich Σ-push-forward | ewa_sandwich.rs; Pillar 6: 10000/10000 PSD-preserving hops; KS bound tightness 1.467× |
| **#288** | 2026-04-29 | jc: Σ-Codebook Viability Probe — rules out CausalEdge64 8→16B expansion | sigma_codebook_probe.rs; R²=0.9949 at k=256; CausalEdge64 stays 8 bytes |
| **#287** | 2026-04-29 | jc: Pillar 5++ — Düker-Zoubouloglou Hilbert-space CLT | dueker_zoubouloglou.rs; Pillar 5++: bundle-of-N in ℝ^16384 → Gaussian limit in ℓ² |
| **#286** | 2026-04-29 | jc: Pillar 5+ — Köstenberger-Stark concentration on Hadamard 2×2 SPD | koestenberger.rs; Pillar 5+: tightness 0.969× on SPD manifold |
| **#285** | 2026-04-29 | Re-land #283 unlocks (Quantum, Disambiguator, verb_table, animal-farm) | Quantum mode, Disambiguator trait, verb_table, animal-farm harness; PhaseTag overflow fix |
| **#284** | 2026-04-29 | Re-land #281 unlocks (PolicyRewriter, DomainProfile) | PolicyRewriter trait, ColumnMaskRewriter, DomainProfile HIPAA thresholds |
| **#282** | 2026-04-29 | fix: Grammar/Markov hardening — slice unification, kernel wiring | CRITICAL slice fix; rotate_right removed; coherence kernel wired; 363 tests |
| **#280** | 2026-04-29 | fix: Foundry hardening — sealed RLS, VecDeque audit, URL decode, Plugin handshake | Sealed RLS default; O(1) audit ring; FNV-1a; URL decode; Plugin handshake; 58 tests |
| **#279** | 2026-04-29 | feat: DeepNSM grammar parser — Markov ±5 bundler, role keys, thinking styles | D0/D2/D3/D4/D5/D6/D7: MarkovBundler, RoleKeySlice, GrammarStyleConfig, 12 YAML configs |
| **#278** | 2026-04-29 | feat: Foundry parity — RLS rewriter, audit log, PostgREST, with_registry | LF-3/DM-7 RLS; LF-90 audit; DM-8 PostgREST stub; LanceMembrane::with_registry; 35 tests |
| **#277** | 2026-04-28 | plan: unified Foundry roadmap for SMB + MedCare (corrects #276 framing) | foundry-roadmap-unified-v1.md; correct scale decisions per FormatBestPractices.md |
| **#276** | 2026-04-28 | plan: Foundry Consumer Parity — shared ontology + UNKNOWN resolutions | foundry-consumer-parity-v1.md; 5 callcenter UNKNOWNs resolved; DM-8 unblocked |
| **#275** | 2026-04-28 | feat: add lancedb 0.27.2 + pin lance =4.0.0 | lancedb=0.27.2 optional dep; lance exact-pinned =4.0.0 for compat |
| **#274** | 2026-04-27 | fix: F-01 identity-tear race + F-08 bounds check + F-09 poison recovery | Single ActorState RwLock; poison recovery; push bounds check |
| **#273** | 2026-04-27 | feat: bump lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 | Version bumps + API break fixes (invalid_input, DeltaTableProvider migration) |
| **#272** | 2026-04-27 | feat: Column H — EntityTypeId on BindSpace (Phase 1 of 4) | EntityTypeId u16 on BindSpace; push_typed(); 1-based index; 4 tests |
| **#271** | 2026-04-27 | plan: BindSpace Columns E/F/G/H — 4→8 SoA integration plan | bindspace-columns-v1.md; 24 deliverables; 7 SOUND / 7 CAUTION / 0 WRONG |
| **#270** | 2026-04-26 | ci: remove typos spell-check job (too many false positives) | Removed crate-ci/typos from style.yml; cargo fmt --check remains |
| **#269** | 2026-04-26 | feat: Distance trait + SIMD Hamming/cosine wiring + PaletteDistanceTable + Dockerfile docs | Distance trait; SIMD Hamming/cosine wiring; PaletteDistanceTable 128KB; Dockerfile.md |

---

## 2026-06-17 — Append: materialized-awareness driver wire (provenance-only) on branch claude/materialize-awareness-f34-loop

(Per APPEND-ONLY rule: new top-of-inventory entry.)

### Current Contract Inventory — new entry

**`lance-graph-contract::cognitive_shader::MaterializeProvenance`** (new type, 2026-06-17): primitive-only Copy record (`first_tactic:u8`, `steps:u16`, `rested:bool`, `final_free_energy:f32`, `fork:u8`) added as a field on `ShaderCrystal`. The `cognitive-shader-driver` runs the `materialize` F→34→F loop **and** the ndarray HHTL `fork_decision` as a **side analysis** over each cycle's already-computed observables (`free_energy`, `std_dev`, MUL, per-hit resonances) and records the outcome here. **Provenance-only — does NOT alter `bus.gate` or persistence** (operator decision 2026-06-17: cycle untouched). Observable→`ThoughtCtx` mapping is faithful (sd←std_dev, confidence←1−F, dissonance←|felt−demonstrated| DK gap); the fork's challenge is a **`std_dev` dispersion proxy (CONJECTURE)** with a std_dev-calibrated floor/σ, pending the real orthogonal `CoarseResidue` magnitude from the codec path and real HHTL cascade depth (depth==max⇒leaf for now). Driver helper `materialize_provenance(...)`; 2 driver tests (confident→Commit, scattered→ForkDomain; dispatch populates provenance) + the ndarray fork ladder (PR #221, merged). `fork:u8` = `ForkAction` (0 Commit /1 DescendDeeper /2 ForkBasin /3 ForkDomain). Note: `cognitive-shader-driver` is not in the default-workspace clippy member set. See `EPIPHANIES.md` E-MATERIALIZED-AWARENESS-1.

---

## 2026-06-16 — Append: `contract::materialize` shipped (branch claude/materialize-awareness-f34-loop)

(Per APPEND-ONLY rule: new top-of-inventory entry.)

### Current Contract Inventory — new entry

**`lance-graph-contract::materialize`** (new module, 2026-06-16): the closed `F→34→F` dispatch loop that makes awareness *materialize* — the missing wire from awareness state to the 34 `recipe_kernels` tactics. Public surface: `select_tactic(&ThoughtCtx) -> u8` (awareness→tactic id, `free_energy`-primary so dispatch tracks awareness), `materialize(&mut ThoughtCtx, max_steps) -> Trace` (select→`Tactic::run`→settle gate→recompute surprise→re-dispatch; rests at CollapseGate FLOW), `recompute_free_energy`, `awareness_is_causal` (the materialization predicate / falsifier), types `Step` / `Trace`, const `HOMEOSTASIS_FLOOR=0.2`. Decision: `recipe_kernels` is the canonical "34" (ndarray `hpc/styles/*` is divergent/registry-less, not canonical). Zero-dep, offline; 6 tests green (+632 prior contract lib), clippy `--all-targets -D warnings` clean. Open: driver-side `ThoughtCtx::from_live` + version-diff provenance wire. See `EPIPHANIES.md` E-MATERIALIZED-AWARENESS-1.

---

## 2026-05-07 — Append: lance-graph-ontology shipped (commit 4cf9a26, branch claude/create-graph-ontology-crate-gkuJG)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table and "Current Contract Inventory" snapshot above. Treat the row below as the new top-of-table entry; treat the inventory paragraph below as a new top-of-inventory entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **(open / pending merge)** | *(open)* | feat(lance-graph-ontology): scaffold OGIT-canonical ontology spine | New workspace member `crates/lance-graph-ontology/` (~3000 LOC, 28 tests = 16 inline + 12 integration). Phases 3-5 of the v4 plan: scaffold + TTL hydration + tenant bridges. Public surface: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. Feature-gated `lance_cache::LanceWriter` (under `lance-cache` feature, gated to keep zero-protoc compile path). Builds on prior commit `edef321` (recon + SPO-1 decision: federated two-layer cache, Option B). |

### Current Contract Inventory — new entry

**`lance-graph-ontology`** (new crate, 2026-05-07): consolidates per-tenant bridge multiplication into one ontology spine. OGIT becomes the canonical TTL ontology source; Lance is the (feature-gated) runtime dictionary cache; tenant bridges become thin scoped views over the shared registry. Public types: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. 28 tests passing (16 inline + 12 integration). Feature-gated Lance persistence under `lance-cache` (kept off by default so the crate compiles without `protoc`, which `lance-encoding`'s build-script requires). Branch `claude/create-graph-ontology-crate-gkuJG`; commit `4cf9a26`; prior recon + decision in `edef321` (`.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`).

---

## 2026-05-07 — Sprint-2: Unified OGIT Architecture synthesis (recently shipped — documentation tier)

> **APPEND-ONLY annotation.** Per the governance rule above, this section augments — does not edit — prior content. Treat as the new top-of-state. Branch: `claude/unified-ogit-architecture-synthesis`.
>
> Sprint-2 was a 12-agent + meta-review coordinated burst. **Zero code changes; documentation tier only.** It captures 16 turns of architectural conversation (2026-05-07) as a unified pattern-recognition framework over already-shipped substrate, plus three concrete next-PR sub-plans and one proof-of-vision plan. The dominant finding: ~80% of the "unified OGIT architecture" we were about to design is **already shipped**; recognising this drops architecture entropy by **−11** with no code written.

### Sprint-2 deliverables (12 workers + meta)

**New plan-docs (4)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/plans/unified-ogit-architecture-v1.md` | ~22 KB | W1 | Master synthesis: 15 patterns (A-O) + Tier 0-4 stack + proof-of-vision. Canonical reference for the unified OGIT architecture. |
| `.claude/plans/ogit-g-context-bundle-v1.md` | ~10 KB | W10 | Tier-1 sub-plan: G-overlay wiring; Patterns A (G-slot) + B (context-bundle) + C (per-cycle cascade). |
| `.claude/plans/compile-time-consumer-binding-v1.md` | ~10 KB | W11 | Tier-2 sub-plan: compile-time consumer binding + ractor; Patterns E (consumer-binding) + F (zero-overhead actor seam). |
| `.claude/plans/anatomy-realtime-v1.md` | ~12 KB | W12 | Proof-of-vision: north-star realtime anatomy demo end-to-end across the unified stack. |

**New knowledge doc (1)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/knowledge/tier-0-pattern-recognition.md` | ~13 KB | W2 | File→pattern map covering ~30 already-shipped files. Read this FIRST in any future session that touches OGIT architecture to avoid the Discovery-Loop anti-pattern. |

**Board appends (5, append-only governance preserved)**

| File | Worker | Append summary |
|---|---|---|
| `.claude/patterns.md` | W3 | Appended **Pattern Recognition Framework**: 15 patterns A-O catalogued + new Anti-Pattern **"Designing What's Already Built"**. |
| `.claude/board/EPIPHANIES.md` | W4 | Appended **17 architectural epiphanies**: E-OGIT-1 through E-RECOGNITION-OVER-DESIGN-17. |
| `.claude/board/TECH_DEBT.md` | W5 | Appended **11 TD entries**: TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11, each with effort estimate. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` | W6 | Appended **5 row reframes** (THINK-1 5→3, HEEL-1 4→2, ADJ-THINK-1 4→2, CRYSTAL-1 4→2, CAM-DIST-1 3→2) + 15-pattern absorption table. **Net entropy delta: −11**. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` | W7 | Appended RECOGNITION-1 meta-finding row + Anti-Pattern surfaced ("Designing What's Already Built"). |

**Index update (1)**

| File | Worker | Update |
|---|---|---|
| `.claude/board/INTEGRATION_PLANS.md` | W8 | Indexed the 4 new plan-docs (W1 master + W10 + W11 + W12). |

**Sprint coordination (CCA2A pattern, `/sprint-log-2`)**

- `.claude/board/sprint-log-2/SPRINT_LOG.md` — master coordination index.
- `.claude/board/sprint-log-2/agents/agent-W{1..12}.md` — per-agent append-only logs (12 files).
- `.claude/board/sprint-log-2/meta-1-review.md` — meta agent brutally-honest review.
- `.claude/board/sprint-log-2/agents/agent-W9.md` — this worker's handover log.

### Aggregate impact

- **15 architectural patterns (A-O)** named and catalogued.
- **~80% of the "unified OGIT architecture" is recognised as already shipped** — Patterns H, M, N, O at substrate level; Pattern F shape proven by gRPC.
- **~20% genuinely new wiring work** captured as TECH_DEBT entries with effort estimates (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11).
- **Net entropy reduction from recognition alone: −11** (no code changes; 5 row reframes + 15-pattern absorption).
- **Totals shipped this sprint:** 4 new plan-docs + 1 knowledge doc + 5 board appends + 1 index update + sprint-log-2 scaffolding (1 master + 12 agent logs + 1 meta review).

### What this enables

Future sessions that read `.claude/knowledge/tier-0-pattern-recognition.md` first will avoid the **Discovery-Loop anti-pattern at architectural scale** — the same anti-pattern `.claude/patterns.md` warns about at cycle level (proposing concepts that already exist in workspace).

The master plan-doc `.claude/plans/unified-ogit-architecture-v1.md` provides the canonical reference for the unified OGIT architecture. The three sub-plans give concrete next-PR scope:

- **Tier 1 next PR** — `.claude/plans/ogit-g-context-bundle-v1.md` (G-overlay wiring).
- **Tier 2 next PR** — `.claude/plans/compile-time-consumer-binding-v1.md` (compile-time consumer binding + ractor).
- **Proof of vision** — `.claude/plans/anatomy-realtime-v1.md` (north-star demo).

### Cross-references

- All sister deliverables listed above (W1–W12 + meta).
- 16-turn architectural conversation (2026-05-07).
- Pre-existing plans absorbed into the unified framework: `lance-graph-ontology-v5` (PR #355), `palantir-parity-cascade-v2` (PR #353), `ogit-cascade-supabase-callcenter-v1` (PR #355).
- Substrate already shipped (Patterns H/M/N/O): see "Current Contract Inventory" and "Current AriGraph Inventory" sections above; especially `lance-graph-ontology` (commit `4cf9a26`), `cognitive-shader-driver` BindSpace SoA (PR #204+ thru #323), `crystal/` Vsa16kF32 sandwich (PR #208/#209), `cam/` codec cascade (PR #225).

### Brutally-honest self-review (W9)

- **In scope:** append-only update to `LATEST_STATE.md`. Did not edit any prior content. Verified the file's existing closing section (`2026-05-07 — Append: lance-graph-ontology shipped`) is preserved verbatim.
- **Risk:** the "~80% already shipped" claim is W1/W2's recognition assertion, not independently re-verified by W9. This section reports it as the synthesis output; the canonical evidence lives in `tier-0-pattern-recognition.md` (W2) and the entropy ledger reframe rows (W6).
- **Governance:** append-only preserved. No deletions. No edits to the prior `## 2026-05-07 — Append: lance-graph-ontology shipped` section. Section heading matches the spec exactly.
- **What this section does NOT do:** it does not edit the top-of-file "Last updated" line (would violate append-only); it does not edit the "Recently Shipped PRs" table (Sprint-2 shipped no PRs); it does not edit "Active Branches" (Sprint-2 is documentation tier on a branch that has not yet merged).

## 2026-05-12 — Sprint-3: Tier-1 Implementation Specs (PR #360 + #361 + post-#360 substrate sweep)

**PR #360** (sprint-3 main): 11 PR-X-1 specs covering 7 design-phase patterns A/B/C/D/E/F/J + 3 trivia closures + supporting docs. ~140 KB across `.claude/specs/`. Engineer can now execute Tier-1 in ~6 working days parallelized (per W10 sequencing).

**PR #361** (post-#360 corrections): PR-F-1 supervisor must skip inert bundles (DOLCE/FMA have consumer_pointer=None by design); PR-E-1 build script must emit data-only (no consumer crate refs) to avoid Cargo dependency cycle. Both fixed via append-only correction sections; inventory-crate self-registration recommended for actor binding.

**Post-#360 substrate-recognition sweep** (this PR): 3 of 11 specs reclassified PARTIALLY SHIPPED:
- Pattern A: SchemaPtr.ontology_context_id + NamespaceRegistry::seed_defaults already ship; PR-A-1 reduces to ~150 LOC / 1 day
- Pattern C: BridgeFromRegistry + 3 impls + woa-rs#2 + medcare-rs#110 consumer scaffolds already ship; PR-C-1 reduces to ~80 LOC / ½ day
- Pattern D: parse_ttl_directory_with_provenance + attach_provenance already ship; PR-D-1 reduces to ~250 LOC / 1-2 days (OWL/RDF-XML adapter only)

Compressed sprint-3 critical path: ~6 days → ~3-4 days parallelized. The genuinely-new ~5-pattern set is B (context bundle), E (manifest-modules), F (ractor port), G (inheritance protocol), J (INT4-32D atoms).

### New knowledge docs (sprint-3 substrate-sweep)

- `.claude/knowledge/pattern-recognition-cross-source.md` — A-O ↔ Pillars 0-4 ↔ `.grok/` ↔ shipped substrate matrix (4 parallel taxonomies cross-referenced)
- `.claude/knowledge/cca2a-sprint-prompt-template.md` — substrate-grep checklist + wrong-repo guardrail + pattern-letter discipline (mandatory pre-spawn template for future sprints)

### Anti-Pattern recurrence captured

The "Designing What's Already Built" anti-pattern (introduced PR #358) recurred in sprint-3's own design (PR-A-1/PR-C-1/PR-D-1 over-scoped because they didn't sweep post-#355 substrate). The correction PR formalizes the substrate-grep checklist as mandatory before any new spec.

### Recurring failure mode: wrong-repo error

Sprint-2 W7 → ndarray; sprint-3 W9 → ada-consciousness. Both corrected via main-thread pygithub recovery. Wrong-repo guardrail snippet now mandatory in every worker prompt (per `.claude/knowledge/cca2a-sprint-prompt-template.md`).

### Cross-references

- `.claude/specs/sprint-3-execution-plan.md` (W1 master)
- `.claude/specs/sprint-3-pr-graph.md` (W10 sequencing — to be updated for compressed timeline)
- `.claude/specs/pr-{a,b,c,d,e,f,j}-1-*.md` (11 PR-ready specs; A/C/D have appended CORRECTION sections)
- `.claude/specs/consumer-crate-template.md` (W8; re-target from hubspo-rs hypothetical to woa-rs/medcare-rs precedent)
- `.claude/specs/ogit-g-smoke-test.md` (W11 validation)
- `.claude/specs/trivia-prs-bundle.md` (W12 — 3 quick wins parallel-shippable)
- `.claude/board/sprint-log-3/{SPRINT_LOG.md,agents/agent-W1..W12.md,meta-1-review.md,sprint-summary.md}`

PR sequence: #360 → #361 → post-#360 substrate-sweep (this PR).

---

## APPEND-ONLY annotation — D-ODOO-1 odoo hydrator (2026-05-27)

> Per the APPEND-ONLY governance rule, this section augments — does not edit — prior content. Treat as the new top-of-state. Branch: `claude/lance-graph-att-activate-Jd2iZ`.

### Current Contract Inventory — new entry

- **`OGIT::ODOO_V1` = (50, 1)** — new OGIT G slot (first manifest-declared slot above SKR03BAU=42). Source: `modules/odoo/manifest.yaml` (`ogit_g: ODOO`, `inherits_from: fibofnd`, 17 entity_types u16=4300..4316). Registered in `crates/lance-graph-contract/build.rs` CANONICAL_SLOTS as `("ODOO", 50)`; build regenerates `OUT_DIR/ogit_namespace.rs` accordingly.

### New module surface (`lance-graph-ontology`)

- **`hydrators::odoo`** — Layer-1 odoo extraction hydrator (four-way alignment seam). `hydrate_odoo(registry)` + `hydrate_odoo_from(paths, registry)`; `inherits_from: Some(OGIT::FIBOFND_V1.0)`; edge whitelist {rdfs:subClassOf, owl:equivalentClass, rdfs:subPropertyOf, owl:equivalentProperty}. Re-exported from `lib.rs`.
- **`hydrators::dolce_odoo`** — odoo DOLCE suffix classifier (Seam decision 2, own module per Open-question 3). `pub fn classify_odoo(iri: &str) -> DolceCategory` + `pub enum DolceCategory { Endurant, Perdurant, Quality, AbstractEntity }`. Re-exported from `lib.rs`. (Doc-noted: canonical DUL renames Endurant→Object / Perdurant→Event.)

### New data artifacts

- `data/ontologies/odoo/odoo-core.ttl` — 17 odoo core classes (`odoo: <https://ada.world/onto/odoo#>`).
- `data/ontologies/odoo/alignment/odoo-to-fibo.ttl` + `odoo-to-skr.ttl` — Layer-2 `owl:equivalentClass`/`owl:equivalentProperty` alignment axioms (Seam decision 1 / Option B: odoo inherits existing FIBO/SKR slots, no new CAM family).

### Tests

`cargo test -p lance-graph-ontology` → 127 passed / 0 failed (+7 odoo integration tests across `tests/odoo_hydrator_smoke.rs` + `tests/odoo_dolce_classifier.rs`, incl. the full 21-row seam classifier matrix; +4 lib unit tests). `cargo test -p lance-graph-contract` → 449 passed / 0 failed.

### Relationship to prior art

`lance-graph-callcenter::odoo_alignment` already ships a parallel `dolce_odoo()` + `DolceMarker` + `ODOO_SEED` table. This is the ontology-side counterpart (TTL hydration into `OntologyRegistry`); consistent doctrine (Option B, same pivots), distinct crate + distinct `DolceCategory` enum per task spec. Cross-crate dedup is a possible follow-up, not done here.

---

## 2026-05-28 — Append: PR #422 shipped (post-merge governance for the #418/#419 review handover)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#422** | 2026-05-28 | docs(handover): PR #418/#419 review + surreal/mailbox/Baton/SoA-as-BindSpace-surrogate plan map | Read-only synthesis handover. New `.claude/handovers/2026-05-28-1200-pr-418-419-surreal-mailbox-baton-plan-map.md` (~310 LOC, 7 sections): §1 PR #418 review (verdict *sound, merge-ready as a spec* + 3 substantive notes on the bare-columns-vs-hot-thought footprint distinction, `E-RUBICON-RACTOR` as honest post-hoc CONJECTURE, OQ-4 doctrinal gating); §2 the **SurrealDB role correction** (Zone-2 cold store → *view over leading LanceDB*, recorded in `E-RUBICON-RACTOR` + plan §2.7); §3 the plan corpus map (8 plans + 9 epiphanies + `PR-NDARRAY-MIRI-COMPLETE → D-CE64-MB-1-impl → D-MBX-1..6` dep chain + `TD-RESONANCEDTO-DUP-1`); §4 brief #419 review (unrelated to surreal/mailbox; the 14 `NEEDS-INPUT` blockers are the real gate for D-ODOO-SAV-4); §5 navigability meta-finding (the surreal POC docs lack a supersedure pointer); §6 action surface; §7 cross-refs. Board appends: `EPIPHANIES.md` ← `E-SURREAL-POC-UNANNOTATED-SUPERSEDURE` (FINDING / navigability); `AGENT_LOG.md` ← session row. **Zero code change**; 3 files; +310/-0. Branch `claude/lance-graph-ontology-review-Pyry3` → `main`. Merge commit `984512b` on top of `a29946b` (the doc commit, rebased onto post-#421 `main` to resolve the AGENT_LOG append-vs-append conflict by keeping both #421's AXIS-B row and this PR's session row in chronological order). |

---

## 2026-05-28 — Append: PR #425 shipped (deps comment cleanup + [patch.crates-io] ndarray declared intent)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#425** | 2026-05-28 | deps(workspace): clean BLOCKED comments; record 6.0.0→6.0.1 block (lancedb 0.29.0 transitive) | Workspace `Cargo.toml` cleanup + finding. Replaces the stale `BLOCKED-(A)/(B)/(D)` comment block (predates #423's 4→6 / 0.27→0.29 / 52→53 / 57→58 bump) with a dated `RESOLVED(A)/(B)/(D)` record pointing to #423 + the live crate-level pins. Records the user-authorised follow-on patch `lance 6.0.0 → 6.0.1` as **CURRENTLY BLOCKED** by `lancedb 0.29.0`'s transitive `lance = "=6.0.0"` requirement (proof: `cargo check` → `versions that meet the requirements '=6.0.0' are: 6.0.0`; resolution paths: wait for lancedb 0.29.1+, drop strict-=, or `[patch.crates-io]` override). Adds `[patch.crates-io] ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", branch = "master" }` per user directive — declared intent; cargo emits `warning: patch ndarray v0.17.2 was not used in the crate graph` because lance-index 6.0.0 pins `ndarray = "0.16.1"` (semver gap, fork at 0.17.2). `Cargo.lock` now contains a `[[patch.unused]]` entry that makes the gap visible at every build. Files `TD-NDARRAY-PATCH-0_16` in `TECH_DEBT.md`. Codex P2 (`59ef97e`) flagged the original false RESOLVED(D) claim; fix in `2e001a5`/`8f3913b`/`1444f78`. Merge commit `1a3abfb8`. |

## 2026-05-28 — Append: PR #427 shipped (bindspace→mailbox migration wave A1-A4)

(Per APPEND-ONLY rule: appended after PR #425's annotation above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#427** | 2026-05-28 | feat(mailbox-soa): bindspace→mailbox migration wave A1-A4 (thoughtspace columns + transitional routing + WitnessTable + plan §10) | First implementation pulse of `bindspace-singleton-to-mailbox-soa-v1` (PR #418 plan). **A1** (`1df12eca`, +103): 4 thoughtspace columns on `MailboxSoA` (`edges`/`qualia`/`meta`/`entity_type`) + 8 row accessors + zero-init in `new()` + reset in `reset_row()`. **A2** (`61b641d5`, +42): transitional `mailboxes: HashMap` + `with_mailbox()` builder + `mailbox()` accessor on `ShaderDriver` — sibling-shape, additive, singleton untouched. **A3** (`ef848a34`, +187): new `WitnessTable` + `WitnessEntry{ mailbox_ref, spo_fact_ref }` primitive in `lance-graph-contract::witness_table` (zero-dep, 3 unit tests, `const fn new`, `get`/`set` bounds-checked). **A4** (`0f448730`, +36): plan §10 "2026-05-28 architectural refinements" appended — 7 ratified findings (SoA-Lance ≠ cascade; cascade is not an index space; 64k-256k mailbox envelope ~360 MB - 1.4 GB RAM-resident; W-slot = per-cohort witness table not corpus pointer; cascade granularities = CPU/cache boundaries 64/256/4096/16384; `simd_soa.rs` introspects per-SoA shape; SoA invariant spawn → commit, two egress modes external/internal) + 2 surviving OQs (OQ-MBX-8 `persisted_row` vs Lance native versioning; OQ-MBX-15′ container scoping granularity). Codex P1 follow-on `f541b280`: widen `WitnessEntry.mailbox_ref` u16 → u32 + correct `Option<u64>` size doc. **457 contract+driver tests passing**, zero new behavioural code outside the columns/builder/primitive. Singleton `Arc<BindSpace>` NOT removed (sibling pattern); cutover in a downstream slice (D-MBX-3/4). Merge commit `84296118`. Author session — this governance row is the post-merge close-out; per-deliverable AGENT_LOG entries D-MBX-A1..A4 already prepended at branch HEAD pre-merge. |

## 2026-05-30 — Append: new standalone crate `lance-graph-arm-discovery` (Aerial+ transcode, D-ARM-13) on branch `claude/jolly-cori-clnf9`

(Per APPEND-ONLY rule: dated annotation augmenting the "Current Contract Inventory" snapshot above. Branch work, not yet merged — recorded so a new session does not re-derive the crate.)

### Current Contract Inventory — new entry

- **`crates/lance-graph-arm-discovery`** (NEW, **excluded** standalone zero-dep crate; build via `cargo test --manifest-path crates/lance-graph-arm-discovery/Cargo.toml`). The **Aerial+** Rust transcode (Karabulut 2025, 2504.19354v1) — the upstream runtime-data proposer leg of `streaming-arm-nars-discovery-v1`. Public surface: `encode::{FeatureSpec, Dataset}`, `rule::{Item, CandidateRule, Proposer}`, `translator::{arm_to_nars, NarsTruth, CandidateTriple, FeedProjector, DebugProjector, NARS_PERSONALITY_K}`, `ndjson::to_ndjson`, and (feature `aerial`, default-on) `aerial::{Rng, AerialAutoencoder, AerialParams, AerialProposer, extract_rules, ExtractParams}`. 35/35 tests, clippy `-D warnings` clean. Emits the `{"s","p","o","f","c"}` ndjson the SPO store loader reads; `(f,c)` == `TruthValue::new(f,c)` == `ruff_spo_triplet::Triple{f,c}`. Determinism boundary: nondeterministic AE is seeded + feature-gated + emits *proposals* only. Synergy map: `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md`. Status board: D-ARM-13 (Shipped on branch) + D-ARM-SYN-1/2/3 (Queued). **Not** in `lance-graph-contract` yet — `rule`/`translator` are the local seam until D-ARM-1/2 land the shared carriers.

> **2026-06-01 — PR-in-flight (autoattended)** (D-EW64-3/4): `lance_graph_contract::episodic_edges` gains `EpisodicEdges64::{coldest, contains, promote_into}` + the `DemotionSink` trait. `coldest()` = the eviction victim (symmetric to `strongest()`); `contains()` = family-discriminating membership; `promote_into(e, sink)` = `promote` routing the evicted (coldest) edge to a `DemotionSink` — the hot→cold connectome exit. `DemotionSink` impls (surreal/LanceDB-LIVE "wingman", `E-SUBSTRATE-IS-THE-SCHEDULER`) are deferred + GATED on OQ-11.6. Zero-dep; contract lib 545 green; default clippy clean; `episodic_edges.rs` pedantic+nursery clean.

> **2026-06-01 — Shipped (autoattended, 5-agent council)** (D-ATOM-4/RawEdge): `contract::counterfactual` wired into `lib.rs` (was orphaned); `RawEdge(i8)` mantissa-only **structural** impl of `EpisodicEdge` (`size_of==1` — a u64 newtype could read plasticity 50–52); `deposit_counterfactual` v2 filled (−6 on split). Closes the counterfactual seam (NOT the prefetch loop). +3 latent scaffold fixes. 550 contract lib green, clippy clean. The council REFUTED the prior "compose `Heel.plasticity` × MRU" ① resolution (`E-BASIN-NOT-EDGE-PLASTICITY`): coarse strength = MRU slot-order (shipped); per-edge Hebbian = per-plane `PlasticityState` (gated).

## 2026-06-14 — Append: `EdgeCodecFlavor` selector + ndarray edge-codec/reliability layer (branch `claude/wonderful-hawking-lodtql`)

(Per APPEND-ONLY rule: new top-of-inventory entry. Branch work; records the contract type so a new session does not re-derive it.)

### Current Contract Inventory — new entry

- **`canonical_node::EdgeCodecFlavor`** (NEW; re-exported from `lib.rs`). Per-class selector for how a node's 16-byte `EdgeBlock` (+ optional value-slab residue) is *interpreted* — `CoarseOnly` (1 B palette index, the canon zero-fallback default), `CoarseResidue` (1 + ⌈D/2⌉ B, value-slab signed-4-bit residue), `Pq32x4` (16 B = 32×4-bit product code, the turbovec PQ model). `bytes_per_vector(dim)` + `is_layout_preserving()` (always `true`). **Iron invariant:** the flavor is interpretation, NOT layout — every variant leaves `NODE_ROW_STRIDE = 512` untouched, so adoption needs no `ENVELOPE_LAYOUT_VERSION` bump (canon "registry-resolved via `classid → ClassView`"). Default `CoarseOnly` matches the all-zero bootstrap. Selection surface: new defaulted `ClassView::edge_codec_flavor(&self, ClassId) -> EdgeCodecFlavor` (non-breaking — `RegistryClassView` inherits the default; per-class override is the follow-up wiring in `lance-graph-ontology`). +3 tests; contract lib 609 green; clippy `-D warnings` clean.
- **`canonical_node::ValueSchema` + `ValueTenant` + `VALUE_TENANTS`** (NEW; re-exported from `lib.rs`). The **value-side analog of `EdgeCodecFlavor`**: per-class presets for which tenants the 480-byte `NodeRow::value` slab materialises. `ValueTenant` = 9 stable append-only positions (discriminant == `FieldMask` bit == `VALUE_TENANTS` index): Meta(`MetaWord`) · Qualia(`QualiaI4_16D`) · MaterializedEdges(4×`CausalEdge64`) · Fingerprint(32 B) · **HelixResidue(48 B)** · **TurbovecResidue(`Pq32x4` 16 B)** · Energy(f32) · Plasticity(u32) · EntityType(u16). `VALUE_TENANTS` = the stable row-relative byte carve `[32,186)` (reserve-don't-reclaim, contiguous, compile-time asserted ≤ 480). Presets: `Bootstrap` (EMPTY, zero-fallback **default**) · `Cognitive` (58 B: hot SoA columns, no codec residues) · `Compressed` (98 B: codec stack + fingerprint, no hot lifecycle) · `Full` (154 B: all 9 tenants). Built on existing `class_view::FieldMask` (presence) + `soa_envelope::ColumnDescriptor` — **no new presence type** (per "refactor into what exists"). `is_layout_preserving()` always `true` (carves WITHIN the reserved slab; `NODE_ROW_STRIDE=512` untouched → no `ENVELOPE_LAYOUT_VERSION` bump). Selection surface: defaulted `ClassView::value_schema(&self, ClassId) -> ValueSchema` (default `Bootstrap`, non-breaking — mirrors `edge_codec_flavor`). **Closes the SoA-extension dilution gap**: the formerly-comment-only helix-48 is now a first-class tenant alongside turbovec-`Pq32x4` + signed-`CoarseResidue`. +6 tests + 3 compile-time canon asserts; contract lib 611 green; clippy `-D warnings` + fmt clean.
- **Encode/measure kernels live in `ndarray` (the hardware layer), not the contract:** `ndarray::hpc::edge_codec` (Codebook k-means, `CoarseResidueCodec`, `ProductQuantizer`, `reconstruct_coarse`) + `ndarray::hpc::reliability` (Pearson r, Spearman ρ, Cronbach α, ICC(2,1), `FidelityReport`). Harness `examples/edge_codec_compare` measures all flavors × {blob, continuous} regimes. **Measured:** CoarseResidue dominates agreement (ICC 0.97–0.99, ρ 0.98, α 0.99); Pq32x4 keeps rank (ρ 0.60–0.67) but not absolute distance (ICC 0.11–0.29); CoarseOnly collapses on continuous (ICC 0.003); AMX `matmul_i8_to_i32` assign 100% vs scalar, 24–28 GMAC/s. ndarray commit `d3b608f`.
- **Deferred (flagged):** turbovec PQ4 *throughput* path (the FastScan nibble-LUT for the Pq32x4 flavor) blocked on the **#493 P2** build break — `lance-graph-turbovec` requests the `ndarray-simd` turbovec feature that was REMOVED (turbovec commit `7fa217c`); the polyfill fns are gone. turbovec's API is end-to-end (owns encode+scan), so it is a *PQ4 flavor*, not a residue-nibble-scan primitive. Fidelity (what ICC/Pearson/α measure) is independent of the fast kernel, so this is throughput-only follow-up.

## 2026-06-29 — Append: `facet::CascadeShape` cascade algebra (branch `claude/odoo-rs-transcode-lf8ya5`)

(Per APPEND-ONLY rule: new top-of-inventory entry. Branch work; records the contract type so a new session does not re-derive it. Part of the OGAR transpile-substrate arc — the compiled-`ClassView`-spine work, `OGAR/docs/OGAR-TRANSPILE-SUBSTRATE.md` §1.5/§1.6.)

### Current Contract Inventory — new entry

- **`facet::CascadeShape` + `facet::ClassArm` + `facet::CASCADE_UNITS` + `FacetCascade::{tier_bytes, cascade_byte, cascade_group_shared}`** (NEW; `lance-graph-contract::facet`, zero-dep `const fn`). Carving = "Both — one cascade algebra"; home = "lance-graph-contract (FacetCascade/ClassView)". `CASCADE_UNITS = 12` = the facet's 6×2 tier-bytes = a 12-field class's fields — unit-agnostic. **Carvings are VIEW rotations, not function layouts** (operator correction 2026-06-29):
  - `CascadeShape::{G6D2, G4D3, G3D4}` carves the 12-unit ladder as `G groups × D levels`, `G·D = 12`. `CascadeShape::ALIGNED = [G3D4, G6D2]` are the byte-aligned **defaults** (`group_of` is a pure SHIFT — `shift()` returns `Some(2)`/`Some(1)` — the canon's "tier-of-level is a shift, never a branch"). `CascadeShape::ROTATIONS = [G3D4, G4D3, G6D2]` is the full **rotation set** a ClassView may rotate through: a ClassView can **always rotate** (read the same bytes under a different carving) per class — the relief valve for **classid-stacking entropy** (rare, e.g. some Odoo classes — rotate instead of minting another classid).
  - **`G4D3` is the WORST CASE, not a co-equal carving** (operator: "4 group_of is a very bad and unwanted example"): it straddles tier boundaries, `shift()` is `None`, `group_of` must DIVIDE; excluded from `ALIGNED`, kept in `ROTATIONS` only as the deliberate rare rotation. `is_byte_aligned()` is the guard (`false` for G4D3) — reject the straddle on the common path.
  - **`ClassArm::{View, Functions}`** — the classid is an **additional switch** (operator: "functions should be encoded using the Classid as an additional switch (functions, view)"). Functions are NOT a facet carving — they are the DO arm (`ActionDef`/`KausalSpec` on the resolved node), reached by switching the classid to `Functions`; carvings address the THINK/`View` arm only. (OGAR THINK/DO split, `OGAR-AST-CONTRACT.md`.) `ClassArm::BOTH`, `is_functions()`.
  - `FacetCascade::tier_bytes()` = the 12 cascade bytes as one coarse→fine ladder (`hi` then `lo` per tier); `cascade_byte(shape,g,l)`; `cascade_group_shared(other,shape,g)` = per-group coarse→fine LCP redout.
  - **`canonical_node::GUIDS_PER_NODE = 32` + the clean/SoC-over-packed doctrine** (operator 2026-06-29): the 512-byte node = `NODE_ROW_STRIDE / 16` = **32 × 16-byte GUID slots** (`key` + `edges` = 2; value slab = 30). Doctrine: when a class needs more than fits cleanly in one slot, **Tetris each concern into its own slot (SoC)** rather than bit-pack — the 32-slot capacity is *why* the `G4D3` straddle / packing is almost never needed (it's the headroom that also lets a ClassView rotate and lets the rare classid-stacking-entropy case spread to a fresh slot instead of minting another classid). Compile-time assert (`GUIDS_PER_NODE == 32 && ·16 == NODE_ROW_STRIDE`) + test `guids_per_node_is_32_slots_clean_soc_over_packed` (asserts the 2+30 split). `facet::CascadeShape` cross-refs it from the `G4D3` worst-case doc.
  This generalises the OGAR GUID `3×4`-vs-`4×3` debate from nibble-units to byte/field-units and lands on the canon's verdict (aligned 3×4 default; straddling 4×3 worst-case). **The shared substrate the three language SDKs (§1.6) all read.** +4 facet tests (`cascade_rotations_are_total_but_only_aligned_are_defaults`, `classid_switch_separates_view_from_functions`, `tier_bytes_ladder_and_per_carving_grouping`, `cascade_group_shared_is_per_group_lcp`) + canonical_node `guids_per_node_*` + 4 compile-time asserts. Lib facet 8/8 + canonical_node 43/43 green; clippy `-D warnings` + rustfmt clean (probe-workspace verified offline — the workspace ndarray git dep is 403 offline).
  - **2026-06-29 correction (operator veto):** the "G4D3 = worst case to prevent" framing above is SOFTENED — **the shape is class-conditioned, not locked**. A ClassView is mapped from the class's *inherited* format and selected by `classid` (the filter); the shape follows: **Rails → `6×2`, other frameworks → `4×3`, the GUID → `3×4`** (operator: "Rails might need 6x2x8bit, others 4x3x8bit"). So `4×3` (`G4D3`) is **legitimate per-class**, not a thing to "reject" — its `group_of` divides (a per-class *cost* a class opts into), and `is_byte_aligned()`/`shift()`/`ALIGNED` now read as "distinguishes the shift fast-path from the divide shape," not "prevent." NEW `CascadeShape::from_levels(d)` — the class-conditioned `D ∈ {2,3,4}` selector (`2→G6D2`/`3→G4D3`/`4→G3D4`), inverse of `levels()`; the classid resolves `D`, never a global lock. Test renamed → `cascade_shapes_are_total_and_class_conditioned` (adds the `from_levels` round-trip). The earlier "quadruplet/4-bucket FieldMask" framing in ruff `soc` was likewise unlocked → byte-cardinality cap, class-conditioned shape (ruff #36). Facet 7/7 + canonical_node 43/43 green post-correction.
  - **2026-06-29 (later) correction — the "(ruff #36)" attribution on the line above is WRONG (append-only, prior line kept as record):** ruff PR #36 (`origin/main` tip `3d04e37` = "Merge PR #36", payload commit `c613094` "soc FieldMask cap 64 -> 256 (quadruplet) + bucket chaining") merged the **pre-veto LOCKED quadruplet** — `FIELD_MASK_BUCKET_BITS = 64`, `FIELD_MASK_MAX_BUCKETS = 4`, `field_mask_buckets()`, `FIELD_MASK_CAP = 64*4 = 256` — **NOT** the unlock. The veto edit was authored but never committed, so #36 shipped the pre-veto code. The actual unlock (`FIELD_MASK_CAP = MAX_SIBLINGS_PER_TIER` = byte cardinality, class-conditioned shape, `quadruplet`→`classview` test renames) lands via ruff commit `101928a` ("apply dropped operator veto"), which is **not yet on ruff `main`** — it is in the PR-to-main on branch `claude/odoo-rs-transcode-lf8ya5` (this arc). Until that ruff PR merges, ruff `soc` on `main` is still the locked quadruplet; the "unlocked (ruff #36)" reading becomes true only after it merges. (Confirmed by adversarial cross-repo audit, both P0 claims unrefuted at high confidence; lance-graph's own `facet.rs` is correctly class-conditioned on `main` and needs no change.)

## 2026-07-02 — Append: cross-session intake arc (PR #632; branch claude/v3-substrate-migration-review-o0yoxv)

(Per APPEND-ONLY rule: new top-of-inventory entries. Companion PR: OGAR #148 — merge OGAR first, then bump this repo's ogar-vocab lock pin so lance_graph_ogar COUNT_FUSE compares 68 == 68.)

### Current Contract Inventory — new entries

- **`codegen_spine::RouteBucketTyped`** (NEW; C6 merged verbatim from op-nexgen's vendored diff, codex-reviewed on nexgen PR #8). Kind-generic sibling of `RouteBucket` (`type Kind: Copy + Eq`) + `?Sized` blanket bridge (`impl<T: RouteBucket + ?Sized> RouteBucketTyped for T { type Kind = OdooMethodKind; }`) so non-Odoo codegen targets bring their own kind enum additively. Coherence rule: a type needing a different Kind skips the legacy trait. 12/12 module tests incl. dyn-object coverage.
- **`emission_scan`** (NEW; op-nexgen L2). Zero-dep typed-DDL adoption counter, `classid_scan`'s design-language sibling: `TypedForm {Typed, AnyTyped, RecordLink, Stub}` (#[non_exhaustive]) + tokenizer `classify_ddl_type` (precedence Stub > RecordLink > AnyTyped > Typed; word-boundary tokens so `many`/`recording` never false-match) + `EmissionCounts` fold with `typed_ratio()` (f64, mirrors `adoption_pct`). 15 tests. Module doc NAMES the contract scan-family pattern (Form enum + classify_* + fold-to-counts): the next governance counter mirrors it.
- **`ogar_codebook` 0x08XX OCR rows** — `unicharset` (0x0801) / `recoder` (0x0802) / `charset` (0x0803) / `network_layer` (0x0804) mirroring OGAR #148's mint (container kinds only; content never becomes concepts — Osint zero-rows precedent). `network_layer` = the KIND "a Tesseract recognizer network layer"; the 27 subclasses live in the classid custom-low half (`NetworkType` ordinal), NOT 27 slots. Drift-guard test extended. CODEBOOK now 69 entries.
- **Rulings + intake record:** EPIPHANIES E-V3-XSESSION-INTAKE-1(+RULINGS), E-V3-GRAPHRAG-INV-1; handover `.claude/handovers/2026-07-02-cross-session-wishlist-intake.md`; plan Addendum-10/11 (per-consumer classid ownership + tripwires ratified; R-1 naming phantom closed — `domain:appid:classview`; R-2 closed — 512-byte row frozen, edges via strided view; L3 new-Arrow-schema design killed; five post-fuse workstreams enumerated). Knowledge: `graphrag-rs-inventory.md`.
