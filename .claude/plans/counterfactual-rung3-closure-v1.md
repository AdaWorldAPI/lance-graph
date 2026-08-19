# Counterfactual rung-3 closure — encoding is specified, runtime is absent — v1

> **⊘ RE-FRAMED 2026-08-19 (operator ruling E).** This plan **survives**, but
> its justification changes and the change is load-bearing. Widening
> `contract::nars::InferenceType` is **not** "making CausalEdge64 better" —
> under ruling E, CE64 is a **codec/projection**, and the contract enum is
> how a reader *decodes what the codec already stores*. Today it decodes
> wrongly: `from_mantissa(−6)` returns `Synthesis` (whose own mantissa is
> **+5**) — a silent direction inversion. That is a **decoder defect in a
> projection**, which is exactly the kind of thing this plan may fix.
>
> **What this plan must NOT become:** a route to adding new CE64 bit-field
> semantics. `.claude/v3/soa_layout/le-contract.md` forbids it and M20
> retired the awareness mantissa on measurement. The canonical meaning of a
> counterfactual belongs in V3 rows + HHTL locality + masks + provenance;
> CE64 then projects it. Cross-ref
> `docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md` §4, and
> §C6 for the measured fact that **`observed` / `inherited` / `extrapolated`
> have no representation anywhere** — the provenance surface this plan's
> `Counterfactual` variant will eventually need to sit beside.

> **Status:** DESIGN. Zero code lands with this doc. Every deliverable is
> pre-registered with a falsifiable gate (can-it-fire **and** can-it-stay-silent,
> per the P0 falsifiability rule); no mechanism lands before its gate is green.
> **The gap, one sentence:** Pearl rung 3 has a *fully specified* encoding
> (mantissa −6, `MASK_SPO`, a deposit path, an iron separate-lane invariant) and
> a *fully absent* runtime (the canonical contract enum cannot name it, two
> bridge sites silently collapse it, and no `abduce → intervene → predict` chain
> executes anywhere).
> **The constraint that shapes everything below:** the V3 ruling retires
> CausalEdge64 *awareness extension*. This plan therefore touches **no CE64
> bit**. It operates on the **already-minted** mantissa semantics plus the
> **contract enum**, which is a Rust type, not a layout.
> **Consumes, does not re-plan:** `causal-rung-standing-wave-v1.md` (D-CSW-1..3 —
> the 2³-views-over-one-`SpoFacet` reimagining), `triangle-tenants-gestalt-separation-v1.md`
> §2a/§3a (D-TRI-5/6 — the emulation apex + the pyramid), `.claude/v3/soa_layout/le-contract.md`
> §"Let go of the cramped 64-bit register".
> **Index:** `.claude/board/INTEGRATION_PLANS.md` (prepend in the same commit as
> the first implementing PR). **D-ids:** D-CFR-1..6 (STATUS_BOARD rows, same commit).

---

## §1 FROZEN DECISIONS (quoted; this plan may not renegotiate any of them)

**F1 — no new CE64 bit semantics.** `.claude/v3/soa_layout/le-contract.md:150-160`,
operator 2026-07-02: *"The prior approach — cramming awareness into a 64-bit
packed edge register and 'hoping for a 3-bit mantissa to mean the whole
awareness' … is **let go**."* And, mechanically: *"Do NOT extend CausalEdge64
bit fields to carry new awareness semantics; new semantics land as facet
layouts (L1–L8 + sanctioned readings)"* (`:159-160`).

**F2 — CE64 is DEMOTED, not deleted; three carriers only.** Same file `:161-166`,
**[H] → RESOLVED 2026-07-18**: the residual role is *"exactly THREE concrete
carriers: (1) the MailboxSoA per-row baton edge (`mailbox_soa.rs:92`), (2) the
perturbation baseline (E-THINKING-TENANTS-V3-1), (3) the p64 palette address
(`p64-bridge/src/lib.rs:30`). Only the awareness-mantissa is retired (M20)."*
`.claude/v3/ENTROPY-MILESTONES.md:45` (M20) carries the same ruling with its
evidence (D-MTS-6 GREEN, k\*=1, 2 bits/edge matches all three awareness proxies).

**F3 — the mantissa that ALREADY EXISTS is not a new semantic.** `causal-edge`
`layout.rs:16-32` fixes `INFER_SHIFT = 46`, 4-bit signed, and its slot table
already names `6 = Intervention(+)/Counterfactual(−)`. **Reading and preserving
a slot that was minted in 2026-05-16 is not "extending CE64".** This is the
narrow doorway F1 leaves open, and this plan stays inside it.

**F4 — no REAL CE64 shrink without D-MTS-6b.** `.claude/v3/VISION.md:109-110`:
*"Fence: D-MTS-6b (driver-integrated fixture) gates any REAL CausalEdge64
shrink — the proxies are proxies [ASPIRATION until 6b]."* Nothing here shrinks.

**F5 — the separate-lane invariant.** `crates/lance-graph-contract/src/counterfactual.rs:21-24`:
*"A counterfactual **stays in a separate lane — it is NEVER written as observed
SPO truth.** … The `InferenceType::Counterfactual` tag is the mechanical
enforcement of that invariant."* **Consequence that makes this plan load-bearing
rather than cosmetic:** the enforcement mechanism is a tag the canonical enum
cannot currently express. The invariant is asserted and unenforceable.

**F6 — I-LEGACY-API-FEATURE-GATED applies to the enum widening** (`CLAUDE.md`).
Widening a public enum is a semver-visible change to the crate every consumer
compiles against; the compat note and the match-exhaustiveness fallout are gate
material, not follow-up.

---

## §2 INPUT INVENTORY (every row read at the cited line this session)

### 2a — The type-width root cause

| Site | State |
|---|---|
| `crates/lance-graph-contract/src/nars.rs:12-23` | `InferenceType` — **5 variants** (Deduction, Induction, Abduction, Revision, Synthesis) |
| `crates/causal-edge/src/edge.rs:11-30` | `InferenceType` — **8 variants**; `Intervention = 5`, `Counterfactual = 6`, `Reserved7 = 7` |
| `crates/lance-graph-contract/src/nars.rs:64-72` | `to_mantissa()` — 5 arms; **no `6` is producible** |
| `crates/lance-graph-contract/src/nars.rs:79-109` | `from_mantissa()` — `mag 6` falls to `_ => Synthesis` |
| `crates/causal-edge/src/edge.rs:90-133` | `from_mantissa()` — `mag 6` → `forward ? Intervention : Counterfactual` |
| `.claude/v3/MODULE-TABLE.md:200` | contract `nars.rs` is *"the contract-canonical copy"*, *"shared by n8n-rs/planner/crewai-rust"* |

The contract is the canonical copy every consumer compiles against, so **rung-3
semantics cannot cross the contract boundary at all.**

> **⚠ SESSION CORRECTION — the collapse is worse than "magnitude discarded".**
> The pre-session framing was that the collapse *preserves the sign and discards
> the magnitude*. Measured, there are **two different** collapses and only one of
> them preserves the sign:
>
> | Path | In | Out | Out's own mantissa | Verdict |
> |---|---|---|---|---|
> | enum-level (planner stopgaps) | `Counterfactual` | `Abduction` | **−1** | sign kept, magnitude 6→1 lost |
> | mantissa-level (`contract::nars::from_mantissa`) | **−6** | `Synthesis` | **+5** | **sign FLIPPED**, magnitude lost |
>
> A CE64 edge stamped −6 by `deposit_counterfactual` that re-enters through the
> contract's own `from_mantissa` comes back as a **forward-chain** rule. That is
> not a lossy narrowing; it is a direction inversion, and it is silent.

### 2b — The stopgaps (all four carry their own admission)

| Site | Code | Self-admission |
|---|---|---|
| `lance-graph-planner/src/nars/inference.rs:71-72` | `Intervention ⇒ Abduction`, `Counterfactual ⇒ Abduction` | `:58-61` *"the 3-step chain's first step (abduce latent context) dominates semiring selection; the intervene + predict sub-steps compose … at the caller"* |
| `lance-graph-planner/src/orchestration_impl.rs:206-207` | same two arms | `:202-205` *"contract InferenceType will gain matching variants in a follow-up PR (W2/meta-r1 scope)"* |
| `lance-graph-planner/src/thinking/nars_dispatch.rs:126-129` | `Counterfactual ⇒ CamWide{top_k:64, window:128}` | `:122-125` *"the intervene and predict sub-steps are handled by the caller composing Intervention + Deduction queries"* |
| `lance-graph-planner/src/cache/nars_engine.rs:242-246` | `f = fa·fb; c = fa·fb·ca·cb·0.70` | `:235-241` *"Implemented as Deduction ×0.70 confidence modifier. TUNED-LATER"* |

**Nothing composes the caller.** `grep 'fn intervene'` over `crates/` returns
one real function (plus two test-fn names matching the substring,
`tests/intervene_counterfactual.rs:27,56`) — `lance-graph/src/graph/arigraph/triplet_graph.rs:789
intervene_on` — and no site chains abduce → intervene → predict. The three
sub-steps are named in four doc comments and executed in none.

> **⚠ SESSION CORRECTION to `triangle-tenants-gestalt-separation-v1.md` §2a.**
> That plan cites *"`lance-graph-cognitive/world/counterfactual.rs::intervene()`
> (Pearl Rung 3 do-calculus on fingerprints, implemented + integration-tested)"*.
> **The function does not exist.** `crates/lance-graph-cognitive/src/world/counterfactual.rs`
> exports exactly `substitute_binding` (`:83`), `worlds_differ` (`:104`),
> `multi_substitute_binding` (`:112`); `grep 'fn intervene' crates/lance-graph-cognitive/src/`
> is empty. The crate is also workspace-**excluded** (`Cargo.toml:42`). The
> substitution machinery is real; the intervention entry point is not. Corrected
> here, not in that plan (append-only; that plan's §2a gets a dated pointer when
> its own next revision lands).

### 2c — What is NOT stuck (the plane half is wired)

`.claude/board/STATUS_BOARD.md:432` — **D-TRI-6**: *"In PR (P3) — ascent loop
WIRED (driver rung→predicate-plane widen; identity-at-base, superset-monotone);
settlement probe green; real-cycle distribution + jc threshold calibration still
open."* So the *plane* half of rung 3 ascends. Only the **rule-identity** half —
which inference rule the ascended rung is executing — is stuck at the enum
boundary. That asymmetry is the whole shape of this plan.

### 2d — The L1 mask conflict (blocks superset-monotone ascent)

| Site | L1 mask |
|---|---|
| `lance-graph-contract/src/cognitive_shader.rs:239-249` | `1 => 0b001` (O) — self-labelled *"CONVENTION, hand-chosen pending its own probe"* |
| `causal-edge/src/pearl.rs:14, 40-42` | `SO = 0b101` — *"**Level 1: Association.** Pure observational correlation"* |
| `lance-graph-contract/src/orchestration_mode.rs:152-153` | `SO = 5` — *"101: Subject×Object — P(Y\|X), Pearl Level 1 (SEE)"* |

Two of three say L1 = `0b101`. Under `0b101`, `L1 ⊄ L2 (0b011)` — the
superset-monotone property D-TRI-6 just certified **breaks at the base**. The
contract's own doc comment already flags this as unprobed. A probe is owed.

### 2e — "Pearl 2³" names THREE unrelated structures

| Structure | Site | What the 8 are |
|---|---|---|
| **CausalMask powerset** | `causal-edge/src/pearl.rs:26-49`; `contract/src/orchestration_mode.rs:141-158` | the 8 subsets of {S,P,O} — *projection planes* |
| **CausalAmbiguity** | `contract/src/grammar/ticket.rs:23-35` | *"the 2³ = 8 possible SPO **role assignments** (subject/object swap, passive, ergative shift)"* — a parse-ambiguity bitmask |
| **pearl_queries** | `contract/src/exploration.rs:24-30` | a **3-way** SEE / DO / IMAGINE decomposition — not 8 at all |

Three different arities (8 planes, 8 permutations, 3 rungs) under one phrase.
`nars_engine.rs:252` weights "the 8 Pearl projections"; `ticket.rs` masks 8
role assignments; a reader cannot tell from the phrase which is meant.

### 2f — Multi-hop and revision fragmentation (the DEFERRED evidence)

| Site | State |
|---|---|
| `lance-graph-planner/src/nars/belief.rs:279` `close_transitive` | **genuine** multi-hop deduction to a true fixed point (`:270-277` argues termination from bounded monotone expectation) |
| `lance-graph-planner/src/strategy/truth_propagation.rs:36-52` | `plan()` returns `Ok(input)` — a **documented no-op**; MODULE-TABLE:325 concurs |
| `lance-graph-planner/src/physical/accumulate.rs` | `AccumulateOp` (`:20`) — **no `fn execute`** anywhere in the file |
| `causal-edge/src/network.rs:61-77` `forward_chain` | composes via `CausalEdge64::forward` (`edge.rs:631-661`) — palette compose tables + a **float `match` on the decoded rule**… |
| `causal-edge/src/network.rs:39` | …while `CausalNetwork` **declares `nars_tables: NarsTables`** (`tables.rs:41-46`: `revision: Vec<[PackedTruth; 256*256]>`, `deduction: [PackedTruth; 256*256]`, lookup at `:122-125`) that `forward_chain` never reads. Dead weight on the hot path. |
| revision formula | **19** `fn revise` sites workspace-wide; at least **3 full reimplementations** of the same evidence-weighted merge: `nars/truth.rs:57-70`, `thinking/sigma_chain.rs:148-159`, `physical/accumulate.rs:160-178` |
| `lance-graph-planner/src/thinking/mod.rs:88` | `let sigma_stage = SigmaStage::Omega; // Start at observation` — hardcoded; `:97` likewise pins `rung: RungLevel::Surface`. **No sigma stage consumes `InferenceType` or a truth type.** |

### 2g — The shipped deposit path, and its structural blocker

`crates/lance-graph-contract/src/counterfactual.rs` (D-ATOM-4, 546 lines):
`deposit_counterfactual` (`:140-158`) writes `edge.set_inference_mantissa(-6)`
with the comment *"`InferenceType::Counterfactual.to_mantissa() == -6` — the
road-not-taken nibble"*. `EpisodicEdge` (`:178-188`) is the zero-dep trait;
`RawEdge` implements it (`:472-479`, clamped to i4 at `:482-490`) so the deposit
path is **testable today without any bridge**. The bridge itself is BLOCKED
(`:172-177`): *"Options: (a) impl in `causal-edge` gated on a
`lance-graph-contract` feature; (b) newtype in a thin bridge crate."*

**Measured constraints on that choice** (new this session — both crates state a
zero-dep invariant in their own manifests):

- `lance-graph-contract/Cargo.toml:11-17`: zero deps *"even of optional path
  deps: as a WORKSPACE MEMBER, any path dep here … breaks EVERY cargo
  invocation in CI (learned 2026-07-07)"*. So the dep can never point contract → causal-edge.
- `causal-edge/Cargo.toml:22-25`: *"No dependencies — this crate is
  self-contained. TrustTexture is defined locally (not imported from
  lance-graph-contract) to preserve the zero-dep invariant."* Option (a) spends
  exactly that invariant.
- `causal-edge` is workspace-**excluded** (`Cargo.toml:38`) yet already path-dep'd
  by two members — `lance-graph-planner/Cargo.toml:27` and
  `cognitive-shader-driver/Cargo.toml:43`. **So availability was never the
  blocker; the zero-dep invariants are.**
- A **third option exists and was not in the doc comment:** the orphan rule
  forbids a bare `impl EpisodicEdge for CausalEdge64` in the planner (both types
  foreign), but a **newtype in an existing dependent** (planner or
  cognitive-shader-driver, each of which already has both crates in scope) costs
  no new crate and no new dep edge.

### 2h — The consumer boundary the widening crosses

`cognitive-shader-driver/src/driver.rs:29` imports **causal-edge's** 8-variant
enum; `:648-657` maps it to `NarsInference` with a `_ => NarsInference::Revision`
catch-all justified by *"style_ord_to_inference never returns Reserved5/6/7"*.
The planner side imports the **contract's** 5-variant enum. `contract/src/plan.rs:19`
puts `inference_type: InferenceType` on `ThinkingContext`, and `orchestration.rs`
(MODULE-TABLE:204) imports it into `UnifiedStep` — the surface `ladybug-rs`
consumes. So the widening is visible to an out-of-tree consumer through
`OrchestrationBridge`.

**The hazard that makes D-CFR-2 mandatory:** `orchestration_impl.rs:144`, `:149`,
`:176` are `_ =>` wildcards. Widening the enum produces **no compile error
there** — the new variants are silently absorbed into `CamExact` / `Boolean`.
The match-exhaustiveness falsifier fires at `nars.rs:45-53`, `nars.rs:64-72`,
`grammar/inference.rs:33-44` and `pearl_junction.rs:126-135`, and **is blind at
exactly the three sites that route** rung-3 traffic.

---

## §3 PROPOSED RESOLUTION (ordered; each step gated before the next starts)

### D-CFR-1 — widen `contract::nars::InferenceType` to 7 and close the mantissa round-trip

Add `Intervention` and `Counterfactual` to `contract/src/nars.rs:12-23`, extend
`to_mantissa` (`+6` / `−6`, matching `causal-edge/src/edge.rs:77-79`) and
`from_mantissa` (`mag 6 → forward ? Intervention : Counterfactual`, matching
`edge.rs:124-130`). Add `QueryStrategy` arms so `default_strategy()` stays total.
Delete the two stopgap mappings. `Reserved7` is **not** added — the contract
models rules, not slots.

*Gate (all four halves required):*
1. **can-it-fire:** `Counterfactual.to_mantissa() == -6` **and**
   `from_mantissa(-6) == Counterfactual` — a full contract round-trip, plus the
   cross-crate value-bridge assertion `contract::to_mantissa(X) == causal_edge::to_mantissa(X)`
   for all 7 shared variants.
2. **disable-run (red):** revert `from_mantissa`'s `mag 6` arm to `_ => Synthesis`
   and the test must go red **naming the sign flip** (`−6` in, `+5` out), not
   merely "wrong variant".
3. **the two stopgap sites are DELETED** — `inference.rs:71-72` and
   `orchestration_impl.rs:206-207` carry real arms; a grep for the string
   *"will gain matching variants in a follow-up PR"* returns zero.
4. **can-it-stay-silent:** the 5 pre-existing variants' `to_mantissa` /
   `from_mantissa` / `default_strategy` results are byte-identical before and
   after — pinned as an explicit 5-row table, so the widening cannot be a
   silent re-encoding of the core set (F6 / I-LEGACY-API-FEATURE-GATED).

### D-CFR-2 — wildcard census + routing distinctness (the falsifier D-CFR-1 is blind to)

Convert `orchestration_impl.rs:139-149` and `:171-177` to exhaustive matches, and
audit every `_ =>` arm over an `InferenceType` in `crates/` for silent absorption.
Ship the compat note (F6): a doc-comment migration pointer on the enum naming the
two added variants and the mantissa they carry.

*Gate:*
- **can-it-fire:** each of the 7 variants routes to a **distinct-by-design**
  `(QueryStrategy, SemiringChoice)` pair, or its non-distinctness is asserted
  deliberately with a one-line reason. Anti-vacuity: at least 5 distinct pairs
  across 7 variants — a mapping that collapses ≥3 variants onto one pair fails.
- **can-it-stay-silent:** a `Counterfactual` step through
  `OrchestrationBridge::resolve_thinking` must **not** produce `DnTreeFull`
  (Abduction's strategy) — the exact pre-fix behaviour, pinned as a negative.
- **disable-run:** restoring any one `_ =>` arm makes the distinctness test red.

### D-CFR-3 — L1 mask probe (`0b001` vs `0b101`)

Pre-register **before** running: measure both L1 candidates on a labelled
observational fixture. Pass criterion is stated as a pair, not a winner —
(i) retrieval/settlement quality, and (ii) the structural check
`L1 ⊆ L2 ⊆ L3` (superset-monotone), which `0b001` satisfies and `0b101` does not.

*Gate:* both metrics recorded; **either outcome is a PASS**, with the handling
fixed in advance — `0b101` wins on quality ⇒ superset-monotone is **falsified**
as a global property and D-TRI-6's ascent claim is re-scoped to L2→L3 (the two
probe-certified rows) rather than silently kept; `0b001` wins ⇒ the CONVENTION
label at `cognitive_shader.rs:240-242` is promoted to FINDING and
`causal-edge/src/pearl.rs:14` gets a dated demarcation note. **A null result
(indistinguishable) leaves the CONVENTION label in place and is recorded as
such** — it does not license promotion.

### D-CFR-4 — Pearl 2³ disambiguation (doc-level, no code)

One knowledge doc naming the three structures of §2e with their arities and
their non-relationship, plus a one-line `which-2³` header comment at each of the
four sites (`pearl.rs`, `orchestration_mode.rs`, `ticket.rs`, `exploration.rs`).

*Gate:* mechanical — every site that says "2³" or "Pearl 2^3" carries the
disambiguating header (grep, `== 4` not `>= 4`). Anti-vacuity: the doc must
state a **falsifiable** distinctness claim (the three arities are 8 / 8 / 3 and
the two 8s index different sets), so a future merge proposal has something to
refute rather than a taxonomy to nod at.

### D-CFR-5 — land the `EpisodicEdge` bridge (gated on §5 OD-1)

Whichever location the operator picks, the deliverable is identical: a real
`CausalEdge64` reaches `deposit_counterfactual`.

*Gate:*
- **can-it-fire:** `deposit_counterfactual(split_verdict, &mut edge)` on a real
  `CausalEdge64` under `causal-edge-v2-layout` → `edge.inference_mantissa() == -6`
  → `contract::InferenceType::from_mantissa(...) == Counterfactual` (depends on
  D-CFR-1 being green).
- **can-it-stay-silent:** a **non-split** verdict returns `false` and leaves the
  mantissa **byte-identical** — the guard must not fire on everything.
- **legacy half (F6):** under `--no-default-features` (v1 layout) the write is a
  documented no-op and the test asserts the no-op *observably*, per
  I-LEGACY-API-FEATURE-GATED's paired-test rule.

### D-CFR-6 — inertness test for the two TUNED-LATER discounts

`nars_engine.rs:232` (`0.85`, Intervention) and `:244` (`0.70`, Counterfactual)
are hand-chosen constants with no test that they do anything.

*Gate (the threshold-inertness rule):* raising the Counterfactual discount toward
`1.0` must make some ranked outcome change; lowering it toward `0` must make a
different one change. If neither moves anything, the knob is decoration and the
finding is recorded as such rather than the constant defended. Both constants
stay labelled hand-tuned until a `jc`/Jirak-derived calibration replaces them
(I-NOISE-FLOOR-JIRAK).

---

## §4 NON-GOALS (explicitly out of scope for this plan)

1. **No new CausalEdge64 bits, fields, or widths** — F1/F2/F4. Slot 6 already
   exists; this plan reads it.
2. **No CE64 shrink** — fenced on D-MTS-6b (F4).
3. **No `NarsTables` rewrite.** Wiring the declared-but-unread tables into
   `forward_chain` (§2f) is named as DEFERRED, not attempted here.
4. **No revision-formula unification.** 19 sites, ≥3 reimplementations — a
   collapse needs the D-TSC-1 discipline (jc-measure agreement first, collapse
   only on measured identity). DEFERRED.
5. **No `abduce → intervene → predict` composition.** D-CFR-1..2 make the rung-3
   *identity* survivable; they do not execute the chain. DEFERRED.
6. **No new inference machinery, semiring, or trait.** The fix is enum variants
   plus a mantissa round-trip.
7. **No sigma-chain rewiring**, no `AccumulateOp::execute`, no
   `TruthPropagation` un-no-op-ing. DEFERRED.
8. **No `Reserved7` in the contract** — a slot is not a rule.
9. **No V3 facet-layout work.** The 96-bit facet is where new awareness
   semantics belong (F1); nothing here mints one.

---

## §5 OPEN OPERATOR DECISIONS

**OD-1 — where does `impl EpisodicEdge for CausalEdge64` live?** Three options,
with this session's measured costs (§2g):

| Option | Cost | Note |
|---|---|---|
| (a) in `causal-edge`, feature-gated on `lance-graph-contract` | spends causal-edge's stated zero-dep invariant (`Cargo.toml:22-25`) | the doc comment's first option; the crate manifest argues against it |
| (b) thin bridge crate | one new crate + one new dep edge; both zero-dep invariants intact | the doc comment's second option |
| (c) **newtype in an existing dependent** (planner or cognitive-shader-driver) | zero new crates, zero new dep edges; both invariants intact | **not in the doc comment**; both crates already hold both deps (`planner/Cargo.toml:27`, `driver/Cargo.toml:43`). Orphan rule forbids a bare impl, so a newtype is required. |

Recommendation offered, not taken: (c) if one consumer needs it, (b) if two or
more will. **The operator decides; D-CFR-5 is blocked until then.**

**OD-2 — how is a D-CFR-3 outcome of "`0b101` wins" handled?** The handling is
pre-registered in D-CFR-3 (re-scope superset-monotone to L2→L3), but the
*consequence* is a live claim on another plan's certified property, so the
operator should confirm the re-scope is acceptable **before** the probe runs —
otherwise a green probe creates pressure to re-interpret the result rather than
accept it. This is the anti-HARKing gate for D-CFR-3.

**OD-3 — CascadeChannels8 tension (flagged, not resolved).**
`.claude/v3/FUTURE-DESIGN.md:101-105` names `layered.rs::CascadeChannels8` the
*"First wiring target — the confirmed NEXT edge"*, gate = M8 parity, and says it
*"collapses into `causal_edge::CausalEdge64`'s signed mantissa slot"*. **That is
the same mantissa slot M20 retires** (`ENTROPY-MILESTONES.md:45`). Wiring a new
per-level cognitive carrier INTO a retired register is a direct F1 collision.
Not this plan's to resolve — but a session that picks up CascadeChannels8 must
route it to a facet layout, or the operator must carve out an exception.

---

## §6 DEFERRED — missing integration (named so it is not mistaken for done)

| # | Gap | Evidence | Why deferred |
|---|---|---|---|
| DEF-1 | **The `abduce → intervene → predict` composition** never runs. Four doc comments delegate it to "the caller"; no caller exists (`grep 'fn intervene'` → 1 real fn, `triplet_graph.rs:789`, + 2 test-name substring hits). | §2b | Needs a composition site + a fixture; D-CFR-1..2 are its prerequisite (the chain cannot be typed today). |
| DEF-2 | **Three unrelated multi-hop paths.** `belief.rs:279` `close_transitive` is real; `truth_propagation.rs:36-52` is a documented no-op; `accumulate.rs` `AccumulateOp` has no `execute()`. | §2f | Unification is a D-TSC-1-shaped measure-then-collapse wave, not a drive-by. |
| DEF-3 | **`NarsTables` is dead weight on the multi-hop path.** `network.rs:39` declares a 256×256 deduction table + revision tables; `forward_chain` (`:61-77`) never reads them, computing truth by float `match` in `edge.rs:656+`. | §2f | Wiring it changes numeric output; needs a parity pin against the current float path first. |
| DEF-4 | **Revision-formula dedup.** 19 `fn revise` sites; ≥3 full reimplementations (`nars/truth.rs:57`, `sigma_chain.rs:148`, `accumulate.rs:162`). | §2f | Collapse only on measured identity (D-TSC-1 lesson); the three differ in what they feed as `f` and are **not** obviously the same function. |
| DEF-5 | **Sigma-chain truth wiring.** `thinking/mod.rs:88` hardcodes `SigmaStage::Omega` and `:97` hardcodes `RungLevel::Surface`; no stage consumes `InferenceType` or a truth type. | §2f | The ascent half is D-TRI-6's; the sigma half has no owner yet. |
| DEF-6 | **CascadeChannels8 first-wiring-target.** `FUTURE-DESIGN.md:103`, gate M8 parity (`ENTROPY-MILESTONES.md:33`, QUEUED). **Tension: it wires INTO the mantissa slot M20 retires.** | OD-3 | Flagged for operator; blocked on the F1 carve-out question, not on effort. |
| DEF-7 | **`ScenarioBranch` / mirror-SoA emulation** (`triangle-tenants…-v1.md` §2a, D-TRI-5) — the apex customer of a working rung 3. | §2b correction | Depends on DEF-1 and on the corrected inventory of what `lance-graph-cognitive` actually exports. |

---

## §7 Gate summary

| D-id | Deliverable | Can-it-fire | Can-it-stay-silent / disable-run |
|---|---|---|---|
| D-CFR-1 | 7-variant contract enum + mantissa round-trip | `−6 ⇄ Counterfactual`; cross-crate `to_mantissa` parity, all 7 | 5 core variants byte-identical; reverting `mag 6` → red **naming the sign flip**; both stopgap strings grep to zero |
| D-CFR-2 | wildcard census + routing distinctness | ≥5 distinct `(strategy, semiring)` pairs over 7 variants | `Counterfactual` must NOT route to `DnTreeFull`; restoring any `_ =>` → red |
| D-CFR-3 | L1 mask probe | both L1 candidates measured on a labelled fixture | outcome handling pre-registered **both ways** (OD-2); null result leaves CONVENTION standing |
| D-CFR-4 | Pearl 2³ disambiguation doc | every "2³" site carries a `which-2³` header (`== 4`) | the doc states a refutable distinctness claim (8/8/3), not a taxonomy |
| D-CFR-5 | `EpisodicEdge` bridge (blocked on OD-1) | real `CausalEdge64` → mantissa `−6` → `Counterfactual` | non-split verdict leaves the edge byte-identical; v1 no-op observably asserted |
| D-CFR-6 | discount inertness | raising `0.70` changes a ranked outcome | lowering it changes a **different** one; neither ⇒ recorded as decoration |

**Ordering:** D-CFR-1 → D-CFR-2 (its blind spot) → D-CFR-5 (needs 1) . D-CFR-3,
D-CFR-4 and D-CFR-6 are independent and may run in parallel. **Nothing in §6
starts until §7 is fully green** — the deferred items all assume a rung-3
identity that survives the contract boundary, and today it does not.
