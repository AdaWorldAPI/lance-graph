# Cognitive Substrate Convergence — i4 Mantissa, Gapless Baton, Active Inference (v2)

> **Status:** ACTIVE — sprint-11 Phase A/B COMPLETE (pending merges); sprint-12 Phase C in flight via Wave F
>
> **Version:** v2 (2026-05-16) — successor to v1 (2026-05-15)
> **Predecessor:** `.claude/plans/cognitive-substrate-convergence-v1.md` (v1 — original proposal, authored during live cross-session A2A discussion 2026-05-15)
>
> **READ BY:** integration-lead, truth-architect, host-glove-designer, palette-engineer, family-codec-smith, certification-officer, bus-compiler, nars-engineer, anyone implementing sprint-12's D-CSV-8..D-CSV-12 or touching `QualiaColumn` / `EdgeColumn` / `MailboxSoA` / `CollapseGate` wire format / `MUL` evaluation path
>
> **CONSOLIDATES:**
> - v1 plan (`.claude/plans/cognitive-substrate-convergence-v1.md`) — all architecture sections UNCHANGED unless annotated
> - sprint-11 implementation outcomes: Wave A (PR #383), Wave B (PR #384), spec patches (PR #381)
> - sprint-11 Wave F partial: D-CSV-5a (PR #385), D-CSV-6a+7 (PR #386), D-CSV-8+9 (PR #387), D-CSV-10 in-flight (W-F1)
> - sprint-12 forward plan: D-CSV-11/12 productized; D-CSV-13/14/15 new entries; on-Think methods sprint-13+
>
> **DOES NOT REPLACE:** per-PR specs in `.claude/specs/pr-ce64-mb-*.md` — those remain implementation-level contracts.
> This v2 is the **architectural anchor** updated with sprint-11 outcomes, carrying the same locking function as v1
> while surfacing what shipped vs what carries forward.

---

## 0. Status delta — sprint-11 outcomes and sprint-12 forward plan [UPDATED 2026-05-16]

### §0.1 What shipped in sprint-11 (Waves A–E and Wave F partial)

**Phase A (substrate primitives) — COMPLETE:**

| D-id | PR | Commit | Outcome |
|---|---|---|---|
| D-CSV-1 | #383 | `03bd175` | `causal-edge` crate v2 layout shipped. OQ-CSV-2 ratified: 6 bits (default). Feature-gated via `causal-edge-v2-layout`; crate bumped 0.1.0 → 0.2.0. |
| D-CSV-2 | #384 | In PR | `QualiaI4_16D` type in `lance-graph-contract::qualia`. OQ-CSV-1 ratified: Option α (convergence-observable vocab — arousal/valence/tension…, dropping dim 16 "integration"). 14 tests pass. |
| D-CSV-3 | #383 | `03bd175` | Signed-mantissa `InferenceType` expansion; PR-LL-1 Intervention/Counterfactual absorbed into canonical edge enum via Reserved5/6. |
| D-CSV-4 | #383 | `03bd175` | `CollapseGateEmission` shipped in contract crate. Vec instead of SmallVec to preserve zero-dep (SmallVec optimization deferred — see §8 annotation). |

**Spec patches (pre-sprint-11 prep) — COMPLETE:**

PR #381 (merged 2026-05-16, commit `a7c0545`). All 8 W2/W3/W4/W5/W6/W7/W10/W11 sprint-10 specs patched. ~1,200 LOC actual (underestimated in v1: W3 codex P1 fix +280 LOC over estimate; W5 full WitnessCorpus section +16 LOC over estimate).

**Phase B (storage and dispatch path) — PARTIAL (in PR / Wave F):**

| D-id | PR | Status |
|---|---|---|
| D-CSV-5a | #385 | QualiaColumn sibling-i4 column in `cognitive-shader-driver` — In PR |
| D-CSV-6a + D-CSV-7 | #386 | `WitnessCorpus` (partial) + `MailboxSoA` W-slot + `apply_edges` — In PR |
| D-CSV-5b | Post-#385 | Cutover (remove `[f32; 18]`) — Queued sprint-12 |
| D-CSV-6b | Post-#386 | Full CAM-PQ-indexed WitnessCorpus | — Queued sprint-12 |

**Phase C (reasoning path) — PARTIAL (Wave F):**

| D-id | PR | Status |
|---|---|---|
| D-CSV-8 + D-CSV-9 | #387 | MUL integer SIMD (scalar path shipped; AVX-512/NEON deferred — TD-D-CSV-8-SIMD-1) + 8ch↔SPO transcoder — Shipped |
| D-CSV-10 | Wave F W-F1 | Σ-tier sigma-tier-router crate — In PR |

### §0.2 What carries over to sprint-12

- D-CSV-5b (QualiaColumn cutover — remove `[f32; 18]` after all consumers migrate via Phase 5a sibling)
- D-CSV-6b (full CAM-PQ-indexed `WitnessCorpus` — 6a ships partial; full unbounded corpus is sprint-12)
- D-CSV-11 productization (vertical streaming structs in ndarray — now in PR via Wave F W-F4/5/6)
- D-CSV-12 (scalar splat ops on i4 substrate — in PR via Wave F W-F7; on-Think methods deferred to sprint-13+)
- D-CSV-13 (NEW: SIMD vectorization of D-CSV-8 i4 MUL evaluation — sprint-12 follow-on to scalar path shipped in #387)
- D-CSV-14 (NEW: on-Think method migration for D-CSV-12 splat ops — sprint-13+ per original estimate)
- D-CSV-15 (NEW: Σ10 Jirak-derived threshold — TD-7 resolution, VAMPE coupled-revival sprint-13+)

### §0.3 New sprint-12 infrastructure observations

Three anti-patterns surfaced during sprint-11 that the sprint-12 fleet must guard against:

- **Subagent isolation pattern** — workers building cognitive-shader-driver crate hit workspace members/exclude conflict (TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1); workaround is `--manifest-path`. Sprint-12 workers should note this.
- **v1-API-under-v2 alias anti-pattern (E-META-10)** — W-A1 left `pack()` writing `temporal << 52` under the v2 feature flag, corrupting the reclaim zone. Caught at main-thread code review; gated on `#[cfg(not(feature = "causal-edge-v2-layout"))]`. Sprint-12 must apply the same gate discipline to any v1-compatible accessors.
- **Two-TrustTexture coexistence (CSI-1 residual)** — `contract::mul::TrustTexture` and `causal_edge::layout::TrustTexture` both exist with incompatible variant names. Sprint-12 cross-crate code bridging MUL assessment to causal-edge layout must fully qualify. Deferred rename tracked as TD-TRUST-TEXTURE-DUPE-1.

---

## 1. One-paragraph thesis

*(UNCHANGED from v1)*

The CausalEdge64 v2 layout, the QualiaColumn quantization, the CollapseGate wire format, the witness-corpus pointer design, the MUL evaluation algebra, the Σ-tier router's Rubicon-resonance orchestration, and the thinking-engine ↔ cognitive-shader-driver SoA reunification are not seven independent design questions — they **converge into one substrate** where (a) signed i4 mantissa is the universal precision family across NARS / Qualia / ThinkingAtom / direction, (b) the i4 payload IS its own CAM key so content equals address, (c) inter-mailbox handoff is discrete baton tuples with zero analog bucket, (d) `Vsa16kF32` is narrowed to intra-tier Markov accumulation + crystal carrier + grammar bind/unbind testing, (e) cycle driver is free-energy gradient (active inference) not request/response, and (f) mailboxes are spatial-temporal meaning accumulators not channels. Autopoiesis of thinking styles and philosophic entanglement across mailboxes fall out of the shared substrate without extra mechanism. This plan is the single canonical reference for sprint-12+ implementation and the continuing locking point for architectural decisions made during sprint-10 + post-sprint-10 cross-session discussion.

---

## 2. Why now — context-dilution gate

*(UNCHANGED from v1 — original rationale still applies; this v2 is the post-sprint-11 refresh before sprint-12 context dilution)*

Sprint-10's 12-worker fleet surfaced findings the parent plan `causaledge64-mailbox-rename-soa-v1.md` did NOT foresee:

1. **Dual `CausalEdge64` types** (E-META-7, in `EPIPHANIES.md`): `causal_edge::CausalEdge64` (SPO-palette layout, `crates/causal-edge/src/edge.rs:60`) ≠ `thinking_engine::layered::CausalEdge64` (8-channel cascade, `crates/thinking-engine/src/layered.rs:45`). Same name, different bit semantics, different consumers.
2. **p64 drift origin pinpointed** at `crates/lance-graph-planner/src/cache/convergence.rs:18-22` `#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring` — the wiring was started, never finished.
3. **Three-zone hot-path mental model** corrects prior "AriGraph reads = µs cold-path joins" framing.
4. **Signed-mantissa NARS** insight — current 3-bit unsigned enum wastes the symmetry; signed i4 carries direction × rule.
5. **i4-as-CAM** insight — i4-16D has 16¹⁶ ≈ 1.8×10¹⁹ unique states, enough entropy to be both content AND CAM address.
6. **Gapless baton model** — `Vsa16kF32` between mailboxes was always over-engineered; discrete `(u16 target, CausalEdge64)` tuples suffice.
7. **Qualia i4-16D** — 9× compression from f32-18D; aligns with NARS mantissa precision family; Wisdom × Staunen → Magnitude becomes one SIMD multiply.

---

## 3. Three findings from sprint-10 that anchor this plan

*(UNCHANGED from v1)*

### 3.1 Dual `CausalEdge64` types (E-META-7)

| Type | Location | Layout | Consumers |
|---|---|---|---|
| `causal_edge::CausalEdge64` | `crates/causal-edge/src/edge.rs:60` | (S/P/O palette + NARS f/c + Pearl mask + direction + inference + plasticity + temporal) | `NarsTables`, `lance-graph-planner::cache::nars_engine`, `cognitive-shader-driver::BindSpace::EdgeColumn`, AriGraph SPO commit |
| `thinking_engine::layered::CausalEdge64` | `crates/thinking-engine/src/layered.rs:45` | 8 channels × 8 bits (BECOMES/CAUSES/SUPPORTS/REFINES/GROUNDS/ABSTRACTS/RELATES/CONTRADICTS) | `TierEngine::emit_causal_edges`, `apply_edges`, downstream tier energy perturbation |

**Reunification path (Option R-3):** transcode 8-channel → SPO-palette at thinking-engine L3 commit boundary. **Sprint-11 outcome:** D-CSV-9 (8ch↔SPO transcoder) shipped via PR #387 alongside D-CSV-8 (MUL evaluation).

### 3.2 p64 drift origin

`crates/lance-graph-planner/src/cache/convergence.rs:18-22`:

```rust
#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring
use super::nars_engine::{CausalEdge64, SpoHead, MASK_SPO};
```

### 3.3 Three-zone hot-path mental model

| Zone | Mechanism | Cost |
|---|---|---|
| **Zone-1** (cycle-speed) | thinking-engine MatVec + AriGraph `entity_index` | **200-500 ns** MatVec + **20-200 ns** HashMap O(1) |
| **Zone-2** (SPO-as-3D-vector ANN) | blasgraph + neighborhood cascade HEEL→HIP→TWIG→LEAF via `zeckf64()` | **20-1200 µs** |
| **Zone-3** (cold path) | `lance-graph-planner` DataFusion projection + columnar joins | **>1 ms** |

---

## 4. The five compressions

*(UNCHANGED from v1)*

### 4.1 Encoding — signed i4 mantissa family

| Field | Encoding | Range |
|---|---|---|
| NARS Inference mantissa | i4 signed | −8..+7 (direction × rule) |
| Qualia 16D dimensions | i4 signed × 16 | −8..+7 per dim (valence × intensity) |
| ThinkingAtom32x4 (`p64-bridge::STYLES`) | i4 signed × 32 | −8..+7 per dim |
| Direction triad | i4 (in CausalEdge64) | sign per S/P/O plane |
| `Vsa16kI8` (CLAUDE.md switchboard tier — quantized fingerprint) | i8 × 16384 | −128..+127 |

### 4.2 Wire format — discrete baton, no analog bucket

Inter-mailbox / inter-tier / inter-cycle wire format is `Vec<(u16 target, CausalEdge64)>` discrete tuples. `Vsa16kF32` does NOT cross mailbox boundaries.

### 4.3 Addressing — i4 IS content AND address (CAM)

i4-16D's 16¹⁶ ≈ 1.8×10¹⁹ unique states. The qualia/mantissa vector serves simultaneously as content AND CAM key.

### 4.4 Temporal axis — structural, not stored

Temporal field in CausalEdge64 dropped. Time is carried by cycle order (`MailboxSoA::cycle: u32`), relative order (position in `SpoWitnessChain` / WitnessCorpus chain), and wall-clock (AriGraph `Triplet.timestamp: u64`).

### 4.5 Cycle driver — entropy-driven, not request-driven

Per CLAUDE.md "The shader can't resist the thinking": active inference is the dispatch mechanism. Free-energy floor (`MUL::homeostasis`) is the rest condition. Σ10 Rubicon resonance threshold is the commit trigger.

---

## 5. Locked architectural decisions (20 items) [UPDATED 2026-05-16]

Each row carries implementation outcome annotations for decisions that have shipped. `🆕 vs v1` marks changed cells.

| # | Decision | Rationale | Lives in | Sprint-11 Outcome 🆕 vs v1 |
|---|---|---|---|---|
| **L-1** | **Keep TWO `CausalEdge64` types** at sprint-11 (transcode at L3 commit boundary, not unify) | Each variant is optimal for its tier | `crates/causal-edge/src/edge.rs:60` (SPO) + `crates/thinking-engine/src/layered.rs:45` (8-channel) | **Confirmed AS SHIPPED.** D-CSV-9 transcoder in PR #387 implements R-3 transcode. |
| **L-2** | **Drop temporal (12 bits)** from CausalEdge64 v2 | Redundant with chain-position + AriGraph anchor | `edge.rs:52-63` field reclaimed | **SHIPPED PR #383 commit `03bd175`.** `temporal()` accessor deprecated; v1 pack() feature-gated. |
| **L-3** | **Drop G-slot (5 bits)** that was being proposed | Three-way redundant | not added | **CONFIRMED — never added in v2 layout.** PR #383. |
| **L-4** | **Expand InferenceType 3→4 bits SIGNED** mantissa (−8..+7) | Direction × rule composition | `edge.rs:46-49` widened | **SHIPPED PR #383 commit `03bd175`.** `inference_mantissa()` i4-signed accessor in `layout.rs`. |
| **L-5** | **Causal mask (3 bits) IS the Pearl-rung axis** — no separate Pearl-3 modifier bit | `causal_mask = 0b111 SPO` already encodes Counterfactual | `pearl.rs:11-49` unchanged | **CONFIRMED** — unchanged. |
| **L-6** | **W-slot 6 bits** = discourse corpus root handle (64 active corpora) | Witness corpus is CAM-PQ-indexed, unbounded; W-slot is the entry pointer | NEW field in CausalEdge64 v2 | **SHIPPED PR #383 commit `03bd175`.** OQ-CSV-2 ratified 6 bits. `w_slot()` + `with_routing()` accessors. |
| **L-7** | **Truth-band lens 2 bits** (4 states incl. "13% ambiguous direction") | Carries committed-vs-ambiguous expressivity | NEW field | **SHIPPED PR #383 commit `03bd175`.** `truth_band_lens()` accessor; 4 states encoded. |
| **L-8** | **KEEP direction (3b) + plasticity (3b) in edge** | Both are load-bearing dispatch payload | unchanged | **CONFIRMED** — unchanged in v2 layout. |
| **L-9** | **PR-LL-1 `Intervention`+`Counterfactual` slot into `Reserved5`+`Reserved6`** of canonical `causal_edge::InferenceType` | PR #375 added to `nars_dispatch.rs` only; canonical edge enum needs to absorb them | `crates/causal-edge/src/edge.rs:22-25` | **SHIPPED PR #383 commit `03bd175`** (D-CSV-3). `InferenceType::to_mantissa/from_mantissa` bidirectional mapping. |
| **L-10** | **QualiaColumn → i4-16D signed** (replaces `[f32; 18]`) | 9× compression; aligns with mantissa family | `cognitive-shader-driver::bindspace.rs` QualiaColumn type | **IN PR #384 (D-CSV-2) + In PR #385 (D-CSV-5a sibling column).** OQ-CSV-1 ratified: Option α vocab. Cutover (5b) sprint-12. |
| **L-11** | **MetaColumn unchanged** — MetaWord bits, 36 ThinkingStyles | Different tier from NARS rule | unchanged | **CONFIRMED** — unchanged. |
| **L-12** | **FingerprintColumns unchanged** — `Vsa16kF32` carrier (64 KB per row) | Preserved for Markov ±5 + crystal carrier + grammar bind/unbind testing | unchanged | **CONFIRMED** — unchanged. |
| **L-13** | **CollapseGate wire format** = `Vec<(u16 target, CausalEdge64)>` + implicit provenance | No `Vsa16kF32` between mailboxes; gapless | `contract::collapse_gate::CollapseGateEmission` | **SHIPPED PR #383 commit `03bd175`** (D-CSV-4). Vec used instead of SmallVec — see §8 annotation. |
| **L-14** | **Mailbox semantics:** spatial-temporal meaning accumulators, NOT channels | Per W6 `MailboxSoA<N>` — each row is a neuron-like accumulator | `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) | **IN PR #386 (D-CSV-7).** W-slot referencing + plasticity accumulator + `apply_edges`. |
| **L-15** | **Σ-tier router orchestration:** Rubicon-resonance, NOT expected-result | Σ10 fires when ΔF < threshold AND resonance > Rubicon-bar | `pr-ce64-mb-6-sigma-tier-router.md` (W7) | **IN PR via Wave F W-F1** (D-CSV-10). sigma-tier-router crate in flight. |
| **L-16** | **Witness chain** is **sorted by emission cycle, drop-oldest truncation** | "Sort witness by time" structural-temporal pattern | `pr-ce64-mb-4-arigraph-spo-g.md` (W5) | **IN PR #386 (D-CSV-6a partial).** Full unbounded corpus (6b) sprint-12. |
| **L-17** | **`SpoWitnessChain<32>` → `WitnessCorpus` (CAM-PQ-indexed, unbounded)** | Bounded chain doesn't scale to discourse | W5 spec patched in PR #381 | **PARTIAL IN PR #386 (D-CSV-6a).** Full CAM-PQ-indexed version (D-CSV-6b) sprint-12. |
| **L-18** | **MUL evaluation in integer SIMD** (i4 × i4 → i8 products) | DK / TrustTexture / FlowState / GateDecision read i4 qualia + signed mantissa | `lance-graph-planner::mul/` | **SHIPPED PR #387 (D-CSV-8).** Scalar path. AVX-512/NEON deferred → D-CSV-13 sprint-12. |
| **L-19** | **8-channel ↔ SPO-palette transcode at L3 commit** (Option R-3) | Signed mantissa makes transcode near-bitcast | `thinking_engine::commit::transcode_to_spo()` | **SHIPPED PR #387 (D-CSV-9).** 16-mapping round-trip. |
| **L-20** | **Vertical streaming structs in ndarray** are the missing layer | `qualia.history(window: 100)`, `inference.trajectory(±5)`, `splat.evolve(steps)` | ndarray struct-method surface | **IN PR via Wave F W-F4/5/6** (D-CSV-11). Productization sprint-12. |

---

## 6. Final CausalEdge64 v2 bit layout (Option F) [UNCHANGED — AS SHIPPED in PR #383 commit `03bd175`] 🆕 vs v1

```text
[ 0:  7]  S palette index           u8       (256 subject archetypes)
[ 8: 15]  P palette index           u8       (256 predicate archetypes)
[16: 23]  O palette index           u8       (256 object archetypes)
[24: 31]  NARS frequency            u8       (f = val/255)
[32: 39]  NARS confidence           u8       (c = val/255)
[40: 42]  Causal mask               3b       (Pearl 2³ — IS the rung axis)
[43: 45]  Direction triad           3b       (sign(palette[idx].dim0) per S/P/O)
[46: 49]  Inference mantissa        4b s     (−8..+7 — direction × NARS rule)
[50: 52]  Plasticity flags          3b       (hot/cold per S/P/O plane)
[53: 58]  W slot                    6b       ← NEW: corpus root handle (64 active)
[59: 60]  Truth-band lens           2b       ← NEW: 4 lens states
[61: 63]  Spare                     3b       ← honest headroom for sprint-12+
                                    ───
                                    64b      zero unused
```

**AS SHIPPED in PR #383 commit `03bd175`** — layout is locked. The feature gate `causal-edge-v2-layout` activates the new accessors; v1 callers compile without the feature and see deprecation warnings on `temporal()` and `inference_type()` (the intended migration signal).

**Reclaim arithmetic:** drop temporal (−12 bits) → spend on Inference mantissa expansion (+1), W slot (+6), Truth-band lens (+2) = 9 spent, 3 spare. Spare remains reserved.

**Signed mantissa encoding rationale:**

| Sign | Direction | Magnitude interpretation |
|---|---|---|
| `+` (0..+7) | forward-chain / compose / commit | Deduction, Synthesis, Revision-positive, Induction |
| `−` (−8..−1) | backward-chain / decompose / refute | Abduction, Contraposition, Revision-negative, Counterfactual |

---

## 7. Column-level changes (BindSpace SoA) [UPDATED 2026-05-16] 🆕 vs v1

### 7.1 EdgeColumn (Planner axis)

**Sprint-11 outcome: SHIPPED (PR #383 commit `03bd175`).** v2 bit layout per §6 active under `causal-edge-v2-layout` feature. `layout.rs` contains all shift constants + masks + `_LAYOUT_COVERAGE` compile-time const-assert.

### 7.2 QualiaColumn (Angle axis)

**Sprint-11 outcome: IN PR (#384 D-CSV-2 + #385 D-CSV-5a sibling column).**

OQ-CSV-1 ratified to **Option α** — convergence-observable vocab, NOT the felt-qualia vocab proposed in v1 §7.2. The qualia-engineer cross-check revealed the canonical surface is observables (arousal, valence, tension, warmth, clarity, boundary, depth, velocity, entropy, coherence, intimacy, presence, assertion, receptivity, groundedness, expansion — first 16 of `AXIS_LABELS`, dropping dim 16 "integration").

| Before | After |
|---|---|
| `[[f32; 18]; N]` (72 B / row) | `[QualiaI4_16D; N]` (8 B / row, packed i4 × 16 signed) |
| Footprint at 1M rows: 72 MB | Footprint at 1M rows: **8 MB** (9× compression) |

Sibling-column (5a) in PR #385 adds `QualiaI4_16D` alongside `[f32; 18]`. Cutover (5b) is sprint-12.

**Updated per-dim table (Option α vocab):**

| Dim idx | Qualia | + means | − means |
|---|---|---|---|
| 0 | Arousal | high energy | low energy |
| 1 | Valence | positive affect | negative affect |
| 2 | Tension | high tension | relaxed |
| 3 | Warmth | warm / affiliative | cold / distancing |
| 4 | Clarity | clear / salient | ambiguous / diffuse |
| 5 | Boundary | well-bounded concept | fuzzy / overlapping |
| 6 | Depth | deep / elaborated | shallow / surface |
| 7 | Velocity | fast-changing | stable / slow |
| 8 | Entropy | high surprise | low surprise (predicted) |
| 9 | Coherence | story holds | story breaks |
| 10 | Intimacy | close / personal | distant / formal |
| 11 | Presence | foregrounded | backgrounded |
| 12 | Assertion | assertive | tentative |
| 13 | Receptivity | receptive | resistant |
| 14 | Groundedness | grounded / embodied | abstract / disembodied |
| 15 | Expansion | expanding frame | contracting frame |

(dim 16 "integration" dropped — recoverable on demand from valence + coherence + cycle-delta.)

### 7.3 MetaColumn (Thinking axis)

**Unchanged.** `MetaWord` bits packing the 36 ThinkingStyle selector + modulation weights.

### 7.4 FingerprintColumns (Topic axis)

**Unchanged.** `Vsa16kF32` carrier (16384 × f32 = 64 KB per row). Per L-12 + L-13.

---

## 8. CollapseGate wire format [UPDATED 2026-05-16] 🆕 vs v1

**AS SHIPPED in PR #383 commit `03bd175`** — `CollapseGateEmission` type in `lance-graph-contract::collapse_gate`.

**Implementation deviation from v1 spec:**

- v1 spec called for `SmallVec<[(u16, CausalEdge64); 8]>` to avoid heap allocation in the hot path.
- Shipped version uses `Vec<CollapseStep>` to preserve the contract crate's **zero-dep guarantee**.
- SmallVec optimization is deferred. Tracked as **TD-COLLAPSE-GATE-SMALLVEC-1** in TECH_DEBT.md.
- Payoff: ~20 LOC + `smallvec` dep addition, or feature-gate if zero-dep must be preserved. Sprint-12+ polish.

### 8.1 Type definition (as shipped)

In `lance-graph-contract::collapse_gate`:

```rust
/// Discrete baton emission from one CollapseGate to downstream consumers.
/// No Vsa16kF32 envelope — payload IS its own format per the gapless-baton model.
/// Vec instead of SmallVec to preserve zero-dep contract crate invariant.
/// (SmallVec optimization deferred — see TD-COLLAPSE-GATE-SMALLVEC-1)
#[repr(C)]
pub struct CollapseGateEmission {
    pub batons: Vec<(u16, CausalEdge64)>,  // discrete baton tuples
    pub source_mailbox: MailboxId,          // MailboxId = u32
    pub chain_position: u32,
    pub merge_mode: MergeMode,             // Bundle | Xor
}
```

### 8.2 Wire-cost budget

*(UNCHANGED from v1)*

- Header (source + chain_pos + mode): 13 bytes
- Per baton: 10 bytes (u16 target + u64 edge)
- 8 inline batons: 80 bytes
- Total typical emission: ~93 bytes

### 8.3 No analog bucket

*(UNCHANGED from v1)* — Three candidates for `Vsa16kF32` between mailboxes, all rebutted.

---

## 9. Mailbox semantics [UPDATED 2026-05-16] 🆕 vs v1

*(v1 §9.1–9.3 UNCHANGED in rationale)*

**Sprint-11 outcome: D-CSV-7 IN PR #386.** `MailboxSoA<N>` with W-slot referencing + per-row plasticity accumulator + `apply_edges` for baton receipt shipped in Wave F alongside D-CSV-6a. Merges pending.

**MailboxSoA shipped via PR #386 if merged.** The spatial-temporal accumulator semantics (§9.1) are fully implemented: each row integrates multi-source batons via `apply_edges`, per-row `plasticity_counter` records integration history, threshold crossing emits via the receiving `CollapseGate`.

Philosophic entanglement (§9.2) and autopoiesis (§9.3) are **unchanged from v1** — the substrate implementation that makes them concrete (W-slot + shared witness corpus root) is the same PR #386.

---

## 10. Active inference framing — Σ-tier driver [UPDATED 2026-05-16] 🆕 vs v1

*(v1 §10.1–10.3 UNCHANGED in rationale)*

**Sprint-11 Wave F Σ-tier router (W-F1):** D-CSV-10 (`SigmaTierRouter` Rubicon-resonance dispatch) is in PR via Wave F worker W-F1. The sigma-tier-router crate implements the ΔF < threshold AND resonance > Rubicon-bar commit trigger.

**OQ-CSV-6 status:** Hand-tuned Rubicon threshold shipped for sprint-11/12 per plan recommendation. Tracked as **TD-SIGMA-TIER-THRESHOLDS-1** (TECH_DEBT.md 2026-05-16). Principled Jirak-derived derivation deferred to VAMPE coupled-revival sprint-13+ (D-CSV-15 NEW entry — see §11).

---

## 11. D-CSV-* deliverable table [UPDATED 2026-05-16] 🆕 vs v1

### Phase A — Substrate primitives

| D-id | Title | Status | PR / Outcome |
|---|---|---|---|
| **D-CSV-1** | `causal-edge` crate v2 layout per §6 | **Shipped** | PR #383 commit `03bd175` |
| **D-CSV-2** | `QualiaI4_16D` type in `lance-graph-contract::qualia` + f32↔i4 migration helpers | **Shipped** | PR #384 (In PR; OQ-CSV-1 ratified Option α) |
| **D-CSV-3** | `InferenceType` signed-mantissa expansion + absorb PR-LL-1 variants | **Shipped** | PR #383 commit `03bd175` |
| **D-CSV-4** | `CollapseGateEmission` wire format spec + impl | **Shipped** | PR #383 commit `03bd175` (Vec not SmallVec — TD-COLLAPSE-GATE-SMALLVEC-1) |

### Phase B — Storage & dispatch path

| D-id | Title | Status | PR / Outcome |
|---|---|---|---|
| **D-CSV-5a** | QualiaColumn sibling-i4 column (`[f32; 18]` stays, `QualiaI4_16D` added alongside) | **In PR** | PR #385 (Wave F) |
| **D-CSV-5b** | QualiaColumn cutover (remove `[f32; 18]` after all consumers migrated) | **Queued** | sprint-12, after #385 merged + consumers updated |
| **D-CSV-6a** | `WitnessCorpus` partial (W-slot anchor + chain invariant) | **In PR** | PR #386 (Wave F, with D-CSV-7) |
| **D-CSV-6b** | `WitnessCorpus` full (CAM-PQ-indexed, unbounded, salience decay) | **Queued** | sprint-12, after #386 merged |
| **D-CSV-7** | `MailboxSoA<N>` integration: W-slot referencing + plasticity accumulator + `apply_edges` | **In PR** | PR #386 (Wave F, with D-CSV-6a) |

### Phase C — Reasoning path

| D-id | Title | Status | PR / Outcome |
|---|---|---|---|
| **D-CSV-8** | MUL evaluation integer SIMD: DK/TrustTexture/FlowState/GateDecision consume i4 qualia + signed mantissa | **Shipped** | PR #387 (scalar path; AVX-512/NEON deferred → D-CSV-13) |
| **D-CSV-9** | 8-channel ↔ SPO-palette transcoder (Option R-3) at L3 commit | **Shipped** | PR #387 (paired with D-CSV-8) |
| **D-CSV-10** | Σ-tier Rubicon-resonance dispatch in `SigmaTierRouter` | **In PR** | Wave F W-F1 (sigma-tier-router crate) |

### Phase D — Streaming infrastructure (productization sprint-12)

| D-id | Title | Status | PR / Outcome |
|---|---|---|---|
| **D-CSV-11** | Vertical streaming structs in ndarray: `QualiaStream`, `InferenceStream`, `SplatFieldStream` + `par_*` rayon variants | **In PR** | Wave F W-F4/5/6 (sprint-12 productization) |
| **D-CSV-12** | Splat shader op fleet on i4: `splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany` (scalar; on-Think methods sprint-13+) | **In PR** | Wave F W-F7 (scalar; on-Think deferred → D-CSV-14) |

### Phase E — Sprint-12 new entries [UPDATED 2026-05-16] 🆕 vs v1

| D-id | Title | Sprint | Status | Rationale |
|---|---|---|---|---|
| **D-CSV-13** | SIMD vectorization of D-CSV-8 i4 MUL evaluation (AVX-512 + NEON intrinsics) | 12 | **Queued** | Scalar path shipped in PR #387; AVX-512/NEON deferred per TD-D-CSV-8-SIMD-1. Est. ~150-300 LOC per ISA. 4-8× throughput gain. |
| **D-CSV-14** | On-Think method migration for D-CSV-12 splat ops | 13+ | **Backlog** | D-CSV-12 ships scalar standalone ops; on-Think methods (struct-method surface per L-20) deferred. Depends on D-CSV-11 vertical streaming. |
| **D-CSV-15** | Σ10 Jirak-derived threshold (TD-7 resolution) | 13+ | **Backlog** | TD-SIGMA-TIER-THRESHOLDS-1: OQ-CSV-6 hand-tuned acceptable through sprint-12; principled Jirak 2016 derivation via VAMPE+Jirak coupled-revival sprint-13+. Resolves `I-NOISE-FLOOR-JIRAK` iron-rule debt. |

---

## 12. OQ gate table [UPDATED 2026-05-16] 🆕 vs v1

| OQ # | Question | Ratification status |
|---|---|---|
| **OQ-CSV-1** | Per-dim qualia layout (15 named dims + 1 spare) | **RATIFIED** — Option α (convergence-observable vocab, first 16 of `AXIS_LABELS`). Sprint-11 qualia-engineer cross-check against `crates/thinking-engine/src/qualia.rs`. Blocks D-CSV-2 LIFTED. |
| **OQ-CSV-2** | W-slot width: 6 (64 corpora) or 8 (256 corpora) bits | **RATIFIED** — 6 bits (default per plan §11 recommendation). OQ resolved at merge time of PR #383. |
| **OQ-CSV-3** | Spare bits (3) — reserved-for-future vs pre-allocate | **DEFAULT APPLIED** — reserved. No ratification required; non-blocking. |
| **OQ-CSV-4** | QualiaI4_16D migration phasing: sibling-then-cutover vs big-bang | **RATIFIED** — sibling-then-cutover (D-CSV-5a sibling in PR #385; D-CSV-5b cutover sprint-12). Lower risk; 1 extra PR accepted. |
| **OQ-CSV-5** | Pre-computed Magnitude i8 sibling column vs on-demand | **DEFAULT APPLIED** — on-demand (`magnitude() = coherence.saturating_mul(valence)` in `QualiaI4_16D`). Non-blocking; 1 SIMD/query. |
| **OQ-CSV-6** | Σ10 Rubicon threshold Jirak-derived vs hand-tuned | **PARTIAL** — hand-tuned accepted for sprint-11/12 per `I-NOISE-FLOOR-JIRAK`. Documented in TD-SIGMA-TIER-THRESHOLDS-1. Jirak-derived resolution forwarded to D-CSV-15 sprint-13+. |

Cross-ref: W-F11 knowledge doc `i4-substrate-decisions.md` captures the full OQ ratification chain with file:line evidence for each decision.

---

## 13. Risk matrix [UPDATED 2026-05-16] 🆕 vs v1

### 13.1 i4 quantization precision (MED — UNCHANGED)

*(UNCHANGED from v1)* — per-dim calibration, bipolar interpretation, i8 fallback if needed.

### 13.2 Reunification transcoder lossiness (LOW-MED — RESOLVED in PR #387)

D-CSV-9 shipped in PR #387. The 8ch→SPO transcode is lossy (per-channel breakdown of constructive operators is not preserved; direction + net magnitude + Pearl rung is). For commit-tier purposes, this is acceptable. Ghost-edge mechanism (W5 spec) preserves cascade history if reversal needed. **Risk remains LOW-MED at the lossiness characterization; the implementation concern is resolved.**

### 13.3 Witness corpus unbounded growth (MED — UNCHANGED)

D-CSV-6a (partial) in PR #386. Full pruning policy (D-CSV-6b) sprint-12. `WitnessCorpusPruningPolicy` spec still needed.

### 13.4 Downstream consumer ABI break from QualiaColumn migration (HIGH → MED) [UPDATED 2026-05-16] 🆕 vs v1

**Risk REDUCED.** Sibling-column approach (D-CSV-5a in PR #385) eliminates the big-bang ABI break. Old `[f32; 18]` stays during sprint-12; consumers opt in via `QualiaI4_16D` sibling. Cutover (5b) only after all consumers confirmed migrated. Residual risk: cognitive-shader-driver crate workspace membership conflict (TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1) may cause friction during D-CSV-5a merge.

### 13.5 Per-tenant codebook divergence (LOW today, MED future — UNCHANGED)

*(UNCHANGED from v1)* — G-slot redundancy argument valid as long as tenants respect OGIT family-prefix convention.

### 13.6 D-CSV-11 ndarray PR coordination (HIGH — UNCHANGED)

Sprint-12 productization via Wave F W-F4/5/6 in PR. Still requires coordinated merge with `AdaWorldAPI/ndarray` upstream (PR #116 hpc-extras gap). The in-PR Wave F work makes the scope concrete; coordination is now time-sensitive.

### 13.7 Subagent isolation — workspace build conflicts [UPDATED 2026-05-16] 🆕 vs v1

**NEW observation from sprint-11.** `cognitive-shader-driver` crate hit a workspace members/exclude conflict during D-CSV-5a work. `cargo <cmd> -p cognitive-shader-driver` fails from workspace root; workaround is `--manifest-path crates/cognitive-shader-driver/Cargo.toml`. Sprint-12 workers must be briefed. Tracked as TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1.

### 13.8 v1-API-under-v2 alias anti-pattern (E-META-10) [UPDATED 2026-05-16] 🆕 vs v1

**NEW observation from sprint-11.** Worker W-A1 left `pack()` writing `temporal << 52` under the v2 feature flag, corrupting the new reclaim zone (bits 53-63). Caught at main-thread code review; corrected via feature-gate. Pattern: any accessor that existed in v1 and writes to reclaimed bits MUST be gated on `#[cfg(not(feature = "causal-edge-v2-layout"))]`. Sprint-12 workers adding any v1-compat accessor must apply this gate.

### 13.9 Two-TrustTexture coexistence (CSI-1 residual) [UPDATED 2026-05-16] 🆕 vs v1

**NEW from sprint-11 Wave F cross-crate review.** Two `TrustTexture` enums coexist with incompatible variants: `contract::mul::TrustTexture` (Calibrated/Overconfident/Underconfident/Volatile/Frozen) and `causal_edge::layout::TrustTexture` (Crystalline/Solid/Porous/Fractured/Molten). Cross-crate code must fully qualify. TYPE_DUPLICATION_MAP (W-F8) records both. Deferred rename tracked as TD-TRUST-TEXTURE-DUPE-1.

### 13.10 Disk quota pressure from parallel cargo builds (MED) [UPDATED 2026-05-16] 🆕 vs v1

**NEW from sprint-11 PR #386 incident.** Workspace hit ENOSPC mid-rebase during heavy parallel cargo builds; freed 21 GB via `cargo clean`. Sprint-12 fleets should run `cargo clean` at sprint-start hygiene. ISSUE filed in ISSUES.md (ISSUE-X3 disk quota).

---

## 14. Test plan extension [UPDATED 2026-05-16] 🆕 vs v1

### Sprint-11 tests landed

Approximate counts from worker reports (Wave A/B/C/D/E workers):

| Wave / D-id | Tests added |
|---|---|
| D-CSV-1 (Wave A W-A1) | 16 unit tests (v2 layout round-trip, field isolation, signed mantissa) |
| D-CSV-3 (Wave A W-A1) | Mantissa-roundtrip + to/from_mantissa mapping (covered in D-CSV-1 tests) |
| D-CSV-4 (Wave A W-A2) | 8 unit tests (CollapseGateEmission: new/push/cost + Bundle/Xor semantics) |
| D-CSV-2 (Wave B W-B1) | 14 unit tests (QualiaI4_16D: size, zero, roundtrip, clamp, isolation, migration) |
| D-CSV-8 + D-CSV-9 (PR #387) | ~20 combined (MUL scalar path parity + 16-mapping transcoder round-trip) |
| Spec-patch (PR #381) | 0 (governance only) |
| **Subtotal sprint-11** | **~58 new tests** |

Miri growth: extends sprint-10-test-plan.md `760→1550` target. Sprint-11 delta: +58 unit tests toward the 1550 gate. Sprint-12 projected: +50+ tests from D-CSV-13 (SIMD paths) + D-CSV-11 productization.

### Sprint-12 test targets

| Phase / D-id | Target | Approach |
|---|---|---|
| D-CSV-13 (SIMD vectorization) | 12+ tests | `is_x86_feature_detected!` gate; AVX-512 scalar parity + NEON parity; throughput bench |
| D-CSV-5b (QualiaColumn cutover) | 15+ tests | Cross-consumer compile-check matrix; clippy `--tests --no-deps -D warnings` as gate |
| D-CSV-6b (full WitnessCorpus) | 10+ benches | CAM-PQ retrieval correctness + Markov ±500 window + salience decay + corpus root anchor |
| D-CSV-11 productization | 18+ tests | QualiaStream / InferenceStream / SplatFieldStream + par_* rayon work-stealing |
| D-CSV-12 (splat ops) | 14+ tests | splat_gaussian + score_hole_closure + replay_coherence + emit_if_epiphany + 4 benches |
| **Sprint-12 projected total** | **~70+ new tests** | — |

**Aggregate test target post-sprint-12:** ~1550 (Miri gate from sprint-10 plan) + ~128 sprint-11/12 new tests = ~1678 heading toward the 1900 target with D-CSV-11 vertical streaming Miri scope added.

---

## 15. Sprint phasing [UPDATED 2026-05-16] 🆕 vs v1

### Phase A (sprint-11): COMPLETE

D-CSV-1 / D-CSV-3 / D-CSV-4 — Shipped via PR #383 commit `03bd175`.
D-CSV-2 — Shipped via PR #384 (In PR; OQ-CSV-1 ratified).

### Phase B (sprint-11): COMPLETE modulo merges

D-CSV-5a — In PR #385 (Wave F).
D-CSV-6a + D-CSV-7 — In PR #386 (Wave F).
D-CSV-5b + D-CSV-6b — Queued sprint-12 (dependent on #385 + #386 merges).

### Phase C (sprint-12): D-CSV-8/9 SHIPPED; D-CSV-10 in PR; D-CSV-13 SIMD vec queued

D-CSV-8 + D-CSV-9 — Shipped via PR #387.
D-CSV-10 — In PR via Wave F W-F1 (sigma-tier-router crate).
D-CSV-13 — Queued sprint-12 (SIMD vectorization of D-CSV-8 scalar path).

### Phase D (sprint-13+): D-CSV-11/12 productized; D-CSV-14 on-Think; D-CSV-15 Jirak threshold

D-CSV-11 — In PR via Wave F W-F4/5/6 (productization sprint-12 primary).
D-CSV-12 — In PR via Wave F W-F7 (scalar; on-Think methods D-CSV-14 sprint-13+).
D-CSV-14 — Backlog sprint-13+ (on-Think method migration for splat ops).
D-CSV-15 — Backlog sprint-13+ (Jirak-derived Σ10 threshold, VAMPE coupled-revival).

---

## 16. Test target growth [UPDATED 2026-05-16] 🆕 vs v1

| Sprint | Tests added | Source | Running total |
|---|---|---|---|
| sprint-10 (spec sprint) | 0 (spec only) | PR #372 + PR #381 | ~760 (Miri baseline) |
| sprint-11 | ~58 | Wave A/B/C workers: D-CSV-1..4 (44 tests) + D-CSV-8/9 (~20 tests); wave F D-CSV-5a/6a/7 count TBD | ~818+ |
| sprint-12 projected | ~70+ | D-CSV-13 + D-CSV-5b + D-CSV-6b + D-CSV-11 productization + D-CSV-12 | ~888+ |
| sprint-13+ target | ~100+ | D-CSV-11 full Miri scope + D-CSV-14 on-Think + D-CSV-15 VAMPE | ~988+ toward 1900 |

Note: v1 §15 projected ~80 tests for sprint-11 (Wave A/B/C/D/E workers). Actual sprint-11 is ~58 confirmed from shipped waves (Wave F D-CSV-5a/6a/7 test counts not yet confirmed — expect +20-30 additional from PR #385 and #386 when merged). Sprint-11 total likely ~78-88, consistent with the v1 estimate.

---

## 17. Cross-references [UPDATED 2026-05-16] 🆕 vs v1

### 17.1 v1 plan (predecessor)

- `.claude/plans/cognitive-substrate-convergence-v1.md` — this v2 supersedes for sprint-12+ planning; v1 remains archival for decision archaeology

### 17.2 sprint-11 meta-review

- `.claude/board/sprint-log-11/meta-review.md` — sprint-11 Opus meta (W-F10 deliverable); cross-ref for sprint-11 grade and E-META observations

### 17.3 i4-substrate-decisions knowledge doc

- `.claude/knowledge/i4-substrate-decisions.md` — W-F11 knowledge doc; captures OQ-CSV-1..6 ratification chain with file:line evidence per decision; READ BY any worker touching qualia quantization or CausalEdge64 v2 layout

### 17.4 Sprint-10 work this consolidates

*(UNCHANGED from v1 §17.1 — all references valid)*

- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — parent plan
- `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — par-tile substrate
- `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2) — bit layout (resolved by §6; patched in PR #381)
- `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` (W3) — PAL8 regression tests (patched PR #381 with codex P1 fix)
- `.claude/specs/pr-ce64-mb-3-bindspace-efgh.md` (W4) — BindSpace columns (patched PR #381)
- `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (W5) — AriGraph SPO-G + WitnessCorpus retrofit (patched PR #381)
- `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — MailboxSoA (patched PR #381)
- `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` (W7) — SigmaTierRouter + Rubicon-resonance (patched PR #381)
- `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` (W9) — bevy proof
- `.claude/specs/pr-ndarray-miri-complete.md` (W8) — Miri coverage
- `.claude/specs/sprint-10-pr-dep-graph.md` (W10) — dep graph (patched PR #381)
- `.claude/specs/sprint-10-test-plan.md` (W11) — unified test plan (patched PR #381)
- `.claude/specs/sprint-10-execution-plan.md` (W12) — sprint-11 fleet definition
- `.claude/board/sprint-log-10/meta-review.md` — Opus meta-review with CSI-1..6 + E-META-1..5

### 17.5 Sprint-10 knowledge corpus

*(UNCHANGED from v1 §17.2)*

- `.claude/knowledge/causal-edge-64-spo-variant.md`
- `.claude/knowledge/causal-edge-64-thinking-engine-variant.md`
- `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md`
- `.claude/knowledge/spo-schema-and-mailbox-sidecar.md`
- `.claude/knowledge/spo-ontology-format-stack.md`
- `.claude/knowledge/ogit-owl-dolce-ontology-compartments.md`
- `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md`
- `.claude/knowledge/splat-shader-rayon-struct-method-vision.md`

### 17.6 Recent PRs this builds on

*(v1 §17.3 EXTENDED)*

- **PR #381** (2026-05-16) — 8-spec patch bundle (cognitive-substrate-convergence spec patches, ~1,200 LOC, commit `a7c0545`)
- **PR #383** (2026-05-??) — sprint-11 Wave A implementation (D-CSV-1/3/4, commit `03bd175`)
- **PR #384** (In PR) — sprint-11 Wave B D-CSV-2 `QualiaI4_16D`
- **PR #385** (In PR, Wave F) — D-CSV-5a sibling QualiaColumn
- **PR #386** (In PR, Wave F) — D-CSV-6a + D-CSV-7 `WitnessCorpus` partial + `MailboxSoA`
- **PR #387** (Shipped) — D-CSV-8 + D-CSV-9 MUL scalar path + 8ch↔SPO transcoder
- **PR #383** (2026-05-14) → also the base for Wave A
- **PR #372** (2026-05-14) — sprint-10 spec corpus (12-worker CCA2A fleet + Opus meta)
- **PR #373** (2026-05-14) — neurosymbolic-rlvr-causal-curriculum-v1.md
- **PR #375** (2026-05-??) — PR-LL-1: NARS Intervention/Counterfactual in `nars_dispatch.rs`
- **PR #379** (2026-05-??) — 4-branch retirement

### 17.7 Doctrinal anchors (CLAUDE.md sections)

*(UNCHANGED from v1 §17.4)*

- **"The Click" (P-1)** — Markov ±5, role-key bind/unbind, free-energy minimization
- **"AGI-as-glove"** — 4 BindSpace columns = AGI surface
- **`I-SUBSTRATE-MARKOV`** iron rule — VSA-bundling guarantees Chapman-Kolmogorov
- **`I-NOISE-FLOOR-JIRAK`** iron rule — Jirak 2016 for σ-thresholds (D-CSV-15 resolution)
- **`I-VSA-IDENTITIES`** iron rule — VSA on identity fingerprints
- **"The shader can't resist the thinking"** — active inference dispatch driver

---

## 18. Status [UPDATED 2026-05-16] 🆕 vs v1

| Field | Value |
|---|---|
| **Status** | ACTIVE — Phase A COMPLETE; Phase B in PR (Wave F); Phase C partially Shipped (#387) + in PR (W-F1); Phase D in PR (Wave F W-F4/5/6/7) |
| **Confidence (2026-05-16)** | HIGH on shipped architecture (D-CSV-1/2/3/4/8/9); HIGH on gapless-baton model (CollapseGateEmission shipped); HIGH on i4-mantissa NARS (OQ-CSV-2 ratified 6-bit W-slot); HIGH on QualiaColumn i4-16D (OQ-CSV-1 ratified Option α); MED on Rubicon-resonance threshold (OQ-CSV-6 hand-tuned → D-CSV-15 Jirak derivation sprint-13+) |
| **Branch** | `claude/sprint-12-wave-f-fleet` (this file) |
| **Predecessor** | `.claude/plans/cognitive-substrate-convergence-v1.md` |
| **Successor** | None (this is v2; v3 to be authored post-sprint-12 if scope warrants) |

---

*End of cognitive-substrate-convergence-v2.md. Authored 2026-05-16 by W-F12 as sprint-12 Wave F forward-plan document. Captures sprint-11 outcomes (Waves A-E + Wave F partial) and locks sprint-12 scope before context dilution. Single canonical reference for sprint-12+ implementation planning.*
