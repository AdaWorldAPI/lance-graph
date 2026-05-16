# i4 Substrate Decisions — Cross-Session Reference (sprint-11 Implementation Outcomes)

> **READ BY:** Any agent touching CausalEdge64, QualiaI4_16D, MUL evaluation,
> splat ops, or the i4 substrate. Tier-1 mandatory for sprint-12+ implementation.
> **Status:** FINDING (sprint-11 implementations have shipped on main; this
> doc captures the decisions + their implementation outcomes for posterity).

---

## 1. The i4 Substrate Doctrine

**One sentence:** sign = direction, |magnitude| = NARS rule slot.

The signed i4 mantissa (−8..+7) encodes direction × rule in a single 4-bit field:

- **Sign `+` (0..+7):** forward-chain / compose / commit — Deduction, Synthesis, Revision-positive, Induction (forward generalization).
- **Sign `−` (−8..−1):** backward-chain / decompose / refute — Abduction, Contraposition, Revision-negative, Counterfactual.

`abs(mantissa)` selects the base NARS rule (8 base slots); `signum(mantissa)` selects the chain direction. 16 distinct directed-inferences, naturally composable with integer arithmetic. This precision family unifies what was scattered across f32 qualia, u8 NARS f/c, u3 inference, and u3 direction into a single algebra: `i4 × i4 → i8`, `i8 × i8 → i16`.

**Companion doctrines (from plan §5 and §4):**

- **L-3 (no G-slot in edge):** G-slot is three-way redundant: tenant via SoA partition, belief via witness corpus root, ontology via palette family-prefix. The edge carries no domain discriminator.
- **L-4 (signed mantissa):** inference mantissa widened 3b unsigned → 4b signed i4. Carries direction × rule composition.
- **L-9 (PR-LL-1 absorption):** `Intervention` and `Counterfactual` from `nars_dispatch.rs` PR-LL-1 are absorbed into the canonical mantissa table at slots +6 / −6. They are NOT separate enum variants in the bit field — they route through `to_mantissa()` / `from_mantissa()`.

---

## 2. Locked Decisions L-1 through L-20

Each row gives the decision, which PR shipped it, and the canonical code site (or a deviation note where implementation diverged from plan).

### Phase A — Substrate primitives

| # | Decision | PR / code site | Notes |
|---|---|---|---|
| **L-1** | Keep TWO `CausalEdge64` types at sprint-11 (transcode at L3 commit, not unify) | PR #383 — `crates/causal-edge/src/edge.rs:60` (SPO) + `crates/thinking-engine/src/layered.rs:45` (8-channel) | Both types co-exist; L-1 held |
| **L-2** | Drop temporal (12 bits) from CausalEdge64 v2 | PR #383 — `layout.rs`: `V1_TEMPORAL_SHIFT` deprecated, bits 52-63 reclaimed | `set_temporal()` is a no-op under v2 feature |
| **L-3** | No G-slot — redundant via tenant SoA + witness corpus + palette family-prefix | PR #383 — no G-shift constant in `layout.rs` | G-slot was proposed; plan §5 L-3 dropped it before implementation |
| **L-4** | Inference mantissa 3→4 bits SIGNED i4 (−8..+7) | PR #383 — `layout.rs::INFER_SHIFT=46, BITS4_MASK=0xF`; `edge.rs::with_inference_mantissa(i8)` + `inference_mantissa() -> i8` | Plasticity shifted from bits 49-51 to 50-52 as a consequence |
| **L-5** | Causal mask (3 bits) IS the Pearl-rung axis — no separate Pearl-3 modifier | PR #383 — `pearl.rs::CausalMask` unchanged | `causal_mask = 0b111 (SPO)` IS the Pearl-3 counterfactual flag |
| **L-6** | W-slot 6 bits = discourse corpus root handle (64 active corpora) | PR #383 — `layout.rs::W_SHIFT=53, BITS6_MASK=0x3F`; OQ-CSV-2 ratified to 6 bits (plan default) | W=0 means no corpus anchor |
| **L-7** | Truth-band lens 2 bits (4 states: Crystalline / Solid / Fuzzy / Murky) | PR #383 — `layout.rs::TRUTH_SHIFT=59, TrustTexture` enum; `with_truth()` + `truth()` accessors | Local `TrustTexture` in causal-edge; canonical contract type is `lance_graph_contract::mul::TrustTexture` |
| **L-8** | Keep direction (3b) + plasticity (3b) in edge | PR #383 — `DIR_SHIFT=43` unchanged; `PLAST_SHIFT=50` (shifted +1 from v1 due to mantissa expansion) | Both are load-bearing dispatch payload per the hot-path analysis |
| **L-9** | Intervention + Counterfactual absorb into Reserved5/Reserved6 of canonical `InferenceType` | PR #383 — `edge.rs::InferenceType::Intervention=5, Counterfactual=6`; `to_mantissa()/from_mantissa()` routing | Slots +6/−6 in the i4 mantissa table |

### Phase B — Storage & dispatch

| # | Decision | PR / code site | Notes |
|---|---|---|---|
| **L-10** | QualiaColumn → i4-16D signed (replaces `[f32; 18]`) | PR #384 — `crates/lance-graph-contract/src/qualia.rs::QualiaI4_16D(pub u64)` | OQ-CSV-1 ratified: canonical convergence-observable vocab (arousal/valence/…/expansion, 16 dims), NOT the plan §7.2 felt-qualia CONJECTURE |
| **L-11** | MetaColumn unchanged — MetaWord bits, 36 ThinkingStyles | No sprint-11 PR; `contract::thinking.rs` unchanged | Different tier from NARS rule; styles dispatch the cycle mode |
| **L-12** | FingerprintColumns unchanged — `Vsa16kF32` carrier | No sprint-11 PR; `Vsa16kF32` preserved for intra-tier Markov + crystal carrier + grammar bind/unbind | Does NOT cross mailbox boundaries |
| **L-13** | CollapseGate wire format = `Vec<(u16 target, CausalEdge64)>` + implicit provenance | PR #383 — `contract::collapse_gate::CollapseGateEmission` (Vec instead of SmallVec; see TD-7) | `SmallVec` deferral preserves contract zero-dep; sprint-12+ optimization |
| **L-14** | Mailbox semantics: spatial-temporal accumulators, NOT channels | Spec shipped in PR #381 (W6 spec); implementation pending D-CSV-7 | `MailboxSoA<N>` each row = neuron-like accumulator with plasticity counter |
| **L-15** | Σ-tier router: Rubicon-resonance, NOT expected-result | PR #387 — `crates/sigma-tier-router/src/lib.rs` (D-CSV-10); hand-tuned bands per TECH_DEBT TD-7 | "Never commit on F-rising" invariant; Jirak-derived threshold sprint-13+ |
| **L-16** | Witness chain: sorted by emission cycle, drop-oldest truncation | Spec shipped PR #381 (W5 spec `W5-INV-CHAIN-ORDER`); implementation pending D-CSV-6 | Timestamp_ns ASC + hash tie-break per iron rule |
| **L-17** | `SpoWitnessChain<32>` → `WitnessCorpus` (CAM-PQ-indexed, unbounded) | Spec shipped PR #381 (W5 spec); implementation pending D-CSV-6 | `Arc<Vec<WitnessEntry>>` with copy-on-write via `Arc::make_mut` |
| **L-18** | MUL evaluation in integer SIMD (i4 × i4 → i8 products) | PR #387 — `crates/lance-graph-planner/src/mul/` i4 evaluation module (D-CSV-8, scalar path; SIMD vectorization TD-7) | Scalar path correct; AVX-512/NEON intrinsic deferred |
| **L-19** | 8-channel ↔ SPO-palette transcode at L3 commit (Option R-3) | PR #387 — `crates/thinking-engine/src/layered.rs::CausalEdge64::to_spo()` + `from_spo()` (D-CSV-9) | Deviation: shipped in thinking-engine as method, not in a separate `thinking_engine::commit` module as originally proposed |
| **L-20** | Vertical streaming structs in ndarray (future) | Not shipped — D-CSV-11 sprint-13+ | Blocks on ndarray PR #116 (hpc-extras upstream gap) |

---

## 3. The Four Columns + i4 Ratifications (AGI-as-Glove Doctrine)

The four `BindSpace` SoA columns remain the AGI surface. Sprint-11 migrated two of the four; two are unchanged.

### EdgeColumn (Planner axis)

**Type:** `CausalEdge64` v2 — `crates/causal-edge/src/edge.rs:60`
**PR:** #383 (Wave A — D-CSV-1 + D-CSV-3 + D-CSV-4)
**Layout:** signed mantissa 4b (bits 46-49), W-slot 6b (53-58), truth-band lens 2b (59-60), spare 3b (61-63); temporal dropped; plasticity shifted to 50-52.

```text
[ 0:  7]  S palette index   u8   (256 subject archetypes)
[ 8: 15]  P palette index   u8   (256 predicate archetypes)
[16: 23]  O palette index   u8   (256 object archetypes)
[24: 31]  NARS frequency    u8   (f = val/255)
[32: 39]  NARS confidence   u8   (c = val/255)
[40: 42]  Causal mask       3b   (Pearl 2³ rung axis)
[43: 45]  Direction triad   3b   (sign per S/P/O plane)
[46: 49]  Inference mantissa 4b s (−8..+7: direction × rule)
[50: 52]  Plasticity flags  3b   (hot/cold per S/P/O)
[53: 58]  W slot            6b   NEW: corpus root handle (0..63)
[59: 60]  Truth-band lens   2b   NEW: Crystalline/Solid/Fuzzy/Murky
[61: 63]  Spare             3b   NEW: sprint-12+ headroom
```

Compile-time const-assert in `layout.rs::_LAYOUT_COVERAGE` verifies all 64 bits covered exactly once (`8+8+8+8+8+3+3+4+3+6+2+3 = 64`).

### QualiaColumn (Angle axis)

**Legacy type:** `QualiaVector = [f32; 17]` (`QUALIA_DIMS = 17`) — `crates/lance-graph-contract/src/qualia.rs` — UNCHANGED in sprint-11.
**New sibling type:** `QualiaI4_16D(pub u64)` — **PR #384** (Wave B — D-CSV-2); `#[repr(C, align(8))]`, 8 bytes / 16 dims / i4 signed per dim.
**PR #385:** QualiaColumn migration Phase 5a (sibling double-write) — In PR / sprint-12 completion.
**D-CSV-5b cutover:** sprint-12; removes the legacy `[f32; 17]` column after all consumers migrated.

OQ-CSV-1 ratified: canonical 16 dims = first 16 of `AXIS_LABELS` (arousal/valence/tension/warmth/clarity/boundary/depth/velocity/entropy/coherence/intimacy/presence/assertion/receptivity/groundedness/expansion). Plan §7.2's felt-qualia CONJECTURE (Wisdom/Trust/Hope/etc.) was **not adopted** — cross-check against `crates/thinking-engine/src/qualia.rs` confirmed the canonical surface is convergence observables.

OQ-CSV-4 ratified: sibling-column-then-cutover (Phase 5a = add sibling; Phase 5b = remove legacy). Lower risk than big-bang.

### MetaColumn (Thinking axis)

**Unchanged.** `MetaWord` bits, 36 ThinkingStyle selector + modulation weights (`contract::thinking.rs`). Per L-11: thinking styles dispatch the cycle's mode; this column carries that selection per SoA row. No i4 migration.

### FingerprintColumns (Topic axis)

**Unchanged.** `Vsa16kF32` carrier (16384 × f32 = 64 KB per row). Per L-12 + L-13: `Vsa16kF32` preserved for intra-cycle Markov bundling + crystal carrier + grammar bind/unbind testing. Does NOT cross mailbox boundaries; the inter-mailbox wire IS discrete batons (`CollapseGateEmission`).

---

## 4. OQ Ratifications (plan §11 gate table)

| OQ | Outcome | Wave / evidence |
|---|---|---|
| **OQ-CSV-1** Qualia 16D per-dim assignment | **Ratified: Option α** — canonical convergence-observable vocab (arousal..expansion, 16 dims). Plan §7.2 felt-qualia CONJECTURE not adopted. | Wave B (D-CSV-2); qualia-engineer cross-check vs `thinking-engine/src/qualia.rs` |
| **OQ-CSV-2** W-slot width 6 vs 8 bits | **Ratified: 6 bits (64 corpora)** — plan §11 default. Promote to 8 in v3 if multi-tenant SaaS demands. | Wave A (D-CSV-1) |
| **OQ-CSV-3** Spare bits allocation | **N/A** — not surfaced as a gate in plan §11; bits 61-63 are "reserved for sprint-12+ probe-derived needs" per plan §6. | Not a user gate |
| **OQ-CSV-4** QualiaColumn migration phasing | **Ratified: sibling-column-then-cutover (5a/5b)** — Phase 5a adds `QualiaI4_16D` as sibling column; Phase 5b cuts over. | Wave C (D-CSV-5a in PR #385); D-CSV-5b sprint-12 |
| **OQ-CSV-5** Pre-computed Magnitude column | **N/A** — ratified as on-demand (1 SIMD multiply per row sweep: `coherence × valence → i8`). Not a blocking gate. | Non-blocking per plan §11 |
| **OQ-CSV-6** Σ10 Rubicon threshold derivation | **Hand-tuned for sprint-11/12 with TECH_DEBT (TD-7)** — bands default to Σk = k × 0.10. Jirak-derived calibration (VAMPE + Jirak coupled revival) deferred to sprint-13+. | PR #387 (sigma-tier-router); TECH_DEBT.md entry |

### Sprint-13 ratifications (OQ-CSV-7..16) — user-ratified 2026-05-16

User batch-ratified all 10 sprint-13 spawn blockers per PP-11 Opus recommendations
in the `.claude/board/sprint-log-13/oq-catalog.md` per protocol §3. Source:
autoattended ratification at the post-PR-#392-merge gate-closing moment.

| OQ | Outcome | Wave / evidence |
|---|---|---|
| **OQ-CSV-7** PP-3 rayon feature gate name | **Ratified: `parallel`** — match ndarray's existing feature naming. Avoids feature-fork in the dependency graph; every existing parallel consumer already opts into `parallel`. | Sprint-13 D-CSV-17 (W-I4) |
| **OQ-CSV-8** PP-3 par_* iteration chunk size | **Ratified: Fixed** — **8 rows for QualiaI4** (8×8B = 64B cache line), **4 rows for SplatField** (4×16B = 64B cache line). Cache-line aligned by construction; deterministic in benches. Auto-detect deferred to sprint-14+ if tuning need surfaces. | Sprint-13 D-CSV-17 (W-I4) |
| **OQ-CSV-9** PP-4 splat_field carrier (Think vs sibling) | **Ratified: Add `splat_field: Vec<SplatField>` to `Think`** — keeps the canonical carrier per CLAUDE.md "Thinking is a struct" doctrine. Sibling `Splat` struct would re-create the free-function-on-state anti-pattern under a new name. Cost: +24B per Think (one Vec header) when splat is unused. | Sprint-13 D-CSV-14 (W-I2) |
| **OQ-CSV-10** PP-4 splat generation source | **Ratified: Derive from `self.cycle`** — single source of truth; no new invariant; cycle already advances on every Think step. Sub-cycle granularity deferred until use case forces it. | Sprint-13 D-CSV-14 (W-I2) |
| **OQ-CSV-11** PP-5 SPO → 256D adapter (VSA bind vs one-hot) | **Ratified: Option A (VSA bind of role-keyed fingerprints)** — `bind(S_key, s) XOR bind(P_key, p) XOR bind(O_key, o)`. Per I-VSA-IDENTITIES iron rule: role information must survive superposition; one-hot fails this litmus. Role keys live as constants in `lance-graph-contract::vsa::roles`. Cost: 3 XOR ops per SPO insert. | Sprint-13 D-CSV-16 (W-I3) |
| **OQ-CSV-12** PP-5 WitnessCorpus cam_pq lazy vs eager | **Ratified: Lazy via `enable_cam_pq()`** — keeps ndarray/cam_pq an optional dep behind a feature gate; HashMap-only users see zero cost. Builders that want similarity query call `corpus.enable_cam_pq()` once after construction. cam_pq state lives in `Option<CamPqState>` field; query methods early-return `None` if disabled. | Sprint-13 D-CSV-16 (W-I3) |
| **OQ-CSV-13** PP-6 SIMD i4 runtime vs compile-time dispatch | **Ratified: Runtime dispatch via `simd_caps()`** — matches ndarray's existing pattern (cf. `ndarray::simd::caps`); single binary that adapts to host; bench-friendly. Cost: one cached `LazyLock<SimdCaps>` and a branch per call site. Branch predictor handles cost trivially after warmup. | Sprint-13 D-CSV-13b (W-I1) |
| **OQ-CSV-14** PP-6 bench speedup floor (SHIP vs LAND) | **Ratified: SHIP gate 4× AVX-512 vs scalar; LAND gate 2×** — below 2× blocks PR; 2×–4× lands with `TD-D-CSV-13b-PERF-FLOOR-1` TECH_DEBT note and follow-up; ≥4× ships. 4× = published expectation (8-wide i4 lanes after vpcompress, ~2× horizontal-sum overhead); 2× = floor at which AVX-512 is worth the binary cost. | Sprint-13 D-CSV-13b (W-I1) |
| **OQ-CSV-15** PP-8 worker-template-v2 workspace member edit ownership | **Ratified: WORKER edits `members =`** — W-G6 proved feasible (PR landed clean, no CSI-7 recurrence). Avoids orphan pattern at root (worker tests crate against workspace; if `members =` is wrong, worker's own `cargo test` catches it). Worker template v2 documents the required edit as a checklist item. | Sprint-13 all new-crate workers (PP-3 splat-field-types, PP-5 witness-index-cam-pq) |
| **OQ-CSV-16** Governance: E-META-10 + iron-rules-doctrine in BOOT.md Tier-1 | **Ratified: YES, add to BOOT.md Tier-1 trigger table** — promotion was already shipped via PR #391 (4 new agent trigger rows include iron-rules-doctrine as mandatory knowledge for PP-13/14/15/16). This ratification confirms the choice. Cost: +1 read per worker session that triggers; benefit: zero iron-rule violations make it past worker self-review. | PR #391 (already shipped); this ratification confirms the design |

**Effect:** sprint-13 worker fleet dispatch (Wave I) is unblocked. W-I1 (D-CSV-13b SIMD) / W-I2 (D-CSV-14 splat-on-Think) / W-I3 (D-CSV-16 WitnessIndexCamPq) / W-I4 (D-CSV-17 rayon par_*) + W-Meta-Opus may now spawn.

---

## 5. Sprint-11 Codex P1 Anti-Pattern: v1-API-Under-v2-Feature Aliasing

The **single recurring failure mode** across all Wave A-E workers. Definition: a v1 API path writes or reads bits in the v1 layout position, but under the `causal-edge-v2-layout` feature those bits are reclaimed for new v2 fields — silently corrupting routing state or producing wrong semantics.

**Documented instances (all caught in PR review before merge):**

1. **`pack(..., temporal=X)` writing to reclaim zone** — Wave A W-A1: `pack()` was still writing `temporal << 52` under v2, corrupting bits 52 (plasticity[2]), 53-58 (W-slot), 59-60 (truth-band lens), 61-63 (spare). **Fix (PR #383):** feature-gate the temporal write; under v2 the `temporal` arg is silently dropped. Tests `test_roundtrip` and `test_temporal_in_msb_gives_sort_order` gated `#[cfg(not(feature = "causal-edge-v2-layout"))]`.

2. **`inference_type()` reading 3 unsigned bits under v2** — reading bits 46-48 as a 3-bit unsigned index when the field is now 4-bit signed i4. `0b1111` under v1 decodes as `Reserved7`; under v2 it is `inference_mantissa() = -1` (Abduction direction). **Fix (PR #383):** `forward()` and any dispatch path routes through `InferenceType::from_mantissa(self.inference_mantissa())` under v2 feature; the deprecated `inference_type()` accessor is NOT called on v2 edges.

3. **`set_temporal()` writing reclaim-zone bits** — `learn()` calls `set_temporal()` which under v1 writes bits 52-63. Under v2 those bits are occupied. **Fix (PR #383):** `set_temporal()` is a complete no-op under v2 feature; `learn()` inherits the no-op transitively. Test `test_set_temporal_no_op_under_v2` in `v2_layout_tests.rs` is the regression guard.

4. **`pack(InferenceType::Counterfactual)` writing raw discriminant 6 into mantissa slot** — the v1 `pack()` stored the enum discriminant directly into bits 46-48 (3 bits). Under v2, `Counterfactual` has discriminant 6 (binary `0b110`) — bits 46-48 = `0b110` + bit 49 = 0, giving `inference_mantissa() = +6` (Intervention), not `−6` (Counterfactual). **Fix (PR #383):** under v2 feature, `pack()` calls `inference.to_mantissa()` and writes the signed i4 result, not the raw discriminant. Test `test_pack_uses_mantissa_mapping_under_v2` in `v2_layout_tests.rs`.

5. **W3 spec Test 1 using `temporal = 1023`** — caught in PR #381 codex review: constructing a v1 edge with `temporal = 1023` sets bits 52-61 in the test fixture, which under v2 aliases to W-slot + truth-band + spare, making the migration test fail on ordinary data instead of testing the zero-default contract. **Fix (PR #381, commit `33509ab`):** Test 1 rewritten to use `temporal = 0`; added Test 1b `pal8_v1_nonzero_temporal_is_blocked_by_version_gate` proving the PAL8 version gate is mandatory.

**The rule:** every v1 API path under v2 feature MUST transparently route through the canonical mapping (`to_mantissa()` / `from_mantissa()`) OR be feature-gated to a documented no-op with a migration pointer. "Silent semantic shift" (wrong discriminant written, wrong bits read) is the failure mode — it compiles, tests with v1 patterns pass, but v2 edges are silently misinterpreted.

---

## 6. The 12-Mapping Transcoder Table (D-CSV-9)

Shipped in **PR #387** — `crates/thinking-engine/src/layered.rs::CausalEdge64::to_spo()`.

The transcoder maps the dominant 8-channel signal to an SPO-palette edge at the L3 commit boundary. Lossy (8 channels collapse to 1 dominant + sign); lossless round-trip is preserved only for the dominant channel.

| 8ch Channel (index) | NARS mantissa slot | Causal mask | Pearl rung | Notes |
|---|---|---|---|---|
| **BECOMES** (0) | +1 / −1 | SPO | 3 (Counterfactual) | Forward = Deduction; backward BECOMES → CONTRADICTS slot in lossy collapse |
| **CAUSES** (1) | +6 / −6 | SPO | 3 (Counterfactual) | +6 = Intervention (do-calculus); −6 = Counterfactual (Pearl-3) |
| **SUPPORTS** (2) | +4 / −4 | PO | 2 (Intervention) | Revision positive / negative |
| **REFINES** (3) | +5 / −5 | PO | 2 (Intervention) | Synthesis / Decomposition |
| **GROUNDS** (4) | +1 / −1 | S | 1 (Association) | S-grounded Deduction; shares mantissa 1 with BECOMES |
| **ABSTRACTS** (5) | +2 / −2 | P | 1 (Association) | Induction / Contraposition |
| **RELATES** (6) | 0 | None | 0 (prior) | Identity/neutral; no causal plane active |
| **CONTRADICTS** (7) | −1 / +1 | SPO | 3 (Counterfactual) | Destructive; sign reversed from BECOMES |

**Lossy-collapse classes:**
- **BECOMES + GROUNDS → mantissa ±1:** both share `|mantissa| = 1`. Positive BECOMES round-trips to BECOMES (dominant=0). Negative BECOMES (backward chain) collapses to CONTRADICTS (mantissa=−1, dominant=7) — semantically correct (a backward-chain transformative signal IS a contradiction in the SPO lattice).
- **REFINES + (ABSTRACTS tilt) → mantissa ±5 / ±2 overlap:** REFINES goes to Synthesis (5); at the tilt, the round-trip `from_spo()` may surface ABSTRACTS instead of REFINES since both are in nearby slots.
- **CAUSES is the only clean round-trip for Pearl-3:** mantissa ±6 maps uniquely back to CAUSES, no shared slot.

**Code sites:**
- `to_spo(s, p, o)` — `layered.rs:161-195` (forward transcode, dominant-channel dispatch)
- `from_spo(spo)` — `layered.rs:197-222` (inverse / round-trip debugging)
- `dominant_channel()` — `layered.rs:138-150` (ties break to lowest index)
- `active_channel_count()` — `layered.rs:153-159` (confidence proxy in transcode)

**Transcoder tests:** `layered.rs:688+` (`transcoder_tests` mod) — 16 tests covering forward, negative, neutral, and round-trip lossy-collapse classes.

---

## 7. Cross-References

- **Canonical plan:** `.claude/plans/cognitive-substrate-convergence-v1.md` — §5 (L-1..L-20), §6 (Option F bit layout), §11 (D-CSV-* deliverables), §12 (spec patch matrix), §14 (OQ-CSV-1..6 gate table)
- **Sprint-10 knowledge trinity:**
  - `.claude/knowledge/causal-edge-64-spo-variant.md` — SPO-palette variant detail
  - `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` — 8-channel cascade variant detail
  - `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md` — reunification Options R-1/R-2/R-3 + drift origin
- **Sprint-log meta-reviews:**
  - `.claude/board/sprint-log-10/meta-review.md` — CSI-1..6, E-META-1..5, sprint-11 gate decision
  - `.claude/board/sprint-log-11/meta-review.md` — sprint-11 Wave A-E outcomes (W-F10 in parallel)
- **Board files:**
  - `.claude/board/STATUS_BOARD.md` — D-CSV-* row status (Shipped / In PR / Queued)
  - `.claude/board/PR_ARC_INVENTORY.md` — per-PR Added/Locked/Deferred history
  - `.claude/board/AGENT_LOG.md` — per-worker Layer-2 blackboard entries
  - `.claude/board/TECH_DEBT.md` — TD-7 entries (sigma-tier-router hand-tuned thresholds; i4 MUL scalar path; CollapseGateEmission SmallVec deferral)
  - `.claude/board/ISSUES.md` — ENOSPC incident (PR #386 rebase); protoc env gap
- **TYPE_DUPLICATION_MAP.md** — `docs/TYPE_DUPLICATION_MAP.md` — lists `CausalEdge64 (2 copies)` + `TrustTexture (2 copies: local in causal-edge, canonical in contract)`
- **EPIPHANIES.md** — E-META-7 (dual CausalEdge64 discovery), E-META-8 (board-hygiene rule violation by PR #381)
- **Iron rules (CLAUDE.md):** `I-SUBSTRATE-MARKOV` (Chapman-Kolmogorov; Bundle not XOR for transitions), `I-NOISE-FLOOR-JIRAK` (weak-dependence Berry-Esseen for σ-thresholds), `I-VSA-IDENTITIES` (CAM-PQ and VSA are separate tools)

---

*Authored 2026-05-16. Worker W-F11 (sprint-12 Wave F). Captures sprint-11 L-1..L-20 locked decisions + actual implementation outcomes for cross-session reference.*
