# Cognitive Substrate Convergence — i4 Mantissa, Gapless Baton, Active Inference (v3)

> **Status:** ACTIVE — sprint-12 Wave F + Wave G complete (grade B+); sprint-13 entry scope locked
>
> **Version:** v3 (2026-05-16) — successor to v2 (2026-05-16, W-F12)
> **Predecessors:**
> - `.claude/plans/cognitive-substrate-convergence-v2.md` (v2 — W-F12 sprint-12 forward plan)
> - `.claude/plans/cognitive-substrate-convergence-v1.md` (v1 — original 2026-05-15 cross-session A2A discussion)
>
> **READ BY:** integration-lead, truth-architect, host-glove-designer, palette-engineer, family-codec-smith, certification-officer, bus-compiler, nars-engineer, anyone implementing sprint-13's D-CSV-13b / D-CSV-14 / D-CSV-15b / D-CSV-16 / D-CSV-17, or touching the SIMD intrinsic gate, the `WitnessIndexCamPq` real wiring, the `par_*_stream` rayon variants, or the on-Think method migration for splat ops.
>
> **CONSOLIDATES:**
> - v2 plan (UNCHANGED unless `[UPDATED v3]` annotation present)
> - sprint-12 Wave F outcomes: D-CSV-5a / D-CSV-6a / D-CSV-7 / D-CSV-10 / D-CSV-11 / D-CSV-12 (PRs #385/#386/#388/#389) — see §0.1
> - sprint-12 Wave G outcomes: W-G2 (`WitnessIndexCamPq` HashMap placeholder), W-G3 (i4 SIMD scalar batch API), CSI-18 (iron-rule doctrine consolidation) — see §0.2
> - sprint-13 forward plan: NEW deliverables D-CSV-13b (SIMD intrinsics), D-CSV-14 (on-Think methods), D-CSV-15b (VAMPE-Jirak), D-CSV-16 (cam_pq real wiring), D-CSV-17 (rayon `par_*` variants)
>
> **DOES NOT REPLACE:** per-PR specs in `.claude/specs/pr-ce64-mb-*.md`. v3 is the **architectural anchor** for sprint-13 spawn — single canonical reference before context dilution.

---

## 0. Status delta — sprint-12 outcomes and sprint-13 forward plan 🆕 v3 (sprint-13)

### §0.1 What shipped in sprint-12 Wave F (and the post-merge codex P2 follow-up)

**Phase B (storage and dispatch) — COMPLETE (post-Wave-F):**

| D-id | PR | Commit | Outcome |
|---|---|---|---|
| D-CSV-5a | #385 | `6f58418` (merge) | QualiaColumn sibling-i4 column shipped. Double-write semantics; legacy `[f32; 17]` retained for cutover safety. [UPDATED v3: was "In PR" in v2 §0.1 — CSI-11 corrected this drift.] |
| D-CSV-6a | #386 | `33110c8` (merge) | `WitnessCorpus` partial (W-slot anchor, chain order invariant W5-INV-CHAIN-ORDER, `Arc<Vec<WitnessEntry>>` copy-on-write). Indexing is a HashMap placeholder pending D-CSV-16. |
| D-CSV-7 | #386 | `33110c8` (merge) | `MailboxSoA<N>` with W-slot referencing, per-row plasticity accumulator, `apply_edges` baton receipt, `last_emission_cycle: u32` with `u32::MAX` sentinel for never-fired. |

**Phase C (reasoning path) — COMPLETE:**

| D-id | PR | Commit | Outcome |
|---|---|---|---|
| D-CSV-10 | #388 | `77f2d26` (merge) | `sigma-tier-router` crate shipped. `SigmaTierRouter` Rubicon-resonance dispatch (ΔF < threshold AND resonance > Rubicon-bar). CSI-7 fixed mid-merge (added to parent workspace `members`; standalone `[workspace]` line removed). |

**Phase D (streaming infrastructure) — PARTIAL:**

| D-id | PR | Commit | Outcome |
|---|---|---|---|
| D-CSV-11 (W-F4) | #388 | `77f2d26` (merge) | `InferenceStream` shipped in ndarray `hpc/stream/inference.rs` + registered in `mod.rs` (W-F5 over-delivered: also registered `qualia` + `splat_field` mod entries). Sibling `QualiaStream` + `SplatFieldStream` files shipped (W-F4/W-F6). **CSI-7/CSI-8** (lance-graph-side lib.rs + workspace orphan) **resolved via lance-graph `d4e5bbc` aggregation commit.** **CSI-9** (ndarray-side `pub mod` registration) **resolved via ndarray PR #147 merge** — ndarray master HEAD = `e956e9d9` (verified 2026-05-16 post-PR-#391-merge). `par_*` rayon variants NOT YET SHIPPED — carries to D-CSV-17 (sprint-13). |
| D-CSV-12 (W-F7) | #388 | `77f2d26` (merge) | `splat_ops` module in thinking-engine: `splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany` as **free functions** on i4 substrate. On-Think carrier methods deferred — carries to D-CSV-14 (sprint-13). |

**Wave F aggregation + codex P2 follow-up — COMPLETE:**

- `d4e5bbc` — W-Meta-Opus honest review committed; CSI-7 / CSI-8 lance-graph-side integration fixes landed in single aggregation commit. **CSI-9 (cross-repo ndarray-side `pub mod` registration) was a separate ndarray-repo fix shipped via PR #147 merge (ndarray master `e956e9d9`), NOT via this lance-graph commit.**
- PR #389 (`b526485` merge) — codex P2: `AttentionMaskBackend` trait impl for `AttentionMaskSoA` + canonical `MailboxId` import (CSI-10 closed). 14/14 attention-mask tests pass.

### §0.2 What shipped in sprint-12 Wave G (post-Wave-F continuation) 🆕 v3

Wave G ran as a smaller follow-on fleet after the W-Meta-Opus aggregation cleared the Wave F integration debt. Three deliverables:

- **W-G1 (governance):** sprint-log-12/meta-review-opus.md sprint-12 honest grade = **B+** (revised up from sprint-11's B because Wave F discipline tightened: aggregation pass scheduled, lib.rs registration debt closed inside the wave instead of after meta-review). Carries forward to v3 §13 risk matrix as the Wave-F-style "main thread aggregates" gap is now closed.
- **W-G2 (`WitnessIndexCamPq` HashMap placeholder):** to unblock D-CSV-6a sibling consumers without holding for the full `ndarray::hpc::cam_pq` wiring, W-G2 shipped a `WitnessIndexCamPq` newtype wrapping `HashMap<QualiaI4_16D, Vec<WitnessHandle>>`. The CAM-PQ contract surface (`add`, `query_nearest`, `evict_oldest`) is correct; the internal storage is HashMap not real CAM-PQ. Carries to **D-CSV-16 (sprint-13)** for real `ndarray::hpc::cam_pq` substitution.
- **W-G3 (i4 SIMD scalar batch API):** to make D-CSV-13 SIMD-portable across ISAs and feature-detect cleanly, W-G3 shipped the **scalar batch API** that AVX-512 / NEON intrinsics will target. The API is `simd_i4_mul_batch(a: &[i8], b: &[i8], out: &mut [i8])` with a portable scalar fallback; the SIMD-feature-gated body is **NOT YET WRITTEN**. Carries to **D-CSV-13b (sprint-13)** for AVX-512 + NEON intrinsic bodies. This split (scalar API in sprint-12, intrinsic bodies in sprint-13) lets downstream consumers compile and test against the API surface without waiting for portability work.
- **CSI-18 (iron-rule doctrine consolidation):** across Wave F + Wave G, three iron-rule-shaped findings recur frequently enough that they now warrant promotion in `CLAUDE.md` next to `I-SUBSTRATE-MARKOV` / `I-NOISE-FLOOR-JIRAK` / `I-VSA-IDENTITIES`. The proposed three are: (a) **I-AGGREGATION-DISCIPLINE** — every fleet wave MUST schedule an explicit aggregation commit before meta-review spawn; (b) **I-FEATURE-GATE-FIELD-ISOLATION** — every v2-style layout change requires field-isolation-matrix tests at the bit boundary; (c) **I-PLAN-GIT-RECONCILE** — plan-author worker MUST run `git log origin/main -20` and reconcile "In PR" cells before commit. PP-2 drafts the doctrine consolidation knowledge doc; integration to CLAUDE.md is a separate PR.

### §0.3 What carries over to sprint-13

- **D-CSV-13b** — SIMD intrinsics (AVX-512 + NEON) targeting the W-G3 scalar batch API. PP-6 drafts the spec. ~150-300 LOC per ISA.
- **D-CSV-14** — On-Think method migration for D-CSV-12 splat ops (sprint-12 free fns → sprint-13 carrier methods on `Think` per L-20). PP-4 drafts the spec.
- **D-CSV-15b** — VAMPE-coupled Jirak threshold refinement (sprint-12 D-CSV-15 hand-derived hand-tuned → sprint-13 VAMPE-validated). Deferred to sprint-13+.
- **D-CSV-16** — `WitnessIndexCamPq` real `ndarray::hpc::cam_pq` wiring; replaces the W-G2 HashMap placeholder. PP-5 drafts.
- **D-CSV-17** — `par_qualia_stream` / `par_inference_stream` / `par_splat_field_stream` rayon variants in ndarray (sprint-12 W-F4/5/6 shipped forward-iter scaffold; rayon `par_*` variants are sprint-13). PP-3 drafts.

### §0.4 Sprint-13 governance carryover

- Worker template v2 (CSI-13 root cause + CSI-18 doctrine consolidation): PP-8 drafts. Bakes in the I-AGGREGATION-DISCIPLINE iron rule via the prompt template (every worker prompt includes "your lib.rs/mod.rs hunk OR an explicit aggregation deliverable").
- Iron rule doctrine consolidation (CSI-18): PP-2 drafts the knowledge doc; CLAUDE.md integration is the follow-on PR.
- Sprint-13 OQ catalog: PP-11 enumerates (OQ-CSV-7..N — see §12).

---

## 1. One-paragraph thesis

*(UNCHANGED from v1/v2)*

The CausalEdge64 v2 layout, the QualiaColumn quantization, the CollapseGate wire format, the witness-corpus pointer design, the MUL evaluation algebra, the Σ-tier router's Rubicon-resonance orchestration, and the thinking-engine ↔ cognitive-shader-driver SoA reunification are not seven independent design questions — they **converge into one substrate** where (a) signed i4 mantissa is the universal precision family across NARS / Qualia / ThinkingAtom / direction, (b) the i4 payload IS its own CAM key so content equals address, (c) inter-mailbox handoff is discrete baton tuples with zero analog bucket, (d) `Vsa16kF32` is narrowed to intra-tier Markov accumulation + crystal carrier + grammar bind/unbind testing, (e) cycle driver is free-energy gradient (active inference) not request/response, and (f) mailboxes are spatial-temporal meaning accumulators not channels. Autopoiesis of thinking styles and philosophic entanglement across mailboxes fall out of the shared substrate without extra mechanism. v3 is the canonical reference for sprint-13 implementation and the continuing locking point for architectural decisions across sprint-10 → sprint-12.

---

## 2. Why now — context-dilution gate

*(UNCHANGED from v1/v2 — original rationale still applies; v3 is the post-sprint-12 refresh before sprint-13 context dilution)*

Sprint-12 outcomes that confirm the original rationale:

- All 20 locked decisions L-1..L-20 either shipped or have a concrete sprint-13 entry (see §5 outcome annotations).
- The dual-CausalEdge64 reunification (Option R-3, D-CSV-9) shipped via PR #387 and the transcoder round-trip is verified in 16 tests.
- The five compressions (§4) hold under sprint-12 implementation outcomes; no compression was rolled back.

What sprint-12 surfaced that sprint-13 must address:

1. **W-G2 HashMap placeholder is a known-shape mismatch** with the real `ndarray::hpc::cam_pq` surface — sprint-13 D-CSV-16 closes this. HIGH risk per §13.
2. **Free-function splat ops are an L-20 violation** — D-CSV-12 shipped free fns to unblock sprint-12, but L-20 ("vertical streaming structs in ndarray, methods on Think carrier") requires sprint-13 D-CSV-14 method migration.
3. **`par_*` rayon variants** are a load-bearing dependency for the splat shader rayon vision (`.claude/knowledge/splat-shader-rayon-struct-method-vision.md`). D-CSV-17 is the sprint-13 productization.
4. **SIMD portability is unfinished business** — W-G3 scalar API is the floor, not the ceiling. D-CSV-13b sprint-13 intrinsics deliver the 4-8× throughput gain the v2 plan promised.

---

## 3. Three findings from sprint-10 that anchor this plan

*(UNCHANGED from v1/v2)*

### 3.1 Dual `CausalEdge64` types (E-META-7) — RESOLVED via transcode

D-CSV-9 transcoder shipped PR #387. Both types co-exist; L3 commit transcode (Option R-3) is the canonical bridge. See `i4-substrate-decisions.md` §6 for the 12-mapping transcoder table.

### 3.2 p64 drift origin — RESOLVED via convergence wiring

`crates/lance-graph-planner/src/cache/convergence.rs:18-22` `#[allow(unused_imports)]` was the original sprint-10 finding. The convergence wiring is now downstream of D-CSV-1 + D-CSV-7 + D-CSV-10; sprint-13 D-CSV-16 closes the final cam_pq surface.

### 3.3 Three-zone hot-path mental model — UNCHANGED

| Zone | Mechanism | Cost |
|---|---|---|
| **Zone-1** (cycle-speed) | thinking-engine MatVec + AriGraph `entity_index` | **200-500 ns** MatVec + **20-200 ns** HashMap O(1) |
| **Zone-2** (SPO-as-3D-vector ANN) | blasgraph + neighborhood cascade HEEL→HIP→TWIG→LEAF via `zeckf64()` | **20-1200 µs** |
| **Zone-3** (cold path) | `lance-graph-planner` DataFusion projection + columnar joins | **>1 ms** |

---

## 4. The five compressions

*(UNCHANGED from v1/v2)*

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

i4-16D's 16¹⁶ ≈ 1.8×10¹⁹ unique states. The qualia/mantissa vector serves simultaneously as content AND CAM key. Sprint-13 D-CSV-16 realizes this as `WitnessIndexCamPq` against real `ndarray::hpc::cam_pq`.

### 4.4 Temporal axis — structural, not stored

Temporal field in CausalEdge64 dropped. Time is carried by cycle order (`MailboxSoA::cycle: u32`), relative order (position in `SpoWitnessChain` / WitnessCorpus chain), and wall-clock (AriGraph `Triplet.timestamp: u64`).

### 4.5 Cycle driver — entropy-driven, not request-driven

Per CLAUDE.md "The shader can't resist the thinking": active inference is the dispatch mechanism. Free-energy floor (`MUL::homeostasis`) is the rest condition. Σ10 Rubicon resonance threshold is the commit trigger (sprint-12 hand-tuned; D-CSV-15b sprint-13 VAMPE-Jirak-derived).

---

## 5. Locked architectural decisions (20 items) [UPDATED v3]

Each row carries sprint-12 implementation outcome annotations. `[UPDATED v3]` marks cells where sprint-12 changed status. Decisions L-1..L-20 themselves are **immutable since v1**; only outcomes update.

| # | Decision | Sprint-12 outcome [UPDATED v3] | Sprint-13 follow-on |
|---|---|---|---|
| **L-1** | Keep TWO `CausalEdge64` types (transcode at L3 commit, not unify) | **Confirmed AS SHIPPED.** D-CSV-9 transcoder PR #387. Both types coexist. | None — transcode is the canonical bridge. |
| **L-2** | Drop temporal (12 bits) | **SHIPPED PR #383 `03bd175`.** `temporal()` deprecated; v1 pack() feature-gated. | None. |
| **L-3** | Drop G-slot (5 bits) | **CONFIRMED — never added in v2.** | None. |
| **L-4** | Expand InferenceType 3→4 bits SIGNED mantissa | **SHIPPED PR #383.** i4-signed accessor in `layout.rs`. | None. |
| **L-5** | Causal mask (3b) IS Pearl-rung axis | **CONFIRMED** — unchanged. | None. |
| **L-6** | W-slot 6 bits | **SHIPPED PR #383.** OQ-CSV-2 ratified 6 bits. | None. |
| **L-7** | Truth-band lens 2 bits | **SHIPPED PR #383.** `truth_band_lens()` accessor. | None. |
| **L-8** | Keep direction (3b) + plasticity (3b) | **CONFIRMED** — unchanged in v2 layout. | None. |
| **L-9** | PR-LL-1 Intervention+Counterfactual into Reserved5+Reserved6 | **SHIPPED PR #383.** `to_mantissa/from_mantissa` bidirectional. | None. |
| **L-10** | QualiaColumn → i4-16D signed | **SHIPPED PR #384 + #385 (D-CSV-5a sibling shipped via merge `6f58418`).** [UPDATED v3 — was "In PR" in v2; sprint-12 closed it.] Cutover (5b) is sprint-13. | **D-CSV-5b** sprint-13 cutover after consumer audit. |
| **L-11** | MetaColumn unchanged | **CONFIRMED** — unchanged. | None. |
| **L-12** | FingerprintColumns unchanged (`Vsa16kF32`) | **CONFIRMED** — unchanged. | None. |
| **L-13** | CollapseGate wire format = Vec<(u16, CausalEdge64)> | **SHIPPED PR #383** (D-CSV-4). Vec not SmallVec — TD-COLLAPSE-GATE-SMALLVEC-1. | SmallVec opt sprint-14+. |
| **L-14** | Mailbox semantics: spatial-temporal accumulators | **SHIPPED PR #386** (D-CSV-7). [UPDATED v3 — was "In PR" in v2.] | None. |
| **L-15** | Σ-tier router: Rubicon-resonance not expected-result | **SHIPPED PR #388** (D-CSV-10). [UPDATED v3 — was "In PR" in v2.] CSI-7 fixed mid-merge. | **D-CSV-15b** sprint-13 VAMPE-Jirak refinement. |
| **L-16** | Witness chain: sorted by emission cycle, drop-oldest | **SHIPPED PR #386** (D-CSV-6a partial). [UPDATED v3.] | **D-CSV-6b** + **D-CSV-16** sprint-13 full CAM-PQ-indexed. |
| **L-17** | SpoWitnessChain<32> → WitnessCorpus | **PARTIAL SHIPPED PR #386 (W-G2 HashMap index placeholder).** [UPDATED v3.] | **D-CSV-16** sprint-13 real ndarray::hpc::cam_pq wiring. |
| **L-18** | MUL evaluation in integer SIMD | **SHIPPED PR #387 (scalar) + W-G3 (scalar batch API).** [UPDATED v3.] AVX-512/NEON deferred. | **D-CSV-13b** sprint-13 SIMD intrinsics. |
| **L-19** | 8-channel ↔ SPO-palette transcode at L3 commit (Option R-3) | **SHIPPED PR #387** (D-CSV-9). 16-mapping round-trip. | None. |
| **L-20** | Vertical streaming structs in ndarray (struct-method surface) | **PARTIAL SHIPPED PR #388** (D-CSV-11 forward-iter scaffold + D-CSV-12 free fns). [UPDATED v3.] | **D-CSV-14** sprint-13 on-Think method migration; **D-CSV-17** sprint-13 par_* rayon variants. |

---

## 6. Final CausalEdge64 v2 bit layout (Option F) [UNCHANGED — AS SHIPPED in PR #383]

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
[53: 58]  W slot                    6b       corpus root handle (64 active)
[59: 60]  Truth-band lens           2b       4 lens states
[61: 63]  Spare                     3b       honest headroom for sprint-13+
                                    ───
                                    64b      zero unused
```

**AS SHIPPED in PR #383 commit `03bd175`** — layout is locked. v3 does not propose any layout change; the spare 3 bits remain reserved.

**Sprint-13 layout consideration (NOT a change):** D-CSV-16 cam_pq integration may surface a need for an in-edge CAM-PQ shard hint (1-2 bits in the spare). This will be evaluated as part of D-CSV-16 design and, if needed, become **OQ-CSV-7** (see §12). The default is "no layout change."

---

## 7. Column-level changes (BindSpace SoA) [UPDATED v3]

### 7.1 EdgeColumn (Planner axis)

**Sprint-11 outcome: SHIPPED (PR #383).** Sprint-12: unchanged. Sprint-13: unchanged baseline; OQ-CSV-7 cam_pq shard hint may pre-allocate 1-2 of the 3 spare bits if D-CSV-16 surfaces the need.

### 7.2 QualiaColumn (Angle axis)

**Sprint-12 outcome: SHIPPED (PR #384 + #385).** [UPDATED v3 — sprint-12 closed D-CSV-5a; cutover D-CSV-5b is sprint-13.] OQ-CSV-1 ratified Option α convergence-observable vocab.

| Before | After |
|---|---|
| `[[f32; 18]; N]` (72 B / row) | `[QualiaI4_16D; N]` (8 B / row, packed i4 × 16 signed) |
| Footprint at 1M rows: 72 MB | Footprint at 1M rows: **8 MB** (9× compression) |

Sprint-13 D-CSV-5b cutover removes the legacy `[f32; 17]` after the consumer audit confirms all readers have migrated.

### 7.3 MetaColumn (Thinking axis)

**Unchanged through sprint-12.** Sprint-13: unchanged.

### 7.4 FingerprintColumns (Topic axis)

**Unchanged through sprint-12.** Sprint-13: unchanged.

---

## 8. CollapseGate wire format [UPDATED v3]

**AS SHIPPED in PR #383** — `CollapseGateEmission` in `lance-graph-contract::collapse_gate`. Sprint-12: no change. Sprint-13: no change; TD-COLLAPSE-GATE-SMALLVEC-1 polish remains sprint-14+.

### 8.1 Type definition (as shipped)

```rust
#[repr(C)]
pub struct CollapseGateEmission {
    pub batons: Vec<(u16, CausalEdge64)>,
    pub source_mailbox: MailboxId,
    pub chain_position: u32,
    pub merge_mode: MergeMode,
}
```

### 8.2 Wire-cost budget (UNCHANGED)

- Header: 13 bytes
- Per baton: 10 bytes (u16 target + u64 edge)
- 8 inline batons: 80 bytes
- Total typical emission: ~93 bytes

### 8.3 No analog bucket (UNCHANGED)

Three candidates for `Vsa16kF32` between mailboxes, all rebutted in v1.

---

## 9. Mailbox semantics [UPDATED v3]

*(v1/v2 §9.1–9.3 UNCHANGED in rationale)*

**Sprint-12 outcome: D-CSV-7 SHIPPED PR #386.** [UPDATED v3 — was "In PR" in v2.] `MailboxSoA<N>` with W-slot referencing + per-row plasticity accumulator + `apply_edges` + `last_emission_cycle: u32` (with `u32::MAX` sentinel for never-fired). Spatial-temporal accumulator semantics fully realized.

**Sprint-13:** no new mailbox-level work scheduled; D-CSV-17 `par_*` rayon variants enable parallel sweep over `MailboxSoA<N>` rows but do not change the per-row semantics.

---

## 10. Active inference framing — Σ-tier driver [UPDATED v3]

*(v1/v2 §10.1–10.3 UNCHANGED in rationale)*

**Sprint-12 outcome: D-CSV-10 SHIPPED PR #388.** [UPDATED v3 — was "In PR" in v2.] `sigma-tier-router` crate (`SigmaTierRouter` Rubicon-resonance dispatch). CSI-7 fixed mid-merge (workspace membership). Hand-tuned thresholds (Σk = k × 0.10) ship as TD-SIGMA-TIER-THRESHOLDS-1.

**Sprint-13 D-CSV-15b:** VAMPE-coupled Jirak threshold refinement. The hand-tuned bands are acceptable through sprint-13 entry but become a Jirak-derivation deliverable once VAMPE coupled-revival activates. Resolves `I-NOISE-FLOOR-JIRAK` iron-rule debt.

---

## 11. D-CSV-* deliverable table [UPDATED v3]

### Phase A — Substrate primitives (sprint-11, complete)

| D-id | Title | Status | Commit |
|---|---|---|---|
| **D-CSV-1** | `causal-edge` crate v2 layout per §6 | **Shipped** | PR #383 `03bd175` |
| **D-CSV-2** | `QualiaI4_16D` type + f32↔i4 migration helpers | **Shipped** | PR #384 `0751a8b` (merge) |
| **D-CSV-3** | `InferenceType` signed-mantissa expansion + PR-LL-1 absorb | **Shipped** | PR #383 `03bd175` |
| **D-CSV-4** | `CollapseGateEmission` wire format | **Shipped** | PR #383 `03bd175` (Vec; TD-COLLAPSE-GATE-SMALLVEC-1) |

### Phase B — Storage & dispatch (sprint-12, complete)

| D-id | Title | Status | Commit |
|---|---|---|---|
| **D-CSV-5a** | QualiaColumn sibling-i4 column | **Shipped** | PR #385 `6f58418` (merge) [UPDATED v3] |
| **D-CSV-5b** | QualiaColumn cutover (drop legacy `[f32; 17]`) | **Queued (sprint-13)** | After consumer audit |
| **D-CSV-6a** | `WitnessCorpus` partial (W-slot anchor + chain invariant) | **Shipped** | PR #386 `33110c8` (merge) [UPDATED v3] |
| **D-CSV-6b** | `WitnessCorpus` full (CAM-PQ-indexed, unbounded, salience decay) | **Queued (sprint-13)** | Depends on D-CSV-16 |
| **D-CSV-7** | `MailboxSoA<N>`: W-slot referencing + plasticity + `apply_edges` | **Shipped** | PR #386 `33110c8` (merge) [UPDATED v3] |

### Phase C — Reasoning path (sprint-12, complete)

| D-id | Title | Status | Commit |
|---|---|---|---|
| **D-CSV-8** | MUL i4 SIMD evaluation (scalar path) | **Shipped** | PR #387 `da8e8f7` |
| **D-CSV-9** | 8-channel ↔ SPO-palette transcoder (Option R-3) | **Shipped** | PR #387 `fdafd1a` |
| **D-CSV-10** | Σ-tier `SigmaTierRouter` Rubicon-resonance dispatch | **Shipped** | PR #388 `77f2d26` (merge) [UPDATED v3] |
| **D-CSV-13** | SIMD scalar batch API (W-G3) — portable API surface for sprint-13 intrinsics | **Shipped (Wave G)** | sprint-12 Wave G [UPDATED v3] |
| **D-CSV-13b** 🆕 v3 | SIMD intrinsics targeting D-CSV-13 scalar API: AVX-512 i4×i4→i8 + NEON i4×i4→i8 | **Queued (sprint-13)** | Spec: PP-6. Est. ~150-300 LOC per ISA. |

### Phase D — Streaming infrastructure (sprint-12 partial; sprint-13 productization)

| D-id | Title | Status | Commit |
|---|---|---|---|
| **D-CSV-11** | Vertical streaming structs in ndarray: `QualiaStream` / `InferenceStream` / `SplatFieldStream` (forward-iter scaffold) | **Shipped (forward-iter)** | PR #388 `77f2d26` (merge); par_* deferred [UPDATED v3] |
| **D-CSV-12** | Splat shader op fleet on i4 (scalar **free functions**) | **Shipped (free fns)** | PR #388 `77f2d26` (merge); on-Think methods deferred [UPDATED v3] |
| **D-CSV-14** 🆕 v3 | On-Think method migration: `Think::splat_gaussian()`, `Think::score_hole_closure()`, `Think::replay_coherence()`, `Think::emit_if_epiphany()` | **Queued (sprint-13)** | Spec: PP-4. Depends on D-CSV-11 + D-CSV-17. Realizes L-20 method-on-carrier semantics. |
| **D-CSV-16** 🆕 v3 | `WitnessIndexCamPq` real `ndarray::hpc::cam_pq` wiring (replaces W-G2 HashMap placeholder) | **Queued (sprint-13)** | Spec: PP-5. Depends on `ndarray::hpc::cam_pq` API shape resolution (see OQ-CSV-9 §12). HIGH risk per §13. |
| **D-CSV-17** 🆕 v3 | `par_qualia_stream` / `par_inference_stream` / `par_splat_field_stream` rayon variants in ndarray | **Queued (sprint-13)** | Spec: PP-3. Builds on D-CSV-11 forward-iter scaffold. ~80 LOC + rayon work-stealing harness. |

### Phase E — Sprint-13+ refinements

| D-id | Title | Sprint | Status | Rationale |
|---|---|---|---|---|
| **D-CSV-15** | Σ10 Jirak-derived threshold (hand-derived, TD-7 resolution candidate) | 13+ | **Hand-tuned** | Sprint-12 ships hand-tuned per TD-SIGMA-TIER-THRESHOLDS-1; D-CSV-15b is the principled refinement. |
| **D-CSV-15b** 🆕 v3 | VAMPE-coupled Jirak threshold refinement (D-CSV-15 hand-derived → VAMPE-validated) | 13+ | **Backlog** | Depends on VAMPE coupled-revival activation. Resolves `I-NOISE-FLOOR-JIRAK` iron-rule debt fully. |

---

## 12. OQ gate table [UPDATED v3]

### Sprint-11/12 OQs (closed)

| OQ # | Question | Ratification status |
|---|---|---|
| **OQ-CSV-1** | Per-dim qualia layout (15 named dims + 1 spare) | **RATIFIED** — Option α (convergence-observable vocab). |
| **OQ-CSV-2** | W-slot width: 6 vs 8 bits | **RATIFIED** — 6 bits. |
| **OQ-CSV-3** | Spare bits (3) — reserved-for-future vs pre-allocate | **DEFAULT APPLIED** — reserved. |
| **OQ-CSV-4** | QualiaI4_16D migration phasing: sibling-then-cutover vs big-bang | **RATIFIED** — sibling-then-cutover. |
| **OQ-CSV-5** | Pre-computed Magnitude i8 sibling column vs on-demand | **DEFAULT APPLIED** — on-demand. |
| **OQ-CSV-6** | Σ10 Rubicon threshold Jirak-derived vs hand-tuned | **PARTIAL** — hand-tuned sprint-11/12 accepted; full Jirak deferred to D-CSV-15b sprint-13+. |

### Sprint-13 OQs (NEW) 🆕 v3

| OQ # | Question | Blocks | Recommendation |
|---|---|---|---|
| **OQ-CSV-7** | Should the 3 spare bits in CausalEdge64 pre-allocate a 1-2 bit cam_pq shard hint for D-CSV-16? | D-CSV-16 layout question | **Default: NO.** Keep spare reserved; D-CSV-16 resolves cam_pq shard at the `WitnessIndexCamPq` layer, not in the edge bit field. Promote to YES only if benchmark shows >10% query throughput from edge-embedded shard hint. PP-11 enumerates. |
| **OQ-CSV-8** | SIMD intrinsic feature gate strategy: per-ISA cargo features (`avx512`, `neon`) vs `is_x86_feature_detected!` + `is_aarch64_feature_detected!` runtime detection | D-CSV-13b | **Default: runtime detection** for the entry path, with `#[target_feature]` annotated function bodies. Per-ISA cargo features remain available for build-time pinning when downstream needs deterministic codegen. Aligns with ndarray's existing `simd_caps()` singleton pattern (CLAUDE.md). |
| **OQ-CSV-9** | `ndarray::hpc::cam_pq` API shape for `WitnessIndexCamPq`: which crate-side API is canonical? | D-CSV-16 | **Default: consult `family-codec-smith` agent and `encoding-ecosystem.md` knowledge doc before drafting D-CSV-16 spec.** PP-5 must enumerate the candidate API surfaces (the gap between W-G2 HashMap interface and the real cam_pq surface is exactly what OQ-CSV-9 resolves). HIGH risk per §13.4. |
| **OQ-CSV-10** | Rayon thread-pool sizing for `par_*_stream`: workspace-global pool vs per-call configurable | D-CSV-17 | **Default: workspace-global with `RAYON_NUM_THREADS` override.** Matches existing rayon usage in ndarray. Per-call configurability is a sprint-14+ polish item if benchmarks show contention. |
| **OQ-CSV-11** | On-Think method placement: methods on `Think` struct directly vs methods via `impl SplatOps for Think` trait | D-CSV-14 | **Default: methods directly on `Think`.** Trait indirection adds dispatch cost and breaks the "thinking is a struct" doctrine (CLAUDE.md "The Click"). The struct's fields ARE the cognitive state; methods on the carrier read its own state. Trait split is sprint-15+ if multiple Think-shaped carriers emerge. |
| **OQ-CSV-12** | CSI-18 iron-rule promotion: which of (I-AGGREGATION-DISCIPLINE, I-FEATURE-GATE-FIELD-ISOLATION, I-PLAN-GIT-RECONCILE) reach CLAUDE.md as iron rules? | Worker template v2 (PP-8) | **Default: all three.** PP-2 drafts the consolidation knowledge doc; CLAUDE.md PR is the follow-on. Sprint-12 evidence supports promotion of all three (CSI-7/8/9 = aggregation, codex P1 ×4 = field-isolation, CSI-11 = plan-git-reconcile). |

Cross-ref: W-F11 knowledge doc `i4-substrate-decisions.md` captures the OQ-CSV-1..6 ratification chain with file:line evidence. PP-11 will enumerate the OQ-CSV-7..12 chain similarly.

---

## 13. Risk matrix [UPDATED v3]

### 13.1 i4 quantization precision (MED — UNCHANGED)

*(UNCHANGED)* — per-dim calibration, bipolar interpretation, i8 fallback if needed.

### 13.2 Reunification transcoder lossiness (LOW-MED — RESOLVED, no change)

D-CSV-9 shipped PR #387. Lossy at the per-channel breakdown but acceptable for commit-tier. Ghost-edge mechanism preserves cascade history.

### 13.3 Witness corpus unbounded growth (MED → HIGH) [UPDATED v3] 🆕 v3

**Risk INCREASED.** D-CSV-6a shipped with W-G2 HashMap placeholder for the index. The HashMap grows unbounded until D-CSV-6b + D-CSV-16 land. Sprint-13 entry MUST schedule D-CSV-16 before any production-scale stress test. Pruning policy spec still needed.

### 13.4 ndarray cam_pq shape mismatch (NEW — HIGH) 🆕 v3

**NEW risk.** The W-G2 `WitnessIndexCamPq` HashMap placeholder was designed against a *contract surface* (`add`, `query_nearest`, `evict_oldest`) that may not align with the real `ndarray::hpc::cam_pq` API. OQ-CSV-9 surfaces this gap. PP-5 drafting D-CSV-16 MUST first enumerate the actual `ndarray::hpc::cam_pq` surface (cross-repo) before committing to a wiring approach. Failure mode: D-CSV-16 implementation discovers the real API doesn't match the placeholder, requiring re-spec mid-sprint. **Mitigation:** require PP-5 to include an explicit "cam_pq API surface inventory" section before drafting the spec; consult `family-codec-smith` agent.

### 13.5 Downstream consumer ABI break from QualiaColumn cutover (MED — UNCHANGED)

D-CSV-5b sprint-13 cutover requires consumer audit. Sibling approach minimized risk; cutover is the last mile.

### 13.6 SIMD portability (NEW — MED) 🆕 v3

**NEW risk.** D-CSV-13b targets AVX-512 + NEON. Per-ISA intrinsic differences (lane widths, mask predicate semantics, saturation behavior) can produce silent semantic divergence between scalar / AVX-512 / NEON paths if not tested against the scalar reference. OQ-CSV-8 resolves the feature-gate strategy. **Mitigation:** mandatory cross-ISA parity test matrix (scalar reference × AVX-512 × NEON) per the D-CSV-8 scalar parity pattern PR #387 established. Estimated ~12+ tests per ISA path.

### 13.7 Rayon thread-pool contention (NEW — LOW-MED) 🆕 v3

**NEW risk.** D-CSV-17 introduces three new rayon `par_*` paths in ndarray that share the workspace global pool with existing rayon consumers. Saturation under heavy parallel sweep is plausible but not benchmarked. OQ-CSV-10 default (workspace-global pool) preserves the current pattern; per-call sizing is sprint-14+ polish. **Mitigation:** baseline benchmark before merge.

### 13.8 D-CSV-11 ndarray PR coordination (HIGH → MED) [UPDATED v3]

**Risk REDUCED.** Sprint-12 Wave F shipped the ndarray `hpc/stream/mod.rs` registration via the **ndarray PR #147 merge** (CSI-9 fix; ndarray master `e956e9d9`). The lance-graph-side d4e5bbc aggregation commit closed CSI-7 (workspace member) + CSI-8 (lance-graph lib.rs orphan), NOT CSI-9. Cross-repo coordination is now established; sprint-13 D-CSV-17 follows the same pattern with lower coordination cost.

### 13.9 Subagent isolation — workspace build conflicts (MED — UNCHANGED) [carries from v2]

TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1. Sprint-13 workers must use `--manifest-path` workaround.

### 13.10 v1-API-under-v2 alias anti-pattern (E-META-10) [UPDATED v3]

**Promoted to iron-rule candidate I-FEATURE-GATE-FIELD-ISOLATION** per CSI-18. PP-2 drafts the consolidation. Sprint-13 worker template v2 (PP-8) bakes the field-isolation-matrix test requirement into every v2-style layout change prompt.

### 13.11 Two-TrustTexture coexistence (CSI-1 residual — MED, UNCHANGED) [carries from v2]

TD-TRUST-TEXTURE-DUPE-1.

### 13.12 Disk quota pressure from parallel cargo builds (MED — UNCHANGED) [carries from v2]

ISSUE-X3. Sprint-13 fleets should `cargo clean` at sprint-start.

### 13.13 Aggregation gap (CSI-13 → CSI-18) (HIGH → LOW) [UPDATED v3] 🆕 v3

**Risk REDUCED via Wave G + CSI-18.** Sprint-12 Wave F shipped CSI-7 / CSI-8 (lance-graph-side) via the d4e5bbc aggregation commit + the PR #389 codex P2 follow-up; CSI-9 (cross-repo ndarray-side) shipped via ndarray PR #147 merge (master `e956e9d9`). Sprint-13 worker template v2 (PP-8) bakes the I-AGGREGATION-DISCIPLINE iron-rule candidate into the prompt template. Mechanism: every worker prompt includes "your lib.rs/mod.rs hunk OR an explicit aggregation deliverable as the wave's final worker."

### 13.14 Plan-git drift (CSI-11) (LOW — UNCHANGED, mitigated via I-PLAN-GIT-RECONCILE candidate) [carries from v2]

W-F12 plan v2 drifted on D-CSV-5a. PP-8 worker template v2 mandates `git log origin/main -20` reconcile step for plan-author workers.

---

## 14. Test plan extension [UPDATED v3]

### Sprint-11 + sprint-12 tests landed

| Wave / D-id | Tests added | Source |
|---|---|---|
| D-CSV-1/3/4 (Wave A) | 16 + 8 (24) | Layout + CollapseGateEmission |
| D-CSV-2 (Wave B) | 14 | QualiaI4_16D |
| D-CSV-8/9 (Wave E, PR #387) | ~20 | MUL scalar parity + transcoder |
| D-CSV-5a (Wave C, PR #385) | ~8 | Sibling double-write |
| D-CSV-6a + D-CSV-7 (Wave D, PR #386) | ~24 | WitnessCorpus + MailboxSoA |
| D-CSV-10 (W-F1, PR #388) | ~24 #[test] markers | sigma-tier-router |
| D-CSV-11 streams (W-F4/5/6, PR #388) | ~36 #[test] markers (3×12) | QualiaStream / InferenceStream / SplatFieldStream |
| D-CSV-12 splat ops (W-F7, PR #388) | 16 | splat_gaussian / score_hole_closure / replay_coherence / emit_if_epiphany |
| AttentionMask SoA + Actor (W-F2/3, PR #388 + #389) | ~28 + 2 backend impl | attention_mask + attention_mask_actor |
| Wave G W-G2 (`WitnessIndexCamPq` HashMap) | ~8 estimated | Placeholder add/query/evict tests |
| Wave G W-G3 (i4 SIMD scalar batch API) | ~12 estimated | Scalar parity matrix |
| **Subtotal sprint-11 + sprint-12** | **~190 new tests** | — |

### Sprint-13 test targets 🆕 v3

| Phase / D-id | Target | Approach |
|---|---|---|
| **D-CSV-5b** (cutover) | 15 | Cross-consumer compile-check matrix + clippy `--tests --no-deps -D warnings` gate |
| **D-CSV-6b** (full WitnessCorpus) | 12 + 4 benches | Pruning policy + salience decay + corpus root anchor |
| **D-CSV-13b** (SIMD intrinsics) | 24+ (12 AVX-512 + 12 NEON) | Per-ISA scalar parity (against D-CSV-13 scalar reference) + throughput bench (4-8× target gain) + `#[target_feature]` correctness + `is_x86_feature_detected!`/`is_aarch64_feature_detected!` runtime gate |
| **D-CSV-14** (on-Think methods) | 16 | 4 methods × (correctness + state-isolation + ndarray-stream-integration + bench) |
| **D-CSV-15b** (VAMPE-Jirak threshold) | 8 + 2 benches | Coupled VAMPE-Jirak derivation + threshold validation against hand-tuned baseline |
| **D-CSV-16** (cam_pq real wiring) | 12 + 4 benches | API parity vs W-G2 HashMap placeholder + CAM-PQ retrieval correctness + Markov ±500 window |
| **D-CSV-17** (par_*_stream rayon) | 12 | 3 streams × (single-thread parity + parallel correctness + work-stealing harness + bench) |
| **Sprint-13 projected** | **~80+ new tests** | — |

**Aggregate test target post-sprint-13:** ~760 (Miri baseline) + ~190 sprint-11/12 + ~80 sprint-13 = ~1030 cumulative on the path to the 1900 Miri scope target.

---

## 15. Sprint phasing [UPDATED v3]

### Phase A (sprint-11): COMPLETE

D-CSV-1 / D-CSV-2 / D-CSV-3 / D-CSV-4 — Shipped (PR #383 + #384).

### Phase B (sprint-12): COMPLETE [UPDATED v3]

D-CSV-5a / D-CSV-6a / D-CSV-7 — Shipped (PR #385 + #386). D-CSV-5b cutover queued for sprint-13.

### Phase C (sprint-12): COMPLETE [UPDATED v3]

D-CSV-8 / D-CSV-9 / D-CSV-10 / D-CSV-13 (scalar batch API) — Shipped (PR #387 + #388).

### Phase D (sprint-12): PARTIAL [UPDATED v3]

D-CSV-11 (forward-iter scaffold) / D-CSV-12 (scalar free fns) — Shipped (PR #388).
D-CSV-14 / D-CSV-17 — Queued sprint-13.

### Phase E (sprint-13): forward plan 🆕 v3

- **D-CSV-5b** — QualiaColumn cutover (remove `[f32; 17]`) after consumer audit.
- **D-CSV-6b** — Full WitnessCorpus (pruning + salience decay) — depends on D-CSV-16.
- **D-CSV-13b** — SIMD intrinsics (AVX-512 + NEON) targeting D-CSV-13 scalar API.
- **D-CSV-14** — On-Think method migration for splat ops (L-20 realization).
- **D-CSV-16** — `WitnessIndexCamPq` real `ndarray::hpc::cam_pq` wiring (replaces W-G2 HashMap).
- **D-CSV-17** — `par_*_stream` rayon variants.

### Phase F (sprint-14+): backlog 🆕 v3

- **D-CSV-15b** — VAMPE-coupled Jirak threshold refinement (resolves `I-NOISE-FLOOR-JIRAK` debt; depends on VAMPE coupled-revival).
- TD-COLLAPSE-GATE-SMALLVEC-1 polish (~20 LOC + smallvec dep or feature-gate).
- TD-TRUST-TEXTURE-DUPE-1 canonical rename across causal-edge + contract crates.
- Trait split for on-Think methods (`impl SplatOps for Think`) if multi-Think-shape emergence demands it (OQ-CSV-11 default rejected this for sprint-13).

---

## 16. Test target growth [UPDATED v3]

| Sprint | Tests added | Source | Running total |
|---|---|---|---|
| sprint-10 (spec sprint) | 0 (spec only) | PR #372 + PR #381 | ~760 (Miri baseline) |
| sprint-11 | ~58 | Waves A/B + PR #387 (D-CSV-8/9 ~20) | ~818 |
| sprint-12 | ~132 (Wave F implementation + W-G2 + W-G3 estimates) | PR #385/#386/#388/#389 + Wave G | ~950 |
| sprint-13 projected | ~80+ | D-CSV-5b + D-CSV-6b + D-CSV-13b + D-CSV-14 + D-CSV-16 + D-CSV-17 + D-CSV-15b initial | ~1030+ |
| sprint-14+ target | ~100+ | D-CSV-15b full + D-CSV-11 par_* Miri scope expansion | ~1130+ toward 1900 |

Note: sprint-12 ~132 is the conservative estimate; some D-CSV-11 stream tests are #[test] markers that decompose into sub-cases — distinct test-function count is the floor.

---

## 17. Cross-references [UPDATED v3]

### 17.1 v2 plan (predecessor)

- `.claude/plans/cognitive-substrate-convergence-v2.md` — W-F12 sprint-12 forward plan. This v3 supersedes for sprint-13+ planning; v2 remains archival for sprint-12 archaeology.

### 17.2 v1 plan (great-grand-predecessor)

- `.claude/plans/cognitive-substrate-convergence-v1.md` — original 2026-05-15 cross-session A2A discussion. Architectural anchor for L-1..L-20.

### 17.3 Sprint-12 meta-reviews

- `.claude/board/sprint-log-11/meta-review.md` — W-F10 Sonnet draft (sprint-11 + Wave F).
- `.claude/board/sprint-log-11/meta-review-opus.md` — W-Meta-Opus honest review (sprint-11 + Wave F; grade B). CSI-7/8/9/10/11/12/13 cross-cutting findings.
- `.claude/board/sprint-log-12/meta-review-opus.md` — W-G1 sprint-12 honest review (grade B+). CSI-18 doctrine consolidation. (PP-7 drafts; not yet on disk at time of v3 authoring — see PP-7 spec for the format precedent.)

### 17.4 Knowledge docs

- `.claude/knowledge/i4-substrate-decisions.md` — W-F11 knowledge doc; OQ-CSV-1..6 ratification chain with file:line evidence.
- `.claude/knowledge/iron-rules-doctrine.md` (PP-2 drafts) 🆕 v3 — consolidation of I-SUBSTRATE-MARKOV + I-NOISE-FLOOR-JIRAK + I-VSA-IDENTITIES with three sprint-12-promoted candidates (I-AGGREGATION-DISCIPLINE, I-FEATURE-GATE-FIELD-ISOLATION, I-PLAN-GIT-RECONCILE).

### 17.5 Sprint-13 spec packages 🆕 v3

PP-3 through PP-6 draft the new D-CSV-* specs. PP-8 drafts the worker template v2.

- **PP-3** — `.claude/specs/pr-csv-17-par-streams.md` — D-CSV-17 spec (rayon `par_*_stream` variants in ndarray).
- **PP-4** — `.claude/specs/pr-csv-14-on-think-splat.md` — D-CSV-14 spec (on-Think method migration for splat ops).
- **PP-5** — `.claude/specs/pr-csv-16-witness-cam-pq.md` — D-CSV-16 spec (`WitnessIndexCamPq` real cam_pq wiring); MUST include "cam_pq API surface inventory" section.
- **PP-6** — `.claude/specs/pr-csv-13b-simd-intrinsics.md` — D-CSV-13b spec (AVX-512 + NEON intrinsics for batch i4).
- **PP-8** — `.claude/specs/worker-template-v2.md` — sprint-13 worker prompt template baking in CSI-18 iron-rule candidates.
- **PP-11** — `.claude/specs/sprint-13-oq-catalog.md` — OQ-CSV-7..12 enumeration with default recommendations.

### 17.6 Sprint-10 spec corpus (parent plan + W1..W12 patches)

*(UNCHANGED from v2 §17.4)* — `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` parent plan; W1-W12 spec patches in `.claude/specs/pr-ce64-mb-*.md` and `.claude/specs/sprint-10-*.md`.

### 17.7 Recent PRs this builds on

*(v2 §17.6 EXTENDED with sprint-12 PRs)*

- **PR #381** (2026-05-16, `a7c0545`) — 8-spec patch bundle pre-sprint-11 prep.
- **PR #383** (sprint-11 Wave A, `03bd175`) — D-CSV-1/3/4 + signed-mantissa NARS.
- **PR #384** (sprint-11 Wave B) — D-CSV-2 QualiaI4_16D.
- **PR #385** (sprint-11 Wave C, `6f58418` merge) — D-CSV-5a sibling QualiaI4Column.
- **PR #386** (sprint-11 Wave D, `33110c8` merge) — D-CSV-6a WitnessCorpus partial + D-CSV-7 MailboxSoA.
- **PR #387** (sprint-11 Wave E) — D-CSV-8 MUL scalar + D-CSV-9 transcoder.
- **PR #388** (sprint-12 Wave F, `77f2d26` merge) — D-CSV-10 sigma-tier-router + D-CSV-11 streams (forward-iter) + D-CSV-12 splat ops free fns + AttentionMask SoA/Actor + 12-worker fleet + W-F10/F11/F12 governance.
- **PR #389** (sprint-12 Wave F codex P2 follow-up, `b526485` merge) — `AttentionMaskBackend` impl for `AttentionMaskSoA` + canonical `MailboxId` import (CSI-10 closed).

### 17.8 Doctrinal anchors (CLAUDE.md sections)

*(UNCHANGED from v1/v2)*

- **"The Click" (P-1)** — Markov ±5, role-key bind/unbind, free-energy minimization.
- **"AGI-as-glove"** — 4 BindSpace columns = AGI surface.
- **`I-SUBSTRATE-MARKOV`** — Chapman-Kolmogorov via VSA bundling.
- **`I-NOISE-FLOOR-JIRAK`** — Jirak 2016 weak-dependence Berry-Esseen.
- **`I-VSA-IDENTITIES`** — identity-fingerprint VSA, not content register.
- **"The shader can't resist the thinking"** — active inference dispatch driver.

### 17.9 Sprint-12 iron-rule candidates (CSI-18 consolidation) 🆕 v3

Per PP-2 knowledge doc + CLAUDE.md follow-on PR:

- **I-AGGREGATION-DISCIPLINE (candidate)** — every fleet wave MUST schedule an explicit aggregation commit before meta-review spawn. Evidence: CSI-7/8/9 from sprint-12 Wave F.
- **I-FEATURE-GATE-FIELD-ISOLATION (candidate)** — every v2-style layout change requires field-isolation-matrix tests at the bit boundary. Evidence: E-META-10 + 4 codex P1 instances in PR #383.
- **I-PLAN-GIT-RECONCILE (candidate)** — plan-author worker MUST run `git log origin/main -20` and reconcile "In PR" cells before commit. Evidence: CSI-11 from sprint-12 Wave F.

---

## 18. Status [UPDATED v3]

| Field | Value |
|---|---|
| **Status** | ACTIVE — Phase A/B/C COMPLETE; Phase D partial (forward-iter scaffold + scalar free fns shipped; on-Think methods + par_* rayon + cam_pq wiring queued); Phase E (sprint-13) scope locked; Phase F (sprint-14+) backlog enumerated. |
| **Confidence (2026-05-16)** | HIGH on shipped architecture (D-CSV-1/2/3/4/5a/6a/7/8/9/10/11-scaffold/12-free-fns + W-G2/W-G3 sprint-12); HIGH on gapless-baton model (CollapseGateEmission); HIGH on i4-mantissa NARS; HIGH on QualiaColumn i4-16D; MED on Rubicon-resonance threshold (hand-tuned → D-CSV-15b sprint-13+); MED on `WitnessIndexCamPq` shape (placeholder → D-CSV-16 sprint-13 OQ-CSV-9 resolution); MED on SIMD portability (scalar API → D-CSV-13b sprint-13 intrinsics). |
| **Branch** | `claude/sprint-13-preflight-planning` (this file) |
| **Predecessor** | `.claude/plans/cognitive-substrate-convergence-v2.md` (v2, W-F12) |
| **Great-grand-predecessor** | `.claude/plans/cognitive-substrate-convergence-v1.md` (v1, 2026-05-15) |
| **Successor** | None (this is v3; v4 to be authored post-sprint-13 if scope warrants). |

---

*End of cognitive-substrate-convergence-v3.md. Authored 2026-05-16 by PP-1 (Opus planner) as sprint-13 preflight forward-plan document. Captures sprint-12 Wave F + Wave G outcomes (PRs #385/#386/#387/#388/#389) and locks sprint-13 scope (D-CSV-5b/6b/13b/14/16/17) before context dilution. Single canonical reference for sprint-13 implementation planning. Sister files: PP-2 iron-rule doctrine consolidation, PP-3..PP-6 new D-CSV-* specs, PP-7 sprint-log-12 meta-review, PP-8 worker template v2, PP-11 sprint-13 OQ catalog.*
