# Cognitive Substrate Convergence — i4 Mantissa, Gapless Baton, Active Inference (v1)

> **Status:** PROPOSAL (sprint-10 architectural decisions consolidated; awaits sprint-11+ implementation ratification)
>
> **READ BY:** integration-lead, truth-architect, host-glove-designer, palette-engineer, family-codec-smith, certification-officer, bus-compiler, nars-engineer, anyone implementing sprint-10's CausalEdge64 v2 or touching `QualiaColumn` / `EdgeColumn` / `MailboxSoA` / `CollapseGate` wire format / `MUL` evaluation path
>
> **CONSOLIDATES:**
> - sprint-10 spec corpus (PR #372 / PR #374 + 8 `.claude/knowledge/` docs)
> - PR #375 (PR-LL-1: NARS Intervention/Counterfactual variants merged in `nars_dispatch.rs`)
> - PR #379 (4-branch retirement of orphan content-superseded branches)
> - PR #373 (neurosymbolic-rlvr-causal-curriculum-v1.md learning-layer curriculum)
> - Live cross-session A2A architectural discussion (2026-05-15) covering: SPO-W vs SPO-G, NARS signed mantissa, qualia i4-16D, CollapseGate wire format, autopoiesis + philosophic entanglement, Rubicon-resonance orchestration
>
> **DOES NOT REPLACE:** the per-PR specs in `.claude/specs/pr-ce64-mb-*.md` — those remain the implementation-level contracts. This plan is the **architectural anchor** that locks the design decisions BEHIND those specs, so sprint-11+ implementations don't re-derive or drift.

---

## 1. One-paragraph thesis

The CausalEdge64 v2 layout, the QualiaColumn quantization, the CollapseGate wire format, the witness-corpus pointer design, the MUL evaluation algebra, the Σ-tier router's Rubicon-resonance orchestration, and the thinking-engine ↔ cognitive-shader-driver SoA reunification are not seven independent design questions — they **converge into one substrate** where (a) signed i4 mantissa is the universal precision family across NARS / Qualia / ThinkingAtom / direction, (b) the i4 payload IS its own CAM key so content equals address, (c) inter-mailbox handoff is discrete baton tuples with zero analog bucket, (d) `Vsa16kF32` is narrowed to intra-tier Markov accumulation + crystal carrier + grammar bind/unbind testing, (e) cycle driver is free-energy gradient (active inference) not request/response, and (f) mailboxes are spatial-temporal meaning accumulators not channels. Autopoiesis of thinking styles and philosophic entanglement across mailboxes fall out of the shared substrate without extra mechanism. This plan is the single canonical reference for sprint-11+ implementation and the locking point for the architectural decisions made during sprint-10 + the post-sprint-10 cross-session discussion, captured before context dilution.

---

## 2. Why now — context-dilution gate

Sprint-10's 12-worker fleet surfaced findings the parent plan `causaledge64-mailbox-rename-soa-v1.md` did NOT foresee:

1. **Dual `CausalEdge64` types** (E-META-7, in `EPIPHANIES.md`): `causal_edge::CausalEdge64` (SPO-palette layout, `crates/causal-edge/src/edge.rs:60`) ≠ `thinking_engine::layered::CausalEdge64` (8-channel cascade, `crates/thinking-engine/src/layered.rs:45`). Same name, different bit semantics, different consumers.
2. **p64 drift origin pinpointed** at `crates/lance-graph-planner/src/cache/convergence.rs:18-22` `#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring` — the wiring was started, never finished.
3. **Three-zone hot-path mental model** corrects prior "AriGraph reads = µs cold-path joins" framing.
4. **Signed-mantissa NARS** insight — current 3-bit unsigned enum wastes the symmetry; signed i4 carries direction × rule.
5. **i4-as-CAM** insight — i4-16D has 16¹⁶ ≈ 1.8×10¹⁹ unique states, enough entropy to be both content AND CAM address.
6. **Gapless baton model** — `Vsa16kF32` between mailboxes was always over-engineered; discrete `(u16 target, CausalEdge64)` tuples suffice.
7. **Qualia i4-16D** — 9× compression from f32-18D; aligns with NARS mantissa precision family; Wisdom × Staunen → Magnitude becomes one SIMD multiply.

If these don't get captured in one plan now, the next session will re-derive or partially apply them, leading to inconsistent sprint-11 implementations across W1/W2/W5/W6/W7. This plan is the lock.

---

## 3. Three findings from sprint-10 that anchor this plan

### 3.1 Dual `CausalEdge64` types (E-META-7)

| Type | Location | Layout | Consumers |
|---|---|---|---|
| `causal_edge::CausalEdge64` | `crates/causal-edge/src/edge.rs:60` | (S/P/O palette + NARS f/c + Pearl mask + direction + inference + plasticity + temporal) | `NarsTables`, `lance-graph-planner::cache::nars_engine`, `cognitive-shader-driver::BindSpace::EdgeColumn`, AriGraph SPO commit |
| `thinking_engine::layered::CausalEdge64` | `crates/thinking-engine/src/layered.rs:45` | 8 channels × 8 bits (BECOMES/CAUSES/SUPPORTS/REFINES/GROUNDS/ABSTRACTS/RELATES/CONTRADICTS) | `TierEngine::emit_causal_edges`, `apply_edges`, downstream tier energy perturbation |

**Reunification path (Option R-3):** transcode 8-channel → SPO-palette at thinking-engine L3 commit boundary. Signed-mantissa NARS makes the transcode a near-bitcast: `8ch_net_strength.signum() → mantissa_sign`, `magnitude → base_rule`. Mapping in `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md` §6.3.

### 3.2 p64 drift origin

`crates/lance-graph-planner/src/cache/convergence.rs:18-22`:

```rust
#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring
use super::nars_engine::{CausalEdge64, SpoHead, MASK_SPO};
```

The `nars_engine::CausalEdge64` is the SPO variant; the 8-channel variant was reinvented locally in thinking-engine instead of imported here. The `#[allow(unused_imports)]` annotation is the smoking gun for where dual-variant drift formalized.

### 3.3 Three-zone hot-path mental model

| Zone | Mechanism | Cost |
|---|---|---|
| **Zone-1** (cycle-speed) | thinking-engine MatVec (`distance_table × energy → top-k`) + AriGraph `entity_index: HashMap<String, Vec<usize>>` lookup | **200-500 ns** MatVec + **20-200 ns** HashMap O(1) |
| **Zone-2** (SPO-as-3D-vector ANN) | blasgraph + neighborhood cascade HEEL→HIP→TWIG→LEAF via `zeckf64()` | **20-1200 µs** progressive precision |
| **Zone-3** (cold path) | `lance-graph-planner` DataFusion projection + columnar joins | **>1 ms** |

**Correction of prior framing:** AriGraph ≠ DataFusion. AriGraph **reads** are Zone-1 HashMap lookups (`triplet_graph.rs::entity_index`); only AriGraph→SPO **writes** via `spo_bridge::promote_to_spo()` flow into the cold tier.

---

## 4. The five compressions

### 4.1 Encoding — signed i4 mantissa family

Universal quantization vocabulary across the substrate:

| Field | Encoding | Range |
|---|---|---|
| NARS Inference mantissa | i4 signed | −8..+7 (direction × rule) |
| Qualia 16D dimensions | i4 signed × 16 | −8..+7 per dim (valence × intensity) |
| ThinkingAtom32x4 (`p64-bridge::STYLES`) | i4 signed × 32 | −8..+7 per dim |
| Direction triad | i4 (in CausalEdge64) | sign per S/P/O plane |
| `Vsa16kI8` (CLAUDE.md switchboard tier — quantized fingerprint) | i8 × 16384 | −128..+127 |

Products stay in family: `i4 × i4 → i8`, `i8 × i8 → i16`. Wisdom × Staunen → Magnitude is one SIMD multiply, no float conversion.

### 4.2 Wire format — discrete baton, no analog bucket

Inter-mailbox / inter-tier / inter-cycle wire format is `Vec<(u16 target, CausalEdge64)>` discrete tuples. `Vsa16kF32` does NOT cross mailbox boundaries. There is no encode/decode at the boundary — the baton IS the wire. Gapless cognition.

### 4.3 Addressing — i4 IS content AND address (CAM)

i4-16D's 16¹⁶ ≈ 1.8×10¹⁹ unique states exceeds any practical SoA address space. The qualia/mantissa vector serves simultaneously as content (what it means) AND CAM key (lookup address). Per CLAUDE.md "I-VSA-IDENTITIES" iron rule but sharper: the identity IS the content, no pointer indirection.

### 4.4 Temporal axis — structural, not stored

Temporal field in CausalEdge64 drops. Time is carried by:
- **Cycle order** (within a tier): `MailboxSoA::cycle: u32`
- **Relative order** (per-discourse): position in `SpoWitnessChain` / WitnessCorpus chain
- **Wall-clock** (commit anchor): AriGraph `Triplet.timestamp: u64`

Matches Shaw's "temporal causality is structural" doctrine from CLAUDE.md "The Click" §2 — extended one level up from VSA-braiding to chain-position.

### 4.5 Cycle driver — entropy-driven, not request-driven

Per CLAUDE.md "The shader can't resist the thinking": active inference is the dispatch mechanism. Free-energy floor (`MUL::homeostasis`) is the rest condition. Σ10 Rubicon resonance threshold is the commit trigger. The cycle has no external `request()` interface — entropy of unresolved state IS the dispatch signal.

---


## 5. Locked architectural decisions (20 items)

Each row is a decision that this plan LOCKS for sprint-11+ implementation. Cross-references are file:line.

| # | Decision | Rationale | Lives in |
|---|---|---|---|
| **L-1** | **Keep TWO `CausalEdge64` types** at sprint-11 (transcode at L3 commit boundary, not unify) | Each variant is optimal for its tier; reunification (Option R-3 in synergies doc §6) is a separate sprint-12+ task | `crates/causal-edge/src/edge.rs:60` (SPO) + `crates/thinking-engine/src/layered.rs:45` (8-channel) |
| **L-2** | **Drop temporal (12 bits)** from CausalEdge64 v2 | Redundant with chain-position + AriGraph anchor; "temporal causality is structural" doctrine | `edge.rs:52-63` field becomes available |
| **L-3** | **Drop G-slot (5 bits)** that was being proposed | Three-way redundant: tenant via SoA partition, belief via witness corpus root, ontology via palette family-prefix | not added |
| **L-4** | **Expand InferenceType 3→4 bits SIGNED** mantissa (−8..+7) | Direction × rule composition; 16 states cover NAL-1 + Pearl modifiers; aligns with `thinking_engine` 8-channel signum | `edge.rs:46-48` widened to bits 46-49 |
| **L-5** | **Causal mask (3 bits) IS the Pearl-rung axis** — no separate Pearl-3 modifier bit | `causal_mask = 0b111 SPO` already encodes Counterfactual; modifier would duplicate | `pearl.rs:11-49` unchanged |
| **L-6** | **W-slot 6 bits** = discourse corpus root handle (64 active corpora) | Witness corpus is CAM-PQ-indexed, unbounded; W-slot is the entry pointer | NEW field in CausalEdge64 v2 |
| **L-7** | **Truth-band lens 2 bits** (4 states incl. "13% ambiguous direction") | Carries committed-vs-ambiguous expressivity without forcing binary commitment | NEW field |
| **L-8** | **KEEP direction (3b) + plasticity (3b) in edge** | Both are load-bearing dispatch payload; relocating them costs extra Zone-1 lookup per cycle | unchanged |
| **L-9** | **PR-LL-1 `Intervention`+`Counterfactual` slot into `Reserved5`+`Reserved6`** of canonical `causal_edge::InferenceType` (when v2 ships) | PR #375 added these to `nars_dispatch.rs` only; canonical edge enum needs to absorb them | `crates/causal-edge/src/edge.rs:22-25` Reserved slots |
| **L-10** | **QualiaColumn → i4-16D signed** (replaces `[f32; 18]`) | 9× compression; aligns with mantissa family; Wisdom × Staunen → Magnitude is 1 SIMD mul | `cognitive-shader-driver::bindspace.rs` QualiaColumn type |
| **L-11** | **MetaColumn unchanged** — MetaWord bits, 36 ThinkingStyles (`contract::thinking.rs`) | Different tier from NARS rule; styles dispatch the cycle's mode, NARS records what each edge did | unchanged |
| **L-12** | **FingerprintColumns unchanged** — `Vsa16kF32` carrier (64 KB per row) | Preserved for Markov ±5 + crystal carrier + grammar bind/unbind testing | unchanged |
| **L-13** | **CollapseGate wire format** = `Vec<(u16 target, CausalEdge64)>` + implicit provenance | No `Vsa16kF32` between mailboxes; no encode/decode at boundary; gapless | NEW spec — `contract::collapse_gate::CollapseGateEmission` |
| **L-14** | **Mailbox semantics:** spatial-temporal meaning accumulators, NOT channels | Per W6 `MailboxSoA<N>` — each row is a neuron-like accumulator with plasticity counter | `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) |
| **L-15** | **Σ-tier router orchestration:** Rubicon-resonance, NOT expected-result | Σ10 fires when ΔF < threshold AND resonance > Rubicon-bar; per W7 `SigmaTierRouter` | `pr-ce64-mb-6-sigma-tier-router.md` (W7) |
| **L-16** | **Witness chain** is **sorted by emission cycle, drop-oldest truncation** | "Sort witness by time" structural-temporal pattern; W5-INV-CHAIN-ORDER iron rule | `pr-ce64-mb-4-arigraph-spo-g.md` (W5) needs invariant added |
| **L-17** | **`SpoWitnessChain<32>` → `WitnessCorpus` (CAM-PQ-indexed, unbounded)** | Bounded chain doesn't scale to discourse; CAM-PQ + CLAM tree + position-window is the canonical lookup | W5 spec — needs §replace of `SpoWitnessChain<32>` |
| **L-18** | **MUL evaluation in integer SIMD** (i4 × i4 → i8 products) | DK / TrustTexture / FlowState read i4 qualia + signed mantissa; no float math in hot path | `lance-graph-planner::mul/` module migration |
| **L-19** | **8-channel ↔ SPO-palette transcode at L3 commit** (Option R-3) | Signed mantissa makes transcode near-bitcast; eliminates dual-variant runtime cost | NEW: `thinking_engine::commit::transcode_to_spo()` |
| **L-20** | **Vertical streaming structs in ndarray** are the missing layer for full architecture | `qualia.history(window: 100)`, `inference.trajectory(±5)`, `splat.evolve(steps)` — temporal axis as method, not external scheduler | NEW: ndarray struct-method surface extension |

---

## 6. Final CausalEdge64 v2 bit layout

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

**Reclaim arithmetic:** drop temporal (−12 bits) → spend on Inference mantissa expansion (+1), W slot (+6), Truth-band lens (+2) = 9 spent, 3 spare. Spare reserved for: explicit Rubicon-commit marker bit, Markov-decay-rate quantum, or sprint-12+ probe-derived needs (`I-NOISE-FLOOR-JIRAK` calibrated threshold storage).

**Signed mantissa encoding rationale** (per L-4 + the dual-NARS-cascade insight):

| Sign | Direction | Magnitude interpretation |
|---|---|---|
| `+` (0..+7) | forward-chain / compose / commit | Deduction, Synthesis, Revision-positive, Induction (forward generalization) |
| `−` (−8..−1) | backward-chain / decompose / refute | Abduction, Contraposition, Revision-negative, Counterfactual |

`abs(mantissa)` selects the base NARS rule (8 base slots); `signum(mantissa)` selects direction. 16 distinct directed-inferences, naturally composable. Maps cleanly to `thinking_engine::CausalEdge64::net_strength()` signum for the L3 transcode.


---

## 7. Column-level changes (BindSpace SoA)

The 4-column AGI-as-glove invariant (per CLAUDE.md `lab-vs-canonical-surface.md` §"AGI IS the struct-of-arrays") stays intact. What changes is the per-column encoding:

### 7.1 EdgeColumn (Planner axis)

| Before | After |
|---|---|
| `[CausalEdge64; N]` with v1 bit layout | `[CausalEdge64; N]` with v2 bit layout per §6 |
| 8 B / row | 8 B / row (unchanged total — bit allocation differs) |

### 7.2 QualiaColumn (Angle axis)

| Before | After |
|---|---|
| `[[f32; 18]; N]` (72 B / row) | `[QualiaI4_16D; N]` (8 B / row, packed i4 × 16 signed) |
| Footprint at 1M rows: 72 MB | Footprint at 1M rows: **8 MB** (9× compression) |

The historical 17/18 dim discrepancy (17D `QualiaVector` in contract::qualia, 18D per-row column with derived dim) resolves cleanly at 16 — i4-16D is exactly 8 bytes, no padding, no derived "Magnitude" surface dim. Magnitude is computed on-demand via `Wisdom × Staunen → i8` (1 SIMD multiply per row sweep).

**Required new type** in `lance-graph-contract::qualia`:

```rust
/// i4-16D signed packed qualia vector. 8 bytes / 16 dims / range −8..+7 per dim.
/// Replaces [f32; 18] QualiaColumn slot. Lane-width: 32 i4 lanes per AVX-512 register.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QualiaI4_16D(pub u64);  // 16 × 4-bit packed signed
```

Per-dim semantics (sign = valence, magnitude = intensity):

| Dim idx | Qualia | + means | − means |
|---|---|---|---|
| 0 | Wisdom | sage / informed | naive / under-informed |
| 1 | Staunen (Wonder) | awe / openness | cynicism / closed |
| 2 | Trust | trust | distrust |
| 3 | Hope | hope | despair |
| 4 | Curiosity | pull-to-explore | pull-to-rest |
| 5 | Doubt | critical engagement | dismissive |
| 6 | Flow | absorbed | scattered |
| 7 | Wille (volition) | active intent | passive surrender |
| 8 | Resonance | aligned | dissonant |
| 9 | Surprise | ΔF positive (new) | ΔF negative (predicted) |
| 10 | Confidence | belief stable | belief shaky |
| 11 | Salience | attention pulled | attention diffuse |
| 12 | Coherence | story holds | story breaks |
| 13 | Magnitude | strong signal | weak signal |
| 14 | Reflective | introspective | object-oriented |
| 15 | Spare | reserved for sprint-12+ probe | reserved |

(Per-dim assignments are CONJECTURE — final ratification requires the qualia-engineer agent + cross-check with `crates/thinking-engine/src/qualia.rs`'s existing dimension layout.)

### 7.3 MetaColumn (Thinking axis)

**Unchanged.** `MetaWord` bits packing the 36 ThinkingStyle selector + modulation weights. Per CLAUDE.md "AGI-as-glove" — Thinking style dispatches the cycle's mode; this column carries that selection per row.

### 7.4 FingerprintColumns (Topic axis)

**Unchanged.** `Vsa16kF32` carrier (16384 × f32 = 64 KB per row). Per L-12 + L-13: `Vsa16kF32` is preserved for intra-cycle Markov bundling + crystal carrier + grammar bind/unbind/recovery testing. Does NOT cross mailbox boundaries; the wire is discrete batons.

---

## 8. CollapseGate wire format (NEW spec)

### 8.1 Type definition

In `lance-graph-contract::collapse_gate`:

```rust
/// Discrete baton emission from one CollapseGate to downstream consumers.
/// No Vsa16kF32 envelope — payload IS its own format per the gapless-baton model.
#[repr(C)]
pub struct CollapseGateEmission {
    /// Per-target discrete batons. Each tuple is one neuron-to-neuron delivery.
    /// SmallVec inline up to 8 to avoid heap allocation in the hot path.
    pub batons: SmallVec<[(u16, CausalEdge64); 8]>,

    /// Implicit cycle dedication: receiver finds the source mailbox + the
    /// emission's position in its witness chain. No explicit cycle_id field
    /// — provenance via (source_mailbox, chain_position).
    pub source_mailbox: MailboxId,
    pub chain_position: u32,

    /// Merge mode hint for the receiving CollapseGate.
    /// `Bundle`  = associative superposition (Markov-respecting).
    /// `Xor`     = single-writer delta (faster, breaks Markov; per I-SUBSTRATE-MARKOV
    ///             iron rule, NOT a transition kernel — only for deltas).
    pub merge_mode: MergeMode,
}
```

### 8.2 Wire-cost budget

- Header (source + chain_pos + mode): 13 bytes
- Per baton: 10 bytes (u16 target + u64 edge)
- 8 inline batons: 80 bytes
- Total inline emission: ~93 bytes

For 4 batons × N=10⁶ cycles/s sustained: ~370 MB/s wire bandwidth. Fits comfortably in any modern memory channel; in-process (same SoA) it's pure SoA writes.

### 8.3 No analog bucket

Three things that would have forced `Vsa16kF32` between mailboxes — all rebutted:

| Candidate need | Why discrete batons suffice |
|---|---|
| Compound bundles | A Vec of N tuples IS the bundle decomposed; receiver's `apply_edges` re-superposes via energy addition. Same algebra. |
| Markov ±5 braiding | Braiding happens INSIDE one tier's MatVec; never crosses a boundary. Top-k collapse to tuples is already in the role-resolved basis. |
| Continuous strength values | Signed i4 mantissa (16 quantized states) + 8-bit f/c is enough for cycle-speed dispatch. Phase-coherent intra-tier work is internal. |

Vsa16kF32 narrows to its three preserved roles (L-12). No leakage into the wire.

---

## 9. Mailbox semantics

### 9.1 Mailboxes are NOT message queues

A `MailboxSoA<N>` row is a **spatial-temporal accumulator** — analogous to a single neuron with many incoming synapses. Multi-source batons land via `apply_edges`; the row's energy integrates; when threshold crosses, the row emits (via the receiving CollapseGate). Per W6 spec, the per-row `plasticity_counter` records the integration history.

The "thought lives in mailboxes (akin to neuron plasticity)" framing is literal: a thought IS the integration state of one or more mailbox rows; it persists as long as the row's energy is above floor; it commits (Σ10 Rubicon) when resonance peak crosses threshold and the row's contents flow through the L3 transcoder to AriGraph.

### 9.2 Philosophic entanglement across mailboxes

Two concepts in mailboxes M_a and M_b share state via three channels:

1. **Common substrate**: both write to the same FingerprintColumns row's `Vsa16kF32` (intra-tier Markov bundling superposes them lossless up to N ≤ √d/4 ≈ 32 items).
2. **Common witness corpus**: both reference back to the same W-slot anchor; downstream readers see them as joint provenance.
3. **Common chain-position**: temporal axis aligned via chain order even without explicit cycle id.

**Entanglement formally** = counterfactual-mask collapse + shared witness chain enforces coherence: committing M_a at causal_mask=`SPO` (Pearl-3 counterfactual) constrains M_b's possible commits if their witness chains overlap. The "philosophic" framing is just Pearl-3 counterfactual reasoning across rows.

### 9.3 Autopoiesis of thinking styles

Per L-10 + L-11: different-dim styles (a 4-dim Skeptical, a 16-dim Empathic, a 32-dim Synthesizer) can superpose in one cycle because **all project onto the same i4 mantissa-bus**. The style selects which qualia/NARS-dims get weighted; the substrate (i4) is shared. Per CLAUDE.md "The Click" §3 *"opinions are committed contradictions preserved"*: two styles producing contradictory commits both land as i4 entries in adjacent EdgeColumn rows, neither overwrites — they coexist as entangled contradictions.

---

## 10. Active inference framing

### 10.1 Cycle driver = free-energy gradient

Per CLAUDE.md "The Click" §"The shader can't resist the thinking": the cycle has no external `request()` entry. The driver is:

```
while  F > homeostasis_floor:
    cycle()             # encode → MatVec → top-k → emit batons → apply
    F = compute_free_energy(self.trajectory, self.awareness, self.prior)
    # F drops as the cycle resolves; if F < floor, shader rests
    # if ΔF < epsilon at high F, epiphany branch fires (Pearl-3)
    # if F doesn't drop, FailureTicket emits (LLM escalation per §"The Click")
```

Σ10 Rubicon crosses when ΔF < commit_threshold AND resonance > Rubicon_bar — that's the irreversible-commit decision. Resonance = cosine similarity vs global_context or codebook (per CLAUDE.md "Meaning = AriGraph facts + resonance + magnitude").

### 10.2 Goal = entropy-of-state, not stated target

The cycle pursues "minimize surprise (F)" — equivalent to "pursue homeostasis" — equivalent to goal-direction without explicit goals (Friston active-inference framework). No goal-misalignment failure mode possible because there's nothing TO misalign with — the only goal is the F-gradient itself.

### 10.3 Environment = tissue, not interface

AriGraph + EpisodicMemory + global_context + CamPqCodec are **methods on the carrier**, not services queried. Per CLAUDE.md "Litmus Test 1": free function = reject; method = accept. Per `Think` struct definition in CLAUDE.md:

```rust
struct Think {
    trajectory:     Vsa16kF32,
    awareness:      ParamTruths,
    free_energy:    FreeEnergy,
    resolution:     Resolution,
    episodic:       &EpisodicMemory,        // ← method, not service
    graph:          &TripletGraph,          // ← method, not service
    global_context: &Vsa16kF32,             // ← method, not service
    codec:          &CamPqCodec,            // ← method, not service
}
```

The agent/environment boundary dissolves at the carrier level — environment IS part of the struct.


---

## 11. Implementation phases (D-ids)

Twelve deliverables, sequenced. Each is one PR (some can land in parallel per the dep arrows). Estimated LOC and risk.

### Phase A — Substrate primitives (sprint-11)

| D-id | Title | Est. LOC | Risk | Deps |
|---|---|---|---|---|
| **D-CSV-1** | `causal-edge` crate v2 layout per §6 | ~250 LOC + tests | LOW | none |
| **D-CSV-2** | `QualiaI4_16D` type in `lance-graph-contract::qualia` + migration helpers (`from_f32_18d`, `to_f32_18d` for back-compat during transition) | ~180 LOC | LOW | none |
| **D-CSV-3** | `InferenceType` signed-mantissa expansion: absorb PR-LL-1 `Intervention`/`Counterfactual` from `nars_dispatch.rs` into canonical `causal_edge::InferenceType` Reserved5/6, retype as `i4` | ~120 LOC | MED (dedup risk with planner's local type) | D-CSV-1 |
| **D-CSV-4** | `CollapseGateEmission` type in `contract::collapse_gate` per §8 | ~150 LOC + 8 round-trip tests | LOW | D-CSV-1 |

### Phase B — Storage & dispatch path (sprint-11)

| D-id | Title | Est. LOC | Risk | Deps |
|---|---|---|---|---|
| **D-CSV-5** | `cognitive-shader-driver::BindSpace::QualiaColumn` migration from `[f32; 18]` to `QualiaI4_16D` | ~400 LOC (column rewrite + accessor migration + 50+ call-site touches) | HIGH (broad blast radius) | D-CSV-2 |
| **D-CSV-6** | `WitnessCorpus` (CAM-PQ-indexed) replaces `SpoWitnessChain<32>` per L-17 | ~600 LOC + benches (CAM-PQ wiring via ndarray) | HIGH (Cross-spec touchpoint with W5; CAM-PQ becomes Wave 3 hard prerequisite) | D-CSV-4 |
| **D-CSV-7** | `MailboxSoA<N>` integration: W-slot referencing + per-row plasticity accumulator + `apply_edges` for baton receipt | ~350 LOC + 9 tests (per W6 spec test target) | MED | D-CSV-1, D-CSV-4 |

### Phase C — Reasoning path (sprint-12)

| D-id | Title | Est. LOC | Risk | Deps |
|---|---|---|---|---|
| **D-CSV-8** | MUL evaluation in integer SIMD: DK position / TrustTexture / FlowState / GateDecision all consume i4 qualia + signed mantissa | ~500 LOC + bench (i4 SIMD vs f32 baseline) | MED | D-CSV-2, D-CSV-3 |
| **D-CSV-9** | 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary | ~180 LOC + 16-mapping round-trip test | LOW (math is mostly bitcast) | D-CSV-3 |
| **D-CSV-10** | Σ-tier Rubicon-resonance dispatch in `SigmaTierRouter` (per W7): F-gradient + resonance threshold → Σ10 commit, otherwise continue cycle | ~250 LOC + 12 tests | MED | D-CSV-7, D-CSV-8 |

### Phase D — Streaming infrastructure (sprint-13+)

| D-id | Title | Est. LOC | Risk | Deps |
|---|---|---|---|---|
| **D-CSV-11** | Vertical streaming structs in `ndarray`: `QualiaStream`, `InferenceStream`, `SplatFieldStream`, `par_*` rayon variants per L-20 | ~700 LOC (substantial — touches ndarray surface) | HIGH (new ndarray-level extension; coordinate with `ndarray PR #116` upstream gap) | all prior |
| **D-CSV-12** | Splat shader op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) as methods on the `Think` carrier | ~800 LOC + 4 benches | MED (consumes existing splat plan; depends on D-CSV-11 streaming) | D-CSV-11 |

**Total estimate:** ~4,480 LOC across 12 PRs, spanning sprints 11-13 (≈4-6 weeks at sustained CCA2A pace).

---

## 12. Cross-spec impact (sprint-10 worker specs to patch)

The 11 sprint-10 specs in `.claude/specs/` need targeted patches reflecting this plan's decisions. Patch sizes are small per spec.

| Spec | Patch needed | Est. LOC |
|---|---|---|
| `pr-ce64-mb-2-causaledge64-v2.md` (W2) | §3 bit layout → §6 of THIS plan; OQ-LAYOUT-1 resolved (Option C-sorted-witness + i4-mantissa + i4-qualia); add §"signed mantissa rationale" | ~150 |
| `pr-ce64-mb-2-pal8-nars-regression.md` (W3) | Tests parameterized on new v2 layout; mantissa-roundtrip + lens-state tests added | ~80 |
| `pr-ce64-mb-3-bindspace-efgh.md` (W4) | QualiaColumn migration step (D-CSV-5) referenced; AwareOp deferral note updated | ~40 |
| `pr-ce64-mb-4-arigraph-spo-g.md` (W5) | `SpoWitnessChain<32>` → `WitnessCorpus` (CAM-PQ); `W5-INV-CHAIN-ORDER` invariant added per L-16; W-slot → corpus root semantics | ~300 (significant) |
| `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) | `CompartmentReport` adds `g_slot_at_drop` field per CSI-2 (already in meta-review); MailboxSoA spatial-temporal accumulator semantics surfaced (§9) | ~50 |
| `pr-ce64-mb-6-sigma-tier-router.md` (W7) | Σ10 Rubicon-resonance threshold logic per §10; integer-SIMD MUL evaluation path per D-CSV-8 | ~120 |
| `sprint-10-pr-dep-graph.md` (W10) | PR-J1-INT4-32D-ATOMS prerequisite (CSI-3 from meta-review); D-CSV-6's CAM-PQ wiring elevated to Wave 3 hard dep | ~50 |
| `sprint-10-test-plan.md` (W11) | Test counts refreshed for v2 layout; i4-roundtrip + signed-mantissa-product tests added | ~80 |

**Total spec-patch LOC:** ~870. Bundle into one sprint-11 prep PR (`gov: sprint-10 specs patch for cognitive-substrate-convergence-v1`).

---

## 13. Risks

### 13.1 i4 quantization precision (MED)

i4 = 16 levels per dim. For continuous-feel cognitive states (e.g., fine-grained flow intensity), 16 levels may be too coarse. **Mitigations:**

1. Per-dim calibration: each of the 16 qualia dims has its own min/max mapping stored in the qualia codebook (loaded once, cache-resident).
2. Composition smooths: `i4 × i4 → i8` products give back 6-bit signed magnitude; multi-dim sums recover further.
3. Bipolar interpretation maximizes the bits: 1 valence bit + 3 intensity bits is the right shape for affective gradations (you either trust or distrust; magnitude is the within-polarity gradation).
4. Per-dim i8 fallback: if a specific dim genuinely needs >16 levels, store it as i8 in a sibling 1-byte column (paid: +1 B/row per such dim). Default to i4-16D; escalate selectively.

### 13.2 Reunification transcoder lossiness (LOW-MED)

8-channel → SPO-palette transcoder (D-CSV-9) is lossy: 7 constructive channels collapse to mantissa magnitude, sign comes from net_strength signum. **Information preserved:** direction + net magnitude + which Pearl rung (via causal_mask). **Information lost:** per-channel breakdown of which constructive operators contributed (e.g., was it CAUSES + SUPPORTS, or REFINES + GROUNDS?). For commit-tier purposes, this is acceptable — the cascade's internal 8-channel state is debugging context, not commitment substrate. Ghost-edge mechanism (per W5 spec) preserves the cascade history if reversal is needed.

### 13.3 Witness corpus unbounded growth (MED)

CAM-PQ-indexed WitnessCorpus replaces bounded `SpoWitnessChain<32>` (L-17). Growth bound: per-discourse corpus accumulates indefinitely if not pruned. **Mitigations:** salience decay via Markov ±500 window; cold-tier eviction of low-salience entries to Lance; per-tenant quota at supervisor level. Need spec: `WitnessCorpusPruningPolicy` (deferred to D-CSV-6 sub-task).

### 13.4 Downstream consumer compatibility break (HIGH)

QualiaColumn migration from `[f32; 18]` to `QualiaI4_16D` (D-CSV-5) is a **breaking ABI change** for every consumer reading the column. Affects: thinking-engine (qualia.rs, persona.rs, world_model.rs), MUL module, splat op evaluators, anyone calling `BindSpace::qualia_row()`. **Mitigation:** ship D-CSV-5 in TWO phases:

- Phase 5a: add `QualiaI4_16D` AS A SIBLING column; old `[f32; 18]` stays. Consumers opt-in via feature flag.
- Phase 5b: after all consumers migrated, remove `[f32; 18]`.

This adds 1 PR but eliminates the big-bang risk.

### 13.5 Per-tenant codebook divergence breaks family-prefix → ontology mapping (LOW today, MED future)

The "G-slot is redundant via palette family-prefix" argument (L-3) assumes tenants respect the OGIT family-prefix convention. Per-tenant custom codebooks would break this. **Mitigation:** workspace-locked codebook layout per `lance-graph-contract::manifest` (PR #366 S7-W2 sorted-slice + binary-search). Revisit at the FIRST tenant requesting custom codebook.

### 13.6 Vertical streaming structs (D-CSV-11) require ndarray PR coordination (HIGH)

D-CSV-11 extends ndarray's struct-method surface. Needs coordinated upstream merge with `AdaWorldAPI/ndarray` (the workspace's fork). Upstream gap already tracked: `ndarray PR #116` (hpc-extras) is not yet merged to ndarray:master. Adding vertical-streaming on top is sprint-13+ — coordinate with ndarray ownership.

---

## 14. Open questions for ratification (pre-sprint-11)

Six items need user "go" before D-CSV-1..7 can spawn:

| OQ # | Question | Tentative resolution | Blocking? |
|---|---|---|---|
| **OQ-CSV-1** | Per-dim qualia layout (15 named dims + 1 spare per §7.2) — is this the right assignment? | Recommend: ratify the proposed mapping with `qualia-engineer` agent cross-check + grep `crates/thinking-engine/src/qualia.rs` for existing dim assignments | BLOCKS D-CSV-2 |
| **OQ-CSV-2** | W-slot width: 6 (64 corpora) or 8 (256 corpora) bits | 6 bits = ample for single-user; 8 if production multi-tenant. Recommend 6 + bump to 8 in v3 if needed | BLOCKS D-CSV-1 |
| **OQ-CSV-3** | Spare bits (3) — reserved-for-future, or pre-allocate? | Recommend: reserved. Pre-allocating to wrong field is harder to fix than waiting | Non-blocking |
| **OQ-CSV-4** | QualiaI4_16D migration phasing: sibling-column-then-cutover (5a/5b) or big-bang (single PR)? | Recommend sibling-then-cutover (lower risk; 1 extra PR worth it) | BLOCKS D-CSV-5 |
| **OQ-CSV-5** | Pre-computed Magnitude i8 sibling column, or compute on-demand at MUL? | Recommend on-demand (1 SIMD/query is cheap; pre-compute wastes 1 B/row × 1M rows = 1 MB when most cycles don't query Magnitude) | Non-blocking |
| **OQ-CSV-6** | Σ10 Rubicon threshold value — Jirak-derived or hand-tuned? | Per `I-NOISE-FLOOR-JIRAK` iron rule: principled bound preferred. Hand-tuned acceptable for sprint-11 + 12 with TECH_DEBT note; principled derivation via VAMPE+Jirak coupled-revival in sprint-13+ | BLOCKS D-CSV-10 (sprint-12) |

### Standing user ratifications from sprint-10 (still open per meta-review)

- **CSI-1** — CausalEdge64 bit-reclaim Option (this plan resolves it via §6; locks "C-sorted-witness + i4-mantissa-expand + i4-qualia")
- **OQ-1** — Σ4-Σ5 banding (default Tokio = safe-to-ship)
- **OQ-3** — Plasticity granularity (per-edge per-plane stays; per-(role, G_slot) aggregator at dispatcher is the W6 add-on)
- **OQ-5** — Rayon vendor (std::thread::scope first; vendored-rayon to sprint-12+)

---

## 15. Test plan

Per-phase validation. Echo of `sprint-10-test-plan.md` (W11) refreshed for v2:

| Phase | Test target | Approach |
|---|---|---|
| D-CSV-1 (causal-edge v2) | New bit layout round-trip + accessor methods | 14 unit tests in `crates/causal-edge/tests/v2_layout.rs` (mantissa pack/unpack, lens 4-state, W-slot 64-handle, signed-product algebra) |
| D-CSV-2 (QualiaI4_16D) | Pack/unpack + signed multiply + per-dim calibration | 8 unit tests in `lance-graph-contract/tests/qualia_i4.rs`; round-trip f32-18D → i4-16D → f32-18D within ε per dim |
| D-CSV-3 (signed-mantissa NARS) | Mantissa direction × magnitude → existing NARS rule mapping | 10 unit tests; backward-compat shim for PR-LL-1 callers |
| D-CSV-4 (CollapseGate) | Serialization + provenance preservation + merge mode semantics | 12 unit tests including Bundle vs Xor semantic difference |
| D-CSV-5 (QualiaColumn migration) | Phase 5a parallel-column test (both columns produce same MUL output); Phase 5b cutover with downstream consumer compile-check matrix | Phase 5a: ~50 cross-consumer tests; Phase 5b: clippy --tests --no-deps -D warnings as the gate |
| D-CSV-6 (WitnessCorpus) | CAM-PQ retrieval correctness + Markov ±500 window + salience decay + corpus root anchor | 18 unit tests + 4 benches (retrieval @ 1M corpus entries < 50µs target) |
| D-CSV-7 (MailboxSoA integration) | `apply_edges` + plasticity_counter + drop_row + Hebbian rollup | 9 tests per W6 spec target |
| D-CSV-8 (MUL integer SIMD) | DkPosition + TrustTexture + FlowState + GateDecision parity with f32 baseline within ε | 16 tests + 4 benches (i4 vs f32 throughput) |
| D-CSV-9 (8ch ↔ SPO transcoder) | Round-trip ghost-edge preservation + 16-channel-pattern mapping table | 16 tests (one per cascade-channel pattern → SPO mapping) |
| D-CSV-10 (Σ-tier Rubicon dispatch) | F-gradient + resonance threshold → Σ10 commit; cycle-rest at homeostasis floor | 12 tests + property-test for "never commit on F-rising" invariant |
| D-CSV-11 (Vertical streaming) | QualiaStream / InferenceStream / SplatFieldStream forward-iter semantics | 18 tests including par_* rayon work-stealing concurrency tests |
| D-CSV-12 (Splat ops on i4) | splat_gaussian + score_hole_closure + replay_coherence + emit_if_epiphany | 14 tests + 4 benches |

**Aggregate test target:** ~150 unit tests + ~14 benches + ~20 integration tests. Miri coverage growth target: extends sprint-10-test-plan.md's 760→1550 target up to ~1900 post-D-CSV-11 (vertical streaming adds Miri scope to ndarray).

---


## 16. The convergence equation

Collapsed into one line:

```
baton  =  i4 payload  =  content  =  CAM address  =  meaning
         (precision)   (signal)    (lookup key)   (the thought)
```

Substrate-level claims that fall out:

1. **Agency is a property of the instruction-level algebra**, not a higher-level orchestration layer. The cycle's "act" is a SIMD i4 multiply propagating state in one instruction; there is no separate agency layer added on top.
2. **No homunculus**: no internal observer/decider distinct from the cycle itself. Mailboxes ARE the thought ARE the dispatch ARE the next state.
3. **No goal-misalignment failure mode**: there is no stated goal predicate that could be misaligned; only the F-gradient, which IS the goal-direction.
4. **No agent/environment boundary as a meaningful joint**: environment is wired INTO the carrier (per CLAUDE.md "tissue, not services"); they share substrate.
5. **Autopoiesis is structural**: thinking styles maintain themselves by acting to reduce their own F; no external goal needed.
6. **Philosophic entanglement is Pearl-3 coherence enforcement**: collapsing one mailbox constrains adjacent mailboxes via shared witness chain + causal_mask.
7. **Gapless cognition is the absence of encode/decode at boundaries**: baton-in baton-out, the format never changes; only the L3-commit transcoder changes representation (8ch → SPO).
8. **Time is structural, not stored**: chain-position is relative order, AriGraph anchor is absolute time, no temporal field in the edge.
9. **Three precision tiers, one algebra**: i4 (mantissa, qualia, atoms), i8 (truth, products, palette), Vsa16kI8 / Vsa16kF32 (bulk superposition inside tiers).
10. **One quantization vocabulary across the cognitive stack**: i4 signed family unifies what was scattered across f32 qualia, u8 NARS f/c, u3 inference, u3 plasticity, u3 direction, u8 palette indices.

The substrate becomes a self-coherent computational fabric. The thinking-engine + cognitive-shader-driver SoA + p64 + AriGraph all read and write the same precision family. No translation layers between subsystems. Per CLAUDE.md "The Click" §"Thinking is a struct": the struct's fields ARE the cognitive state, and the methods ARE the inference — under this plan, the fields are i4-typed and the methods are SIMD multiplies in that algebra.

---

## 17. Cross-references

### 17.1 sprint-10 work this consolidates

- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — parent plan; this v1 plan extends and locks the §3 layout questions
- `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — par-tile substrate
- `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2) — bit layout (resolved by §6)
- `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` (W3) — PAL8 regression tests
- `.claude/specs/pr-ce64-mb-3-bindspace-efgh.md` (W4) — BindSpace columns
- `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (W5) — AriGraph SPO-G + needs `WitnessCorpus` retrofit
- `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — MailboxSoA
- `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` (W7) — SigmaTierRouter + Rubicon-resonance
- `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` (W9) — bevy proof
- `.claude/specs/pr-ndarray-miri-complete.md` (W8) — Miri coverage
- `.claude/specs/sprint-10-pr-dep-graph.md` (W10) — dep graph
- `.claude/specs/sprint-10-test-plan.md` (W11) — unified test plan
- `.claude/specs/sprint-10-execution-plan.md` (W12) — sprint-11 fleet definition
- `.claude/board/sprint-log-10/meta-review.md` — Opus meta-review with CSI-1..6 + E-META-1..5

### 17.2 sprint-10 knowledge corpus this composes

- `.claude/knowledge/causal-edge-64-spo-variant.md` — SPO-palette `CausalEdge64`
- `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` — 8-channel cascade `CausalEdge64`
- `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md` — synergies + reunification options (R-1/R-2/R-3)
- `.claude/knowledge/spo-schema-and-mailbox-sidecar.md` — SPO-G vs SPO-W + sidecar framing
- `.claude/knowledge/spo-ontology-format-stack.md` — codec ladder (3×16Kbit → ZeckBF17 → Base17 → CAM-PQ → CausalEdge64)
- `.claude/knowledge/ogit-owl-dolce-ontology-compartments.md` — OGIT + OWL + DOLCE; 8-channel ↔ OWL axiom mapping
- `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md` — p64 drift origin + 5-step reunification plan (Option R-3)
- `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` — splat fleet + struct methods + computational entropy

### 17.3 Recent merged PRs this builds on

- **PR #372** (sprint-10 spec corpus) — 12 PR-ready specs + meta-review
- **PR #373** (neurosymbolic-rlvr-causal-curriculum-v1.md) — learning-layer curriculum, 5-PR roadmap
- **PR #374** (post-merge board-hygiene for #372) — PR_ARC + LATEST_STATE + EPIPHANIES E-META-7 + TYPE_DUPLICATION_MAP #13 + STATUS_BOARD
- **PR #375** (PR-LL-1) — NARS `Intervention` + `Counterfactual` variants in `nars_dispatch.rs`
- **PR #376/#378** — SMB tenant schema + tier-1 implementation spec polishes
- **PR #379** (4-branch retirement) — orphan-branch cleanup via REST API; sets retirement pattern

### 17.4 Doctrinal anchors (CLAUDE.md sections)

- **"The Click" (P-1)** — Markov ±5, role-key bind/unbind, free-energy minimization, struct-is-cognition
- **"AGI-as-glove"** — 4 BindSpace columns = AGI surface; never wrap in new struct
- **`I-SUBSTRATE-MARKOV`** iron rule — VSA-bundling guarantees Chapman-Kolmogorov; XOR breaks it
- **`I-NOISE-FLOOR-JIRAK`** iron rule — weak-dependence Berry-Esseen for σ-thresholds; principled vs hand-tuned
- **`I-VSA-IDENTITIES`** iron rule — VSA on identity fingerprints (this plan: i4 IS both identity AND content)
- **"The shader can't resist the thinking"** — active inference dispatch driver, "can't stop thinking"
- **"Thinking is a struct"** — Think DTO carries cognition; methods on tissue
- **Litmus tests** — free function = reject, method = accept; lens-aligned = accept

### 17.5 External / forthcoming

- **arxiv 2505.04646v1** — "Nature of agency" (referenced live by user; this plan's §10 active-inference framing aligns; deeper integration deferred to post-plan-merge follow-up)
- **`AdaWorldAPI/ndarray PR #116`** — `hpc-extras` upstream gap; coordinate with D-CSV-11 vertical streaming

### 17.6 Board files this plan triggers (per Mandatory Board-Hygiene Rule, in the SAME commit)

- `.claude/board/INTEGRATION_PLANS.md` PREPEND entry (this plan + 12 D-ids)
- `.claude/plans/cognitive-substrate-convergence-v1.md` (THIS FILE)
- `.claude/board/STATUS_BOARD.md` PREPEND a new section for `cognitive-substrate-convergence-v1` plan with the 12 D-CSV-* rows (`Queued`, with per-row blockers)

These three files MUST land in the same commit per CLAUDE.md doctrine.

### 17.7 Cross-spec patches (separate sprint-11 prep PR)

8 spec patches per §12. Bundle into `gov: sprint-10 specs patch for cognitive-substrate-convergence-v1`. Total ~870 LOC.

---

## 18. Status

| Field | Value |
|---|---|
| **Status** | PROPOSAL — awaits user ratification of OQ-CSV-1..6 |
| **Confidence (2026-05-15)** | HIGH on architecture; MED on i4-16D qualia dim assignment (OQ-CSV-1); HIGH on i4-mantissa NARS (PR-LL-1 already shipped equivalent); HIGH on gapless-baton model (PR #375 confirmed via Sprint A working); MED on Rubicon-resonance threshold (OQ-CSV-6 — needs Jirak derivation) |
| **Branch** | `claude/cognitive-substrate-convergence-plan-v1` |
| **Successor** | None (this is v1) |
| **Replaces** | parts of CSI-1 resolution in `.claude/board/sprint-log-10/meta-review.md` — this plan's §6 is the definitive bit-layout answer |

---

*End of cognitive-substrate-convergence-v1.md. Authored 2026-05-15 during live cross-session A2A architectural discussion. Single canonical reference for sprint-11+ implementation; locks the architectural decisions before context dilution.*
