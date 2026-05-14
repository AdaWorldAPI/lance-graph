# CausalEdge64 — Synergies, Trade-offs, and PR Trajectory

> **READ BY:** integration-lead, truth-architect, anyone planning sprint-10+ CausalEdge64 v2 work
>
> **PREREQUISITES:** read `causal-edge-64-spo-variant.md` and `causal-edge-64-thinking-engine-variant.md` first; this doc compares them.
>
> **Status:** FINDING (verified 2026-05-14) + CONJECTURE on reunification path

---

## 1. The Two Invariants at a Glance

| Aspect | SPO-Palette Variant (`causal-edge` crate) | 8-Channel Cascade Variant (`thinking-engine` crate) |
|---|---|---|
| **u64 contents** | Self-addressing: (S, P, O, f, c, mask, dir, infer, plast, t) | Pure strength vector: 8 channels × 8 bits each |
| **Addressing** | IN the u64 (S/P/O indices) | OUT of the u64: `(target: u16, edge: CausalEdge64)` tuple |
| **Truth model** | NARS frequency + confidence inline | No truth — pure energy perturbation |
| **Causal model** | Pearl 2³ mask + 5 NARS rules | 7 constructive + 1 destructive channels |
| **Update semantics** | `forward()` composes via palette tables; `learn()` revises | `apply_edges()` adds energy; tier re-normalizes |
| **Primary producer** | NarsTables compose, AriGraph→SPO promotion, palette dispatch | ThinkingEngine top-k after MatVec |
| **Primary consumer** | Planner cache, BindSpace::EdgeColumn, cognitive-shader-driver | Downstream TierEngine, AriGraph orchestrator, AttentionEdges log |
| **Hot-path tier** | Zone-1 (SoA sweep, ~50-200 ns/row) | Zone-1 (200-500 ns MatVec emission + ~20-200 ns/edge consumption) |
| **Persistence** | Promotes to AriGraph SPO via `spo_bridge::promote_to_spo()` | Logged in `ContextBlackboard.attention_edges: Vec<u64>` |

---

## 2. What Each Does BETTER

### 2.1 SPO-Palette Variant — what it does better

**Self-contained statements.** The u64 IS a complete causal proposition. Hash it, store it, retrieve it — no surrounding tuple key needed. This makes it the canonical wire format for AriGraph promotion, p64 cache hot lookups, and CAM-PQ-style indexed retrieval.

**Pearl's ladder explicit.** The 3-bit causal mask is a precise Pearl-rung indicator. Predicate-pushdown query optimization can filter by `matches_causal(0b011)` (= Level-2 interventional) in a single bitmask AND. This is load-bearing for the planner's strategy selection.

**NARS truth grounded.** The f/c pair plus inference type means revision behavior is unambiguous — given two edges with the same (S, P, O), `Revision` semantics merge them exactly per Wang 2013 NAL formulas. The 8-channel variant has no truth values, only strengths.

**Composition primitive.** `forward()` is the **BNN-style inference step** — three 256×256 palette table lookups + one NARS truth update = O(1) per edge. This is what makes the p64 hot path possible (per `cache/convergence.rs`: "4096 heads → CausalEdge64 forward/learn = O(1) per head").

**Per-plane plasticity.** Three independent plasticity bits let the system say "S-archetype is stable, but O-archetype is still under revision" — finer than a global "uncertain" flag.

### 2.2 8-Channel Cascade Variant — what it does better

**Multi-channel simultaneity.** A single edge can carry CAUSES + SUPPORTS + REFINES + CONTRADICTS all non-zero at once. This expresses **emotionally compound** causal relationships ("I cause this, support it, AND have some lingering doubt"). The SPO variant's single 3-bit InferenceType slot can name only one rule.

**Energy-additive composition.** `apply_edges()` adds strengths directly to an energy vector and re-normalizes. No truth-revision arithmetic. This makes it the right primitive for **cascade routing** — moving attention/energy through a tier hierarchy without committing to specific facts.

**Decoupled addressing.** Because source/target live in the tuple key, the same `CausalEdge64` channel-pattern can be applied to ANY (source, target) pair. This makes it composable across tiers with different vocabularies (L1 = 64-atom routing tier; L2 = 256-atom role tier; L3 = 4096-atom COCA tier).

**Naturally aligned to subjective channels.** BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS map cleanly to **cognitive operators** the system actually uses in language understanding and inference. The SPO variant's "direction triad" is much more abstract (sign(dim0) per plane).

**Cycle-speed emission and consumption.** Both producer (`emit_causal_edges`) and consumer (`apply_edges`) are pure SIMD/register operations on the strength bytes. No table lookups, no palette composition, no truth revision math. Fastest possible cycle-speed path.

---

## 3. Mapping — What Functions Each Has in the Thinking-Engine

Per `crates/thinking-engine/src/` source roster and CLAUDE.md "Thinking Engine":

| Function area | SPO variant role | 8-channel variant role |
|---|---|---|
| **`engine.rs`** (core MatVec, `think()`) | Not used directly | Internal: drives energy vector evolution |
| **`layered.rs`** (3-tier cascade L1→L2→L3) | Not present | **DEFINED HERE.** All emit/apply happens in this module |
| **`bridge.rs` / `l4_bridge.rs`** (downstream egress) | The SPO variant is what gets emitted as the **canonical fact** when L3 commits a thought to AriGraph | The 8-channel variant is the **interior dispatch language** that drove the L1→L2→L3 cascade leading up to commit |
| **`cognitive_stack.rs`** (working memory) | Stored as `pending_triplets: Vec<Triplet>` after promotion | Stored as `attention_edges: Vec<u64>` log during cascade |
| **`ghosts.rs`** (alternative-world tracking) | Each ghost = a CausalEdge64 (SPO variant) with `causal_mask = SPO` (Pearl-3 counterfactual flag) | Each ghost-cascade run produces 8-channel edges; CONTRADICTS channel marks rejection paths |
| **`qualia.rs` / `world_model.rs`** (subjective state) | Read SPO-variant truth values to grade qualia (e.g., Wisdom × Staunen) | Read 8-channel energy levels to grade resonance/contradiction depth |
| **`persona.rs`** (style modulation) | Modulates inference_type selection in `forward()` | Modulates which channels get amplified during `apply_edges` (e.g., persona "Skeptic" amplifies CONTRADICTS) |
| **`sensors/jina_lens.rs`, `bge_m3_lens.rs`, `reranker_lens.rs`** | Produce embeddings that become SPO-variant edges via palette quantization | Produce energy injections that become 8-channel edges via `emit_causal_edges` |

The split is: **8-channel = interior dispatch grammar; SPO = exterior commit grammar.** The cascade reasons internally with the 8-channel form; when a thought commits to AriGraph or planner cache, it crystallizes as an SPO-variant edge.

---

## 4. Bit-Reclaim Trade Analysis (Sprint-10 CausalEdge64 v2)

Sprint-10's `causaledge64-mailbox-rename-soa-v1` plan targets the **SPO-variant only**. Recommendation evolution across this session:

### 4.1 First framing (wrong-ish) — "drop redundant fields"

Initial proposal: drop temporal + direction + inference (= 18 bits) because:
- temporal duplicated by AriGraph timestamp
- direction was a cache of palette `dim0.sign()`
- inference movable to AttentionMask::style_slots

**Why it was wrong:** the Explore agent's hot-path mapping showed `forward()` propagates direction through composition (`edge.rs:457` TODO confirms it's load-bearing, not derived); plasticity per-plane is finer than confidence; the bits ARE the cycle-speed dispatch payload, not relocatable metadata.

### 4.2 Second framing (better) — "drop only truly redundant"

Conservative reclaim: drop **temporal (12 bits)** only.
- Chain-position in `SpoWitnessChain` gives relative order (per the "sort witness by time" insight).
- AriGraph `Triplet.timestamp: u64` at the chain root gives absolute time.
- The 12-bit cycle bucket is the only field with NO load-bearing inference semantics — it's a cache, not state.

### 4.3 Third framing (current) — "drop temporal + G-slot redundant via witness corpus"

Drop temporal (12 b) + G_slot-would-have-been-new (5 b) = 17 b freed total.
- G_slot is implicit in the witness-corpus root's `domain` field — per the per-tenant SoA architecture (W7 SigmaTierRouter is per-tenant), all edges in one SoA share a tenant; cross-domain reasoning goes through corpus traversal at supervisor/AriGraph level, not at edge level.

### 4.4 Allocate 17 freed bits

| New field | Bits | Purpose |
|---|---|---|
| **W slot** | 6-8 | Discourse-corpus handle: 64-256 active discourse roots (CAM-PQ-indexed witness corpus per `spo-ontology-format-stack.md`) |
| **Truth-band lens** | 2 | 4 states incl. "13% ambiguous direction" (per the user's "transfer it forcefully into causality even if 13% are still unclear" insight) |
| **Spare** | 7-9 | Honest headroom for sprint-12+ probes |

### 4.5 The 8-channel variant — UNTOUCHED by v2

The 8-channel cascade variant has zero unused bits already (8 channels × 8 = 64 bits, full). It cannot gain G-slot, W-slot, or truth-band lens without removing channels. **Either the v2 evolution is SPO-variant-only, or both variants must converge into a wider unified representation** (see §6 reunification).

---

## 5. Last PR Trajectory — What the Recent Work Was Acting On

From `.claude/board/PR_ARC_INVENTORY.md` (top of file, 2026-05-13 most-recent):

### PR #366 — Sprint-7 implementation wave (merged 2026-05-13)

**The single most consequential recent landing for CausalEdge64.** Wired together:

- **S7-W5** `pr-f1-thinking-engine-wire`: `CognitiveBridgeGate` trait in `thinking-engine` + `UnifiedBridgeGate<B: NamespaceBridge>` in `lance-graph-callcenter`. **This created the security gate that fires before any CausalEdge64 (8-channel variant) crosses tenant boundaries.** Chinese-wall check on `tenant_id` mismatch before policy. No circular dep (callcenter → thinking-engine only). 329 thinking-engine + 114 callcenter + 12 new gate tests.
- **S7-W3** `pr-g2-ractor-supervisor`: new crate `lance-graph-supervisor` with `CallcenterSupervisor` (one-for-one supervision, exponential backoff). **This is the ractor topology that hosts the mailboxes that carry CausalEdge64 dispatches** (sprint-10 will extend this with `SigmaTierRouter` per W7 spec).
- **S7-W2** `pr-g1-manifest-modules`: zero-dep YAML→Rust codegen for consumer manifests. **This is the contract layer that consumers of CausalEdge64 use to know which domain a tenant lives in** (medcare / smb-office / q2-cockpit / fma / hubspot).

**Status of CausalEdge64 work after #366:** the 8-channel variant has its security gate; the ractor topology is ready to dispatch edges between actors; the contract crate knows what tenants look like. **What was NOT done:** unifying the two CausalEdge64 variants; defining the v2 SPO-variant layout; wiring the SPOW tetrahedron W-slot.

### PR #365 — Sprint-5-6 specs (merged 2026-05-13)

13 PR-ready specs (~300 KB) covering MedCare super-domain, woa-rs extraction, conformance harness, OGIT/SMB TTL hydration. **Set up the spec corpus that PR #366 implemented.** Did not directly touch CausalEdge64 layout.

### PR #364 (referenced, not in the recent inventory section)

`ndarray = features = ["hpc-extras"]` switchover (referenced by MedCare#118 cross-ref). **Made the per-plane palette distance tables available** (`SpoDistanceMatrices` in `ndarray::hpc::palette_distance`). The p64 convergence (`cache/convergence.rs::PlaneDistance`) depends on this — without it, the SPO-variant's `distance_masked()` path falls back to bit-ops.

### Trajectory Summary

The recent PRs (#364, #365, #366) collectively **built the integration scaffolding around CausalEdge64**:
- Hot-path palette distance tables (#364)
- Spec corpus for sprint-7 (#365)
- Ractor topology + cognitive bridge gate + manifest contracts (#366)

What they did NOT do:
- Re-unify the two CausalEdge64 variants
- Define the SPO-variant v2 bit layout
- Wire the SPOW tetrahedron W-slot
- Connect thinking-engine 8-channel emissions to BindSpace::EdgeColumn (which expects the SPO variant)

These are the gaps that sprint-10 (`causaledge64-mailbox-rename-soa-v1`) was opened to address — and where the cross-spec consistency issue (CSI-1) surfaced.

---

## 6. Reunification Path (CONJECTURE)

The p64 convergence comment at `cache/convergence.rs:1-23` declares:

> *"p64 IS the bridge between them. Cold path BUILDS the graph; Hot path SERVES the graph."*

The implicit promise: **one CausalEdge64 that lives at the bridge — both addressing-rich and channel-rich.**

Three proposed reunification options:

### Option R-1: SPO-variant absorbs 8 channels into u128

```text
CausalEdge128 {
    u64 spo_bits:      same as current SPO-variant
    u64 channel_bits:  8 channels × 8 bits = current 8-channel variant
}
```

- **Pro:** zero information loss. Both variants are preserved bit-for-bit. Two u64 register reads per edge (cache-line aligned).
- **Con:** doubles edge memory; halves SoA throughput per-row; breaks current cache-line layout in BindSpace::EdgeColumn.

### Option R-2: 8-channel-variant absorbs SPO via paired tuple

```text
(SpoAddress, CausalEdge64_8ch)
where SpoAddress = u32 { s_idx: u8, p_idx: u8, o_idx: u8, pearl_rung: u8 }
```

- **Pro:** addressing externalized (matches current 8-channel design with `(target, edge)` tuples); SoA structure-of-arrays-friendly.
- **Con:** loses NARS f/c inline; loses direction/plasticity. Major API break for planner cache.

### Option R-3 ⭐ (recommended): Strict tier separation + lens

```text
Cascade tier (thinking-engine):  uses 8-channel variant unchanged
Commit tier (causal-edge):       uses SPO-palette variant with W-slot, lens
Bridge: thinking-engine's 8-channel edges, on L3-commit, are **transcoded** to
        SPO-variant edges via a deterministic mapping:
            CAUSES → InferenceType::Deduction
            SUPPORTS → InferenceType::Revision (positive)
            CONTRADICTS → InferenceType::Revision (negative)
            REFINES → InferenceType::Abduction
            ABSTRACTS → InferenceType::Induction
            BECOMES / GROUNDS / RELATES → InferenceType::Synthesis with channel-strength → f/c mapping
```

- **Pro:** each variant stays optimal for its tier; no register width change; the transcoding is the natural commit-point (matches per `cache/convergence.rs` "hot path serves the graph" doctrine).
- **Con:** information loss in the 8-channel → SPO transcoding (cannot reverse without ghost-edge preservation).

**Where the drift originally happened:** thinking-engine evolved its cascade primitive (8-channel) independently of causal-edge's NARS/Pearl primitive (SPO). The p64 convergence work bridged them through `nars_engine.rs`'s `CausalEdge64` import (the SPO variant) but **the 8-channel variant was never imported back into the planner** — it lives in thinking-engine and is consumed by AriGraph orchestrator without ever going through the planner's `forward()` path.

**Where reunification roots:** the `p64::convergence` doctrine ("4096 heads → CausalEdge64 forward/learn = O(1) per head") implies a single CausalEdge64 at the bridge. Today that singular bridge-edge is **the SPO variant**; the 8-channel variant exists as an upstream-only dispatch artifact that doesn't survive into the persistent cache. Option R-3 formalizes this: the 8-channel variant has a **lifetime bounded by the cognitive cycle**; after commit, only SPO-variant edges persist.

See `cognitive-shader-driver-thinking-engine-reunification.md` for the full reunification plan.

---

## 7. Recommendation for Sprint-10 CSI-1 Resolution

Given the corrected hot-path analysis and the two-variants finding:

1. **Sprint-10 v2 work targets the SPO-palette variant only.** The 8-channel variant stays at u64 with 8 channels.
2. **Bit-reclaim:** drop temporal (12 b) + drop G-slot-would-have-been-new (5 b) = 17 b freed.
3. **Allocate:** W-slot (6-8 b for discourse corpus) + truth-band lens (2 b for ambiguity expressivity) + spare (7-9 b).
4. **Cross-variant invariant:** the 8-channel variant remains the **interior dispatch language** of thinking-engine; the SPO variant remains the **exterior commit language** of AriGraph/planner cache. The transcoding step at L3-commit (Option R-3) is the canonical bridge.
5. **Add type-duplication entry:** `TYPE_DUPLICATION_MAP.md` should list `CausalEdge64 (2 copies — different semantics)` so future sessions don't conflate them.
6. **PREPEND to `EPIPHANIES.md`** the dual-CausalEdge64 discovery as E-META-7 (or whatever number is next) so it survives context resets.

---

## 8. Cross-references

- `causal-edge-64-spo-variant.md` — the SPO-palette variant in detail
- `causal-edge-64-thinking-engine-variant.md` — the 8-channel cascade variant in detail
- `cognitive-shader-driver-thinking-engine-reunification.md` — the full reunification plan + drift origin analysis
- `spo-schema-and-mailbox-sidecar.md` — mailbox-level implications of which variant travels in messages
- `spo-ontology-format-stack.md` — how SPO storage formats (3×16Kbit / CAM-PQ / bgz17 / bgz-hhtl-d) relate to each variant
- `lab-vs-canonical-surface.md` — AGI-as-glove doctrine; EdgeColumn = CausalEdge64
- `encoding-ecosystem.md` — codec stack context
- `.claude/board/PR_ARC_INVENTORY.md` — PR #366 (Sprint-7 wave), #365 (specs), #364 (ndarray hpc-extras)
- `.claude/board/sprint-log-10/meta-review.md` — CSI-1 (the central plan/code-gap finding)
- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — sprint-10 parent plan
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` — SPOW tetrahedron source
- `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md` — splat shader vision

---

*Last verified: 2026-05-14. Trajectory section reflects PR_ARC_INVENTORY state as of merge of PR #366.*
