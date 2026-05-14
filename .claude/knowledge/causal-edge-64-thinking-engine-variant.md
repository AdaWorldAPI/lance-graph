# CausalEdge64 — 8-Channel Cascade Variant (thinking-engine crate)

> **READ BY:** integration-lead, truth-architect, bus-compiler, anyone touching `thinking-engine::layered` / `CognitiveBridgeGate` / cognitive cascade L1→L2→L3 routing
>
> **PAIRED WITH:** `causal-edge-64-spo-variant.md` (the OTHER `CausalEdge64` type — same name, different bit layout); cross-comparison in `causal-edge-64-synergies-and-pr-trajectory.md`
>
> **Status:** FINDING (verified against shipped source 2026-05-14)

---

## 1. Identity

**Type:** `thinking_engine::layered::CausalEdge64` (the "8-channel cascade variant", "cognitive-cascade variant", "BECOMES-edge").

**Definition site:** `crates/thinking-engine/src/layered.rs:45`

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalEdge64(pub u64);
```

**Crate:** `thinking-engine` (in `crates/thinking-engine/` per CLAUDE.md "Thinking Engine"; standalone-ish, depended on by `lance-graph-callcenter` via `CognitiveBridgeGate`).

**One-line role:** the 64-bit **dispatch payload** emitted by `ThinkingEngine` after MatVec — encodes 7 constructive + 1 destructive energy channels that perturb downstream tier engines or modify AriGraph entity state.

**Doctrine line** (from `layered.rs:1-11`):
> *"Layered thinking cascade with CausalEdge64 upstream propagation. Three tiers of ThinkingEngine, connected by causal edges: L1 (small, routing) → L2 (mid, role resonance) → L3 (full thought). CausalEdge64 packs 8 channels (7 constructive + 1 destructive) into a u64. Each channel is one byte (0-255). Constructive channels add energy; the CONTRADICTS channel subtracts energy."*

---

## 2. Bit Layout

From `layered.rs:33-46`:

```text
[ 0:  7]  channel 0 (BECOMES)        u8   (transformative resonance)
[ 8: 15]  channel 1 (CAUSES)         u8   (forward causation)
[16: 23]  channel 2 (SUPPORTS)       u8   (corroborative evidence)
[24: 31]  channel 3 (REFINES)        u8   (specialization / refinement)
[32: 39]  channel 4 (GROUNDS)        u8   (foundational basis)
[40: 47]  channel 5 (ABSTRACTS)      u8   (generalization upward)
[48: 55]  channel 6 (RELATES)        u8   (lateral semantic relation)
[56: 63]  channel 7 (CONTRADICTS)    u8   (destructive / refutation)

Total                                64b  (zero unused; 8 bytes; little-endian)
```

**Channel constants** (`layered.rs:23-30`):

```rust
pub const CHANNEL_BECOMES:     u8 = 0;
pub const CHANNEL_CAUSES:      u8 = 1;
pub const CHANNEL_SUPPORTS:    u8 = 2;
pub const CHANNEL_REFINES:     u8 = 3;
pub const CHANNEL_GROUNDS:     u8 = 4;
pub const CHANNEL_ABSTRACTS:   u8 = 5;
pub const CHANNEL_RELATES:     u8 = 6;
pub const CHANNEL_CONTRADICTS: u8 = 7;
```

---

## 3. Critical Distinction — Source/Target NOT in u64

From `layered.rs:88-89`:

> *"Source and target are NOT stored inside the u64 (all 64 bits are channels).
> They are carried alongside as the tuple key `(u16, CausalEdge64)`."*

This is a **fundamental contrast** with the SPO-palette variant:

| Variant | What the u64 encodes | Where the addressing lives |
|---|---|---|
| **SPO-palette** (`causal_edge` crate) | (S, P, O, f, c, mask, dir, infer, plast, t) — addressing IS in the u64 | Self-addressing via palette indices |
| **8-channel cascade** (`thinking-engine` crate) | (BECOMES, CAUSES, SUPPORTS, REFINES, GROUNDS, ABSTRACTS, RELATES, CONTRADICTS) — only the strengths | Addressing is external: `(target: u16, edge: CausalEdge64)` tuple |

So the 8-channel variant is **a strength vector**, not a self-contained statement. The target u16 (the atom-index in the receiving tier) is carried separately. Source is implicit from the emitting tier engine.

---

## 4. Key Methods

### 4.1 Channel I/O

`layered.rs:50-70`:

```rust
pub fn new() -> Self                              { CausalEdge64(0) }
pub fn set_channel(&mut self, channel: u8, value: u8)
pub fn get_channel(&self, channel: u8) -> u8
```

Each channel is 1 byte; shift = `channel * 8`. Little-endian byte order within the u64.

### 4.2 Aggregate strengths

`layered.rs:73-85`:

```rust
pub fn constructive_strength(&self) -> u16  // sum of channels 0..=6
pub fn contradiction_strength(&self) -> u8  // channel 7 only
pub fn net_strength(&self) -> i16           // constructive − destructive (can be negative)
```

### 4.3 Convenience constructor

`layered.rs:87-94`:

```rust
pub fn with_source_target(_source: u16, _target: u16, strength: u8) -> Self {
    let mut e = Self::new();
    e.set_channel(CHANNEL_CAUSES, strength);
    e
}
```

Note the underscore prefixes — source/target args are accepted but discarded (they live in the tuple key, not the u64). Strength is set on the CAUSES channel by default.

---

## 5. The Three-Tier Cascade

`layered.rs:9-11` + `TierEngine` struct at `layered.rs:108`:

```text
L1 (small, routing)
   distance_table N×N (small N, e.g., 64)
   MatVec → top-k atoms
   emit_causal_edges() → Vec<(u16 target, CausalEdge64)>
      ↓ upstream propagation
L2 (mid, role resonance)
   distance_table M×M (mid M, e.g., 256)
   apply_edges(from_L1) → energy perturbation
   MatVec → top-k atoms
   emit_causal_edges() → Vec<(u16 target, CausalEdge64)>
      ↓ upstream propagation
L3 (full thought)
   distance_table 4096×4096 (full COCA vocabulary)
   apply_edges(from_L2) → energy perturbation
   MatVec → top-k atoms
   final ThoughtResult
```

Each `TierEngine` (`layered.rs:108-170+`):

```rust
pub struct TierEngine {
    engine: ThinkingEngine,
    distance_table: Vec<u8>,        // N×N shadow copy
    tier_name: String,
    size: usize,                    // N
}

impl TierEngine {
    pub fn new(distance_table: Vec<u8>, name: &str) -> Self
    pub fn think(&mut self, max_cycles: usize)
    pub fn top_k(&self, k: usize) -> Vec<(u16, f32)>
    pub fn emit_causal_edges(&self, k: usize) -> Vec<(u16, CausalEdge64)>
    pub fn apply_edges(&mut self, edges: &[(u16, CausalEdge64)])
    pub fn reset(&mut self)
    pub fn size(&self) -> usize
    pub fn perturb(&mut self, indices: &[u16])
    pub fn engine(&self) -> &ThinkingEngine
}
```

---

## 6. `emit_causal_edges()` — the Producer

`layered.rs:165-203`:

```rust
pub fn emit_causal_edges(&self, k: usize) -> Vec<(u16, CausalEdge64)> {
    let peaks = self.top_k(k);
    let mut edges = Vec::new();

    for &(peak_idx, peak_energy) in &peaks {
        if peak_energy < 1e-15 { continue; }

        let pi = peak_idx as usize;
        let row_offset = pi * self.size;

        // Collect (neighbor_index, similarity) excluding self.
        let mut neighbors: Vec<(usize, u8)> = (0..self.size)
            .filter(|&j| j != pi)
            .map(|j| (j, self.distance_table[row_offset + j]))
            .collect();

        // Sort by similarity descending to find nearest neighbors.
        neighbors.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top 4 neighbors.
        for &(neighbor_idx, sim) in neighbors.iter().take(4) {
            let strength = ((sim as f32 / 255.0) * peak_energy * 255.0)
                .round().clamp(0.0, 255.0) as u8;
            if strength == 0 { continue; }
            let mut edge = CausalEdge64::new();
            edge.set_channel(CHANNEL_CAUSES, strength);
            edges.push((neighbor_idx as u16, edge));
        }
    }
    edges
}
```

**Algorithm:** for each of the top-k MatVec peaks, find its top-4 nearest neighbors in the distance table, emit a CAUSES-channel edge to each, strength proportional to `(similarity / 255) × peak_energy × 255`.

**Output cardinality:** `up to k × 4` edges per emission. With k = 8 and 4 neighbors each = 32 edges per cycle.

---

## 7. `apply_edges()` — the Consumer

`layered.rs:206-225`:

```rust
pub fn apply_edges(&mut self, edges: &[(u16, CausalEdge64)]) {
    for &(target, edge) in edges {
        let idx = target as usize;
        if idx >= self.size { continue; }
        let net = edge.net_strength();          // constructive − destructive
        let delta = net as f32 / 255.0;
        self.engine.energy[idx] += delta;
        if self.engine.energy[idx] < 0.0 {
            self.engine.energy[idx] = 0.0;
        }
    }
    // Re-normalize.
    let total: f32 = self.engine.energy.iter().sum();
    if total > 1e-15 {
        for e in &mut self.engine.energy {
            *e /= total;
        }
    }
}
```

**Algorithm:** for each incoming `(target, edge)`, compute `net_strength()`, scale to f32, add to `energy[target]`, clamp at zero floor, re-normalize the full energy vector.

This is **energy perturbation** — the cognitive cascade's mechanism for upstream tiers to steer downstream tiers' thinking by injecting/subtracting energy at specific atom indices.

---

## 8. Hot-Path Role

**Zone-1 (cycle-speed):** YES — this IS the dispatch payload of the cognitive cycle.

The end-to-end cycle (per the Explore agent's mapping):

```text
text → tokenizer → codebook index → perturb energy vector
    ↓
ThinkingEngine.think(N cycles):  distance_table × energy → energy_next  (MatVec)
    Current:  F32x16 SIMD, ~500 ns per cycle on Zen 4
    Future:   AMX TDPBUSD, ~200 ns per cycle (1 cycle for 256 MACs)
    ↓
top_k(k=8) atoms identified
    ↓
emit_causal_edges(8) → Vec<(u16 target, CausalEdge64)>  (~32 edges)
    ↓
AriGraph orchestrator receives edges
    For each (target, edge):
        HashMap.get(entity_index[target_name])  — O(1), ~20-200 ns
        Read channels (BECOMES / CAUSES / etc.) — direct register access
        Update Triplet (revise / add / delete)
    ↓
(Optional) promote_to_spo(triplet) if truth confidence passes gate
```

**Total cycle budget:** ~500 ns (MatVec) + ~32 × 200 ns (entity_index lookups) ≈ **6 µs per full cognitive cycle** — well under the ~17 KB/s thought throughput target per CLAUDE.md.

**The edge channels are read AT the entity_index lookup.** Direction, plasticity, inference equivalents (if mapped) travel WITH the edge from MatVec emission to AriGraph consumption. There is no separate metadata fetch.

---

## 9. Channel Semantics — Cognitive Operators

The 7 constructive channels + 1 destructive channel form a small **operator algebra** over thought-atom relationships:

| Channel | Operator | Semantic role |
|---|---|---|
| **BECOMES** (0) | transformative resonance | "this atom is becoming that atom" — identity-shift |
| **CAUSES** (1) | forward causation | "this atom causes activation of that atom" — directed flow |
| **SUPPORTS** (2) | corroborative evidence | "this atom supports the truth of that atom" — NARS revision direction |
| **REFINES** (3) | specialization | "this atom refines that atom" — taxonomic descent |
| **GROUNDS** (4) | foundational basis | "this atom grounds that atom" — abductive justification |
| **ABSTRACTS** (5) | generalization upward | "this atom abstracts to that atom" — taxonomic ascent |
| **RELATES** (6) | lateral relation | "this atom relates to that atom" — semantic neighborhood |
| **CONTRADICTS** (7) | destructive refutation | "this atom contradicts that atom" — energy subtracts |

Each channel can independently carry a strength of 0-255. **A single edge can carry multiple channels active simultaneously** — e.g., an edge with `CAUSES = 200, SUPPORTS = 100, CONTRADICTS = 50` says "I cause this strongly, I support it medium, but I also have some contradicting force."

This is **richer** than a single InferenceType enum (the SPO-palette variant's 3-bit inference field, which can only name one rule at a time). The cost is that the 8-channel variant has no SPO addressing of its own — addressing must come from the tuple key or the surrounding context.

---

## 10. Consumers

### 10.1 Cascade tier engines

The primary consumer is `TierEngine::apply_edges()` itself. Edges emitted by tier N flow into `apply_edges()` of tier N+1, perturbing its energy vector before its next `think()` cycle.

### 10.2 `CognitiveBridgeGate` (sprint-7, PR #366)

Per the Sprint-7 wiring (PR #366):

> *S7-W5 `pr-f1-thinking-engine-wire` — `CognitiveBridgeGate` trait in `thinking-engine` + `UnifiedBridgeGate<B: NamespaceBridge>` impl in `lance-graph-callcenter`. Chinese-wall check fires before policy on `tenant_id` mismatch.*

The `CognitiveBridgeGate` trait sits in thinking-engine and provides the surface that `lance-graph-callcenter`'s `UnifiedBridgeGate<B>` implements. Tenant isolation gating runs **before** any CausalEdge64 emission crosses into another tenant's working memory. The 8-channel CausalEdge64 is the payload that this gate authorizes.

### 10.3 `lance-graph::graph::arigraph::orchestrator`

`ContextBlackboard` (per the Explore agent's mapping) holds:

```rust
pub attention_edges: Vec<u64>,    // CausalEdge64 attention log (last inference pass)
pub graph_context:   Vec<String>, // Retrieved graph context (formatted triplets)
pub pending_triplets: Vec<Triplet>, // From LLM extraction
```

The orchestrator reads `attention_edges`, dereferences each edge's target via `entity_index`, and updates AriGraph triplets accordingly.

---

## 11. Cross-references

- **Paired with:** `causal-edge-64-spo-variant.md` (the OTHER `CausalEdge64` type — same name, different bit layout, different consumers).
- **Synergies + drift analysis:** `causal-edge-64-synergies-and-pr-trajectory.md`.
- **AGI-as-glove doctrine:** `lab-vs-canonical-surface.md` (the 4 BindSpace columns; CausalEdge64 = EdgeColumn).
- **Sprint-7 wiring:** PR #366 entry in `PR_ARC_INVENTORY.md` (CognitiveBridgeGate trait, UnifiedBridgeGate).
- **Thinking-engine roster:** CLAUDE.md "Thinking Engine (crates/thinking-engine/)" — full module list.
- **Three-zone hot-path mental model:** `cognitive-shader-driver-thinking-engine-reunification.md` (this knowledge doc set).
- **`emit_causal_edges` callers:** `thinking-engine/src/layered.rs` examples + downstream `dispatcher` integration points.

---

## 12. Open Questions This Variant Raises for Sprint-10 v2 Work

The sprint-10 `causaledge64-mailbox-rename-soa-v1` plan targets the **SPO-palette variant** for bit-reclaim. **This variant is not in scope.** That's a real consistency issue:

1. **Two CausalEdge64 types with the same name** but different semantics is a footgun. The workspace's `TYPE_DUPLICATION_MAP.md` does not list this duplication.
2. **The plan §3 layout discussion implicitly assumes "the" CausalEdge64** — which one?
3. **If the SPO-palette variant gains G-slot, W-slot, truth-band lens** (per CSI-1 in `.claude/board/sprint-log-10/meta-review.md`), the 8-channel variant **does not** automatically gain corresponding fields. Should it?
4. **Reunification path** — see `cognitive-shader-driver-thinking-engine-reunification.md` for the proposed convergence of both variants into a unified `CausalEdge64` that carries (SPO addressing) + (8 channels) + (Pearl rung) in a wider register, or a paired-register pattern.

---

*Last verified: 2026-05-14 against shipped `crates/thinking-engine/src/layered.rs`.*
