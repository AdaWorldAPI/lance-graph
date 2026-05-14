# PR-CE64-MB-4 — AriGraph SPO-G Quad Upgrade + Ghost-Edge Persistence + SpoWitnessChain

> **Status:** Spec (2026-05-14) — sprint-log-10 W5 output
> **Scope deliverable:** D-OGIT-G-1 (SPO-G quad) + ghost-edge persistence + SpoWitness64 + SpoWitnessChain<N>
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 (lance-graph::arigraph row) + §7 (PR-CE64-MB-4 entry)
> **Primary references:**
> - `.claude/plans/ogit-g-context-bundle-v1.md` §D-OGIT-G-1 — SPO-G u32 slot spec
> - `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1 §2 §3 §8 §9
> **Depends on:** PR-CE64-MB-1 (par-tile crate apex); can land in parallel with PR-CE64-MB-2 + PR-CE64-MB-3
> **LOC estimate:** ~600 LOC
> **Iron rules:** I-SUBSTRATE-MARKOV preserved · I-VSA-IDENTITIES preserved · I-NOISE-FLOOR-JIRAK noted

---

## §1 Statement of Scope

This PR upgrades AriGraph from SPO triples to SPO-G quads by adding a fifth architectural
position — a `u32` OGIT domain pointer — implementing ghost-edge persistence for unresolved
hole-forms from the SPOW tetrahedron (cited from §8 of oxigraph-arigraph-cognitive-shader-
soa-merge-v1.md), and introducing two complementary witness shapes for peer and
parent-supervisor edges.

**What this PR adds:**

1. **SPO-G quad** — `Triplet` in `triplet_graph.rs` gains `g: u32` (architectural OGIT
   domain pointer, cold form). The 5-bit hot-slot lives in `CausalEdge64` (PR-CE64-MB-2).
   Lance MVCC versioning provides the temporal axis: `(G, lance_version)` per
   `ogit-g-context-bundle-v1.md` §D-OGIT-G-1.

2. **Ghost-edge persistence** — when a compartment hits temporal-window-end OR
   budget-exhausted without resolving a SPOW hole-form, AriGraph stores a ghost at
   Pearl rung 3 (counterfactual) or 7 (full-cf). Ghosts persist FOREVER in AriGraph;
   only the `AttentionMask` hot-slot evicts.

3. **`SpoWitness64`** — u64 packed, Copy, 8 bytes. Peer mailbox edges.

4. **`SpoWitnessChain<N>`** — `Box<[SpoWitness64; N]>` (default N=32). Parent-supervisor
   + AriGraph commit edges.

**Out of scope for this PR:**
- 5-bit G hot-slot in CausalEdge64 (PR-CE64-MB-2)
- AttentionMask rename lookups (PR-CE64-MB-5)
- SigmaTierRouter dispatch (PR-CE64-MB-6)

---

## §2 SPO-G Quad Shape

### 2.1 Current SPO Triple (confirmed from triplet_graph.rs)

```rust
// crates/lance-graph/src/graph/arigraph/triplet_graph.rs — CURRENT
pub struct Triplet {
    pub subject: String,
    pub object: String,
    pub relation: String,
    pub truth: TruthValue,    // NARS frequency + confidence
    pub timestamp: u64,
}
```

Note: current AriGraph is warm string-keyed L1. Cold fingerprint-keyed L2 is `SpoStore`
populated by `spo_bridge::promote_to_spo`. The SPO-G upgrade extends L1 first.

### 2.2 New SPO-G Quad (Triplet extended)

```rust
/// A knowledge triplet with OGIT domain context: subject -[relation]-> object | G.
///
/// `g` is the architectural OGIT domain pointer (cold u32 form).
/// The 5-bit hot-slot lives in CausalEdge64.g_slot (PR-CE64-MB-2) and resolves
/// to this u32 via AttentionMask::resolve_g (PR-CE64-MB-5).
///
/// Per ogit-g-context-bundle-v1.md §D-OGIT-G-1:
///   g = 0 = UNROUTED (default for existing SPO triples, backward-compat)
///   Lance MVCC provides temporal axis: (G, lance_version)
#[derive(Debug, Clone)]
pub struct Triplet {
    pub subject: String,
    pub object: String,
    pub relation: String,
    pub truth: TruthValue,
    pub timestamp: u64,
    /// OGIT domain pointer (u32). 0 = UNROUTED (legacy default, backward-compat).
    pub g: u32,
    /// Pearl causal rung (0-7). 3 = counterfactual ghost; 7 = full-cf ghost.
    pub pearl_rung: u8,
    /// Witness reference (FNV-1a hash of (G, S_hash, P_hash, O_hash)).
    /// 0 = no witness. Points into WitnessChainStore.
    /// Per oxigraph-arigraph-cognitive-shader-soa-merge-v1.md §2 line ~150.
    pub witness_ref: u64,
}
```

### 2.3 Storage Migration and Backward Compatibility

- **Existing SPO triples**: `g = 0`, `pearl_rung = 0`, `witness_ref = 0` on load (UNROUTED defaults).
- **Queries without G filter**: union scan over all G OR G=0 fallback; planner choice deferred
  to W10 PR-dep-graph spec per §7 sequencing.
- **`promote_to_spo` bridge**: gains `g: u32` with default 0 — existing callers compile unchanged.
  Implemented as `promote_to_spo_g(triplet, gate, spo, g)` + keep original forwarding with g=0.
- **Lance schema version**: bump `SCHEMA_VERSION: u32 = 2 → 3` in
  `crates/lance-graph-ontology/src/lance_cache.rs`. Add `g u32`, `pearl_rung u8`,
  `witness_ref u64` columns. Update `schema_version_pinned` test to expect version 3 and
  new column count. Follow `lance_cache_invalidate_*` test pattern (stale meta triggers
  rebuild, not panic).
- **`witness_ref` derivation**: use `contract::hash::fnv1a` (canonical per PR #307) — avoids
  hash inconsistency across crates.

### 2.4 SpoGQuad Query Result Type

```rust
/// G-filtered query result returned by TripletGraph::query_by_g.
#[derive(Debug, Clone)]
pub struct SpoGQuad {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub g: u32,
    pub pearl_rung: u8,
    pub witness_ref: u64,
    pub truth: TruthValue,
    pub timestamp: u64,
}
```

New method on `TripletGraph`:
```rust
pub fn query_by_g(&self, g: u32) -> Vec<SpoGQuad> { ... }
```

---

## §3 Witness Lane Upgrade — SPOW Tetrahedron

DELTA: cites and refines `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §8.
Do NOT re-derive the SPOW tetrahedron — cite §8.

### 3.1 SPOW Hole-Form Taxonomy (from §8)

```
W = witness / why / worldline / evidence / provenance

Tetrahedron hole-forms:
  SP_ asks O      (subject + predicate → find object)
  S_O asks P      (subject + object → find predicate)
  _PO asks S      (predicate + object → find subject)
  SPO asks W      (full triad → find witness/provenance)
  SPW asks O
  SOW asks P
  POW asks S
```

A ghost edge is emitted when a compartment cannot resolve its hole-form within the
temporal window. The ghost preserves the unsolved (S, P, O, W, G) tuple.

### 3.2 SpoWitness64 — Packed Peer-Mailbox Form

```rust
// crates/lance-graph/src/graph/arigraph/witness.rs (NEW FILE)

/// Packed witness for peer mailbox edges. Copy, 8 bytes, cache-line friendly.
///
/// Bit layout (64 bits):
///   0- 7: s_idx     (8b) subject palette index — mirrors CausalEdge64 bits 3-10
///   8-15: p_idx     (8b) predicate palette index
///  16-23: o_idx     (8b) object palette index
///  24-29: w_palette (6b) witness palette slot — mirrors CausalEdge64 W slot
///  30-31: truth_band (2b) TrustTexture collapse — mirrors CausalEdge64 bits 62-63
///  32-34: pearl_rung (3b) Pearl 2^3 rung — mirrors CausalEdge64 bits 0-2
///  35-50: temporal  (16b) cycle index — mirrors CausalEdge64 bits 27-42
///  51-63: reserved  (13b) zero
///
/// Size invariant: std::mem::size_of::<SpoWitness64>() == 8
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SpoWitness64(pub u64);

impl SpoWitness64 {
    pub fn new(
        s_idx: u8, p_idx: u8, o_idx: u8,
        w_palette: u8, truth_band: u8, pearl_rung: u8, temporal: u16,
    ) -> Self {
        debug_assert!(w_palette < 64);
        debug_assert!(truth_band < 4);
        debug_assert!(pearl_rung < 8);
        Self(
            (s_idx as u64)
            | ((p_idx as u64) << 8)
            | ((o_idx as u64) << 16)
            | ((w_palette as u64) << 24)
            | ((truth_band as u64) << 30)
            | ((pearl_rung as u64) << 32)
            | ((temporal as u64) << 35)
        )
    }

    pub fn s_idx(self) -> u8       { (self.0 & 0xFF) as u8 }
    pub fn p_idx(self) -> u8       { ((self.0 >> 8) & 0xFF) as u8 }
    pub fn o_idx(self) -> u8       { ((self.0 >> 16) & 0xFF) as u8 }
    pub fn w_palette(self) -> u8   { ((self.0 >> 24) & 0x3F) as u8 }
    pub fn truth_band(self) -> u8  { ((self.0 >> 30) & 0x3) as u8 }
    pub fn pearl_rung(self) -> u8  { ((self.0 >> 32) & 0x7) as u8 }
    pub fn temporal(self) -> u16   { ((self.0 >> 35) & 0xFFFF) as u16 }

    /// True if pearl_rung is 3 (counterfactual) or 7 (full-cf) — ghost witness.
    pub fn is_ghost(self) -> bool { matches!(self.pearl_rung(), 3 | 7) }
}
```

### 3.3 SpoWitnessChain<N> — Owned Parent-Supervisor Form

```rust
/// Owned witness chain for parent-supervisor and AriGraph commit edges.
///
/// Default N = 32 — matches Markov bundle limit √d/4 ≈ 32 at d=16384
/// (per CLAUDE.md "Markov bundle ≤ √d/4"). When chain exceeds N, oldest
/// half is NARS-summarized into history_summary and chain truncates.
///
/// Conversion: SpoWitnessChain::from_single(w: SpoWitness64) -> Self
/// for the common single-witness-emission path (OQ-8 parent plan).
pub struct SpoWitnessChain<const N: usize = 32> {
    entries: [SpoWitness64; N],
    len: usize,
    /// NARS-summarized epoch witness from pre-truncation period.
    /// None until chain has overflowed once.
    history_summary: Option<SpoWitness64>,
}

impl<const N: usize> SpoWitnessChain<N> {
    pub fn empty() -> Self { ... }
    pub fn from_single(w: SpoWitness64) -> Self { ... }
    pub fn push(&mut self, w: SpoWitness64) { /* truncate + NARS summarize if full */ }
    pub fn len(&self) -> usize { self.len }
    pub fn as_slice(&self) -> &[SpoWitness64] { &self.entries[..self.len] }
    pub fn history_summary(&self) -> Option<SpoWitness64> { self.history_summary }
}

impl<const N: usize> From<SpoWitness64> for SpoWitnessChain<N> {
    fn from(w: SpoWitness64) -> Self { Self::from_single(w) }
}
```

`WitnessChainStore` (lives alongside `TripletGraph`):
```rust
pub struct WitnessChainStore {
    chains: HashMap<u64, SpoWitnessChain<32>>,
}
impl WitnessChainStore {
    pub fn push_witness(&mut self, witness_ref: u64, w: SpoWitness64) { ... }
    pub fn get_chain(&self, witness_ref: u64) -> Option<&SpoWitnessChain<32>> { ... }
}
```

---

## §4 Ghost-Edge Persistence

### 4.1 Emission Policy

When `MailboxSoA` compartment reaches temporal-window-end OR budget-exhausted:

1. Compartment calls `GhostStore::emit_ghost(subject, relation, object, g, reason, cycle)`.
2. Ghost stored as `Triplet` with `pearl_rung = reason.pearl_rung()`:
   - `GhostReason::TemporalWindowEnd` → rung 3 (counterfactual)
   - `GhostReason::BudgetExhausted` → rung 7 (full-cf)
3. No schema change to Arrow/Lance table — `pearl_rung` is the extended `Triplet` field (§2.2).

### 4.2 Ghost Types (ghost.rs NEW FILE)

```rust
// crates/lance-graph/src/graph/arigraph/ghost.rs

pub enum GhostReason {
    TemporalWindowEnd,   // → pearl_rung 3
    BudgetExhausted,     // → pearl_rung 7
}
impl GhostReason {
    pub fn pearl_rung(self) -> u8 {
        match self {
            Self::TemporalWindowEnd => 3,
            Self::BudgetExhausted => 7,
        }
    }
}

/// Event fired when new concrete evidence matches a ghosted (G, S, P, O) tuple.
/// SigmaTierRouter (PR-CE64-MB-6, W7 spec) subscribes to spawn a fresh compartment.
pub struct GhostReactivationEvent {
    pub g: u32,
    pub subject: String,
    pub relation: String,
    pub object: String,   // the previously-unresolved object (ghost placeholder)
    pub ghost_rung: u8,
    pub evidence: Triplet, // concrete new evidence
}

/// Ghost-edge store: wraps TripletGraph for ghost-specific operations.
///
/// Hibernation policy (causaledge64-mailbox-rename-soa-v1.md §4 + §11 OQ-2):
/// Ghosts persist in AriGraph FOREVER. AttentionMask hot-slot eviction does
/// NOT delete ghosts. Ghost edges ARE the long-term memory.
pub struct GhostStore<'a> {
    graph: &'a mut TripletGraph,
}

impl<'a> GhostStore<'a> {
    pub fn new(graph: &'a mut TripletGraph) -> Self { Self { graph } }

    pub fn emit_ghost(
        &mut self,
        subject: &str, relation: &str, object: &str,
        g: u32, reason: GhostReason, cycle: u64,
    ) -> usize { ... }

    /// Iterator over active (non-deleted) ghosts in domain G.
    pub fn pending_ghosts(&self, g: u32) -> impl Iterator<Item = &Triplet> {
        self.graph.triplets.iter().filter(move |t| {
            t.g == g && matches!(t.pearl_rung, 3 | 7) && !t.is_deleted()
        })
    }

    /// Check if concrete evidence reactivates a ghost.
    /// Matches on (G, subject, relation) — object is the open hole-position.
    pub fn check_reactivation(&self, evidence: &Triplet) -> Option<GhostReactivationEvent> { ... }
}
```

### 4.3 NARS Ghost Decay (OQ-2 Resolution)

Per parent plan §11 OQ-2: "NARS-revised, low-frequency" — batched at AriGraph-commit
boundaries, not per-cycle.

```rust
/// Low-frequency batch: NARS-revise ghost confidence using newer contradicting triples.
///
/// For each ghost in G: count non-ghost triples sharing (G, subject, relation).
/// Contradiction count > 0 → decay ghost confidence by 1/(1+contradiction_count).
/// confidence < 0.05 → soft-delete ghost.
///
/// I-NOISE-FLOOR-JIRAK: the 0.05 floor is hand-tuned (documented here).
/// Principled threshold: use Jirak 2016 rate bounds when available.
pub fn nars_revise_ghosts(graph: &mut TripletGraph, g: u32) { ... }
```

---

## §5 Christmas-Tree Decoration Mechanics

### 5.1 Decoration Pipeline (DELTA from §9 of oxigraph plan)

```
Compartment emits CausalEdge64
    ↓
SigmaTierRouter (PR-CE64-MB-6) → AriGraph commit if intent.is_some() OR σ_tier ≥ Σ7
    ↓
AriGraph::commit_edge(edge: CausalEdge64, mask: &AttentionMask)
  resolve G:   mask.resolve_g(edge.g_slot()) → OgitDomainId → u32
  resolve W:   mask.resolve_w(edge.w_slot()) → WitnessId → u64
  resolve S/P/O: palette indices → strings via PaletteSemiring
    ↓
TripletGraph::add_triplets(&[Triplet { g, pearl_rung, witness_ref, ... }])
    ↓
WitnessChainStore::push_witness(witness_ref, SpoWitness64::from(edge))
```

### 5.2 Context Separation Law (§3 of merge plan, load-bearing)

Per `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §3:

```
ontology_context_id (= g: u32) = semantic domain/namespace boundary
witness_ref (= u64 FNV-1a)     = why/how/source this assertion is supported
```

Never collapse context and witness into one field. G says WHERE. witness_ref says WHY.
`WitnessChainStore` is indexed by `witness_ref`; `TripletGraph` is queryable by `g`.

### 5.3 Chain Truncation at N=32

When `SpoWitnessChain::push` overflows N=32:
1. NARS-summarize oldest N/2 entries → `history_summary: SpoWitness64`
2. Compact newest N/2 entries to front
3. Append new witness at position N/2 + 1

This matches the 32-bundle-limit from CLAUDE.md "Markov bundle ≤ √d/4" at d=16384.

---

## §6 Files-to-Touch Table

| File | Change | LOC delta |
|---|---|---|
| `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` | Extend `Triplet` with `g`, `pearl_rung`, `witness_ref`; add `SpoGQuad`; add `query_by_g`; update `add_triplets` dedup to include `g` | +200 LOC |
| `crates/lance-graph/src/graph/arigraph/witness.rs` | NEW — `SpoWitness64` + `SpoWitnessChain<N>` + `WitnessChainStore` | +150 LOC |
| `crates/lance-graph/src/graph/arigraph/ghost.rs` | NEW — `GhostReason` + `GhostStore` + `GhostReactivationEvent` + `nars_revise_ghosts` | +120 LOC |
| `crates/lance-graph/src/graph/arigraph/orchestrator.rs` | Integrate SPO-G commit path + ghost emission on temporal-window-end/budget-exhausted + GhostReactivationEvent | +80 LOC |
| `crates/lance-graph/src/graph/arigraph/mod.rs` | `pub mod witness; pub mod ghost;` | +2 LOC |
| `crates/lance-graph/src/graph/arigraph/spo_bridge.rs` | `promote_to_spo_g(triplet, gate, spo, g)` + keep original forwarding | +15 LOC |
| `crates/lance-graph-ontology/src/lance_cache.rs` | `SCHEMA_VERSION 2 → 3`; add `g u32`, `pearl_rung u8`, `witness_ref u64` columns; update `schema_version_pinned` test | +30 LOC |
| Tests (inline) | 7 new test functions | ~120 LOC |

**No changes to:** `episodic.rs`, `retrieval.rs`, `sensorium.rs`, `xai_client.rs`, `language.rs`,
`crates/causal-edge/` (g_slot hot-path is PR-CE64-MB-2).

---

## §7 Test Plan

### T1: `spo_g_quad_round_trip`
Write SPO-G quad with G=42; read back via `query_by_g(42)`; assert field equality.
Also write legacy triple (g=0) and assert `query_by_g(0)` returns it.

### T2: `spo_g_query_g_filter`
Write quads in G=1 and G=2; query WHERE g=1; assert only G=1 quads returned.
Query g=99; assert empty.

### T3: `witness64_to_witnesschain_packing`
Assert `std::mem::size_of::<SpoWitness64>() == 8`.
Construct witness, assert all accessor fields round-trip.
Construct `SpoWitnessChain::from_single(w)`, assert `len == 1`.
Assert chain owns its array (not borrowed).

### T4: `ghost_persistence_after_compartment_drop`
Spawn `GhostStore`, emit ghost in G=42, drop store (simulates compartment drop).
Re-open `GhostStore`, query `pending_ghosts(42)`.
Assert ghost present with `pearl_rung == 3` (TemporalWindowEnd).

### T5: `ghost_reactivation_on_evidence`
Emit ghost at (G=42, "alice", "knows", "?").
Construct concrete evidence (G=42, "alice", "knows", "bob").
Call `GhostStore::check_reactivation(&evidence)`.
Assert `GhostReactivationEvent { ghost_rung: 3, evidence.object: "bob" }`.

### T6: `nars_revised_ghost_decay`
Emit ghost at (G=42, "alice", "knows", "?", BudgetExhausted).
Assert initial confidence == 0.1.
Add 5 non-ghost triples sharing (G=42, "alice", "knows").
Call `nars_revise_ghosts(&mut graph, 42)`.
Assert ghost confidence < 0.1 (decay applied).

### T7: `christmas_tree_chain_truncation`
Push 50 witnesses to `SpoWitnessChain<32>`.
Assert `chain.len() <= 32`.
Assert `chain.history_summary().is_some()`.
Assert most-recent witnesses are preserved in chain.

---

## §8 Risk Matrix

| Risk | Severity | Mitigation |
|---|---|---|
| **Lance schema bump breaks existing AriGraph data** | HIGH | Add new columns with Arrow defaults (g=0, pearl_rung=0, witness_ref=0). Follow `lance_cache_invalidate_*` test pattern from `lance_cache.rs`. SCHEMA_VERSION 2→3. Update `schema_version_pinned` test. Existing rows read UNROUTED defaults. |
| **`promote_to_spo` API break on g parameter** | HIGH | Add `promote_to_spo_g(triplet, gate, spo, g)` new function; keep original `promote_to_spo` unchanged (forwards with g=0). Zero breaking change for existing callers. |
| **`SpoWitnessChain<N>` sizing** | MED | N=32 matches Markov bundle √d/4 limit at d=16384 (CLAUDE.md). If traces show typical chains < 8 entries, smaller N reduces footprint. If chains routinely hit 32, NARS truncation cost is visible. Ratify N=32 via OQ with W10. |
| **NARS ghost decay rate** | MED | `1 / (1 + contradiction_count)` is hand-tuned. I-NOISE-FLOOR-JIRAK: when principled threshold needed, use Jirak 2016 rate bounds. The 0.05 confidence floor is also hand-tuned — documented in code comment per iron rule. |
| **Pearl rung 3 vs 7 encoding** | LOW | Rung 3 = do-modified counterfactual (temporal window end); rung 7 = full-cf (budget exhausted). Matches Pearl 2^3 hierarchy in causaledge64 plan §3. `GhostReason::pearl_rung()` is single source of truth. |
| **TripletGraph in-memory only** | MED | Ghost edges accumulate without bound in Vec. Lance-backed persistence is follow-on (OQ-W5-1). This PR establishes ghost API + in-memory semantics; next PR adds Lance persistence via `LanceWriter`. |
| **Ghost-ghost collision on same (G, S, P, ?)** | LOW | Two compartments racing same hole-form emit two ghosts with same placeholder object. Extend dedup key in `add_triplets` to include `g` and `pearl_rung`. Documented in §6 change to `add_triplets`. |

---

## §9 Open Questions for Meta-Review

**OQ-W5-1 — Lance persistence for ghost edges**
Current `TripletGraph` is fully in-memory. For production "ghosts persist FOREVER" durability,
ghost edges must be written to Lance via `LanceWriter`. Should this PR implement Lance-backed
ghost persistence, or defer to a follow-on (PR-CE64-MB-4b)?
**Recommendation: defer** — this PR establishes ghost API and in-memory semantics; follow-on
adds Lance persistence using the `lance_cache.rs` SCHEMA_VERSION pattern.

**OQ-W5-2 — `promote_to_spo` API evolution**
Two options: (a) separate `promote_to_spo_g` function (zero breaking change, API surface
duplication), or (b) builder pattern on `PromoteGate`. Recommend (a) for this PR; (b) as
follow-on refactor.

**OQ-W5-3 — `witness_ref` derivation function**
Should use `contract::hash::fnv1a` (canonical per PR #307) for consistency across crates.
Avoids hash inconsistency. Confirm that `lance-graph` can depend on `lance-graph-contract`
(it already does — TruthValue is from contract).

---

## §10 Iron Rule Compliance

| Iron Rule | Compliance |
|---|---|
| **I-SUBSTRATE-MARKOV** | Preserved. SPO-G quads are identity tuples (subject/relation/object = string-keyed identities; g = domain pointer identity; palette indices = archetype identities). No VSA bundling of SPO content in AriGraph. Vsa16kF32 remains single-cycle-only. |
| **I-VSA-IDENTITIES** | Preserved. `SpoWitness64.s_idx/p_idx/o_idx` = palette indices (identities). `w_palette` = witness palette slot (identity). `witness_ref` = FNV-1a pointer (identity). No VSA superposition of content occurs in this module. |
| **I-NOISE-FLOOR-JIRAK** | Noted. NARS ghost decay `confidence < 0.05` floor is hand-tuned (documented in code comment). When principled threshold needed: use Jirak 2016 rate bounds. Initial ghost confidence 0.1 also hand-tuned — documented. |
| **I1 (BindSpace read-only)** | Preserved. AriGraph SPO-G quads and ghost edges are Zone-2 cold storage; not BindSpace columns. No CollapseGate involvement in AriGraph persistence. |
| **Method-on-carrier discipline** | Preserved. `TripletGraph::query_by_g`, `GhostStore::emit_ghost`, `GhostStore::pending_ghosts`, `GhostStore::check_reactivation`, `SpoWitnessChain::push`, `SpoWitnessChain::from_single` are all carrier methods. `nars_revise_ghosts` is a module-level batch utility (acceptable for batch-processing operations — not a carrier state query). |
| **Zero-dep invariant (contract crate)** | Preserved. No new external crate deps added to `lance-graph-contract`. New types live in `lance-graph/graph/arigraph/` (not contract). |

---

## §11 Cross-Crate Coordination

**Consuming workers (downstream from this PR):**
- **W6** (`pr-ce64-mb-5-mailbox-soa-attentionmask.md`): `MailboxSoA` lifecycle step 4
  calls `AriGraph::commit_edge` — stub the call site pending this PR.
- **W7** (`pr-ce64-mb-6-sigma-tier-router.md`): `SigmaTierRouter` subscribes to
  `GhostReactivationEvent` to spawn fresh compartments.

**This PR exports for downstream:**
- `Triplet` with `g: u32 + pearl_rung: u8 + witness_ref: u64`
- `SpoGQuad` query result type
- `SpoWitness64` (Copy, 8 bytes)
- `SpoWitnessChain<N>` (default N=32)
- `GhostReactivationEvent`
- `GhostStore::pending_ghosts(g)` iterator
- `nars_revise_ghosts(graph, g)` batch utility
- `SCHEMA_VERSION` bump 2 → 3 in `lance_cache.rs`

---

*End of spec PR-CE64-MB-4 — AriGraph SPO-G Quad Upgrade.*
*~600 LOC estimate. Plans cited: causaledge64-mailbox-rename-soa-v1.md §6+§7,*
*ogit-g-context-bundle-v1.md §D-OGIT-G-1, oxigraph-arigraph-cognitive-shader-soa-merge-v1.md §1-§9.*
