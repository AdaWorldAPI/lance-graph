# PR-CE64-MB-4 — AriGraph SPO-G Quad Upgrade + Ghost-Edge Persistence + WitnessCorpus

> **Status:** Spec (2026-05-14; patched 2026-05-16 per cognitive-substrate-convergence-v1.md) — sprint-log-10 W5 output
> **Scope deliverable:** D-OGIT-G-1 (SPO-G quad) + ghost-edge persistence + SpoWitness64 + WitnessCorpus (CAM-PQ-indexed, replaces SpoWitnessChain<32>)
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 (lance-graph::arigraph row) + §7 (PR-CE64-MB-4 entry)
> **Architectural anchor:** `.claude/plans/cognitive-substrate-convergence-v1.md` §5 L-16, L-17 · §6 W-slot · §11 D-CSV-6 · §12 (W5 patch row)
> **Primary references:**
> - `.claude/plans/ogit-g-context-bundle-v1.md` §D-OGIT-G-1 — SPO-G u32 slot spec
> - `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1 §2 §3 §8 §9
> - `.claude/knowledge/spo-ontology-format-stack.md` — CAM-PQ codec context; WitnessCorpus indexing
> **Depends on:** PR-CE64-MB-1 (par-tile crate apex); can land in parallel with PR-CE64-MB-2 + PR-CE64-MB-3
> **LOC estimate:** ~900 LOC (was ~600; +~300 for WitnessCorpus design per cognitive-substrate-convergence-v1.md §12)
> **Iron rules:** I-SUBSTRATE-MARKOV preserved · I-VSA-IDENTITIES preserved · I-NOISE-FLOOR-JIRAK noted
> **New invariant added:** W5-INV-CHAIN-ORDER (see §3A)

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

3. **`SpoWitness64`** — u64 packed, Copy, 8 bytes. Each witness becomes a corpus row in
   `WitnessCorpus` (per L-17 of cognitive-substrate-convergence-v1.md).

4. **`WitnessCorpus`** — CAM-PQ-indexed (via `ndarray::hpc::cam_pq`), unbounded, with
   salience-decay eviction. Replaces the bounded `SpoWitnessChain<32>` linked-list which
   does not scale to discourse-level reasoning (per plan L-17). Indexed lookup in ≤50 µs
   at 1M corpus entries (D-CSV-6 benchmark target).

**Out of scope for this PR:**
- 5-bit G hot-slot in CausalEdge64 (PR-CE64-MB-2)
- AttentionMask rename lookups (PR-CE64-MB-5)
- SigmaTierRouter dispatch (PR-CE64-MB-6)

> **NOTE (2026-05-16 patch):** `SpoWitnessChain<32>` is **retired** by this patch and replaced
> throughout by `WitnessCorpus`. See §3A (WitnessCorpus design), §3B (W-slot semantics in
> CausalEdge64 v2), and invariant `W5-INV-CHAIN-ORDER` below. Deliverable for `WitnessCorpus`
> implementation is D-CSV-6 per `.claude/plans/cognitive-substrate-convergence-v1.md` §11.

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
    /// 0 = no witness. Points into WitnessCorpus (CAM-PQ-indexed, unbounded).
    /// Per cognitive-substrate-convergence-v1.md L-17 + oxigraph-arigraph-cognitive-
    /// shader-soa-merge-v1.md §2 line ~150.
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

### 3.3 WitnessCorpus Design (replaces SpoWitnessChain<32>)

> **Architectural rationale (cognitive-substrate-convergence-v1.md §5 L-17):** The bounded
> `SpoWitnessChain<32>` linked-list does not scale to discourse-level reasoning. The `<32>` cap
> collides with the Markov-bundle floor (√d/4 ≈ 32 items at d=16384) but not with the witness
> corpus size needed for multi-turn discourse (hundreds of entries). Per `L-3`, the G-slot in
> v1 CausalEdge64 was redundant: per-tenant SoA partition already encodes tenant, and palette
> family-prefix already encodes ontological family. Witness corpus replaces `SpoWitnessChain<32>`
> as the canonical witness-tracking structure. The W-slot in CausalEdge64 v2 is the entry pointer
> into the corpus (corpus root handle — 64 active corpora at 6 bits per §6 of
> cognitive-substrate-convergence-v1.md), NOT a G-slot encoding tenant.

#### 3.3.1 WitnessEntry — corpus row

```rust
// crates/lance-graph/src/graph/arigraph/witness.rs (extended)

/// A single witness row in the WitnessCorpus.
/// Unbounded: no `<32>` cap. One entry per witnessed SPO event.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Packed SPO triple identity (FNV-1a of palette indices).
    pub spo: u64,
    /// Wall-clock nanoseconds at witness emission.
    /// Primary sort key per W5-INV-CHAIN-ORDER.
    pub timestamp_ns: u64,
    /// Optional provenance URL / source document reference.
    /// Carries identity fingerprint per I-VSA-IDENTITIES (not raw content).
    pub source_url: Option<String>,
    /// Arbitrary evidence blob (serialized context, citation, or proof chunk).
    /// Content stored here; identity pointer lives in `spo` + `timestamp_ns`.
    pub evidence_blob: bytes::Bytes,
}
```

#### 3.3.2 WitnessCorpus — CAM-PQ-indexed store

```rust
// crates/lance-graph/src/graph/arigraph/witness.rs

/// CAM-PQ-indexed witness corpus. Replaces SpoWitnessChain<32>.
///
/// INVARIANTS:
///   W5-INV-CHAIN-ORDER: entries sorted by timestamp_ns ASC; same-timestamp
///     tie-break by source_url.hash(). Insertion via Arc::make_mut (copy-on-write).
///   W5-INV-WITNESS-UNBOUNDED: no `<32>` cap; growth bounded only by
///     salience-decay eviction policy (WitnessCorpusPruningPolicy — D-CSV-6 sub-task).
///   W5-INV-CAM-PQ-INDEX: cam_pq_index is the canonical search structure;
///     direct Vec scan is reserved for Miri/test contexts.
///
/// Indexed lookup target: ≤50 µs at 1M corpus entries (D-CSV-6 benchmark).
pub struct WitnessCorpus {
    /// Sorted entry store. Arc for copy-on-write semantics on insertion.
    pub entries: Arc<Vec<WitnessEntry>>,
    /// CAM-PQ index over (s, p, o) triple. O(log N) point queries,
    /// O(1) for hot subjects via palette family-prefix bucket.
    pub cam_pq_index: CamPqIndex,
}

/// Opaque handle returned by WitnessCorpus::insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessId(pub u64);

/// Query selector for fuzzy / family-prefix lookups.
pub struct SpoQuery {
    pub s: Option<u8>,      // subject palette index (None = wildcard)
    pub p: Option<u8>,      // predicate palette index (None = wildcard)
    pub o: Option<u8>,      // object palette index (None = wildcard)
}

impl WitnessCorpus {
    /// Insert a new witness entry. Maintains W5-INV-CHAIN-ORDER via binary
    /// search insertion on timestamp_ns; tie-breaks by source_url hash.
    /// Uses Arc::make_mut for copy-on-write — cheap when refcount == 1.
    ///
    /// Returns WitnessId (index into sorted entries at time of insert;
    /// stable until corpus compaction).
    pub fn insert(
        &mut self,
        spo: u64,
        timestamp_ns: u64,
        source_url: Option<String>,
        evidence_blob: bytes::Bytes,
    ) -> WitnessId { ... }

    /// Point query: return all entries matching this exact SPO triple,
    /// sorted by timestamp_ns ASC (W5-INV-CHAIN-ORDER).
    /// O(log N) via cam_pq_index; O(1) amortized for hot-subject buckets.
    pub fn query(&self, spo: u64) -> impl Iterator<Item = &WitnessEntry> { ... }

    /// Fuzzy / family-prefix lookup via CAM-PQ index. Returns top-k WitnessIds
    /// ranked by palette family-prefix similarity (ontological family proximity).
    /// Used when exact SPO triple not found and family-level inference is needed.
    pub fn cam_pq_search(&self, query: SpoQuery, k: usize) -> Vec<WitnessId> { ... }

    /// Evict entries whose salience has decayed below threshold.
    /// Salience = NARS frequency × confidence × recency_weight.
    /// Hand-tuned threshold (I-NOISE-FLOOR-JIRAK: principled derivation deferred
    /// to D-CSV-6 sub-task WitnessCorpusPruningPolicy).
    pub fn evict_stale(&mut self, threshold: f32) -> usize { ... }
}
```

#### 3.3.3 Per-tenant lookup flow (replaces G-slot tenant routing)

Per `cognitive-substrate-convergence-v1.md` §5 L-3 + L-9, tenant routing no longer requires
a dedicated G-slot in CausalEdge64. The canonical lookup flow is:

```text
1. SoA scan: identify per-tenant rows via MailboxSoA partition key
   (tenant boundary is enforced at the SoA partition level, not via a bit in the edge)

2. Palette family-prefix filter: SPO palette indices encode ontological family
   via OGIT family-prefix convention (workspace-locked codebook per contract::manifest).
   Triples in the same family share the upper-4-bits of their palette index.

3. Witness CAM-PQ lookup: WitnessCorpus::cam_pq_search(SpoQuery { s, p, o }, k=8)
   retrieves the k nearest witnesses in O(log N); hot subjects are O(1) via
   family-prefix bucket already populated in cam_pq_index.

4. CausalEdge64 v2 decode: W-slot (6 bits) → corpus root handle (0..63) → WitnessCorpus.
   (W-slot is dispatch metadata, NOT epistemic confidence — that lives in NARS
   frequency bits 24-31 and confidence bits 32-39 of CausalEdge64 v2.)
```

**Time as helper (§5 L-9):** `timestamp_ns` in `WitnessEntry` supplies causality direction
hints without requiring a temporal bit in CausalEdge64. The `causal_mask` (3 bits, Pearl-2³)
covers Pearl-3 counterfactual reasoning directly; temporal ordering in the corpus supplies
the "happened-before" direction hint structurally, matching the "temporal causality is
structural, not stored" doctrine from CLAUDE.md "The Click" §2.

`WitnessCorpusStore` (lives alongside `TripletGraph`, replaces `WitnessChainStore`):
```rust
/// Global witness corpus store. Keyed by corpus root handle (W-slot value, 0..63).
/// Replaces WitnessChainStore + SpoWitnessChain<32>.
pub struct WitnessCorpusStore {
    corpora: [Option<WitnessCorpus>; 64],
}

impl WitnessCorpusStore {
    pub fn corpus_mut(&mut self, root: u8) -> &mut WitnessCorpus { ... }
    pub fn corpus(&self, root: u8) -> Option<&WitnessCorpus> { ... }

    /// Insert a witness into the corpus identified by root handle.
    pub fn push_witness(
        &mut self, root: u8,
        spo: u64, timestamp_ns: u64,
        source_url: Option<String>,
        evidence_blob: bytes::Bytes,
    ) -> WitnessId { ... }
}
```

### 3.4 W-Slot Semantics in CausalEdge64 v2

> **Source:** `cognitive-substrate-convergence-v1.md` §6 bit layout (bits 53-58) + §5 L-6.

The W-slot in `CausalEdge64` v2 encodes **dispatch metadata** for the witness corpus, NOT
epistemic confidence. The 6-bit W-slot covers three sub-fields packed within those 6 bits:

| Sub-field | Width | Values | Semantics |
|---|---|---|---|
| **Tier** | 3 bits | `Sharp(0)` / `Wide(1)` / `Quenched(2)` / `Recall(3)` / `Compress(4)` / `Lambda(5)` + 2 reserved | Dispatch tier for corpus root access |
| **Plasticity** | 2 bits | `Frozen(0)` / `Slow(1)` / `Fast(2)` / `Volatile(3)` | Write-plasticity of the witness corpus at this handle |
| **State** | 1 bit | `0 = Witnessed` / `1 = Hypothetical` | Whether the associated triple has a concrete witness (`0`) or is a hypothesis pending evidence (`1`) |

**Separation of concerns:**

```text
W-slot (6b, bits 53-58):  DISPATCH METADATA — tier / plasticity / hypothetical flag
NARS frequency (8b, 24-31): EPISTEMIC CONFIDENCE — how often true
NARS confidence (8b, 32-39): EPISTEMIC CONFIDENCE — how much evidence
```

The W-slot `State=1 (Hypothetical)` correlates with Pearl rung 3/7 ghost edges: a ghost
emitted by `GhostStore::emit_ghost` sets `State=Hypothetical` in the outbound `CausalEdge64`;
once `GhostStore::check_reactivation` finds concrete evidence, the corpus is updated and
the W-slot `State` is flipped to `Witnessed`.

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

### 5.1 Decoration Pipeline — SoA Partition + WitnessCorpus Flow

> **Updated 2026-05-16:** G-slot tenant routing retired per `cognitive-substrate-convergence-v1.md`
> §5 L-3. Replaced by per-tenant SoA partition + palette family-prefix + WitnessCorpus lookup.

```
Compartment emits CausalEdge64 (v2 layout)
    ↓
SoA scan: tenant boundary = MailboxSoA partition key (NOT G-slot in edge)
    ↓
Palette family-prefix filter: SPO indices → ontological family via OGIT convention
  (upper-4 bits of palette index encode family; workspace-locked codebook)
    ↓
SigmaTierRouter (PR-CE64-MB-6) → AriGraph commit if intent.is_some() OR σ_tier ≥ Σ7
    ↓
AriGraph::commit_edge(edge: CausalEdge64, mask: &AttentionMask)
  resolve W-slot: edge.w_slot() → corpus root handle (0..63) → WitnessCorpusStore::corpus()
  resolve S/P/O:  palette indices → strings via PaletteSemiring
  resolve G:      mask.resolve_g_domain() → OgitDomainId → u32 (from SoA partition, not edge)
    ↓
TripletGraph::add_triplets(&[Triplet { g, pearl_rung, witness_ref, ... }])
    ↓
WitnessCorpusStore::push_witness(
    root = edge.w_slot(),
    spo  = fnv1a(s_idx, p_idx, o_idx),
    timestamp_ns = AriGraph::wall_clock_ns(),
    source_url   = None,          // filled by provenance annotator if present
    evidence_blob = Bytes::new(), // filled by evidence packager if present
)
    ↓
Witness CAM-PQ index updated: cam_pq_index.insert(spo, WitnessId)
```

### 5.2 Context Separation Law (§3 of merge plan, load-bearing)

Per `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §3:

```
ontology_context_id (= g: u32)  = semantic domain/namespace boundary (WHERE)
witness_ref (= u64 FNV-1a)      = why/how/source this assertion is supported (WHY)
W-slot (= 6b in CausalEdge64)   = dispatch metadata: corpus root handle + tier + plasticity (HOW)
```

Never collapse context and witness into one field. G says WHERE. witness_ref says WHY.
W-slot says HOW to retrieve it. `WitnessCorpusStore` is keyed by W-slot root handle (0..63);
`TripletGraph` is queryable by `g`; CAM-PQ index is the search interface.

### 5.3 WitnessCorpus Invariants Applied at Decoration Time

When `WitnessCorpusStore::push_witness` is called during decoration:

1. **W5-INV-CHAIN-ORDER:** Binary-search insertion keeps entries sorted by `timestamp_ns ASC`.
   Same-timestamp tie-break uses `source_url.as_deref().map(|s| fxhash::hash(s)).unwrap_or(0)`.
   `Arc::make_mut` is called on the inner `Vec` — copy-on-write, cheap when refcount == 1.

2. **W5-INV-WITNESS-UNBOUNDED:** No truncation at insertion time. The corpus grows until
   `WitnessCorpus::evict_stale(threshold)` is called (policy-driven, not insertion-driven).
   This eliminates the information-loss risk of the old `SpoWitnessChain<32>` truncation.

3. **W5-INV-CAM-PQ-INDEX:** After insertion, `cam_pq_index.insert(spo, WitnessId)` is called
   to keep the index consistent. The index is the canonical search structure; raw Vec iteration
   is reserved for Miri/test contexts only.

> **Note on §5.3 replacement:** Prior `SpoWitnessChain<32>` chain-truncation at N=32 is
> **retired**. The old policy (NARS-summarize oldest N/2 → `history_summary`) is superseded
> by the salience-decay eviction policy in `WitnessCorpus::evict_stale`. The 32-bundle-limit
> from CLAUDE.md "Markov bundle ≤ √d/4" applies to VSA superposition inside one cycle's
> MatVec, NOT to the persistent witness corpus.

---

## §6 Files-to-Touch Table

| File | Change | LOC delta |
|---|---|---|
| `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` | Extend `Triplet` with `g`, `pearl_rung`, `witness_ref`; add `SpoGQuad`; add `query_by_g`; update `add_triplets` dedup to include `g` | +200 LOC |
| `crates/lance-graph/src/graph/arigraph/witness.rs` | NEW — `SpoWitness64` + `WitnessEntry` + `WitnessCorpus` (CAM-PQ-indexed, unbounded) + `WitnessCorpusStore` | +280 LOC |
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

### T3: `witness64_to_corpus_insertion_and_query`
Assert `std::mem::size_of::<SpoWitness64>() == 8`.
Construct witness, assert all accessor fields round-trip.
Construct `WitnessCorpus` (default empty). Insert 3 entries with distinct `timestamp_ns`.
Call `WitnessCorpus::query(spo)`. Assert returned iterator yields entries in `timestamp_ns` ASC order
(W5-INV-CHAIN-ORDER). Assert `entries.len() == 3` (W5-INV-WITNESS-UNBOUNDED — no cap).
Call `WitnessCorpus::cam_pq_search(SpoQuery { s: Some(idx), p: None, o: None }, k=2)`.
Assert at most 2 results, all matching subject palette index.

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

### T7: `witness_corpus_order_invariant`
Construct `WitnessCorpus`. Insert 50 witnesses with shuffled `timestamp_ns` values.
Assert `corpus.entries.len() == 50` (W5-INV-WITNESS-UNBOUNDED — no truncation).
Assert entries are sorted by `timestamp_ns` ASC (W5-INV-CHAIN-ORDER).
Insert 2 witnesses with the same `timestamp_ns` but different `source_url`.
Assert tie-break order is deterministic (hash-based — same result on re-run).
Assert CAM-PQ index has 52 entries (W5-INV-CAM-PQ-INDEX).

### T8: `witness_corpus_cam_pq_lookup_performance`
Build `WitnessCorpus` with 10_000 entries (random SPO palette indices).
Benchmark `WitnessCorpus::query(spo)` — assert p99 < 50 µs (D-CSV-6 target).
Benchmark `WitnessCorpus::cam_pq_search(query, k=8)` — assert p99 < 100 µs.
(Benchmark uses `std::time::Instant`; not a criterion bench — inline perf assertion.)

### T9: `witness_corpus_store_multi_root`
Construct `WitnessCorpusStore`. Insert witnesses into root handles 0, 1, and 63.
Assert `corpus(0)`, `corpus(1)`, `corpus(63)` each return a non-None corpus.
Assert `corpus(2)` returns None (not inserted).
Assert cross-root isolation: inserting into root 0 does not affect root 1.

---

## §8 Risk Matrix

| Risk | Severity | Mitigation |
|---|---|---|
| **Lance schema bump breaks existing AriGraph data** | HIGH | Add new columns with Arrow defaults (g=0, pearl_rung=0, witness_ref=0). Follow `lance_cache_invalidate_*` test pattern from `lance_cache.rs`. SCHEMA_VERSION 2→3. Update `schema_version_pinned` test. Existing rows read UNROUTED defaults. |
| **`promote_to_spo` API break on g parameter** | HIGH | Add `promote_to_spo_g(triplet, gate, spo, g)` new function; keep original `promote_to_spo` unchanged (forwards with g=0). Zero breaking change for existing callers. |
| **`WitnessCorpus` unbounded growth** | MED | Replaces bounded `SpoWitnessChain<32>`. Growth is bounded only by `WitnessCorpusPruningPolicy` (salience-decay eviction — D-CSV-6 sub-task). Mitigations: (1) `evict_stale(threshold)` called at AriGraph-commit boundaries; (2) salience = NARS f × c × recency weight decays old entries; (3) per-tenant quota at supervisor level if needed. The `<32>` cap is NOT the mitigation — it was the problem. |
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
| **W5-INV-CHAIN-ORDER** | **Iron rule.** `WitnessCorpus.entries` MUST be sorted by `timestamp_ns` ASC at all times. Same-timestamp tie-break is `source_url.hash()` (deterministic). Insertion uses `Arc::make_mut` + binary-search splice. Any violation causes non-deterministic causality direction — forbidden. Cross-ref: cognitive-substrate-convergence-v1.md §5 L-16. |
| **W5-INV-WITNESS-UNBOUNDED** | **Iron rule.** `WitnessCorpus` has NO `<N>` cap. Growth is bounded only by `WitnessCorpusPruningPolicy` (salience-decay eviction — D-CSV-6 sub-task). Code that re-introduces a `<32>` or any fixed-size cap on the corpus MUST be rejected in review. Cross-ref: cognitive-substrate-convergence-v1.md §5 L-17. |
| **W5-INV-CAM-PQ-INDEX** | **Iron rule.** `WitnessCorpus.cam_pq_index` is the canonical search structure for witness retrieval. Raw `Vec` iteration over `entries` is forbidden in production paths; acceptable only in Miri/test contexts. Every `insert()` call MUST update `cam_pq_index` atomically. Cross-ref: spo-schema-and-mailbox-sidecar.md §2.4 (W cardinality unbounded → CAM-PQ needed). |
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
- `WitnessEntry` — corpus row (spo, timestamp_ns, source_url, evidence_blob)
- `WitnessCorpus` — CAM-PQ-indexed, unbounded (replaces `SpoWitnessChain<32>`)
- `WitnessCorpusStore` — array of 64 corpora keyed by W-slot root handle
- `WitnessId` — opaque corpus handle (u64)
- `SpoQuery` — fuzzy lookup selector (s/p/o wildcard)
- `GhostReactivationEvent`
- `GhostStore::pending_ghosts(g)` iterator
- `nars_revise_ghosts(graph, g)` batch utility
- `SCHEMA_VERSION` bump 2 → 3 in `lance_cache.rs`

**Retired (no longer exported):**
- `SpoWitnessChain<N>` (retired per L-17; remove all uses after this PR lands)
- `WitnessChainStore` (replaced by `WitnessCorpusStore`)

**Cross-references for downstream consumers (D-CSV-6, D-CSV-7):**
- `cognitive-substrate-convergence-v1.md` §5 L-9, L-16, L-17 · §6 W-slot bit layout · §11 D-CSV-6 + D-CSV-7
- `spo-schema-and-mailbox-sidecar.md` §2.4 (SPO-W cardinality + CAM-PQ rationale)
- `CLAUDE.md` iron rule `I-VSA-IDENTITIES` (witness `source_url` and `evidence_blob` use identity fingerprints, not raw content superposition)

---

*End of spec PR-CE64-MB-4 — AriGraph SPO-G Quad Upgrade + WitnessCorpus.*
*~900 LOC estimate (+300 for WitnessCorpus per cognitive-substrate-convergence-v1.md §12 W5 patch row).*
*Plans cited: causaledge64-mailbox-rename-soa-v1.md §6+§7, ogit-g-context-bundle-v1.md §D-OGIT-G-1,*
*oxigraph-arigraph-cognitive-shader-soa-merge-v1.md §1-§9, cognitive-substrate-convergence-v1.md §5 L-3/L-6/L-9/L-16/L-17 + §6 + §11 D-CSV-6/D-CSV-7.*
*Invariants: W5-INV-CHAIN-ORDER (iron), W5-INV-WITNESS-UNBOUNDED (iron), W5-INV-CAM-PQ-INDEX (iron).*
*Knowledge ref: spo-schema-and-mailbox-sidecar.md · CLAUDE.md I-VSA-IDENTITIES.*
