# PR-CE64-MB-3: BindSpace Columns E/F/G/H — Implementation Spec

> **Status:** Draft (sprint-log-10, W4)
> **Author:** W4 (Sonnet 4.6), session 2026-05-14
> **Scope:** Extend BindSpace SoA from 4 to 8 column families.
> Columns E (OntologyDelta), F (AwarenessColumn), G (ModelRef), H
> (EntityTypeId — already partially wired, completing the per-row
> Column H SoA slice + driver.rs:311 wiring + FIX-5 trust_below_floor
> closing test).
> **Parent plan §:** bindspace-columns-v1.md §1-§5 (Phase 2 impl) +
> causaledge64-mailbox-rename-soa-v1.md §6 (cognitive-shader-driver
> row) + §7 (PR-CE64-MB-3 entry, ~800 LOC estimate).
> **LOC estimate:** ~760 LOC across 10 files.
> **Closes:** PR 355 #6 (per-row context_ids wired via Column H),
> FIX-5 (trust_below_floor per-row wiring test),
> bindspace-columns-v1.md Phase 2.
> **Gated by:** PR-CE64-MB-1 (par-tile crate, BindSpaceView type).

---

## §1 Statement of Scope

**Delta from parent plan:** bindspace-columns-v1.md §1-§3 defines the
column types, storage sizes, and scientific invariants. This spec
does NOT redefine them. It specifies the **implementation shape**:
file locations, write-token pattern, Superposition semantics, tests.

### SoA before and after (non-cycle columns only)

The parent plan §1-§2 cites ~6,212 B/row (current) → ~6,578 B/row
(target). These predate PR #323 (sigma column). Ground-truth from
the live bindspace.rs byte_footprint() test: 71,777 B/row (1 row).
The cycle plane (Vsa16kF32, 65,536 B/row) dominates; this is swept
row-by-row, not held simultaneously in L3.

Non-cycle column breakdown (matching plan §1/§2 framing):

```
Current non-cycle subtotal:   ~6,297 B/row
  A: FingerprintColumns (content/topic/angle)  = 6,144 B (unchanged)
  B: QualiaColumn [f32; 18]                    =    72 B (unchanged)
     ↳ NOTE: QualiaColumn is migrating from [f32; 18] → QualiaI4_16D (8 B/row,
       packed i4×16 signed) per cognitive-substrate-convergence-v1.md §7.2 and
       plan decision L-10. The BindSpace E/F/G/H work in THIS spec is designed
       to coexist with BOTH representations:
         • Phase 5a (sibling-column): both [f32;18] and QualiaI4_16D present;
           columns E/F/G/H access qualia via the existing f32 path unchanged.
         • Phase 5b (post-cutover, D-CSV-5): [f32;18] dropped; columns E/F/G/H
           must read QualiaI4_16D only. No E/F/G/H changes required at cutover
           because qualia is accessed only through QualiaColumn accessors, not
           packed inline.
       See D-CSV-5 in cognitive-substrate-convergence-v1.md §11 Phase B for the
       full migration phasing and blast-radius analysis.
     ↳ i4-16D Magnitude computation: Under the QualiaI4_16D substrate, Magnitude
       (qualia dim 13 per §7.2 dim-assignment table in cognitive-substrate-
       convergence-v1.md) is NOT stored — it is computed on-demand as a single
       SIMD multiply: Wisdom_i4 × Staunen_i4 → i8 product (one AVX-512 lane op
       per row sweep). Per CLAUDE.md "The Click" §3 and plan §4.1: `i4 × i4 → i8`
       stays in the integer precision family; no float conversion needed.
       This applies post-Phase-5b cutover; during Phase-5a the f32 path
       derives Magnitude via the existing accessor.
  C: MetaColumn u32 + sigma u8                 =     5 B (unchanged)
  D: EdgeColumn [u64; 8]                       =    64 B (unchanged)
     temporal u64 + expert u16 + entity u16    =    12 B (unchanged)

Target non-cycle subtotal:    ~6,589 B/row (+292 B, +4.6%)
  E: OntologyColumn (OntologyDelta 32 B)       =    32 B (NEW)
  F: AwarenessColumn [u8; 256]                 =   256 B (NEW)
  G: ModelBindingColumn (ModelRef u32)          =     4 B (NEW)
  H: TypeColumn EntityTypeId u16 (already
     allocated, this PR completes semantics)   =     0 B additional
```

After this PR: byte_footprint() = 71,777 + 32 + 256 + 4 = 72,069 B/row.
Non-cycle cache workspace at 4096 rows: ~26.9 MB (plan §2: "still fits
L3 cache on Sapphire Rapids"). cite: bindspace-columns-v1.md §1 and §2.

**Column H status (PR #272 shipped):** entity_type: Box<[u16]> is
allocated. push_typed() writes it. driver.rs:308-317 reads it.
Tests entity_type_defaults_to_untyped, entity_type_set_and_get,
builder_push_typed_sets_entity_type all passing. This PR adds the
TypeColumn named wrapper, BindSpaceView accessor, and FIX-5 test.

---

## §2 Column E — OntologyColumn

**Parent plan ref:** bindspace-columns-v1.md §3 Column E, Phase 2
deliverables D-E1 through D-E7.

**Implementation:** crates/cognitive-shader-driver/src/columns/ontology.rs (NEW)

**Contract placement:** OntologyDelta in
crates/lance-graph-contract/src/ontology.rs alongside EntityTypeId.
Preserves zero-dep invariant.

### OntologyDelta (verbatim from plan §3)

```rust
// lance-graph-contract::ontology
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct OntologyDelta {
    pub entity_type_id: u16,    // 0 = no delta
    pub relation_type_id: u16,  // 0 = no delta
    pub kind: u8,               // DeltaKind 0..=4
    pub pearl_rung: u8,         // CRITICAL: 0=obs, 1-6=do/cf, 7=full-cf
    pub frequency: u8,          // NARS frequency [0, 255]
    pub confidence: u8,         // NARS confidence [0, 255]
    pub s_idx: u8,              // Subject palette index
    pub p_idx: u8,              // Predicate palette index
    pub o_idx: u8,              // Object palette index
    pub style_ord: u8,          // ThinkingStyle ordinal
    pub temporal: u16,          // matches CausalEdge64 temporal field
    _reserved: [u8; 18],        // alignment + future use = 32 bytes total
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DeltaKind { #[default] None=0, Confirm=1, Extend=2, Contradict=3, Refine=4 }

// Mandatory compile-time size assertion:
const _: () = assert!(
    std::mem::size_of::<OntologyDelta>() == 32,
    "OntologyDelta must be exactly 32 bytes for cache line alignment"
);
```

### Pearl observational/interventional separation invariant (B2)

From bindspace-columns-v1.md §3 + §7 risk matrix B2 (SOUND if gated):

- pearl_rung = 0: observational (P(Y|X) consumers)
- pearl_rung >= 1: interventional/counterfactual (P(Y|do(X)) consumers)

Any consumer performing causal inference MUST filter by pearl_rung
before combining OntologyDeltas. Without the tag, the
observational/interventional separation required by Pearl's do-calculus
is violated. This filter is MANDATORY in all causal inference consumers.

Driver mapping (RungLevel -> pearl_rung per plan §8):

| RungLevel | pearl_rung | Pearl regime |
|---|---|---|
| Surface/Shallow | 0 | Observational |
| Contextual/Analogical | 1-2 | do(S) / do(P) |
| Abstract/Structural | 3-4 | do(O) / cf(S') |
| Counterfactual/Meta | 5-6 | cf(P') / cf(O') |
| Recursive/Transcendent | 7 | Full counterfactual |

LF cross-refs: LF-20 FunctionSpec, LF-23 NotificationSpec, LF-14 Lineage.

### OntologyColumn struct (driver-side)

```rust
// crates/cognitive-shader-driver/src/columns/ontology.rs
#[derive(Debug)]
pub struct OntologyColumn(pub Box<[OntologyDelta]>);

impl OntologyColumn {
    pub fn zeros(len: usize) -> Self {
        Self(vec![OntologyDelta::default(); len].into_boxed_slice())
    }
    #[inline] pub fn get(&self, row: usize) -> &OntologyDelta { &self.0[row] }
    #[inline] pub fn set(&mut self, row: usize, d: OntologyDelta) { self.0[row] = d; }
    #[inline] pub fn clear(&mut self, row: usize) { self.0[row] = OntologyDelta::default(); }
    pub fn active_deltas(&self) -> impl Iterator<Item=(usize, &OntologyDelta)> {
        self.0.iter().enumerate().filter(|(_, d)| d.kind != DeltaKind::None as u8)
    }
}
```

### OntologyDeltaSink trait (contract extension)

```rust
// crates/lance-graph-contract/src/cognitive_shader.rs
// Extends ShaderSink for AriGraph + callcenter consumers.
pub trait OntologyDeltaSink: ShaderSink {
    fn on_ontology_delta(&mut self, row: u32, delta: &OntologyDelta) -> bool {
        let _ = (row, delta); true
    }
}
```

### AccumulatedOntology (D-E7, driver-side)

Per bindspace-columns-v1.md §11 corrected D-E3 (vertical temporal axis):

```rust
// crates/cognitive-shader-driver/src/driver.rs — field on ShaderDriver
pub struct AccumulatedOntology {
    entity_counts: HashMap<u16, u32>,
    relation_counts: HashMap<u16, u32>,
    triplet_truths: HashMap<(u8, u8, u8), (u8, u8)>,
}
// Methods: classify_entity, classify_relation, has_entity, contradicts, apply_delta
// This is the vertical read-back state: persists across cycles on ShaderDriver
// (NOT per-row — accumulated ontology IS correctly global per plan §11).
```

---

## §3 Column F — AwarenessColumn

**Parent plan ref:** bindspace-columns-v1.md §3 Column F, Phase 3
deliverables D-F1 through D-F10.

**Implementation:** crates/cognitive-shader-driver/src/columns/awareness.rs (NEW)

**Type alias in contract:** pub type AwarenessColumn = [u8; 256]; in
lance-graph-contract::cognitive_shader (or ::distance per plan §4).

**OQ-7 resolution (locked in plan):** Column F STAYS full-BindSpace-width
(256 B/row). Compression would lose per-word granularity, breaking the
AwareOp derivation table.

### Storage field on BindSpace

```rust
// bindspace.rs — Column F field
pub awareness_column: AwarenessColumnStore,  // Box<[u8]>, len * 256 bytes
```

### AwareOp trait (contract-side, D-F2)

```rust
pub trait AwareOp {
    fn awareness_of(result: &[u64; 256], inputs: &[&[u64; 256]]) -> [u8; 256];
}
```

### Awareness derivation per operation (verbatim from plan §3)

| Operation | Derivation | Cost |
|---|---|---|
| vsa_bind(a,b) | per-word: 255 - abs(popcount(word)-32) | 1 popcnt/word |
| vsa_bundle(items) | per-word: max_tally - second_tally | accumulator |
| hamming(a,b) | per-word: 255*(1-abs(hw-32)/32) | 1 popcnt/word |
| cosine(a,b) | scalar: min(n_a,n_b)/max(n_a,n_b)*255 | 0 extra |
| palette_lookup(idx) | 255*(d2-d1)/d2, d1=nearest, d2=2nd | 1 extra lookup |
| cam_pq_decode(code) | 255-(residual_norm*255/max_res).min(255) | ADC |

### Vertical composition (D-F10, verbatim from plan §11)

```rust
// seed_awareness_from_prior — EMA alpha=0.7 recent, 0.3 prior
fn seed_awareness_from_prior(prior: &[u8; 256], current: &mut [u8; 256]) {
    for (cur, &prev) in current.iter_mut().zip(prior.iter()) {
        *cur = ((*cur as u16 * 179 + prev as u16 * 76) / 255) as u8;
    }
}
// Called at START of each dispatch cycle in ShaderDriver::dispatch().
// Driver holds prior_awareness: [u8; 256] (global per ShaderDriver).
```

### Scientific invariants (E1, D1/D2, D3, C1 from plan §3 and §7)

- E1: Column F is a strict sidecar — does NOT modulate bind/bundle
  strength. Kan extension optimality (Shaw 2501.05368) holds because
  awareness is invisible to ring operations.
- D1/D2: Awareness in PARALLEL array, never packed into carrier words.
  VSA algebra operates ONLY on Column A. Column F invisible to algebra.
  Preserves Kleyko et al. bipolar/binary VSA guarantees.
- D3: After unbind, awareness MUST be recomputed from result (D-F8).
  Pre-bind epistemic state ≠ post-unbind state. vsa_unbind calls
  AwareOp::awareness_of(result, &[bundle, key]).
- C1: Column F is pre-NARS signal, NOT replacement for <f,c> revision.
  NARS revision operates at statement level (Column D edge truth).

### AwarenessColumnStore struct (driver-side)

```rust
// crates/cognitive-shader-driver/src/columns/awareness.rs
#[derive(Debug)]
pub struct AwarenessColumnStore(pub Box<[u8]>);

impl AwarenessColumnStore {
    pub fn zeros(len: usize) -> Self {
        Self(vec![0u8; len * 256].into_boxed_slice())
    }
    #[inline]
    pub fn row(&self, row: usize) -> &[u8; 256] {
        self.0[row*256..(row+1)*256].try_into().unwrap()
    }
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [u8; 256] {
        (&mut self.0[row*256..(row+1)*256]).try_into().unwrap()
    }
    pub fn set_row(&mut self, row: usize, awareness: &[u8; 256]) {
        self.0[row*256..(row+1)*256].copy_from_slice(awareness);
    }
}
```

**AwareOp default stub (Phase 2 no-op):** Returns [128u8; 256]
(midpoint = no information) until ndarray D-F4/D-F5 impls land.
This unblocks end-to-end wiring and CI tests without blocking on
the ndarray follow-on PR (see OQ-W4-3 in §13).

**AwareOp deferral — carry-over to sprint-12+, NOT sprint-11 scope:**
The no-op stub `[128u8; 256]` for D-F4/D-F5 (ndarray-backed AwareOp
per-word derivation table) is an **explicit carry-over to sprint-12+**.
Sprint-11 will NOT implement D-F4/D-F5. The stub is intentional and
CI-safe; replacing it with real per-word derivations requires the
vertical streaming structs from D-CSV-11 (sprint-13+), which in turn
depend on the `AdaWorldAPI/ndarray PR #116` hpc-extras upstream merge.
Any sprint-11 PR touching awareness.rs MUST NOT attempt to implement
D-F4/D-F5 — leave the stub intact and document it as TECH_DEBT pending
D-CSV-11. See cognitive-substrate-convergence-v1.md §11 Phase D (D-CSV-11)
and §13.6 for the ndarray coordination requirement.

---

## §4 Column G — ModelBindingColumn

**Parent plan ref:** bindspace-columns-v1.md §3 Column G, Phase 4
deliverables D-G1 through D-G5.

**Implementation:** crates/cognitive-shader-driver/src/columns/model_binding.rs (NEW)

**Type alias in contract:** pub type ModelRef = u32; in
lance-graph-contract::ontology (or ::model — defer to W1).
ModelRegistry + ModelProvider traits also in contract.

### ModelRef and traits (verbatim from plan §3)

```rust
pub type ModelRef = u32;

pub trait ModelRegistry: Send + Sync {
    fn get(&self, model_ref: ModelRef) -> Option<&dyn ModelProvider>;
}

pub trait ModelProvider: Send + Sync {
    fn classify(&self, fingerprint: &[u64; 256]) -> (u8, f32);
}
```

### ModelBindingColumn struct (driver-side)

```rust
// crates/cognitive-shader-driver/src/columns/model_binding.rs
#[derive(Debug)]
pub struct ModelBindingColumn(pub Box<[ModelRef]>);

impl ModelBindingColumn {
    pub fn zeros(len: usize) -> Self { Self(vec![0u32; len].into_boxed_slice()) }
    #[inline] pub fn get(&self, row: usize) -> ModelRef { self.0[row] }
    #[inline] pub fn set(&mut self, row: usize, r: ModelRef) { self.0[row] = r; }
    #[inline] pub fn is_bound(&self, row: usize) -> bool { self.0[row] != 0 }
}
```

LF cross-refs: LF-50 ModelRegistry, LF-51 ModelDeployment, LF-52 LlmProvider.
Column G enables per-row model provenance: "which ONNX model produced
this row's classification?" — architecturally novel, Foundry/Vertex
does not have this (plan §6 table, star rows).

---

## §5 Column H — TypeColumn (completing per-row wiring)

**Parent plan ref:** bindspace-columns-v1.md §3 Column H, Phase 1
deliverables D-H1 through D-H4 — PARTLY SHIPPED (PR #272).

### Shipped state (PR #272)

- EntityTypeId = u16 in lance-graph-contract::ontology (line 81).
- BindSpace.entity_type: Box<[u16]> allocated.
- push_typed(entity_type: u16) builder method.
- driver.rs:308-317 reads entity_type[r] + OntologyRegistry for ctx_id
  and MulThresholdProfile. trust_below_floor check at line 317.
- Tests passing: entity_type_defaults_to_untyped, entity_type_set_and_get,
  builder_push_typed_sets_entity_type.

### This PR's Column H delta

1. TypeColumn named wrapper struct (get/set/zeros — cosmetic consistency).
2. read_column_h_at + write_column_h on BindSpaceView (see §7).
3. FIX-5 trust_below_floor_wiring_per_row_ctx test (see §9 H-1).

### PR 355 #6 close (per-row context_ids)

Current driver.rs:305-315 comment: "per-row context column is
Wave-3.5 follow-up (gate is one-per-dispatch today)."
Current implementation: ctx_id = passed_rows.first().copied()...
(one context per dispatch, first passing row only).

Target: replace with per-row context lookup inside the emission loop
so each emitted edge reads its specific row's Column H (entity_type[r]).
This resolves the Wave-3.5 caveat.

Cross-refs:
- palantir-parity-cascade-v2.md D-PARITY-V2-12 (column extension)
- ogit-cascade-supabase-callcenter-v1.md D-CASCADE-V1-2 (SchemaPtr.context_id,
  already shipped in lance-graph-ontology)
- PR #272 Foundry parity: Column H already at contract level;
  this PR completes Phase 2 semantics.

---

## §6 CollapseGate MergeMode::Superposition

**Status: ALREADY SHIPPED** in collapse_gate.rs line 29:
  Superposition = 2, plus GateDecision::FLOW_SUPER constant.

**This PR's delta for Superposition:**

1. Document semantics in a dedicated block comment in collapse_gate.rs.
2. Wire the Superposition path in dispatch emission loop for Column E
   case (contradictory edges tagged DeltaKind::Contradict).
3. Ship the S-1 Superposition test (see §9).

### Superposition semantics (implementation contract)

From causaledge64-mailbox-rename-soa-v1.md §6 + bindspace-columns-v1.md §3:
"opinions are committed contradictions preserved"

Trigger: two compartments emit edges e1 and e2 where:
- e1 XOR e2 != 0 (they differ at the bit level)
- AND 2-bit truth band (bits 62-63, v2 layout) = 11 (Murky/Dissonant)
  for at least one edge
- AND e2 contradicts a Bundle-committed edge at the same row

Action: preserve BOTH deltas in Column D as separate EdgeColumn slots
(8 slots per row in the ShardedEdgeColumn target layout). Tag Column E
OntologyDelta.kind = DeltaKind::Contradict for the newer edge.

Iron rule compliance (I-SUBSTRATE-MARKOV): Superposition is a
REFINEMENT of Xor — it NAMES the both-preserved case. It does NOT
replace the Markov-preserving Bundle path. Chapman-Kolmogorov holds
on the augmented state space: both edges coexist as separate u64
slots (no semigroup violation). Per plan §3: "2-bit truth Murky-or-
Dissonant when superposition active."

---

## §7 BindSpaceView Accessor (D-CE64-MB-8)

**Parent plan ref:** causaledge64-mailbox-rename-soa-v1.md §5 D-CE64-MB-8.

**Implementation:** crates/cognitive-shader-driver/src/bindspace_view.rs (NEW)
OR deferred to crates/par-tile/src/bindspace_view.rs if W1's par-tile
spec places it there. Coordination note: if par-tile defines
PlanarBorrow<R, C>, BindSpaceView implements it. See OQ-W4-1 in §13.

### BindSpaceView shape (matching plan §5)

```rust
// crates/cognitive-shader-driver/src/bindspace_view.rs

/// Column mask — bitmask over 8 column families (A=bit0 ... H=bit7).
#[derive(Clone, Copy, Debug, Default)]
pub struct ColumnMask(pub u8);

impl ColumnMask {
    pub const ALL: Self = Self(0xFF);
    pub const EFGH_ONLY: Self = Self(0b1111_0000);
    pub const ABCD_ONLY: Self = Self(0b0000_1111);
    #[inline] pub fn has_column_e(&self) -> bool { self.0 & 0b0001_0000 != 0 }
    #[inline] pub fn has_column_f(&self) -> bool { self.0 & 0b0010_0000 != 0 }
    #[inline] pub fn has_column_g(&self) -> bool { self.0 & 0b0100_0000 != 0 }
    #[inline] pub fn has_column_h(&self) -> bool { self.0 & 0b1000_0000 != 0 }
}

/// Zero-copy borrow into shared BindSpace columns.
/// Arc<BindSpace> + Range<usize> rows + ColumnMask filter.
/// No allocation on construction.
#[derive(Clone)]
pub struct BindSpaceView {
    pub columns: Arc<BindSpace>,
    pub rows: Range<usize>,
    pub column_mask: ColumnMask,
}

impl BindSpaceView {
    // --- Read accessors ---
    pub fn read_column_e_at(&self, idx: usize) -> &OntologyDelta {
        debug_assert!(self.rows.contains(&idx));
        self.columns.ontology_column.get(idx)
    }
    pub fn read_column_f_at(&self, idx: usize) -> &[u8; 256] {
        debug_assert!(self.rows.contains(&idx));
        self.columns.awareness_column.row(idx)
    }
    pub fn read_column_g_at(&self, idx: usize) -> ModelRef {
        debug_assert!(self.rows.contains(&idx));
        self.columns.model_binding.get(idx)
    }
    pub fn read_column_h_at(&self, idx: usize) -> EntityTypeId {
        debug_assert!(self.rows.contains(&idx));
        self.columns.entity_type[idx]
    }

    // --- Write-token pattern (single-mutation-point, I1 invariant) ---
    pub fn write_delta_to_e(&self, idx: usize, delta: OntologyDelta)
        -> WriteToken<OntologyDelta>
    {
        debug_assert!(self.rows.contains(&idx));
        WriteToken { row: idx, payload: delta }
    }

    pub fn commit_with_token<T: ApplyToColumn>(
        &mut self,
        token: WriteToken<T>,
        gate: &GateDecision,
    ) -> GateDecision {
        if gate.is_flow() {
            token.payload.apply_to(self, token.row, gate.merge);
        }
        *gate
    }
}

/// Owned write token — consumed exactly once via commit_with_token.
pub struct WriteToken<T> {
    pub(crate) row: usize,
    pub(crate) payload: T,
}

pub trait ApplyToColumn {
    fn apply_to(self, view: &mut BindSpaceView, row: usize, merge: MergeMode);
}
```

The write-token pattern enforces single-mutation-point per I1: no
free functions write BindSpace columns. All writes gate through
commit_with_token which requires a GateDecision::is_flow() check.

---

## §8 BindSpace Struct Extension

File: crates/cognitive-shader-driver/src/bindspace.rs

New column fields added to BindSpace (after entity_type):

```rust
pub ontology_column: OntologyColumn,        // Column E (NEW)
pub awareness_column: AwarenessColumnStore, // Column F (NEW)
pub model_binding: ModelBindingColumn,      // Column G (NEW)
```

Updated zeros(len) initializes all three. Updated byte_footprint() adds:

```rust
let ontology_bytes = self.len * 32;       // Column E
let awareness_bytes = self.len * 256;     // Column F
let model_binding_bytes = self.len * 4;   // Column G
```

Updated bindspace_footprint_adds_columns test asserts 72,069 B/row:

```rust
// 71777 (pre-PR) + 32 (E) + 256 (F) + 4 (G) = 72069
assert_eq!(bs.byte_footprint(), 72_069);
```

---

## §9 Files-to-Touch Table

| File | Status | Change | LOC |
|---|---|---|---|
| crates/lance-graph-contract/src/ontology.rs | Exists | Add OntologyDelta + DeltaKind | +80 |
| crates/cognitive-shader-driver/src/bindspace.rs | Exists | Add E/F/G fields + zeros/footprint | +80 |
| crates/cognitive-shader-driver/src/columns/ontology.rs | NEW | OntologyColumn + AccumulatedOntology stub | +150 |
| crates/cognitive-shader-driver/src/columns/awareness.rs | NEW | AwarenessColumnStore + AwareOp stub | +120 |
| crates/cognitive-shader-driver/src/columns/model_binding.rs | NEW | ModelBindingColumn | +50 |
| crates/cognitive-shader-driver/src/columns/type_column.rs | NEW | TypeColumn named wrapper | +30 |
| crates/cognitive-shader-driver/src/bindspace_view.rs | NEW | BindSpaceView + ColumnMask + WriteToken | +180 |
| crates/lance-graph-contract/src/cognitive_shader.rs | Exists | OntologyDeltaSink trait | +30 |
| crates/cognitive-shader-driver/src/driver.rs | Exists | Per-row Column H ctx loop (PR 355 #6) | +30 |
| crates/cognitive-shader-driver/src/columns/mod.rs | NEW | Module exports | +10 |

**Total: ~760 LOC** (within ~800 LOC target per causaledge64 plan §7).

New module structure:
```
crates/cognitive-shader-driver/src/
  bindspace.rs           (extended)
  bindspace_view.rs      (NEW)
  driver.rs              (extended)
  columns/
    mod.rs               (NEW)
    ontology.rs          (NEW)
    awareness.rs         (NEW)
    model_binding.rs     (NEW)
    type_column.rs       (NEW)
```

---

## §10 Test Plan

### Column E Tests

**E-1: OntologyDelta round-trip**
```rust
#[test]
fn ontology_delta_round_trip() {
    let delta = OntologyDelta {
        entity_type_id: 42, relation_type_id: 7,
        kind: DeltaKind::Extend as u8, pearl_rung: 2,
        frequency: 180, confidence: 200,
        s_idx: 10, p_idx: 5, o_idx: 3, style_ord: 1, temporal: 1234,
        _reserved: [0u8; 18],
    };
    let mut col = OntologyColumn::zeros(4);
    col.set(2, delta);
    assert_eq!(col.get(2).entity_type_id, 42);
    assert_eq!(col.get(2).pearl_rung, 2);
    assert_eq!(col.get(2).kind, DeltaKind::Extend as u8);
    assert_eq!(col.get(0).kind, DeltaKind::None as u8);
}
```

**E-2: Pearl rung filter separates observational from interventional**
```rust
#[test]
fn pearl_rung_filter_separates_obs_interventional() {
    let mut col = OntologyColumn::zeros(3);
    col.set(0, OntologyDelta { pearl_rung: 0, kind: DeltaKind::Confirm as u8,
        ..OntologyDelta::default() });
    col.set(1, OntologyDelta { pearl_rung: 2, kind: DeltaKind::Extend as u8,
        ..OntologyDelta::default() });
    col.set(2, OntologyDelta { pearl_rung: 7, kind: DeltaKind::Contradict as u8,
        ..OntologyDelta::default() });
    let obs: Vec<_> = col.active_deltas()
        .filter(|(_, d)| d.pearl_rung == 0).collect();
    let interv: Vec<_> = col.active_deltas()
        .filter(|(_, d)| d.pearl_rung >= 1).collect();
    assert_eq!(obs.len(), 1);
    assert_eq!(interv.len(), 2);
}
```

**E-3: OntologyDelta is exactly 32 bytes**
```rust
#[test]
fn ontology_delta_is_32_bytes() {
    assert_eq!(std::mem::size_of::<OntologyDelta>(), 32);
}
```

### Column F Tests

**F-1: AwarenessColumnStore round-trip**
```rust
#[test]
fn awareness_column_round_trip() {
    let mut col = AwarenessColumnStore::zeros(3);
    let mut aw = [0u8; 256];
    aw[0] = 200; aw[127] = 128; aw[255] = 42;
    col.set_row(1, &aw);
    assert_eq!(col.row(1)[0], 200);
    assert_eq!(col.row(1)[127], 128);
    assert_eq!(col.row(1)[255], 42);
    assert!(col.row(0).iter().all(|&b| b == 0));
    assert!(col.row(2).iter().all(|&b| b == 0));
}
```

**F-2: Awareness composes through 3 cascade levels via EMA**
```rust
#[test]
fn awareness_composes_through_3_levels() {
    let mut current = [128u8; 256];
    let level1 = [200u8; 256];
    let level2 = [50u8; 256];
    let level3 = [180u8; 256];
    seed_awareness_from_prior(&level1, &mut current);
    seed_awareness_from_prior(&level2, &mut current);
    seed_awareness_from_prior(&level3, &mut current);
    for &b in current.iter() {
        assert!(b >= 40 && b <= 210,
            "awareness must stay bounded after 3-level EMA composition");
    }
}
```

### Column G Tests

**G-1: ModelBindingColumn default unbound + set/get**
```rust
#[test]
fn model_binding_default_unbound_then_set() {
    let mut col = ModelBindingColumn::zeros(4);
    for i in 0..4 { assert!(!col.is_bound(i)); }
    col.set(2, 99);
    assert!(col.is_bound(2));
    assert_eq!(col.get(2), 99);
    assert!(!col.is_bound(0));
}
```

**G-2: ModelRef stability — overwrite replaces, no accumulation**
```rust
#[test]
fn model_ref_overwrite_replaces_not_accumulates() {
    let mut col = ModelBindingColumn::zeros(2);
    col.set(0, 5); col.set(0, 7);
    assert_eq!(col.get(0), 7);
    assert_eq!(col.get(1), 0);
}
```

### Column H Tests (FIX-5)

**H-1: trust_below_floor_wiring_per_row_ctx (FIX-5 closing test)**
```rust
/// FIX-5: trust_below_floor fires per-row when Column H is Healthcare
/// EntityTypeId AND qualia produce Fuzzy TrustTexture.
///
/// Spawn a Healthcare-domain row (entity_type = Patient EntityTypeId 1),
/// attach OntologyRegistry that maps it to Healthcare context_id,
/// set qualia to produce Fuzzy trust, run dispatch on that row,
/// assert gate = HOLD (trust_below_floor fired).
#[test]
fn trust_below_floor_wiring_per_row_ctx() {
    let mut bs = BindSpace::zeros(2);
    let patient_entity_type_id: u16 = 1; // Healthcare Patient, 1-based
    bs.entity_type[0] = patient_entity_type_id;
    let reg = Arc::new(build_healthcare_test_registry());
    bs.set_ontology(reg);
    let mut qualia = [0.0f32; QUALIA_DIMS];
    qualia[0] = 0.2; // low valence -> Fuzzy TrustTexture in MUL
    qualia[1] = 0.2; // low arousal
    bs.qualia.set(0, &qualia);

    let driver = ShaderDriver::new_with_bindspace(bs);
    let dispatch = ShaderDispatch {
        rows: ColumnWindow::new(0, 1),
        ..ShaderDispatch::default()
    };
    let crystal = driver.dispatch(&dispatch);
    assert!(crystal.bus.gate.is_hold(),
        "trust_below_floor must produce HOLD for Healthcare row with Fuzzy trust");
}
```

### BindSpaceView Tests

**V-1: Zero-copy borrow of 100-row range**
```rust
#[test]
fn bindspace_view_borrow_100_rows_no_alloc() {
    let bs = Arc::new(BindSpace::zeros(200));
    let view = BindSpaceView {
        columns: Arc::clone(&bs),
        rows: 50..150,
        column_mask: ColumnMask::ALL,
    };
    let e = view.read_column_e_at(50);
    assert_eq!(e.kind, DeltaKind::None as u8);
    let h = view.read_column_h_at(100);
    assert_eq!(h, 0);
    let g = view.read_column_g_at(75);
    assert_eq!(g, 0);
}
```

### Superposition Test

**S-1: Complementary edges produce Contradict tag on Column E**
```rust
#[test]
fn superposition_preserves_contradictory_edges_with_contradict_tag() {
    let e1 = 0x00_FF_00_FF_00_FF_00_FFu64;
    let e2 = 0xFF_00_FF_00_FF_00_FF_00u64;
    assert_ne!(e1 ^ e2, 0); // precondition

    // When Superposition fires, Column E must carry DeltaKind::Contradict.
    let mut col = OntologyColumn::zeros(1);
    col.set(0, OntologyDelta {
        kind: DeltaKind::Contradict as u8,
        pearl_rung: 0,
        ..OntologyDelta::default()
    });
    assert_eq!(col.get(0).kind, DeltaKind::Contradict as u8,
        "Superposition must tag contradicting edge as DeltaKind::Contradict");
    // Full 8-slot ShardedEdgeColumn storage tested in PR-CE64-MB-5
    // MailboxSoA integration tests (see OQ-W4-2).
}
```

---

## §11 Risk Matrix

| Risk | Severity | Prob | Mitigation |
|---|---|---|---|
| SoA layout regression on Column D padding | HIGH | Low | Gated by W3 edge_column_layout_invariant_64b_per_row test; footprint asserts 72,069 B/row after PR |
| OntologyDelta size drift (non-32 bytes) | HIGH | Low | const _ = assert!(size_of::<OntologyDelta>()==32) panics at compile time |
| Superposition violates I-SUBSTRATE-MARKOV | MED | Low | Superposition refines Xor, does NOT replace Bundle. Chapman-Kolmogorov holds on augmented state space (both edges in separate EdgeColumn slots). |
| Column F composition rate divergence | MED | Med | F-2 test covers 3-level EMA. Same alpha=0.7 formula MUST be identical at all cascade levels. |
| Column H u16 to u32 future widening | LOW | Low | Planned but NOT this PR. Document in type_column.rs. Widening is a future breaking change. |
| AccumulatedOntology HashMap unbounded growth | LOW | Med | Cap max_entity_types in AccumulatedOntology with LRU eviction of least-seen entries (D-E7). |
| BindSpaceView Arc::get_mut exclusivity | MED | Low | Write-token pattern requires CollapseGate exclusive lock before commit_with_token. Panic on Arc aliasing is the correct safety guarantee. |
| AwareOp ndarray impls not in this PR | MED | Certain | Ships stub returning [128u8;256]. Ndarray D-F4/D-F5 are follow-on. No CI blocker. |

---

## §12 Iron Rule Compliance Audit

| Iron Rule | Status | Notes |
|---|---|---|
| I-SUBSTRATE-MARKOV | COMPLIANT | Superposition refines Xor, does NOT replace Bundle on Markov-transition paths. Column F invisible to VSA ring operations (E1). |
| I-NOISE-FLOOR-JIRAK | COMPLIANT | No new noise-floor claims. Column F is per-word heuristic. Future F-floor thresholds must cite Jirak 2016 bounds. |
| I-VSA-IDENTITIES | COMPLIANT | Column F in PARALLEL array, never packed into carrier words (D1). Column E is separate from Column A. Column G and H are typed handles, not VSA representations. |
| I1 (single-mutation-point) | COMPLIANT | Write-token pattern in BindSpaceView::write_delta_to_e / commit_with_token enforces single mutation point. No free functions write columns. |
| Method-on-carrier | COMPLIANT | All new operations are methods on OntologyColumn, AwarenessColumnStore, ModelBindingColumn, BindSpaceView, AccumulatedOntology. Zero free functions. |

---

## §13 Open Questions for Meta-Review

**OQ-W4-1 — BindSpaceView placement (W1 coordination, BLOCKER for MB-5):**
Should BindSpaceView live in crates/par-tile/ (W1 spec) or
crates/cognitive-shader-driver/? If par-tile defines PlanarBorrow<R,C>,
BindSpaceView implements it. Must read W1 spec before finalizing
bindspace_view.rs. This is the primary cross-worker dependency.

**OQ-W4-2 — EdgeColumn 8-slot ShardedEdgeColumn scope:**
Plan references "8 x CausalEdge64 = 64 B/row" for Column D. Current
EdgeColumn is 1 u64/row (8 bytes). Superposition's full storage requires
8 slots. Recommend deferring ShardedEdgeColumn to PR-CE64-MB-5 to avoid
footprint assertion churn in this PR. If deferred: footprint stays 72,069
B/row; ShardedEdgeColumn adds 56 B/row in MB-5 to reach 72,125 B/row.

**OQ-W4-3 — AwareOp stub vs ndarray dependency:**
Should AwareOp have a default no-op impl (returns [128u8;256]) so Column F
is wired end-to-end before ndarray D-F4/D-F5 land? Recommend YES: unblocks
F-2 cascade composition test without blocking on ndarray. The ndarray impls
override the default when they arrive (PR-CE64-MB-5 or separate ndarray PR).
The stub is explicitly scoped to sprint-12+ per the AwareOp deferral note
in §3 above.

---

## §13 Cross-References

**Architectural anchor plan:**
- `.claude/plans/cognitive-substrate-convergence-v1.md` — the substrate plan
  locking the v2 cognitive substrate this spec composes with. Specifically:
  - §5 L-10: QualiaColumn → QualiaI4_16D decision (replaces [f32;18])
  - §7.2: QualiaColumn column-level encoding change + dim-assignment table
    (Wisdom=dim0, Staunen=dim1, Magnitude=dim13; computed via i4×i4→i8 SIMD)
  - §11 D-CSV-5: Migration phasing (Phase 5a sibling-column, Phase 5b cutover)
  - §12 row for pr-ce64-mb-3-bindspace-efgh.md: ~40 LOC patch scope
  - §13.6: ndarray PR #116 coordination requirement for D-CSV-11 (blocks
    AwareOp D-F4/D-F5 real impls, which are sprint-12+ carry-over)

**Parent plans:**
- `bindspace-columns-v1.md` (§1-§5, §7-§11) — column types, storage, invariants
- `causaledge64-mailbox-rename-soa-v1.md` (§6, §7) — PR-CE64-MB-3 LOC estimate

---

## §14 Dependency Graph Position

```
PR-CE64-MB-1 (par-tile crate apex — W1)
  |
  v
PR-CE64-MB-3 (this PR — Columns E/F/G/H)
  |-- uses EntityTypeId (contract::ontology, shipped)
  |-- uses OntologyRegistry (lance-graph-ontology, shipped)
  |-- uses MergeMode::Superposition (collapse_gate, shipped)
  |-- defers BindSpaceView placement to W1 par-tile decision
  |-- closes: PR 355 #6, FIX-5, bindspace-columns-v1 Phase 2
  |
  v
PR-CE64-MB-5 (MailboxSoA + AttentionMaskActor — consumes BindSpaceView)
```

Can land in parallel with: PR-CE64-MB-2 (causal-edge v2, orthogonal
crate), PR-CE64-MB-4 (AriGraph SPO-G, orthogonal crate).

---

*Spec authored by W4 (Sonnet 4.6), sprint-log-10, 2026-05-14.*
*Plans cited: bindspace-columns-v1.md (§1-§5, §7-§11) +*
*causaledge64-mailbox-rename-soa-v1.md (§6, §7).*
*Source files confirmed: bindspace.rs (71,777 B/row footprint),*
*cognitive_shader.rs, collapse_gate.rs (Superposition=2 shipped),*
*driver.rs (FIX-5 target at line 311-317),*
*lance-graph-contract/src/ontology.rs (EntityTypeId at line 81).*
