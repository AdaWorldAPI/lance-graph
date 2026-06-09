# Cognitive Write = Round-Trip Substrate Hardening — v1

> **Status:** PLAN (spec). Unblocked except P5 (BLOCKED(C) surrealdb fork coords).
> **Thesis:** Express the cold-path write (mailbox SoA → SPO nodes + CausalEdge64
> edges) as a `codegen_spine::TripletProjection`. Then `roundtrip_eq` over it is
> not a *separate* test — it is the **commit's own gate**. Every commit becomes a
> substrate proof: if the LE bytes / class addressing lost an iota, the commit
> fails. This is the probe that answers the original multi-session question —
> *which member-LE vs container-LE contracts are sound vs brittle in ndarray AND
> lance-graph combined* — by exercising them on real cognitive-cycle data.

## 0. Why this exists (the one-paragraph frame)

The scalar emitter `ExternalMembrane::project(bus, meta) -> CognitiveEventRow`
(lance_membrane.rs:381) *"strips all VSA state"* — it is an audit witness, NOT
round-trippable (you cannot decompile `fp[0]`/`fp[255]` back to the tensor). The
hardening path is the **node/edge** projection, which IS lossless by construction
because it carries only the **exact-LE substrate** (`CausalEdge64` u64 ⊕ `class_id`
u16 ⊕ `MetaWord` u32) and deliberately EXCLUDES the lossy CAM-PQ fingerprint
(certified separately by rank-correlation ρ, never by equality).

## 1. Grounded inventory (file:line evidence)

| Surface | Status | Evidence |
|---|---|---|
| `TripletProjection` + `roundtrip_eq` → `RoundTripFailure` | **[G]** | codegen_spine.rs:107-183 |
| `Triple { s,p,o: String, f,c: f32 }` | **[G]** | codegen_spine.rs:74 |
| `SoaEnvelope` trait (columns/row_stride/n_rows/cycle/as_le_bytes/verify_layout) | **[G]** | soa_envelope.rs:139 |
| `SoaEnvelope` **real implementors** | **[H]** | ZERO — only `TestEnvelope` in tests |
| `ColumnDescriptor{name_id:u16,kind,elems_per_row:u16,row_offset:u32}` / `ColumnKind` | **[G]** | soa_envelope.rs:54-97 |
| `MailboxSoaView`: `edges_raw()->&[u64]`, `meta_raw()->&[u32]`, `class_id()->&[u16]`, `energy()->&[f32]` | **[G]** | soa_view.rs:28-105 |
| `MailboxSoaOwner: MailboxSoaView` (`try_advance_phase`, Rubicon DAG) | **[G]** | soa_view.rs:112 |
| `commit_event(row)->u64` sole-writer (ticks version) | **[G]** | lance_membrane.rs:315 |
| Two-layer gate `CommitFilter` + `MembraneGate` (write unconditional, fan-out gated) | **[G]** | lance_membrane.rs:415-429 |
| CausalEdge64 NARS: freq 10b ×1023 [19:10], conf 10b ×1023 [9:0] → tol = 1/1023 | **[G]** | ndarray causal_diff.rs:143-169 |
| `MailboxSoA<N>` owner lives in `cognitive-shader-driver` | **[G]** | soa_view.rs:8 |
| KausalSpec DO-enforcement runtime (fires odoo pragmatics on commit) | **[ABSENT]** | OGAR state_machine in ractor_actors, not wired |
| surreal_container SurrealQL read glove | **[BLOCKED(C)]** | surreal_container/src/lib.rs:30 (fork coords) |

## 2. Maps

### MAP 1 — the round-trip (the whole point)

```
   WRITE  (project = encode)                 READ-BACK (decompile = decode)
   Vec<Triple>                               Vec<Triple>
      │  intern (s,p,o) → ids (dict)            ▲  dict reverse: ids → (s,p,o)
      │  quantize (f,c) → NARS bits             │  dequantize NARS → (f,c)
      ▼                                         │
   CausalEdge64 u64  +  class_id u16  ──────────┘
      │  lay out via MAILBOX_COLUMNS
      ▼
   SoaEnvelope LE bytes  ── commit_event (sole-writer, tick version) ──► Lance

   roundtrip_eq:  in(s,p,o) == out(s,p,o)        [exact,  truth_tolerance ignored for identity]
                  in(f,c)   ≈  out(f,c)          [tol = 1/1023, the CausalEdge64 NARS grid]
        PASS ⟹ substrate sound for this cycle | FAIL ⟹ the brittle contract, NAMED
```

### MAP 2 — SoA column → SPO role → cold byte → read-back

| MailboxSoaView accessor | LE member | name_id | ColumnKind | SPO role | round-trips? |
|---|---|---|---|---|---|
| `edges_raw() -> &[u64]` | `CausalEdge64` | Edge=1 | U64 | **(S,P,O) + NARS** — one packed triple | **exact** (s,p,o) / **±1/1023** (f,c) |
| `class_id() -> &[u16]` | `EntityTypeId` | ClassId=3 | U16 | subject's **class** (OGIT/OGAR shape) | **exact** |
| `meta_raw() -> &[u32]` | `MetaWord` | Meta=2 | U32 | thinking/awareness modal (not SPO identity) | **exact** |
| `energy() -> &[f32]` | spatial-temporal accum | Energy=0 | F32 | not SPO — provenance scalar | **exact** (bit) |
| (excluded) fingerprint tensor | CAM-PQ | — | — | lives in ShaderBus ONLY | **NO** → ρ-cert |

### MAP 3 — failure-mode → brittle contract (the diagnostic value)

| roundtrip_eq failure | What broke | Layer |
|---|---|---|
| `missing/extraneous (s,p,o)` ≠ 0 | `CausalEdge64` pack/unpack OR `SymbolDict` not a bijection | member-LE **or** register |
| `(f,c)` off by > 1/1023 | NARS quantization grid mismatch (two CausalEdge64s disagree) | member-LE |
| `verify_layout()` `StrideMismatch`/`ColumnOverlap`/`OutOfBounds` | `MAILBOX_COLUMNS` descriptor table wrong | container-LE (envelope) |
| `PacketSizeMismatch` | `as_le_bytes().len()` ≠ stride×rows (torn snapshot / wrong N) | container-LE |
| `LayoutVersionMismatch` | v1 packet under v2 reader | I-LEGACY-API version gate |

### MAP 4 — three layers, two witnesses (do NOT conflate)

| Layer | Member | Witness | Equality? |
|---|---|---|---|
| exact-LE members | `CausalEdge64`/`EpisodicEdges64`/`MetaWord` `to_le_bytes` | `roundtrip_eq` tol=0 | yes (proven sound) |
| container envelope | `SoaEnvelope` geometry (zero impls today) | `roundtrip_eq` once P1 lands | yes (provable) |
| lossy codec | CAM-PQ fingerprint (ρ<1.0 by design) | Pearson/Spearman ρ (certification-officer) | **no — tolerance/correlation, NOT round-trip** |

### MAP 5 — THINK/DO (Semantik/Pragmatik) both round-trip as triples

```
  Class (shape, subClassOf)      ──► triples: (subj rdf:type ogit:ObjectType), (field depends_on …)   THINK
  ActionDef (DO, object_class→Class, OdooMethodKind, KausalSpec) ──► triples: (amount_total emitted_by _compute_amount)   DO
        │                                                                          │
        └─────────────── both are just Triples in the projection ─────────────────┘
   ⟹ round-trip proves the DO-*declarations* survive the write — WITHOUT needing the
     [ABSENT] KausalSpec runtime that *fires* them. Harden now; enforce later.
```

## 3. Type spec (compiles in the head)

### P0 — contract, zero-dep: the mailbox's canonical column catalogue

```rust
// crates/lance-graph-contract/src/soa_envelope.rs  (ADD — enum + const only, no deps)

/// Stable per-row column identity for the mailbox SoA envelope.
/// Discriminant == ColumnDescriptor.name_id. Append-only (I-LEGACY-API).
#[repr(u16)]
pub enum MailboxColumn { Energy = 0, Edge = 1, Meta = 2, ClassId = 3 }
// future (deferred-accessor pattern, soa_view.rs:71-95): Qualia = 4, EpisodicWitness = 5

/// Canonical packed geometry. Physical order = u64-alignment-first; name_id = logical id.
/// summed = 8+4+4+2 = 18 = stride (verify_layout() asserts this).
/// If alignment padding is ever needed, model it as an explicit `Pad` U8 column so
/// `summed == stride` still holds — never an implicit gap.
pub const MAILBOX_COLUMNS: [ColumnDescriptor; 4] = [
    ColumnDescriptor { name_id: 1, kind: ColumnKind::U64, elems_per_row: 1, row_offset: 0  }, // Edge   [0,8)
    ColumnDescriptor { name_id: 0, kind: ColumnKind::F32, elems_per_row: 1, row_offset: 8  }, // Energy [8,12)
    ColumnDescriptor { name_id: 2, kind: ColumnKind::U32, elems_per_row: 1, row_offset: 12 }, // Meta   [12,16)
    ColumnDescriptor { name_id: 3, kind: ColumnKind::U16, elems_per_row: 1, row_offset: 16 }, // Class  [16,18)
];
pub const MAILBOX_ROW_STRIDE: usize = 18;
```

### P1 — cognitive-shader-driver: SoaEnvelope over the real backing store (zero-copy)

```rust
impl<const N: usize> SoaEnvelope for MailboxSoA<N> {
    fn columns(&self)    -> &[ColumnDescriptor] { &MAILBOX_COLUMNS }
    fn row_stride(&self) -> usize { MAILBOX_ROW_STRIDE }
    fn n_rows(&self)     -> usize { self.n_rows() }            // MailboxSoaView::n_rows
    fn cycle(&self)      -> u32   { self.current_cycle() }     // MailboxSoaView::current_cycle
    fn as_le_bytes(&self)-> &[u8] { self.backing_le_slice() }  // R1: pointer INTO store, NO re-encode
}
// DoD assert (mirrors soa_view.rs:222): as_le_bytes().as_ptr() == backing.as_ptr()
```

### P2 — lance-graph: the cognitive write AS a lossless projection

```rust
// crates/lance-graph/src/graph/cognitive_write.rs  (NEW — has causal-edge + contract deps)
use lance_graph_contract::codegen_spine::{Triple, TripletProjection};
use lance_graph_contract::soa_envelope::{ColumnDescriptor, MAILBOX_COLUMNS, MAILBOX_ROW_STRIDE};

pub struct CognitiveWriteProjection;

/// The const form: the committed cold packet (structural) + the IRI↔id bijection (symbolic).
/// Splitting these SEPARATES the two failure modes (MAP 3).
#[derive(Clone)]
pub struct CognitiveCommitPacket {
    pub envelope: Vec<u8>,             // SoaEnvelope::as_le_bytes() snapshot — exact-LE substrate
    pub columns:  Vec<ColumnDescriptor>,
    pub row_stride: usize,
    pub n_rows:   usize,
    pub cycle:    u32,
    pub dict:     SymbolDict,          // register, NOT VSA (I-VSA-IDENTITIES Test 0)
}

/// IRI ↔ dense-id bijection. A HashMap/BTreeMap because (s,p,o) are exact-match keys —
/// the register, not a fingerprint bundle (I-VSA-IDENTITIES: "lazy VSA check").
#[derive(Clone, Default)]
pub struct SymbolDict { iri_to_id: std::collections::BTreeMap<String, u32>, id_to_iri: Vec<String> }

impl TripletProjection for CognitiveWriteProjection {
    type Const = CognitiveCommitPacket;

    /// CausalEdge64 stores NARS as 10-bit freq/conf ×1023 (ndarray causal_diff.rs:143-169).
    /// GROUNDING TODO @P2: confirm crates/causal-edge CausalEdge64 uses the SAME grid;
    /// if it differs, set this to that crate's LSB. A mismatch here is itself a finding.
    fn truth_tolerance() -> f32 { 1.0 / 1023.0 }

    // project:   for each Triple → intern (s,p,o)→ids, pack CausalEdge64(s,p,o, f*1023, c*1023)
    //            → edges_raw u64; class_id = subject's class id; lay out via MAILBOX_COLUMNS.
    fn project(triples: &[Triple]) -> CognitiveCommitPacket { /* encode */ unimplemented!() }

    // decompile: walk edges_raw u64 → CausalEdge64::from(raw) → (s_id,p_id,o_id, freq,conf)
    //            → dict reverse → IRIs, NARS/1023 → (f,c) → Triple.
    fn decompile(c: &CognitiveCommitPacket) -> Vec<Triple> { /* decode */ unimplemented!() }
}
```

## 4. Phase plan (each phase = one landable PR, DoD-gated)

| Phase | Crate | Deliverable | DoD | Invariant respected |
|---|---|---|---|---|
| **P0** | lance-graph-contract | `MailboxColumn` ordinals + `MAILBOX_COLUMNS` + `MAILBOX_ROW_STRIDE` | a `verify_layout()` unit test over the const table is green; **no new deps**; clippy/fmt clean | BBB zero-dep |
| **P1** | cognitive-shader-driver | `impl SoaEnvelope for MailboxSoA<N>` (zero-copy) | `as_le_bytes().as_ptr() == backing.as_ptr()`; `verify_layout()` green; round-trips a hand-built SoA | R1 (one SoA never transformed) |
| **P2** | lance-graph | `CognitiveWriteProjection : TripletProjection` + `SymbolDict` | `roundtrip_eq` PASSES on the `account.move` fixture; a deliberately-corrupted CausalEdge64 pack FAILS it (negative test, cf. `LossyDropFrequency`); (f,c) within 1/1023; **grounding-TODO closed** (causal-edge NARS grid confirmed) | I-VSA-IDENTITIES (dict=register; codec excluded) |
| **P3** | lance-graph-callcenter | `project_graph` sibling of `ExternalMembrane::project` → emits `NodeRecord`/`EdgeRecord` through `commit_event` + `CommitFilter` + `MembraneGate` | a committed cycle is queryable as nodes/edges via `MetadataStore` (DataFusion); version ticks; RBAC/tenant fan-out applies | sole-writer; gate unchanged |
| **P4** | callcenter + shader-driver | auto-fire on Rubicon `Committed` (`try_advance_phase` → `project_graph` → `commit_event`), with `roundtrip_eq` as the commit's own debug-gate | advancing a mailbox to `Committed` emits the graph commit; **every commit runs the round-trip assertion** (debug build) — a lossy commit panics in test, is logged in release | I-SUBSTRATE-MARKOV (read→owned→gated write) |
| **P5** | surreal_container | SurrealQL read glove over the committed envelope (nodes/edges) | DEFERRED | — |

**P5 is the ONLY fork-gated phase.** P0–P4 need no surrealdb coords. P5 is
**BLOCKED(C)** on the `AdaWorldAPI/surrealdb` fork git URL + branch/tag + `kv-lance`
feature name — a human input (P0 rule: do not guess fork coords).

## 5. Invariants & risk register

- **BBB zero-dep:** P0 is enum + const ONLY in the contract crate. The projection
  (P2) needs `causal-edge` → it lives in `lance-graph` (which has both deps), NEVER
  in the contract. `SymbolDict` uses `BTreeMap` (std), no alloc-heavy dep.
- **R1 (one SoA never transformed):** P1's `as_le_bytes()` returns a borrow into the
  backing store. If it ever `.to_vec()`s, R1 is violated and the "commit = visibility
  flip, not a copy" property is lost. The pointer-equality DoD assert is the guard.
- **I-VSA-IDENTITIES:** the `SymbolDict` is the register (exact-match lookup). The
  CAM-PQ fingerprint is EXCLUDED from the packet — bundling identities not content.
  If a future hand reaches for VSA-cosine to resolve an IRI, that's register laziness.
- **I-SUBSTRATE-MARKOV:** the write path stays single-writer-gated. `project_graph`
  READS the SoA (`MailboxSoaView`, `&self`), builds an OWNED packet, and `commit_event`
  is the gated write-back. No `&mut` during compute; no second writer (the three-wall
  result). A SurrealQL `UPDATE` against the live mailbox remains **[REJECT]**.
- **I-LEGACY-API version gate:** `ENVELOPE_LAYOUT_VERSION` (=1) is stamped; a v1 packet
  under a v2 reader is refused by `verify_layout()` before mis-decode. Reclaiming a
  `name_id` requires a version bump + field-isolation tests.

## 6. The decoupling (why this is buildable NOW)

The odoo DOs are **declared as triples** (`emitted_by`/`depends_on` edges). The
round-trip proves *those declaration edges survive the write* — it does **not**
require the `KausalSpec` enforcement runtime (**[ABSENT]**) that *fires* the compute/
validate at commit time. So: **harden the substrate now** (P0–P4, round-trip the
declarations) and **build the firing runtime later** — they do not block each other.
"Shape inherits; behavior composes; and the *declarations* of both are round-trippable
before either *executes*."

## 7. Provenance (grounded this session)

codegen_spine.rs:74,107-183 · soa_envelope.rs:54-97,139 · soa_view.rs:8,28-105,112,222 ·
lance_membrane.rs:315,356,381,415-429 · external_intent.rs:113 · version_watcher.rs ·
ndarray causal_diff.rs:143-169 · surreal_container/src/lib.rs:30 · sla.rs:31-101.
