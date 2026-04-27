# BindSpace Columns E/F/G/H — Integration Plan v1

> **Status:** Active
> **Author:** main thread (Opus 4.7 1M), session 2026-04-26
> **Scope:** Extend BindSpace SoA from 4 → 8 column families to deliver
> Foundry-Vertex parity + inline awareness + ontology enrichment + ONNX binding
> **Scientific review:** 7 SOUND, 7 CAUTION, 0 WRONG (see §7 Risk Matrix)
> **Depends on:** PR #270 (merged), TD-DIST-1 (shipped), TD-AWARENESS-INLINE-1 (queued)

---

## §1 Current State (4 columns)

```
BindSpace SoA (cognitive-shader-driver/src/bindspace.rs):
  A: FingerprintColumns  — [u64; 256] × 3 planes (content/topic/angle) = 6 KB/row
  B: QualiaColumn        — [f32; 18] = 72 B/row
  C: MetaColumn          — MetaWord u32 = 4 B/row
  D: EdgeColumn          — [u64; 8] (CausalEdge64 × 8) = 64 B/row
  ─────────────────────────────────────────────────────────────
  Total per row: ~6,212 bytes
  Rows: 4096 (BindSpace address space)
  Total footprint: ~25 MB (fits L3 cache on Sapphire Rapids)
```

## §2 Target State (8 columns)

```
BindSpace SoA (proposed):
  A: FingerprintColumns  — [u64; 256] × 3 planes           = 6,144 B/row  (unchanged)
  B: QualiaColumn        — [f32; 18]                        =    72 B/row  (unchanged)
  C: MetaColumn          — MetaWord u32                     =     4 B/row  (unchanged)
  D: EdgeColumn          — [u64; 8] (CausalEdge64 × 8)     =    64 B/row  (unchanged)
  E: OntologyColumn      — OntologyDelta                    =    32 B/row  (NEW)
  F: AwarenessColumn     — [u8; 256] (per-word mantissa)    =   256 B/row  (NEW)
  G: ModelBindingColumn  — ModelRef u32                     =     4 B/row  (NEW)
  H: TypeColumn          — EntityTypeId u16                 =     2 B/row  (NEW)
  ─────────────────────────────────────────────────────────────
  Total per row: ~6,578 bytes (+366 bytes, +5.9% overhead)
  Total footprint: ~26.2 MB (still fits L3 cache)
```

## §3 Column Specifications

### Column E — OntologyColumn

**Purpose:** Per-cycle ontology delta. When the shader cycle discovers that
a triplet extends, contradicts, or refines the ontology, the delta is written
here. Downstream consumers (AriGraph, callcenter, q2) read deltas the same
way they read emitted edges.

```rust
/// Per-row ontology delta emitted during the shader cycle.
/// 32 bytes = 256 bits. Fits one cache line with MetaWord + TypeColumn.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct OntologyDelta {
    /// Entity type observed or inferred (0 = no delta).
    pub entity_type_id: u16,
    /// Relation type observed or inferred (0 = no delta).
    pub relation_type_id: u16,
    /// Delta kind: 0=none, 1=confirm, 2=extend, 3=contradict, 4=refine.
    pub kind: u8,
    /// Pearl rung that produced this delta (0=observational, 1-6=do/cf, 7=full-cf).
    /// CRITICAL: must respect observational/interventional separation (§7 B2).
    pub pearl_rung: u8,
    /// NARS frequency of the ontology assertion [0, 255].
    pub frequency: u8,
    /// NARS confidence of the ontology assertion [0, 255].
    pub confidence: u8,
    /// Subject palette index from the producing CausalEdge64.
    pub s_idx: u8,
    /// Predicate palette index.
    pub p_idx: u8,
    /// Object palette index.
    pub o_idx: u8,
    /// Thinking style ordinal that produced this delta.
    pub style_ord: u8,
    /// Temporal index (matches CausalEdge64 temporal field).
    pub temporal: u16,
    /// Reserved for alignment + future use.
    _reserved: [u8; 18],
}
```

**Scientific constraint (B2):** Observational deltas (pearl_rung=0) and
interventional deltas (pearl_rung≥1) MUST be tagged. Downstream consumers
that compute `P(Y|X)` vs `P(Y|do(X))` filter by this field. Without the
tag, the observational/interventional separation required by Pearl's
do-calculus would be violated.

**LF cross-ref:** LF-20 FunctionSpec, LF-23 NotificationSpec, LF-14 Lineage.

### Column F — AwarenessColumn

**Purpose:** Per-word (per-u64) inline awareness mantissa. Computed by every
stream operation as a structural byproduct — bit-purity, distribution shape,
match strength, residual norm. Composes through the cascade.

```rust
/// Per-row awareness: one byte per u64 word in the fingerprint.
/// 256 bytes = one cache line group (4 × 64-byte lines).
///
/// SCIENTIFIC NOTE (D1/D2): awareness lives in a PARALLEL array, not
/// packed into the carrier words. The VSA algebra (bind/unbind/bundle)
/// operates ONLY on Column A. Column F is invisible to the algebra.
/// This preserves Kleyko et al.'s bipolar/binary VSA guarantees.
///
/// SCIENTIFIC NOTE (D3): after unbind, awareness MUST be recomputed
/// from the result, NOT inherited from pre-bind state. Pre-bind
/// epistemic state ≠ post-unbind state.
///
/// SCIENTIFIC NOTE (A3): the Markov state is now (value, awareness),
/// not just value. Chapman-Kolmogorov holds on the augmented space.
pub type AwarenessColumn = [u8; 256];

/// Awareness derivation functions — one per stream operation.
pub trait AwareOp {
    /// Compute awareness annotation for this operation's result.
    /// Returns 0 = no information, 255 = maximum confidence.
    fn awareness_of(result: &[u64; 256], inputs: &[&[u64; 256]]) -> [u8; 256];
}
```

**Awareness derivation per operation:**

| Operation | Awareness derivation | Cost | Grounding |
|---|---|---|---|
| `vsa_bind(a, b)` | per-word: `255 - abs(popcount(word) - 32)` (distance from 50% density) | 1 popcnt/word | Kleyko: balanced density = maximum capacity |
| `vsa_bundle(items)` | per-word: `max_tally - second_tally` (margin of majority vote) | tracked in accumulator | Kleyko: margin = signal strength |
| `hamming(a, b)` | per-word: `255 * (1 - abs(hw - 32) / 32)` (uniformity of difference distribution) | 1 popcnt/word | Jirak: uniform = well-behaved; clustered = suspicious |
| `cosine(a, b)` | scalar: `min(norm_a, norm_b) / max(norm_a, norm_b)` scaled to [0,255] | 0 extra (norms already computed) | Low norm = low info content |
| `palette_lookup(idx)` | `255 * (d2 - d1) / d2` where d1=nearest, d2=2nd nearest | 1 extra lookup | Large gap = unambiguous match |
| `cam_pq_decode(code)` | `255 - (residual_norm * 255 / max_residual).min(255)` | tracked in ADC | Low residual = good reconstruction |

**Scientific constraint (C1/C3):** Per-word awareness is a pre-NARS signal,
NOT a replacement for NARS `<f, c>` truth pairs. NARS revision operates at
the statement level (Column D edge truth). Column F feeds the awareness
*seed* for the next cycle's F-landscape but does not replace the formal
revision step.

**Scientific constraint (E1):** Awareness is strictly a sidecar — it does
NOT modulate bind/bundle strength. The Kan extension optimality (Shaw
2501.05368) holds because awareness is invisible to the ring operations.
If a future design requires awareness-weighted binding, the Shaw proof
must be re-derived for the weighted ring.

**LF cross-ref:** TD-DIST-1 Distance trait (extends with `_with_awareness`),
TD-AWARENESS-INLINE-1.

### Column G — ModelBindingColumn

**Purpose:** Per-row optional reference to an ONNX model for the L4→L1
feedback loop. When set, the shader cycle consults the referenced model
during dispatch (style oracle). When 0, pure algebraic dispatch.

```rust
/// Per-row model binding. 0 = no model (algebraic dispatch only).
/// Non-zero = index into the ModelRegistry (LF-50).
///
/// The ONNX classifier IS a type-system citizen: each row knows which
/// model produced its classification, enabling provenance queries
/// ("which model influenced this edge?").
pub type ModelRef = u32;

/// Trait for model registry providers (LF-50).
pub trait ModelRegistry: Send + Sync {
    /// Look up model by ref. Returns None for ref=0 or unknown ref.
    fn get(&self, model_ref: ModelRef) -> Option<&dyn ModelProvider>;
}

/// Trait for model inference providers (LF-52).
pub trait ModelProvider: Send + Sync {
    /// Classify a fingerprint into a style ordinal + confidence.
    fn classify(&self, fingerprint: &[u64; 256]) -> (u8, f32);
}
```

**LF cross-ref:** LF-50 ModelRegistry, LF-51 ModelDeployment, LF-52 LlmProvider.

### Column H — TypeColumn

**Purpose:** Per-row Foundry "Object Type" link. The entity type this row's
fingerprint belongs to, enabling queries to filter by type without schema
re-parsing.

```rust
/// Per-row entity type binding. 0 = untyped.
/// Non-zero = index into `OntologySpec.entity_types` (contract::ontology).
///
/// This IS the Palantir Vertex "Object Type" equivalent. Every row in
/// BindSpace is typed, enabling Object Explorer scrolling, property
/// view selection (LF-22 ObjectView), and type-filtered search (LF-40).
pub type EntityTypeId = u16;
```

**LF cross-ref:** LF-22 ObjectView (shipped), LF-20 FunctionSpec, LF-40 Search.


---

## §4 Object-Oriented Design — Trait Hierarchy

```
                    Distance (shipped TD-DIST-1)
                        │
                    Aware (NEW)
                    fn awareness(&self) -> Self::Awareness
                        │
         ┌──────────────┼───────────────┐
         │              │               │
     [u64; 256]     [u8; 6]         [u8; 3]
     Binary16K      CamPqCode       PaletteEdge
     Hamming+       L1+             L1+
     bit-purity     residual-norm   gap-strength
         │
     Annotated<T: Aware>
     { value: T, awareness: T::Awareness }
         │
     ┌───┴───┐
     │       │
  Distance   AwareOp
  ::distance_with_awareness()
  → (u32, u8)
```

**Key principle:** `Annotated<T>` is a *product functor* `T × Awareness`,
not a natural transformation (scientific review E2). The forgetful
projection `Annotated<T> → T` IS a natural transformation. Code that
needs bare values calls `.value`; code that needs epistemic context
reads `.awareness`.

**Contract placement:** All traits and types in `lance-graph-contract`
(zero deps). Implementations in ndarray (SIMD kernels) or carrier crates.

---

## §5 Build Order (Wedge Sequence)

Each phase is one PR. Each PR passes clippy + tests before merge.

### Phase 1: Column H (EntityTypeId) — Pure DTO

**Effort:** S (small)
**Blocked by:** nothing
**Unblocks:** LF-22 ObjectView usage, LF-40 type-filtered search

Deliverables:
- D-H1: `pub type EntityTypeId = u16` in `contract::ontology`
- D-H2: `type_id: EntityTypeId` field on `BindSpace` SoA (one `Box<[u16]>`)
- D-H3: `ShaderDriver::dispatch()` writes `type_id` from the matched
         `OntologySpec` entry (or 0 if untyped)
- D-H4: 3 tests (set/get type_id, filter by type, untyped default)

### Phase 2: Column E (OntologyDelta) — Structural Learning

**Effort:** M (medium)
**Blocked by:** Phase 1 (uses EntityTypeId)
**Unblocks:** LF-23 NotificationSpec triggers, SPO Pearl 2³ enrichment

Deliverables:
- D-E1: `OntologyDelta` struct in `contract::ontology` (32 bytes, repr(C))
- D-E2: `ontology: Box<[OntologyDelta]>` field on BindSpace SoA
- D-E3: `ShaderDriver::dispatch()` emits OntologyDelta when the cycle
         discovers a novel triplet pattern (new entity type, new relation,
         contradiction with existing schema)
- D-E4: `pearl_rung` field gates observational vs interventional deltas
         (scientific constraint B2)
- D-E5: `OntologyDeltaSink` trait on `ShaderSink` (parallel to `on_crystal`)
- D-E6: 5 tests (emit confirm, emit extend, emit contradict, pearl_rung
         filtering, empty delta for routine cycles)

### Phase 3: Column F (AwarenessColumn) — Inline Mantissa

**Effort:** L (large — touches ndarray SIMD ops + shader cascade)
**Blocked by:** Phase 2 conceptually; can start in parallel on ndarray side
**Unblocks:** TD-AWARENESS-INLINE-1, principled meta_confidence

Deliverables:
- D-F1: `Aware` trait + `Annotated<T>` in `contract::distance`
- D-F2: `AwareOp` trait in `contract::distance`
- D-F3: `awareness: Box<[u8; 256 * N]>` field on BindSpace SoA
- D-F4: `AwareOp` impls for `vsa_bind`, `vsa_bundle` in ndarray
- D-F5: `AwareOp` impls for `hamming`, `cosine` in ndarray
- D-F6: `Distance::distance_with_awareness()` default method
- D-F7: `ShaderDriver::dispatch()` composes awareness through cascade;
         final awareness integral replaces the current
         `(1 - free_energy.total)` meta_confidence
- D-F8: Awareness recomputation after unbind (scientific constraint D3):
         `vsa_unbind` MUST call `AwareOp::awareness_of(result, &[bundle, key])`
- D-F9: 8 tests (bind awareness, bundle awareness, hamming awareness,
         cosine awareness, compose through 3-step cascade, unbind
         recomputes, meta_confidence from integral, edge case: zero input)

### Phase 4: Column G (ModelRef) — ONNX Binding

**Effort:** M (medium, depends on LF-50/52 trait surface)
**Blocked by:** LF-50 ModelRegistry + LF-52 LlmProvider traits
**Unblocks:** L4→ONNX→L1 feedback loop as a type-system citizen

Deliverables:
- D-G1: `ModelRef = u32` in `contract::ontology` (or `contract::model`)
- D-G2: `ModelRegistry` trait + `ModelProvider` trait in contract
- D-G3: `model_ref: Box<[u32]>` field on BindSpace SoA
- D-G4: `ShaderDriver::dispatch()` consults `ModelRegistry` when
         `model_ref[row] != 0` for style oracle classification
- D-G5: 4 tests (no model = algebraic, model ref resolves, model
         classification influences style, provenance query)

---

## §6 Foundry-Vertex Parity Cross-Reference

| Vertex Feature | Our Column/Trait | Phase | LF | Status |
|---|---|---|---|---|
| **Object Type system** | Column H `EntityTypeId` | 1 | — | NEW |
| **Property views** | `Schema::ObjectView` | — | LF-22 | ✅ shipped |
| **Ontology functions** | `FunctionSpec` | — | LF-20 | queued |
| **Action triggers** | `ActionSpec` | — | — | ✅ shipped |
| **Search (full-text + facets)** | Column H enables type-filtered | 1 | LF-40/41 | queued |
| **Notifications** | Column E emits triggers | 2 | LF-23 | queued |
| **Time travel** | `EntityStore::scan_as_of` | — | LF-31 | queued |
| **Branches / scenarios** | `ScenarioBranch` | — | LF-70/72 | ✅ in-PR |
| **Model registry** | Column G `ModelRef` | 4 | LF-50 | queued |
| **Model deployment** | `ModelDeployment` | 4 | LF-51 | queued |
| **LLM provider** | `LlmProvider` | 4 | LF-52 | queued |
| **Decisions / approvals** | `Approval` workflow | — | LF-60 | queued |
| **Lineage** | Column E per-row provenance | 2 | LF-14 | queued |
| **Ontology learning** ★ | Column E inline delta | 2 | — | NOVEL (beyond Foundry) |
| **Inline confidence** ★ | Column F awareness mantissa | 3 | — | NOVEL (beyond Foundry) |
| **Model provenance** ★ | Column G per-row model ref | 4 | — | NOVEL (beyond Foundry) |

★ = architecturally novel; Foundry/Vertex does not have this.

---

## §7 Scientific Risk Matrix

From the Opus scientific cross-check (Jirak, Pearl, NARS, Kleyko, Shaw):

| # | Claim | Verdict | Risk | Mitigation in this plan |
|---|---|---|---|---|
| A1 | Inline awareness doesn't add new weak dependence | SOUND | — | §3 Column F: awareness is measurable function of carrier bits, no new dependence |
| A2 | 11% overhead fits same cache line | CAUTION | SoA columns are separate allocations → separate prefetch | §2: total +5.9% overhead; cache-line co-location is aspirational, not guaranteed. Benchmark after Phase 3. |
| A3 | CK holds on augmented state (value, awareness) | SOUND | Joint state space is larger | §3: documented in Column F scientific notes |
| B1 | 8 Pearl perspectives map to 7 semirings | CAUTION | Actually 8-to-8 cross-crate | §3 Column E: `pearl_rung` field is 3 bits (8 values); mapping to semirings is per-crate dispatch, not a constraint |
| B2 | Inline ontology violates obs/interventional separation | SOUND (if gated) | Must tag deltas by Pearl rung | §3 Column E: `pearl_rung` field is CRITICAL; D-E4 test enforces it |
| B3 | Do and observe simultaneously | SOUND | Causal mask is the regime selector | §3: CausalEdge64 bits [40:42] already encode the regime |
| C1 | Inline awareness compatible with NARS revision | CAUTION | Per-word ≠ per-statement | §3 Column F: awareness is pre-NARS seed, NOT replacement for `<f,c>` revision |
| C2 | Inline changes revision order | CAUTION | Unquantified impact | Phase 3 D-F9 test: compare meta_confidence with/without inline awareness to measure delta |
| C3 | Per-word truth lacks formal NARS grounding | CAUTION | Engineering heuristic, not formal | Acknowledged in Column F notes; formal paper bridge is future work |
| D1 | 7 bits per word breaks VSA algebra | SOUND | Parallel array, not packed | §3 Column F: separate column, invisible to algebra |
| D2 | Awareness orthogonal to VSA space | SOUND | Correlated but harmless | §3: functionally orthogonal, mathematically correlated |
| D3 | Unbind requires awareness recomputation | CAUTION | Unspecified in original epiphany | §5 Phase 3 D-F8: explicit requirement in deliverables |
| E1 | Kan extension holds with sidecar | SOUND | Only if sidecar doesn't modulate | §3 Column F: strict sidecar; if future weighted-bind, re-derive |
| E2 | `Annotated<T>` is a natural transformation | CAUTION | It's a product functor, not nat-trans | §4: corrected terminology; forgetful projection IS the nat-trans |

**Summary:** 0 WRONG. 7 risks mitigated by design constraints in this plan.
Remaining unmitigated: C2 (revision order impact) — measured empirically in Phase 3.

---

## §8 Pearl 2³ × Semiring Mapping (Corrected)

The scientific review noted the mapping is 8-to-8 (not 8-to-7) and cross-crate.
Corrected table with crate residence:

| Pearl rung | Intervention | Semiring | Crate | What it computes |
|---|---|---|---|---|
| 0 | Observational | HammingMin | `blasgraph` | How similar to what I've seen? |
| 1 | do(S) | XorBundle | `blasgraph` | What changes if S differs? |
| 2 | do(P) | Resonance | `blasgraph` | What changes if P differs? |
| 3 | do(O) | SimilarityMax | `blasgraph` | What changes if O differs? |
| 4 | cf(S') | TruthPropagating | `planner` | Had S been different, would conclusion hold? |
| 5 | cf(P') | NarsTruth | `contract::nars` | Had P been different, would confidence change? |
| 6 | cf(O') | Boolean | `contract::nars` | Had O been different, would edge exist? |
| 7 | Full cf | CamPqAdc | `ndarray::hpc::cam_pq` | Distance in alternative universe's codebook |

**Cross-crate dispatch:** The shader cycle selects the semiring via the
`pearl_rung` field. Rungs 0-3 dispatch to `blasgraph` (in-process graph
algebra). Rungs 4-6 dispatch to contract/planner traits (lightweight
truth-table lookups via `NarsTables`). Rung 7 dispatches to ndarray's
CAM-PQ codec for distance computation.

---

## §9 Semantic Kernel Column Layout

The semantic kernel (Markov + CAM-PQ, per `soa-review.md`) runs across
all 8 columns simultaneously:

```
StreamDto ingress
  │
  ▼
Column A: encode (RoleKey bind → TEKAMOLO → Vsa16kF32 trajectory)
  │
  ├── Column H: type binding (EntityTypeId from OntologySpec)
  │
  ▼
Column C: meta prefilter (MetaWord selects style + rung)
  │
  ▼
Column A×B: cascade (fingerprint compare × qualia distance × FreeEnergy)
  │
  ├── Column F: awareness derivation (per-word mantissa per cascade step)
  │
  ▼
Column D: edge emission (CausalEdge64 per strong hit)
  │
  ├── Column E: ontology delta (if novel pattern detected, tagged by pearl_rung)
  │
  ├── Column G: model consultation (if model_ref ≠ 0, L4→ONNX→L1 feedback)
  │
  ▼
NARS revision (statement-level, Column D truth pairs)
  │
  ▼
Column F integral → meta_confidence (replaces 1 - F.total)
  │
  ▼
Next cycle's F landscape is different (Column F awareness feeds back)
```

---

## §10 Deliverable Summary

| D-id | Phase | Effort | Crate | Description |
|---|---|---|---|---|
| D-H1 | 1 | S | contract | `EntityTypeId = u16` |
| D-H2 | 1 | S | shader-driver | `type_id` field on BindSpace |
| D-H3 | 1 | S | shader-driver | dispatch writes type_id |
| D-H4 | 1 | S | shader-driver | 3 tests |
| D-E1 | 2 | M | contract | `OntologyDelta` struct (32B repr(C)) |
| D-E2 | 2 | S | shader-driver | `ontology` field on BindSpace |
| D-E3 | 2 | M | shader-driver | dispatch emits delta on novel pattern |
| D-E4 | 2 | S | shader-driver | pearl_rung gating (B2 constraint) |
| D-E5 | 2 | S | contract | `OntologyDeltaSink` trait |
| D-E6 | 2 | M | shader-driver | 5 tests |
| D-F1 | 3 | S | contract | `Aware` trait + `Annotated<T>` |
| D-F2 | 3 | S | contract | `AwareOp` trait |
| D-F3 | 3 | S | shader-driver | awareness field on BindSpace |
| D-F4 | 3 | M | ndarray | `AwareOp` for vsa_bind, vsa_bundle |
| D-F5 | 3 | M | ndarray | `AwareOp` for hamming, cosine |
| D-F6 | 3 | S | contract | `distance_with_awareness()` default method |
| D-F7 | 3 | L | shader-driver | compose awareness through cascade |
| D-F8 | 3 | M | ndarray | awareness recomputation after unbind |
| D-F9 | 3 | M | shader-driver | 8 tests |
| D-G1 | 4 | S | contract | `ModelRef = u32` |
| D-G2 | 4 | M | contract | `ModelRegistry` + `ModelProvider` traits |
| D-G3 | 4 | S | shader-driver | model_ref field on BindSpace |
| D-G4 | 4 | M | shader-driver | dispatch consults registry |
| D-G5 | 4 | M | shader-driver | 4 tests |

**Total:** 24 deliverables. Phase 1 = 4 (all S). Phase 2 = 6 (2S + 3M + 1S).
Phase 3 = 9 (3S + 4M + 1L + 1M). Phase 4 = 5 (1S + 2M + 1S + 1M).
