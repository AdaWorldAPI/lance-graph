# SPO Ontology Storage Format Stack

> **READ BY:** family-codec-smith, palette-engineer, integration-lead, anyone choosing storage format for SPO triples / CausalEdge64 records / witness corpus
>
> **Status:** FINDING (existing formats) + CONJECTURE (selection guidance)

The workspace ships multiple SPO storage formats, each optimized for a different access pattern. This doc maps them, names their codec invariants, and gives selection guidance for the sprint-10 v2 work.

---

## 1. The Format Ladder (cold → warm → hot)

Per `docs/CODEC_COMPRESSION_ATLAS.md`:

```text
Full planes (16Kbit, ρ=1.000)                           ← Cold canonical
  → ZeckBF17 (48 B, ρ=0.982)                            ← Cold projection
  → Base17 (34 B, ρ=0.965)                              ← Warm projection
  → PaletteEdge (3 B, ρ=0.937) / CAM-PQ (6 B, varies)   ← Warm/Hot quantized
  → Scent (1 B, ρ=0.937)                                ← Hot index byte
  → CausalEdge64 (8 B, addressing + truth)              ← Hot dispatch
```

ρ = reconstruction fidelity (Spearman correlation against full planes; see `signed-session-findings.md`). The ladder trades bits for speed: from 16384 bits per fingerprint (lossless) down to 64 bits per causal edge (addressed + truth-aware).

---

## 2. Format 1: 3×16Kbit SPO (lossless cold canonical)

**Structure:** each triple stores three 16384-bit fingerprints — one per S, P, O plane.

```rust
struct LosslessSpo {
    s_plane: Box<[u64; 256]>,  // 16384 bits = 256 × u64 = 2 KB
    p_plane: Box<[u64; 256]>,  // 2 KB
    o_plane: Box<[u64; 256]>,  // 2 KB
}
// Total per triple: 6 KB
```

**Where it lives:** Lance dataset persistence, archival storage, "ground truth" comparison anchor.

**Operations supported:**
- Exact match (Hamming distance via SIMD popcount; ~1-5 µs per pair for 16K bits)
- Reversible bit-set ops (intersection, union, XOR for compositional inference)
- Bipolar projection (sign-extracted to ±1 per bit)
- Slot extraction (extract role-keyed sub-fingerprint via slice indexing)

**Cycle cost:** ~1-5 µs per Hamming comparison (50+ µs for 1000 triples).

**When to use:**
- Cold-load fallback for cascade LEAF stage (per `neighborhood::leaf`)
- Re-encoding validation (proving a quantized format is faithful)
- Cross-format conversion (decompress quantized format → 16K planes → re-quantize differently)

**When NOT to use:**
- Cycle-speed dispatch (way too slow)
- Multi-tenant per-cycle storage (way too big — 6 KB × 1M triples = 6 GB)

---

## 3. Format 2: CAM-PQ (Compressed Approximate Membership Pseudo-Quantization)

Per CLAUDE.md "ndarray Integration Policy" + `contract::cam::CamCodecContract`:

**Structure:** compress each fingerprint via product quantization — 1 codebook per M sub-vector lanes, 256 centroids per codebook.

```rust
struct CamPqCode {
    codes: [u8; M],  // M sub-vector lane assignments, each ∈ [0, 256)
}
// Typical M = 6, giving 6 bytes per fingerprint (vs 2 KB lossless = ~340x compression)
```

**Where it lives:** `ndarray::hpc::cam_pq` (canonical impl), `lance-graph` re-export via `CamCodecContract`.

**Operations supported:**
- ADC (Asymmetric Distance Computation): compare a full f32 query vector to a compressed code in O(M) lookups
- IVF (Inverted File): coarse-quantize into K cells, search only relevant cells
- Symmetric distance: compare two compressed codes via codebook centroid distances

**Cycle cost:** ~50-200 ns per ADC comparison; ~20 ns per IVF cell probe.

**When to use:**
- Witness corpus retrieval ("find me prior witnesses similar to this query")
- Anaphora binding (CAM-PQ-ranked + position-window + morph filter)
- k-NN over discourse-scale corpora (1M-100M witnesses indexed)
- Hot-path SPO similarity when 1-3% accuracy loss is acceptable

**When NOT to use:**
- Exact match required
- Sparse high-entropy distributions where centroids cluster badly
- Workloads where the codebook can't be precomputed (e.g., per-cycle ad-hoc data)

---

## 4. Format 3: bgz17 (Palette-Compressed VSA, 34 B Base17)

Per CLAUDE.md "bgz17" + `crates/bgz17/`:

**Structure:** 17-bit-per-symbol VSA over 4096-codebook centroids:

```rust
struct Base17 {
    code: [u8; 34],  // 17 bits × 16 symbols packed = 272 bits ≈ 34 B
}
```

Each "symbol" is a 17-bit codebook index; 16 symbols per Base17 give a 16-dimensional VSA-style superposition.

**Codec:** 121 tests in `crates/bgz17/` validate roundtrip + composition correctness. PaletteSemiring + PaletteMatrix support sparse-matrix mxm operations on Base17-encoded data.

**Where it lives:** `crates/bgz17/` (standalone, 0 external deps, in workspace `exclude` list per CLAUDE.md).

**Operations supported:**
- Palette composition (256×256 distance-tabulated mxm)
- Sparse CSR/CSC matrix ops with PaletteSemiring
- SIMD batch palette distance
- HHTL cascade integration (per bgz-tensor)

**Cycle cost:** ~50-100 ns per palette mxm (cache-resident distance tables).

**When to use:**
- SPO graph as PaletteMatrix (semiring multiplication for multi-hop traversal)
- Encoding stack intermediate (between full 16K planes and 8-byte CausalEdge64)
- Container packing for I/O (compact wire format)

**When NOT to use:**
- Direct CausalEdge64 substitute (palette indices don't carry NARS truth or Pearl rung)
- Real-time inference where palette assignments are still being learned

---

## 5. Format 4: HHTL-bgz / bgz-hhtl-d (HHTL Cascade Codec)

Per CLAUDE.md "bgz-tensor" + `crates/bgz-tensor/src/hhtl_cache.rs`:

**Structure:** Hierarchical High-Throughput Lookup (HHTL) — multi-level cascaded codec:

```text
Level 1: Scent (1 byte)         ρ=0.937   ← coarsest, 256 distinct values
Level 2: PaletteEdge (3 byte)   ρ=0.937   ← refines via 24-bit palette code
Level 3: Base17 (34 byte)       ρ=0.965   ← Base17 from §4
Level 4: ZeckBF17 (48 byte)     ρ=0.982   ← Zeckendorf-Beta17 projection
Level 5: full 16K planes        ρ=1.000   ← exact (cold)
```

Each level offers progressive precision. The cascade short-circuits early when coarse precision is sufficient.

**HipCache** (`bgz-tensor/src/hhtl_cache.rs`):
- k=64 nearest-neighbor cache per query
- RouteAction enum: `Skip / Attend / Compose / Escalate` per query
- 95% of pairs skip past Scent — only escalate to higher-bit levels when needed

**Where it lives:** `crates/bgz-tensor/` + `lance-graph::neighborhood` (the cascade caller).

**Cycle cost:** 
- HEEL (byte-0 scent only): ~20 µs per neighborhood, 94% precision
- HIP (next 1-2 bytes): ~500 µs, 99% precision
- TWIG (full 8 bytes): ~500 µs, 99.5% precision  
- LEAF (16K planes, cold-load): ~100 µs for ~50 candidates

**When to use:**
- SPO-as-3D-vector neighborhood search (Zone-2 hot path per the three-zone model)
- Progressive-precision queries ("give me close-enough fast; refine if needed")
- Bandwidth-constrained transmission (send Scent first, escalate on demand)

**When NOT to use:**
- Single-edge dispatch (the cascade overhead isn't worth it for 1 edge)
- Workloads where the recall-precision tradeoff at 94% breaks correctness

---

## 6. Format 5: CausalEdge64 (8 B, hot dispatch)

Already documented in `causal-edge-64-spo-variant.md`. Summary in this context:

**Structure:** 8 bytes packed: (S, P, O, f, c, mask, dir, infer, plast, t).

**Cycle cost:** ~50-200 ns for `forward()` (three palette-table lookups + truth math).

**When to use:**
- Cycle-speed BindSpace::EdgeColumn dispatch
- ractor mailbox messages
- AriGraph commit handoff (`promote_to_spo()`)

**When NOT to use:**
- Lossless storage (it's a projection; some info is irrecoverably quantized)
- Discourse corpus retrieval (use CAM-PQ or HHTL instead)

---

## 7. Format Selection Matrix

| Use case | Recommended format | Why |
|---|---|---|
| **Cold archival / ground truth** | 3×16Kbit lossless | Reversible; precision = 1.0 |
| **Cross-cycle persistent triple store** | AriGraph `Triplet` (strings + truth + g + w) | Queryable via Oxigraph SPARQL; commit point |
| **Witness corpus retrieval** | CAM-PQ over Base17 codes | k-NN at ms scale; matches SPOW tetrahedron W-vertex |
| **Multi-hop traversal (graph mxm)** | bgz17 PaletteMatrix | Sparse semiring ops with palette compose tables |
| **Neighborhood search (SPO as 3D vector)** | HHTL cascade (bgz-tensor) | Progressive precision; 94% recall at ~20µs |
| **Cycle-speed BindSpace dispatch** | CausalEdge64 (SPO variant) | 8-byte u64; `forward()` is O(1) per edge |
| **Cognitive cascade interior dispatch** | CausalEdge64 (8-channel variant) | 8 channels × 8 bits; emit/apply pure register ops |
| **Mailbox-to-mailbox message** | CausalEdge64 (SPO variant) wrapped per Σ-tier | Self-contained wire format |
| **Anaphora resolution (Relativpronomen)** | Witness corpus query: CAM-PQ + position-window + morph filter | Composes vector-sort with order-sort with symbolic filter |

---

## 8. How the Formats Map to SPO-G / SPO-W / CausalEdge64

| Format | SPO-G representation | SPO-W representation | CausalEdge64 link |
|---|---|---|---|
| **3×16Kbit lossless** | Three planes per (S, P, O); G is metadata column | W is metadata column (witness ref) | Re-encodes to CausalEdge64 via palette quantization + NARS truth packing |
| **CAM-PQ** | One M-byte code per S, P, O (M typically 6); G filters at IVF cell level | W = corpus-root anchor; CAM-PQ corpus IS the witness store | CausalEdge64 references CAM-PQ entry via W-slot pointer (6-8 bits) |
| **bgz17 Base17** | 34-byte palette VSA per (S, P, O); G is graph-id namespace prefix | W = palette-encoded witness fingerprint | CausalEdge64 = bgz17 Base17 → quantized to 8-byte palette indices + NARS truth |
| **HHTL bgz-hhtl-d** | Multi-level cascade per S, P, O plane; G filters at coarse level | W = highest level not yet escalated (Scent for "approximate witness ref") | CausalEdge64 = result of full cascade (Scent→PaletteEdge→Base17→full) collapsed to 8 B |
| **CausalEdge64 itself** | G implicit (per-tenant SoA partition); explicit G-slot in v2 ALTERNATE proposal | W-slot (6-8 bits) → corpus root anchor → witness chain | This IS the cycle-speed home |

---

## 9. Recommended Format Stack for Sprint-10 v2

Given the corrected hot-path analysis (`causal-edge-64-synergies-and-pr-trajectory.md` §4):

```text
ZONE-1 HOT PATH (cycle-speed, 200-500 ns):
  - thinking-engine 8-channel CausalEdge64 (interior cascade)
  - SPO-palette CausalEdge64 with v2 layout (after transcode at L3 commit)
  - AriGraph entity_index HashMap (O(1) string-keyed lookup)

ZONE-2 WITNESS CORPUS PATH (10s of µs):
  - CAM-PQ codes (M=6 bytes per witness fingerprint)
  - bgz17 Base17 for multi-hop semiring traversal
  - bgz-hhtl-d cascade for progressive-precision retrieval

ZONE-3 PERSISTENT PATH (ms+):
  - AriGraph Triplet (SPO-G + SPO-W combined)
  - Lance dataset for archival (3×16Kbit lossless)
  - DataFusion for analytical queries
```

The v2 CausalEdge64 W-slot points into Zone-2 (CAM-PQ witness corpus); the v2 G is implicit in the SoA partition (Zone-3 AriGraph still carries G explicitly for cross-partition queries).

---

## 10. Cross-references

- `encoding-ecosystem.md` — MANDATORY codec map (read before any codec work)
- `signed-session-findings.md` — ρ values for each format
- `vsa-switchboard-architecture.md` — Three-layer VSA architecture (carrier / role catalogue / content store)
- `causal-edge-64-spo-variant.md` — CausalEdge64 SPO layout
- `spo-schema-and-mailbox-sidecar.md` — SPO-G + SPO-W schema design
- `ogit-owl-dolce-ontology-compartments.md` — How OGIT/OWL/DOLCE map to G partitions
- `cognitive-shader-driver-thinking-engine-reunification.md` — Where the formats are consumed
- `splat-shader-rayon-struct-method-vision.md` — Future shader ops over format stack
- `docs/CODEC_COMPRESSION_ATLAS.md` — Full codec chain documentation

---

*Authored 2026-05-14. Format-ladder ρ values from signed-session-findings.md; ndarray hpc-extras availability per ndarray:claude/burn-A1-dep-gating (open ndarray#116 — not yet merged to master).*
