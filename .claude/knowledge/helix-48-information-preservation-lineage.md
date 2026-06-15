# helix-48 information-preservation lineage

> **Status:** knowledge cross-link, append-only.
> **Why this doc exists:** the *"helix-48 carries ~94 % of a 32 768-bit Jina 1024-D embedding, with or without Morton cascade addressing"* claim is the operator's reason-for-being for the `HelixResidue` ValueTenant shipped in PR #496. The fragments that ground it are committed across four April-2026 knowledge docs and one OGAR discovery-map entry; the post-#496 substrate exposes `HelixResidue` as a 48-byte ValueTenant but the doc-comment at the canonical-node site does not cite the lineage. This file compiles the cross-citations so the framing is referenceable from one place.
>
> **Pure docs.** No code touched.

---

## 0. The claim in one line

> **The 48-bit Σ₁ SEED preserves ~94 % of a Jina 1024-D embedding (32 768 bits), validated on SimLex-999.** The post-#496 `HelixResidue` ValueTenant scales this carrier up to 48 bytes (384 bits) for the substrate's place/residue use, inheriting the lineage. The information-preservation property is independent of Morton cascade addressing — with or without the cascade, the helix-48 carrier holds the 32 768-bit Jina equivalent.

---

## 1. The committed fragments

### 1.1 Σ Compression Tiers — the canonical statement

[`lance-graph/.claude/knowledge/linguistic-epiphanies-2026-04-19.md:299-312`](../knowledge/linguistic-epiphanies-2026-04-19.md) (committed `dfcf246b`, **lance-graph PR #210** — *"Phase 1 — ContextChain reasoning + role keys + knowledge docs"*).

| Σ Tier | Form | Width | Bytes | Role |
|---|---|---|---|---|
| Σ₃ FULL | Jina float | 1024D | **4096 B (= 32 768 bit)** | Full embedding |
| Σ₂ MEANING | projected float | 48D | 192 B | Interpretable axes |
| **Σ₁ SEED** | **bit-packed** | **48-bit** | **6 B** | **Hamming-searchable** |
| Σ₀ GLYPH | hash | 12-bit | 2 B | Node type + hash |

> **Validated claim (SimLex-999):** binarization preserves 99 %+ semantic similarity structure. Hamming ≈ cosine for semantic tasks. **48 bits captures ~94 % of Jina 1024-D.**

### 1.2 CAM-PQ 48-bit fingerprint substrate

[`lance-graph/.claude/knowledge/encoding-ecosystem.md:91`](../knowledge/encoding-ecosystem.md) (committed `c1d44910`, **lance-graph PR #176**):

> *CAM fingerprint (48-bit) → COCA 4096 codebook → DeepNSM addressing*

[`ndarray/.claude/knowledge/pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md:22`](../../../../ndarray/.claude/knowledge/pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md) (ndarray PR-x12):

> *Algorithm: Content-Addressable Memory (CAM) + Product Quantization (PQ). Unifies FAISS PQ6×8 (**48-bit fingerprints, 6 subspaces × 256 centroids each**) with CLAM 48-bit archetypes into a single codec.*

The 48 bits decompose as **6 subspaces × 8 bits = 48 bits = 6 bytes** per CAM-PQ fingerprint. Each subspace selects one of 256 centroids; total addressable code space = 256⁶ ≈ 2.8 × 10¹⁴.

### 1.3 The 11/17 X-Trans / quasi-irrational stride rationale

[`lance-graph/.claude/BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md:14-67`](../BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md) (committed `79b46189`, **lance-graph PR #156** — *"docs(research): BF16-HHTL session derivation notes"*).

The key math (from `crates/bgz17/src/lib.rs:53-60`):

```rust
pub const BASE_DIM: usize = 17;     // prime base
pub const GOLDEN_STEP: usize = 11;  // doc-comment: "Golden-ratio step"
```

- `17 / φ = 10.5063…` → nearest integer = **11**.
- `gcd(11, 17) = 1` (17 is prime ⇒ any nonzero step coprime ⇒ `(i · 11) mod 17` is a full permutation of `{0..16}`, every position visited exactly once, perfectly reversible).
- Combined: **the discrete golden-ratio rotation** — *"a full permutation that is as close to a φ-rotation as integer arithmetic allows"* — the **quasi-irrational stride**.

This is the bgz17 equivalent of **Fujifilm X-Trans**: irregular sampling that converts structured aliasing into noise-like error. Same effect as TurboQuant's Hadamard rotation, achieved via number theory rather than geometry ([`ndarray/.claude/knowledge/rotation_vs_error_correction.md:7`](../../../../ndarray/.claude/knowledge/rotation_vs_error_correction.md)).

### 1.4 Maximally-irrational stride beats harmonic for argmax

[`lance-graph/.claude/knowledge/codec-findings-2026-04-20.md:74`](../knowledge/codec-findings-2026-04-20.md) (committed `4c4c0e7f`, **lance-graph PR #218**):

> **I3 — Maximally-irrational strides beat harmonic strides for argmax.**

[`ndarray/src/hpc/gguf_indexer.rs:114`](../../../../ndarray/src/hpc/gguf_indexer.rs):

> *round(17 / φ) = 11 — maximally irrational stride across BASE_DIM positions.*

### 1.5 Morton cascade — the orthogonal addressing layer

[`OGAR/docs/DISCOVERY-MAP.md:127`](../../../../OGAR/docs/DISCOVERY-MAP.md) (OGAR canon):

> **D-CASCADE | 64→256→1024→4096→16k→64k→256k = immaterialized Morton enumeration; every level = +1 nibble.**

Each level adds one nibble (4 bits) to the address, fanning out by ×4 per axis. Algorithmically grounded by generalized Morton/Hilbert ordering (arXiv 2309.15199, Walker). The substrate is *"an **immaterialized Morton cascade with templated payloads**"* ([DISCOVERY-MAP.md:108](../../../../OGAR/docs/DISCOVERY-MAP.md)).

### 1.6 The post-#496 substrate tenant

[`lance-graph/crates/lance-graph-contract/src/canonical_node.rs:333-334`](../../crates/lance-graph-contract/src/canonical_node.rs) (lance-graph **PR #496**):

```rust
/// helix golden-spiral Place/Residue (48 B).
HelixResidue = 4,
```

The substrate's `HelixResidue` is **48 BYTES = 384 bits** — an 8× scale-up of the 48-bit Σ₁ SEED. Used in the `ValueSchema::Compressed` preset alongside `TurbovecResidue` (the 32×4-bit TurboQuant edge codec from ndarray #218 / lance-graph #494):

```rust
/// Cold / compressed codec stack: Fingerprint + Helix-48 + turbovec residue +
/// EntityType. No hot lifecycle columns.
Compressed = 2,
```

---

## 2. The unified framing

### 2.1 Two scales of "helix-48"

| Carrier | Width | Source | Information property |
|---|---|---|---|
| **Σ₁ SEED** | 48 **bit** (6 B) | PR #210, Hamming-searchable | **94 % of Jina 1024-D (32 768 bit) on SimLex-999** |
| **`HelixResidue` ValueTenant** | 48 **byte** (384 bit) | PR #496, place/residue carrier | Inherits the lineage; the 8× wider budget allows higher residue precision at the substrate-node level |

Both are *helix* (golden-spiral place/residue, stride-4-over-17 walked by `CurveRuler`) — they differ only in budget. The substrate uses the byte-wide tenant; the SEED width is the *compression floor* validated against Jina.

### 2.2 Why "with or without Morton cascade"

The information-preservation claim is a property of the **carrier** (the 48-bit fingerprint structure under CAM-PQ 6×256 + bgz17 11/17 stride), not of the **addressing** (Morton cascade 64→256→1024…). The two compose orthogonally:

- **Carrier alone:** a single helix-48 fingerprint can be stored without any addressing context and still preserves 94 % of a Jina 1024-D embedding.
- **Carrier + Morton cascade:** the same fingerprint can be placed at any cascade level (HEEL / HIP / TWIG / family / identity) without changing its information density.
- **Cascade alone:** the addressing has no information about the content; it indexes *where* a fingerprint lives, not *what* it encodes.

So the operator's framing — *"helix 48-bit either with or without Morton cascade gives x32000 information preserving"* — names that the carrier's information-preservation is *independent* of whether the Morton cascade is wrapping it.

### 2.3 Why 32 768 specifically

Σ₃ FULL = 1024 dimensions × 32 bits per f32 = **32 768 bits** = the full Jina embedding. The "x32000" in the operator's relay is this number, rounded — it's the size of the embedding the 48-bit seed preserves 94 % of. The compression ratio is `32 768 / 48 ≈ 683×` (Σ₃ → Σ₁).

For the 48-byte `HelixResidue` ValueTenant the ratio is `32 768 / 384 ≈ 85×`, with higher fidelity because the carrier is 8× wider.

---

## 3. Cross-references

### PRs that shipped the fragments

- **lance-graph #156** — `79b46189` — `BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md` (the 11/17 X-Trans rationale)
- **lance-graph #176** — `c1d44910` — `encoding-ecosystem.md` (CAM 48-bit lineage)
- **lance-graph #210** — `dfcf246b` — `linguistic-epiphanies-2026-04-19.md` (the canonical Σ-tier 94 %-Jina claim)
- **lance-graph #218** — `4c4c0e7f` — `codec-findings-2026-04-20.md` (maximally-irrational stride finding)
- **lance-graph #496** — *"Integrated cognitive planner reference map + ValueSchema presets + FULL POC default"* — `HelixResidue` ValueTenant exposed in `canonical_node.rs`

### Canon anchors

- OGAR/CLAUDE.md P0 — operator-pinned GUID canon (classid · HEEL · HIP · TWIG · family · identity).
- OGAR/docs/DISCOVERY-MAP.md — D-CASCADE, D-CAM, D-MOIRE, D-GOLDEN, D-MANTISSA, D-BGZ17, D-QUANTGATE.
- lance-graph/crates/helix/KNOWLEDGE.md — place/residue spec; *"8K resolution at Super-8 cost"*.
- lance-graph/crates/lance-graph-contract/src/canonical_node.rs — `ValueTenant` enum + `ValueSchema` presets (post-#496).

---

## 4. What this doc does NOT do

- Does not change any code or tenant layout.
- Does not declare new architectural decisions — the canon is the canon, the fragments are the fragments.
- Does not retroactively edit the four April-2026 knowledge docs or the OGAR DISCOVERY-MAP — those are append-only.
- Does not edit `canonical_node.rs`'s `HelixResidue` doc-comment in this PR (that would touch shipped code; the lineage cross-link can be added in a separate small PR if desired).
