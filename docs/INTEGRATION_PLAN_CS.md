# Integration Plan: Cognitive Shader + BindSpace Address Space

> Updated 2026-04-18 with correct BindSpace semantics.

## BindSpace = Read-Only Shared Memory

BindSpace is NOT a mutable database. It's a **read-only fingerprint substrate**
where writers hold owned microcopies and merge back via gated protocols.

### The Three Data Patterns

| Pattern | Ownership | Mutation | Example |
|---|---|---|---|
| **Slice window** | `&[T]` borrowed, N-aligned | SIMD batch read | F32x16 / U8x64 `array_window` |
| **Microcopies** | Owned `Copy` values | Stack-allocated | CausalEdge64, Band, TruthValue |
| **Write-back** | Through gate | XOR or Bundle | Spine update, consolidation |

**Slice-of-array for SIMD batching** is the third leg:
- SIMD units need 16 (F32x16) or 64 (U8x64) contiguous elements
- `array_window(data, N)` yields aligned chunks
- Zero-copy: the window IS a `&[T]` view into the column
- One cascade level = one `array_window` iteration pattern
- Hamming popcount: `&[u64]` windowed as `U64x8` (AVX-512 VPOPCNTDQ)
- Base17 L1: `&[i16]` windowed as `I16x32`
- Palette lookup: `&[u8]` windowed as `U8x64`

**The same object, sliced at multiple SIMD widths** — the 256×256 palette
semiring is ONE table (65,536 entries, 64-byte aligned) but must be
addressable at the three native SIMD widths simultaneously:

```rust
pub struct PaletteTable {
    // One backing store, 64-byte aligned for AVX-512
    data: Arc<[u64; 8192]>,  // 65,536 bytes = 256×256 u8 distances
}

impl PaletteTable {
    // Same bytes, three SIMD views:

    pub fn as_u8x64(&self) -> &[U8x64; 1024]    // byte lookups (distance)
    pub fn as_f16x32(&self) -> &[F16x32; 1024]  // half-precision compose
    pub fn as_f32x16(&self) -> &[F32x16; 2048]  // single-precision combine
    pub fn as_f64x8(&self) -> &[F64x8; 4096]    // double for calibration
}
```

No conversion. No copy. The same contiguous bytes reinterpreted through
different SIMD lane-width views. The CognitiveShader picks the lane
width per op:
- Distance lookup → U8x64 (palette index to u8 distance, 64 at a time)
- Soft compose → F16x32 (fused intermediate, 32 at a time)
- Exact dot → F32x16 (single-precision final, 16 at a time)
- Calibration → F64x8 (drift detection, 8 at a time)

The `Fingerprint<256>` column works the same way:
```rust
impl Fingerprint<256> {
    pub fn as_bytes(&self)    -> &[u8; 2048]   // Hamming popcount
    pub fn as_u64(&self)      -> &[u64; 256]   // XOR bind
    pub fn as_u8x64(&self)    -> &[U8x64; 32]  // SIMD popcount batch
}
```

This is the fourth data pattern: **same object, multiple SIMD lane views.**
The BindSpace address points to ONE Arc'd byte region. The consumer
chooses the lane width based on the operation. Zero-copy, zero branch.

```
BindSpace column (read-only, Arc<[u64; 256 * N]>)
  │
  ▼ zero-copy slice window
&[u64; batch_size]   ← SIMD kernel input
  │
  ▼ SIMD op (popcount / AND / gather)
Microcopy result     ← stack-allocated Band / u32 distance
  │
  ▼ through Luftschleuse
BindSpace commit     ← XOR or Bundle merge
```

### The Luftschleuse (Airgap) Protocol

Writers never mutate BindSpace directly. They:

1. **Read** fingerprints as `&[u8]` slices (zero-copy)
2. **Compute** on owned microcopies (Copy, stack-only)
3. **Submit** deltas through the airgap (gated write)
4. **Merge** via XOR (single writer) or Bundle (multi-writer superposition)

```
         ┌─────────────────────────────────────────┐
         │        BindSpace (read-only)            │
         │   Fingerprint columns, Arc<[u64]>       │
         └────┬────────────────────────┬───────────┘
              │ &[u8] slices            │ &[u8] slices
              ▼                         ▼
      ┌───────────────┐         ┌───────────────┐
      │  Shader A     │         │  Shader B     │
      │  microcopies  │         │  microcopies  │
      │  (Copy only)  │         │  (Copy only)  │
      └───────┬───────┘         └───────┬───────┘
              │ delta + gate            │ delta + gate
              ▼                         ▼
         ┌─────────────────────────────────────────┐
         │      Luftschleuse (write airlock)       │
         │  Single writer: XOR commit              │
         │  Multi writer: Bundle (majority vote)   │
         │  Superposition: ALL deltas sum          │
         └────────────────┬────────────────────────┘
                          │ committed delta
                          ▼
         ┌─────────────────────────────────────────┐
         │        BindSpace (next generation)      │
         └─────────────────────────────────────────┘
```

### Superposition of Overlapping Writers

Two shaders writing to the same address at the same cycle:

```
Shader A writes: delta_A = target_addr ⊕ value_A
Shader B writes: delta_B = target_addr ⊕ value_B

Single-target XOR merge: new = old ⊕ delta_A ⊕ delta_B
  → ordering doesn't matter (XOR is commutative + associative)
  → both changes preserved as superposition

Multi-target Bundle merge: new = majority_vote([old, value_A, value_B])
  → single winner per bit
  → ambiguity filtered by consensus
```

No locks. No races. XOR is its own inverse — you can always back out.

## Integration Plan (prioritized by era)

### Phase 1 — Harden Foundation (Era 6 + 7)
**Keep the bedrock solid before building up.**

1. **Unify Fingerprint type**: kill holograph `BitpackedVector`, use
   `ndarray::hpc::fingerprint::Fingerprint<256>` everywhere.
2. **VectorWidth consumer wiring**: `vector_config()` LazyLock read
   at serialization boundaries only (hot path never branches).
3. **Complete ndarray Fingerprint API**: already done —
   get/set_bit, bind, and, not, permute, random, from_content, density.
4. **CognitiveShader → thinking-engine** wire-through: shader dispatch
   from `thinking-engine::cognitive_stack` to `p64-bridge::CognitiveShader`
   to `bgz17::palette_semiring::compose`.

### Phase 2 — BindSpace Address Substrate (new — era 9)
**Make the connective tissue work.**

5. **Port Container/CogRecord** to `lance-graph-contract` (16K width).
   Read-only. `Arc<[u64; 256]>` columns. No mutation APIs.
6. **Define Luftschleuse trait** in contract:
   ```rust
   pub trait Luftschleuse {
       type Delta: Copy;
       fn submit(&self, delta: Self::Delta);  // non-blocking
       fn commit(&mut self) -> Generation;    // merge all pending
   }
   ```
7. **Microcopy types**: confirm CausalEdge64, Band, TruthValue,
   ThinkingStyle are all Copy + small (≤16 bytes).
8. **Write-back gates**: `gated_xor` (single target),
   `majority_bundle` (multi target), `superposition_merge`
   (ambiguous — keep all).

### Phase 3 — Struct-of-Arrays Columns (era 8)
**The AGI address dimensions.**

9. **Column types** in contract:
   - `ContentColumn` (Fingerprint<256> array)
   - `TopicColumn` (Fingerprint<256> array)
   - `AngleColumn` (Fingerprint<256> array)
   - `CausalityColumn` (CausalEdge64 array)
   - `QualiaColumn` ([f32; 18] array)
   - `TemporalColumn` (u64 array)
   - `ShaderColumn` (u8 array — which shader emitted)
10. **Cascade per column**: Hamming sweep on fingerprint cols,
    range filter on scalar cols. Intersect survivors across dims.
11. **Column storage**: Arrow FixedSizeBinary for Fingerprint cols,
    Lance columnar format for scalars. Read-only, mmap'd.

### Phase 4 — Shader Stream (era 7+8 convergence)
**The 5D per-cycle stream.**

12. **Cycle loop**:
    ```
    for cycle in 0..:
        // Read ONE column per cascade level
        let topic_hits = topic_col.hamming_sweep(query_topic);
        let angle_hits = angle_col.hamming_sweep(query_angle);
        let causal_hits = causal_col.filter_rung(rung);
        let qualia_hits = qualia_col.range_match(qualia);
        
        // Intersect (bitmap AND)
        let survivors = topic_hits & angle_hits & causal_hits & qualia_hits;
        
        // Exact step on survivors
        for idx in survivors.iter() {
            let edge = shader.compute(content_col[idx], ...);
            airlock.submit(edge);
        }
        
        // Commit deltas
        next_gen = airlock.commit();
    ```

13. **CognitiveShader dispatch**: per cycle, the shader selects which
    columns to sweep and in what order (analytical shader might skip
    qualia; creative shader might skip causality).

### Phase 5 — GGUF Hydration (era 8 endgame)
**Weights as seeds for holographic memory.**

14. **Hydration pipeline**:
    - Load GGUF shard
    - kmeans per weight matrix → 256 archetypes → palette
    - Per archetype: Fingerprint<256> for Hamming cascade
    - Per cluster: holographic residual (slot-encoded phase+mag)
    - Emit CausalEdge64 wiring (layer → S/P/O palette indices)
    - Store in BindSpace columns (read-only after bake)
15. **Inference = cascade over hydrated columns**. No matmul. No FP.
    Just XOR/popcount/lookup per shader cycle.

## What Migrates vs What Stays

### Migrate into BindSpace columns
- Weight archetypes (GGUF hydration)
- CausalEdge64 outputs (inference)
- COCA verbs (cam_ops 4096)
- Thinking styles (contract 36)
- Grammar triangles (spectroscopy output)
- Dream consolidation results

### Stays as cold-path (DataFusion)
- Historical logs
- Training data
- User session history
- Analytics queries
- Batch jobs

### Stays as microcopy (hot path, Copy types)
- CausalEdge64 in shader inner loop
- TruthValue in NARS inference
- Band in cascade routing
- ThinkingStyle (3 bytes) in shader dispatch

## Priority Ordering

1. **P0** — Unify Fingerprint type (ndarray canonical)
2. **P0** — Port Container/CogRecord (read-only addressing)
3. **P1** — Luftschleuse trait + XOR/Bundle gates
4. **P1** — CognitiveShader → thinking-engine wire-through
5. **P2** — Column types in contract (AGI dimensions)
6. **P2** — Cascade per column implementation
7. **P3** — GGUF hydration pipeline
8. **P3** — Cognitive shader inference loop

## Success Criteria

- All programs (codecs, shaders, learning, grammar, search) emit
  and consume the same 64-bit BindSpace addresses
- No locks. No `&mut` during computation. Only Luftschleuse commits.
- Hot path: 0.3ns per XOR, 2400M lookups/sec, zero FP.
- Cold path: DataFusion SQL/Cypher on Lance columnar.
- Inference: 5 cascades per cycle × ~2ms each = ~10ms per token
  on CPU with cascade acceleration.
