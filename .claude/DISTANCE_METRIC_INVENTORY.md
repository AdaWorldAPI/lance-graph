# Distance Metric Inventory

> Where, why, and which distance metric is used across the stack.
> Updated: 2026-04-03
>
> **Rule**: Popcount/Hamming is NEVER valid on bgz17 i16[17] or BF16 stacked data.
> Use L1, PCDVQ-weighted L1, or palette lookup instead.

---

## Valid Metrics by Data Type

| Data type | Valid metrics | Invalid metrics | Why |
|-----------|--------------|-----------------|-----|
| bgz17 Base17 i16[17] | L1, PCDVQ-weighted L1, palette lookup | Hamming, popcount | Non-uniform bits — Hamming counts bit flips, meaningless on integers |
| StackedBF16 u64[17] | L1 (on hydrated f32), cosine, palette L1 | Hamming, popcount | BF16 is sign/exponent/mantissa — bit distance ≠ value distance |
| SearchKey17 u8[17] | L1 (on collapsed bytes) | — | Collapsed key with structured bits (sign+magnitude+hash) |
| Binary planes u64[N] | Hamming (popcount of XOR) | L1 | Uniform binary — each bit equally meaningful |
| Scent bytes u8/u32 | Hamming (popcount of XOR) | L1 | 7-bit Boolean lattice — designed for Hamming |
| Fingerprint\<256\> 16Kbit | Hamming (popcount of XOR) | — | Binary fingerprint planes — Hamming IS the metric |
| BitVec/Bitmap | Hamming (popcount) | — | Sparse graph structure — set membership |
| ThinkingStyleFingerprint u64 | Cosine proxy (bit agreement), texture resonance check | Ranking distance | Structured cognitive bits — valid as gate, not as ordering |

---

## Inventory by File

### ndarray (hardware — SIMD kernels)

| File | Function | Metric | Data type | Valid? |
|------|----------|--------|-----------|--------|
| `hpc/bgz17_bridge.rs` | `l1_avx512/avx2/scalar` | **L1** | Base17 i16[17] | ✓ Correct |
| `hpc/bgz17_bridge.rs` | `l1_weighted_avx512/avx2/scalar` | **PCDVQ L1** | Base17 i16[17] | ✓ Correct (20×/3×/1× weighting) |
| `hpc/bgz17_bridge.rs` | `sign_agreement_avx512/avx2/scalar` | Sign compare | Base17 i16[17] | ✓ Compares sign of i16 values, not bit patterns |
| `hpc/bgz17_bridge.rs` | `xor_bind_avx512/avx2/scalar` | XOR bind | Base17 i16[17] | ✓ Algebraic operation (superposition), not distance |
| `hpc/cascade.rs` | `Cascade::calibrate/observe/expose` | **Welford σ** | u32 distances | ✓ Statistical envelope on any metric |
| `hpc/cascade.rs` | `query()` Stroke 1-2 | **Hamming** | u8[] binary | ✓ Binary fingerprint data |
| `hpc/cascade.rs` | `query_precise()` Stroke 3 | **Cosine** | f32/i8/BF16 | ✓ Precision tier on hydrated data |
| `hpc/cascade.rs` | `bf16_hamming_scalar` | **Weighted Hamming** | BF16 bytes | ⚠️ Valid only as BF16Hamming precision mode, not for bgz17 |
| `hpc/palette_distance.rs` | `nearest_avx512/avx2/scalar` | **Palette L1** | Base17 i16[17] | ✓ Uses L1 kernel per candidate |
| `hpc/palette_distance.rs` | `DistanceTables` | **Precomputed L1** | u16 table | ✓ O(1) lookup, precomputed from L1 |
| `hpc/cam_pq.rs` | `distance_batch_avx512` | **VPGATHERDD** | CAM indices | ✓ Product quantization gather |
| `hpc/cam_pq.rs` | `squared_l2` | **L2²** | f32 subspaces | ✓ Euclidean distance |
| `hpc/clam.rs` | `hamming_inline` | **Hamming** | Binary planes | ✓ CLAM on binary fingerprints |
| `hpc/heel_f64x8.rs` | `heel_weighted_distance` | **F64x8 dot** | f64[8] | ✓ Weighted distance via SIMD polyfill |
| `hpc/heel_f64x8.rs` | `heel_plane_distances` | **Hamming** | u64[8] HEEL planes | ✓ Binary planes — Hamming correct |
| `hpc/heel_f64x8.rs` | `cosine_f64_simd` | **Cosine** | f64 slices | ✓ SIMD FMA cosine |
| `hpc/heel_f64x8.rs` | `cosine_f32_to_f64_simd` | **Cosine** | f32→f64 | ✓ Precision-widening cosine |
| `hpc/gguf_indexer.rs` | `project_row_to_base17` | **Golden-step avg** | f32→i16[17] | ✓ Encoding, not distance |
| `hpc/gguf_indexer.rs` | `project_row_bf16_direct` | **Golden-step avg** | BF16→i16[17] | ✓ Encoding, not distance |

### lance-graph / bgz-tensor (consumer — encoding pipeline)

| File | Function | Metric | Data type | Valid? |
|------|----------|--------|-----------|--------|
| `bgz-tensor/projection.rs` | `Base17::l1` | **L1** | i16[17] | ✓ Correct |
| `bgz-tensor/projection.rs` | `Base17::l1_weighted` | **PCDVQ L1** | i16[17] | ✓ Correct |
| `bgz-tensor/projection.rs` | `Base17::cosine` | **Cosine** | i16[17] as f64 | ✓ Correct |
| `bgz-tensor/projection.rs` | `Base17::attention_proxy` | **L1** | i16[17] | ✓ Returns raw L1 |
| `bgz-tensor/stacked_n.rs` | `StackedN::cosine` | **Cosine** | BF16→f32 | ✓ Correct |
| `bgz-tensor/stacked_n.rs` | `StackedN::l1_f32` | **L1** | BF16→f32 | ✓ Correct |
| `bgz-tensor/stacked.rs` | `StackedBF16x4::cosine` | **Cosine** | BF16→f64 | ✓ Correct |
| `bgz-tensor/stacked.rs` | `StackedBF16x4::vedic_upper_distance` | **L1** | i16 packed | ✓ Coarse L1 on upper 32 bits |
| `bgz-tensor/stacked.rs` | `StackedBF16x4::full_distance` | **L1** | i16 packed | ✓ Full L1 across all slots |
| `bgz-tensor/stacked.rs` | `SearchKey17::l1` | **L1** | u8[17] | ✓ Collapsed key L1 |
| `bgz-tensor/stacked.rs` | `SearchKey17::sign_agreement` | **Sign compare** | 1-bit per dim | ✓ Majority-voted sign, not raw BF16 |
| `bgz-tensor/attention.rs` | `AttentionTable::distance` | **Palette L1** | u16 table | ✓ O(1) precomputed L1 |
| `bgz-tensor/similarity.rs` | `SimilarityTable::similarity` | **CDF lookup** | u32→f32 | ✓ Calibrated from L1 distribution |
| `bgz-tensor/hdr_belichtung.rs` | `heel_distance_l1` | **L1** | Base17 i16[17] | ✓ Correct |
| `bgz-tensor/hdr_belichtung.rs` | `heel_distance_palette` | **Palette L1** | u16 table | ✓ O(1) lookup |
| `bgz-tensor/hdr_belichtung.rs` | `hip_palette_candidates` | **Palette ranking** | u16 table + ¼σ bands | ✓ Band-classified palette distance |
| `bgz-tensor/hdr_belichtung.rs` | `twig_distance_l1` | **L1** | Base17 i16[17] | ✓ Correct |
| `bgz-tensor/hdr_belichtung.rs` | `leaf_hydrate_cosine` | **Cosine** | StackedN→f32 | ✓ Exact via hydration |
| `bgz-tensor/hdr_belichtung.rs` | `PaletteCascade` | **ndarray Cascade** | Welford σ | ✓ Delegates to ndarray |
| `bgz-tensor/neuron_hetero.rs` | `ThinkingStyleFingerprint::texture_resonance_check` | **Bit agreement** | u64 structured | ✓ Binary gate (pass/fail) |
| `bgz-tensor/neuron_hetero.rs` | `ThinkingStyleFingerprint::cosine_proxy` | **Bit agreement** | u64 structured | ✓ Cosine replacement on cognitive bits |
| `bgz-tensor/neuron_hetero.rs` | `ThinkingStyleFingerprint::bit_disagreements` | **Bit count** | u64 | ✓ Diagnostics only |
| `bgz-tensor/euler_fold.rs` | `euler_gamma_fold/unfold` | **γ-rotation** | StackedN f32 | ✓ Not a distance — encoding operation |
| `bgz-tensor/belichtungsmesser.rs` | `Belichtungsmesser::classify` | **¼σ band** | u32 L1 distances | ✓ Band classification on L1 |
| `bgz-tensor/belichtungsmesser.rs` | `three_stroke` | **Plane L1** | Base17 subranges | ✓ L1 on dim subsets |
| `bgz-tensor/quality.rs` | `pearson/spearman/mae/rmse` | **Correlation** | f64 | ✓ Statistical metrics |
| `bgz-tensor/codebook4096.rs` | `Codebook4096::assign` | **Full distance** | StackedBF16x4 | ✓ L1 on stacked |
| `bgz-tensor/variance.rs` | `compute_variance` | **L1 to centroid** | Base17 i16[17] | ✓ Correct |
| `bgz-tensor/jina.rs` | `cosine_f32` | **Cosine** | f32 embeddings | ✓ Ground truth |

### lance-graph / bgz17 (palette semiring codec)

| File | Function | Metric | Data type | Valid? |
|------|----------|--------|-----------|--------|
| `bgz17/base17.rs` | L1 distance | **L1** | i16[17] | ✓ Correct |
| `bgz17/palette.rs` | `Palette::nearest` | **L1** | Base17 i16[17] | ✓ Correct |
| `bgz17/distance_matrix.rs` | Precomputed K×K | **L1** | u16 table | ✓ O(1) lookup |
| `bgz17/layered.rs:61` | Scent XOR + count_ones | **Hamming** | Scent u32 | ✓ Scent = Boolean lattice |
| `bgz17/container.rs:692` | Fingerprint XOR | **Hamming** | 16Kbit planes | ✓ Binary planes |
| `bgz17/rabitq_compat.rs:174` | XOR + popcount | **Hamming** | RaBitQ binary | ✓ Binary by design |
| `bgz17/clam_bridge.rs:128` | Scent distance | **Hamming** | Scent u32 | ✓ Boolean lattice |
| `bgz17/prefetch.rs:230` | Scent prefetch | **Hamming** | Scent u32 | ✓ Boolean lattice |
| `bgz17/bridge.rs:111` | Plane XOR | **Hamming** | Binary plane | ✓ Binary |

### lance-graph / p64-bridge (topology ↔ metric convergence)

| File | Function | Metric | Data type | Valid? |
|------|----------|--------|-----------|--------|
| `p64-bridge/lib.rs:420` | Plane density count | **Popcount** | u64[64] palette rows | ✓ Binary topology density |
| `p64-bridge/lib.rs:505` | NNZ count | **Popcount** | u64[64] palette rows | ✓ Sparsity measurement |
| `p64-bridge/lib.rs:511` | Per-layer density | **Popcount** | u64[64] per layer | ✓ Binary topology |
| `p64-bridge/lib.rs` | `Blumenstrauss::cascade` | **Palette L1** | bgz17 PaletteSemiring | ✓ Delegates to bgz17 |

### lance-graph / blasgraph (graph algebra)

| File | Function | Metric | Data type | Valid? |
|------|----------|--------|-----------|--------|
| `blasgraph/types.rs:179` | `BitVec::popcount` | **Popcount** | 16Kbit binary | ✓ Binary fingerprint |
| `blasgraph/ndarray_bridge.rs:174` | `dispatch_popcount` | **Popcount** | u8[] binary | ✓ SIMD-dispatched (AVX-512/AVX2/scalar) |
| `blasgraph/semiring.rs:140,319` | Semiring MIN | **Popcount** | BitVec comparison | ✓ Binary lattice ordering |
| `blasgraph/ops.rs:232` | Edge weight | **Popcount** | BitVec | ✓ Binary density |
| `graph/sparse.rs:51` | `bitmap_popcount` | **Popcount** | Bitmap u64[] | ✓ Sparse structure |
| `graph/fingerprint.rs:44` | Density check | **Popcount** | u64[] fingerprint | ✓ Binary density cap |
| `graph/falkor_compat.rs:164` | Distance proxy | **Popcount** | BitVec | ✓ FalkorDB compat |

---

## Cascade Architecture

```
HEEL:  palette_table[a_idx * 256 + b_idx]    O(1) lookup, precomputed L1
       → ¼σ band classification via ndarray Cascade Welford σ
       → reject if band > heel_max_band

HIP:   same palette distance, tighter band threshold
       → reject if band > hip_max_band

TWIG:  Base17 L1 (17 i16 subtracts)           O(17) actual metric
       → reject if band > twig_max_band

LEAF:  BF16→f32 hydration → exact cosine      O(SPD×17) precision
       → final answer
```

Popcount appears NOWHERE in this cascade. Every stage uses L1 or palette L1.

---

## When Popcount IS Correct

1. **Binary fingerprint planes** (16Kbit) — each bit independently meaningful
2. **Scent bytes** (7-bit Boolean lattice) — designed for XOR+popcount
3. **p64 HEEL planes** (u64[8]) — BNN attention Q AND K
4. **BitVec/Bitmap** — sparse graph structure, set membership
5. **RaBitQ binary codes** — binary quantization by design
6. **ThinkingStyleFingerprint** — ONLY as texture resonance gate or cosine proxy

## When Popcount Is WRONG

1. **bgz17 Base17 i16[17]** — integer coordinates, use L1
2. **StackedBF16 u64[17]** — packed BF16 values, use L1 on hydrated f32
3. **BF16 raw bytes** — sign/exponent/mantissa structure, bit flips ≠ value distance
4. **Any f32/f64 data** — continuous values, use cosine or L1
