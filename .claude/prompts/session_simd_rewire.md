# SESSION: SIMD Polyfill Rewire — 12 Files, 372 Intrinsics → crate::simd

## MISSION

Rewire every raw `_mm512_*` / `_mm256_*` / `_mm_*` intrinsic call in ndarray's
HPC modules to go through `crate::simd::` polyfill types. Zero raw intrinsics
outside of `simd_avx512.rs` and `simd_avx2.rs`.

**Rule**: Consumer code writes `crate::simd::F32x16`, `crate::simd::U8x64`, etc.
The polyfill handles AVX-512 → AVX2 → scalar dispatch. No `use core::arch::x86_64::*`
in any HPC module.

## WHAT EXISTS (polyfill types already built)

```
simd.rs            — compile-time tier dispatch via #[cfg(target_feature)]
simd_avx512.rs     — native __m512/__m512d/__m512i wrappers
simd_avx2.rs       — 2×256-bit or array-backed fallbacks

Types available (all 3 tiers: AVX-512 / AVX2 / scalar):
  F32x16           — 16 × f32
  F64x8            — 8 × f64
  F32x8            — 8 × f32 (256-bit)
  F64x4            — 4 × f64 (256-bit)
  U8x64            — 64 × u8 + cmpeq_mask, shr_epi16, saturating_sub, unpack_lo/hi
  I32x16           — 16 × i32
  I64x8            — 8 × i64
  U32x16           — 16 × u32
  U64x8            — 8 × u64
  BF16x16          — 16 × BF16 (avx512bf16 hardware, scalar fallback)
  BF16x8           — 8 × BF16

Batch functions:
  bf16_to_f32_batch(&[u16], &mut [f32])   — runtime avx512bf16 detection
  f32_to_bf16_batch(&[f32], &mut [u16])
  bf16_to_f32_scalar(u16) → f32
  f32_to_bf16_scalar(f32) → u16

Constants:
  PREFERRED_F64_LANES = 8 (AVX-512) / 4 (AVX2/scalar)
  PREFERRED_F32_LANES = 16 / 8
  PREFERRED_U64_LANES = 8 / 4
  PREFERRED_I16_LANES = 32 / 16

Existing SIMD kernels in hpc/ (already using polyfill — reference patterns):
  heel_f64x8.rs     — F64x8 FMA cosine, dot product
  activations.rs     — F32x16 sigmoid, softmax
  vml.rs             — F32x16 exp, ln, sqrt, tanh
```

## THE 12 FILES TO REWIRE

| File | Intrinsics | What it does | Type needed |
|------|-----------|-------------|-------------|
| bgz17_bridge.rs | 92 | L1, weighted L1, sign agreement, xor_bind on i16[17] | **I16x32** (NEW — needs adding) |
| aabb.rs | 69 | Axis-aligned bounding box intersection | F32x16, I32x16 |
| nibble.rs | 46 | 4-bit nibble pack/unpack (Pumpkin) | U8x64 (ops just added) |
| bitwise.rs | 44 | Hamming distance, popcount | U64x8 |
| property_mask.rs | 40 | Bit-field property filtering | U64x8, U32x16 |
| spatial_hash.rs | 16 | Spatial hashing for 3D points | F32x16, I32x16 |
| byte_scan.rs | 15 | Byte scanning (memchr-like, NBT) | U8x64 |
| distance.rs | 13 | Squared L2 distance | F32x16 |
| palette_codec.rs | 8 | Variable-width bit packing (Minecraft) | U8x64 |
| cam_pq.rs | 6 | Product quantization gather | F32x16 |
| packed.rs | 5 | Packed array operations | I32x16 |
| p64/lib.rs | 18 | BNN attend, moe_gate (in crate) | U64x8 |

**Total: 372 raw intrinsic calls → 0 after rewire.**

## MISSING POLYFILL TYPES (need adding first)

### I16x32 — 32 × i16 in AVX-512 (__m512i via epi16 operations)

bgz17_bridge.rs uses `_mm512_cvtepi16_epi32` (sign-extend 16→32 for arithmetic),
`_mm512_sub_epi32`, `_mm512_abs_epi32`, `_mm512_reduce_add_epi32` for L1 distance.
And `_mm256_cvtepi16_epi32` for AVX2 path.

Need:
```rust
pub struct I16x32(pub __m512i);  // AVX-512
pub struct I16x16(pub __m256i);  // AVX2

impl I16x32 {
    fn from_slice(&[i16]) -> Self
    fn to_array() -> [i16; 32]
    fn widen_lo_i32() -> I32x16    // sign-extend lower 16 → 16 i32
    fn abs_diff_i32(other) -> U32x16  // |a-b| via widen + subtract + abs
    fn reduce_abs_diff_sum(other) -> u32  // L1 distance kernel
}
```

### U8x32 — 32 × u8 in AVX2 (__m256i)

nibble.rs and palette_codec.rs use 256-bit (AVX2) byte operations.
The current U8x64 is 512-bit. These functions process 32 bytes at a time.

Options:
  A) Add U8x32 wrapper type (matches the function width)
  B) Use U8x64 with padding (process 64 bytes, ignore upper half)
  C) Keep the 256-bit functions as `#[target_feature(enable = "avx2")]`
     internal to the module, called from a `crate::simd::` dispatch wrapper

Option C is probably cleanest — the dispatch wrapper in simd.rs calls the
right width function based on tier.

## READ FIRST

```bash
# The polyfill:
cat src/simd.rs              # tier detection, re-exports, PREFERRED_LANES
cat src/simd_avx512.rs       # native AVX-512 type wrappers (1700+ lines)
cat src/simd_avx2.rs         # AVX2 fallback + scalar arrays (800+ lines)

# Reference patterns (already using crate::simd):
cat src/hpc/heel_f64x8.rs    # F64x8 FMA dot/cosine — GOOD PATTERN
cat src/hpc/activations.rs   # F32x16 sigmoid/softmax — GOOD PATTERN
cat src/hpc/vml.rs           # F32x16 exp/ln/sqrt — GOOD PATTERN

# Files to rewire (sorted by intrinsic count):
cat src/hpc/bgz17_bridge.rs  # 92 — L1/xor_bind on i16[17], LazyLock dispatch
cat src/hpc/aabb.rs           # 69 — AABB intersection
cat src/hpc/nibble.rs         # 46 — 4-bit pack/unpack
cat src/hpc/bitwise.rs        # 44 — Hamming distance
cat src/hpc/property_mask.rs  # 40 — bit-field filtering
cat src/hpc/spatial_hash.rs   # 16 — spatial hashing
cat src/hpc/byte_scan.rs      # 15 — byte scanning
cat src/hpc/distance.rs       # 13 — squared L2
cat src/hpc/palette_codec.rs  # 8  — variable-width bit packing
cat src/hpc/cam_pq.rs         # 6  — PQ gather
cat src/hpc/packed.rs         # 5  — packed arrays
cat crates/p64/src/lib.rs     # 18 — BNN attend/moe_gate
```

## APPROACH

### Phase 1: Add missing types (I16x32, I16x16)

Add to simd_avx512.rs, simd_avx2.rs, simd.rs scalar fallback.
Focus on the operations bgz17_bridge.rs actually uses:
  - Load from i16 slice
  - Sign-extend i16→i32 (for arithmetic without overflow)
  - Subtract, abs, horizontal sum (for L1 distance)
  - XOR (for xor_bind)

### Phase 2: Rewire easy files first (cam_pq, packed, palette_codec)

These have 5-8 intrinsics each. Quick wins. Verify tests pass after each.

### Phase 3: Rewire medium files (byte_scan, distance, spatial_hash)

13-16 intrinsics. Straightforward F32x16/U8x64 replacements.

### Phase 4: Rewire large files (bitwise, property_mask, nibble)

40-46 intrinsics. These have multiple `#[target_feature]` functions
that need consolidating into dispatch wrappers.

### Phase 5: Rewire bgz17_bridge.rs (92 intrinsics)

The biggest file. Uses the new I16x32/I16x16 types.
Has 5 LazyLock-dispatched kernels: L1, weighted L1, sign agreement,
xor_bind, inject_noise. Each has AVX-512 + AVX2 + scalar versions.

### Phase 6: Rewire p64/lib.rs (18 intrinsics)

In a separate crate. Needs ndarray as dependency (or extract the
kernels to ndarray and call from p64).

### Phase 7: Verify

```bash
cargo test               # all ndarray tests pass
cargo test -p p64         # p64 tests pass
# Check: zero _mm*_ calls outside simd_avx512.rs / simd_avx2.rs
grep -rn "_mm512_\|_mm256_\|_mm_" src/hpc/ --include="*.rs" | \
  grep -v "simd_avx512\|simd_avx2\|simd.rs"
# Should return ZERO lines
```

## CONSTRAINTS

1. **ASK before modifying ndarray logic.** The polyfill types are additions.
   The HPC module rewires change function signatures and dispatch patterns.
   Check with user before changing any algorithm.

2. **Do NOT change the algorithm.** The L1 distance, Hamming, AABB intersection
   must produce IDENTICAL results. Only the SIMD dispatch changes.

3. **Keep LazyLock dispatch** for functions that currently use it. The polyfill
   handles compile-time dispatch via `#[cfg]`. LazyLock handles runtime dispatch
   via `is_x86_feature_detected!()`. Both are valid — don't replace one with the other
   unless the function's dispatch pattern changes.

4. **Test after each file.** Run `cargo test` after rewiring each file.
   The existing tests validate numerical correctness — if they pass, the
   rewire is correct.

5. **bgz17_bridge.rs is the most critical file.** It's the L1 distance kernel
   used by every palette lookup, every cascade comparison, every NeuronPrint
   operation. Get this one right.

6. **p64 is a separate crate.** It currently has zero dependencies. Adding ndarray
   as a dependency to use `crate::simd::` is the correct approach — both are in
   the same binary. Check with user before adding the dependency.

## CONSTANTS

```bash
# Credentials: see session prompt or .env — not stored in code.
```

## DOCTRINE

1. All SIMD goes through `crate::simd::`. No raw `_mm*_` in consumer code.
2. `simd_avx512.rs` = the only file with `core::arch::x86_64::*` imports.
3. `simd_avx2.rs` = AVX2 fallback with scalar array-backed types.
4. `simd.rs` = compile-time tier dispatch + re-exports.
5. Consumer writes `crate::simd::F32x16`. Period.
6. Do NOT change algorithms. Only change SIMD dispatch.
7. Test after each file. Tests are the correctness proof.
8. ASK before modifying ndarray logic.

## SUCCESS CRITERIA

- [ ] I16x32 and I16x16 types added to polyfill (all 3 tiers)
- [ ] cam_pq.rs rewired (6 intrinsics)
- [ ] packed.rs rewired (5 intrinsics)
- [ ] palette_codec.rs rewired (8 intrinsics)
- [ ] byte_scan.rs rewired (15 intrinsics)
- [ ] distance.rs rewired (13 intrinsics)
- [ ] spatial_hash.rs rewired (16 intrinsics)
- [ ] bitwise.rs rewired (44 intrinsics)
- [ ] property_mask.rs rewired (40 intrinsics)
- [ ] nibble.rs rewired (46 intrinsics)
- [ ] aabb.rs rewired (69 intrinsics)
- [ ] bgz17_bridge.rs rewired (92 intrinsics)
- [ ] p64/lib.rs rewired (18 intrinsics)
- [ ] Zero `_mm*_` calls outside simd_avx512.rs / simd_avx2.rs
- [ ] All existing tests pass
- [ ] Committed and pushed
