# Step 4: Fisher z + BGZ-HHTL-D in ndarray with SIMD

> Agent: savant-research (codec) + sentinel-qa (unsafe audit)
> READ FIRST: ndarray/CLAUDE.md, ndarray/src/hpc/audio/, simd.rs
> Depends on: Step 1 (fisher_z.rs API settled)

## What to Do

Add `src/hpc/fisher_z.rs` and `src/hpc/bgz_hhtl_d.rs` to ndarray.
These are the SIMD-accelerated versions of the bgz-tensor primitives.

### fisher_z.rs (ndarray)

SIMD-accelerated batch encode/decode:

```rust
use crate::simd::{F32x16, f32x16};

/// Batch encode 16 cosines at once via F32x16.
pub fn fisher_z_encode_batch(cosines: &[f32], gamma: &FamilyGamma, out: &mut [i8]) {
    // Process 16 at a time using F32x16
    for chunk in cosines.chunks(16) {
        let v = F32x16::from_slice(chunk);
        // arctanh via log: 0.5 * ln((1+x)/(1-x))
        // ... SIMD implementation
    }
}

/// Batch decode 16 i8 values at once.
pub fn fisher_z_decode_batch(values: &[i8], gamma: &FamilyGamma, out: &mut [f32]) {
    // F32x16 tanh approximation
}

/// Table lookup: k×k i8 table, batch of (a, b) pairs.
/// Uses array_windows for streaming access.
pub fn fisher_z_table_lookup_batch(
    table: &[i8],
    k: usize,
    pairs: &[(u8, u8)],
    out: &mut [i8],
) {
    for (pair, dst) in pairs.iter().zip(out.iter_mut()) {
        *dst = table[pair.0 as usize * k + pair.1 as usize];
    }
}
```

### bgz_hhtl_d.rs (ndarray)

SIMD-accelerated HHTL-D entry parsing:

```rust
/// Parse 16 HhtlDEntry (64 bytes) at once via U8x64.
/// Extract basin, HIP, centroid, polarity in parallel.
pub fn parse_entries_batch(bytes: &[u8; 64]) -> [ParsedEntry; 16] {
    // 4 bytes per entry × 16 = 64 bytes = one U8x64 load
}
```

### Key Rules

- Use `crate::simd::F32x16` for 16-wide float ops
- Use `array_windows` pattern for streaming table access
- All `unsafe` blocks need `// SAFETY:` comments
- LazyLock dispatch: AVX-512 → AVX2 → scalar
- Tests must pass on the CI VM (AVX2 only, no AVX-512)

## Tests

- `fisher_z_batch_matches_scalar`: SIMD output == scalar output
- `table_lookup_batch_correct`: batch lookup matches individual lookups
- `parse_entries_16_at_once`: batch parse matches sequential parse

## Pass Criteria

- cargo test --lib fisher_z passes (ndarray)
- No SIGILL on AVX2-only machines
- Benchmark: batch encode ≥ 100M values/sec
