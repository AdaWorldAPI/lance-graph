# KNOWLEDGE.md — ZeckBF17 Research Agent Reference

## What This Crate Is

A standalone research crate testing whether 16,384-dimensional i8 accumulator
planes can be compressed to 48 bytes (341:1) by treating the dimensions as
17 base classes × 964 octaves, where only ~14 octaves carry independent information.

**The one number that matters:** Spearman rank correlation ρ between pairwise
distances computed on full i8[16384] planes vs i16[17] base patterns.
If ρ > 0.937, ZeckBF17 eliminates the dead zone in the Pareto frontier.

## Architecture (the production system this feeds into)

### Three-Layer Search Cascade

```
L1  neighborhood/zeckf64.rs    ZeckF64 u64 per edge    ρ=0.937  ~1ms
    - Byte 0: scent (7 Boolean masks: S close? P? O? SP? SO? PO? SPO?)
    - Bytes 1-7: distance quantiles per mask (0=identical, 255=max different)
    - 19 of 128 patterns are legal (Boolean lattice constraint)
    - Distance: L1 on bytes (Manhattan). NOT Hamming.

L2  blasgraph/types.rs         BitVec 16Kbit integrated   ρ=0.834
    - ONE vector: majority-vote bundle of S ⊕ P ⊕ O
    - Distance: Hamming on the bundled vector
    - LOSES plane separation → scores LOWER than 1-byte scent
    - This is NOT a good baseline for ZeckBF17

L3  spo/                       exact S+P+O planes         ρ=1.000
    - THREE separate 16384-bit planes
    - Distance: ds + dp + do (sum of per-plane Hamming)
```

### CRITICAL INSIGHT: Why L2 < L1

The integrated BitVec (ρ=0.834) scores LOWER than the scent byte (ρ=0.937)
because bundling S+P+O into one vector DESTROYS the plane-separation
information that the 7 scent bits preserve. The scent encodes WHICH planes
are close. The integrated BitVec encodes whether the COMBINED signal is
close but can't say which plane caused the difference.

ZeckBF17 stores THREE separate base patterns (i16[17] each), preserving
plane separation like the scent. It should compare against ρ=0.937 (scent)
and ρ=0.982 (full ZeckF64), NOT against ρ=0.834 (integrated BitVec).

### Correct Pareto Frontier

```
Encoding              Bytes   Preserves S/P/O?   ρ vs exact S+P+O
─────────────────────────────────────────────────────────────────────
Scent byte              1     YES (7 masks)       0.937
ZeckBF17 bases        102     YES (3 × i16[17])   ? ← MEASURE
ZeckBF17 edge         116     YES (+ envelope)     ? ← MEASURE
Full ZeckF64            8     YES (7 + quantiles)  0.982
Integrated BitVec    2048     NO (bundled S⊕P⊕O)  0.834 ← WRONG CURVE
Exact S+P+O         6144     YES (3 planes)       1.000

The integrated BitVec is on a DIFFERENT Pareto curve (bundled metric).
ZeckBF17 and the scent byte are on the SAME curve (per-plane metric).
```

## ZeckBF17 Format

### Why 17

17 is prime. Golden-ratio step = round(17/φ) = 11. gcd(11,17) = 1 →
visits all 17 residues. This is the discrete golden-angle / X-Trans pattern.

**WARNING:** An earlier version claimed Fibonacci mod 17 visits all 17.
WRONG. Fibonacci mod 17 visits only 13 (missing {6,7,10,11}).
Fibonacci mod p has full coverage only for p ∈ {2, 3, 5, 7}.
The golden-ratio STEP (not Fibonacci SEQUENCE) is the correct traversal.

### Encoding

```rust
ZeckBF17Plane {           // 48 bytes
    base: [i16; 17],      // 34 bytes: mean per base dim, fixed-point ×256
    envelope: [u8; 14],   // 14 bytes: amplitude scale per independent octave
}

ZeckBF17Edge {            // 116 bytes
    subject: [i16; 17],   // 34 bytes
    predicate: [i16; 17], // 34 bytes
    object: [i16; 17],    // 34 bytes
    envelope: [u8; 14],   // 14 bytes (shared)
}
```

### Why i16 (not BF16)

BF16: 1 sign + 8 exponent + 7 mantissa. Wastes 8 exponent bits on dynamic
range never used (source is i8). Mean of 0.2 → stores 0.0 → LOSES SIGN.

i16 fixed-point (×256): mean 0.2 → stores 51. Mean -0.003 → stores -1.
256× finer quantization. Native SIMD. Same 34 bytes.

### Distance: L1 on i16 (not Hamming)

```rust
fn base_l1(a: &BasePattern, b: &BasePattern) -> u32 {
    a.dims.iter().zip(b.dims.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}
```

Matches production `zeckf64_distance()` which is L1 on quantile bytes.
`zeckf64_from_base()` produces a full u64 with the same byte layout
as `neighborhood/zeckf64.rs::zeckf64()`.

### Golden-Step Traversal

```
GOLDEN_POS_17 = [0, 11, 5, 16, 10, 4, 15, 9, 3, 14, 8, 2, 13, 7, 1, 12, 6]

Encode: for each octave (0..964), for each base_idx (0..17):
  dim = octave * 17 + GOLDEN_POS_17[base_idx]
  base[base_idx] += accumulator[dim]  (then average and scale to i16)

Decode: reverse — distribute base pattern across octaves, scaled by envelope.
```

## Codec Session Epiphanies (for context)

```
[3]  Alpha(|value|) = GAIN, sign bits = SHAPE. The Pareto frontier's 3 points
     are gain-only (8 bits), gain+coarse shape (57 bits), gain+exact shape (49K bits).
     L1/L2/L3 are ONE codec at three bitrates.

[6]  Semantic folding: crystallized planes predict uncrystallized ones.
     ZeckBF17 eliminates the dead zone by CHANGING THE BASIS.

[9]  σ-3 crystallization threshold IS a Lagrange multiplier for R-D optimization.
     Making σ adaptive per-scope: σ = c·2^((density-12)/3).

[12] Octave 0/1/2 hierarchy IS residual vector quantization.
     Combined with 3-layer cascade: 3×3 = 9-level refinement grid.
```

## Known Bugs (as of PR #21)

### FIXED in this branch:
- ✓ Fibonacci → golden step (all 17 residues)
- ✓ BF16 → i16 fixed-point
- ✓ BF16 Hamming → L1 on i16
- ✓ Unused deps removed (hound, half)
- ✓ Accumulator cyclic shift removed (shift mismatch with crystallize/unbind)

### Remaining:
- iMDCT reconstruction incomplete in transform.rs
- FftPlanner::new() per-call in transform.rs (perf)
- Duplicate pearson() in accumulator.rs and universal_perception.rs
- metrics.rs white noise RNG produces correlated data
- decode_hybrid_scent_only unimplemented in hybrid.rs
- Crate not in workspace Cargo.toml (by design — standalone)

## File Map

```
src/zeckbf17.rs             643 lines  THE codec. Run fidelity_experiment().
src/accumulator.rs          370 lines  Streaming bundle. Shift bug FIXED.
src/diamond.rs              289 lines  Diamond Markov extraction.
src/universal_perception.rs 561 lines  Noise floor vs masking hypothesis.
src/transform.rs            267 lines  MDCT, Bark bands, psychoacoustic mask.
src/bands.rs                161 lines  BF16 pack/unpack, weighted Hamming.
src/perframe.rs             147 lines  Strategy A: per-frame MDCT.
src/hybrid.rs               235 lines  Strategy C: combined.
src/metrics.rs              359 lines  4-strategy comparison.
src/lib.rs                  114 lines  Types, module declarations.
```

## How To Run

```bash
cd crates/lance-graph-codec-research
cargo test -- --nocapture

# THE critical test:
cargo test test_fidelity_vs_encounters -- --nocapture

# Page curve (after fidelity validates):
cargo test test_diamond -- --nocapture
cargo test test_accumulation_convergence -- --nocapture
cargo test test_universal_perception -- --nocapture
```

## What Success Looks Like

```
encounters | fidelity | ρ(rank) | scent%
    5      |  > 0.6   | > 0.5   | > 50%
   10      |  > 0.7   | > 0.7   | > 60%
   20      |  > 0.8   | > 0.85  | > 70%
   50      |  > 0.85  | > 0.93  | > 80%   ← THIS is the target
  100      |  > 0.9   | > 0.95  | > 85%

If ρ at 50 encounters > 0.937: ZeckBF17 beats scent-only.
The dead zone disappears. Proceed to Page curve measurement.
```
