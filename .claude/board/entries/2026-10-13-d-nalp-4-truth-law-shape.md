# 2026-10-13 — D-NALP-4: which realization of the per-row truth laws is exact (W4)

**Status:** MEASURED (shape exactness); the lane-producing kernel and terminal are OPEN.

**MEASURED:** `cargo run --release --manifest-path crates/lance-graph-nal/Cargo.toml --example w4_truth_law_probe`
on an AVX-512 host (the probe prints the tier; it is a scalar probe, so it
measures exactness, not speed). Every input, `(f1, c1, f2, c2)` ∈ 256⁴ = 2³²
rows per law, against the exact integer law rounded once, half up
(`crates/lance-graph-nal/tests/common/truth_law.rs`):

| law | table (256² `u8` laws) | `f32`, codes decoded to `x/255` (`isa::truth`) | `f32`, integer-valued | `f64`, integer-valued |
|---|---|---|---|---|
| deduction | 419,227,709 rows off | 2,413 | 1,368 | **0** |
| analogy | 521,640,704 | 512 | 0 | **0** |
| induction | 775,161,856 | 4,352 | 7,680 | **0** |
| abduction | 775,161,856 | 4,352 | 7,680 | **0** |

Every miss is one code on the confidence; the frequency never differs.

**What follows:**
- The table shape (plan §4 C candidate 1, `MapPairU8`) rounds at every stage
  and misses 10–18 % of all rows. Rejected.
- `f32` misses a few thousand rows per law, on either encoding.
- Integer-valued `f64` (products of codes, one division, one rounding) is
  exact on all 2³² rows of all four laws. It is also the shape that covers
  the variable division of `w / (w + k)`, which a multiply-high by a constant
  (§4 C candidate 2) cannot.

**OPEN:**
- Speed: no SIMD kernel exists yet; `F64x8` has `floor` and `mul_add` but no
  lane conversion from the `I32` truth lanes. Cost per element is measured on
  the kernel, not on this scalar probe.
- The plan's gate "one shape serves SQL nullable arithmetic too" is not met by
  this measurement: `f64` is exact for `i32` sums and differences, not for
  every `i32` product (past 2⁵³).
- Which premise is "1" in induction/abduction is W5's join (ONA swaps the
  arguments of its `Truth_Induction`); the laws here are two-input laws.
