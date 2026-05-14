# PR-NDARRAY-MIRI-COMPLETE — Spec

> **Status:** Draft (2026-05-14, W8 sprint-log-10)
> **Target repo:** `/home/user/ndarray` (AdaWorldAPI ndarray fork) — NOT lance-graph.
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §8 (ndarray-side prerequisites)
> **Sequence:** MUST land BEFORE par-tile crate (PR-CE64-MB-1) so par-tile SIMD paths are Miri-checkable.
> **Deliverable size:** ~200 LOC additions + ~70 LOC cfg-reroute + ~200 LOC new tests + ~10 LOC script update. ~700 LOC total delta, 1 file deletion.
> **Confirmed shipped (do not re-land):** `e0907cd` cfg(miri) cpuid bypass; `6590b9e` miri-tests.sh constrained scope; `530ffaa` "5 of 30 types" stale comment correction.

---

## §1 Statement of scope

This PR is the ndarray-side prerequisite that unblocks par-tile (W1's spec, PR-CE64-MB-1) when par-tile types include Miri-checkable SIMD paths. It is **fully standalone** from the lance-graph sprint-10 PRs; it lands on the ndarray repo on a focused branch and has no cross-spec dependencies within this sprint.

Two independent deliverables within one PR:

**Part A — Close u-word and i-word method gaps**: `simd_nightly/u_word_types.rs` and `simd_nightly/i_word_types.rs` currently expose only `cmpeq_mask`, `cmpgt_mask`, `copy_to_slice`, `from_array`, `from_slice`, `reduce_max/min/sum`, `saturating_add/sub`, `simd_max/min`, `splat`, and `to_array`. The full comparison + mask + clamp + select family that `f32_types.rs` has is absent. Par-tile and cognitive-shader code that works with integer SIMD vectors (byte-scan, quantized activations, U16x32 codebook distance, U32x16 index arithmetic) cannot be Miri-tested without this family.

**Part B — Route `crate::simd::*` through `simd_nightly` under `cfg(miri)`**: The consumer-facing `src/simd.rs` re-exports from `simd_avx512` / `simd_avx2` / `simd_neon` unconditionally. Miri rejects every call into those backends with "calling a function that requires unavailable target features: avx". The polyfill (`simd_nightly`) is Miri-clean and has full 24-type parity per PR #146. The missing piece is a `cfg(miri)` preempting block in `simd.rs` that redirects `pub use` from `simd_nightly` instead. Once landed, every `use ndarray::simd::F32x16` call site becomes Miri-checkable without any change to consumer code.

**Delta against parent plan §8**: the plan already lists these two items as "OPEN — needs follow-up PR". This spec provides the implementation shape, test plan, risk matrix, and sequencing rationale. No new architectural decisions are introduced; only implementation detail.

---

## §2 Source audit — confirmed current state (2026-05-14)

### 2.1 u_word_types.rs — what exists per direct read

Types: `U64x8`, `U64x4`, `U32x16`, `U32x8`, `U16x32`.

Methods present on all five (confirmed via file read):
- `splat`, `from_slice`, `from_array`, `to_array`, `copy_to_slice`
- `reduce_sum`, `reduce_min`, `reduce_max`
- `simd_min`, `simd_max`
- `cmpeq_mask(other) -> u{N}` — raw bitmask primitive
- `cmpgt_mask(other) -> u{N}` — raw bitmask primitive
- `saturating_add`, `saturating_sub` — on U16x32 only (not U32x16, U64x8, U64x4, U32x8)
- `Default` impl (via `splat(0)`)

**Confirmed absent on U16x32, U32x16, U64x8** (the three production-width types; U64x4 and U32x8 are companion bit-reinterpretation types with lower gap priority):
- `simd_eq(self, other) -> {mask type}` — typed mask wrapper (not raw bits)
- `simd_ne(self, other) -> {mask type}`
- `simd_ge(self, other) -> {mask type}` / `simd_gt` / `simd_le` / `simd_lt`
- `simd_clamp(self, lo, hi) -> Self`
- `select(self, mask, other) -> Self` — mask-driven blend
- `to_bitmask(mask) -> u64` — uniform u64 extraction (production API naming)
- `zero() -> Self` — explicit zero constructor (Default exists but production API names it `zero`)
- U16x32-specific: `from_u8x64_lo`, `from_u8x64_hi`, `pack_saturate_u8`, `mullo`, `shl`, `shr`

### 2.2 i_word_types.rs — what exists per direct read

Types: `I16x16`, `I16x32`, `I32x16`, `I64x8`.

Methods present on all four (confirmed):
- `splat`, `from_slice`, `from_array`, `to_array`, `copy_to_slice`
- `reduce_sum`, `reduce_min`, `reduce_max`
- `simd_min`, `simd_max`
- `saturating_add`, `saturating_sub`
- `cmpeq_mask(other) -> u{N}`, `cmpgt_mask(other) -> u{N}`
- `PartialEq`, `Display`

**Confirmed absent on all four i-word types**: same simd_eq/ne/ge/gt/le/lt typed-mask family, plus `simd_clamp`, `select`, `to_bitmask`, `zero`.

### 2.3 simd.rs — current re-export block (lines 200-291, confirmed)

Lines 205-220 already document the Miri gap precisely (the comment says "The remaining work for Miri-clean coverage of `hpc::*` is wiring this file's `pub use crate::simd_{avx512,avx2,neon}::*` re-exports to route through `simd_nightly` under `cfg(miri)`"). The four existing `pub use` blocks carry NO `not(miri)` guard — they fire unconditionally on x86_64 regardless of Miri context.

### 2.4 miri-tests.sh — current filter (confirmed from direct read)

Three exclusion clauses active:
1. `!(test(/^hpc::/) - test(/^hpc::byte_scan/))` — skips all hpc::* except byte_scan
2. `!test(/^simd::tests::/)` — skips simd::tests::* (AVX intrinsics calls)
3. `!test(/^hpc::framebuffer::pyramid_tests::/)` — keeps out 3 pyramid tests (19+ min each)

The script's own comment (lines 44-58) accurately diagnoses the root cause: "The missing piece is a `cfg(miri)` switch in `src/simd.rs`."

### 2.5 _original_draft.rs — confirmed dead

Present in the `src/simd_nightly/` directory (confirmed via `ls`). Not listed in `mod.rs` pub modules. Deletion safe pending grep confirmation.

---

## §3 Part A — u-word method gaps: implementation shape

### 3.1 Mask type strategy

`f32_types.rs` uses typed `F32Mask16` wrappers (from `masks.rs`). The u-word / i-word types have no such wrappers. **Chosen approach (Option B — minimal surface, lower risk)**: return `core::simd::Mask<i{W}, N>` directly from the typed comparison methods. Consumer calls `.to_bitmask() as u64` explicitly. The existing `cmpeq_mask` / `cmpgt_mask` already do this internally — Option B extends that pattern uniformly.

Escalation path: if consumer code fails to compile under `cfg(miri)` because production intrinsics return a typed mask struct and polyfill returns a raw `Mask<..>`, promote to typed `UMask{N}` / `IMask{N}` wrappers in `masks.rs`. This is a localised change.

### 3.2 U16x32 / U32x16 / U64x8 — methods to add

Mask type per production-width type:
- `U32x16` (16 lanes u32): `Mask<i32, 16>`, bitmask primitive `u16`
- `U16x32` (32 lanes u16): `Mask<i16, 32>`, bitmask primitive `u32`
- `U64x8` (8 lanes u64): `Mask<i64, 8>`, bitmask primitive `u8`

Code sketch (U32x16 shown; U16x32 / U64x8 are symmetric with appropriate Mask type):

```rust
// In impl U32x16:

// ── Typed comparison family ──────────────────────────────────────────

pub fn simd_eq(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_eq(other.0)
}
pub fn simd_ne(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_ne(other.0)
}
pub fn simd_lt(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_lt(other.0)
}
pub fn simd_le(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_le(other.0)
}
pub fn simd_gt(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_gt(other.0)
}
pub fn simd_ge(self, other: Self) -> core::simd::Mask<i32, 16> {
    self.0.simd_ge(other.0)
}

// ── Clamp / select / zero ────────────────────────────────────────────

pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
    Self(self.0.simd_clamp(lo.0, hi.0))
}

// Mask-driven blend. `mask` is a raw bitmask (u16 for U32x16).
pub fn select(self, mask: u16, other: Self) -> Self {
    use core::simd::prelude::Select;
    let m = core::simd::Mask::<i32, 16>::from_bitmask(mask as u64);
    Self(m.select(self.0, other.0))
}

// Uniform zero constructor.
pub fn zero() -> Self { Self::splat(0) }

// Uniform u64 bitmask extraction from a typed comparison result.
pub fn to_bitmask(mask: core::simd::Mask<i32, 16>) -> u64 {
    mask.to_bitmask() as u64
}
```

All seven comparisons require `SimdPartialEq` and `SimdPartialOrd` — already imported at file top (`use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd}`). `simd_clamp` requires `SimdOrd` — also already imported. `select` requires adding `use core::simd::prelude::Select` at file top (not currently present in `u_word_types.rs`; `masks.rs` imports it).

### 3.3 U16x32 additional methods (widening / narrowing / arithmetic)

These are needed for Miri coverage of CAM-PQ codebook accumulation paths (byte → u16 widen, accumulate, narrow).

```rust
// Widen lower 32 bytes of U8x64 to u16. Scalar loop (polyfill; prod = _mm512_cvtepu8_epi16).
pub fn from_u8x64_lo(a: super::u8_types::U8x64) -> Self {
    let bytes = a.to_array();
    let mut out = [0u16; 32];
    for i in 0..32 { out[i] = bytes[i] as u16; }
    Self::from_array(out)
}

// Widen upper 32 bytes of U8x64 to u16.
pub fn from_u8x64_hi(a: super::u8_types::U8x64) -> Self {
    let bytes = a.to_array();
    let mut out = [0u16; 32];
    for i in 0..32 { out[i] = bytes[i + 32] as u16; }
    Self::from_array(out)
}

// Narrow two U16x32 to U8x64 with saturation. Scalar loop (prod = _mm512_packus_epi16).
pub fn pack_saturate_u8(self, other: Self) -> super::u8_types::U8x64 {
    let a = self.to_array();
    let b = other.to_array();
    let mut out = [0u8; 64];
    for i in 0..32 { out[i]      = a[i].min(255) as u8; }
    for i in 0..32 { out[i + 32] = b[i].min(255) as u8; }
    super::u8_types::U8x64::from_array(out)
}

// Low-half multiply (core::simd Mul gives this natively for u16x32).
pub fn mullo(self, other: Self) -> Self {
    Self(self.0 * other.0)
}

// Variable-amount left shift.
pub fn shl(self, amt: Self) -> Self { Self(self.0 << amt.0) }

// Variable-amount right shift (logical, unsigned).
pub fn shr(self, amt: Self) -> Self { Self(self.0 >> amt.0) }
```

### 3.4 I16x16 / I16x32 / I32x16 / I64x8 — symmetric gaps

Same simd_eq/ne/lt/le/gt/ge, simd_clamp, select, to_bitmask, zero family. `simd_clamp` on signed types uses `SimdOrd::simd_clamp` — already valid (signed integers implement `SimdOrd` in `core::simd`).

Mask types:
- `I16x16`: `Mask<i16, 16>`, bitmask `u16`
- `I16x32`: `Mask<i16, 32>`, bitmask `u32`
- `I32x16`: `Mask<i32, 16>`, bitmask `u16`
- `I64x8`: `Mask<i64, 8>`, bitmask `u8`

I-word types do NOT need widening/narrowing helpers (U16x32-specific).

---

## §4 Part A — test plan

Add to `/home/user/ndarray/src/simd_nightly/tests.rs` (confirmed in directory). All tests under `#[cfg(feature = "nightly-simd")]`.

Target: ~48 new tests. Run command:
```bash
cargo +nightly miri nextest run -p ndarray --features nightly-simd simd_nightly::tests
```

### 4.1 U32x16 tests (12 new tests)

| Test name | What it checks |
|---|---|
| `u32x16_simd_eq_full_match` | All lanes equal → bitmask = 0xFFFF |
| `u32x16_simd_eq_no_match` | No lanes equal → bitmask = 0 |
| `u32x16_simd_ne_partial` | One lane differs → bit 0 set, rest clear |
| `u32x16_simd_lt_threshold` | Lanes 0..7 < 8 → 0x00FF |
| `u32x16_simd_gt_threshold` | Lanes 8..15 > 7 → 0xFF00 |
| `u32x16_simd_le_all_equal` | All lanes le equal value → all bits set |
| `u32x16_simd_ge_all_equal` | All lanes ge equal value → all bits set |
| `u32x16_simd_clamp_basic` | Values 0..15 clamped to [3, 10] |
| `u32x16_simd_clamp_in_range` | Value already in range → unchanged |
| `u32x16_select_blend_lane0` | Blend: mask bit 0 set → lane 0 from self, rest from other |
| `u32x16_zero_is_zero` | zero() → reduce_sum() == 0 |
| `u32x16_to_bitmask_uniform` | to_bitmask(simd_eq(b)) returns u64 |

### 4.2 U16x32 tests (12 new tests)

| Test name | What it checks |
|---|---|
| `u16x32_simd_eq_full_match` | All 32 lanes equal → 0xFFFF_FFFF |
| `u16x32_simd_lt_partial` | Lanes below threshold |
| `u16x32_simd_clamp_basic` | Clamp on 32-lane vector |
| `u16x32_select_blend` | Mask-driven blend |
| `u16x32_zero` | zero() → all zero |
| `u16x32_from_u8x64_lo` | Lower 32 bytes widened correctly |
| `u16x32_from_u8x64_hi` | Upper 32 bytes widened correctly |
| `u16x32_pack_saturate_identity` | Values 0..31 → same values after round-trip |
| `u16x32_pack_saturate_clamp` | Values > 255 → saturated to 255 |
| `u16x32_mullo_basic` | 2 * 3 = 6 per lane |
| `u16x32_shl_basic` | 1 << 2 = 4 per lane |
| `u16x32_shr_basic` | 8 >> 1 = 4 per lane |

### 4.3 U64x8 tests (8 new tests)

| Test name | What it checks |
|---|---|
| `u64x8_simd_eq_full_match` | All 8 lanes equal → 0xFF |
| `u64x8_simd_ne_none` | All equal → simd_ne bitmask = 0 |
| `u64x8_simd_lt_partial` | Threshold test, 4-bit boundary |
| `u64x8_simd_clamp_basic` | Clamp on 8-lane u64 |
| `u64x8_select_blend` | Blend via u8 mask |
| `u64x8_zero` | zero() → all zero |
| `u64x8_simd_gt_full` | All lanes greater → 0xFF |
| `u64x8_to_bitmask_uniform` | Returns u64, top bits zero for 8-lane type |

### 4.4 I-word tests (16 new tests, 4 per type)

Each of `I16x16`, `I16x32`, `I32x16`, `I64x8` gets:
- `{type}_simd_eq_full_match`
- `{type}_simd_lt_partial`
- `{type}_simd_clamp_negative` (negative values — specifically tests signed semantics)
- `{type}_zero_is_zero`

---

## §5 Part B — cfg(miri) dispatch reroute in src/simd.rs

### 5.1 New cfg(miri) block — insert before line 222

```rust
// ── Miri: route crate::simd::* through simd_nightly (core::simd polyfill) ──
//
// Miri cannot execute _mm*_* / vget*_* intrinsics (reports
// "calling a function that requires unavailable target features: avx").
// Under cfg(miri), all types come from `simd_nightly` (core::simd wrappers,
// fully Miri-executable). Consumer code using `use ndarray::simd::F32x16`
// requires zero changes.
//
// `nightly-simd` feature MUST be enabled for Miri runs; enforced by
// scripts/miri-tests.sh (--features nightly-simd).
#[cfg(all(miri, feature = "nightly-simd"))]
pub use crate::simd_nightly::{
    F32x16, F32x8,
    F64x4, F64x8,
    I8x32, I8x64,
    I16x16, I16x32,
    I32x16,
    I64x8,
    U8x32, U8x64,
    U16x32,
    U32x8, U32x16,
    U64x4, U64x8,
    F32Mask16, F32Mask8,
    F64Mask4, F64Mask8,
    // lowercase aliases (added to simd_nightly/mod.rs in this PR)
    f32x16, f32x8,
    f64x4, f64x8,
    i8x32, i8x64,
    i16x16, i16x32,
    i32x16, i64x8,
    u8x32, u8x64,
    u16x32, u32x8, u32x16,
    u64x4, u64x8,
};
```

BF16x16 / BF16x8: gated separately — see OQ-2.

### 5.2 Add `not(miri)` to all existing cfg guards

Six blocks gain `not(miri)` as an additional predicate. Example:
- Before: `#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]`
- After: `#[cfg(all(not(miri), target_arch = "x86_64", target_feature = "avx512f"))]`

Apply to: AVX-512 block, BF16 batch block, BF16 RNE block, BF16 avx512bf16 block, AVX2 fallback blocks (×2), U8x32 block.

### 5.3 simd_nightly/mod.rs — add lowercase type aliases

Append to current re-exports:

```rust
// Lowercase type aliases — matches simd_avx512/simd_avx2 namespace surface
// Required for cfg(miri) dispatch in src/simd.rs.
pub type f32x16 = F32x16;
pub type f32x8  = F32x8;
pub type f64x4  = F64x4;
pub type f64x8  = F64x8;
pub type i8x32  = I8x32;
pub type i8x64  = I8x64;
pub type i16x16 = I16x16;
pub type i16x32 = I16x32;
pub type i32x16 = I32x16;
pub type i64x8  = I64x8;
pub type u8x32  = U8x32;
pub type u8x64  = U8x64;
pub type u16x32 = U16x32;
pub type u32x8  = U32x8;
pub type u32x16 = U32x16;
pub type u64x4  = U64x4;
pub type u64x8  = U64x8;
```

---

## §6 Part B — test plan

### 6.1 Update miri-tests.sh

After Part B lands, remove clauses 1 and 2:

```bash
cargo +nightly miri nextest run -v \
    --no-fail-fast \
    -p ndarray -p ndarray-rand \
    --features approx,serde,nightly-simd \
    -E '!test(/^hpc::framebuffer::pyramid_tests::/)'
```

Comment update: replace the three-clause rationale with a one-clause rationale explaining that pyramid_tests remain excluded only for runtime cost (19+ min each under Miri; not a UB signal).

### 6.2 Confirm-pass representative tests

Previously excluded tests that should now pass:
- `simd::tests::f32x16_splat_reduce` and all other simd::tests::*
- `hpc::activations::tests::test_sigmoid_f32_tail`
- `hpc::fingerprint::*`
- `hpc::quantized::*` (if present)
- `hpc::byte_scan::*` (already passing — confirm not regressed)

### 6.3 CI gate promotion

The PR author should update `.github/workflows/ci.yaml` Miri job from conditional/optional to required after confirming exit 0 on the new script. This is the "miri promotes from optional → required" gate documented in miri-tests.sh.

---

## §7 Files-to-touch table

| File | Change type | Estimated delta |
|---|---|---|
| `/home/user/ndarray/src/simd_nightly/u_word_types.rs` | Add methods on U16x32/U32x16/U64x8 | +110 LOC |
| `/home/user/ndarray/src/simd_nightly/i_word_types.rs` | Add methods on I16x16/I16x32/I32x16/I64x8 | +80 LOC |
| `/home/user/ndarray/src/simd_nightly/mod.rs` | Add 17 lowercase type aliases | +20 LOC |
| `/home/user/ndarray/src/simd_nightly/tests.rs` | Add ~48 new tests | +200 LOC |
| `/home/user/ndarray/src/simd.rs` | cfg(miri) block + not(miri) on 6 existing guards | +40 LOC |
| `/home/user/ndarray/scripts/miri-tests.sh` | Remove 2 exclusion clauses; update comment | -20 LOC |
| `/home/user/ndarray/src/simd_nightly/_original_draft.rs` | **DELETE** (dead, not in mod.rs) | -700 LOC |

Net: +450 LOC, -720 LOC.

---

## §8 Risk matrix

| Risk | Severity | Mitigation |
|---|---|---|
| **cfg(miri) name parity**: simd_nightly methods must match simd_avx512 API exactly. | HIGH | Confirm by running `RUSTFLAGS="--cfg miri" cargo +nightly check --features nightly-simd`. Any mismatch surfaces as compile error. Parity confirmed for float types from direct read; integer types require §3's additions to achieve parity. |
| **select() signature mismatch**: production may use typed `__mmask16`; polyfill uses raw integer bitmask. | HIGH | Run `grep -n "fn select" /home/user/ndarray/src/simd_avx512.rs` before committing. If production takes typed mask, add typed UMask/IMask wrappers. This is OQ-1. |
| **pyramid_tests still slow**: 19+ min each. | MEDIUM | Clause 3 KEPT in miri-tests.sh. Not removed. |
| **BF16x16/BF16x8 feature gating**: may require separate `bf16` feature in cfg(miri) block. | MEDIUM | OQ-2: check Cargo.toml before committing BF16 inclusion in cfg(miri) block. |
| **U16x32 scalar loops under Miri**: `from_u8x64_lo/hi`, `pack_saturate_u8` are slow. | LOW | Correctness not performance is the goal. Document in method comments. |
| **_original_draft.rs hidden references**: file may be referenced outside mod.rs. | LOW | `grep -r "_original_draft" /home/user/ndarray/src/` before deletion. Zero hits = safe. |
| **`core::simd::prelude::Select` import**: not currently imported in u_word_types.rs. | LOW | Add at file top alongside existing `use core::simd::cmp::*` imports. |

---

## §9 Iron rule compliance

- **I-SUBSTRATE-MARKOV** — not applicable (SIMD wrapper layer, not VSA bundling).
- **I-VSA-IDENTITIES** — not applicable (integer SIMD is hardware layer).
- **I-NOISE-FLOOR-JIRAK** — not applicable (no statistical threshold changes).
- **Method-on-carrier**: all new methods are instance methods or associated functions on the U*/I* struct — no free functions violating "object speaks for itself."
- **Zero-cost abstractions**: every new method is `#[inline(always)]` forwarding to `core::simd`. No allocations, no `Box<dyn>`.
- **No serde**: no serde derives added. Debug/Display already present where needed.

---

## §10 Sequencing and cross-spec dependencies

Fully standalone. No dependency on any other sprint-10 worker spec (W1-W12).

Landing order:
1. This PR (PR-NDARRAY-MIRI-COMPLETE) on ndarray repo.
2. PR-CE64-MB-1 par-tile crate (W1 spec) — consumes ndarray SIMD under Miri.
3. PR-CE64-MB-2 through PR-CE64-MB-7 — no ndarray SIMD dependency ordering.

**Branch suggestion**: `claude/ndarray-miri-complete`, based on the ndarray HEAD carrying `e0907cd` and `6590b9e`.

**Approximate implementation time**: 2-4 hours for a focused engineer. Part A is mechanical. Part B is ~30 lines of cfg changes. Tests are the majority of the writing time.

---

## §11 Open questions for meta-review

**OQ-1 — select() API parity (HIGH priority)**: Does `simd_avx512::U32x16::select()` exist, and if so, what is its signature? If production takes a typed `__mmask16` struct, the polyfill must accept the same type. Resolve: `grep -n "fn select" /home/user/ndarray/src/simd_avx512.rs`. If missing from production entirely, select() on polyfill can use any reasonable signature. If present with typed mask, this spec's §3.2 must be revised to use typed `UMask{N}` wrappers.

**OQ-2 — BF16 feature gating under cfg(miri) (MEDIUM priority)**: `BF16x16` and `BF16x8` are conditionally compiled behind `target_feature = "avx512bf16"` in production (line 272). Their polyfill in `bf16_types.rs` may require a `bf16` or `half` feature. The cfg(miri) block in §5.1 excludes BF16 from the primary block; they may need a separate `#[cfg(all(miri, feature = "nightly-simd", feature = "bf16"))]` block. Resolve: `grep -n "BF16\|half\|bf16" /home/user/ndarray/Cargo.toml`.

**OQ-3 — F16x16 inclusion (LOW priority)**: `simd_nightly/mod.rs` exports `F16x16` but `simd_avx512.rs` likely doesn't export it (F16 intrinsics are post-AVX-512). The cfg(miri) block correctly omits F16x16 — confirm: `grep -n "F16x16\|f16" /home/user/ndarray/src/simd_avx512.rs`.

---

## §12 Janitorial cleanup (same PR)

- **Delete** `src/simd_nightly/_original_draft.rs` after grep confirms zero references.
- **Update** simd.rs comment block (lines 205-220): change "The remaining work for Miri-clean coverage" to "The cfg(miri) dispatch below closes this gap; the remaining open item is hpc::framebuffer::pyramid_tests (runtime cost, see miri-tests.sh)."
- **Add** one-line doc update to `simd_nightly/mod.rs` module doc: "Under `cfg(miri)`, `crate::simd::*` routes here automatically via `src/simd.rs` dispatch."

---

## §13 Confidence summary

| Area | Confidence | Notes |
|---|---|---|
| Part A — u-word/i-word method list | HIGH | Direct file read confirms all gaps. |
| Part A — implementation correctness | HIGH | All methods are 1-line forwards to core::simd traits already imported. |
| Part A — U16x32 widening helpers | HIGH | Scalar loops are trivially correct; production paths unaffected. |
| Part B — cfg(miri) block structure | HIGH | Standard Rust cfg attribute, well-understood. |
| Part B — name parity | MEDIUM | Confirmed for float types; OQ-1 (select signature) could require §3.2 revision. |
| Part B — BF16 gating | MEDIUM | OQ-2 unresolved; may need feature-gated subblock. |
| Overall PR scope | HIGH | Well-bounded, 2-4 hour implementation, zero risk to existing tests. |
