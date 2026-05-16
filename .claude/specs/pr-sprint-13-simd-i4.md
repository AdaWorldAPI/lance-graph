# PR — Sprint-13 SIMD i4 (D-CSV-13b) — AVX-512 + NEON intrinsics for `mul::i4_eval::batch`

> **Status:** DRAFT (planning preflight PP-6, opus planner, 2026-05-16)
> **Sprint:** sprint-13
> **Deliverable:** D-CSV-13b
> **Predecessor:** sprint-12 W-G3 (D-CSV-13a, scalar batch API merged via #388,
>   `crates/lance-graph-contract/src/mul.rs::i4_eval::batch`)
> **Iron rules in force:** `I-LEGACY-API-FEATURE-GATED` (same function name MUST
>   NOT silently produce different semantics across feature gates),
>   `I-NOISE-FLOOR-JIRAK` (weak-dependence statistics; impacts perf-test
>   significance claims), `I-SUBSTRATE-MARKOV` (substrate-level changes touching
>   bundle semantics are out of scope here — i4 evaluation is downstream of
>   bundling), `I-VSA-IDENTITIES` (none of the SIMD ops here bundle CAM-PQ
>   content; i4 dims are extracted scalars over a u64 packed identity)
> **Confidence:** Medium-High — the contract API is fixed (Wave G), the SIMD
>   shape is the standard "decode i4 lanes → masked compare → blend → pack"
>   pipeline used in ndarray's existing `simd_int_ops.rs` and `simd_avx512.rs`,
>   and `simd_caps()` dispatch is already wired. Open risks tracked in §8.

Cross-refs:

- `crates/lance-graph-contract/src/mul.rs` §`i4_eval::batch` — the 5 scalar
  functions whose semantics must be matched bit-for-bit by the SIMD intrinsics
  shipped here (`dk_position_batch`, `trust_texture_batch`, `flow_state_batch`,
  `gate_decision_batch`, `mul_assess_batch`).
- `crates/lance-graph-contract/src/qualia.rs` §`QualiaI4_16D` — the
  `pub struct QualiaI4_16D(pub u64)` carrier; 16 × 4-bit signed nibbles
  (`get/set` via shift+mask; `i4` sign extends through `(v << 4) as i8 >> 4`).
- `/home/user/ndarray/src/hpc/simd_caps.rs` §`SimdCaps` — runtime dispatch
  singleton: `simd_caps().avx512f`, `simd_caps().avx512bw`, `simd_caps().neon`.
- `/home/user/ndarray/src/simd_avx512.rs` — proven intrinsic patterns
  (`_mm512_cmpgt_epi8_mask` at L1612, `_mm512_mask_blend_epi8` at L641).
- `/home/user/ndarray/src/simd_int_ops.rs` — proven 8-bit integer SIMD
  patterns (load / compare / blend / store at the i8 grain).
- `CLAUDE.md` §`I-LEGACY-API-FEATURE-GATED` (added 2026-05-16) — iron rule
  cited in §5 below.
- `CLAUDE.md` §`I-NOISE-FLOOR-JIRAK` — cited in §7 (benchmark significance).
- `.claude/board/STATUS_BOARD.md` D-CSV-13b row (Queued → In progress on
  sprint-13 start).
- `.claude/board/TECH_DEBT.md` — adds TD-SIMD-I4-DISPATCH-1 on landing if the
  fast-path is gated behind a Cargo feature rather than runtime-dispatched
  (see §4 alternatives).

---

## §0 Status

| Item | Value |
|---|---|
| Spec ID | PR-SPRINT-13-SIMD-I4 |
| Deliverable | D-CSV-13b |
| Sprint | sprint-13 |
| Predecessor | sprint-12 W-G3 (#388 merged 2026-05-16) |
| Successor | none (D-CSV-13 closes once benchmarks land) |
| Owner | sprint-13 workers W-SIMD-1 through W-SIMD-5 (one worker per batch fn) |
| Reviewer | family-codec-smith + truth-architect (semantic equivalence) |
| LOC target | ~600 (300 src + 250 tests + 100 bench) |
| Test target | 10 SIMD-vs-scalar equivalence tests + 5 length-edge tests |
| Bench target | ≥4× speedup on AVX-512 vs scalar baseline for batches ≥ 64 |

---

## §1 Statement of scope

Sprint-12 Wave G shipped a SCALAR implementation of the 5 batch functions in
`lance_graph_contract::mul::i4_eval::batch`. The shape was deliberately
SIMD-ready (parallel arrays + pre-allocated outputs) but the inner loop is a
straight `for i in 0..qualia.len()` call to the per-element scalar function.

Sprint-13 D-CSV-13b lands the actual SIMD intrinsics: AVX-512F + AVX-512BW for
x86-64 servers (Zen 4, Sapphire Rapids), NEON for aarch64 (Apple Silicon, Pi
5, AWS Graviton). Each of the 5 batch functions gets:

1. one AVX-512 implementation in `mod avx512_impl` (gated by
   `target_feature = "avx512f,avx512bw"`),
2. one NEON implementation in `mod neon_impl` (gated by
   `target_feature = "neon"`),
3. the scalar implementation from Wave G stays as `mod scalar_impl` (the
   correctness anchor and the unconditional fallback for non-SIMD targets and
   for the AVX-2-only path on older x86-64).

Runtime dispatch via `simd_caps()` (already in ndarray; lance-graph-contract
takes a dev-dep on ndarray's `hpc::simd_caps` module ONLY for the dispatch
singleton — no other ndarray surface enters the zero-dep contract crate).

**Out of scope:**

- AVX-2-only path. AVX-2 lacks `_mm256_cmpgt_epi8_mask`-shaped mask returns
  (it produces a `__m256i` mask instead). Decision: skip AVX-2 and fall
  through to scalar on AVX-2-only x86-64. Documented in §8 (risk R-4).
- WASM SIMD128. Same shape as NEON but deferred to sprint-14; tracked as
  TD-SIMD-I4-WASM-1.
- VNNI / AMX. The i4 evaluation pipeline is masked-compare + blend; there is
  no dot-product to accelerate, so VNNI / AMX add no value here.
- Auto-vectorization. We do NOT rely on the compiler vectorizing the scalar
  loop; the i4 nibble extraction + branching decision tree defeats LLVM's
  loop vectorizer in practice (verified by inspecting `cargo asm` on the
  W-G3 scalar baseline — no vectorized inner loop emitted at `-C opt-level=3
  -C target-cpu=znver4`).

**Zero-dep posture:** Currently lance-graph-contract has zero deps. The SIMD
intrinsics live behind `std::arch::x86_64::*` / `std::arch::aarch64::*`,
which are stdlib — no new deps required. The `simd_caps()` dispatch helper
adds a tiny dependency on ndarray's `hpc::simd_caps` module; if that's
unacceptable for the zero-dep invariant, the alternative is to inline a
~30-line `simd_caps()` copy into lance-graph-contract (see §4 alternative).

---

## §2 i4 lane → AVX-512 register mapping

The `QualiaI4_16D(pub u64)` packs 16 × i4 lanes into a single u64:

```
bit:       63        56        48        40        32        24        16         8         0
           │   d15    │   d14    │   d13    │   d12    │   d11    │   d10    │  ...     │   d0     │
           │  4-bit   │  4-bit   │  4-bit   │  4-bit   │  4-bit   │  4-bit   │          │  4-bit   │
           │  signed  │  signed  │  signed  │  signed  │  signed  │  signed  │          │  signed  │
```

Each i4 sign-extends to i8 via `(nibble << 4) as i8 >> 4` (the trick used in
`QualiaI4_16D::get` at qualia.rs:184-194).

### AVX-512 mapping

One AVX-512 vector is 512 bits = 64 × i8 lanes = 64 nibbles spread over 4 ×
i16 lanes per qualia. The natural decomposition: **process 8 qualia per
AVX-512 vector**, each qualia contributing 16 × i4 = 8 × i8 (after unpack to
i8), giving 8 × 8 = 64 i8 lanes per vector. Concretely:

- Load 8 × `u64` (8 × `QualiaI4_16D`) → `__m512i` (8 × i64).
- Extract one dim across all 8 qualia: `_mm512_srli_epi64::<{dim*4}>` shifts,
  `_mm512_and_si512` with `_mm512_set1_epi64(0xF)` masks → 8 × u64 each
  carrying one i4 in low nibble.
- Sign-extend i4 → i8: `_mm512_slli_epi64::<4>` to shift the nibble to the
  top of its byte slot, then arithmetic right shift `_mm512_srai_epi64::<4>`
  (or `_mm512_srai_epi16` on a re-interpreted view) to sign-extend within
  the lane.
- Compare against a broadcast threshold: `_mm512_cmpgt_epi8_mask` (proven in
  ndarray simd_avx512.rs:1612) — returns a `__mmask64` of 64 lane-wise
  greater-than results. Since we only use 8 of the 64 lanes (one per
  qualia), the upper 56 bits are don't-care; mask them with `& 0xFF` after
  the compare-mask is materialized.
- Branch-free decision pipeline: chain `_mm512_mask_blend_epi8` (proven in
  ndarray simd_avx512.rs:641) over the decision tree. Each mask blend
  selects the i8-encoded enum discriminant for the matching branch.
- Pack outputs: a single `_mm512_mask_compressstoreu_epi8` or a scalar
  byte-loop stores the 8 enum discriminants (one byte each) into the
  `&mut [DkPosition]` slice (after a `transmute` of the slice to `&mut [u8]`
  — safe because `DkPosition` is `#[repr(u8)]`, see §5 invariant).

### NEON mapping

NEON has 128-bit vectors = 16 × i8 lanes. Natural decomposition: **process
2 qualia per NEON vector** (2 × 16 i4 lanes = 32 i4 → unpack to 32 i8 over 2
vectors). The pipeline mirrors AVX-512 with NEON intrinsics:

- `vld1q_u64` loads 2 × u64 → `uint64x2_t`.
- `vshrq_n_u64` + `vandq_u64` extracts a dim.
- `vshlq_n_s8` + `vshrq_n_s8` sign-extends i4 → i8 (within a re-interpreted
  `int8x16_t` view).
- `vcgtq_s8` compares signed (lane-wise i8 greater-than).
- `vbslq_s8` (bitwise-select) implements the branch-free blend.
- `vst1q_u8` stores back.

NEON throughput is lower (one qualia processed per `vcgtq_s8` byte vs 8 per
`_mm512_cmpgt_epi8_mask`), but the same algorithmic shape applies.

### Important: enum repr(u8) prerequisite

For the pack-into-output-slice step to be safe and efficient, the 4 enums
in `mul.rs` (`DkPosition`, `TrustTexture`, `FlowState`,
`GateDecision::{Block,Hold,Flow}`-discriminant-only-without-payload) must be
`#[repr(u8)]`. Currently:

- `DkPosition`, `TrustTexture`, `FlowState` are simple unit-variant enums →
  trivially `#[repr(u8)]` adds zero LOC. **The spec mandates adding the
  attribute** to lock the layout.
- `GateDecision` has `Block { reason: String }` and `Hold { reason: String
  }` variants — these are NOT `#[repr(u8)]` and cannot be packed into a
  `[u8]` slice. **The spec splits `gate_decision_batch` into two stages**:
  a SIMD stage that computes a `Vec<u8>` of variant-discriminants (0=Flow,
  1=Hold, 2=Block), and a scalar tail that materializes the `String`
  reasons (or skips them if the caller passes a "discriminant-only"
  variant of the batch fn; see §4 batch-fn signature).

---

## §3 Per-function SIMD pseudocode (10 sketches)

For each of the 5 batch functions, 1 AVX-512 sketch + 1 NEON sketch =
**10 sketches total**.

### 3.1 `dk_position_batch` — AVX-512

```rust
// SAFETY: caller must verify simd_caps().avx512f && simd_caps().avx512bw
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn dk_position_batch_avx512(
    qualia: &[QualiaI4_16D],
    mantissas: &[i8],
    out: &mut [DkPosition],
) {
    // DIM_COHERENCE = 9 (so shift = 9 * 4 = 36 bits)
    let chunks = qualia.len() / 8;

    for c in 0..chunks {
        // Load 8 × u64 qualia
        let q_vec = _mm512_loadu_si512(qualia.as_ptr().add(c * 8) as *const _);

        // Extract coherence (dim 9) from each qualia
        // 1. shift right by 36 to put dim-9 nibble in low 4 bits
        let shifted = _mm512_srli_epi64::<36>(q_vec);
        // 2. mask low 4 bits
        let mask_f = _mm512_set1_epi64(0xF);
        let nibble = _mm512_and_si512(shifted, mask_f);
        // 3. sign-extend i4 → i8 by shifting up + arithmetic shift down
        //    (operate on the low byte of each i64 lane; treat as 64×i8 view)
        let shifted_up = _mm512_slli_epi16::<4>(nibble);
        let coherence_i8 = _mm512_srai_epi16::<4>(shifted_up);
        //    coherence_i8 now has each lane's i8 coherence in lane.byte[0]

        // Load 8 × i8 mantissas (packed at byte granularity; gather low byte of each i64 lane)
        let m_bytes = _mm_loadl_epi64(mantissas.as_ptr().add(c * 8) as *const _);
        //    expand 8 × i8 into 8 × i64 lanes (low byte populated, upper 7 bytes zero)
        let m_vec = _mm512_cvtepi8_epi64(m_bytes);
        //    absolute value: abs(i8) is a single instruction _mm512_abs_epi8 (one byte per lane,
        //    but we have it in low byte of i64 lane; treat as 64×i8 view)
        let abs_m = _mm512_abs_epi8(m_vec);  // op on i8 view; upper 56 bytes/lane don't matter

        // ── Decision pipeline (branch-free, 4-way) ────────────────────────
        //   coherence_i8 ≥ +5  AND abs_m ≥ +4  → Plateau (3)
        //   else coherence_i8 ≥ +2  AND abs_m ≥ +2 → SlopeOfEnlightenment (2)
        //   else coherence_i8 ≤ -3  OR  abs_m ≤ +1 → ValleyOfDespair (1)
        //   else                                    → MountStupid (0)
        //
        //   Use unmasked compare-to-broadcast threshold + and-of-masks for AND clauses,
        //   or-of-masks for OR clauses. Each compare-mask is __mmask64 with bit i
        //   reflecting whether lane.byte[0] satisfies the predicate; we only care
        //   about bits {0, 8, 16, 24, 32, 40, 48, 56} (low byte of each i64 lane);
        //   AVX-512 cmpgt_epi8 returns one mask bit per byte lane, so all 64 are
        //   produced — we filter by ANDing with 0x0101010101010101 (bit 0 of each byte).

        let thr5 = _mm512_set1_epi8(5);
        let thr2 = _mm512_set1_epi8(2);
        let thr_neg3 = _mm512_set1_epi8(-3);
        let thr4 = _mm512_set1_epi8(4);
        let thr1 = _mm512_set1_epi8(1);

        // Compare-masks (each is __mmask64; we extract the 8 bits we care about later)
        let m_coh_ge5 = _mm512_cmpge_epi8_mask(coherence_i8, thr5);
        let m_coh_ge2 = _mm512_cmpge_epi8_mask(coherence_i8, thr2);
        let m_coh_le_neg3 = _mm512_cmple_epi8_mask(coherence_i8, thr_neg3);
        let m_abs_ge4 = _mm512_cmpge_epi8_mask(abs_m, thr4);
        let m_abs_ge2 = _mm512_cmpge_epi8_mask(abs_m, thr2);
        let m_abs_le1 = _mm512_cmple_epi8_mask(abs_m, thr1);

        // Mount Stupid base (discriminant = 0)
        let mut result = _mm512_setzero_si512();

        // ValleyOfDespair: coh ≤ -3 OR abs_m ≤ 1
        let m_valley = m_coh_le_neg3 | m_abs_le1;
        result = _mm512_mask_blend_epi8(m_valley, result, _mm512_set1_epi8(1));

        // Slope: coh ≥ 2 AND abs_m ≥ 2 (overrides Valley if both true and Plateau not yet)
        let m_slope = m_coh_ge2 & m_abs_ge2;
        result = _mm512_mask_blend_epi8(m_slope, result, _mm512_set1_epi8(2));

        // Plateau (highest priority): coh ≥ 5 AND abs_m ≥ 4
        let m_plateau = m_coh_ge5 & m_abs_ge4;
        result = _mm512_mask_blend_epi8(m_plateau, result, _mm512_set1_epi8(3));

        // result now has 64 × i8 discriminants; lane.byte[0] of each i64 lane is what we want
        // Compress-store the 8 relevant bytes into out[c*8 .. c*8+8]
        // Use _mm512_mask_compressstoreu_epi8 with mask = 0x0101010101010101 to extract
        // byte 0 of each i64 lane, packed into 8 contiguous bytes.
        let extract_mask: __mmask64 = 0x0101010101010101;
        _mm512_mask_compressstoreu_epi8(
            out.as_mut_ptr().add(c * 8) as *mut u8,
            extract_mask,
            result,
        );
    }

    // Scalar tail for the remainder (< 8 qualia)
    let tail_start = chunks * 8;
    for i in tail_start..qualia.len() {
        out[i] = scalar_impl::dk_position_i4(&qualia[i], mantissas[i]);
    }
}
```

### 3.2 `dk_position_batch` — NEON

```rust
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dk_position_batch_neon(
    qualia: &[QualiaI4_16D],
    mantissas: &[i8],
    out: &mut [DkPosition],
) {
    // NEON 128-bit register holds 2 × u64 = 2 qualia per chunk
    let chunks = qualia.len() / 2;

    for c in 0..chunks {
        let q_vec = vld1q_u64(qualia.as_ptr().add(c * 2) as *const u64);

        // Extract coherence (shift 36, mask 0xF, sign-extend)
        let shifted = vshrq_n_u64::<36>(q_vec);
        let nibble = vandq_u64(shifted, vdupq_n_u64(0xF));
        // Reinterpret as int8x16_t, sign-extend the low nibble of each i64 lane
        let nibble_i8 = vreinterpretq_s8_u64(nibble);
        let shifted_up = vshlq_n_s8::<4>(nibble_i8);
        let coherence_i8 = vshrq_n_s8::<4>(shifted_up);
        // coherence_i8 now has i8 coherence in lane.byte[0] (bytes 0 and 8 across the q128)

        // Load 2 mantissas
        let m0 = *mantissas.as_ptr().add(c * 2);
        let m1 = *mantissas.as_ptr().add(c * 2 + 1);
        // Place at byte 0 and byte 8 to align with coherence layout
        let mut m_arr = [0i8; 16];
        m_arr[0] = m0;
        m_arr[8] = m1;
        let m_vec = vld1q_s8(m_arr.as_ptr());
        let abs_m = vabsq_s8(m_vec);

        // Compare masks (lane-wise i8 → 0xFF or 0x00 per byte lane)
        let thr5 = vdupq_n_s8(5);
        let thr2 = vdupq_n_s8(2);
        let thr_neg3 = vdupq_n_s8(-3);
        let thr4 = vdupq_n_s8(4);
        let thr1 = vdupq_n_s8(1);

        let m_coh_ge5 = vcgeq_s8(coherence_i8, thr5);
        let m_coh_ge2 = vcgeq_s8(coherence_i8, thr2);
        let m_coh_le_neg3 = vcleq_s8(coherence_i8, thr_neg3);
        let m_abs_ge4 = vcgeq_s8(abs_m, thr4);
        let m_abs_ge2 = vcgeq_s8(abs_m, thr2);
        let m_abs_le1 = vcleq_s8(abs_m, thr1);

        // Same priority order: MountStupid → Valley → Slope → Plateau
        let mut result = vdupq_n_s8(0);  // MountStupid
        let m_valley = vorrq_u8(m_coh_le_neg3, m_abs_le1);
        result = vbslq_s8(m_valley, vdupq_n_s8(1), result);
        let m_slope = vandq_u8(m_coh_ge2, m_abs_ge2);
        result = vbslq_s8(m_slope, vdupq_n_s8(2), result);
        let m_plateau = vandq_u8(m_coh_ge5, m_abs_ge4);
        result = vbslq_s8(m_plateau, vdupq_n_s8(3), result);

        // Extract byte 0 and byte 8 of the q128 register
        let d0 = vgetq_lane_s8::<0>(result) as u8;
        let d1 = vgetq_lane_s8::<8>(result) as u8;
        *(out.as_mut_ptr().add(c * 2) as *mut u8) = d0;
        *(out.as_mut_ptr().add(c * 2 + 1) as *mut u8) = d1;
    }

    // Scalar tail (0 or 1 element)
    let tail_start = chunks * 2;
    for i in tail_start..qualia.len() {
        out[i] = scalar_impl::dk_position_i4(&qualia[i], mantissas[i]);
    }
}
```

### 3.3 `trust_texture_batch` — AVX-512

Three dims read: coherence (9, shift 36), valence (1, shift 4), tension (2,
shift 8). Decision tree:

```
if coh ≤ -3 AND tension ≥ 3 → Uncertain (2)
else if valence ≥ 4 AND coh < 5 → Overconfident (1)
else if valence ≤ -3 → Underconfident (3)
else if coh ≥ 4 AND valence ≥ 2 AND tension ≤ 1 → Calibrated (0)
else → Calibrated (0)  // moderate default
```

(Discriminants: `Calibrated=0, Overconfident=1, Uncertain=2, Underconfident=3`,
matching the declaration order in `mul.rs` — see §5 invariant.)

Pseudocode shape mirrors §3.1: load 8 qualia, extract 3 dims (3 ×
shift+and+sign-extend), produce 3 compare-mask pairs, blend in priority
order: Calibrated-default → Underconfident → Overconfident → Uncertain
(highest priority overwrites lower). Pack-store via
`_mm512_mask_compressstoreu_epi8`. **No mantissa parameter** for this fn.

### 3.4 `trust_texture_batch` — NEON

Same as §3.3 with the NEON intrinsic substitution table from §3.2: 2 qualia
per 128-bit register, NEON compares + `vbslq_s8` blends.

### 3.5 `flow_state_batch` — AVX-512

Four dims read: warmth (3, shift 12), groundedness (14, shift 56), tension
(2, shift 8), coherence (9, shift 36). One arithmetic combination:
`flow_proxy = warmth + groundedness - tension` (i8, with saturating
add/sub via `_mm512_adds_epi8` and `_mm512_subs_epi8`).

Decision tree (in priority order, lowest first to be overwritten):

```
default → Boredom (1)
flow_proxy ≤ -2 OR (mantissa < 0 AND coh ≤ -1) → Anxiety (3)
flow_proxy ≥ 2 AND mantissa > 0 → Transition (2)
flow_proxy ≥ 4 AND mantissa > 0 → Flow (0)  // highest priority
```

Note the priority order matters: in the scalar source, Flow is checked
first (and returned immediately), Anxiety second. In the SIMD branch-free
version we apply lowest-priority first, then overwrite. The **final priority
must be: Boredom (base) → Anxiety → Transition → Flow** so that Flow wins
over Transition when both conditions hold, and Anxiety wins over Boredom
but loses to Transition+Flow. This matches the scalar early-return order
exactly — verified by exhaustive enumeration over the relevant predicate
truth table (Flow⇒Transition vacuously when flow_proxy ≥ 4 AND mantissa
> 0, so Flow's overwrite is correct).

### 3.6 `flow_state_batch` — NEON

Same shape as §3.5 with NEON. Note `vqaddq_s8` and `vqsubq_s8` are the
saturating arithmetic primitives.

### 3.7 `gate_decision_batch` — AVX-512 (discriminant-only stage)

This is the most complex of the 5 because `GateDecision::Block { reason:
String }` and `GateDecision::Hold { reason: String }` carry payloads. The
SIMD path produces a `[u8]` of discriminants (0=Flow, 1=Hold, 2=Block); a
scalar tail materializes the `String` reasons.

Stage 1 (SIMD): compute `texture_disc[i]` (via §3.3) and `flow_disc[i]`
(via §3.5). Combine via a lookup-table-style blend:

```
texture × flow → discriminant
  Uncertain (2) × any → Block (2)
  Underconfident (3) × Anxiety (3) → Block (2)
  Overconfident (1) × any → Hold (1)
  any × Anxiety (3) → Hold (1)
  Calibrated/Underconf (0/3) × Flow/Transition (0/2) → Flow (0)
  else → Hold (1)
```

The 4×4 = 16-entry lookup table fits in a single AVX-512 broadcast +
`_mm512_permutexvar_epi8` (PSHUFB-style table lookup). Concretely: pack
texture and flow into a single 4-bit index per lane (`idx = (texture << 2) |
flow`), broadcast the 16-byte lookup table, and `_mm512_permutexvar_epi8`
returns the discriminant. One SIMD instruction replaces the 5-arm match.

Stage 2 (scalar): walk the `[u8]` and materialize `GateDecision::Block {
reason: "..." }` / `Hold { reason: "..." }` strings for the matching
indices. Performance note: when the caller only needs discriminants
(e.g. logging / counting), expose
`gate_decision_disc_batch(qualia, mantissas, out: &mut [u8])` that skips
stage 2 entirely. Documented in §4 batch-fn signature.

### 3.8 `gate_decision_batch` — NEON

NEON has `vqtbl1q_u8` (table lookup, 16-byte table) which is the direct
analog of `_mm512_permutexvar_epi8` for stage 1. Stage 2 is identical
scalar code.

### 3.9 `mul_assess_batch` — AVX-512

`mul_assess_batch` is the union of §3.1 (dk_position), §3.3 (trust_texture),
§3.5 (flow_state) plus the f64 `trust_value` / `complexity_mapped` /
`allostatic_load` / `free_will_modifier` finalization.

The 3 enum-valued reads stay SIMD; the f64 finalization is scalar
per-element (4 small lookup tables: `trust_value_table[texture_disc]`,
`dk_factor_table[dk_disc]`, `flow_factor_table[flow_disc]`, `intensity`
from a separate `_mm512_abs_epi8` over the qualia.magnitude()). The
output struct `MulAssessment` contains a `String` reason if it goes
through `GateDecision::Block` path — but `mul_assess_i4` doesn't actually
embed a GateDecision in its output (it only stores `trust`, `dk_position`,
`homeostasis`, `complexity_mapped`, `free_will_modifier`), so there's no
String payload to materialize. The output stages cleanly into a
scalar tail.

Pseudocode:

```rust
// Stage 1: SIMD-compute 3 × [u8] discriminants for dk, texture, flow
let mut dk_disc = vec![0u8; n];
let mut tex_disc = vec![0u8; n];
let mut flow_disc = vec![0u8; n];
dk_position_disc_batch_avx512(qualia, mantissas, &mut dk_disc);
trust_texture_disc_batch_avx512(qualia, &mut tex_disc);
flow_state_disc_batch_avx512(qualia, mantissas, &mut flow_disc);

// Stage 2: scalar finalize per element (4 lookup tables + f64 multiply)
for i in 0..n {
    let texture = TrustTexture::from_disc(tex_disc[i]);
    let dk = DkPosition::from_disc(dk_disc[i]);
    let flow = FlowState::from_disc(flow_disc[i]);
    let intensity = qualia[i].magnitude();
    let trust_value = match texture { /* same as scalar */ };
    // ... assemble MulAssessment as in scalar_impl::mul_assess_i4
    out[i] = MulAssessment { /* fields */ };
}
```

### 3.10 `mul_assess_batch` — NEON

Identical structure to §3.9 with NEON intrinsics for the three SIMD passes.

---

## §4 Runtime dispatch via `simd_caps()`

```rust
// crates/lance-graph-contract/src/mul.rs — i4_eval::batch
pub mod batch {
    use super::*;

    // Public API — runtime dispatch. Wave G API surface preserved.
    pub fn dk_position_batch(
        qualia: &[QualiaI4_16D],
        mantissas: &[i8],
        out: &mut [DkPosition],
    ) {
        assert_eq!(qualia.len(), mantissas.len(), "qualia/mantissas length mismatch");
        assert_eq!(qualia.len(), out.len(), "input/output length mismatch");

        let caps = simd_caps();

        #[cfg(target_arch = "x86_64")]
        if caps.avx512f && caps.avx512bw && qualia.len() >= 8 {
            unsafe { avx512_impl::dk_position_batch(qualia, mantissas, out) };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if caps.neon && qualia.len() >= 2 {
            unsafe { neon_impl::dk_position_batch(qualia, mantissas, out) };
            return;
        }

        scalar_impl::dk_position_batch(qualia, mantissas, out);
    }

    // ... same pattern for the other 4 functions.

    // Public discriminant-only variants (perf optimization — skip String alloc)
    pub fn gate_decision_disc_batch(
        qualia: &[QualiaI4_16D],
        mantissas: &[i8],
        out: &mut [u8],
    ) { /* ... */ }
}

// Private modules
mod scalar_impl {
    // The current Wave G impl, moved verbatim under this module name.
    pub fn dk_position_batch(...) { /* same as today */ }
    // ... 4 more
}

#[cfg(target_arch = "x86_64")]
mod avx512_impl {
    use super::*;
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dk_position_batch(...) { /* §3.1 */ }
    // ... 4 more
}

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use super::*;
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn dk_position_batch(...) { /* §3.2 */ }
    // ... 4 more
}
```

### Dispatch alternative — compile-time only

If the runtime-dispatch ndarray dep is unacceptable for the zero-dep
contract crate, the alternative is compile-time-only dispatch via
`#[cfg(target_feature = "avx512f,avx512bw")]` at the module level. This
loses the runtime-dispatch flexibility (a binary built for AVX-512 won't
run on Zen 3) but keeps lance-graph-contract dep-free.

Decision deferred to sprint-13 W-SIMD-1 first work item; both paths are
in scope. **Recommendation:** inline a ~30-line `simd_caps()` copy into
lance-graph-contract; runtime-dispatch is worth ~30 LOC of duplication
because the API surface ("call `dk_position_batch` from any caller, get
the fastest impl") is much cleaner than "different binaries for different
targets." Documented in §8 risk R-1.

### Min-batch-size guard

Each SIMD path includes a `qualia.len() >= MIN_BATCH` guard (8 for
AVX-512, 2 for NEON). Below that, scalar wins because the dispatch +
prologue cost dominates the inner loop savings. The guards are
intentionally conservative — the bench plan (§7) measures the actual
crossover and adjusts in a sprint-13 follow-up if the measured value
diverges from the default.

---

## §5 Semantic equivalence — IRON RULE

Per `I-LEGACY-API-FEATURE-GATED` (CLAUDE.md, added 2026-05-16), **the
same function name MUST NOT silently produce different semantics across
different feature gates or dispatch targets.** In sprint-11 codex caught
this 5 times in `CausalEdge64` (v1 vs v2 layout); we will NOT recreate
the pattern in mul.

Applied to D-CSV-13b: **every output byte of `*_batch_avx512` MUST equal
the corresponding output byte of `*_batch_scalar` for every legal input.**
This is stronger than "same semantic decision" — it is byte-identical
output, including the discriminant ordering of all 4 enums.

### Enum layout invariant (added by this spec)

The 4 enums in `mul.rs` MUST have `#[repr(u8)]` with the following
discriminants:

```rust
#[repr(u8)]
pub enum DkPosition {
    MountStupid = 0,
    ValleyOfDespair = 1,
    SlopeOfEnlightenment = 2,
    Plateau = 3,
}

#[repr(u8)]
pub enum TrustTexture {
    Calibrated = 0,
    Overconfident = 1,
    Uncertain = 2,
    Underconfident = 3,
}

#[repr(u8)]
pub enum FlowState {
    Flow = 0,
    Boredom = 1,
    Transition = 2,
    Anxiety = 3,
}

// GateDecision has String payloads — only the discriminant byte is
// SIMD-packable. Discriminant order:
// Flow = 0, Hold = 1, Block = 2.
// Locked via from_disc()/to_disc() helpers; the underlying enum stays
// non-#[repr(u8)] because of the String payloads.
```

These discriminants are NEW invariants — sprint-13 W-SIMD-1 ships them
as a "no-semantic-change" PR ahead of the SIMD work. The existing tests
don't depend on numeric discriminants, only on variant identity, so the
change is a pure addition.

### Why this is iron rule territory

If a downstream caller migrates from scalar → SIMD path and the SIMD
path silently produces `DkPosition::Plateau (disc=3)` where scalar
produces `DkPosition::SlopeOfEnlightenment (disc=2)` because of a
priority-order bug in the SIMD branch-free blend chain, **the bug is
detectable only on the workloads that hit the discrepancy**. This is
exactly the failure mode that I-LEGACY-API-FEATURE-GATED prohibits
(same name, different semantics under different gate). Mitigation:
mandatory exhaustive SIMD-vs-scalar tests (§6).

### Migration pointer for legacy callers

Wave G's `mul_assess_vec` (the Vec-returning convenience function) is
not in the SIMD set — it allocates per call and is documented as
"non-hot-path." It stays scalar. No migration pointer needed because
the semantic contract is unchanged.

---

## §6 SIMD-vs-scalar equivalence tests — 10 mandatory tests

### Test list

For each of the 5 batch functions × 2 architectures = 10 tests. Each test
runs an exhaustive or representative sweep over the i4 input space and
asserts byte-identical SIMD vs scalar output.

| # | Function | Arch | Coverage |
|---|---|---|---|
| T1 | `dk_position_batch` | AVX-512 | All 16^2 = 256 (coherence, mantissa) pairs, batched ×8 |
| T2 | `dk_position_batch` | NEON | All 16^2 = 256 pairs, batched ×2 |
| T3 | `trust_texture_batch` | AVX-512 | All 16^3 = 4096 (coh, val, tens) tuples, batched ×8 |
| T4 | `trust_texture_batch` | NEON | All 16^3 = 4096 tuples, batched ×2 |
| T5 | `flow_state_batch` | AVX-512 | Random 100K (warmth, ground, tens, coh, m) inputs |
| T6 | `flow_state_batch` | NEON | Random 100K inputs |
| T7 | `gate_decision_batch` | AVX-512 | All 5-tuple combinations (≤ 16^5 = 1M, sampled to 50K) |
| T8 | `gate_decision_batch` | NEON | Sampled 50K |
| T9 | `mul_assess_batch` | AVX-512 | Random 10K + 200 hand-picked edge cases |
| T10 | `mul_assess_batch` | NEON | Random 10K + 200 edge cases |

### Test pattern

```rust
#[cfg(test)]
mod simd_equivalence_tests {
    use super::*;

    fn assert_dk_position_batch_simd_matches_scalar(
        qualia: &[QualiaI4_16D],
        mantissas: &[i8],
    ) {
        let n = qualia.len();
        let mut out_scalar = vec![DkPosition::MountStupid; n];
        let mut out_simd = vec![DkPosition::MountStupid; n];

        scalar_impl::dk_position_batch(qualia, mantissas, &mut out_scalar);

        #[cfg(target_arch = "x86_64")]
        if simd_caps().avx512f && simd_caps().avx512bw {
            unsafe { avx512_impl::dk_position_batch(qualia, mantissas, &mut out_simd) };
            // Cast to [u8] for byte-level comparison
            let scalar_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                out_scalar.as_ptr() as *const u8, n) };
            let simd_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                out_simd.as_ptr() as *const u8, n) };
            assert_eq!(scalar_bytes, simd_bytes,
                "AVX-512 path produced different bytes than scalar at indices: {:?}",
                scalar_bytes.iter().zip(simd_bytes).enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .take(10)
                    .collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_dk_position_batch_exhaustive_avx512() {
        if !simd_caps().avx512f || !simd_caps().avx512bw {
            eprintln!("skipping: no AVX-512");
            return;
        }
        let mut qualia = Vec::new();
        let mut mantissas = Vec::new();
        for coh_raw in 0u8..16 {
            for m_raw in 0u8..16 {
                let coh = ((coh_raw << 4) as i8) >> 4;
                let m = ((m_raw << 4) as i8) >> 4;
                let mut q = QualiaI4_16D::ZERO;
                q.set(DIM_COHERENCE, coh);
                qualia.push(q);
                mantissas.push(m);
            }
        }
        assert_dk_position_batch_simd_matches_scalar(&qualia, &mantissas);
    }

    // ... 9 more analogous tests
}
```

### Length-edge tests (5 additional, not counted in the 10)

| E1 | Empty slice — no SIMD lane consumed, scalar tail handles 0 elems |
| E2 | Exactly MIN_BATCH (8 for AVX, 2 for NEON) — one full SIMD chunk, zero tail |
| E3 | MIN_BATCH + 1 — one chunk + 1-elem scalar tail |
| E4 | 2 × MIN_BATCH − 1 — one chunk + MIN_BATCH−1 tail |
| E5 | 10K — large batch, validates accumulation correctness over many chunks |

Each edge test runs over all 5 batch functions (any one arch — equivalence
to scalar is sufficient).

### Cross-arch parity test (1, manual on physical hardware)

In CI we cannot test both AVX-512 and NEON simultaneously (no
cross-compilation runtime). The cross-arch test is **manual**: a sprint-13
worker runs the test suite on (a) an AVX-512 Linux box and (b) an Apple
Silicon Mac and confirms byte-identical output across the two
architectures. Logged in `.claude/board/AGENT_LOG.md` as W-SIMD-VERIFY-1.

---

## §7 Benchmark plan

### Criterion suite

`crates/lance-graph-contract/benches/i4_batch.rs` (NEW), with the
following 5 benches (one per function, 4 batch sizes each):

| Bench | Function | Sizes |
|---|---|---|
| bench_dk_position | dk_position_batch | 8, 64, 1024, 16384 |
| bench_trust_texture | trust_texture_batch | 8, 64, 1024, 16384 |
| bench_flow_state | flow_state_batch | 8, 64, 1024, 16384 |
| bench_gate_decision_disc | gate_decision_disc_batch | 8, 64, 1024, 16384 |
| bench_mul_assess | mul_assess_batch | 8, 64, 1024, 16384 |

Each bench measures scalar (Wave G impl) vs SIMD (sprint-13 impl) by
toggling a build feature or by directly calling the private modules
through a `#[cfg(bench)]` re-export.

### Target speedups (AVX-512, batch size 1024)

| Function | Scalar baseline | SIMD target | Speedup |
|---|---|---|---|
| dk_position_batch | ~5 ns/elem | ≤ 1.2 ns/elem | ≥ 4× |
| trust_texture_batch | ~8 ns/elem | ≤ 2 ns/elem | ≥ 4× |
| flow_state_batch | ~12 ns/elem | ≤ 3 ns/elem | ≥ 4× |
| gate_decision_disc_batch | ~15 ns/elem | ≤ 3.5 ns/elem | ≥ 4× |
| mul_assess_batch | ~80 ns/elem | ≤ 30 ns/elem | ≥ 2.5× (limited by f64 finalize) |

NEON targets are typically ~1.5× over scalar (8-wide SIMD on 2-wide
register; throughput-bound), not 4×. NEON bench reports speedup but
does not gate the PR — NEON correctness gates, NEON perf does not.

### Significance per I-NOISE-FLOOR-JIRAK

Criterion's default outlier detection assumes IID samples; benchmark
runs in this workspace are weakly-dependent (cache effects, branch
predictor warmup). For any claim of the form "AVX-512 is N× faster than
scalar with p < 0.05," cite Jirak 2016 rate `n^(p/2−1)` rather than
classical Berry-Esseen. The PR landing notes should state speedups as
point estimates with criterion confidence intervals; statistical-
significance claims beyond that are out of scope (the speedups are large
enough that significance is not in question).

### Bench regression gate

Sprint-14 may add a "perf regression CI" job that re-runs these benches
on each PR. If the scalar baseline regresses by > 10% or the SIMD speedup
drops below 3× on AVX-512, the job flags the PR for review. Tracked as
TD-SIMD-PERF-REGRESSION-CI-1 (out of scope for sprint-13).

---

## §8 Risk matrix

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | ndarray dev-dep for `simd_caps()` violates zero-dep posture | Medium | Low | Inline 30-line copy if rejected; PR W-SIMD-1 first work item |
| R-2 | AVX-512 microarchitecture differences (Zen 4 vs Sapphire Rapids vs Tiger Lake vs Ice Lake) — frequency throttling, port pressure | Medium | Medium | (a) target `avx512f,avx512bw` only (no VBMI / VNNI) — these are universally supported on AVX-512-capable CPUs; (b) bench on at least 2 microarchitectures before PR merge; (c) document the bench rig in PR description |
| R-3 | NEON pipeline shape (Apple M1/M2/M3 vs Graviton vs Pi 5) — NEON throughput varies 2×–4× across these | High | Low | Correctness only; perf is a "nice to have" on NEON. PR target: byte-identical to scalar on all three |
| R-4 | Compiler vectorization regression — LLVM may auto-vectorize the scalar fallback differently across rustc versions, changing the byte-output cliff between scalar and SIMD | Low | Medium | The scalar impl is the correctness anchor and is hand-written branch-tree, not auto-vectorized. Lock rustc MSRV in `lance-graph-contract/Cargo.toml`; assert byte equivalence in CI |
| R-5 | Enum `#[repr(u8)]` change breaks downstream serialization | Low | High | Audit consumers before W-SIMD-1: grep for `bincode::serialize(&dk_position)` and `serde_json::to_value(&trust_texture)` across the workspace + sibling repos (crewai-rust, n8n-rs). If discriminants are serialized, lock them via explicit `= N` annotations matching the current default order |
| R-6 | `_mm512_mask_compressstoreu_epi8` requires AVX-512BW + AVX-512VBMI2 on some toolchains; minimum-feature-set claim of "avx512f + avx512bw only" is wrong | Medium | Medium | Verified: compressstoreu_epi8 needs AVX-512VBMI2 (Ice Lake+). Either add VBMI2 to the feature set (loses Skylake-X / Cascade Lake support) OR use scalar byte-extract loop (lose ~5% of the 4× speedup). **Decision:** use scalar byte-extract for portability; sprint-14+ can add a VBMI2-fast-path if profiling demands |
| R-7 | `GateDecision::Block { reason: String }` allocation in stage 2 dominates SIMD savings for the full-fidelity (not disc-only) variant | High | Medium | Expose `gate_decision_disc_batch` (already in §4); document in PR that `gate_decision_batch` is "SIMD-first-stage + scalar-string-stage" and not 4× faster end-to-end |
| R-8 | SIMD-vs-scalar tests don't run on the CI runner (Linux x86-64 without AVX-512; only AVX-2) | High | Low | Tests skip with `eprintln!` if `simd_caps()` doesn't report the required features — same pattern as ndarray's existing SIMD tests. Cross-arch parity test §6 runs manually before merge |
| R-9 | Sprint-13 scope creep — temptation to also wire AVX-2 path while we're at it | Medium | Low | Iron-discipline: AVX-2 is out of scope per §1. The current scalar fallback runs on AVX-2 hardware correctly (just not as fast as it could). Tracked as TD-SIMD-I4-AVX2-1 |
| R-10 | Iron rule violation — accidentally producing different bytes from SIMD vs scalar on a code path not covered by the 10 equivalence tests | Medium | Critical | (a) exhaustive enumeration for dk_position (256 inputs) and trust_texture (4096 inputs) — both fit; (b) sampled tests with seeded RNG for flow_state / gate / mul_assess; (c) cite I-LEGACY-API-FEATURE-GATED in the PR description so reviewers know the iron rule is in force |

---

## §9 LOC estimate

| Component | LOC | Notes |
|---|---|---|
| `mul.rs` §`batch::scalar_impl` | 0 (refactor) | Move existing W-G3 code into `mod scalar_impl` — pure rename |
| `mul.rs` §`batch::avx512_impl` | ~250 | 5 functions × ~40 LOC + ~50 LOC shared helpers (extract_i4_from_qualia_vec, broadcast_i8_threshold, mask_compressstore_epi8_scalar_tail) |
| `mul.rs` §`batch::neon_impl` | ~200 | 5 functions × ~35 LOC + ~25 LOC shared helpers |
| `mul.rs` §`batch` public dispatch wrappers | ~60 | 5 functions × ~12 LOC of `if caps.avx512 { ... } else if caps.neon { ... } else { scalar }` |
| Enum `#[repr(u8)]` annotations | ~10 | 3 enums + GateDecision from_disc/to_disc helpers |
| `simd_caps()` shim (if no ndarray dep) | ~30 | Inlined 3-bool struct + LazyLock + is_x86_feature_detected |
| **Source subtotal** | **~550** | within "~300 LOC × 2 archs" envelope (550 ≈ 250 + 200 + 100) |
| Tests — 10 SIMD-vs-scalar equivalence | ~180 | 10 × ~18 LOC including the shared assertion helper |
| Tests — 5 length-edge tests | ~50 | 5 × ~10 LOC |
| Tests — 1 cross-arch parity guard (CI marker) | ~20 | conditional `#[cfg]` test stub + log assertion |
| **Test subtotal** | **~250** | matches "~250 LOC tests" envelope exactly |
| Bench — `benches/i4_batch.rs` | ~100 | 5 criterion bench fns × ~20 LOC including input generation |
| **Bench subtotal** | **~100** | matches "~100 LOC bench" envelope exactly |
| **TOTAL** | **~900** | exceeds spec target of 600 if both archs land in one PR |

### PR slicing

If 900 LOC is too large for one PR (workspace policy is < 700 LOC for
non-spec PRs), slice along:

- **W-SIMD-1** (~250 LOC): scalar refactor + dispatch scaffolding + AVX-512
  `dk_position_batch` + 1 equivalence test + 1 bench.
- **W-SIMD-2** (~150 LOC): AVX-512 `trust_texture_batch` + AVX-512
  `flow_state_batch` + 2 tests + 2 benches.
- **W-SIMD-3** (~150 LOC): AVX-512 `gate_decision_disc_batch` + AVX-512
  `mul_assess_batch` + 2 tests + 2 benches.
- **W-SIMD-4** (~300 LOC): all 5 NEON impls + 5 NEON equivalence tests +
  cross-arch parity test marker.
- **W-SIMD-5** (~50 LOC): bench polish + sprint-13 closeout + STATUS_BOARD +
  PR_ARC update.

Total: 5 PRs × ~180 LOC average, all comfortably under the 700 LOC ceiling.

---

## §10 Acceptance gates (sprint-13 closeout)

D-CSV-13b is "Shipped" when ALL of the following are true:

1. The 5 batch fns in `lance_graph_contract::mul::i4_eval::batch` exist
   with `#[cfg(target_arch = ...)] mod {avx512,neon}_impl` modules behind
   them, runtime-dispatched via `simd_caps()`.
2. The 10 SIMD-vs-scalar equivalence tests pass on a CI runner with
   AVX-512 (or the test asserts byte-identity by reading scalar output
   alongside SIMD output and the test is no-op on non-AVX-512 hardware
   with a logged-skip marker).
3. The 5 length-edge tests pass on the default CI runner.
4. The cross-arch parity test (§6 W-SIMD-VERIFY-1) is logged in
   `.claude/board/AGENT_LOG.md` with the AVX-512 box's `lscpu` output
   and the Apple Silicon Mac's `sysctl -a | grep machdep.cpu.brand`
   output, plus the byte-identity assertion passing on both.
4. Criterion bench `crates/lance-graph-contract/benches/i4_batch.rs`
   reports ≥ 4× speedup on AVX-512 for `dk_position_batch`,
   `trust_texture_batch`, `flow_state_batch`, `gate_decision_disc_batch`
   at batch size 1024 vs the Wave G scalar baseline.
5. `mul_assess_batch` reports ≥ 2.5× speedup (lower target because the
   f64 finalize stage is scalar).
6. PR description cites I-LEGACY-API-FEATURE-GATED, I-NOISE-FLOOR-JIRAK,
   §3 priority-order analysis for flow_state, and §6 R-5 audit of
   downstream enum serialization (the audit MUST conclude that
   `#[repr(u8)]` discriminant locking is safe — i.e. nothing depends on
   the unannotated default order).
7. `.claude/board/LATEST_STATE.md` Current Contract Inventory updated
   with "i4_eval::batch SIMD intrinsics shipped (avx512f + avx512bw +
   neon paths runtime-dispatched via simd_caps; scalar fallback for all
   other targets)."
8. `.claude/board/PR_ARC_INVENTORY.md` entry PREPENDED with Added /
   Locked / Deferred / Docs / Confidence.
9. `STATUS_BOARD.md` D-CSV-13b row flipped to "Shipped."
10. No new TECH_DEBT entries on the critical path; the documented
    deferrals (AVX-2 fast path, WASM SIMD128, VBMI2 compressstore,
    bench regression CI) are tracked as TD-SIMD-I4-* entries with
    explicit "deferred to sprint-14+" annotations.

---

## §11 Open questions for user ratification

| OQ | Question | Default if no answer |
|---|---|---|
| OQ-SIMD-1 | runtime dispatch via ndarray dev-dep, OR inline 30-line simd_caps shim into lance-graph-contract? | inline shim (preserves zero-dep posture) |
| OQ-SIMD-2 | enum `#[repr(u8)]` discriminants — accept the natural declaration order (§5), or require explicit `= N` annotations to lock against future reordering? | explicit annotations (defensive against future PR re-ordering) |
| OQ-SIMD-3 | `gate_decision_batch` SIMD-first-stage + scalar-string-stage — acceptable, or should we make `GateDecision` itself a 2-variant `(Disc, Option<String>)` to avoid the second stage entirely? | acceptable as-is; deeper refactor out of scope |
| OQ-SIMD-4 | NEON correctness gates PR but NEON perf does not — confirm? | confirmed |
| OQ-SIMD-5 | bench rig — sprint-13 worker should provision a Sapphire Rapids + Zen 4 + M2 + Pi 5 quad? Or just Sapphire Rapids + M2? | Sapphire Rapids + M2 only; Zen 4 + Pi 5 are sprint-14 |

---

## §12 Cross-references and prior art

- **Sprint-11 Wave E D-CSV-8** (commit `da8e8f7`) — "MUL i4 SIMD evaluation
  (scalar i4 path, sprint-12 SIMD vec deferred)." The scalar i4
  per-element fns from Wave E are the inner-loop body of the Wave G batch
  fns; sprint-13 lands the SIMD that Wave E deferred.
- **Sprint-12 Wave G W-G3** (commit `7d7b537` → `03ce219` → #388 merged
  2026-05-16) — the scalar batch API shape this PR vectorizes.
- **ndarray `simd_avx512.rs`** — proven AVX-512 patterns:
  `_mm512_cmpgt_epi8_mask` (L1612), `_mm512_mask_blend_epi8` (L641).
  Sprint-13 reuses these intrinsics inside lance-graph-contract.
- **CLAUDE.md `I-LEGACY-API-FEATURE-GATED`** (added 2026-05-16 via W-G5,
  commit `03ce219`) — the iron rule that makes §5 mandatory.
- **CLAUDE.md `I-NOISE-FLOOR-JIRAK`** — cited in §7 for benchmark
  significance claims.
- **`.claude/knowledge/i4-substrate-decisions.md`** (sprint-12 W-F11) — the
  5-instance catalogue of legacy-API drift that motivates the iron rule.
- **family-codec-smith agent card** — reviewer for §3 (codec-shaped
  SIMD pipelines).
- **truth-architect agent card** — reviewer for §5 (semantic equivalence
  enforcement).

---

## §13 Status footer

- **Drafted:** 2026-05-16 by opus planner (sprint-13 preflight PP-6)
- **Branch:** `claude/sprint-13-preflight-planning`
- **Next action:** user ratifies §11 OQs; sprint-13 W-SIMD-1 begins
  with the scalar refactor + dispatch scaffolding (~250 LOC PR).
- **Predecessor merged:** #388 (Wave G, scalar batch API), 2026-05-16
- **Successor:** none (D-CSV-13b is terminal for the SIMD i4 line;
  AVX-2 + WASM tracked as TD entries deferred to sprint-14+)
- **Confidence:** Medium-High — scalar API surface is locked; SIMD
  shape is the standard "load + extract + compare-mask + blend + pack"
  pipeline with proven intrinsics in adjacent ndarray code; the main
  risks (R-6 VBMI2 portability, R-5 enum-serde audit, R-2 microarch
  variance) are tractable and documented.
