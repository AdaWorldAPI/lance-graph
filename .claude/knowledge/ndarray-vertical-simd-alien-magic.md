# KNOWLEDGE: ndarray Vertical SIMD — the "Alien Magic" Surface

## READ BY:
- Any worker spawning to write SIMD code in a consumer crate (lance-graph, lance-graph-contract, lance-graph-planner, bgz17, holograph, thinking-engine, sigker, etc.)
- Any worker spawning to add a primitive to `adaworldapi/ndarray` `src/simd_*.rs`
- The `simd-savant` agent (`.claude/agents/simd-savant.md`) — this doc is the WHAT; the savant card is the GATE
- PP-14 convergence-architect — when proposing cross-slice synergies in SIMD workloads
- Any session reading `EPIPHANIES.md E-SIMD-SWEEP-1`

## P0 TRIGGERS:
- About to write `_mm*` / `vld1q_*` / `core::arch::*` in any crate → STOP, read this first
- About to file a PR against `adaworldapi/ndarray` adding `pub fn` to `src/simd*.rs` → STOP, read this first
- About to design a new vertical SIMD workload (palette gather, splat blend, signature kernel, etc.) → STOP, the surface design is here

---

## The Click

**ndarray's SIMD surface is designed to fit exactly what our stack vertically needs — not as a generic library that consumers wrap, but as struct methods on typed wrappers (`I8x16`, `U8x32`, `F32x16`, `U64x8`, …) plus closure-parameterized batch primitives that absorb the consumer's domain semantics.**

The pattern has two halves:

1. **Struct methods on typed wrappers.** `I8x16::from_i4_packed_u64(packed) -> Self`, `I8x16::saturating_abs(self) -> Self`, `U8x32::lut_lookup_64(table) -> Self`, `U64x8::xor_popcount(self, other) -> u32`. Each method maps to a single SIMD instruction on the target arch and falls back to a fused scalar loop on unsupported targets. Bounds-aware loads come for free — the wrapper knows its lane count.

2. **Closure-parameterized batch primitives.** `ndarray::simd::batch_packed_i4_16<E, F>(packed, aux, out, f: F)` where `F: Fn(I8x16, i8) -> E`. The polyfill owns the runtime feature dispatch, lane chunking, tail handling, scalar fallback. The consumer provides the domain semantics as a closure — its enum (`DkPosition`, `TrustTexture`, …), its classification logic, its thresholds.

The consumer crate sees **zero raw intrinsics, zero `cfg(target_arch)` blocks, zero runtime feature detection.** It uses `ndarray::simd::*` types and the closure-batch entry points. The polyfill is the channel; everything below stays inside `ndarray/src/simd_*.rs`.

This is the "alien magic" — ndarray is designed AS-IF the architects had clairvoyant knowledge of our exact workloads: palette streams L1-L4, spatial splat fields, blasgraph-over-palette, i4-packed qualia, hamming over u64x8, path-signature kernels. The struct methods + batch primitives are the cellular chemistry; the consumer crates assemble organs from them.

---

## Why this shape (not other shapes)

### NOT: free functions in `ndarray::simd::*`

`ndarray::simd::unpack_i4_signed_u64(packed: u64) -> I8x16` works but lives nowhere natural — it's a global function with no host. When the consumer wants a different lane width (i4 × 32, i4 × 64), or a different sign convention (unsigned nibble), or a different upcast (i4 → i16), the surface fragments into a free-function zoo. Struct methods on `I8x16`, `I8x32`, `I8x64` give a uniform call shape and let the compiler dispatch by type.

### NOT: traits the consumer impls on its own structs

`pub trait I4Packed16 { fn raw(&self) -> u64; fn extract_i8x16(&self) -> I8x16 { … } }` and then `impl I4Packed16 for QualiaI4_16D { fn raw(&self) -> u64 { self.0 } }` looks clean, but it scatters the SIMD-correctness burden across consumers — every domain struct decides for itself what `extract_i8x16` means. The polyfill loses its single-channel property.

### YES: struct methods on the polyfill's typed wrappers + closure-batch primitives

The wrapper owns the SIMD; the consumer owns the domain. The closure is the boundary. `bytemuck::cast_slice(&qualia)` (or `repr(transparent)` zero-cost view) gets the consumer's `&[QualiaI4_16D]` to a `&[u64]` view; ndarray takes it from there.

---

## Per-workload surface table

Each row maps one stack workload to (a) the ndarray struct methods it needs and (b) the consumer crate that calls them. Rows marked **W1a** are wave-1 ndarray PRs (this session's plan); rows marked **W1.5** are deferred behind their certification gate.

| Workload | ndarray struct surface | Consumer | Wave |
|---|---|---|---|
| i4 × 16 packed qualia | `I8x16::from_i4_packed_u64(packed: u64) -> Self`, `I8x16::lane_i8::<const N: usize>(self) -> i8`, `I8x16::saturating_abs(self) -> Self`, `batch_packed_i4_16<E, F>(packed, aux, out, f)` | `lance-graph-contract::mul::i4_eval::batch` (5 batch fns) | **W1a** |
| Hamming over u64 lanes | `U64x8::xor_popcount(self, other) -> u64`, batched `hamming_distance_raw` (exists) | `holograph/hamming.rs`, `lance-graph/blasgraph/types.rs`, `blasgraph/ndarray_bridge.rs` | **W1a** |
| Palette gather (u16 indices → u8 values) | `U16x8::gather_u16(indices, table)`, `palette_lookup_u8x8(idx_v, lut)` | `bgz17/src/simd.rs` | **W1a** |
| Cross-arch prefetch hint | `prefetch_read_t0(ptr: *const u8)`, `prefetch_read_t1`, `prefetch_read_t2` (no-op on unsupported) | `bgz17/src/prefetch.rs` | **W1a** |
| VPSHUFB / TBL nibble-popcount LUT | `U8x64::nibble_popcount_lut(self) -> Self` (already exists), `U8x32::nibble_popcount_lut(self) -> Self` (parity) | `blasgraph/ndarray_bridge.rs` (currently hand-rolled twice — AP-SIMD-7) | **W1b** (consume existing) |
| Palette L1-L4 stream sweep | `U8x32::lut_lookup_64(table) -> Self`, `U8x32::splat_byte(b) -> Self`, `U8x32::compress(mask)` | `bgz17`, future palette codec consumers | **W1a** + **W1b** |
| Spatial splat field merge (mean/var/energy/gen AoS) | `F32x16::splat_blend(means, vars, energy, gen)`, `F32x16::masked_add(mask, addend)`, `F32x16::reduce_top_k::<const K: usize>()` | `thinking-engine::splat_ops`, `ndarray::hpc::stream::splat_field` | **W1b** (after `W1a` saturating-abs + lane extract) |
| BLAS over palette CSR/CSC (palette-typed edge SpMV) | `U8x32::palette_csr_spmv(mat_row, vec, table)`, `U8x32::gather_scatter_palette(idx, vals)` | `lance-graph/blasgraph` palette semiring | **W1b** |
| Path-signature truncated tensor algebra | `F32x16::shuffle_product_lift(a, b, depth)`, banded 2D PDE kernel sweep `signature_pde_sweep(x, y, kernel_fn)` | `sigker::signature`, `sigker::kernel` | **W1.5** (gated on `jc Pillar 11`) |
| Randomized signature projection | `F32x16::random_proj_step(state, seed, depth)` | `sigker::randomized` | **W1.5** |
| Log-signature Lyndon basis compression | `I16x16::lyndon_pack(basis_idx)`, batched `lyndon_unpack_batch` | `sigker::log_signature` | **W1.5** |
| BF16 tables (existing) | `BF16x16` typed wrapper + ops (already in ndarray) | `thinking-engine::bf16_engine`, `engine.rs:504` (VNNI dispatch) | **W1b** (route existing VNNI through `ndarray::simd_amx`) |
| AMX matmul (existing) | `ndarray::simd_amx::*` (already in ndarray) | `engine.rs:504`, future tile workloads | **W1b** |

---

## Wave plan (sprint-13 / sprint-14)

### W1a — ndarray primitive additions (parallel small PRs against `adaworldapi/ndarray`)

Five small primitives, each on its own branch, each auditable by `simd-savant` before merge:

1. **`TD-NDARRAY-SIMD-UNPACK-I4-16D`** — `I8x16::from_i4_packed_u64` + `I8x16::lane_i8::<N>` + `batch_packed_i4_16<E, F>` closure-batch entry. AVX-512 path via `_mm512_cvtepi8_epi16` + nibble shuffle; NEON via `vshl_n_s8` / `vqshl_n_s8`; scalar via fused-loop fallback. Bounds-aware tail.
2. **`TD-NDARRAY-SIMD-SATURATING-ABS-I8`** — `I8x16::saturating_abs(self) -> Self`. **AVX-512 needs `_mm512_min_epu8(_mm512_abs_epi8(x), _mm512_set1_epi8(0x7f))`** — VPABSB alone does NOT saturate (`abs(0x80) = 0x80`, still `i8::MIN`); the VPMINUB clamp is required to remap `0x80 → 0x7f`. NEON `vqabsq_s8` (the `q` suffix means saturating); scalar `i8::saturating_abs`. Closes codex P2 i8::MIN divergence on PR #398 by giving consumers a single source-of-truth for "true saturating abs" across all three backends.
3. **`TD-NDARRAY-SIMD-GATHER`** — `U16x8::gather_u16(indices, table)`. AVX2 `_mm256_i32gather_epi32` + downcast; NEON loop (no native gather); scalar `indices.iter().map(|&i| table[i])`. Closes `bgz17/src/simd.rs:88` raw `_mm256_i32gather_epi32`.
4. **`TD-NDARRAY-SIMD-PREFETCH`** — `prefetch_read_t0(ptr: *const u8)`, `prefetch_read_t1`, `prefetch_read_t2`. AVX `_mm_prefetch`; NEON `__builtin_prefetch`-equivalent; no-op on unsupported. Closes `bgz17/src/prefetch.rs:96` / `:100`.
5. **`TD-NDARRAY-SIMD-POPCOUNT-U64`** — `U64x8::popcnt(self) -> Self` (lane-wise 64-bit popcount). AVX-512 VPOPCNTDQ `_mm512_popcnt_epi64`; NEON `vcntq_u8` + horizontal-sum; scalar `u64::count_ones`. Closes holograph + blasgraph hamming raw-intrinsic blocks.

### W1b — consumer migrations (gated on W1a merge)

Each consumer crate migrates separately, one PR per crate. The `simd-savant` runs PRE-MERGE on each.

- `lance-graph-contract/src/mul.rs` — consume #1 + #2; fix codex P1 (NEON OOB at `len==2`, eliminated by polyfill bounds-aware loads); fix codex P2 (i8::MIN, direction B via `I8x16::saturating_abs`). Hoist the `batch_classify_qualia<E, F>` generic to reduce the 5 batch fns to closures.
- `bgz17/src/simd.rs` — consume #3 + retire bespoke `SimdLevel` enum + `detect_simd()` (AP-SIMD-8).
- `bgz17/src/prefetch.rs` — consume #4.
- `blasgraph/types.rs` + `blasgraph/ndarray_bridge.rs` — consume #5 + existing `U8x64::nibble_popcount_lut` + `hamming_distance_raw`; drop the duplicate hand-rolled LUTs (AP-SIMD-7).
- `holograph/hamming.rs` — consume #5 + `hamming_distance_raw`.
- `thinking-engine/src/engine.rs:504` — route VNNI dispatch through `ndarray::simd_amx` (already imported at line 160).

### W1.5 — sigker primitives (gated on `jc Pillar 11` activation)

Three deferred ndarray PRs for path-signature workloads. **Do not spawn until sigker is benchmarked at production carrier widths and `jc Pillar 11 (Hambly-Lyons signature uniqueness)` activates.** When that gate trips:

6. **`TD-NDARRAY-SIMD-SIGNATURE-PDE-SWEEP`** — banded 2D PDE solver kernel for signature inner-product `〈S(X), S(Y)〉` via Goursat (O(T₁·T₂) flops, no signature materialization).
7. **`TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`** — fixed-width random-matrix-vector for Cuchiero-Schmocker-Teichmann randomized signatures.
8. **`TD-NDARRAY-SIMD-LYNDON-PACK`** — Lyndon-basis packing/unpacking for log-signatures (7-13× compression, lossless).

---

## sigker positioning — the Index-regime third lane

`crates/sigker` is the workspace's path-signature codec. Read its `lib.rs` once; key properties:

- **Pure scalar Rust, zero raw intrinsics, zero `ndarray` dep today.** Cleanest exemplar of the "domain crate composes via closures" pattern.
- **Algebraic peer to bgz17 (palette-distance) and deepnsm (NSM tiling).** Third encoding lane in the codec routing table.
- **Index regime**, not Argmax — by Hambly-Lyons 2010 uniqueness, path signatures are EXACT on tree-quotient classes of paths. **Bypasses the Jirak 2016 noise floor** (`I-NOISE-FLOOR-JIRAK` iron rule in CLAUDE.md) that hits VSA bundling.
- **Certification gate:** `jc Pillar 11` (Hambly-Lyons signature uniqueness on lance-graph paths) — currently DEFERRED. Activates once sigker is benchmarked at production carrier widths.

When `jc Pillar 11` activates, sigker becomes a first-class consumer of ndarray vertical SIMD. The W1.5 wave catalogue above is its primitive shopping list.

**Architectural implication:** the W1a primitives must be designed broad enough that the W1.5 primitives compose naturally with them. Specifically, `F32x16` + `BF16x16` operations (already in ndarray) plus the closure-batch primitive shape (introduced in W1a-#1) are the foundation that signature kernels build on. The randomized projection in W1.5-#7 is a `batch_packed<E, F>` over an `F32x16` state with a Gaussian-random closure — same shape as W1a-#1, different lane type.

---

## Cross-references

- `.claude/agents/simd-savant.md` — the 5th-slot agent card enforcing this surface as the canonical SIMD channel. AP-SIMD-1..8 anti-patterns + grep audits.
- `.claude/knowledge/autoattended-multiagent-pattern.md` § 14 — the lance-graph adapter section that introduced the simd-savant slot.
- `.claude/board/EPIPHANIES.md` § E-SIMD-SWEEP-1 (2026-05-16) — the 158-violation finding that triggered the wave.
- `.claude/board/TECH_DEBT.md` — the 5 W1a + 3 W1.5 `TD-NDARRAY-SIMD-*` entries.
- `CLAUDE.md` § Substrate-level iron rules — `I-SUBSTRATE-MARKOV`, `I-NOISE-FLOOR-JIRAK`, `I-VSA-IDENTITIES`. The W1.5 sigker positioning is the architectural counterpart to `I-NOISE-FLOOR-JIRAK` (sigker bypasses what Jirak quantifies).
- `crates/sigker/src/lib.rs` — the path-signature codec; the W1.5 consumer.
- `/home/user/ndarray/src/simd.rs` — the polyfill's public re-export surface. Read before designing any W1a primitive.
- `/home/user/ndarray/src/simd_avx512.rs`, `simd_neon.rs`, `simd_int_ops.rs` — the per-arch implementations.

---

## Litmus test for any proposed change to this surface

> **Does the new primitive go on a typed-wrapper struct, or as a free function?**
> Free function = reject; the surface fragments. Struct method = accept.

> **Does the consumer-side call route through `ndarray::simd::*` types, or does it import `core::arch::*`?**
> `core::arch::*` in a consumer crate = AP-SIMD-1/2 violation; route through the polyfill.

> **Does the batch primitive take a closure for domain semantics, or hard-code the enum?**
> Hard-coded enum = the polyfill doesn't know `DkPosition` exists; closure = the boundary is clean.

> **Is the workload palette-stream, spatial-splat, blasgraph-over-palette, hamming, signature, or random-projection?**
> If none of these, propose the new workload first against this doc + the simd-savant card. New workloads extend the table in §"Per-workload surface table" via a doc PR, not by inlining intrinsics in a consumer crate.
