# key16-contiguous-match-v1 — a contiguous fast path for the 16-byte key match

**Status:** PROPOSAL (D-KEY16-0..5). No kernel change is authorized until the
D-KEY16-2 probe clears its pre-registered threshold.
**Date:** 2026-10-11. **Repos:** ndarray (kernel, parity, probe), lance-graph
(consumers: mask-risc, quack, dir-sim). lance-graph-java is unaffected (see §6).

## 1. Overview

`ndarray::simd::ternary_match_strided16_to_mask` answers "does this 16-byte
key match `pattern` on the `care` bits?" for every row of a lane, read in
place. mask-risc lowers `Pred::MatchFacet16Strided` to it; quack exposes it as
`Cmp::MatchFacet16Strided`; dir-sim's `subtree` query runs it over the
population's packed `[[u8; 16]]` Dn128 lane (stride 16) to select an OU
subtree.

The compare is already efficient: one `XOR_AND` ternlog per 8-byte half on
`U64x8`, the halves ORed and tested for zero. The cost is the gather in front
of it. For every group of 16 rows the kernel does 32 scalar 8-byte loads into
two stack arrays, then `U64x8::from_array` (`simd_masking_ops.rs`, the loop
after `let groups = count / 16`). It takes that path even when
`stride_bytes == 16`, where the keys are packed back to back and one 512-bit
load already holds four whole keys.

This plan adds a contiguous fast path for `stride_bytes == 16`, measures a
gather variant for wider strides, and closes a parity gap found while
scoping it.

## 2. Decisions

- **The 1.99 `u128`-in-vector-register `asm!` path is NOT used.** Passing one
  key per `xmm` register is one key per instruction, against eight 64-bit
  halves per `U64x8`. ndarray's 69 `asm!` sites were checked on 2026-10-11:
  none passes a 128-bit value and none works around the old limit. REVISIT
  WHEN: a 16-byte-key kernel needs an instruction `std::arch` does not expose.
- **The fast path is written against the facade, not a backend.** It uses
  a byte-level unaligned load (below), `U64x8::ternlog::<{ ternlog::XOR_AND }>`
  and `U64x8::cmpeq_mask` (returns `u8`), so every backend realizes it the way
  it realizes the existing kernel. No new intrinsic outside `simd_{arch}.rs`.
- **The load takes bytes, never a reinterpreted `&[u64]`.** The kernel's input
  is `&[u8]` and the first offset may be 1..15, so the address is not
  `u64`-aligned; building a `&[u64]` from it for `U64x8::from_slice` is
  undefined behaviour regardless of which instruction the backend emits. The
  facade has no `U8x64 → U64x8` bitcast and no byte load for `U64x8` today
  (checked 2026-10-11), so W3 adds one, `U64x8::from_le_bytes(&[u8; 64])`,
  on every backend (an unaligned vector load where one exists, eight
  `u64::from_le_bytes` otherwise), covered by its own parity case at offsets
  0..63. The current kernel already reads each half with `u64::from_le_bytes`,
  so the scalar fallback is the same arithmetic. (Codex review on #1475.)
- **The branch is on the runtime argument** (`stride_bytes == 16`), never on
  `cfg(target_feature)`. The function signature does not change, so no
  consumer changes and no lgj ABI minor bump.
- **Semantics are frozen first.** Parity coverage for the existing kernel
  lands and is green before any kernel change (TDD; §4 W1).

## 3. The fast path (design to be measured, not adopted)

With 16-byte keys packed back to back, one `U64x8` load holds
`[k0.lo, k0.hi, k1.lo, k1.hi, k2.lo, k2.hi, k3.lo, k3.hi]`. Splat the pattern
and care as alternating vectors `[plo, phi, plo, phi, …]` and
`[clo, chi, clo, chi, …]`. Then:

```text
v = ternlog::<XOR_AND>(x, p_alt, c_alt)      // (x ^ p) & c per 64-bit half
m = v.cmpeq_mask(zero)                       // 8 bits, one per half
keys = m & (m >> 1) & 0b0101_0101            // a key matches iff both its halves do
```

The four key bits sit at even positions; they are compacted to 4 consecutive
output bits (portable bit fold, or `pext` behind the facade if a backend
offers it). Every row is still tested; nothing is skipped. Tail rows
(`count % 4`, and the partial final word) use the existing scalar tail.

Alignment: the lane base is not guaranteed aligned to anything. The load is
`U64x8::from_le_bytes`, which takes bytes and is defined at every address;
the probe's arm B uses the same byte load so W2 measures what W3 ships.

## 4. Waves

| Wave | D-id | Repo | Work | Gate (all must pass) |
|---|---|---|---|---|
| W0 | D-KEY16-0 | lance-graph | This plan + board rows | — |
| W1 | D-KEY16-1 | ndarray | Add `ternary_match_strided16_to_mask` to `crates/simd-masking-parity` against a bytewise reference: strides 16 / 20 / 32 / 512, first offsets 0..15, care bits in bytes 12..16 (the in-crate test `ternary_match_strided16_sees_bytes_past_twelve` shape), tails 0 / 1 / 3 / 4 / 5 / 15 / 16 / 17 / 63 / 64 / 65 / 130 | green on the CURRENT kernel on every `simd-matrix.yaml` realization row (avx512, avx2, avx, v2, neon, wasm + scalar, nightly); a deliberately wrong reference makes it red (can-fire) |
| W2 | D-KEY16-2 | ndarray | `examples/key16_match_probe.rs`: arm A current kernel; arm B contiguous fast path (prototype inside the probe); arm C AVX-512 64-bit gather for strides > 16 (prototype, `cfg`-gated in the probe only). N = 65,536 (dir-sim population) and 1,048,576; strides 16 / 32 / 512; selectivity ~1 % and ~50 % | answers bit-identical across arms BEFORE timing; release, quiet machine, median of 101 runs, avx512 and avx2 hosts reported separately; results committed to the probe's doc |
| W3 | D-KEY16-3 | ndarray | Only if W2 shows arm B ≥ 1.5× arm A at stride 16 on BOTH avx512 and avx2, with no arm slower at strides 32 / 512: add `U64x8::from_le_bytes` to the facade and every backend, then the `stride_bytes == 16` path to the kernel | W1 parity green on every matrix row, including first offsets 1..15; a `from_le_bytes` parity case at offsets 0..63 against `u64::from_le_bytes`; `scripts/codegen-witness.sh` unchanged; disable run 1: route stride 16 back to the gather path, probe speedup disappears; disable run 2: fold on the low half only (`m & 0x55`), parity goes red |
| W4 | D-KEY16-4 | ndarray | Only if W2 shows arm C ≥ 1.2× arm A at strides 32 and 512 on the avx512 CI tier: gather path in `simd_avx512.rs` behind the facade; other backends keep the scalar gather | as W3, plus the avx2 / neon / wasm rows still take the old path (asserted, not assumed) |
| W5 | D-KEY16-5 | lance-graph | Consumer verification after the ndarray merge: `cargo test` for mask-risc, quack, dir-sim; the Identity workflow's 87 pinned dir-sim names; dir-sim `subtree` timing before/after at 65,536 nodes, recorded | no consumer source change; all green; timing recorded with method |

**Kill conditions.** W2 below threshold for arm B → close D-KEY16-3 as
measured won't-fix and keep W1 (parity coverage stands on its own). Arm C
below threshold → close D-KEY16-4 the same way. A W1 failure on the current
kernel is a real backend bug: fix it first, as its own PR.

## 5. Order and ownership

W1 → W2 → (W3, W4 independently) → W5. W1 and W2 are independent of each
other and may run in parallel; W3 and W4 need W2's numbers. ndarray work
needs a session with push access to `AdaWorldAPI/ndarray` (this planning
session has read access only). lance-graph consumes ndarray from git `master`
(mask-risc's manifest), so W5 starts once the ndarray PR merges; no pin bump
is needed.

## 6. Non-goals and scope notes

- lance-graph-java does not call this kernel (lgj-abi uses
  `ternary_match_strided_to_mask`, `ternary_match_u32_to_mask` and
  `ternary_match_u64_to_mask`; no `strided16` reference in `native/lgj-abi/src`), so lgj is out
  of scope. Its column-wise store measured 3.3–4.8× faster hops from
  contiguous lanes alone (lgj `docs/abi.md` §18), which is the reason to expect
  the stride-16 path to pay; it is not evidence that it will.
- No change to mask-risc's fuser, its 256-word tiles or the ternlog dispatch;
  mask chains carry no 128-bit values.
- No new consumer API. If a consumer later wants an explicit contiguous
  16-byte lane type, that is a separate plan.

## 7. Files

- ndarray: `src/simd_masking_ops.rs` (`ternary_match_strided16_to_mask`),
  `crates/simd-masking-parity/src/lib.rs`, `examples/key16_match_probe.rs`
  (new), `.github/workflows/simd-matrix.yaml` (existing rows run W1),
  optionally `src/simd_avx512.rs` (W4).
- lance-graph: none changed; verified are `crates/lance-graph-mask-risc`
  (`exec.rs`, the `Pred::MatchFacet16Strided` arm), `crates/lance-graph-quack`,
  `crates/lance-graph-dir-sim/src/lib.rs` (`subtree`).
