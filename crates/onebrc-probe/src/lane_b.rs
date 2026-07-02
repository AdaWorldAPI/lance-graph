//! Lane B — SIMD delimiter scan via `ndarray::simd`.
//!
//! Per Addendum-13 lane B (see `README.md` §3), this lane measures vectorized
//! `;`/`\n` scanning against Lane A's byte-wise scalar scan, on identical
//! parse + accumulate logic. The workspace's SIMD iron rule is "all SIMD from
//! `ndarray::simd`" (`simd-savant` agent,
//! `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`) — this module
//! uses `ndarray::simd::{U8x32, U8x64}::cmpeq_mask` exclusively, never a raw
//! `core::arch` intrinsic, `pulp`, `wide`, or `memchr`.
//!
//! **Width is compile-time-dispatched** (`SimdByte`): under AVX-512
//! (`target-cpu=x86-64-v4` / `native`) the scan strides 64-byte `zmm`
//! blocks via `U8x64` (`cmpeq_mask -> u64`); under the AVX2 default
//! (`x86-64-v3`, the CI baseline) it strides 32-byte `ymm` blocks via
//! `U8x32` (`cmpeq_mask -> u32`; one `__m256i`, see
//! `ndarray/src/simd_avx2.rs`). For each block,
//! `cmpeq_mask(SimdByte::splat(b'\n'))` and `cmpeq_mask(SimdByte::splat(b';'))`
//! produce a `SimdByte::LANES`-bit mask with bit `i` set iff `block[i]`
//! matches. The set bits of the combined mask (newline and semicolon bytes
//! never coincide, so `nl_mask | semi_mask` has no lost information) are
//! walked in ascending order via the classic `mask & (mask - 1)` "clear
//! lowest set bit" trick, recovering the ordered sequence of delimiter
//! events for the block. The walk is generic over the `u32`/`u64` mask
//! width, so the same body serves both dispatched widths.
//!
//! **Parse remains scalar** (SWAR/branchless parse deliberately deferred —
//! see `README.md` §1 "NOT reimplemented here"): `parse_temp_tenths` is the
//! same byte-scan integer parser Lane A uses. This lane's speedup, if any,
//! comes purely from vectorized delimiter-finding, not from vectorized
//! parsing.
//!
//! ## Cross-block record state
//!
//! A record's `;` and its `\n` are not guaranteed to land in the same
//! block (short station names put the `;` near a block's end and the `\n`
//! in the next block, or vice versa). Two scalars carry state across block
//! boundaries:
//!
//! - `line_start: usize` — the byte offset where the current (in-progress)
//!   station name begins.
//! - `pending_semi: Option<usize>` — `Some(offset)` once this record's `;`
//!   has been seen but its `\n` has not yet arrived; `None` while still
//!   scanning for the `;`.
//!
//! The tail (fewer than one full `SimdByte`-wide block remaining after the
//! last full block) is finished with a plain byte-wise scalar loop — the
//! same station-name / temp-field extraction shape as `lane_a_scalar`,
//! continuing from whatever `line_start` / `pending_semi` state the SIMD
//! pass left behind.

use crate::{parse_temp_tenths, Stats};
use ndarray::simd::array_chunks;
use std::collections::BTreeMap;

// Compile-time SIMD byte-width dispatch — both widths are `ndarray::simd`
// types (the iron rule; never a raw `core::arch` intrinsic). AVX-512
// targets (`target-cpu=x86-64-v4` / `native`) scan in 64-byte `zmm`
// strides via `U8x64`; the AVX2 default (`x86-64-v3`, the CI baseline)
// scans in 32-byte `ymm` strides via `U8x32`. The stride, the needle
// width, and the `array_chunks` const-generic all key off
// `SimdByte::LANES`, so the same body strides the widest lane the target
// actually provides. `cmpeq_mask` returns a `u64` (avx512) / `u32`
// (avx2) mask; the ascending set-bit walk below is generic over both.
#[cfg(not(target_feature = "avx512f"))]
use ndarray::simd::U8x32 as SimdByte;
#[cfg(target_feature = "avx512f")]
use ndarray::simd::U8x64 as SimdByte;

/// The SIMD block width lane B actually strides, in bytes — the dispatched
/// `SimdByte::LANES` (64 under avx512, 32 under avx2). Exposed so tests can
/// assert against the real block boundary instead of a hardcoded width.
#[cfg(test)]
pub(crate) const SIMD_LANES: usize = SimdByte::LANES;

/// Lane B — SIMD delimiter scan. One pass over `data` in `SimdByte::LANES`-wide
/// strides (64-byte `zmm` under avx512, 32-byte `ymm` under avx2) using
/// `ndarray::simd::{U8x64, U8x32}::cmpeq_mask` to locate `;` and `\n` bytes;
/// scalar integer temp parse (see module doc); `BTreeMap<String, Stats>`
/// accumulation identical in shape to `lane_a_scalar`.
pub fn lane_b_simd(data: &[u8]) -> BTreeMap<String, Stats> {
    let mut map: BTreeMap<String, Stats> = BTreeMap::new();
    let len = data.len();

    // `line_start` — offset where the in-progress station name begins.
    // `pending_semi` — `Some(offset)` once `;` has been seen for the
    // in-progress record but its `\n` has not yet arrived.
    let mut line_start = 0usize;
    let mut pending_semi: Option<usize> = None;

    let nl_needle = SimdByte::splat(b'\n');
    let semi_needle = SimdByte::splat(b';');

    // The non-overlapping `SimdByte::LANES`-wide stride walk routes through
    // `ndarray::simd::array_chunks` (simd_ops.rs) — the W1a batch-walk
    // primitive; `array_windows` is its OVERLAPPING sibling (GEMM-style
    // row windows) and is deliberately NOT used here: delimiter scanning
    // never re-reads bytes. The const-generic is `{ SimdByte::LANES }`, so
    // the chunk width tracks the dispatched lane width (64 under avx512,
    // 32 under avx2) with no hardcoded stride.
    let aligned_end = (len / SimdByte::LANES) * SimdByte::LANES;
    for (chunk_idx, chunk) in
        array_chunks::<u8, { SimdByte::LANES }>(&data[..aligned_end]).enumerate()
    {
        let pos = chunk_idx * SimdByte::LANES;
        let block = SimdByte::from_slice(chunk);
        let nl_mask = block.cmpeq_mask(nl_needle);
        let semi_mask = block.cmpeq_mask(semi_needle);
        // `;` and `\n` never occupy the same byte, so OR-ing loses no
        // information and gives one ascending walk over both event kinds.
        let mut combined = nl_mask | semi_mask;
        while combined != 0 {
            let bit = combined.trailing_zeros() as usize;
            let abs = pos + bit;
            if (nl_mask >> bit) & 1 == 1 {
                // Newline event: closes the temp field started at the
                // most recent pending semicolon.
                let semi = pending_semi
                    .take()
                    .expect("newline event must be preceded by a pending semicolon");
                let name = std::str::from_utf8(&data[line_start..semi])
                    .expect("station name is valid utf8");
                let tenths = parse_temp_tenths(&data[semi + 1..abs]);
                match map.get_mut(name) {
                    Some(stats) => stats.observe(tenths),
                    None => {
                        map.insert(name.to_string(), Stats::single(tenths));
                    }
                }
                line_start = abs + 1;
            } else {
                // Semicolon event: closes the station name, opens the temp
                // field.
                pending_semi = Some(abs);
            }
            combined &= combined - 1; // clear the lowest set bit
        }
    }

    // Tail — fewer than `SimdByte::LANES` bytes remain. Finish with a plain
    // scalar scan, continuing from whatever `line_start` / `pending_semi`
    // state the SIMD pass left behind (mirrors `lane_a_scalar`'s per-record
    // shape).
    let mut i = aligned_end;
    while i < len {
        match pending_semi {
            None => {
                while data[i] != b';' {
                    i += 1;
                }
                pending_semi = Some(i);
                i += 1;
            }
            Some(semi) => {
                while data[i] != b'\n' {
                    i += 1;
                }
                let name = std::str::from_utf8(&data[line_start..semi])
                    .expect("station name is valid utf8");
                let tenths = parse_temp_tenths(&data[semi + 1..i]);
                match map.get_mut(name) {
                    Some(stats) => stats.observe(tenths),
                    None => {
                        map.insert(name.to_string(), Stats::single(tenths));
                    }
                }
                i += 1;
                line_start = i;
                pending_semi = None;
            }
        }
    }

    map
}
