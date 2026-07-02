//! Lane B — SIMD delimiter scan via `ndarray::simd`.
//!
//! Per Addendum-13 lane B (see `README.md` §3), this lane measures vectorized
//! `;`/`\n` scanning against Lane A's byte-wise scalar scan, on identical
//! parse + accumulate logic. The workspace's SIMD iron rule is "all SIMD from
//! `ndarray::simd`" (`simd-savant` agent,
//! `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`) — this module
//! uses `ndarray::simd::U8x32::cmpeq_mask` exclusively, never a raw
//! `core::arch` intrinsic, `pulp`, `wide`, or `memchr`.
//!
//! `U8x32` is the AVX2-native byte width (one `__m256i` = 32 bytes; see
//! `ndarray/src/simd_avx2.rs` module doc). The corpus is scanned in 32-byte
//! strides: for each block, `cmpeq_mask(U8x32::splat(b'\n'))` and
//! `cmpeq_mask(U8x32::splat(b';'))` produce 32-bit masks with bit `i` set
//! iff `block[i]` matches. The set bits of the combined mask (newline and
//! semicolon bytes never coincide, so `nl_mask | semi_mask` has no lost
//! information) are walked in ascending order via the classic
//! `mask & (mask - 1)` "clear lowest set bit" trick, recovering the ordered
//! sequence of delimiter events for the block.
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
//! 32-byte block (short station names put the `;` near a block's end and
//! the `\n` in the next block, or vice versa). Two scalars carry state
//! across block boundaries:
//!
//! - `line_start: usize` — the byte offset where the current (in-progress)
//!   station name begins.
//! - `pending_semi: Option<usize>` — `Some(offset)` once this record's `;`
//!   has been seen but its `\n` has not yet arrived; `None` while still
//!   scanning for the `;`.
//!
//! The tail (fewer than 32 bytes remaining after the last full block) is
//! finished with a plain byte-wise scalar loop — the same station-name /
//! temp-field extraction shape as `lane_a_scalar`, continuing from whatever
//! `line_start` / `pending_semi` state the SIMD pass left behind.

use crate::{parse_temp_tenths, Stats};
use ndarray::simd::U8x32;
use std::collections::BTreeMap;

/// Lane B — SIMD delimiter scan. One pass over `data` in 32-byte strides
/// using `ndarray::simd::U8x32::cmpeq_mask` to locate `;` and `\n` bytes;
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

    let nl_needle = U8x32::splat(b'\n');
    let semi_needle = U8x32::splat(b';');

    let aligned_end = (len / U8x32::LANES) * U8x32::LANES;
    let mut pos = 0usize;
    while pos < aligned_end {
        let block = U8x32::from_slice(&data[pos..pos + U8x32::LANES]);
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
        pos += U8x32::LANES;
    }

    // Tail — fewer than 32 bytes remain. Finish with a plain scalar scan,
    // continuing from whatever `line_start` / `pending_semi` state the SIMD
    // pass left behind (mirrors `lane_a_scalar`'s per-record shape).
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
