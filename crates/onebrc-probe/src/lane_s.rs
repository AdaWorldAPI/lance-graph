//! Lane S — the SWAR lane: the actual 1BRC-frontier compute levers on top of
//! the flat open-addressed table (lane F's group-by, which was already right).
//!
//! What the real 1BRC winners do that lanes A/C/F/R do NOT (see the answer in
//! the session log): find delimiters with SWAR (8 bytes at a time in a `u64`)
//! instead of a byte-by-byte `while data[i] != b';'` loop, and parse the
//! temperature branchlessly. Those are the two measurable levers here.
//!
//! - **(b) SWAR delimiter find** — the classic `haszero` bit trick:
//!   `x = word ^ needle_bcast; (x - 0x0101…) & !x & 0x8080…` sets the high bit
//!   of each byte where `word == needle`. `trailing_zeros() >> 3` gives the
//!   first match position. Replaces the scalar name/temp scans.
//! - **(b) branchless temp parse** — the field is `-?\d?\d.\d`; parse to fixed
//!   -point tenths with no data-dependent branches beyond the sign/width.
//! - **(c) name compare** — kept as `&[u8] == &[u8]`, which LLVM already lowers
//!   to a vectorized `memcmp`; a hand-rolled 8-byte-chunk compare is the same
//!   instruction sequence, so this is (c) already.
//!
//! **(a) mmap is deliberately NOT here and is NOT measurable in this harness:**
//! `main.rs` does `fs::read` BEFORE `Instant::now()`, so the `mrows_s` metric is
//! compute-only. mmap is a wall-clock / 13 GB-allocation lever for the full 1B
//! file, on an axis this timer does not observe. Claiming an mmap speedup on
//! `mrows_s` would be false; measuring it needs an end-to-end wall-clock mode.
//!
//! Reuses lane F's `SoaTable` (flat open-addressed accumulator) verbatim — the
//! ONLY variable vs lane F is the scan+parse. Same `chunk_bounds`/`merge_maps`
//! threading. std-only; keeps the crate's zero-dep contract.

use crate::lane_f::{fnv1a64, morton_slot, table_to_map, SoaTable};
use crate::{chunk_bounds, merge_maps, Stats};
use std::collections::BTreeMap;

const SEMI_BCAST: u64 = 0x3B3B_3B3B_3B3B_3B3B; // b';' in every byte
const NL_BCAST: u64 = 0x0A0A_0A0A_0A0A_0A0A; // b'\n' in every byte
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// SWAR "has zero byte" applied to `word ^ bcast`: a set 0x80 in each byte
/// where `word`'s byte equals `bcast`'s byte.
#[inline(always)]
fn match_mask(word: u64, bcast: u64) -> u64 {
    let x = word ^ bcast;
    x.wrapping_sub(ONES) & !x & HIGH
}

/// First index `>= i` where `data[idx]` equals `bcast`'s byte — SWAR 8 bytes at
/// a time, scalar tail. Reads only within `data` (the `i + 8 <= len` guard).
#[inline(always)]
fn find(data: &[u8], mut i: usize, bcast: u64) -> usize {
    let len = data.len();
    while i + 8 <= len {
        let word = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
        let m = match_mask(word, bcast);
        if m != 0 {
            return i + ((m.trailing_zeros() >> 3) as usize);
        }
        i += 8;
    }
    let b = bcast as u8;
    while i < len && data[i] != b {
        i += 1;
    }
    i
}

/// Parse `-?\d?\d.\d` to fixed-point tenths. Branch only on sign and 1-vs-2
/// integer digits (both perfectly predicted at ~50/50 and ~90/10).
#[inline(always)]
fn parse_tenths(s: &[u8]) -> i32 {
    let neg = s[0] == b'-';
    let d = if neg { &s[1..] } else { s };
    let v = if d.len() == 3 {
        // d.d
        (d[0] - b'0') as i32 * 10 + (d[2] - b'0') as i32
    } else {
        // dd.d
        (d[0] - b'0') as i32 * 100 + (d[1] - b'0') as i32 * 10 + (d[3] - b'0') as i32
    };
    if neg {
        -v
    } else {
        v
    }
}

/// Scan `data` with the SWAR find + branchless parse, folding into lane F's
/// flat `SoaTable`. The ONLY difference from `lane_f::accumulate_table` is the
/// scan/parse — same hash, same slot fn, same table.
fn accumulate_swar(data: &[u8]) -> SoaTable {
    let mut table = SoaTable::new();
    let len = data.len();
    let mut i = 0usize;
    while i < len {
        let name_start = i;
        let semi = find(data, i, SEMI_BCAST);
        let name = &data[name_start..semi];
        let nl = find(data, semi + 1, NL_BCAST);
        let tenths = parse_tenths(&data[semi + 1..nl]);
        let h = fnv1a64(name);
        table.observe(morton_slot(h), h, name, tenths);
        i = nl + 1;
    }
    table
}

/// Lane S — SWAR delimiter scan + branchless parse over lane F's flat table.
pub fn lane_s_swar(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let bounds = chunk_bounds(data, workers);
    let results: Vec<BTreeMap<String, Stats>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                scope.spawn(move || table_to_map(accumulate_swar(slice)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("lane S worker panicked"))
            .collect()
    });
    merge_maps(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swar_find_matches_scalar() {
        let data = b"hello;12.3\nab;-4.5\n";
        // ';' after "hello" (idx 5), '\n' after "12.3" (idx 10), etc.
        assert_eq!(find(data, 0, SEMI_BCAST), 5);
        assert_eq!(find(data, 6, NL_BCAST), 10);
        assert_eq!(find(data, 11, SEMI_BCAST), 13);
        assert_eq!(find(data, 14, NL_BCAST), 18);
    }

    #[test]
    fn parse_tenths_all_shapes() {
        assert_eq!(parse_tenths(b"1.0"), 10);
        assert_eq!(parse_tenths(b"12.3"), 123);
        assert_eq!(parse_tenths(b"-4.5"), -45);
        assert_eq!(parse_tenths(b"-99.9"), -999);
        assert_eq!(parse_tenths(b"0.0"), 0);
    }

    #[test]
    fn lane_s_agrees_with_lane_a() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_s_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 97).expect("gen");
        assert_eq!(result.rows, 50_000);
        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let s = lane_s_swar(&data, 3);
        assert_eq!(a, s, "SWAR lane must produce identical aggregates to lane A");
        assert!(!a.is_empty());
    }
}
