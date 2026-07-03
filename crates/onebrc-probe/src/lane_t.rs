//! Lane T — the HHTL **trie** lane: group-by as a prefix DESCENT, not a hash.
//!
//! Where lane F/R hash the station name (FNV-1a) into a flat table slot and
//! linear-probe on collision, lane T makes the **name itself the path**: it
//! descends an arena-backed trie one symbol per level to a terminal node that
//! holds the station's accumulator. Collision-free by construction — distinct
//! names reach distinct terminal nodes, shared prefixes share internal nodes.
//! No hash pass, no probe chain, no tag/name re-verification.
//!
//! This is the operational form of the canon's `panCAKES ≡ radix trie ≡ HHTL`
//! (`contract::hhtl::NiblePath`): the keys ARE the tree, routing is pure
//! index arithmetic on the key, and the descent never touches a value until
//! the terminal fold. Two variants, measured side by side:
//!
//! - [`lane_t_trie`] — **16-ary nibble trie** (HHTL-faithful: `FAN_OUT = 16`,
//!   one nibble per level, high-then-low per byte → 2 levels per name byte).
//! - [`lane_t_byte`] — **256-ary byte trie** (one level per byte → half the
//!   descent depth, larger nodes). The honest "is the 16-ary descent depth or
//!   the trie idea itself the cost?" control.
//!
//! Same scalar `;`/`\n` byte scan and same `chunk_bounds`/`merge_maps` threaded
//! driver as lanes A/C/F/R — the ONLY variable vs lane F is the accumulator
//! (trie descent instead of hash+slot+probe). std-only; keeps the crate's
//! zero-dep contract.

use crate::{chunk_bounds, merge_maps, parse_temp_tenths, Stats};
use std::collections::BTreeMap;

/// HHTL fan-out: 16 children per level (one nibble). Matches `contract::hhtl`.
const FANOUT16: usize = 16;
/// Byte-trie fan-out: 256 children per level (one byte, half the depth).
const FANOUT256: usize = 256;

/// Arena-backed trie over the station-name bytes, generic in fan-out via the
/// two `observe_*` descents. `children[node * fanout + sym] = child index`
/// (`0` = empty; node `0` is the root and is never a child, so `0` is a safe
/// empty sentinel). SoA accumulators are one slot per node; only terminal
/// nodes are folded.
struct Trie {
    fanout: usize,
    children: Vec<u32>,
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
    names: Vec<Vec<u8>>,
}

impl Trie {
    fn new(fanout: usize) -> Self {
        // node 0 = root, with `fanout` empty children.
        Self {
            fanout,
            children: vec![0u32; fanout],
            mins: vec![i32::MAX],
            maxs: vec![i32::MIN],
            sums: vec![0],
            counts: vec![0],
            names: vec![Vec::new()],
        }
    }

    /// Follow (or create) the child of `node` at symbol `sym`, returning the
    /// child node index.
    #[inline(always)]
    fn descend(&mut self, node: usize, sym: usize) -> usize {
        let idx = node * self.fanout + sym;
        let child = self.children[idx];
        if child != 0 {
            child as usize
        } else {
            let new = self.counts.len();
            self.children.extend(std::iter::repeat(0u32).take(self.fanout));
            self.mins.push(i32::MAX);
            self.maxs.push(i32::MIN);
            self.sums.push(0);
            self.counts.push(0);
            self.names.push(Vec::new());
            self.children[idx] = new as u32;
            new
        }
    }

    /// Fold one observation into the terminal node reached for `name`.
    #[inline(always)]
    fn fold(&mut self, node: usize, name: &[u8], tenths: i32) {
        if self.counts[node] == 0 {
            self.names[node] = name.to_vec();
        }
        if tenths < self.mins[node] {
            self.mins[node] = tenths;
        }
        if tenths > self.maxs[node] {
            self.maxs[node] = tenths;
        }
        self.sums[node] += tenths as i64;
        self.counts[node] += 1;
    }

    /// 16-ary descent: high nibble then low nibble of each byte.
    #[inline(always)]
    fn observe_nibble(&mut self, name: &[u8], tenths: i32) {
        let mut node = 0usize;
        for &b in name {
            node = self.descend(node, (b >> 4) as usize);
            node = self.descend(node, (b & 0x0F) as usize);
        }
        self.fold(node, name, tenths);
    }

    /// 256-ary descent: one byte per level.
    #[inline(always)]
    fn observe_byte(&mut self, name: &[u8], tenths: i32) {
        let mut node = 0usize;
        for &b in name {
            node = self.descend(node, b as usize);
        }
        self.fold(node, name, tenths);
    }

    fn into_map(self) -> BTreeMap<String, Stats> {
        let mut out = BTreeMap::new();
        for node in 0..self.counts.len() {
            if self.counts[node] > 0 {
                let name = String::from_utf8(self.names[node].clone()).expect("station name utf8");
                out.insert(
                    name,
                    Stats {
                        min: self.mins[node],
                        max: self.maxs[node],
                        sum: self.sums[node],
                        count: self.counts[node],
                    },
                );
            }
        }
        out
    }
}

/// Scan `data` (the same scalar `;`/`\n` byte scan as lane F), routing every
/// record through the trie `descend` closure into an owned [`Trie`].
#[inline]
fn accumulate_trie(data: &[u8], fanout: usize, nibble: bool) -> Trie {
    let mut trie = Trie::new(fanout);
    let len = data.len();
    let mut i = 0usize;
    while i < len {
        let name_start = i;
        while data[i] != b';' {
            i += 1;
        }
        let name = &data[name_start..i];
        i += 1; // skip ';'
        let temp_start = i;
        while data[i] != b'\n' {
            i += 1;
        }
        let tenths = parse_temp_tenths(&data[temp_start..i]);
        i += 1; // skip '\n'
        if nibble {
            trie.observe_nibble(name, tenths);
        } else {
            trie.observe_byte(name, tenths);
        }
    }
    trie
}

fn lane_trie_threads(data: &[u8], workers: usize, fanout: usize, nibble: bool) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let bounds = chunk_bounds(data, workers);
    let results: Vec<BTreeMap<String, Stats>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                scope.spawn(move || accumulate_trie(slice, fanout, nibble).into_map())
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("lane T worker panicked"))
            .collect()
    });
    merge_maps(results)
}

/// Lane T — HHTL **16-ary nibble trie**: the name descends one nibble per
/// level (high-then-low per byte), the terminal node IS the accumulator.
pub fn lane_t_trie(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_trie_threads(data, workers, FANOUT16, true)
}

/// Lane T (byte) — **256-ary byte trie**: one level per byte (half the descent
/// depth). The control for "is the 16-ary depth or the trie itself the cost?".
pub fn lane_t_byte(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_trie_threads(data, workers, FANOUT256, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sharing_and_prefix_of_another_name() {
        // "ab" is a prefix of "abc" — the "ab" terminal is ALSO an internal
        // node on the "abc" path. Both must accumulate independently.
        let corpus = b"ab;1.0\nabc;2.0\nab;3.0\nabc;-4.0\nz;0.5\n";
        for nibble in [true, false] {
            let fanout = if nibble { FANOUT16 } else { FANOUT256 };
            let trie = accumulate_trie(corpus, fanout, nibble);
            let map = trie.into_map();
            assert_eq!(map.len(), 3, "three stations (nibble={nibble})");
            assert_eq!(
                map["ab"],
                Stats { min: 10, max: 30, sum: 40, count: 2 },
                "nibble={nibble}"
            );
            assert_eq!(
                map["abc"],
                Stats { min: -40, max: 20, sum: -20, count: 2 },
                "nibble={nibble}"
            );
            assert_eq!(map["z"], Stats { min: 5, max: 5, sum: 5, count: 1 });
        }
    }

    #[test]
    fn both_tries_agree_with_lane_a_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_t_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 71).expect("gen");
        assert_eq!(result.rows, 50_000);
        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let t16 = lane_t_trie(&data, 3);
        let t256 = lane_t_byte(&data, 3);
        assert_eq!(a, t16, "16-ary nibble trie must equal lane A");
        assert_eq!(a, t256, "256-ary byte trie must equal lane A");
        assert!(!a.is_empty());
    }
}
