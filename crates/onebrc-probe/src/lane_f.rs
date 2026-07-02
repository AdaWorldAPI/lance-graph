//! Lane F — the substrate-native lane: group-by-identity as a prefix
//! ROUTE, aggregation as a gated indexed write into SoA-shaped
//! accumulators. Plus Lane R, its honest control.
//!
//! Per Addendum-13 lane F (operator: "process it as cognitive shader in a
//! morton tile cascaded batch"): a station's identity is hashed to a key,
//! the key is read as **two axis bytes**, and the axes are
//! **nibble-interleaved** into a 16-bit Morton tile position — the
//! 256×256 centroid-tile read of the GUID canon (OGAR `CLAUDE.md` § "Tier
//! interpretation": each tier's 16 bits = two 256-entry axes,
//! nibble-interleaved, coarse→fine = alternating-axis refinement, "Morton
//! in centroid space"). That tile position IS the accumulator address:
//! records route into flat parallel arrays (`min[]/max[]/sum[]/count[]` —
//! SoA-shaped, one slot per tile) and aggregation is an **indexed gated
//! write** into the worker's OWN arrays (single-writer by ownership),
//! with a commutative BUNDLE merge across workers at the end — the
//! borrow-strategy write-back discipline, applied to raw OLAP.
//!
//! ## The honest control — Lane R
//!
//! The Morton route is radix bucketing wearing our address. The fastest
//! known 1BRC entries are radix/perfect-hash designs, so lane F alone
//! proves nothing about the ADDRESS — only about flat-table-vs-BTreeMap.
//! Lane R (`lane_r_radix`) is byte-identical to lane F except for ONE
//! line: the slot function is the plain low 16 bits of the same hash
//! (`h & 0xFFFF`), no interleave. Therefore:
//!
//! - **R vs C** prices the open-addressed SoA flat table against the
//!   `BTreeMap` accumulation (the data-structure win, address-agnostic).
//! - **F vs R** prices the Morton addressing itself (the interleave ALU
//!   cost + any cache-distribution difference) — the ADDRESSING TAX
//!   Addendum-13 sends this lane to isolate. F ≈ R validates
//!   addressing-is-aggregation (the semantic address costs nothing over
//!   a plain radix bucket); F < R prices the address layer.
//!
//! ## What is deliberately NOT here
//!
//! - No per-tile record bucketing + cascade-ordered sweep: with ~400
//!   groups the scatter/gather of a two-pass tile batch costs more than
//!   it buys; the direct-indexed write already exercises the
//!   route-then-accumulate shape. (A 100M-row, high-cardinality corpus is
//!   where tile-batched sweeps would earn their keep — noted for a
//!   follow-up, not smuggled into this measurement.)
//! - No kanban scheduling of tile batches: lane E measured that tax
//!   (~66 µs/card, within noise at chunk granularity) — re-adding it
//!   here would only blur F−R.
//! - No SIMD scan: that is lane B's variable. F/R use the same scalar
//!   byte scan as lanes A/C so the ONLY variable vs lane C is the
//!   accumulator (and, between F and R, the slot function). std-only —
//!   lanes A/C/F/R keep the crate's zero-dep contract.
//!
//! ## Hash
//!
//! FNV-1a 64 over the station-name bytes (own impl, no dep). Both slot
//! functions consume the SAME hash value, so the slot fn is the only
//! difference between F and R. Name-byte equality is verified on every
//! tag hit (not just the 64-bit tag), so a hash collision degrades to a
//! linear probe, never to a wrong merge — both lanes pay the identical
//! verification cost, keeping F−R clean.

use crate::{parse_temp_tenths, Stats};
use std::collections::BTreeMap;

/// Number of accumulator slots — the full 16-bit tile space (256×256).
pub(crate) const SLOTS: usize = 1 << 16;

/// FNV-1a 64-bit over the station name bytes.
#[inline(always)]
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Morton tile position: the hash's two low axis bytes, nibble-interleaved
/// coarse→fine (`x_hi y_hi x_lo y_lo`) — the 256×256 centroid-tile read of
/// the GUID canon (alternating-axis refinement; each byte's nibbles are the
/// axis's coarse→fine ancestry).
#[inline(always)]
pub(crate) fn morton_slot(h: u64) -> u16 {
    let x = (h & 0xFF) as u16;
    let y = ((h >> 8) & 0xFF) as u16;
    ((x & 0xF0) << 8) | ((y & 0xF0) << 4) | ((x & 0x0F) << 4) | (y & 0x0F)
}

/// Radix control slot: the plain low 16 bits of the same hash — identical
/// pipeline, no interleave. The one-line difference that isolates the
/// addressing tax.
#[inline(always)]
fn radix_slot(h: u64) -> u16 {
    (h & 0xFFFF) as u16
}

/// One worker's owned accumulator: SoA parallel arrays indexed by tile
/// slot, open-addressed (linear probe) on collision. Single-writer by
/// ownership — each worker builds its own table; cross-worker combination
/// is the commutative BUNDLE merge in [`table_to_map`] + `merge_maps`.
pub(crate) struct SoaTable {
    /// Full 64-bit hash tag per slot; `None`-ness is carried by `names`.
    pub(crate) tags: Vec<u64>,
    /// Station name owned per occupied slot (empty vec = unoccupied).
    /// Verified byte-for-byte on every tag hit — see module doc "Hash".
    pub(crate) names: Vec<Vec<u8>>,
    // SoA value arrays — the "SoA-shaped accumulators" of Addendum-13:
    // one field per column, indexed by slot, updated by gated indexed
    // writes (min/max/sum/count — each write is a fold, never a blind
    // overwrite of foreign state).
    pub(crate) mins: Vec<i32>,
    pub(crate) maxs: Vec<i32>,
    pub(crate) sums: Vec<i64>,
    pub(crate) counts: Vec<u32>,
}

impl SoaTable {
    pub(crate) fn new() -> Self {
        Self {
            tags: vec![0; SLOTS],
            names: vec![Vec::new(); SLOTS],
            mins: vec![i32::MAX; SLOTS],
            maxs: vec![i32::MIN; SLOTS],
            sums: vec![0; SLOTS],
            counts: vec![0; SLOTS],
        }
    }

    /// Route `name` to its slot (slot fn + linear probe) and fold one
    /// observation into the SoA columns at that address. Returns the
    /// RESOLVED slot index (post-probe) so callers that track dirty slots
    /// (lane G's morsel extraction) can record it; lanes F/R ignore it.
    #[inline(always)]
    pub(crate) fn observe(&mut self, slot0: u16, h: u64, name: &[u8], tenths: i32) -> usize {
        let mut s = slot0 as usize;
        loop {
            if self.counts[s] == 0 {
                // First occupancy of this slot: claim it for `name`.
                self.tags[s] = h;
                self.names[s] = name.to_vec();
                break;
            }
            if self.tags[s] == h && self.names[s] == name {
                break;
            }
            // Collision (different station hashed/probed here): linear
            // probe to the next slot, wrapping. ~400 stations in 65536
            // slots keeps probe chains ≈ 1.
            s = (s + 1) & (SLOTS - 1);
        }
        // The gated indexed write — folds, never blind assignment.
        if tenths < self.mins[s] {
            self.mins[s] = tenths;
        }
        if tenths > self.maxs[s] {
            self.maxs[s] = tenths;
        }
        self.sums[s] += tenths as i64;
        self.counts[s] += 1;
        s
    }
}

/// Scan `data` (the same scalar `;`/`\n` byte scan as lane A) routing every
/// record through `slot_of` into an owned [`SoaTable`]. Generic over the
/// slot function so F and R monomorphize separately (zero-cost, no fn-ptr
/// indirection in the hot loop).
fn accumulate_table(data: &[u8], slot_of: impl Fn(u64) -> u16 + Copy) -> SoaTable {
    let mut table = SoaTable::new();
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

        let h = fnv1a64(name);
        table.observe(slot_of(h), h, name, tenths);
    }
    table
}

/// Sweep a worker's table into the common `BTreeMap<String, Stats>` output
/// shape (occupied slots only) so cross-worker combination reuses the same
/// commutative `merge_maps` BUNDLE step every other lane uses — and so the
/// parity tests compare like with like.
pub(crate) fn table_to_map(table: SoaTable) -> BTreeMap<String, Stats> {
    let mut out = BTreeMap::new();
    for s in 0..SLOTS {
        if table.counts[s] > 0 {
            let name = String::from_utf8(table.names[s].clone()).expect("station name utf8");
            out.insert(
                name,
                Stats {
                    min: table.mins[s],
                    max: table.maxs[s],
                    sum: table.sums[s],
                    count: table.counts[s],
                },
            );
        }
    }
    out
}

/// Shared threaded driver for F/R: `chunk_bounds` split (identical to lane
/// C), each worker accumulates its OWN [`SoaTable`] over its slice, tables
/// sweep to maps, maps BUNDLE-merge. Only `slot_of` differs between lanes.
fn lane_table_threads(
    data: &[u8],
    workers: usize,
    slot_of: impl Fn(u64) -> u16 + Copy + Send,
) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let bounds = crate::chunk_bounds(data, workers);
    let results: Vec<BTreeMap<String, Stats>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                scope.spawn(move || table_to_map(accumulate_table(slice, slot_of)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("lane F/R worker panicked"))
            .collect()
    });
    crate::merge_maps(results)
}

/// Lane F — Morton-tile routed SoA accumulation (see module doc). The
/// station key's two axis bytes, nibble-interleaved into the 256×256 tile
/// space, ARE the accumulator address.
pub fn lane_f_morton(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_table_threads(data, workers, morton_slot)
}

/// Lane R — the plain-radix control: identical pipeline, slot = low 16
/// hash bits, no interleave. F−R isolates the addressing tax.
pub fn lane_r_radix(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_table_threads(data, workers, radix_slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_slot_nibble_interleaves_coarse_to_fine() {
        // x = 0xAB (h low byte), y = 0xCD (h next byte) →
        // x_hi y_hi x_lo y_lo = 0xA C B D.
        let h = 0xCDAB_u64; // low byte 0xAB → x; next byte 0xCD → y
        assert_eq!(morton_slot(h), 0xACBD);
        // Degenerate corners.
        assert_eq!(morton_slot(0x0000), 0x0000);
        assert_eq!(morton_slot(0xFFFF), 0xFFFF);
        // Only the low 16 bits of the hash participate.
        assert_eq!(morton_slot(0xDEAD_BEEF_0000_CDAB), 0xACBD);
    }

    #[test]
    fn forced_collisions_probe_correctly() {
        // A constant slot function forces EVERY station into slot 0 —
        // the probe chain must still keep stations separate (tag + full
        // name-byte verification), never merge two stations' stats.
        let corpus = b"aa;1.0\nbb;2.0\naa;3.0\ncc;-4.0\nbb;0.5\n";
        let table = accumulate_table(corpus, |_| 0u16);
        let map = table_to_map(table);
        assert_eq!(map.len(), 3, "three stations despite total slot collision");
        assert_eq!(
            map["aa"],
            Stats {
                min: 10,
                max: 30,
                sum: 40,
                count: 2
            }
        );
        assert_eq!(
            map["bb"],
            Stats {
                min: 5,
                max: 20,
                sum: 25,
                count: 2
            }
        );
        assert_eq!(
            map["cc"],
            Stats {
                min: -40,
                max: -40,
                sum: -40,
                count: 1
            }
        );
    }

    #[test]
    fn lane_f_and_lane_r_agree_with_lane_a_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_f_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 33).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let f = lane_f_morton(&data, 3);
        let r = lane_r_radix(&data, 3);
        assert_eq!(a, f, "lane F must produce identical aggregates to lane A");
        assert_eq!(a, r, "lane R must produce identical aggregates to lane A");
        assert!(!a.is_empty());
    }
}
