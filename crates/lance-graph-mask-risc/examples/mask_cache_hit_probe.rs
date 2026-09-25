//! What does a cached-mask HIT cost? (`D-WFL-CACHE`'s `C_lookup`.)
//!
//! The plan's cache economics rest on "cached mask -> zero-copy peek -> ~11 ns",
//! recorded as a PREMISE. This probe measures it.
//!
//! 4096 cached masks, each one 64k-row tile (1024 words = 8 KB; 32 MB total, far
//! past L2). A query names a mask by key and peeks a row in it.
//!
//! - lookup: `slot` = direct index into a `Vec`; `map` = `HashMap<u64, usize>`
//!   from a mask key (e.g. an aperture id) to its slot.
//! - peek: `bit` = test one row's bit; `word` = popcount the row's 64-bit word.
//! - hot: every query hits the same mask; cold: a uniformly random mask of 4096.
//! - latency: each query's key and row depend on the previous answer, so no two
//!   queries overlap (the honest "callable in N ns"). throughput: independent.
//!
//! A second table measures the HHTL partial mask applied vertically: one node's
//! 6-byte tier path (HEEL/HIP/TWIG, half of the 12-byte facet) compared under a
//! per-byte care mask, over a working set of 1, 4 or 64 tiles.
//!
//! Measured 2026-09-25 (Xeon @ 2.8 GHz, median of 5, 2 runs):
//!
//! | query | hot | cold |
//! |---|---|---|
//! | cached mask, slot + peek (throughput) | 1.8-2.3 ns | 18.6-22.4 ns |
//! | cached mask, slot + peek (latency) | 5.5-6.0 ns | 142-157 ns |
//! | cached mask, HashMap + peek (throughput) | 13.6 ns | 85-104 ns |
//! | HHTL half partial mask (throughput): 1 / 4 / 64 tiles | 9.9-10.3 ns | 11.6-19.3 / 48.6 ns |
//! | HHTL half partial mask (latency): 1 / 4 / 64 tiles | 30 ns | 55 / 160 ns |
//!
//! The throughput figures include the query generator (one `mix`, a few ns).
//! Queries hit uniformly random nodes; HHTL-ordered access would be kinder.
//!
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example mask_cache_hit_probe`

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

const MASKS: usize = 4096;
const WORDS: usize = 1024;
const Q: usize = 1 << 20;

fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    (x ^ (x >> 29)).wrapping_mul(0xc4ce_b9fe_1a85_ec53)
}

#[derive(Clone, Copy, PartialEq)]
enum Lookup {
    Slot,
    Map,
}
#[derive(Clone, Copy, PartialEq)]
enum Peek {
    Bit,
    Word,
}

struct Cache {
    masks: Vec<u64>,
    keys: Vec<u64>,
    map: HashMap<u64, usize>,
}

impl Cache {
    #[inline(always)]
    fn slot(&self, lookup: Lookup, i: usize) -> usize {
        match lookup {
            Lookup::Slot => i,
            Lookup::Map => self.map[&self.keys[i]],
        }
    }
    #[inline(always)]
    fn peek(&self, s: usize, row: usize, peek: Peek) -> u64 {
        let w = self.masks[s * WORDS + (row >> 6)];
        match peek {
            Peek::Bit => (w >> (row & 63)) & 1,
            Peek::Word => u64::from(w.count_ones()),
        }
    }
}

/// ns per query, dependent chain (latency).
fn latency(c: &Cache, lookup: Lookup, peek: Peek, hot: bool) -> f64 {
    let mut x = 0x1234_5678u64;
    let t = Instant::now();
    for _ in 0..Q {
        let i = if hot { 7 } else { (x as usize) % MASKS };
        let row = ((x >> 16) as usize) % (WORDS * 64);
        let s = c.slot(lookup, i);
        let v = c.peek(s, row, peek);
        x = mix(x ^ v);
    }
    black_box(x);
    t.elapsed().as_nanos() as f64 / Q as f64
}

/// ns per query, independent queries (throughput).
fn throughput(c: &Cache, lookup: Lookup, peek: Peek, hot: bool, qs: &[(usize, usize)]) -> f64 {
    let t = Instant::now();
    let mut acc = 0u64;
    for &(i, row) in qs {
        let i = if hot { 7 } else { i };
        acc = acc.wrapping_add(c.peek(c.slot(lookup, i), row, peek));
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / qs.len() as f64
}

/// HHTL partial mask, vertically: one node's 6-byte tier path (HEEL/HIP/TWIG,
/// 3 x u8:u8 = half of the 12-byte facet) compared under a per-byte care mask.
/// `rows` bounds the working set: a small one stays hot, a large one is cold.
fn hhtl_half(store: &[u8], rows: usize, care: u64, pat: u64, dependent: bool) -> f64 {
    const REC: usize = 16;
    let mut x = 0x9E37_79B9u64;
    let mut acc = 0u64;
    let t = Instant::now();
    for q in 0..Q as u64 {
        let seed = if dependent { x } else { mix(q) };
        let r = (seed as usize) % rows;
        let o = r * REC + 4;
        let mut b = [0u8; 8];
        b[..6].copy_from_slice(&store[o..o + 6]);
        let hit = u64::from((u64::from_le_bytes(b) ^ pat) & care == 0);
        if dependent {
            x = mix(x ^ hit);
        } else {
            acc = acc.wrapping_add(hit);
        }
    }
    black_box((x, acc));
    t.elapsed().as_nanos() as f64 / Q as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let masks: Vec<u64> = (0..MASKS * WORDS).map(|i| mix(i as u64)).collect();
    let keys: Vec<u64> = (0..MASKS).map(|i| mix(i as u64 ^ 0xA9E8)).collect();
    let map = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let c = Cache { masks, keys, map };
    let qs: Vec<(usize, usize)> = (0..Q as u64)
        .map(|q| {
            let h = mix(q ^ 0x5EED);
            ((h as usize) % MASKS, ((h >> 20) as usize) % (WORDS * 64))
        })
        .collect();
    println!(
        "{MASKS} cached masks x {} KB = {} MB; {Q} queries, median of 5",
        WORDS * 8 / 1024,
        (MASKS * WORDS * 8) >> 20
    );
    println!(
        "{:>6} {:>5} {:>11} {:>11} {:>11} {:>11}",
        "lookup", "peek", "hot lat", "cold lat", "hot thru", "cold thru"
    );
    for lookup in [Lookup::Slot, Lookup::Map] {
        for peek in [Peek::Bit, Peek::Word] {
            let r = |f: &dyn Fn() -> f64| median((0..5).map(|_| f()).collect());
            let hl = r(&|| latency(&c, lookup, peek, true));
            let cl = r(&|| latency(&c, lookup, peek, false));
            let ht = r(&|| throughput(&c, lookup, peek, true, &qs));
            let ct = r(&|| throughput(&c, lookup, peek, false, &qs));
            println!(
                "{:>6} {:>5} {hl:>9.2}ns {cl:>9.2}ns {ht:>9.2}ns {ct:>9.2}ns",
                if lookup == Lookup::Slot {
                    "slot"
                } else {
                    "map"
                },
                if peek == Peek::Bit { "bit" } else { "word" }
            );
        }
    }

    // HHTL half-length partial masks over a 16-byte-record store.
    let store: Vec<u8> = (0..(1usize << 22) * 16)
        .map(|i| mix(i as u64) as u8)
        .collect();
    println!("\nHHTL partial mask, 6-byte half path, per node");
    println!("{:>28} {:>11} {:>11}", "working set", "latency", "thru");
    for (name, rows) in [
        ("1 tile (64k rows, 1 MB)", 1usize << 16),
        ("4 tiles (256k rows, 4 MB)", 1 << 18),
        ("64 tiles (4M rows, 64 MB)", 1 << 22),
    ] {
        // HEEL + HIP (4 of the 6 path bytes) cared, TWIG free: a partial mask.
        let care = 0x0000_FFFF_FFFFu64;
        let pat = 0x0000_1234_5678u64;
        let r = |f: &dyn Fn() -> f64| median((0..5).map(|_| f()).collect());
        let l = r(&|| hhtl_half(&store, rows, care, pat, true));
        let t = r(&|| hhtl_half(&store, rows, care, pat, false));
        println!("{name:>28} {l:>9.2}ns {t:>9.2}ns");
    }
}
