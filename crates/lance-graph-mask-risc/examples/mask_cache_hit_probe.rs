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
//!   queries overlap. That chain runs through `mix()`, which costs ~3.1 ns on
//!   its own (printed); subtract it to get the lookup's own latency.
//!   throughput: independent queries.
//!
//! A second table measures the HHTL partial mask applied vertically: one node's
//! 6-byte tier path (HEEL/HIP/TWIG, half of the 12-byte facet) compared under a
//! per-byte care mask. Its latency is a pointer chase: a single random cycle is
//! written into each member's record, so the next row comes from the bytes just
//! loaded and nothing else is on the chain.
//!
//! Measured 2026-09-25 (Xeon @ 2.8 GHz, median of 5, 2 runs):
//!
//! | query | hot | cold |
//! |---|---|---|
//! | cached mask, slot + peek (throughput) | 1.9-2.3 ns | 17-21 ns |
//! | cached mask, slot + peek (latency, incl. ~3.1 ns mix) | 5.6-5.8 ns | 135-156 ns |
//! | cached mask, HashMap + peek (throughput) | 15.3-15.8 ns | 99-109 ns |
//!
//! HHTL half partial mask, per node, 16-byte records:
//!
//! | members | latency (pointer chase) | throughput |
//! |---|---|---|
//! | <= 2k rows (<= 32 KB, L1d) | 3.4-3.6 ns | 2.0-2.5 ns |
//! | 4k-8k rows (64-128 KB) | 5.3-6.1 ns | 2.6-2.7 ns |
//! | 32k rows, packed (512 KB) | 8.1-8.9 ns | 3.3-3.4 ns |
//! | 32k members scattered over a 64k tile | 11.4-11.9 ns | 4.1-4.3 ns |
//! | 64k rows (1 MB, one tile) | 12.9 ns | 5.2-5.5 ns |
//! | 256k rows (4 MB) | 28-30 ns | 8.2-8.9 ns |
//! | 4M rows (64 MB) | 160-162 ns | 50 ns |
//!
//! At the real 512-byte `NodeRow` stride, read in place through a strided view
//! (#1284: no second SoA copy), each row costs its own cache line (latency):
//! 64 rows 3.4 ns, 512 rows 6.6-7.4 ns, 4k-8k rows 26-27 ns, 32k rows 43-55 ns,
//! 64k rows 138-139 ns.
//!
//! Queries hit uniformly random members; HHTL-ordered access would be kinder.
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

/// Write a single random cycle (Sattolo) through `members` into bytes 12..16
/// of each member's record: the next row to visit. A dependent query reads
/// its next address out of the record it just loaded (a pointer chase), so
/// latency contains the load and nothing else.
fn link_cycle(store: &mut [u8], rec: usize, members: &[u32], seed: u64) {
    let mut order = members.to_vec();
    let mut x = seed;
    for i in (1..order.len()).rev() {
        x = mix(x);
        let j = (x as usize) % i; // Sattolo: j < i gives one cycle
        order.swap(i, j);
    }
    for k in 0..order.len() {
        let from = order[k] as usize;
        let to = order[(k + 1) % order.len()];
        store[from * rec + 12..from * rec + 16].copy_from_slice(&to.to_le_bytes());
    }
}

#[inline(always)]
fn half_hit(store: &[u8], o: usize, care: u64, pat: u64) -> u64 {
    let mut b = [0u8; 8];
    b[..6].copy_from_slice(&store[o + 4..o + 10]);
    u64::from((u64::from_le_bytes(b) ^ pat) & care == 0)
}

/// HHTL partial mask, vertically: one node's 6-byte tier path (HEEL/HIP/TWIG,
/// 3 x u8:u8 = half of the 12-byte facet) compared under a per-byte care mask.
/// Latency: follow the cycle `link_cycle` wrote (each next row comes from the
/// loaded record). Throughput: independent queries over the same members.
fn hhtl_half(
    store: &[u8],
    rec: usize,
    members: &[u32],
    care: u64,
    pat: u64,
    dependent: bool,
) -> f64 {
    let mut acc = 0u64;
    let t = Instant::now();
    if dependent {
        let mut r = members[0] as usize;
        for _ in 0..Q {
            let o = r * rec;
            acc = acc.wrapping_add(half_hit(store, o, care, pat));
            r = u32::from_le_bytes(store[o + 12..o + 16].try_into().unwrap()) as usize;
        }
    } else {
        assert!(members.len().is_power_of_two());
        for q in 0..Q as u64 {
            let r = members[(mix(q) as usize) & (members.len() - 1)] as usize;
            acc = acc.wrapping_add(half_hit(store, r * rec, care, pat));
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / Q as f64
}

/// The address generator alone, with no load: what the first version of this
/// probe folded into every "latency" figure.
fn mix_chain_ns() -> f64 {
    let mut x = 0x9E37_79B9u64;
    let t = Instant::now();
    for _ in 0..Q {
        x = mix(x);
    }
    black_box(x);
    t.elapsed().as_nanos() as f64 / Q as f64
}

/// `n` distinct rows drawn uniformly from `0..domain`, sorted.
fn scattered(n: usize, domain: usize, seed: u64) -> Vec<u32> {
    let mut all: Vec<u32> = (0..domain as u32).collect();
    let mut x = seed;
    for i in 0..n {
        x = mix(x);
        let j = i + (x as usize) % (domain - i);
        all.swap(i, j);
    }
    all.truncate(n);
    all.sort_unstable();
    all
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

    let r = |f: &dyn Fn() -> f64| median((0..5).map(|_| f()).collect());
    // HEEL + HIP (4 of the 6 path bytes) cared, TWIG free: a partial mask.
    let care = 0x0000_FFFF_FFFFu64;
    let pat = 0x0000_1234_5678u64;
    println!(
        "\nmix() chain alone, no load (the old latency loop's floor): {:.2}ns",
        r(&|| mix_chain_ns())
    );

    // HHTL half-length partial masks over a 16-byte-record store.
    let mut store: Vec<u8> = (0..(1usize << 22) * 16)
        .map(|i| mix(i as u64) as u8)
        .collect();
    println!("\nHHTL partial mask, 6-byte half path, per node (latency = pointer chase)");
    println!("{:>34} {:>11} {:>11}", "members", "latency", "thru");
    for (name, rows) in [
        ("1k rows (16 KB)", 1usize << 10),
        ("2k rows (32 KB = L1d)", 1 << 11),
        ("4k rows (64 KB)", 1 << 12),
        ("8k rows (128 KB)", 1 << 13),
        ("32k rows, packed (512 KB)", 1 << 15),
        ("1 tile (64k rows, 1 MB)", 1 << 16),
        ("4 tiles (256k rows, 4 MB)", 1 << 18),
        ("64 tiles (4M rows, 64 MB)", 1 << 22),
    ] {
        let members: Vec<u32> = (0..rows as u32).collect();
        link_cycle(&mut store, 16, &members, 0x5A77);
        let l = r(&|| hhtl_half(&store, 16, &members, care, pat, true));
        let t = r(&|| hhtl_half(&store, 16, &members, care, pat, false));
        println!("{name:>34} {l:>9.2}ns {t:>9.2}ns");
    }
    // Positive/negative selection caps a tile at 32k MEMBERS, but the members
    // keep their rows: scattered over the tile they still span its 1 MB.
    let members = scattered(1 << 15, 1 << 16, 0xC0DE);
    link_cycle(&mut store, 16, &members, 0x5A78);
    let l = r(&|| hhtl_half(&store, 16, &members, care, pat, true));
    let t = r(&|| hhtl_half(&store, 16, &members, care, pat, false));
    println!(
        "{:>34} {l:>9.2}ns {t:>9.2}ns",
        "32k members scattered in 64k tile"
    );

    // The same partial mask at the real NodeRow stride (512 B), read in place
    // through a strided view (#1284): no second SoA copy, but one cache line
    // per row touched.
    drop(store);
    let rows_max = 1usize << 18;
    let mut big: Vec<u8> = (0..rows_max * 512).map(|i| mix(i as u64) as u8).collect();
    println!("\nsame, 512-byte NodeRow stride (in place, one 64-byte line per row)");
    println!("{:>34} {:>11} {:>11}", "members", "latency", "thru");
    for (name, rows) in [
        ("64 rows (32 KB)", 1usize << 6),
        ("512 rows (256 KB)", 1 << 9),
        ("4k rows (2 MB)", 1 << 12),
        ("8k rows (4 MB)", 1 << 13),
        ("32k rows (16 MB)", 1 << 15),
        ("64k rows (32 MB)", 1 << 16),
        ("256k rows (128 MB)", 1 << 18),
    ] {
        let members: Vec<u32> = (0..rows as u32).collect();
        link_cycle(&mut big, 512, &members, 0x5A79);
        let l = r(&|| hhtl_half(&big, 512, &members, care, pat, true));
        let t = r(&|| hhtl_half(&big, 512, &members, care, pat, false));
        println!("{name:>34} {l:>9.2}ns {t:>9.2}ns");
    }
}
