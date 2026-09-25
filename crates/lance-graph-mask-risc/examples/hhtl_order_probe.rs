//! Does HHTL-ordered access beat random access for a vertical partial mask?
//!
//! Same query as `mask_cache_hit_probe`: one node's 6-byte tier path (half the
//! facet) compared under a care mask. What changes is the ORDER the member
//! nodes are visited in:
//!
//! - `random`: a uniformly random permutation of the members;
//! - `ordered`: ascending address order, which is what a prefix walk over a
//!   sealed (address-ordered) lane produces;
//! - `clustered`: 256-row blocks (one HEEL.hi cell) visited in random order,
//!   rows in address order inside each block — a frontier that finishes one
//!   subtree before it jumps.
//!
//! Every case visits max(Q, members) nodes, so the whole cycle is walked.
//! Latency is a pointer chase: bytes 12..16 of each member's record hold the
//! next row in the chosen order, so nothing but the load is on the chain.
//! Throughput walks the same order from a list drawn before the clock starts.
//!
//! Measured 2026-09-25 (Xeon @ 2.8 GHz, median of 5, 3 runs), ns per node:
//!
//! | members (16-byte records) | random lat | ordered lat | clustered lat |
//! |---|---|---|---|
//! | 32k rows, packed (512 KB) | 7.8-8.7 | 3.2-3.3 | 3.2-3.3 |
//! | 32k scattered over a 64k tile | 12.3-16.5 | 4.2-4.6 | 4.3-4.5 |
//! | one tile (64k, 1 MB) | 13.1-14.6 | 3.2-3.3 | 3.3-3.5 |
//! | four tiles (256k, 4 MB) | 28.5-30.5 (one run 84) | 3.2-3.4 | 3.4 |
//! | 64 tiles (4M, 64 MB), walked end to end | 157-163 | 3.8-3.9 | 4.4-4.6 |
//! | 256k scattered over 4M (1/16) | 138-151 | 77-91 | 99-105 |
//!
//! | members (512-byte `NodeRow`, in place) | random lat | ordered lat | clustered lat |
//! |---|---|---|---|
//! | 4k rows (2 MB span) | 26-28 | 14.0-14.6 | 14.1-16.7 |
//! | 32k rows (16 MB span) | 65-125 | 22-24 | 18-27 |
//! | 64k rows (32 MB span) | 125-152 | 57-71 | 54-70 |
//! | 4k scattered over 64k | 40-42 | 36-40 | 36-41 |
//!
//! Ordered access over the packed key lane stays at ~3-4 ns per node at every
//! working-set size up to 64 MB: the hardware prefetcher streams it. Walking
//! 256-row subtrees in random order costs almost nothing extra. What defeats
//! the prefetcher is sparsity: at 1 member per 16 rows (4 cache lines apart)
//! the ordered walk is still ~80-90 ns, and on the 512-byte stride a row is 8
//! lines from the next, so ordered access only roughly halves the cost.
//!
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example hhtl_order_probe`

use std::hint::black_box;
use std::time::Instant;

const Q: usize = 1 << 20;

fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    (x ^ (x >> 29)).wrapping_mul(0xc4ce_b9fe_1a85_ec53)
}

fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut x = seed;
    for i in (1..v.len()).rev() {
        x = mix(x);
        v.swap(i, (x as usize) % (i + 1));
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Order {
    Random,
    Ordered,
    Clustered,
}

/// The visit order over sorted `members`.
fn visit_order(members: &[u32], order: Order, seed: u64) -> Vec<u32> {
    let mut v = members.to_vec();
    match order {
        Order::Random => shuffle(&mut v, seed),
        Order::Ordered => {}
        Order::Clustered => {
            let mut blocks: Vec<&[u32]> = v.chunk_by(|a, b| a >> 8 == b >> 8).collect();
            shuffle(&mut blocks, seed);
            v = blocks.concat();
        }
    }
    v
}

/// Link `order` into one cycle through bytes 12..16 of each record.
fn link(store: &mut [u8], rec: usize, order: &[u32]) {
    for k in 0..order.len() {
        let from = order[k] as usize * rec;
        let to = order[(k + 1) % order.len()];
        store[from + 12..from + 16].copy_from_slice(&to.to_le_bytes());
    }
}

#[inline(always)]
fn half_hit(store: &[u8], o: usize) -> u64 {
    const CARE: u64 = 0x0000_FFFF_FFFF;
    const PAT: u64 = 0x0000_1234_5678;
    let mut b = [0u8; 8];
    b[..6].copy_from_slice(&store[o + 4..o + 10]);
    u64::from((u64::from_le_bytes(b) ^ PAT) & CARE == 0)
}

fn latency(store: &[u8], rec: usize, start: u32, visits: usize) -> f64 {
    let (mut r, mut acc) = (start as usize, 0u64);
    let t = Instant::now();
    for _ in 0..visits {
        let o = r * rec;
        acc = acc.wrapping_add(half_hit(store, o));
        r = u32::from_le_bytes(store[o + 12..o + 16].try_into().unwrap()) as usize;
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / visits as f64
}

fn throughput(store: &[u8], rec: usize, queries: &[u32]) -> f64 {
    let mut acc = 0u64;
    let t = Instant::now();
    for &r in queries {
        acc = acc.wrapping_add(half_hit(store, r as usize * rec));
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / queries.len() as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// `n` distinct rows of `0..domain`, sorted.
fn members(n: usize, domain: usize, seed: u64) -> Vec<u32> {
    if n == domain {
        return (0..domain as u32).collect();
    }
    let mut all: Vec<u32> = (0..domain as u32).collect();
    shuffle(&mut all, seed);
    all.truncate(n);
    all.sort_unstable();
    all
}

fn run(store: &mut [u8], rec: usize, label: &str, m: &[u32]) {
    let mut row = format!("{label:>40}");
    for order in [Order::Random, Order::Ordered, Order::Clustered] {
        let o = visit_order(m, order, 0x5A77);
        link(store, rec, &o);
        // At least Q visits, and never fewer than the whole cycle: a case with
        // more members than Q must be walked end to end, or a big working set
        // is measured by its first Q members only.
        let visits = Q.max(o.len());
        let qs: Vec<u32> = (0..visits).map(|i| o[i % o.len()]).collect();
        let l = median((0..5).map(|_| latency(store, rec, o[0], visits)).collect());
        let t = median((0..5).map(|_| throughput(store, rec, &qs)).collect());
        row += &format!(" {:>7.2} {:>6.2}", l, t);
    }
    println!("{row}");
}

fn header(title: &str) {
    println!("\n{title}");
    println!(
        "{:>40} {:>14} {:>14} {:>14}",
        "", "random", "ordered", "clustered"
    );
    println!(
        "{:>40} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6}",
        "members", "lat", "thru", "lat", "thru", "lat", "thru"
    );
}

fn main() {
    println!("ns per node, median of 5; lat = pointer chase, thru = independent");
    let max_rows = 1usize << 22;
    let mut store: Vec<u8> = (0..max_rows * 16).map(|i| mix(i as u64) as u8).collect();
    header("16-byte records (packed key lane)");
    for (label, n, domain) in [
        ("2k rows (L1d)", 1usize << 11, 1usize << 11),
        ("32k rows, packed (512 KB)", 1 << 15, 1 << 15),
        ("32k scattered over a 64k tile", 1 << 15, 1 << 16),
        ("4k scattered over a 64k tile", 1 << 12, 1 << 16),
        ("one tile (64k, 1 MB)", 1 << 16, 1 << 16),
        ("four tiles (256k, 4 MB)", 1 << 18, 1 << 18),
        ("64 tiles (4M, 64 MB)", 1 << 22, 1 << 22),
        ("256k scattered over 4M (1/16)", 1 << 18, 1 << 22),
    ] {
        run(
            &mut store,
            16,
            label,
            &members(n, domain, 0xC0DE ^ n as u64),
        );
    }
    drop(store);

    let rows = 1usize << 16;
    let mut big: Vec<u8> = (0..rows * 512).map(|i| mix(i as u64) as u8).collect();
    header("512-byte NodeRow stride, read in place (one line per row)");
    for (label, n, domain) in [
        ("512 rows (256 KB span)", 1usize << 9, 1usize << 9),
        ("4k rows (2 MB span)", 1 << 12, 1 << 12),
        ("32k rows (16 MB span)", 1 << 15, 1 << 15),
        ("64k rows (32 MB span)", 1 << 16, 1 << 16),
        ("4k scattered over 64k", 1 << 12, 1 << 16),
    ] {
        run(&mut big, 512, label, &members(n, domain, 0xC0DE ^ n as u64));
    }
}
