//! D-DRF-P2 follow-up: does the 65 536-row build cost track distinct 4 KiB pages touched?
//!
//! `p2_breakeven` measured the warm gather at 3.8 µs for 1 024 rows and 23 µs for 65 536 rows
//! with the same 4 096 slots. This probe holds the slot count fixed, builds maps that touch a
//! chosen number P of distinct pages, and reports the gather cost against P. Reported, not gated.
//!
//! PRE-REGISTERED (before the first run), per build on a cold manager, warm caches:
//!   H_TLB   build ns rises with P once P passes the second-level TLB reach (about 1.5-2 K
//!           entries on this class of host), so the P = 4096 build is clearly above the
//!           P = 512 build, even though both touch 4096 distinct cache lines.
//!   H_other build ns is flat in P over 512..4096 (cache lines, not pages, set the cost).
//! Within P >= 512 the distinct cache-line count is constant at 4096, so that range isolates
//! the page effect. Below 512 the line count also falls; those rows are shown but do not
//! decide. Page counts are computed from the real row addresses, not assumed from the row size.
//!   cargo run --release --example tlb_probe
use drf_focus_probe::*;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const OFF: usize = 16;
const NROWS: usize = 65536;
const PAGE: usize = 4096;

fn median_ns(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[reps / 2]
}

fn evict(buf: &mut [u64]) {
    let mut acc = 0u64;
    for (i, v) in buf.iter_mut().enumerate() {
        *v = v.wrapping_add(i as u64);
        acc ^= *v;
    }
    black_box(acc);
}

fn median_cold_ns(reps: usize, buf: &mut [u64], mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            evict(buf);
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[reps / 2]
}

/// Kernel page size and huge-page bytes for the mapping holding `addr`, read from smaps. Policy
/// (`transparent_hugepage/enabled`) does not say how a given buffer is actually backed.
fn backing(addr: usize) -> String {
    let Ok(text) = std::fs::read_to_string("/proc/self/smaps") else {
        return "smaps unreadable".into();
    };
    let mut hit = false;
    let mut out = String::from("mapping not found");
    for line in text.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((lo, hi)) = range.split_once('-')
            && let (Ok(lo), Ok(hi)) = (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16))
        {
            hit = (lo..hi).contains(&addr);
            continue;
        }
        if hit && (line.starts_with("KernelPageSize") || line.starts_with("AnonHugePages")) {
            if out == "mapping not found" {
                out.clear();
            }
            out.push_str(&line.split_whitespace().collect::<Vec<_>>().join(" "));
            out.push_str("; ");
        }
    }
    out
}

fn main() {
    let thp = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
        .unwrap_or_else(|_| "unreadable".into());
    println!("transparent_hugepage: {}", thp.trim());
    let mut evict_buf = vec![0u64; 64 << 20];
    let rows: Vec<Row> = (0..NROWS)
        .map(|i| {
            let mut r = [0u8; ROW];
            r[OFF..OFF + 4].copy_from_slice(&((i as i32) * 3 - 7).to_le_bytes());
            r
        })
        .collect();
    println!("row buffer backing: {}", backing(rows.as_ptr() as usize));
    let soa = Soa {
        rows: &rows,
        id: SourceId {
            dataset: 1,
            branch: 1,
            version: 1,
            index_seal: 7,
        },
    };
    let facet = Facet::new(OFF).unwrap();
    println!(
        "slots={SLOTS} rows={NROWS} ({} MiB) facet=i32@{OFF}",
        (NROWS * ROW) >> 20
    );
    println!(
        "{:>6} {:>8} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "P_req", "pages", "lines", "build_w", "build_c", "ns/slot_w", "ns/page_w"
    );
    // Two generators over the same P. "skewed": slot i reads row (i % P) * 8 + (i / P) % 8, so the
    // number of distinct within-page offsets shrinks as P grows (8 at P=512, 1 at P=4096) and
    // cache-set coverage changes with P. "balanced": the offset is rotated by page, so every
    // offset is used and the offsets-per-page spread no longer depends on P.
    for arm in ["skewed", "balanced", "shuffled"] {
        println!("-- {arm}");
        println!(
            "{:>6} {:>7} {:>7} {:>5} {:>9} {:>9} {:>9}",
            "P_req", "pages", "lines", "offs", "build_w", "build_c", "ns/slot_w"
        );
        for p_req in [64usize, 128, 256, 512, 1024, 1536, 2048, 3072, 4096] {
            let mut ord = Box::new([0u16; SLOTS]);
            for (i, o) in ord.iter_mut().enumerate() {
                let (page, visit) = (i % p_req, i / p_req);
                let off = if arm == "skewed" {
                    visit % 8
                } else {
                    (visit + page % 8) % 8
                };
                *o = ((page * 8 + off) % NROWS) as u16;
            }
            if arm == "shuffled" {
                // Same set of ordinals as `balanced`, seeded Fisher-Yates visit order.
                let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ p_req as u64;
                for i in (1..SLOTS).rev() {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ord.swap(i, (x % (i as u64 + 1)) as usize);
                }
            }
            let addr = |o: u16| rows.as_ptr() as usize + o as usize * ROW + OFF;
            let pages: HashSet<usize> = ord.iter().map(|&o| addr(o) / PAGE).collect();
            let lines: HashSet<usize> = ord.iter().map(|&o| addr(o) / 64).collect();
            let offs: HashSet<usize> = ord.iter().map(|&o| (addr(o) % PAGE) / 64).collect();
            let mut hist = [0usize; 8];
            for &o in ord.iter() {
                hist[(addr(o) % PAGE) / 512] += 1;
            }
            let map = Arc::new(SealedMap::new(MapId(p_req as u64), 7, ord));
            let warm = median_ns(401, || {
                let mgr = FocusManager::new();
                black_box(mgr.load(&soa, &map, Scope(1), facet).unwrap());
            });
            let cold = median_cold_ns(21, &mut evict_buf, || {
                let mgr = FocusManager::new();
                black_box(mgr.load(&soa, &map, Scope(1), facet).unwrap());
            });
            println!(
                "{:>6} {:>7} {:>7} {:>5} {:>9.0} {:>9.0} {:>9.2}",
                p_req,
                pages.len(),
                lines.len(),
                offs.len(),
                warm,
                cold,
                warm / SLOTS as f64
            );
            if p_req == 4096 || p_req == 512 {
                println!("       offset-bin histogram (8 x 512 B): {hist:?}");
            }
        }
    }
}
