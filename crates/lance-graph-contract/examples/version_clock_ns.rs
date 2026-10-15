//! D-WAL-6: cost of resolving a `TemporalPov` to a row range, per call, in ns.
//! Compared with a per-row predicate scan over the same rows (the shape of a
//! window that filters every row).
//!
//! `cargo run --release -p lance-graph-contract --example version_clock_ns`

use std::hint::black_box;
use std::time::Instant;

use lance_graph_contract::temporal_pov::{TemporalPov, VersionClock, VersionRange};

fn main() {
    println!(
        "tier: avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    let versions = 100_000u64;
    let rows_per_version = 4u64;
    let mut clock = VersionClock::new(0, 0);
    let mut row_version = Vec::with_capacity((versions * rows_per_version) as usize);
    for v in 1..=versions {
        for _ in 0..rows_per_version {
            row_version.push(v);
        }
        clock.record(v, row_version.len() as u64).unwrap();
    }
    // Pseudo-random reads so the branch predictor and cache see real spread.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let povs: Vec<TemporalPov> = (0..4096)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let a = x % versions;
            let b = a + (x >> 32) % 1000;
            TemporalPov::new(VersionRange::new(a, b), 0)
        })
        .collect();

    let iters = 20_000_000usize;
    let t = Instant::now();
    let mut acc = 0u64;
    for i in 0..iters {
        let r = clock.rows_at(black_box(&povs[i & 4095])).unwrap();
        acc = acc.wrapping_add(r.end - r.start);
    }
    let clock_ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
    black_box(acc);

    let scans = 200usize;
    let t = Instant::now();
    let mut acc = 0u64;
    for i in 0..scans {
        let pov = black_box(&povs[i & 4095]);
        acc += row_version.iter().filter(|&&v| pov.admits(v)).count() as u64;
    }
    let scan_ns = t.elapsed().as_secs_f64() * 1e9 / scans as f64;
    black_box(acc);

    println!(
        "rows {} versions {}  clock.rows_at: {clock_ns:.2} ns/call  per-row scan: {:.0} ns/call ({:.2} ns/row)  clock memory: {} B",
        row_version.len(),
        versions,
        scan_ns,
        scan_ns / row_version.len() as f64,
        (versions + 1) * 8
    );
}
