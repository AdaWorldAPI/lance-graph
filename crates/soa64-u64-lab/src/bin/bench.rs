//! Timing harness: ns and µs only. Every timed result is checked against the
//! oracle first; speed is reported only for answers that are right.
//!
//! The allocation column counts every heap allocation made during the warm
//! samples (a counting wrapper over the system allocator, the only `unsafe` in
//! the crate). The sample buffers are allocated before the counter is read.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call unchanged to `System`; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;
use std::time::Instant;

use soa64_u64_lab::arms::{
    demanded_fields, Aos, Arm, HashRow, Query, SoaBitpack, SoaNarrow, SoaU64,
};
use soa64_u64_lab::fixture::{probes, Fixture, G2L};
use soa64_u64_lab::oracle::oracle;

const SEED: u64 = 0x5EED_0064;
const SIZES: [usize; 5] = [256, 513, 4096, 65536, 262144];
const WARM_SAMPLES: usize = 31;
const COLD_SAMPLES: usize = 11;
const POINT_PROBES: usize = 1024;
const SCRATCH_BYTES: usize = 64 << 20;

fn median(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn us_since(t: Instant) -> f64 {
    t.elapsed().as_nanos() as f64 / 1000.0
}

/// Stream through a buffer larger than the last-level cache. The buffer is
/// filled with non-zero data so the pages are real, not the shared zero page.
fn evict(scratch: &[u64]) {
    let mut s = 0u64;
    for &x in scratch {
        s = s.wrapping_add(x);
    }
    black_box(s);
}

/// Analytic cache lines touched by `q` on `arm` (64-byte lines).
fn lines_touched(arm: &dyn Arm, q: &Query<'_>, n: usize) -> u64 {
    let fields = demanded_fields(q);
    let is_point = matches!(q, Query::Point { .. });
    let probes = match q {
        Query::Point { probes } => probes.len() as u64,
        _ => 0,
    };
    if arm.storage_bits(0).is_some() {
        if is_point {
            // Upper bound: one line per probe per demanded lane.
            return probes * fields.len() as u64;
        }
        let mut lines: u64 = fields
            .iter()
            .map(|&f| {
                let bits = u64::from(arm.storage_bits(f).unwrap_or(64));
                (n as u64 * bits).div_ceil(8).div_ceil(64)
            })
            .sum();
        if matches!(q, Query::SumPresentC) {
            lines += (n as u64).div_ceil(8).div_ceil(64);
        }
        lines
    } else if is_point {
        probes
    } else {
        // 512-byte rows: 8 lines each.
        n as u64 * 512 / 64
    }
}

fn env_line() {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "env: cpu={cpu} avx2={} avx512f={} rustc={}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f"),
        option_env!("RUSTC_VERSION").unwrap_or("unknown"),
    );
}

fn run_size(n: usize, scratch: &[u64]) {
    println!("\n## n = {n}\n");
    let fx = Fixture::new(SEED, n);
    let pr = probes(SEED, n, POINT_PROBES);

    let t = Instant::now();
    let aos = Aos::build(&fx);
    let ti_aos = us_since(t);
    let t = Instant::now();
    let soa = SoaU64::build(&fx);
    let ti_soa = us_since(t);
    let t = Instant::now();
    let narrow = SoaNarrow::build(&fx);
    let ti_narrow = us_since(t);
    let t = Instant::now();
    let pack = SoaBitpack::build(&fx);
    let ti_pack = us_since(t);
    let t = Instant::now();
    let hash = HashRow::build(&fx);
    let ti_hash = us_since(t);

    let arms: [(&dyn Arm, f64); 5] = [
        (&aos, ti_aos),
        (&soa, ti_soa),
        (&narrow, ti_narrow),
        (&pack, ti_pack),
        (&hash, ti_hash),
    ];

    println!("| arm | ingress µs | bytes resident |");
    println!("|---|---|---|");
    for (arm, ti) in &arms {
        println!("| {} | {:.1} | {} |", arm.name(), ti, arm.bytes_resident());
    }
    println!();

    let mut queries: Vec<(String, Query<'_>)> = vec![
        ("CountWhere".to_string(), Query::CountWhere),
        ("SumWhere".to_string(), Query::SumWhere),
        ("SumPresentC".to_string(), Query::SumPresentC),
    ];
    for d in [1usize, 3, 8, 32, 64] {
        queries.push((format!("DemandFold d={d}"), Query::DemandFold { d }));
    }
    queries.push((
        format!("Point {POINT_PROBES}"),
        Query::Point { probes: &pr },
    ));
    let expected: Vec<u128> = queries.iter().map(|(_, q)| oracle(&fx, q)).collect();

    println!("| arm | query | warm ns/id | cold ns/id | warm µs total | allocs | lines touched |");
    println!("|---|---|---|---|---|---|---|");
    for (arm, _) in &arms {
        for ((label, q), exp) in queries.iter().zip(&expected) {
            let got = arm.run(q);
            assert_eq!(got, *exp, "arm {} query {} n {}", arm.name(), label, n);

            black_box(arm.run(black_box(q)));
            let mut warm: Vec<u64> = Vec::with_capacity(WARM_SAMPLES);
            let allocs_before = ALLOCS.load(Ordering::Relaxed);
            for _ in 0..WARM_SAMPLES {
                let t = Instant::now();
                black_box(arm.run(black_box(q)));
                warm.push(t.elapsed().as_nanos() as u64);
            }
            let allocs = ALLOCS.load(Ordering::Relaxed) - allocs_before;
            let mut cold: Vec<u64> = Vec::with_capacity(COLD_SAMPLES);
            for _ in 0..COLD_SAMPLES {
                evict(scratch);
                let t = Instant::now();
                black_box(arm.run(black_box(q)));
                cold.push(t.elapsed().as_nanos() as u64);
            }
            let warm_med = median(&mut warm);
            let cold_med = median(&mut cold);
            let ids = (if matches!(q, Query::Point { .. }) {
                POINT_PROBES
            } else {
                n
            }) as f64;
            println!(
                "| {} | {} | {:.3} | {:.3} | {:.2} | {} | {} |",
                arm.name(),
                label,
                warm_med as f64 / ids,
                cold_med as f64 / ids,
                warm_med as f64 / 1000.0,
                allocs,
                lines_touched(*arm, q, n),
            );
        }
    }

    // Identity lookups: ingress goes through G2L once; the positional sweep
    // takes none, the per-element-identity null takes one per row.
    let g2l = G2L::new(&fx);
    for &g in &fx.guids {
        black_box(g2l.lookup(g));
    }
    let ingress_lookups = g2l.lookups.get();
    let before = g2l.lookups.get();
    black_box(soa.run(&Query::SumWhere));
    let soa_sweep = g2l.lookups.get() - before;
    hash.reset_lookups();
    black_box(hash.run(&Query::SumWhere));
    let hash_sweep = hash.lookups();
    println!(
        "\nidentity lookups: ingress via G2L = {ingress_lookups}; \
         SumWhere sweep: soa-u64 = {soa_sweep}, hash-row = {hash_sweep}"
    );
}

fn main() {
    env_line();
    let scratch: Vec<u64> = (0..SCRATCH_BYTES / 8).map(|i| i as u64 | 1).collect();
    for n in SIZES {
        run_size(n, &scratch);
    }
}
