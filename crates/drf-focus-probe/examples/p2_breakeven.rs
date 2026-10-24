//! D-DRF-P2 break-even: focus (gather once + N folds) vs direct (N in-place reads).
//!
//! Reported, not gated. One focus = 4096 slots; every timing is per call, median of
//! repeated runs, in ns. The crossover N is the smallest fold count at which the
//! focus total (build + N*fold) is below N*direct.
//!   cargo run --release --example p2_breakeven
use drf_focus_probe::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const OFF: usize = 16;
type MapGen = Box<dyn Fn(usize, &mut u64) -> u16>;

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

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

/// Stream a buffer twice the size of the last-level cache, outside the timed
/// region, so the rows a cold rep touches come from DRAM.
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

/// Smallest N with focus (build + N folds) below direct (first read at `d0`,
/// later reads at `d`). `None` if it never crosses within 10 000.
fn crossover(build: f64, fold: f64, d0: f64, d: f64) -> Option<u64> {
    (1..=10_000u64).find(|&n| build + n as f64 * fold < d0 + (n - 1) as f64 * d)
}

fn main() {
    // 512 MiB: twice the 260 MiB L3 on the measuring host.
    let mut evict_buf = vec![0u64; 64 << 20];
    let isa = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "scalar"
    };
    println!("isa={isa} slots={SLOTS} row={ROW}B facet=i32@{OFF}");
    println!("warm = repeated reps, rows stay cached; cold = 512 MiB evicted before each rep");
    println!(
        "{:<6} {:>8} {:<10} {:>5} {:>9} {:>7} {:>9} {:>9} {:>11} {:>6} {:>6} {:>7}",
        "rows",
        "lineKiB",
        "map",
        "dens",
        "build_w",
        "fold_w",
        "direct_w",
        "build_c",
        "direct_c",
        "N*_w",
        "N*_c",
        "A2@N*c"
    );
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    for nrows in [1024usize, 65536] {
        let rows: Vec<Row> = (0..nrows)
            .map(|i| {
                let mut r = [0u8; ROW];
                r[OFF..OFF + 4].copy_from_slice(&((i as i32) * 3 - 7).to_le_bytes());
                r
            })
            .collect();
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
        let maps: [(&str, MapGen); 2] = [
            (
                "random",
                Box::new(move |_, s| (xs(s) % nrows as u64) as u16),
            ),
            ("clustered", Box::new(move |i, _| (i % nrows) as u16)),
        ];
        for (id, (mname, make)) in maps.iter().enumerate() {
            let mut ord = Box::new([0u16; SLOTS]);
            for (i, o) in ord.iter_mut().enumerate() {
                *o = make(i, &mut st);
            }
            let map = Arc::new(SealedMap::new(MapId(id as u64), 7, ord));
            for dens in [1.0f64, 0.25] {
                let mut mask = [0u64; TILE_WORDS];
                for w in mask.iter_mut() {
                    *w = if dens >= 1.0 {
                        u64::MAX
                    } else {
                        xs(&mut st) & xs(&mut st)
                    };
                }
                let pop: u32 = mask.iter().map(|w| w.count_ones()).sum();
                // build = a cold manager per call, so every rep gathers.
                let build = median_ns(201, || {
                    let mgr = FocusManager::new();
                    black_box(mgr.load(&soa, &map, Scope(1), facet).unwrap());
                });
                let mgr = FocusManager::new();
                let focus = mgr.load(&soa, &map, Scope(1), facet).unwrap();
                let fold = median_ns(2001, || {
                    black_box(focus.sum_where(black_box(&mask)));
                });
                let direct = median_ns(201, || {
                    black_box(direct_sum_where(&soa, &map, facet, black_box(&mask), 16).unwrap());
                });
                // Same answer on both paths, or the timing is meaningless.
                assert_eq!(
                    focus.sum_where(&mask),
                    direct_sum_where(&soa, &map, facet, &mask, 16).unwrap()
                );
                let build_c = median_cold_ns(21, &mut evict_buf, || {
                    let mgr = FocusManager::new();
                    black_box(mgr.load(&soa, &map, Scope(1), facet).unwrap());
                });
                let direct_c = median_cold_ns(21, &mut evict_buf, || {
                    black_box(direct_sum_where(&soa, &map, facet, black_box(&mask), 16).unwrap());
                });
                let show = |n: Option<u64>| n.map_or("never".to_string(), |n| n.to_string());
                let nw = crossover(build, fold, direct, direct);
                let nc = crossover(build_c, fold, direct_c, direct);
                let a2 = nc.map_or("-".into(), |n| {
                    let r = n as f64 * pop as f64;
                    format!("{:.3}", r / (r + SLOTS as f64))
                });
                let mut seen = vec![false; nrows];
                let lines = map
                    .ord
                    .iter()
                    .filter(|&&o| !std::mem::replace(&mut seen[o as usize], true))
                    .count();
                println!(
                    "{:<6} {:>8} {:<10} {:>5.2} {:>9.0} {:>7.0} {:>9.0} {:>9.0} {:>11.0} {:>6} {:>6} {:>7}",
                    nrows,
                    lines * 64 / 1024,
                    mname,
                    pop as f64 / SLOTS as f64,
                    build,
                    fold,
                    direct,
                    build_c,
                    direct_c,
                    show(nw),
                    show(nc),
                    a2
                );
            }
        }
    }
}
