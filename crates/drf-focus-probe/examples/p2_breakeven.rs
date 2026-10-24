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

fn main() {
    let isa = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "scalar"
    };
    println!("isa={isa} slots={SLOTS} row={ROW}B facet=i32@{OFF}");
    println!(
        "{:<10} {:>7} {:<10} {:>6} {:>11} {:>9} {:>11} {:>7} {:>12}",
        "rows", "MiB", "map", "dens", "build_ns", "fold_ns", "direct_ns", "N*", "A2@N*"
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
                let nstar = if direct > fold {
                    format!("{}", (build / (direct - fold)).ceil() as u64)
                } else {
                    "never".into()
                };
                let a2 = match nstar.parse::<f64>() {
                    Ok(n) => format!("{:.3}", n * pop as f64 / (n * pop as f64 + SLOTS as f64)),
                    Err(_) => "-".into(),
                };
                println!(
                    "{:<10} {:>7.1} {:<10} {:>6.2} {:>11.0} {:>9.0} {:>11.0} {:>7} {:>12}",
                    nrows,
                    (nrows * ROW) as f64 / (1 << 20) as f64,
                    mname,
                    pop as f64 / SLOTS as f64,
                    build,
                    fold,
                    direct,
                    nstar,
                    a2
                );
            }
        }
    }
}
