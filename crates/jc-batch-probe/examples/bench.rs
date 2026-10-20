//! U1 benchmark: one statistical battery answered request by request versus
//! from shared power-sum passes. Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example bench
//! ```
//!
//! `BENCH_N` rows (default 1,000,000), 7 `I32` statistic lanes, a selection
//! lane, a category lane, an 8-value group lane and a zero lane. The battery
//! is the one of `tests/batch.rs` over two selections: every mean, variance
//! and ANOVA, and every covariance, Pearson, R² and regression line (both
//! directions) of every pair: 252 requests. Both paths must return equal
//! answers before either is timed. Times are medians of `BENCH_REPS`
//! (default 5) warm runs.

use std::hint::black_box;
use std::time::{Duration, Instant};

use jc_batch_probe::{answer_batched, answer_separately, Request, Stat, Table};
use lance_graph_mask_risc::LaneRef;
use lance_graph_relalg::{Cmp, Rel};

const STAT_LANES: u16 = 7;
const SEL: u16 = 7;
const CAT: u16 = 8;
const GROUP: u16 = 9;
const ZERO: u16 = 10;
const GROUPS: usize = 8;
const VERSION: u64 = 1;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn battery(mut select: impl FnMut() -> Rel) -> Vec<Request> {
    let mut r = Vec::new();
    let mut push = |stat| {
        r.push(Request {
            version: VERSION,
            select: select(),
            stat,
        })
    };
    for x in 0..STAT_LANES {
        push(Stat::Mean(x));
        push(Stat::Variance(x));
        push(Stat::Anova(x));
    }
    for x in 0..STAT_LANES {
        for y in (x + 1)..STAT_LANES {
            push(Stat::Covariance(x, y));
            push(Stat::Pearson(x, y));
            push(Stat::RSquared(x, y));
            push(Stat::Regression(x, y));
            push(Stat::Regression(y, x));
        }
    }
    r
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let n = env("BENCH_N", 1_000_000);
    let reps = env("BENCH_REPS", 5);
    let mut s = 0xBE7C_u64;
    let base: Vec<i32> = (0..n)
        .map(|_| (lcg(&mut s) % 2_001) as i32 - 1_000)
        .collect();
    let stats: Vec<Vec<i32>> = (0..STAT_LANES as i32)
        .map(|j| {
            base.iter()
                .map(|&b| b * j + (lcg(&mut s) % 401) as i32 - 200)
                .collect()
        })
        .collect();
    let sel: Vec<i32> = (0..n).map(|_| (lcg(&mut s) % 100) as i32).collect();
    let cat: Vec<u32> = (0..n).map(|_| (lcg(&mut s) % 4) as u32).collect();
    let group: Vec<u32> = (0..n)
        .map(|_| (lcg(&mut s) % GROUPS as u64) as u32)
        .collect();
    let zero = vec![0u32; n];
    let mut lanes: Vec<LaneRef<'_>> = stats.iter().map(|v| LaneRef::I32(v)).collect();
    lanes.extend([
        LaneRef::I32(&sel),
        LaneRef::U32(&cat),
        LaneRef::U32(&group),
        LaneRef::U32(&zero),
    ]);
    let t = Table {
        version: VERSION,
        n_rows: n,
        lanes: &lanes,
        group: GROUP,
        groups: GROUPS,
        zero: ZERO,
    };

    let low = || Rel::i32(SEL, Cmp::Lt, 60);
    let mut reqs = battery(low);
    reqs.extend(battery(|| Rel::And(vec![low(), Rel::eq_u32(CAT, 2)])));

    let (a_sep, c_sep) = answer_separately(&t, &reqs).unwrap();
    let (a_bat, c_bat, red) = answer_batched(&t, &reqs).unwrap();
    assert_eq!(a_sep, a_bat, "the two paths must agree before timing");

    let time = |f: &dyn Fn()| {
        let v: Vec<Duration> = (0..reps)
            .map(|_| {
                let st = Instant::now();
                f();
                st.elapsed()
            })
            .collect();
        median(v)
    };
    let t_sep = time(&|| {
        black_box(answer_separately(&t, &reqs).unwrap());
    });
    let t_bat = time(&|| {
        black_box(answer_batched(&t, &reqs).unwrap());
    });

    println!("rows {n}, requests {}, {red:?}", reqs.len());
    println!(
        "{:>10} {:>9} {:>9} {:>9} {:>14} {:>14} {:>12}",
        "path", "ms", "programs", "sum pass", "rows trav.", "values read", "mask bytes"
    );
    for (name, ms, c) in [("separate", t_sep, c_sep), ("batched", t_bat, c_bat)] {
        println!(
            "{:>10} {:>9.1} {:>9} {:>9} {:>14} {:>14} {:>12}",
            name,
            ms.as_secs_f64() * 1e3,
            c.programs,
            c.sum_passes,
            c.rows_traversed,
            c.values_read,
            c.mask_bytes
        );
    }
    println!(
        "ratio separate/batched: time {:.1}x, rows {:.1}x, values {:.1}x, masks {:.1}x",
        t_sep.as_secs_f64() / t_bat.as_secs_f64(),
        c_sep.rows_traversed as f64 / c_bat.rows_traversed as f64,
        c_sep.values_read as f64 / c_bat.values_read as f64,
        c_sep.mask_bytes as f64 / c_bat.mask_bytes as f64
    );
}
