//! D-TJ-3: a bound predicate narrows the ADDRESS RANGE, not just the result.
//!
//! An adjacency table stored in timestamp order (the JanusGraph sort-key
//! slice, as a local SoA lane). Query: `ts ∈ [a, b)` → `count`, `sum(w)`.
//!
//! - `sweep` : `GeI32(a) ∧ LtI32(b)` over every row (`quack::lower`) — what a
//!   planner without ordering evidence must emit;
//! - `slice` : `partition_point` on the ordered lane gives `[lo, hi)`, lowered
//!   as `Cmp::Range { lo, hi }` — no row is compared. The binary search is
//!   INSIDE the timed region (it is the bind/bundle-time cost a prepared plan
//!   would pay per parameter value).
//!
//! The lane is sorted by construction here; in production that ordering must
//! be ATTESTED (`ordered_lane::OrderedLaneWitness` does this for facet keys —
//! there is no witness type for an ordered scalar property yet).
//!
//! ```bash
//! cargo run --release -p lance-graph-benches --example bounded_slice_probe
//! ```

use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};

const TS: Col = Col(0);
const W: Col = Col(1);
const LIVE: Mask = Mask(0);

fn median_us(reps: usize, mut f: impl FnMut() -> (u64, i64)) -> (f64, (u64, i64)) {
    let mut last = (0, 0);
    let mut v: Vec<u128> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            last = std::hint::black_box(f());
            t0.elapsed().as_nanos()
        })
        .collect();
    v.sort_unstable();
    (v[v.len() / 2] as f64 / 1e3, last)
}

fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        *w.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
    }
    w
}

fn run(p: &Program, ts: &[i32], w: &[i32], live: &[u64]) -> i64 {
    let lanes = [LaneRef::I32(ts), LaneRef::I32(w)];
    let masks: [&[u64]; 1] = [live];
    let planes = Planes {
        n_rows: ts.len(),
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(p, ts.len()).expect("scratch");
    match execute_into(p, &planes, &Foreign::NONE, &mut scratch, Out::None).expect("executes") {
        Value::Count(c) => c as i64,
        Value::SumI64(s) => s,
        v => panic!("{v:?}"),
    }
}

fn main() {
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31);
    println!("rows\tselectivity\tmatches\tsweep_us\tslice_us\tspeedup");
    for n in [100_000usize, 1_000_000] {
        // non-decreasing timestamps with duplicates (several edges per tick)
        let ts: Vec<i32> = (0..n as i32).map(|i| i / 3).collect();
        let w: Vec<i32> = (0..n as i32).map(|i| (i * 7919) % 101).collect();
        let live = ones(n);
        let span = ts[n - 1] + 1;
        for sel in [0.5, 0.1, 0.01, 0.001] {
            let a = span / 3;
            let b = a + ((span as f64 * sel) as i32).max(1);
            let want: (u64, i64) = ts
                .iter()
                .zip(&w)
                .filter(|(&t, _)| t >= a && t < b)
                .fold((0, 0), |(c, s), (_, &x)| (c + 1, s + x as i64));
            let sweep_f = Filter::and([
                Filter::plane(LIVE),
                Filter::cmp(TS, Cmp::GeI32(a)),
                Filter::cmp(TS, Cmp::LtI32(b)),
            ]);
            let sweep_c = lower(&Query {
                filter: sweep_f.clone(),
                agg: Agg::Count,
            })
            .unwrap();
            let sweep_s = lower(&Query {
                filter: sweep_f,
                agg: Agg::SumI32(W),
            })
            .unwrap();
            let (sweep_us, got) = median_us(reps, || {
                (
                    run(&sweep_c, &ts, &w, &live) as u64,
                    run(&sweep_s, &ts, &w, &live),
                )
            });
            assert_eq!(got, want, "sweep n={n} sel={sel}");
            let (slice_us, got) = median_us(reps, || {
                let lo = ts.partition_point(|&t| t < a) as u32;
                let hi = ts.partition_point(|&t| t < b) as u32;
                let f = Filter::and([Filter::plane(LIVE), Filter::Cmp(TS, Cmp::Range { lo, hi })]);
                let c = lower(&Query {
                    filter: f.clone(),
                    agg: Agg::Count,
                })
                .unwrap();
                let s = lower(&Query {
                    filter: f,
                    agg: Agg::SumI32(W),
                })
                .unwrap();
                (run(&c, &ts, &w, &live) as u64, run(&s, &ts, &w, &live))
            });
            assert_eq!(got, want, "slice n={n} sel={sel}");
            println!(
                "{n}\t{sel}\t{}\t{sweep_us:.1}\t{slice_us:.1}\t{:.1}x",
                want.0,
                sweep_us / slice_us
            );
        }
    }
}
