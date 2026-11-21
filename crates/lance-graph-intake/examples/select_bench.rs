//! D-INTAKE-2D: selection through the intake facade, cold and warm, beside
//! DuckDB 2.0.1 (`duckdb_select.py`, same data, same predicates).
//!
//! Row `r` has `id = r + 1`, `a = (r * 7919) % 1000`, `b = NULL when r % 5 ==
//! 0 else (r * 31) % 50 - 25`, `t = 'v' || ((r * 13) % 40)`.
//!
//! | Predicate | Definition |
//! |---|---|
//! | P1 | `a < 100` |
//! | P2 | `a < 100 AND b >= 0`, three-valued |
//! | P3 | a bag of 4096 keys `1 + (k * 2654435761) % n` |
//!
//! Every answer is asserted against a scalar oracle before any time is
//! reported. Times are medians of 11 runs, in nanoseconds:
//!
//! - `count`: `count_where`, which does the work of a fresh query (bound
//!   check, validity folds, predicate, count). This compares with DuckDB's
//!   warm `COUNT(*)`.
//! - `select_ref`: the same, plus keeping the positions under the current
//!   version.
//! - `count_of`: a warm replay of the kept selection (frame check plus kept
//!   length). This is amortization, not a faster selection vector.
//! - `rows_of(k)`: rows decoded from a kept selection of k positions, the
//!   fixed per-call overhead (bound check, reader open) plus k × 4 cells.
//!
//! Cold admission (`from_rows`) is reported in µs as a cold, out-of-fold cost,
//! not a gate.
//!
//! ```text
//! cargo run --release -p lance-graph-intake --example select_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use lance_graph_intake::{ShapeConfig, Table};
use lance_graph_quack::{Cmp, Col, Filter};

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> u128 {
    let mut out = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        black_box(f());
        out.push(t0.elapsed().as_nanos());
    }
    median(out)
}

fn rows(n: usize) -> Vec<[Option<String>; 4]> {
    (0..n)
        .map(|r| {
            [
                Some((r + 1).to_string()),
                Some(((r * 7919) % 1000).to_string()),
                (r % 5 != 0).then(|| (((r * 31) % 50) as i64 - 25).to_string()),
                Some(format!("v{}", (r * 13) % 40)),
            ]
        })
        .collect()
}

fn main() {
    for n in [100_000usize, 1_000_000] {
        let owned = rows(n);
        let grid: Vec<Vec<Option<&str>>> = owned
            .iter()
            .map(|r| r.iter().map(|c| c.as_deref()).collect())
            .collect();
        let columns = ["id", "a", "b", "t"];

        let t0 = Instant::now();
        let mut table = Table::from_rows(
            "bench",
            0x2B04_0001,
            &columns,
            &grid,
            &ShapeConfig::default(),
        )
        .expect("admit");
        let admit_us = t0.elapsed().as_micros();

        let p1 = Filter::cmp(Col(1), Cmp::LtI32(100));
        let p2 = Filter::and([
            Filter::cmp(Col(1), Cmp::LtI32(100)),
            Filter::cmp(Col(2), Cmp::GeI32(0)),
        ]);
        let want1 = (0..n).filter(|r| (r * 7919) % 1000 < 100).count();
        let want2 = (0..n)
            .filter(|&r| (r * 7919) % 1000 < 100 && r % 5 != 0 && ((r * 31) % 50) >= 25)
            .count();
        assert_eq!(table.count_where(&p1).unwrap(), want1);
        assert_eq!(table.count_where(&p2).unwrap(), want2);

        let count1 = time(11, || table.count_where(&p1).unwrap());
        let count2 = time(11, || table.count_where(&p2).unwrap());
        let ref1 = time(11, || table.select_ref(&p1).unwrap());
        let kept = table.select_ref(&p1).unwrap();
        assert_eq!(kept.len(), want1);
        let replay = time(11, || table.count_of(&kept).unwrap());

        let bag: Vec<u64> = (0..4096u64)
            .map(|k| 1 + (k * 2654435761) % n as u64)
            .collect();
        let mut overhead = Vec::new();
        for k in [1usize, 64, 4096] {
            let r = table.select_keys_ref(&bag[..k]).unwrap();
            let got = table.rows_of(&r).unwrap();
            assert_eq!(got.rows.len(), k);
            assert_eq!(got.rows[0][0].as_deref(), Some(bag[0].to_string().as_str()));
            overhead.push((k, time(11, || table.rows_of(&r).unwrap())));
        }

        println!("n = {n}: admit {admit_us} us (cold, reported, not a gate)");
        println!(
            "  P1 count {count1} ns ({:.2} ns/selected, {want1} selected); select_ref {ref1} ns; warm count_of {replay} ns",
            count1 as f64 / want1 as f64
        );
        println!(
            "  P2 count {count2} ns ({:.2} ns/selected, {want2} selected)",
            count2 as f64 / want2 as f64
        );
        for (k, ns) in overhead {
            println!(
                "  rows_of({k}) {ns} ns ({:.1} ns/row, 4 cells each)",
                ns as f64 / k as f64
            );
        }
    }
}
