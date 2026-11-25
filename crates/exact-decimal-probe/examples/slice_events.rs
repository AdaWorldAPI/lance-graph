//! D-AED-4 measurement: Reseal against Append on one write pattern.
//! Run: `cargo run --release --example slice_events`.
//!
//! Workload: 256 batches of 256 amounts at scale 2 (|c| < 2.55, PM9);
//! every 16th batch also carries a value at scale 3 (finer), every 32nd a
//! value past PM17 (wider). Every 9th cell is absent.

use std::hint::black_box;
use std::time::Instant;

use exact_decimal_probe::{Exp, Policy, SliceLog};

fn batch(b: usize) -> Vec<Option<String>> {
    (0..256)
        .map(|i| {
            if i % 9 == 0 {
                return None;
            }
            Some(if i == 1 && b % 16 == 15 {
                format!("0.{:03}", (b * 7) % 1000 + 1)
            } else if i == 2 && b % 32 == 31 {
                "700000.25".to_owned()
            } else {
                format!("{}.{:02}", (b + i) % 2, (b * 31 + i) % 100)
            })
        })
        .collect()
}

/// Best of `reps`, in ns.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    let batches: Vec<Vec<Option<String>>> = (0..256).map(batch).collect();
    let finer = batches
        .iter()
        .filter(|b| {
            b[1].as_deref()
                .is_some_and(|t| t.len() == 5 && t.starts_with("0."))
        })
        .count();
    let wide = batches
        .iter()
        .filter(|b| b[2].as_deref() == Some("700000.25"))
        .count();
    assert_eq!((finer, wide), (16, 8), "the events are in the workload");
    let mut results = Vec::new();
    for policy in [Policy::Reseal, Policy::Append] {
        let mut log = SliceLog::new(policy, Exp::scale(3));
        let t = Instant::now();
        for b in &batches {
            let cells: Vec<Option<&str>> = b.iter().map(|c| c.as_deref()).collect();
            log.append(&cells).unwrap();
        }
        let write_ns = t.elapsed().as_nanos() as f64;
        let values: usize = log.slices().iter().map(|s| s.len).sum();
        let combined = log.combined().unwrap();
        // Baseline: rescale every value to the finest exponent, then sum.
        let finest = log.slices().iter().map(|s| s.exp).min().unwrap();
        let per_value = || {
            let mut n = 0u64;
            let mut s = 0i128;
            for p in log.slices() {
                let k = 10i128.pow((p.exp.0 - finest.0) as u32);
                for c in (0..p.len).filter_map(|i| p.get(i)) {
                    n += 1;
                    s += i128::from(c) * k;
                }
            }
            (n, s)
        };
        assert_eq!(per_value(), (combined.n, combined.s), "{policy:?}: exact");
        let align = best(200, || {
            black_box(log.combined().unwrap());
        });
        let rescale = best(20, || {
            black_box(per_value());
        });
        results.push(combined);
        println!(
            "{policy:?}: slices {:>3}  encoded {:>7} B ({:.3} B/value)  rewritten {:>7} values  \
             write {:>8.1} ns/value  aggregate: per-slice {:>9.0} ns ({} alignments)  per-value rescale {:>9.0} ns",
            log.slices().len(),
            log.bytes(),
            log.bytes() as f64 / values as f64,
            log.rewritten,
            write_ns / values as f64,
            align,
            combined.alignments,
            rescale,
        );
    }
    assert_eq!(
        (results[0].n, results[0].s, results[0].q),
        (results[1].n, results[1].s, results[1].q),
        "both policies hold the same values"
    );
}
