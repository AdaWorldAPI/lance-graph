//! Staged binding against in-place binding: bind cost and fold cost, with no
//! residual. Every answer is checked against the row-at-a-time oracle before
//! it is timed.
//!
//! `cargo run --release --manifest-path crates/dr-select-k-lab/Cargo.toml --example inplace_report`

use std::hint::black_box;
use std::time::Instant;

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::{self, Pattern};
use dr_select_k_lab::oracle::{self, Query};
use dr_select_k_lab::source::KeyRange;
use dr_select_k_lab::{Bound, InPlace, PreparedFolds, View};

const N: usize = 1_000_000;
const DUP: u32 = 4;
const RUNS: usize = 41;
const T: i32 = 500;

fn median(mut f: impl FnMut()) -> u64 {
    for _ in 0..3 {
        f();
    }
    let mut s = [0u64; RUNS];
    for slot in &mut s {
        let t0 = Instant::now();
        f();
        *slot = t0.elapsed().as_nanos() as u64;
    }
    s.sort_unstable();
    s[RUNS / 2]
}

fn main() {
    println!(
        "# N={N} runs={RUNS} ns, medians; filter selection, no residual, count(b>{T}) and sum(a)"
    );
    println!("k\tpattern\tarm\tbind_ns\tbind_bytes\tcount_ns\tsum_ns\tbreak_even_folds");
    for pattern in [Pattern::Clustered, Pattern::Random] {
        let src = fixture::source(N, pattern, DUP, 0x5EED_1234);
        for k in [256usize, 65_536, 750_000] {
            let range = KeyRange {
                lo: 0,
                hi: (k / DUP as usize) as i32 - 1,
            };
            let q = Query { range, t_a: None };
            let (want_count, want_sum) = (oracle::count_b_gt(&src, q, T), oracle::sum_a(&src, q));
            let c = Counters::new();
            let sel = src.select_filter(range, &c);
            assert_eq!(sel.rows.len(), k);

            // Staged: K rows of every lane copied, folds over the copy.
            let before = c.snapshot()[4];
            let bound = Bound::bind(&src, sel.clone(), &c).unwrap();
            let staged_bytes = c.snapshot()[4] - before;
            let mut folds = PreparedFolds::new(&bound);
            assert_eq!(folds.count_b_gt(&bound, View::All, T, &c), want_count);
            assert_eq!(folds.sum_a(&bound, View::All, &c), want_sum);
            let staged_bind = median(|| {
                black_box(Bound::bind(&src, black_box(sel.clone()), &c).unwrap());
            });
            let clone_only = median(|| {
                black_box(black_box(&sel).clone());
            });
            let staged_bind = staged_bind.saturating_sub(clone_only);
            let staged_count = median(|| {
                black_box(folds.count_b_gt(&bound, View::All, black_box(T), &c));
            });
            let staged_sum = median(|| {
                black_box(folds.sum_a(&bound, View::All, &c));
            });

            // In place: one bit per source row, no lane value copied.
            let before = c.snapshot()[4];
            let mut ip = InPlace::bind(&src, sel.clone(), &c).unwrap();
            let ip_bytes = c.snapshot()[4] - before;
            assert_eq!(ip.count_b_gt(&src, T, &c), Ok(want_count));
            assert_eq!(ip.sum_a(&src, &c), Ok(want_sum));
            assert_eq!(ip.via_positions_count_b_gt(&src, T, &c), Ok(want_count));
            assert_eq!(ip.via_positions_sum_a(&src, &c), Ok(want_sum));
            let ip_bind = median(|| {
                black_box(InPlace::bind(&src, black_box(sel.clone()), &c).unwrap());
            })
            .saturating_sub(clone_only);
            let plane_count = median(|| {
                black_box(ip.count_b_gt(&src, black_box(T), &c).unwrap());
            });
            let plane_sum = median(|| {
                black_box(ip.sum_a(&src, &c).unwrap());
            });
            let pos_count = median(|| {
                black_box(ip.via_positions_count_b_gt(&src, black_box(T), &c).unwrap());
            });
            let pos_sum = median(|| {
                black_box(ip.via_positions_sum_a(&src, &c).unwrap());
            });

            let name = if pattern == Pattern::Clustered {
                "clustered"
            } else {
                "random"
            };
            // Folds after which the staged bind has paid for itself against
            // this arm: extra bind cost over the per-fold saving.
            let even = |arm_bind: u64, arm_fold: u64| -> String {
                if arm_fold <= staged_count {
                    "never".into()
                } else {
                    let extra = staged_bind.saturating_sub(arm_bind) as f64;
                    format!("{:.1}", extra / (arm_fold - staged_count) as f64)
                }
            };
            println!("{k}\t{name}\tstaged\t{staged_bind}\t{staged_bytes}\t{staged_count}\t{staged_sum}\t-");
            println!(
                "{k}\t{name}\tplane-over-n\t{ip_bind}\t{ip_bytes}\t{plane_count}\t{plane_sum}\t{}",
                even(ip_bind, plane_count)
            );
            println!(
                "{k}\t{name}\tpositions\t{ip_bind}\t{ip_bytes}\t{pos_count}\t{pos_sum}\t{}",
                even(ip_bind, pos_count)
            );
            if ip.is_run() {
                assert_eq!(ip.run_count_b_gt(&src, T, &c), Ok(want_count));
                assert_eq!(ip.run_sum_a(&src, &c), Ok(want_sum));
                let run_count = median(|| {
                    black_box(ip.run_count_b_gt(&src, black_box(T), &c).unwrap());
                });
                let run_sum = median(|| {
                    black_box(ip.run_sum_a(&src, &c).unwrap());
                });
                println!(
                    "{k}\t{name}\trun-slice\t{ip_bind}\t{ip_bytes}\t{run_count}\t{run_sum}\t{}",
                    even(ip_bind, run_count)
                );
            }
        }
    }
}
