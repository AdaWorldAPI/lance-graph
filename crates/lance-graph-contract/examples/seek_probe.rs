//! D-ASM-11 pre-work, P-1 timed: four cursor seeks over a sorted `u32` lane.
//!
//! `m` increasing targets spread evenly over `N` keys; each seek starts at the
//! previous answer. Reports ns per seek and ns per key (`N`) for:
//! galloping, a fresh whole-lane binary search, a linear step, and the hybrid
//! `seek_partition_point` given the true gap as its hint. Every method's
//! answers are checked against `partition_point` before any timing counts.
//! Only a v4 run (`avx512f=true` printed below) may be cited.
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 RUSTFLAGS=-Ctarget-cpu=native \
//!   cargo run --release -p lance-graph-contract --example seek_probe
//! ```

use std::hint::black_box;
use std::time::Instant;

use lance_graph_contract::ordered_lane::{gallop_partition_point, seek_partition_point};

fn best(reps: usize, mut f: impl FnMut() -> u64) -> (u64, f64) {
    let mut t_best = f64::INFINITY;
    let mut out = 0;
    for _ in 0..reps {
        let t = Instant::now();
        out = black_box(f());
        t_best = t_best.min(t.elapsed().as_secs_f64());
    }
    (out, t_best)
}

fn main() {
    println!(
        "tier: avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("N | N/m | gallop ns/seek | fresh ns/seek | linear ns/seek | hybrid ns/seek");
    for log_n in [20u32, 24] {
        let n = 1usize << log_n;
        let keys: Vec<u32> = (0..n as u32).map(|i| 2 * i).collect();
        for ratio in [2usize, 16, 256, 4096, 65536] {
            let m = n / ratio;
            let targets: Vec<u32> = (0..m).map(|j| ((2 * n * j) / m) as u32 + 1).collect();
            let want: u64 = targets
                .iter()
                .map(|&t| keys.partition_point(|&k| k < t) as u64)
                .sum();
            let (g, tg) = best(5, || {
                let mut c = 0;
                let mut s = 0u64;
                for &t in &targets {
                    c = gallop_partition_point(&keys, c, |&k| k < t);
                    s += c as u64;
                }
                s
            });
            let (f, tf) = best(5, || {
                targets
                    .iter()
                    .map(|&t| keys.partition_point(|&k| k < t) as u64)
                    .sum()
            });
            let (l, tl) = if ratio <= 4096 {
                best(3, || {
                    let mut c = 0usize;
                    let mut s = 0u64;
                    for &t in &targets {
                        while c < n && keys[c] < t {
                            c += 1;
                        }
                        s += c as u64;
                    }
                    s
                })
            } else {
                (want, f64::NAN)
            };
            let (h, th) = best(5, || {
                let mut c = 0;
                let mut s = 0u64;
                for &t in &targets {
                    c = seek_partition_point(&keys, c, ratio, |&k| k < t);
                    s += c as u64;
                }
                s
            });
            assert_eq!((g, f, l, h), (want, want, want, want), "N={n} N/m={ratio}");
            let per = |t: f64| t * 1e9 / m as f64;
            println!(
                "2^{log_n} | {ratio} | {:.2} | {:.2} | {:.2} | {:.2}",
                per(tg),
                per(tf),
                per(tl),
                per(th)
            );
        }
    }
}
