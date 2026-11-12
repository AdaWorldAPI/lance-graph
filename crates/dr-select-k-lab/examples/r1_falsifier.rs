//! The pre-registered falsifier of row R1 in
//! `.claude/plans/duckdb-to-v3-translation-matrix-v1.md`:
//!
//! "Measure `masked_sum_i32` against an index-list sum at
//! s in {1/64, 1/256, 1/1024}; if the mask loses at any of them, R1
//! downgrades to ADAPT with a documented sparse threshold."
//!
//! Both arms answer the same question over the same `i32` lane: the sum of
//! the selected values. The mask arm is the bare ndarray kernel, with no
//! program, scratch or dispatch around it. The list arm reads a sorted
//! `u32` position list, which is what a `SelectionVector` is.
//!
//! Each cell is checked against a row-at-a-time oracle before it is timed.
//! Prints one TSV table on stdout. Run with
//! `cargo run --release --example r1_falsifier`.

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::masked_sum_i32;

const WARMUP: usize = 5;
const RUNS: usize = 101;
const NS: [usize; 2] = [65_536, 1_000_000];
const DENOMS: [usize; 3] = [64, 256, 1024];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Sorted positions of exactly `k` selected rows out of `n`.
fn positions(n: usize, k: usize, clustered: bool, rng: &mut Rng) -> Vec<u32> {
    if clustered {
        // One contiguous run, so the mask has long stretches of zero words.
        let start = (n - k) / 2;
        return (start..start + k).map(|r| r as u32).collect();
    }
    // Partial Fisher-Yates over the row ids, then sort.
    let mut ids: Vec<u32> = (0..n as u32).collect();
    for i in 0..k {
        let j = i + (rng.next() as usize) % (n - i);
        ids.swap(i, j);
    }
    ids.truncate(k);
    ids.sort_unstable();
    ids
}

fn mask_of(n: usize, pos: &[u32]) -> Vec<u64> {
    let mut m = vec![0u64; n.div_ceil(64)];
    for &p in pos {
        m[p as usize / 64] |= 1u64 << (p % 64);
    }
    m
}

#[inline(never)]
fn list_sum(values: &[i32], pos: &[u32]) -> i64 {
    pos.iter().map(|&p| i64::from(values[p as usize])).sum()
}

/// Reference arm: the same mask read by skipping zero words and walking set
/// bits with `trailing_zeros`. It bounds what a mask consumer could reach on
/// sparse input without a position list. It is a measurement arm, not a
/// kernel anyone calls.
#[inline(never)]
fn bitwalk_sum(values: &[i32], mask: &[u64]) -> i64 {
    let mut sum = 0i64;
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let r = w * 64 + bits.trailing_zeros() as usize;
            sum += i64::from(values[r]);
            bits &= bits - 1;
        }
    }
    sum
}

fn median_ns<F: FnMut() -> i64>(mut f: F) -> u128 {
    for _ in 0..WARMUP {
        black_box(f());
    }
    let mut t = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let s = Instant::now();
        black_box(f());
        t.push(s.elapsed().as_nanos());
    }
    t.sort_unstable();
    t[RUNS / 2]
}

fn main() {
    println!(
        "# R1 falsifier: bare masked_sum_i32 against an index-list sum; medians of {RUNS} after {WARMUP} warm-up, ns; compiled avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
    );
    println!("n\tpattern\ts\tk\tmask_ns\tbitwalk_ns\tlist_ns\tmask_over_list\tbitwalk_over_list\tmask_bytes_read\tlist_bytes_read\tmask_loses\tbitwalk_loses");
    let mut rng = Rng(0x5EED_0001);
    for n in NS {
        let values: Vec<i32> = (0..n).map(|_| (rng.next() % 100) as i32).collect();
        for clustered in [false, true] {
            for d in DENOMS {
                let k = n / d;
                let pos = positions(n, k, clustered, &mut rng);
                let mask = mask_of(n, &pos);

                // Oracle: one row at a time over the whole lane.
                let want: i64 = (0..n)
                    .filter(|&r| mask[r / 64] >> (r % 64) & 1 == 1)
                    .map(|r| i64::from(values[r]))
                    .sum();
                assert_eq!(pos.len(), k);
                assert_eq!(masked_sum_i32(&values, &mask), want);
                assert_eq!(list_sum(&values, &pos), want);
                assert_eq!(bitwalk_sum(&values, &mask), want);

                let m = median_ns(|| masked_sum_i32(black_box(&values), black_box(&mask)));
                let b = median_ns(|| bitwalk_sum(black_box(&values), black_box(&mask)));
                let l = median_ns(|| list_sum(black_box(&values), black_box(&pos)));
                println!(
                    "{n}\t{}\t1/{d}\t{k}\t{m}\t{b}\t{l}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}",
                    if clustered { "clustered" } else { "random" },
                    m as f64 / l.max(1) as f64,
                    b as f64 / l.max(1) as f64,
                    mask.len() * 8,
                    pos.len() * 4,
                    if m > l { "yes" } else { "no" },
                    if b > l { "yes" } else { "no" },
                );
            }
        }
    }
}
