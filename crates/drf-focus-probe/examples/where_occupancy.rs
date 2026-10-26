//! D-DR256-WHERE P1 lab: what selection statistics cost over a demanded (Keep)
//! mask, and whether a shared block schedule pays for multi-property masked
//! reads. Lab only, reported, not gated. No production path, no new ABI.
//!
//! Universe: one DR256 global population, 65,536 rows, 1,024 u64 words.
//! A 256-row block is 4 words.
//!
//! Q1, occupancy. `popcount` alone (what `Agg::Count` folds today) against one
//! pass that also counts active 64-row words and active 256-row blocks.
//! Correctness: all three equal a BTreeSet oracle on every fixture.
//!
//! Q2, multi-property masked reads. P properties, each a presence plane and an
//! i32 column. Answer: per property, the sum of the column over R ∧ Presence[p].
//! - Arm S (today's composition): P independent full passes, each
//!   `mask_and` + `masked_sum_i32` over all 65,536 rows.
//! - Arm B (shared block schedule): R's active 256-row blocks are found once;
//!   each property folds only those blocks (`mask_and` + `masked_sum_i32` on the
//!   256-row slice). Dead blocks are never read.
//!
//!   The block walk reads R's summary bits, not a row-id list, so it is not a
//!   selection vector (fold law 5). Both arms must give the same P sums.
//!
//! Limits, stated: ndarray-level only. mask-risc's own survivor skip inside a
//! tile is NOT exercised here, so Arm S overstates what a Quack program pays;
//! the Quack-path measurement is the next rung (P2).

use ndarray::simd::{mask_and, mask_ternlog_popcount, masked_sum_i32, popcount_batch_u64};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 65_536;
const WORDS: usize = N / 64;
const BLOCK_WORDS: usize = 4;
const BLOCKS: usize = WORDS / BLOCK_WORDS;

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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn mask_of(rows: &BTreeSet<usize>) -> Vec<u64> {
    let mut m = vec![0u64; WORDS];
    for &r in rows {
        m[r >> 6] |= 1 << (r & 63);
    }
    m
}

/// The candidate statistic, one pass: (popcount, active 64-row words, active
/// 256-row blocks). Lab implementation only.
fn occupancy(m: &[u64]) -> (u64, u32, u32) {
    let (mut pop, mut words, mut blocks) = (0u64, 0u32, 0u32);
    for b in m.chunks_exact(BLOCK_WORDS) {
        let mut any = 0u64;
        for &w in b {
            pop += w.count_ones() as u64;
            words += (w != 0) as u32;
            any |= w;
        }
        blocks += (any != 0) as u32;
    }
    (pop, words, blocks)
}

fn fixtures() -> Vec<(String, BTreeSet<usize>)> {
    let mut out = Vec::new();
    let mut rng = Rng(0xD256);
    for &k in &[0usize, 1, 63, 64, 65, 255, 256, 257, 4096, 32_768, 65_536] {
        out.push((format!("contiguous k={k}"), (0..k).collect()));
        if k > 0 && k < N {
            let mut u = BTreeSet::new();
            while u.len() < k {
                u.insert(rng.below(N));
            }
            out.push((format!("uniform k={k}"), u));
            // Clustered: whole 64-row runs at random word positions, last run partial.
            let mut c = BTreeSet::new();
            while c.len() < k {
                let w = rng.below(WORDS);
                for r in w * 64..w * 64 + 64 {
                    if c.len() < k {
                        c.insert(r);
                    }
                }
            }
            out.push((format!("clustered k={k}"), c));
        }
    }
    // The two 4096 fixtures: 64 words / 16 blocks against one row per 16.
    out.push(("4096 dense-64w".into(), (0..4096).collect()));
    out.push(("4096 scattered-1per16".into(), (0..N).step_by(16).collect()));
    // HHTL prefix: everything inside one 4,096-row prefix, half-dense.
    out.push((
        "hhtl-prefix 2048".into(),
        (8192..8192 + 4096).step_by(2).collect(),
    ));
    out
}

fn main() {
    let reps = 301;
    println!("# Q1 occupancy, ns per call (median of {reps}); oracle-checked");
    println!("fixture | pop | words | blocks | popcount ns | occupancy ns");
    let fx = fixtures();
    for (name, rows) in &fx {
        let m = mask_of(rows);
        let (pop, w, b) = occupancy(&m);
        let ow: BTreeSet<usize> = rows.iter().map(|r| r >> 6).collect();
        let ob: BTreeSet<usize> = rows.iter().map(|r| r >> 8).collect();
        assert_eq!(pop as usize, rows.len(), "{name}: popcount");
        assert_eq!(w as usize, ow.len(), "{name}: active words");
        assert_eq!(b as usize, ob.len(), "{name}: active blocks");
        assert_eq!(popcount_batch_u64(&m), pop, "{name}: ndarray popcount");
        let t_pop = median_ns(reps, || {
            black_box(popcount_batch_u64(black_box(&m)));
        });
        let t_occ = median_ns(reps, || {
            black_box(occupancy(black_box(&m)));
        });
        println!("{name} | {pop} | {w} | {b} | {t_pop:.0} | {t_occ:.0}");
    }

    println!();
    println!("# Q2 multi-property masked reads, ns per workload (median of {reps})");
    println!("fixture | P | logical reads | S full passes ns | B shared blocks ns | B/S");
    let mut rng = Rng(0xB10C);
    let pmax = 64usize;
    // Skewed presence: property p is present on ~ (p+1)/(pmax+1) of rows... reversed,
    // so low p is dense and high p is sparse.
    let presence: Vec<Vec<u64>> = (0..pmax)
        .map(|p| {
            let keep = pmax - p; // 64 .. 1 out of 65
            (0..WORDS)
                .map(|_| {
                    let mut w = 0u64;
                    for bit in 0..64 {
                        if rng.below(pmax + 1) < keep {
                            w |= 1 << bit;
                        }
                    }
                    w
                })
                .collect()
        })
        .collect();
    let cols: Vec<Vec<i32>> = (0..pmax)
        .map(|_| (0..N).map(|_| (rng.next() % 2001) as i32 - 1000).collect())
        .collect();
    let ones = vec![u64::MAX; WORDS];
    let mut tmp = vec![0u64; WORDS];
    let mut tmp_b = [0u64; BLOCK_WORDS];
    let mut sums_s = vec![0i64; pmax];
    let mut sums_b = vec![0i64; pmax];
    let picks = [
        "contiguous k=256",
        "uniform k=256",
        "clustered k=4096",
        "4096 dense-64w",
        "4096 scattered-1per16",
        "uniform k=32768",
        "contiguous k=65536",
    ];
    for want in picks {
        let (name, rows) = fx.iter().find(|(n, _)| n == want).unwrap();
        let r = mask_of(rows);
        let active: Vec<usize> = (0..BLOCKS)
            .filter(|&b| r[b * BLOCK_WORDS..(b + 1) * BLOCK_WORDS].iter().any(|&w| w != 0))
            .collect();
        for &p in &[1usize, 2, 4, 8, 16, 32, 64] {
            // Logical reads and the oracle sums.
            let mut logical = 0u64;
            for q in 0..p {
                // a & b, c ignored: ternlog immediate 0xC0.
                let c = mask_ternlog_popcount::<0xC0>(&r, &presence[q], &ones);
                let mut oracle_c = 0u64;
                let mut oracle_s = 0i64;
                for &row in rows {
                    if presence[q][row >> 6] >> (row & 63) & 1 == 1 {
                        oracle_c += 1;
                        oracle_s += cols[q][row] as i64;
                    }
                }
                assert_eq!(c, oracle_c, "{name} p={q}: ternlog count");
                logical += c;
                sums_s[q] = oracle_s; // checked against both arms below
            }
            let oracle: Vec<i64> = sums_s[..p].to_vec();

            let mut arm_s = |sums_s: &mut [i64]| {
                for q in 0..p {
                    mask_and(&r, &presence[q], &mut tmp);
                    sums_s[q] = black_box(masked_sum_i32(&cols[q], &tmp));
                }
            };
            arm_s(&mut sums_s);
            assert_eq!(&sums_s[..p], &oracle[..], "{name}: arm S");
            let t_s = median_ns(reps, || {
                arm_s(&mut sums_s);
            });

            let mut arm_b = |sums_b: &mut [i64]| {
                for q in 0..p {
                    let mut acc = 0i64;
                    for &b in &active {
                        let w = b * BLOCK_WORDS;
                        mask_and(
                            &r[w..w + BLOCK_WORDS],
                            &presence[q][w..w + BLOCK_WORDS],
                            &mut tmp_b,
                        );
                        acc += masked_sum_i32(&cols[q][b * 256..b * 256 + 256], &tmp_b);
                    }
                    sums_b[q] = black_box(acc);
                }
            };
            arm_b(&mut sums_b);
            assert_eq!(&sums_b[..p], &oracle[..], "{name}: arm B");
            let t_b = median_ns(reps, || {
                arm_b(&mut sums_b);
            });
            println!(
                "{name} | {p} | {logical} | {t_s:.0} | {t_b:.0} | {:.3}",
                t_b / t_s
            );
        }
    }
}
