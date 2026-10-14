//! D-MTP-1b — the hop as a range (plan §6c).
//!
//! MeTTa: `(match &kb (, (parent $x $y) (parent $y $z)) ($x $z))`, the same
//! two-hop join as D-MTP-1. When the link lane `dst` is non-decreasing over
//! rows, `Semijoin(dst, valid)` restricted to `valid` is an intersection of
//! interval sets: `valid`'s runs and their preimages under `dst` (found by
//! `partition_point`). The count is interval arithmetic; pairs are emitted by
//! walking the kept intervals. A non-monotone lane is refused and the caller
//! keeps the gather program.
//!
//! Arms: G-scattered / G-clustered (the D-MTP-1 gather program), R-scan
//! (`valid`'s runs from a word scan), R-declared (runs recorded at ingest).
//! Oracle: datafrog (AdaWorldAPI fork).
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 RUSTFLAGS=-Ctarget-cpu=native \
//!   cargo run --release --manifest-path crates/lance-graph-metta/Cargo.toml --example join_range_probe
//! ```

mod common;

use common::{best, oracle, total, Relation2};

/// Depth of node `x` in the fanout-4 BFS tree (`parent(x) = (x-1)/4`).
fn depth(mut x: u32) -> u32 {
    let mut d = 0;
    while x > 0 {
        x = (x - 1) / 4;
        d += 1;
    }
    d
}

/// F3 (`banded = false`) or F4 (`banded = true`: no facts for nodes whose
/// depth is a multiple of 3), in address order.
fn tree(n: u32, banded: bool) -> Vec<(u32, u32)> {
    (1..n)
        .filter(|&x| !banded || depth(x) % 3 != 0)
        .map(|x| (x, (x - 1) / 4))
        .collect()
}

/// A deterministic random permutation of `0..n` (xorshift + Fisher–Yates).
fn permutation(n: u32) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n).collect();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..n as usize).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        p.swap(i, (s % (i as u64 + 1)) as usize);
    }
    p
}

/// The same facts under a permutation of addresses (the scattered twin).
fn scatter(n: u32, facts: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let p = permutation(n);
    facts.iter().map(|&(x, y)| (p[x as usize], p[y as usize])).collect()
}

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::intersect;

    const SIZES: [u32; 3] = [100, 1000, 4097];

    fn gather_pairs(r: &Relation2) -> Vec<(u32, u32)> {
        let mut kept = vec![0u64; r.valid.len()];
        let mut out = Vec::new();
        r.pairs(&mut kept, &mut out);
        sorted(out)
    }

    fn range_pairs(r: &Relation2, runs: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let kept = r.kept_ranges(runs).expect("monotone");
        let mut out = Vec::new();
        r.range_pairs(&kept, &mut out);
        sorted(out)
    }

    #[test]
    fn every_arm_equals_datafrog() {
        for n in SIZES {
            for banded in [false, true] {
                let f = tree(n, banded);
                let want = sorted(oracle(&f));
                assert!(!want.is_empty());
                let r = Relation2::ingest(n as usize, &f).unwrap();
                assert!(r.dst_monotone() && r.monotone);
                let declared = r.declared_runs.clone().expect("address order");
                let scanned = r.scanned_runs();
                assert_eq!(declared, scanned, "n={n} banded={banded}");
                assert_eq!(gather_pairs(&r), want, "G-clustered n={n}");
                assert_eq!(range_pairs(&r, &scanned), want, "R n={n} banded={banded}");
                assert_eq!(r.count(), want.len());
                let kept = r.kept_ranges(&scanned).unwrap();
                assert_eq!(total(&kept), want.len());

                let s = scatter(n, &f);
                let rs = Relation2::ingest(n as usize, &s).unwrap();
                assert_eq!(gather_pairs(&rs).len(), want.len(), "G-scattered n={n}");
                assert_eq!(sorted(oracle(&s)), gather_pairs(&rs));
            }
        }
    }

    #[test]
    fn a_non_monotone_lane_is_refused() {
        let s = scatter(1000, &tree(1000, true));
        let r = Relation2::ingest(1000, &s).unwrap();
        assert!(!r.dst_monotone() && !r.monotone);
        assert!(r.declared_runs.is_none());
        assert!(r.kept_ranges(&r.scanned_runs()).is_none());
    }

    /// Both halves of the intersection must matter on F4, or the disable runs
    /// in plan §6c prove nothing.
    #[test]
    fn f4_needs_both_halves_of_the_intersection() {
        let n = 4097;
        let r = Relation2::ingest(n as usize, &tree(n, true)).unwrap();
        let runs = r.scanned_runs();
        // Levels 1-2 and 4-5 merge: [1, 21) and [85, 1365) at n = 4097.
        assert_eq!(runs, vec![(1, 21), (85, 1365)], "F4 valid runs");
        let want = total(&r.kept_ranges(&runs).unwrap());
        assert_ne!(total(&r.preimage(&runs)), want, "preimage alone");
        assert_ne!(total(&runs), want, "valid alone");
        // F3: one run, and the preimage alone already equals the answer.
        let r3 = Relation2::ingest(n as usize, &tree(n, false)).unwrap();
        let runs3 = r3.scanned_runs();
        assert_eq!(runs3.len(), 1);
        assert_eq!(
            intersect(&runs3, &r3.preimage(&runs3)),
            r3.kept_ranges(&runs3).unwrap()
        );
    }
}

fn main() {
    println!(
        "tier: avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    const REPS: usize = 5;
    for n in [1_000u32, 10_000, 100_000, 1_000_000] {
        for (name, banded) in [("F3", false), ("F4", true)] {
            let f = tree(n, banded);
            let s = scatter(n, &f);
            let (want, t_oracle) = best(REPS, || oracle(&f));
            let want = sorted(want);
            let pairs = want.len();

            let (r, t_ingest) = best(REPS, || Relation2::ingest(n as usize, &f).unwrap());
            let (rs, t_ingest_s) = best(REPS, || Relation2::ingest(n as usize, &s).unwrap());
            let (_, t_mono) = best(REPS, || std::hint::black_box(std::hint::black_box(&r).dst_monotone()));

            let (c, t_gc) = best(REPS, || r.count());
            let (cs, t_gs) = best(REPS, || rs.count());
            let mut kept = vec![0u64; r.valid.len()];
            let mut out = Vec::with_capacity(pairs);
            let (_, t_gc_pairs) = best(REPS, || r.pairs(&mut kept, &mut out));
            assert_eq!(sorted(out.clone()), want);
            let (_, t_gs_pairs) = best(REPS, || rs.pairs(&mut kept, &mut out));
            assert_eq!(out.len(), pairs);

            let (rc, t_rscan) = best(REPS, || total(&r.kept_ranges(&r.scanned_runs()).unwrap()));
            let declared = r.declared_runs.clone().unwrap();
            let (rd, t_rdecl) = best(REPS, || total(&r.kept_ranges(&declared).unwrap()));
            let kept_r = r.kept_ranges(&declared).unwrap();
            let (_, t_r_pairs) = best(REPS, || r.range_pairs(&kept_r, &mut out));
            assert_eq!(sorted(out.clone()), want);
            assert!(c == pairs && cs == pairs && rc == pairs && rd == pairs);

            let per_row = |t: f64| t * 1e9 / n as f64;
            let per_pair = |t: f64| t * 1e9 / pairs.max(1) as f64;
            println!(
                "{name} n={n:>8} pairs={pairs:>8} runs={:>3} | count ns/row: G-scat {:.3} G-clus {:.3} R-scan {:.4} R-decl {:.5} \
                 | emit ns/pair: G-scat {:.2} G-clus {:.2} R {:.2} | ingest ns/row: clus {:.2} scat {:.2} mono {:.3} \
                 | datafrog ns/pair {:.1}",
                declared.len(),
                per_row(t_gs),
                per_row(t_gc),
                per_row(t_rscan),
                per_row(t_rdecl),
                per_pair(t_gs_pairs),
                per_pair(t_gc_pairs),
                per_pair(t_r_pairs),
                per_row(t_ingest),
                per_row(t_ingest_s),
                per_row(t_mono),
                per_pair(t_oracle),
            );
        }
    }
}
