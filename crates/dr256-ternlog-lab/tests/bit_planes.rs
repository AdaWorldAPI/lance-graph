//! A one-byte code held as eight resident bit-planes: band subsets, fused
//! compares, and the sum as eight popcounts.
//! Pass conditions: `.claude/board/entries/2026-11-13-weather-bit-planes.md`
//! B1, B2, B3.
//!
//! Oracles are row loops over the codes; where one is reused across many
//! thresholds it is a histogram built by one row loop (region rows per code).

use dr256_ternlog_lab::bit_planes::*;
use lance_graph_mask_risc::{
    execute_into, fuse_program, BoolExpr, Foreign, Operand, Out, Planes, Program, Scratch,
    Terminal, Value,
};

const GRID: usize = 721 * 1440;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn region(i: usize) -> bool {
    !mix(i as u64 ^ 0x11).is_multiple_of(3)
}

fn plane_of(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut words = vec![0u64; n.div_ceil(64)];
    for i in (0..n).filter(|&i| set(i)) {
        words[i / 64] |= 1 << (i % 64);
    }
    words
}

struct Fixture {
    codes: Vec<u8>,
    bp: BitPlanes,
    region_w: Vec<u64>,
    /// Region rows per code value.
    by_code: [usize; 256],
}

fn fixture() -> Fixture {
    let codes: Vec<u8> = (0..GRID as u64).map(|i| (mix(i) % 256) as u8).collect();
    let bp = BitPlanes::of(&codes);
    let region_w = plane_of(GRID, region);
    let mut by_code = [0usize; 256];
    for (i, &c) in codes.iter().enumerate() {
        if region(i) {
            by_code[usize::from(c)] += 1;
        }
    }
    Fixture {
        codes,
        bp,
        region_w,
        by_code,
    }
}

/// Plane indices 0..=7 are the code bits, 8 is the region.
fn count(program: &Program, bp: &BitPlanes, region_w: &[u64]) -> usize {
    let mut masks: Vec<&[u64]> = bp.planes.iter().map(|p| p.as_slice()).collect();
    masks.push(region_w);
    let planes = Planes {
        n_rows: bp.rows,
        masks: &masks,
        lanes: &[],
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &[],
    };
    let mut scratch = Scratch::for_program(program, bp.rows).unwrap();
    match execute_into(program, &planes, &foreign, &mut scratch, Out::None).unwrap() {
        Value::Count(c) => c,
        other => panic!("not a count: {other:?}"),
    }
}

fn region_leaf() -> BoolExpr {
    BoolExpr::Leaf(Operand::Plane(8))
}

fn fused_count(e: &BoolExpr, f: &Fixture) -> usize {
    let p = fuse_program(e, 0, |m| Terminal::Count { mask: m }).unwrap();
    count(&p, &f.bp, &f.region_w)
}

/// Region rows with `lo < code < hi`, from the per-code histogram.
fn between(f: &Fixture, lo: usize, hi: usize) -> usize {
    (lo + 1..hi).map(|c| f.by_code[c]).sum()
}

#[test]
fn every_band_subset_is_one_ternlog_immediate() {
    let f = fixture();
    // Region rows per band, from one row loop over the codes.
    let mut band = [0usize; 8];
    for (i, &c) in f.codes.iter().enumerate() {
        if region(i) {
            band[usize::from(c >> 5)] += 1;
        }
    }
    let pop: usize = band.iter().sum();
    assert!(pop > 0);
    let mut inside = 0;
    for subset in 0..=255u8 {
        let want: usize = (0..8)
            .filter(|b| (subset >> b) & 1 == 1)
            .map(|b| band[b])
            .sum();
        let got = count(&band_program(subset, Operand::Plane(8)), &f.bp, &f.region_w);
        assert_eq!(got, want, "subset {subset:#04x}");
        if got > 0 && got < pop {
            inside += 1;
        }
    }
    assert!(
        inside >= 200,
        "only {inside} of 256 answers are non-trivial"
    );
    assert_eq!(
        count(&band_program(0, Operand::Plane(8)), &f.bp, &f.region_w),
        0
    );
    assert_eq!(
        count(&band_program(0xFF, Operand::Plane(8)), &f.bp, &f.region_w),
        pop
    );
}

#[test]
fn the_fused_compare_equals_the_row_loop_at_every_threshold() {
    let f = fixture();
    for t in 0..=255u8 {
        let t_us = usize::from(t);
        let want_gt = between(&f, t_us, 256);
        match gt(t, plane_op) {
            Some(e) => assert_eq!(
                fused_count(&and(e, region_leaf()), &f),
                want_gt,
                "code > {t}"
            ),
            None => assert_eq!(want_gt, 0, "gt {t} has no expression but rows satisfy it"),
        }
        let want_lt: usize = f.by_code[..t_us].iter().sum();
        match lt(t, plane_op) {
            Some(e) => assert_eq!(
                fused_count(&and(e, region_leaf()), &f),
                want_lt,
                "code < {t}"
            ),
            None => assert_eq!(want_lt, 0, "lt {t} has no expression but rows satisfy it"),
        }
    }

    // The range of the first probe, checked against a direct row loop.
    let range = and(
        and(gt(145, plane_op).unwrap(), lt(183, plane_op).unwrap()),
        region_leaf(),
    );
    let got = fused_count(&range, &f);
    let want = f
        .codes
        .iter()
        .enumerate()
        .filter(|&(i, &c)| region(i) && 145 < c && c < 183)
        .count();
    assert_eq!(got, want);
    assert!(got > 0 && got < GRID / 5, "range count {got}");
}

#[test]
fn the_sum_is_eight_popcounts() {
    let f = fixture();
    let selection = and(
        and(gt(145, plane_op).unwrap(), lt(183, plane_op).unwrap()),
        region_leaf(),
    );
    let mut sum = 0u64;
    for k in 0..8 {
        let e = and(selection.clone(), BoolExpr::Leaf(plane_op(k)));
        sum += (fused_count(&e, &f) as u64) << k;
    }
    let want: u64 = f
        .codes
        .iter()
        .enumerate()
        .filter(|&(i, &c)| region(i) && 145 < c && c < 183)
        .map(|(_, &c)| u64::from(c))
        .sum();
    assert_eq!(sum, want);
    assert!(sum > 0);
}

#[test]
fn a_band_edge_is_the_only_bound_a_band_subset_can_state() {
    let f = fixture();
    // Bands 5 and 6 are codes 160..=223.
    let bands_5_6 = count(&band_program(0x60, Operand::Plane(8)), &f.bp, &f.region_w);
    assert_eq!(bands_5_6, between(&f, 159, 224));
    assert!(bands_5_6 > 0);

    // 145 < code < 183 starts and ends inside bands; no subset states it.
    let want = between(&f, 145, 183);
    assert!(want > 0);
    let mut band = [0usize; 8];
    for c in 0..256usize {
        band[c >> 5] += f.by_code[c];
    }
    for subset in 0..=255u8 {
        let n: usize = (0..8)
            .filter(|b| (subset >> b) & 1 == 1)
            .map(|b| band[b])
            .sum();
        assert_ne!(n, want, "subset {subset:#04x} equals the range");
    }
}
