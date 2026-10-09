//! D-RPF-CONF-0, R3: the disagreement matrix over the DISTINCT LIVE revision
//! laws, measured on the u8 truth grid.
//!
//! The law list is the G4 census result
//! (`.claude/board/entries/2026-10-09-d-rpf-conf-0-confidence-surface.md`), not
//! "every function named revise". Each law is called through its real public
//! entry point; nothing is reimplemented here. Laws that compute in f32 are
//! quantized with ONE declared quantizer, `q` below, which is the same rule
//! `CausalEdge64::set_*_u8` applies (clamp, `*255`, round half away from zero).
//! A and B carry their own native u8 encoding and are not re-quantized.
//!
//! Laws compared (letters from the census):
//! - A    `CausalEdge64::revision` (ISA f32, k=1, `w = MAX` at c >= 0.999)
//! - B1   `NarsTables::build(1).revise` (the planner's replay path)
//! - B16  `NarsTables::build(16).revise`
//! - C    `ndarray::hpc::nars::nars_revision` (evidence split, c clamped to 0.9999)
//! - D    planner `nars::truth::TruthValue::revise` (`w = MAX` only at c >= 1)
//! - E    contract `exploration::NarsTruth::revision` (`+1e-9`, cap 0.99)
//! - LIN  contract `grammar::revise_truth` (c used directly as the weight)
//!
//! Not compared, with the reason recorded in the board entry: F and G (not
//! reachable from the planner and not on its path), H (private, `wip`),
//! J (`PathTruth::revise` takes a raw weight, so comparing it needs a declared
//! c->w rule that no caller fixes).

use causal_edge::edge::CausalEdge64;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use lance_graph_contract::crystal::TruthValue as CrystalTruth;
use lance_graph_contract::exploration::NarsTruth as ContractTruth;
use lance_graph_contract::grammar::revise_truth;
use lance_graph_planner::nars::truth::TruthValue as PlannerTruth;
use ndarray::hpc::nars::{nars_revision, NarsTruth as NdTruth};

/// The declared quantizer for f32 laws: the rule `CausalEdge64::set_*_u8` uses.
fn q(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn d(x: u8) -> f32 {
    x as f32 / 255.0
}

type Law<'a> = Box<dyn Fn(u8, u8, u8, u8) -> (u8, u8) + 'a>;

fn edge(f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

fn law_a() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r = edge(f1, c1).revision(edge(f2, c2));
        (r.frequency_u8(), r.confidence_u8())
    })
}

fn law_table(t: &NarsTables) -> Law<'_> {
    Box::new(move |f1, c1, f2, c2| {
        let p = t.revise(f1, c1, f2, c2);
        (unpack_f(p), unpack_c(p))
    })
}

fn law_c() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r = nars_revision(NdTruth::new(d(f1), d(c1)), NdTruth::new(d(f2), d(c2)));
        (q(r.frequency), q(r.confidence))
    })
}

fn law_d() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r = PlannerTruth::new(d(f1), d(c1)).revise(&PlannerTruth::new(d(f2), d(c2)));
        (q(r.frequency), q(r.confidence))
    })
}

fn law_e() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r = ContractTruth::new(d(f1), d(c1)).revision(&ContractTruth::new(d(f2), d(c2)));
        (q(r.frequency), q(r.confidence))
    })
}

fn law_lin() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r = revise_truth(CrystalTruth::new(d(f1), d(c1)), d(f2), d(c2));
        (q(r.frequency), q(r.confidence))
    })
}

/// Frequency pairs sampled; confidence is swept over the full 256 x 256 grid.
const FS: [u8; 5] = [0, 51, 128, 204, 255];

/// A confidence code at the edge of the scale, where laws differ by policy
/// rather than by arithmetic.
fn is_boundary(c: u8) -> bool {
    c == 0 || c >= 254
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Pair {
    differ: u64,
    differ_broad: u64,
    max_df: u8,
    max_dc: u8,
    /// The same maxima over broad cells only (no c in {0, 254, 255}).
    max_df_broad: u8,
    max_dc_broad: u8,
    /// Sums of |dF| and |dC| over every compared cell (mean = sum / CELLS).
    sum_df: u64,
    sum_dc: u64,
}

const CELLS: u64 = (FS.len() * FS.len() * 256 * 256) as u64;

/// One law's outputs over the whole sampled domain, in a fixed order.
fn surface(law: &Law<'_>) -> Vec<(u8, u8)> {
    let mut v = Vec::with_capacity(CELLS as usize);
    for &f1 in &FS {
        for &f2 in &FS {
            for c1 in 0..=255u8 {
                for c2 in 0..=255u8 {
                    v.push(law(f1, c1, f2, c2));
                }
            }
        }
    }
    v
}

fn compare(a: &[(u8, u8)], b: &[(u8, u8)]) -> Pair {
    let mut p = Pair::default();
    let mut i = 0;
    for _f1 in &FS {
        for _f2 in &FS {
            for c1 in 0..=255u8 {
                for c2 in 0..=255u8 {
                    let ((fa, ca), (fb, cb)) = (a[i], b[i]);
                    i += 1;
                    let df = fa.abs_diff(fb);
                    let dc = ca.abs_diff(cb);
                    p.sum_df += df as u64;
                    p.sum_dc += dc as u64;
                    p.max_df = p.max_df.max(df);
                    p.max_dc = p.max_dc.max(dc);
                    if df != 0 || dc != 0 {
                        p.differ += 1;
                        if !is_boundary(c1) && !is_boundary(c2) {
                            p.differ_broad += 1;
                            p.max_df_broad = p.max_df_broad.max(df);
                            p.max_dc_broad = p.max_dc_broad.max(dc);
                        }
                    }
                }
            }
        }
    }
    p
}

/// How many distinct output confidences a law produces over the c grid,
/// with f fixed. 1 means the law ignores confidence.
fn distinct_c(law: &Law<'_>) -> usize {
    let mut seen = [false; 256];
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            seen[law(128, c1, 128, c2).1 as usize] = true;
        }
    }
    seen.iter().filter(|s| **s).count()
}

const NAMES: [&str; 7] = ["A", "B1", "B16", "C", "D", "E", "LIN"];

fn all_surfaces() -> Vec<Vec<(u8, u8)>> {
    let t1 = NarsTables::build(1);
    let t16 = NarsTables::build(16);
    let laws: Vec<Law<'_>> = vec![
        law_a(),
        law_table(&t1),
        law_table(&t16),
        law_c(),
        law_d(),
        law_e(),
        law_lin(),
    ];
    laws.iter().map(surface).collect()
}

fn matrix(s: &[Vec<(u8, u8)>]) -> Vec<(usize, usize, Pair)> {
    let mut out = Vec::new();
    for i in 0..s.len() {
        for j in (i + 1)..s.len() {
            out.push((i, j, compare(&s[i], &s[j])));
        }
    }
    out
}

/// Distinct output confidences at f = (128, 128), per law in `NAMES` order.
const DISTINCT_C: [usize; 7] = [256, 1, 92, 256, 256, 253, 171];

/// Per pair (i, j): (cells differing, of those broad, broad max dF, broad max dC).
/// Domain: f in `FS` x `FS`, c over the full 256 x 256 grid (1,638,400 cells).
#[rustfmt::skip]
const MATRIX: [(usize, usize, u64, u64, u8, u8); 21] = [
    (0, 1, 1636550, 1598385, 128, 168),
    (0, 2, 1585273, 1547346, 113,  14),
    (0, 3,     391,     130,   1,   1),
    (0, 4,       0,       0,   0,   0),
    (0, 5,   38897,   13327,  15,   2),
    (0, 6, 1636238, 1598618, 107, 126),
    (1, 2, 1630720, 1592545, 128, 155),
    (1, 3, 1636552, 1598387, 128, 168),
    (1, 4, 1636550, 1598385, 128, 168),
    (1, 5, 1636550, 1598385, 128, 168),
    (1, 6, 1638119, 1600200, 127, 168),
    (2, 3, 1585277, 1547350, 113,  14),
    (2, 4, 1585273, 1547346, 113,  14),
    (2, 5, 1585273, 1547346, 113,  14),
    (2, 6, 1637830, 1599655, 112, 123),
    (3, 4,     391,     130,   1,   1),
    (3, 5,   39025,   13455,  15,   2),
    (3, 6, 1636240, 1598620, 107, 126),
    (4, 5,   38897,   13327,  15,   2),
    (4, 6, 1636238, 1598618, 107, 126),
    (5, 6, 1636218, 1598618, 105, 125),
];

#[test]
fn r3_report() {
    let t1 = NarsTables::build(1);
    let t16 = NarsTables::build(16);
    let laws: Vec<Law<'_>> = vec![
        law_a(),
        law_table(&t1),
        law_table(&t16),
        law_c(),
        law_d(),
        law_e(),
        law_lin(),
    ];
    for (n, l) in NAMES.iter().zip(&laws) {
        println!("distinct_c {n}: {}", distinct_c(l));
    }
    let s: Vec<_> = laws.iter().map(surface).collect();
    for (i, j, p) in matrix(&s) {
        println!(
            "{:>3} vs {:<3} differ={:>7} broad={:>7} max_dF={:>3} max_dC={:>3} \
             broad_max=({},{}) mean_dF={:.4} mean_dC={:.4}",
            NAMES[i],
            NAMES[j],
            p.differ,
            p.differ_broad,
            p.max_df,
            p.max_dc,
            p.max_df_broad,
            p.max_dc_broad,
            p.sum_df as f64 / CELLS as f64,
            p.sum_dc as f64 / CELLS as f64
        );
    }
}

#[test]
fn r3_distinct_confidence_is_pinned() {
    let t1 = NarsTables::build(1);
    let t16 = NarsTables::build(16);
    let laws: Vec<Law<'_>> = vec![
        law_a(),
        law_table(&t1),
        law_table(&t16),
        law_c(),
        law_d(),
        law_e(),
        law_lin(),
    ];
    let got: Vec<usize> = laws.iter().map(distinct_c).collect();
    assert_eq!(got, DISTINCT_C);
}

#[test]
fn r3_matrix_is_pinned() {
    let got: Vec<_> = matrix(&all_surfaces())
        .into_iter()
        .map(|(i, j, p)| {
            (
                i,
                j,
                p.differ,
                p.differ_broad,
                p.max_df_broad,
                p.max_dc_broad,
            )
        })
        .collect();
    assert_eq!(got, MATRIX.to_vec());
}

/// The falsifier the matrix answers: a pair is a cosmetic difference when no
/// broad cell differs by more than one code. Pinned so a change to any law
/// that moves a pair across that line is loud.
#[test]
fn r3_cosmetic_pairs_are_pinned() {
    // Read from the measurement, not from `MATRIX`: a check over the pinned
    // constant could never fail.
    let cosmetic: Vec<(&str, &str)> = matrix(&all_surfaces())
        .into_iter()
        .filter(|(_, _, p)| p.max_df_broad <= 1 && p.max_dc_broad <= 1)
        .map(|(i, j, _)| (NAMES[i], NAMES[j]))
        .collect();
    assert_eq!(cosmetic, vec![("A", "C"), ("A", "D"), ("C", "D")]);
}

/// Anti-vacuity: the measure is not blind. A law compared with itself is
/// zero everywhere, and a confidence-inert law (B1) is detected as far from A
/// in the broad domain.
#[test]
fn r3_falsifier_the_measure_separates_identical_from_inert() {
    let a = surface(&law_a());
    assert_eq!(compare(&a, &a), Pair::default());
    let t1 = NarsTables::build(1);
    let b1 = surface(&law_table(&t1));
    let p = compare(&a, &b1);
    assert!(p.differ_broad * 2 > CELLS, "{p:?}");
    assert!(p.max_dc_broad > 100, "{p:?}");
}
