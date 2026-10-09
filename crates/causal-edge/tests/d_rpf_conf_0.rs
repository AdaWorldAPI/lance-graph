//! D-RPF-CONF-0 — the exhaustive confidence surface of CE64 revision.
//!
//! Measurement only. Production semantics are not changed here and no
//! confidence policy is chosen; the policy candidates below are probe
//! variants that live in this file.
//!
//! - **T1**: `CausalEdge64::revision` over all 65,536 `(c1, c2)` against the
//!   exact integer form. With integer codes `c ∈ 0..=255` and
//!   `w = c / (255 − c)`, the pooled confidence is
//!   `c_out·255 = 255·N/D` with `N = c1(255−c2) + c2(255−c1)` and
//!   `D = 255² − c1·c2`. It is defined on 65,535 cells; `(255, 255)` gives
//!   `0/0` and is the singular cell, kept out of the ordinary comparison.
//! - **T3**: revision against a zero-weight operand (confidence code 0),
//!   over every representable `(f, c)`, in both operand orders.
//! - **R1**: four c=255 policy candidates measured side by side.
//! - **NarsTables**: whether the table path's output depends on confidence.
//!
//! Every pinned set below was discovered by running this file against the
//! Rust code, then written down. The numpy model that preceded it is not the
//! source of any constant here.
#![cfg(feature = "causal-edge-v2-layout")]

use std::collections::BTreeSet;

use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use causal_edge::{CausalEdge64, IsaFault};

// ─── the production path ────────────────────────────────────────────────

fn edge(f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

/// `(f, c)` codes of `edge(f1, c1).revision(edge(f2, c2))`.
fn prod(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    let r = edge(f1, c1).revision(edge(f2, c2));
    (r.frequency_u8(), r.confidence_u8())
}

// ─── the exact integer reference ────────────────────────────────────────

/// The exact pooled confidence code, rounded half away from zero (the
/// production rounding of a non-negative value), and whether the exact value
/// sits on a `.5` tie. `None` at `(255, 255)`, where `N = D = 0`.
fn exact_c(c1: u8, c2: u8) -> Option<(u8, bool)> {
    let (c1, c2) = (u64::from(c1), u64::from(c2));
    let n = c1 * (255 - c2) + c2 * (255 - c1);
    let d = 255 * 255 - c1 * c2;
    if d == 0 {
        return None;
    }
    let code = (2 * 255 * n + d) / (2 * d);
    let tie = (2 * 255 * n) % (2 * d) == d;
    Some((code as u8, tie))
}

/// The exact pooled frequency code, and whether it is a `.5` tie. `None`
/// when neither side carries evidence (`N = 0`), which includes `(255, 255)`.
fn exact_f(f1: u8, c1: u8, f2: u8, c2: u8) -> Option<(u8, bool)> {
    let (f1, c1, f2, c2) = (u64::from(f1), u64::from(c1), u64::from(f2), u64::from(c2));
    let a = c1 * (255 - c2);
    let b = c2 * (255 - c1);
    let n = a + b;
    if n == 0 {
        return None;
    }
    let num = f1 * a + f2 * b;
    let code = (2 * num + n) / (2 * n);
    let tie = (2 * num) % (2 * n) == n;
    Some((code as u8, tie))
}

// ─── T1 ─────────────────────────────────────────────────────────────────

/// One cell `(c1, c2, got, want)` where a confidence surface disagrees with
/// its reference.
type Dev = (u8, u8, u8, u8);

#[derive(Debug, PartialEq, Eq)]
struct Surface {
    compared: usize,
    deviations: Vec<Dev>,
    max_delta: u8,
    ties: Vec<(u8, u8)>,
    monotonicity_violations: Vec<(u8, u8)>,
    symmetry_violations: usize,
    singular: (u8, u8),
}

/// Measures a confidence surface `c(c1, c2)` against a reference. The
/// surface is evaluated at frequency 128 on both sides; `t1_confidence_*`
/// checks that it does not depend on frequency.
fn measure_c(
    surface: impl Fn(u8, u8) -> u8,
    reference: impl Fn(u8, u8) -> Option<(u8, bool)>,
) -> Surface {
    let mut grid = vec![[0u8; 256]; 256];
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            grid[c1 as usize][c2 as usize] = surface(c1, c2);
        }
    }
    let mut s = Surface {
        compared: 0,
        deviations: Vec::new(),
        max_delta: 0,
        ties: Vec::new(),
        monotonicity_violations: Vec::new(),
        symmetry_violations: 0,
        singular: (0, 0),
    };
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            let got = grid[c1 as usize][c2 as usize];
            if grid[c2 as usize][c1 as usize] != got {
                s.symmetry_violations += 1;
            }
            // Non-decreasing in c1 for fixed c2: more evidence never lowers
            // the pooled confidence.
            // The singular cell is kept out: it has no ordinary neighbour
            // relation and is reported on its own.
            if c1 > 0
                && reference(c1, c2).is_some()
                && reference(c1 - 1, c2).is_some()
                && got < grid[c1 as usize - 1][c2 as usize]
            {
                s.monotonicity_violations.push((c1, c2));
            }
            match reference(c1, c2) {
                None => s.singular = (c1, c2),
                Some((want, tie)) => {
                    s.compared += 1;
                    if tie {
                        s.ties.push((c1, c2));
                    }
                    if got != want {
                        s.deviations.push((c1, c2, got, want));
                        s.max_delta = s.max_delta.max(got.abs_diff(want));
                    }
                }
            }
        }
    }
    s
}

/// What the current Rust revision does, discovered by running `t1_report`
/// and then written down.
///
/// The numpy f32 model that preceded this file predicted
/// `{(45,45), (85,165), (165,85)}`. Rust disagrees on two of the three:
/// `(85,165)` is an exact tie that Rust rounds correctly, and
/// `(146,148)`/`(148,146)` are not ties at all. Their exact value is
/// `8097270/43417 ≈ 186.499988`; the f32 path lands just above `.5` and rounds
/// up. `(45,45)` is an exact tie (`76.5`) that f32 lands just below.
const T1_DEVIATIONS: &[Dev] = &[(45, 45, 76, 77), (146, 148, 187, 186), (148, 146, 187, 186)];

/// The ordinary cells whose exact value is a `.5` tie.
const T1_TIES: &[(u8, u8)] = &[
    (3, 75),
    (45, 45),
    (75, 3),
    (85, 85),
    (85, 165),
    (153, 225),
    (165, 85),
    (225, 153),
];

fn prod_c(c1: u8, c2: u8) -> u8 {
    prod(128, c1, 128, c2).1
}

#[test]
fn t1_report() {
    let s = measure_c(prod_c, exact_c);
    eprintln!("T1 ordinary cells compared: {}", s.compared);
    eprintln!("T1 deviations: {:?}", s.deviations);
    eprintln!("T1 max delta: {}", s.max_delta);
    eprintln!("T1 ties: {:?}", s.ties);
    eprintln!(
        "T1 monotonicity violations: {:?}",
        s.monotonicity_violations
    );
    eprintln!("T1 symmetry violations: {}", s.symmetry_violations);
    let (c1, c2) = s.singular;
    eprintln!(
        "T1 singular ({c1},{c2}): current -> {:?}",
        prod(128, c1, 128, c2)
    );
}

#[test]
fn t1_confidence_surface_is_pinned() {
    let s = measure_c(prod_c, exact_c);
    assert_eq!(s.compared, 65_535, "ordinary domain");
    assert_eq!(s.singular, (255, 255), "the one undefined cell");
    assert_eq!(s.deviations, T1_DEVIATIONS);
    assert_eq!(s.ties, T1_TIES);
    assert!(s.monotonicity_violations.is_empty());
    assert_eq!(s.symmetry_violations, 0);
}

#[test]
fn t1_confidence_does_not_depend_on_frequency() {
    // `ws` never reads f, so c(c1, c2) must be the same for every frequency
    // pair. A probe that silently depended on f = 128 would hide a defect.
    for &(f1, f2) in &[(0u8, 0u8), (255, 255), (0, 255), (77, 200)] {
        for c1 in 0..=255u8 {
            for c2 in 0..=255u8 {
                if (c1, c2) == (255, 255) {
                    continue; // NaN path: f and c both collapse; see t1_singular
                }
                assert_eq!(
                    prod(f1, c1, f2, c2).1,
                    prod_c(c1, c2),
                    "({f1},{c1},{f2},{c2})"
                );
            }
        }
    }
}

#[test]
fn t1_singular_cell_is_recorded_not_endorsed() {
    // Current behaviour only: both operands at 255 pool two f32::MAX weights,
    // overflow to infinity, compute inf/inf = NaN, and the saturating cast
    // stores 0 for both fields. That 0 is not claimed to be correct.
    for &(f1, f2) in &[(0u8, 0u8), (128, 128), (255, 255), (10, 250)] {
        assert_eq!(prod(f1, 255, f2, 255), (0, 0), "f1={f1} f2={f2}");
    }
    assert_eq!(exact_c(255, 255), None);
    assert_eq!(exact_f(0, 255, 0, 255), None);
}

#[test]
fn t1_falsifier_a_perturbed_reference_changes_the_set() {
    // Anti-vacuity: if the comparison were blind (e.g. both sides computed by
    // the same function), corrupting the reference would not move the result.
    let perturbed = |c1: u8, c2: u8| {
        exact_c(c1, c2).map(|(v, t)| {
            if (c1, c2) == (100, 37) {
                (v ^ 1, t)
            } else {
                (v, t)
            }
        })
    };
    let s = measure_c(prod_c, perturbed);
    assert_ne!(s.deviations, T1_DEVIATIONS);
    assert!(s.deviations.iter().any(|d| (d.0, d.1) == (100, 37)));
}

#[test]
fn t1_frequency_side_against_exact_on_a_confidence_sample() {
    // The full f-side is 2^32 cells; this samples the confidence plane and
    // takes every (f1, f2). Reported, then pinned.
    let cs = [1u8, 64, 128, 200, 254];
    let mut dev = 0usize;
    let mut ties = 0usize;
    let mut tie_dev = 0usize;
    let mut max_delta = 0u8;
    let mut compared = 0usize;
    for &c1 in &cs {
        for &c2 in &cs {
            for f1 in 0..=255u8 {
                for f2 in 0..=255u8 {
                    let (want, tie) = exact_f(f1, c1, f2, c2).expect("evidence on both sides");
                    let got = prod(f1, c1, f2, c2).0;
                    compared += 1;
                    ties += usize::from(tie);
                    if got != want {
                        dev += 1;
                        tie_dev += usize::from(tie);
                        max_delta = max_delta.max(got.abs_diff(want));
                    }
                }
            }
        }
    }
    eprintln!(
        "T1-f compared {compared} deviations {dev} (of which ties {tie_dev}) ties {ties} max delta {max_delta}"
    );
    assert_eq!(compared, 25 * 65_536);
    // Every f-side deviation on this sample is an exact .5 tie that f32 lands
    // just below; no non-tie cell deviates.
    assert_eq!((dev, tie_dev, ties, max_delta), (6_932, 6_932, 163_840, 1));
}

// ─── T3 ─────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    drift: Vec<(u8, u8, u8, u8)>, // (f, c, got_f, got_c)
    max_delta: u8,
}

/// Revision of `x = (f, c)` against a zero-weight operand `(g, 0)`, both
/// orders, every representable `x`, three values of `g`.
fn measure_identity(rev: impl Fn(u8, u8, u8, u8) -> (u8, u8)) -> Identity {
    let mut out = Identity {
        drift: Vec::new(),
        max_delta: 0,
    };
    for f in 0..=255u8 {
        for c in 0..=255u8 {
            let first = rev(f, c, 0, 0);
            for &g in &[0u8, 128, 255] {
                let l = rev(f, c, g, 0);
                let r = rev(g, 0, f, c);
                // The zero-weight operand's frequency must not matter, and
                // the operand order must not matter.
                assert_eq!(l, first, "g leaks into ({f},{c})");
                assert_eq!(r, first, "order leaks into ({f},{c})");
            }
            if first != (f, c) {
                out.drift.push((f, c, first.0, first.1));
                out.max_delta = out
                    .max_delta
                    .max(first.0.abs_diff(f))
                    .max(first.1.abs_diff(c));
            }
        }
    }
    out
}

#[test]
fn t3_report() {
    let id = measure_identity(prod);
    let mut by_c: BTreeSet<u8> = BTreeSet::new();
    for d in &id.drift {
        by_c.insert(d.1);
    }
    eprintln!("T3 drift cells: {}", id.drift.len());
    eprintln!("T3 max delta: {}", id.max_delta);
    eprintln!("T3 confidence codes with drift: {:?}", by_c);
    eprintln!(
        "T3 first drift cells: {:?}",
        &id.drift[..id.drift.len().min(12)]
    );
}

#[test]
fn t3_zero_weight_identity_is_pinned() {
    let id = measure_identity(prod);
    // The only drift is semantic, not quantisation: with confidence 0 on both
    // sides there is no evidence, revision returns unknown `(0.5, 0.0)`, and
    // every frequency except 128 moves to 128. For every c > 0 the zero-weight
    // operand is an exact identity (0 off-by-one cells).
    assert_eq!(id.drift.len(), 255);
    assert!(id
        .drift
        .iter()
        .all(|&(f, c, gf, gc)| c == 0 && f != 128 && (gf, gc) == (128, 0)));
    assert_eq!(id.max_delta, 128);
}

#[test]
fn t3_falsifier_an_injected_drift_is_detected() {
    let base = measure_identity(prod);
    let drifting = |f1: u8, c1: u8, f2: u8, c2: u8| {
        let (f, c) = prod(f1, c1, f2, c2);
        // One code of drift on one ordinary cell, whichever operand it is.
        if (f1, c1) == (90, 40) || (f2, c2) == (90, 40) {
            (f.wrapping_add(1), c)
        } else {
            (f, c)
        }
    };
    let id = measure_identity(drifting);
    assert_eq!(id.drift.len(), base.drift.len() + 1);
    assert!(id.drift.iter().any(|d| (d.0, d.1) == (90, 40)));
}

#[test]
fn t3_forward_with_an_all_zero_weight_faults() {
    // The documented null operand of `forward`: mantissa 0 decodes to no
    // instruction, so it must refuse rather than run something.
    let tab = Box::new([0u8; 256 * 256]);
    let got = edge(200, 200).forward(CausalEdge64(0), &tab, &tab, &tab);
    assert_eq!(got, Err(IsaFault::Unsupported { mantissa: 0 }));
}

// ─── R1: c=255 policy candidates (probe variants, not production) ───────

/// Candidate policies for confidence code 255. None of them is adopted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    /// A: the current code path, replicated step by step.
    StatusQuo,
    /// B: cap the decoded confidence at 0.9999 and compute `c / (1 − c)`
    /// directly, as the f64 reference in `ce64_isa_golden.rs` does. (Capping
    /// first and then calling `evidence_weight` would change nothing, because
    /// 0.9999 is still above its 0.999 saturation threshold.)
    CapBeforeWeight,
    /// B2: read input code 255 as 254 before the weight; outputs unchanged.
    CapInputAt254,
    /// C: 255 is not a representable confidence: inputs read 255 as 254 and
    /// outputs are clamped to 254.
    Saturate254,
    /// D: status quo everywhere except the singular cell, which gets a
    /// declared answer: `c = 255` and `f` = the mean of the two frequencies.
    /// Added after the first R1 run showed that every cap candidate changes
    /// the one-sided boundary, where the status quo is already exact. It
    /// isolates "define one cell" from "change the c→w map". Not adopted.
    SingularOnly,
}

const POLICIES: [Policy; 5] = [
    Policy::StatusQuo,
    Policy::CapBeforeWeight,
    Policy::CapInputAt254,
    Policy::Saturate254,
    Policy::SingularOnly,
];

/// `(f_code, c_code, produced_nan)` for one revision under `pol`, in the same
/// f32 arithmetic and the same order as `isa::truth::revision` + `set_*`.
fn revise_with(pol: Policy, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8, bool) {
    if pol == Policy::SingularOnly && (c1, c2) == (255, 255) {
        let f = (f32::from(f1) / 255.0 + f32::from(f2) / 255.0) / 2.0;
        return ((f * 255.0).round() as u8, 255, false);
    }
    let read = |c: u8| match pol {
        Policy::CapInputAt254 | Policy::Saturate254 => c.min(254),
        _ => c,
    };
    let (f1, f2) = (f32::from(f1) / 255.0, f32::from(f2) / 255.0);
    let (c1, c2) = (f32::from(read(c1)) / 255.0, f32::from(read(c2)) / 255.0);
    let weight = |c: f32| match pol {
        Policy::CapBeforeWeight => {
            let c = c.min(0.9999);
            c / (1.0 - c)
        }
        _ => causal_edge::isa::truth::evidence_weight(c),
    };
    let (w1, w2) = (weight(c1), weight(c2));
    let ws = w1 + w2;
    let (f, c) = if ws > f32::EPSILON {
        ((f1 * w1 + f2 * w2) / ws, ws / (ws + 1.0))
    } else {
        (0.5, 0.0)
    };
    let nan = f.is_nan() || c.is_nan();
    let enc = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    let (fo, mut co) = (enc(f), enc(c));
    if pol == Policy::Saturate254 {
        co = co.min(254);
    }
    (fo, co, nan)
}

#[derive(Debug)]
struct PolicyRow {
    nan_c_surface: usize,
    nan_f_boundary: usize,
    monotonicity: usize,
    symmetry: usize,
    exact_mismatch: usize,
    exact_max_err: u8,
    changed_vs_status_quo: usize,
    singular: (u8, u8),
    /// f-side on the one-sided boundary `c1 = 255`, `c2 < 255` (ordinary
    /// cells: the exact limit is `f = f1`): mismatches against the exact
    /// form, max error, and cells changed against the status quo.
    f_onesided_exact_mismatch: usize,
    f_onesided_max_err: u8,
    f_onesided_changed: usize,
}

fn measure_policy(rev: impl Fn(u8, u8, u8, u8) -> (u8, u8, bool)) -> PolicyRow {
    let mut grid = vec![[0u8; 256]; 256];
    let mut row = PolicyRow {
        nan_c_surface: 0,
        nan_f_boundary: 0,
        monotonicity: 0,
        symmetry: 0,
        exact_mismatch: 0,
        exact_max_err: 0,
        changed_vs_status_quo: 0,
        singular: (0, 0),
        f_onesided_exact_mismatch: 0,
        f_onesided_max_err: 0,
        f_onesided_changed: 0,
    };
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            let (_, c, nan) = rev(128, c1, 128, c2);
            grid[c1 as usize][c2 as usize] = c;
            row.nan_c_surface += usize::from(nan);
            if c != prod_c(c1, c2) {
                row.changed_vs_status_quo += 1;
            }
            match exact_c(c1, c2) {
                Some((want, _)) if c != want => {
                    row.exact_mismatch += 1;
                    row.exact_max_err = row.exact_max_err.max(c.abs_diff(want));
                }
                Some(_) => {}
                None => row.singular = (rev(128, c1, 128, c2).0, c),
            }
        }
    }
    #[allow(clippy::needless_range_loop)] // both (c1,c2) and (c2,c1) index the grid
    for c1 in 0..=255usize {
        for c2 in 0..=255usize {
            // Here the singular cell IS included: a policy is judged on the
            // whole surface, and a dip at (255,255) is exactly what it must fix.
            if c1 > 0 && grid[c1][c2] < grid[c1 - 1][c2] {
                row.monotonicity += 1;
            }
            if grid[c1][c2] != grid[c2][c1] {
                row.symmetry += 1;
            }
        }
    }
    for c1 in 253..=255u8 {
        for c2 in 253..=255u8 {
            for f1 in 0..=255u8 {
                for f2 in 0..=255u8 {
                    row.nan_f_boundary += usize::from(rev(f1, c1, f2, c2).2);
                }
            }
        }
    }
    for c2 in [0u8, 1, 64, 127, 200, 253, 254] {
        for f1 in 0..=255u8 {
            for f2 in 0..=255u8 {
                let got = rev(f1, 255, f2, c2).0;
                let want = exact_f(f1, 255, f2, c2).expect("one-sided evidence").0;
                if got != want {
                    row.f_onesided_exact_mismatch += 1;
                    row.f_onesided_max_err = row.f_onesided_max_err.max(got.abs_diff(want));
                }
                row.f_onesided_changed += usize::from(got != prod(f1, 255, f2, c2).0);
            }
        }
    }
    row
}

#[test]
fn r1_replica_of_status_quo_is_faithful() {
    // The policy variants are only meaningful if the replica they modify IS
    // the production path: every (f1,c1,f2,c2) on the boundary block and the
    // whole c-surface must agree bit for bit.
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            let (f, c, _) = revise_with(Policy::StatusQuo, 128, c1, 128, c2);
            assert_eq!((f, c), prod(128, c1, 128, c2), "({c1},{c2})");
        }
    }
    for c1 in 250..=255u8 {
        for c2 in 250..=255u8 {
            for f1 in (0..=255u8).step_by(5) {
                for f2 in (0..=255u8).step_by(3) {
                    let (f, c, _) = revise_with(Policy::StatusQuo, f1, c1, f2, c2);
                    assert_eq!((f, c), prod(f1, c1, f2, c2));
                }
            }
        }
    }
}

#[test]
fn r1_report() {
    eprintln!("R1 policy            NaN(c) NaN(f@253-255) mono sym exact-mismatch max-err changed singular | f@c1=255: exact-mismatch max-err changed");
    for pol in POLICIES {
        let r = measure_policy(|a, b, c, d| revise_with(pol, a, b, c, d));
        eprintln!(
            "R1 {:<18} {:>6} {:>14} {:>4} {:>3} {:>14} {:>7} {:>7} {:?} | {:>6} {:>4} {:>6}",
            format!("{pol:?}"),
            r.nan_c_surface,
            r.nan_f_boundary,
            r.monotonicity,
            r.symmetry,
            r.exact_mismatch,
            r.exact_max_err,
            r.changed_vs_status_quo,
            r.singular,
            r.f_onesided_exact_mismatch,
            r.f_onesided_max_err,
            r.f_onesided_changed
        );
    }
    for pol in POLICIES {
        let mut line = format!("R1 boundary {:<18}", format!("{pol:?}"));
        for c1 in [253u8, 254, 255] {
            for c2 in [0u8, 1, 127, 253, 254, 255] {
                line += &format!(" {c1}/{c2}={}", revise_with(pol, 128, c1, 128, c2).1);
            }
        }
        eprintln!("{line}");
    }
}

#[test]
fn r1_falsifier_an_invalid_boundary_is_detected() {
    // A candidate that lowers confidence when an operand reaches 255 must be
    // caught by the monotonicity gate; one that returns NaN must be counted.
    let dip = measure_policy(|f1, c1, f2, c2| {
        let (f, c, n) = revise_with(Policy::CapInputAt254, f1, c1, f2, c2);
        if c1 == 255 && c2 == 10 {
            (f, 0, n)
        } else {
            (f, c, n)
        }
    });
    assert!(dip.monotonicity > 0);
    let nan = measure_policy(|f1, c1, f2, c2| {
        let (f, c, _) = revise_with(Policy::CapInputAt254, f1, c1, f2, c2);
        (f, c, c1 == 254 && c2 == 254)
    });
    assert!(nan.nan_c_surface > 0 && nan.nan_f_boundary > 0);
}

// ─── NarsTables: does the table path read confidence? ───────────────────

/// Distinct `(f, c)` outputs over all 65,536 `(c1, c2)` for fixed `(f1, f2)`.
fn distinct_over_confidence(rev: impl Fn(u8, u8, u8, u8) -> (u8, u8), f1: u8, f2: u8) -> usize {
    let mut seen = BTreeSet::new();
    for c1 in 0..=255u8 {
        for c2 in 0..=255u8 {
            seen.insert(rev(f1, c1, f2, c2));
        }
    }
    seen.len()
}

/// The table path as a revision function on codes.
fn table_rev(t: &NarsTables) -> impl Fn(u8, u8, u8, u8) -> (u8, u8) + '_ {
    move |f1, c1, f2, c2| {
        let p = t.revise(f1, c1, f2, c2);
        (unpack_f(p), unpack_c(p))
    }
}

#[test]
fn nars_tables_report_and_pin() {
    let t1 = NarsTables::build(1);
    let t16 = NarsTables::build(16);
    let via = table_rev;
    let pairs = [(0u8, 255u8), (64, 192), (200, 30), (128, 128)];
    for &(f1, f2) in &pairs {
        let d1 = distinct_over_confidence(via(&t1), f1, f2);
        let d16 = distinct_over_confidence(via(&t16), f1, f2);
        let dp = distinct_over_confidence(prod, f1, f2);
        eprintln!("NarsTables f=({f1},{f2}): build(1) {d1} | build(16) {d16} | isa revision {dp}");
        // build(1): every confidence lands in bucket 0, so the output is one
        // constant pair for any (c1, c2). Confidence is inert.
        assert_eq!(d1, 1, "build(1) must be confidence-blind");
        assert!(d16 > 1 && dp > d16, "confidence-aware paths must vary");
    }
    // What the one constant pair is: both representative confidences are 0.5,
    // so c = 2/3 -> 170 and f = the mean of the two codes.
    let (f, c) = via(&t1)(0, 7, 255, 250);
    assert_eq!((f, c), (128, 170));
}

#[test]
fn r1_table_is_pinned() {
    // (NaN c-surface, NaN f-boundary, monotonicity, symmetry, exact mismatch,
    //  exact max err, changed cells, singular (f,c), one-sided f: exact
    //  mismatch, max err, changed). Measured; no policy is chosen by this.
    type Row = (
        usize,
        usize,
        usize,
        usize,
        usize,
        u8,
        usize,
        (u8, u8),
        usize,
        u8,
        usize,
    );
    let want: [(Policy, Row); 5] = [
        (
            Policy::StatusQuo,
            (1, 65_536, 1, 0, 3, 1, 0, (0, 0), 0, 0, 0),
        ),
        (
            Policy::CapBeforeWeight,
            (0, 0, 0, 0, 3, 1, 1, (128, 255), 101_900, 6, 101_900),
        ),
        (
            Policy::CapInputAt254,
            (0, 0, 0, 0, 513, 1, 511, (128, 254), 194_671, 128, 194_671),
        ),
        (
            Policy::Saturate254,
            (0, 0, 0, 0, 513, 1, 511, (128, 254), 194_671, 128, 194_671),
        ),
        (
            Policy::SingularOnly,
            (0, 0, 0, 0, 3, 1, 1, (128, 255), 0, 0, 0),
        ),
    ];
    for (pol, w) in want {
        let r = measure_policy(|a, b, c, d| revise_with(pol, a, b, c, d));
        let got = (
            r.nan_c_surface,
            r.nan_f_boundary,
            r.monotonicity,
            r.symmetry,
            r.exact_mismatch,
            r.exact_max_err,
            r.changed_vs_status_quo,
            r.singular,
            r.f_onesided_exact_mismatch,
            r.f_onesided_max_err,
            r.f_onesided_changed,
        );
        assert_eq!(got, w, "{pol:?}");
    }
}
