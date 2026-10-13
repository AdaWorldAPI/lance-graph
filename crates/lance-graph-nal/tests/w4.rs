//! W4 (D-NALP-4): the per-row truth laws a lane-producing terminal must meet.
//!
//! The oracle (`common/truth_law.rs`) computes each law exactly in integers
//! and rounds once. These tests pin it against values computed by hand, the
//! OpenNARS outputs recorded in `.claude/knowledge/nal-recipe-audit/oracles.md`
//! (pinned `4d1bdb0`), and `causal_edge::isa::truth`, the in-tree scalar
//! formulas it must agree with up to `f32` rounding.

mod common;

use common::truth_law::{abduction, analogy, deduction, induction, LAWS};

fn decode(code: u8) -> f64 {
    f64::from(code) / 255.0
}

/// Hand-computed: f = 128·128/255 = 64.25 → 64; c = 128⁴/255³ = 16.19 → 16.
#[test]
fn deduction_matches_hand_computed_codes() {
    assert_eq!(deduction(128, 128, 128, 128), (64, 16));
    // f = 255·200/255 = 200; c = 255·200·255·100/255³ = 78.43 → 78.
    assert_eq!(deduction(255, 255, 200, 100), (200, 78));
}

/// Analogy: f = f1·f2, c = c1·c2·f2. With f2 = 0 the conclusion carries no
/// confidence, while deduction with the same premises also has none: the two
/// differ where f1 = 0 instead (analogy keeps c, deduction drops it).
#[test]
fn analogy_and_deduction_differ_on_the_first_frequency() {
    assert_eq!(analogy(0, 200, 255, 200), (0, 157));
    assert_eq!(deduction(0, 200, 255, 200), (0, 0));
}

/// Induction and abduction take the conclusion's frequency from opposite
/// premises and weigh the other one. Swapping premises must change the
/// answer, or a direction bug would pass unseen (the W5 hazard, ledger R1).
#[test]
fn induction_and_abduction_are_mirror_images() {
    let (f1, c1, f2, c2) = (255, 230, 51, 200);
    assert_eq!(induction(f1, c1, f2, c2), abduction(f2, c1, f1, c2));
    assert_ne!(induction(f1, c1, f2, c2), abduction(f1, c1, f2, c2));
}

/// The round-once rule is half up. All-ones evidence gives w = 1, so
/// c = 1/2 = 127.5 codes, which rounds to 128; rounding down gives 127. That
/// tie cannot reject ties-to-even (128 is even), so a second tie has an even
/// lower integer: f1 = 195 gives c = 255·195/450 = 110.5 codes, 111 half up
/// and 110 ties-to-even.
#[test]
fn a_tie_rounds_up() {
    assert_eq!(induction(255, 255, 255, 255), (255, 128));
    assert_eq!(abduction(255, 255, 255, 255), (255, 128));
    assert_eq!(induction(195, 255, 255, 255), (255, 111));
    assert_eq!(abduction(255, 255, 195, 255), (255, 111));
}

/// Zero evidence yields zero confidence for every law.
#[test]
fn no_evidence_no_confidence() {
    for (name, law) in LAWS {
        assert_eq!(law(255, 0, 255, 255).1, 0, "{name}");
        assert_eq!(law(255, 255, 255, 0).1, 0, "{name}");
    }
}

/// OpenNARS at `4d1bdb0`, nal1.1: `<bird --> animal>. %1.00;0.90%` and
/// `<robin --> bird>. %1.00;0.90%` give `<robin --> animal>. %1.00;0.81%`
/// (deduction) and `<animal --> robin>. %1.00;0.45%` (the weak reverse). The
/// printout has two decimals; 0.9 is code 230.
#[test]
fn matches_opennars_nal1_1() {
    let (f, c) = deduction(255, 230, 255, 230);
    assert_eq!(
        (format!("{:.2}", decode(f)), format!("{:.2}", decode(c))),
        ("1.00".into(), "0.81".into())
    );
    let (f, c) = induction(255, 230, 255, 230);
    assert_eq!(
        (format!("{:.2}", decode(f)), format!("{:.2}", decode(c))),
        ("1.00".into(), "0.45".into())
    );
}

/// Pack the way `CausalEdge64` does: `(v.clamp(0, 1) · 255).round()`.
fn pack(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn isa(name: &str, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    use causal_edge::isa::truth;
    let d = |x: u8| f32::from(x) / 255.0;
    let (f, c) = match name {
        "deduction" => truth::deduction(d(f1), d(c1), d(f2), d(c2)),
        "induction" => truth::induction(d(f1), d(c1), d(f2), d(c2)),
        "abduction" => truth::abduction(d(f1), d(c1), d(f2), d(c2)),
        // `isa::truth` has no analogy; the planner's `TruthValue::analogy`
        // formula, in the same f32 arithmetic.
        "analogy" => (d(f1) * d(f2), d(c1) * d(c2) * d(f2)),
        _ => unreachable!(),
    };
    (pack(f), pack(c))
}

/// The exact law and the in-tree `f32` formulas agree within one code on a
/// stratified grid (every 5th code, ≈7.3M rows per law). One code is the most
/// a single `f32` rounding near a half can move.
#[test]
fn agrees_with_isa_truth_within_one_code() {
    let grid: Vec<u8> = (0..=255u16).step_by(5).map(|x| x as u8).collect();
    for (name, law) in LAWS {
        let mut differ = 0u64;
        for &f1 in &grid {
            for &c1 in &grid {
                for &f2 in &grid {
                    for &c2 in &grid {
                        let (ef, ec) = law(f1, c1, f2, c2);
                        let (sf, sc) = isa(name, f1, c1, f2, c2);
                        let df = (i16::from(ef) - i16::from(sf)).abs();
                        let dc = (i16::from(ec) - i16::from(sc)).abs();
                        assert!(
                            df <= 1 && dc <= 1,
                            "{name}({f1},{c1},{f2},{c2}): exact ({ef},{ec}) isa ({sf},{sc})"
                        );
                        differ += u64::from(df + dc > 0);
                    }
                }
            }
        }
        eprintln!("{name}: {differ} grid rows differ by one code");
    }
}
