//! The exact truth codes of deduction, induction and abduction.
//!
//! A register holds `f` and `c` as `u8` codes (`x / 255`). The conclusion is
//! the law's exact value rounded once, half up, into a code. These tests pin
//! the crate against an oracle written independently of it
//! (`common/truth_law.rs`), against hand-computed values, and against the
//! OpenNARS output for nal1.1; and they check that the register paths
//! (`execute`, `forward`, `syllogize`) pack exactly those codes.

mod common;

use causal_edge::isa::{truth, Opcode};
use causal_edge::{CausalEdge64, Compose};
use common::truth_law::{self as law, Law};

const LAWS: [(Opcode, Law); 3] = [
    (Opcode::Deduction, law::deduction),
    (Opcode::Induction, law::induction),
    (Opcode::Abduction, law::abduction),
];

fn code_fn(op: Opcode) -> Law {
    match op {
        Opcode::Deduction => truth::code::deduction,
        Opcode::Induction => truth::code::induction,
        Opcode::Abduction => truth::code::abduction,
        _ => unreachable!(),
    }
}

/// Hand-computed: f = 128·128/255 = 64.25 → 64; c = 128⁴/255³ = 16.19 → 16.
/// f = 255·200/255 = 200; c = 255·200·255·100/255³ = 78.43 → 78.
#[test]
fn deduction_matches_hand_computed_codes() {
    assert_eq!(truth::code::deduction(128, 128, 128, 128), (64, 16));
    assert_eq!(truth::code::deduction(255, 255, 200, 100), (200, 78));
}

/// The rounding is half up. All-ones evidence gives w = 1, c = 127.5 codes,
/// 128 half up (127 rounding down). That tie cannot reject ties-to-even, so
/// a second one has an even lower integer: f1 = 195 gives c = 255·195/450 =
/// 110.5 codes, 111 half up and 110 ties-to-even.
#[test]
fn a_tie_rounds_up() {
    assert_eq!(truth::code::induction(255, 255, 255, 255), (255, 128));
    assert_eq!(truth::code::abduction(255, 255, 255, 255), (255, 128));
    assert_eq!(truth::code::induction(195, 255, 255, 255), (255, 111));
    assert_eq!(truth::code::abduction(255, 255, 195, 255), (255, 111));
}

/// Induction and abduction take the frequency from opposite premises; a
/// premise swap must change the answer.
#[test]
fn induction_and_abduction_are_mirror_images() {
    let (f1, c1, f2, c2) = (255, 230, 51, 200);
    assert_eq!(
        truth::code::induction(f1, c1, f2, c2),
        truth::code::abduction(f2, c1, f1, c2)
    );
    assert_ne!(
        truth::code::induction(f1, c1, f2, c2),
        truth::code::abduction(f1, c1, f2, c2)
    );
}

#[test]
fn no_evidence_no_confidence() {
    for (op, _) in LAWS {
        let f = code_fn(op);
        assert_eq!(f(255, 0, 255, 255).1, 0, "{op:?}");
        assert_eq!(f(255, 255, 255, 0).1, 0, "{op:?}");
    }
}

/// OpenNARS at `4d1bdb0`, nal1.1: `%1.00;0.90%` premises give `%1.00;0.81%`
/// by deduction and `%1.00;0.45%` by induction. The printout has two
/// decimals; 0.9 is code 230.
#[test]
fn matches_opennars_nal1_1() {
    let two = |x: u8| format!("{:.2}", f64::from(x) / 255.0);
    let (f, c) = truth::code::deduction(255, 230, 255, 230);
    assert_eq!((two(f), two(c)), ("1.00".into(), "0.81".into()));
    let (f, c) = truth::code::induction(255, 230, 255, 230);
    assert_eq!((two(f), two(c)), ("1.00".into(), "0.45".into()));
}

fn grid() -> Vec<u8> {
    (0..=255u16).step_by(5).map(|x| x as u8).collect()
}

/// The crate's codes equal the independent oracle on a stratified grid
/// (every 5th code, 52⁴ ≈ 7.3M rows per law). The exhaustive 2³² check is
/// `examples/truth_exact_probe.rs`.
#[test]
fn codes_equal_the_oracle_on_a_grid() {
    let g = grid();
    for (op, oracle) in LAWS {
        let mine = code_fn(op);
        for &f1 in &g {
            for &c1 in &g {
                for &f2 in &g {
                    for &c2 in &g {
                        assert_eq!(
                            mine(f1, c1, f2, c2),
                            oracle(f1, c1, f2, c2),
                            "{op:?}({f1},{c1},{f2},{c2})"
                        );
                    }
                }
            }
        }
    }
}

/// The `f32` arithmetic the register paths used before: decode `x / 255`,
/// apply the law, pack with `(v · 255).round()`.
fn f32_path(op: Opcode, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    let d = |x: u8| f32::from(x) / 255.0;
    let (f, c) = op.truth(d(f1), d(c1), d(f2), d(c2));
    let p = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    (p(f), p(c))
}

/// Rows where the old `f32` path and the exact law disagree. These are the
/// rows the register tests must cover; without them a register path still on
/// `f32` would pass.
///
/// Deduction: the every-5th grid (it disagrees on 12 of its rows).
/// Induction and abduction disagree on no grid row, so their search is
/// exhaustive over the three codes their confidence depends on, with the
/// passed-through frequency fixed at 200.
fn disagreeing_rows(op: Opcode, oracle: Law) -> Vec<[u8; 4]> {
    let mut rows = Vec::new();
    let mut check = |r: [u8; 4]| {
        let [f1, c1, f2, c2] = r;
        if f32_path(op, f1, c1, f2, c2) != oracle(f1, c1, f2, c2) {
            rows.push(r);
        }
    };
    match op {
        Opcode::Deduction => {
            let g = grid();
            for &f1 in &g {
                for &c1 in &g {
                    for &f2 in &g {
                        for &c2 in &g {
                            check([f1, c1, f2, c2]);
                        }
                    }
                }
            }
        }
        Opcode::Induction | Opcode::Abduction => {
            for a in 0..=255u8 {
                for c1 in 0..=255u8 {
                    for c2 in 0..=255u8 {
                        // Induction weighs f1, abduction f2.
                        check(if op == Opcode::Induction {
                            [a, c1, 200, c2]
                        } else {
                            [200, c1, a, c2]
                        });
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    rows
}

fn edge(s: u8, o: u8, f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_s_idx(s);
    e.set_p_idx(7);
    e.set_o_idx(o);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

fn fc(e: CausalEdge64) -> (u8, u8) {
    (e.frequency_u8(), e.confidence_u8())
}

/// `execute` (what `forward` and the named register methods run) packs the
/// exact codes, on every grid row where the `f32` path was off by one.
#[test]
fn register_methods_pack_the_exact_codes() {
    let t: Box<[u8; 65536]> = vec![0u8; 65536].into_boxed_slice().try_into().unwrap();
    let c = Compose {
        s: &t,
        p: &t,
        o: &t,
    };
    for (op, oracle) in LAWS {
        let rows = disagreeing_rows(op, oracle);
        // Anti-vacuity: the f32 path really is off on these inputs.
        assert!(!rows.is_empty(), "{op:?}: no disagreeing row");
        eprintln!("{op:?}: {} rows where f32 was off", rows.len());
        for [f1, c1, f2, c2] in rows {
            let (a, b) = (edge(1, 2, f1, c1), edge(2, 3, f2, c2));
            assert_eq!(
                fc(a.execute(op, b, c)),
                oracle(f1, c1, f2, c2),
                "{op:?}({f1},{c1},{f2},{c2})"
            );
        }
    }
}

/// `syllogize` packs the exact codes for each figure's rule, premise order
/// (self, other), on the same rows.
#[test]
fn syllogize_packs_the_exact_codes() {
    for (op, oracle) in LAWS {
        for [f1, c1, f2, c2] in disagreeing_rows(op, oracle) {
            // Chain (o1 == s2) → deduction; shared subject → induction;
            // shared object → abduction.
            let (a, b) = match op {
                Opcode::Deduction => (edge(1, 2, f1, c1), edge(2, 3, f2, c2)),
                Opcode::Induction => (edge(1, 2, f1, c1), edge(1, 3, f2, c2)),
                Opcode::Abduction => (edge(1, 3, f1, c1), edge(2, 3, f2, c2)),
                _ => unreachable!(),
            };
            let s = a.syllogize(b).expect("a figure");
            assert_eq!(
                fc(s.conclusion),
                oracle(f1, c1, f2, c2),
                "{op:?}({f1},{c1},{f2},{c2})"
            );
        }
    }
}
