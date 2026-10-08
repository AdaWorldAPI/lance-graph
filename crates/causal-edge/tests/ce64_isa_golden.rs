//! CE64 ISA golden vectors on raw `u64` words.
//!
//! Two sets, kept apart on purpose:
//!
//! - **Legacy capture** (`ce64_legacy_capture.txt`): what `forward`, `learn`
//!   and `syllogize` returned BEFORE the ISA decoder, defects included,
//!   recorded from that code. The ledger test says exactly which of them the
//!   ISA changed (only the unsupported opcodes, which now refuse) and that
//!   every other word is reproduced bit for bit.
//! - **Normative vectors**: the subset accepted as canonical, checked through
//!   the register methods (`a.deduction(b, ..)` etc.) and, independently,
//!   against an f64 reference of the NARS truth functions. The max-confidence
//!   revision defect is NOT in this set; it is pinned as
//!   [`KNOWN_BAD_REVISION_MAX_CONFIDENCE`].
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::isa::{Field, IsaFault, Opcode};
use causal_edge::{CausalEdge64, Compose};

const LEGACY: &str = include_str!("ce64_legacy_capture.txt");

/// `(a, b)` operands of the max-confidence revision defect: both edges at
/// confidence 255, weight opcode Revision. The legacy path, and the current
/// one, return frequency 0 and confidence 0.
const KNOWN_BAD_REVISION_MAX_CONFIDENCE: (u64, u64) =
    (0x6ab8_2fff_ff03_0201, 0xb54d_13ff_ff06_0504);

fn tables() -> [Box<[u8; 65536]>; 3] {
    let mk = |a: usize, b: usize| {
        let mut t = vec![0u8; 65536].into_boxed_slice();
        for x in 0..256 {
            for y in 0..256 {
                t[x * 256 + y] = (x * a + y * b) as u8;
            }
        }
        t.try_into().unwrap()
    };
    [mk(31, 17), mk(7, 13), mk(11, 29)]
}

fn compose(t: &[Box<[u8; 65536]>; 3]) -> Compose<'_> {
    Compose {
        s: &t[0],
        p: &t[1],
        o: &t[2],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Forward(u8),
    Learn,
    Syllogize(u8),
}

struct Vector {
    kind: Kind,
    a: u64,
    b: u64,
    expected: u64,
}

fn legacy() -> Vec<Vector> {
    LEGACY
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
            let kind = match f[0] {
                "F" => Kind::Forward(f[1].parse().unwrap()),
                "L" => Kind::Learn,
                _ => Kind::Syllogize(f[1].parse().unwrap()),
            };
            Vector {
                kind,
                a: hex(f[2]),
                b: hex(f[3]),
                expected: hex(f[4]),
            }
        })
        .collect()
}

fn sign_extend(nibble: u8) -> i8 {
    ((nibble << 4) as i8) >> 4
}

fn run(v: &Vector, c: Compose<'_>) -> Result<u64, IsaFault> {
    let (a, b) = (CausalEdge64(v.a), CausalEdge64(v.b));
    Ok(match v.kind {
        Kind::Forward(_) => a.forward(b, c.s, c.p, c.o)?.0,
        Kind::Learn => {
            let mut e = a;
            e.learn(b, 0);
            e.0
        }
        Kind::Syllogize(_) => a.syllogize(b).unwrap().conclusion.0,
    })
}

/// Revision with a c = 255 operand is outside the normative domain until the
/// saturation rule is decided: both operands at 255 collapse to (0, 0)
/// ([`KNOWN_BAD_REVISION_MAX_CONFIDENCE`]); one operand at 255 takes that
/// side's frequency outright (its weight saturates to `f32::MAX`), where the
/// capped f64 reference gives a weighted mean.
fn is_known_bad(v: &Vector) -> bool {
    let revises = matches!(v.kind, Kind::Learn | Kind::Forward(4));
    revises && (Field::Confidence.get(v.a) == 255 || Field::Confidence.get(v.b) == 255)
}

// ─── Legacy ledger ──────────────────────────────────────────────────────

/// Every pre-ISA word is reproduced exactly, except the forward rows whose
/// weight carries an opcode with no implementation: those now refuse with
/// that exact code. Nothing else moved.
#[test]
fn the_isa_changed_exactly_the_unsupported_opcodes() {
    let t = tables();
    let vectors = legacy();
    assert_eq!(vectors.len(), 126, "capture file");
    let (mut same, mut refused) = (0, 0);
    for v in &vectors {
        match (v.kind, run(v, compose(&t))) {
            (_, Ok(w)) => {
                assert_eq!(w, v.expected, "{:?} {:#018x} {:#018x}", v.kind, v.a, v.b);
                same += 1;
            }
            (Kind::Forward(n), Err(IsaFault::Unsupported { mantissa })) => {
                assert_eq!(mantissa, sign_extend(n));
                assert!(Opcode::decode(mantissa).is_err());
                refused += 1;
            }
            (k, Err(e)) => panic!("{k:?} faulted unexpectedly: {e}"),
        }
    }
    // 16 nibbles, 5 supported: 11 x 5 forward rows refuse.
    assert_eq!((same, refused), (126 - 55, 55));
}

// ─── Normative vectors through the register methods ─────────────────────

fn opcode_of(nibble: u8) -> Option<Opcode> {
    Opcode::decode(sign_extend(nibble)).ok()
}

/// Every accepted forward vector, reproduced through the named register
/// method, not through `forward`'s decoder.
#[test]
fn normative_vectors_reproduce_through_register_methods() {
    let t = tables();
    let c = compose(&t);
    let mut n = 0;
    for v in legacy() {
        let Kind::Forward(nib) = v.kind else { continue };
        let Some(op) = opcode_of(nib) else { continue };
        if is_known_bad(&v) {
            continue;
        }
        let (a, b) = (CausalEdge64(v.a), CausalEdge64(v.b));
        let out = match op {
            Opcode::Deduction => a.deduction(b, c),
            Opcode::Induction => a.induction(b, c),
            Opcode::Abduction => a.abduction(b, c),
            Opcode::Revision => a.execute(Opcode::Revision, b, c),
            Opcode::Synthesis => a.synthesis(b, c),
        };
        assert_eq!(out.0, v.expected, "{op:?}");
        n += 1;
    }
    // 25 executable rows, minus the 2 revision rows with a c = 255 operand.
    assert_eq!(n, 23);
}

/// f64 reference of the NARS truth functions (evidence horizon k = 1).
/// Confidence is capped at 0.9999 before revision, as the canonical upstream
/// (`ndarray::hpc::nars::NarsTruth::new`) does, so `c = 1` stays defined.
fn reference(op: Opcode, f1: f64, c1: f64, f2: f64, c2: f64) -> Option<(f64, f64)> {
    Some(match op {
        Opcode::Deduction => (f1 * f2, f1 * f2 * c1 * c2),
        Opcode::Induction => {
            let w = f1 * c1 * c2;
            (f2, w / (w + 1.0))
        }
        Opcode::Abduction => {
            let w = f2 * c1 * c2;
            (f1, w / (w + 1.0))
        }
        Opcode::Revision => {
            let (c1, c2) = (c1.min(0.9999), c2.min(0.9999));
            let (w1, w2) = (c1 / (1.0 - c1), c2 / (1.0 - c2));
            if w1 + w2 == 0.0 {
                return None;
            }
            ((f1 * w1 + f2 * w2) / (w1 + w2), (w1 + w2) / (w1 + w2 + 1.0))
        }
        Opcode::Synthesis => ((f1 + f2) / 2.0, (c1 + c2) / 2.0),
    })
}

fn fc(word: u64) -> (u8, u8) {
    (
        Field::Frequency.get(word) as u8,
        Field::Confidence.get(word) as u8,
    )
}

fn close(got: (u8, u8), want: (f64, f64)) -> bool {
    let q = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as i32;
    (i32::from(got.0) - q(want.0)).abs() <= 1 && (i32::from(got.1) - q(want.1)).abs() <= 1
}

/// The normative vectors are not self-referential: their truth agrees with an
/// independent f64 reference, and every other field follows the declared
/// contract.
#[test]
fn normative_vectors_agree_with_an_independent_reference() {
    for v in legacy() {
        let Kind::Forward(nib) = v.kind else { continue };
        let Some(op) = opcode_of(nib) else { continue };
        if is_known_bad(&v) {
            continue;
        }
        let u = |x: u64| x as f64 / 255.0;
        let (fa, ca) = fc(v.a);
        let (fb, cb) = fc(v.b);
        let want = reference(op, u(fa.into()), u(ca.into()), u(fb.into()), u(cb.into()))
            .unwrap_or((0.5, 0.0));
        assert!(
            close(fc(v.expected), want),
            "{op:?} {:?} vs {want:?}",
            fc(v.expected)
        );
        let e = v.expected;
        assert_eq!(
            Field::Pearl.get(e),
            Field::Pearl.get(v.a) & Field::Pearl.get(v.b)
        );
        assert_eq!(Field::Direction.get(e), Field::Direction.get(v.b));
        assert_eq!(Field::Plasticity.get(e), Field::Plasticity.get(v.b));
        assert_eq!(sign_extend(Field::Inference.get(e) as u8), op.encoding());
        assert_eq!((Field::Witness.get(e), Field::Epistemic.get(e)), (0, 0));
    }
}

// ─── The max-confidence revision defect ─────────────────────────────────

/// MEASURED DEFECT, NOT NORMATIVE. Revising two edges at confidence 255
/// returns f = 0, c = 0, in `revision` and in `learn` alike.
///
/// The path, step by step (each step asserted below):
/// 1. `c = 255/255 = 1.0 >= 0.999`, so `evidence_weight` returns `f32::MAX`
///    for both operands;
/// 2. `ws = MAX + MAX` overflows to `+inf`;
/// 3. `f = (f1·MAX + f2·MAX) / inf`: with `f1 = f2 = 1` the numerator is
///    also `inf`, so `f = inf/inf = NaN`; `c = inf / (inf + 1) = NaN`;
/// 4. packing: `NaN.clamp(0, 1)` is `NaN`, and the float-to-`u8` cast
///    saturates `NaN` to `0`.
///
/// Not a u8 overflow, a table bin, or a zero-denominator fallback: the
/// confidence-to-weight transform saturates to a finite value whose SUM
/// overflows. The intended result (f64 reference, c capped at 0.9999) is the
/// mean frequency at confidence 255.
///
/// FAILS IF: the defect is fixed. Then move this case into the normative set
/// with the reference result.
#[test]
fn known_bad_revision_max_confidence_collapses_to_zero() {
    use causal_edge::isa::truth;
    // 1-3, on the canonical truth function.
    assert_eq!(truth::evidence_weight(1.0), f32::MAX);
    assert_eq!(f32::MAX + f32::MAX, f32::INFINITY);
    let (f, c) = truth::revision(1.0, 1.0, 1.0, 1.0).unwrap();
    assert!(f.is_nan() && c.is_nan());
    // 4.
    assert_eq!((f.clamp(0.0, 1.0) * 255.0).round() as u8, 0);

    let t = tables();
    let (a, b) = KNOWN_BAD_REVISION_MAX_CONFIDENCE;
    let out = CausalEdge64(a).revision(CausalEdge64(b));
    let step = CausalEdge64(a).execute(Opcode::Revision, CausalEdge64(b), compose(&t));
    assert_eq!(fc(step.0), (0, 0));
    assert_eq!(fc(out.0), (0, 0));
    let mut l = CausalEdge64(a);
    l.learn(CausalEdge64(b), 0);
    assert_eq!(fc(l.0), (0, 0));

    // What it should be.
    let want = reference(Opcode::Revision, 1.0, 1.0, 1.0, 1.0).unwrap();
    assert!(close((255, 255), want));
    assert!(!close(fc(out.0), want));
}

/// MEASURED, NOT NORMATIVE. With ONE operand at c = 255 the result is
/// finite, but that operand's weight saturates to `f32::MAX` and its
/// frequency wins outright. The capped reference gives a weighted mean.
/// Which of the two is canonical is the same open decision as the collapse.
#[test]
fn one_saturated_operand_dominates_revision() {
    use causal_edge::isa::truth;
    let (f, c) = truth::revision(1.0, 1.0, 1.0 / 255.0, 254.0 / 255.0).unwrap();
    assert!(f.is_finite() && c.is_finite());
    let got = ((f * 255.0).round() as u8, (c * 255.0).round() as u8);
    assert_eq!(got, (255, 255));
    let want = reference(Opcode::Revision, 1.0, 1.0, 1.0 / 255.0, 254.0 / 255.0).unwrap();
    assert_eq!((want.0 * 255.0).round() as u8, 249);
}

// ─── Revision boundary contract ─────────────────────────────────────────

/// Numeric revision at the confidence boundaries, for identical, asymmetric
/// and contradictory frequencies, through both the register method and
/// `learn`. All agree with the f64 reference except the double-saturated
/// pair. (A single c = 255 operand against c = 0 agrees: the other side has
/// no weight to lose.)
#[test]
fn revision_boundaries_match_the_reference_except_the_known_defect() {
    let confidences = [(0u8, 0u8), (0, 255), (1, 1), (254, 254), (255, 255)];
    let frequencies = [(200u8, 200u8), (230, 40), (255, 0)];
    let mut bad = Vec::new();
    for &(ca, cb) in &confidences {
        for &(fa, fb) in &frequencies {
            let a = CausalEdge64(0)
                .with_inference_mantissa(1)
                .with_w_slot(9)
                .with_epistemic_raw5(5);
            let mut a = a;
            a.set_frequency_u8(fa);
            a.set_confidence_u8(ca);
            let mut b = CausalEdge64(0);
            b.set_frequency_u8(fb);
            b.set_confidence_u8(cb);
            let u = |x: u8| f64::from(x) / 255.0;
            let want = reference(Opcode::Revision, u(fa), u(ca), u(fb), u(cb));

            let m = fc(a.revision(b).0);
            let mut l = a;
            l.learn(b, 0);
            let learned = fc(l.0);
            match want {
                None => {
                    // No evidence on either side: the method reports
                    // "unknown", learn leaves the edge untouched.
                    assert_eq!(m, (128, 0));
                    assert_eq!(learned, (fa, ca));
                }
                Some(w) => {
                    if !close(m, w) || !close(learned, w) {
                        bad.push(((fa, ca), (fb, cb), m, learned));
                    }
                }
            }
            // learn's revision preserves the register's other state.
            assert_eq!((l.w_slot(), l.epistemic_raw5()), (9, 5));
        }
    }
    // Only the double-saturated pair disagrees, for every frequency mix.
    assert_eq!(bad.len(), 3, "{bad:?}");
    assert!(bad
        .iter()
        .all(|&((_, ca), (_, cb), _, _)| (ca, cb) == (255, 255)));
}
