//! Amendment 6.5 P0 positive tests and negative controls.
//!
//! Rule for every negative control: the payload bytes are IDENTICAL in the accepted and the refused
//! arm. Only the consumer (or one receipt field) changes. A refusal caused by different input
//! values would not show that admission works.
use admit::*;
use ndarray::hpc::codec8::{Asym, Calibration, Codec8, Codec8Error, NaNAtMin, NoNaN, Sym, TiesAway, TiesEven, I8, U8};

struct One;
impl Calibration for One {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 0;
}
struct Half;
impl Calibration for Half {
    const SCALE: f32 = 0.5;
    const ZERO_POINT: i32 = 0;
}
struct OneZp5;
impl Calibration for OneZp5 {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 5;
}

type SymNaN = Codec8<I8, Sym, TiesAway, NaNAtMin>;

const OP: &str = "lab.op";
fn src() -> Option<SourceIdentity> {
    Some(SourceIdentity { revision: "5f64e4ad".into(), backend: "reference".into() })
}
fn asym_payload() -> Vec<i8> {
    vec![-128, -127, -1, 0, 127]
}
fn asym_artifact() -> Artifact {
    produce::<I8, Asym, TiesAway, NoNaN, One>(&asym_payload(), OP, 7, src())
}

// ---------------------------------------------------------------- positive

#[test]
fn p1_p2_asym_minus128_and_neighbours_are_ordinary_values() {
    let a = asym_artifact();
    let b = bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src())).unwrap();
    let mut out = [0f32; 5];
    b.decode(&mut out).unwrap();
    assert_eq!(out, [-128.0, -127.0, -1.0, 0.0, 127.0]);
}

#[test]
fn p3_symnan_minus128_is_the_declared_sentinel() {
    let a = produce::<I8, Sym, TiesAway, NaNAtMin, One>(&[-128, -127, 0, 127], OP, 7, None);
    let b = bind(&a, &Expect::<I8, Sym, TiesAway, NaNAtMin, One>::new(OP, 7, None)).unwrap();
    let mut out = [0f32; 4];
    b.decode(&mut out).unwrap();
    assert!(out[0].is_nan());
    assert_eq!(&out[1..], &[-127.0, 0.0, 127.0]);
}

#[test]
fn p4_sym_minus127_to_127_all_valid() {
    let codes: Vec<i8> = (-127..=127).collect();
    let a = produce::<I8, Sym, TiesAway, NoNaN, One>(&codes, OP, 7, None);
    let b = bind(&a, &Expect::<I8, Sym, TiesAway, NoNaN, One>::new(OP, 7, None)).unwrap();
    let mut out = vec![0f32; codes.len()];
    b.decode(&mut out).unwrap();
    assert!(out.iter().zip(&codes).all(|(&v, &c)| v == c as f32));
}

#[test]
fn p5_tflm_symmetric_weight_still_rejects_minus128() {
    // A producer that CLAIMS Sym/NoNaN (TfliteWeight) but stores -128: refused at bind, by the
    // code-range scan, with the position.
    let a = produce::<I8, Sym, TiesAway, NoNaN, One>(&[3, -128, 4], OP, 7, None);
    let r = bind(&a, &Expect::<I8, Sym, TiesAway, NoNaN, One>::new(OP, 7, None));
    assert_eq!(r.err(), Some(Refusal::InvalidCode { index: 1, code: -128 }));
    // and the decoder's own check agrees, unchanged
    let mut out = [0f32; 3];
    assert_eq!(Codec8::<I8, Sym, TiesAway, NoNaN>::decode_checked::<One>(&[3, -128, 4], &mut out), Err(Codec8Error::InvalidCode { index: 1 }));
}

#[test]
fn p6_signed_and_biased_indexing_agree_on_every_code() {
    use ndarray::hpc::codec8::Stores;
    // a 256-entry table indexed by the biased byte; the signed path indexes it with code + 128
    let table: Vec<u16> = (0..256u16).map(|i| i.wrapping_mul(40503)).collect();
    for c in -128..=127i32 {
        let biased = <U8 as Stores<Asym>>::store(c);
        let signed = <I8 as Stores<Asym>>::store(c);
        assert_eq!(biased, (signed as u8) ^ 0x80, "storage relation at {c}");
        assert_eq!(table[biased as usize], table[(i32::from(signed) + 128) as usize], "index at {c}");
        assert_eq!(<U8 as Stores<Asym>>::load(biased), c);
        assert_eq!(<I8 as Stores<Asym>>::load(signed), c);
        if c > -128 {
            assert_eq!(<U8 as Stores<Sym>>::load(<U8 as Stores<Sym>>::store(c)), c);
        }
    }
}

// ---------------------------------------------------------------- negative controls

/// The original F4: the bare decoder applied to an Asym payload ACCEPTS it and reads -128 as NaN.
#[test]
fn f4_before_wrong_interpretation_accepted() {
    let p = asym_payload();
    let mut out = [0f32; 5];
    SymNaN::decode_checked::<One>(&p, &mut out).unwrap();
    assert!(out[0].is_nan(), "baseline: the bare decoder turns an Asym value into NaN");
}

/// F4 after: the same bytes through `bind` are refused before interpretation, and the identical
/// payload binds to its own consumer (two-sided).
#[test]
fn n1_f4_after_asym_payload_to_symnan_refused() {
    let a = asym_artifact();
    let r = bind(&a, &Expect::<I8, Sym, TiesAway, NaNAtMin, One>::new(OP, 7, src())).err();
    assert_eq!(r, Some(Refusal::CodeSpaceMismatch { artifact: (-128, 127), consumer: (-127, 127) }));
    assert!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src())).is_ok());
}

#[test]
fn n2_symnan_payload_to_asym_consumer_refused() {
    let a = produce::<I8, Sym, TiesAway, NaNAtMin, One>(&[-128, 1, 2], OP, 7, None);
    assert!(matches!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, None)).err(), Some(Refusal::CodeSpaceMismatch { .. })));
    // Sym/NoNaN consumer: same range, different sentinel
    assert_eq!(
        bind(&a, &Expect::<I8, Sym, TiesAway, NoNaN, One>::new(OP, 7, None)).err(),
        Some(Refusal::NanPolicyMismatch { artifact: Some(-128), consumer: None })
    );
    assert!(bind(&a, &Expect::<I8, Sym, TiesAway, NaNAtMin, One>::new(OP, 7, None)).is_ok());
}

#[test]
fn n3_biased_u8_payload_to_raw_signed_consumer_refused() {
    let codes: Vec<u8> = asym_payload().iter().map(|&c| <U8 as ndarray::hpc::codec8::Stores<Asym>>::store(i32::from(c))).collect();
    let a = produce::<U8, Asym, TiesAway, NoNaN, One>(&codes, OP, 7, None);
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, None)).err(), Some(Refusal::StorageMismatch));
    let b = bind(&a, &Expect::<U8, Asym, TiesAway, NoNaN, One>::new(OP, 7, None)).unwrap();
    let mut out = [0f32; 5];
    b.decode(&mut out).unwrap();
    assert_eq!(out, [-128.0, -127.0, -1.0, 0.0, 127.0]);
}

#[test]
fn n4_wrong_codebook_generation_refused() {
    let a = asym_artifact();
    assert_eq!(
        bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 8, src())).err(),
        Some(Refusal::GenerationMismatch { artifact: 7, consumer: 8 })
    );
}

#[test]
fn n5_wrong_rounding_refused() {
    let a = asym_artifact();
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesEven, NoNaN, One>::new(OP, 7, src())).err(), Some(Refusal::RoundingMismatch));
}

#[test]
fn n6_wrong_scale_or_zero_point_refused() {
    let a = asym_artifact();
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, Half>::new(OP, 7, src())).err(), Some(Refusal::ScaleMismatch));
    assert_eq!(
        bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, OneZp5>::new(OP, 7, src())).err(),
        Some(Refusal::ZeroPointMismatch { artifact: 0, consumer: 5 })
    );
}

#[test]
fn n7_same_bytes_incompatible_receipt_refused() {
    // Two artifacts with identical payload bytes and digests, prepared for different operators.
    let a = asym_artifact();
    let b = produce::<I8, Asym, TiesAway, NoNaN, One>(&asym_payload(), "other.op", 7, src());
    assert_eq!(a.payload, b.payload);
    assert_eq!(a.receipt.payload_sha256, b.receipt.payload_sha256);
    let e = Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src());
    assert!(bind(&a, &e).is_ok());
    assert_eq!(bind(&b, &e).err(), Some(Refusal::OperatorMismatch));
}

#[test]
fn n8_one_corrupted_byte_refused() {
    let mut a = asym_artifact();
    a.payload[2] ^= 0x01;
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src())).err(), Some(Refusal::IntegrityMismatch));
}

#[test]
fn n9_source_identity_required_and_mismatched_refused() {
    let a = produce::<I8, Asym, TiesAway, NoNaN, One>(&asym_payload(), OP, 7, None);
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src())).err(), Some(Refusal::SourceMismatch));
    let wrong = Some(SourceIdentity { revision: "8f6596d".into(), backend: "reference".into() });
    let b = asym_artifact();
    assert_eq!(bind(&b, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, wrong)).err(), Some(Refusal::SourceMismatch));
}

#[test]
fn n10_truncated_payload_refused() {
    let mut a = asym_artifact();
    a.payload.pop();
    assert_eq!(bind(&a, &Expect::<I8, Asym, TiesAway, NoNaN, One>::new(OP, 7, src())).err(), Some(Refusal::LengthMismatch { receipt: 5, payload: 4 }));
}
