//! D-TFLP Amendment 6.5 P0: the smallest safe binding contract between an 8-bit artifact and the
//! consumer that interprets it. Lab adapter only.
//!
//! The defect (Amendment 6.4 F4 / N8.1) is not in the decoder. `Codec8<I8, Sym, TiesAway, NaNAtMin>`
//! correctly decodes -128 as NaN, because that is its declared policy. The defect is that a payload
//! is a bare `&[i8]`: the producer's code space is carried only by the producer's Rust type and is
//! gone once the bytes are stored, so any consumer type can be applied to them.
//!
//! The contract here: an artifact carries a [`Receipt`]; a consumer states what it requires as an
//! [`Expect`]; [`bind`] compares the two field by field and checks integrity and the payload's
//! code range BEFORE any value is interpreted, and refuses with a named [`Refusal`]. Binding happens
//! once, at preparation time; a [`Bound`] view then decodes without further checks.
//!
//! No registry is introduced. A codec's identity is a [`CodecSignature`] DERIVED from `Codec8`'s
//! own sealed traits: the code range (`CodeSpace::MIN/MAX`), the NaN code (`NanPolicy::NAN_CODE`),
//! the raw bytes `Stores::store` writes for -128 and 0 (which separates I8 from biased U8), and the
//! rounding observed on four half-ties (which separates TiesAway, TiesEven and Floor).
//!
//! Scope of the identity claim: TESTED_COMPATIBLE_FOR_PINNED_CODEC_SET. The pinned set is every
//! `Codec8<T, S, R, N>` instantiation the sealed traits admit in ndarray `src/hpc/codec8.rs` as of
//! master `af4910a1`: 5 `Stores` pairs x 3 roundings with `NoNaN`, plus the 2 `Sym` pairs x 3
//! roundings with `NaNAtMin`, i.e. 21 types. `tests/admission.rs` checks that their 21 signatures
//! are pairwise distinct. The signature is therefore injective on that set, nothing more: a new
//! `Rounding`, `NanPolicy`, `CodeSpace` or `Stores` impl is not covered until it is added to that
//! test (four half-ties cannot separate every conceivable rounding rule).
//!
//! Byte equality is not identity: [`bind`] compares receipts, so two artifacts with identical
//! payload bytes and different receipts bind to different consumers.

use core::marker::PhantomData;
use ndarray::hpc::codec8::{Calibration, Codec8, Codec8Error, CodeSpace, Linear, NanPolicy, Rounding, Stores};
use sha2::{Digest, Sha256};

/// Behavioural identity of a `Codec8<T, S, R, N>` type, read from the codec's own traits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodecSignature {
    /// Raw stored byte for code -128 (I8: 0x80, biased U8: 0x00).
    pub raw_of_min: u8,
    /// Raw stored byte for code 0 (I8: 0x00, biased U8: 0x80).
    pub raw_of_zero: u8,
    /// Smallest value code.
    pub min: i32,
    /// Largest value code.
    pub max: i32,
    /// The NaN sentinel code, if the codec reserves one.
    pub nan_code: Option<i32>,
    /// `R::round` of 0.5, 1.5, 2.5, -0.5 (as i32): TiesAway 1,2,3,-1; TiesEven 0,2,2,0; Floor 0,1,2,-1.
    pub rounding: [i32; 4],
}

fn raw_byte<E: Copy>(e: E) -> u8 {
    assert_eq!(core::mem::size_of::<E>(), 1, "an 8-bit codec stores one byte per code");
    // SAFETY: E is a one-byte Copy type (i8 or u8), checked above; reading it as u8 is a bit copy.
    unsafe { core::mem::transmute_copy::<E, u8>(&e) }
}

/// The signature of `Codec8<T, S, R, N>`. Only the types' own associated items are used.
pub fn signature<T, S, R, N>() -> CodecSignature
where
    T: Stores<S>,
    S: CodeSpace,
    R: Rounding,
    N: NanPolicy<S>,
{
    CodecSignature {
        raw_of_min: raw_byte(T::store(-128)),
        raw_of_zero: raw_byte(T::store(0)),
        min: S::MIN,
        max: S::MAX,
        nan_code: N::NAN_CODE,
        rounding: [R::round(0.5) as i32, R::round(1.5) as i32, R::round(2.5) as i32, R::round(-0.5) as i32],
    }
}

/// Where the bytes came from, when SourceParity needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    /// Source revision, e.g. the TFLM checkout.
    pub revision: String,
    /// Kernel set / backend, e.g. "reference".
    pub backend: String,
}

/// What an artifact says about its payload. Everything a consumer needs to interpret the bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// Logical code space, storage, NaN policy, rounding and saturation (saturation is the code
    /// range: encoders clamp to `min..=max`).
    pub codec: CodecSignature,
    /// `f32::to_bits` of the calibration scale.
    pub scale_bits: u32,
    /// Calibration zero point.
    pub zero_point: i32,
    /// The operator contract the bytes were prepared for (e.g. "fc.weight.per_tensor").
    pub operator: String,
    /// Calibration / codebook generation the bytes belong to.
    pub codebook_generation: u64,
    /// Source identity, when the artifact claims SourceParity.
    pub source: Option<SourceIdentity>,
    /// Number of codes in the payload.
    pub len: usize,
    /// SHA-256 of the payload bytes.
    pub payload_sha256: [u8; 32],
}

/// A stored artifact: a receipt and its raw payload bytes.
#[derive(Debug, Clone)]
pub struct Artifact {
    /// The receipt.
    pub receipt: Receipt,
    /// The payload exactly as stored.
    pub payload: Vec<u8>,
}

fn sha256(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

/// Produce an artifact from codes encoded by `Codec8<T, S, R, N>` with calibration `C`.
pub fn produce<T, S, R, N, C>(codes: &[T::Elem], operator: &str, codebook_generation: u64, source: Option<SourceIdentity>) -> Artifact
where
    T: Stores<S>,
    S: CodeSpace,
    R: Rounding,
    N: NanPolicy<S>,
    C: Calibration,
{
    let payload: Vec<u8> = codes.iter().map(|&e| raw_byte(e)).collect();
    Artifact {
        receipt: Receipt {
            codec: signature::<T, S, R, N>(),
            scale_bits: C::SCALE.to_bits(),
            zero_point: C::ZERO_POINT,
            operator: operator.to_string(),
            codebook_generation,
            source,
            len: payload.len(),
            payload_sha256: sha256(&payload),
        },
        payload,
    }
}

/// Why a binding was refused. Every variant is decided before any payload value is interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Payload length differs from the receipt.
    LengthMismatch { receipt: usize, payload: usize },
    /// Payload bytes do not hash to the receipt's digest.
    IntegrityMismatch,
    /// Same code range but a different stored byte layout (e.g. biased U8 vs raw I8).
    StorageMismatch,
    /// Different code range.
    CodeSpaceMismatch { artifact: (i32, i32), consumer: (i32, i32) },
    /// Different NaN sentinel.
    NanPolicyMismatch { artifact: Option<i32>, consumer: Option<i32> },
    /// Different rounding mode.
    RoundingMismatch,
    /// Different calibration scale.
    ScaleMismatch,
    /// Different calibration zero point.
    ZeroPointMismatch { artifact: i32, consumer: i32 },
    /// Prepared for a different operator contract.
    OperatorMismatch,
    /// Belongs to a different calibration / codebook generation.
    GenerationMismatch { artifact: u64, consumer: u64 },
    /// Source identity required and missing or different.
    SourceMismatch,
    /// A stored code outside the code space that is not the NaN code.
    InvalidCode { index: usize, code: i32 },
}

/// What a consumer requires, stated from its own codec type `Codec8<T, S, R, N>` and calibration `C`.
pub struct Expect<T, S, R, N, C> {
    /// Operator contract the consumer implements.
    pub operator: String,
    /// The generation it was prepared against.
    pub codebook_generation: u64,
    /// Required source identity (SourceParity consumers); `None` accepts any source.
    pub source: Option<SourceIdentity>,
    _m: PhantomData<fn() -> (T, S, R, N, C)>,
}

impl<T, S, R, N, C> Expect<T, S, R, N, C> {
    /// A consumer requirement.
    pub fn new(operator: &str, codebook_generation: u64, source: Option<SourceIdentity>) -> Self {
        Expect { operator: operator.to_string(), codebook_generation, source, _m: PhantomData }
    }
}

/// An artifact admitted for one consumer type. Holds the payload; decodes through the consumer's codec.
pub struct Bound<'a, T, S, R, N, C> {
    payload: &'a [u8],
    _m: PhantomData<fn() -> (T, S, R, N, C)>,
}

/// Admit `a` for the consumer described by `e`. Checks, in order: length and integrity, codec
/// signature (storage, code space, NaN policy, rounding), calibration, operator, generation,
/// source, then the payload's code range. Returns the first refusal.
pub fn bind<'a, T, S, R, N, C>(a: &'a Artifact, e: &Expect<T, S, R, N, C>) -> Result<Bound<'a, T, S, R, N, C>, Refusal>
where
    T: Stores<S>,
    S: Linear,
    R: Rounding,
    N: NanPolicy<S>,
    C: Calibration,
{
    let r = &a.receipt;
    if r.len != a.payload.len() {
        return Err(Refusal::LengthMismatch { receipt: r.len, payload: a.payload.len() });
    }
    if sha256(&a.payload) != r.payload_sha256 {
        return Err(Refusal::IntegrityMismatch);
    }
    let want = signature::<T, S, R, N>();
    if (r.codec.min, r.codec.max) != (want.min, want.max) {
        return Err(Refusal::CodeSpaceMismatch { artifact: (r.codec.min, r.codec.max), consumer: (want.min, want.max) });
    }
    if (r.codec.raw_of_min, r.codec.raw_of_zero) != (want.raw_of_min, want.raw_of_zero) {
        return Err(Refusal::StorageMismatch);
    }
    if r.codec.nan_code != want.nan_code {
        return Err(Refusal::NanPolicyMismatch { artifact: r.codec.nan_code, consumer: want.nan_code });
    }
    if r.codec.rounding != want.rounding {
        return Err(Refusal::RoundingMismatch);
    }
    if r.scale_bits != C::SCALE.to_bits() {
        return Err(Refusal::ScaleMismatch);
    }
    if r.zero_point != C::ZERO_POINT {
        return Err(Refusal::ZeroPointMismatch { artifact: r.zero_point, consumer: C::ZERO_POINT });
    }
    if r.operator != e.operator {
        return Err(Refusal::OperatorMismatch);
    }
    if r.codebook_generation != e.codebook_generation {
        return Err(Refusal::GenerationMismatch { artifact: r.codebook_generation, consumer: e.codebook_generation });
    }
    if let Some(src) = &e.source {
        if r.source.as_ref() != Some(src) {
            return Err(Refusal::SourceMismatch);
        }
    }
    // Code-range scan: the signature matched, so the layout is the consumer's own.
    for (index, &b) in a.payload.iter().enumerate() {
        let code = load_raw::<T, S>(b);
        if (code < S::MIN || code > S::MAX) && N::NAN_CODE != Some(code) {
            return Err(Refusal::InvalidCode { index, code });
        }
    }
    Ok(Bound { payload: &a.payload, _m: PhantomData })
}

fn load_raw<T: Stores<S>, S: CodeSpace>(b: u8) -> i32 {
    // T::Elem is one byte (checked when the artifact was produced and by `raw_byte`).
    assert_eq!(core::mem::size_of::<T::Elem>(), 1);
    // SAFETY: T::Elem is a one-byte Copy type (i8 or u8); this is a bit copy.
    let e: T::Elem = unsafe { core::mem::transmute_copy::<u8, T::Elem>(&b) };
    T::load(e)
}

impl<T, S, R, N, C> Bound<'_, T, S, R, N, C>
where
    T: Stores<S>,
    S: Linear,
    R: Rounding,
    N: NanPolicy<S>,
    C: Calibration,
{
    /// Decode the admitted payload with the consumer's own codec.
    pub fn decode(&self, dst: &mut [f32]) -> Result<(), Codec8Error> {
        let elems: Vec<T::Elem> = self
            .payload
            .iter()
            .map(|&b| {
                // SAFETY: as in `load_raw`.
                unsafe { core::mem::transmute_copy::<u8, T::Elem>(&b) }
            })
            .collect();
        Codec8::<T, S, R, N>::decode_checked::<C>(&elems, dst)
    }
}
