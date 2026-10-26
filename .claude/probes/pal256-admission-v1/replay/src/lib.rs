//! D-TFLP Amendment 6.5 P2: replay the committed softmax reciprocal fixtures without TFLM.
//!
//! A port of the source int8 SOFTMAX output path (TFLM 5f64e4ad) for one element of a row:
//! sum of prepared weights B over admitted elements, `GetReciprocal(S, 12)` (common.h:968-982) with
//! `one_over_one_plus_x_for_x_in_0_1` (fixedpoint.h:832-853), then
//! `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, A[d]), exponent) - 128`, clamped
//! (reference/softmax.h:131-142).
//!
//! Every `+`/`-` that the C++ performs on int32 through `Add`/`Sub` (undefined on overflow) is a
//! `checked_*` here: a fixture whose replay would need an undefined operation is refused, never
//! reproduced. The supported sum domain is [2^19, 2^28) (6.4 follow-up N11).

use std::fmt;

/// Why a replay refused a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A row code is not admitted (difference below diff_min) — not part of the supported row model.
    UnadmittedCode(i32),
    /// A multiplicity below 1.
    NonPositiveMultiplicity(i32),
    /// The row has no element at the maximum.
    NoMaximum,
    /// The exact sum is outside [2^19, 2^28).
    UnsupportedDomain(i64),
    /// An int32 operation the source performs with undefined overflow would overflow.
    UndefinedArithmetic(&'static str),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The prepared source weights of one calibration (trace63 `softexp`), indexed by difference + 255.
#[derive(Debug, Clone)]
pub struct Table {
    /// Admitted flag.
    pub admitted: [bool; 256],
    /// exp_on_negative_values raw (Q0.31).
    pub a: [i32; 256],
    /// Rescale<12> raw (Q12.19).
    pub b: [i32; 256],
}

/// fixedpoint.h:326-338.
pub fn srdhm(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = i64::from(a) * i64::from(b);
    let nudge: i64 = if ab >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// fixedpoint.h:356-367, exponent in 0..=31.
pub fn rdbpot(x: i32, e: u32) -> Result<i32, Refusal> {
    if e > 31 {
        return Err(Refusal::UndefinedArithmetic("rounding shift exponent > 31"));
    }
    let mask = ((1i64 << e) - 1) as i32;
    let rem = x & mask;
    let thr = (mask >> 1) + i32::from(x < 0);
    (x >> e).checked_add(i32::from(rem > thr)).ok_or(Refusal::UndefinedArithmetic("rdbpot add"))
}

/// Saturating left shift (fixedpoint.h:380-406, exponent > 0).
fn satshl(x: i32, e: u32) -> i32 {
    let thr = (1i32 << (31 - e)) - 1;
    if x > thr {
        i32::MAX
    } else if x < -thr {
        i32::MIN
    } else {
        x << e
    }
}

/// fixedpoint.h:235-241.
fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let s = i64::from(a) + i64::from(b);
    let sign = if s >= 0 { 1 } else { -1 };
    ((s + sign) / 2) as i32
}

/// fixedpoint.h:832-853, raw in, raw out.
fn one_over_one_plus_x(a: i32) -> Result<i32, Refusal> {
    let ub = |what| Refusal::UndefinedArithmetic(what);
    let hd = rounding_half_sum(a, i32::MAX);
    let mut x = 1515870810i32.checked_add(srdhm(hd, -1010580540)).ok_or(ub("constant + hd*c"))?;
    for _ in 0..3 {
        let hdx = srdhm(hd, x);
        let om = (1i32 << 29).checked_sub(hdx).ok_or(ub("one - hd*x"))?;
        x = x.checked_add(satshl(srdhm(x, om), 2)).ok_or(ub("x + rescale"))?;
    }
    Ok(satshl(x, 1))
}

/// The source reciprocal: (raw, num_bits_over_unit). common.h:968-982 with x_integer_digits = 12.
pub fn source_reciprocal(s: i32) -> Result<(i32, i32), Refusal> {
    if s <= 0 {
        return Err(Refusal::UndefinedArithmetic("reciprocal of a non-positive sum (shift by 32 at S = 0)"));
    }
    let hpo = (s as u32).leading_zeros();
    let shifted = ((s as u32) << hpo).wrapping_sub(1u32 << 31) as i32; // unsigned in the source: defined
    Ok((one_over_one_plus_x(shifted)?, 12 - hpo as i32))
}

/// The correctly rounded alternative: round(2^62 / (S << clz(S))), half up, clipped to i32::MAX.
pub fn exact_reciprocal(s: i32) -> Result<(i32, i32), Refusal> {
    if s <= 0 {
        return Err(Refusal::UndefinedArithmetic("reciprocal of a non-positive sum"));
    }
    let hpo = (s as u32).leading_zeros();
    let den = (s as u128) << hpo;
    let q = ((1u128 << 63) + den) / (2 * den);
    Ok((q.min(i32::MAX as u128) as i32, 12 - hpo as i32))
}

/// Which reciprocal a replay uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recip {
    /// The source Newton-Raphson reciprocal (SourceParity).
    Source,
    /// The correctly rounded reciprocal (the cheaper candidate the gate decides on).
    Exact,
}

/// Row as (code, multiplicity) pairs. Returns the exact sum S after the admission checks.
pub fn row_sum(t: &Table, row: &[(i32, i32)]) -> Result<i64, Refusal> {
    let max = row.iter().map(|&(c, _)| c).max().ok_or(Refusal::NoMaximum)?;
    let mut s: i64 = 0;
    for &(c, n) in row {
        if n < 1 {
            return Err(Refusal::NonPositiveMultiplicity(n));
        }
        let i = (c - max + 255) as usize;
        if !t.admitted[i] {
            return Err(Refusal::UnadmittedCode(c));
        }
        s += i64::from(n) * i64::from(t.b[i]);
    }
    if !((1i64 << 19)..(1i64 << 28)).contains(&s) {
        return Err(Refusal::UnsupportedDomain(s));
    }
    Ok(s)
}

/// Output byte of the element with code `code` in `row`.
pub fn byte_of(t: &Table, row: &[(i32, i32)], code: i32, recip: Recip) -> Result<i8, Refusal> {
    let s = row_sum(t, row)?;
    let max = row.iter().map(|&(c, _)| c).max().ok_or(Refusal::NoMaximum)?;
    let (sh, nb) = match recip {
        Recip::Source => source_reciprocal(s as i32)?,
        Recip::Exact => exact_reciprocal(s as i32)?,
    };
    let ex = nb + 23;
    if !(0..=31).contains(&ex) {
        return Err(Refusal::UndefinedArithmetic("exponent outside the source CHECK"));
    }
    let i = (code - max + 255) as usize;
    let v = rdbpot(srdhm(sh, t.a[i]), ex as u32)? - 128;
    Ok(v.clamp(-128, 127) as i8)
}
