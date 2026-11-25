//! Exact decimals as a mathematical reading of an integer coefficient:
//! `value = coefficient × 10^exponent`, with `exponent = −scale`. Not a new
//! physical datatype and not a decimal runtime: the coefficient rides the
//! existing `i32` lane with its presence plane, filters are VRL recipes over
//! it, and sufficient statistics are mask-risc's `GroupPowerSumsI32`.
//!
//! Three parameters stay independent:
//!
//! - **coefficient width** — numerical capacity ([`PmWidth`], the ndarray
//!   #438 capacities);
//! - **exponent** (decimal scale) — numerical resolution, homogeneous per
//!   field or sealed slice ([`Exp`]);
//! - **presentation precision** — formatting only ([`format`]): 250.00 is
//!   coefficient 250 at exponent 0, shown with two fraction digits.
//!
//! The `i32` lane here is a semantic reference, not the storage decision.
//! The destination is adaptive width and scale bound through the ISA
//! register contract; this crate pins the arithmetic that destination must
//! reproduce. See `.claude/knowledge/exact-decimal-reconciliation.md`.
//!
//! Nothing here rounds. A value that is not exact at the asked exponent is
//! refused, and every refusal names its reason.

use lance_graph_contract::class_view::{ClassId, ClassView};
use lance_graph_dt64::PresenceSource;
use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
use lance_graph_mask_risc::{
    Foreign, GroupKey, LaneRef, MaskOp, Operand, Out, Planes, PowerSums, Pred, Program, Terminal,
};
use lance_graph_vrl::{Prepared, Recipe, Witness};

/// A decimal exponent: `value = coefficient × 10^exp`. Scale `s` is
/// `Exp(-s)`; a positive exponent is a coarser unit (thousands, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exp(pub i8);

impl Exp {
    /// Scale `s` as an exponent. Scales above 128 have no `i8` exponent
    /// and panic: an exponent is always the negated scale, never wrapped.
    pub fn scale(s: u8) -> Exp {
        let e = i8::try_from(-i16::from(s)).expect("a scale above 128 has no i8 exponent");
        Exp(e)
    }
}

/// Why an exact-decimal step refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimalRefusal {
    /// Not a plain decimal: empty, a second point, an exponent marker, a
    /// stray character.
    Malformed(String),
    /// The text has nonzero digits finer than the asked exponent; it would
    /// have to be rounded.
    NotExact { text: String, exp: Exp },
    /// The coefficient leaves the carrier (`i32` here).
    Overflow { text: String },
    /// Presentation precision would cut nonzero digits.
    PresentationCuts {
        coefficient: i64,
        exp: Exp,
        digits: u8,
    },
    /// Aligning aggregates to a common exponent left `i128`.
    AlignmentOverflow,
    /// An `f64` is not finite.
    NotFinite,
}

/// The digits of a plain decimal text, sign apart.
struct Digits {
    negative: bool,
    /// Integer and fraction digits, concatenated, leading zeros kept.
    all: String,
    /// How many of `all` are fraction digits.
    frac: usize,
}

fn digits(text: &str) -> Result<Digits, DecimalRefusal> {
    let malformed = || DecimalRefusal::Malformed(text.to_string());
    let t = text.trim();
    let (negative, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (int, frac) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    if (int.is_empty() && frac.is_empty())
        || !int.bytes().all(|b| b.is_ascii_digit())
        || !frac.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(malformed());
    }
    Ok(Digits {
        negative,
        all: format!("{int}{frac}"),
        frac: frac.len(),
    })
}

/// The canonical exact reading of `text`: the smallest exponent-free
/// coefficient, `(c, e)` with `c` holding no trailing zero unless it is 0.
/// `"1.20"` and `"1.2"` both read `(12, Exp(-1))`; `"25000"` reads
/// `(25, Exp(3))`; `"0.00"` and `"-0"` read `(0, Exp(0))` (established zero).
pub fn canonical(text: &str) -> Result<(i128, Exp), DecimalRefusal> {
    let d = digits(text)?;
    let trimmed = d.all.trim_start_matches('0');
    if trimmed.is_empty() {
        return Ok((0, Exp(0)));
    }
    let zeros = trimmed.len() - trimmed.trim_end_matches('0').len();
    let core = &trimmed[..trimmed.len() - zeros];
    let c: i128 = core.parse().map_err(|_| DecimalRefusal::Overflow {
        text: text.to_string(),
    })?;
    let e = zeros as i64 - d.frac as i64;
    let e = i8::try_from(e).map_err(|_| DecimalRefusal::Overflow {
        text: text.to_string(),
    })?;
    Ok((if d.negative { -c } else { c }, Exp(e)))
}

/// `c × 10^k` for `k >= 0`, or `None` when it leaves `i128`.
fn shift(c: i128, k: u32) -> Option<i128> {
    10i128.checked_pow(k).and_then(|p| c.checked_mul(p))
}

/// The coefficient of `text` at exponent `exp`, exactly: fewer fraction
/// digits are padded (`"1.2"` at scale 2 is 120), more are admitted only
/// when they are zeros (`"1.200"` is 120, `"1.205"` is refused). Never
/// rounds; refuses on leaving `i32`.
pub fn parse_exact(text: &str, exp: Exp) -> Result<i32, DecimalRefusal> {
    let (c, e) = canonical(text)?;
    let k = i32::from(e.0) - i32::from(exp.0);
    if c != 0 && k < 0 {
        return Err(DecimalRefusal::NotExact {
            text: text.to_string(),
            exp,
        });
    }
    let v = if c == 0 {
        0
    } else {
        shift(c, k as u32).ok_or_else(|| DecimalRefusal::Overflow {
            text: text.to_string(),
        })?
    };
    i32::try_from(v).map_err(|_| DecimalRefusal::Overflow {
        text: text.to_string(),
    })
}

/// Where a coefficient came from. An `f64` converted under
/// [`F64Policy::ShortestRoundTrip`] is an admitted conversion, never proof
/// that the original decimal was exactly that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Parsed from the source's own decimal text.
    Lexical,
    /// Converted from a binary `f64` through its shortest round-trip text.
    F64ShortestRoundTrip,
}

/// How a binary `f64` may become a coefficient. Explicit: there is no
/// implicit conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F64Policy {
    /// Take the shortest text that round-trips to the same `f64` and parse
    /// it exactly at the field's exponent; refuse when that text is not
    /// exact there (`0.1 + 0.2` at scale 2 is refused).
    ShortestRoundTrip,
}

/// A coefficient with its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Converted {
    pub coefficient: i32,
    pub provenance: Provenance,
}

pub fn from_lexical(text: &str, exp: Exp) -> Result<Converted, DecimalRefusal> {
    Ok(Converted {
        coefficient: parse_exact(text, exp)?,
        provenance: Provenance::Lexical,
    })
}

pub fn from_f64(v: f64, exp: Exp, policy: F64Policy) -> Result<Converted, DecimalRefusal> {
    if !v.is_finite() {
        return Err(DecimalRefusal::NotFinite);
    }
    match policy {
        // Rust's `Display` for `f64` is the shortest text that round-trips,
        // without an exponent marker.
        F64Policy::ShortestRoundTrip => Ok(Converted {
            coefficient: parse_exact(&format!("{v}"), exp)?,
            provenance: Provenance::F64ShortestRoundTrip,
        }),
    }
}

/// Render `coefficient × 10^exp` with exactly `digits` fraction digits.
/// Presentation only: padding is free, cutting a nonzero digit is refused
/// (rounding for display is a separate, explicit policy not modelled here).
pub fn format(coefficient: i64, exp: Exp, digits: u8) -> Result<String, DecimalRefusal> {
    let refuse = || DecimalRefusal::PresentationCuts {
        coefficient,
        exp,
        digits,
    };
    let d = usize::from(digits);
    // Place the point and pad with characters: no power of ten is computed,
    // so no presentation precision can overflow.
    let mag = coefficient.unsigned_abs().to_string();
    let (int, frac) = if exp.0 >= 0 {
        let mut int = mag;
        if int != "0" {
            int.push_str(&"0".repeat(exp.0 as usize));
        }
        (int, String::new())
    } else {
        let f = usize::from(exp.0.unsigned_abs());
        let padded = format!("{mag:0>width$}", width = f + 1);
        let (i, fr) = padded.split_at(padded.len() - f);
        (i.to_string(), fr.to_string())
    };
    let frac = if frac.len() > d {
        if frac[d..].bytes().any(|b| b != b'0') {
            return Err(refuse());
        }
        frac[..d].to_string()
    } else {
        format!("{frac:0<d$}")
    };
    let sign = if coefficient < 0 { "-" } else { "" };
    Ok(if d == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{frac}")
    })
}

/// The ndarray #438 P×M widths: polarity bit plus a magnitude of 8, 16, 24
/// or 32 bits. A capacity statement, not a native SIMD width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PmWidth {
    Pm9,
    Pm17,
    Pm25,
    Pm33,
}

impl PmWidth {
    pub fn capacity(self) -> u64 {
        match self {
            PmWidth::Pm9 => 255,
            PmWidth::Pm17 => 65_535,
            PmWidth::Pm25 => 16_777_215,
            PmWidth::Pm33 => 4_294_967_295,
        }
    }

    /// The narrowest width holding magnitude `m`; `None` past PM33.
    pub fn for_magnitude(m: u64) -> Option<PmWidth> {
        [PmWidth::Pm9, PmWidth::Pm17, PmWidth::Pm25, PmWidth::Pm33]
            .into_iter()
            .find(|w| m <= w.capacity())
    }
}

/// One homogeneous sealed slice: every value at the slice's exponent, which
/// is the coarsest exponent exact for all of them. Redundant powers of ten
/// are removed before the magnitude width is chosen: a slice of whole
/// amounts seals at exponent 0 (or higher) even when its field is declared
/// at scale 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedSlice {
    pub exp: Exp,
    /// Coefficients at `exp`; 0 where absent.
    pub coefficients: Vec<i32>,
    /// One bit per value: established (1) or absent (0).
    pub present: Vec<bool>,
    pub max_magnitude: u64,
    pub width: Option<PmWidth>,
}

/// Seal `cells` (a blank is `None`, an absent value): find the coarsest
/// common exponent, at most `limit` (the field's declared resolution, so a
/// slice never claims a finer one than the field admits), and read every
/// value exactly at it.
pub fn seal(cells: &[Option<&str>], limit: Exp) -> Result<SealedSlice, DecimalRefusal> {
    let mut read = Vec::with_capacity(cells.len());
    let mut exp = None::<Exp>;
    for c in cells {
        match c {
            None => read.push(None),
            Some(t) => {
                let (v, e) = canonical(t)?;
                if v != 0 {
                    exp = Some(exp.map_or(e, |x| x.min(e)));
                }
                read.push(Some(*t));
            }
        }
    }
    // An all-zero or empty slice needs no resolution at all.
    let exp = exp.unwrap_or(Exp(0)).min(Exp(i8::MAX)).max(limit);
    let mut coefficients = Vec::with_capacity(cells.len());
    let mut present = Vec::with_capacity(cells.len());
    let mut max_magnitude = 0u64;
    for t in read {
        match t {
            None => {
                coefficients.push(0);
                present.push(false);
            }
            Some(t) => {
                let c = parse_exact(t, exp)?;
                max_magnitude = max_magnitude.max(u64::from(c.unsigned_abs()));
                coefficients.push(c);
                present.push(true);
            }
        }
    }
    Ok(SealedSlice {
        exp,
        coefficients,
        present,
        max_magnitude,
        width: PmWidth::for_magnitude(max_magnitude),
    })
}

/// The sufficient statistics of one slice in its own exponent: `n`
/// established values, `Σc` and `Σc²` of their coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceStats {
    pub exp: Exp,
    pub sums: PowerSums,
}

/// N, S, Q of several slices combined at one exponent, exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Combined {
    pub exp: Exp,
    pub n: u64,
    /// `Σc` at `exp`: the sum is `s × 10^exp`.
    pub s: i128,
    /// `Σc²` at `exp`: the sum of squares is `q × 10^(2·exp)`.
    pub q: u128,
    /// How many scale alignments were applied (one per slice whose exponent
    /// differs from the target), against one per value for rescaling the
    /// population.
    pub alignments: u32,
}

impl Combined {
    /// Population variance as the exact rational `(num, den)` in units of
    /// `10^(2·exp)`: `(N·Q − S², N²)`. `None` when `n == 0` or it leaves
    /// `i128`.
    pub fn variance_parts(&self) -> Option<(i128, i128)> {
        if self.n == 0 {
            return None;
        }
        let n = i128::from(self.n);
        let q = i128::try_from(self.q).ok()?;
        let num = n.checked_mul(q)?.checked_sub(self.s.checked_mul(self.s)?)?;
        Some((num, n.checked_mul(n)?))
    }
}

/// Combine per-slice aggregates by aligning the aggregates, not the values:
/// the target is the finest exponent among non-empty slices, and each
/// slice's `Σc` is scaled by `10^(e−t)` and `Σc²` by `10^(2(e−t))`. One
/// multiplication per slice instead of one per value.
pub fn combine(slices: &[SliceStats]) -> Result<Combined, DecimalRefusal> {
    let target = slices
        .iter()
        .filter(|s| s.sums.n > 0)
        .map(|s| s.exp)
        .min()
        .unwrap_or(Exp(0));
    let mut out = Combined {
        exp: target,
        n: 0,
        s: 0,
        q: 0,
        alignments: 0,
    };
    for sl in slices.iter().filter(|s| s.sums.n > 0) {
        let k = (i32::from(sl.exp.0) - i32::from(target.0)) as u32;
        if k > 0 {
            out.alignments += 1;
        }
        // A zero aggregate stays zero at any exponent: no factor is needed,
        // so none can overflow.
        let s = match sl.sums.sum {
            0 => 0,
            v => shift(i128::from(v), k).ok_or(DecimalRefusal::AlignmentOverflow)?,
        };
        let q = match sl.sums.sum_sq {
            0 => 0,
            v => 10u128
                .checked_pow(2 * k)
                .and_then(|p| v.checked_mul(p))
                .ok_or(DecimalRefusal::AlignmentOverflow)?,
        };
        out.n += sl.sums.n;
        out.s = out
            .s
            .checked_add(s)
            .ok_or(DecimalRefusal::AlignmentOverflow)?;
        out.q = out
            .q
            .checked_add(q)
            .ok_or(DecimalRefusal::AlignmentOverflow)?;
    }
    Ok(out)
}

/// Per-slice N, S, Q in ONE mask-risc pass: `GroupPowerSumsI32` keyed by
/// the slice-id lane, masked by the value's presence (`presence ∨ c ≠ 0`,
/// the EXO-294 law), so `n` counts established values, not rows.
/// `coefficients[i]` is in its slice's exponent; `slice_of[i] < slices`.
pub fn slice_sums(
    coefficients: &[i32],
    presence: &[u64],
    slice_of: &[u32],
    slices: usize,
) -> Result<Vec<PowerSums>, lance_graph_mask_risc::ExecError> {
    let n_rows = coefficients.len();
    let program = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::NeI32 { lane: 0, v: 0 },
                under: None,
                dst: 0,
            },
            MaskOp::Or {
                a: Operand::Scratch(0),
                b: Operand::Plane(0),
                dst: 0,
            },
        ],
        Terminal::GroupPowerSumsI32 {
            mask: Operand::Scratch(0),
            key: GroupKey::Lane(1),
            val: 0,
        },
    );
    let masks = [presence];
    let lanes = [LaneRef::I32(coefficients), LaneRef::U32(slice_of)];
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&program, n_rows)?;
    let mut out = vec![PowerSums::default(); slices];
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::PowerSums(&mut out),
    )?;
    Ok(out)
}

/// The established-value mask of the coefficient lane: `presence ∨ c ≠ 0`.
fn established_ops() -> Vec<MaskOp> {
    vec![
        MaskOp::Pred {
            pred: Pred::NeI32 { lane: 0, v: 0 },
            under: None,
            dst: 0,
        },
        MaskOp::Or {
            a: Operand::Scratch(0),
            b: Operand::Plane(0),
            dst: 0,
        },
    ]
}

/// The extents `bounds` names: slice `k` is rows `bounds[k]..bounds[k + 1]`.
/// `bounds` starts at 0, ends at `n_rows` and never decreases.
fn extents(
    bounds: &[usize],
    n_rows: usize,
) -> Result<impl Iterator<Item = std::ops::Range<usize>> + '_, lance_graph_mask_risc::ExecError> {
    let ordered = bounds.first() == Some(&0)
        && bounds.last() == Some(&n_rows)
        && bounds.windows(2).all(|w| w[0] <= w[1]);
    if !ordered {
        return Err(lance_graph_mask_risc::ExecError::ExtentOutOfRange {
            lo: bounds.first().copied().unwrap_or(0),
            hi: bounds.last().copied().unwrap_or(0),
            n_rows,
        });
    }
    Ok(bounds.windows(2).map(|w| w[0]..w[1]))
}

/// Per-slice N and S with no slice-id lane: two extent passes per sealed
/// slice (`Count`, `MaskedSumI32`) over the lanes in place. Slice `k` is
/// rows `bounds[k]..bounds[k + 1]`; boundaries need not be word-aligned.
pub fn slice_counts_sums(
    coefficients: &[i32],
    presence: &[u64],
    bounds: &[usize],
) -> Result<Vec<(u64, i64)>, lance_graph_mask_risc::ExecError> {
    let n_rows = coefficients.len();
    let count = Program::new(
        established_ops(),
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let sum = Program::new(
        established_ops(),
        Terminal::MaskedSumI32 {
            mask: Operand::Scratch(0),
            lane: 0,
        },
    );
    let masks = [presence];
    let lanes = [LaneRef::I32(coefficients)];
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&count, n_rows)?;
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    for extent in extents(bounds, n_rows)? {
        let n = match execute_extent(
            &count,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::None,
            extent.clone(),
        )? {
            lance_graph_mask_risc::Value::Count(n) => n as u64,
            other => unreachable!("Count returns a count, got {other:?}"),
        };
        let s = match execute_extent(
            &sum,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::None,
            extent,
        )? {
            lance_graph_mask_risc::Value::SumI64(s) => s,
            other => unreachable!("MaskedSumI32 returns a sum, got {other:?}"),
        };
        out.push((n, s));
    }
    Ok(out)
}

/// Per-slice N, S, Q by sealed-slice boundaries instead of a slice-id lane:
/// one `GroupPowerSumsI32` extent pass per slice. Σx² exists only grouped,
/// so the pass is keyed by `zeros`, one group for every row: a lane the
/// caller allocates once per table length and reuses for every slice and
/// every boundary change, where the slice-id lane is rebuilt whenever a
/// boundary moves. `zeros.len()` must be `coefficients.len()`.
pub fn slice_sums_by_bounds(
    coefficients: &[i32],
    presence: &[u64],
    bounds: &[usize],
    zeros: &[u32],
) -> Result<Vec<PowerSums>, lance_graph_mask_risc::ExecError> {
    let n_rows = coefficients.len();
    let program = Program::new(
        established_ops(),
        Terminal::GroupPowerSumsI32 {
            mask: Operand::Scratch(0),
            key: GroupKey::Lane(1),
            val: 0,
        },
    );
    let masks = [presence];
    let lanes = [LaneRef::I32(coefficients), LaneRef::U32(zeros)];
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&program, n_rows)?;
    let mut out = vec![PowerSums::default(); bounds.len().saturating_sub(1)];
    for (sink, extent) in out.iter_mut().zip(extents(bounds, n_rows)?) {
        execute_extent(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::PowerSums(std::slice::from_mut(sink)),
            extent,
        )?;
    }
    Ok(out)
}

/// A comparison of a decimal field against a literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// What `value <op> literal` becomes on the coefficient at the field's
/// exponent, decided when the program is prepared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threshold {
    /// Compare the coefficient with this integer.
    Coefficient(Op, i32),
    /// True for no established value.
    Never,
    /// True for every established value (absent ones stay undecided).
    EveryValue,
}

/// Resolve `value <op> literal` at `exp` exactly, never through `f64`. A
/// literal between two representable coefficients takes the bound that
/// keeps the comparison's meaning: `x < 1.205` at scale 2 is `c ≤ 120`,
/// `x == 1.205` holds for no value, `x != 1.205` for every value.
pub fn threshold(op: Op, literal: &str, exp: Exp) -> Result<Threshold, DecimalRefusal> {
    let (c, e) = canonical(literal)?;
    let k = i32::from(e.0) - i32::from(exp.0);
    // The literal as the rational num / den in coefficient units.
    let (num, den) = if k >= 0 {
        (
            shift(c, k as u32).ok_or(DecimalRefusal::AlignmentOverflow)?,
            1i128,
        )
    } else {
        (
            c,
            shift(1, (-k) as u32).ok_or(DecimalRefusal::AlignmentOverflow)?,
        )
    };
    let exact = num % den == 0;
    let floor = num.div_euclid(den);
    // Saturating: a bound at the edge of i128 is far outside i32, where
    // `lt` and `gt` already decide, so saturation cannot change an answer.
    let ceil = if exact {
        floor
    } else {
        floor.saturating_add(1)
    };
    let lo = i128::from(i32::MIN);
    let hi = i128::from(i32::MAX);
    // `c < t` over i32 coefficients, with `t` possibly out of range.
    let lt = |t: i128| {
        if t > hi {
            Threshold::EveryValue
        } else if t <= lo {
            Threshold::Never
        } else {
            Threshold::Coefficient(Op::Lt, t as i32)
        }
    };
    let gt = |t: i128| {
        if t < lo {
            Threshold::EveryValue
        } else if t >= hi {
            Threshold::Never
        } else {
            Threshold::Coefficient(Op::Gt, t as i32)
        }
    };
    Ok(match op {
        Op::Eq | Op::Ne if !exact || num < lo || num > hi => {
            if op == Op::Eq {
                Threshold::Never
            } else {
                Threshold::EveryValue
            }
        }
        Op::Eq | Op::Ne => Threshold::Coefficient(op, num as i32),
        // x < L  ⇔  c < ceil(L)
        Op::Lt => lt(ceil),
        // x <= L ⇔  c < floor(L) + 1
        Op::Le => lt(floor.saturating_add(1)),
        // x > L  ⇔  c > floor(L)
        Op::Gt => gt(floor),
        // x >= L ⇔  c > ceil(L) − 1
        Op::Ge => gt(ceil.saturating_sub(1)),
    })
}

/// The VRL drop term for "drop the rows where `path <op> literal`", with the
/// literal resolved at `exp`. `Never` and `EveryValue` keep a term on the
/// field (one that is never, or always, true) so an absent value stays
/// undecided rather than kept.
pub fn drop_term(path: &str, op: Op, literal: &str, exp: Exp) -> Result<String, DecimalRefusal> {
    let sym = |o: Op| match o {
        Op::Eq => "==",
        Op::Ne => "!=",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
    };
    Ok(match threshold(op, literal, exp)? {
        Threshold::Coefficient(o, t) => format!("if {path} {} {t} {{ abort }}", sym(o)),
        Threshold::Never => format!("if {path} < {} {{ abort }}", i32::MIN),
        Threshold::EveryValue => format!("if {path} >= {} {{ abort }}", i32::MIN),
    })
}

/// The owner-side declaration of a field's exponent, by predicate IRI. In
/// production this is a property of the source's column descriptor
/// (intake's column shape), witnessed with the binding; the probe's tests
/// stand in for that owner. Not a second schema store.
pub trait FieldExps {
    fn exp_of(&self, predicate_iri: &str) -> Option<Exp>;
}

/// A VRL recipe whose decimal literals were resolved at the exponents the
/// owner declared, remembering them so a run under a different declaration
/// is refused even when class, fields and row count are unchanged.
pub struct DecimalRecipe {
    pub recipe: lance_graph_vrl::Recipe,
    resolved: Vec<(String, Exp)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RecipeRefusal {
    Decimal(DecimalRefusal),
    Recipe(lance_graph_vrl::Refusal),
    /// The owner declares no exponent for a filtered field.
    Undeclared {
        predicate_iri: String,
    },
    /// The owner's exponent for a field differs from the one the literals
    /// were resolved at.
    ExpChanged {
        predicate_iri: String,
        resolved: Exp,
        now: Exp,
    },
}

/// One decimal filter: drop rows where `path <op> literal`; `predicate_iri`
/// names the field's owner-side declaration.
pub struct DecimalFilter<'s> {
    pub path: &'s str,
    pub predicate_iri: &'s str,
    pub op: Op,
    pub literal: &'s str,
}

impl DecimalRecipe {
    /// Compile `projection` (plain VRL statements) plus `filters`, each
    /// literal resolved at the exponent `exps` declares for its field.
    pub fn compile<V: ClassView + ?Sized>(
        projection: &str,
        filters: &[DecimalFilter<'_>],
        exps: &dyn FieldExps,
        view: &V,
        class: ClassId,
    ) -> Result<DecimalRecipe, RecipeRefusal> {
        let mut src = projection.to_string();
        let mut resolved = Vec::with_capacity(filters.len());
        for f in filters {
            let exp = exps
                .exp_of(f.predicate_iri)
                .ok_or_else(|| RecipeRefusal::Undeclared {
                    predicate_iri: f.predicate_iri.to_string(),
                })?;
            src.push('\n');
            src.push_str(&drop_term(f.path, f.op, f.literal, exp).map_err(RecipeRefusal::Decimal)?);
            resolved.push((f.predicate_iri.to_string(), exp));
        }
        let recipe = Recipe::compile(&src, view, class).map_err(RecipeRefusal::Recipe)?;
        Ok(DecimalRecipe { recipe, resolved })
    }

    /// The owner still declares every filtered field at the exponent its
    /// literal was resolved at.
    fn check(&self, exps: &dyn FieldExps) -> Result<(), RecipeRefusal> {
        for (iri, resolved) in &self.resolved {
            let now = exps.exp_of(iri).ok_or_else(|| RecipeRefusal::Undeclared {
                predicate_iri: iri.clone(),
            })?;
            if now != *resolved {
                return Err(RecipeRefusal::ExpChanged {
                    predicate_iri: iri.clone(),
                    resolved: *resolved,
                    now,
                });
            }
        }
        Ok(())
    }

    /// Prepare a count. The prepared program keeps the exponents it was
    /// resolved at, through this recipe.
    pub fn prepare_count(&self, n_rows: usize) -> Result<Prepared, RecipeRefusal> {
        self.recipe
            .prepare_count(n_rows)
            .map_err(RecipeRefusal::Recipe)
    }

    /// Run a prepared program, refusing first when the owner's declaration
    /// moved: a fresh witness with the same class, fields and row count but
    /// another exponent must not run a program resolved at the old one.
    pub fn run_into<'a, V, S>(
        &self,
        p: &Prepared,
        w: &Witness<'a, V, S>,
        exps: &dyn FieldExps,
        scratch: &mut Scratch<'_>,
        out: Out<'_>,
    ) -> Result<lance_graph_mask_risc::Value, RecipeRefusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        self.check(exps)?;
        self.recipe
            .run_into(p, w, scratch, out)
            .map_err(RecipeRefusal::Recipe)
    }

    /// Rows the recipe keeps (`COUNT(rows)` after its filters).
    pub fn count<'a, V, S>(
        &self,
        w: &Witness<'a, V, S>,
        exps: &dyn FieldExps,
    ) -> Result<u64, RecipeRefusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        self.check(exps)?;
        self.recipe.count(w).map_err(RecipeRefusal::Recipe)
    }

    /// Rows a filter could not decide because the field is absent.
    pub fn count_unknown<'a, V, S>(
        &self,
        w: &Witness<'a, V, S>,
        exps: &dyn FieldExps,
    ) -> Result<u64, RecipeRefusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        self.check(exps)?;
        self.recipe.count_unknown(w).map_err(RecipeRefusal::Recipe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_128_is_the_finest_exponent_and_never_positive() {
        assert_eq!(Exp::scale(0), Exp(0));
        assert_eq!(Exp::scale(2), Exp(-2));
        assert_eq!(Exp::scale(128), Exp(-128));
        assert!(std::panic::catch_unwind(|| Exp::scale(129)).is_err());
    }

    #[test]
    fn format_pads_any_precision_without_overflow() {
        assert_eq!(
            format(1, Exp(0), 39).unwrap(),
            format!("1.{}", "0".repeat(39))
        );
        assert_eq!(
            format(1, Exp(-128), 128).unwrap(),
            format!("0.{}1", "0".repeat(127))
        );
        assert_eq!(format(i64::MIN, Exp(0), 0).unwrap(), "-9223372036854775808");
        assert_eq!(
            format(7, Exp(120), 0).unwrap(),
            format!("7{}", "0".repeat(120))
        );
        assert!(matches!(
            format(1, Exp(-128), 127),
            Err(DecimalRefusal::PresentationCuts { .. })
        ));
        assert_eq!(format(0, Exp(5), 2).unwrap(), "0.00");
    }

    #[test]
    fn a_zero_aggregate_aligns_to_any_exponent() {
        let zero = SliceStats {
            exp: Exp(0),
            sums: PowerSums {
                n: 1,
                sum: 0,
                sum_sq: 0,
            },
        };
        let fine = SliceStats {
            exp: Exp(-20),
            sums: PowerSums {
                n: 1,
                sum: 1,
                sum_sq: 1,
            },
        };
        let c = combine(&[zero, fine]).unwrap();
        assert_eq!((c.n, c.s, c.q, c.exp), (2, 1, 1, Exp(-20)));
    }

    #[test]
    fn a_threshold_at_the_edge_of_i128_does_not_overflow() {
        let max = i128::MAX.to_string();
        let min = (-i128::MAX).to_string();
        // One past: no i128 magnitude, refused rather than wrapped.
        assert!(matches!(
            threshold(Op::Ge, &i128::MIN.to_string(), Exp(0)),
            Err(DecimalRefusal::Overflow { .. })
        ));
        assert_eq!(threshold(Op::Le, &max, Exp(0)), Ok(Threshold::EveryValue));
        assert_eq!(threshold(Op::Lt, &max, Exp(0)), Ok(Threshold::EveryValue));
        assert_eq!(threshold(Op::Ge, &min, Exp(0)), Ok(Threshold::EveryValue));
        assert_eq!(threshold(Op::Gt, &max, Exp(0)), Ok(Threshold::Never));
    }
}

// ---- D-AED-2: a sealed slice at its physical width ------------------------

use ndarray::hpc::abakus::pm::{touched, Pm};
use ndarray::simd::U8x64;

/// The lanes of a [`PhysicalSlice`]. P×M follows the DR512 law: `P=0, M=0`
/// is untouched (absent), `P=0, M>0` is `-M`, `P=1` is `+M`, so `P=1, M=0`
/// is the established `+0` and `-0` is never stored.
#[derive(Clone, Debug, PartialEq)]
pub enum Lanes {
    /// PM9: one magnitude byte and one polarity bit per value, packed.
    /// [`PhysicalSlice::pm_block`] lends 64 of them as an Abakus [`Pm`]
    /// register. A resident `Pm` is padded to its SIMD alignment (128 bytes
    /// on AVX-512), so it is the arithmetic operand, not the storage.
    Pm9 { mag: Vec<u8>, pos: Vec<u64> },
    /// PM17: a `u16` magnitude per value and one polarity bit per value.
    /// Storage only: Abakus has no PM17 arithmetic (an ndarray #438
    /// deliverable).
    Pm17 { mag: Vec<u16>, pos: Vec<u64> },
    /// Past PM17: the `i32` coefficient and a presence plane, as today.
    I32 { coef: Vec<i32>, present: Vec<u64> },
}

/// A sealed slice stored at the narrowest width that holds it. Identity
/// order is the row order: value `i` is lane `i`.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalSlice {
    pub exp: Exp,
    pub len: usize,
    pub lanes: Lanes,
}

/// Notional bytes of a slice descriptor: exponent, width tag and length.
pub const DESCRIPTOR_BYTES: usize = 1 + 1 + 8;

impl PhysicalSlice {
    /// Store `slice` at its width: PM9, PM17, else `i32`.
    pub fn store(slice: &SealedSlice) -> PhysicalSlice {
        let len = slice.coefficients.len();
        let words = len.div_ceil(64);
        // (polarity, magnitude) of value `i`.
        let pm = |i: usize| -> (bool, u32) {
            let c = slice.coefficients[i];
            (slice.present[i] && c >= 0, c.unsigned_abs())
        };
        let lanes = match slice.width {
            Some(PmWidth::Pm9) => {
                let mut pos = vec![0u64; words];
                let mag = (0..len)
                    .map(|i| {
                        let (p, v) = pm(i);
                        pos[i / 64] |= u64::from(p) << (i % 64);
                        v as u8
                    })
                    .collect();
                Lanes::Pm9 { mag, pos }
            }
            Some(PmWidth::Pm17) => {
                let mut pos = vec![0u64; words];
                let mag = (0..len)
                    .map(|i| {
                        let (p, v) = pm(i);
                        pos[i / 64] |= u64::from(p) << (i % 64);
                        v as u16
                    })
                    .collect();
                Lanes::Pm17 { mag, pos }
            }
            _ => {
                let mut present = vec![0u64; words];
                for (i, &p) in slice.present.iter().enumerate() {
                    present[i / 64] |= u64::from(p) << (i % 64);
                }
                Lanes::I32 {
                    coef: slice.coefficients.clone(),
                    present,
                }
            }
        };
        PhysicalSlice {
            exp: slice.exp,
            len,
            lanes,
        }
    }

    /// The width the lanes are stored at.
    pub fn width(&self) -> &'static str {
        match self.lanes {
            Lanes::Pm9 { .. } => "PM9",
            Lanes::Pm17 { .. } => "PM17",
            Lanes::I32 { .. } => "i32",
        }
    }

    /// Value `i`: `None` when untouched, else the coefficient at `exp`.
    pub fn get(&self, i: usize) -> Option<i32> {
        assert!(i < self.len, "value {i} out of range");
        let (p, m) = match &self.lanes {
            Lanes::Pm9 { mag, pos } => (pos[i / 64] >> (i % 64) & 1 == 1, u32::from(mag[i])),
            Lanes::Pm17 { mag, pos } => (pos[i / 64] >> (i % 64) & 1 == 1, u32::from(mag[i])),
            Lanes::I32 { coef, present } => {
                return (present[i / 64] >> (i % 64) & 1 == 1).then_some(coef[i]);
            }
        };
        match (p, m) {
            (false, 0) => None,
            (false, m) => Some(-(m as i32)),
            (true, m) => Some(m as i32),
        }
    }

    /// Presence, one bit per value: `P ∨ M ≠ 0`. For PM9 this is Abakus's
    /// `touched` per block.
    pub fn presence(&self) -> Vec<u64> {
        let tail = |w: usize, bits: u64| {
            let used = self.len - w * 64;
            if used >= 64 {
                bits
            } else {
                bits & ((1u64 << used) - 1)
            }
        };
        match &self.lanes {
            Lanes::Pm9 { pos, .. } => (0..pos.len())
                .map(|w| tail(w, touched(self.pm_block(w).expect("PM9"))))
                .collect(),
            Lanes::Pm17 { mag, pos } => pos
                .iter()
                .enumerate()
                .map(|(w, &p)| {
                    let nonzero = mag[w * 64..(w * 64 + 64).min(self.len)]
                        .iter()
                        .enumerate()
                        .fold(0u64, |acc, (j, &m)| acc | u64::from(m != 0) << j);
                    tail(w, p | nonzero)
                })
                .collect(),
            Lanes::I32 { present, .. } => present.clone(),
        }
    }

    /// Values `64·w .. 64·w + 64` of a PM9 slice as an Abakus register,
    /// the tail padded untouched. `None` for another width.
    pub fn pm_block(&self, w: usize) -> Option<Pm> {
        let Lanes::Pm9 { mag, pos } = &self.lanes else {
            return None;
        };
        let lo = w * 64;
        let hi = (lo + 64).min(mag.len());
        let mut lanes = [0u8; 64];
        lanes[..hi - lo].copy_from_slice(&mag[lo..hi]);
        Some(Pm {
            mag: U8x64::from_array(lanes),
            pos: pos[w],
        })
    }

    /// Back to the sealed slice it was stored from.
    pub fn load(&self) -> SealedSlice {
        let mut coefficients = Vec::with_capacity(self.len);
        let mut present = Vec::with_capacity(self.len);
        let mut max_magnitude = 0u64;
        for i in 0..self.len {
            let v = self.get(i);
            present.push(v.is_some());
            let c = v.unwrap_or(0);
            max_magnitude = max_magnitude.max(u64::from(c.unsigned_abs()));
            coefficients.push(c);
        }
        SealedSlice {
            exp: self.exp,
            coefficients,
            present,
            max_magnitude,
            width: PmWidth::for_magnitude(max_magnitude),
        }
    }

    /// Encoded cost in bytes: a notional descriptor ([`DESCRIPTOR_BYTES`]),
    /// the magnitudes and the polarity or presence plane. It excludes `Vec`
    /// headers and struct padding, so it is not the memory the value holds.
    pub fn bytes(&self) -> usize {
        DESCRIPTOR_BYTES
            + match &self.lanes {
                Lanes::Pm9 { mag, pos } => mag.len() + pos.len() * 8,
                Lanes::Pm17 { mag, pos } => mag.len() * 2 + pos.len() * 8,
                Lanes::I32 { coef, present } => coef.len() * 4 + present.len() * 8,
            }
    }
}

// ---- D-AED-7: schema-on-write detection of exact decimals -----------------

/// Significant digits a binary64 printer can emit without the text being
/// a decimal the source chose: up to 15, every decimal round-trips through
/// `f64` unchanged, so a value that short says nothing about a float origin.
pub const F64_EXACT_DIGITS: usize = 15;

/// What a column of text reads as under the exact-decimal reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimalReading {
    /// Every present value parses exactly at `exp`, the coarsest common
    /// exponent, and at least one is written with a decimal point.
    Decimal {
        exp: Exp,
        min: i32,
        max: i32,
        width: Option<PmWidth>,
        provenance: Provenance,
        /// Present values read.
        support: usize,
    },
    /// No present value has a decimal point: an integer or text column,
    /// left to intake's own detection.
    NotDecimal,
    /// Some value is not an exact decimal; `row` is the first one.
    Refused {
        row: usize,
        why: DecimalDetectRefusal,
    },
}

/// Why a column is refused the decimal reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimalDetectRefusal {
    /// Not a plain decimal (exponent notation, stray characters, ...).
    Malformed,
    /// More significant digits than [`F64_EXACT_DIGITS`], and the text is
    /// exactly what a binary64 printer writes for the nearest `f64`: the
    /// value was produced by float arithmetic, not chosen as a decimal.
    FloatArtefact,
    /// The coefficient at the common exponent leaves `i32`.
    Overflow,
}

/// Whether `text` is the shortest round-trip rendering of an `f64` with more
/// significant digits than any decimal the source would pick.
fn float_artefact(text: &str) -> bool {
    let Ok((c, _)) = canonical(text) else {
        return false;
    };
    if c.unsigned_abs().to_string().len() <= F64_EXACT_DIGITS {
        return false;
    }
    match text.trim().parse::<f64>() {
        Ok(f) if f.is_finite() => canonical(&format!("{f}")) == canonical(text),
        _ => false,
    }
}

/// Detect the exact-decimal reading of one column (`None` is absent).
/// Nothing is rounded: a column is a decimal only if every present value
/// reads exactly at one exponent within `i32`.
pub fn detect_decimal(cells: &[Option<&str>]) -> DecimalReading {
    let mut pointed = false;
    for (row, cell) in cells.iter().enumerate() {
        let Some(text) = cell else { continue };
        if let Err(e) = canonical(text) {
            let why = match e {
                DecimalRefusal::Overflow { .. } => DecimalDetectRefusal::Overflow,
                _ => DecimalDetectRefusal::Malformed,
            };
            return DecimalReading::Refused { row, why };
        }
        if float_artefact(text) {
            return DecimalReading::Refused {
                row,
                why: DecimalDetectRefusal::FloatArtefact,
            };
        }
        pointed |= text.contains('.');
    }
    if !pointed {
        return DecimalReading::NotDecimal;
    }
    let sealed = match seal(cells, Exp(i8::MIN)) {
        Ok(s) => s,
        Err(_) => {
            // The exponent `seal` chose: the finest canonical one.
            let exp = cells
                .iter()
                .flatten()
                .filter_map(|t| canonical(t).ok())
                .filter(|(c, _)| *c != 0)
                .map(|(_, e)| e)
                .min()
                .unwrap_or(Exp(0));
            let row = cells
                .iter()
                .position(|c| c.is_some_and(|t| parse_exact(t, exp).is_err()))
                .unwrap_or(0);
            return DecimalReading::Refused {
                row,
                why: DecimalDetectRefusal::Overflow,
            };
        }
    };
    let present = || {
        sealed
            .coefficients
            .iter()
            .zip(&sealed.present)
            .filter(|(_, p)| **p)
            .map(|(c, _)| *c)
    };
    DecimalReading::Decimal {
        exp: sealed.exp,
        min: present().min().unwrap_or(0),
        max: present().max().unwrap_or(0),
        width: sealed.width,
        provenance: Provenance::Lexical,
        support: present().count(),
    }
}

/// A hand declaration for a decimal-looking column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecimalDecl {
    /// Keep the column as text.
    Text,
    /// Read it as a decimal at this exponent.
    Decimal(Exp),
    /// Read it through the explicit `f64` method; never a silent decimal.
    F64,
}

/// What a column becomes once the declaration is merged over detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecimalColumn {
    Text,
    Decimal(Exp),
    F64,
}

/// How the merged column relates to detection, as intake's `Verdict` does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimalVerdict {
    Agrees,
    Differs { detected: DecimalReading },
    DetectedOnly,
}

/// Merge `decl` over `detected`. Config wins, but a decimal declaration
/// must still read every value exactly: a declared exponent coarser than
/// the data is refused, never rounded.
pub fn merge_decimal(
    cells: &[Option<&str>],
    detected: &DecimalReading,
    decl: Option<DecimalDecl>,
) -> Result<(DecimalColumn, DecimalVerdict), DecimalRefusal> {
    let column = |d: &DecimalReading| match d {
        DecimalReading::Decimal { exp, .. } => DecimalColumn::Decimal(*exp),
        _ => DecimalColumn::Text,
    };
    let Some(decl) = decl else {
        return Ok((column(detected), DecimalVerdict::DetectedOnly));
    };
    let merged = match decl {
        DecimalDecl::Text => DecimalColumn::Text,
        DecimalDecl::F64 => DecimalColumn::F64,
        DecimalDecl::Decimal(exp) => {
            for text in cells.iter().flatten() {
                parse_exact(text, exp)?;
            }
            DecimalColumn::Decimal(exp)
        }
    };
    let verdict = if merged == column(detected) {
        DecimalVerdict::Agrees
    } else {
        DecimalVerdict::Differs {
            detected: detected.clone(),
        }
    };
    Ok((merged, verdict))
}
