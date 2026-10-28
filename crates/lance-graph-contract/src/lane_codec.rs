//! Epistemic lane layout of a class, chosen by the consumer at compile time.
//!
//! A class that carries quantized values in its row declares, once, how its
//! lanes are stored: their width, the code space, and where the value bytes
//! and the presence bits sit. The consumer picks this at compile time,
//! typically from the quantization annotations it already has (ONNX
//! `QuantizeLinear` / TFLite tensor quantization: a zero point of 0 is
//! [`CodeSpace::Sym`], any other zero point is [`CodeSpace::Asym`]), and
//! states it as a constant:
//!
//! ```
//! use lance_graph_contract::lane_codec::{
//!     checked, CodeSpace, EpiLaneClass, EpiLaneSpec, LaneWidth,
//! };
//!
//! /// A class whose lanes are ONNX uint8 values with zero point 128.
//! struct SensorReading;
//! impl EpiLaneClass for SensorReading {
//!     const SPEC: EpiLaneSpec = checked(EpiLaneSpec {
//!         width: LaneWidth::Bits8,
//!         space: CodeSpace::from_zero_point(128),
//!         values: 64,
//!         presence: 96,
//!         lanes: 16,
//!     });
//! }
//! assert_eq!(SensorReading::SPEC.space, CodeSpace::Asym);
//! ```
//!
//! A combination that has no meaning does not compile:
//!
//! ```compile_fail,E0080
//! use lance_graph_contract::lane_codec::{
//!     checked, CodeSpace, EpiLaneClass, EpiLaneSpec, LaneWidth,
//! };
//! struct Bad;
//! impl EpiLaneClass for Bad {
//!     // A 9-bit lane is sign-magnitude: it has no zero point.
//!     const SPEC: EpiLaneSpec = checked(EpiLaneSpec {
//!         width: LaneWidth::Bits9,
//!         space: CodeSpace::Asym,
//!         values: 64,
//!         presence: 96,
//!         lanes: 16,
//!     });
//! }
//! let _ = Bad::SPEC;
//! ```
//!
//! [`ClassView::epi_lanes`](crate::class_view::ClassView::epi_lanes) names the
//! same constant for a class, so a reader holding only a classid can resolve
//! it, and a writer generic over [`EpiLaneClass`] can refuse a class that
//! declares a different layout.
//!
//! # Absent
//!
//! Every lane has a presence bit; the presence plane is not optional. A lane
//! is absent (never touched: NaN, `$null`, UNKNOWN) when its presence bit is 0
//! **and** its value is 0, at 8 and 9 bits alike. At 9 bits the presence bit
//! is also the sign (DR512, `ndarray` `hpc::codec9`), so presence 0 with a
//! nonzero magnitude is a negative value. At 8 bits presence 0 with a nonzero
//! value cannot arise except by reading a 9-bit lane through an 8-bit layout;
//! the width in this spec is what rules that out.
//!
//! A sparse write leaves an absent lane unwritten, so it keeps the value read
//! before. This module describes layout only; it stores nothing and adds no
//! bytes to a row.

/// Bits per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneWidth {
    /// One value byte and one presence bit per lane.
    Bits8,
    /// One magnitude byte per lane plus its presence/sign bit (DR512).
    Bits9,
}

/// The code space of a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeSpace {
    /// Zero point 0: values are symmetric around code 0.
    Sym,
    /// A nonzero zero point.
    Asym,
}

impl CodeSpace {
    /// The code space of an ONNX / TFLite quantization annotation.
    #[must_use]
    pub const fn from_zero_point(zero_point: i32) -> Self {
        if zero_point == 0 {
            Self::Sym
        } else {
            Self::Asym
        }
    }
}

/// Where a class keeps its epistemic lanes, and how they are coded.
///
/// Offsets are bytes into the 512-byte row (key, edge block, value slab).
/// Presence bits are little-endian, bit `i` for lane `i`, in
/// `lanes.div_ceil(8)` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpiLaneSpec {
    pub width: LaneWidth,
    pub space: CodeSpace,
    /// First value byte; one byte per lane.
    pub values: u16,
    /// First presence byte.
    pub presence: u16,
    /// Number of lanes, `1..=64`.
    pub lanes: u8,
}

/// Bytes of a row.
pub const ROW_BYTES: u16 = 512;
/// First byte of the value slab: lanes never overlap the key or edge block.
pub const VALUE_SLAB_START: u16 = 32;

/// Why an [`EpiLaneSpec`] is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecError {
    /// `lanes` is 0 or above 64.
    LaneCount,
    /// A region starts before the value slab or ends past the row.
    OutOfSlab,
    /// The value bytes and the presence bytes overlap.
    Overlap,
    /// A 9-bit lane is sign-magnitude and therefore symmetric.
    NineBitIsSym,
}

impl EpiLaneSpec {
    /// Bytes of the presence plane.
    #[must_use]
    pub const fn presence_bytes(&self) -> u16 {
        (self.lanes as u16).div_ceil(8)
    }

    /// Checks the spec against the rules above.
    ///
    /// # Errors
    /// The first rule broken.
    pub const fn validate(&self) -> Result<(), SpecError> {
        if self.lanes == 0 || self.lanes > 64 {
            return Err(SpecError::LaneCount);
        }
        let v_end = self.values + self.lanes as u16;
        let p_end = self.presence + self.presence_bytes();
        if self.values < VALUE_SLAB_START
            || v_end > ROW_BYTES
            || self.presence < VALUE_SLAB_START
            || p_end > ROW_BYTES
        {
            return Err(SpecError::OutOfSlab);
        }
        if self.presence < v_end && self.values < p_end {
            return Err(SpecError::Overlap);
        }
        match (self.width, self.space) {
            (LaneWidth::Bits9, CodeSpace::Asym) => Err(SpecError::NineBitIsSym),
            _ => Ok(()),
        }
    }

    /// Whether a lane holding `value` with presence bit `present` is absent:
    /// presence 0 and value 0.
    #[must_use]
    pub const fn absent(value: u8, present: bool) -> bool {
        !present && value == 0
    }

    /// Lane `lane`'s value byte in `row`.
    #[must_use]
    pub fn value(&self, row: &[u8], lane: usize) -> u8 {
        row[self.values as usize + lane]
    }

    /// Lane `lane`'s presence bit in `row`.
    #[must_use]
    pub fn present(&self, row: &[u8], lane: usize) -> bool {
        (row[self.presence as usize + lane / 8] >> (lane % 8)) & 1 == 1
    }
}

/// Returns `spec`, or fails to compile when it is used in a constant and
/// breaks a rule of [`EpiLaneSpec::validate`].
#[must_use]
pub const fn checked(spec: EpiLaneSpec) -> EpiLaneSpec {
    match spec.validate() {
        Ok(()) => spec,
        Err(SpecError::LaneCount) => panic!("EpiLaneSpec: lanes must be 1..=64"),
        Err(SpecError::OutOfSlab) => panic!("EpiLaneSpec: lanes must lie in the value slab"),
        Err(SpecError::Overlap) => panic!("EpiLaneSpec: value and presence bytes overlap"),
        Err(SpecError::NineBitIsSym) => panic!("EpiLaneSpec: a 9-bit lane is Sym"),
    }
}

/// A class's lane layout, fixed at compile time.
pub trait EpiLaneClass {
    /// Build with [`checked`] so an invalid layout fails to compile.
    const SPEC: EpiLaneSpec;
}

#[cfg(test)]
mod tests {
    use super::*;

    const DR512: EpiLaneSpec = EpiLaneSpec {
        width: LaneWidth::Bits9,
        space: CodeSpace::Sym,
        values: 64,
        presence: 96,
        lanes: 16,
    };

    #[test]
    fn valid_layouts_pass() {
        assert_eq!(DR512.validate(), Ok(()));
        for space in [CodeSpace::Sym, CodeSpace::Asym] {
            let b8 = EpiLaneSpec {
                width: LaneWidth::Bits8,
                space,
                ..DR512
            };
            assert_eq!(b8.validate(), Ok(()));
        }
    }

    #[test]
    fn each_rule_refuses() {
        let e = |s: EpiLaneSpec| s.validate().unwrap_err();
        assert_eq!(e(EpiLaneSpec { lanes: 0, ..DR512 }), SpecError::LaneCount);
        assert_eq!(e(EpiLaneSpec { lanes: 65, ..DR512 }), SpecError::LaneCount);
        assert_eq!(
            e(EpiLaneSpec {
                values: 16,
                ..DR512
            }),
            SpecError::OutOfSlab
        );
        assert_eq!(
            e(EpiLaneSpec {
                values: 500,
                ..DR512
            }),
            SpecError::OutOfSlab
        );
        assert_eq!(
            e(EpiLaneSpec {
                presence: 16,
                ..DR512
            }),
            SpecError::OutOfSlab
        );
        assert_eq!(
            e(EpiLaneSpec {
                presence: 511,
                ..DR512
            }),
            SpecError::OutOfSlab
        );
        assert_eq!(
            e(EpiLaneSpec {
                presence: 70,
                ..DR512
            }),
            SpecError::Overlap
        );
        assert_eq!(
            e(EpiLaneSpec {
                presence: 63,
                ..DR512
            }),
            SpecError::Overlap
        );
        assert_eq!(
            e(EpiLaneSpec {
                space: CodeSpace::Asym,
                ..DR512
            }),
            SpecError::NineBitIsSym
        );
    }

    /// Adjacent regions do not overlap: presence ending where the values
    /// start, and presence starting where the values end, both pass.
    #[test]
    fn adjacent_regions_pass() {
        assert_eq!(
            EpiLaneSpec {
                presence: 62,
                ..DR512
            }
            .validate(),
            Ok(())
        );
        assert_eq!(
            EpiLaneSpec {
                presence: 80,
                ..DR512
            }
            .validate(),
            Ok(())
        );
    }

    #[test]
    fn presence_bytes_round_up() {
        assert_eq!(EpiLaneSpec { lanes: 1, ..DR512 }.presence_bytes(), 1);
        assert_eq!(EpiLaneSpec { lanes: 9, ..DR512 }.presence_bytes(), 2);
        assert_eq!(EpiLaneSpec { lanes: 64, ..DR512 }.presence_bytes(), 8);
    }

    #[test]
    fn zero_point_selects_the_code_space() {
        assert_eq!(CodeSpace::from_zero_point(0), CodeSpace::Sym);
        assert_eq!(CodeSpace::from_zero_point(128), CodeSpace::Asym);
        assert_eq!(CodeSpace::from_zero_point(-3), CodeSpace::Asym);
    }

    /// Presence 0 and value 0 is the only absent code; a 9-bit negative
    /// (presence 0, value > 0) and +0 (presence 1, value 0) are values.
    #[test]
    fn absent_is_presence_zero_and_value_zero() {
        assert!(EpiLaneSpec::absent(0, false));
        assert!(!EpiLaneSpec::absent(5, false), "9-bit -5");
        assert!(!EpiLaneSpec::absent(0, true), "+0");
        assert_eq!(
            (0..=255u8)
                .flat_map(|v| [(v, false), (v, true)])
                .filter(|&(v, p)| EpiLaneSpec::absent(v, p))
                .count(),
            1
        );
    }

    #[test]
    fn presence_bits_are_read_little_endian_past_the_first_byte() {
        let mut row = [0u8; 512];
        row[96] = 0b0000_0100;
        row[97] = 0b0000_0010;
        assert!(DR512.present(&row, 2));
        assert!(DR512.present(&row, 9));
        assert!(!DR512.present(&row, 1));
        assert!(!DR512.present(&row, 8));
    }
}
