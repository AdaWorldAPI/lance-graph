//! `register128` — the 128-bit working register with no classid inside it
//! (`D-LXC-29`).
//!
//! # What it is
//!
//! 16 raw bytes in a value-slab rail ([`ValueTenant::Register0`],
//! [`ValueTenant::Register1`]). The register is content-blind: the bytes say
//! nothing about what they mean.
//!
//! # Where its meaning comes from
//!
//! The semantic identity of a register is the SPOG context its population was
//! resolved under, never something stored in the payload:
//!
//! - the concept (classid) comes from [`crate::spog_tenants::graph_of`] of the
//!   row key, resolved through [`crate::hotplug::Activation::resolve_for_context`];
//! - the node is the row's own [`NodeGuid`](crate::canonical_node::NodeGuid);
//! - rung / thought layers stay the sparse alpha mechanism
//!   ([`crate::alpha`]); a register is not a dense per-layer expansion.
//!
//! This is the difference from the Facet96 lanes (`Tekamolo`,
//! `CausalWitness`): their first four bytes ARE a classid. Facet96 is not
//! touched by this module and keeps its meaning.
//!
//! # Binding, once
//!
//! A slab whose metadata declares
//! [`SlabReading::Register128`](crate::hotplug::SlabReading::Register128) is
//! bound by [`ResolvedReading::bind_register128`](crate::hotplug::ResolvedReading::bind_register128),
//! which checks the declaration and the value schema ONCE and returns
//! [`RegisterLanes`]. The hot path reads and writes registers through those
//! lanes; it never resolves a reading, looks up a class or touches a label.
//!
//! # Readings of the bytes
//!
//! Which reading a consumer applies (for example bounded statistics, whose
//! word layout lives with the fold in `ndarray::simd`) is the consumer's
//! choice under its bound context. [`Register128::words`] is the only
//! interpretation this crate provides: four little-endian `u32` words.

use crate::canonical_node::{NodeRow, ValueTenant};

/// One 128-bit working register: 16 raw little-endian bytes, no classid.
///
/// # Examples
///
/// ```
/// use lance_graph_contract::register128::Register128;
///
/// let r = Register128::from_words([1, 2, 3, 0]);
/// assert_eq!(r.words(), [1, 2, 3, 0]);
/// assert_eq!(r.0[..4], 1u32.to_le_bytes());
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Register128(pub [u8; 16]);

/// Width of one register in bytes, equal to its value-slab rail.
pub const REGISTER_BYTES: usize = 16;

impl Register128 {
    /// The empty register.
    pub const ZERO: Self = Self([0; REGISTER_BYTES]);

    /// The register as four little-endian `u32` words.
    #[must_use]
    pub const fn words(&self) -> [u32; 4] {
        let b = &self.0;
        [
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        ]
    }

    /// A register from four `u32` words, stored little-endian.
    #[must_use]
    pub const fn from_words(w: [u32; 4]) -> Self {
        let mut b = [0u8; REGISTER_BYTES];
        let mut i = 0;
        while i < 4 {
            let le = w[i].to_le_bytes();
            b[4 * i] = le[0];
            b[4 * i + 1] = le[1];
            b[4 * i + 2] = le[2];
            b[4 * i + 3] = le[3];
            i += 1;
        }
        Self(b)
    }
}

/// How many register rails a reading needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterRails {
    /// [`ValueTenant::Register0`] only.
    One,
    /// [`ValueTenant::Register0`] and [`ValueTenant::Register1`].
    Two,
}

impl RegisterRails {
    /// The tenants this many rails occupy, in rail order.
    #[must_use]
    pub const fn tenants(self) -> &'static [ValueTenant] {
        match self {
            Self::One => &[ValueTenant::Register0],
            Self::Two => &[ValueTenant::Register0, ValueTenant::Register1],
        }
    }
}

/// The register rails of one bound population, produced ONCE by
/// [`ResolvedReading::bind_register128`](crate::hotplug::ResolvedReading::bind_register128).
///
/// Holds plain numbers: the concept the population was resolved under, and
/// how many rails the binding granted. `Copy`, no references, no strings: it
/// is meant to be handed to the population loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterLanes {
    concept: u16,
    rails: RegisterRails,
}

impl RegisterLanes {
    /// Built only by the binding, which has already checked the slab.
    pub(crate) const fn new(concept: u16, rails: RegisterRails) -> Self {
        Self { concept, rails }
    }

    /// The SPOG concept this population was resolved under. The semantic
    /// identity of every register in it; never read from a payload.
    #[must_use]
    pub const fn concept(&self) -> u16 {
        self.concept
    }

    /// The rails this binding granted.
    #[must_use]
    pub const fn rails(&self) -> RegisterRails {
        self.rails
    }

    /// The value-slab byte range of `rail`, or `None` if this binding did not
    /// grant it.
    #[must_use]
    pub const fn rail_range(&self, rail: usize) -> Option<core::ops::Range<usize>> {
        let tenants = self.rails.tenants();
        if rail >= tenants.len() {
            return None;
        }
        let start = tenants[rail].value_offset();
        Some(start..start + REGISTER_BYTES)
    }

    /// Read `rail` of `row`, or `None` if this binding did not grant it.
    #[must_use]
    pub fn get(&self, row: &NodeRow, rail: usize) -> Option<Register128> {
        let r = self.rail_range(rail)?;
        let mut b = [0u8; REGISTER_BYTES];
        b.copy_from_slice(&row.value[r]);
        Some(Register128(b))
    }

    /// Write `rail` of `row`. Returns `false`, writing nothing, if this
    /// binding did not grant the rail. A successful write is counted against
    /// the rail's tenant, like every other tenant setter
    /// ([`crate::tenant_counter::tenant_update`], a no-op unless the
    /// `tenant-counters` feature is on).
    #[must_use]
    pub fn set(&self, row: &mut NodeRow, rail: usize, reg: Register128) -> bool {
        match self.rail_range(rail) {
            Some(r) => {
                row.value[r].copy_from_slice(&reg.0);
                crate::tenant_counter::tenant_update(self.rails.tenants()[rail]);
                true
            }
            None => false,
        }
    }
}

/// How the bytes of a signed register were written. Declared by the slab
/// ([`SlabReading::RegisterI4x32`](crate::hotplug::SlabReading::RegisterI4x32),
/// [`SlabReading::RegisterI8x16`](crate::hotplug::SlabReading::RegisterI8x16)),
/// never chosen by a reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterCarving {
    /// 32 signed `i4`, range `-8..=7`. Dim `2k` is the low nibble of byte
    /// `k`, dim `2k+1` the high nibble ([`crate::atoms::I4x32`]).
    I4x32,
    /// 16 signed `i8`, range `-128..=127`, dim `k` in byte `k`.
    I8x16,
}

impl RegisterCarving {
    /// Number of dimensions in one register.
    #[must_use]
    pub const fn dims(self) -> usize {
        match self {
            Self::I4x32 => 32,
            Self::I8x16 => 16,
        }
    }
}

/// What the signed values of a bound population mean. Two authorities, which
/// must agree: the concept's law, declared by the
/// [`Activation`](crate::hotplug::Activation)
/// ([`register_law_for`](crate::hotplug::Activation::register_law_for)), and the
/// law the slab recorded when its bytes were written
/// ([`SlabReading::RegisterI4x32`](crate::hotplug::SlabReading::RegisterI4x32)).
/// The binder chooses neither. Checked again on every read and write.
///
/// Each law is a different kind of quantity, so a value written under one
/// is not a value under another even when the bytes are identical. That is
/// the mixing an earlier 24×i4 accumulation got wrong, and why no law here
/// converts into another.
///
/// Epistemic state is deliberately absent: it is defined on `CausalEdge64`
/// bits 59..63 (`EpistemicState5`) and is not re-encoded as a signed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterLaw {
    /// A relative address in a local window: `-3` means "three positions
    /// back". An address, not a strength. A referent outside the window is
    /// not encoded here; it is reached through a basin or graph edge.
    RelativeOffset,
    /// A signed position on a declared semantic axis (e.g. `terrestrial`:
    /// fox `+7`, whale `-5`). Independent of the `is_a` taxonomy: a negative
    /// position never removes a class membership.
    AxisPosition,
    /// Evidence stance: positive supports, zero is unresolved, negative
    /// falsifies.
    Support,
}

/// Why a signed register read or write was refused. A refused write leaves
/// the row unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignedRegisterRefusal {
    /// The binding did not grant this rail.
    RailAbsent {
        /// The rail asked for.
        rail: usize,
    },
    /// The caller asked for a different carving than the slab declared.
    CarvingMismatch {
        /// The carving the slab declared.
        bound: RegisterCarving,
        /// The carving the caller asked for.
        asked: RegisterCarving,
    },
    /// The caller asked for a different law than the binding declared.
    LawMismatch {
        /// The law the population was bound under.
        bound: RegisterLaw,
        /// The law the caller asked for.
        asked: RegisterLaw,
    },
    /// A value does not fit the carving (an `i4` outside `-8..=7`).
    OutOfRange {
        /// The dimension holding it.
        dim: usize,
        /// The value.
        value: i8,
    },
}

/// The signed register rails of one bound population, produced ONCE by
/// [`ResolvedReading::bind_signed_register`](crate::hotplug::ResolvedReading::bind_signed_register).
///
/// Carries the concept, the granted rails, the slab's carving and the
/// declared law. Reads decode one rail into an owned array of 16 or 32
/// values; nothing else is copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignedRegisterLanes {
    concept: u16,
    rails: RegisterRails,
    carving: RegisterCarving,
    law: RegisterLaw,
}

impl SignedRegisterLanes {
    /// Built only by the binding, which has already checked the slab.
    pub(crate) const fn new(
        concept: u16,
        rails: RegisterRails,
        carving: RegisterCarving,
        law: RegisterLaw,
    ) -> Self {
        Self {
            concept,
            rails,
            carving,
            law,
        }
    }

    /// The SPOG concept this population was resolved under.
    #[must_use]
    pub const fn concept(&self) -> u16 {
        self.concept
    }

    /// The rails this binding granted.
    #[must_use]
    pub const fn rails(&self) -> RegisterRails {
        self.rails
    }

    /// The carving the slab declared.
    #[must_use]
    pub const fn carving(&self) -> RegisterCarving {
        self.carving
    }

    /// The law this population was bound under.
    #[must_use]
    pub const fn law(&self) -> RegisterLaw {
        self.law
    }

    /// The value-slab byte range of `rail`, after checking rail, carving and
    /// law in that order.
    fn check(
        &self,
        rail: usize,
        carving: RegisterCarving,
        law: RegisterLaw,
    ) -> Result<core::ops::Range<usize>, SignedRegisterRefusal> {
        let tenants = self.rails.tenants();
        if rail >= tenants.len() {
            return Err(SignedRegisterRefusal::RailAbsent { rail });
        }
        if carving != self.carving {
            return Err(SignedRegisterRefusal::CarvingMismatch {
                bound: self.carving,
                asked: carving,
            });
        }
        if law != self.law {
            return Err(SignedRegisterRefusal::LawMismatch {
                bound: self.law,
                asked: law,
            });
        }
        let start = tenants[rail].value_offset();
        Ok(start..start + REGISTER_BYTES)
    }

    /// Read `rail` as 32 signed `i4` under `law`.
    ///
    /// # Errors
    ///
    /// [`SignedRegisterRefusal`] when the rail is absent, the slab is not
    /// carved as [`RegisterCarving::I4x32`], or `law` is not the bound law.
    pub fn read_i4x32(
        &self,
        row: &NodeRow,
        rail: usize,
        law: RegisterLaw,
    ) -> Result<[i8; 32], SignedRegisterRefusal> {
        let r = self.check(rail, RegisterCarving::I4x32, law)?;
        let mut b = [0u8; REGISTER_BYTES];
        b.copy_from_slice(&row.value[r]);
        Ok(crate::atoms::I4x32::from_bytes(b).unpack())
    }

    /// Read `rail` as 16 signed `i8` under `law`.
    ///
    /// # Errors
    ///
    /// As [`read_i4x32`](Self::read_i4x32), for [`RegisterCarving::I8x16`].
    pub fn read_i8x16(
        &self,
        row: &NodeRow,
        rail: usize,
        law: RegisterLaw,
    ) -> Result<[i8; 16], SignedRegisterRefusal> {
        let r = self.check(rail, RegisterCarving::I8x16, law)?;
        let bytes = &row.value[r];
        Ok(core::array::from_fn(|i| bytes[i] as i8))
    }

    /// Write `rail` as 32 signed `i4` under `law`. Every value must lie in
    /// `-8..=7`; nothing is saturated, and nothing is written unless all
    /// checks pass.
    ///
    /// # Errors
    ///
    /// As [`read_i4x32`](Self::read_i4x32), plus
    /// [`SignedRegisterRefusal::OutOfRange`].
    pub fn write_i4x32(
        &self,
        row: &mut NodeRow,
        rail: usize,
        law: RegisterLaw,
        values: &[i8; 32],
    ) -> Result<(), SignedRegisterRefusal> {
        let r = self.check(rail, RegisterCarving::I4x32, law)?;
        if let Some((dim, &value)) = values
            .iter()
            .enumerate()
            .find(|(_, v)| !(-8..=7).contains(*v))
        {
            return Err(SignedRegisterRefusal::OutOfRange { dim, value });
        }
        let packed = crate::atoms::I4x32::pack(values);
        row.value[r].copy_from_slice(packed.as_bytes());
        crate::tenant_counter::tenant_update(self.rails.tenants()[rail]);
        Ok(())
    }

    /// Write `rail` as 16 signed `i8` under `law`.
    ///
    /// # Errors
    ///
    /// As [`read_i8x16`](Self::read_i8x16). Nothing is written on refusal.
    pub fn write_i8x16(
        &self,
        row: &mut NodeRow,
        rail: usize,
        law: RegisterLaw,
        values: &[i8; 16],
    ) -> Result<(), SignedRegisterRefusal> {
        let r = self.check(rail, RegisterCarving::I8x16, law)?;
        for (dst, &v) in row.value[r].iter_mut().zip(values.iter()) {
            *dst = v as u8;
        }
        crate::tenant_counter::tenant_update(self.rails.tenants()[rail]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    fn patterned_row() -> NodeRow {
        NodeRow {
            key: crate::canonical_node::NodeGuid::new(0x0901_0000, 1, 2, 3, 0x66, 11),
            edges: Default::default(),
            value: core::array::from_fn(|i| (i % 251) as u8),
        }
    }

    const LAWS: [RegisterLaw; 3] = [
        RegisterLaw::RelativeOffset,
        RegisterLaw::AxisPosition,
        RegisterLaw::Support,
    ];

    /// FAILS IF: an i4 value does not round-trip, or the nibble layout
    /// differs from `atoms::I4x32` (dim 2k low nibble of byte k).
    #[test]
    fn i4x32_round_trips_every_value_in_every_dim() {
        let lanes = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::Two,
            RegisterCarving::I4x32,
            RegisterLaw::AxisPosition,
        );
        for v in -8i8..=7 {
            for dim in 0..32 {
                let mut values = [0i8; 32];
                values[dim] = v;
                values[(dim + 1) % 32] = -v.clamp(-7, 7);
                let mut row = patterned_row();
                lanes
                    .write_i4x32(&mut row, 1, RegisterLaw::AxisPosition, &values)
                    .unwrap();
                assert_eq!(
                    lanes.read_i4x32(&row, 1, RegisterLaw::AxisPosition),
                    Ok(values)
                );
                let r = lanes
                    .check(1, RegisterCarving::I4x32, RegisterLaw::AxisPosition)
                    .unwrap();
                let byte = row.value[r.start + dim / 2];
                let nibble = if dim % 2 == 0 { byte & 0xF } else { byte >> 4 };
                assert_eq!(nibble, (v as u8) & 0xF, "dim {dim} nibble");
            }
        }
    }

    /// FAILS IF: an i8 value does not round-trip, or dim k is not byte k.
    #[test]
    fn i8x16_round_trips_and_dim_k_is_byte_k() {
        let lanes = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::Two,
            RegisterCarving::I8x16,
            RegisterLaw::RelativeOffset,
        );
        for base in [-128i8, -100, -1, 0, 1, 77, 127] {
            let values: [i8; 16] =
                core::array::from_fn(|k| base.wrapping_add((k as i8).wrapping_mul(17)));
            let mut row = patterned_row();
            lanes
                .write_i8x16(&mut row, 1, RegisterLaw::RelativeOffset, &values)
                .unwrap();
            assert_eq!(
                lanes.read_i8x16(&row, 1, RegisterLaw::RelativeOffset),
                Ok(values)
            );
            let r = lanes.rails.tenants()[1].value_offset();
            for (k, &v) in values.iter().enumerate() {
                assert_eq!(row.value[r + k] as i8, v);
            }
        }
    }

    /// FAILS IF: the carving is not load-bearing: the same 16 bytes must
    /// mean different values under the two carvings.
    #[test]
    fn the_same_bytes_differ_under_the_two_carvings() {
        let mut row = patterned_row();
        let i8s = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::Two,
            RegisterCarving::I8x16,
            RegisterLaw::Support,
        );
        let i4s = SignedRegisterLanes {
            carving: RegisterCarving::I4x32,
            ..i8s
        };
        let mut v8 = [0i8; 16];
        v8[0] = 0xF1u8 as i8; // -15 as i8; nibbles (+1, -1) as i4
        i8s.write_i8x16(&mut row, 1, RegisterLaw::Support, &v8)
            .unwrap();
        let as4 = i4s.read_i4x32(&row, 1, RegisterLaw::Support).unwrap();
        assert_eq!(v8[0], -15);
        assert_eq!((as4[0], as4[1]), (1, -1));
    }

    /// FAILS IF: a read or write under a law other than the bound one is
    /// accepted, or a refused write changes the row.
    #[test]
    fn a_different_law_is_refused_and_writes_nothing() {
        for bound in LAWS {
            for asked in LAWS {
                for carving in [RegisterCarving::I4x32, RegisterCarving::I8x16] {
                    let lanes =
                        SignedRegisterLanes::new(0x0901, RegisterRails::Two, carving, bound);
                    let mut row = patterned_row();
                    let before = row.value;
                    let (read_ok, write) = match carving {
                        RegisterCarving::I4x32 => (
                            lanes.read_i4x32(&row, 1, asked).is_ok(),
                            lanes.write_i4x32(&mut row, 1, asked, &[3; 32]),
                        ),
                        RegisterCarving::I8x16 => (
                            lanes.read_i8x16(&row, 1, asked).is_ok(),
                            lanes.write_i8x16(&mut row, 1, asked, &[3; 16]),
                        ),
                    };
                    if asked == bound {
                        assert!(read_ok && write.is_ok());
                    } else {
                        assert!(!read_ok);
                        assert_eq!(
                            write,
                            Err(SignedRegisterRefusal::LawMismatch { bound, asked })
                        );
                        assert_eq!(row.value, before, "a refused write wrote");
                    }
                }
            }
        }
    }

    /// FAILS IF: a reader can choose a carving other than the slab's.
    #[test]
    fn a_different_carving_is_refused() {
        let lanes = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::Two,
            RegisterCarving::I4x32,
            RegisterLaw::Support,
        );
        let mut row = patterned_row();
        let before = row.value;
        let refused = Err(SignedRegisterRefusal::CarvingMismatch {
            bound: RegisterCarving::I4x32,
            asked: RegisterCarving::I8x16,
        });
        assert_eq!(
            lanes.read_i8x16(&row, 1, RegisterLaw::Support).map(|_| ()),
            refused
        );
        assert_eq!(
            lanes.write_i8x16(&mut row, 1, RegisterLaw::Support, &[1; 16]),
            refused
        );
        assert_eq!(row.value, before);
    }

    /// FAILS IF: an out-of-range i4 is saturated or partly written instead
    /// of refused. The bad value is in the LAST dim so a write that checks
    /// as it goes would already have written the others.
    #[test]
    fn an_out_of_range_i4_is_refused_before_writing() {
        let lanes = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::Two,
            RegisterCarving::I4x32,
            RegisterLaw::AxisPosition,
        );
        for bad in [8i8, 9, 127, -9, -128] {
            let mut values = [5i8; 32];
            values[31] = bad;
            let mut row = patterned_row();
            let before = row.value;
            assert_eq!(
                lanes.write_i4x32(&mut row, 1, RegisterLaw::AxisPosition, &values),
                Err(SignedRegisterRefusal::OutOfRange {
                    dim: 31,
                    value: bad
                })
            );
            assert_eq!(row.value, before);
        }
    }

    /// FAILS IF: a one-rail binding reaches rail 1, or a write touches any
    /// byte outside its rail.
    #[test]
    fn signed_writes_stay_inside_their_rail() {
        let one = SignedRegisterLanes::new(
            0x0901,
            RegisterRails::One,
            RegisterCarving::I8x16,
            RegisterLaw::Support,
        );
        let mut row = patterned_row();
        assert_eq!(
            one.write_i8x16(&mut row, 1, RegisterLaw::Support, &[1; 16]),
            Err(SignedRegisterRefusal::RailAbsent { rail: 1 })
        );
        let two = SignedRegisterLanes {
            rails: RegisterRails::Two,
            ..one
        };
        let before = row.value;
        two.write_i8x16(&mut row, 1, RegisterLaw::Support, &[-1; 16])
            .unwrap();
        let r = two
            .check(1, RegisterCarving::I8x16, RegisterLaw::Support)
            .unwrap();
        for (i, (a, b)) in before.iter().zip(row.value.iter()).enumerate() {
            if !r.contains(&i) {
                assert_eq!(a, b, "byte {i} outside rail 1 changed");
            }
        }
    }

    use super::*;
    use crate::canonical_node::VALUE_TENANTS;

    /// The register width is the declared rail width, for both rails.
    #[test]
    fn the_register_is_exactly_one_declared_rail() {
        for t in [ValueTenant::Register0, ValueTenant::Register1] {
            assert_eq!(
                VALUE_TENANTS[t as usize].col_bytes_per_row(),
                REGISTER_BYTES
            );
            assert_eq!(t.byte_len(), REGISTER_BYTES);
        }
        assert_eq!(core::mem::size_of::<Register128>(), 16);
    }

    /// Words are little-endian and round-trip, with every word distinct so a
    /// swapped pair cannot pass.
    #[test]
    fn words_round_trip_little_endian() {
        let w = [0x0403_0201, 0x0807_0605, 0x0C0B_0A09, 0x100F_0E0D];
        let r = Register128::from_words(w);
        assert_eq!(r.0, core::array::from_fn::<u8, 16, _>(|i| i as u8 + 1));
        assert_eq!(r.words(), w);
    }

    /// FAILS IF: the two rails overlap each other or any other tenant, or do
    /// not append exactly where `EpisodicBasin` ends (field isolation for a
    /// layout that gains lanes, `I-LEGACY-API-FEATURE-GATED`).
    #[test]
    fn the_rails_touch_no_other_tenant() {
        let range = |t: ValueTenant| {
            let d = VALUE_TENANTS[t as usize];
            let s = d.row_offset as usize;
            s..s + d.col_bytes_per_row()
        };
        assert_eq!(range(ValueTenant::Register0), 252..268);
        assert_eq!(range(ValueTenant::Register1), 268..284);
        assert_eq!(
            range(ValueTenant::EpisodicBasin).end,
            252,
            "additive, not reclaiming"
        );
        for rail in [ValueTenant::Register0, ValueTenant::Register1] {
            let r = range(rail);
            for other in VALUE_TENANTS {
                if other.name_id == rail as u16 {
                    continue;
                }
                let (os, oe) = (
                    other.row_offset as usize,
                    other.row_offset as usize + other.col_bytes_per_row(),
                );
                assert!(
                    r.end <= os || oe <= r.start,
                    "{rail:?} overlaps tenant {}",
                    other.name_id
                );
            }
        }
    }

    /// FAILS IF: writing a rail touches any other byte of the row, or a
    /// one-rail binding can reach rail 1.
    #[test]
    fn writing_a_rail_leaves_every_other_byte_alone() {
        let lanes = RegisterLanes::new(0x0901, RegisterRails::Two);
        let mut row = NodeRow {
            key: crate::canonical_node::NodeGuid::new(0x0901_0000, 1, 2, 3, 0x66, 7),
            edges: Default::default(),
            value: core::array::from_fn(|i| (i % 251) as u8),
        };
        let before = row.value;
        let reg = Register128::from_words([u32::MAX, 1, 2, 3]);
        assert!(lanes.set(&mut row, 1, reg));
        assert_eq!(lanes.get(&row, 1), Some(reg));
        let r1 = lanes.rail_range(1).unwrap();
        for (i, (a, b)) in before.iter().zip(row.value.iter()).enumerate() {
            if !r1.contains(&i) {
                assert_eq!(a, b, "byte {i} outside rail 1 changed");
            }
        }
        let one = RegisterLanes::new(0x0901, RegisterRails::One);
        assert_eq!(one.get(&row, 1), None);
        assert!(!one.set(&mut row, 1, Register128::ZERO));
        assert_eq!(
            lanes.get(&row, 1),
            Some(reg),
            "a refused write wrote nothing"
        );
    }

    /// FAILS IF: a successful register write is not counted against its
    /// tenant, or a refused one is. This is the only test in the crate that
    /// writes `Register0` (every other register test writes rail 1), so the
    /// delta is exact even with tests running in parallel. Keep it that way.
    #[cfg(feature = "tenant-counters")]
    #[test]
    fn register_writes_are_counted_per_tenant() {
        use crate::tenant_counter::tenant_count;
        let mut row = NodeRow {
            key: crate::canonical_node::NodeGuid::new(0x0901_0000, 1, 2, 3, 0x66, 9),
            edges: Default::default(),
            value: [0; 480],
        };
        let before = tenant_count(ValueTenant::Register0);
        let one = RegisterLanes::new(0x0901, RegisterRails::One);
        assert!(one.set(&mut row, 0, Register128::from_words([1, 2, 3, 0])));
        assert!(one.set(&mut row, 0, Register128::ZERO));
        assert!(!one.set(&mut row, 1, Register128::ZERO), "refused");
        assert_eq!(tenant_count(ValueTenant::Register0), before + 2);
    }
}
