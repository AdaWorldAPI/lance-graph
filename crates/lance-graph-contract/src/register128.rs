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

#[cfg(test)]
mod tests {
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
    /// tenant, or a refused one is. Only this test writes rail 0, so the
    /// delta is exact even with tests running in parallel.
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
