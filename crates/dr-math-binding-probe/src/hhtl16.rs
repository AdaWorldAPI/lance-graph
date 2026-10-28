//! D-DRCA-1: HHTL16 Cartesian direct routing over the DR64 register.
//!
//! **Lab only.** This module tests the 2026-10-28 DR/DT contract
//! (`.claude/plans/d-drdt-dr-dt-ownership-amendment-v1.md`): DR is coordinate
//! mapping and fold-based projection; it carries no class meaning and no
//! behaviour. Nothing here is a production path or a new ABI.
//!
//! # The address
//!
//! A cell of a bound 64 × 64 register is addressed by the existing
//! [`Morton8x8`] code: `x` is the row coordinate, `y` the column coordinate.
//! That code is the HHTL16 Cartesian address the contract already has (each
//! hex nibble is one 4 × 4 refinement, fan-out 16, `crate::hhtl::FAN_OUT`). No
//! second address system is defined here; this module only reads through it.
//!
//! # Storage stays where it is
//!
//! The relation bits are the borrowed [`BoundDr64`] rows (`rows[x]` bit `y`).
//! Nothing is copied into a second orientation. Rotation is an address map
//! over that one storage, the rule `lance-graph-mask-risc/tests/rotation.rs`
//! sets ("rotation is a coordinate system, not a mask op"):
//!
//! - **Identity orientation** reads `rows[i]`: the properties of identity `i`.
//! - **YZ rotation** swaps the two Cartesian axes of the code
//!   ([`rotate_yz`]). Reading the rotated address `(p, i)` through the map
//!   reads the same bit as `(i, p)`; the addressed relationship does not change.
//! - **Property orientation** ([`column`]) reads every row at one column. The
//!   result is one `u64` of identities, a selection mask, not a copied plane.
//!
//! # Projections
//!
//! | orientation | what | how |
//! |---|---|---|
//! | Identity × Identity (optional) | same-table identity relation | a binding whose row and column axes are both `Identity`; read like any other |
//! | Identity × Property (primary) | fold join through coordinate mapping | [`row`], [`column`], [`cell`] |
//! | Property × Property (derived) | co-occurrence | [`cooccurrence`] (Boolean `Rᵀ ∘ R` from canonical rows, outer-product form, no transpose copy); [`cooccurrence_count`] (one pair, a terminal) |
//! | cross-table Identity × Property | foreign key as a local Cartesian projection | `FK ∘ R_B` through [`crate::relcomp::Chain`], whose domain typing refuses equal ordinals of different tables |
//!
//! Co-occurrence is a derived analytical count. It is not a dependence or a
//! causal claim.
//!
//! # Global → local
//!
//! [`LocalMap`] is the counted global→local preparation (EXO-46). It is built
//! once per frame, holds only the pointer table (local → global, and a sorted
//! index for the inverse), and counts every lookup so a test can prove the
//! projections never call it.
use std::cell::Cell;

use lance_graph_contract::morton8x8::Morton8x8;

use crate::relcomp::Rows;
use crate::{AxisKind, BoundDr64, Population, Truth};

/// Local coordinates are `0..64` (one DR64 register).
pub const LOCAL: usize = 64;

/// Why a local mapping or an address was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// More than 64 identities offered to one 64-slot focus.
    TooMany,
    /// The same global id offered twice.
    Duplicate,
    /// A global id that this map does not hold.
    Unmapped,
    /// A Morton code outside the 64 × 64 register.
    OutOfRegister,
}

/// The prepared global → local pointer table for one table under one frame.
///
/// It holds identity references and local positions only, never values.
/// Lookups are counted (EXO-46: a counted membrane, not a hidden cost).
#[derive(Debug)]
pub struct LocalMap {
    /// local → global (`globals[local]`), in the order the source supplied.
    globals: Vec<u64>,
    /// Globals sorted, with the local position each one maps to.
    sorted: Vec<u64>,
    sorted_local: Vec<u8>,
    lookups: Cell<u64>,
}

impl LocalMap {
    /// Prepare the map. Refuses more than 64 ids and duplicates.
    pub fn prepare(globals: &[u64]) -> Result<Self, MapRefusal> {
        if globals.len() > LOCAL {
            return Err(MapRefusal::TooMany);
        }
        let mut order: Vec<u8> = (0..globals.len() as u8).collect();
        order.sort_unstable_by_key(|&l| globals[l as usize]);
        let sorted: Vec<u64> = order.iter().map(|&l| globals[l as usize]).collect();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(MapRefusal::Duplicate);
        }
        Ok(Self {
            globals: globals.to_vec(),
            sorted,
            sorted_local: order,
            lookups: Cell::new(0),
        })
    }

    /// global → local. Counted.
    pub fn local(&self, global: u64) -> Result<u8, MapRefusal> {
        self.lookups.set(self.lookups.get() + 1);
        self.sorted
            .binary_search(&global)
            .map(|at| self.sorted_local[at])
            .map_err(|_| MapRefusal::Unmapped)
    }

    /// local → global (the inverse). Not counted: it is the pointer table read.
    pub fn global(&self, local: u8) -> Option<u64> {
        self.globals.get(local as usize).copied()
    }

    pub fn len(&self) -> usize {
        self.globals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.globals.is_empty()
    }

    /// How many global → local lookups this map has served.
    pub fn lookups(&self) -> u64 {
        self.lookups.get()
    }

    /// FNV-1a over the local → global table, in local order. A frame states
    /// which coordinate assignment it was sealed under by carrying this value
    /// as its `coordinates` revision, so a selection projected through one
    /// assignment cannot be read under another (D-DRCA-2). A digest, not an
    /// identity: two maps over the same ids in the same order are the same
    /// assignment.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &g in &self.globals {
            for b in g.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// The HHTL16 Cartesian address of cell `(x, y)` of a 64 × 64 register.
#[inline]
pub fn address(x: u8, y: u8) -> Result<Morton8x8, MapRefusal> {
    if usize::from(x) >= LOCAL || usize::from(y) >= LOCAL {
        return Err(MapRefusal::OutOfRegister);
    }
    Ok(Morton8x8::from_xy(x, y))
}

/// The inverse of [`address`]: the `(x, y)` a code addresses.
#[inline]
pub fn coordinates(code: Morton8x8) -> Result<(u8, u8), MapRefusal> {
    let (x, y) = (code.x(), code.y());
    if usize::from(x) >= LOCAL || usize::from(y) >= LOCAL {
        return Err(MapRefusal::OutOfRegister);
    }
    Ok((x, y))
}

/// YZ rotation: swap the two Cartesian axes of a code. An address map; it
/// moves no bits of the stored relation, and
/// it is its own inverse.
#[inline]
pub fn rotate_yz(code: Morton8x8) -> Morton8x8 {
    Morton8x8::from_xy(code.y(), code.x())
}

/// Read the relation bit a code addresses, in the stored orientation.
#[inline]
pub fn cell(rows: &Rows, code: Morton8x8) -> Result<bool, MapRefusal> {
    let (x, y) = coordinates(code)?;
    Ok(rows[x as usize] >> y & 1 == 1)
}

/// Read the relation bit a code addresses in the ROTATED orientation: the
/// code names `(column, row)`, and the read goes through [`rotate_yz`] to the
/// stored cell. Same storage, other coordinates.
#[inline]
pub fn cell_rotated(rows: &Rows, rotated: Morton8x8) -> Result<bool, MapRefusal> {
    cell(rows, rotate_yz(rotated))
}

/// Identity orientation: the column set of row `x` (borrowed word). `None`
/// outside the register, never folded onto a valid row.
#[inline]
pub fn row(r: &BoundDr64<'_>, x: u8) -> Option<u64> {
    r.rows.get(usize::from(x)).copied()
}

/// Property orientation: the rows holding column `y`, read through the
/// address map. One `u64` result, the plane is not copied.
/// `None` outside the register.
#[inline]
pub fn column(rows: &Rows, y: u8) -> Option<u64> {
    if usize::from(y) >= LOCAL {
        return None;
    }
    Some(
        rows.iter()
            .enumerate()
            .fold(0u64, |acc, (x, &w)| acc | ((w >> y) & 1) << x),
    )
}

/// Validity of the column, for a three-valued binding: which rows KNOW
/// column `y`. `None` for a two-valued binding.
#[inline]
pub fn column_validity(r: &BoundDr64<'_>, y: u8) -> Option<u64> {
    r.validity.and_then(|v| column(v, y))
}

/// Property × Property, Boolean: `P[p] = OR over rows x holding p of rows[x]`,
/// i.e. `Rᵀ ∘ R` computed from the canonical rows in outer-product form. No
/// transposed copy is built. Refuses a binding that is not Identity × Property,
/// and one that is not two-valued and distinct: the fold reads payload bits
/// only, so an UNKNOWN cell would count as present (or, zero-filled, as
/// absent), and a bag's multiplicity would be lost. The same rule as
/// `relcomp::Chain::new`.
pub fn cooccurrence(r: &BoundDr64<'_>) -> Option<Rows> {
    if r.method.row.kind != AxisKind::Identity || r.method.column.kind != AxisKind::Property {
        return None;
    }
    if r.method.truth != Truth::TwoValued || r.method.population != Population::Distinct {
        return None;
    }
    let mut out = [0u64; LOCAL];
    for &w in r.rows.iter() {
        let mut bits = w;
        while bits != 0 {
            out[bits.trailing_zeros() as usize] |= w;
            bits &= bits - 1;
        }
    }
    Some(out)
}

/// How many identities hold both properties `p` and `q`: a count terminal
/// over two column masks.
#[inline]
pub fn cooccurrence_count(rows: &Rows, p: u8, q: u8) -> Option<u32> {
    Some((column(rows, p)? & column(rows, q)?).count_ones())
}

/// Count of a selection mask: the same answer in either orientation, since a
/// reducer is invariant under the address map.
#[inline]
pub fn count(rows: &Rows) -> u32 {
    rows.iter().map(|w| w.count_ones()).sum()
}
