//! D-DRCA-2: `DR256::WHERE` × DR#3, lab only.
//!
//! The invariant under test:
//!
//! ```text
//! global selection → optional amortized global→local mapping
//!                  → HHTL16 Cartesian projection → native fold
//! ```
//!
//! # What this module does not do
//!
//! It evaluates no predicate. The global selection is the output of Quack's
//! existing typed WHERE (`Query { filter, agg: Agg::Rows }` → mask-risc
//! `Keep`, three-valued through `Filter::sql_where`), handed in as one
//! borrowed 8 KiB mask: the rows where the criteria are TRUE. This module
//! has no Quack or mask-risc dependency; the tests and the stage example
//! wire the real engine to it.
//!
//! It stores nothing. A [`LocalSelection`] is a caller-owned value, the
//! projection of one selection into one frame, with the guards it was made
//! under. There is no cache, map of selections or persistent mask store.
//!
//! # The three steps
//!
//! 1. **Global → local** ([`project`]): for each local ordinal `l` of the
//!    table's [`LocalMap`], test bit `global(l)` of the WHERE mask.
//!    The work is `map.len()` probes, never a pass over 65,536 rows, and a
//!    selected identity outside the frame is not promoted into it. The frame
//!    must carry `map.digest()` as its `coordinates` revision, so a selection
//!    cannot be projected through one assignment and read under another.
//! 2. **Admission** ([`LocalSelection::admit`]): every read re-checks the
//!    global source seal (owner, source revision, codebook, address space),
//!    the local frame, the focus, the scope, the identity axis, the DR
//!    method and the criteria. Any mismatch refuses; nothing is silently
//!    reused. The only way to read is through the [`Admitted`] it returns.
//! 3. **Fold** ([`selected_count`], [`selected_column`],
//!    [`selected_column_k3`], [`selected_cell`]): the selection is a `u64`
//!    over the identity axis, ANDed into reads of the borrowed rows. No
//!    Cartesian product is built and no row is copied. The Boolean folds
//!    refuse a three-valued or bag register; [`selected_column_k3`] reads
//!    value and validity and serves both.
//!
//! # Epistemic absence
//!
//! `$null`, NaN, SQL NULL, an unset Arrow validity bit and K3 UNKNOWN are one
//! logical value here: no TRUE/FALSE judgment has been established. It says
//! nothing about whether the entity or relationship exists, and it carries no
//! finer epistemic state (that is CausalEdge64 Epi5, under its own evidence
//! contract, and is not derived from absence alone).
//!
//! - WHERE selects TRUE. A slot that is not selected is FALSE **or** UNKNOWN;
//!   [`Admitted`] gives no complement, because the complement of a selection
//!   would read UNKNOWN as FALSE.
//! - `NOT UNKNOWN` is UNKNOWN. A negated selection is a second WHERE over the
//!   negated criteria, evaluated by Quack's typed three-valued `sql_where`,
//!   never the complement of the first.
//! - A three-valued property keeps the existing value + validity
//!   representation (Arrow-style, `BoundDr64::validity`); no separate FALSE
//!   mask is introduced.
use crate::hhtl16::{self, LocalMap};
use crate::relcomp::Rows;
use crate::{
    Axis, BoundDr64, Focus, Frame, FrameStore, Guards, MethodSignature, Population, Refusal, Scope,
    Truth,
};

/// Addresses in one DR256 global population.
pub const GLOBAL: usize = 65_536;
/// `u64` words in one global mask (8 KiB).
pub const GLOBAL_WORDS: usize = GLOBAL / 64;

/// What the global selection was bound under, besides the source seal: the
/// identity of the criteria (a caller-supplied digest of the bound query).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Criteria {
    pub id: u64,
}

/// Why a projection was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectRefusal {
    /// A global mask that is not exactly [`GLOBAL_WORDS`] words.
    WrongWidth,
    /// A local slot whose global id lies outside the 65,536 addresses.
    OutOfUniverse,
    /// The local frame was not sealed under this map's coordinate assignment.
    MapNotFrame,
    /// The global source seal is not current.
    Global(Refusal),
    /// The local frame is not current.
    Local(Refusal),
}

/// Why a reuse was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReuseRefusal {
    /// The global source moved: another owner (`ForeignFrame`), new data
    /// (`SourceChanged`), a new codebook (`MapChanged`), a new address space
    /// (`CoordinatesChanged`) or a reseal.
    Global(Refusal),
    /// The local frame moved, the same five ways.
    Local(Refusal),
    /// The register was bound under a different frame issue.
    DifferentFrame,
    /// The register's row axis is not the axis the selection was projected
    /// onto (equal local ordinals of two tables).
    AxisMismatch,
    /// The register's DR method is not the one the selection was made for.
    SchemaChanged,
    FocusChanged,
    ScopeChanged,
    CriteriaChanged,
}

/// One global selection projected into one table's frame. Caller-owned;
/// readable only through [`LocalSelection::admit`].
#[derive(Clone, Debug)]
pub struct LocalSelection {
    truth: u64,
    slots: u64,
    axis: Axis,
    method: MethodSignature,
    global: Frame,
    local: Frame,
    focus: Focus,
    scope: Scope,
    criteria: Criteria,
    probes: u32,
}

/// The selection, after every guard held. No public constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admitted {
    truth: u64,
    slots: u64,
    /// The register admitted against is two-valued and distinct, so the
    /// Boolean folds may read its payload bits as presence.
    boolean: bool,
}

impl Admitted {
    /// Identities where the criteria are TRUE: what WHERE keeps. The other
    /// occupied slots are FALSE or UNKNOWN, and this type does not say which.
    pub fn truth(&self) -> u64 {
        self.truth
    }
    /// The occupied local slots of the frame.
    pub fn slots(&self) -> u64 {
        self.slots
    }
}

fn bit(mask: &[u64], g: u64) -> bool {
    (mask[(g >> 6) as usize] >> (g & 63)) & 1 == 1
}

/// What one global selection is projected under.
#[derive(Clone, Copy, Debug)]
pub struct Under<'a> {
    /// The owner of the 65,536-address source, and its seal.
    pub global_store: &'a FrameStore,
    pub global: &'a Frame,
    /// The owner of the DR frame, and the guards of the target register.
    pub local_store: &'a FrameStore,
    pub guards: &'a Guards,
    /// The identity axis and DR method the selection is for.
    pub axis: Axis,
    pub method: MethodSignature,
    pub criteria: Criteria,
}

/// Project a global WHERE selection into one table's frame.
///
/// `truth` is the `Keep` output of WHERE: the rows where the criteria are
/// TRUE. Work: `map.len()` probes into it.
pub fn project(
    map: &LocalMap,
    truth: &[u64],
    under: Under<'_>,
) -> Result<LocalSelection, ProjectRefusal> {
    if truth.len() != GLOBAL_WORDS {
        return Err(ProjectRefusal::WrongWidth);
    }
    under
        .global_store
        .check(under.global)
        .map_err(ProjectRefusal::Global)?;
    under
        .local_store
        .check(&under.guards.frame)
        .map_err(ProjectRefusal::Local)?;
    if under.guards.frame.revisions().coordinates != map.digest() {
        return Err(ProjectRefusal::MapNotFrame);
    }
    let (mut t, mut slots) = (0u64, 0u64);
    for l in 0..map.len() as u8 {
        let g = map.global(l).ok_or(ProjectRefusal::OutOfUniverse)?;
        if g >= GLOBAL as u64 {
            return Err(ProjectRefusal::OutOfUniverse);
        }
        t |= u64::from(bit(truth, g)) << l;
        slots |= 1 << l;
    }
    Ok(LocalSelection {
        truth: t,
        slots,
        axis: under.axis,
        method: under.method,
        global: under.global.clone(),
        local: under.guards.frame.clone(),
        focus: under.guards.focus,
        scope: under.guards.scope,
        criteria: under.criteria,
        probes: map.len() as u32,
    })
}

impl LocalSelection {
    /// Mask probes the projection made (the global → local work).
    pub fn probes(&self) -> u32 {
        self.probes
    }

    /// Re-check every guard against the register about to be read.
    pub fn admit(
        &self,
        global_store: &FrameStore,
        local_store: &FrameStore,
        register: &BoundDr64<'_>,
        criteria: Criteria,
    ) -> Result<Admitted, ReuseRefusal> {
        global_store
            .check(&self.global)
            .map_err(ReuseRefusal::Global)?;
        local_store
            .check(&self.local)
            .map_err(ReuseRefusal::Local)?;
        if !register.frame.same_issue(&self.local) {
            return Err(ReuseRefusal::DifferentFrame);
        }
        if register.method.row != self.axis {
            return Err(ReuseRefusal::AxisMismatch);
        }
        if register.method != self.method {
            return Err(ReuseRefusal::SchemaChanged);
        }
        if register.focus != self.focus {
            return Err(ReuseRefusal::FocusChanged);
        }
        if register.scope != self.scope {
            return Err(ReuseRefusal::ScopeChanged);
        }
        if criteria != self.criteria {
            return Err(ReuseRefusal::CriteriaChanged);
        }
        Ok(Admitted {
            truth: self.truth,
            slots: self.slots,
            boolean: register.method.truth == Truth::TwoValued
                && register.method.population == Population::Distinct,
        })
    }
}

// The Boolean folds below read payload bits as presence. They refuse (return
// `None`) when the selection was admitted against a three-valued or bag
// register, where a payload bit under an UNKNOWN cell would count as present
// and a zero-filled one as absent: the rule `hhtl16::cooccurrence` and
// `relcomp::Chain::new` apply. `rows` must be the admitted register's rows or
// a Boolean derivation of them (a `Chain` composition).

/// Identity × Property cells over the selected identities: Σ over set bits
/// `x` of the TRUE selection of `popcount(rows[x])`.
pub fn selected_count(rows: &Rows, a: &Admitted) -> Option<u32> {
    if !a.boolean {
        return None;
    }
    let (mut s, mut n) = (a.truth, 0u32);
    while s != 0 {
        let x = s.trailing_zeros() as usize;
        n += rows[x].count_ones();
        s &= s - 1;
    }
    Some(n)
}

/// The selected identities holding property `y`: one column read, ANDed.
pub fn selected_column(rows: &Rows, a: &Admitted, y: u8) -> Option<u64> {
    if !a.boolean {
        return None;
    }
    hhtl16::column(rows, y).map(|c| c & a.truth)
}

/// Column `y` over the selected identities in the register's own
/// representation: `(value, validity)`, both restricted to the selection.
/// A selected identity whose validity bit is clear is UNKNOWN, whatever its
/// value bit holds; the value bit is masked by validity so a stale payload
/// under an UNKNOWN cell is never returned as TRUE. A two-valued register is
/// valid at every selected identity.
pub fn selected_column_k3(r: &BoundDr64<'_>, a: &Admitted, y: u8) -> Option<(u64, u64)> {
    let sel = a.truth;
    let col = hhtl16::column(r.rows, y)?;
    let known = match r.validity {
        Some(_) => hhtl16::column_validity(r, y)?,
        None => u64::MAX,
    };
    Some((col & known & sel, known & sel))
}

/// One cell `(x, y)`, read only if `x` is selected. `None` outside the
/// register or for a non-Boolean admission.
pub fn selected_cell(rows: &Rows, a: &Admitted, x: u8, y: u8) -> Option<bool> {
    if !a.boolean {
        return None;
    }
    let code = hhtl16::address(x, y).ok()?;
    Some((a.truth >> x) & 1 == 1 && hhtl16::cell(rows, code).ok()?)
}
