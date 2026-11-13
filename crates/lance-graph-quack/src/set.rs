//! A contract [`SetExpr`] as a Quack [`Filter`] (D-VAS-C4, C6).
//!
//! `lance_graph_contract::set_algebra` says what a set means. This module
//! spells that meaning in Quack's vocabulary, so a consumer that holds a
//! `SetExpr` (a DT64 guard, a relalg selection) lowers it through
//! [`lower`](crate::lower) / [`lower_fused`](crate::lower_fused) like any
//! other filter. Pure translation: no lane, plane or row is read and nothing
//! is evaluated.
//!
//! The reading is the one the existing library lowering uses
//! (`lance-graph`'s `fold::set_lower`): a residual's `lane` names a value
//! lane, a [`SetExpr::Sym`] names a resident mask plane by its number. What
//! differs is addressing: the caller says which lanes it supplies and in
//! which order, and column `i` of the filter reads the `i`-th of them. A
//! consumer that binds only the lanes a set names supplies
//! [`SetFilter::lanes`]; one whose planes are indexed by lane number supplies
//! `0..n`.
//!
//! # Logic
//!
//! The logic is fixed when the set is lowered and cannot be swapped later:
//!
//! - [`Logic::TwoValued`]: the expression is normalized with
//!   [`normalize`] and handed out by [`SetFilter::two_valued`].
//! - [`Logic::Kleene`]: the expression is normalized with
//!   [`normalize_kleene`], whose result is valid only where TRUE rows are
//!   read, and handed out by [`SetFilter::where_true`], which applies
//!   [`Filter::sql_where`] over the validity planes the caller names. A lane
//!   the caller does not list is read as never NULL; that is the caller's
//!   statement, this module cannot check it.
//!
//! A two-valued normal form is not rewritten for NULLs afterwards (the
//! two-valued rewrites already turned UNKNOWN rows into TRUE ones), and a
//! Kleene set is not handed out as a plain filter.
//!
//! Not here: who issues the planes a `Sym` names or the validity planes, and
//! whether they belong to the population the lanes belong to. Those are the
//! source's to attest.

use lance_graph_contract::set_algebra::{
    normalize, normalize_kleene, Cmp as SetCmp, Logic, Res, SetExpr,
};

use crate::{Cmp, Col, Filter, Mask};

/// Why a set could not be lowered, or a lowered set not handed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetLowerError {
    /// A `Sym` number does not fit a plane index.
    SymOutOfRange(u32),
    /// `All` / `Empty` inside a junction after normalization. Quack has no
    /// constant leaf; the normal forms absorb constants, so this names an
    /// expression they did not reduce.
    NestedConstant,
    /// The set names a lane the caller did not supply.
    LaneNotSupplied { lane: u16 },
    /// More lanes were supplied than a column index can name.
    TooManyLanes { supplied: usize },
    /// A three-valued set was asked for as a two-valued filter.
    ThreeValued,
    /// A two-valued set was asked for as a NULL-aware filter.
    NotThreeValued,
}

/// A lowered set. Quack has no unfiltered table and no constant leaf, so the
/// two constant sets are answers of their own, never filters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetLowered {
    /// Every row.
    All,
    /// No row.
    Empty,
    Filter(SetFilter),
}

/// A normalized, non-constant set, with the lanes and planes it names. The
/// filter is built when the caller says which lanes it supplies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetFilter {
    normal: SetExpr,
    logic: Logic,
    lanes: Vec<u16>,
    planes: Vec<u16>,
}

/// Normalize `e` under `logic` and prepare it as a filter.
pub fn lower_set(e: &SetExpr, logic: Logic) -> Result<SetLowered, SetLowerError> {
    let normal = match logic {
        Logic::TwoValued => normalize(e),
        Logic::Kleene => normalize_kleene(e).into_where_true(),
    };
    match normal {
        SetExpr::All => return Ok(SetLowered::All),
        SetExpr::Empty => return Ok(SetLowered::Empty),
        _ => {}
    }
    let (mut lanes, mut planes) = (Vec::new(), Vec::new());
    names(&normal, &mut lanes, &mut planes)?;
    lanes.sort_unstable();
    lanes.dedup();
    planes.sort_unstable();
    planes.dedup();
    Ok(SetLowered::Filter(SetFilter {
        normal,
        logic,
        lanes,
        planes,
    }))
}

fn names(e: &SetExpr, lanes: &mut Vec<u16>, planes: &mut Vec<u16>) -> Result<(), SetLowerError> {
    match e {
        SetExpr::Res(Res::I32 { lane, .. }) | SetExpr::Res(Res::U32 { lane, .. }) => {
            lanes.push(*lane)
        }
        SetExpr::Sym(sym) => planes.push(plane(*sym)?.0),
        SetExpr::Not(c) => names(c, lanes, planes)?,
        SetExpr::And(cs) | SetExpr::Or(cs) => {
            for c in cs {
                names(c, lanes, planes)?;
            }
        }
        SetExpr::All | SetExpr::Empty => return Err(SetLowerError::NestedConstant),
    }
    Ok(())
}

fn plane(sym: u32) -> Result<Mask, SetLowerError> {
    u16::try_from(sym)
        .map(Mask)
        .map_err(|_| SetLowerError::SymOutOfRange(sym))
}

impl SetFilter {
    /// The logic the set was normalized under.
    pub fn logic(&self) -> Logic {
        self.logic
    }

    /// The value lanes the set names, ascending and distinct.
    pub fn lanes(&self) -> &[u16] {
        &self.lanes
    }

    /// The resident mask planes the set names (`Sym` numbers), ascending and
    /// distinct. The filter reads plane `p` as `Mask(p)`.
    pub fn planes(&self) -> &[u16] {
        &self.planes
    }

    /// The filter of a two-valued set, over `lanes`: column `i` reads
    /// `lanes[i]`.
    pub fn two_valued(&self, lanes: &[u16]) -> Result<Filter, SetLowerError> {
        if self.logic != Logic::TwoValued {
            return Err(SetLowerError::ThreeValued);
        }
        self.build(lanes)
    }

    /// The filter that keeps the rows where a three-valued set is TRUE, over
    /// `lanes` (column `i` reads `lanes[i]`). `nullable` names each lane that
    /// may hold NULL with its resident validity plane.
    pub fn where_true(
        &self,
        lanes: &[u16],
        nullable: &[(u16, Mask)],
    ) -> Result<Filter, SetLowerError> {
        if self.logic != Logic::Kleene {
            return Err(SetLowerError::NotThreeValued);
        }
        let filter = self.build(lanes)?;
        let mut columns = Vec::with_capacity(nullable.len());
        for &(lane, valid) in nullable {
            // A validity plane for a lane that is not supplied gates nothing.
            if let Ok(col) = column(lanes, lane) {
                columns.push((col, valid));
            }
        }
        Ok(filter.sql_where(&columns))
    }

    fn build(&self, lanes: &[u16]) -> Result<Filter, SetLowerError> {
        if lanes.len() > usize::from(u16::MAX) + 1 {
            return Err(SetLowerError::TooManyLanes {
                supplied: lanes.len(),
            });
        }
        filter(&self.normal, lanes)
    }
}

fn column(lanes: &[u16], lane: u16) -> Result<Col, SetLowerError> {
    lanes
        .iter()
        .position(|&l| l == lane)
        .and_then(|i| u16::try_from(i).ok())
        .map(Col)
        .ok_or(SetLowerError::LaneNotSupplied { lane })
}

fn filter(e: &SetExpr, lanes: &[u16]) -> Result<Filter, SetLowerError> {
    Ok(match e {
        SetExpr::Res(Res::I32 { lane, op, t }) => Filter::Cmp(
            column(lanes, *lane)?,
            match op {
                SetCmp::Lt => Cmp::LtI32(*t),
                SetCmp::Le => Cmp::LeI32(*t),
                SetCmp::Gt => Cmp::GtI32(*t),
                SetCmp::Ge => Cmp::GeI32(*t),
                SetCmp::Eq => Cmp::EqI32(*t),
                SetCmp::Ne => Cmp::NeI32(*t),
            },
        ),
        SetExpr::Res(Res::U32 { lane, eq, v }) => Filter::Cmp(
            column(lanes, *lane)?,
            if *eq { Cmp::EqU32(*v) } else { Cmp::NeU32(*v) },
        ),
        SetExpr::Sym(sym) => Filter::Plane(plane(*sym)?),
        SetExpr::Not(c) => Filter::Not(Box::new(filter(c, lanes)?)),
        SetExpr::And(cs) => Filter::And(junction(cs, lanes)?),
        SetExpr::Or(cs) => Filter::Or(junction(cs, lanes)?),
        SetExpr::All | SetExpr::Empty => return Err(SetLowerError::NestedConstant),
    })
}

fn junction(cs: &[SetExpr], lanes: &[u16]) -> Result<Vec<Filter>, SetLowerError> {
    cs.iter().map(|c| filter(c, lanes)).collect()
}
