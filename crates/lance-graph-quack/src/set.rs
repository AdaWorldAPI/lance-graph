//! A contract [`SetExpr`] as a Quack [`Filter`] (D-VAS-C4, C6).
//!
//! Pure translation: no lane, plane or row is read and nothing is evaluated.
//! Not implemented yet: see [`lower_set`].

use lance_graph_contract::set_algebra::{Logic, SetExpr};

use crate::{Filter, Mask};

/// Why a set could not be lowered, or a lowered set not handed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetLowerError {
    SymOutOfRange(u32),
    NestedConstant,
    LaneNotSupplied { lane: u16 },
    ThreeValued,
    NotThreeValued,
}

/// A lowered set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetLowered {
    All,
    Empty,
    Filter(SetFilter),
}

/// See [`lower_set`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetFilter {}

impl SetFilter {
    pub fn lanes(&self) -> &[u16] {
        &[]
    }
    pub fn planes(&self) -> &[u16] {
        &[]
    }
    pub fn two_valued(&self, _lanes: &[u16]) -> Result<Filter, SetLowerError> {
        Err(SetLowerError::NestedConstant)
    }
    pub fn where_true(
        &self,
        _lanes: &[u16],
        _nullable: &[(u16, Mask)],
    ) -> Result<Filter, SetLowerError> {
        Err(SetLowerError::NestedConstant)
    }
}

pub fn lower_set(_e: &SetExpr, _logic: Logic) -> Result<SetLowered, SetLowerError> {
    Ok(SetLowered::Filter(SetFilter {}))
}
