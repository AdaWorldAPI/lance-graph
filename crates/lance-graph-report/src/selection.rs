//! Lazy selection: WHICH rows, as a description — never as a bitmap.
//!
//! **Predicate existence is not bitmap existence.** A [`Selection`] is a
//! tree of field ids and fixed-width literals. It lowers into a Quack [`Filter`], and Quack
//! lowers that into straight-line mask ops the evaluator runs TILE BY TILE
//! (at most `TILE_WORDS` words of scratch per slot, whatever the population).
//! No population-sized mask is ever produced because a filter exists.
//!
//! A top-level row range is special-cased for scalar folds: it becomes the
//! evaluator's EXECUTION EXTENT (an outer restriction, no mask op at all), so
//! `Range → Count` folds the validity plane's words over the span and writes
//! nothing (falsifier F4).

use lance_graph_quack::{Cmp, Col, Filter, Mask};

use crate::batch::{AbiBatch, LaneData};
use crate::ids::{FieldId, MaskId};
use crate::ReportError;

/// A half-open row-ordinal range `[lo, hi)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowRange {
    /// First row.
    pub lo: u32,
    /// One past the last row.
    pub hi: u32,
}

/// A fixed-width literal. There is deliberately no text variant: a textual
/// categorical value is resolved to its ordinal at the boundary
/// ([`crate::boundary::CamLabels::ordinal`]) BEFORE it enters a plan (S4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// A signed integer literal (for `I32` fields).
    Int(i64),
    /// A coordinate ordinal / label id (for `U32` fields).
    Ordinal(u32),
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// `=`
    Eq,
    /// `<>`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// A predicate over one field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PredicatePlan {
    /// `column <op> value`.
    Cmp {
        /// Field.
        field: FieldId,
        /// Operator.
        op: CmpOp,
        /// Literal.
        value: Scalar,
    },
    /// `column IN (values)`.
    In {
        /// Field.
        field: FieldId,
        /// Literals.
        values: Vec<Scalar>,
    },
}

/// Which rows participate — lazily. Nothing is evaluated until a terminal
/// fold runs, and then only tile by tile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Selection {
    /// Every valid row (the validity plane).
    All,
    /// A row-ordinal range.
    Range(RowRange),
    /// A column predicate.
    Predicate(PredicatePlan),
    /// A resident mask of the batch (a reused facet mask, a focus).
    Mask(MaskId),
    /// Both.
    And(Box<Selection>, Box<Selection>),
    /// Either.
    Or(Box<Selection>, Box<Selection>),
    /// The first but not the second.
    AndNot(Box<Selection>, Box<Selection>),
}

impl Selection {
    /// `self AND other`, with `All` as the identity (so a rewrite chain that
    /// starts at `All` never grows a no-op conjunct).
    pub fn and(self, other: Selection) -> Selection {
        match (self, other) {
            (Selection::All, s) | (s, Selection::All) => s,
            (a, b) => Selection::And(Box::new(a), Box::new(b)),
        }
    }

    /// `self OR other`.
    pub fn or(self, other: Selection) -> Selection {
        Selection::Or(Box::new(self), Box::new(other))
    }

    /// `self AND NOT other`.
    pub fn and_not(self, other: Selection) -> Selection {
        Selection::AndNot(Box::new(self), Box::new(other))
    }

    /// `field <op> value` shorthand.
    pub fn cmp(field: FieldId, op: CmpOp, value: Scalar) -> Selection {
        Selection::Predicate(PredicatePlan::Cmp { field, op, value })
    }

    /// `field IN (values)` shorthand.
    pub fn is_in(field: FieldId, values: impl IntoIterator<Item = Scalar>) -> Selection {
        Selection::Predicate(PredicatePlan::In {
            field,
            values: values.into_iter().collect(),
        })
    }

    /// Split off the top-level row ranges (the conjuncts reachable through
    /// `And` alone), intersected, from the remainder.
    ///
    /// The range half becomes an execution extent; the remainder a filter.
    /// `None` for the range means no top-level range; `None` for the
    /// remainder means nothing but ranges (read the validity plane).
    pub(crate) fn split_range(&self) -> (Option<RowRange>, Option<Selection>) {
        match self {
            Selection::Range(r) => (Some(*r), None),
            Selection::And(a, b) => {
                let (ra, sa) = a.split_range();
                let (rb, sb) = b.split_range();
                let range = match (ra, rb) {
                    (Some(x), Some(y)) => {
                        let lo = x.lo.max(y.lo);
                        Some(RowRange {
                            lo,
                            hi: x.hi.min(y.hi).max(lo),
                        })
                    }
                    (x, y) => x.or(y),
                };
                let rest = match (sa, sb) {
                    (Some(x), Some(y)) => Some(x.and(y)),
                    (x, y) => x.or(y),
                };
                (range, rest)
            }
            other => (None, Some(other.clone())),
        }
    }

    /// Lower to Quack's filter vocabulary, resolving field ids to lanes
    /// against `batch`. Structural: O(tree size), reads no row.
    pub(crate) fn lower(&self, batch: &AbiBatch) -> Result<Filter, ReportError> {
        Ok(match self {
            Selection::All => Filter::Plane(Mask(0)),
            // Row-ordinal range: `Pred::Range` reads no lane; the `Col` is
            // provenance only. Here the range IS a row range by definition,
            // so no ordered-lane witness is involved.
            Selection::Range(r) => {
                if r.hi as usize > batch.n_rows() || r.lo > r.hi {
                    return Err(ReportError::RangeOutOfBounds {
                        lo: r.lo,
                        hi: r.hi,
                        n_rows: batch.n_rows(),
                    });
                }
                Filter::Cmp(Col(0), Cmp::Range { lo: r.lo, hi: r.hi })
            }
            Selection::Mask(id) => Filter::Plane(Mask(batch.resolve_plane(*id)?)),
            Selection::Predicate(p) => lower_predicate(p, batch)?,
            Selection::And(a, b) => Filter::and([a.lower(batch)?, b.lower(batch)?]),
            Selection::Or(a, b) => Filter::or([a.lower(batch)?, b.lower(batch)?]),
            Selection::AndNot(a, b) => {
                Filter::and([a.lower(batch)?, Filter::negate(b.lower(batch)?)])
            }
        })
    }
}

fn lower_predicate(p: &PredicatePlan, batch: &AbiBatch) -> Result<Filter, ReportError> {
    match p {
        PredicatePlan::Cmp { field, op, value } => leaf(*field, *op, *value, batch),
        PredicatePlan::In { field, values } => {
            if values.is_empty() {
                return Err(ReportError::EmptyIn(*field));
            }
            Ok(Filter::or(
                values
                    .iter()
                    .map(|v| leaf(*field, CmpOp::Eq, *v, batch))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
    }
}

fn leaf(field: FieldId, op: CmpOp, value: Scalar, batch: &AbiBatch) -> Result<Filter, ReportError> {
    let (lane, col) = batch
        .column(field)
        .ok_or(ReportError::UnknownField(field))?;
    let bad = ReportError::BadLiteral { field, value };
    let cmp = match (&col.lane, value) {
        (LaneData::I32(_), Scalar::Int(v)) => {
            let v = i32::try_from(v).map_err(|_| bad)?;
            match op {
                CmpOp::Eq => Cmp::EqI32(v),
                CmpOp::Ne => Cmp::NeI32(v),
                CmpOp::Lt => Cmp::LtI32(v),
                CmpOp::Le => Cmp::LeI32(v),
                CmpOp::Gt => Cmp::GtI32(v),
                CmpOp::Ge => Cmp::GeI32(v),
            }
        }
        (LaneData::U32(_), Scalar::Ordinal(v)) => match op {
            CmpOp::Eq => Cmp::EqU32(v),
            CmpOp::Ne => Cmp::NeU32(v),
            // Ordinals are categorical: the IR has no ordered compare on an
            // unsigned lane. Refused, never approximated.
            _ => return Err(ReportError::OrderedCompareOnOrdinal(field)),
        },
        _ => return Err(bad),
    };
    Ok(Filter::Cmp(Col(lane), cmp))
}
