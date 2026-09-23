//! The report plan: control plane only.
//!
//! A [`ReportPlan`] names a source, a lazy [`Selection`], a set of axes with
//! ROLES, and measures. Every rewrite on it — filter, add/remove an axis,
//! reassign a role, pivot, rotate — is a metadata edit that touches no row.
//!
//! **Pivot defines coordinates. Fold populates coordinates. Materialization
//! renders coordinates.**
//!
//! The split that makes rotation free is [`ReportPlan::physical_key`]: the
//! fold's work depends on WHICH axes exist, never on which role each plays.
//! Two plans that differ only in roles share a physical key, so a result
//! computed for one is reinterpreted for the other without a rescan
//! (falsifiers F13 / F14).

use crate::ids::{FieldId, SourceId};
use crate::selection::Selection;

/// The role an axis plays in the presented coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisRole {
    /// Down the side.
    Row,
    /// Across the top.
    Column,
    /// One page (facet) per member.
    Page,
}

/// A coordinate provider — WHERE in the aggregate space a row lands.
///
/// Two providers, and neither knows what it measures:
///
/// * [`CoordSpec::Field`] — a resident `U32` code lane with a declared domain.
///   It can serve as the fold KEY (the one dimension a single substrate pass
///   scatters into).
/// * [`CoordSpec::Bucket`] — a DERIVED coordinate: `floor((v - origin) /
///   width)` over an `I32` lane, `count` members. It is never materialized as
///   a population lane: each member is a pair of tile-evaluated range
///   predicates (`origin + b·width <= v < origin + (b+1)·width`) applied
///   during the fold. A value outside every bucket lies outside the domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CoordSpec {
    /// A resident ordinal field.
    Field(FieldId),
    /// A derived equal-width bucketing of a signed value field.
    Bucket {
        /// The `I32` field bucketed.
        field: FieldId,
        /// Lower edge of bucket 0.
        origin: i64,
        /// Bucket width (> 0).
        width: i64,
        /// Number of buckets.
        count: u32,
    },
}

impl CoordSpec {
    /// The field this coordinate reads.
    pub fn field(&self) -> FieldId {
        match self {
            CoordSpec::Field(f) | CoordSpec::Bucket { field: f, .. } => *f,
        }
    }
}

impl From<FieldId> for CoordSpec {
    fn from(f: FieldId) -> Self {
        CoordSpec::Field(f)
    }
}

/// An axis: a coordinate provider plus its presented role.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxisSpec {
    /// The coordinate.
    pub coord: CoordSpec,
    /// Its presented role.
    pub role: AxisRole,
}

/// What a measure computes per cell.
///
/// The coordinate/pivot engine never matches on this. It asks a measure for
/// its fold states ([`Measure::folds`]), folds each into the aggregate space,
/// merges them along axes for totals ([`FoldState::merge`]) and hands the
/// merged states back to [`Measure::finalize`]. A new measure is a new arm
/// HERE — `folds` + `finalize` — and nothing in the engine changes (A11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasureKind {
    /// `COUNT(*)`.
    Count,
    /// `SUM(field)` — NULL for an empty cell, as in SQL.
    Sum,
    /// `MIN(field)`.
    Min,
    /// `MAX(field)`.
    Max,
    /// `MEAN(field)` — finalized from the SUM and COUNT states.
    Mean,
}

/// One mergeable accumulator the substrate folds per cell. Every state has a
/// merge law, which is what makes totals, parallel partial folds and cached
/// re-aggregation possible without rescanning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FoldState {
    /// Number of selected rows in the cell. Always present: it defines the
    /// EMPTY cell (count 0) for every other state.
    Count,
    /// Σ of an `I32` field, widened to `i64`.
    Sum(FieldId),
    /// Minimum of an `I32` field.
    Min(FieldId),
    /// Maximum of an `I32` field.
    Max(FieldId),
}

impl FoldState {
    /// The identity element of the merge.
    pub fn identity(&self) -> i64 {
        match self {
            FoldState::Count | FoldState::Sum(_) => 0,
            FoldState::Min(_) => i64::MAX,
            FoldState::Max(_) => i64::MIN,
        }
    }

    /// The merge law (associative, commutative, [`FoldState::identity`] is
    /// its identity).
    pub fn merge(&self, a: i64, b: i64) -> i64 {
        match self {
            FoldState::Count | FoldState::Sum(_) => a.wrapping_add(b),
            FoldState::Min(_) => a.min(b),
            FoldState::Max(_) => a.max(b),
        }
    }
}

/// A finalized cell value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellValue {
    /// An integer.
    Int(i64),
    /// A real.
    Real(f64),
    /// No value: the cell is empty (no selected row) or absent (the
    /// coordinate was never observed). Pinned here, not by a renderer.
    Null,
}

/// A measure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Measure {
    /// What it computes.
    pub kind: MeasureKind,
    /// The `I32` value field (`None` only for `Count`).
    pub field: Option<FieldId>,
}

impl Measure {
    /// `COUNT(*)`.
    pub fn count() -> Self {
        Self {
            kind: MeasureKind::Count,
            field: None,
        }
    }

    /// `kind(field)`.
    pub fn of(kind: MeasureKind, field: FieldId) -> Self {
        Self {
            kind,
            field: Some(field),
        }
    }

    /// The fold's operator name (diagnostics / terminal headers).
    pub fn op_name(&self) -> &'static str {
        match self.kind {
            MeasureKind::Count => "COUNT",
            MeasureKind::Sum => "SUM",
            MeasureKind::Min => "MIN",
            MeasureKind::Max => "MAX",
            MeasureKind::Mean => "MEAN",
        }
    }

    /// The fold states this measure needs (beyond the always-present
    /// [`FoldState::Count`]).
    pub fn folds(&self) -> Vec<FoldState> {
        let f = || self.field.unwrap_or(FieldId(u32::MAX));
        match self.kind {
            MeasureKind::Count => vec![],
            MeasureKind::Sum => vec![FoldState::Sum(f())],
            MeasureKind::Min => vec![FoldState::Min(f())],
            MeasureKind::Max => vec![FoldState::Max(f())],
            MeasureKind::Mean => vec![FoldState::Sum(f())],
        }
    }

    /// Finalize from merged states. `state(s)` reads one merged state;
    /// `count` is the merged COUNT. Empty (`count == 0`) is NULL for every
    /// measure but COUNT itself, whose empty answer is a real 0.
    pub fn finalize(&self, count: i64, state: &dyn Fn(&FoldState) -> i64) -> CellValue {
        if self.kind == MeasureKind::Count {
            return CellValue::Int(count);
        }
        if count == 0 {
            return CellValue::Null;
        }
        let f = &self.folds()[0];
        match self.kind {
            MeasureKind::Mean => CellValue::Real(state(f) as f64 / count as f64),
            _ => CellValue::Int(state(f)),
        }
    }
}

/// Keep the top `k` row-keys by one measure's row total (a result-sized
/// ordering applied to the aggregate, never to the population).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopK {
    /// Index into [`ReportPlan::measures`].
    pub measure: usize,
    /// How many row keys to keep.
    pub k: usize,
    /// Largest first.
    pub descending: bool,
}

/// The source a plan reads: a published batch, by identity and generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRef {
    /// Published identity.
    pub id: SourceId,
    /// The generation the plan was minted against.
    pub generation: u32,
}

/// A report plan. Immutable by convention: rewrites return a new plan, so a
/// plan shared by a DAG fan-out (behind an `Arc`) is never mutated under a
/// sibling branch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReportPlan {
    /// What to read.
    pub source: SourceRef,
    /// Which rows.
    pub selection: Selection,
    /// The coordinate axes, with roles. Order within a role is presentation
    /// order (outermost first).
    pub axes: Vec<AxisSpec>,
    /// What each cell holds.
    pub measures: Vec<Measure>,
    /// Optional result-side ordering.
    pub top_k: Option<TopK>,
}

/// The part of a plan the FOLD depends on. Roles, axis order and top-k are
/// absent on purpose: they are presentation, and changing them must never
/// cause population work.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalKey {
    /// The source.
    pub source: SourceRef,
    /// The selection.
    pub selection: Selection,
    /// The coordinates, sorted (a SET, not a sequence): the canonical
    /// dimension order of the aggregate space.
    pub coords: Vec<CoordSpec>,
    /// The measures, in plan order.
    pub measures: Vec<Measure>,
}

impl ReportPlan {
    /// A plan over every valid row of `source`, no axes, no measures.
    pub fn over(source: SourceRef) -> Self {
        Self {
            source,
            selection: Selection::All,
            axes: Vec::new(),
            measures: Vec::new(),
            top_k: None,
        }
    }

    /// AND a selection into the plan.
    pub fn filter(mut self, s: Selection) -> Self {
        self.selection = std::mem::replace(&mut self.selection, Selection::All).and(s);
        self
    }

    /// Add (or re-role) an axis. An existing axis on the same coordinate
    /// changes role and moves to the end of its new role's order.
    pub fn axis(mut self, coord: impl Into<CoordSpec>, role: AxisRole) -> Self {
        let coord = coord.into();
        self.axes.retain(|a| a.coord != coord);
        self.axes.push(AxisSpec { coord, role });
        self
    }

    /// Remove an axis. This changes the coordinate SET, hence the physical
    /// key: the cells it separated must merge.
    pub fn without_axis(mut self, coord: &CoordSpec) -> Self {
        self.axes.retain(|a| &a.coord != coord);
        self
    }

    /// Replace the whole role assignment in one step (rows, columns, pages,
    /// each outermost first). Every coordinate named must already be an
    /// axis; axes not named keep their role at the end. Pure metadata.
    pub fn view(mut self, rows: &[CoordSpec], columns: &[CoordSpec], pages: &[CoordSpec]) -> Self {
        let mut axes = Vec::with_capacity(self.axes.len());
        for (list, role) in [
            (rows, AxisRole::Row),
            (columns, AxisRole::Column),
            (pages, AxisRole::Page),
        ] {
            for c in list {
                axes.push(AxisSpec {
                    coord: c.clone(),
                    role,
                });
            }
        }
        for a in self.axes.drain(..) {
            if !axes.iter().any(|b| b.coord == a.coord) {
                axes.push(a);
            }
        }
        self.axes = axes;
        self
    }

    /// Add a measure.
    pub fn measure(mut self, m: Measure) -> Self {
        self.measures.push(m);
        self
    }

    /// Set the result ordering.
    pub fn top_k(mut self, t: TopK) -> Self {
        self.top_k = Some(t);
        self
    }

    /// Pivot: `rows` become the row axes and `columns` the column axes, in
    /// the given order. Any other axis keeps its role. Pure metadata.
    pub fn pivot(mut self, rows: &[CoordSpec], columns: &[CoordSpec]) -> Self {
        for r in rows {
            self = self.axis(r.clone(), AxisRole::Row);
        }
        for c in columns {
            self = self.axis(c.clone(), AxisRole::Column);
        }
        self
    }

    /// Rotate: every row axis becomes a column axis and vice versa. Pure
    /// metadata; the physical key is unchanged.
    pub fn rotate(mut self) -> Self {
        for a in &mut self.axes {
            a.role = match a.role {
                AxisRole::Row => AxisRole::Column,
                AxisRole::Column => AxisRole::Row,
                AxisRole::Page => AxisRole::Page,
            };
        }
        self
    }

    /// Coordinates playing `role`, in presentation order.
    pub fn axes_in(&self, role: AxisRole) -> Vec<&CoordSpec> {
        self.axes
            .iter()
            .filter(|a| a.role == role)
            .map(|a| &a.coord)
            .collect()
    }

    /// The fold-relevant part of the plan.
    pub fn physical_key(&self) -> PhysicalKey {
        let mut coords: Vec<CoordSpec> = self.axes.iter().map(|a| a.coord.clone()).collect();
        coords.sort();
        PhysicalKey {
            source: self.source,
            selection: self.selection.clone(),
            coords,
            measures: self.measures.clone(),
        }
    }
}
