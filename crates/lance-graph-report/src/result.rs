//! The aggregate coordinate space and its views.
//!
//! **Never transpose data to pivot. Transpose meaning.**
//!
//! A [`CellSpace`] holds fold states against CANONICAL coordinate identities —
//! a tuple over the plan's coordinate SET in its canonical (sorted) order.
//! Where a cell physically lives (a dense offset, or a slot in a sparse
//! column set) is private to the space and is never a presentation position.
//!
//! A [`ReportResult`] is `Arc<CellSpace>` + a [`View`]: which canonical
//! dimensions play rows, columns and pages, plus an optional result-sized
//! presentation order. Rotating, re-rolling, re-ordering or sorting builds a
//! new `View` over the SAME `Arc` — the cell payload is never moved, copied
//! or recomputed (falsifiers F6 / F7 / F14, A3 / A4 / A5 / A8).
//!
//! Semantics pinned here, for every adapter alike:
//!
//! * a cell whose COUNT is 0 is **empty**: COUNT reads 0, every other measure
//!   reads [`CellValue::Null`];
//! * a coordinate never observed in a sparse space is **absent**: every
//!   measure, COUNT included, reads `Null` — nothing was folded there;
//! * a row whose coordinate lies outside a dimension's domain (a code past
//!   the domain, a value outside every bucket) belongs to no cell, so it is
//!   in no total either: totals are totals OVER THE COORDINATE SPACE;
//! * a filtered-out row contributes nothing anywhere.

use std::collections::HashMap;
use std::sync::Arc;

use crate::plan::{AxisRole, CellValue, CoordSpec, FoldState, Measure, PhysicalKey, ReportPlan};
use crate::ReportError;

/// One canonical dimension of the space.
#[derive(Debug, Clone)]
pub struct DimMeta {
    /// The coordinate provider.
    pub coord: CoordSpec,
    /// Domain size (ordinals `0..domain`). Labels are NOT here: they are
    /// resolved from the CAM label store at the terminal, for the members a
    /// view actually presents.
    pub domain: u32,
}

/// How cells are stored. Private to the space: no API exposes which one.
#[derive(Debug)]
pub(crate) enum Layout {
    /// Every coordinate of the (bounded) product has a slot; `offset = Σ
    /// coord[d] · strides[d]` in canonical dimension order.
    Dense { strides: Vec<usize> },
    /// Only observed coordinates have a slot: per-dimension coordinate
    /// columns (SoA) plus an index from coordinate tuple to slot.
    Sparse {
        coords: Vec<Box<[u32]>>,
        index: HashMap<Box<[u32]>, u32>,
    },
}

/// The computed aggregate space.
#[derive(Debug)]
pub struct CellSpace {
    pub(crate) key: PhysicalKey,
    pub(crate) dims: Vec<DimMeta>,
    pub(crate) layout: Layout,
    /// Fold states; `states[0]` is always [`FoldState::Count`].
    pub(crate) states: Vec<FoldState>,
    /// One value column per state, one slot per cell.
    pub(crate) values: Vec<Box<[i64]>>,
}

/// Whether a coordinate holds a folded cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Rows were folded here.
    Present,
    /// The coordinate exists but no selected row landed on it.
    Empty,
    /// The coordinate was never observed (sparse spaces only).
    Absent,
}

impl CellSpace {
    /// The physical key this space answers.
    pub fn key(&self) -> &PhysicalKey {
        &self.key
    }

    /// The canonical dimensions.
    pub fn dims(&self) -> &[DimMeta] {
        &self.dims
    }

    /// Number of stored cells (dense: the product; sparse: observed).
    pub fn stored_cells(&self) -> usize {
        self.values[0].len()
    }

    /// Whether storage is sparse (diagnostic only — semantics never differ).
    pub fn is_sparse(&self) -> bool {
        matches!(self.layout, Layout::Sparse { .. })
    }

    /// Address of the COUNT column's first slot — the identity the
    /// zero-copy falsifiers compare across views.
    pub fn payload_addr(&self) -> usize {
        self.values[0].as_ptr() as usize
    }

    /// Bytes held by fold-state columns (the aggregate allocation).
    pub fn accumulator_bytes(&self) -> usize {
        self.values.iter().map(|v| v.len() * 8).sum()
    }

    fn slot(&self, coord: &[u32]) -> Option<usize> {
        match &self.layout {
            Layout::Dense { strides } => {
                let mut off = 0;
                for (d, &c) in coord.iter().enumerate() {
                    if c >= self.dims[d].domain {
                        return None;
                    }
                    off += c as usize * strides[d];
                }
                Some(off)
            }
            Layout::Sparse { index, .. } => index.get(coord).map(|&s| s as usize),
        }
    }

    /// The state of the cell at a canonical coordinate.
    pub fn state(&self, coord: &[u32]) -> CellState {
        match self.slot(coord) {
            None => CellState::Absent,
            Some(s) if self.values[0][s] == 0 => CellState::Empty,
            Some(_) => CellState::Present,
        }
    }

    fn state_index(&self, s: &FoldState) -> usize {
        self.states
            .iter()
            .position(|x| x == s)
            .expect("measure state folded")
    }

    /// Merge every stored cell matching `fixed` (a partial coordinate:
    /// `Some(member)` pins a dimension) and finalize `measure`. O(stored
    /// cells) — result-sized, never population-sized. `None` on no match
    /// (every matching slot absent).
    pub fn merged(&self, measure: &Measure, fixed: &[Option<u32>]) -> CellValue {
        let mut acc: Vec<i64> = self.states.iter().map(FoldState::identity).collect();
        let mut any = false;
        let mut visit = |slot: usize| {
            any = true;
            for (i, st) in self.states.iter().enumerate() {
                acc[i] = st.merge(acc[i], self.values[i][slot]);
            }
        };
        match &self.layout {
            Layout::Dense { strides } => {
                // Enumerate the sub-product the fixed coordinates leave free.
                let free: Vec<usize> = (0..self.dims.len())
                    .filter(|&d| fixed[d].is_none())
                    .collect();
                let base: usize = (0..self.dims.len())
                    .filter_map(|d| fixed[d].map(|m| m as usize * strides[d]))
                    .sum();
                if fixed
                    .iter()
                    .enumerate()
                    .any(|(d, f)| f.is_some_and(|m| m >= self.dims[d].domain))
                {
                    return CellValue::Null;
                }
                let mut ctr = vec![0u32; free.len()];
                'outer: loop {
                    let off = base
                        + free
                            .iter()
                            .zip(&ctr)
                            .map(|(&d, &c)| c as usize * strides[d])
                            .sum::<usize>();
                    visit(off);
                    for i in (0..free.len()).rev() {
                        ctr[i] += 1;
                        if ctr[i] < self.dims[free[i]].domain {
                            continue 'outer;
                        }
                        ctr[i] = 0;
                    }
                    break;
                }
            }
            Layout::Sparse { coords, .. } => {
                #[allow(clippy::needless_range_loop)] // `slot` indexes N coordinate columns at once
                for slot in 0..self.stored_cells() {
                    if fixed
                        .iter()
                        .enumerate()
                        .all(|(d, f)| f.is_none_or(|m| coords[d][slot] == m))
                    {
                        visit(slot);
                    }
                }
            }
        }
        if !any {
            return CellValue::Null;
        }
        let count = acc[0];
        measure.finalize(count, &|s| acc[self.state_index(s)])
    }
}

/// Which canonical dimensions play which role, and in what order — pure
/// presentation metadata over a [`CellSpace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    /// Canonical dimension indices down the side, outermost first.
    pub rows: Vec<usize>,
    /// Across the top.
    pub columns: Vec<usize>,
    /// One page per member tuple.
    pub pages: Vec<usize>,
    /// Presentation order of the row keys (indices into
    /// [`ReportResult::row_keys`]); `None` = canonical order. Result-sized.
    pub row_order: Option<Arc<[u32]>>,
}

/// A computed report: a shared aggregate space seen through a view.
#[derive(Debug, Clone)]
pub struct ReportResult {
    pub(crate) space: Arc<CellSpace>,
    pub(crate) view: View,
}

/// The shape a result presents as (for adapters that care).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportShape {
    /// No coordinates.
    Scalar,
    /// One coordinate.
    Series,
    /// Two or more coordinates.
    Matrix,
}

impl ReportResult {
    pub(crate) fn new(space: Arc<CellSpace>, plan: &ReportPlan) -> Result<Self, ReportError> {
        let r = ReportResult {
            view: View {
                rows: vec![],
                columns: vec![],
                pages: vec![],
                row_order: None,
            },
            space,
        };
        r.reinterpret(plan)
    }

    /// The shared aggregate space.
    pub fn space(&self) -> &Arc<CellSpace> {
        &self.space
    }

    /// The current view.
    pub fn view(&self) -> &View {
        &self.view
    }

    /// Presented shape.
    pub fn shape(&self) -> ReportShape {
        match self.space.dims.len() {
            0 => ReportShape::Scalar,
            1 => ReportShape::Series,
            _ => ReportShape::Matrix,
        }
    }

    /// The measures, in plan order.
    pub fn measures(&self) -> &[Measure] {
        &self.space.key.measures
    }

    /// View this space through another plan's roles, order and top-k. The
    /// plans must share a physical key — otherwise the fold differs and the
    /// caller must execute. No population work, no cell copy.
    pub fn reinterpret(&self, plan: &ReportPlan) -> Result<Self, ReportError> {
        if plan.physical_key() != self.space.key {
            return Err(ReportError::PhysicalKeyMismatch);
        }
        let idx = |c: &CoordSpec| {
            self.space
                .dims
                .iter()
                .position(|d| &d.coord == c)
                .expect("coord in space")
        };
        let pick = |role| plan.axes_in(role).into_iter().map(idx).collect::<Vec<_>>();
        let mut out = ReportResult {
            space: Arc::clone(&self.space),
            view: View {
                rows: pick(AxisRole::Row),
                columns: pick(AxisRole::Column),
                pages: pick(AxisRole::Page),
                row_order: None,
            },
        };
        if let Some(t) = plan.top_k {
            let m = self
                .measures()
                .get(t.measure)
                .ok_or(ReportError::BadTopK)?
                .clone();
            out = out.order_rows_by(&m, t.descending, Some(t.k));
        }
        Ok(out)
    }

    /// Swap rows and columns. Metadata only.
    pub fn rotate(&self) -> Self {
        let mut v = self.view.clone();
        std::mem::swap(&mut v.rows, &mut v.columns);
        v.row_order = None;
        ReportResult {
            space: Arc::clone(&self.space),
            view: v,
        }
    }

    /// Re-role in one step, by canonical coordinate. Metadata only.
    pub fn with_roles(
        &self,
        rows: &[CoordSpec],
        columns: &[CoordSpec],
        pages: &[CoordSpec],
    ) -> Result<Self, ReportError> {
        let idx = |c: &CoordSpec| {
            self.space
                .dims
                .iter()
                .position(|d| &d.coord == c)
                .ok_or(ReportError::PhysicalKeyMismatch)
        };
        let conv = |l: &[CoordSpec]| l.iter().map(idx).collect::<Result<Vec<_>, _>>();
        Ok(ReportResult {
            space: Arc::clone(&self.space),
            view: View {
                rows: conv(rows)?,
                columns: conv(columns)?,
                pages: conv(pages)?,
                row_order: None,
            },
        })
    }

    /// Order row keys by the merged value of `measure` over each row (a
    /// presentation permutation; identity untouched), optionally keeping `k`.
    pub fn order_rows_by(&self, measure: &Measure, descending: bool, k: Option<usize>) -> Self {
        let rows = self.row_keys();
        let key = |v: CellValue| match v {
            CellValue::Int(i) => i as f64,
            CellValue::Real(r) => r,
            CellValue::Null => f64::NEG_INFINITY,
        };
        let mut order: Vec<u32> = (0..rows.len() as u32).collect();
        let vals: Vec<f64> = rows
            .iter()
            .map(|r| key(self.row_total(measure, r)))
            .collect();
        order.sort_by(|&a, &b| {
            let (x, y) = (vals[a as usize], vals[b as usize]);
            let o = x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal);
            if descending { o.reverse() } else { o }.then(a.cmp(&b))
        });
        if let Some(k) = k {
            order.truncate(k);
        }
        let mut v = self.view.clone();
        v.row_order = Some(order.into());
        ReportResult {
            space: Arc::clone(&self.space),
            view: v,
        }
    }

    /// Member tuples along `dims` that the view enumerates: the full product
    /// for a dense space, the observed projections for a sparse one.
    fn keys(&self, dims: &[usize]) -> Vec<Vec<u32>> {
        if dims.is_empty() {
            return vec![vec![]];
        }
        match &self.space.layout {
            Layout::Dense { .. } => {
                let mut out = vec![vec![]];
                for &d in dims {
                    let n = self.space.dims[d].domain;
                    out = out
                        .into_iter()
                        .flat_map(|p: Vec<u32>| (0..n).map(move |m| [p.clone(), vec![m]].concat()))
                        .collect();
                }
                out
            }
            Layout::Sparse { coords, .. } => {
                let mut set: Vec<Vec<u32>> = (0..self.space.stored_cells())
                    .map(|s| dims.iter().map(|&d| coords[d][s]).collect())
                    .collect();
                set.sort();
                set.dedup();
                set
            }
        }
    }

    /// Row keys in presentation order.
    pub fn row_keys(&self) -> Vec<Vec<u32>> {
        let all = self.keys(&self.view.rows);
        match &self.view.row_order {
            None => all,
            Some(o) => o.iter().map(|&i| all[i as usize].clone()).collect(),
        }
    }

    /// Column keys.
    pub fn column_keys(&self) -> Vec<Vec<u32>> {
        self.keys(&self.view.columns)
    }

    /// Page keys.
    pub fn page_keys(&self) -> Vec<Vec<u32>> {
        self.keys(&self.view.pages)
    }

    fn fixed(&self, page: &[u32], row: &[u32], col: &[u32]) -> Vec<Option<u32>> {
        let mut f = vec![None; self.space.dims.len()];
        for (dims, key) in [
            (&self.view.pages, page),
            (&self.view.rows, row),
            (&self.view.columns, col),
        ] {
            for (&d, &m) in dims.iter().zip(key) {
                f[d] = Some(m);
            }
        }
        f
    }

    /// One presented cell. Dimensions in no role are merged over (they are
    /// in the space but not shown).
    pub fn value(&self, measure: &Measure, page: &[u32], row: &[u32], col: &[u32]) -> CellValue {
        self.space.merged(measure, &self.fixed(page, row, col))
    }

    /// Total over every column (and hidden dimension) for one row, across
    /// all pages.
    pub fn row_total(&self, measure: &Measure, row: &[u32]) -> CellValue {
        self.space.merged(measure, &self.fixed(&[], row, &[]))
    }

    /// Total over every row for one column, across all pages.
    pub fn column_total(&self, measure: &Measure, col: &[u32]) -> CellValue {
        self.space.merged(measure, &self.fixed(&[], &[], col))
    }

    /// Grand total.
    pub fn grand_total(&self, measure: &Measure) -> CellValue {
        self.space
            .merged(measure, &vec![None; self.space.dims.len()])
    }

    /// `value / grand_total` for a cell (derived from aggregated cells, no
    /// rescan). `None` when either side is NULL or the total is 0.
    pub fn share_of_total(
        &self,
        measure: &Measure,
        page: &[u32],
        row: &[u32],
        col: &[u32],
    ) -> Option<f64> {
        ratio(
            self.value(measure, page, row, col),
            self.grand_total(measure),
        )
    }

    /// `value / row_total`.
    pub fn share_of_row(&self, measure: &Measure, row: &[u32], col: &[u32]) -> Option<f64> {
        ratio(
            self.value(measure, &[], row, col),
            self.row_total(measure, row),
        )
    }

    /// `value / column_total`.
    pub fn share_of_column(&self, measure: &Measure, row: &[u32], col: &[u32]) -> Option<f64> {
        ratio(
            self.value(measure, &[], row, col),
            self.column_total(measure, col),
        )
    }
}

fn ratio(a: CellValue, b: CellValue) -> Option<f64> {
    let f = |v| match v {
        CellValue::Int(i) => Some(i as f64),
        CellValue::Real(r) => Some(r),
        CellValue::Null => None,
    };
    let (a, b) = (f(a)?, f(b)?);
    (b != 0.0).then(|| a / b)
}
