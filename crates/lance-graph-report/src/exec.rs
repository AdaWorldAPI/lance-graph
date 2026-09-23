//! Physical planning and the fold.
//!
//! ```text
//!   ReportPlan (control plane, ids only)
//!        │ plan_physical        — structural; reads no row
//!        ▼
//!   PhysicalPlan  (selection carrier · coordinate providers · fold key ·
//!        │         accumulator strategy · pass count · lowered programs)
//!        │ run                  — the only population work
//!        ▼
//!   CellSpace (Arc)  ── View ──▶ ReportResult ──▶ terminal adapter
//! ```
//!
//! **Pivot defines coordinates. Fold populates coordinates. Materialization
//! renders coordinates.** Nothing in this file knows what a field means; it
//! knows lanes, ordinals, domains, selections and fold states.
//!
//! Every program executed here is a Quack lowering ([`lance_graph_quack::lower`])
//! run by mask-risc's one evaluator — this crate has no aggregate semantics
//! of its own (F10) and no row loop (F2): its loops run over PASSES and over
//! result cells, never over rows.
//!
//! # The physical choices, and what drives each
//!
//! * **Fold key** — one ordinal coordinate the substrate scatters into
//!   directly (`GroupSumI32` / `GroupReduce`). Chosen as the widest
//!   field-backed coordinate whose domain fits the domain-buffer budget.
//! * **Partitions** — every other coordinate. A partition member is a
//!   selection conjunct (`field = m`, or a derived bucket's two range
//!   compares) evaluated tile by tile during the pass; nothing is
//!   materialized for it (A6). One pass per partition-member tuple.
//! * **Accumulator** — dense when the coordinate product fits the dense-cell
//!   budget; otherwise sparse, with partition members DISCOVERED first by
//!   one domain-sized count pass each, so the product is never allocated (A7).
//! * **Selection carrier** — the validity plane (no ops), an execution
//!   extent (scalar folds over a top-level range: zero mask words, F4),
//!   tile-local mask ops (the default), or ONE reused population mask when
//!   the pass × state count makes re-evaluating the predicate the costlier
//!   side. The reused mask is the one population-sized allocation this file
//!   can make, it is counted, and it is named in the explain output.
//!
//! The honest limit: a partition costs one pass per member tuple, so a plan
//! whose partition side is high-cardinality AND densely observed exceeds the
//! pass budget and is REFUSED with [`ReportError::PassBudget`] rather than
//! run slowly or allocated densely. The primitive that would lift it — a
//! composite-key (multi-lane) group fold in `ndarray::simd` / mask-risc — is
//! a named substrate gap, not something to hand-roll here.

use std::collections::HashMap;
use std::sync::Arc;

use lance_graph_mask_risc::{
    execute_extent, scratch_words_for, tile_words_for, words_for, Foreign, Out, Planes, Program,
    Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, GroupAddr, GroupAgg, Mask, Query};

use crate::batch::{AbiBatch, LaneData};
use crate::plan::{CoordSpec, FoldState, PhysicalKey, ReportPlan, SourceRef};
use crate::result::{CellSpace, DimMeta, Layout, ReportResult};
use crate::selection::RowRange;
use crate::ReportError;

/// The knobs the physical planner reads. Every one changes a decision (each
/// has an inertness test in `tests/`), none changes a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerPolicy {
    /// Largest coordinate product stored densely.
    pub dense_cell_budget: u64,
    /// Largest single-coordinate domain a fold-key / discovery buffer may span.
    pub domain_buffer_budget: u32,
    /// Most passes over the population one report may make.
    pub pass_budget: u64,
    /// Reuse ONE materialized selection mask once the plan would otherwise
    /// evaluate the selection predicate in at least this many programs.
    pub reuse_mask_min_programs: u64,
}

impl Default for PlannerPolicy {
    fn default() -> Self {
        Self {
            dense_cell_budget: 1 << 20,
            domain_buffer_budget: 1 << 22,
            pass_budget: 1 << 16,
            reuse_mask_min_programs: 3,
        }
    }
}

/// How a coordinate places a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// A resident ordinal lane.
    OrdinalLane {
        /// Lane index in the batch.
        lane: u16,
    },
    /// A derived bucketing of a signed lane; each member is a pair of range
    /// compares, never a materialized lane.
    DerivedBucket {
        /// Lane index of the bucketed field.
        lane: u16,
    },
}

/// One canonical dimension as planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimPlan {
    /// The coordinate.
    pub coord: CoordSpec,
    /// Domain size.
    pub domain: u32,
    /// How rows are placed on it.
    pub provider: Provider,
}

/// The carrier the planner chose for the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionCarrier {
    /// Every valid row: the validity plane, read in place, zero ops.
    ValidityPlane,
    /// Mask ops evaluated tile by tile (at most `TILE_WORDS` words per slot).
    TileLocal,
    /// One population mask, materialized once and read as a resident plane
    /// by every later program.
    ReusedMask,
}

/// The accumulator strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accumulator {
    /// One cell.
    Scalar,
    /// Every coordinate of the product has a slot.
    Dense {
        /// Slots allocated.
        cells: u64,
    },
    /// Only observed coordinates get a slot; the product is never allocated.
    Sparse {
        /// The product of domains (for the explain output only).
        product: u128,
    },
}

/// The explainable physical plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPlan {
    /// The source.
    pub source: SourceRef,
    /// Population size.
    pub n_rows: usize,
    /// Selection carrier.
    pub selection: SelectionCarrier,
    /// Execution extent (scalar folds over a top-level range only).
    pub extent: Option<RowRange>,
    /// Canonical dimensions.
    pub dims: Vec<DimPlan>,
    /// Canonical index of the fold-key dimension, if any.
    pub fold_key: Option<usize>,
    /// Canonical indices of the partition dimensions, in pass order.
    pub partitions: Vec<usize>,
    /// Fold states (index 0 is COUNT).
    pub states: Vec<FoldState>,
    /// Accumulator strategy.
    pub accumulator: Accumulator,
    /// Passes over the population (dense: exact; sparse: an upper bound
    /// before discovery).
    pub passes: u64,
    /// The programs of the FIRST pass, as lowered — the physical identity
    /// two front-ends must agree on (plan-equivalence tests compare these).
    pub first_pass: Vec<Program>,
}

/// What a run actually did — the counters that catch hidden materialization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecStats {
    /// Programs executed over the population (each is one scan).
    pub population_scans: u64,
    /// Of those, discovery scans (sparse accumulators only).
    pub discovery_scans: u64,
    /// Mask ops executed (each writes tile-local scratch, never a population).
    pub tile_mask_ops: u64,
    /// Population-sized masks materialized (0 unless [`SelectionCarrier::ReusedMask`]).
    pub mask_materializations: u64,
    /// Bytes of those masks.
    pub mask_bytes: u64,
    /// Largest tile scratch any one program needed.
    pub scratch_bytes_peak: u64,
    /// Fold-state bytes allocated for the result.
    pub accumulator_bytes: u64,
    /// Transient domain-sized buffer bytes (fold key / discovery), peak.
    pub domain_buffer_bytes: u64,
    /// Cells stored in the result.
    pub result_cells: u64,
}

struct Resolved {
    dims: Vec<DimPlan>,
    fold_key: Option<usize>,
    partitions: Vec<usize>,
    states: Vec<FoldState>,
    state_lanes: Vec<Option<u16>>,
}

fn member_filter(d: &DimPlan, m: u32) -> Filter {
    match (&d.provider, &d.coord) {
        (Provider::OrdinalLane { lane }, _) => Filter::Cmp(Col(*lane), Cmp::EqU32(m)),
        (Provider::DerivedBucket { lane }, CoordSpec::Bucket { origin, width, .. }) => {
            let lo = origin + width * i64::from(m);
            let hi = lo + width;
            let mut parts = Vec::with_capacity(2);
            if lo > i64::from(i32::MIN) {
                parts.push(Filter::Cmp(
                    Col(*lane),
                    Cmp::GeI32(lo.min(i64::from(i32::MAX)) as i32),
                ));
            }
            if hi <= i64::from(i32::MAX) {
                parts.push(Filter::Cmp(
                    Col(*lane),
                    Cmp::LtI32(hi.max(i64::from(i32::MIN)) as i32),
                ));
            }
            if lo > i64::from(i32::MAX) {
                // Wholly above the lane's range: the member is empty.
                parts = vec![Filter::Cmp(Col(*lane), Cmp::GtI32(i32::MAX))];
            }
            match parts.len() {
                0 => Filter::Plane(Mask(0)),
                1 => parts.pop().expect("one part"),
                _ => Filter::And(parts),
            }
        }
        (Provider::DerivedBucket { .. }, CoordSpec::Field(_)) => {
            unreachable!("bucket provider on a field")
        }
    }
}

fn resolve(
    key: &PhysicalKey,
    batch: &AbiBatch,
    policy: &PlannerPolicy,
) -> Result<Resolved, ReportError> {
    let mut dims = Vec::with_capacity(key.coords.len());
    for c in &key.coords {
        let f = c.field();
        let (lane, col) = batch.column(f).ok_or(ReportError::UnknownField(f))?;
        let dim = match c {
            CoordSpec::Field(_) => DimPlan {
                coord: c.clone(),
                domain: col.domain.ok_or(ReportError::NotACoordinate(f))?,
                provider: Provider::OrdinalLane { lane },
            },
            CoordSpec::Bucket { width, count, .. } => {
                if !matches!(col.lane, LaneData::I32(_)) || *width <= 0 {
                    return Err(ReportError::BadBucket(f));
                }
                DimPlan {
                    coord: c.clone(),
                    domain: *count,
                    provider: Provider::DerivedBucket { lane },
                }
            }
        };
        dims.push(dim);
    }
    let fold_key = dims
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            matches!(d.provider, Provider::OrdinalLane { .. })
                && d.domain <= policy.domain_buffer_budget
        })
        .max_by_key(|(i, d)| (d.domain, std::cmp::Reverse(*i)))
        .map(|(i, _)| i);
    let partitions = (0..dims.len()).filter(|&i| Some(i) != fold_key).collect();

    let mut states = vec![FoldState::Count];
    for m in &key.measures {
        for s in m.folds() {
            if !states.contains(&s) {
                states.push(s);
            }
        }
    }
    let state_lanes = states
        .iter()
        .map(|s| match s {
            FoldState::Count => Ok(None),
            FoldState::Sum(f) | FoldState::Min(f) | FoldState::Max(f) => {
                let (lane, col) = batch.column(*f).ok_or(ReportError::UnknownField(*f))?;
                if !matches!(col.lane, LaneData::I32(_)) {
                    return Err(ReportError::NotAMeasure(*f));
                }
                Ok(Some(lane))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Resolved {
        dims,
        fold_key,
        partitions,
        states,
        state_lanes,
    })
}

/// The aggregate a fold state lowers to, keyed or scalar.
fn agg_for(state: &FoldState, lane: Option<u16>, key: Option<u16>) -> Agg {
    let v = lane.map(Col);
    match (key.map(Col), state) {
        (Some(k), FoldState::Count) => Agg::GroupReduce {
            key: GroupAddr::Local(k),
            agg: GroupAgg::Count,
        },
        (Some(k), FoldState::Sum(_)) => Agg::GroupSumI32 {
            key: k,
            val: v.expect("lane"),
        },
        (Some(k), FoldState::Min(_)) => Agg::GroupReduce {
            key: GroupAddr::Local(k),
            agg: GroupAgg::MinI32(v.expect("lane")),
        },
        (Some(k), FoldState::Max(_)) => Agg::GroupReduce {
            key: GroupAddr::Local(k),
            agg: GroupAgg::MaxI32(v.expect("lane")),
        },
        (None, FoldState::Count) => Agg::Count,
        (None, FoldState::Sum(_)) => Agg::SumI32(v.expect("lane")),
        (None, FoldState::Min(_)) => Agg::MinI32(v.expect("lane")),
        (None, FoldState::Max(_)) => Agg::MaxI32(v.expect("lane")),
    }
}

fn scalar_of(state: &FoldState, v: Value) -> i64 {
    match v {
        Value::Count(n) => n as i64,
        Value::SumI64(s) => s,
        Value::OptI32(Some(x)) => i64::from(x),
        _ => state.identity(),
    }
}

/// `base AND member₁ AND …` — the base (selection or plane) leads, so it
/// gates every member compare under the survivor skip.
fn conj(base: &Filter, members: &[Filter]) -> Filter {
    if members.is_empty() {
        return base.clone();
    }
    Filter::And(
        std::iter::once(base.clone())
            .chain(members.iter().cloned())
            .collect(),
    )
}

/// Mixed-radix iteration over member tuples.
fn for_each_tuple(
    radices: &[Vec<u32>],
    mut f: impl FnMut(&[u32]) -> Result<(), ReportError>,
) -> Result<(), ReportError> {
    if radices.iter().any(Vec::is_empty) {
        return Ok(());
    }
    let mut ctr = vec![0usize; radices.len()];
    let mut tuple: Vec<u32> = radices.iter().map(|r| r[0]).collect();
    loop {
        f(&tuple)?;
        let mut i = radices.len();
        loop {
            if i == 0 {
                return Ok(());
            }
            i -= 1;
            ctr[i] += 1;
            if ctr[i] < radices[i].len() {
                tuple[i] = radices[i][ctr[i]];
                break;
            }
            ctr[i] = 0;
            tuple[i] = radices[i][0];
        }
    }
}

impl ReportPlan {
    fn check_source(&self, batch: &AbiBatch) -> Result<(), ReportError> {
        if self.source.id != batch.source() || self.source.generation != batch.generation() {
            return Err(ReportError::StaleSource);
        }
        Ok(())
    }

    /// Plan physically, without touching a row.
    pub fn explain(
        &self,
        batch: &AbiBatch,
        policy: &PlannerPolicy,
    ) -> Result<PhysicalPlan, ReportError> {
        self.check_source(batch)?;
        let key = self.physical_key();
        let r = resolve(&key, batch, policy)?;
        let product: u128 = r.dims.iter().map(|d| u128::from(d.domain)).product();
        let accumulator = if r.dims.is_empty() {
            Accumulator::Scalar
        } else if product <= u128::from(policy.dense_cell_budget) {
            Accumulator::Dense {
                cells: product as u64,
            }
        } else {
            Accumulator::Sparse { product }
        };
        let passes: u128 = r
            .partitions
            .iter()
            .map(|&i| u128::from(r.dims[i].domain))
            .product();
        let passes = passes.min(u128::from(u64::MAX)) as u64;

        // Scalar folds take a top-level range as an execution extent.
        let (extent, base_sel) = if r.dims.is_empty() {
            self.selection.split_range()
        } else {
            (None, Some(self.selection.clone()))
        };
        if let Some(e) = extent {
            if e.hi as usize > batch.n_rows() || e.lo > e.hi {
                return Err(ReportError::RangeOutOfBounds {
                    lo: e.lo,
                    hi: e.hi,
                    n_rows: batch.n_rows(),
                });
            }
        }
        let base = match &base_sel {
            None => Filter::Plane(Mask(0)),
            Some(s) => s.lower(batch)?,
        };
        let sel_ops = lower(&Query {
            filter: base.clone(),
            agg: Agg::Count,
        })?
        .ops
        .len();
        let programs = passes.saturating_mul(r.states.len() as u64);
        let selection = if sel_ops == 0 {
            SelectionCarrier::ValidityPlane
        } else if programs >= policy.reuse_mask_min_programs {
            SelectionCarrier::ReusedMask
        } else {
            SelectionCarrier::TileLocal
        };
        let first_members: Vec<Filter> = r
            .partitions
            .iter()
            .map(|&i| member_filter(&r.dims[i], 0))
            .collect();
        let fbase = if selection == SelectionCarrier::ReusedMask {
            Filter::Plane(Mask(u16::MAX))
        } else {
            base
        };
        let key_lane = r.fold_key.map(|i| match r.dims[i].provider {
            Provider::OrdinalLane { lane } => lane,
            Provider::DerivedBucket { lane } => lane,
        });
        let first_pass = r
            .states
            .iter()
            .zip(&r.state_lanes)
            .map(|(s, &l)| {
                lower(&Query {
                    filter: conj(&fbase, &first_members),
                    agg: agg_for(s, l, key_lane),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PhysicalPlan {
            source: self.source,
            n_rows: batch.n_rows(),
            selection,
            extent,
            dims: r.dims,
            fold_key: r.fold_key,
            partitions: r.partitions,
            states: r.states,
            accumulator,
            passes,
            first_pass,
        })
    }

    /// Plan and fold. The only population work in the crate.
    pub fn execute(
        &self,
        batch: &AbiBatch,
        policy: &PlannerPolicy,
    ) -> Result<(ReportResult, ExecStats), ReportError> {
        let pp = self.explain(batch, policy)?;
        let key = self.physical_key();
        let r = resolve(&key, batch, policy)?;
        let mut stats = ExecStats::default();
        let n = batch.n_rows();
        let extent = pp.extent.map_or(0..n, |e| e.lo as usize..e.hi as usize);

        let (mut masks, lanes) = batch.views();
        let base_sel = if pp.dims.is_empty() {
            self.selection.split_range().1
        } else {
            Some(self.selection.clone())
        };
        let mut base = match &base_sel {
            None => Filter::Plane(Mask(0)),
            Some(s) => s.lower(batch)?,
        };

        // ── selection carrier ────────────────────────────────────────────
        let reused: Vec<u64>;
        if pp.selection == SelectionCarrier::ReusedMask {
            let mut buf = vec![0u64; words_for(n)];
            let prog = lower(&Query {
                filter: base.clone(),
                agg: Agg::Rows,
            })?;
            {
                let planes = Planes {
                    n_rows: n,
                    masks: &masks,
                    lanes: &lanes,
                };
                let mut scratch = Scratch::for_program(&prog, n)?;
                execute_extent(
                    &prog,
                    &planes,
                    &Foreign::NONE,
                    &mut scratch,
                    Out::Mask(&mut buf),
                    extent.clone(),
                )?;
            }
            stats.population_scans += 1;
            stats.tile_mask_ops += prog.ops.len() as u64;
            stats.mask_materializations += 1;
            stats.mask_bytes += (buf.len() * 8) as u64;
            reused = buf;
            let idx = u16::try_from(masks.len()).map_err(|_| ReportError::TooManyPlanes)?;
            masks.push(&reused);
            base = Filter::Plane(Mask(idx));
        }
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };

        let key_lane = r.fold_key.map(|i| match r.dims[i].provider {
            Provider::OrdinalLane { lane } | Provider::DerivedBucket { lane } => lane,
        });
        let key_domain = r.fold_key.map_or(1, |i| r.dims[i].domain as usize);

        // One program run: lowers, sizes branch-private scratch, executes.
        let run = |filter: Filter,
                   s: usize,
                   out: Option<&mut [i64]>,
                   stats: &mut ExecStats|
         -> Result<Value, ReportError> {
            let prog = lower(&Query {
                filter,
                agg: agg_for(&r.states[s], r.state_lanes[s], key_lane),
            })?;
            let mut scratch = Scratch::for_program(&prog, n)?;
            if prog.requires_scratch() {
                let b = scratch_words_for(tile_words_for(n), prog.scratch_slots as usize)
                    .unwrap_or(0) as u64
                    * 8;
                stats.scratch_bytes_peak = stats.scratch_bytes_peak.max(b);
            }
            stats.population_scans += 1;
            stats.tile_mask_ops += prog.ops.len() as u64;
            let out = match out {
                Some(o) => Out::I64(o),
                None => Out::None,
            };
            Ok(execute_extent(
                &prog,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                out,
                extent.clone(),
            )?)
        };

        let dims_meta: Vec<DimMeta> = pp
            .dims
            .iter()
            .map(|d| DimMeta {
                coord: d.coord.clone(),
                domain: d.domain,
            })
            .collect();
        let nstates = r.states.len();

        let space = match pp.accumulator {
            Accumulator::Scalar => {
                let mut values: Vec<Box<[i64]>> = Vec::with_capacity(nstates);
                for s in 0..nstates {
                    let v = run(base.clone(), s, None, &mut stats)?;
                    values.push(vec![scalar_of(&r.states[s], v)].into_boxed_slice());
                }
                CellSpace {
                    key,
                    dims: dims_meta,
                    layout: Layout::Dense { strides: vec![] },
                    states: r.states.clone(),
                    values,
                }
            }
            Accumulator::Dense { cells } => {
                if pp.passes > policy.pass_budget {
                    return Err(ReportError::PassBudget {
                        passes: pp.passes,
                        budget: policy.pass_budget,
                    });
                }
                // Storage order: partitions (pass order), then the fold key
                // innermost so each pass writes one contiguous run.
                let mut strides = vec![0usize; r.dims.len()];
                let mut stride = 1usize;
                if let Some(k) = r.fold_key {
                    strides[k] = 1;
                    stride = key_domain;
                }
                for &p in r.partitions.iter().rev() {
                    strides[p] = stride;
                    stride *= r.dims[p].domain as usize;
                }
                let mut values: Vec<Box<[i64]>> = r
                    .states
                    .iter()
                    .map(|s| vec![s.identity(); cells as usize].into_boxed_slice())
                    .collect();
                let radices: Vec<Vec<u32>> = r
                    .partitions
                    .iter()
                    .map(|&p| (0..r.dims[p].domain).collect())
                    .collect();
                for_each_tuple(&radices, |tuple| {
                    let members: Vec<Filter> = r
                        .partitions
                        .iter()
                        .zip(tuple)
                        .map(|(&p, &m)| member_filter(&r.dims[p], m))
                        .collect();
                    let at: usize = r
                        .partitions
                        .iter()
                        .zip(tuple)
                        .map(|(&p, &m)| m as usize * strides[p])
                        .sum();
                    for (s, col) in values.iter_mut().enumerate() {
                        let f = conj(&base, &members);
                        if r.fold_key.is_some() {
                            run(f, s, Some(&mut col[at..at + key_domain]), &mut stats)?;
                        } else {
                            let v = run(f, s, None, &mut stats)?;
                            col[at] = scalar_of(&r.states[s], v);
                        }
                    }
                    Ok(())
                })?;
                CellSpace {
                    key,
                    dims: dims_meta,
                    layout: Layout::Dense { strides },
                    states: r.states.clone(),
                    values,
                }
            }
            Accumulator::Sparse { .. } => {
                // Discovery: which members of each ordinal partition occur
                // under the selection — one domain-sized count pass each.
                let mut radices: Vec<Vec<u32>> = Vec::with_capacity(r.partitions.len());
                for &p in &r.partitions {
                    let d = &r.dims[p];
                    match d.provider {
                        Provider::OrdinalLane { lane } => {
                            if d.domain > policy.domain_buffer_budget {
                                return Err(ReportError::DomainBudget {
                                    domain: d.domain,
                                    budget: policy.domain_buffer_budget,
                                });
                            }
                            let mut buf = vec![0i64; d.domain as usize];
                            stats.domain_buffer_bytes =
                                stats.domain_buffer_bytes.max((buf.len() * 8) as u64);
                            let prog = lower(&Query {
                                filter: base.clone(),
                                agg: Agg::GroupReduce {
                                    key: GroupAddr::Local(Col(lane)),
                                    agg: GroupAgg::Count,
                                },
                            })?;
                            let mut scratch = Scratch::for_program(&prog, n)?;
                            execute_extent(
                                &prog,
                                &planes,
                                &Foreign::NONE,
                                &mut scratch,
                                Out::I64(&mut buf),
                                extent.clone(),
                            )?;
                            stats.population_scans += 1;
                            stats.discovery_scans += 1;
                            stats.tile_mask_ops += prog.ops.len() as u64;
                            radices.push((0..d.domain).filter(|&m| buf[m as usize] > 0).collect());
                        }
                        Provider::DerivedBucket { .. } => radices.push((0..d.domain).collect()),
                    }
                }
                let passes: u128 = radices.iter().map(|r| r.len() as u128).product();
                if passes > u128::from(policy.pass_budget) {
                    return Err(ReportError::PassBudget {
                        passes: passes.min(u128::from(u64::MAX)) as u64,
                        budget: policy.pass_budget,
                    });
                }
                let ndims = r.dims.len();
                let mut coords: Vec<Vec<u32>> = vec![Vec::new(); ndims];
                let mut vals: Vec<Vec<i64>> = vec![Vec::new(); nstates];
                let mut inner: Vec<Vec<i64>> = if r.fold_key.is_some() {
                    stats.domain_buffer_bytes = stats
                        .domain_buffer_bytes
                        .max((key_domain * 8 * nstates) as u64);
                    r.states
                        .iter()
                        .map(|s| vec![s.identity(); key_domain])
                        .collect()
                } else {
                    Vec::new()
                };
                for_each_tuple(&radices, |tuple| {
                    let members: Vec<Filter> = r
                        .partitions
                        .iter()
                        .zip(tuple)
                        .map(|(&p, &m)| member_filter(&r.dims[p], m))
                        .collect();
                    let f = conj(&base, &members);
                    let mut emit = |k: Option<u32>, st: &dyn Fn(usize) -> i64| {
                        for (&p, &m) in r.partitions.iter().zip(tuple) {
                            coords[p].push(m);
                        }
                        if let (Some(kd), Some(k)) = (r.fold_key, k) {
                            coords[kd].push(k);
                        }
                        for (s, v) in vals.iter_mut().enumerate() {
                            v.push(st(s));
                        }
                    };
                    if r.fold_key.is_some() {
                        for (s, buf) in inner.iter_mut().enumerate() {
                            buf.fill(r.states[s].identity());
                            run(f.clone(), s, Some(buf), &mut stats)?;
                        }
                        for (k, &count) in inner[0].iter().enumerate() {
                            if count > 0 {
                                emit(Some(k as u32), &|s| inner[s][k]);
                            }
                        }
                    } else {
                        let got = (0..nstates)
                            .map(|s| {
                                run(f.clone(), s, None, &mut stats)
                                    .map(|v| scalar_of(&r.states[s], v))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if got[0] > 0 {
                            emit(None, &|s| got[s]);
                        }
                    }
                    Ok(())
                })?;
                let coords: Vec<Box<[u32]>> =
                    coords.into_iter().map(Vec::into_boxed_slice).collect();
                let cells = vals[0].len();
                let mut index = HashMap::with_capacity(cells);
                for slot in 0..cells {
                    let c: Box<[u32]> = coords.iter().map(|col| col[slot]).collect();
                    index.insert(c, slot as u32);
                }
                CellSpace {
                    key,
                    dims: dims_meta,
                    layout: Layout::Sparse { coords, index },
                    states: r.states.clone(),
                    values: vals.into_iter().map(Vec::into_boxed_slice).collect(),
                }
            }
        };
        stats.accumulator_bytes = space.accumulator_bytes() as u64;
        stats.result_cells = space.stored_cells() as u64;
        Ok((ReportResult::new(Arc::new(space), self)?, stats))
    }
}
