//! The PostgreSQL row polyfill for consumers (D-QSP-VROW; D-REGC-4, -10).
//!
//! A PostgreSQL row is a reading over source-owned lanes at one position,
//! never a stored row (EXO-332: `identity[index].key : identity[index].value`).
//! This crate gives a consumer the four reads a PostgreSQL-shaped answer
//! needs, each over lanes lent by ONE DT64 check per table:
//!
//! - [`Reader::open`] / [`Reader::open_selected`]: one
//!   `LaneBinding::borrow_all_present_bound` (or
//!   `borrow_all_selected_present_bound`), then every bound lane with its
//!   presence plane, plus the table's live plane ([`LiveSource`]). No
//!   `ClassView` is consulted after bind: a registered class changes only by
//!   a new versioned reference (register-contract-reconciliation C1, C2).
//! - [`Reader::keep_where`] / [`Reader::count_where`]: a Quack [`Filter`]
//!   over the lent lanes, made SQL three-valued by [`Filter::sql_where`],
//!   lowered by `lance-graph-quack` and executed by `lance-graph-mask-risc`.
//!   This crate builds programs and never evaluates a predicate itself.
//! - [`Reader::rows_at`] / [`Reader::rows_in`]: the rows of an issued
//!   selection (a bag, in list order) or of a kept mask (ascending), read in
//!   place.
//! - [`read_bag`]: a join's rows as a virtual tuple bag: one position list
//!   per side, aligned by index, row `i = (A[posA[i]], B[posB[i]], …)`. The
//!   bag is never built; a side with no row at `i` (an outer join's missing
//!   side) is −0 on every column.
//!
//! Cells are spelled as PostgreSQL text only at the end, by
//! [`lance_graph_contract::pg_row::write_cell`]; the returned [`PgRows`] are
//! the only materialization.
//!
//! # Absent, live, dead
//!
//! A cell is absent iff its presence bit is 0 and its value is 0 (−0, the
//! lane socket's rule). A row is live iff its bit in the source's live plane
//! is 1. A dead position is refused by every read ([`PgRowRefusal::DeadPosition`]),
//! and every `WHERE` is intersected with the live plane, because a deleted
//! row may still hold its old values (`.claude/knowledge/register-contract-reconciliation.md`
//! C3, C4).
//!
//! # What is not here
//!
//! - Ordering (`ORDER BY`), aggregates beyond `COUNT(*)`, and a client
//!   protocol. Aggregates stay Quack's (`lower_sum`, `lower_group_by`, …).
//! - Strided and 64-bit lanes: refused as [`PgRowRefusal::LaneKind`]
//!   (register-contract-reconciliation C8).
//! - Who issues a join's positions, and where an outer side's present plane
//!   comes from: open (EXO-329, D-REGC-4). [`read_bag`] takes both from the
//!   caller and checks them.

/// The streaming cell writer (D-INTAKE-2G). A child module, so it reads the
/// reader's lanes in place.
pub mod sink;

#[cfg(all(
    not(debug_assertions),
    any(
        feature = "disable-pg-row-presence",
        feature = "disable-pg-row-live",
        feature = "disable-pg-row-bag-present"
    )
))]
compile_error!("disable-pg-row-* features are for disable runs in debug builds only");

use lance_graph_contract::pg_row::{
    write_cell, CellRefusal, CodeText, PgColumn, PgTypeRefusal, Written,
};
use lance_graph_dt64::dr::FrameStore;
use lance_graph_dt64::{
    BindStore, CoordSpace, LaneBinding, LaneRefusal, LaneSource, PresenceSource,
};
use lance_graph_mask_risc::{
    execute_into, words_for, ExecError, Foreign, LaneRef, Out, Planes, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, LowerError, Mask, Query};

/// The fourth socket a polyfilled source implements: which of its rows are
/// live.
///
/// One bit per row, LSB-first `u64` words, in the row order `space` names.
/// The plane is the source's own and must not be copied. `None` means the
/// source holds no live plane for that row order, which is refused, never
/// read as "all live".
///
/// DT64 has no such socket yet; whether liveness becomes one there, or a
/// delete writes −0 so that a dead row reads as absent, is D-REGC-3.
pub trait LiveSource: LaneSource {
    fn borrow_live(&self, space: CoordSpace) -> Option<&[u64]>;
}

/// A lent lane, as mask-risc reads it.
pub trait IntoLaneRef<'s> {
    fn lane_ref(self) -> LaneRef<'s>;
}

impl<'s> IntoLaneRef<'s> for LaneRef<'s> {
    fn lane_ref(self) -> LaneRef<'s> {
        self
    }
}

impl<'s> IntoLaneRef<'s> for &'s [u32] {
    fn lane_ref(self) -> LaneRef<'s> {
        LaneRef::U32(self)
    }
}

impl<'s> IntoLaneRef<'s> for &'s [i32] {
    fn lane_ref(self) -> LaneRef<'s> {
        LaneRef::I32(self)
    }
}

/// One output column, in the binding's field order.
#[derive(Clone, Copy)]
pub struct ColumnSpec<'c> {
    pub name: &'c str,
    pub col: PgColumn,
    /// The code table of a text column, at the frame the lanes are read
    /// under.
    pub codes: Option<&'c dyn CodeText>,
}

/// A result as PostgreSQL prints it: `\gdesc` type names and text cells,
/// `None` for NULL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PgRows {
    pub types: Vec<String>,
    pub rows: Vec<Vec<Option<String>>>,
}

/// Why a read refused. Nothing is returned with a refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum PgRowRefusal {
    /// The DT64 check refused (frame, route, basis, mapping, scope, row
    /// order, selection, presence).
    Lane(LaneRefusal),
    /// A column's type, encoding or absent spelling is not admitted.
    Column {
        name: String,
        refusal: PgTypeRefusal,
    },
    /// The columns do not match the bound lanes one to one.
    Arity { columns: usize, lanes: usize },
    /// A lane kind the polyfill does not read (64-bit, strided).
    LaneKind { name: String },
    /// Lanes, presence or live plane do not span the same rows.
    Span { name: String },
    /// The source holds no live plane for the binding's row order.
    LiveUnbound,
    /// A position past the lanes.
    Position { at: u32 },
    /// A position whose row is not live.
    DeadPosition { at: u32 },
    /// A cell could not be spelled.
    Cell {
        name: String,
        at: u32,
        refusal: CellRefusal,
    },
    /// The filter names something other than this table's columns.
    FilterNotAdmitted(&'static str),
    /// Quack refused to lower the filter.
    Lower(LowerError),
    /// mask-risc refused the program.
    Exec(ExecError),
    /// A bag's sides disagree in length, or a present plane is short.
    Bag(&'static str),
}

/// The lanes of one table, lent under one check.
pub struct Reader<'s, 'c> {
    cols: &'c [ColumnSpec<'c>],
    lanes: Vec<(LaneRef<'s>, &'s [u64])>,
    live: &'s [u64],
    rows: usize,
}

impl<'s, 'c> Reader<'s, 'c> {
    /// One `borrow_all_present_bound` over `binding` (no `ClassView`: the
    /// read goes through the basis recorded at bind), then the live plane.
    pub fn open<S>(
        binding: &LaneBinding,
        store: &BindStore,
        frames: &FrameStore,
        source: &'s S,
        cols: &'c [ColumnSpec<'c>],
    ) -> Result<Self, PgRowRefusal>
    where
        S: PresenceSource + LiveSource,
        S::Lane<'s>: IntoLaneRef<'s>,
    {
        let lanes = binding
            .borrow_all_present_bound(store, frames, source)
            .map_err(PgRowRefusal::Lane)?;
        let mut lent = Vec::with_capacity(lanes.len());
        for item in lanes {
            let (_, lane, plane) = item.map_err(PgRowRefusal::Lane)?;
            lent.push((lane.lane_ref(), plane));
        }
        Self::new(binding, source, cols, lent)
    }

    /// One `borrow_all_selected_present_bound` over `binding` and the
    /// source's selection `sel`, then the live plane. Returns the positions
    /// the source issued, a bag in list order.
    pub fn open_selected<S>(
        binding: &LaneBinding,
        store: &BindStore,
        frames: &FrameStore,
        source: &'s S,
        sel: u16,
        cols: &'c [ColumnSpec<'c>],
    ) -> Result<(Self, &'s [u32]), PgRowRefusal>
    where
        S: lance_graph_dt64::SelectionSource + PresenceSource + LiveSource,
        S::Lane<'s>: IntoLaneRef<'s>,
    {
        let (lanes, positions) = binding
            .borrow_all_selected_present_bound(store, frames, source, sel)
            .map_err(PgRowRefusal::Lane)?;
        let mut lent = Vec::with_capacity(lanes.len());
        for item in lanes {
            let (_, lane, plane) = item.map_err(PgRowRefusal::Lane)?;
            lent.push((lane.lane_ref(), plane));
        }
        Ok((Self::new(binding, source, cols, lent)?, positions))
    }

    fn new<S: LiveSource>(
        binding: &LaneBinding,
        source: &'s S,
        cols: &'c [ColumnSpec<'c>],
        lanes: Vec<(LaneRef<'s>, &'s [u64])>,
    ) -> Result<Self, PgRowRefusal> {
        if cols.len() != lanes.len() {
            return Err(PgRowRefusal::Arity {
                columns: cols.len(),
                lanes: lanes.len(),
            });
        }
        let rows = lanes.first().map_or(0, |(l, _)| l.len());
        let words = words_for(rows);
        for (spec, (lane, plane)) in cols.iter().zip(&lanes) {
            spec.col.admit().map_err(|refusal| PgRowRefusal::Column {
                name: spec.name.to_owned(),
                refusal,
            })?;
            if !matches!(lane, LaneRef::U32(_) | LaneRef::I32(_)) {
                return Err(PgRowRefusal::LaneKind {
                    name: spec.name.to_owned(),
                });
            }
            if lane.len() != rows || plane.len() < words {
                return Err(PgRowRefusal::Span {
                    name: spec.name.to_owned(),
                });
            }
        }
        let live = source
            .borrow_live(binding.selection())
            .ok_or(PgRowRefusal::LiveUnbound)?;
        if live.len() < words {
            return Err(PgRowRefusal::Span {
                name: "live".to_owned(),
            });
        }
        Ok(Reader {
            cols,
            lanes,
            live,
            rows,
        })
    }

    /// Rows the lanes span.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The columns' `\gdesc` type names.
    pub fn types(&self) -> Vec<String> {
        self.cols.iter().map(|c| c.col.ty.gdesc()).collect()
    }

    fn is_live(&self, at: usize) -> bool {
        cfg!(feature = "disable-pg-row-live") || bit(self.live, at)
    }

    fn check_position(&self, at: u32) -> Result<usize, PgRowRefusal> {
        let p = at as usize;
        if p >= self.rows {
            return Err(PgRowRefusal::Position { at });
        }
        if !self.is_live(p) {
            return Err(PgRowRefusal::DeadPosition { at });
        }
        Ok(p)
    }

    /// Append the cells of row `p` (already checked) to `row`.
    fn push_row(&self, p: usize, row: &mut Vec<Option<String>>) -> Result<(), PgRowRefusal> {
        for (spec, (lane, plane)) in self.cols.iter().zip(&self.lanes) {
            let value = match lane {
                LaneRef::U32(v) => v[p],
                LaneRef::I32(v) => v[p] as u32,
                _ => {
                    return Err(PgRowRefusal::LaneKind {
                        name: spec.name.to_owned(),
                    })
                }
            };
            let present = cfg!(feature = "disable-pg-row-presence") || bit(plane, p);
            row.push(self.spell(spec, present, value, p as u32)?);
        }
        Ok(())
    }

    /// Append −0 on every column: the row of a missing side.
    fn push_absent(&self, row: &mut Vec<Option<String>>) -> Result<(), PgRowRefusal> {
        for spec in self.cols {
            row.push(self.spell(spec, false, 0, u32::MAX)?);
        }
        Ok(())
    }

    fn spell(
        &self,
        spec: &ColumnSpec<'_>,
        present: bool,
        value: u32,
        at: u32,
    ) -> Result<Option<String>, PgRowRefusal> {
        let mut text = String::new();
        match write_cell(&spec.col, present, value, spec.codes, &mut text) {
            Ok(Written::Null) => Ok(None),
            Ok(Written::Text) => Ok(Some(text)),
            Err(refusal) => Err(PgRowRefusal::Cell {
                name: spec.name.to_owned(),
                at,
                refusal,
            }),
        }
    }

    /// The rows at `positions`, in list order, duplicates kept.
    pub fn rows_at(&self, positions: &[u32]) -> Result<PgRows, PgRowRefusal> {
        let mut rows = Vec::with_capacity(positions.len());
        for &at in positions {
            let p = self.check_position(at)?;
            let mut row = Vec::with_capacity(self.cols.len());
            self.push_row(p, &mut row)?;
            rows.push(row);
        }
        Ok(PgRows {
            types: self.types(),
            rows,
        })
    }

    /// The rows whose bit is set in `mask`, ascending.
    pub fn rows_in(&self, mask: &[u64]) -> Result<PgRows, PgRowRefusal> {
        if mask.len() < words_for(self.rows) {
            return Err(PgRowRefusal::Span {
                name: "mask".to_owned(),
            });
        }
        let mut rows = Vec::new();
        for (w, &word) in mask.iter().enumerate().take(words_for(self.rows)) {
            let mut bits = word;
            while bits != 0 {
                let p = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = self.check_position(p as u32)?;
                let mut row = Vec::with_capacity(self.cols.len());
                self.push_row(p, &mut row)?;
                rows.push(row);
            }
        }
        Ok(PgRows {
            types: self.types(),
            rows,
        })
    }

    /// The live rows `filter` holds for, under SQL three-valued logic.
    ///
    /// `Col(k)` names the `k`-th column. The filter may use comparisons,
    /// `AND`, `OR` and `NOT` only; a plane, a foreign lane or a semijoin is
    /// refused, because their operands are not this reader's.
    pub fn keep_where(&self, filter: &Filter) -> Result<Vec<u64>, PgRowRefusal> {
        self.keep_where_inner(filter, None)
    }

    /// [`Reader::keep_where`] with each column's validity plane supplied by
    /// the caller (from [`Reader::validity_planes`], kept for the data version
    /// they were computed under). No validity fold runs. `validity` holds one
    /// plane per column, in column order.
    pub fn keep_where_valid(
        &self,
        filter: &Filter,
        validity: &[&[u64]],
    ) -> Result<Vec<u64>, PgRowRefusal> {
        self.keep_where_inner(filter, Some(validity))
    }

    /// [`Reader::count_where`] with supplied validity planes, as
    /// [`Reader::keep_where_valid`].
    pub fn count_where_valid(
        &self,
        filter: &Filter,
        validity: &[&[u64]],
    ) -> Result<usize, PgRowRefusal> {
        match self.run_where(filter, Agg::Count, Out::None, Some(validity))? {
            Value::Count(n) => Ok(n),
            _ => Err(PgRowRefusal::Bag("Count returned no count")),
        }
    }

    /// The rows a query already bound by `lance_graph_quack::bind` keeps, as
    /// a mask over this reader's positions. No name, literal or codebook is
    /// touched here: the query is numeric, and its filter is already three
    /// valued.
    ///
    /// The query's planes follow [`BOUND_LIVE`] and [`bound_validity`]: plane
    /// 0 is the live plane, plane `1 + k` is column `k`'s validity, supplied
    /// in column order as for [`Reader::keep_where_valid`]. The live plane is
    /// conjoined again here, so a query that forgot it still keeps no deleted
    /// row. Only `Agg::Rows` is admitted.
    pub fn keep_bound(&self, query: &Query, validity: &[&[u64]]) -> Result<Vec<u64>, PgRowRefusal> {
        if query.agg != Agg::Rows {
            return Err(PgRowRefusal::FilterNotAdmitted("bound query is not Rows"));
        }
        admit_bound(&query.filter, self.cols.len())?;
        let words = words_for(self.rows);
        if validity.len() != self.cols.len() || validity.iter().any(|p| p.len() < words) {
            return Err(PgRowRefusal::Span {
                name: "validity".to_owned(),
            });
        }
        let lanes: Vec<LaneRef<'s>> = self.lanes.iter().map(|(l, _)| *l).collect();
        let mut masks: Vec<&[u64]> = Vec::with_capacity(1 + validity.len());
        masks.push(self.live);
        masks.extend(validity.iter().copied());
        let q = Query {
            filter: Filter::and([Filter::plane(BOUND_LIVE), query.filter.clone()]),
            agg: Agg::Rows,
        };
        let planes = Planes {
            n_rows: self.rows,
            masks: &masks,
            lanes: &lanes,
        };
        let mut keep = vec![0u64; words];
        match run(&q, &planes, Out::Mask(&mut keep))? {
            Value::Mask(_) => Ok(keep),
            _ => Err(PgRowRefusal::Bag("Keep returned no mask")),
        }
    }

    /// Every column's validity plane (presence OR value != 0), one mask-risc
    /// fold per column. A plane depends only on the lanes and their presence,
    /// so a caller that keeps them for a data version reuses them in
    /// [`Reader::keep_where_valid`].
    pub fn validity_planes(&self) -> Result<Vec<Vec<u64>>, PgRowRefusal> {
        let all: Vec<usize> = (0..self.cols.len()).collect();
        self.validity(&all)
    }

    fn keep_where_inner(
        &self,
        filter: &Filter,
        given: Option<&[&[u64]]>,
    ) -> Result<Vec<u64>, PgRowRefusal> {
        let words = words_for(self.rows);
        let mut keep = vec![0u64; words];
        match self.run_where(filter, Agg::Rows, Out::Mask(&mut keep), given)? {
            Value::Mask(_) => Ok(keep),
            _ => Err(PgRowRefusal::Bag("Keep returned no mask")),
        }
    }

    /// `COUNT(*)` of the live rows `filter` holds for.
    pub fn count_where(&self, filter: &Filter) -> Result<usize, PgRowRefusal> {
        match self.run_where(filter, Agg::Count, Out::None, None)? {
            Value::Count(n) => Ok(n),
            _ => Err(PgRowRefusal::Bag("Count returned no count")),
        }
    }

    /// Only the columns `filter` reads need a validity plane: `sql_where`
    /// rewrites only the comparisons on them, so the other columns' planes
    /// are never read. They are taken from `given` or computed here.
    fn run_where(
        &self,
        filter: &Filter,
        agg: Agg,
        out: Out<'_>,
        given: Option<&[&[u64]]>,
    ) -> Result<Value, PgRowRefusal> {
        admit_filter(filter, self.cols.len())?;
        let lanes: Vec<LaneRef<'s>> = self.lanes.iter().map(|(l, _)| *l).collect();
        let read = filter_columns(filter);
        let words = words_for(self.rows);
        let computed;
        let planes_of: Vec<&[u64]> = match given {
            Some(planes) => {
                if planes.len() != self.cols.len() || planes.iter().any(|p| p.len() < words) {
                    return Err(PgRowRefusal::Span {
                        name: "validity".to_owned(),
                    });
                }
                read.iter().map(|&k| planes[k]).collect()
            }
            None => {
                computed = self.validity(&read)?;
                computed.iter().map(Vec::as_slice).collect()
            }
        };
        let mut masks: Vec<&[u64]> = Vec::with_capacity(1 + planes_of.len());
        masks.push(self.live);
        masks.extend(planes_of.iter().copied());
        let nullable: Vec<(Col, Mask)> = read
            .iter()
            .enumerate()
            .map(|(i, &k)| (Col(k as u16), Mask(i as u16 + 1)))
            .collect();
        let sql = filter.sql_where(&nullable);
        let filter = if cfg!(feature = "disable-pg-row-live") {
            sql
        } else {
            Filter::and([Filter::plane(Mask(0)), sql])
        };
        let planes = Planes {
            n_rows: self.rows,
            masks: &masks,
            lanes: &lanes,
        };
        run(&Query { filter, agg }, &planes, out)
    }

    /// Per column, the rows whose cell is not absent: presence OR value ≠ 0,
    /// as a mask-risc fold over the column's lane and its presence plane.
    fn validity(&self, columns: &[usize]) -> Result<Vec<Vec<u64>>, PgRowRefusal> {
        let words = words_for(self.rows);
        let mut out = Vec::with_capacity(columns.len());
        for &k in columns {
            let (lane, plane) = (&self.lanes[k].0, self.lanes[k].1);
            let mut keep = vec![0u64; words];
            if cfg!(feature = "disable-pg-row-presence") {
                keep.iter_mut().for_each(|w| *w = u64::MAX);
                if let Some(last) = keep.last_mut() {
                    let tail = self.rows % 64;
                    if tail != 0 {
                        *last = (1u64 << tail) - 1;
                    }
                }
                out.push(keep);
                continue;
            }
            let nonzero = match lane {
                LaneRef::U32(_) => Cmp::NeU32(0),
                _ => Cmp::NeI32(0),
            };
            let q = Query {
                filter: Filter::or([Filter::plane(Mask(0)), Filter::cmp(Col(0), nonzero)]),
                agg: Agg::Rows,
            };
            let planes = Planes {
                n_rows: self.rows,
                masks: &[plane],
                lanes: core::slice::from_ref(lane),
            };
            run(&q, &planes, Out::Mask(&mut keep))?;
            out.push(keep);
        }
        Ok(out)
    }
}

/// The columns a filter compares, ascending, each once.
fn filter_columns(f: &Filter) -> Vec<usize> {
    fn walk(f: &Filter, out: &mut Vec<usize>) {
        match f {
            Filter::Cmp(Col(c), _) => out.push(usize::from(*c)),
            Filter::And(parts) | Filter::Or(parts) => parts.iter().for_each(|p| walk(p, out)),
            Filter::Not(inner) => walk(inner, out),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(f, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn run(q: &Query, planes: &Planes<'_>, out: Out<'_>) -> Result<Value, PgRowRefusal> {
    let program = lower(q).map_err(PgRowRefusal::Lower)?;
    let mut scratch = Scratch::new(words_for(planes.n_rows), program.scratch_slots as usize);
    execute_into(&program, planes, &Foreign::NONE, &mut scratch, out).map_err(PgRowRefusal::Exec)
}

/// The live plane of a query bound for [`Reader::keep_bound`].
pub const BOUND_LIVE: Mask = Mask(0);

/// The validity plane of column `k` in a query bound for
/// [`Reader::keep_bound`].
pub const fn bound_validity(k: u16) -> Mask {
    Mask(1 + k)
}

/// A bound filter may read this reader's columns and its planes `0..=columns`
/// only; a foreign lane or a semijoin has no operand here.
fn admit_bound(f: &Filter, columns: usize) -> Result<(), PgRowRefusal> {
    match f {
        Filter::Cmp(Col(c), _) if usize::from(*c) < columns => Ok(()),
        Filter::Cmp(..) => Err(PgRowRefusal::FilterNotAdmitted("column out of range")),
        Filter::Plane(Mask(m)) if usize::from(*m) <= columns => Ok(()),
        Filter::Plane(_) => Err(PgRowRefusal::FilterNotAdmitted("plane out of range")),
        Filter::And(parts) | Filter::Or(parts) => {
            parts.iter().try_for_each(|p| admit_bound(p, columns))
        }
        Filter::Not(inner) => admit_bound(inner, columns),
        Filter::EqU32Via { .. } => Err(PgRowRefusal::FilterNotAdmitted("foreign lane")),
        Filter::Semijoin { .. } => Err(PgRowRefusal::FilterNotAdmitted("semijoin")),
    }
}

fn admit_filter(f: &Filter, columns: usize) -> Result<(), PgRowRefusal> {
    match f {
        Filter::Cmp(Col(c), _) if usize::from(*c) < columns => Ok(()),
        Filter::Cmp(..) => Err(PgRowRefusal::FilterNotAdmitted("column out of range")),
        Filter::And(parts) | Filter::Or(parts) => {
            parts.iter().try_for_each(|p| admit_filter(p, columns))
        }
        Filter::Not(inner) => admit_filter(inner, columns),
        Filter::Plane(_) => Err(PgRowRefusal::FilterNotAdmitted("plane")),
        Filter::EqU32Via { .. } => Err(PgRowRefusal::FilterNotAdmitted("foreign lane")),
        Filter::Semijoin { .. } => Err(PgRowRefusal::FilterNotAdmitted("semijoin")),
    }
}

/// One side of a virtual tuple bag.
pub struct BagSide<'r, 's, 'c> {
    pub reader: &'r Reader<'s, 'c>,
    /// The side's positions, aligned with every other side by index.
    pub positions: &'r [u32],
    /// Bit `i` is 1 when this side has a row at bag index `i`. `None` means
    /// every index has one (an inner side). Where an outer side's plane comes
    /// from is D-REGC-4; at a 0 bit the position is not read.
    pub present: Option<&'r [u64]>,
}

/// A join's rows: bag index `i` reads every side at its `positions[i]`, or
/// −0 on every column of a side whose present bit is 0. The columns are the
/// sides' columns, in side order. Nothing is built before the cells.
pub fn read_bag(sides: &[BagSide<'_, '_, '_>]) -> Result<PgRows, PgRowRefusal> {
    let Some(first) = sides.first() else {
        return Err(PgRowRefusal::Bag("no sides"));
    };
    let len = first.positions.len();
    for side in sides {
        if side.positions.len() != len {
            return Err(PgRowRefusal::Bag("sides differ in length"));
        }
        if side.present.is_some_and(|p| p.len() < words_for(len)) {
            return Err(PgRowRefusal::Bag("present plane shorter than the bag"));
        }
    }
    let width: usize = sides.iter().map(|s| s.reader.cols.len()).sum();
    let mut rows = Vec::with_capacity(len);
    for i in 0..len {
        let mut row = Vec::with_capacity(width);
        for side in sides {
            let here = cfg!(feature = "disable-pg-row-bag-present")
                || side.present.is_none_or(|p| bit(p, i));
            if here {
                let p = side.reader.check_position(side.positions[i])?;
                side.reader.push_row(p, &mut row)?;
            } else {
                side.reader.push_absent(&mut row)?;
            }
        }
        rows.push(row);
    }
    Ok(PgRows {
        types: sides.iter().flat_map(|s| s.reader.types()).collect(),
        rows,
    })
}

#[inline]
fn bit(plane: &[u64], at: usize) -> bool {
    plane.get(at / 64).is_some_and(|w| w >> (at % 64) & 1 == 1)
}
