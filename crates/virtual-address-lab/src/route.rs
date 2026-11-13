//! Folds over view columns. The recipe is consulted once, when the fold is
//! planned; a row-order reversal is dropped when the terminal does not care
//! about row order, and kept when the output is in row order.

use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Scratch, Terminal, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, LowerError, Query};

use crate::grid::Grid;
use crate::recipe::Access;
use crate::view::{Trace, VirtualTable};

/// Why a fold was not planned or run.
#[derive(Debug, Clone, PartialEq)]
pub enum Stop {
    /// The view has no such column.
    NoColumn,
    /// The columns cross lanes: no bound operand holds them. See
    /// [`AcrossPlan`] for the unadmitted strided form and
    /// [`sum_where_eq_cell_by_cell`] for the bound one.
    CrossLane,
    /// The two columns are not in the same physical row order.
    Misaligned,
    Lower(LowerError),
    Exec(ExecError),
    Value(Value),
}

/// A planned fold over bound lanes.
pub struct FoldPlan {
    pub program: Program,
    scratch: Scratch<'static>,
    /// Whether the view's row order is the reverse of the lanes'.
    pub reversed: bool,
    /// Whether that reversal was dropped because the terminal is
    /// permutation-invariant.
    pub reversal_cancelled: bool,
}

fn lane_of(table: &VirtualTable<'_>, col: usize) -> Result<(usize, bool), Stop> {
    let column = table.column(col).ok_or(Stop::NoColumn)?;
    match column.access {
        Access::Lane(lane) => Ok((lane, column.reversed)),
        Access::AcrossLanes { .. } => Err(Stop::CrossLane),
    }
}

fn plan(
    table: &VirtualTable<'_>,
    where_col: usize,
    v: i32,
    agg_col: Option<usize>,
    agg: fn(Col) -> Agg,
    keeps_order: bool,
) -> Result<FoldPlan, Stop> {
    let (where_lane, reversed) = lane_of(table, where_col)?;
    let agg_lane = match agg_col {
        Some(col) => {
            let (lane, r) = lane_of(table, col)?;
            if r != reversed {
                return Err(Stop::Misaligned);
            }
            lane
        }
        None => where_lane,
    };
    let program = lower(&Query {
        filter: Filter::cmp(Col(where_lane as u16), Cmp::EqI32(v)),
        agg: agg(Col(agg_lane as u16)),
    })
    .map_err(Stop::Lower)?;
    let scratch = Scratch::for_program(&program, table.len()).map_err(Stop::Exec)?;
    Ok(FoldPlan {
        program,
        scratch,
        reversed,
        reversal_cancelled: reversed && !keeps_order,
    })
}

/// `SUM(view column sum_col) WHERE view column where_col = v`.
pub fn plan_sum_where_eq(
    table: &VirtualTable<'_>,
    sum_col: usize,
    where_col: usize,
    v: i32,
) -> Result<FoldPlan, Stop> {
    plan(table, where_col, v, Some(sum_col), Agg::SumI32, false)
}

/// `COUNT(*) WHERE view column where_col = v`.
pub fn plan_count_eq(table: &VirtualTable<'_>, where_col: usize, v: i32) -> Result<FoldPlan, Stop> {
    plan(table, where_col, v, None, |_| Agg::Count, false)
}

/// The rows `WHERE view column where_col = v`, as a mask. Order-sensitive.
pub fn plan_rows_eq(table: &VirtualTable<'_>, where_col: usize, v: i32) -> Result<FoldPlan, Stop> {
    plan(table, where_col, v, None, |_| Agg::Rows, true)
}

impl FoldPlan {
    fn run(&mut self, table: &VirtualTable<'_>, out: Out<'_>) -> Result<Value, Stop> {
        let planes = Planes {
            n_rows: table.len(),
            masks: &[],
            lanes: table.operands(),
        };
        Trace::bump(&table.trace.folds, 1);
        execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            out,
        )
        .map_err(Stop::Exec)
    }

    pub fn sum(&mut self, table: &VirtualTable<'_>) -> Result<i64, Stop> {
        match self.run(table, Out::None)? {
            Value::SumI64(s) => Ok(s),
            other => Err(Stop::Value(other)),
        }
    }

    pub fn count(&mut self, table: &VirtualTable<'_>) -> Result<usize, Stop> {
        match self.run(table, Out::None)? {
            Value::Count(c) => Ok(c),
            other => Err(Stop::Value(other)),
        }
    }

    /// The selected rows into `words`, in PHYSICAL row order, with the
    /// recipe's reversal attached instead of applied.
    pub fn rows<'w>(
        &mut self,
        table: &VirtualTable<'_>,
        words: &'w mut [u64],
    ) -> Result<MaskView<'w>, Stop> {
        self.run(table, Out::Mask(words))?;
        Ok(MaskView {
            words,
            rows: table.len(),
            reversed: self.reversed,
        })
    }
}

/// A row mask in physical order plus the view's row-order law.
#[derive(Debug, Clone, Copy)]
pub struct MaskView<'w> {
    words: &'w [u64],
    rows: usize,
    reversed: bool,
}

impl MaskView<'_> {
    /// Whether VIEW row `row` is selected. One address resolution, no move.
    pub fn has(&self, row: usize) -> bool {
        if row >= self.rows {
            return false;
        }
        let at = if self.reversed {
            self.rows - 1 - row
        } else {
            row
        };
        self.words[at / 64] >> (at % 64) & 1 == 1
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// The mask in view row order as its own buffer: a physical transform of
    /// the MASK (never of a lane), for a consumer that insists on one.
    pub fn to_view_order(&self, trace: &Trace) -> Vec<u64> {
        let mut out = vec![0u64; self.words.len()];
        for row in (0..self.rows).filter(|&r| self.has(r)) {
            out[row / 64] |= 1u64 << (row % 64);
        }
        Trace::bump(&trace.transforms, 1);
        out
    }
}

/// `SUM(column sum_pos) WHERE column where_pos = v` for a view whose columns
/// cross lanes, as ONE program over two strided operands of the source's
/// slab. Unadmitted: it reads [`Grid::across`], which no binding issues.
pub struct AcrossPlan {
    pub program: Program,
    scratch: Scratch<'static>,
}

impl AcrossPlan {
    pub fn sum_where_eq(grid: &Grid, v: u32) -> Result<AcrossPlan, Stop> {
        let program = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 1, v },
                under: None,
                dst: 0,
            }],
            Terminal::MaskedStridedGroupSum {
                mask: Operand::Scratch(0),
                lane: 0,
                groups: 1,
                group_bytes: 4,
            },
        );
        let scratch = Scratch::for_program(&program, grid.lanes()).map_err(Stop::Exec)?;
        Ok(AcrossPlan { program, scratch })
    }

    /// Positions `sum_pos` and `where_pos` of every lane.
    pub fn run(&mut self, grid: &Grid, sum_pos: usize, where_pos: usize) -> Result<i64, Stop> {
        let lanes: [LaneRef<'_>; 2] = [grid.across(sum_pos), grid.across(where_pos)];
        let planes = Planes {
            n_rows: grid.lanes(),
            masks: &[],
            lanes: &lanes,
        };
        match execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::None,
        )
        .map_err(Stop::Exec)?
        {
            Value::StridedSum(Some(s)) => Ok(s),
            other => Err(Stop::Value(other)),
        }
    }
}

/// The same demand for a lane-crossing view through the BOUND table: one
/// resolved read per cell, no operand, no copy.
pub fn sum_where_eq_cell_by_cell(
    table: &VirtualTable<'_>,
    sum_col: usize,
    where_col: usize,
    v: i32,
) -> Option<i64> {
    let (rows, _) = table.dims();
    let mut sum = 0i64;
    for r in 0..rows {
        if table.value(r, where_col)? == v {
            sum += i64::from(table.value(r, sum_col)?);
        }
    }
    Some(sum)
}
