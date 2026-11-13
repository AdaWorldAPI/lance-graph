//! A virtual table: bound lanes plus a recipe. Constructing it, a row object
//! or a column descriptor reads no value.

use std::cell::Cell;

use lance_graph_dt64::{LaneBinding, LaneRefusal};
use lance_graph_mask_risc::LaneRef;

use crate::grid::{Grid, World};
use crate::recipe::{Column, Recipe};

/// What a route did.
#[derive(Debug, Default)]
pub struct Trace {
    /// Address recipes resolved to a physical cell or run.
    pub resolutions: Cell<u64>,
    /// Single values read on demand.
    pub reads: Cell<u64>,
    /// Indexed reads through a position list.
    pub gathers: Cell<u64>,
    /// Physical rearrangements of a population (a view built, a mask
    /// reversed).
    pub transforms: Cell<u64>,
    /// Values written into a new buffer.
    pub copied: Cell<u64>,
    /// Bulk programs run over lanes.
    pub folds: Cell<u64>,
}

impl Trace {
    pub fn bump(cell: &Cell<u64>, by: u64) {
        cell.set(cell.get() + by);
    }

    /// `(resolutions, reads, gathers, transforms, copied, folds)`.
    pub fn snapshot(&self) -> [u64; 6] {
        [
            self.resolutions.get(),
            self.reads.get(),
            self.gathers.get(),
            self.transforms.get(),
            self.copied.get(),
            self.folds.get(),
        ]
    }
}

/// Why a virtual table could not be formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableRefusal {
    Lanes(LaneRefusal),
    /// The binding does not bind every lane as an `i32` lane of one length.
    Shape,
}

/// Bound lanes seen through a recipe.
pub struct VirtualTable<'a> {
    lanes: Vec<LaneRef<'a>>,
    len: usize,
    recipe: Recipe,
    pub trace: &'a Trace,
}

impl<'a> VirtualTable<'a> {
    /// Take every lane through `binding`'s one check and attach `recipe`.
    /// Lane `l` is operand `l` (every field is bound, so rank is position).
    pub fn bound(
        world: &World,
        binding: &LaneBinding,
        grid: &'a Grid,
        recipe: Recipe,
        trace: &'a Trace,
    ) -> Result<VirtualTable<'a>, TableRefusal> {
        let mut lanes = Vec::with_capacity(grid.lanes());
        for item in binding
            .borrow_all(&world.store, &world.view, grid.frame_store(), grid)
            .map_err(TableRefusal::Lanes)?
        {
            let (_field, lane) = item.map_err(TableRefusal::Lanes)?;
            lanes.push(lane);
        }
        let ok = lanes.len() == grid.lanes()
            && lanes
                .iter()
                .all(|l| matches!(l, LaneRef::I32(v) if v.len() == grid.len()));
        if !ok {
            return Err(TableRefusal::Shape);
        }
        Ok(VirtualTable {
            lanes,
            len: grid.len(),
            recipe,
            trace,
        })
    }

    pub fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// The same lanes through `recipe` composed after this table's own.
    pub fn through(&self, next: Recipe) -> VirtualTable<'a> {
        VirtualTable {
            lanes: self.lanes.clone(),
            len: self.len,
            recipe: self.recipe.then(next),
            trace: self.trace,
        }
    }

    /// The bound lanes in operand order.
    pub fn operands(&self) -> &[LaneRef<'a>] {
        &self.lanes
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.lanes.is_empty()
    }

    /// `(rows, cols)` of the view.
    pub fn dims(&self) -> (usize, usize) {
        self.recipe.view_dims(self.lanes.len(), self.len)
    }

    /// Row `row` as an object. Reads nothing.
    pub fn row(&self, row: usize) -> Option<RowObject<'_, 'a>> {
        (row < self.dims().0).then_some(RowObject { table: self, row })
    }

    /// One demanded cell: one resolution, one read.
    pub fn value(&self, row: usize, col: usize) -> Option<i32> {
        let (lane, pos) = self.recipe.resolve(self.lanes.len(), self.len, row, col)?;
        Trace::bump(&self.trace.resolutions, 1);
        let LaneRef::I32(values) = self.lanes[lane] else {
            return None;
        };
        Trace::bump(&self.trace.reads, 1);
        Some(values[pos])
    }

    /// Where view column `col` lies. One resolution, no read.
    pub fn column(&self, col: usize) -> Option<Column> {
        let column = self.recipe.column(self.lanes.len(), self.len, col)?;
        Trace::bump(&self.trace.resolutions, 1);
        Some(column)
    }

    /// The reference baseline: build the whole view, row-major. One physical
    /// transform, every cell copied.
    pub fn materialize(&self) -> Vec<i32> {
        let (rows, cols) = self.dims();
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let (lane, pos) = self
                    .recipe
                    .resolve(self.lanes.len(), self.len, r, c)
                    .expect("inside the view");
                let LaneRef::I32(values) = self.lanes[lane] else {
                    unreachable!("bound() admits i32 lanes only")
                };
                out.push(values[pos]);
            }
        }
        Trace::bump(&self.trace.transforms, 1);
        Trace::bump(&self.trace.copied, (rows * cols) as u64);
        out
    }
}

/// One row of a virtual table: a row number and a borrow. No property value
/// exists until [`RowObject::get`] is called for it.
#[derive(Clone, Copy)]
pub struct RowObject<'t, 'a> {
    table: &'t VirtualTable<'a>,
    row: usize,
}

impl RowObject<'_, '_> {
    /// Property count.
    pub fn properties(&self) -> usize {
        self.table.dims().1
    }

    /// One demanded property.
    pub fn get(&self, property: usize) -> Option<i32> {
        self.table.value(self.row, property)
    }
}
