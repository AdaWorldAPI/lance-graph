//! The virtual cube's method: a selection name and an orientation.
//!
//! Selection and mapping are existing DR/DT64 concerns. This module defines
//! neither: the positions are the source's, lent under
//! `LaneBinding::selected`'s joint witness, and field -> lane is the
//! binding's. The method only NAMES the selection and carries the recipe.
//!
//! The tile the recipe acts on is `fields x ordinals`; an ordinal indexes the
//! issued selection. A recipe that reverses the position axis reverses the
//! traversal of that selection and names no row the source did not issue.

use lance_graph_dt64::LaneBinding;
use lance_graph_mask_risc::LaneRef;

use crate::grid::{Grid, World};
use crate::recipe::Recipe;
use crate::view::{Ledger, Refused};

/// A method over the cube. Holds no borrow, no value, no length, no frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Method {
    /// The name of a source-issued selection.
    pub selection: u16,
    pub orientation: Recipe,
}

/// `f(x)`: a field of the class basis and an ordinal of the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub field: usize,
    pub ordinal: usize,
}

impl Method {
    pub const fn new(selection: u16, orientation: Recipe) -> Method {
        Method {
            selection,
            orientation,
        }
    }

    /// This method seen through one more recipe. The selection is unchanged.
    pub const fn then(self, next: Recipe) -> Method {
        Method {
            selection: self.selection,
            orientation: self.orientation.then(next),
        }
    }

    /// The address function over a tile of `fields` x `selected` ordinals.
    /// Pure: reads nothing.
    pub fn cell(self, fields: usize, selected: usize, row: usize, col: usize) -> Option<Cell> {
        let (field, ordinal) = self.orientation.resolve(fields, selected, row, col)?;
        Some(Cell { field, ordinal })
    }

    /// The field of view cell `(row, col)`. Needs neither the selection nor a
    /// lane: it is what ON is decided on.
    pub fn field(self, fields: usize, row: usize, col: usize) -> Option<u8> {
        self.orientation
            .lane_at(fields, row, col)
            .and_then(|f| u8::try_from(f).ok())
    }

    /// The value at `(field, ordinal)` of this method's selection: the read a
    /// demand names once behaviour has been admitted.
    pub fn read_cell(
        self,
        world: &World,
        binding: &LaneBinding,
        grid: &Grid,
        cell: Cell,
        ledger: &Ledger,
    ) -> Result<i32, Refused> {
        let field = u8::try_from(cell.field).map_err(|_| Refused::OutsideTheView)?;
        let (lane, positions) = binding
            .selected(
                &world.store,
                &world.view,
                grid.frame_store(),
                grid,
                field,
                self.selection,
            )
            .map_err(Refused::Lane)?;
        let LaneRef::I32(values) = lane else {
            return Err(Refused::LaneKind);
        };
        let position = *positions.get(cell.ordinal).ok_or(Refused::OutsideTheView)?;
        let value = values
            .get(position as usize)
            .ok_or(Refused::PositionOutsideLane {
                at: cell.ordinal,
                position,
            })?;
        ledger.reads.set(ledger.reads.get() + 1);
        Ok(*value)
    }

    /// One demanded cell. The binding's check and the source's selection are
    /// taken through `LaneBinding::selected`; the recipe is evaluated once
    /// for both coordinates; one value is read.
    pub fn read(
        self,
        world: &World,
        binding: &LaneBinding,
        grid: &Grid,
        row: usize,
        col: usize,
        ledger: &Ledger,
    ) -> Result<i32, Refused> {
        let field = self
            .orientation
            .lane_at(grid.lanes(), row, col)
            .ok_or(Refused::OutsideTheView)?;
        let (lane, positions) = binding
            .selected(
                &world.store,
                &world.view,
                grid.frame_store(),
                grid,
                u8::try_from(field).map_err(|_| Refused::OutsideTheView)?,
                self.selection,
            )
            .map_err(Refused::Lane)?;
        let cell = self
            .cell(grid.lanes(), positions.len(), row, col)
            .ok_or(Refused::OutsideTheView)?;
        debug_assert_eq!(cell.field, field);
        ledger.resolutions.set(ledger.resolutions.get() + 1);
        let LaneRef::I32(values) = lane else {
            return Err(Refused::LaneKind);
        };
        let position = positions[cell.ordinal];
        let value = values
            .get(position as usize)
            .ok_or(Refused::PositionOutsideLane {
                at: cell.ordinal,
                position,
            })?;
        ledger.reads.set(ledger.reads.get() + 1);
        Ok(*value)
    }
}
