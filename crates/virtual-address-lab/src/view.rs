//! A virtual property: a recipe applied at the moment a value is demanded,
//! and a recipe chain normalised before the existing selection fold is
//! called. This module owns no lane, no position and no order.

use std::cell::Cell;

use lance_graph_dt64::{LaneBinding, LaneRefusal};
use lance_graph_mask_risc::{LaneRef, Value};
use lance_graph_report::bound_sel::{fold_bound_selection, BoundFold, BoundSelRefusal};

use crate::grid::{Grid, World};
use crate::recipe::Recipe;

/// What demand-driven reads did.
#[derive(Debug, Default)]
pub struct Ledger {
    /// Recipes resolved to a physical `(field, position)`.
    pub resolutions: Cell<u64>,
    /// Values read.
    pub reads: Cell<u64>,
}

/// Why a virtual read or fold did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The view has no such cell.
    OutsideTheView,
    /// The binding's check, or a field it does not bind.
    Lane(LaneRefusal),
    /// The lane is not an `i32` lane.
    LaneKind,
    /// The view property is one position of every lane, not one field.
    /// `LaneBinding::selected` borrows exactly one field's lane, so no
    /// existing operand holds this column; nothing was run.
    PropertyCrossesFields { recipe: Recipe },
    /// The existing boundary refused.
    Fold(BoundSelRefusal),
    /// A selected position is not a row of the lent lane.
    PositionOutsideLane { at: usize, position: u32 },
}

/// One row of a virtual table: a recipe and a row number. Sixteen bytes, no
/// borrow, no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowObject {
    pub recipe: Recipe,
    pub row: usize,
}

impl RowObject {
    /// One demanded property: the recipe resolved, the field's lane taken
    /// through the binding's check, one value read.
    pub fn get(
        &self,
        world: &World,
        binding: &LaneBinding,
        grid: &Grid,
        property: usize,
        ledger: &Ledger,
    ) -> Result<i32, Refused> {
        let (field, pos) = self
            .recipe
            .resolve(grid.lanes(), grid.len(), self.row, property)
            .ok_or(Refused::OutsideTheView)?;
        ledger.resolutions.set(ledger.resolutions.get() + 1);
        let lane = binding
            .lane(
                &world.store,
                &world.view,
                grid.frame_store(),
                grid,
                field as u8,
            )
            .map_err(Refused::Lane)?;
        let LaneRef::I32(values) = lane else {
            return Err(Refused::LaneKind);
        };
        ledger.reads.set(ledger.reads.get() + 1);
        Ok(values[pos])
    }
}

/// The field a chain of views makes of `property`, after normalisation.
/// Reads nothing and allocates nothing.
pub fn field_through(chain: &[Recipe], grid: &Grid, property: usize) -> Result<u8, Refused> {
    field_in(chain, grid.lanes(), grid.len(), property)
}

/// [`field_through`] over a tile of `lanes` fields and `len` rows.
pub fn field_in(
    chain: &[Recipe],
    lanes: usize,
    len: usize,
    property: usize,
) -> Result<u8, Refused> {
    let recipe = Recipe::normalize(chain);
    let (_, properties) = recipe.view_dims(lanes, len);
    if property >= properties {
        return Err(Refused::OutsideTheView);
    }
    recipe
        .field_of(lanes, len, property)
        .map(|f| f as u8)
        .ok_or(Refused::PropertyCrossesFields { recipe })
}

/// `fold` of view property `property` (seen through `chain`) over the
/// source-issued selection `sel`. The chain only chooses the `field` handed
/// to the existing boundary; the selection is the source's, unchanged.
pub fn fold_property(
    world: &World,
    binding: &LaneBinding,
    grid: &Grid,
    chain: &[Recipe],
    property: usize,
    sel: u16,
    fold: BoundFold,
) -> Result<Value, Refused> {
    let field = field_through(chain, grid, property)?;
    fold_bound_selection(
        binding,
        &world.store,
        &world.view,
        grid.frame_store(),
        grid,
        field,
        sel,
        fold,
    )
    .map_err(Refused::Fold)
}
