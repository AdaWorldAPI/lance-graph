//! A witnessed selection, folded (D-DR-SELECT-K-SV-2).
//!
//! `lance_graph_mask_risc::execute_sel` folds a list of row positions over a
//! lane and knows a frame only as a row count. This is the boundary in front
//! of it: the lane and the positions are taken, together, from one source
//! through one checked [`LaneBinding`], and only then folded.
//!
//! It adds no authority. The witness is DT64's
//! ([`LaneBinding::selected`]): the frame is the source's own store's current
//! seal, the route and the basis are as bound, and the selection is one the
//! source holds under that frame, in the binding's row order and for its
//! reader scope. The fold is mask-RISC's, called exactly once. Nothing here
//! stores a position, maps a coordinate or reads a second time.
//!
//! ## What a pass does and does not mean
//!
//! - **Means:** the positions and the lane come from the same owner under
//!   its current frame and one row order, and every position is a row of
//!   that lane (`execute_sel` refuses otherwise).
//! - **Does not mean authorized.** A `Scope` is a plain value with no issuer
//!   (see [`crate::lane_source`]'s "Reader scope" note). Scope equality here
//!   is a consistency check; a revoked grant is not observed.
//! - **Does not mean ordered.** The list is a bag in list order. No fold
//!   here depends on order, and no order witness is consulted.
//! - **Does not read history.** Only the current frame is admitted.

use lance_graph_contract::class_view::ClassView;
use lance_graph_dt64::dr::FrameStore;
use lance_graph_dt64::{BindStore, LaneBinding, LaneRefusal, SelectionSource};
use lance_graph_mask_risc::{execute_sel, ExecError, LaneRef, Planes, SelFold, Value};

/// The folds a witnessed selection admits: the two `execute_sel` admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundFold {
    /// How many selected rows hold a value greater than `t` (signed, strict).
    CountGtI32 { t: i32 },
    /// The sum of the field over the selected rows, widened to `i64`.
    SumI32,
}

/// Why a witnessed fold was refused. Either way no value was folded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundSelRefusal {
    /// The joint witness failed: frame, route, basis, mapping, row order,
    /// scope or the selection itself. Nothing was borrowed.
    Binding(LaneRefusal),
    /// The witness held, and the fold refused its operands: the field's lane
    /// is not `i32`, or a position is not a row of the lane.
    Fold(ExecError),
}

/// Fold `field` over the source's selection `sel`, under `binding`'s witness.
///
/// One [`LaneBinding::selected`] (one check, then both borrows), then one
/// `execute_sel`. A position listed twice is folded twice. Allocates nothing.
#[allow(clippy::too_many_arguments)]
pub fn fold_bound_selection<'s, V, S>(
    binding: &LaneBinding,
    store: &BindStore,
    view: &V,
    frames: &FrameStore,
    source: &'s S,
    field: u8,
    sel: u16,
    fold: BoundFold,
) -> Result<Value, BoundSelRefusal>
where
    V: ClassView + ?Sized,
    S: SelectionSource<Lane<'s> = LaneRef<'s>>,
{
    let (lane, positions) = binding
        .selected(store, view, frames, source, field, sel)
        .map_err(BoundSelRefusal::Binding)?;
    let lanes = [lane];
    let planes = Planes {
        n_rows: lanes[0].len(),
        masks: &[],
        lanes: &lanes,
    };
    let fold = match fold {
        BoundFold::CountGtI32 { t } => SelFold::CountGtI32 { lane: 0, t },
        BoundFold::SumI32 => SelFold::SumI32 { lane: 0 },
    };
    execute_sel(fold, &planes, positions).map_err(BoundSelRefusal::Fold)
}
