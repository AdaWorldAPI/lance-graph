//! D-DRF W4–W7: hand flushed row images to the real `CycleWriter` as owned
//! casts (feature `handoff`).
//!
//! A [`RowImage`] owns its bytes, so once [`Writeback::flush`] returns, the
//! source rows and the focus can be dropped or changed while the writer is
//! still sealing. The compiler checks that; the tests check that what lands is
//! the flushed snapshot.
//!
//! One cast per distinct row. The writer's freeze folds same-row casts by
//! replacing the whole payload (later stream position wins), so two casts for
//! one row would drop one of them. [`Writeback::flush`] already coalesces per
//! row, which is why the handoff takes its output and nothing else.
//!
//! [`Writeback::flush`]: crate::Writeback::flush

use lance_graph_planner::persist_sink::{CycleId, MailboxId, SweepSlot};

use crate::RowImage;

impl RowImage {
    /// This image as one cast for `owner`. The row bytes are moved, not copied.
    pub fn into_slot(self, cycle: CycleId, stream_position: u64, owner: MailboxId) -> SweepSlot {
        SweepSlot {
            cycle,
            stream_position,
            owner,
            row: u64::from(self.ord),
            paired_move: None,
            payload: self.row,
        }
    }
}

/// One cast per image, in the images' ordinal order, at stream positions
/// `next_position..next_position + images.len()`. Returns the casts and the
/// next free position. Positions must keep increasing across cycles (the
/// writer's recovery watermark depends on it), so pass the returned value to
/// the next cycle.
pub fn into_slots(
    images: Vec<RowImage>,
    cycle: CycleId,
    owner: MailboxId,
    next_position: u64,
) -> (Vec<SweepSlot>, u64) {
    let n = images.len() as u64;
    let slots = images
        .into_iter()
        .zip(next_position..)
        .map(|(img, pos)| img.into_slot(cycle, pos, owner))
        .collect();
    (slots, next_position + n)
}
