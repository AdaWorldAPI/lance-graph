//! The three counters the plan asks for, plus what a conversion costs.

use core::cell::Cell;

/// Counts kept apart on purpose: a fold reading staged lanes is operand
/// access, not a storage access.
#[derive(Debug, Default)]
pub struct Counters {
    /// Calls into the source's selection or staging API.
    pub storage: Cell<u64>,
    /// Bindings made.
    pub binds: Cell<u64>,
    /// Rows of K-local lanes a fold was handed (per call, not per element).
    pub operand_rows: Cell<u64>,
    /// Conversions between views of one selection (mask to positions, and so on).
    pub conversions: Cell<u64>,
    /// Bytes copied out of the source or into another view.
    pub bytes_copied: Cell<u64>,
}

impl Counters {
    pub fn new() -> Counters {
        Counters::default()
    }

    pub(crate) fn bump(cell: &Cell<u64>, by: u64) {
        cell.set(cell.get() + by);
    }

    /// A copy of the five counts: storage, binds, operand rows, conversions,
    /// bytes copied.
    pub fn snapshot(&self) -> [u64; 5] {
        [
            self.storage.get(),
            self.binds.get(),
            self.operand_rows.get(),
            self.conversions.get(),
            self.bytes_copied.get(),
        ]
    }
}
