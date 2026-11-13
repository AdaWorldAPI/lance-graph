//! A versioned coordinate for a mailbox row (D-VAS-B2), on DT64's existing
//! [`MapTable`].
//!
//! `MapTable` is the versioned global -> local table: per tile, an
//! append-only history of maps from a 64-bit global id to a local position.
//! Here the tile is one mailbox and the local position is a row of it. This
//! module adds no table, no id storage and no version counter; it pairs a
//! map frame with a lane binding so that a read is refused when either has
//! moved.
//!
//! Two versions, kept apart:
//!
//! - the **source** version is the publication of the mailbox
//!   ([`crate::MailboxLanes`]); it moves when bytes may have changed;
//! - the **coordinate** version is the map's; it moves when the SAME bytes
//!   are to be read through another assignment of ids to rows. Advancing the
//!   map writes nothing to the mailbox.
//!
//! What this is NOT: a `NodeGuid` resolver (D-VAS-B1, BLOCKED). The id is the
//! map's own `u64`. `MailboxSoA` holds no key column, its
//! `MailboxSoaView::row_for_local_key` answers `None`, and `MapTable` cannot
//! hold a 128-bit identity. Whoever fills the map decides which id sits on
//! which row; no authority for that exists for a mailbox, so in this crate's
//! tests the test fills it.

use lance_graph_dt64::dr::{Frame, FrameRefusal, MapRead, MapRefusal, MapTable, TileId};

/// Why a coordinate did not resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordRefusal {
    /// The frame was issued by another table.
    ForeignMap,
    /// The map has advanced since the frame was sealed. The old assignment
    /// is still readable as history through `MapTable::map`; it is not
    /// applied to current bytes.
    StaleMapping,
    /// The table holds no map for the tile.
    NoSuchTile(TileId),
    /// The map holds no such id.
    Unmapped,
    /// The map names a row the lane does not have.
    RowOutsideLane { row: u32, rows: usize },
}

/// A coordinate binding: one tile of one map, at one sealed version.
#[derive(Debug, Clone)]
pub struct Coordinates {
    tile: TileId,
    frame: Frame,
}

impl Coordinates {
    /// Bind `tile` at the table's CURRENT version.
    #[must_use]
    pub fn bind(table: &MapTable, tile: TileId) -> Self {
        Self {
            tile,
            frame: table.seal(),
        }
    }

    /// The map frame this binding was sealed under.
    #[must_use]
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// The row `global` sits on, in a lane of `rows` rows. Refused unless
    /// this binding's frame is the table's current one.
    pub fn row_of(&self, table: &MapTable, global: u64, rows: usize) -> Result<u32, CoordRefusal> {
        let map = table.map(&self.frame, self.tile).map_err(|e| match e {
            MapRead::Stale(FrameRefusal::ForeignFrame) | MapRead::Stale(_) => {
                CoordRefusal::ForeignMap
            }
            MapRead::NoSuchTile(t) => CoordRefusal::NoSuchTile(t),
            MapRead::FutureVersion { .. } => CoordRefusal::StaleMapping,
        })?;
        if !self.frame.same_issue(&table.seal()) {
            return Err(CoordRefusal::StaleMapping);
        }
        let row = u32::from(map.local(global).map_err(|e| match e {
            MapRefusal::Unmapped
            | MapRefusal::TooMany
            | MapRefusal::Duplicate
            | MapRefusal::OutOfTile => CoordRefusal::Unmapped,
        })?);
        if row as usize >= rows {
            return Err(CoordRefusal::RowOutsideLane { row, rows });
        }
        Ok(row)
    }
}
