//! A thought whose rows are positions of a DT64 [`LocalMap`].
//!
//! A [`BoundThought`] records a [`MapStamp`] of the map it was written under
//! (width, length, digest). It is admitted later only under that same map, or
//! under a map that only appended to it.
//!
//! What a pass means: the rows of the thought are positions of one coordinate
//! assignment, and for those positions the assignment is unchanged.
//!
//! What a pass does not mean:
//!
//! - It is not an authorization. Nothing here says who may write or read.
//! - It is not a frame or version witness. The caller still owns whether the
//!   map it passes is the current one.
//! - The digest is not collision-proof. It is [`LocalMap::digest`] (FNV-1a),
//!   reused as is; this module defines no second digest.

use std::collections::BTreeSet;

use lance_graph_dt64::dr::{LocalMap, Width};
use thought_git_probe::fixtures::{Cell, Outcome, Thought};

/// What a thought records about the map it was written under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapStamp {
    /// Positions per axis of the tile the map was prepared for.
    pub width: Width,
    /// How many ids the map held.
    pub len: usize,
    /// [`LocalMap::digest`] of the map at that length.
    pub digest: u64,
}

/// How a current map relates to the map a thought was written under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admitted {
    /// The current map is the map the thought was written under.
    Same,
    /// The current map only appended `appended` ids to it.
    Extended {
        /// How many ids were appended.
        appended: usize,
    },
}

/// Why a thought was refused, at build time or at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceRefusal {
    /// The current map is prepared for another tile width.
    WidthChanged {
        /// The width the thought was written under.
        written: Width,
        /// The width of the current map.
        current: Width,
    },
    /// The current map holds fewer ids than the thought was written under.
    Shrunk {
        /// The length the thought was written under.
        written: usize,
        /// The length of the current map.
        current: usize,
    },
    /// The first `written` positions of the current map are not the ids the
    /// thought was written under.
    Reordered,
    /// A write names a row the map does not hold.
    RowOutOfMap {
        /// The offending row.
        row: u32,
        /// How many positions the map holds.
        len: usize,
    },
    /// A write names a lane at or past the lane count.
    LaneOutOfRange {
        /// The offending lane.
        lane: u8,
        /// The lane count of the space.
        lanes: u8,
    },
    /// A cell is written twice.
    DuplicateCell(Cell),
}

impl MapStamp {
    /// The stamp of `map`: its width, length and digest.
    #[must_use]
    pub fn of(map: &LocalMap) -> Self {
        Self {
            width: map.width(),
            len: map.len(),
            digest: map.digest(),
        }
    }

    /// Whether `current` is the map this stamp was taken from, or only
    /// appended to it. Only the first `len` positions are compared; what is
    /// appended past them is not looked at.
    ///
    /// A digest is not an identity: [`LocalMap::digest`] is FNV-1a, so two
    /// different id tables can collide, and a pass is only as strong as that.
    ///
    /// # Errors
    ///
    /// [`SpaceRefusal::WidthChanged`], [`SpaceRefusal::Shrunk`] or
    /// [`SpaceRefusal::Reordered`], checked in that order.
    pub fn admits(&self, current: &LocalMap) -> Result<Admitted, SpaceRefusal> {
        if current.width() != self.width {
            return Err(SpaceRefusal::WidthChanged {
                written: self.width,
                current: current.width(),
            });
        }
        if current.len() < self.len {
            return Err(SpaceRefusal::Shrunk {
                written: self.len,
                current: current.len(),
            });
        }
        let prefix = current
            .view()
            .globals()
            .get(..self.len)
            .ok_or(SpaceRefusal::Reordered)?;
        // The prefix of a valid map is valid, so `prepare` does not fail here;
        // an impossible failure is reported as a mismatch, never unwrapped.
        let prefix_digest = LocalMap::prepare(self.width, prefix)
            .map_err(|_| SpaceRefusal::Reordered)?
            .digest();
        if prefix_digest != self.digest {
            return Err(SpaceRefusal::Reordered);
        }
        if current.len() == self.len {
            Ok(Admitted::Same)
        } else {
            Ok(Admitted::Extended {
                appended: current.len() - self.len,
            })
        }
    }
}

/// A thought over the local positions of one [`LocalMap`], with the stamp of
/// that map.
#[derive(Debug, Clone)]
pub struct BoundThought {
    thought: Thought,
    stamp: MapStamp,
    lanes: u8,
}

impl BoundThought {
    /// Builds a thought whose rows are positions of `map`.
    ///
    /// Every write is checked before anything is built: its row must be a
    /// position the map holds (not merely a position of the tile), its lane
    /// must be below `lanes`, and no cell may be written twice.
    ///
    /// # Errors
    ///
    /// [`SpaceRefusal::RowOutOfMap`], [`SpaceRefusal::LaneOutOfRange`] or
    /// [`SpaceRefusal::DuplicateCell`] for the first offending write.
    pub fn new(
        id: u32,
        map: &LocalMap,
        lanes: u8,
        writes: Vec<(Cell, u64)>,
        outcome: Outcome,
    ) -> Result<Self, SpaceRefusal> {
        let mut seen = BTreeSet::new();
        for &(c, _) in &writes {
            if usize::try_from(c.row).unwrap_or(usize::MAX) >= map.len() {
                return Err(SpaceRefusal::RowOutOfMap {
                    row: c.row,
                    len: map.len(),
                });
            }
            if c.lane >= lanes {
                return Err(SpaceRefusal::LaneOutOfRange {
                    lane: c.lane,
                    lanes,
                });
            }
            if !seen.insert(c) {
                return Err(SpaceRefusal::DuplicateCell(c));
            }
        }
        let rows = u32::try_from(map.len()).unwrap_or(u32::MAX);
        Ok(Self {
            thought: Thought::in_space(id, rows, lanes, writes, outcome),
            stamp: MapStamp::of(map),
            lanes,
        })
    }

    /// The stamp of the map the thought was written under.
    #[must_use]
    pub fn stamp(&self) -> MapStamp {
        self.stamp
    }

    /// The lane count of the thought's space.
    #[must_use]
    pub fn lanes(&self) -> u8 {
        self.lanes
    }

    /// The thought, only under a map the stamp admits.
    ///
    /// # Errors
    ///
    /// Whatever [`MapStamp::admits`] refuses.
    pub fn admit(&self, current: &LocalMap) -> Result<&Thought, SpaceRefusal> {
        self.stamp.admits(current).map(|_| &self.thought)
    }
}
