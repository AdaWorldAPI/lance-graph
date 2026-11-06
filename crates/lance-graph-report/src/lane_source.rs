//! The DT64 lane socket over a resident [`AbiBatch`] (D-LANESOCK-2).
//!
//! [`AbiLanes`] implements [`LaneSource`] for lanes this crate already holds.
//! It binds nothing itself and owns no field table:
//!
//! - **WHAT / WHICH BEHAVIOUR** is DT64's: `BindStore::bind_lanes` resolves the
//!   class, its schema revision and its field basis, and `BindStore::admit`
//!   admits the method.
//! - **WHERE** is the frame's: one [`FrameStore`] per source, whose revisions
//!   are `(SourceId, relation map, generation)`. Publishing a replacement batch
//!   advances it, so every binding sealed under the old batch is refused.
//! - **VALUES** stay in the batch: `borrow` hands out the batch's own `Arc`
//!   lane as a `LaneRef`. Nothing is copied.
//! - **HOW** is Quack / mask-RISC, over the borrowed lanes.
//!
//! A field resolves from the basis entry DT64 validated and hands to
//! `lane_of` (its `predicate_iri`), then the [`Catalog`] (name → [`FieldId`]),
//! then [`AbiBatch::column`] (FieldId → lane index). The adapter holds no
//! ClassView of its own, so it cannot disagree with DT64 about which field a
//! position names. Name lookup happens per bind and per check, never per row.
//!
//! ## Admission scope
//!
//! - **Value kinds:** `I32` and `U32` lanes only. A `U64` reference lane is
//!   unbound. Exact decimal and workbook values are not admitted here.
//! - **Reader scope: NOT production-admitted.** DR has no authoritative issuer
//!   of a reader [`Scope`] (it is a plain `{ id }` value), so this adapter
//!   cannot certify which scope a batch serves. Its only constructor,
//!   `AbiLanes::unwitnessed`, is `#[cfg(test)]`: it exists only inside this
//!   crate's own unit tests, and no build of the library (any feature set)
//!   can construct an `AbiLanes`.
//! - **Writer:** none. This is a read adapter over resident lanes; durable
//!   value ingress is out of scope.

use std::sync::Arc;

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev};
use lance_graph_mask_risc::LaneRef;

use crate::batch::{AbiBatch, LaneData};
use crate::boundary::Catalog;
use crate::ids::SourceId;

/// The relation-map revision of every frame this adapter seals. The adapter
/// holds no relation map; the revision is fixed so that only a change of
/// source or generation moves the frame.
pub const RELATION_MAP: u64 = 0;

/// Why a replacement batch was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishRefusal {
    /// The replacement belongs to another source.
    ForeignSource { bound: SourceId, offered: SourceId },
    /// The replacement's generation does not exceed the current one.
    StaleGeneration { current: u32, offered: u32 },
}

/// A resident batch presented to DT64's lane socket.
#[derive(Debug)]
pub struct AbiLanes<'c> {
    catalog: &'c Catalog,
    batch: Arc<AbiBatch>,
    frames: FrameStore,
    scope: Scope,
}

impl<'c> AbiLanes<'c> {
    /// TEST ONLY. The caller asserts which reader scope the batch serves; no
    /// DR authority stands behind it, which is why this exists only in this
    /// crate's unit tests.
    #[cfg(test)]
    pub(crate) fn unwitnessed(catalog: &'c Catalog, batch: Arc<AbiBatch>, scope: Scope) -> Self {
        let frames = FrameStore::new(revisions_of(&batch));
        Self {
            catalog,
            batch,
            frames,
            scope,
        }
    }

    /// The frame owner for this source. Pass it to `BindStore::bind_lanes`
    /// and `LaneBinding::check`.
    pub fn frames(&self) -> &FrameStore {
        &self.frames
    }

    /// The current frame.
    pub fn seal(&self) -> Frame {
        self.frames.seal()
    }

    /// The row order of the current batch.
    pub fn space(&self) -> CoordSpace {
        space_of(&self.batch)
    }

    /// The resident batch.
    pub fn batch(&self) -> &Arc<AbiBatch> {
        &self.batch
    }

    /// Replace the batch with a later generation of the same source and
    /// advance the frame. Every binding sealed before is refused afterwards.
    pub fn publish(&mut self, batch: Arc<AbiBatch>) -> Result<Frame, PublishRefusal> {
        if batch.source() != self.batch.source() {
            return Err(PublishRefusal::ForeignSource {
                bound: self.batch.source(),
                offered: batch.source(),
            });
        }
        if batch.generation() <= self.batch.generation() {
            return Err(PublishRefusal::StaleGeneration {
                current: self.batch.generation(),
                offered: batch.generation(),
            });
        }
        self.frames.advance(revisions_of(&batch));
        self.batch = batch;
        Ok(self.frames.seal())
    }
}

fn revisions_of(batch: &AbiBatch) -> FrameRevisions {
    FrameRevisions {
        source: u64::from(batch.source().0),
        relation_map: RELATION_MAP,
        coordinates: u64::from(batch.generation()),
    }
}

fn space_of(batch: &AbiBatch) -> CoordSpace {
    CoordSpace {
        population: u64::from(batch.source().0),
        coordinates: u64::from(batch.generation()),
    }
}

impl LaneSource for AbiLanes<'_> {
    type Lane<'a>
        = LaneRef<'a>
    where
        Self: 'a;

    fn lane_of(
        &self,
        _class: u32,
        _rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        // Only this source's current frame addresses its lanes.
        self.frames.check(frame).ok()?;
        let id = self.catalog.field(&basis.predicate_iri)?;
        let (lane, column) = self.batch.column(id)?;
        match column.lane {
            LaneData::I32(_) | LaneData::U32(_) => Some(LaneAddr {
                space: space_of(&self.batch),
                lane,
                scope: self.scope,
            }),
            LaneData::U64(_) => None,
        }
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != space_of(&self.batch) || addr.scope != self.scope {
            return None;
        }
        match &self.batch.columns().get(usize::from(addr.lane))?.lane {
            LaneData::I32(v) => Some(LaneRef::I32(v)),
            LaneData::U32(v) => Some(LaneRef::U32(v)),
            LaneData::U64(_) => None,
        }
    }
}

#[cfg(test)]
mod tests;
