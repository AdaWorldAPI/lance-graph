//! A read-only DT64 lane socket over one publication of a [`MailboxSoA`]
//! (D-VAS-A1, D-VAS-A2).
//!
//! - **Bytes** stay the mailbox's. [`MailboxLanes::borrow`] lends the owner's
//!   own arrays as [`LaneRef`]s; nothing is copied, reordered or allocated.
//! - **Frame.** The mailbox has no revision that accounts for its writes
//!   (the field setters and the public columns move neither `current_cycle`
//!   nor `last_write_cycle`). What it has is [`Published`], the shared borrow
//!   issued by the owner, during which no column can change. This socket
//!   holds ONE [`FrameStore`] per `Published`, so a frame, a binding or a
//!   lane address is valid for that publication only and is `ForeignFrame`
//!   under any other, including a later publication of the same mailbox.
//!   Nothing here survives the borrow, so nothing here is replayable.
//! - **Which field is which column.** No contract names the mailbox's
//!   columns. [`Column`] names the ones this socket lends, by a predicate
//!   IRI of its own (`urn:mailbox:<column>`). DT64 resolves the class field
//!   through the `ClassView` it validated and passes that field's
//!   `predicate_iri`; the socket maps the IRI and never a position.
//! - **Kinds.** `LaneRef` has `u32` and `u64` lanes, so `meta`,
//!   `last_active_cycle`, `last_write_cycle`, `edges` and `temporal` are
//!   lent. `energy` (`f32`), `entity_type` and `expert` (`u16`), `sigma` and
//!   `plasticity_counter` (`u8`), `qualia`, the style lanes and the identity
//!   planes are refused by name ([`ColumnRefusal::UnsupportedKind`]).
//! - **Rows.** A lane is the populated prefix (`MailboxSoA::populated`).
//!
//! Not provided: a reader-[`Scope`] issuer (DR has none; the scope here is
//! the caller's assertion, compared for consistency only), selections,
//! `NodeGuid` resolution, any write.

use cognitive_shader_driver::mailbox_soa::Published;
use lance_graph_contract::ontology::FieldRef;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev};
use lance_graph_mask_risc::LaneRef;

/// A mailbox column this socket lends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Column {
    /// `meta`, one packed `u32` per row.
    Meta = 0,
    /// `last_active_cycle`, `u32`.
    LastActiveCycle = 1,
    /// `last_write_cycle`, `u32`.
    LastWriteCycle = 2,
    /// `edges`, one `u64` per row.
    Edges = 3,
    /// `temporal`, `u64`.
    Temporal = 4,
}

/// Why a predicate names no lendable column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRefusal {
    /// Not a column of the mailbox.
    Unknown,
    /// A column of the mailbox whose element kind `LaneRef` cannot lend.
    UnsupportedKind(&'static str),
}

impl Column {
    /// Every lendable column, in lane-number order.
    pub const ALL: [Column; 5] = [
        Column::Meta,
        Column::LastActiveCycle,
        Column::LastWriteCycle,
        Column::Edges,
        Column::Temporal,
    ];

    /// The predicate IRI this socket reads the column under.
    #[must_use]
    pub const fn iri(self) -> &'static str {
        match self {
            Column::Meta => "urn:mailbox:meta",
            Column::LastActiveCycle => "urn:mailbox:last_active_cycle",
            Column::LastWriteCycle => "urn:mailbox:last_write_cycle",
            Column::Edges => "urn:mailbox:edges",
            Column::Temporal => "urn:mailbox:temporal",
        }
    }

    /// The column a predicate IRI names.
    pub fn from_iri(iri: &str) -> Result<Column, ColumnRefusal> {
        if let Some(c) = Column::ALL.into_iter().find(|c| c.iri() == iri) {
            return Ok(c);
        }
        let kind = match iri {
            "urn:mailbox:energy" => "f32",
            "urn:mailbox:entity_type" | "urn:mailbox:expert" => "u16",
            "urn:mailbox:sigma" | "urn:mailbox:plasticity_counter" => "u8",
            "urn:mailbox:qualia" => "QualiaI4_16D",
            "urn:mailbox:frozen_style"
            | "urn:mailbox:learned_style"
            | "urn:mailbox:explore_style" => "[u8; 12]",
            "urn:mailbox:content" | "urn:mailbox:topic" | "urn:mailbox:angle" => "identity plane",
            _ => return Err(ColumnRefusal::Unknown),
        };
        Err(ColumnRefusal::UnsupportedKind(kind))
    }

    fn from_lane(lane: u16) -> Option<Column> {
        Column::ALL.into_iter().find(|c| *c as u16 == lane)
    }
}

/// One publication of a mailbox, presented to DT64's lane socket.
pub struct MailboxLanes<'a, const N: usize> {
    published: Published<'a, N>,
    frames: FrameStore,
    scope: Scope,
}

impl<'a, const N: usize> MailboxLanes<'a, N> {
    /// A socket over `published`, with a frame store of its own. `scope` is
    /// the reader scope the caller asserts; nothing authorizes it.
    #[must_use]
    pub fn new(published: Published<'a, N>, scope: Scope) -> Self {
        let frames = FrameStore::new(FrameRevisions {
            source: u64::from(published.owner().mailbox_id()),
            relation_map: 0,
            coordinates: 0,
        });
        Self {
            published,
            frames,
            scope,
        }
    }

    /// The frame store of this publication. Same store as
    /// [`LaneSource::frames`].
    #[must_use]
    pub fn frame_store(&self) -> &FrameStore {
        &self.frames
    }

    /// The current frame of this publication.
    #[must_use]
    pub fn seal(&self) -> Frame {
        self.frames.seal()
    }

    /// The mailbox's own row order: its slot order, for this mailbox.
    #[must_use]
    pub fn space(&self) -> CoordSpace {
        CoordSpace {
            population: u64::from(self.published.owner().mailbox_id()),
            coordinates: 0,
        }
    }

    /// Rows a lane holds: the populated prefix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.published.owner().populated()
    }
}

impl<const N: usize> LaneSource for MailboxLanes<'_, N> {
    type Lane<'s>
        = LaneRef<'s>
    where
        Self: 's;

    fn lane_of(
        &self,
        _class: u32,
        _rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        let column = Column::from_iri(&basis.predicate_iri).ok()?;
        Some(LaneAddr {
            space: self.space(),
            lane: column as u16,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != self.space() || addr.scope != self.scope {
            return None;
        }
        let owner = self.published.owner();
        let rows = owner.populated();
        Some(match Column::from_lane(addr.lane)? {
            Column::Meta => LaneRef::U32(owner.meta_raw().get(..rows)?),
            Column::LastActiveCycle => LaneRef::U32(owner.last_active_cycle.get(..rows)?),
            Column::LastWriteCycle => LaneRef::U32(owner.last_write_cycle.get(..rows)?),
            Column::Edges => LaneRef::U64(owner.edges_raw().get(..rows)?),
            Column::Temporal => LaneRef::U64(owner.temporal.get(..rows)?),
        })
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}
