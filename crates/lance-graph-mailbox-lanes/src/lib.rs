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
//! - **Selections (D-VAS-A3).** The socket is also a [`SelectionSource`]. It
//!   issues position lists itself, from the published columns: an order by a
//!   column ([`MailboxLanes::issue_order`]), a filter by a column comparison
//!   ([`MailboxLanes::issue_where`]) and the concatenation of two issued
//!   selections ([`MailboxLanes::issue_concat`], the only way a bag with
//!   repeats arises). A caller cannot hand in positions. The lists are views
//!   of this publication: they live in the socket, in the mailbox's row
//!   order, and end with it. The mailbox itself has no home for an order.
//!
//! Not provided: a reader-[`Scope`] issuer (DR has none; the scope here is
//! the caller's assertion, compared for consistency only), `NodeGuid`
//! resolution, any write.

use cognitive_shader_driver::mailbox_soa::Published;
use lance_graph_contract::ontology::FieldRef;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev, SelAddr, SelectionSource};
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

/// How a column value is compared when a selection filters by it. Values
/// are compared as unsigned integers of the column's own width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    Eq(u64),
    Ne(u64),
    Lt(u64),
    Ge(u64),
}

impl Keep {
    fn holds(self, v: u64) -> bool {
        match self {
            Keep::Eq(t) => v == t,
            Keep::Ne(t) => v != t,
            Keep::Lt(t) => v < t,
            Keep::Ge(t) => v >= t,
        }
    }
}

/// Why the socket issued no selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionRefusal {
    /// All 65,536 selection names of this publication are taken.
    NamesExhausted,
    /// The named selection was not issued by this socket.
    Unknown { sel: u16 },
}

/// One publication of a mailbox, presented to DT64's lane socket.
pub struct MailboxLanes<'a, const N: usize> {
    published: Published<'a, N>,
    frames: FrameStore,
    scope: Scope,
    /// Position lists this socket issued, by selection name.
    selections: Vec<Vec<u32>>,
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
            selections: Vec::new(),
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

    /// The value of `column` at `row`, widened to `u64`.
    fn value(&self, column: Column, row: usize) -> u64 {
        let owner = self.published.owner();
        match column {
            Column::Meta => u64::from(owner.meta_raw()[row]),
            Column::LastActiveCycle => u64::from(owner.last_active_cycle[row]),
            Column::LastWriteCycle => u64::from(owner.last_write_cycle[row]),
            Column::Edges => owner.edges_raw()[row],
            Column::Temporal => owner.temporal[row],
        }
    }

    fn issue(&mut self, positions: Vec<u32>) -> Result<u16, SelectionRefusal> {
        let name =
            u16::try_from(self.selections.len()).map_err(|_| SelectionRefusal::NamesExhausted)?;
        self.selections.push(positions);
        Ok(name)
    }

    /// Issue the populated rows ordered by `column`, ascending; rows with
    /// equal values keep their row order.
    pub fn issue_order(&mut self, column: Column) -> Result<u16, SelectionRefusal> {
        let mut rows: Vec<u32> = (0..self.rows() as u32).collect();
        rows.sort_by_key(|&r| self.value(column, r as usize));
        self.issue(rows)
    }

    /// Issue the populated rows where `column` satisfies `keep`, in row
    /// order.
    pub fn issue_where(&mut self, column: Column, keep: Keep) -> Result<u16, SelectionRefusal> {
        let rows: Vec<u32> = (0..self.rows() as u32)
            .filter(|&r| keep.holds(self.value(column, r as usize)))
            .collect();
        self.issue(rows)
    }

    /// Issue `first` followed by `second`: a bag in list order. A row both
    /// hold appears once per occurrence.
    pub fn issue_concat(&mut self, first: u16, second: u16) -> Result<u16, SelectionRefusal> {
        let list = |sel: u16| {
            self.selections
                .get(usize::from(sel))
                .ok_or(SelectionRefusal::Unknown { sel })
        };
        let mut rows = list(first)?.clone();
        rows.extend_from_slice(list(second)?);
        self.issue(rows)
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

impl<const N: usize> SelectionSource for MailboxLanes<'_, N> {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: self.space(),
            sel,
            scope: self.scope,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        if addr.space != self.space() || addr.scope != self.scope {
            return None;
        }
        self.selections
            .get(usize::from(addr.sel))
            .map(Vec::as_slice)
    }
}
