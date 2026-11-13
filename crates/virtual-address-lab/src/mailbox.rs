//! The read-only socket between a mailbox owner and DT64's lane contract.
//!
//! **This is the seam the production code does not have**, built here only so
//! the address law can be shown on a real owner. It follows
//! `lance-graph-report`'s `AbiLanes`: it holds a shared borrow of the owner
//! and no value byte; `borrow` hands out the owner's own column.
//!
//! Because it borrows the owner, the owner cannot be mutated while a socket
//! lives. A socket made after a mutation has another `FrameStore`, so every
//! binding sealed under the earlier one is refused.
//!
//! It adds no authority: the frame revisions are read from the owner
//! (`mailbox_id`, `current_cycle`), the row order is the owner's, and the
//! `Scope` is a plain consistency value.

use std::cell::Cell;

use dr256_ternlog_lab::order_store::{KeyLane, SealedOrder};
use dr256_ternlog_lab::{OrderIndex, Table};
use lance_graph_contract::ontology::FieldRef;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore};
use lance_graph_dt64::{
    BindStore, CoordSpace, LaneAddr, LaneBinding, LaneSource, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::LaneRef;

use crate::grid::{View, CLASS, READER};
use crate::recipe::Recipe;
use crate::view::Refused;

/// The class basis the socket serves, in field order.
pub const FIELDS: [(&str, &str); 2] = [("urn:mailbox/meta", "meta"), ("urn:mailbox/edge", "edge")];
/// Field position of the `meta` column (`u32`).
pub const META: u8 = 0;
/// Field position of the `edge` column (`u64`).
pub const EDGE: u8 = 1;
/// The population number of an order index's own row order.
const INDEX_POPULATION: u64 = 0x1D;
/// The tag of the computed order key (the `meta` value).
const KEY_META: u64 = 0x4D45_5441;

enum Positions {
    Order(SealedOrder),
    List(Vec<u32>),
}

struct Issued {
    positions: Positions,
    space: CoordSpace,
}

/// What the owner's frame is derived from.
pub fn revisions_of<V: MailboxSoaView>(owner: &V) -> FrameRevisions {
    FrameRevisions {
        source: u64::from(owner.mailbox_id()),
        relation_map: 0,
        coordinates: u64::from(owner.current_cycle()),
    }
}

/// A mailbox owner presented to DT64's lane and selection sockets.
pub struct MailboxLanes<'m, V: MailboxSoaView> {
    owner: &'m V,
    frames: FrameStore,
    space: CoordSpace,
    selections: Vec<Issued>,
    /// TEST SEAM: serve the two columns under each other's field.
    swapped: Cell<bool>,
    /// How often each column was lent (`[meta, edge]`).
    borrows: [Cell<u64>; 2],
}

impl<'m, V: MailboxSoaView> MailboxLanes<'m, V> {
    pub fn new(owner: &'m V) -> Self {
        MailboxLanes {
            owner,
            frames: FrameStore::new(revisions_of(owner)),
            space: CoordSpace {
                population: u64::from(owner.mailbox_id()),
                coordinates: u64::from(owner.current_cycle()),
            },
            selections: Vec::new(),
            swapped: Cell::new(false),
            borrows: [Cell::new(0), Cell::new(0)],
        }
    }

    /// Populated rows of the owner; the lanes lent span exactly these.
    pub fn rows(&self) -> usize {
        self.owner.n_rows()
    }

    pub fn space(&self) -> CoordSpace {
        self.space
    }

    /// How often the `meta` and the `edge` column were lent.
    pub fn borrows(&self) -> [u64; 2] {
        [self.borrows[0].get(), self.borrows[1].get()]
    }

    /// Serve the two columns under each other's field from now on.
    pub fn swap_columns(&self) {
        self.swapped.set(!self.swapped.get());
    }

    fn issue(&mut self, positions: Positions, space: CoordSpace) -> u16 {
        self.selections.push(Issued { positions, space });
        (self.selections.len() - 1) as u16
    }

    /// The order of the populated rows by their `meta` value, ties by row:
    /// `OrderIndex::by_key`, sealed. The index is held here because the owner
    /// has no home for one. `None` if a key does not fit `i32`.
    pub fn issue_order_by_meta(&mut self) -> Option<u16> {
        let meta = &self.owner.meta_raw()[..self.rows()];
        if meta.iter().any(|&m| i32::try_from(m).is_err()) {
            return None;
        }
        let table = Table::new(
            CLASS,
            self.space.population,
            self.rows(),
            self.frames.revisions().source,
            READER,
            &[],
        );
        let index = OrderIndex::by_key(&table, INDEX_POPULATION, |row| meta[row] as i32)?;
        let sealed = SealedOrder::seal(index, &table, KeyLane::Computed(KEY_META), 0).ok()?;
        Some(self.issue(Positions::Order(sealed), self.space))
    }

    /// The source issues a list as it is: a bag, in list order.
    pub fn issue_list(&mut self, rows: Vec<u32>) -> u16 {
        self.issue(Positions::List(rows), self.space)
    }

    /// The source issues a list whose positions are in another row order.
    pub fn issue_list_in(&mut self, rows: Vec<u32>, space: CoordSpace) -> u16 {
        self.issue(Positions::List(rows), space)
    }

    /// The positions of `sel` with no witness, to compare addresses.
    pub fn raw_positions(&self, sel: u16) -> &[u32] {
        match &self.selections[usize::from(sel)].positions {
            Positions::Order(sealed) => sealed.rows(),
            Positions::List(rows) => rows,
        }
    }
}

impl<V: MailboxSoaView> LaneSource for MailboxLanes<'_, V> {
    type Lane<'a>
        = LaneRef<'a>
    where
        Self: 'a;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        if class != CLASS || rev != SchemaRev(1) {
            return None;
        }
        let column = FIELDS
            .iter()
            .position(|(iri, _)| *iri == basis.predicate_iri.as_str())?;
        let lane = if self.swapped.get() {
            1 - column
        } else {
            column
        };
        Some(LaneAddr {
            space: self.space,
            lane: lane as u16,
            scope: READER,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != self.space {
            return None;
        }
        let n = self.rows();
        let lane = usize::from(addr.lane);
        let lent = match lane {
            0 => LaneRef::U32(&self.owner.meta_raw()[..n]),
            1 => LaneRef::U64(&self.owner.edges_raw()[..n]),
            _ => return None,
        };
        self.borrows[lane].set(self.borrows[lane].get() + 1);
        Some(lent)
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl<V: MailboxSoaView> SelectionSource for MailboxLanes<'_, V> {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        let issued = self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: issued.space,
            sel,
            scope: READER,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        let issued = self.selections.get(usize::from(addr.sel))?;
        if issued.space != addr.space {
            return None;
        }
        Some(match &issued.positions {
            Positions::Order(sealed) => sealed.rows(),
            Positions::List(rows) => rows,
        })
    }
}

/// One demanded value, as the owner stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell64 {
    U32(u32),
    U64(u64),
}

/// Visit view property `property` (seen through `chain`) of every identity of
/// the source-issued selection `sel`, in the selection's order.
///
/// The chain is normalised first and only chooses the field. The lane and the
/// positions come from ONE `LaneBinding::selected`; each visited value is one
/// position-indexed load from the owner's column. Nothing is staged. A
/// position that is not a populated row stops the visit with a refusal.
/// Returns how many values were read.
#[allow(clippy::too_many_arguments)]
pub fn visit_selected<'s, S>(
    store: &BindStore,
    view: &View,
    binding: &LaneBinding,
    source: &'s S,
    rows: usize,
    chain: &[Recipe],
    property: usize,
    sel: u16,
    mut visit: impl FnMut(u32, Cell64),
) -> Result<usize, Refused>
where
    S: SelectionSource<Lane<'s> = LaneRef<'s>>,
{
    let field = crate::view::field_in(chain, FIELDS.len(), rows, property)?;
    let (lane, positions) = binding
        .selected(store, view, source.frames(), source, field, sel)
        .map_err(Refused::Lane)?;
    for (at, &p) in positions.iter().enumerate() {
        let i = p as usize;
        let cell = match lane {
            LaneRef::U32(v) => v.get(i).copied().map(Cell64::U32),
            LaneRef::U64(v) => v.get(i).copied().map(Cell64::U64),
            _ => return Err(Refused::LaneKind),
        };
        match cell {
            Some(c) => visit(p, c),
            None => return Err(Refused::PositionOutsideLane { at, position: p }),
        }
    }
    Ok(positions.len())
}
