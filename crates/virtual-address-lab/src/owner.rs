//! P0: one demanded property, read from a mailbox owner through its existing
//! borrow contract ([`MailboxSoaView`]) and nothing else.
//!
//! Three things are kept apart:
//!
//! - **class interpretation**: the class's field basis names a property by
//!   its predicate ([`BASIS`]); the recipe chain says which basis position a
//!   view property is;
//! - **physical addressing**: [`column_of`] maps a predicate to the column the
//!   owner lends for it;
//! - **the read**: one indexed load from that borrowed column.
//!
//! This module holds no frame, no selection and no witness. The identity is a
//! source-local row position supplied by the caller: the owner resolves no
//! `NodeGuid` (`row_for_local_key` answers `None`), and nothing here pretends
//! otherwise.

use std::cell::Cell;

use lance_graph_contract::soa_view::MailboxSoaView;

use crate::recipe::Recipe;
use crate::view::{field_in, Refused};

/// The class basis: predicate and label per field position.
pub const BASIS: [(&str, &str); 2] = [("urn:mailbox/meta", "meta"), ("urn:mailbox/edge", "edge")];

/// A column the owner lends as a typed slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    /// `MailboxSoaView::meta_raw`, `&[u32]`.
    Meta,
    /// `MailboxSoaView::edges_raw`, `&[u64]`.
    Edge,
}

/// Physical addressing: the column the owner lends for a predicate.
pub fn column_of(predicate: &str) -> Option<Column> {
    match predicate {
        "urn:mailbox/meta" => Some(Column::Meta),
        "urn:mailbox/edge" => Some(Column::Edge),
        _ => None,
    }
}

/// One demanded value, as the owner stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    U32(u32),
    U64(u64),
}

/// What the reads did.
#[derive(Debug, Default)]
pub struct Reads {
    /// Borrows of the `meta` column.
    pub meta_borrows: Cell<u64>,
    /// Borrows of the `edge` column.
    pub edge_borrows: Cell<u64>,
    /// Values loaded.
    pub values: Cell<u64>,
}

/// The virtual property: a source-local row, a view property and the recipe
/// chain it is seen through. Describes where to read; holds no value.
#[derive(Debug, Clone, Copy)]
pub struct VirtualProperty<'c> {
    pub row: usize,
    pub property: usize,
    pub chain: &'c [Recipe],
}

impl VirtualProperty<'_> {
    /// The column this property resolves to. Reads nothing.
    pub fn column(&self, rows: usize) -> Result<Column, Refused> {
        let field = field_in(self.chain, BASIS.len(), rows, self.property)?;
        column_of(BASIS[usize::from(field)].0).ok_or(Refused::OutsideTheView)
    }

    /// Resolve and read: the chain is normalised, the basis position names a
    /// predicate, the predicate names a column, and ONE value is loaded from
    /// the slice the owner lends. A row that is not populated is refused; the
    /// zero padding of the owner's capacity rows is never returned.
    pub fn read<V: MailboxSoaView>(&self, owner: &V, reads: &Reads) -> Result<Value, Refused> {
        let rows = owner.n_rows();
        let column = self.column(rows)?;
        if self.row >= rows {
            return Err(Refused::PositionOutsideLane {
                at: 0,
                position: self.row as u32,
            });
        }
        let value = match column {
            Column::Meta => {
                reads.meta_borrows.set(reads.meta_borrows.get() + 1);
                Value::U32(owner.meta_raw()[self.row])
            }
            Column::Edge => {
                reads.edge_borrows.set(reads.edge_borrows.get() + 1);
                Value::U64(owner.edges_raw()[self.row])
            }
        };
        reads.values.set(reads.values.get() + 1);
        Ok(value)
    }
}
