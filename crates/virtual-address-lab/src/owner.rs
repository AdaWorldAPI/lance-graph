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

/// A LAB-LOCAL predicate table, in field order. It is not a `ClassView`
/// binding: no class registry resolved it, so a read through it shows the
/// mechanics of "basis position -> predicate -> column", not a class-bound
/// interpretation.
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

/// The virtual property: a source-local row ORDINAL (an address, not an
/// identity), a view property and the recipe chain it is seen through.
/// Describes where to read; holds no value and NO source version, so it is
/// not version-bound.
#[derive(Debug, Clone, Copy)]
pub struct VirtualProperty<'c> {
    pub row: usize,
    pub property: usize,
    pub chain: &'c [Recipe],
}

/// Where a virtual property lives in the owner: `f(x)`. A coordinate, not a
/// permission and not a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    pub column: Column,
    /// Source-local position within that column.
    pub position: usize,
}

impl VirtualProperty<'_> {
    /// The address function: the chain is normalised and view cell
    /// `(row, property)` is mapped to a column and a source-local position.
    /// Pure: reads nothing and decides nothing about whether a read is
    /// allowed. Refuses only a cell outside the view and a property that is
    /// not one field.
    pub fn address(&self, rows: usize) -> Result<Address, Refused> {
        let field = field_in(self.chain, BASIS.len(), rows, self.property)?;
        let (_, position) = Recipe::normalize(self.chain)
            .resolve(BASIS.len(), rows, self.row, self.property)
            .ok_or(Refused::PositionOutsideLane {
                at: 0,
                position: self.row as u32,
            })?;
        let column = column_of(BASIS[usize::from(field)].0).ok_or(Refused::OutsideTheView)?;
        Ok(Address { column, position })
    }

    /// Admission, separate from the address function. The owner has issued no
    /// coordinate or selection witness, so P0 admits only a recipe that
    /// leaves source positions in place; one that reverses them is refused
    /// whatever the row.
    pub fn admit(&self, rows: usize) -> Result<Address, Refused> {
        let address = self.address(rows)?;
        let recipe = Recipe::normalize(self.chain);
        if recipe.reverses_positions() {
            return Err(Refused::RowCoordinatesChanged { recipe });
        }
        Ok(address)
    }

    /// The column an admitted read borrows. Reads nothing.
    pub fn column(&self, rows: usize) -> Result<Column, Refused> {
        self.admit(rows).map(|a| a.column)
    }

    /// `S[f(x)]` for an admitted address: ONE value is loaded from the slice
    /// the owner lends, at the position the address function computed. A row
    /// that is not populated is outside the view; the zero padding of the
    /// owner's capacity rows is never returned.
    pub fn read<V: MailboxSoaView>(&self, owner: &V, reads: &Reads) -> Result<Value, Refused> {
        let Address { column, position } = self.admit(owner.n_rows())?;
        let value = match column {
            Column::Meta => {
                reads.meta_borrows.set(reads.meta_borrows.get() + 1);
                Value::U32(owner.meta_raw()[position])
            }
            Column::Edge => {
                reads.edge_borrows.set(reads.edge_borrows.get() + 1);
                Value::U64(owner.edges_raw()[position])
            }
        };
        reads.values.set(reads.values.get() + 1);
        Ok(value)
    }
}
