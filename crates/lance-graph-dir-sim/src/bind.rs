//! The developer world for IAM queries: a field name and a text literal,
//! bound once through the generic Quack frontend (`lance_graph_quack::bind`).
//!
//! ```text
//!   where_eq(view, "smtp", "Alice@X.de")
//!      │  table("users").where_eq(..).bind(&UserBinder)   one key lookup
//!      ▼
//!   Query  ──lower──►  key_eq_program(key)   (the same Program, test-pinned)
//!      │
//!      ▼
//!   users_with_key's executor: base lane, overrides, created rows
//! ```
//!
//! This module is the only place in the crate where a query holds text
//! (`tests/string_fence.rs` keeps the execution modules text-free). A field
//! binds by its **comparison key** (`KeyId`): `ALICE@x.DE` and `alice@x.de`
//! are one value, as the directory treats them.

use std::cell::Cell;

use lance_graph_quack::bind::{table, BindError, Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Col, LowerError, Mask};
use ogar_dir_sim::{Attribute, KeyId};

use crate::{users_matching, Dicts, Kept, View};

/// The user table's catalog and codebook over a store's dictionaries.
///
/// Fields `upn` and `smtp` bind to the attribute's key lane (`Col(0)`, the
/// lane [`users_with_key`](crate::users_with_key) supplies) and resolve a
/// literal with [`Dicts::key_lookup`]. It mints nothing. It records the
/// attribute and key it issued, because the executor needs them to merge a
/// version's overrides and created users.
pub struct UserBinder<'a> {
    dicts: &'a Dicts,
    field: Cell<Option<Attribute>>,
    issued: Cell<Option<(Attribute, KeyId)>>,
}

impl<'a> UserBinder<'a> {
    /// A binder over `dicts`.
    #[must_use]
    pub fn new(dicts: &'a Dicts) -> Self {
        Self {
            dicts,
            field: Cell::new(None),
            issued: Cell::new(None),
        }
    }

    /// The `(attribute, key)` the last bind resolved, if any.
    #[must_use]
    pub fn issued(&self) -> Option<(Attribute, KeyId)> {
        self.issued.get()
    }
}

impl Binder for UserBinder<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "users").then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        let a = match name {
            "upn" => Attribute::Upn,
            "smtp" => Attribute::PrimarySmtp,
            _ => return None,
        };
        self.field.set(Some(a));
        Some(BoundField {
            col: Col(0),
            kind: FieldKind::Code,
            // Dictionary ids are never NULL; an absent attribute is the NONE
            // id, which no literal resolves to.
            validity: None,
        })
    }
    fn code(&self, _: TableId, _: Col, literal: &str) -> Option<u32> {
        let a = self.field.get()?;
        let k = self.dicts.key_lookup(literal)?;
        self.issued.set(Some((a, k)));
        Some(k.0)
    }
}

/// Why [`where_eq`] could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhereEqError {
    /// The field or literal did not bind (unknown field, unknown value, …).
    Bind(BindError),
    /// The bound query did not lower.
    Lower(LowerError),
}

/// `SELECT users WHERE <field> = <literal>` over a version: bound once, then
/// executed numerically by the same executor as
/// [`users_with_key`](crate::users_with_key).
///
/// # Errors
///
/// [`WhereEqError::Bind`] when the field is not `upn`/`smtp` or no observed
/// value has the literal's key; [`WhereEqError::Lower`] if lowering fails.
pub fn where_eq(v: &View<'_>, field: &str, literal: &str) -> Result<Kept, WhereEqError> {
    let b = UserBinder::new(v.dicts);
    let q = table("users")
        .where_eq(field, literal)
        .bind(&b)
        .map_err(WhereEqError::Bind)?;
    let p = lower(&q).map_err(WhereEqError::Lower)?;
    // A successful bind of a text literal against a coded field went
    // through `code`, which records what it issued.
    let Some((a, key)) = b.issued() else {
        return Err(WhereEqError::Bind(BindError::UnknownField {
            table: "users".into(),
            field: field.into(),
        }));
    };
    Ok(users_matching(v, a, key, &p))
}
