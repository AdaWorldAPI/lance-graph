//! The class-qualified call, as a caller's `sql(query)` carries it.
//!
//! No SQL frontend reaches Quack today: `lance-graph`'s SQL path is
//! DataFusion, which is in its grace period and gains nothing new. This
//! module recognises exactly ONE form and refuses everything else:
//!
//! ```text
//!   SELECT <concept>::<method>(<table>) FROM <table>
//! ```
//!
//! It is the seam a real SQL frontend hands its parsed call to, not a SQL
//! parser. The text is resolved once, cold, into a typed [`MethodCall`]:
//!
//! - `<concept>` through OGAR's codebook (`canonical_concept_id`), never a
//!   local name table;
//! - `<method>` through the [`MethodTable`]'s labels, into a [`MethodKey`];
//! - the payload names the row the method reads, which must be the `FROM`
//!   table, which must be the table the methods were bound against.
//!
//! Nothing is substituted back into text: the call carries numbers only.
use lance_graph_ogar::ogar_vocab::canonical_concept_id;

use crate::method::{MethodKey, MethodTable};

/// A resolved class-qualified call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodCall {
    /// Canonical concept id.
    pub concept: u16,
    pub method: MethodKey,
}

/// Why a call text did not resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallParseError {
    /// Not of the one recognised form.
    Unrecognised,
    /// OGAR's codebook has no concept of that name.
    UnknownConcept(String),
    /// The concept is not the one whose methods the table holds.
    WrongConcept { called: u16, table: u16 },
    /// The concept has no method of that name.
    UnknownMethod(String),
    /// The payload is not the `FROM` table.
    PayloadMismatch,
    /// The `FROM` table is not the one the methods were bound against.
    WrongTable(String),
}

/// Resolve `SELECT <concept>::<method>(<table>) FROM <table>` against `t`.
///
/// # Errors
///
/// [`CallParseError`] naming the first part that did not resolve.
pub fn resolve_call(sql: &str, t: &MethodTable) -> Result<MethodCall, CallParseError> {
    let words: Vec<&str> = sql.split_whitespace().collect();
    let [select, call, from, table] = words.as_slice() else {
        return Err(CallParseError::Unrecognised);
    };
    if !select.eq_ignore_ascii_case("select") || !from.eq_ignore_ascii_case("from") {
        return Err(CallParseError::Unrecognised);
    }
    let (concept, rest) = call.split_once("::").ok_or(CallParseError::Unrecognised)?;
    let (method, payload) = rest
        .strip_suffix(')')
        .and_then(|r| r.split_once('('))
        .ok_or(CallParseError::Unrecognised)?;
    let ident =
        |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if ![concept, method, payload, table].into_iter().all(ident) {
        return Err(CallParseError::Unrecognised);
    }
    let id = canonical_concept_id(concept)
        .ok_or_else(|| CallParseError::UnknownConcept(concept.to_owned()))?;
    if id != t.concept() {
        return Err(CallParseError::WrongConcept {
            called: id,
            table: t.concept(),
        });
    }
    if payload != *table {
        return Err(CallParseError::PayloadMismatch);
    }
    if *table != t.table() {
        return Err(CallParseError::WrongTable((*table).to_owned()));
    }
    let method = t
        .resolve(method)
        .ok_or_else(|| CallParseError::UnknownMethod(method.to_owned()))?;
    Ok(MethodCall {
        concept: id,
        method,
    })
}
