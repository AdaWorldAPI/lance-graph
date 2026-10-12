//! Cross-glove parity probe (D-XGP-1, `.claude/plans/cross-glove-business-parity-v1.md` §C).
//!
//! Two shipped interfaces meet here, and nothing else is added:
//!
//! - semantic identity: `OntologyRegistry` maps `(bridge_id, public_name)` to
//!   an OGIT URI, and one URI is one `entity_type_id` across bridges;
//! - executable binding: Quack's `Binder` maps a field name to a lane.
//!
//! [`native_field`] answers "which native field of this glove means this
//! canonical attribute"; [`CanonicalBinder`] puts that answer in front of a
//! glove's own `Binder`, so a `Draft` written in canonical names binds to the
//! glove's lanes. This is a FIELD MAPPING (bind time, fields → lanes), not a
//! report `pivot()` (plan §C.6.1).
pub mod basis;

use lance_graph_ontology::namespace::SchemaKind;
use lance_graph_ontology::OntologyRegistry;
use lance_graph_quack::bind::{Binder, BoundField, TableId};
use lance_graph_quack::{Col, Mask};

/// The glove's native name for `canonical_uri`, from its own bridge's row.
///
/// `None` unless a row exists for `bridge` that is an `Attribute` carrying
/// that URI's identity with `confidence == 1.0`. A hypothesized row
/// (`confidence < 1.0`) is kept in the registry and never bound (plan §C.6.2).
///
/// `MappingRow::active` is not consulted: the registry has no path that sets
/// it false today, so a guard on it could not be tested. Add it together with
/// the first deactivation path and a test that exercises it.
#[must_use]
pub fn native_field(reg: &OntologyRegistry, bridge: &str, canonical_uri: &str) -> Option<String> {
    let id = reg.resolve_uri(canonical_uri)?.entity_type_id();
    reg.rows_with_entity_type(id)
        .into_iter()
        .find(|r| r.bridge_id == bridge && r.kind == SchemaKind::Attribute && r.confidence >= 1.0)
        .map(|r| r.public_name)
}

/// A glove's `Binder` addressed in canonical names.
pub struct CanonicalBinder<'a> {
    reg: &'a OntologyRegistry,
    bridge: &'a str,
    concept: &'a str,
    native_table: &'a str,
    inner: &'a dyn Binder,
}

impl<'a> CanonicalBinder<'a> {
    /// `concept` is the canonical table name a `Draft` uses; `native_table` is
    /// the glove's own table name, handed to `inner`.
    #[must_use]
    pub fn new(
        reg: &'a OntologyRegistry,
        bridge: &'a str,
        concept: &'a str,
        native_table: &'a str,
        inner: &'a dyn Binder,
    ) -> Self {
        Self {
            reg,
            bridge,
            concept,
            native_table,
            inner,
        }
    }
}

impl Binder for CanonicalBinder<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        if name == self.concept {
            self.inner.table(self.native_table)
        } else {
            None
        }
    }
    fn live(&self, table: TableId) -> Mask {
        self.inner.live(table)
    }
    fn field(&self, table: TableId, name: &str) -> Option<BoundField> {
        let native = native_field(self.reg, self.bridge, name)?;
        self.inner.field(table, &native)
    }
    fn code(&self, table: TableId, field: Col, literal: &str) -> Option<u32> {
        self.inner.code(table, field, literal)
    }
}
