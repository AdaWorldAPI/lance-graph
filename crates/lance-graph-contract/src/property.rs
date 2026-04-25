//! Property specifications for graph entities.
//!
//! Defines the shape, optionality, and data-classification of properties
//! that attach to vertices and edges. Outside the BBB this is a boring
//! schema layer; inside it feeds the cognitive shader's metadata columns.

// ═══════════════════════════════════════════════════════════════════════════
// PROPERTY KIND
// ═══════════════════════════════════════════════════════════════════════════

/// The scalar kind of a property value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    Bool,
    I64,
    F64,
    String,
    Bytes,
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA CLASSIFICATION (GDPR)
// ═══════════════════════════════════════════════════════════════════════════

/// Data classification marking for GDPR compliance.
/// Determines retention policy, access audit requirements, and
/// cross-border transfer restrictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marking {
    Public,
    Internal,
    Pii,
    Financial,
    Restricted,
}

impl Default for Marking {
    fn default() -> Self { Marking::Internal }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROPERTY SPEC
// ═══════════════════════════════════════════════════════════════════════════

/// Specification for a single property on a vertex or edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropertySpec {
    pub name: &'static str,
    pub kind: PropertyKind,
    pub required: bool,
    pub default_value: Option<&'static str>,
    pub marking: Marking,
}

impl PropertySpec {
    /// A required property (must be present on every entity).
    pub const fn required(name: &'static str, kind: PropertyKind) -> Self {
        Self {
            name,
            kind,
            required: true,
            default_value: None,
            marking: Marking::Internal,
        }
    }

    /// An optional property with a default value.
    pub const fn optional(name: &'static str, kind: PropertyKind, default_value: &'static str) -> Self {
        Self {
            name,
            kind,
            required: false,
            default_value: Some(default_value),
            marking: Marking::Internal,
        }
    }

    /// A free-form property (optional, no default).
    pub const fn free(name: &'static str, kind: PropertyKind) -> Self {
        Self {
            name,
            kind,
            required: false,
            default_value: None,
            marking: Marking::Internal,
        }
    }

    /// Set the data-classification marking.
    pub const fn with_marking(mut self, marking: Marking) -> Self {
        self.marking = marking;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LINEAGE HANDLE
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to an entity's lineage chain.
/// Tracks who created/modified what, when, and from which source.
/// Outside the BBB this is a boring audit trail; inside it feeds
/// CausalEdge64 provenance bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageHandle {
    pub entity_type: &'static str,
    pub entity_id: u64,
    pub version: u64,
    pub source_system: &'static str,
    pub timestamp_ms: u64,
}

impl LineageHandle {
    pub const fn new(
        entity_type: &'static str,
        entity_id: u64,
        version: u64,
        source_system: &'static str,
        timestamp_ms: u64,
    ) -> Self {
        Self { entity_type, entity_id, version, source_system, timestamp_ms }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ENTITY STORE — STREAMING SCAN (LF-4)
// ═══════════════════════════════════════════════════════════════════════════

/// Streaming-capable entity scan API for tables that exceed the
/// in-memory capacity (~50K rows).
///
/// Implementations (in `lance-graph-callcenter` and friends) use Arrow
/// `RecordBatch` chunks rather than collected `Vec<Row>` so that very
/// large entity tables (call logs, conversation transcripts, audit
/// trails) can be processed without materializing the whole result set.
///
/// The contract crate is zero-dep, so the row batch and error types are
/// associated types — the caller binds them to concrete Arrow / Lance
/// types at the impl site.
pub trait EntityStore: Send + Sync {
    /// One streamed batch of rows. The implementor picks the concrete
    /// shape (typically `arrow::record_batch::RecordBatch` or a typed
    /// row vector); the contract surface stays Arrow-free.
    type RowBatch: Send;

    /// Error produced by stream setup or per-batch reads.
    type Error: Send + 'static;

    /// Iterator returned by `scan_stream`. Each call to `next()` yields
    /// one batch or a per-batch error. The implementor chooses the
    /// batch size based on backend characteristics (Lance fragments,
    /// DataFusion partitions, etc).
    type ScanStream: Iterator<Item = Result<Self::RowBatch, Self::Error>> + Send;

    /// Stream rows for an entity type.
    ///
    /// The `entity_type` argument matches `LineageHandle::entity_type`
    /// — e.g. `"customer"`, `"call_event"`, `"steering_intent"`.
    /// Implementations should prefer streaming over `Vec` collection
    /// once the row count exceeds ~50K, where holding the whole result
    /// in memory becomes wasteful.
    fn scan_stream(&self, entity_type: &str) -> Result<Self::ScanStream, Self::Error>;
}

// ═══════════════════════════════════════════════════════════════════════════
// ENTITY WRITER — UPSERT WITH LINEAGE (LF-5)
// ═══════════════════════════════════════════════════════════════════════════

/// Writer trait for entities with provenance tracking.
///
/// Every upsert produces a [`LineageHandle`] the caller can persist
/// alongside the data for audit purposes — who created/modified what,
/// when, and from which source system.
///
/// Like [`EntityStore`], the row payload is an associated type so the
/// contract crate can stay zero-dep; concrete impls in
/// `lance-graph-callcenter` bind it to `arrow::record_batch::RecordBatch`
/// or a typed row struct.
pub trait EntityWriter: Send + Sync {
    /// Error produced by the upsert operation.
    type Error: Send + 'static;

    /// One row's worth of data the implementation knows how to encode.
    type Row: Send;

    /// Upsert a row and emit a [`LineageHandle`] for the version produced.
    ///
    /// The handle is the audit-trail record — persist it next to the
    /// row in a sidecar lineage table or feed it into a `CausalEdge64`
    /// provenance bit stream.
    fn upsert_with_lineage(
        &self,
        entity_type: &'static str,
        entity_id: u64,
        row: Self::Row,
        source_system: &'static str,
    ) -> Result<LineageHandle, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marking_default_is_internal() {
        assert_eq!(Marking::default(), Marking::Internal);
    }

    #[test]
    fn required_property_spec() {
        let spec = PropertySpec::required("name", PropertyKind::String);
        assert!(spec.required);
        assert!(spec.default_value.is_none());
        assert_eq!(spec.marking, Marking::Internal);
    }

    #[test]
    fn optional_property_spec() {
        let spec = PropertySpec::optional("active", PropertyKind::Bool, "true");
        assert!(!spec.required);
        assert_eq!(spec.default_value, Some("true"));
    }

    #[test]
    fn free_property_spec() {
        let spec = PropertySpec::free("notes", PropertyKind::String);
        assert!(!spec.required);
        assert!(spec.default_value.is_none());
    }

    #[test]
    fn with_marking_builder() {
        let spec = PropertySpec::required("ssn", PropertyKind::String)
            .with_marking(Marking::Pii);
        assert_eq!(spec.marking, Marking::Pii);
    }

    #[test]
    fn lineage_handle_const_new() {
        let h = LineageHandle::new("customer", 42, 1, "crm", 1700000000000);
        assert_eq!(h.entity_type, "customer");
        assert_eq!(h.entity_id, 42);
        assert_eq!(h.version, 1);
        assert_eq!(h.source_system, "crm");
        assert_eq!(h.timestamp_ms, 1700000000000);
    }

    // ─────────────────────────────────────────────────────────────────
    // LF-4 / LF-5 — trait surface compile checks
    //
    // These tests don't exercise behaviour; they prove that the trait
    // surface is sound — an implementor can satisfy both `EntityStore`
    // and `EntityWriter` simultaneously with reasonable associated
    // types. If the trait bounds drift in a way that breaks
    // implementability, this test stops compiling.
    // ─────────────────────────────────────────────────────────────────

    /// Trivial in-memory backing struct used only by the trait-surface
    /// compile tests below. Holds nothing — the goal is to prove the
    /// `impl` blocks type-check, not to exercise behaviour.
    struct DummyStore;

    /// Row payload for `DummyStore`'s `EntityStore` and `EntityWriter`
    /// impls — a single tagged tuple stand-in for an Arrow batch.
    #[derive(Debug, PartialEq, Eq)]
    struct DummyBatch(u64);

    /// Empty error type — the dummy impls never fail.
    #[derive(Debug)]
    struct DummyError;

    impl EntityStore for DummyStore {
        type RowBatch = DummyBatch;
        type Error = DummyError;
        type ScanStream = std::vec::IntoIter<Result<DummyBatch, DummyError>>;

        fn scan_stream(&self, entity_type: &str) -> Result<Self::ScanStream, Self::Error> {
            // One batch tagged with the entity_type's length so the
            // argument is observably consumed.
            let batch = DummyBatch(entity_type.len() as u64);
            Ok(vec![Ok(batch)].into_iter())
        }
    }

    impl EntityWriter for DummyStore {
        type Error = DummyError;
        type Row = DummyBatch;

        fn upsert_with_lineage(
            &self,
            entity_type: &'static str,
            entity_id: u64,
            _row: Self::Row,
            source_system: &'static str,
        ) -> Result<LineageHandle, Self::Error> {
            Ok(LineageHandle::new(entity_type, entity_id, 1, source_system, 0))
        }
    }

    #[test]
    fn entity_store_scan_stream_compiles_and_yields() {
        let store = DummyStore;
        let mut stream = store.scan_stream("customer").expect("scan_stream");
        let first = stream.next().expect("one batch").expect("ok batch");
        // "customer" has 8 bytes.
        assert_eq!(first, DummyBatch(8));
        assert!(stream.next().is_none());
    }

    #[test]
    fn entity_writer_upsert_with_lineage_emits_handle() {
        let store = DummyStore;
        let handle = store
            .upsert_with_lineage("call_event", 7, DummyBatch(0), "asterisk")
            .expect("upsert");
        assert_eq!(handle.entity_type, "call_event");
        assert_eq!(handle.entity_id, 7);
        assert_eq!(handle.version, 1);
        assert_eq!(handle.source_system, "asterisk");
    }

    /// Compile-time check: a single struct can implement both traits
    /// at once. If the bounds ever conflict, this stops compiling.
    #[test]
    fn store_and_writer_compose_on_one_type() {
        fn assert_both<T: EntityStore + EntityWriter>(_: &T) {}
        assert_both(&DummyStore);
    }
}
