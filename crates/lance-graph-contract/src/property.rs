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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

impl Marking {
    /// Returns the most restrictive marking from the slice.
    ///
    /// GDPR precedence: Public < Internal < Pii < Financial < Restricted.
    /// If any property on a row is `Pii`, the row inherits `Pii` (or higher).
    /// Empty slice returns `Public` (least restrictive).
    pub fn most_restrictive(markings: &[Marking]) -> Marking {
        markings.iter().copied().max().unwrap_or(Marking::Public)
    }
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

    /// Merge two lineage handles for the same entity.
    ///
    /// Takes the higher version, the later timestamp, and the newer
    /// handle's `source_system`. Because `source_system` is `&'static str`,
    /// we cannot dynamically concatenate two values (e.g. `"mongo+imap"`).
    /// The caller can use a pre-interned combined string if a merged
    /// source label is required.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that `entity_type` and `entity_id` match between
    /// the two handles. Merging handles for different entities is a
    /// logic error.
    pub fn merge(self, other: Self) -> Self {
        debug_assert_eq!(self.entity_type, other.entity_type);
        debug_assert_eq!(self.entity_id, other.entity_id);
        let (newer, older) = if self.version >= other.version {
            (self, other)
        } else {
            (other, self)
        };
        Self {
            entity_type: newer.entity_type,
            entity_id: newer.entity_id,
            version: newer.version,
            source_system: newer.source_system,
            timestamp_ms: newer.timestamp_ms.max(older.timestamp_ms),
        }
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

// ═══════════════════════════════════════════════════════════════════════════
// MOCK STORE — IN-MEMORY ENTITYSTORE + ENTITYWRITER FOR TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// Test-only in-memory store implementing both [`EntityStore`] and
/// [`EntityWriter`].
///
/// **Not for production use.** This module exists as a copy-paste template
/// for SMB integration tests. It uses `RefCell` for interior mutability so
/// it satisfies the `&self` signature of `EntityWriter::upsert_with_lineage`.
/// A production implementation would use `RwLock` or take `&mut self`.
pub mod mock_store {
    use super::*;
    use std::sync::RwLock;

    /// In-memory test store implementing both [`EntityStore`] and
    /// [`EntityWriter`].
    ///
    /// Rows are stored as `(entity_id, payload)` pairs. The version counter
    /// auto-increments on each upsert. Uses `RwLock` for interior mutability
    /// so the `Send + Sync` bounds on `EntityStore` / `EntityWriter` are
    /// satisfied.
    ///
    /// **Not for production use.** This is a copy-paste template for SMB
    /// integration tests.
    pub struct VecStore {
        pub rows: RwLock<Vec<(u64, Vec<u8>)>>,
        version_counter: RwLock<u64>,
    }

    impl VecStore {
        pub fn new() -> Self {
            Self {
                rows: RwLock::new(Vec::new()),
                version_counter: RwLock::new(0),
            }
        }
    }

    impl EntityStore for VecStore {
        type RowBatch = Vec<(u64, Vec<u8>)>;
        type Error = &'static str;
        type ScanStream = std::vec::IntoIter<Result<Self::RowBatch, Self::Error>>;

        fn scan_stream(&self, _entity_type: &str) -> Result<Self::ScanStream, Self::Error> {
            let batch = self.rows.read().map_err(|_| "lock poisoned")?.clone();
            Ok(vec![Ok(batch)].into_iter())
        }
    }

    impl EntityWriter for VecStore {
        type Error = &'static str;
        type Row = Vec<u8>;

        fn upsert_with_lineage(
            &self,
            entity_type: &'static str,
            entity_id: u64,
            row: Self::Row,
            source_system: &'static str,
        ) -> Result<LineageHandle, Self::Error> {
            let mut ver = self.version_counter.write().map_err(|_| "lock poisoned")?;
            *ver += 1;
            let version = *ver;
            self.rows.write().map_err(|_| "lock poisoned")?.push((entity_id, row));
            Ok(LineageHandle::new(entity_type, entity_id, version, source_system, 0))
        }
    }
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

    // ─────────────────────────────────────────────────────────────────
    // W-1 — LineageHandle::merge
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn merge_takes_higher_version() {
        let v1 = LineageHandle::new("customer", 42, 1, "mongo", 1000);
        let v3 = LineageHandle::new("customer", 42, 3, "imap", 900);
        let merged = v1.merge(v3);
        assert_eq!(merged.version, 3);
        assert_eq!(merged.source_system, "imap"); // newer handle's source
    }

    #[test]
    fn merge_takes_later_timestamp() {
        let a = LineageHandle::new("order", 7, 2, "crm", 5000);
        let b = LineageHandle::new("order", 7, 1, "erp", 9000);
        let merged = a.merge(b);
        // a has higher version (2), b has later timestamp (9000)
        assert_eq!(merged.version, 2);
        assert_eq!(merged.source_system, "crm");
        assert_eq!(merged.timestamp_ms, 9000);
    }

    #[test]
    fn merge_equal_versions_keeps_self() {
        let a = LineageHandle::new("ticket", 1, 5, "src_a", 100);
        let b = LineageHandle::new("ticket", 1, 5, "src_b", 200);
        let merged = a.merge(b);
        // self.version >= other.version, so self is "newer"
        assert_eq!(merged.source_system, "src_a");
        assert_eq!(merged.timestamp_ms, 200);
    }

    // ─────────────────────────────────────────────────────────────────
    // W-2 — Marking::most_restrictive
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn most_restrictive_empty_is_public() {
        assert_eq!(Marking::most_restrictive(&[]), Marking::Public);
    }

    #[test]
    fn most_restrictive_single() {
        assert_eq!(Marking::most_restrictive(&[Marking::Pii]), Marking::Pii);
    }

    #[test]
    fn most_restrictive_mixed() {
        let markings = [
            Marking::Public,
            Marking::Internal,
            Marking::Pii,
            Marking::Financial,
            Marking::Internal,
        ];
        assert_eq!(Marking::most_restrictive(&markings), Marking::Financial);
    }

    #[test]
    fn most_restrictive_all_public() {
        let markings = [Marking::Public, Marking::Public, Marking::Public];
        assert_eq!(Marking::most_restrictive(&markings), Marking::Public);
    }

    #[test]
    fn most_restrictive_restricted_wins() {
        let markings = [Marking::Pii, Marking::Restricted, Marking::Financial];
        assert_eq!(Marking::most_restrictive(&markings), Marking::Restricted);
    }

    #[test]
    fn marking_ord_matches_gdpr_precedence() {
        assert!(Marking::Public < Marking::Internal);
        assert!(Marking::Internal < Marking::Pii);
        assert!(Marking::Pii < Marking::Financial);
        assert!(Marking::Financial < Marking::Restricted);
    }

    // ─────────────────────────────────────────────────────────────────
    // W-3 + W-4 — VecStore mock
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn vec_store_scan_empty() {
        let store = mock_store::VecStore::new();
        let mut stream = store.scan_stream("any").expect("scan");
        let batch = stream.next().expect("one batch").expect("ok");
        assert!(batch.is_empty());
        assert!(stream.next().is_none());
    }

    #[test]
    fn vec_store_upsert_and_scan() {
        let store = mock_store::VecStore::new();
        let h1 = store
            .upsert_with_lineage("customer", 1, vec![0xAA], "crm")
            .expect("upsert 1");
        let h2 = store
            .upsert_with_lineage("customer", 2, vec![0xBB], "crm")
            .expect("upsert 2");

        assert_eq!(h1.version, 1);
        assert_eq!(h2.version, 2);
        assert_eq!(h1.entity_id, 1);
        assert_eq!(h2.entity_id, 2);

        let mut stream = store.scan_stream("customer").expect("scan");
        let batch = stream.next().expect("one batch").expect("ok");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], (1, vec![0xAA]));
        assert_eq!(batch[1], (2, vec![0xBB]));
    }

    #[test]
    fn vec_store_implements_both_traits() {
        fn assert_both<T: EntityStore + EntityWriter>(_: &T) {}
        assert_both(&mock_store::VecStore::new());
    }
}
