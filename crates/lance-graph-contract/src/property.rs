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
}
