//! Property classification for AriGraph SPO predicates.
//!
//! Each predicate in the triple store carries a `PropertySpec` that
//! determines: (1) whether absence triggers a `FailureTicket` (Required),
//! (2) how the object value is stored — lossless Index or compressed
//! CAM-PQ Argmax, and (3) the NARS truth floor below which the system
//! escalates.
//!
//! The bardioc Required/Optional/Free concept maps to the I1 Codec
//! Regime Split (ADR-0002): Required = Passthrough (identity must
//! round-trip), Optional = configurable, Free = CamPq (similarity
//! search over schema-free attributes).

use crate::cam::CodecRoute;

/// Classification of an SPO predicate's cardinality and schema obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    /// MUST exist for the entity to be valid. Absence triggers
    /// FailureTicket via FreeEnergy escalation. Always Index regime
    /// (lossless, exact match). Examples: tax_id, customer_name, IBAN.
    Required,
    /// MAY exist. Adds value when present but absence does not
    /// escalate. Codec route is configurable per predicate —
    /// address = Index, industry_description = CamPq.
    Optional,
    /// Schema-free. Any predicate name accepted. Default codec
    /// route is CamPq (Argmax) for similarity search across
    /// tenants. User-defined tags, notes, custom fields.
    Free,
}

/// Specification for a single predicate in the AriGraph SPO store.
///
/// Ties the predicate name to its property kind, codec route, and
/// NARS truth floor. The truth floor is the minimum (frequency,
/// confidence) below which the system treats the property as
/// "effectively absent" — for Required properties, this triggers
/// a FailureTicket.
#[derive(Clone, Debug)]
pub struct PropertySpec {
    /// Predicate name in the SPO triple (e.g. "tax_id", "address", "note").
    pub predicate: &'static str,
    /// Required / Optional / Free classification.
    pub kind: PropertyKind,
    /// How the object value is stored/searched. Derived from kind
    /// by default but overridable per predicate.
    pub codec_route: CodecRoute,
    /// Minimum (frequency, confidence) as u8 pair (0..255 each).
    /// Below this floor, Required properties trigger FailureTicket.
    /// None = no floor check (typical for Free properties).
    pub nars_floor: Option<(u8, u8)>,
}

impl PropertySpec {
    /// Create a Required property spec. Default codec: Passthrough (Index).
    /// Default NARS floor: (128, 128) — moderate confidence required.
    pub const fn required(predicate: &'static str) -> Self {
        Self {
            predicate,
            kind: PropertyKind::Required,
            codec_route: CodecRoute::Passthrough,
            nars_floor: Some((128, 128)),
        }
    }

    /// Create an Optional property spec. Caller must specify codec route.
    /// No NARS floor by default (absence doesn't escalate).
    pub const fn optional(predicate: &'static str, codec_route: CodecRoute) -> Self {
        Self {
            predicate,
            kind: PropertyKind::Optional,
            codec_route,
            nars_floor: None,
        }
    }

    /// Create a Free property spec. Default codec: CamPq (Argmax).
    /// No NARS floor (schema-free, always accepted).
    pub const fn free(predicate: &'static str) -> Self {
        Self {
            predicate,
            kind: PropertyKind::Free,
            codec_route: CodecRoute::CamPq,
            nars_floor: None,
        }
    }

    /// Override the NARS truth floor.
    pub const fn with_nars_floor(mut self, frequency: u8, confidence: u8) -> Self {
        self.nars_floor = Some((frequency, confidence));
        self
    }

    /// Override the codec route.
    pub const fn with_codec_route(mut self, route: CodecRoute) -> Self {
        self.codec_route = route;
        self
    }

    /// Check whether a given (frequency, confidence) pair is below this
    /// property's truth floor. Returns true if escalation is warranted.
    pub const fn below_floor(&self, frequency: u8, confidence: u8) -> bool {
        match self.nars_floor {
            Some((min_f, min_c)) => frequency < min_f || confidence < min_c,
            None => false,
        }
    }
}

/// A property schema — a collection of PropertySpecs for a given entity type.
/// Used by AriGraph to validate triples on insert and to route codec
/// decisions per predicate.
#[derive(Clone, Debug)]
pub struct PropertySchema {
    /// Entity type name (e.g. "Customer", "Invoice", "TaxDeclaration").
    pub entity_type: &'static str,
    /// Ordered list of property specs. Required properties come first
    /// by convention (not enforced).
    pub properties: &'static [PropertySpec],
}

impl PropertySchema {
    /// Look up a property spec by predicate name.
    pub fn get(&self, predicate: &str) -> Option<&PropertySpec> {
        self.properties.iter().find(|p| p.predicate == predicate)
    }

    /// Return all Required properties.
    pub fn required(&self) -> impl Iterator<Item = &PropertySpec> {
        self.properties.iter().filter(|p| p.kind == PropertyKind::Required)
    }

    /// Return all predicates that are missing from a given set of
    /// predicate names. Only checks Required properties.
    /// Returns predicate names that should trigger FailureTicket.
    pub fn missing_required<'a>(&'a self, present: &'a [&str]) -> impl Iterator<Item = &'static str> + 'a {
        self.required()
            .filter(move |p| !present.contains(&p.predicate))
            .map(|p| p.predicate)
    }

    /// Determine the codec route for a predicate. If the predicate is
    /// not in the schema, it's treated as Free (CamPq).
    pub fn codec_route_for(&self, predicate: &str) -> CodecRoute {
        self.get(predicate)
            .map(|p| p.codec_route)
            .unwrap_or(CodecRoute::CamPq)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Schema builder — declarative API for SMB tenants
// ═══════════════════════════════════════════════════════════════════════════

/// Owned property schema built at runtime via the builder API.
/// Complement to `PropertySchema` (which is `&'static`-only for const schemas).
#[derive(Clone, Debug)]
pub struct Schema {
    pub name: &'static str,
    pub properties: Vec<PropertySpec>,
}

impl Schema {
    pub fn builder(name: &'static str) -> SchemaBuilder {
        SchemaBuilder { name, properties: Vec::new() }
    }

    pub fn get(&self, predicate: &str) -> Option<&PropertySpec> {
        self.properties.iter().find(|p| p.predicate == predicate)
    }

    pub fn required_props(&self) -> impl Iterator<Item = &PropertySpec> {
        self.properties.iter().filter(|p| p.kind == PropertyKind::Required)
    }

    pub fn missing_required<'a>(&'a self, present: &'a [&str]) -> impl Iterator<Item = &'static str> + 'a {
        self.required_props()
            .filter(move |p| !present.contains(&p.predicate))
            .map(|p| p.predicate)
    }

    pub fn codec_route_for(&self, predicate: &str) -> CodecRoute {
        self.get(predicate)
            .map(|p| p.codec_route)
            .unwrap_or(CodecRoute::CamPq)
    }

    /// Validate a set of present predicates. Returns a list of missing
    /// Required predicate names. Empty = valid.
    pub fn validate(&self, present: &[&str]) -> Vec<&'static str> {
        self.missing_required(present).collect()
    }
}

pub struct SchemaBuilder {
    name: &'static str,
    properties: Vec<PropertySpec>,
}

impl SchemaBuilder {
    /// Add a Required property (Passthrough codec, NARS floor 128/128).
    pub fn required(mut self, predicate: &'static str) -> Self {
        self.properties.push(PropertySpec::required(predicate));
        self
    }

    /// Add an Optional property with Passthrough (exact match) codec.
    pub fn optional(mut self, predicate: &'static str) -> Self {
        self.properties.push(PropertySpec::optional(predicate, CodecRoute::Passthrough));
        self
    }

    /// Add an Optional property with CamPq (similarity search) codec.
    pub fn searchable(mut self, predicate: &'static str) -> Self {
        self.properties.push(PropertySpec::optional(predicate, CodecRoute::CamPq));
        self
    }

    /// Add a Free property (CamPq codec, no NARS floor).
    pub fn free(mut self, predicate: &'static str) -> Self {
        self.properties.push(PropertySpec::free(predicate));
        self
    }

    /// Add a custom PropertySpec directly.
    pub fn property(mut self, spec: PropertySpec) -> Self {
        self.properties.push(spec);
        self
    }

    pub fn build(self) -> Schema {
        Schema { name: self.name, properties: self.properties }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Example schemas — SMB domain (const)
// ═══════════════════════════════════════════════════════════════════════════

/// Customer entity property schema.
pub const CUSTOMER_SCHEMA: PropertySchema = PropertySchema {
    entity_type: "Customer",
    properties: &[
        // Required — identity, lossless
        PropertySpec::required("customer_name"),
        PropertySpec::required("tax_id"),
        // Optional — exact match
        PropertySpec::optional("address", CodecRoute::Passthrough),
        PropertySpec::optional("iban", CodecRoute::Passthrough),
        PropertySpec::optional("phone", CodecRoute::Passthrough),
        PropertySpec::optional("email", CodecRoute::Passthrough),
        // Optional — similarity search
        PropertySpec::optional("industry", CodecRoute::CamPq),
        PropertySpec::optional("description", CodecRoute::CamPq),
        // Free — anything goes, similarity indexed
        PropertySpec::free("tag"),
        PropertySpec::free("note"),
    ],
};

/// Invoice entity property schema.
pub const INVOICE_SCHEMA: PropertySchema = PropertySchema {
    entity_type: "Invoice",
    properties: &[
        PropertySpec::required("invoice_number"),
        PropertySpec::required("date"),
        PropertySpec::required("total_amount"),
        PropertySpec::required("currency"),
        PropertySpec::required("customer_ref"),
        PropertySpec::optional("due_date", CodecRoute::Passthrough),
        PropertySpec::optional("payment_terms", CodecRoute::Passthrough),
        PropertySpec::optional("line_items_hash", CodecRoute::Passthrough),
        PropertySpec::free("note"),
        PropertySpec::free("tag"),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_defaults() {
        let p = PropertySpec::required("tax_id");
        assert_eq!(p.kind, PropertyKind::Required);
        assert_eq!(p.codec_route, CodecRoute::Passthrough);
        assert!(p.nars_floor.is_some());
    }

    #[test]
    fn optional_inherits_codec() {
        let p = PropertySpec::optional("industry", CodecRoute::CamPq);
        assert_eq!(p.kind, PropertyKind::Optional);
        assert_eq!(p.codec_route, CodecRoute::CamPq);
        assert!(p.nars_floor.is_none());
    }

    #[test]
    fn free_defaults_to_campq() {
        let p = PropertySpec::free("note");
        assert_eq!(p.kind, PropertyKind::Free);
        assert_eq!(p.codec_route, CodecRoute::CamPq);
        assert!(p.nars_floor.is_none());
    }

    #[test]
    fn below_floor_required() {
        let p = PropertySpec::required("tax_id");
        // Default floor is (128, 128)
        assert!(p.below_floor(100, 200)); // frequency too low
        assert!(p.below_floor(200, 100)); // confidence too low
        assert!(!p.below_floor(200, 200)); // both above
    }

    #[test]
    fn below_floor_free_always_false() {
        let p = PropertySpec::free("note");
        assert!(!p.below_floor(0, 0)); // no floor = never below
    }

    #[test]
    fn schema_missing_required() {
        let present = ["customer_name", "address", "tag"];
        let missing: Vec<_> = CUSTOMER_SCHEMA.missing_required(&present).collect();
        assert!(missing.contains(&"tax_id"));
        assert!(!missing.contains(&"customer_name"));
    }

    #[test]
    fn schema_codec_route_known_predicate() {
        assert_eq!(CUSTOMER_SCHEMA.codec_route_for("tax_id"), CodecRoute::Passthrough);
        assert_eq!(CUSTOMER_SCHEMA.codec_route_for("industry"), CodecRoute::CamPq);
    }

    #[test]
    fn schema_codec_route_unknown_predicate_defaults_to_campq() {
        assert_eq!(CUSTOMER_SCHEMA.codec_route_for("unknown_field"), CodecRoute::CamPq);
    }

    #[test]
    fn invoice_schema_has_five_required() {
        let count = INVOICE_SCHEMA.required().count();
        assert_eq!(count, 5);
    }

    #[test]
    fn with_nars_floor_override() {
        let p = PropertySpec::free("note").with_nars_floor(50, 50);
        assert!(p.below_floor(40, 60));
        assert!(!p.below_floor(60, 60));
    }

    // ── Schema builder tests ──

    #[test]
    fn schema_builder_declarative() {
        let s = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id")
            .optional("address")
            .searchable("industry")
            .free("note")
            .build();
        assert_eq!(s.name, "Customer");
        assert_eq!(s.properties.len(), 5);
    }

    #[test]
    fn schema_validate_missing_required() {
        let s = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id")
            .optional("address")
            .build();
        let missing = s.validate(&["customer_name", "address"]);
        assert_eq!(missing, vec!["tax_id"]);
    }

    #[test]
    fn schema_validate_all_present() {
        let s = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id")
            .build();
        let missing = s.validate(&["customer_name", "tax_id"]);
        assert!(missing.is_empty());
    }

    #[test]
    fn schema_searchable_is_campq() {
        let s = Schema::builder("Test")
            .searchable("description")
            .build();
        assert_eq!(s.codec_route_for("description"), CodecRoute::CamPq);
    }

    #[test]
    fn schema_unknown_predicate_defaults_campq() {
        let s = Schema::builder("Test").build();
        assert_eq!(s.codec_route_for("anything"), CodecRoute::CamPq);
    }

    #[test]
    fn schema_optional_is_passthrough() {
        let s = Schema::builder("Test")
            .optional("address")
            .build();
        assert_eq!(s.codec_route_for("address"), CodecRoute::Passthrough);
    }
}
