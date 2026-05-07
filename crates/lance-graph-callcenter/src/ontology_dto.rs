//! External ontology DTO surface — the "Foundry outside" layer.
//!
//! Projects the canonical `OntologyRegistry` (the SoA, per
//! `ogit-cascade-supabase-callcenter-v1.md` Pillar 0) into consumer-facing
//! bilingual DTOs that PostgREST, Phoenix, and downstream apps consume.
//! This is the ONE surface both SMB and MedCare consumers see.
//!
//! Per Pillar 3 of v1 cascade, the per-tenant factories (`medcare_ontology`,
//! `smb_ontology`) collapsed from hand-rolled DTO builders to 2-line
//! projections over `OntologyRegistry::enumerate(namespace)`. The bridges
//! stay 15-20 LOC; the heavy lifting moves into the registry.
//!
//! Internal types (`BindSpace`, `FingerprintColumns`, `CausalEdge64`) never
//! leak through this module. The BBB invariant (from `external_membrane.rs`)
//! holds: VSA/semiring types stay inside; scalar/typed DTOs cross outside.

// classification: bridge-projection
// (per .claude/plans/palantir-parity-cascade-v2.md DTO ladder Tier-4)

use lance_graph_contract::ontology::{EntityTypeId, Label, Locale, Ontology};
use lance_graph_contract::property::{
    ActionTrigger, Cardinality, Marking, PropertyKind, SemanticType,
};
use lance_graph_ontology::{MappingRow, OntologyRegistry, SchemaPtr};
use lance_graph_ontology::namespace::SchemaKind;

/// External-facing ontology view. Projects the registry through a locale
/// lens, stripping internal implementation details.
#[derive(Clone, Debug)]
pub struct OntologyDto {
    pub key: &'static str,
    pub name: String,
    pub locale: Locale,
    pub entity_types: Vec<EntityTypeDto>,
    pub link_types: Vec<LinkTypeDto>,
    pub action_types: Vec<ActionTypeDto>,
}

#[derive(Clone, Debug)]
pub struct EntityTypeDto {
    pub id: EntityTypeId,
    pub key: String,
    pub name: String,
    pub properties: Vec<PropertyDto>,
    pub required_count: usize,
}

#[derive(Clone, Debug)]
pub struct PropertyDto {
    pub key: String,
    pub kind: &'static str,
    pub semantic_type: String,
    pub marking: &'static str,
}

#[derive(Clone, Debug)]
pub struct LinkTypeDto {
    pub subject_type: String,
    pub predicate: String,
    pub object_type: String,
    pub cardinality: &'static str,
}

#[derive(Clone, Debug)]
pub struct ActionTypeDto {
    pub name: String,
    pub entity_type: String,
    pub target_predicate: String,
    pub trigger: &'static str,
}

impl OntologyDto {
    /// Projection over a registry namespace — the v1 cascade Pillar 0
    /// canonical constructor. Walks `registry.enumerate(namespace)` once
    /// and drops each row into the matching kind bucket.
    ///
    /// Per the v1 cascade plan: `MappingRow` carries dictionary metadata
    /// (kind / semantic_type / marking), not full Schema property layouts.
    /// Property + link/action body fields populate when D-CASCADE-V1-7
    /// (codec-cascade columns) lands; today they remain projections of
    /// what the registry knows.
    pub fn project(
        registry: &OntologyRegistry,
        namespace: &str,
        key: &'static str,
        label: Label,
        locale: Locale,
    ) -> Self {
        let rows = registry.enumerate(namespace);
        let mut entity_types: Vec<EntityTypeDto> = Vec::new();
        let mut link_types: Vec<LinkTypeDto> = Vec::new();
        let mut action_types: Vec<ActionTypeDto> = Vec::new();

        for row in rows {
            match row.kind {
                SchemaKind::Entity => entity_types.push(entity_dto(&row)),
                SchemaKind::Edge => link_types.push(link_dto(&row)),
                SchemaKind::Attribute => action_types.push(action_dto(&row)),
            }
        }

        OntologyDto {
            key,
            name: label.display(locale).to_string(),
            locale,
            entity_types,
            link_types,
            action_types,
        }
    }

    /// Legacy projection over a hand-rolled `Ontology`. Retained for
    /// `transcode::CachedOntology` and any consumer that already carries
    /// a fully-formed `Ontology` literal. New code should use
    /// [`OntologyDto::project`] over the canonical registry instead.
    pub fn from_ontology(ontology: &Ontology, locale: Locale) -> Self {
        let entity_types: Vec<EntityTypeDto> = ontology
            .schemas
            .iter()
            .enumerate()
            .map(|(idx, schema)| EntityTypeDto {
                id: (idx + 1) as EntityTypeId,
                key: schema.name.to_string(),
                name: schema.name.to_string(),
                required_count: schema.required_props().count(),
                properties: schema
                    .properties
                    .iter()
                    .map(|p| PropertyDto {
                        key: p.predicate.to_string(),
                        kind: kind_str(p.kind),
                        semantic_type: semantic_type_str(&p.semantic_type),
                        marking: marking_str(p.marking),
                    })
                    .collect(),
            })
            .collect();

        let link_types: Vec<LinkTypeDto> = ontology
            .links
            .iter()
            .map(|l| LinkTypeDto {
                subject_type: l.subject_type.to_string(),
                predicate: l.predicate.to_string(),
                object_type: l.object_type.to_string(),
                cardinality: cardinality_str(l.cardinality),
            })
            .collect();

        let action_types: Vec<ActionTypeDto> = ontology
            .actions
            .iter()
            .map(|a| ActionTypeDto {
                name: a.name.to_string(),
                entity_type: a.entity_type.to_string(),
                target_predicate: a.target_predicate.to_string(),
                trigger: trigger_str(a.trigger),
            })
            .collect();

        OntologyDto {
            key: ontology.name,
            name: ontology.label.display(locale).to_string(),
            locale,
            entity_types,
            link_types,
            action_types,
        }
    }

    pub fn entity_type(&self, key: &str) -> Option<&EntityTypeDto> {
        self.entity_types.iter().find(|e| e.key == key)
    }

    pub fn links_from(&self, subject_type: &str) -> Vec<&LinkTypeDto> {
        self.link_types
            .iter()
            .filter(|l| l.subject_type == subject_type)
            .collect()
    }

    pub fn actions_for(&self, entity_type: &str) -> Vec<&ActionTypeDto> {
        self.action_types
            .iter()
            .filter(|a| a.entity_type == entity_type)
            .collect()
    }
}

// ── Per-row projection helpers ───────────────────────────────────────────────

fn entity_dto(row: &MappingRow) -> EntityTypeDto {
    let SchemaPtr { .. } = row.schema_ptr; // structural binding only
    let id = row.schema_ptr.entity_type_id();
    let name = row
        .ogit_uri
        .name()
        .unwrap_or(&row.public_name)
        .to_string();
    EntityTypeDto {
        id,
        key: row.public_name.clone(),
        name,
        // Properties land in this slot when D-CASCADE-V1-7 wires the
        // codec-cascade columns; today MappingRow only knows kind+marking.
        properties: Vec::new(),
        required_count: 0,
    }
}

fn link_dto(row: &MappingRow) -> LinkTypeDto {
    LinkTypeDto {
        subject_type: String::new(),
        predicate: row.public_name.clone(),
        object_type: String::new(),
        cardinality: "many_to_many",
    }
}

fn action_dto(row: &MappingRow) -> ActionTypeDto {
    ActionTypeDto {
        name: row.public_name.clone(),
        entity_type: String::new(),
        target_predicate: row
            .ogit_uri
            .name()
            .unwrap_or(&row.public_name)
            .to_string(),
        trigger: "manual",
    }
}

fn kind_str(k: PropertyKind) -> &'static str {
    match k {
        PropertyKind::Required => "required",
        PropertyKind::Optional => "optional",
        PropertyKind::Free => "free",
    }
}

fn marking_str(m: Marking) -> &'static str {
    match m {
        Marking::Public => "public",
        Marking::Internal => "internal",
        Marking::Pii => "pii",
        Marking::Financial => "financial",
        Marking::Restricted => "restricted",
    }
}

fn cardinality_str(c: Cardinality) -> &'static str {
    match c {
        Cardinality::OneToOne => "one_to_one",
        Cardinality::OneToMany => "one_to_many",
        Cardinality::ManyToMany => "many_to_many",
    }
}

fn trigger_str(t: ActionTrigger) -> &'static str {
    match t {
        ActionTrigger::Manual => "manual",
        ActionTrigger::Auto => "auto",
        ActionTrigger::Suggested => "suggested",
    }
}

fn semantic_type_str(st: &SemanticType) -> String {
    match st {
        SemanticType::PlainText => "text".into(),
        SemanticType::Iban => "iban".into(),
        SemanticType::Currency(code) => format!("currency:{code}"),
        SemanticType::Email => "email".into(),
        SemanticType::Phone => "phone".into(),
        SemanticType::Date(p) => format!("date:{p:?}").to_lowercase(),
        SemanticType::Geo(f) => format!("geo:{f:?}").to_lowercase(),
        SemanticType::Address => "address".into(),
        SemanticType::File(mime) => format!("file:{mime}"),
        SemanticType::Image => "image".into(),
        SemanticType::Url => "url".into(),
        SemanticType::TaxId => "tax_id".into(),
        SemanticType::CustomerId => "customer_id".into(),
        SemanticType::InvoiceNumber => "invoice_number".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bilingual SMB + MedCare projection factories — the bridge collapse.
//
// Per `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` Pillar 3:
// the per-tenant factories are 2-line projections over the canonical
// `OntologyRegistry` enumerate. The hand-rolled `Ontology` literals these
// used to build were displaced when v4 hydration started populating the
// registry from `OGIT/NTO/Healthcare/` and `OGIT/NTO/SMB/` TTL.
// ═══════════════════════════════════════════════════════════════════════════

/// SMB DTO — projection over the `SMB` namespace of the canonical registry.
pub fn smb_ontology(registry: &OntologyRegistry) -> OntologyDto {
    OntologyDto::project(
        registry,
        "SMB",
        "smb",
        Label::new("smb", "Tax Practice", "Steuerberatungskanzlei"),
        Locale::De,
    )
}

/// MedCare DTO — projection over the `Healthcare` namespace of the
/// canonical registry.
pub fn medcare_ontology(registry: &OntologyRegistry) -> OntologyDto {
    OntologyDto::project(
        registry,
        "Healthcare",
        "medcare",
        Label::new("medcare", "Medical Practice", "Arztpraxis"),
        Locale::De,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::Schema;
    use lance_graph_ontology::namespace::OgitUri;
    use lance_graph_ontology::{MappingProposal, MappingProposalKind};

    fn entity_proposal(bridge: &str, public: &str, uri: &str) -> MappingProposal {
        let parsed = OgitUri::parse(uri).unwrap();
        let ns = parsed.namespace().unwrap().to_string();
        let name = parsed.name().unwrap().to_string();
        MappingProposal {
            public_name: public.to_string(),
            bridge_id: bridge.to_string(),
            ogit_uri: parsed,
            namespace: ns,
            kind: MappingProposalKind::Entity {
                schema: Schema::builder(Box::leak(name.into_boxed_str())).build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: format!("test://{uri}"),
            checksum: format!("ck-{uri}"),
            created_by: "test".into(),
        }
    }

    fn smb_registry() -> OntologyRegistry {
        let reg = OntologyRegistry::new_in_memory();
        reg.append_mapping(entity_proposal("smb", "Customer", "ogit.SMB:Customer")).unwrap();
        reg.append_mapping(entity_proposal("smb", "Invoice", "ogit.SMB:Invoice")).unwrap();
        reg.append_mapping(entity_proposal("smb", "TaxDeclaration", "ogit.SMB:TaxDeclaration")).unwrap();
        reg
    }

    fn medcare_registry() -> OntologyRegistry {
        let reg = OntologyRegistry::new_in_memory();
        reg.append_mapping(entity_proposal("medcare", "Patient", "ogit.Healthcare:Patient")).unwrap();
        reg.append_mapping(entity_proposal("medcare", "Diagnosis", "ogit.Healthcare:Diagnosis")).unwrap();
        reg.append_mapping(entity_proposal("medcare", "LabResult", "ogit.Healthcare:LabResult")).unwrap();
        reg.append_mapping(entity_proposal("medcare", "Prescription", "ogit.Healthcare:Prescription")).unwrap();
        reg
    }

    #[test]
    fn smb_projects_three_entities() {
        let reg = smb_registry();
        let dto = smb_ontology(&reg);
        assert_eq!(dto.key, "smb");
        assert_eq!(dto.name, "Steuerberatungskanzlei");
        assert_eq!(dto.entity_types.len(), 3);
        assert!(dto.entity_type("Customer").is_some());
    }

    #[test]
    fn medcare_projects_four_entities() {
        let reg = medcare_registry();
        let dto = medcare_ontology(&reg);
        assert_eq!(dto.key, "medcare");
        assert_eq!(dto.name, "Arztpraxis");
        assert_eq!(dto.entity_types.len(), 4);
    }

    #[test]
    fn unknown_namespace_yields_empty_dto() {
        let reg = OntologyRegistry::new_in_memory();
        let dto = OntologyDto::project(
            &reg,
            "Nonexistent",
            "x",
            Label::en_only("x"),
            Locale::En,
        );
        assert!(dto.entity_types.is_empty());
        assert!(dto.link_types.is_empty());
        assert!(dto.action_types.is_empty());
    }

    #[test]
    fn from_ontology_legacy_path_still_works() {
        // Legacy projection path retained for `transcode::CachedOntology`.
        let ontology = Ontology::builder("test")
            .label(Label::new("test", "Test", "Test"))
            .schema(Schema::builder("Customer").required("name").build())
            .build();
        let dto = OntologyDto::from_ontology(&ontology, Locale::En);
        assert_eq!(dto.entity_types.len(), 1);
        assert_eq!(dto.entity_types[0].required_count, 1);
    }
}
