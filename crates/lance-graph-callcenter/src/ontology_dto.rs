//! External ontology DTO surface — the "Foundry outside" layer.
//!
//! Projects the internal `contract::ontology::Ontology` into consumer-facing
//! bilingual DTOs that PostgREST, Phoenix, and downstream apps consume.
//! This is the ONE surface both SMB and MedCare consumers see.
//!
//! Internal types (`BindSpace`, `FingerprintColumns`, `CausalEdge64`) never
//! leak through this module. The BBB invariant (from `external_membrane.rs`)
//! holds: VSA/semiring types stay inside; scalar/typed DTOs cross outside.

use lance_graph_contract::ontology::{EntityTypeId, Label, Locale, Ontology};
use lance_graph_contract::property::{
    ActionSpec, ActionTrigger, Cardinality, LinkSpec, Marking, PropertyKind, Schema, SemanticType,
};

/// External-facing ontology view. Projects the full `Ontology` through
/// a locale lens, stripping internal implementation details.
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
    pub key: &'static str,
    pub name: String,
    pub properties: Vec<PropertyDto>,
    pub required_count: usize,
}

#[derive(Clone, Debug)]
pub struct PropertyDto {
    pub key: &'static str,
    pub kind: &'static str,
    pub semantic_type: String,
    pub marking: &'static str,
}

#[derive(Clone, Debug)]
pub struct LinkTypeDto {
    pub subject_type: &'static str,
    pub predicate: &'static str,
    pub object_type: &'static str,
    pub cardinality: &'static str,
}

#[derive(Clone, Debug)]
pub struct ActionTypeDto {
    pub name: &'static str,
    pub entity_type: &'static str,
    pub target_predicate: &'static str,
    pub trigger: &'static str,
}

impl OntologyDto {
    pub fn from_ontology(ontology: &Ontology, locale: Locale) -> Self {
        let entity_types: Vec<EntityTypeDto> = ontology
            .schemas
            .iter()
            .enumerate()
            .map(|(idx, schema)| EntityTypeDto {
                id: (idx + 1) as EntityTypeId,
                key: schema.name,
                name: schema.name.to_string(),
                required_count: schema.required_props().count(),
                properties: schema
                    .properties
                    .iter()
                    .map(|p| PropertyDto {
                        key: p.predicate,
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
                subject_type: l.subject_type,
                predicate: l.predicate,
                object_type: l.object_type,
                cardinality: cardinality_str(l.cardinality),
            })
            .collect();

        let action_types: Vec<ActionTypeDto> = ontology
            .actions
            .iter()
            .map(|a| ActionTypeDto {
                name: a.name,
                entity_type: a.entity_type,
                target_predicate: a.target_predicate,
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
// Bilingual SMB + MedCare example ontologies
// ═══════════════════════════════════════════════════════════════════════════

pub fn smb_ontology() -> Ontology {
    Ontology::builder("smb")
        .label(Label::new("smb", "Tax Practice", "Steuerberatungskanzlei"))
        .locale(Locale::De)
        .schema(
            Schema::builder("Customer")
                .required("customer_name")
                .required("tax_id")
                .optional("address")
                .optional("iban")
                .searchable("industry")
                .free("note")
                .build(),
        )
        .schema(
            Schema::builder("Invoice")
                .required("invoice_number")
                .required("date")
                .required("total_amount")
                .required("currency")
                .required("customer_ref")
                .optional("due_date")
                .free("note")
                .build(),
        )
        .schema(
            Schema::builder("TaxDeclaration")
                .required("declaration_id")
                .required("tax_year")
                .required("customer_ref")
                .required("declaration_type")
                .optional("filing_date")
                .optional("status")
                .build(),
        )
        .link(LinkSpec::one_to_many("Customer", "issued", "Invoice"))
        .link(LinkSpec::one_to_many("Customer", "filed", "TaxDeclaration"))
        .action(ActionSpec::manual("approve", "Invoice", "status"))
        .action(ActionSpec::auto("classify", "Customer", "industry"))
        .action(ActionSpec::manual("submit", "TaxDeclaration", "status"))
        .build()
}

pub fn medcare_ontology() -> Ontology {
    Ontology::builder("medcare")
        .label(Label::new("medcare", "Medical Practice", "Arztpraxis"))
        .locale(Locale::De)
        .schema(
            Schema::builder("Patient")
                .required("patient_id")
                .required("name")
                .required("geburtsdatum")
                .optional("versichertennummer")
                .optional("krankenkasse")
                .optional("address")
                .free("note")
                .build(),
        )
        .schema(
            Schema::builder("Diagnosis")
                .required("icd10_code")
                .required("patient_ref")
                .required("date")
                .optional("description")
                .optional("severity")
                .build(),
        )
        .schema(
            Schema::builder("LabResult")
                .required("lab_id")
                .required("patient_ref")
                .required("parameter")
                .required("value")
                .required("unit")
                .optional("reference_range")
                .optional("date")
                .build(),
        )
        .schema(
            Schema::builder("Prescription")
                .required("prescription_id")
                .required("patient_ref")
                .required("medication")
                .required("dosage")
                .optional("duration")
                .optional("refills")
                .build(),
        )
        .link(LinkSpec::one_to_many(
            "Patient",
            "diagnosed_with",
            "Diagnosis",
        ))
        .link(LinkSpec::one_to_many("Patient", "lab_result", "LabResult"))
        .link(LinkSpec::one_to_many(
            "Patient",
            "prescribed",
            "Prescription",
        ))
        .link(LinkSpec::one_to_many(
            "Diagnosis",
            "confirmed_by",
            "LabResult",
        ))
        .action(ActionSpec::auto("triage", "Patient", "urgency"))
        .action(ActionSpec::suggested("prescribe", "Patient", "medication"))
        .action(ActionSpec::manual(
            "approve_prescription",
            "Prescription",
            "status",
        ))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smb_dto_german_display_name() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        assert_eq!(dto.name, "Steuerberatungskanzlei");
        assert_eq!(dto.key, "smb");
    }

    #[test]
    fn smb_dto_english_display_name() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::En);
        assert_eq!(dto.name, "Tax Practice");
    }

    #[test]
    fn smb_entity_types() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        assert_eq!(dto.entity_types.len(), 3);
        let customer = dto.entity_type("Customer").unwrap();
        assert_eq!(customer.id, 1);
        assert_eq!(customer.required_count, 2);
        assert_eq!(customer.properties.len(), 6);
    }

    #[test]
    fn smb_links_from_customer() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        let links = dto.links_from("Customer");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].predicate, "issued");
        assert_eq!(links[1].predicate, "filed");
    }

    #[test]
    fn smb_actions_for_invoice() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        let actions = dto.actions_for("Invoice");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].trigger, "manual");
    }

    #[test]
    fn medcare_dto_bilingual() {
        let ont = medcare_ontology();
        let de = OntologyDto::from_ontology(&ont, Locale::De);
        let en = OntologyDto::from_ontology(&ont, Locale::En);
        assert_eq!(de.name, "Arztpraxis");
        assert_eq!(en.name, "Medical Practice");
    }

    #[test]
    fn medcare_entity_types() {
        let ont = medcare_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        assert_eq!(dto.entity_types.len(), 4);
        let patient = dto.entity_type("Patient").unwrap();
        assert_eq!(patient.required_count, 3);
    }

    #[test]
    fn medcare_links() {
        let ont = medcare_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        assert_eq!(dto.link_types.len(), 4);
        let patient_links = dto.links_from("Patient");
        assert_eq!(patient_links.len(), 3);
    }

    #[test]
    fn medcare_actions() {
        let ont = medcare_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::De);
        assert_eq!(dto.action_types.len(), 3);
        let prescribe = dto.actions_for("Patient");
        assert_eq!(prescribe.len(), 2);
    }

    #[test]
    fn property_marking_exposed() {
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::En);
        let customer = dto.entity_type("Customer").unwrap();
        let tax_id = customer
            .properties
            .iter()
            .find(|p| p.key == "tax_id")
            .unwrap();
        assert_eq!(tax_id.marking, "internal");
        assert_eq!(tax_id.kind, "required");
    }

    #[test]
    fn entity_type_id_matches_dto_id() {
        use lance_graph_contract::ontology::entity_type_id;
        let ont = smb_ontology();
        let dto = OntologyDto::from_ontology(&ont, Locale::En);
        let customer_id = entity_type_id(&ont, "Customer");
        let dto_customer = dto.entity_type("Customer").unwrap();
        assert_eq!(customer_id, dto_customer.id);
    }

    #[test]
    fn env_var_api_key_pattern() {
        // Railway pattern: API key from env, never hardcoded
        let key = std::env::var("RAILWAY_API_KEY").unwrap_or_default();
        // In tests, key is empty (no Railway); in CI/Railway, it's set
        assert!(key.is_empty() || key.len() > 8);
    }
}
