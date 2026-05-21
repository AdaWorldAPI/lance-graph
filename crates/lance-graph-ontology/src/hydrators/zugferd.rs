//! ZUGFeRD / Factur-X (EN 16931) hydration glue.
//!
//! L3 e-invoicing schemas — the German hybrid PDF/A-3 + XML invoice format
//! aligned with the EU directive's EN 16931 reference data model. ZUGFeRD
//! and Factur-X share the same UN/CEFACT Cross-Industry Invoice (CII) XML
//! schema; this hydrator targets the EN16931 profile (the EU-aligned one).
//!
//! Ships as XSD, not OWL — so this hydrator uses [`super::xsd::XsdHydrator`]
//! rather than [`super::owl::OwlHydrator`]. The same minimal-name-extraction
//! pattern applies: every named `xs:element` / `xs:complexType` / etc. in
//! the CII schemas is interned as `{targetNamespace}#{name}`, giving the
//! cognitive shader a stable u32 entity id for every CII concept.
//!
//! Declares `inherits_from: Some(OGIT::DOLCE_V1.0)`. Alignment to upstream
//! ontologies (FIBO MonetaryAmount, DUL Description, schema.org Invoice)
//! is left as an explicit downstream assertion, not baked into the XSD.

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::HydrateErr;
use super::schematron::SchematronHydrator;
use super::xsd::{collect_xsd_files, XsdHydrator};
use crate::registry::OntologyRegistry;

const ZUGFERD_DIR_RELATIVE_PATH: &str = "data/ontologies/zugferd";
const ZUGFERD_SCH_RELATIVE_PATH: &str = "data/ontologies/zugferd/FACTUR-X_EN16931.sch";

/// Stable URN prefix for the Factur-X EN16931 Schematron rule namespace.
/// Used as the base for every interned rule / assert / report / pattern
/// IRI. The string is not a URL — it's a stable identifier that downstream
/// alignment can map to PEPPOL / EN16931 / FeRD rule registries.
const ZUGFERD_SCH_BASE_IRI: &str = "urn:schematron:factur-x-1.08-en16931";

/// Cascade edge whitelist for ZUGFeRD/CII. XSD doesn't have OWL-style
/// `rdfs:subClassOf` edges natively — the type hierarchy is implicit in
/// `xs:extension base="..."` references which this minimal hydrator does
/// NOT resolve into explicit edges (see the xsd.rs module docs for the
/// deferred follow-up). We register the canonical CII relational
/// predicates so downstream consumers can attach business-level edges via
/// `OntologyRegistry::register_edge_types` extensions later.
const ZUGFERD_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    // CII relational containers — the cognitive-shader traversal will hop
    // through these when projecting an invoice into the SPO graph.
    "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100#ExchangedDocument",
    "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100#ExchangedDocumentContext",
    "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100#SupplyChainTradeTransaction",
    // Trade-party / agreement / delivery / settlement predicates
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#ApplicableHeaderTradeAgreement",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#ApplicableHeaderTradeDelivery",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#ApplicableHeaderTradeSettlement",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#IncludedSupplyChainTradeLineItem",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SellerTradeParty",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#BuyerTradeParty",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#PostalTradeAddress",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedTradeTax",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedTradePaymentTerms",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedTradeSettlementHeaderMonetarySummation",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedLineTradeAgreement",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedLineTradeDelivery",
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100#SpecifiedLineTradeSettlement",
];

/// Hydrate the ZUGFeRD / Factur-X EN16931 profile as `OGIT::ZUGFERD_V1`.
///
/// Walks every `.xsd` file under `data/ontologies/zugferd/`, interns the
/// CII type / element / attribute names, and registers the cascade
/// whitelist of relational predicates.
pub fn hydrate_zugferd(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_zugferd_from(&zugferd_dir(), registry)
}

/// Test-friendly variant: hydrate ZUGFeRD from an explicit directory.
pub fn hydrate_zugferd_from(
    dir: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let files = collect_xsd_files(dir)?;
    let path_refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    let hydrator = XsdHydrator {
        g: OGIT::ZUGFERD_V1.0,
        version: OGIT::ZUGFERD_V1.1,
        domain_name: "zugferd".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(&path_refs, registry)?;
    registry
        .register_edge_types(OGIT::ZUGFERD_V1.0, ZUGFERD_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::ZUGFERD_V1.0,
            reason,
        })?;
    Ok(OGIT::ZUGFERD_V1.0)
}

fn zugferd_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(ZUGFERD_DIR_RELATIVE_PATH)
}

/// Hydrate the ZUGFeRD / Factur-X EN16931 Schematron business rules as
/// `OGIT::ZUGFERDRULES_V1`. Walks `FACTUR-X_EN16931.sch` and interns:
///
/// - Every `<assert id="FX-SCH-A-...">` and `<report id="FX-SCH-R-...">` as
///   `urn:schematron:factur-x-1.08-en16931/{assert,report}/{id}`
/// - Every `<pattern id="...">` (if `@id` present) as
///   `urn:schematron:factur-x-1.08-en16931/pattern/{id}`
/// - Every bracketed EN16931 / PEPPOL / CO / DE business-rule ID found in
///   the message text (e.g. `[BR-52]`, `[BR-CO-03]`, `[BR-DE-1]`,
///   `[PEPPOL-EN16931-R008]`) as
///   `urn:schematron:factur-x-1.08-en16931/rule/{rule-id}`
///
/// Declares `inherits_from: Some(OGIT::ZUGFERD_V1.0)` — the rule namespace
/// is meaningless without the structural CII namespace.
pub fn hydrate_zugferd_rules(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_zugferd_rules_from(&zugferd_sch_path(), registry)
}

pub fn hydrate_zugferd_rules_from(
    sch_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = SchematronHydrator {
        g: OGIT::ZUGFERDRULES_V1.0,
        version: OGIT::ZUGFERDRULES_V1.1,
        domain_name: "zugferd-rules".to_string(),
        inherits_from: Some(OGIT::ZUGFERD_V1.0),
        starting_entity_id: 100,
        base_iri: ZUGFERD_SCH_BASE_IRI.to_string(),
    };
    hydrator.hydrate_many(&[sch_path], registry)?;
    Ok(OGIT::ZUGFERDRULES_V1.0)
}

fn zugferd_sch_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(ZUGFERD_SCH_RELATIVE_PATH)
}
