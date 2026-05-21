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
use super::xsd::{collect_xsd_files, XsdHydrator};
use crate::registry::OntologyRegistry;

const ZUGFERD_DIR_RELATIVE_PATH: &str = "data/ontologies/zugferd";

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
