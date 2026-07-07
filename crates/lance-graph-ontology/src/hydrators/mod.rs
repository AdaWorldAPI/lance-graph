//! Pattern D — Meta-Structure Hydration.
//!
//! Each per-ontology hydrator is *data + ~50 LOC of glue*, never a bespoke
//! crate. The generic [`OwlHydrator`] reads OWL/Turtle into a typed
//! [`ContextBundle`] keyed by an OGIT `G` slot; per-ontology glue picks the
//! parser, declares the `G` slot, names the parent (`inherits_from`), and
//! whitelists the cascade edge IRIs.
//!
//! Hydrators that ship today:
//! - [`hydrate_dolce`] — L1 upper ontology (DOLCE+DUL).
//! - [`hydrate_owltime`] — L2 universal temporal ontology (OWL-Time).
//! - [`hydrate_provo`] — L2 universal provenance ontology (PROV-O).
//! - [`hydrate_qudt`] — L2 quantities/units/dimensions (QUDT 2.1).
//! - [`hydrate_schemaorg`] — L3 commercial-web (schema.org).
//!
//! Hydrators that follow this scaffold (remaining bO-* series): SKOS,
//! FIBO-FND, FIBO-BE, AEC3PO, XBRL GL, IFRS, UBL, ISO 20022, SKR03/SKR04,
//! HGB / UStG, GoBD, XRechnung, ZUGFeRD. Each lands as one `hydrate_*()`
//! glue + one TTL artifact.

pub mod dolce;
pub mod dolce_odoo;
pub mod fibo;
pub mod odoo;
pub mod owl;
pub mod owltime;
pub mod provo;
pub mod qudt;
pub mod schemaorg;
pub mod schematron;
pub mod skos;
pub mod skr;
pub mod skr_datev;
pub mod xsd;
pub mod zugferd;

pub use dolce::{hydrate_dolce, hydrate_dolce_from, hydrate_dolce_from_many};
pub use dolce_odoo::{classify_odoo, DolceCategory};
pub use fibo::{hydrate_fibo_be, hydrate_fibo_be_from, hydrate_fibo_fnd, hydrate_fibo_fnd_from};
pub use odoo::{hydrate_odoo, hydrate_odoo_from};
pub use owl::{
    ContextBundle, EntityId, HydrateErr, MetaStructureHydrator, OntologySlot, OwlHydrator,
};
pub use owltime::{hydrate_owltime, hydrate_owltime_from};
pub use provo::{hydrate_provo, hydrate_provo_from};
pub use qudt::{hydrate_qudt, hydrate_qudt_from};
pub use schemaorg::{hydrate_schemaorg, hydrate_schemaorg_from};
pub use schematron::SchematronHydrator;
pub use skos::{hydrate_skos, hydrate_skos_from};
pub use skr::SkrHydrator;
pub use skr_datev::{
    hydrate_skr03, hydrate_skr03_bau, hydrate_skr03_bau_from, hydrate_skr03_from, hydrate_skr04,
    hydrate_skr04_from, SKR03_BAU_IRI_PREFIX, SKR03_IRI_PREFIX, SKR04_IRI_PREFIX,
};
pub use xsd::{collect_xsd_files, XsdHydrator};
pub use zugferd::{
    hydrate_zugferd, hydrate_zugferd_from, hydrate_zugferd_rules, hydrate_zugferd_rules_from,
};
