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
pub mod fibo;
pub mod owl;
pub mod owltime;
pub mod provo;
pub mod qudt;
pub mod schemaorg;
pub mod skos;
pub mod xsd;
pub mod zugferd;

pub use dolce::{hydrate_dolce, hydrate_dolce_from, hydrate_dolce_from_many};
pub use fibo::{hydrate_fibo_be, hydrate_fibo_be_from, hydrate_fibo_fnd, hydrate_fibo_fnd_from};
pub use owl::{ContextBundle, EntityId, HydrateErr, MetaStructureHydrator, OntologySlot, OwlHydrator};
pub use owltime::{hydrate_owltime, hydrate_owltime_from};
pub use provo::{hydrate_provo, hydrate_provo_from};
pub use qudt::{hydrate_qudt, hydrate_qudt_from};
pub use schemaorg::{hydrate_schemaorg, hydrate_schemaorg_from};
pub use skos::{hydrate_skos, hydrate_skos_from};
pub use xsd::{collect_xsd_files, XsdHydrator};
pub use zugferd::{hydrate_zugferd, hydrate_zugferd_from};
