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
//!
//! Hydrators that follow this scaffold (PR-bO-2 through PR-bO-16):
//! OWL-Time, PROV-O, QUDT, SKOS, FIBO-FND, FIBO-BE, schema.org subset, XBRL
//! GL, IFRS, UBL, ISO 20022, SKR03/SKR04, HGB / UStG, GoBD, XRechnung,
//! ZUGFeRD. Each lands as one `hydrate_*()` glue + one TTL artifact.

pub mod dolce;
pub mod owl;

pub use dolce::{hydrate_dolce, hydrate_dolce_from};
pub use owl::{ContextBundle, EntityId, HydrateErr, MetaStructureHydrator, OntologySlot, OwlHydrator};
