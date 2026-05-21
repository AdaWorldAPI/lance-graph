//! QUDT (`http://qudt.org/2.1/schema/qudt`) hydration glue.
//!
//! L2 universal quantities / units / dimensions ontology — defines
//! `Quantity`, `QuantityKind`, `Unit`, `QuantityValue`, `Dimension`,
//! `SystemOfUnits`, plus 2900+ unit individuals and the SI quantity-kind
//! catalogue.
//!
//! Multi-file hydration: QUDT ships as separate TTL artifacts (core schema,
//! units catalogue, quantitykinds catalogue). All three merge into a single
//! `ContextBundle` keyed by `OGIT::QUDT_V1.0` via
//! [`OwlHydrator::hydrate_many`].

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, OwlHydrator};
use crate::registry::OntologyRegistry;

const QUDT_CORE_RELATIVE_PATH: &str = "data/ontologies/qudt-core.ttl";
const QUDT_UNITS_RELATIVE_PATH: &str = "data/ontologies/qudt-units.ttl";

const QUDT_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    // Quantity-Kind / Unit binding
    "http://qudt.org/schema/qudt/hasQuantityKind",
    "http://qudt.org/schema/qudt/quantityKind",
    "http://qudt.org/schema/qudt/hasUnit",
    "http://qudt.org/schema/qudt/unit",
    "http://qudt.org/schema/qudt/applicableUnit",
    // Dimensional structure
    "http://qudt.org/schema/qudt/hasDimensionVector",
    "http://qudt.org/schema/qudt/dimensionVectorForSI",
    // Conversion / scaling
    "http://qudt.org/schema/qudt/conversionMultiplier",
    "http://qudt.org/schema/qudt/conversionOffset",
    "http://qudt.org/schema/qudt/hasFactorUnit",
    // System-of-units association
    "http://qudt.org/schema/qudt/hasUnitSystem",
    "http://qudt.org/schema/qudt/systemDerivedQuantityKinds",
    "http://qudt.org/schema/qudt/baseUnitOfSystem",
    // Value carriers
    "http://qudt.org/schema/qudt/numericValue",
    "http://qudt.org/schema/qudt/value",
    "http://qudt.org/schema/qudt/quantityValue",
];

/// Hydrate QUDT as `OGIT::QUDT_V1` (L2 quantities/units/dimensions).
///
/// Hydrates the core schema + units catalogue. The quantitykinds catalogue
/// can be added via [`hydrate_qudt_from`] when its file is available.
pub fn hydrate_qudt(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let core = qudt_core_path();
    let units = qudt_units_path();
    hydrate_qudt_from(&[&core, &units], registry)
}

/// Test-friendly variant: hydrate QUDT from explicit paths. Accepts a slice
/// of TTL paths so callers can compose `[core, units, quantitykinds, ...]`.
pub fn hydrate_qudt_from(
    ttl_paths: &[&Path],
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::QUDT_V1.0,
        version: OGIT::QUDT_V1.1,
        domain_name: "qudt".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(ttl_paths, registry)?;
    registry
        .register_edge_types(OGIT::QUDT_V1.0, QUDT_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::QUDT_V1.0,
            reason,
        })?;
    Ok(OGIT::QUDT_V1.0)
}

fn qudt_core_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(QUDT_CORE_RELATIVE_PATH)
}

fn qudt_units_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(QUDT_UNITS_RELATIVE_PATH)
}
