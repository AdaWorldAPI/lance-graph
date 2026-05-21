//! OWL-Time (W3C `http://www.w3.org/2006/time#`) hydration glue.
//!
//! L2 universal temporal ontology — defines `Instant`, `Interval`,
//! `ProperInterval`, `DateTimeDescription`, `Duration`, and the Allen 13
//! interval relations (`before`, `after`, `meets`, `overlaps`, …).
//!
//! Declares `inherits_from: Some(OGIT::DOLCE_V1.0)` and aligns with DUL via
//! `time:TemporalEntity ⊑ dul:Event` (an axiom downstream consumers can
//! assert when they need cross-G traversal).

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, MetaStructureHydrator, OwlHydrator};
use crate::registry::OntologyRegistry;

const TIME_TTL_RELATIVE_PATH: &str = "data/ontologies/time.ttl";

const TIME_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    // Interval-positional (Allen relations)
    "http://www.w3.org/2006/time#before",
    "http://www.w3.org/2006/time#after",
    "http://www.w3.org/2006/time#intervalBefore",
    "http://www.w3.org/2006/time#intervalAfter",
    "http://www.w3.org/2006/time#intervalMeets",
    "http://www.w3.org/2006/time#intervalMetBy",
    "http://www.w3.org/2006/time#intervalOverlaps",
    "http://www.w3.org/2006/time#intervalOverlappedBy",
    "http://www.w3.org/2006/time#intervalStarts",
    "http://www.w3.org/2006/time#intervalStartedBy",
    "http://www.w3.org/2006/time#intervalFinishes",
    "http://www.w3.org/2006/time#intervalFinishedBy",
    "http://www.w3.org/2006/time#intervalDuring",
    "http://www.w3.org/2006/time#intervalContains",
    "http://www.w3.org/2006/time#intervalEquals",
    "http://www.w3.org/2006/time#intervalIn",
    // Interval boundaries
    "http://www.w3.org/2006/time#hasBeginning",
    "http://www.w3.org/2006/time#hasEnd",
    "http://www.w3.org/2006/time#hasTime",
    "http://www.w3.org/2006/time#hasDuration",
    "http://www.w3.org/2006/time#hasTemporalDuration",
];

/// Hydrate OWL-Time as `OGIT::TIME_V1` (L2 universal temporal ontology).
pub fn hydrate_owltime(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_owltime_from(&time_ttl_path(), registry)
}

/// Test-friendly variant: hydrate OWL-Time from an explicit path.
pub fn hydrate_owltime_from(
    ttl_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::TIME_V1.0,
        version: OGIT::TIME_V1.1,
        domain_name: "owltime".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate(ttl_path, registry)?;
    registry
        .register_edge_types(OGIT::TIME_V1.0, TIME_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::TIME_V1.0,
            reason,
        })?;
    Ok(OGIT::TIME_V1.0)
}

fn time_ttl_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(TIME_TTL_RELATIVE_PATH)
}
