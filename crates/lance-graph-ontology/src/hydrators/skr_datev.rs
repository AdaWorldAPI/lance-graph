//! Wiring for the two DATEV SKR schemes.
//!
//! - `hydrate_skr03` → `OGIT::SKR03_V1`, base IRI `urn:datev:skr03:account`.
//!   Reads `data/ontologies/skr-datev/skr03.csv` (canonical 4-digit accounts).
//! - `hydrate_skr04` → `OGIT::SKR04_V1`, base IRI `urn:datev:skr04:account`.
//!   Reads `data/ontologies/skr-datev/skr04.csv`.
//! - `hydrate_skr03_bau` → also `OGIT::SKR03_V1`, but with the 6-digit Bau
//!   variant. Test-only entry point; the default `hydrate_skr03` uses the
//!   canonical 4-digit form because cross-scheme alignment (FIBO, ZUGFeRD)
//!   anchors to 4-digit numbers.
//!
//! Both schemes declare `inherits_from: Some(OGIT::DOLCE_V1.0)` — accounts
//! are abstract economic objects, anchored to DUL Object via the cognitive
//! shader's downstream alignment axioms (not baked in here).

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::HydrateErr;
use super::skr::SkrHydrator;
use crate::registry::OntologyRegistry;

const SKR03_CSV_RELATIVE_PATH: &str = "data/ontologies/skr-datev/skr03.csv";
const SKR03_BAU_CSV_RELATIVE_PATH: &str = "data/ontologies/skr-datev/skr03-bau.csv";
const SKR04_CSV_RELATIVE_PATH: &str = "data/ontologies/skr-datev/skr04.csv";

pub const SKR03_IRI_PREFIX: &str = "urn:datev:skr03:account";
pub const SKR04_IRI_PREFIX: &str = "urn:datev:skr04:account";
pub const SKR03_BAU_IRI_PREFIX: &str = "urn:datev:skr03-bau:account";

/// Hydrate canonical SKR 03 (4-digit accounts) as `OGIT::SKR03_V1`.
pub fn hydrate_skr03(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_skr03_from(&skr03_csv_path(), registry)
}

pub fn hydrate_skr03_from(
    csv_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let h = SkrHydrator {
        g: OGIT::SKR03_V1.0,
        version: OGIT::SKR03_V1.1,
        domain_name: "skr03".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
        iri_prefix: SKR03_IRI_PREFIX.to_string(),
    };
    h.hydrate(csv_path, registry)?;
    Ok(OGIT::SKR03_V1.0)
}

/// Hydrate the SKR 03 Bau und Handwerk extension (6-digit accounts) into
/// `OGIT::SKR03_V1`. Uses a SEPARATE IRI prefix
/// (`urn:datev:skr03-bau:account`) so it doesn't clash with the canonical
/// 4-digit accounts hydrated via [`hydrate_skr03`]. Intended for callers
/// that need the trade-specific extensions explicitly.
pub fn hydrate_skr03_bau(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_skr03_bau_from(&skr03_bau_csv_path(), registry)
}

pub fn hydrate_skr03_bau_from(
    csv_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let h = SkrHydrator {
        g: OGIT::SKR03_V1.0,
        version: OGIT::SKR03_V1.1,
        domain_name: "skr03-bau".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
        iri_prefix: SKR03_BAU_IRI_PREFIX.to_string(),
    };
    h.hydrate(csv_path, registry)?;
    Ok(OGIT::SKR03_V1.0)
}

/// Hydrate SKR 04 (4-digit accounts, balance-sheet-oriented) as `OGIT::SKR04_V1`.
pub fn hydrate_skr04(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_skr04_from(&skr04_csv_path(), registry)
}

pub fn hydrate_skr04_from(
    csv_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let h = SkrHydrator {
        g: OGIT::SKR04_V1.0,
        version: OGIT::SKR04_V1.1,
        domain_name: "skr04".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
        iri_prefix: SKR04_IRI_PREFIX.to_string(),
    };
    h.hydrate(csv_path, registry)?;
    Ok(OGIT::SKR04_V1.0)
}

fn skr03_csv_path() -> PathBuf {
    workspace_relative(SKR03_CSV_RELATIVE_PATH)
}

fn skr03_bau_csv_path() -> PathBuf {
    workspace_relative(SKR03_BAU_CSV_RELATIVE_PATH)
}

fn skr04_csv_path() -> PathBuf {
    workspace_relative(SKR04_CSV_RELATIVE_PATH)
}

fn workspace_relative(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel)
}
