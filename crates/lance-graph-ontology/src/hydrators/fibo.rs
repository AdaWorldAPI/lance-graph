//! FIBO (Financial Industry Business Ontology) hydration glue.
//!
//! FIBO ships as ~250 modular RDF/XML files organized into module suites:
//!
//! - **FND** (Foundations, `OGIT::FIBOFND_V1`): legal persons, parties,
//!   contracts, addresses, dates, identifiers, currency amounts — the
//!   ontology layer every higher FIBO module sits on.
//! - **BE** (Business Entities, `OGIT::FIBOBE_V1`): corporations, LLCs,
//!   partnerships, trusts, cooperatives, ownership / control structures.
//!
//! Both modules ship as `.rdf` files (RDF/XML, not Turtle). The generic
//! `OwlHydrator` already dispatches by extension via `detect_format` and
//! routes RDF/XML through `oxrdfxml`, so per-FIBO glue is again ~50 LOC.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, OwlHydrator};
use crate::registry::OntologyRegistry;

const FIBO_FND_RELATIVE_PATH: &str = "data/ontologies/fibo-fnd";
const FIBO_BE_RELATIVE_PATH: &str = "data/ontologies/fibo-be";

/// FIBO-FND cascade whitelist. Covers the foundational predicates used
/// by every downstream FIBO module: subsumption, the FIBO Commons
/// has*/is* pattern, and dcterms metadata.
const FIBO_FND_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    // FIBO Commons (OMG) — every FIBO module declares against these.
    "https://www.omg.org/spec/Commons/Relations/hasPart",
    "https://www.omg.org/spec/Commons/Relations/isPartOf",
    "https://www.omg.org/spec/Commons/Classifiers/isClassifiedBy",
    "https://www.omg.org/spec/Commons/Classifiers/classifies",
    "https://www.omg.org/spec/Commons/Identifiers/hasIdentifier",
    "https://www.omg.org/spec/Commons/Identifiers/identifies",
    "https://www.omg.org/spec/Commons/Designators/isDesignatedBy",
    "https://www.omg.org/spec/Commons/Designators/designates",
    "https://www.omg.org/spec/Commons/DatesAndTimes/hasDate",
    "https://www.omg.org/spec/Commons/DatesAndTimes/hasObservedDateTime",
    // Party / agreement
    "https://spec.edmcouncil.org/fibo/ontology/FND/Parties/Parties/hasParty",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Parties/Parties/hasPartyInRole",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Agreements/hasContractParty",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Contracts/hasContractualElement",
    // Address / location
    "https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/hasAddress",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Places/Locations/isLocatedAt",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Places/Locations/hasLocation",
    // Currency / amounts
    "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasMonetaryAmount",
    "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasAmount",
];

/// FIBO-BE cascade whitelist. Inherits the FND surface and adds the
/// ownership / control / corporate-structure predicates.
const FIBO_BE_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    // Ownership / control
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Ownership/hasOwnership",
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Ownership/hasOwner",
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Ownership/isOwnedBy",
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateControl/hasControl",
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateControl/controls",
    "https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateControl/isControlledBy",
    // Corporate body / governance
    "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/CorporateBodies/hasGoverningBody",
    "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/CorporateBodies/hasOfficer",
    "https://spec.edmcouncil.org/fibo/ontology/BE/Corporations/Corporations/hasShareholder",
    // Partner / trustee / member roles
    "https://spec.edmcouncil.org/fibo/ontology/BE/Partnerships/Partnerships/hasPartner",
    "https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/hasTrustee",
    "https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/hasBeneficiary",
    "https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/hasSettlor",
    // FND-inherited (so a BE-only cascade traversal still reaches the
    // foundation predicates)
    "https://www.omg.org/spec/Commons/Relations/hasPart",
    "https://www.omg.org/spec/Commons/Identifiers/hasIdentifier",
];

/// Hydrate FIBO-FND as `OGIT::FIBOFND_V1` (L3 finance/business foundation).
pub fn hydrate_fibo_fnd(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_fibo_fnd_from(&fibo_fnd_dir(), registry)
}

/// Test-friendly variant: hydrate FIBO-FND from an explicit directory.
pub fn hydrate_fibo_fnd_from(
    dir: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let files = collect_rdf_files(dir)?;
    let path_refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    let hydrator = OwlHydrator {
        g: OGIT::FIBOFND_V1.0,
        version: OGIT::FIBOFND_V1.1,
        domain_name: "fibofnd".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(&path_refs, registry)?;
    registry
        .register_edge_types(OGIT::FIBOFND_V1.0, FIBO_FND_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::FIBOFND_V1.0,
            reason,
        })?;
    Ok(OGIT::FIBOFND_V1.0)
}

/// Hydrate FIBO-BE as `OGIT::FIBOBE_V1` (L3 business entities).
pub fn hydrate_fibo_be(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_fibo_be_from(&fibo_be_dir(), registry)
}

/// Test-friendly variant: hydrate FIBO-BE from an explicit directory.
pub fn hydrate_fibo_be_from(
    dir: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let files = collect_rdf_files(dir)?;
    let path_refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    let hydrator = OwlHydrator {
        g: OGIT::FIBOBE_V1.0,
        version: OGIT::FIBOBE_V1.1,
        domain_name: "fibobe".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(&path_refs, registry)?;
    registry
        .register_edge_types(OGIT::FIBOBE_V1.0, FIBO_BE_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::FIBOBE_V1.0,
            reason,
        })?;
    Ok(OGIT::FIBOBE_V1.0)
}

/// Recursively collect every `.rdf` file under `root`, sorted for stable
/// IRI-interning order.
fn collect_rdf_files(root: &Path) -> Result<Vec<PathBuf>, HydrateErr> {
    let mut out: Vec<PathBuf> = Vec::new();
    walk(root, &mut out).map_err(|e| HydrateErr::Io {
        path: root.to_path_buf(),
        source: e,
    })?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if !d.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("FIBO directory not found: {}", d.display()),
            ));
        }
        if !d.is_dir() {
            return Ok(());
        }
        for entry in fs::read_dir(&d)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension() == Some(OsStr::new("rdf")) {
                out.push(p);
            }
        }
    }
    Ok(())
}

fn fibo_fnd_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(FIBO_FND_RELATIVE_PATH)
}

fn fibo_be_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(FIBO_BE_RELATIVE_PATH)
}
