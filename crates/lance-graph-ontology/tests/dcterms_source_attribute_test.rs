//! D-ONTO-V5-1: per-attribute `dcterms:source` provenance.
//!
//! After `AdaWorldAPI/OGIT#2` merged, every per-attribute predicate in a
//! WorkOrder entity TTL carries its own `dcterms:source` literal pointing
//! at the WoA Python source line that defines the column (e.g.
//! `ogit.WorkOrder:fahrtKm` → `"AdaWorldAPI/WoA/models.py:Customer.fahrt_km"`).
//!
//! This test hydrates the merged Customer.ttl from `/home/user/OGIT` and
//! asserts that the parser extracts every `(predicate_iri, source_uri)`
//! pair through the new `ProvenanceBundle` sibling structure on
//! `proposal.rs`. Closes ledger row TTL-PROBE-5 (Wave 1 extraction half;
//! Wave 3 will thread the pairs into a new `MappingRow` column).

use lance_graph_ontology::proposal::ProvenanceBundle;
use lance_graph_ontology::ttl_parse::TtlSource;
use std::path::Path;

/// Locate the OGIT WorkOrder Customer.ttl that OGIT#2 just merged.
/// `OGIT_FORK_PATH` overrides; otherwise defaults to `/home/user/OGIT`.
fn customer_ttl_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("OGIT_FORK_PATH") {
        let p = Path::new(&p).join("NTO/WorkOrder/entities/Customer.ttl");
        if p.exists() {
            return Some(p);
        }
    }
    let canonical = Path::new("/home/user/OGIT/NTO/WorkOrder/entities/Customer.ttl");
    if canonical.exists() {
        return Some(canonical.to_path_buf());
    }
    None
}

#[test]
fn dcterms_source_attribute_pairs_surface_for_customer() {
    let Some(path) = customer_ttl_path() else {
        eprintln!("SKIP: Customer.ttl not found at /home/user/OGIT (set OGIT_FORK_PATH)");
        return;
    };

    let src = TtlSource::from_path(&path).expect("Customer.ttl must read");
    let bundles = src
        .parse_provenance()
        .expect("Customer.ttl must parse for provenance");

    // The entity bundle is keyed by the entity URI.
    let customer = bundles
        .iter()
        .find(|b| b.entity_uri == "ogit.WorkOrder:Customer")
        .expect("Customer entity bundle must be present");

    // Entity-level dcterms:source must point at the Python source.
    assert_eq!(
        customer.entity_source_uri, "AdaWorldAPI/WoA/models.py:Customer",
        "entity-level dcterms:source must verbatim-match the OGIT#2 literal"
    );

    // Customer.ttl declares 18 per-attribute dcterms:source lines pointing
    // at concrete Customer.<field> entries (kdnr, firma, vorname, nachname,
    // anrede, mailAnrede, telefon, strasse, adresszusatz, plz, ort,
    // zahlungsziel, stundensatz, fahrtKm, fahrtKosten, notizen, aktiv,
    // createdAt) plus 2 attributes whose dcterms:source falls back to the
    // entity (iban, taxId). The plan-doc lower bound is 8 attribute pairs;
    // we assert >= 8 so the test still passes if a future TTL trim drops
    // some attributes, while logging the actual count for visibility.
    assert!(
        customer.attribute_count() >= 8,
        "expected at least 8 per-attribute dcterms:source pairs, got {}",
        customer.attribute_count()
    );

    // Spot-check a representative pair from the OGIT#2 task description.
    let fahrt_km = customer
        .source_for("ogit.WorkOrder:fahrtKm")
        .expect("fahrtKm must carry per-attribute provenance");
    assert_eq!(
        fahrt_km, "AdaWorldAPI/WoA/models.py:Customer.fahrt_km",
        "fahrtKm must point at the snake_case Python column name"
    );

    // Every recorded source URI must be non-empty and prefixed with the
    // expected WoA path. This is the registry-exposed shape the Wave 3
    // column extension will persist.
    for pair in &customer.attribute_sources {
        assert!(
            pair.predicate_iri.starts_with("ogit.WorkOrder:"),
            "predicate_iri must be canonical OGIT URI form: {pair:?}"
        );
        assert!(
            !pair.source_uri.is_empty(),
            "source_uri must not be empty: {pair:?}"
        );
        assert!(
            pair.source_uri.starts_with("AdaWorldAPI/WoA/models.py:"),
            "source_uri must point at the WoA Python source: {pair:?}"
        );
    }
}

#[test]
fn provenance_bundle_lookup_is_consistent() {
    // Synthetic, in-memory TTL that mirrors the OGIT#2 shape — kept here so
    // the test gate doesn't depend on /home/user/OGIT being checked out.
    const FIXTURE: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.Synth:             <http://www.purl.org/ogit/Synth/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms:                <http://purl.org/dc/terms/> .

ogit.Synth:Item
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Item";
    dcterms:source "synth/source.py:Item";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit.Synth:alpha ) ;
    ogit:optional-attributes ( ogit.Synth:beta ogit.Synth:gamma ) ;
.

ogit.Synth:alpha
    a rdfs:Property ;
    rdfs:label "alpha" ;
    dcterms:source "synth/source.py:Item.alpha" .

ogit.Synth:beta
    a rdfs:Property ;
    rdfs:label "beta" ;
    dcterms:source "synth/source.py:Item.beta" .

ogit.Synth:gamma
    a rdfs:Property ;
    rdfs:label "gamma" .
"#;

    let src = TtlSource::from_bytes(
        std::path::PathBuf::from("synth.ttl"),
        FIXTURE.as_bytes().to_vec(),
    );
    let bundles: Vec<ProvenanceBundle> = src.parse_provenance().expect("synth TTL must parse");
    let item = bundles
        .iter()
        .find(|b| b.entity_uri == "ogit.Synth:Item")
        .expect("Item bundle must be present");
    assert_eq!(item.entity_source_uri, "synth/source.py:Item");
    // alpha + beta carry dcterms:source; gamma does not — it must be
    // omitted (NOT recorded with empty source).
    assert_eq!(item.attribute_count(), 2);
    assert_eq!(
        item.source_for("ogit.Synth:alpha"),
        Some("synth/source.py:Item.alpha")
    );
    assert_eq!(
        item.source_for("ogit.Synth:beta"),
        Some("synth/source.py:Item.beta")
    );
    assert!(item.source_for("ogit.Synth:gamma").is_none());
}
