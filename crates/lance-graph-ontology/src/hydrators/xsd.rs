//! Generic XSD → `ContextBundle` hydrator.
//!
//! Where [`super::owl::OwlHydrator`] handles OWL ontologies in Turtle and
//! RDF/XML serializations, `XsdHydrator` handles XML Schema (XSD) artifacts.
//! Used by ZUGFeRD / Factur-X (this PR), and intended for UBL, ISO 20022,
//! and any other e-business schema delivered as XSD.
//!
//! Pattern D, minimal-name-extraction shape:
//!
//! - Walk every `<xs:element name="...">`, `<xs:complexType name="...">`,
//!   `<xs:simpleType name="...">`, `<xs:attribute name="...">` and
//!   `<xs:attributeGroup name="...">` element in each input file.
//! - Compose each named declaration into a stable IRI by joining the
//!   nearest `xs:schema/@targetNamespace` with `#{name}` (e.g.
//!   `urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100#CrossIndustryInvoice`).
//! - Intern each IRI exactly once into the [`super::owl::OntologySlot`]
//!   keyed by the chosen `G` slot.
//!
//! Out of scope for this PR (deferred to a full XSD type-graph projection
//! follow-up):
//!
//! - `<xs:extension base="...">` / `<xs:restriction base="...">` resolution
//!   into rdfs:subClassOf-equivalent edges.
//! - `<xs:element ref="...">` cross-references between files.
//! - Substitution groups, attribute groups beyond name interning.
//! - Schematron / business rules — those live in `.sch` files alongside the
//!   `.xsd` files and would be a separate Schematron hydrator.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use quick_xml::events::Event;
use quick_xml::Reader;

use super::owl::{ContextBundle, EntityId, HydrateErr, OntologySlot};
use crate::registry::OntologyRegistry;

/// Closure type for the IRI interning callback. Factored out to keep
/// `walk_xsd`'s signature inside clippy's `type_complexity` budget.
type InternFn<'a> = &'a mut dyn FnMut(
    String,
    &mut HashMap<String, EntityId>,
    &mut EntityId,
) -> EntityId;

/// XSD hydrator. Reusable for every XSD-shaped business-document schema.
pub struct XsdHydrator {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    pub starting_entity_id: EntityId,
}

impl XsdHydrator {
    /// Multi-file variant. Walks every input XSD, interns named element /
    /// type / attribute / attributeGroup declarations as
    /// `{targetNamespace}#{name}` IRIs, and registers a [`ContextBundle`]
    /// keyed by `self.g`.
    pub fn hydrate_many(
        &self,
        xsd_paths: &[&Path],
        registry: &OntologyRegistry,
    ) -> Result<u32, HydrateErr> {
        let mut iri_to_id: HashMap<String, EntityId> = HashMap::new();
        let mut next_id: EntityId = self.starting_entity_id;

        let mut intern = |iri: String,
                          map: &mut HashMap<String, EntityId>,
                          n: &mut EntityId|
         -> EntityId {
            if let Some(&id) = map.get(&iri) {
                return id;
            }
            let id = *n;
            *n += 1;
            map.insert(iri, id);
            id
        };

        for path in xsd_paths {
            let bytes = fs::read(path).map_err(|e| HydrateErr::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            walk_xsd(path, &bytes, &mut iri_to_id, &mut next_id, &mut intern)?;
        }

        let entity_count = iri_to_id.len() as u32;
        let ontology = OntologySlot {
            entity_count,
            iri_to_id,
        };
        let bundle = ContextBundle {
            g: self.g,
            version: self.version,
            domain_name: self.domain_name.clone(),
            inherits_from: self.inherits_from,
            ontology: Some(Arc::new(ontology)),
            edge_types: Vec::new(),
        };
        registry.register_bundle(bundle);
        Ok(self.g)
    }
}

/// Walk one XSD file's events, finding the schema's `targetNamespace` and
/// interning every named declaration found beneath it.
fn walk_xsd(
    path: &Path,
    bytes: &[u8],
    iri_to_id: &mut HashMap<String, EntityId>,
    next_id: &mut EntityId,
    intern: InternFn<'_>,
) -> Result<(), HydrateErr> {
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut target_ns: Option<String> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => {
                return Err(HydrateErr::Parse {
                    path: path.to_path_buf(),
                    message: format!("XSD: {e}"),
                });
            }
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                // `e.name()` returns a borrowed view tied to `e`; capture
                // the local-name bytes into an owned slice to avoid the
                // temporary-borrow issue under quick-xml 0.37.
                let qname = e.name();
                let local: Vec<u8> = local_name(qname.as_ref()).to_vec();
                if local == b"schema" && target_ns.is_none() {
                    target_ns = attr_value(&e, b"targetNamespace");
                } else if matches!(
                    local.as_slice(),
                    b"element" | b"complexType" | b"simpleType" | b"attribute" | b"attributeGroup"
                ) {
                    if let Some(name) = attr_value(&e, b"name") {
                        let ns = target_ns.as_deref().unwrap_or("");
                        let iri = if ns.is_empty() {
                            // Schema with no targetNamespace: fall back to
                            // the bare local name so the IRI is still
                            // resolvable (used only by exotic XSDs; CII
                            // always declares a targetNamespace).
                            name
                        } else {
                            format!("{ns}#{name}")
                        };
                        intern(iri, iri_to_id, next_id);
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(())
}

fn local_name(qname: &[u8]) -> &[u8] {
    match qname.iter().position(|b| *b == b':') {
        Some(i) => &qname[i + 1..],
        None => qname,
    }
}

fn attr_value(e: &quick_xml::events::BytesStart, attr_name: &[u8]) -> Option<String> {
    for attr in e.attributes().with_checks(false).flatten() {
        if local_name(attr.key.as_ref()) == attr_name {
            return std::str::from_utf8(attr.value.as_ref())
                .ok()
                .map(|s| s.to_string());
        }
    }
    None
}

/// Recursively collect every `.xsd` file under `root`, sorted for stable
/// IRI-interning order.
pub fn collect_xsd_files(root: &Path) -> Result<Vec<PathBuf>, HydrateErr> {
    let mut out: Vec<PathBuf> = Vec::new();
    walk_dir(root, &mut out).map_err(|e| HydrateErr::Io {
        path: root.to_path_buf(),
        source: e,
    })?;
    out.sort();
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("XSD directory not found: {}", dir.display()),
        ));
    }
    if !dir.is_dir() {
        return Ok(());
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().map(|s| s == "xsd").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY_XSD: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="urn:example:test">
  <xs:element name="Foo" type="xs:string"/>
  <xs:complexType name="BarType">
    <xs:sequence>
      <xs:element name="baz" type="xs:int"/>
    </xs:sequence>
  </xs:complexType>
  <xs:simpleType name="Quux"/>
  <xs:attribute name="anAttr" type="xs:string"/>
</xs:schema>
"#;

    #[test]
    fn tiny_xsd_interns_named_declarations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.xsd");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_XSD.as_bytes()).unwrap();
        drop(f);

        let reg = OntologyRegistry::new_in_memory();
        let h = XsdHydrator {
            g: 9999,
            version: 1,
            domain_name: "tiny-xsd".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
        };
        h.hydrate_many(&[&path], &reg).expect("hydrate tiny XSD");

        let bundle = reg.bundle_for(9999).expect("bundle");
        // Foo (element) + BarType (complexType) + baz (nested element) +
        // Quux (simpleType) + anAttr (attribute) = 5 names.
        assert!(bundle.entity_count() >= 5);
        assert!(bundle
            .resolve_iri("urn:example:test#Foo")
            .is_some());
        assert!(bundle
            .resolve_iri("urn:example:test#BarType")
            .is_some());
        assert!(bundle
            .resolve_iri("urn:example:test#baz")
            .is_some());
        assert!(bundle
            .resolve_iri("urn:example:test#anAttr")
            .is_some());
    }
}
