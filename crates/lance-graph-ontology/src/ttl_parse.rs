//! TTL → MappingProposal pipeline.
//!
//! Walks an OGIT-shaped TTL directory, parses each file via `oxttl`, and
//! emits `MappingProposal`s for every `ogit:Entity` subclass it finds. Verbs
//! (subclasses of `ogit:Verb`) become edge proposals; standalone attributes
//! (`owl:DatatypeProperty`) become attribute proposals.
//!
//! Lists declared via `( a b c )` syntax are stored by oxttl as RDF lists
//! (`rdf:first` / `rdf:rest` / `rdf:nil`). We re-assemble those after
//! collecting triples.
//!
//! ## What this parser handles today
//!
//! - Entities: `<X> a rdfs:Class; rdfs:subClassOf ogit:Entity` → entity
//!   proposal with `Schema` derived from `ogit:mandatory-attributes` (→
//!   `Schema::required(...)`) and `ogit:optional-attributes` (→
//!   `Schema::optional(...)`).
//! - Verbs: `<X> a rdfs:Class; rdfs:subClassOf ogit:Verb` → edge proposal.
//!   The `LinkSpec` is built with placeholder subject/object types (the
//!   `ogit:from-to` / `ogit:allowed` constraints are not yet expanded; they
//!   round-trip via the dictionary `source_uri` for now).
//! - Attributes: `<X> a owl:DatatypeProperty` → attribute proposal with the
//!   `SemanticType` looked up from `semantic_types.toml`.
//!
//! ## What it does NOT yet handle
//!
//! - Bilingual labels (`@de`/`@en` annotations).
//! - The detailed `ogit:allowed` constraint blocks for verbs (those become
//!   edge proposals with `LinkSpec::one_to_many` placeholders).
//! - Anything past Turtle (RDF/XML, N-Quads, etc.) — that is the remit of
//!   the future `lance-graph-rdf` crate.
//!
//! Carrier-method doctrine: `TtlSource` carries the parsing logic; the
//! free `parse_ttl_directory` is a thin convenience wrapper.

use crate::error::{Error, Result};
use crate::namespace::OgitUri;
use crate::proposal::{HydrationFailure, MappingProposal, MappingProposalKind};
use crate::semantic_types::SemanticTypeMap;
use lance_graph_contract::cam::CodecRoute;
use lance_graph_contract::property::{Cardinality, LinkSpec, Marking, PropertySpec, Schema};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const OGIT_BASE: &str = "http://www.purl.org/ogit/";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
const OGIT_ENTITY: &str = "http://www.purl.org/ogit/Entity";
const OGIT_VERB: &str = "http://www.purl.org/ogit/Verb";
const OGIT_NODE: &str = "http://www.purl.org/ogit/Node";
const OGIT_SCOPE: &str = "http://www.purl.org/ogit/scope";
const OGIT_MANDATORY: &str = "http://www.purl.org/ogit/mandatory-attributes";
const OGIT_OPTIONAL: &str = "http://www.purl.org/ogit/optional-attributes";
const OGIT_INDEXED: &str = "http://www.purl.org/ogit/indexed-attributes";

/// One TTL source — typically a single `.ttl` file in the OGIT NTO tree.
pub struct TtlSource {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl TtlSource {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path, bytes })
    }

    pub fn from_bytes(path: impl AsRef<Path>, bytes: Vec<u8>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            bytes,
        }
    }

    /// SHA256 of the source bytes — drives idempotent re-hydration.
    pub fn checksum(&self) -> String {
        let mut h = Sha256::new();
        h.update(&self.bytes);
        format!("{:x}", h.finalize())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Walk the bytes via oxttl, group by subject, and emit
    /// `MappingProposal`s. The `bridge_id` is the value the proposals will
    /// be filed under in the registry (typically `"ogit"` for raw OGIT
    /// hydration; tenant bridges add their own proposals separately via
    /// `MappingProposal::with_bridge_id`).
    pub fn parse_into_proposals(
        &self,
        bridge_id: &str,
        sem: &SemanticTypeMap,
    ) -> std::result::Result<Vec<MappingProposal>, HydrationFailure> {
        // Pass 1: read all triples into memory. OGIT TTL files are small
        // (median ~50 lines, max <1 KB) so this is cheap.
        use oxttl::TurtleParser;

        let parser = TurtleParser::new()
            .with_base_iri("http://www.purl.org/ogit/")
            .map_err(|e| HydrationFailure {
                source: format!("{}", self.path.display()),
                reason: format!("base IRI: {e}"),
            })?
            .for_slice(&self.bytes);

        let mut triples: Vec<RawTriple> = Vec::new();
        for item in parser {
            match item {
                Ok(t) => {
                    let s = subject_to_string(&t.subject);
                    let p = t.predicate.as_str().to_string();
                    let o = term_to_value(&t.object);
                    triples.push(RawTriple {
                        subject: s,
                        predicate: p,
                        object: o,
                    });
                }
                Err(e) => {
                    return Err(HydrationFailure {
                        source: format!("{}", self.path.display()),
                        reason: format!("oxttl: {e}"),
                    });
                }
            }
        }

        // Pass 2: index triples by subject; pre-resolve RDF lists.
        let by_subject: HashMap<String, Vec<(String, RdfValue)>> =
            triples.into_iter().fold(HashMap::new(), |mut acc, t| {
                acc.entry(t.subject)
                    .or_default()
                    .push((t.predicate, t.object));
                acc
            });

        let mut proposals = Vec::new();

        let checksum = self.checksum();
        let source_uri = format!("file:{}", self.path.display());

        for (subject_uri, props) in &by_subject {
            // Skip non-OGIT subjects (blank nodes, anonymous list cells).
            if !subject_uri.starts_with(OGIT_BASE) {
                continue;
            }
            let canonical = canonical_ogit_uri(subject_uri);
            let ogit_uri = match OgitUri::parse(&canonical) {
                Ok(u) => u,
                Err(_) => continue, // root vocabulary terms etc.
            };
            let namespace = ogit_uri.namespace().unwrap_or("").to_string();

            let kind_class = classify(props);
            match kind_class {
                SubjectKind::Entity => {
                    let schema = build_entity_schema(&ogit_uri, props, &by_subject, sem);
                    proposals.push(MappingProposal {
                        public_name: canonical.clone(),
                        bridge_id: bridge_id.to_string(),
                        ogit_uri: ogit_uri.clone(),
                        namespace: namespace.clone(),
                        kind: MappingProposalKind::Entity { schema },
                        marking: default_marking_for_namespace(&namespace),
                        confidence: 1.0,
                        source_uri: source_uri.clone(),
                        checksum: checksum.clone(),
                        created_by: "ogit_hydrator_v1".to_string(),
                    });
                }
                SubjectKind::Verb => {
                    // Edge proposal — the from-to constraints become a
                    // generic Node->Node placeholder; consumers that need
                    // typed link constraints can re-derive them from the
                    // raw TTL via `source_uri`.
                    let predicate = ogit_uri.name().unwrap_or("relates");
                    let link = LinkSpec {
                        subject_type: "ogit.Node",
                        predicate: leak_static(predicate),
                        object_type: "ogit.Node",
                        cardinality: Cardinality::ManyToMany,
                        codec_route: CodecRoute::Passthrough,
                    };
                    proposals.push(MappingProposal {
                        public_name: canonical.clone(),
                        bridge_id: bridge_id.to_string(),
                        ogit_uri: ogit_uri.clone(),
                        namespace: namespace.clone(),
                        kind: MappingProposalKind::Edge { link },
                        marking: default_marking_for_namespace(&namespace),
                        confidence: 1.0,
                        source_uri: source_uri.clone(),
                        checksum: checksum.clone(),
                        created_by: "ogit_hydrator_v1".to_string(),
                    });
                }
                SubjectKind::Attribute => {
                    let semantic_type = sem.lookup(&canonical);
                    proposals.push(MappingProposal {
                        public_name: canonical.clone(),
                        bridge_id: bridge_id.to_string(),
                        ogit_uri: ogit_uri.clone(),
                        namespace: namespace.clone(),
                        kind: MappingProposalKind::Attribute {
                            predicate: ogit_uri.name().unwrap_or("").to_string(),
                            semantic_type,
                        },
                        marking: default_marking_for_namespace(&namespace),
                        confidence: 1.0,
                        source_uri: source_uri.clone(),
                        checksum: checksum.clone(),
                        created_by: "ogit_hydrator_v1".to_string(),
                    });
                }
                SubjectKind::Other => {}
            }
        }

        Ok(proposals)
    }
}

/// Walk a directory tree, parse every `*.ttl` file, return all proposals.
/// Failed files return a `HydrationFailure` rather than aborting.
pub fn parse_ttl_directory(
    root: &Path,
    bridge_id: &str,
    sem: &SemanticTypeMap,
    namespace_filter: &[&str],
) -> Result<(Vec<MappingProposal>, Vec<HydrationFailure>)> {
    let mut proposals = Vec::new();
    let mut failures = Vec::new();

    walk_ttl_files(root, &mut |path| {
        // Apply namespace filter — directory under root is the namespace.
        if !namespace_filter.is_empty() {
            let rel = path.strip_prefix(root).unwrap_or(path);
            let ns = rel
                .components()
                .next()
                .and_then(|c| c.as_os_str().to_str())
                .unwrap_or("");
            if !namespace_filter.iter().any(|f| *f == ns) {
                return Ok(());
            }
        }
        match TtlSource::from_path(path) {
            Ok(src) => match src.parse_into_proposals(bridge_id, sem) {
                Ok(mut p) => proposals.append(&mut p),
                Err(f) => failures.push(f),
            },
            Err(e) => failures.push(HydrationFailure {
                source: format!("{}", path.display()),
                reason: format!("io: {e}"),
            }),
        }
        Ok(())
    })?;

    Ok((proposals, failures))
}

/// Compute the SHA256 of the concatenated sorted contents of every TTL
/// file under `root`. Used to short-circuit hydration when nothing has
/// changed.
pub fn ttl_root_checksum(root: &Path) -> Result<String> {
    let mut paths: Vec<PathBuf> = Vec::new();
    walk_ttl_files(root, &mut |p| {
        paths.push(p.to_path_buf());
        Ok(())
    })?;
    paths.sort();

    let mut h = Sha256::new();
    for p in paths {
        let bytes = fs::read(&p).map_err(|source| Error::Io {
            path: p.clone(),
            source,
        })?;
        h.update(p.to_string_lossy().as_bytes());
        h.update(&[0u8]);
        h.update(&bytes);
        h.update(&[0u8]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn walk_ttl_files(
    root: &Path,
    visit: &mut dyn FnMut(&Path) -> Result<()>,
) -> Result<()> {
    if !root.exists() {
        return Err(Error::Io {
            path: root.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "TTL root does not exist",
            ),
        });
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let read = fs::read_dir(&dir).map_err(|source| Error::Io {
            path: dir.clone(),
            source,
        })?;
        for entry in read {
            let entry = entry.map_err(|source| Error::Io {
                path: dir.clone(),
                source,
            })?;
            let p = entry.path();
            if p.is_dir() {
                // Skip hidden / version-control directories.
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.starts_with('.') {
                    continue;
                }
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("ttl") {
                visit(&p)?;
            }
        }
    }
    Ok(())
}

// ---------- Triple shape helpers ----------

#[derive(Clone, Debug)]
struct RawTriple {
    subject: String,
    predicate: String,
    object: RdfValue,
}

#[derive(Clone, Debug)]
enum RdfValue {
    Iri(String),
    Blank(String),
    // `Literal(String)`'s payload is captured for completeness and round-trip;
    // the current entity-classifier doesn't read it. TTL-PROBE-5 (TECH_DEBT)
    // tracks the follow-up that wires `dcterms:source` literals through to
    // `MappingProposal::source_uri`. Don't strip the field — its presence is
    // load-bearing for the future fix.
    #[allow(dead_code)]
    Literal(String),
}

fn subject_to_string(s: &oxrdf::Subject) -> String {
    match s {
        oxrdf::Subject::NamedNode(n) => n.as_str().to_string(),
        oxrdf::Subject::BlankNode(b) => format!("_:{}", b.as_str()),
    }
}

fn term_to_value(t: &oxrdf::Term) -> RdfValue {
    match t {
        oxrdf::Term::NamedNode(n) => RdfValue::Iri(n.as_str().to_string()),
        oxrdf::Term::BlankNode(b) => RdfValue::Blank(format!("_:{}", b.as_str())),
        oxrdf::Term::Literal(l) => RdfValue::Literal(l.value().to_string()),
    }
}

fn canonical_ogit_uri(raw: &str) -> String {
    // Normalise: ogit IRIs in the TTL come back as
    // `http://www.purl.org/ogit/Network/IPAddress` (slash) or `:` form.
    // OGIT URIs throughout the registry use the `:` form. Convert here.
    if let Some(rest) = raw.strip_prefix(OGIT_BASE) {
        if let Some((ns, name)) = rest.rsplit_once('/') {
            return format!("ogit.{}:{}", ns.replace('/', "."), name);
        }
        return format!("ogit:{rest}");
    }
    raw.to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubjectKind {
    Entity,
    Verb,
    Attribute,
    Other,
}

fn classify(props: &[(String, RdfValue)]) -> SubjectKind {
    let mut is_class = false;
    let mut is_attribute_class = false;
    let mut subclass_of_entity = false;
    let mut subclass_of_verb = false;
    for (p, o) in props {
        if p == RDF_TYPE {
            if let RdfValue::Iri(ref iri) = o {
                if iri == RDFS_CLASS {
                    is_class = true;
                }
                if iri == OWL_DATATYPE_PROPERTY {
                    is_attribute_class = true;
                }
            }
        }
        if p == RDFS_SUBCLASS_OF {
            if let RdfValue::Iri(ref iri) = o {
                if iri == OGIT_ENTITY {
                    subclass_of_entity = true;
                }
                if iri == OGIT_VERB {
                    subclass_of_verb = true;
                }
            }
        }
    }
    match (is_class, subclass_of_entity, subclass_of_verb, is_attribute_class) {
        (true, true, _, _) => SubjectKind::Entity,
        (true, _, true, _) => SubjectKind::Verb,
        (_, _, _, true) => SubjectKind::Attribute,
        _ => SubjectKind::Other,
    }
}

fn build_entity_schema(
    uri: &OgitUri,
    props: &[(String, RdfValue)],
    by_subject: &HashMap<String, Vec<(String, RdfValue)>>,
    _sem: &SemanticTypeMap,
) -> Schema {
    // Static name leak: we need `&'static str` for SchemaBuilder.
    let entity_name = leak_static(uri.name().unwrap_or("Unknown"));
    let mut builder = Schema::builder(entity_name);

    for (p, o) in props {
        let attrs = match p.as_str() {
            OGIT_MANDATORY => walk_rdf_list(o, by_subject),
            OGIT_OPTIONAL => walk_rdf_list(o, by_subject),
            OGIT_INDEXED => walk_rdf_list(o, by_subject),
            _ => continue,
        };

        let is_required = p == OGIT_MANDATORY;

        for attr in attrs {
            // Strip the namespace prefix to get the predicate's local name.
            let local = attr_local_name(&attr);
            let leaked: &'static str = leak_static(local);
            if is_required {
                builder = builder.property(PropertySpec::required(leaked));
            } else {
                builder = builder.property(PropertySpec::optional(
                    leaked,
                    CodecRoute::Passthrough,
                ));
            }
        }
    }

    builder.build()
}

fn walk_rdf_list(
    head: &RdfValue,
    by_subject: &HashMap<String, Vec<(String, RdfValue)>>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: String = match head {
        RdfValue::Blank(b) => b.clone(),
        RdfValue::Iri(iri) if iri == RDF_NIL => return out,
        _ => return out,
    };

    // Bound the walk to keep malformed / cyclic lists from blocking forever.
    for _ in 0..1024 {
        let triples = match by_subject.get(&current) {
            Some(t) => t,
            None => break,
        };
        let mut first: Option<String> = None;
        let mut next: Option<String> = None;
        for (p, o) in triples {
            if p == RDF_FIRST {
                if let RdfValue::Iri(iri) = o {
                    first = Some(iri.clone());
                }
            }
            if p == RDF_REST {
                match o {
                    RdfValue::Iri(iri) if iri == RDF_NIL => break,
                    RdfValue::Blank(b) => next = Some(b.clone()),
                    _ => {}
                }
            }
        }
        if let Some(f) = first {
            out.push(f);
        }
        match next {
            Some(n) => current = n,
            None => break,
        }
    }
    out
}

fn attr_local_name(uri: &str) -> &str {
    if let Some(after_colon) = uri.rsplit_once(':') {
        return after_colon.1;
    }
    if let Some(after_slash) = uri.rsplit_once('/') {
        return after_slash.1;
    }
    uri
}

fn default_marking_for_namespace(namespace: &str) -> Marking {
    match namespace {
        "Auth" | "Compliance" | "Person" => Marking::Pii,
        "FinancialAccounting" | "FinancialMarket" | "Cost" | "Credit" | "Price" => {
            Marking::Financial
        }
        _ => Marking::Internal,
    }
}

/// Leak a string into a `&'static str`. Necessary because contract types
/// like `Schema` and `PropertySpec` hold `&'static str` references. The
/// leaked memory is bounded — TTL sources are small and re-hydration
/// short-circuits via the root checksum.
fn leak_static(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Suppress unused-warning for the constants we reserve for future
/// `ogit:allowed` constraint expansion.
#[allow(dead_code)]
const _RESERVED: &[&str] = &[OGIT_NODE, OGIT_SCOPE, RDFS_LABEL];

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_TTL: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.Test:              <http://www.purl.org/ogit/Test/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms:                <http://purl.org/dc/terms/> .

ogit.Test:Widget
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Widget";
    dcterms:description "A round thing." ;
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes (
        ogit:name
    );
.
"#;

    #[test]
    fn parse_tiny_ttl_yields_one_entity() {
        let src = TtlSource::from_bytes(PathBuf::from("tiny.ttl"), TINY_TTL.as_bytes().to_vec());
        let sem = SemanticTypeMap::defaults();
        let proposals = src.parse_into_proposals("ogit", sem).unwrap();
        let entity = proposals
            .iter()
            .find(|p| matches!(p.kind, MappingProposalKind::Entity { .. }))
            .expect("expected one entity proposal");
        assert_eq!(entity.namespace, "Test");
        assert_eq!(entity.public_name, "ogit.Test:Widget");
    }

    #[test]
    fn checksum_changes_on_content_change() {
        let a = TtlSource::from_bytes(PathBuf::from("a.ttl"), b"ogit:foo a rdfs:Class .".to_vec());
        let b = TtlSource::from_bytes(PathBuf::from("a.ttl"), b"ogit:bar a rdfs:Class .".to_vec());
        assert_ne!(a.checksum(), b.checksum());
    }
}
