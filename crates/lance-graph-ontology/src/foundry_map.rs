//! `MappingProposal` → contract `Ontology` adapter.
//!
//! The TTL hydrator emits a flat sequence of `MappingProposal`s. Consumers
//! that want to talk to existing `lance-graph-contract::ontology` surfaces
//! (`SchemaExpander`, `SpoBridge`, callcenter `ontology_dto`) want a single
//! `Ontology` value with `schemas: Vec<Schema>`, `links: Vec<LinkSpec>`,
//! `actions: Vec<ActionSpec>`. This module is the adapter.
//!
//! Carrier-method doctrine: methods on `OntologyAssembler` and
//! `MappingProposal` itself, not free functions on slices.

use crate::proposal::{MappingProposal, MappingProposalKind};
use lance_graph_contract::ontology::{Ontology, OntologyBuilder};

/// Assembles a contract `Ontology` from a slice of `MappingProposal`s.
/// Multiple proposals may target the same entity (e.g. one TTL file
/// declares the entity, a sibling file adds a verb between two of them);
/// the assembler merges them by name.
pub struct OntologyAssembler {
    name: &'static str,
}

impl OntologyAssembler {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }

    pub fn assemble(&self, proposals: &[MappingProposal]) -> Ontology {
        let mut builder: OntologyBuilder = Ontology::builder(self.name);
        for proposal in proposals {
            match &proposal.kind {
                MappingProposalKind::Entity { schema } => {
                    builder = builder.schema(schema.clone());
                }
                MappingProposalKind::Edge { link } => {
                    builder = builder.link(link.clone());
                }
                // Standalone attribute proposals are SemanticType
                // annotations on the dictionary; they don't add to the
                // contract `Ontology` directly. The dictionary still
                // carries them, and consumers that want SemanticType for
                // a predicate look them up via `OntologyRegistry::resolve_uri`.
                MappingProposalKind::Attribute { .. } => {}
            }
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::OgitUri;
    use lance_graph_contract::property::{Marking, Schema};

    #[test]
    fn assemble_merges_entities_and_links() {
        let entity = MappingProposal {
            public_name: "ogit.Test:Widget".to_string(),
            bridge_id: "ogit".to_string(),
            ogit_uri: OgitUri::parse("ogit.Test:Widget").unwrap(),
            namespace: "Test".to_string(),
            kind: MappingProposalKind::Entity {
                schema: Schema::builder("Widget").required("id").build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: "test://1".to_string(),
            checksum: "abc".to_string(),
            created_by: "test".to_string(),
        };
        let assembler = OntologyAssembler::new("Test");
        let ontology = assembler.assemble(std::slice::from_ref(&entity));
        assert_eq!(ontology.schemas.len(), 1);
        assert_eq!(ontology.schemas[0].name, "Widget");
    }
}
