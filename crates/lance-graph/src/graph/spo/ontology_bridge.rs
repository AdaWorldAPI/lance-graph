// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ontology → SPO bridge — exercises the `SchemaExpander` trait
//! defined in `lance-graph-contract` against the canonical
//! `Ontology` builder.
//!
//! Each PropertySpec.predicate becomes an SPO triple
//! `(entity:{type}:{id}, {predicate}, value:{hash})`.
//! Each LinkSpec becomes an edge triple
//! `(entity:{subject_type}:{subject_id}, {link.predicate},
//! entity:{object_type}:{object_id})`.
//!
//! The trait impl itself lives next to `Ontology` in the contract
//! crate (orphan-rule constraint — both the trait and the receiver
//! type are defined there). This module's role is to validate the
//! integration end-to-end and to host follow-on Phase-2 helpers
//! that turn `ExpandedTriple`s into `SpoRecord`s for the SPO store.

#[cfg(test)]
mod tests {
    use lance_graph_contract::ontology::{Ontology, SchemaExpander};
    use lance_graph_contract::property::{LinkSpec, Schema};

    #[test]
    fn expand_entity_produces_triples_for_each_property() {
        let customer = Schema::builder("Customer")
            .required("name")
            .required("tax_id")
            .build();
        let ontology = Ontology::builder("Test").schema(customer).build();

        let triples =
            ontology.expand_entity("Customer", 42, &[("name", b"Alice"), ("tax_id", b"DE123")]);
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].subject_label, "entity:Customer:42");
        assert_eq!(triples[0].predicate, "name");
        assert_eq!(triples[1].predicate, "tax_id");
        assert_eq!(triples[0].entity_type_id, 1);
    }

    #[test]
    fn expand_required_property_uses_nars_floor() {
        let schema = Schema::builder("Customer").required("tax_id").build();
        let ontology = Ontology::builder("Test").schema(schema).build();
        let triples = ontology.expand_entity("Customer", 1, &[("tax_id", b"x")]);
        // Required default floor is (128, 128) → (0.502, 0.502)
        assert!(triples[0].truth.0 > 0.4 && triples[0].truth.0 < 0.6);
    }

    #[test]
    fn expand_unknown_entity_type_returns_empty() {
        let ontology = Ontology::builder("Test").build();
        let triples = ontology.expand_entity("Nonexistent", 1, &[("foo", b"bar")]);
        assert!(triples.is_empty());
    }

    #[test]
    fn expand_link_produces_edge_triple() {
        let ontology = Ontology::builder("Test")
            .schema(Schema::builder("Customer").build())
            .schema(Schema::builder("Invoice").build())
            .link(LinkSpec::one_to_many("Customer", "issued", "Invoice"))
            .build();

        let link = &ontology.links[0];
        let triple = ontology.expand_link(link, 42, 100);
        assert_eq!(triple.subject_label, "entity:Customer:42");
        assert_eq!(triple.object_label, "entity:Invoice:100");
        assert_eq!(triple.predicate, "issued");
    }

    #[test]
    fn expand_smb_ontology_produces_expected_triples() {
        // This validates the integration with the actual smb ontology
        // from callcenter::ontology_dto. We test it by building a minimal
        // ontology that mirrors the smb structure.
        let ontology = Ontology::builder("SMB")
            .schema(
                Schema::builder("Customer")
                    .required("name")
                    .required("tax_id")
                    .build(),
            )
            .schema(
                Schema::builder("Invoice")
                    .required("number")
                    .required("amount")
                    .build(),
            )
            .link(LinkSpec::one_to_many("Customer", "issued", "Invoice"))
            .build();

        let cust_triples =
            ontology.expand_entity("Customer", 1, &[("name", b"X"), ("tax_id", b"Y")]);
        assert_eq!(cust_triples.len(), 2);

        let link_triple = ontology.expand_link(&ontology.links[0], 1, 100);
        assert_eq!(link_triple.predicate, "issued");
    }
}
