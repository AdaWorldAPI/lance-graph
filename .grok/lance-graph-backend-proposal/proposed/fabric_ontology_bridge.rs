//! Bridge between lance-graph-ontology (OGIT+DOLCE spine) and
//! lance-graph-cognitive fabric / Firefly Frame
//!
//! This allows Firefly Frames (especially NARS and Causal language prefixes)
//! to carry properly typed, DOLCE-grounded ontology information in their
//! CONTEXT and DATA payload fields.

use crate::ogit_dolce_spine::{OgitDolceSpine, DolceCategory};

/// Payload that can be embedded into a Firefly Frame's DATA or CONTEXT section
#[derive(Debug, Clone)]
pub struct OntologyPayload {
    pub node_id: String,
    pub ogit_type: String,
    pub dolce_category: DolceCategory,
    pub truth_value: Option<(f32, f32)>,   // NARS <f,c>
    pub qualia: Option<Vec<i8>>,           // Directly maps to Firefly CONTEXT qualia vector
}

/// Convert an OgitNode into a payload suitable for Firefly Frame
pub fn node_to_firefly_payload(node_id: &str, spine: &OgitDolceSpine) -> Option<OntologyPayload> {
    // In real code you would look up the node in the spine
    // Here we show the shape
    Some(OntologyPayload {
        node_id: node_id.to_string(),
        ogit_type: "ogit:Event".to_string(),
        dolce_category: DolceCategory::Perdurant,
        truth_value: Some((0.85, 0.92)),
        qualia: Some(vec![10, -5, 20, 0, 15, -8, 5, 30]), // example 8-dim qualia
    })
}

/// Example: How a NARS DEDUCE operation (language prefix 0x3) could use the ontology
pub fn nars_deduce_with_ontology(
    premise_a: &str,
    premise_b: &str,
    spine: &mut OgitDolceSpine,
) -> OntologyPayload {
    // 1. Look up or create nodes in the OGIT+DOLCE spine
    // 2. Perform NARS-style revision / deduction
    // 3. Return payload ready to be packed into a Firefly Frame

    let conclusion_id = format!("nars:deduced:{}->{}", premise_a, premise_b);
    spine.create_perdurant_event(conclusion_id.clone(), "NARS deduction result");

    // Annotate with truth value (example)
    spine.annotate_with_nars_context(&conclusion_id, 0.78, 0.65, vec![5, 12, -3, 8, 0, 20, -10, 15]);

    node_to_firefly_payload(&conclusion_id, spine).unwrap()
}

// TODO: Add functions for:
// - Packing OntologyPayload into the 384-bit DATA or CONTEXT field of Firefly Frame
// - Reading ontology info from incoming Firefly Frames
// - Causal reasoning (Pearl rungs) using DOLCE perdurants
