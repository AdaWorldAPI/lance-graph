//! OGIT + DOLCE Canonical Ontology Spine for lance-graph-ontology
//!
//! This is a proposed foundation for the ontology crate.
//! It aligns the existing OGIT spine scaffolding with DOLCE foundational categories.
//! Goal: Provide strong cognitive/NARS grounding while keeping OGIT as the project canonical layer.

use std::collections::HashMap;

/// Core DOLCE-inspired categories (simplified for Rust embedding)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DolceCategory {
    /// Endurant - objects that persist through time (e.g. "Apple", "Person")
    Endurant,
    /// Perdurant - events/processes (e.g. "Eating", "ReasoningStep")
    Perdurant,
    /// Quality - properties/qualia (e.g. color, truth-value, emotional state)
    Quality,
    /// Abstract - non-spatio-temporal (e.g. numbers, types, NARS terms)
    Abstract,
    /// Particular - concrete instances
    Particular,
}

/// OGIT Spine Node — the canonical internal representation
#[derive(Debug, Clone)]
pub struct OgitNode {
    pub id: String,
    pub ogit_type: String,                    // e.g. "ogit:Concept", "ogit:Event"
    pub dolce_category: DolceCategory,        // Grounding in DOLCE
    pub labels: HashMap<String, String>,      // Bilingual labels (en, de, ...)
    pub properties: HashMap<String, String>,  // Arbitrary props
    pub truth_value: Option<(f32, f32)>,      // NARS-style <f, c>
    pub qualia: Option<Vec<i8>>,              // 8-dimensional emotional/qualia vector (for Firefly CONTEXT)
}

/// Relation in the OGIT spine
#[derive(Debug, Clone)]
pub struct OgitRelation {
    pub id: String,
    pub relation_type: String,                // e.g. "ogit:causes", "ogit:partOf"
    pub dolce_category: DolceCategory,
    pub source: String,                       // source node id
    pub target: String,                       // target node id
    pub properties: HashMap<String, String>,
}

/// The main Ontology Spine — this is what should live in lance-graph-ontology/src/ontology/
pub struct OgitDolceSpine {
    nodes: HashMap<String, OgitNode>,
    relations: HashMap<String, OgitRelation>,
    // Future: OWL import cache, DOLCE alignment table, etc.
}

impl OgitDolceSpine {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            relations: HashMap::new(),
        }
    }

    /// Create a new node grounded in both OGIT and DOLCE
    pub fn create_node(
        &mut self,
        id: String,
        ogit_type: String,
        dolce: DolceCategory,
        labels: HashMap<String, String>,
    ) -> &OgitNode {
        let node = OgitNode {
            id: id.clone(),
            ogit_type,
            dolce_category: dolce,
            labels,
            properties: HashMap::new(),
            truth_value: None,
            qualia: None,
        };
        self.nodes.insert(id.clone(), node);
        self.nodes.get(&id).unwrap()
    }

    /// Add NARS-style truth value + qualia (directly usable by Firefly Frame CONTEXT field)
    pub fn annotate_with_nars_context(&mut self, node_id: &str, f: f32, c: f32, qualia: Vec<i8>) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.truth_value = Some((f, c));
            node.qualia = Some(qualia);
        }
    }

    /// Example: Create a perdurant event (very useful for NARS + causal reasoning in Firefly)
    pub fn create_perdurant_event(&mut self, id: String, label_en: &str) -> &OgitNode {
        let mut labels = HashMap::new();
        labels.insert("en".to_string(), label_en.to_string());
        self.create_node(id, "ogit:Event".to_string(), DolceCategory::Perdurant, labels)
    }

    // TODO: Add methods for:
    // - DOLCE alignment lookup table
    // - Export to OWL
    // - Import from OWL
    // - Mapping to Firefly Frame payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_perdurant() {
        let mut spine = OgitDolceSpine::new();
        let node = spine.create_perdurant_event("evt:reasoning-42".to_string(), "NARS inference step");
        assert_eq!(node.dolce_category, DolceCategory::Perdurant);
        assert_eq!(node.ogit_type, "ogit:Event");
    }
}
