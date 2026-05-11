//! Proposed addition: Semantic enrichment for DnPath
//!
//! This can be added to crates/lance-graph-callcenter/src/dn_path.rs
//! or as a separate module. It is fully backward compatible.

use lance_graph_contract::ontology::Label;

/// Semantic context that can be attached to a DnPath without changing its compact form.
#[derive(Clone, Debug, Default)]
pub struct SemanticContext {
    /// OGIT schema labels
    pub ogit_labels: Vec<Label>,

    /// OWL class or DOLCE category (e.g. "Endurant", "Perdurant", "Quality", "Role")
    pub owl_dolce_category: Option<String>,

    /// Optional references to CAM codebook centroids (for hybrid retrieval)
    pub cam_centroid_refs: Vec<u8>,
}

/// Extended version of DnPath that carries semantic information.
#[derive(Clone, Debug)]
pub struct DnPathWithSemantics {
    pub path: super::DnPath,
    pub semantic: SemanticContext,
}

impl DnPathWithSemantics {
    pub fn new(path: super::DnPath) -> Self {
        Self {
            path,
            semantic: SemanticContext::default(),
        }
    }

    pub fn with_ogit_labels(mut self, labels: Vec<Label>) -> Self {
        self.semantic.ogit_labels = labels;
        self
    }

    pub fn with_owl_dolce_category(mut self, category: impl Into<String>) -> Self {
        self.semantic.owl_dolce_category = Some(category.into());
        self
    }

    pub fn with_cam_centroids(mut self, centroids: Vec<u8>) -> Self {
        self.semantic.cam_centroid_refs = centroids;
        self
    }

    /// Convenience accessor for the original compact scent
    pub fn scent(&self) -> u8 {
        self.path.scent()
    }
}
