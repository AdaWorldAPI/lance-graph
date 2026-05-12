//! ogit_owl_dolce — Semantic layer bridging CAM-PQ / DnPath with OGIT + OWL + DOLCE
//!
//! This module maps CAM archetypes and codebook labels into a richer
//! ontological space using DOLCE top-level categories and OGIT schema labels.

use lance_graph_contract::ontology::Label;

/// Extended DOLCE category mapping for CAM archetypes.
///
/// These mappings are intentionally coarse-grained at first.
/// They can be refined per domain (e.g. medical, CRM, industrial).
pub fn archetype_to_dolce_category(archetype: &str) -> Option<&'static str> {
    match archetype {
        // HEEL → stable, identity-preserving entities
        "HEEL" => Some("Endurant"),

        // BRANCH → processes, events, state changes that unfold in time
        "BRANCH" => Some("Perdurant"),

        // TWIG_* → dependent characteristics / attributes
        "TWIG_A" | "TWIG_B" => Some("Quality"),

        // LEAF → functional or social roles played by endurants
        "LEAF" => Some("Role"),

        // GAMMA → abstract, meta, or cross-cutting concepts
        "GAMMA" => Some("Abstract"),

        _ => None,
    }
}

/// More fine-grained DOLCE classification (optional deeper layer).
///
/// This can be used when more precision is needed (e.g. for reasoning).
pub fn archetype_to_dolce_fine(archetype: &str) -> Option<&'static str> {
    match archetype {
        "HEEL"     => Some("PhysicalEndurant"),
        "BRANCH"   => Some("Event"),
        "TWIG_A"   => Some("PhysicalQuality"),
        "TWIG_B"   => Some("MentalQuality"),
        "LEAF"     => Some("SocialRole"),
        "GAMMA"    => Some("AbstractEntity"),
        _ => archetype_to_dolce_category(archetype),
    }
}

/// Convert a raw CAM codebook label into a proper OGIT schema label.
pub fn codebook_label_to_ogit(label: &str) -> Label {
    // TODO: In production this should do a proper lookup/creation against the OGIT registry.
    Label {
        name: label.to_string(),
        namespace: "cam".to_string(),
        // version, description, etc. can be added later
    }
}

/// Rich semantic enrichment result from CAM codebook + archetype.
#[derive(Debug, Clone, Default)]
pub struct SemanticEnrichment {
    pub dolce_category: Option<&'static str>,
    pub dolce_fine: Option<&'static str>,
    pub ogit_label: Option<Label>,
    pub confidence: f32, // 0.0 – 1.0
}

/// Enrich a context using both archetype and optional codebook label.
pub fn enrich_from_cam_codebook(
    archetype: &str,
    codebook_label: Option<&str>,
) -> SemanticEnrichment {
    let dolce = archetype_to_dolce_category(archetype);
    let dolce_fine = archetype_to_dolce_fine(archetype);
    let ogit = codebook_label.map(codebook_label_to_ogit);

    // Simple heuristic confidence: higher when both archetype and label are present
    let confidence = match (dolce, codebook_label) {
        (Some(_), Some(_)) => 0.9,
        (Some(_), None)    => 0.7,
        _                  => 0.5,
    };

    SemanticEnrichment {
        dolce_category: dolce,
        dolce_fine,
        ogit_label: ogit,
        confidence,
    }
}

/// Helper to attach semantic enrichment directly to a DnPath context.
pub fn attach_semantics_to_dn_path(
    archetype: &str,
    codebook_label: Option<&str>,
) -> SemanticEnrichment {
    enrich_from_cam_codebook(archetype, codebook_label)
}
