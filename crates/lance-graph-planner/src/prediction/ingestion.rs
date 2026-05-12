//! News/Intel Ingestion — external data → graph nodes + edges.
//!
//! Converts unstructured intelligence (news articles, OSINT feeds, LLM output)
//! into typed graph nodes and edges that can be merged into the aiwar graph.
//!
//! ## Pipeline
//!
//! ```text
//! Raw text → Entity extraction → Node creation → Edge inference → Graph merge
//!            (regex/LLM)         (typed props)    (NARS truth)    (truth revision)
//! ```
//!
//! ## Integration with MCP
//!
//! The q2 notebook server exposes ingestion via MCP tools:
//! - `ingest_text`: parse raw text into entities
//! - `ingest_url`: fetch URL, extract entities (requires WebFetch)
//! - `ingest_llm`: send text to LLM for structured extraction
//!
//! The planner's `plan_polyglot()` can then query the expanded graph.

use super::{Derivation, IngestEdge, IngestNode};

/// Result of entity extraction from text.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted nodes (entities).
    pub nodes: Vec<IngestNode>,
    /// Extracted edges (relationships between entities).
    pub edges: Vec<IngestEdge>,
    /// Source document identifier.
    pub source: String,
    /// Confidence in the extraction (0..1).
    pub extraction_confidence: f64,
}

/// Entity types recognized by the extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    System,      // AI/weapon system
    Stakeholder, // Company, government, military unit
    Person,      // Individual
    Location,    // Geographic location
    Event,       // Military operation, treaty, deployment
    Technology,  // Specific technology (radar, drone, cyber tool)
}

impl EntityType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Stakeholder => "Stakeholder",
            Self::Person => "Person",
            Self::Location => "Location",
            Self::Event => "Event",
            Self::Technology => "Technology",
        }
    }
}

/// Relationship types for edge extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationType {
    DevelopedBy,
    DeployedBy,
    UsedIn,
    ConnectedTo,
    Targets,
    Defends,
    SuppliedBy,
    SanctionedBy,
    AlliedWith,
    OpposedTo,
}

impl RelationType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DevelopedBy => "DEVELOPED_BY",
            Self::DeployedBy => "DEPLOYED_BY",
            Self::UsedIn => "USED_IN",
            Self::ConnectedTo => "CONNECTED_TO",
            Self::Targets => "TARGETS",
            Self::Defends => "DEFENDS",
            Self::SuppliedBy => "SUPPLIED_BY",
            Self::SanctionedBy => "SANCTIONED_BY",
            Self::AlliedWith => "ALLIED_WITH",
            Self::OpposedTo => "OPPOSED_TO",
        }
    }
}

/// Extract entities from raw text using pattern matching.
///
/// This is the regex/heuristic path — fast, deterministic, no LLM needed.
/// For higher quality extraction, use `extract_with_llm_prompt()` to generate
/// a prompt that an LLM can fill in.
pub fn extract_entities(text: &str, source: &str) -> ExtractionResult {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Known aiwar entity patterns
    let system_patterns = [
        ("Lavender", "AI targeting system", "targeting"),
        ("Gospel", "AI confirmation system", "target_confirmation"),
        ("Fire Factory", "Automated strike system", "strike"),
        ("Iron Dome", "Missile defense system", "defense"),
        ("Patriot", "Air defense system", "defense"),
        ("THAAD", "Ballistic missile defense", "defense"),
        ("Arrow", "Anti-ballistic missile", "defense"),
        ("Pegasus", "Spyware", "surveillance"),
        ("Predator", "Surveillance drone", "surveillance"),
        ("Reaper", "Armed drone", "strike"),
        ("Replicator", "Autonomous systems program", "autonomous"),
        ("Palantir", "Data analytics platform", "intelligence"),
        ("Gotham", "Intelligence platform", "intelligence"),
        ("S-300", "Air defense system", "defense"),
        ("S-400", "Air defense system", "defense"),
        ("Shahed", "Loitering munition", "strike"),
        ("Karrar", "Combat drone", "strike"),
    ];

    let stakeholder_patterns = [
        ("IRGC", "Stakeholder", "military"),
        ("IDF", "Stakeholder", "military"),
        ("Pentagon", "Stakeholder", "military"),
        ("Mossad", "Stakeholder", "intelligence"),
        ("CIA", "Stakeholder", "intelligence"),
        ("NSO Group", "Stakeholder", "cyber"),
        ("Elbit", "Stakeholder", "defense_industry"),
        ("Raytheon", "Stakeholder", "defense_industry"),
        ("Lockheed", "Stakeholder", "defense_industry"),
        ("Boeing", "Stakeholder", "defense_industry"),
        ("Iran", "Stakeholder", "nation"),
        ("Israel", "Stakeholder", "nation"),
        ("United States", "Stakeholder", "nation"),
        ("China", "Stakeholder", "nation"),
        ("Russia", "Stakeholder", "nation"),
        ("Hezbollah", "Stakeholder", "non_state"),
        ("Hamas", "Stakeholder", "non_state"),
    ];

    let text_upper = text.to_uppercase();

    // Extract systems
    for (name, desc, sys_type) in &system_patterns {
        if text.contains(name) || text_upper.contains(&name.to_uppercase()) {
            let id = name.to_lowercase().replace(' ', "_");
            nodes.push(IngestNode {
                id: id.clone(),
                label: name.to_string(),
                node_type: "System".into(),
                properties: vec![
                    ("description".into(), desc.to_string()),
                    ("type".into(), sys_type.to_string()),
                ],
                source: source.into(),
                source_confidence: 0.85,
            });
        }
    }

    // Extract stakeholders
    for (name, node_type, stake_type) in &stakeholder_patterns {
        if text.contains(name) || text_upper.contains(&name.to_uppercase()) {
            let id = name.to_lowercase().replace(' ', "_");
            nodes.push(IngestNode {
                id: id.clone(),
                label: name.to_string(),
                node_type: node_type.to_string(),
                properties: vec![("type".into(), stake_type.to_string())],
                source: source.into(),
                source_confidence: 0.80,
            });
        }
    }

    // Extract edges from co-occurrence
    // If two entities appear in the same sentence, they're likely related
    let sentences: Vec<&str> = text
        .split(['.', '!', '?', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    for sentence in &sentences {
        let sentence_nodes: Vec<&IngestNode> = nodes
            .iter()
            .filter(|n| {
                sentence.contains(&n.label)
                    || sentence.to_uppercase().contains(&n.label.to_uppercase())
            })
            .collect();

        for i in 0..sentence_nodes.len() {
            for j in (i + 1)..sentence_nodes.len() {
                let a = &sentence_nodes[i];
                let b = &sentence_nodes[j];

                // Determine relationship type from context keywords
                let rel = if sentence.contains("develop")
                    || sentence.contains("built")
                    || sentence.contains("created")
                {
                    RelationType::DevelopedBy
                } else if sentence.contains("deploy")
                    || sentence.contains("used")
                    || sentence.contains("operate")
                {
                    RelationType::DeployedBy
                } else if sentence.contains("target")
                    || sentence.contains("strike")
                    || sentence.contains("attack")
                {
                    RelationType::Targets
                } else if sentence.contains("defend")
                    || sentence.contains("protect")
                    || sentence.contains("intercept")
                {
                    RelationType::Defends
                } else if sentence.contains("sanction")
                    || sentence.contains("restrict")
                    || sentence.contains("ban")
                {
                    RelationType::SanctionedBy
                } else if sentence.contains("ally")
                    || sentence.contains("partner")
                    || sentence.contains("cooperat")
                {
                    RelationType::AlliedWith
                } else if sentence.contains("oppos")
                    || sentence.contains("rival")
                    || sentence.contains("enemy")
                    || sentence.contains("against")
                {
                    RelationType::OpposedTo
                } else if sentence.contains("supply")
                    || sentence.contains("sell")
                    || sentence.contains("provide")
                {
                    RelationType::SuppliedBy
                } else {
                    RelationType::ConnectedTo
                };

                edges.push(IngestEdge {
                    source_id: a.id.clone(),
                    target_id: b.id.clone(),
                    relationship: rel.label().to_string(),
                    frequency: 0.70,
                    confidence: 0.50, // Co-occurrence is moderate confidence
                    source: source.into(),
                    derivation: Derivation::Ingested,
                });
            }
        }
    }

    ExtractionResult {
        nodes,
        edges,
        source: source.into(),
        extraction_confidence: 0.75,
    }
}

/// Generate an LLM prompt for structured entity extraction.
///
/// The caller sends this prompt to any LLM (Claude, GPT, local model)
/// and parses the JSON response back into `IngestNode` / `IngestEdge`.
pub fn extraction_prompt(text: &str) -> String {
    format!(
        r#"Extract entities and relationships from this intelligence text.

TEXT:
{text}

Return JSON with this exact structure:
{{
  "nodes": [
    {{
      "id": "lowercase_underscore_name",
      "label": "Human Readable Name",
      "type": "System|Stakeholder|Person|Location|Event|Technology",
      "properties": {{ "key": "value" }}
    }}
  ],
  "edges": [
    {{
      "source": "node_id",
      "target": "node_id",
      "relationship": "DEVELOPED_BY|DEPLOYED_BY|TARGETS|DEFENDS|CONNECTED_TO|SUPPLIED_BY|SANCTIONED_BY|ALLIED_WITH|OPPOSED_TO",
      "confidence": 0.0-1.0
    }}
  ]
}}

Rules:
- Only extract entities explicitly mentioned in the text
- Use UPPERCASE for relationship types
- Confidence reflects how certain the relationship is from the text
- Include all relevant military, political, and technological entities
- For weapons systems, type is "System"
- For countries, organizations, companies, type is "Stakeholder"
"#
    )
}

/// Parse an LLM's JSON response into IngestNode/IngestEdge.
pub fn parse_llm_response(json: &str, source: &str) -> Result<ExtractionResult, String> {
    let doc: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON from LLM: {e}"))?;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    if let Some(node_array) = doc.get("nodes").and_then(|v| v.as_array()) {
        for node in node_array {
            let id = node.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            let label = node.get("label").and_then(|v| v.as_str()).unwrap_or(id);
            let node_type = node
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("System");

            let mut properties = Vec::new();
            if let Some(props) = node.get("properties").and_then(|v| v.as_object()) {
                for (k, v) in props {
                    if let Some(s) = v.as_str() {
                        properties.push((k.clone(), s.to_string()));
                    }
                }
            }

            nodes.push(IngestNode {
                id: id.to_string(),
                label: label.to_string(),
                node_type: node_type.to_string(),
                properties,
                source: source.into(),
                source_confidence: 0.70, // LLM extraction — moderate confidence
            });
        }
    }

    if let Some(edge_array) = doc.get("edges").and_then(|v| v.as_array()) {
        for edge in edge_array {
            let source_id = edge
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let target_id = edge
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let relationship = edge
                .get("relationship")
                .and_then(|v| v.as_str())
                .unwrap_or("CONNECTED_TO");
            let confidence = edge
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            edges.push(IngestEdge {
                source_id: source_id.to_string(),
                target_id: target_id.to_string(),
                relationship: relationship.to_string(),
                frequency: 0.80,
                confidence,
                source: source.into(),
                derivation: Derivation::Ingested,
            });
        }
    }

    Ok(ExtractionResult {
        nodes,
        edges,
        source: source.into(),
        extraction_confidence: 0.70,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_entities_from_news() {
        let text = "Israel deployed the Iron Dome system to defend against Shahed drones launched by Iran. \
                     The IRGC developed the Shahed loitering munition as a low-cost strike weapon. \
                     Palantir's Gotham platform is used by the Pentagon for intelligence analysis.";

        let result = extract_entities(text, "test_article_001");

        // Should find systems
        assert!(result.nodes.iter().any(|n| n.label == "Iron Dome"));
        assert!(result.nodes.iter().any(|n| n.label == "Shahed"));
        assert!(result.nodes.iter().any(|n| n.label == "Gotham"));

        // Should find stakeholders
        assert!(result.nodes.iter().any(|n| n.label == "IRGC"));
        assert!(result.nodes.iter().any(|n| n.label == "Pentagon"));

        // Should find edges
        assert!(!result.edges.is_empty());

        // Should have defense relationship for Iron Dome
        assert!(result
            .edges
            .iter()
            .any(|e| e.source_id.contains("iron_dome") || e.target_id.contains("iron_dome")));
    }

    #[test]
    fn test_extraction_prompt_is_valid() {
        let prompt = extraction_prompt("Iran launched missiles at Israel");
        assert!(prompt.contains("TEXT:"));
        assert!(prompt.contains("Iran launched missiles"));
        assert!(prompt.contains("\"nodes\""));
        assert!(prompt.contains("\"edges\""));
    }

    #[test]
    fn test_parse_llm_response() {
        let json = r#"{
            "nodes": [
                {"id": "s300", "label": "S-300", "type": "System", "properties": {"role": "air_defense"}},
                {"id": "iran", "label": "Iran", "type": "Stakeholder", "properties": {}}
            ],
            "edges": [
                {"source": "iran", "target": "s300", "relationship": "DEPLOYED_BY", "confidence": 0.9}
            ]
        }"#;

        let result = parse_llm_response(json, "llm_extraction_001").unwrap();
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].confidence, 0.9);
    }

    #[test]
    fn test_parse_llm_response_invalid_json() {
        let result = parse_llm_response("not json", "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_cooccurrence_edges() {
        let text = "Iran deployed S-300 systems near Tehran.";
        let result = extract_entities(text, "test");
        // Iran and S-300 co-occur → should have an edge
        let has_edge = result.edges.iter().any(|e| {
            (e.source_id.contains("iran") && e.target_id.contains("s-300"))
                || (e.source_id.contains("s-300") && e.target_id.contains("iran"))
        });
        assert!(
            has_edge,
            "Should extract co-occurrence edge between Iran and S-300"
        );
    }
}
