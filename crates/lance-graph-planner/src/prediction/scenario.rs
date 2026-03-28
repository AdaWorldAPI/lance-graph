//! Scenario generation — thinking-style-dependent future prediction.
//!
//! Each thinking style explores the causal graph differently:
//! - Analytical/Convergent: deep sequential chains, high confidence threshold
//! - Creative/Divergent: lateral connections, low threshold, find surprises
//! - Focused/Attention: single kill chain, narrow fan-out
//! - Intuitive/Speed: fast heuristic, top-1 most likely

use super::{CausalStep, Derivation, Scenario};
use crate::thinking::style::{ThinkingStyle, ThinkingCluster};
use crate::elevation::budget::PatienceBudget;

/// Generate scenarios from seed edges through a specific thinking style lens.
pub fn generate_scenarios(
    question: &str,
    seed_edges: &[(String, String, String, f64, f64)], // (src, rel, tgt, freq, conf)
    style: ThinkingStyle,
    budget: &PatienceBudget,
) -> Vec<Scenario> {
    let cluster = style.cluster();

    // Convert seed edges to CausalSteps
    let steps: Vec<CausalStep> = seed_edges.iter().map(|(src, rel, tgt, freq, conf)| {
        CausalStep {
            source: src.clone(),
            relationship: rel.clone(),
            target: tgt.clone(),
            frequency: *freq,
            confidence: *conf,
            derivation: Derivation::Observed,
        }
    }).collect();

    match cluster {
        ThinkingCluster::Convergent => generate_deep_chains(question, &steps, style, budget),
        ThinkingCluster::Divergent => generate_lateral_scenarios(question, &steps, style, budget),
        ThinkingCluster::Attention => generate_focused_chain(question, &steps, style, budget),
        ThinkingCluster::Speed => generate_heuristic_scenario(question, &steps, style, budget),
    }
}

/// Convergent: follow the longest chain, high confidence threshold.
/// "If A→B→C→D, what happens at D?"
fn generate_deep_chains(
    question: &str,
    steps: &[CausalStep],
    style: ThinkingStyle,
    budget: &PatienceBudget,
) -> Vec<Scenario> {
    let min_confidence = 0.5;
    let max_chain_length = (budget.result_threshold / 1000).max(3).min(10);

    // Build adjacency: source → [(target, step)]
    let mut adj: std::collections::HashMap<&str, Vec<&CausalStep>> = std::collections::HashMap::new();
    for step in steps {
        adj.entry(&step.source).or_default().push(step);
    }

    // Find all chains from any node
    let mut scenarios = Vec::new();
    for step in steps {
        let mut chain = vec![step.clone()];
        let mut current = &step.target;
        let mut visited = std::collections::HashSet::new();
        visited.insert(step.source.as_str());

        // Follow chain
        while chain.len() < max_chain_length {
            if visited.contains(current.as_str()) { break; }
            visited.insert(current);

            if let Some(next_steps) = adj.get(current.as_str()) {
                // Pick highest confidence next step
                if let Some(best) = next_steps.iter()
                    .filter(|s| s.confidence >= min_confidence)
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
                {
                    chain.push((*best).clone());
                    current = &best.target;
                } else {
                    break;
                }
            } else {
                // No outgoing edges — try NARS deduction to extend
                let inferred = infer_next_step(current, steps, min_confidence);
                if let Some(inferred_step) = inferred {
                    let target = inferred_step.target.clone();
                    chain.push(inferred_step);
                    // Can't follow further without owned string
                    break;
                } else {
                    break;
                }
            }
        }

        if chain.len() >= 2 {
            let overall_confidence = chain.iter()
                .map(|s| s.confidence)
                .product::<f64>();

            scenarios.push(Scenario {
                name: format!("Sequential: {} → ... → {}",
                    chain.first().map(|s| s.source.as_str()).unwrap_or("?"),
                    chain.last().map(|s| s.target.as_str()).unwrap_or("?")),
                description: format!("{} (deep chain, {} steps)", question, chain.len()),
                chain,
                confidence: overall_confidence,
                style,
                time_horizon: max_chain_length,
                blind_spots: vec![
                    "Assumes sequential escalation without de-escalation".into(),
                    "Does not account for third-party intervention".into(),
                ],
            });
        }
    }

    // Sort by chain length × confidence
    scenarios.sort_by(|a, b| {
        let score_a = a.chain.len() as f64 * a.confidence;
        let score_b = b.chain.len() as f64 * b.confidence;
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    scenarios.truncate(3); // Top 3 deep chains
    scenarios
}

/// Divergent: find lateral connections, low threshold, surprise discovery.
/// "What unexpected connections exist?"
fn generate_lateral_scenarios(
    question: &str,
    steps: &[CausalStep],
    style: ThinkingStyle,
    budget: &PatienceBudget,
) -> Vec<Scenario> {
    let min_confidence = 0.2; // Low threshold — find weak signals
    let mut scenarios = Vec::new();

    // Find nodes that appear as both source and target of different relationships
    let mut as_source: std::collections::HashMap<&str, Vec<&CausalStep>> = std::collections::HashMap::new();
    let mut as_target: std::collections::HashMap<&str, Vec<&CausalStep>> = std::collections::HashMap::new();

    for step in steps {
        as_source.entry(&step.source).or_default().push(step);
        as_target.entry(&step.target).or_default().push(step);
    }

    // Abduction: A→B and C→B ⟹ A and C are related
    for (target, incoming) in &as_target {
        if incoming.len() >= 2 {
            for i in 0..incoming.len() {
                for j in (i + 1)..incoming.len() {
                    let a = incoming[i];
                    let b = incoming[j];
                    // Abductive inference
                    let freq = a.frequency;
                    let conf = a.confidence * b.confidence * b.frequency;
                    if conf >= min_confidence {
                        let chain = vec![
                            a.clone(),
                            CausalStep {
                                source: a.source.clone(),
                                relationship: "ABDUCED_LINK".into(),
                                target: b.source.clone(),
                                frequency: freq,
                                confidence: conf,
                                derivation: Derivation::Abduction,
                            },
                            b.clone(),
                        ];
                        scenarios.push(Scenario {
                            name: format!("Lateral: {} ↔ {} via {}",
                                a.source, b.source, target),
                            description: format!("{} (abductive discovery)", question),
                            chain,
                            confidence: conf,
                            style,
                            time_horizon: 2,
                            blind_spots: vec![
                                "Abductive inference is the weakest form — may be spurious".into(),
                                "Shared target doesn't imply shared cause".into(),
                            ],
                        });
                    }
                }
            }
        }
    }

    // Induction: A→B and A→C ⟹ B and C are related
    for (source, outgoing) in &as_source {
        if outgoing.len() >= 2 {
            for i in 0..outgoing.len() {
                for j in (i + 1)..outgoing.len() {
                    let a = outgoing[i];
                    let b = outgoing[j];
                    let freq = b.frequency;
                    let conf = a.confidence * b.confidence * a.frequency;
                    if conf >= min_confidence {
                        let chain = vec![
                            a.clone(),
                            CausalStep {
                                source: a.target.clone(),
                                relationship: "INDUCED_LINK".into(),
                                target: b.target.clone(),
                                frequency: freq,
                                confidence: conf,
                                derivation: Derivation::Induction,
                            },
                            b.clone(),
                        ];
                        scenarios.push(Scenario {
                            name: format!("Induced: {} ↔ {} from {}",
                                a.target, b.target, source),
                            description: format!("{} (inductive discovery)", question),
                            chain,
                            confidence: conf,
                            style,
                            time_horizon: 3,
                            blind_spots: vec![
                                "Induction from shared source is speculative".into(),
                            ],
                        });
                    }
                }
            }
        }
    }

    scenarios.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    scenarios.truncate(5); // Top 5 lateral discoveries
    scenarios
}

/// Attention/Focused: single strongest chain, narrow fan-out.
fn generate_focused_chain(
    question: &str,
    steps: &[CausalStep],
    style: ThinkingStyle,
    _budget: &PatienceBudget,
) -> Vec<Scenario> {
    if steps.is_empty() {
        return vec![];
    }

    // Find the single strongest path
    let mut best_chain: Vec<CausalStep> = Vec::new();
    let mut best_confidence = 0.0;

    // Start from each step, follow highest confidence
    for start in steps {
        let mut chain = vec![start.clone()];
        let mut current = &start.target;
        let mut conf_product = start.confidence;

        for other in steps {
            if other.source == *current && conf_product * other.confidence > best_confidence {
                chain.push(other.clone());
                conf_product *= other.confidence;
                current = &other.target;
            }
        }

        if conf_product > best_confidence {
            best_confidence = conf_product;
            best_chain = chain;
        }
    }

    if best_chain.is_empty() {
        return vec![];
    }

    vec![Scenario {
        name: format!("Kill chain: {} → {}",
            best_chain.first().map(|s| s.source.as_str()).unwrap_or("?"),
            best_chain.last().map(|s| s.target.as_str()).unwrap_or("?")),
        description: format!("{} (focused, single strongest path)", question),
        chain: best_chain,
        confidence: best_confidence,
        style,
        time_horizon: 1,
        blind_spots: vec![
            "Only follows strongest signal — misses parallel chains".into(),
            "No consideration of defensive responses".into(),
        ],
    }]
}

/// Speed/Intuitive: fast heuristic, top-1 most likely step.
fn generate_heuristic_scenario(
    question: &str,
    steps: &[CausalStep],
    style: ThinkingStyle,
    _budget: &PatienceBudget,
) -> Vec<Scenario> {
    // Just pick the highest confidence step
    let best = steps.iter()
        .max_by(|a, b| (a.frequency * a.confidence)
            .partial_cmp(&(b.frequency * b.confidence))
            .unwrap_or(std::cmp::Ordering::Equal));

    match best {
        Some(step) => vec![Scenario {
            name: format!("Snap: {} → {}", step.source, step.target),
            description: format!("{} (System 1 heuristic)", question),
            chain: vec![step.clone()],
            confidence: step.frequency * step.confidence,
            style,
            time_horizon: 0,
            blind_spots: vec![
                "System 1 response — no deliberation".into(),
                "Single step, no causal chain".into(),
                "May be anchoring bias on most salient edge".into(),
            ],
        }],
        None => vec![],
    }
}

/// NARS deduction: given a target node, find if any two edges can chain to it.
fn infer_next_step(
    from: &str,
    steps: &[CausalStep],
    min_confidence: f64,
) -> Option<CausalStep> {
    // Find edges FROM this node
    let outgoing: Vec<&CausalStep> = steps.iter()
        .filter(|s| s.source == from)
        .collect();

    // Find edges whose source matches any of our outgoing targets
    for out in &outgoing {
        for step in steps {
            if step.source == out.target && step.source != from {
                let freq = out.frequency * step.frequency;
                let conf = out.confidence * step.confidence * out.frequency * step.frequency;
                if conf >= min_confidence {
                    return Some(CausalStep {
                        source: from.to_string(),
                        relationship: format!("DEDUCED_{}", step.relationship),
                        target: step.target.clone(),
                        frequency: freq,
                        confidence: conf,
                        derivation: Derivation::Deduction,
                    });
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevation::budget::budget_for_cluster;

    fn kill_chain_edges() -> Vec<(String, String, String, f64, f64)> {
        vec![
            ("Lavender".into(), "TARGETS".into(), "Gaza_Combatant".into(), 0.95, 0.87),
            ("Gospel".into(), "CONFIRMS".into(), "Lavender_Target".into(), 0.90, 0.82),
            ("Fire_Factory".into(), "STRIKES".into(), "Confirmed_Target".into(), 0.88, 0.79),
            ("Iron_Dome".into(), "INTERCEPTS".into(), "Incoming_Rocket".into(), 0.92, 0.85),
            ("Patriot".into(), "DEFENDS".into(), "Military_Base".into(), 0.85, 0.78),
            ("Iran_AD".into(), "PROTECTS".into(), "Tehran".into(), 0.80, 0.70),
            ("F35".into(), "PENETRATES".into(), "Iran_AD".into(), 0.75, 0.65),
            ("Iran_AD".into(), "RETALIATES".into(), "Israeli_Base".into(), 0.60, 0.45),
        ]
    }

    #[test]
    fn test_analytical_produces_deep_chains() {
        let edges = kill_chain_edges();
        let budget = budget_for_cluster(ThinkingCluster::Convergent);
        let scenarios = generate_scenarios(
            "Iran strike scenario",
            &edges,
            ThinkingStyle::Analytical,
            &budget,
        );
        assert!(!scenarios.is_empty());
        // Analytical should produce multi-step chains
        assert!(scenarios.iter().any(|s| s.chain.len() >= 2));
    }

    #[test]
    fn test_creative_finds_lateral_connections() {
        let edges = kill_chain_edges();
        let budget = budget_for_cluster(ThinkingCluster::Divergent);
        let scenarios = generate_scenarios(
            "What connections does everyone miss?",
            &edges,
            ThinkingStyle::Creative,
            &budget,
        );
        // Creative should find abductive/inductive links
        assert!(scenarios.iter().any(|s|
            s.chain.iter().any(|step|
                step.derivation == Derivation::Abduction
                || step.derivation == Derivation::Induction
            )
        ));
    }

    #[test]
    fn test_focused_produces_single_chain() {
        let edges = kill_chain_edges();
        let budget = budget_for_cluster(ThinkingCluster::Attention);
        let scenarios = generate_scenarios(
            "Kill chain",
            &edges,
            ThinkingStyle::Focused,
            &budget,
        );
        assert_eq!(scenarios.len(), 1);
    }

    #[test]
    fn test_intuitive_is_fast() {
        let edges = kill_chain_edges();
        let budget = budget_for_cluster(ThinkingCluster::Speed);
        let scenarios = generate_scenarios(
            "Quick read",
            &edges,
            ThinkingStyle::Intuitive,
            &budget,
        );
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].chain.len(), 1); // Single step
        assert!(!scenarios[0].blind_spots.is_empty()); // Acknowledges limitations
    }

    #[test]
    fn test_different_styles_different_blind_spots() {
        let edges = kill_chain_edges();
        let analytical = generate_scenarios(
            "test", &edges, ThinkingStyle::Analytical,
            &budget_for_cluster(ThinkingCluster::Convergent),
        );
        let creative = generate_scenarios(
            "test", &edges, ThinkingStyle::Creative,
            &budget_for_cluster(ThinkingCluster::Divergent),
        );

        // Different styles should identify different blind spots
        let analytical_spots: Vec<&str> = analytical.iter()
            .flat_map(|s| s.blind_spots.iter().map(|b| b.as_str()))
            .collect();
        let creative_spots: Vec<&str> = creative.iter()
            .flat_map(|s| s.blind_spots.iter().map(|b| b.as_str()))
            .collect();
        assert_ne!(analytical_spots, creative_spots);
    }
}
