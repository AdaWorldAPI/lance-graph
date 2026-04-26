//! Strategy selection: who decides which strategies participate.

use crate::traits::{PlanStrategy, PlanContext, PipelinePhase};

/// Strategy selection mode.
#[derive(Debug, Clone)]
pub enum StrategySelector {
    /// User explicitly names strategies: "use dp_join + sigma_scan"
    Explicit(Vec<String>),

    /// AGI chooses based on thinking style + MUL assessment.
    Resonance {
        /// 23D sparse thinking style vector.
        thinking_style: Vec<f64>,
        /// Free Will Modifier from MUL gate.
        mul_modifier: f64,
        /// Compass score (if navigating unknown territory).
        compass_score: f64,
    },

    /// Auto-select: each strategy reports affinity, top-N compose.
    Auto {
        /// Maximum strategies per pipeline phase.
        max_per_phase: usize,
        /// Minimum affinity to participate.
        min_affinity: f32,
    },
}

impl Default for StrategySelector {
    fn default() -> Self {
        Self::Auto {
            max_per_phase: 2,
            min_affinity: 0.3,
        }
    }
}

/// Select strategies from the registry based on the selector mode.
pub fn select_strategies<'a>(
    selector: &StrategySelector,
    strategies: &'a [Box<dyn PlanStrategy>],
    context: &PlanContext,
) -> Vec<&'a dyn PlanStrategy> {
    match selector {
        StrategySelector::Explicit(names) => {
            // Return only explicitly named strategies, in pipeline order.
            let mut selected: Vec<&dyn PlanStrategy> = strategies.iter()
                .filter(|s| names.iter().any(|n| n.eq_ignore_ascii_case(s.name())))
                .map(|s| s.as_ref())
                .collect();
            selected.sort_by_key(|s| s.capability().phase());
            selected
        }

        StrategySelector::Resonance { thinking_style, mul_modifier, compass_score: _ } => {
            // Score each strategy: affinity * mul_modifier * style_alignment
            let mut scored: Vec<(&dyn PlanStrategy, f32)> = strategies.iter()
                .map(|s| {
                    let base_affinity = s.affinity(context);
                    let modifier = *mul_modifier as f32;
                    // Style alignment: if the strategy's capability matches the thinking style,
                    // boost its score. Exploratory styles boost VectorScan/TruthPropagation.
                    let style_boost = style_alignment(s.capability().phase(), thinking_style);
                    let score = base_affinity * modifier * (1.0 + style_boost);
                    (s.as_ref(), score)
                })
                .filter(|(_, score)| *score > 0.2)
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top strategies, ensuring at least one per phase
            let mut selected = Vec::new();
            let mut phase_counts = std::collections::HashMap::new();

            for (strategy, _score) in scored {
                let phase = strategy.capability().phase();
                let count = phase_counts.entry(phase).or_insert(0);
                if *count < 2 {
                    selected.push(strategy);
                    *count += 1;
                }
            }

            selected.sort_by_key(|s| s.capability().phase());
            selected
        }

        StrategySelector::Auto { max_per_phase, min_affinity } => {
            // Each strategy scores affinity, top-N per phase compose.
            let mut by_phase: std::collections::HashMap<PipelinePhase, Vec<(&dyn PlanStrategy, f32)>> =
                std::collections::HashMap::new();

            for strategy in strategies {
                let affinity = strategy.affinity(context);
                if affinity >= *min_affinity {
                    by_phase.entry(strategy.capability().phase())
                        .or_default()
                        .push((strategy.as_ref(), affinity));
                }
            }

            let mut selected = Vec::new();

            // Sort phases and pick top strategies per phase
            let mut phases: Vec<_> = by_phase.keys().copied().collect();
            phases.sort();

            for phase in phases {
                if let Some(candidates) = by_phase.get_mut(&phase) {
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    for (strategy, _) in candidates.iter().take(*max_per_phase) {
                        selected.push(*strategy);
                    }
                }
            }

            selected
        }
    }
}

/// Compute thinking style alignment with a pipeline phase.
fn style_alignment(phase: PipelinePhase, thinking_style: &[f64]) -> f32 {
    if thinking_style.is_empty() {
        return 0.0;
    }

    // Map style dimensions to phase preferences.
    // Indices into the 23D sparse vector (from crewai-rust):
    // [0]=depth, [1]=somatic, [2]=emotional, [3]=creative, [4]=analytical,
    // [5]=flow, [6]=vitality, [7]=transcendent, ...
    let analytical = thinking_style.get(4).copied().unwrap_or(0.0) as f32;
    let creative = thinking_style.get(3).copied().unwrap_or(0.0) as f32;
    let depth = thinking_style.first().copied().unwrap_or(0.0) as f32;

    match phase {
        PipelinePhase::Parse => 0.0,
        PipelinePhase::Plan => analytical * 0.5 + depth * 0.3,
        PipelinePhase::Optimize => analytical * 0.7,
        PipelinePhase::Physicalize => creative * 0.3 + depth * 0.4,
        PipelinePhase::Execute => 0.1, // Always needed
    }
}
