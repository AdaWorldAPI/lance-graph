//! Auto-detect thinking style from the 18D qualia vector.
//!
//! EmbedAnything pattern: "auto_detect by architecture". Here the
//! "architecture" is qualia shape. The driver hands the top-row qualia
//! to `auto_style` which returns an ordinal in 0..12.
//!
//! Rules (coarse but deterministic, no forward pass):
//!
//! ```text
//! certainty (Q4) dominant  + low urgency (Q5)   → Analytical/Deliberate
//! arousal (Q6)   dominant  + high exploration   → Creative/Exploratory
//! urgency (Q5)   dominant  + high activation    → Focused/Intuitive
//! depth (Q3)     dominant                       → Metacognitive/Abstract
//! valence (Q0)   low       + activation high    → Divergent
//! ```
//!
//! If nothing dominates (`max < threshold`), fall back to Deliberate.

use lance_graph_contract::cognitive_shader::StyleSelector;
use crate::bindspace::QUALIA_DIMS;

/// Mapping from qualia shape to a style ordinal (0..11 matches
/// `thinking_engine::cognitive_stack::ThinkingStyle::all()`).
pub const DELIBERATE: u8 = 0;
pub const ANALYTICAL: u8 = 1;
pub const CONVERGENT: u8 = 2;
pub const SYSTEMATIC: u8 = 3;
pub const CREATIVE: u8 = 4;
pub const DIVERGENT: u8 = 5;
pub const EXPLORATORY: u8 = 6;
pub const FOCUSED: u8 = 7;
pub const DIFFUSE: u8 = 8;
pub const PERIPHERAL: u8 = 9;
pub const INTUITIVE: u8 = 10;
pub const METACOGNITIVE: u8 = 11;

/// Qualia → style ordinal.
pub fn style_from_qualia(q: &[f32]) -> u8 {
    if q.len() < QUALIA_DIMS {
        return DELIBERATE;
    }
    let valence = q[0];
    let activation = q[1];
    let _dominance = q[2];
    let depth = q[3];
    let certainty = q[4];
    let urgency = q[5];
    let arousal = q.get(6).copied().unwrap_or(0.0);

    // Rank the dominant axis.
    let score = |x: f32| x.clamp(-1.0, 1.0).abs();
    let (dom_axis, dom_value) = [
        ("certainty", score(certainty)),
        ("arousal", score(arousal)),
        ("urgency", score(urgency)),
        ("depth", score(depth)),
        ("valence", score(valence)),
    ]
    .into_iter()
    .fold(("deliberate", 0.0), |acc, (name, v)| if v > acc.1 { (name, v) } else { acc });

    if dom_value < 0.25 {
        return DELIBERATE;
    }

    match dom_axis {
        "certainty" if urgency.abs() < 0.3 => ANALYTICAL,
        "certainty"                        => CONVERGENT,
        "arousal" if activation > 0.5       => CREATIVE,
        "arousal"                          => EXPLORATORY,
        "urgency" if activation > 0.5       => FOCUSED,
        "urgency"                          => INTUITIVE,
        "depth" if certainty < 0.3         => METACOGNITIVE,
        "depth"                            => SYSTEMATIC,
        "valence" if activation > 0.5       => DIVERGENT,
        "valence"                          => DIFFUSE,
        _ => DELIBERATE,
    }
}

/// Resolve a `StyleSelector` against a qualia row.
pub fn resolve(sel: StyleSelector, qualia_row: &[f32]) -> u8 {
    match sel {
        StyleSelector::Ordinal(n) => n % 12,
        StyleSelector::Named(name) => match name {
            "analytical" => ANALYTICAL,
            "convergent" => CONVERGENT,
            "systematic" => SYSTEMATIC,
            "creative" => CREATIVE,
            "divergent" => DIVERGENT,
            "exploratory" => EXPLORATORY,
            "focused" => FOCUSED,
            "diffuse" => DIFFUSE,
            "peripheral" => PERIPHERAL,
            "intuitive" => INTUITIVE,
            "deliberate" => DELIBERATE,
            "metacognitive" => METACOGNITIVE,
            _ => DELIBERATE,
        },
        StyleSelector::Auto => style_from_qualia(qualia_row),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(vals: &[(usize, f32)]) -> [f32; QUALIA_DIMS] {
        let mut out = [0.0f32; QUALIA_DIMS];
        for &(i, v) in vals { out[i] = v; }
        out
    }

    #[test]
    fn high_certainty_low_urgency_is_analytical() {
        let qv = q(&[(4, 0.9), (5, 0.1)]);
        assert_eq!(style_from_qualia(&qv), ANALYTICAL);
    }

    #[test]
    fn high_arousal_and_activation_is_creative() {
        let qv = q(&[(6, 0.9), (1, 0.8)]);
        assert_eq!(style_from_qualia(&qv), CREATIVE);
    }

    #[test]
    fn high_urgency_is_intuitive_by_default() {
        let qv = q(&[(5, 0.8), (1, 0.2)]);
        assert_eq!(style_from_qualia(&qv), INTUITIVE);
    }

    #[test]
    fn depth_without_certainty_is_metacognitive() {
        let qv = q(&[(3, 0.8), (4, 0.1)]);
        assert_eq!(style_from_qualia(&qv), METACOGNITIVE);
    }

    #[test]
    fn flat_qualia_falls_back_to_deliberate() {
        let qv = [0.0f32; QUALIA_DIMS];
        assert_eq!(style_from_qualia(&qv), DELIBERATE);
    }

    #[test]
    fn resolve_respects_explicit_ordinal() {
        let qv = [0.0f32; QUALIA_DIMS];
        assert_eq!(resolve(StyleSelector::Ordinal(CREATIVE), &qv), CREATIVE);
    }

    #[test]
    fn resolve_names_are_case_sensitive() {
        let qv = [0.0f32; QUALIA_DIMS];
        assert_eq!(resolve(StyleSelector::Named("focused"), &qv), FOCUSED);
        // unknown names fall back
        assert_eq!(resolve(StyleSelector::Named("UPPER"), &qv), DELIBERATE);
    }
}
