//! Trust Qualia — the felt-sense of knowing.
//!
//! Four orthogonal dimensions (geometric mean composite):
//! - Competence: "can I do this?"
//! - Source: "is the information reliable?"
//! - Environment: "is context stable?"
//! - Calibration: "are my estimates accurate?"
//!
//! Maps to texture: crystalline | solid | fuzzy | murky | dissonant

use super::SituationInput;

/// Trust assessment result.
#[derive(Debug, Clone)]
pub struct TrustQualia {
    /// "Can I do this?" (0..1, from Brier score history).
    pub competence: f64,
    /// "Is the information reliable?" (0..1, NARS confidence).
    pub source: f64,
    /// "Is context stable?" (0..1, input entropy).
    pub environment: f64,
    /// "Are my estimates accurate?" (0..1, Brier error).
    pub calibration: f64,
    /// Texture level derived from composite.
    pub texture: TrustTexture,
}

/// Texture level — the felt quality of trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustTexture {
    /// ≥ 0.85 — frozen, reliable.
    Crystalline,
    /// 0.65..0.85 — stable, testable.
    Solid,
    /// 0.45..0.65 — uncertain, revisable.
    Fuzzy,
    /// 0.25..0.45 — suspect, needs repair.
    Murky,
    /// < 0.25 — unreliable, conflict.
    Dissonant,
}

impl TrustQualia {
    /// Composite trust = geometric mean of 4 dimensions.
    /// Weakest link property: single low dimension drags composite down.
    pub fn composite_score(&self) -> f64 {
        (self.competence * self.source * self.environment * self.calibration).powf(0.25)
    }
}

impl TrustTexture {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.85 {
            Self::Crystalline
        } else if score >= 0.65 {
            Self::Solid
        } else if score >= 0.45 {
            Self::Fuzzy
        } else if score >= 0.25 {
            Self::Murky
        } else {
            Self::Dissonant
        }
    }

    /// Whether this texture level is safe for autonomous operation.
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::Crystalline | Self::Solid)
    }
}

/// Assess trust from situation input.
pub fn assess(input: &SituationInput) -> TrustQualia {
    let competence = input.demonstrated_competence;
    let source = input.source_reliability;
    let environment = input.environment_stability;
    let calibration = input.calibration_accuracy;

    let composite = (competence * source * environment * calibration).powf(0.25);
    let texture = TrustTexture::from_score(composite);

    TrustQualia {
        competence,
        source,
        environment,
        calibration,
        texture,
    }
}
