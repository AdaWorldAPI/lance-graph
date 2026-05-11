//! Dunning-Kruger Detector — the humility engine.
//!
//! Tracks the gap between felt_competence and demonstrated_competence
//! to determine DK position on the curve.

use super::SituationInput;

/// Position on the Dunning-Kruger curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DkPosition {
    /// HIGH confidence, LOW experience — DANGEROUS.
    MountStupid,
    /// Aware of gaps, cautious.
    ValleyOfDespair,
    /// Building real competence.
    SlopeOfEnlightenment,
    /// Calibrated confidence.
    PlateauOfMastery,
}

impl DkPosition {
    /// Humility factor: how much to discount confidence based on DK position.
    pub fn humility_factor(&self) -> f64 {
        match self {
            Self::MountStupid => 0.3,           // Heavily discounted
            Self::ValleyOfDespair => 0.7,       // Cautious but aware
            Self::SlopeOfEnlightenment => 0.85, // Building competence
            Self::PlateauOfMastery => 1.0,      // Full confidence earned
        }
    }

    /// Whether this position is safe for autonomous action.
    pub fn is_safe(&self) -> bool {
        !matches!(self, Self::MountStupid)
    }
}

/// Detect DK position from the gap between felt and demonstrated competence.
pub fn detect(input: &SituationInput) -> DkPosition {
    let gap = input.felt_competence - input.demonstrated_competence;
    let demonstrated = input.demonstrated_competence;

    // Mount Stupid: high felt, low demonstrated (big positive gap)
    if gap > 0.3 && demonstrated < 0.4 {
        DkPosition::MountStupid
    }
    // Valley: low felt, becoming aware of gaps
    else if input.felt_competence < 0.4 && demonstrated < 0.5 {
        DkPosition::ValleyOfDespair
    }
    // Plateau: both high, well-calibrated (small gap)
    else if demonstrated > 0.7 && gap.abs() < 0.15 {
        DkPosition::PlateauOfMastery
    }
    // Slope: demonstrated growing, felt catching up
    else {
        DkPosition::SlopeOfEnlightenment
    }
}

/// DK detector with history tracking (for learning loop).
pub struct DkDetector {
    history: Vec<(f64, f64)>, // (felt, demonstrated) pairs
}

impl Default for DkDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DkDetector {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    pub fn record(&mut self, felt: f64, demonstrated: f64) {
        self.history.push((felt, demonstrated));
        // Keep last 100 observations
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Trend: is the gap narrowing (learning) or widening (overconfidence)?
    pub fn trend(&self) -> DkTrend {
        if self.history.len() < 2 {
            return DkTrend::Stable;
        }

        let recent = &self.history[self.history.len().saturating_sub(10)..];
        let gaps: Vec<f64> = recent.iter().map(|(f, d)| f - d).collect();

        if gaps.len() < 2 {
            return DkTrend::Stable;
        }

        let first_half: f64 = gaps[..gaps.len() / 2].iter().sum::<f64>() / (gaps.len() / 2) as f64;
        let second_half: f64 =
            gaps[gaps.len() / 2..].iter().sum::<f64>() / (gaps.len() - gaps.len() / 2) as f64;

        let delta = second_half - first_half;

        if delta < -0.05 {
            DkTrend::Learning // Gap narrowing
        } else if delta > 0.05 {
            DkTrend::Overconfident // Gap widening
        } else {
            DkTrend::Stable
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DkTrend {
    Learning,
    Overconfident,
    Stable,
}
