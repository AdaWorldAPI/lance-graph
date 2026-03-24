//! Sigma Chain: Ω → Δ → Φ → Θ → Λ
//!
//! Five cognitive operations tracking the epistemic lifecycle of each thinking atom.
//! From bighorn/extensions/agi_stack thinking_atom.py.
//!
//! - Ω (OMEGA): Observation — grounded in direct experience (Rung 2)
//! - Δ (DELTA): Insight/Hypothesis — abductive leap (Rung 3)
//! - Φ (PHI): Belief — evaluated proposition with confidence (Rung 4)
//! - Θ (THETA): Integration — synthesizes multiple beliefs (Rung 5)
//! - Λ (LAMBDA): Trajectory — action tendency or prediction (Rung 5+)

/// Sigma chain stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SigmaStage {
    /// Ω: Observation. Grounded, highest confidence (0.9). No derives_from.
    Omega,
    /// Δ: Insight. Abductive leap, confidence starts at 0.5.
    Delta,
    /// Φ: Belief. Evaluated proposition, confidence 0.7-0.85.
    Phi,
    /// Θ: Integration. Synthesizes multiple beliefs, averages confidence.
    Theta,
    /// Λ: Trajectory. Action tendency, confidence × 0.8.
    Lambda,
}

impl SigmaStage {
    /// Default confidence for this stage.
    pub fn default_confidence(&self) -> f64 {
        match self {
            Self::Omega => 0.9,
            Self::Delta => 0.5,
            Self::Phi => 0.75,
            Self::Theta => 0.7,
            Self::Lambda => 0.6,
        }
    }

    /// Epistemic rung for this stage (from spectroscopy/rungs.py).
    pub fn rung(&self) -> u8 {
        match self {
            Self::Omega => 2,  // Perceptual
            Self::Delta => 3,  // Conceptual
            Self::Phi => 4,    // Metacognitive
            Self::Theta => 5,  // Systems
            Self::Lambda => 6, // Meta-Systems
        }
    }

    /// Next stage in the chain (if any).
    pub fn next(&self) -> Option<SigmaStage> {
        match self {
            Self::Omega => Some(Self::Delta),
            Self::Delta => Some(Self::Phi),
            Self::Phi => Some(Self::Theta),
            Self::Theta => Some(Self::Lambda),
            Self::Lambda => None,
        }
    }

    /// Temperature range for this rung (from rungs.py).
    pub fn temperature_range(&self) -> (f64, f64) {
        match self {
            Self::Omega => (0.4, 0.7),  // Perceptual
            Self::Delta => (0.2, 0.4),  // Conceptual
            Self::Phi => (0.3, 0.5),    // Metacognitive
            Self::Theta => (0.5, 0.7),  // Systems
            Self::Lambda => (0.5, 0.8), // Meta-Systems
        }
    }
}

/// A thinking atom — epistemic object with confidence, derivation, trajectory.
#[derive(Debug, Clone)]
pub struct ThinkingAtom {
    /// Distinguished name.
    pub dn: String,
    /// Human-readable content.
    pub content: String,
    /// Current sigma stage.
    pub stage: SigmaStage,
    /// Epistemic confidence (0..1).
    pub confidence: f64,
    /// Known propositions with confidence.
    pub knowns: Vec<Proposition>,
    /// Open questions.
    pub unknowns: Vec<String>,
    /// Required assumptions.
    pub assumptions: Vec<String>,
    /// Parent atom(s) this was derived from.
    pub derives_from: Vec<String>,
    /// Anticipated next state.
    pub leads_to: Option<String>,
}

/// A proposition with confidence.
#[derive(Debug, Clone)]
pub struct Proposition {
    pub content: String,
    pub confidence: f64,
    pub source: Option<String>,
}

impl ThinkingAtom {
    /// Create an Omega (observation) atom — ground truth.
    pub fn observe(content: String) -> Self {
        let dn = format!("DN:Atom.Ω.{:08x}", fxhash(&content));
        Self {
            dn,
            content,
            stage: SigmaStage::Omega,
            confidence: 0.9,
            knowns: Vec::new(),
            unknowns: Vec::new(),
            assumptions: Vec::new(),
            derives_from: Vec::new(),
            leads_to: None,
        }
    }

    /// Escalate to the next sigma stage.
    pub fn escalate(&self) -> Option<ThinkingAtom> {
        let next_stage = self.stage.next()?;
        let confidence = match next_stage {
            SigmaStage::Delta => 0.5,
            SigmaStage::Phi => self.confidence * 0.85 + 0.15 * 0.75,
            SigmaStage::Theta => self.confidence * 0.9,
            SigmaStage::Lambda => self.confidence * 0.8,
            SigmaStage::Omega => unreachable!(),
        };

        Some(ThinkingAtom {
            dn: format!("DN:Atom.{:?}.{:08x}", next_stage, fxhash(&self.dn)),
            content: self.content.clone(),
            stage: next_stage,
            confidence,
            knowns: self.knowns.clone(),
            unknowns: self.unknowns.clone(),
            assumptions: self.assumptions.clone(),
            derives_from: vec![self.dn.clone()],
            leads_to: None,
        })
    }

    /// NARS revision: merge two atoms with evidence combination.
    /// f_revised = (f1*c1 + f2*c2) / (c1 + c2)
    /// c_revised = (c1 + c2) / (c1 + c2 + 1)
    pub fn revise(&self, other: &ThinkingAtom) -> ThinkingAtom {
        let c1 = self.confidence;
        let c2 = other.confidence;
        let f1 = self.knowns.first().map(|p| p.confidence).unwrap_or(0.5);
        let f2 = other.knowns.first().map(|p| p.confidence).unwrap_or(0.5);

        let f_revised = if c1 + c2 > 0.0 {
            (f1 * c1 + f2 * c2) / (c1 + c2)
        } else {
            0.5
        };
        let c_revised = (c1 + c2) / (c1 + c2 + 1.0);

        ThinkingAtom {
            dn: format!("DN:Atom.Rev.{:08x}", fxhash(&format!("{}{}", self.dn, other.dn))),
            content: format!("Revision of {} and {}", self.content, other.content),
            stage: self.stage,
            confidence: c_revised,
            knowns: vec![Proposition {
                content: format!("Revised: {}", self.content),
                confidence: f_revised,
                source: Some("nars_revision".into()),
            }],
            unknowns: Vec::new(),
            assumptions: Vec::new(),
            derives_from: vec![self.dn.clone(), other.dn.clone()],
            leads_to: None,
        }
    }
}

/// Simple hash for DN generation.
fn fxhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for b in s.bytes() {
        h = h.wrapping_mul(0x9E3779B9).wrapping_add(b as u32);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_chain() {
        let atom = ThinkingAtom::observe("test observation".into());
        assert_eq!(atom.stage, SigmaStage::Omega);
        assert_eq!(atom.confidence, 0.9);

        let delta = atom.escalate().unwrap();
        assert_eq!(delta.stage, SigmaStage::Delta);
        assert_eq!(delta.confidence, 0.5);
        assert_eq!(delta.derives_from, vec![atom.dn.clone()]);

        let phi = delta.escalate().unwrap();
        assert_eq!(phi.stage, SigmaStage::Phi);

        let theta = phi.escalate().unwrap();
        assert_eq!(theta.stage, SigmaStage::Theta);

        let lambda = theta.escalate().unwrap();
        assert_eq!(lambda.stage, SigmaStage::Lambda);

        assert!(lambda.escalate().is_none()); // End of chain
    }

    #[test]
    fn test_nars_revision() {
        let a = ThinkingAtom {
            dn: "a".into(),
            content: "claim A".into(),
            stage: SigmaStage::Phi,
            confidence: 0.8,
            knowns: vec![Proposition { content: "A is true".into(), confidence: 0.7, source: None }],
            unknowns: vec![],
            assumptions: vec![],
            derives_from: vec![],
            leads_to: None,
        };
        let b = ThinkingAtom {
            dn: "b".into(),
            content: "claim B".into(),
            stage: SigmaStage::Phi,
            confidence: 0.6,
            knowns: vec![Proposition { content: "A is true".into(), confidence: 0.9, source: None }],
            unknowns: vec![],
            assumptions: vec![],
            derives_from: vec![],
            leads_to: None,
        };

        let revised = a.revise(&b);
        // c_revised = (0.8 + 0.6) / (0.8 + 0.6 + 1.0) = 1.4 / 2.4 ≈ 0.583
        assert!(revised.confidence > 0.5 && revised.confidence < 0.7);
        assert_eq!(revised.derives_from.len(), 2);
    }
}
