//! Pearl's Causal Hierarchy as 3-bit Mask.
//!
//! The 2³ decomposition of SPO gives 8 projections,
//! mapping directly to Pearl's three levels of causal reasoning.

/// The 3-bit causal projection mask.
///
/// Each bit enables one SPO plane in distance computations.
/// Switching Pearl levels = flipping bits.
///
/// ## Pearl's Ladder
///
/// ```text
/// S_O (0b101) = Level 1: Association     P(Y|X)
///   "Patients (S) with good joints (O)" — pure correlation
///
/// _PO (0b011) = Level 2: Intervention    P(Y|do(X))
///   "Prophylaxis (P) → joint improvement (O)" — confounders (S) projected out
///
/// SPO (0b111) = Level 3: Counterfactual  P(Y_x|X',Y')
///   "THIS patient (S) under THAT treatment (P) → predicted outcome (O)"
///
/// SP_ (0b110) = Confounder Detection
///   "Which patients (S) get which treatments (P)?" — confounding by indication
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CausalMask {
    /// No planes active. Aggregate prior.
    None = 0b000,
    /// Object only. Outcome marginal.
    O = 0b001,
    /// Predicate only. Intervention marginal.
    P = 0b010,
    /// Predicate + Object. **Level 2: Intervention (do-calculus).**
    /// Subject confounders projected out by construction.
    PO = 0b011,
    /// Subject only. Entity marginal.
    S = 0b100,
    /// Subject + Object. **Level 1: Association.**
    /// Pure observational correlation.
    SO = 0b101,
    /// Subject + Predicate. **Confounder detection.**
    /// Reveals confounding by indication.
    SP = 0b110,
    /// All planes active. **Level 3: Counterfactual.**
    /// Full structural causal model.
    SPO = 0b111,
}

impl CausalMask {
    /// Construct from raw 3-bit value.
    #[inline]
    pub fn from_bits(v: u8) -> Self {
        match v & 0b111 {
            0b000 => Self::None,
            0b001 => Self::O,
            0b010 => Self::P,
            0b011 => Self::PO,
            0b100 => Self::S,
            0b101 => Self::SO,
            0b110 => Self::SP,
            _ => Self::SPO,
        }
    }

    /// Pearl level name.
    pub fn pearl_level(self) -> &'static str {
        match self {
            Self::None => "Prior",
            Self::O => "Outcome Marginal",
            Self::P => "Intervention Marginal",
            Self::PO => "Level 2: Intervention P(Y|do(X))",
            Self::S => "Entity Marginal",
            Self::SO => "Level 1: Association P(Y|X)",
            Self::SP => "Confounder Detection",
            Self::SPO => "Level 3: Counterfactual",
        }
    }

    /// Number of active planes (0-3).
    #[inline]
    pub fn active_count(self) -> u8 {
        (self as u8).count_ones() as u8
    }

    /// Is this an interventional projection (do-calculus)?
    #[inline]
    pub fn is_interventional(self) -> bool {
        self == Self::PO
    }

    /// Is this a counterfactual projection?
    #[inline]
    pub fn is_counterfactual(self) -> bool {
        self == Self::SPO
    }

    /// Detect Simpson's Paradox potential:
    /// S_O and _PO disagree in direction → confounding present.
    /// Requires checking both projections' direction triads.
    pub fn simpsons_paradox_risk(so_direction: u8, po_direction: u8) -> bool {
        // If S_O shows negative association (O pathological)
        // but _PO shows positive effect (O healthy under P),
        // Simpson's Paradox is present.
        let so_o_path = so_direction & 0b100 != 0; // O pathological in S_O
        let po_o_path = po_direction & 0b100 != 0;  // O pathological in _PO
        so_o_path != po_o_path // disagree → Simpson's
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_eight_projections() {
        for v in 0u8..=7 {
            let mask = CausalMask::from_bits(v);
            assert_eq!(mask as u8, v);
        }
    }

    #[test]
    fn test_pearl_levels() {
        assert!(CausalMask::SO.pearl_level().contains("Association"));
        assert!(CausalMask::PO.pearl_level().contains("Intervention"));
        assert!(CausalMask::SPO.pearl_level().contains("Counterfactual"));
    }

    #[test]
    fn test_simpsons_detection() {
        // S_O: outcome pathological (bit 2 set)
        // _PO: outcome healthy (bit 2 clear)
        assert!(CausalMask::simpsons_paradox_risk(0b100, 0b000));
        // Same direction → no Simpson's
        assert!(!CausalMask::simpsons_paradox_risk(0b100, 0b100));
    }
}
