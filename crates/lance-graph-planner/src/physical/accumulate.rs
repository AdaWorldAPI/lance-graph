//! ACCUMULATE: Propagate values along graph edges using a semiring.
//!
//! This is the missing piece that nobody has: truth values get accumulated
//! DURING traversal, not as a post-hoc filter.
//!
//! The semiring determines how values combine:
//! - Boolean: AND/OR (reachability)
//! - HammingMin: min Hamming distance along path
//! - Tropical: (min, +) for shortest paths
//! - XorBundle: XOR superposition for resonance
//! - TruthPropagating: NARS deduction at each hop, revision at merge
//! - Palette: bgz17 palette codec accumulation

#[allow(unused_imports)] // Morsel, ColumnData intended for accumulate execution wiring
use super::{ColumnData, Morsel, PhysicalOperator};
use crate::ir::logical_op::SemiringType;

/// ACCUMULATE physical operator.
#[derive(Debug)]
pub struct AccumulateOp {
    /// Semiring to use for accumulation.
    pub semiring: SemiringType,
    /// Child operator (typically ScanOp).
    pub child: Box<dyn PhysicalOperator>,
    /// Estimated output cardinality.
    pub estimated_cardinality: f64,
}

/// Semiring operations trait.
pub trait Semiring: std::fmt::Debug + Send + Sync {
    /// Identity element for addition (⊕).
    fn zero(&self) -> SemiringValue;
    /// Identity element for multiplication (⊗).
    fn one(&self) -> SemiringValue;
    /// Addition: combine two values at a merge point.
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue;
    /// Multiplication: propagate a value along an edge.
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue;
}

/// A value in the semiring.
#[derive(Debug, Clone)]
pub enum SemiringValue {
    Bool(bool),
    Distance(u32),
    Float(f64),
    /// XOR bundle: superposition of fingerprints.
    Fingerprint(Vec<u64>),
    /// NARS truth value: (frequency, confidence).
    Truth {
        frequency: f64,
        confidence: f64,
    },
}

/// Boolean semiring: AND/OR.
#[derive(Debug)]
pub struct BooleanSemiring;

impl Semiring for BooleanSemiring {
    fn zero(&self) -> SemiringValue {
        SemiringValue::Bool(false)
    }
    fn one(&self) -> SemiringValue {
        SemiringValue::Bool(true)
    }
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Bool(a), SemiringValue::Bool(b)) => SemiringValue::Bool(*a || *b),
            _ => SemiringValue::Bool(false),
        }
    }
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Bool(a), SemiringValue::Bool(b)) => SemiringValue::Bool(*a && *b),
            _ => SemiringValue::Bool(false),
        }
    }
}

/// Tropical semiring: (min, +) for shortest paths.
#[derive(Debug)]
pub struct TropicalSemiring;

impl Semiring for TropicalSemiring {
    fn zero(&self) -> SemiringValue {
        SemiringValue::Float(f64::INFINITY)
    }
    fn one(&self) -> SemiringValue {
        SemiringValue::Float(0.0)
    }
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Float(a), SemiringValue::Float(b)) => SemiringValue::Float(a.min(*b)),
            _ => SemiringValue::Float(f64::INFINITY),
        }
    }
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Float(a), SemiringValue::Float(b)) => SemiringValue::Float(a + b),
            _ => SemiringValue::Float(f64::INFINITY),
        }
    }
}

/// XOR Bundle semiring: superposition algebra for resonance.
#[derive(Debug)]
pub struct XorBundleSemiring;

impl Semiring for XorBundleSemiring {
    fn zero(&self) -> SemiringValue {
        SemiringValue::Fingerprint(vec![0u64; 256])
    }
    fn one(&self) -> SemiringValue {
        SemiringValue::Fingerprint(vec![u64::MAX; 256])
    }
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        // Bundle: majority vote (simplified as OR for binary)
        match (a, b) {
            (SemiringValue::Fingerprint(a), SemiringValue::Fingerprint(b)) => {
                let result: Vec<u64> = a.iter().zip(b.iter()).map(|(x, y)| x | y).collect();
                SemiringValue::Fingerprint(result)
            }
            _ => self.zero(),
        }
    }
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        // Bind: XOR (from VSA algebra)
        match (a, b) {
            (SemiringValue::Fingerprint(a), SemiringValue::Fingerprint(b)) => {
                let result: Vec<u64> = a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect();
                SemiringValue::Fingerprint(result)
            }
            _ => self.zero(),
        }
    }
}

/// NARS Truth-Propagating semiring.
/// multiply = NARS deduction (propagate truth along edge)
/// add = NARS revision (merge evidence at node)
#[derive(Debug)]
pub struct TruthPropagatingSemiring;

impl Semiring for TruthPropagatingSemiring {
    fn zero(&self) -> SemiringValue {
        SemiringValue::Truth {
            frequency: 0.5,
            confidence: 0.0,
        }
    }
    fn one(&self) -> SemiringValue {
        SemiringValue::Truth {
            frequency: 1.0,
            confidence: 0.9,
        }
    }

    /// NARS revision: merge evidence from two paths.
    /// f_revised = (f1*c1 + f2*c2) / (c1 + c2)
    /// c_revised = (c1 + c2) / (c1 + c2 + 1)
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (
                SemiringValue::Truth {
                    frequency: f1,
                    confidence: c1,
                },
                SemiringValue::Truth {
                    frequency: f2,
                    confidence: c2,
                },
            ) => {
                let denom = c1 + c2;
                if denom < 1e-10 {
                    return self.zero();
                }
                let f_revised = (f1 * c1 + f2 * c2) / denom;
                let c_revised = denom / (denom + 1.0);
                SemiringValue::Truth {
                    frequency: f_revised,
                    confidence: c_revised,
                }
            }
            _ => self.zero(),
        }
    }

    /// NARS deduction: propagate truth along an edge.
    /// f_conclusion = f_premise * f_edge
    /// c_conclusion = c_premise * c_edge * f_premise * f_edge
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (
                SemiringValue::Truth {
                    frequency: f1,
                    confidence: c1,
                },
                SemiringValue::Truth {
                    frequency: f2,
                    confidence: c2,
                },
            ) => {
                let f_conclusion = f1 * f2;
                let c_conclusion = c1 * c2 * f1 * f2;
                SemiringValue::Truth {
                    frequency: f_conclusion,
                    confidence: c_conclusion,
                }
            }
            _ => self.zero(),
        }
    }
}

/// Hamming-min semiring: track minimum Hamming distance along path.
#[derive(Debug)]
pub struct HammingMinSemiring;

impl Semiring for HammingMinSemiring {
    fn zero(&self) -> SemiringValue {
        SemiringValue::Distance(u32::MAX)
    }
    fn one(&self) -> SemiringValue {
        SemiringValue::Distance(0)
    }
    fn add(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Distance(a), SemiringValue::Distance(b)) => {
                SemiringValue::Distance((*a).min(*b))
            }
            _ => self.zero(),
        }
    }
    fn multiply(&self, a: &SemiringValue, b: &SemiringValue) -> SemiringValue {
        match (a, b) {
            (SemiringValue::Distance(a), SemiringValue::Distance(b)) => {
                SemiringValue::Distance(a.saturating_add(*b))
            }
            _ => self.zero(),
        }
    }
}

/// Create a semiring implementation from a SemiringType.
pub fn create_semiring(semiring_type: SemiringType) -> Box<dyn Semiring> {
    match semiring_type {
        SemiringType::Boolean => Box::new(BooleanSemiring),
        SemiringType::Tropical => Box::new(TropicalSemiring),
        SemiringType::XorBundle => Box::new(XorBundleSemiring),
        SemiringType::TruthPropagating => Box::new(TruthPropagatingSemiring),
        SemiringType::HammingMin => Box::new(HammingMinSemiring),
        SemiringType::Palette => Box::new(BooleanSemiring), // TODO: bgz17 palette semiring
        SemiringType::Custom(_) => Box::new(BooleanSemiring), // Fallback
    }
}

impl PhysicalOperator for AccumulateOp {
    fn name(&self) -> &str {
        "Accumulate"
    }
    fn cardinality(&self) -> f64 {
        self.estimated_cardinality
    }
    fn is_pipeline_breaker(&self) -> bool {
        false
    }
    fn children(&self) -> Vec<&dyn PhysicalOperator> {
        vec![&*self.child]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_propagating_revision() {
        let sr = TruthPropagatingSemiring;
        let a = SemiringValue::Truth {
            frequency: 0.8,
            confidence: 0.7,
        };
        let b = SemiringValue::Truth {
            frequency: 0.6,
            confidence: 0.5,
        };
        let revised = sr.add(&a, &b);

        if let SemiringValue::Truth {
            frequency,
            confidence,
        } = revised
        {
            // f = (0.8*0.7 + 0.6*0.5) / (0.7+0.5) = (0.56+0.30)/1.2 = 0.7167
            assert!((frequency - 0.7167).abs() < 0.01);
            // c = 1.2 / (1.2 + 1.0) = 0.5455
            assert!((confidence - 0.5455).abs() < 0.01);
        } else {
            panic!("Expected Truth value");
        }
    }

    #[test]
    fn test_truth_propagating_deduction() {
        let sr = TruthPropagatingSemiring;
        let premise = SemiringValue::Truth {
            frequency: 0.9,
            confidence: 0.8,
        };
        let edge = SemiringValue::Truth {
            frequency: 0.7,
            confidence: 0.9,
        };
        let conclusion = sr.multiply(&premise, &edge);

        if let SemiringValue::Truth {
            frequency,
            confidence,
        } = conclusion
        {
            assert!((frequency - 0.63).abs() < 0.01); // 0.9 * 0.7
            assert!((confidence - 0.4536).abs() < 0.01); // 0.8 * 0.9 * 0.9 * 0.7
        } else {
            panic!("Expected Truth value");
        }
    }

    #[test]
    fn test_xor_bundle_bind() {
        let sr = XorBundleSemiring;
        let a = SemiringValue::Fingerprint(vec![0xFF00FF00u64; 4]);
        let b = SemiringValue::Fingerprint(vec![0x00FF00FFu64; 4]);
        let bound = sr.multiply(&a, &b);

        if let SemiringValue::Fingerprint(fp) = bound {
            assert_eq!(fp[0], 0xFFFFFFFF); // XOR produces all-ones
        } else {
            panic!("Expected Fingerprint");
        }
    }

    #[test]
    fn test_tropical_shortest_path() {
        let sr = TropicalSemiring;
        // Two paths: cost 3 and cost 5
        let path_a = SemiringValue::Float(3.0);
        let path_b = SemiringValue::Float(5.0);
        let shortest = sr.add(&path_a, &path_b);

        if let SemiringValue::Float(v) = shortest {
            assert_eq!(v, 3.0); // min(3, 5) = 3
        }

        // Extend path_a by edge of cost 2
        let edge = SemiringValue::Float(2.0);
        let extended = sr.multiply(&path_a, &edge);

        if let SemiringValue::Float(v) = extended {
            assert_eq!(v, 5.0); // 3 + 2 = 5
        }
    }
}
