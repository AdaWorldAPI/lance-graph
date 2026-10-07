//! The P7a sealed model for the examples (D-GSO-7a, #1369).
//!
//! The model, its satisfaction folds and its builder live in production as
//! `lance_graph_contract::certification` (D-PEARL-PROD-0). This module only
//! re-exports them under the names the examples were written against, adds the
//! fixtures, and reads certifications as the examples' working-name `Contract`.
//! Each including example uses a subset, hence the `dead_code` allowance.
#![allow(dead_code)]

use crate::certification_reading::Contract;
use lance_graph_contract::causal_audit::SupportBasis;
#[allow(unused_imports)] // each including example uses a subset
pub use lance_graph_contract::certification::{
    compare, CertificationModel as Model, ModelBuilder as Builder, NotGrounded, Sat,
};

/// The production certifications, read as the examples' `Contract` (same
/// codes).
pub trait Certify {
    /// `CertificationModel::certification`.
    fn certify(&self) -> Contract;
    /// `CertificationModel::observational_certification`.
    fn certify_observational(&self) -> Contract;
}

impl Certify for Model {
    fn certify(&self) -> Contract {
        Contract::from_certification(self.certification())
    }
    fn certify_observational(&self) -> Contract {
        Contract::from_certification(self.observational_certification())
    }
}

/// A robust single-stratum association: 4/5 exposed vs 1/5 unexposed.
pub fn robust_association() -> Builder {
    let mut b = Builder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

/// Within each stratum A raises Y; pooled, A lowers it (Simpson).
pub fn simpson_population() -> Builder {
    let mut b = Builder::new();
    // stratum 0: mostly exposed, low base rate
    b.cell(0, true, 8, 2); // 0.25
    b.cell(0, false, 2, 0); // 0.00
                            // stratum 1: mostly unexposed, high base rate
    b.cell(1, true, 2, 2); // 1.00
    b.cell(1, false, 8, 6); // 0.75
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

/// A positive randomized trial with two independent intervention sources.
pub fn add_positive_trial(b: &mut Builder) {
    b.arm(true, 6, 5);
    b.arm(false, 6, 1);
    b.sources(SupportBasis::InterventionBacked, &[10, 11]);
}
