//! Bayesian predictive uncertainty is not NARS confidence by renaming.
use causal_edge::CausalEdge64;

#[derive(Clone, Copy)]
struct Posterior {
    successes: u32,
    trials: u32,
}

impl Posterior {
    fn mean(self) -> f64 {
        (self.successes as f64 + 0.5) / (self.trials as f64 + 1.0)
    }
    fn variance(self) -> f64 {
        let a = self.successes as f64 + 0.5;
        let b = (self.trials - self.successes) as f64 + 0.5;
        a * b / ((a + b) * (a + b) * (a + b + 1.0))
    }
}

#[test]
fn same_frequency_does_not_imply_same_uncertainty() {
    let few = Posterior {
        successes: 7,
        trials: 10,
    };
    let many = Posterior {
        successes: 700,
        trials: 1000,
    };
    assert!((7.0_f64 / 10.0 - 700.0 / 1000.0).abs() < f64::EPSILON);
    assert!(many.variance() < few.variance() / 50.0);
    assert!((few.mean() - many.mean()).abs() < 0.02);
}

#[test]
fn ce64_expectation_is_a_quantized_projection_not_the_count_carrier() {
    let sample = Posterior {
        successes: 7,
        trials: 10,
    };
    let mut edge = CausalEdge64(0);
    edge.set_frequency_u8((255.0_f64 * 7.0 / 10.0).round() as u8);
    edge.set_confidence_u8((255.0_f64 * 10.0 / 11.0).round() as u8);
    assert!((edge.expectation() as f64 - sample.mean()).abs() < 0.01);
    assert!(sample.variance() > 0.0);
    // The sample size must be carried separately; the two u8 codes
    // cannot recover an arbitrary exact 1000-observation history.
}

#[test]
fn polarity_and_evidence_mass_are_independent_axes() {
    let a = Posterior {
        successes: 1,
        trials: 2,
    };
    let b = Posterior {
        successes: 500,
        trials: 1000,
    };
    assert!((a.mean() - b.mean()).abs() < 1e-12);
    assert!(a.variance() > b.variance() * 100.0);
}
