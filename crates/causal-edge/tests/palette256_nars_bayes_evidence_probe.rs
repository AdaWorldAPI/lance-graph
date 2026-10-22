//! Distinguish NARS independent evidence count, Bayesian posterior, and
//! quantized palette projection. A similarity score is not a sample size.
#[derive(Clone, Copy)]
struct Evidence {
    positive: u64,
    negative: u64,
    sources: u64,
}
impl Evidence {
    fn merge(self, other: Self) -> Option<Self> {
        (self.sources & other.sources == 0).then_some(Self {
            positive: self.positive.checked_add(other.positive)?,
            negative: self.negative.checked_add(other.negative)?,
            sources: self.sources | other.sources,
        })
    }
    fn frequency(self) -> f64 {
        self.positive as f64 / (self.positive + self.negative) as f64
    }
    fn nars_confidence(self) -> f64 {
        let n = self.positive + self.negative;
        n as f64 / (n as f64 + 1.0)
    }
    fn jeffreys_posterior_mean(self) -> f64 {
        (self.positive as f64 + 0.5) / ((self.positive + self.negative) as f64 + 1.0)
    }
}
#[test]
fn independent_equal_frequency_groups_do_not_increase_frequency() {
    let a = Evidence { positive: 7, negative: 3, sources: 1 };
    let b = Evidence { positive: 7, negative: 3, sources: 2 };
    let both = a.merge(b).unwrap();
    assert!((both.frequency() - 0.7).abs() < 1e-12);
    assert!((both.nars_confidence() - (20.0 / 21.0)).abs() < 1e-12);
    assert!((both.jeffreys_posterior_mean() - 14.5 / 21.0).abs() < 1e-12);
}
#[test]
fn frequency_is_not_evidence_mass_and_duplicate_sources_do_not_count() {
    let low = Evidence { positive: 1, negative: 1, sources: 1 };
    let high = Evidence { positive: 50, negative: 50, sources: 2 };
    assert_eq!(low.frequency(), high.frequency());
    assert!(high.nars_confidence() > low.nars_confidence());
    assert!(low.merge(low).is_none(), "do not double-count the same source");
}
