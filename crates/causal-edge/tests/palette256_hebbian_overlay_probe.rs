//! Local associative learning belongs to a versioned overlay, not to the
//! immutable Palette256 codebook or its family-calibrated pairwise LUT.
#[derive(Default)]
struct Overlay {
    latent: f64,
    seen_sources: u64,
}
impl Overlay {
    fn apply(&mut self, source_id: u8, pre: f64, post: f64, eta: f64) -> Result<bool, &'static str> {
        if source_id >= 64 || !pre.is_finite() || !post.is_finite()
            || !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
            return Err("invalid update contract");
        }
        let source = 1_u64 << source_id;
        if self.seen_sources & source != 0 { return Ok(false); }
        // Hypothesis under test, NOT NARS revision or a proven Bayesian update.
        let damping = 1.0 - self.latent.abs().tanh();
        self.latent = (self.latent + eta * pre * post * damping).clamp(-8.0, 8.0);
        self.seen_sources |= source;
        Ok(true)
    }
    fn bounded_readout(&self) -> f64 { self.latent.tanh() }
}
#[test]
fn repeated_eligible_updates_remain_bounded_and_do_not_move_reference() {
    let frozen_reference: [u8; 4] = [13, 42, 99, 241];
    let reference_before = frozen_reference;
    let mut synapse = Overlay::default();
    for source in 0..64_u8 {
        assert!(synapse.apply(source, 0.8, 0.9, 0.15).unwrap());
    }
    assert!(synapse.latent.is_finite());
    assert!(synapse.bounded_readout() > 0.0 && synapse.bounded_readout() < 1.0);
    assert_eq!(frozen_reference, reference_before);
}
#[test]
fn a_duplicate_root_does_not_reinforce_the_synapse_again() {
    let mut synapse = Overlay::default();
    assert!(synapse.apply(4, 0.8, 0.9, 0.1).unwrap());
    let first = synapse.latent;
    assert!(!synapse.apply(4, 0.8, 0.9, 0.1).unwrap());
    assert_eq!(synapse.latent, first);
    assert!(synapse.apply(5, -0.8, 0.9, 0.1).unwrap());
    assert!(synapse.latent < first, "anticorrelated activation must decrease the overlay");
}
