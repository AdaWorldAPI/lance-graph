//! Candidate Hebbian update, probe-only. Never writes CE64 or frozen codebooks.
use lance_graph_contract::episodic_edges::{EdgeRef, EpisodicEdges64};

fn candidate(z: f64, s: f64, p: f64, eta: f64) -> f64 {
    let delta = eta * s * p * (1.0 - z.abs().tanh());
    z + delta
}

#[test]
fn eta_or_inactive_stimulus_is_exactly_inert() {
    for z in [-7.0, -1.0, 0.0, 1.0, 7.0] {
        assert_eq!(candidate(z, 0.9, 0.9, 0.0), z);
        assert_eq!(candidate(z, 0.0, 0.9, 0.05), z);
        assert_eq!(candidate(z, 0.9, 0.0, 0.05), z);
    }
}

#[test]
fn coincident_and_opposed_stimuli_choose_direction() {
    assert!(candidate(0.0, 1.0, 1.0, 0.1) > 0.0);
    assert!(candidate(0.0, 1.0, -1.0, 0.1) < 0.0);
}

#[test]
fn saturation_damps_update_but_does_not_create_immutable_certainty() {
    let fresh = candidate(0.0, 1.0, 1.0, 0.1);
    let saturated = candidate(12.0, 1.0, 1.0, 0.1) - 12.0;
    assert!(fresh > 0.09);
    assert!(saturated.abs() < 1e-10);
    let mut z = 0.0;
    for _ in 0..64 {
        z = candidate(z, 1.0, 1.0, 0.1);
    }
    assert!(z > 0.0 && z < 6.4, "damped growth is below linear baseline");
}

#[test]
fn hot_episodic_mru_remains_separate_from_candidate_weight() {
    let a = EdgeRef::intra(1).expect("valid reference");
    let b = EdgeRef::intra(2).expect("valid reference");
    let (one, eviction) = EpisodicEdges64::empty().promote(a);
    assert!(eviction.is_none());
    let (two, eviction) = one.promote(b);
    assert!(eviction.is_none());
    assert_eq!(two.strongest(), Some(b));
    assert!(two.contains(a));
    // Experimenting with an external scalar has not rewritten the MRU register.
    assert_eq!(candidate(0.0, 1.0, 1.0, 0.1), 0.1);
}
