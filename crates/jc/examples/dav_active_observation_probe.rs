//! DAV × EWA × Revision — first end-to-end active-observation wiring.
//!
//! This is a diagnostic probe, NOT a JC pillar and NOT a production policy.
//!
//! It asks one deliberately small question:
//!
//! > Can an EWA-bounded candidate frontier plus deterministic multi-route
//! > disagreement choose a useful withheld observation, feed it through the
//! > real revision contract as a new independent root, and then show that the
//! > selected epistemic hole collapses on replay?
//!
//! Existing pieces only:
//!
//! - EWA transport: `lance_graph_contract::sigma_propagation::ewa_sandwich`
//! - disagreement normalization: `jc::quorum::max_u8_variance`
//! - write-back court of appeal: `lance_graph_contract::revision::GadamerRevision`
//!
//! No diffusion model, LPIPS, new 0..63 ordinal, new fold primitive, or
//! production write path is introduced here.
//!
//! Run:
//!
//! ```text
//! cargo run --manifest-path crates/jc/Cargo.toml --example dav_active_observation_probe
//! ```

use jc::quorum::max_u8_variance;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionKind, RevisionPolicy,
};
use lance_graph_contract::sigma_propagation::{ewa_sandwich, Spd2};

const N: usize = 21;
const SEED: usize = 0;
const TARGET: usize = 10;

// This is intentionally a probe threshold, not a substrate constant.
// It makes a=0.55's isotropic EWA field just reach hop 10:
// 0.55^10 ~= 0.002533.
const MIN_EWA_WEIGHT: f64 = 0.0025;
const APERTURES: [f64; 5] = [0.45, 0.50, 0.55, 0.60, 0.65];

#[derive(Clone, Copy, Debug)]
struct Candidate {
    idx: usize,
    disagreement: f64,
    ewa_weight: f64,
    score: f64,
}

fn truth(idx: usize) -> bool {
    // One planted regime change at TARGET, then a second one farther right.
    // TARGET is the useful withheld fact the probe should discover.
    (TARGET..16).contains(&idx)
}

fn initial_observations() -> Vec<Option<bool>> {
    let mut observed: Vec<Option<bool>> = (0..N).map(|idx| Some(truth(idx))).collect();

    // Four withheld facts. Only TARGET sits on a regime boundary and therefore
    // makes the deterministic completion routes disagree.
    for idx in [6usize, TARGET, 13, 18] {
        observed[idx] = None;
    }
    observed
}

fn nearest_left(observed: &[Option<bool>], idx: usize) -> bool {
    (0..idx)
        .rev()
        .find_map(|j| observed[j])
        .unwrap_or(false)
}

fn nearest_right(observed: &[Option<bool>], idx: usize) -> bool {
    ((idx + 1)..observed.len())
        .find_map(|j| observed[j])
        .unwrap_or(false)
}

fn local_majority(observed: &[Option<bool>], idx: usize, radius: usize) -> bool {
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius + 1).min(observed.len());
    let mut yes = 0usize;
    let mut no = 0usize;

    for value in observed[lo..hi].iter().flatten() {
        if *value {
            yes += 1;
        } else {
            no += 1;
        }
    }

    // Stable tie-break: false. The physical storage / iteration order cannot
    // decide a semantic tie.
    yes > no
}

fn route_votes(observed: &[Option<bool>], idx: usize) -> [u8; 3] {
    if let Some(v) = observed[idx] {
        let x = if v { 255 } else { 0 };
        return [x, x, x];
    }

    [
        if nearest_left(observed, idx) { 255 } else { 0 },
        if nearest_right(observed, idx) { 255 } else { 0 },
        if local_majority(observed, idx, 3) { 255 } else { 0 },
    ]
}

/// Complement of the quorum agreement normalization:
///
/// disagreement = sqrt(var(votes) / max_attainable_var(k))
///
/// 0 = all routes coincide, 1 = maximally split for this k.
fn route_disagreement(observed: &[Option<bool>], idx: usize) -> f64 {
    let votes = route_votes(observed, idx);
    let mean = votes.iter().map(|&v| f64::from(v)).sum::<f64>() / votes.len() as f64;
    let var = votes
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / votes.len() as f64;

    let max_var = max_u8_variance(votes.len());
    if max_var == 0.0 {
        0.0
    } else {
        (var / max_var).sqrt().clamp(0.0, 1.0)
    }
}

/// Isotropic special case of the real EWA sandwich.
///
/// M = sqrt(a) I, Sigma_0 = I
/// Sigma_h = M Sigma_(h-1) M^T = a^h I
///
/// Reading the mean diagonal therefore gives the aperture field weight at
/// exactly `hops` hops while still executing the certified ABI kernel.
fn ewa_aperture_weight(aperture: f64, hops: usize) -> f64 {
    assert!((0.0..=1.0).contains(&aperture));
    let root = aperture.sqrt();
    let m = Spd2 {
        a: root,
        b: 0.0,
        c: root,
    };
    let mut sigma = Spd2::I;
    for _ in 0..hops {
        sigma = ewa_sandwich(&m, &sigma);
    }
    assert!(sigma.is_spd(1e-15));
    0.5 * (sigma.a + sigma.c)
}

fn rank_candidates(observed: &[Option<bool>], aperture: f64) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = observed
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if value.is_some() {
                return None;
            }

            let hops = idx.abs_diff(SEED);
            let ewa_weight = ewa_aperture_weight(aperture, hops);
            if ewa_weight < MIN_EWA_WEIGHT {
                return None;
            }

            let disagreement = route_disagreement(observed, idx);
            Some(Candidate {
                idx,
                disagreement,
                ewa_weight,
                score: disagreement * ewa_weight,
            })
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.idx.cmp(&b.idx))
    });
    candidates
}

fn prior_horizon() -> InterpretiveHorizon<u64, u64> {
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: 0,
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: 0,
        independent_roots: 0,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn revise_with_observation(idx: usize, value: bool) {
    let bit = 1u64 << idx;
    let prior = prior_horizon();
    let proposed_claims = if value { bit } else { 0 };

    let encounter = EncounterEvidence {
        proposed_claims,
        independent_roots: bit,
        inherited_roots: 0,
        resistance: bit,
        contradictions: 0,
        affected_parts: bit,
    };
    let ancestry = BasisView {
        ancestry_independent_roots: 0,
        ancestry_derived_roots: 0,
        ancestor_claims: 0,
        closes_cycle: false,
    };

    let delta = GadamerRevision.revise(&prior, &encounter, &ancestry);

    assert_eq!(
        delta.kind,
        RevisionKind::HorizonExpansion,
        "a newly observed true boundary fact should expand the horizon"
    );
    assert_eq!(
        delta.evidential_effect,
        EvidentialEffect::IncreaseEligible,
        "the observation is a genuinely new independent root"
    );
    assert_eq!(delta.new_independent_roots, bit);
    assert_eq!(delta.resulting.independent_roots, bit);
    assert_eq!(delta.resulting.projected_claims, bit);
}

fn main() {
    let mut observed = initial_observations();

    println!("DAV x EWA x Revision active-observation probe");
    println!("seed={SEED}, planted useful withheld target={TARGET}");
    println!("frontier floor={MIN_EWA_WEIGHT:.6}");
    println!();

    let mut best_efficiency = (0.0f64, 0.0f64);

    for aperture in APERTURES {
        let ranked = rank_candidates(&observed, aperture);
        let touched = ranked.len();
        let top = ranked.first().copied();
        let hit = top.is_some_and(|c| c.idx == TARGET);
        let efficiency = if touched == 0 {
            0.0
        } else {
            (if hit { 1.0 } else { 0.0 }) / touched as f64
        };

        if efficiency > best_efficiency.1 {
            best_efficiency = (aperture, efficiency);
        }

        println!(
            "a={aperture:.2}  hop10={:.6}  touched={touched}  top={:?}  hit={}  efficiency={efficiency:.3}",
            ewa_aperture_weight(aperture, 10),
            top.map(|c| (c.idx, c.disagreement, c.ewa_weight, c.score)),
            hit
        );
    }

    // Synthetic falsifier shape:
    // .45 cannot reach hop 10 at this frontier floor.
    assert!(
        rank_candidates(&observed, 0.45)
            .first()
            .map_or(true, |c| c.idx != TARGET),
        "narrow aperture unexpectedly reached the planted hop-10 target"
    );

    // .55 is the first sweep point whose EWA field reaches hop 10 and the
    // disagreement rank should put that useful fact first.
    let ranked_055 = rank_candidates(&observed, 0.55);
    let selected = ranked_055
        .first()
        .copied()
        .expect("a=0.55 should expose at least one candidate");
    assert_eq!(
        selected.idx, TARGET,
        "DAV ranking failed to choose the planted useful observation"
    );
    assert!(
        selected.disagreement > 0.0,
        "selected target must be epistemically unresolved before observation"
    );

    // Deterministic-random baseline = lowest candidate ordinal. At this exact
    // frontier it touches node 6 first, so k=1 misses the useful target.
    let mut ordinal_baseline: Vec<usize> = ranked_055.iter().map(|c| c.idx).collect();
    ordinal_baseline.sort_unstable();
    assert_eq!(ordinal_baseline.first().copied(), Some(6));
    assert_ne!(ordinal_baseline[0], TARGET);

    // Observation is the only place where hidden truth enters.
    let revealed = truth(selected.idx);
    assert!(revealed, "the planted boundary target is a true fact");

    // Existing revision.rs is the court of appeal. A physical observation is
    // presented as a new independent root; DAV itself never mints evidence.
    revise_with_observation(selected.idx, revealed);

    // Replay with the observation present. All three deterministic routes now
    // read the same observed value at the selected site, so the epistemic hole
    // must collapse.
    observed[selected.idx] = Some(revealed);
    let after = route_disagreement(&observed, selected.idx);
    assert_eq!(
        after, 0.0,
        "observation + revision did not collapse the selected disagreement"
    );

    println!();
    println!(
        "selected hop-{} target with a=.55: disagreement {:.3} -> {:.3} after observation",
        TARGET.abs_diff(SEED),
        selected.disagreement,
        after
    );
    println!(
        "best synthetic recovered-fact/touched-candidate efficiency in sweep: a={:.2} ({:.3})",
        best_efficiency.0, best_efficiency.1
    );
    println!("PASS: EWA frontier -> DAV disagreement -> observation -> revision -> replay collapse");
}
