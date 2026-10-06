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
//! - disagreement: `jc::quorum::pairwise_agreement_u8`, the quorum dispersion
//!   `1 − σ/σ_max(k)`, read as its complement
//! - write-back court of appeal: `lance_graph_contract::revision::GadamerRevision`
//!
//! # The belief state is the revision horizon
//!
//! What the probe believes is an `InterpretiveHorizon<u64, u64>`: bit `i` of
//! `independent_roots` says site `i` has been observed, and bit `i` of
//! `projected_claims` says what was observed there. The routes read that
//! horizon and nothing else. A revealed observation changes the belief state
//! only by being presented to `GadamerRevision` and adopted when the revision
//! returns `EvidentialEffect::IncreaseEligible`; replay then reads
//! `delta.resulting`. Remove the revision call and the hole cannot close.
//!
//! The counterfactual docket (`RevisionVerdict::is_acceptable`) is NOT walked:
//! this updates the probe's working horizon, not actual-world state.
//!
//! No diffusion model, LPIPS, new 0..63 ordinal, new fold primitive, or
//! production write path is introduced here.
//!
//! Run:
//!
//! ```text
//! cargo run --manifest-path crates/jc/Cargo.toml --example dav_active_observation_probe
//! ```

use jc::quorum::pairwise_agreement_u8;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionDelta, RevisionPolicy,
};
use lance_graph_contract::sigma_propagation::{ewa_sandwich, Spd2};

type Horizon = InterpretiveHorizon<u64, u64>;

const N: usize = 21;
const SEED: usize = 0;
const TARGET: usize = 10;
const WITHHELD: [usize; 4] = [6, TARGET, 13, 18];

// A probe threshold, not a substrate constant. It places the reach boundary
// for hop 10 between apertures 0.50 and 0.55 (0.50^10 ~= 0.000977,
// 0.55^10 ~= 0.002533). The sweep therefore demonstrates the reach rule
// `a^h >= floor`; it is not evidence that 0.55 is special.
const MIN_EWA_WEIGHT: f64 = 0.0025;
const APERTURES: [f64; 5] = [0.45, 0.50, 0.55, 0.60, 0.65];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    idx: usize,
    /// `0` = the routes coincide, `255` = maximally split.
    disagreement: u8,
    ewa_weight: f64,
    score: f64,
}

/// The hidden world. Read only by `initial_horizon` (for the prior evidence)
/// and by the reveal step; ranking never sees it.
fn truth(idx: usize) -> bool {
    // One planted regime change at TARGET, then a second one farther right.
    (TARGET..16).contains(&idx)
}

fn bit(idx: usize) -> u64 {
    1u64 << idx
}

/// Prior evidence: every site except `WITHHELD` was observed independently.
fn initial_horizon() -> Horizon {
    let mut roots = 0u64;
    let mut claims = 0u64;
    for idx in (0..N).filter(|i| !WITHHELD.contains(i)) {
        roots |= bit(idx);
        if truth(idx) {
            claims |= bit(idx);
        }
    }
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: 0,
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: claims,
        independent_roots: roots,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

/// The read the routes are allowed: what the horizon says was observed.
fn observations(h: &Horizon) -> [Option<bool>; N] {
    core::array::from_fn(|idx| {
        (h.independent_roots & bit(idx) != 0).then_some(h.projected_claims & bit(idx) != 0)
    })
}

fn nearest_left(observed: &[Option<bool>], idx: usize) -> bool {
    (0..idx).rev().find_map(|j| observed[j]).unwrap_or(false)
}

fn nearest_right(observed: &[Option<bool>], idx: usize) -> bool {
    ((idx + 1)..observed.len())
        .find_map(|j| observed[j])
        .unwrap_or(false)
}

fn local_majority(observed: &[Option<bool>], idx: usize, radius: usize) -> bool {
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius + 1).min(observed.len());
    let (mut yes, mut no) = (0usize, 0usize);
    for value in observed[lo..hi].iter().flatten() {
        if *value {
            yes += 1;
        } else {
            no += 1;
        }
    }
    // Stable tie-break: false. Iteration order cannot decide a semantic tie.
    yes > no
}

fn vote(v: bool) -> u8 {
    if v {
        255
    } else {
        0
    }
}

fn route_votes(observed: &[Option<bool>], idx: usize) -> [u8; 3] {
    if let Some(v) = observed[idx] {
        return [vote(v); 3];
    }
    [
        vote(nearest_left(observed, idx)),
        vote(nearest_right(observed, idx)),
        vote(local_majority(observed, idx, 3)),
    ]
}

/// Route disagreement at one site, through jc's quorum dispersion.
///
/// `pairwise_agreement_u8` scores the pairs of `k` square tables. Each route
/// contributes a 2×2 table whose single off-diagonal pair holds its vote, so
/// cell `[0][1]` is exactly the quorum score of the three votes.
fn route_disagreement(observed: &[Option<bool>], idx: usize) -> u8 {
    let tables = route_votes(observed, idx).map(|v| [255u8, v, v, 255]);
    let refs: [&[u8]; 3] = [&tables[0], &tables[1], &tables[2]];
    let agreement = pairwise_agreement_u8(&refs, 2).expect("three 2x2 tables");
    255 - agreement[1]
}

/// Isotropic special case of the real EWA sandwich.
///
/// M = sqrt(a) I, Sigma_0 = I, so Sigma_h = a^h I. The mean diagonal is the
/// aperture field weight at `hops` hops, computed by the contract kernel.
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

/// Rank the unobserved sites inside the EWA frontier, enumerating them in
/// `order`. The result must not depend on `order`.
fn rank_in(
    observed: &[Option<bool>],
    aperture: f64,
    order: impl Iterator<Item = usize>,
) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = order
        .filter(|&idx| observed[idx].is_none())
        .filter_map(|idx| {
            let ewa_weight = ewa_aperture_weight(aperture, idx.abs_diff(SEED));
            (ewa_weight >= MIN_EWA_WEIGHT).then(|| {
                let disagreement = route_disagreement(observed, idx);
                Candidate {
                    idx,
                    disagreement,
                    ewa_weight,
                    score: f64::from(disagreement) / 255.0 * ewa_weight,
                }
            })
        })
        .collect();
    // Explicit, total order: higher score first, then the semantic site
    // ordinal. Enumeration order never decides.
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.idx.cmp(&b.idx)));
    candidates
}

fn rank(observed: &[Option<bool>], aperture: f64) -> Vec<Candidate> {
    rank_in(observed, aperture, 0..N)
}

/// Present an encounter to `GadamerRevision`, with the prior horizon as the
/// ancestry, and adopt the result only if it is eligible to increase support.
fn revise(
    prior: &Horizon,
    encounter: &EncounterEvidence<u64>,
) -> (Horizon, RevisionDelta<u64, u64>) {
    let ancestry = BasisView {
        ancestry_independent_roots: prior.independent_roots,
        ancestry_derived_roots: prior.inherited_roots,
        ancestor_claims: prior.projected_claims,
        closes_cycle: false,
    };
    let delta = GadamerRevision.revise(prior, encounter, &ancestry);
    let next = if delta.evidential_effect == EvidentialEffect::IncreaseEligible {
        delta.resulting.clone()
    } else {
        prior.clone()
    };
    (next, delta)
}

fn claims_with(prior: &Horizon, idx: usize, value: bool) -> u64 {
    if value {
        prior.projected_claims | bit(idx)
    } else {
        prior.projected_claims & !bit(idx)
    }
}

/// The reveal step: the only place hidden truth enters after the prior. The
/// physical observation is a new independent root.
fn observe(prior: &Horizon, idx: usize) -> (Horizon, RevisionDelta<u64, u64>) {
    let value = truth(idx);
    revise(
        prior,
        &EncounterEvidence {
            proposed_claims: claims_with(prior, idx, value),
            independent_roots: bit(idx),
            inherited_roots: 0,
            resistance: 0,
            contradictions: 0,
            affected_parts: bit(idx),
        },
    )
}

/// What DAV must NOT be able to do: turn its own route consensus into
/// evidence. The consensus is a derived interpretation, so it arrives as an
/// inherited root, never an independent one.
fn adopt_route_consensus(prior: &Horizon, idx: usize) -> (Horizon, RevisionDelta<u64, u64>) {
    let votes = route_votes(&observations(prior), idx);
    let value = votes.iter().filter(|&&v| v == 255).count() * 2 > votes.len();
    revise(
        prior,
        &EncounterEvidence {
            proposed_claims: claims_with(prior, idx, value),
            independent_roots: 0,
            inherited_roots: bit(idx),
            resistance: 0,
            contradictions: 0,
            affected_parts: bit(idx),
        },
    )
}

/// Total route disagreement over every still-withheld site.
fn residual_disagreement(h: &Horizon) -> u32 {
    let observed = observations(h);
    (0..N)
        .filter(|&i| observed[i].is_none())
        .map(|i| u32::from(route_disagreement(&observed, i)))
        .sum()
}

#[derive(Debug, PartialEq)]
struct Cycle {
    selected: usize,
    before: u8,
    after: u8,
    residual_after: u32,
    resulting: Horizon,
}

/// One closed loop: observe `selected`, revise, replay from the horizon.
fn run_cycle(prior: &Horizon, selected: usize) -> Cycle {
    let before = route_disagreement(&observations(prior), selected);
    let (next, delta) = observe(prior, selected);
    assert_eq!(
        delta.evidential_effect,
        EvidentialEffect::IncreaseEligible,
        "a revealed observation must enter as a new independent root"
    );
    assert_eq!(delta.new_independent_roots, bit(selected));
    let after = route_disagreement(&observations(&next), selected);
    Cycle {
        selected,
        before,
        after,
        residual_after: residual_disagreement(&next),
        resulting: next,
    }
}

fn main() {
    let prior = initial_horizon();
    let observed = observations(&prior);

    println!("DAV x EWA x Revision active-observation probe");
    println!("seed={SEED}, planted useful withheld target={TARGET}, withheld={WITHHELD:?}");
    println!("frontier floor={MIN_EWA_WEIGHT:.6} (probe threshold, not a constant)");
    println!();

    // The frontier follows the reach rule, at every aperture. `powi` is an
    // independent computation of a^h, not the kernel under test.
    for aperture in APERTURES {
        let ranked = rank(&observed, aperture);
        let reaches_target = aperture.powi(TARGET as i32) >= MIN_EWA_WEIGHT;
        let top = ranked.first().copied();
        println!(
            "a={aperture:.2}  hop10={:.6}  frontier={:?}  top={:?}",
            ewa_aperture_weight(aperture, TARGET),
            ranked.iter().map(|c| c.idx).collect::<Vec<_>>(),
            top.map(|c| (c.idx, c.disagreement))
        );
        assert_eq!(
            ranked.iter().any(|c| c.idx == TARGET),
            reaches_target,
            "a={aperture}: the EWA frontier disagrees with a^h >= floor"
        );
        assert_eq!(
            top.is_some_and(|c| c.idx == TARGET),
            reaches_target,
            "a={aperture}: a reachable disagreeing target must rank first"
        );
        // Ranking must not depend on enumeration order. At a=0.65 sites 6
        // and 13 tie at score 0, so the tie-break is what this checks.
        assert_eq!(
            rank_in(&observed, aperture, (0..N).rev()),
            ranked,
            "a={aperture}: candidate enumeration order changed the ranking"
        );
    }
    let tied = rank(&observed, 0.65);
    assert!(
        tied.windows(2).any(|w| w[0].score == w[1].score),
        "anti-vacuity: the sweep must contain a score tie for the order check"
    );

    let aperture = 0.55;
    let ranked = rank(&observed, aperture);

    let dav = ranked[0].idx;
    let ordinal = ranked
        .iter()
        .map(|c| c.idx)
        .min()
        .expect("non-empty frontier");
    println!();
    println!("a={aperture:.2}: DAV k=1 picks {dav}, ordinal baseline k=1 picks {ordinal}");

    // Before observation the selected site is an open hole.
    assert!(
        ranked[0].disagreement > 0,
        "selected site must be unresolved"
    );

    // DAV cannot close the hole with its own consensus: revision refuses an
    // inherited root, the horizon is unchanged, and the hole stays open.
    let (laundered, delta) = adopt_route_consensus(&prior, dav);
    assert_ne!(delta.evidential_effect, EvidentialEffect::IncreaseEligible);
    assert_eq!(laundered, prior, "route consensus changed the belief state");
    assert_eq!(
        route_disagreement(&observations(&laundered), dav),
        ranked[0].disagreement
    );

    // The closed loop, for DAV and for the baseline.
    let dav_cycle = run_cycle(&prior, dav);
    let ordinal_cycle = run_cycle(&prior, ordinal);
    for (name, c) in [("DAV", &dav_cycle), ("ordinal", &ordinal_cycle)] {
        println!(
            "{name:>7}: site {} disagreement {} -> {}, residual over withheld sites {} -> {}",
            c.selected,
            c.before,
            c.after,
            residual_disagreement(&prior),
            c.residual_after
        );
    }
    assert_eq!(
        dav_cycle.after, 0,
        "replay did not collapse the selected hole"
    );
    assert_eq!(dav_cycle.residual_after, 0);
    assert!(
        ordinal_cycle.residual_after > dav_cycle.residual_after,
        "the baseline closed as much as DAV, so DAV bought nothing here"
    );

    // Replay is deterministic.
    assert_eq!(run_cycle(&prior, dav), dav_cycle);

    println!();
    println!("PASS: EWA frontier -> disagreement -> observation -> revision -> replay collapse");
}
