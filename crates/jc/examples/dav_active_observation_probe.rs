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
//! # Carrier invariance (Round 1)
//!
//! `carrier_differential` runs the same world through two carriers: the
//! reference `[Option<bool>; N]` and the Quack / mask-risc shape, a value lane
//! plus a validity plane. The validity plane is the horizon's own
//! `independent_roots` word, borrowed. Candidate selection is one Quack
//! filter, `is_null(VALID) AND Range(SITE)`, run by the mask-risc evaluator.
//! The routes, scoring and revision are shared code. Every output must match
//! for four NULL payloads (including the hidden truth) and both enumeration
//! orders. The frontier range is bound to the site-ordinal axis, never to the
//! nullable value column: `sql_where` would gate it by validity and drop every
//! candidate. The probe asserts both halves of that.
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
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};

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

/// What the routes may read: whether a site was observed, and if so what.
///
/// Two carriers implement it — the reference `[Option<bool>; N]` and the
/// Quack value lane + validity plane — and the routes are written once
/// against this trait, so a carrier change cannot change the route code.
trait Observed {
    fn get(&self, idx: usize) -> Option<bool>;
    fn len(&self) -> usize;
}

impl Observed for [Option<bool>; N] {
    fn get(&self, idx: usize) -> Option<bool> {
        self[idx]
    }
    fn len(&self) -> usize {
        N
    }
}

fn nearest_left(observed: &dyn Observed, idx: usize) -> bool {
    (0..idx)
        .rev()
        .find_map(|j| observed.get(j))
        .unwrap_or(false)
}

fn nearest_right(observed: &dyn Observed, idx: usize) -> bool {
    ((idx + 1)..observed.len())
        .find_map(|j| observed.get(j))
        .unwrap_or(false)
}

fn local_majority(observed: &dyn Observed, idx: usize, radius: usize) -> bool {
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius + 1).min(observed.len());
    let (mut yes, mut no) = (0usize, 0usize);
    for value in (lo..hi).filter_map(|j| observed.get(j)) {
        if value {
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

fn route_votes(observed: &dyn Observed, idx: usize) -> [u8; 3] {
    if let Some(v) = observed.get(idx) {
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
fn route_disagreement(observed: &dyn Observed, idx: usize) -> u8 {
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
    observed: &[Option<bool>; N],
    aperture: f64,
    order: impl Iterator<Item = usize>,
) -> Vec<Candidate> {
    let candidates = order
        .filter(|&idx| observed[idx].is_none())
        .filter(|&idx| ewa_aperture_weight(aperture, idx.abs_diff(SEED)) >= MIN_EWA_WEIGHT);
    score_and_sort(observed, aperture, candidates)
}

/// Score a candidate set and put it in the explicit total order. Shared by
/// both carriers: only how the candidate SET is produced differs.
fn score_and_sort(
    observed: &dyn Observed,
    aperture: f64,
    candidates: impl Iterator<Item = usize>,
) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = candidates
        .map(|idx| {
            let ewa_weight = ewa_aperture_weight(aperture, idx.abs_diff(SEED));
            let disagreement = route_disagreement(observed, idx);
            Candidate {
                idx,
                disagreement,
                ewa_weight,
                score: f64::from(disagreement) / 255.0 * ewa_weight,
            }
        })
        .collect();
    // Explicit, total order: higher score first, then the semantic site
    // ordinal. Enumeration order never decides.
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.idx.cmp(&b.idx)));
    candidates
}

fn rank(observed: &[Option<bool>; N], aperture: f64) -> Vec<Candidate> {
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
fn adopt_route_consensus(
    carrier: Carrier,
    prior: &Horizon,
    idx: usize,
) -> (Horizon, RevisionDelta<u64, u64>) {
    let votes = with_reading(carrier, prior, |o| route_votes(o, idx));
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
fn residual_disagreement(carrier: Carrier, h: &Horizon) -> u32 {
    with_reading(carrier, h, |o| {
        (0..N)
            .filter(|&i| o.get(i).is_none())
            .map(|i| u32::from(route_disagreement(o, i)))
            .sum()
    })
}

#[derive(Debug, PartialEq)]
struct Cycle {
    selected: usize,
    before: u8,
    after: u8,
    residual_after: u32,
    delta: RevisionDelta<u64, u64>,
    resulting: Horizon,
}

/// One closed loop: observe `selected`, revise, replay from the horizon.
///
/// The carrier is only ever a READING of a horizon: it is rebuilt from
/// `prior` and from `delta.resulting`, never written. The only write is the
/// revision call inside `observe`.
fn run_cycle(carrier: Carrier, prior: &Horizon, selected: usize) -> Cycle {
    let before = with_reading(carrier, prior, |o| route_disagreement(o, selected));
    let (next, delta) = observe(prior, selected);
    assert_eq!(
        delta.evidential_effect,
        EvidentialEffect::IncreaseEligible,
        "a revealed observation must enter as a new independent root"
    );
    assert_eq!(delta.new_independent_roots, bit(selected));
    let after = with_reading(carrier, &next, |o| route_disagreement(o, selected));
    Cycle {
        selected,
        before,
        after,
        residual_after: residual_disagreement(carrier, &next),
        delta,
        resulting: next,
    }
}

// ---------------------------------------------------------------------------
// Carrier B: the Quack / mask-risc representation — value lane + validity
// plane. The cognitive semantics above are unchanged; only the carrier is.
// ---------------------------------------------------------------------------

/// `Planes::masks[0]`: the validity plane. A set bit = the site was observed.
const VALID: Mask = Mask(0);
/// `Planes::lanes[0]`: the value lane. Read only where `VALID` is set.
const VALUE: Col = Col(0);
/// The SITE-ORDINAL axis: provenance for `Cmp::Range`, which reads the row
/// ordinal and no lane (`Pred::Range` carries no lane index). It is
/// deliberately NOT the value column: the address of an unobserved site is
/// known even though its value is not, and naming `VALUE` here would let
/// `sql_where` gate the frontier by `VALID` and remove every candidate.
const SITE: Col = Col(1);

/// What a NULL row's value-lane payload holds. A correct carrier is
/// indifferent to it; varying it is the leak test.
#[derive(Clone, Copy, Debug)]
enum Poison {
    Zero,
    /// The hidden truth itself sits under the NULL bit.
    Truth,
    NotTruth,
    Garbage,
}

const POISONS: [Poison; 4] = [
    Poison::Zero,
    Poison::Truth,
    Poison::NotTruth,
    Poison::Garbage,
];

#[derive(Clone, Copy, Debug)]
enum Carrier {
    /// `[Option<bool>; N]`, the #1344 carrier.
    Reference,
    /// value lane + validity plane.
    Quack(Poison),
    /// A deliberately wrong Quack reader that treats a NULL row with a
    /// non-zero payload as observed-true. Exists only to prove the poison
    /// test can fire.
    LeakyQuack(Poison),
}

/// A reading of the horizon as value lane + validity plane.
///
/// `valid` BORROWS the horizon's own `independent_roots` word — the revision
/// state's root mask is already a packed `u64` validity plane, so no copy is
/// made. `value` is the one materialised lane: `N` `u32`s built from
/// `projected_claims`, with `poison` written under every NULL bit.
struct QuackCarrier<'h> {
    valid: &'h [u64],
    value: [u32; N],
    leaky: bool,
}

impl<'h> QuackCarrier<'h> {
    fn read(h: &'h Horizon, poison: Poison, leaky: bool) -> Self {
        let value = core::array::from_fn(|idx| {
            if h.independent_roots & bit(idx) != 0 {
                u32::from(h.projected_claims & bit(idx) != 0)
            } else {
                match poison {
                    Poison::Zero => 0,
                    Poison::Truth => u32::from(truth(idx)),
                    Poison::NotTruth => u32::from(!truth(idx)),
                    Poison::Garbage => 0xA5A5_A5A5,
                }
            }
        });
        Self {
            valid: core::slice::from_ref(&h.independent_roots),
            value,
            leaky,
        }
    }

    fn is_valid(&self, idx: usize) -> bool {
        self.valid[idx / 64] >> (idx % 64) & 1 == 1
    }

    /// Run one Quack filter over this carrier and return the kept mask.
    fn select(&self, filter: &Filter) -> Vec<u64> {
        let program = lower(&Query {
            filter: filter.clone(),
            agg: Agg::Rows,
        })
        .expect("lowers");
        let masks: [&[u64]; 1] = [self.valid];
        let lanes = [LaneRef::U32(&self.value)];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let mut scratch = Scratch::for_program(&program, N).expect("carves");
        let mut kept = vec![0u64; words_for(N)];
        match execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(&mut kept),
        )
        .expect("runs")
        {
            Value::Mask(_) => kept,
            other => panic!("not a kept mask: {other:?}"),
        }
    }
}

impl Observed for QuackCarrier<'_> {
    fn get(&self, idx: usize) -> Option<bool> {
        if self.is_valid(idx) {
            Some(self.value[idx] != 0)
        } else if self.leaky && self.value[idx] == 1 {
            Some(true)
        } else {
            None
        }
    }
    fn len(&self) -> usize {
        N
    }
}

fn with_reading<R>(carrier: Carrier, h: &Horizon, f: impl FnOnce(&dyn Observed) -> R) -> R {
    match carrier {
        Carrier::Reference => f(&observations(h)),
        Carrier::Quack(p) => f(&QuackCarrier::read(h, p, false)),
        Carrier::LeakyQuack(p) => f(&QuackCarrier::read(h, p, true)),
    }
}

/// The EWA frontier as a contiguous ordinal interval `[lo, hi)`.
///
/// In the isotropic case the kernel weight `a^h` is non-increasing in the hop
/// count for `a <= 1`, so `weight >= floor` holds exactly on
/// `|idx - SEED| <= h_max`. `h_max` is found by walking the KERNEL, not by
/// `powi`; monotonicity is asserted, not assumed.
fn frontier_range(aperture: f64) -> (u32, u32) {
    let mut h_max = 0usize;
    let mut prev = f64::INFINITY;
    for hops in 0..N {
        let w = ewa_aperture_weight(aperture, hops);
        assert!(
            w <= prev,
            "EWA weight rose with hop count; frontier not contiguous"
        );
        prev = w;
        if w >= MIN_EWA_WEIGHT {
            h_max = hops;
        }
    }
    let lo = SEED.saturating_sub(h_max);
    let hi = (SEED + h_max + 1).min(N);
    (lo as u32, hi as u32)
}

/// Candidate selection as ONE Quack filter: `is_null(value) AND frontier`.
fn candidate_filter(aperture: f64) -> Filter {
    let (lo, hi) = frontier_range(aperture);
    Filter::and([
        Filter::is_null(VALID),
        Filter::cmp(SITE, Cmp::Range { lo, hi }),
    ])
}

/// Set bits of a kept mask, ascending or descending — two physical
/// enumeration orders of the same candidate set.
fn set_bits(mask: &[u64], descending: bool) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..N)
        .filter(|&i| mask[i / 64] >> (i % 64) & 1 == 1)
        .collect();
    if descending {
        rows.reverse();
    }
    rows
}

fn rank_quack(h: &Horizon, aperture: f64, poison: Poison, descending: bool) -> Vec<Candidate> {
    let carrier = QuackCarrier::read(h, poison, false);
    let kept = carrier.select(&candidate_filter(aperture));
    score_and_sort(&carrier, aperture, set_bits(&kept, descending).into_iter())
}

/// The Round-1 differential: the same deterministic world through the
/// reference carrier (A) and the Quack carrier (B), every semantic output
/// compared.
fn carrier_differential(prior: &Horizon) {
    println!();
    println!("carrier differential: A = [Option<bool>; N], B = value lane + validity plane");
    let reference = observations(prior);

    // F5, can-fire: attaching the frontier to the NULLABLE value column lets
    // `sql_where` gate the range by validity, which removes exactly the NULL
    // rows that are the candidates.
    let trap = Filter::and([
        Filter::is_null(VALID),
        Filter::cmp(
            VALUE,
            Cmp::Range {
                lo: 0,
                hi: N as u32,
            },
        ),
    ])
    .sql_where(&[(VALUE, VALID)]);
    let quack = QuackCarrier::read(prior, Poison::Zero, false);
    assert_eq!(
        set_bits(&quack.select(&trap), false),
        Vec::<usize>::new(),
        "F5 can-fire: a range on the nullable value lane must lose every NULL candidate"
    );

    for aperture in APERTURES {
        let a = rank(&reference, aperture);
        assert!(
            !a.is_empty(),
            "anti-vacuity: empty frontier at a={aperture}"
        );

        // F5, silence: on the SITE axis the frontier is not nullable, so
        // `sql_where` leaves the filter — and the candidates — unchanged.
        let filter = candidate_filter(aperture);
        let nullable = filter.sql_where(&[(VALUE, VALID)]);
        assert_eq!(
            nullable, filter,
            "sql_where touched a non-nullable frontier"
        );

        for poison in POISONS {
            for descending in [false, true] {
                let b = rank_quack(prior, aperture, poison, descending);
                // F1 / F2 / F3 / F6: identity, disagreement, EWA weight and
                // score (bit-for-bit) and order all equal the reference.
                assert_eq!(
                    b, a,
                    "a={aperture} poison={poison:?} desc={descending}: carrier changed ranking"
                );
            }
        }
        println!(
            "a={aperture:.2}  frontier={:?}  candidates={:?}  A==B for {} poisons x 2 orders",
            frontier_range(aperture),
            a.iter().map(|c| c.idx).collect::<Vec<_>>(),
            POISONS.len()
        );
    }

    // F2/F3 can-fire: a reader that lets a NULL payload through DOES change
    // the reasoning once the hidden truth is the payload — so the equality
    // above is a measurement, not a tautology.
    let aperture = 0.55;
    let leaked = |p| {
        with_reading(Carrier::LeakyQuack(p), prior, |o| {
            (0..N).map(|i| route_disagreement(o, i)).collect::<Vec<_>>()
        })
    };
    assert_ne!(
        leaked(Poison::Truth),
        leaked(Poison::NotTruth),
        "F2/F3 can-fire: the leaky reader did not leak, so the poison test is vacuous"
    );

    // The closed loop, both carriers, every poison.
    let dav = rank(&reference, aperture)[0].idx;
    let ordinal = rank(&reference, aperture)
        .iter()
        .map(|c| c.idx)
        .min()
        .expect("non-empty frontier");
    for selected in [dav, ordinal] {
        let a = run_cycle(Carrier::Reference, prior, selected);
        for poison in POISONS {
            let b = run_cycle(Carrier::Quack(poison), prior, selected);
            // pre-observation disagreement, verdict, delta, resulting
            // horizon, post-replay disagreement, residual.
            assert_eq!(b, a, "site {selected} poison={poison:?}: cycle differs");
            // final observed/value state, read through each carrier.
            let final_a = observations(&a.resulting);
            let final_b = QuackCarrier::read(&b.resulting, poison, false);
            for (i, &want) in final_a.iter().enumerate() {
                assert_eq!(final_b.get(i), want, "final state differs at {i}");
            }
        }
        println!(
            "site {selected}: before {} -> after {}, residual {}, verdict {:?}/{:?}: A==B",
            a.before, a.after, a.residual_after, a.delta.kind, a.delta.evidential_effect
        );
    }

    // F4: the Quack path cannot bypass revision. Its own consensus is
    // refused exactly as the reference's is, and the carrier — a reading of
    // the unchanged horizon — still shows the hole.
    for poison in POISONS {
        let (laundered, delta) = adopt_route_consensus(Carrier::Quack(poison), prior, dav);
        assert_ne!(delta.evidential_effect, EvidentialEffect::IncreaseEligible);
        assert_eq!(
            &laundered, prior,
            "Quack consensus changed the belief state"
        );
        assert_eq!(
            rank_quack(&laundered, aperture, poison, false),
            rank(&reference, aperture)
        );
    }
    println!("PASS: carrier invariance (A == B), NULL payload inert, revision not bypassed");
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
    let (laundered, delta) = adopt_route_consensus(Carrier::Reference, &prior, dav);
    assert_ne!(delta.evidential_effect, EvidentialEffect::IncreaseEligible);
    assert_eq!(laundered, prior, "route consensus changed the belief state");
    assert_eq!(
        route_disagreement(&observations(&laundered), dav),
        ranked[0].disagreement
    );

    // The closed loop, for DAV and for the baseline.
    let dav_cycle = run_cycle(Carrier::Reference, &prior, dav);
    let ordinal_cycle = run_cycle(Carrier::Reference, &prior, ordinal);
    for (name, c) in [("DAV", &dav_cycle), ("ordinal", &ordinal_cycle)] {
        println!(
            "{name:>7}: site {} disagreement {} -> {}, residual over withheld sites {} -> {}",
            c.selected,
            c.before,
            c.after,
            residual_disagreement(Carrier::Reference, &prior),
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
    assert_eq!(run_cycle(Carrier::Reference, &prior, dav), dav_cycle);

    println!();
    println!("PASS: EWA frontier -> disagreement -> observation -> revision -> replay collapse");

    carrier_differential(&prior);
}
