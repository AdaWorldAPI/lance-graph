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
//! # Resident reading (Round 2)
//!
//! `resident_differential` adds carrier C: the horizon's own
//! `independent_roots` and `projected_claims` words read as two borrowed
//! bit-planes, with no value lane built. It requires A == B == C on the same
//! outputs. C is a physical-carrier test only. A mask plane is never nullable
//! to Quack, so C has no SQL NULL semantics; B remains that test. For `k = 1`,
//! top-1 is a streaming fold over the kept mask with one `Option<Candidate>`
//! of state, no row-id list, no candidate list and no sort. The one derived
//! population object left on the C path is the kept mask, a caller-owned
//! stack bitmap. A heap meter pins candidate selection at 0 bytes.
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
    execute_into, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Heap meter (Round 2, falsifier 6). The pattern of
// `lance-graph-mask-risc/tests/no_alloc.rs`: count only allocations made
// while the current thread is inside `measure`.
// ---------------------------------------------------------------------------

struct Counting;

static HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.try_with(Cell::get).unwrap_or(false) {
            HEAP_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Heap bytes `f` allocated on this thread.
fn measure<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = HEAP_BYTES.load(Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    (r, HEAP_BYTES.load(Ordering::Relaxed) - before)
}

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
    /// Carrier C: the horizon's two resident words read as bit-planes,
    /// validity = `independent_roots`, value = `projected_claims`. The
    /// poison is written into `projected_claims` at UNOBSERVED sites —
    /// projected but not observed.
    Resident(Poison),
    /// C with the NULL gate broken: a projected bit at an unobserved site is
    /// read as an observed `true` (falsifier R2.5-1).
    LeakyResident(Poison),
    /// C with the validity gate removed entirely: every projected bit is read
    /// as an observation (falsifier R2.5-5).
    UngatedResident(Poison),
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
        Carrier::Resident(p) => with_resident(h, p, Gate::Valid, f),
        Carrier::LeakyResident(p) => with_resident(h, p, Gate::Leaky, f),
        Carrier::UngatedResident(p) => with_resident(h, p, Gate::None, f),
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

// ---------------------------------------------------------------------------
// Carrier C (Round 2): the horizon's own resident words, read as planes.
// No value lane is built. C has NO SQL NULL semantics: a plane is never
// nullable to Quack, so B stays the semantic NULL test and C is the
// physical-carrier test.
// ---------------------------------------------------------------------------

/// How a resident reading treats the validity plane.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gate {
    /// Correct: a value bit is read only where the validity bit is set.
    Valid,
    /// Broken: a set value bit at an unobserved site reads as observed-true.
    Leaky,
    /// Broken: validity ignored.
    None,
}

/// Carrier C: two borrowed `&[u64]` planes over the horizon's words.
struct ResidentCarrier<'h> {
    valid: &'h [u64],
    value: &'h [u64],
    gate: Gate,
}

impl<'h> ResidentCarrier<'h> {
    fn read(h: &'h Horizon, gate: Gate) -> Self {
        Self {
            valid: core::slice::from_ref(&h.independent_roots),
            value: core::slice::from_ref(&h.projected_claims),
            gate,
        }
    }
}

fn plane_bit(plane: &[u64], idx: usize) -> bool {
    plane[idx / 64] >> (idx % 64) & 1 == 1
}

impl Observed for ResidentCarrier<'_> {
    fn get(&self, idx: usize) -> Option<bool> {
        let (valid, value) = (plane_bit(self.valid, idx), plane_bit(self.value, idx));
        match self.gate {
            Gate::Valid => valid.then_some(value),
            Gate::Leaky => (valid || value).then_some(value),
            Gate::None => Some(value),
        }
    }
    fn len(&self) -> usize {
        N
    }
}

/// `h` with `poison` written into `projected_claims` at every UNOBSERVED
/// site, leaving `independent_roots` alone. This is test fixture
/// construction — a projection the horizon legally can carry — not part of
/// any measured path. `Poison::Zero` returns the horizon unchanged.
fn poisoned(h: &Horizon, poison: Poison) -> Horizon {
    let unobserved = !h.independent_roots & ((1u64 << N) - 1);
    let projected: u64 = (0..N)
        .filter(|&i| match poison {
            Poison::Zero => false,
            Poison::Truth => truth(i),
            Poison::NotTruth => !truth(i),
            Poison::Garbage => true,
        })
        .fold(0, |acc, i| acc | bit(i));
    let mut out = h.clone();
    out.projected_claims = (h.projected_claims & !unobserved) | (projected & unobserved);
    out
}

fn with_resident<R>(
    h: &Horizon,
    poison: Poison,
    gate: Gate,
    f: impl FnOnce(&dyn Observed) -> R,
) -> R {
    if let Poison::Zero = poison {
        // The unpoisoned case borrows the live horizon itself.
        f(&ResidentCarrier::read(h, gate))
    } else {
        let p = poisoned(h, poison);
        f(&ResidentCarrier::read(&p, gate))
    }
}

/// Words of the candidate mask — `words_for(N)`, here 1.
const W: usize = N.div_ceil(64);
/// Stack scratch for the candidate program (`Scratch::over_for_program`):
/// tile-local, never heap.
const SCRATCH_WORDS: usize = 8;

/// Run an already-lowered candidate program over the RESIDENT validity
/// plane, writing the kept mask into a caller-owned stack array. No lane is
/// borrowed: `Pred::Range` reads none, and `is_null` reads only the plane.
///
/// The `[u64; W]` it returns is the one derived population object on the C
/// path: a caller-owned bitmap, which per `stay-on-the-lane.md` §6 is still
/// materialization.
fn resident_candidates(valid: &[u64], program: &Program) -> [u64; W] {
    let masks: [&[u64]; 1] = [valid];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let mut buf = [0u64; SCRATCH_WORDS];
    let mut scratch = Scratch::over_for_program(&mut buf, program, N).expect("stack scratch");
    let mut kept = [0u64; W];
    match execute_into(
        program,
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

/// Visit the set bits of `mask` in ascending or descending ordinal, one word
/// at a time — a visitation, not a row-id list.
fn for_each_set_bit(mask: &[u64], descending: bool, mut f: impl FnMut(usize)) {
    let mut visit = |w: usize| {
        let mut word = mask[w];
        while word != 0 {
            let b = if descending {
                63 - word.leading_zeros() as usize
            } else {
                word.trailing_zeros() as usize
            };
            word &= !(1u64 << b);
            f(w * 64 + b);
        }
    };
    if descending {
        (0..mask.len()).rev().for_each(&mut visit);
    } else {
        (0..mask.len()).for_each(&mut visit);
    }
}

fn score_one(observed: &dyn Observed, aperture: f64, idx: usize) -> Candidate {
    let ewa_weight = ewa_aperture_weight(aperture, idx.abs_diff(SEED));
    let disagreement = route_disagreement(observed, idx);
    Candidate {
        idx,
        disagreement,
        ewa_weight,
        score: f64::from(disagreement) / 255.0 * ewa_weight,
    }
}

/// Top-1 as a streaming fold over the candidate aperture: state is ONE
/// `Option<Candidate>`, no row-id list, no candidate list, no sort.
///
/// With `tie_break`, the order is the same total order `score_and_sort`
/// uses (score desc, then ordinal asc), so the result cannot depend on
/// visitation order. Without it, a strict `>` keeps the FIRST visited of a
/// tie — falsifier R2.5-7.
fn top1_fold(
    observed: &dyn Observed,
    kept: &[u64],
    aperture: f64,
    descending: bool,
    tie_break: bool,
) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for_each_set_bit(kept, descending, |idx| {
        let c = score_one(observed, aperture, idx);
        let wins = match best {
            None => true,
            Some(b) => c.score > b.score || (tie_break && c.score == b.score && c.idx < b.idx),
        };
        if wins {
            best = Some(c);
        }
    });
    best
}

/// The ordinal baseline's pick: the lowest set bit — O(1), no fold at all.
fn lowest_set_bit(mask: &[u64]) -> Option<usize> {
    mask.iter()
        .enumerate()
        .find(|(_, &w)| w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
}

/// C's full ranking — only for the ranking differential. The cycle does not
/// need it (see `resident_differential`).
fn rank_resident(h: &Horizon, aperture: f64, poison: Poison, descending: bool) -> Vec<Candidate> {
    let program = lower(&Query {
        filter: candidate_filter(aperture),
        agg: Agg::Rows,
    })
    .expect("lowers");
    with_resident(h, poison, Gate::Valid, |o| {
        let kept = resident_candidates(core::slice::from_ref(&h.independent_roots), &program);
        let mut ids = Vec::new();
        for_each_set_bit(&kept, descending, |i| ids.push(i));
        score_and_sort(o, aperture, ids.into_iter())
    })
}

/// Round 2: A == B == C, plus the population-boundary measurements.
fn resident_differential(prior: &Horizon) {
    println!();
    println!("resident differential: C = borrowed independent_roots + projected_claims planes");
    let reference = observations(prior);

    // Plans are lowered once, outside every measured window: a plan is
    // query-sized, not population-sized.
    let programs: Vec<(f64, Program)> = APERTURES
        .iter()
        .map(|&a| {
            let program = lower(&Query {
                filter: candidate_filter(a),
                agg: Agg::Rows,
            })
            .expect("lowers");
            (a, program)
        })
        .collect();

    for (aperture, program) in &programs {
        let aperture = *aperture;
        let a = rank(&reference, aperture);
        for poison in POISONS {
            for descending in [false, true] {
                let b = rank_quack(prior, aperture, poison, descending);
                let c = rank_resident(prior, aperture, poison, descending);
                assert_eq!(b, a, "a={aperture} {poison:?}: B != A");
                assert_eq!(c, a, "a={aperture} {poison:?} desc={descending}: C != A");

                // k=1 needs no candidate list: the streaming fold agrees.
                let top = with_resident(prior, poison, Gate::Valid, |o| {
                    let kept = resident_candidates(
                        core::slice::from_ref(&prior.independent_roots),
                        program,
                    );
                    top1_fold(o, &kept, aperture, descending, true)
                });
                assert_eq!(
                    top,
                    a.first().copied(),
                    "a={aperture}: streaming top-1 != A"
                );
            }
        }
        let kept = resident_candidates(core::slice::from_ref(&prior.independent_roots), program);
        assert_eq!(
            lowest_set_bit(&kept),
            a.iter().map(|c| c.idx).min(),
            "a={aperture}: ordinal pick"
        );
        println!(
            "a={aperture:.2}  lowering={:?}  A==B==C ranking, streaming top-1 == A[0]",
            program.lowering()
        );
    }

    // Cycles: DAV and ordinal, every poison through C.
    let aperture = 0.55;
    let ranked = rank(&reference, aperture);
    for selected in [
        ranked[0].idx,
        ranked.iter().map(|c| c.idx).min().expect("frontier"),
    ] {
        let a = run_cycle(Carrier::Reference, prior, selected);
        for poison in POISONS {
            assert_eq!(run_cycle(Carrier::Quack(poison), prior, selected), a);
            let c = run_cycle(Carrier::Resident(poison), prior, selected);
            assert_eq!(c, a, "site {selected} {poison:?}: C cycle differs");
            let final_a = observations(&a.resulting);
            with_resident(&c.resulting, poison, Gate::Valid, |o| {
                for (i, &want) in final_a.iter().enumerate() {
                    assert_eq!(o.get(i), want, "final state differs at {i}");
                }
            });
        }
    }
    println!("cycles: DAV + ordinal, 4 poisons: A == B == C (delta, resulting, replay, residual)");

    // F4 (court of appeal) through C: its own consensus is refused.
    for poison in POISONS {
        let (laundered, delta) =
            adopt_route_consensus(Carrier::Resident(poison), prior, ranked[0].idx);
        assert_ne!(delta.evidential_effect, EvidentialEffect::IncreaseEligible);
        assert_eq!(&laundered, prior);
    }

    // R2.5-1 can-fire: a projected-but-unobserved bit read as evidence
    // changes the reasoning once the hidden truth is projected.
    let routes = |carrier| {
        with_reading(carrier, prior, |o| {
            (0..N).map(|i| route_disagreement(o, i)).collect::<Vec<_>>()
        })
    };
    assert_ne!(
        routes(Carrier::LeakyResident(Poison::Truth)),
        routes(Carrier::Resident(Poison::Truth)),
        "R2.5-1 can-fire: the leaky resident reader did not leak"
    );
    // R2.5-5 can-fire: without the validity gate, projected truth reaches the
    // routes — the unobserved hidden truth would close the selected hole.
    let target_disagreement =
        |carrier| with_reading(carrier, prior, |o| route_disagreement(o, ranked[0].idx));
    assert_eq!(target_disagreement(Carrier::Resident(Poison::Truth)), 255);
    assert_eq!(
        target_disagreement(Carrier::UngatedResident(Poison::Truth)),
        0,
        "R2.5-5 can-fire: ungated reading did not expose projected truth"
    );

    // R2.5-7 can-fire: after the DAV cycle every remaining candidate scores
    // 0, so top-1 is a pure tie. With the ordinal tie-break, visitation
    // order is inert; without it, it decides.
    let after = run_cycle(Carrier::Reference, prior, ranked[0].idx).resulting;
    let program = &programs.iter().find(|(a, _)| *a == 0.65).expect("a=0.65").1;
    let kept = resident_candidates(core::slice::from_ref(&after.independent_roots), program);
    let pick = |descending, tie_break| {
        with_resident(&after, Poison::Zero, Gate::Valid, |o| {
            top1_fold(o, &kept, 0.65, descending, tie_break).map(|c| c.idx)
        })
    };
    let tied = rank(&observations(&after), 0.65);
    assert!(
        tied.len() >= 2 && tied[0].score == tied[1].score,
        "anti-vacuity: top-1 must be a tie for R2.5-7"
    );
    assert_eq!(
        pick(false, true),
        pick(true, true),
        "tie-break must make order inert"
    );
    assert_eq!(pick(false, true), Some(tied[0].idx));
    assert_ne!(
        pick(false, false),
        pick(true, false),
        "R2.5-7 can-fire: without the tie-break visitation order must decide"
    );
    println!(
        "R2.5-7: tied top-1 at a=0.65 {:?}; no tie-break: asc {:?} / desc {:?}",
        tied.iter().map(|c| c.idx).collect::<Vec<_>>(),
        pick(false, false),
        pick(true, false)
    );

    // R2.5-6: the heap meter. Plans are already lowered.
    let (aperture, program) = (0.55, &programs[2].1);
    assert_eq!(programs[2].0, aperture);
    let valid = core::slice::from_ref(&prior.independent_roots);
    let (count, select_heap) = measure(|| {
        let kept = resident_candidates(valid, program);
        let mut n = 0usize;
        for_each_set_bit(&kept, false, |_| n += 1);
        n
    });
    let (top, top1_heap) = measure(|| {
        with_resident(prior, Poison::Zero, Gate::Valid, |o| {
            let kept = resident_candidates(valid, program);
            top1_fold(o, &kept, aperture, false, true)
        })
    });
    let (_, rank_c_heap) = measure(|| rank_resident(prior, aperture, Poison::Zero, false));
    let (_, rank_b_heap) = measure(|| rank_quack(prior, aperture, Poison::Zero, false));
    let (_, one_score_heap) = measure(|| {
        with_resident(prior, Poison::Zero, Gate::Valid, |o| {
            route_disagreement(o, 10)
        })
    });
    assert_eq!(top.map(|c| c.idx), Some(ranked[0].idx));
    assert_eq!(select_heap, 0, "C candidate selection allocated");
    assert_eq!(
        top1_heap,
        count * one_score_heap,
        "C top-1 heap must be exactly the route scorer's per-candidate allocation"
    );
    assert!(
        rank_b_heap > top1_heap && rank_c_heap > top1_heap,
        "R2.5-6 can-fire: the meter must see the list-building paths allocate"
    );
    println!(
        "heap bytes @a=0.55 ({count} candidates): C select+visit {select_heap}, \
         C streaming top-1 {top1_heap} (= {count} x {one_score_heap} route-scorer bytes), \
         C full rank {rank_c_heap}, B full rank {rank_b_heap}"
    );
    println!(
        "derived population bytes on C: kept mask {} (stack, caller-owned)",
        W * core::mem::size_of::<u64>()
    );
    println!("PASS: A == B == C; k=1 needs no candidate list; revision still the only write");
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
    resident_differential(&prior);
}
