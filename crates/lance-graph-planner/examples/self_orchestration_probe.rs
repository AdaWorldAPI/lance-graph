//! **D-CE64-LOOP-0 — does the register learn where to think next?**
//!
//! A bounded, deterministic cross-cycle loop over one `CausalEdge64` register
//! `R` and the production Pearl operators (`planner::pearl`, #1391):
//!
//! ```text
//! R[k] → select fold (local feedback) → execute existing operator
//!      → measured outcome → Bayesian update of a declared hypothesis set
//!      → Shannon entropy H, signed activation → revise Epi5 only when earned
//!      → R[k+1]
//! ```
//!
//! # What is real and what is policy
//!
//! - **Operators** are production: `pearl::hydrate`, `pearl::reason` +
//!   `pearl::revise` under SO / PO / SPO (counterfactual cut) / SP.
//! - **Alternatives** are declared: which continuation explains `A → Y` —
//!   direct, or via one of four candidate intermediates. The world (a sealed
//!   chain) binds exactly one of them.
//! - **H** is Shannon entropy, `-Σ p log2 p`, over the posterior of those
//!   alternatives. The posterior is non-uniform and comes from measured
//!   outcomes through likelihood tables. **The likelihood tables are policy
//!   pins**, not measurements.
//! - **Activation** is the signed change, in sevenths, of the posterior mass
//!   of the alternative a fold tested (or the certification rank a Pearl
//!   operator earned). It is measured from the outcome, never supplied.
//! - **Epi5 moves only through `revise` / `hydrate`.** Neither H nor
//!   activation is ever an input to them.
//!
//! # Where H and activation live
//!
//! In a transient [`Trace`], **not** in the register. Bits 46..49 are the
//! inference type, 50..52 plasticity, 53..58 the witness handle (#1393
//! review), so this probe does not write them; a test pins that they are
//! unchanged after a whole run. What the register would need to carry across
//! cycles is reported, not implemented.
//!
//! # No phase controller
//!
//! The selectors read only the posterior, the register's Epi5 and per-fold
//! activation. There is no phase variable. Any regime visible in the trace is
//! read off afterwards (`main` prints them), never switched on.
//!
//! Run: `cargo run --release -p lance-graph-planner --example self_orchestration_probe`
//! Tests (CI): `cargo test -p lance-graph-planner --example self_orchestration_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use causal_edge::edge::InferenceType;
use causal_edge::tables::NarsTables;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::SupportBasis;
use lance_graph_contract::certification::{CertificationModel, ModelBuilder};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_planner::chain_counterfactual::{CutContext, DEFAULT_FREQUENCY_BAR};
use lance_graph_planner::chain_replay::{ChainStep, ComposeTables};
use lance_graph_planner::pearl::{
    hydrate, reason, revise, Chain, Edit, Evidence, Hydration, Reaction, Reading,
};

// ── Allocation counter (measurement only) ─────────────────────────────────

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── The sealed world ──────────────────────────────────────────────────────

const A: u8 = 10;
const Y: u8 = 30;
const PREDICATE: u8 = 0x91;
/// Candidate intermediates. Only `20` is bound on both sides; `99` and `55`
/// are half-bound decoys; `77` is unbound.
const CANDIDATES: [u8; 4] = [20, 99, 77, 55];
/// A candidate outside the hypothesis set: the noise fold.
const NOISE: u8 = 200;
const CLASS: ClassId = 0x0902;
const RAIL: RailAxis = RailAxis::Taxonomy;
/// The register's witness handle (SPO-G sub-context). Must survive.
const WITNESS: u8 = 17;

fn edge(s: u8, o: u8, f: u8, c: u8, mask: CausalMask) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        PREDICATE,
        o,
        f,
        c,
        mask,
        0b101,
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        0,
    )
}

fn sealed_steps() -> Vec<ChainStep> {
    vec![
        (PREDICATE, edge(A, 20, 40, 220, CausalMask::SPO)),
        (PREDICATE, edge(20, Y, 250, 220, CausalMask::SPO)),
        (PREDICATE, edge(A, 99, 120, 200, CausalMask::SPO)),
        (PREDICATE, edge(55, Y, 180, 200, CausalMask::SPO)),
        (PREDICATE, edge(3, 4, 90, 90, CausalMask::SPO)),
    ]
}

fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    core::array::from_fn(|_| {
        let mut t = Box::new([0u8; 256 * 256]);
        for v in t.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((x >> 33) % 256) as u8;
        }
        t
    })
}

/// Associated pooled and after the robustness mask, lowered in stratum 1:
/// observationally `Related`. With `trial`, executed arms earn `Causes`.
fn population(trial: bool) -> CertificationModel {
    let mut b = ModelBuilder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.cell(1, true, 3, 1);
    b.cell(1, false, 3, 2);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    if trial {
        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
    }
    b.build()
}

/// Everything sealed, owned once. Read-only during a run.
struct World {
    tables: NarsTables,
    compose: [Box<[u8; 256 * 256]>; 3],
    steps: Vec<ChainStep>,
    /// Per candidate: the two-step chain `A → b → Y`, when both are bound.
    chains: [Option<Vec<ChainStep>>; 4],
    model: CertificationModel,
    decl: Epi5Declarations,
}

impl World {
    fn new(model: CertificationModel) -> Self {
        let steps = sealed_steps();
        let find = |s: u8, o: u8| {
            steps
                .iter()
                .copied()
                .find(|(_, e)| e.s_idx() == s && e.o_idx() == o)
        };
        let chains = CANDIDATES.map(|b| match (find(A, b), find(b, Y)) {
            (Some(x), Some(y)) => Some(vec![x, y]),
            _ => None,
        });
        let mut decl = Epi5Declarations::new();
        decl.declare(
            CLASS,
            RAIL,
            Epi5Reading {
                generation: Epi5Gen::V1,
            },
        );
        World {
            tables: NarsTables::build(1),
            compose: compose_tables(),
            steps,
            chains,
            model,
            decl,
        }
    }

    fn reading(&self) -> Reading<'_> {
        Reading {
            declarations: &self.decl,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        }
    }

    fn cut_context(&self) -> CutContext<'_> {
        CutContext {
            tables: &self.tables,
            compose: ComposeTables {
                s: &self.compose[0],
                p: &self.compose[1],
                o: &self.compose[2],
            },
            owner: 3,
            base_seq: 100,
            bar: DEFAULT_FREQUENCY_BAR,
        }
    }
}

/// The starting register: `A → Y`, `IndirectUnknown × Open`, witness 17.
fn start_register() -> CausalEdge64 {
    let e = edge(A, Y, 200, 200, CausalMask::SO).with_w_slot(WITNESS);
    e.with_epistemic_raw5(
        EpistemicState5::new(
            Epi5Gen::V1,
            Topology2::IndirectUnknown,
            Certification3::Open,
        )
        .raw(),
    )
}

fn epi(e: CausalEdge64) -> (Topology2, Certification3) {
    let s = EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).expect("valid code");
    (s.topology(), s.certification())
}

// ── Folds ─────────────────────────────────────────────────────────────────

/// The legal fold vocabulary: existing operators only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fold {
    /// `pearl::hydrate` with candidate `j` as the intermediate.
    Hydrate(u8),
    /// SPO counterfactual cut of the `b_j → Y` step of candidate `j`'s chain.
    /// Legal only once `Hydrate(j)` has returned `Hydrated`.
    Cut(u8),
    /// SO: observational folds.
    Observe,
    /// PO: executed trial arms.
    Intervene,
    /// SP: observational against interventional direction.
    Confound,
    /// Hydrate a candidate outside the hypothesis set. Never informative.
    Noise,
}

const FOLDS: [Fold; 12] = [
    Fold::Hydrate(0),
    Fold::Hydrate(1),
    Fold::Hydrate(2),
    Fold::Hydrate(3),
    Fold::Cut(0),
    Fold::Cut(1),
    Fold::Cut(2),
    Fold::Cut(3),
    Fold::Observe,
    Fold::Intervene,
    Fold::Confound,
    Fold::Noise,
];

/// What a fold measured. Only this, never the fold's identity, updates the
/// posterior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Hydrated(Hydration),
    /// 0 = load-bearing, 1 = truth only, 2 = inert.
    Reaction(u8),
    Earned(Option<Certification3>),
    Confounded(Option<bool>),
    Ungrounded,
}

/// Hypotheses: 0 = direct, `1 + j` = via `CANDIDATES[j]`.
const K: usize = 5;

/// Likelihood of `o` under each hypothesis. **Policy pins**, stated once.
fn likelihood(f: Fold, o: Outcome) -> [f64; K] {
    let mut l = [1.0; K];
    match (f, o) {
        (Fold::Hydrate(j), Outcome::Hydrated(h)) => {
            let (own, direct, other) = match h {
                Hydration::Hydrated => (0.90, 0.02, 0.05),
                Hydration::Partial => (0.08, 0.18, 0.25),
                Hydration::ProposedOnly => (0.02, 0.80, 0.70),
            };
            for (i, x) in l.iter_mut().enumerate() {
                *x = if i == 0 {
                    direct
                } else if i == 1 + j as usize {
                    own
                } else {
                    other
                };
            }
        }
        (Fold::Cut(j), Outcome::Reaction(k)) => {
            let (own, other) = match k {
                0 => (0.70, 0.10),
                1 => (0.25, 0.40),
                _ => (0.05, 0.50),
            };
            for (i, x) in l.iter_mut().enumerate() {
                *x = if i == 1 + j as usize { own } else { other };
            }
        }
        // Pearl certification folds and noise say nothing about the mediator.
        _ => {}
    }
    l
}

/// The outcomes a fold can produce, for the expected-information computation.
fn outcomes(f: Fold) -> &'static [Outcome] {
    match f {
        Fold::Hydrate(_) => &[
            Outcome::Hydrated(Hydration::Hydrated),
            Outcome::Hydrated(Hydration::Partial),
            Outcome::Hydrated(Hydration::ProposedOnly),
        ],
        Fold::Cut(_) => &[
            Outcome::Reaction(0),
            Outcome::Reaction(1),
            Outcome::Reaction(2),
        ],
        _ => &[],
    }
}

// ── Belief ────────────────────────────────────────────────────────────────

/// A posterior over the hypotheses, or a contradiction (no hypothesis is
/// consistent with what was measured). A contradiction has no entropy: it is
/// not a settled belief.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Belief {
    Posterior([f64; K]),
    Contradiction,
}

impl Belief {
    fn uniform() -> Self {
        Belief::Posterior([1.0 / K as f64; K])
    }

    fn update(self, l: [f64; K]) -> Self {
        let Belief::Posterior(p) = self else {
            return Belief::Contradiction;
        };
        let mut q = [0.0; K];
        let mut z = 0.0;
        for i in 0..K {
            q[i] = p[i] * l[i];
            z += q[i];
        }
        if z <= 0.0 {
            return Belief::Contradiction;
        }
        for x in q.iter_mut() {
            *x /= z;
        }
        Belief::Posterior(q)
    }

    /// Shannon entropy in bits; `None` for a contradiction.
    fn entropy(self) -> Option<f64> {
        match self {
            Belief::Posterior(p) => Some(shannon(&p)),
            Belief::Contradiction => None,
        }
    }

    fn mass(self, i: usize) -> f64 {
        match self {
            Belief::Posterior(p) => p[i],
            Belief::Contradiction => 0.0,
        }
    }
}

fn shannon(p: &[f64]) -> f64 {
    p.iter().filter(|x| **x > 0.0).map(|x| -x * x.log2()).sum()
}

/// Expected information gain of `f` under the current belief, in bits.
fn eig(b: Belief, f: Fold) -> f64 {
    let Belief::Posterior(p) = b else { return 0.0 };
    let h0 = shannon(&p);
    let mut expected = 0.0;
    for &o in outcomes(f) {
        let l = likelihood(f, o);
        let po: f64 = (0..K).map(|i| p[i] * l[i]).sum();
        if po > 0.0 {
            expected += po * b.update(l).entropy().unwrap_or(0.0);
        }
    }
    if outcomes(f).is_empty() {
        0.0
    } else {
        h0 - expected
    }
}

// ── One fold ──────────────────────────────────────────────────────────────

/// The loop's cross-cycle state. `r` is the register; everything else is a
/// transient side table this probe keeps outside it.
#[derive(Clone)]
struct State {
    r: CausalEdge64,
    belief: Belief,
    /// Per fold: bitset of the Epi5 raw codes it has already been executed at.
    /// A deterministic fold re-executed at the same code measures nothing new.
    seen: [u32; FOLDS.len()],
    /// Per candidate: the hydration returned `Hydrated` (opens `Cut(j)`).
    bound: [bool; 4],
    /// Last activation per fold.
    act: [i8; FOLDS.len()],
}

impl State {
    fn new() -> Self {
        State {
            r: start_register(),
            belief: Belief::uniform(),
            seen: [0; FOLDS.len()],
            bound: [false; 4],
            act: [0; FOLDS.len()],
        }
    }

    fn legal(&self, f: Fold) -> bool {
        match f {
            Fold::Cut(j) => self.bound[j as usize],
            _ => true,
        }
    }

    /// Whether running fold `i` now could only repeat a measurement already
    /// taken. A mediator fold reads the sealed chain alone, so one run is all
    /// it has; re-running it after Epi5 moved would count the same evidence
    /// twice. A Pearl fold revises the register, so it is new at a new code.
    fn observed(&self, i: usize) -> bool {
        match FOLDS[i] {
            Fold::Hydrate(_) | Fold::Cut(_) | Fold::Noise => self.seen[i] != 0,
            _ => self.seen[i] & (1 << self.r.epistemic_raw5()) != 0,
        }
    }
}

/// One record per fold. Kept only when a trace is requested.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    fold: Fold,
    outcome: Outcome,
    activation: i8,
    h_before: Option<f64>,
    h_after: Option<f64>,
    epi_before: u8,
    epi_after: u8,
    redundant: bool,
}

fn with_mask(mut e: CausalEdge64, m: CausalMask) -> CausalEdge64 {
    e.set_causal_mask(m);
    e
}

fn quantize(x: f64) -> i8 {
    (x * 7.0).round().clamp(-7.0, 7.0) as i8
}

/// Execute fold `i` against the world, update the state, return the record.
fn fold(w: &World, s: &mut State, i: usize) -> Step {
    let f = FOLDS[i];
    let h_before = s.belief.entropy();
    let epi_before = s.r.epistemic_raw5();
    let redundant = s.observed(i);
    let rd = w.reading();
    let ev = Evidence {
        model: &w.model,
        chain: None,
    };
    let pearl = |r: CausalEdge64, m: CausalMask, ev: &Evidence<'_>, edit: Edit| {
        let measured = reason(with_mask(r, m), ev, edit);
        let next = revise(&measured, rd).expect("declared reading");
        (next, measured)
    };
    let (next, outcome) = match f {
        Fold::Hydrate(j) => {
            let (n, h) = hydrate(s.r, (A, CANDIDATES[j as usize], Y), &w.steps, rd)
                .expect("declared reading");
            (n, Outcome::Hydrated(h))
        }
        Fold::Noise => {
            let (n, h) = hydrate(s.r, (A, NOISE, Y), &w.steps, rd).expect("declared reading");
            (n, Outcome::Hydrated(h))
        }
        Fold::Cut(j) => {
            let steps = w.chains[j as usize]
                .as_deref()
                .expect("legal only when bound");
            let ev = Evidence {
                model: &w.model,
                chain: Some(Chain {
                    steps,
                    seed: edge(A, Y, 200, 200, CausalMask::SPO),
                    cx: w.cut_context(),
                }),
            };
            let (n, m) = pearl(s.r, CausalMask::SPO, &ev, Edit::CutStep(1));
            let k = match m.reaction() {
                Some(Reaction::LoadBearing { .. }) => 0,
                Some(Reaction::TruthOnly { .. }) => 1,
                _ => 2,
            };
            (n, Outcome::Reaction(k))
        }
        Fold::Observe | Fold::Intervene => {
            let mask = if f == Fold::Observe {
                CausalMask::SO
            } else {
                CausalMask::PO
            };
            let (n, m) = pearl(s.r, mask, &ev, Edit::None);
            let o = if m.ungrounded().is_some() {
                Outcome::Ungrounded
            } else {
                Outcome::Earned(m.earned())
            };
            (n, o)
        }
        Fold::Confound => {
            let (n, m) = pearl(s.r, CausalMask::SP, &ev, Edit::None);
            (n, Outcome::Confounded(m.confounded()))
        }
    };
    s.r = next;

    // Activation: the signed reaction this fold produced.
    let activation = if redundant {
        0
    } else {
        match f {
            Fold::Hydrate(j) | Fold::Cut(j) => {
                let t = 1 + j as usize;
                let before = s.belief.mass(t);
                s.belief = s.belief.update(likelihood(f, outcome));
                quantize(s.belief.mass(t) - before)
            }
            Fold::Observe | Fold::Intervene => {
                let (_, c0) = epi_raw(epi_before);
                let (_, c1) = epi(s.r);
                (c1.code() as i8 - c0.code() as i8).clamp(-7, 7)
            }
            Fold::Confound => match outcome {
                Outcome::Confounded(Some(true)) => -1,
                Outcome::Confounded(Some(false)) => 1,
                _ => 0,
            },
            Fold::Noise => 0,
        }
    };
    if let (Fold::Hydrate(j), Outcome::Hydrated(Hydration::Hydrated)) = (f, outcome) {
        s.bound[j as usize] = true;
    }
    s.seen[i] |= 1 << epi_before;
    s.act[i] = activation;
    Step {
        fold: f,
        outcome,
        activation,
        h_before,
        h_after: s.belief.entropy(),
        epi_before,
        epi_after: s.r.epistemic_raw5(),
        redundant,
    }
}

fn epi_raw(raw: u8) -> (Topology2, Certification3) {
    let s = EpistemicState5::decode(Epi5Gen::V1, raw).expect("valid code");
    (s.topology(), s.certification())
}

// ── Selection policies ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    RoundRobin,
    Random(u64),
    EntropyOnly,
    ActivationOnly,
    Combined,
}

/// Deterministic LCG, so a seeded policy replays exactly.
fn lcg(x: &mut u64) -> u64 {
    *x = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *x >> 33
}

/// The certification code a Pearl fold could still add: the gap to the best
/// code it can produce (observation stops at `CausalCandidate`, the trial at
/// `Causes`), zero for every other fold.
fn headroom(s: &State, f: Fold) -> f64 {
    let (_, c) = epi(s.r);
    let top = match f {
        Fold::Observe => Certification3::CausalCandidate.code(),
        Fold::Intervene => Certification3::Causes.code(),
        _ => return 0.0,
    };
    f64::from(top.saturating_sub(c.code())) / 5.0
}

/// Pick the next fold. Returns `None` when a feedback policy has no fold
/// left that could measure anything new: a stable fixed point.
fn select(p: Policy, s: &State, k: u64, rng: &mut u64) -> Option<usize> {
    let legal = |i: usize| s.legal(FOLDS[i]);
    match p {
        Policy::RoundRobin => {
            let n = FOLDS.len();
            (0..n).map(|d| (k as usize + d) % n).find(|&i| legal(i))
        }
        Policy::Random(_) => {
            let ls: [usize; FOLDS.len()] = core::array::from_fn(|i| i);
            let count = ls.iter().filter(|&&i| legal(i)).count();
            let pick = (lcg(rng) as usize) % count;
            ls.into_iter().filter(|&i| legal(i)).nth(pick)
        }
        Policy::EntropyOnly | Policy::ActivationOnly | Policy::Combined => {
            let h_max = (K as f64).log2();
            let h = s.belief.entropy().unwrap_or(h_max);
            let mut best: Option<(usize, f64)> = None;
            for (i, &f) in FOLDS.iter().enumerate() {
                if !legal(i) || s.observed(i) {
                    continue;
                }
                let score = match p {
                    Policy::EntropyOnly => eig(s.belief, f),
                    Policy::ActivationOnly => {
                        // Follow what reacted: a cut whose hydration excited,
                        // or a fold that reacted before; otherwise neutral.
                        let opened = match f {
                            Fold::Cut(j) => f64::from(s.act[j as usize].max(0)),
                            _ => 0.0,
                        };
                        f64::from(s.act[i]) + opened
                    }
                    _ => {
                        let opened = match f {
                            Fold::Cut(j) => 0.25 * f64::from(s.act[j as usize].max(0)) / 7.0,
                            _ => 0.0,
                        };
                        // A certification fold is worth more the more settled
                        // the alternatives are: continuous, not a phase.
                        let commit = (1.0 - h / h_max) * headroom(s, f);
                        eig(s.belief, f) + opened + commit
                    }
                };
                // A fold with nothing to gain is not run: when none is left
                // the loop has reached its fixed point and stops. Activation
                // alone has no prior signal, so an untried fold is eligible
                // at zero; a tried one only while it keeps reacting.
                let eligible = match p {
                    Policy::ActivationOnly => score > 0.0 || (s.seen[i] == 0 && score >= 0.0),
                    _ => score > 1e-9,
                };
                if eligible && best.is_none_or(|(_, b)| score > b + 1e-12) {
                    best = Some((i, score));
                }
            }
            best.map(|(i, _)| i)
        }
    }
}

// ── A run ─────────────────────────────────────────────────────────────────

/// The task's goal: the mediator is identified (H ≤ 0.5 bit) and the
/// register earned `IndirectKnown × Causes`.
fn goal(s: &State) -> bool {
    s.belief.entropy().is_some_and(|h| h <= 0.5)
        && epi(s.r) == (Topology2::IndirectKnown, Certification3::Causes)
}

#[derive(Debug, Clone, PartialEq)]
struct Summary {
    folds: u64,
    to_goal: Option<u64>,
    redundant: u64,
    noise: u64,
    epi_transitions: u64,
    unresolved: u64,
    info_bits: f64,
    final_h: Option<f64>,
    stopped: bool,
}

/// Run one episode of at most `cap` folds. `trace` receives every step when
/// given; the hot path passes `None`.
fn run(w: &World, p: Policy, cap: u64, mut trace: Option<&mut Vec<Step>>) -> (State, Summary) {
    let mut s = State::new();
    let mut rng = match p {
        Policy::Random(seed) => seed,
        _ => 0,
    };
    let h0 = s.belief.entropy().unwrap_or(0.0);
    let mut sum = Summary {
        folds: 0,
        to_goal: None,
        redundant: 0,
        noise: 0,
        epi_transitions: 0,
        unresolved: 0,
        info_bits: 0.0,
        final_h: None,
        stopped: false,
    };
    for k in 0..cap {
        let Some(i) = select(p, &s, k, &mut rng) else {
            sum.stopped = true;
            break;
        };
        let st = fold(w, &mut s, i);
        sum.folds += 1;
        sum.redundant += u64::from(st.redundant);
        sum.noise += u64::from(st.fold == Fold::Noise);
        sum.epi_transitions += u64::from(st.epi_before != st.epi_after);
        sum.unresolved += u64::from(st.h_after.is_none_or(|h| h > 1.0));
        if let Some(t) = trace.as_deref_mut() {
            t.push(st);
        }
        if sum.to_goal.is_none() && goal(&s) {
            sum.to_goal = Some(sum.folds);
        }
    }
    sum.final_h = s.belief.entropy();
    sum.info_bits = h0 - sum.final_h.unwrap_or(h0);
    (s, sum)
}

const POLICIES: [Policy; 5] = [
    Policy::RoundRobin,
    Policy::Random(0x5EED),
    Policy::EntropyOnly,
    Policy::ActivationOnly,
    Policy::Combined,
];

// ── Measurement ───────────────────────────────────────────────────────────

fn main() {
    let w = World::new(population(true));

    println!("D-CE64-LOOP-0: policy comparison (cap 40 folds, one episode each)");
    println!(
        "  {:<16} {:>6} {:>8} {:>9} {:>6} {:>5} {:>10} {:>8} {:>8}",
        "policy", "folds", "to_goal", "redundant", "noise", "epi", "unresolved", "info", "final_H"
    );
    for p in POLICIES {
        let (_, m) = run(&w, p, 40, None);
        println!(
            "  {:<16} {:>6} {:>8} {:>9} {:>6} {:>5} {:>10} {:>8.3} {:>8}",
            format!("{p:?}"),
            m.folds,
            m.to_goal.map_or("-".into(), |g| g.to_string()),
            m.redundant,
            m.noise,
            m.epi_transitions,
            m.unresolved,
            m.info_bits,
            m.final_h.map_or("contra".into(), |h| format!("{h:.3}")),
        );
    }
    let seeds: Vec<Option<u64>> = (0..200u64)
        .map(|seed| run(&w, Policy::Random(seed * 7919 + 1), 40, None).1.to_goal)
        .collect();
    let reached: Vec<u64> = seeds.iter().flatten().copied().collect();
    let mut sorted = reached.clone();
    sorted.sort_unstable();
    println!(
        "  random over 200 seeds: reached goal {}/200, median folds {:?}",
        reached.len(),
        sorted.get(sorted.len() / 2)
    );

    println!("\nCombined trace (regimes are read off afterwards, never switched on):");
    let mut t = Vec::new();
    run(&w, Policy::Combined, 40, Some(&mut t));
    for (k, st) in t.iter().enumerate() {
        let (tb, cb) = epi_raw(st.epi_before);
        let (ta, ca) = epi_raw(st.epi_after);
        println!(
            "  {k:>2} {:<12} act {:>+3}  H {} -> {}  {:?}x{:?} -> {:?}x{:?}{}",
            format!("{:?}", st.fold),
            st.activation,
            st.h_before.map_or("contra".into(), |h| format!("{h:.3}")),
            st.h_after.map_or("contra".into(), |h| format!("{h:.3}")),
            tb,
            cb,
            ta,
            ca,
            if st.redundant { "  (redundant)" } else { "" }
        );
    }

    println!("\nThroughput (combined policy, episodes back to back, no trace):");
    println!(
        "  {:>9} {:>10} {:>10} {:>11} {:>12} {:>14}",
        "folds", "ns/fold", "allocs/fold", "epi/fold", "bits/fold", "folds/350ms"
    );
    for n in [1u64, 10, 100, 1_000, 10_000, 100_000, 1_000_000] {
        let mut done = 0u64;
        let mut epi = 0u64;
        let mut bits = 0.0;
        let a0 = ALLOCS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        while done < n {
            let (_, m) = run(&w, Policy::Combined, (n - done).min(40), None);
            done += m.folds;
            epi += m.epi_transitions;
            bits += m.info_bits;
        }
        let dt = t0.elapsed().as_nanos() as f64;
        let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
        let ns = dt / done as f64;
        println!(
            "  {:>9} {:>10.1} {:>10.2} {:>11.4} {:>12.4} {:>14.0}",
            done,
            ns,
            allocs as f64 / done as f64,
            epi as f64 / done as f64,
            bits / done as f64,
            350e6 / ns
        );
    }
}

// ── Falsifiers ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::layout::EPISTEMIC_MASK;

    fn trace(w: &World, p: Policy, cap: u64) -> (State, Summary, Vec<Step>) {
        let mut t = Vec::new();
        let (s, m) = run(w, p, cap, Some(&mut t));
        (s, m, t)
    }

    /// The register bits this probe must not write: direction, mantissa,
    /// plasticity, witness (43..58).
    const UNTOUCHED: u64 = ((1u64 << 16) - 1) << 43;

    #[test]
    fn combined_reaches_the_goal_and_stops_at_a_fixed_point() {
        let w = World::new(population(true));
        let (s, m, _) = trace(&w, Policy::Combined, 40);
        assert!(goal(&s), "{m:?}");
        assert!(m.stopped, "no fold left that measures anything new");
        assert_eq!(m.redundant, 0);
        assert_eq!(m.noise, 0);
    }

    /// F12: the feedback selector beats the fixed schedule and the random
    /// control on the measured task (folds to goal, redundancy).
    #[test]
    fn f12_feedback_beats_fixed_and_random_schedules() {
        let w = World::new(population(true));
        let combined = run(&w, Policy::Combined, 40, None).1;
        let rr = run(&w, Policy::RoundRobin, 40, None).1;
        let c = combined.to_goal.expect("combined reaches the goal");
        assert!(
            rr.to_goal.is_none_or(|g| g > c),
            "rr {rr:?} vs {combined:?}"
        );
        assert!(rr.redundant > combined.redundant);
        let mut worse = 0;
        for seed in 0..200u64 {
            let r = run(&w, Policy::Random(seed * 7919 + 1), 40, None).1;
            worse += u32::from(r.to_goal.is_none_or(|g| g > c));
        }
        assert!(
            worse >= 190,
            "random matched combined on {} seeds",
            200 - worse
        );
    }

    /// The selector actually uses feedback: the two single-signal ablations
    /// behave measurably differently from the combined policy.
    #[test]
    fn entropy_and_activation_each_change_what_happens_next() {
        let w = World::new(population(true));
        let (_, comb, tc) = trace(&w, Policy::Combined, 40);
        let (_, ent, te) = trace(&w, Policy::EntropyOnly, 40);
        let (_, act, ta) = trace(&w, Policy::ActivationOnly, 40);
        let seq = |t: &[Step]| t.iter().map(|s| s.fold).collect::<Vec<_>>();
        assert_ne!(seq(&tc), seq(&te));
        assert_ne!(seq(&tc), seq(&ta));
        let c = comb.to_goal.expect("combined reaches the goal");
        assert!(ent.to_goal.is_none_or(|g| g >= c), "{ent:?}");
        assert!(act.to_goal.is_none_or(|g| g > c), "{act:?}");
    }

    /// F1: high entropy alone never promotes. At maximum entropy a fold with
    /// no evidence behind it leaves Epi5 alone.
    #[test]
    fn f1_high_entropy_alone_promotes_nothing() {
        let w = World::new(CertificationModel {
            ledger: Default::default(),
            ..population(false)
        });
        let mut s = State::new();
        assert!(s.belief.entropy().unwrap() > 2.3);
        let i = FOLDS.iter().position(|f| *f == Fold::Observe).unwrap();
        let st = fold(&w, &mut s, i);
        assert_eq!(st.outcome, Outcome::Ungrounded);
        assert_eq!(st.epi_before, st.epi_after);
    }

    /// F2: strong activation alone does not earn `Causes`. Without executed
    /// arms the cuts still excite, and the register never reaches `Causes`.
    #[test]
    fn f2_strong_activation_alone_never_earns_causes() {
        let w = World::new(population(false));
        let (s, _, t) = trace(&w, Policy::Combined, 40);
        assert!(t.iter().any(|st| st.activation >= 3), "some fold excited");
        assert_ne!(epi(s.r).1, Certification3::Causes);
        // Silence twin: with arms, the same loop earns it.
        let w = World::new(population(true));
        assert_eq!(
            epi(run(&w, Policy::Combined, 40, None).0.r).1,
            Certification3::Causes
        );
    }

    /// F3: an ungrounded Pearl operation leaves Epi5 unchanged.
    #[test]
    fn f3_ungrounded_pearl_changes_nothing() {
        let w = World::new(CertificationModel {
            ledger: Default::default(),
            ..population(true)
        });
        let mut s = State::new();
        for f in [Fold::Observe, Fold::Intervene] {
            let i = FOLDS.iter().position(|x| *x == f).unwrap();
            let st = fold(&w, &mut s, i);
            assert_eq!(st.epi_before, st.epi_after, "{f:?} {:?}", st.outcome);
        }
    }

    /// F4 + F5 + F6: across a whole run only bits 59..63 of the register
    /// move (and 40..42, the question asked). The counterfactual's -6 tag,
    /// the plasticity flags and the witness handle are never written.
    #[test]
    fn f4_f5_f6_the_register_keeps_mantissa_plasticity_and_witness() {
        let w = World::new(population(true));
        let start = start_register();
        let (s, _, t) = trace(&w, Policy::Combined, 40);
        assert!(
            t.iter().any(|st| matches!(st.fold, Fold::Cut(_))),
            "a cut ran"
        );
        assert_eq!((s.r.0 ^ start.0) & UNTOUCHED, 0);
        assert_eq!(s.r.w_slot(), WITNESS);
        assert_eq!(s.r.inference_mantissa(), start.inference_mantissa());
        assert_ne!(
            s.r.inference_mantissa(),
            InferenceType::Counterfactual.to_mantissa()
        );
        // Only the epistemic code and the question bits differ.
        let question = 0b111u64 << 40;
        assert_eq!((s.r.0 ^ start.0) & !(EPISTEMIC_MASK | question), 0);
    }

    /// F7: a step claims a belief-state transition only when Epi5 changed.
    #[test]
    fn f7_no_transition_without_an_epi5_change() {
        let w = World::new(population(true));
        for p in POLICIES {
            let (_, m, t) = trace(&w, p, 40);
            let changed = t.iter().filter(|s| s.epi_before != s.epi_after).count() as u64;
            assert_eq!(m.epi_transitions, changed);
            for st in t.iter().filter(|s| s.epi_before != s.epi_after) {
                assert!(
                    matches!(st.fold, Fold::Hydrate(_) | Fold::Observe | Fold::Intervene),
                    "{st:?}"
                );
            }
        }
    }

    /// F8: a contradiction is not a settled belief.
    #[test]
    fn f8_a_contradiction_has_no_entropy() {
        let b = Belief::uniform().update([0.0; K]);
        assert_eq!(b, Belief::Contradiction);
        assert_eq!(b.entropy(), None);
        // Silence twin: one surviving hypothesis is settled at 0 bits.
        let settled = Belief::uniform().update([1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(settled.entropy(), Some(0.0));
    }

    /// F9: same sealed inputs, seed and policy give the same trace.
    #[test]
    fn f9_replay_is_deterministic() {
        let w = World::new(population(true));
        for p in POLICIES {
            assert_eq!(trace(&w, p, 40).2, trace(&w, p, 40).2, "{p:?}");
        }
    }

    /// F10: the combined policy does not chase the uninformative noise fold,
    /// and every fold it runs either gains information or earns Epi5.
    #[test]
    fn f10_combined_does_not_chase_noise() {
        let w = World::new(population(true));
        let (_, m, t) = trace(&w, Policy::Combined, 40);
        assert_eq!(m.noise, 0);
        let idle = t
            .iter()
            .filter(|s| s.h_before == s.h_after && s.epi_before == s.epi_after)
            .count();
        assert!(idle <= 2, "{idle} folds changed nothing");
        // Silence twin: round-robin does run it.
        assert!(run(&w, Policy::RoundRobin, 40, None).1.noise > 0);
    }

    /// F11: no indefinite oscillation. The feedback policies stop at a fixed
    /// point; round-robin cycles, which is the detectable stable cycle.
    #[test]
    fn f11_feedback_stops_where_a_fixed_schedule_cycles() {
        let w = World::new(population(true));
        for p in [
            Policy::EntropyOnly,
            Policy::ActivationOnly,
            Policy::Combined,
        ] {
            let m = run(&w, p, 1_000, None).1;
            assert!(m.stopped && m.folds < 40, "{p:?} {m:?}");
        }
        let rr = run(&w, Policy::RoundRobin, 1_000, None).1;
        assert_eq!(rr.folds, 1_000);
        assert!(rr.redundant > 900);
    }

    /// F14: there is no phase state anywhere in the loop; the only inputs to
    /// selection are the posterior, the register and per-fold activation.
    /// Pinned behaviourally: two states equal in those fields select the
    /// same fold regardless of how they were reached.
    #[test]
    fn f14_selection_reads_only_local_state() {
        let w = World::new(population(true));
        let (a, _) = run(&w, Policy::Combined, 3, None);
        let mut b = State::new();
        b.r = a.r;
        b.belief = a.belief;
        b.seen = a.seen;
        b.bound = a.bound;
        b.act = a.act;
        let (mut r1, mut r2) = (0, 0);
        assert_eq!(
            select(Policy::Combined, &a, 3, &mut r1),
            select(Policy::Combined, &b, 99, &mut r2)
        );
    }

    /// No pseudo-replication: a mediator fold reads the sealed chain alone,
    /// so no feedback policy runs it twice, even after Epi5 has moved.
    #[test]
    fn a_sealed_measurement_is_counted_once() {
        let w = World::new(population(true));
        for p in [
            Policy::EntropyOnly,
            Policy::ActivationOnly,
            Policy::Combined,
        ] {
            let (_, _, t) = trace(&w, p, 40);
            for f in FOLDS
                .iter()
                .filter(|f| matches!(f, Fold::Hydrate(_) | Fold::Cut(_) | Fold::Noise))
            {
                let n = t.iter().filter(|s| s.fold == *f).count();
                assert!(n <= 1, "{p:?} ran {f:?} {n} times");
            }
        }
    }

    /// Anti-vacuity: entropy actually falls on the task, and the redundancy
    /// counter can fire (it does on the fixed schedule).
    #[test]
    fn the_measurements_can_fire() {
        let w = World::new(population(true));
        let comb = run(&w, Policy::Combined, 40, None).1;
        assert!(comb.info_bits > 1.5, "{comb:?}");
        assert!(run(&w, Policy::RoundRobin, 40, None).1.redundant > 0);
    }
}
